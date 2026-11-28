"""Handoffs in the public composio_swekit SWE-bench trajectories, and their token alignment. CPU only.

A trajectory file is a JSON list of sub-runs (one per model call); each sub-run lists LangChain message
dicts under `kwargs`, and a response may instead be a LangChain LLMResult node. The serving model is echoed
per response. A HANDOFF is a change of serving model between two consecutive responses of one trajectory:
the sender context S is every message up to the sender's last response, and the receiver prompt R is the
prompt of the receiving model's own call. Both are joined with a newline.

The alignment is `difflib.SequenceMatcher(None, S_ids, R_ids, autojunk=False).get_matching_blocks()`:
deterministic, a common subsequence, in general shorter than the longest one, so the matched set is a floor.

Provenance: UPSTREAM.md.
"""
import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_KEYS = ("model", "model_id", "model_name")

_ROLE_BY_LC_TYPE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


@dataclass(frozen=True)
class Msg:
    role: str                      # system | user | assistant | tool
    model: str | None = None       # per-step serving model, None when the trace has none
    request: int | None = None     # the model call (sub-run) this message belongs to


@dataclass(frozen=True)
class Trajectory:
    agent: str                     # the submission's folder name
    traj_id: str
    messages: tuple[Msg, ...] = field(default_factory=tuple)


def models_in(obj, keys=MODEL_KEYS, _depth=0) -> list[str]:
    """Every serving-identity string anywhere in `obj`, in document order (duplicates kept)."""
    out: list[str] = []
    if _depth > 12:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, str) and v and v.lower() != "default":
                out.append(v)
            out.extend(models_in(v, keys, _depth + 1))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(models_in(v, keys, _depth + 1))
    return out


def _lc_text(kw: dict) -> str:
    c = kw.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):   # content blocks
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


def _lc_role(kw: dict, node: dict) -> str:
    """The role of a LangChain message node, from its `type` or the class name ending its `id`. Matched on
    the whole word: a substring test would read every `langchain...` id as "ai". An unknown node is refused."""
    names = [kw.get("type")]
    if isinstance(node.get("id"), list) and node["id"]:
        names.append(str(node["id"][-1]).removesuffix("Message").removesuffix("Chunk"))
    for name in names:
        if isinstance(name, str) and name.lower() in _ROLE_BY_LC_TYPE:
            return _ROLE_BY_LC_TYPE[name.lower()]
    raise ValueError(f"message node of unknown role: id={node.get('id')!r} type={kw.get('type')!r}")


def _flatten_nodes(sub) -> list[dict]:
    """Every message/LLMResult node of one sub-run, in document order. One submission lists message
    nodes directly; another nests the whole prompt as one list node. Nesting is flattened at any depth."""
    out: list[dict] = []
    stack = list(sub) if isinstance(sub, list) else []
    stack.reverse()
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            out.append(node)
        elif isinstance(node, list):
            stack.extend(reversed(node))
    return out


def _llmresult_texts(node: dict) -> list[str]:
    """Response texts from a LangChain LLMResult node ({llm_output, run, generations})."""
    if "generations" not in node:
        return []
    out = []
    for outer in node.get("generations") or []:
        for leaf in (outer if isinstance(outer, list) else [outer]):
            if isinstance(leaf, dict) and isinstance(leaf.get("text"), str):
                out.append(leaf["text"])
    return out


def load_trajectory(path: Path, agent: str) -> tuple[Trajectory, list[str], list[bool]]:
    """One trajectory file -> (trajectory, the text of each message, whether each message is a recorded
    RESPONSE of its call rather than a message of its prompt)."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, list):
        raise ValueError(f"{path}: expected a JSON list of sub-runs, got {type(doc).__name__}")
    messages: list[Msg] = []
    texts: list[str] = []
    is_response: list[bool] = []
    for si, sub in enumerate(doc):
        found = models_in(sub)
        model = found[0] if found else None
        for node in _flatten_nodes(sub):
            kw = node.get("kwargs")
            if isinstance(kw, dict):
                messages.append(Msg(role=_lc_role(kw, node), model=model, request=si))
                texts.append(_lc_text(kw))
                is_response.append(False)
                continue
            for text in _llmresult_texts(node):
                resp_model = (node.get("llm_output") or {}).get("model_name") or model
                messages.append(Msg(role="assistant", model=resp_model, request=si))
                texts.append(text)
                is_response.append(True)
    if not messages:
        raise ValueError(f"{path}: no LangChain message nodes found; wrong adapter for this shape")
    return Trajectory(agent=agent, traj_id=f"{agent}/{path.stem}", messages=tuple(messages)), texts, is_response


@dataclass(frozen=True)
class Handoff:
    handoff_id: str          # "<submission>/<file stem>#<switch index>"
    sender_model: str
    receiver_model: str
    sender_text: str
    receiver_text: str
    response_text: str       # what the receiving model answered, as the trace records it


def handoffs_from(traj: Trajectory, texts: list[str], is_response: list[bool]) -> list[Handoff]:
    """Walk the messages that are not user messages (model, tool and system messages), in order. A handoff
    is a RECORDED RESPONSE whose serving model differs from the model attached to the one before it. S is
    every message up to and including that earlier one; R is the receiver's own request prompt (the
    messages of its call that precede its response), never the trajectory prefix. A change of model that
    lands on prompt history rather than on a recorded response is not a handoff: nothing was answered."""
    if not len(texts) == len(is_response) == len(traj.messages):
        raise ValueError("texts and response flags must align with the messages")
    out = []
    asst = [(i, m) for i, m in enumerate(traj.messages) if m.role != "user"]
    for k in range(1, len(asst)):
        (i_prev, prev), (i_cur, cur) = asst[k - 1], asst[k]
        if prev.model is None or cur.model is None or prev.model == cur.model or not is_response[i_cur]:
            continue
        if cur.request is None:
            raise ValueError(f"{traj.traj_id}: switch at turn {k} without a request boundary")
        prompt_idx = [j for j in range(i_cur) if traj.messages[j].request == cur.request]
        out.append(Handoff(handoff_id=f"{traj.traj_id}#{k}", sender_model=prev.model, receiver_model=cur.model,
                           sender_text="\n".join(texts[:i_prev + 1]),
                           receiver_text="\n".join(texts[j] for j in prompt_idx),
                           response_text=texts[i_cur]))
    return out


def load_handoffs(submission_dirs: list[Path]) -> list[Handoff]:
    out = []
    for sub in submission_dirs:
        for f in sorted(Path(sub).glob("*.json")):
            out.extend(handoffs_from(*load_trajectory(f, Path(sub).name)))
    return out


def matching_pairs(a: list[int], b: list[int]) -> np.ndarray:
    """[[p_S, p_R], ...] over every token in the matching blocks, in order."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pairs = []
    for i, j, n in sm.get_matching_blocks():
        for t in range(n):
            pairs.append((i + t, j + t))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
