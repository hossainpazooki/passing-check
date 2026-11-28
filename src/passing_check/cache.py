"""Assemble a reused cache at a handoff and read the reader under it. Same model on both sides.

The reader's key and value states for the matched tokens are taken from its prefill of the sender context,
the keys are moved from their sender positions to their receiver positions under the model's own rotary
schedule, and every unmatched receiver token is computed against the cache assembled so far.

Provenance: UPSTREAM.md.
"""
import numpy as np
import torch
from transformers import DynamicCache


def half_rotate(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def relocate(keys, source_positions, destination_positions, inv_freq):
    # Undo the source rotation, then apply the destination rotation under the SAME static schedule.
    # The amplitude already in the keys is preserved. Two rotations avoid the long-position fp32
    # phase-rounding error of computing only (destination - source) * inv_freq.
    def phase(positions):
        angles = torch.outer(torch.as_tensor(positions, device=keys.device).float(),
                             inv_freq.to(keys.device).float())
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()
    source_cos, source_sin = phase(source_positions)
    dest_cos, dest_sin = phase(destination_positions)
    x = keys.float()
    content = x * source_cos - half_rotate(x) * source_sin
    return (content * dest_cos + half_rotate(content) * dest_sin).to(keys.dtype)


def validate_pairs(sender, receiver, pairs):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ValueError("pairs must be an integer [n, 2] array")
    if len(pairs) and (np.any(pairs < 0) or np.any(pairs[:, 0] >= len(sender))
                       or np.any(pairs[:, 1] >= len(receiver))):
        raise ValueError("matched position outside prompt")
    if len(pairs) and (len(np.unique(pairs[:, 0])) != len(pairs)
                       or np.any(np.diff(pairs[:, 1]) <= 0)):
        raise ValueError("matched positions must be unique and receiver-ordered")
    if not np.array_equal(np.asarray(sender)[pairs[:, 0]], np.asarray(receiver)[pairs[:, 1]]):
        raise ValueError("matched tokens differ")


def blocks(pairs):
    # Contiguous runs of matched rows; one-token matches are kept.
    rows = np.arange(len(pairs))
    if not len(rows):
        return []
    cuts = np.flatnonzero((np.diff(pairs[rows, 0]) != 1) | (np.diff(pairs[rows, 1]) != 1)) + 1
    return np.split(rows, cuts)


@torch.inference_mode()
def prefill(model, tokens, chunk, cache=None, last_logits=False):
    """Extend (or start) a cache with `tokens`. With `last_logits`, also return the logits after the last
    token, which predict the next one; there must then be at least one token."""
    cache = DynamicCache(config=model.config) if cache is None else cache
    tokens = torch.as_tensor(tokens, dtype=torch.long, device=model.device).reshape(1, -1)
    if last_logits and not tokens.shape[1]:
        raise ValueError("no token to compute: the end of the prompt must be computed, not copied")
    out = None
    for start in range(0, tokens.shape[1], chunk):
        out = model(tokens[:, start:start + chunk], past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
    return (cache, out.logits[0, -1].float().cpu()) if last_logits else cache


def matched(cache, positions):
    # Keep only the selected states, on CPU, so a long cache can be freed.
    return [(layer.keys.index_select(-2, torch.as_tensor(positions, device=layer.keys.device)).cpu(),
             layer.values.index_select(-2, torch.as_tensor(positions, device=layer.values.device)).cpu())
            for layer in cache.layers]


def relocated_candidate(source, pairs, inv_freq):
    return [(relocate(k, pairs[:, 0], pairs[:, 1], inv_freq), v) for k, v in source]


def random_candidate(fresh, reused, rng, scale=1.0):
    # Isotropic noise on the fresh states, with the reused cache's K and V error norm copied exactly per
    # token, layer and head. Removing position is a rotation and keeps norms, so this candidate reads the
    # same as the reused cache on the key deviation. `scale` shrinks the noise: the key deviation then
    # reads scale squared times the reused cache's, along the same drawn direction.
    result = []
    for layer_fresh, layer_reused in zip(fresh, reused, strict=True):
        output = []
        for reference, candidate in zip(layer_fresh, layer_reused, strict=True):
            noise = torch.from_numpy(rng.standard_normal(reference.shape).astype(np.float32))
            norm = torch.linalg.vector_norm((candidate - reference).float(), dim=-1, keepdim=True)
            noise = noise / torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
            output.append(reference + (noise * norm * scale).to(reference.dtype))
        result.append(tuple(output))
    return result


def unrelated_candidate(fresh, pairs, inv_freq):
    """Each matched token given ANOTHER matched token's fresh key (the one half the matched set away), moved
    from that token's position to this one's, so that removing this token's position leaves the other key's
    content and no stray rotation. Values are left as they are; this candidate is only read on its keys."""
    shift = len(pairs) // 2
    origin = np.roll(pairs[:, 1], shift)
    return [(relocate(k.roll(shift, dims=-2), origin, pairs[:, 1], inv_freq), v) for k, v in fresh]


def key_deviation(fresh, candidate, pairs, inv_freq):
    """Per matched token, layer and head: the squared distance between the candidate's key and the fresh
    key, both with position removed, as a multiple of that layer-head's mean squared distance of a fresh
    matched key from the matched-set mean. Shape [n, L, H], float64. 0 is identity; 1 is the spread of the
    fresh keys themselves."""
    zero = np.zeros(len(pairs), dtype=np.int64)
    layers = []
    for reference, other in zip(fresh, candidate, strict=True):
        ref = relocate(reference[0], pairs[:, 1], zero, inv_freq).double()
        old = relocate(other[0], pairs[:, 1], zero, inv_freq).double()
        denominator = ((ref - ref.mean(-2, keepdim=True)) ** 2).sum((-2, -1))
        if torch.any(denominator <= 0):
            raise ValueError("degenerate matched reference variance")
        squares = ((old - ref) ** 2).sum(-1)
        layers.append((squares / (denominator[..., None] / len(pairs)))[0].T)
    return torch.stack(layers, 1).numpy()


@torch.inference_mode()
def construct(model, receiver, pairs, candidate, chunk):
    # Copy the candidate states for each matched run, then compute each gap against the assembled cache.
    # Returns (cache, the logits after the last prompt token). The prompt must end with computed tokens.
    cache = DynamicCache(config=model.config)
    position = 0
    for rows in blocks(pairs):
        destination = int(pairs[rows[0], 1])
        cache = prefill(model, receiver[position:destination], chunk, cache)
        if cache.get_seq_length() != destination:
            raise ValueError("cache length differs from destination position")
        for layer, (keys, values) in enumerate(candidate):
            cache.update(keys[:, :, rows].to(model.device), values[:, :, rows].to(model.device), layer)
        position = destination + len(rows)
    return prefill(model, receiver[position:], chunk, cache, last_logits=True)


def fork(cache):
    """A copy of a cache that a forward pass may extend without touching the original."""
    out = DynamicCache()
    for layer, state in enumerate(cache.layers):
        out.update(state.keys.clone(), state.values.clone(), layer)
    return out


@torch.inference_mode()
def continuation_logits(model, cache, continuation, chunk):
    # Condition on C[0]; the logits after C[:-1] predict C[1:]. C[0] is unscored. The cache is extended.
    if len(continuation) < 2:
        raise ValueError("at least two continuation tokens are needed")
    inputs = torch.as_tensor(continuation[:-1], dtype=torch.long, device=model.device)[None]
    logits = []
    for start in range(0, inputs.shape[1], chunk):
        out = model(inputs[:, start:start + chunk], past_key_values=cache, use_cache=True,
                    logits_to_keep=0)
        cache = out.past_key_values
        logits.append(out.logits[0].float().cpu())
    return torch.cat(logits)


@torch.inference_mode()
def greedy(model, cache, prompt_logits, n_tokens):
    """Greedy decode of n tokens from the bare prompt: the first is the argmax of the logits after the
    prompt's last token, each later one the argmax after feeding the one before. The cache is extended by
    n - 1 tokens."""
    out_ids = [int(prompt_logits.argmax())]
    for _ in range(n_tokens - 1):
        token = torch.as_tensor([[out_ids[-1]]], dtype=torch.long, device=model.device)
        out = model(token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        out_ids.append(int(out.logits[0, -1].argmax()))
    return np.asarray(out_ids, dtype=np.int64)


def compare_logits(fresh, candidate):
    if (fresh.shape != candidate.shape or not torch.isfinite(fresh).all()
            or not torch.isfinite(candidate).all()):
        raise ValueError("invalid continuation logits")
    # Float64 on CPU makes a tiny negative KL from roundoff negligible.
    p = fresh.double().log_softmax(-1)
    q = candidate.double().log_softmax(-1)
    return {"kl": (p.exp() * (p - q)).sum(-1).clamp_min(0).numpy(),
            "top1_agrees": (fresh.argmax(-1) == candidate.argmax(-1)).numpy(),
            "max_logit_error": float((fresh - candidate).abs().max())}
