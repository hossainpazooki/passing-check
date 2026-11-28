"""The cache assembly on a tiny randomly initialised model, CPU only. Proves the construction, not a claim."""
import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from passing_check.cache import (blocks, compare_logits, construct, continuation_logits, fork, greedy,
                                 half_rotate, key_deviation, matched, prefill, random_candidate, relocate,
                                 relocated_candidate, unrelated_candidate, validate_pairs)
from passing_check.stats import make_rng


@pytest.fixture
def model():
    torch.manual_seed(3)
    cfg = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                     max_position_embeddings=81920, attention_dropout=0,
                     rope_parameters={"rope_type": "yarn", "rope_theta": 1000000.0,
                                      "factor": 2.5, "original_max_position_embeddings": 32768})
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval()


def test_pairs_are_validated_and_split_into_runs():
    sender, receiver = np.array([1, 2, 3, 4, 5]), np.array([1, 2, 9, 3, 4, 5])
    pairs = np.array([[0, 0], [1, 1], [2, 3], [3, 4], [4, 5]])
    validate_pairs(sender, receiver, pairs)
    assert [r.tolist() for r in blocks(pairs)] == [[0, 1], [2, 3, 4]]
    for invalid in (np.array([[0, 0], [1, 0]]), np.array([[99, 0]]), np.array([[0, 2]])):
        with pytest.raises(ValueError):
            validate_pairs(sender, receiver, invalid)


def test_relocation_at_long_positions_keeps_the_amplitude(model):
    freq = model.model.rotary_emb.inv_freq
    scale = model.model.rotary_emb.attention_scaling
    content = torch.randn(1, 2, 4, 8)
    source, destination = np.array([32768, 50000, 70000, 80110]), np.array([2, 121, 1234, 19000])

    def direct(positions):
        angles = torch.outer(torch.as_tensor(positions).float(), freq)
        angles = torch.cat((angles, angles), dim=-1)
        return (content * angles.cos() + half_rotate(content) * angles.sin()) * scale
    actual = relocate(direct(source), source, destination, freq)
    torch.testing.assert_close(actual, direct(destination), atol=1e-6, rtol=1e-6)
    assert scale > 1      # a second amplitude multiplication would fail the comparison above


def test_a_full_copy_of_the_readers_own_states_reproduces_the_fresh_pass(model):
    tokens, continuation = np.array([2, 3, 4, 5, 6]), np.array([9, 10, 11, 12])
    pairs = np.column_stack((np.arange(5), np.arange(5)))
    original = prefill(model, tokens, 2)
    states = matched(original, np.arange(5))
    expected = continuation_logits(model, fork(original), continuation, 2)
    with pytest.raises(ValueError, match="must be computed"):
        construct(model, tokens, pairs, states, 2)          # a prompt whose last token is copied
    assembled, at_end = construct(model, tokens, pairs[:4], [(k[:, :, :4], v[:, :, :4]) for k, v in states], 2)
    torch.testing.assert_close(at_end, prefill(model, tokens, 2, last_logits=True)[1], atol=1e-6, rtol=1e-6)
    actual = continuation_logits(model, assembled, continuation, 2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    full = model(torch.tensor([tokens.tolist() + continuation[:-1].tolist()]), logits_to_keep=0).logits[0]
    torch.testing.assert_close(expected, full[len(tokens):], atol=1e-6, rtol=1e-6)
    identical = compare_logits(expected, expected)
    assert not identical["kl"].any() and identical["top1_agrees"].all()
    assert original.get_seq_length() == len(tokens)      # the fork took the extension, not the original


def test_a_gap_is_computed_against_the_reused_states_before_it(model):
    source, receiver = np.array([15, 3, 4, 5, 6]), np.array([2, 3, 9, 5, 7])
    pairs = np.array([[1, 1], [3, 3]])
    freq = model.model.rotary_emb.inv_freq
    candidate = relocated_candidate(matched(prefill(model, source, 2), pairs[:, 0]), pairs, freq)
    assembled, _ = construct(model, receiver, pairs, candidate, 2)
    fresh = prefill(model, receiver, 2)
    assert assembled.get_seq_length() == len(receiver)
    assert not torch.equal(assembled.layers[1].values[:, :, 2], fresh.layers[1].values[:, :, 2])
    different = compare_logits(continuation_logits(model, fresh, np.array([7, 8, 9]), 2),
                               continuation_logits(model, assembled, np.array([7, 8, 9]), 2))
    assert different["kl"].max() > 0


def test_the_random_candidate_reads_the_same_as_the_reused_one_on_the_key_deviation(model):
    source, receiver = np.array([15, 3, 4, 5, 6, 7]), np.array([2, 3, 4, 9, 6, 7])
    pairs = np.array([[1, 1], [2, 2], [4, 4], [5, 5]])
    freq = model.model.rotary_emb.inv_freq
    fresh = matched(prefill(model, receiver, 2), pairs[:, 1])
    reused = relocated_candidate(matched(prefill(model, source, 2), pairs[:, 0]), pairs, freq)
    a, b = random_candidate(fresh, reused, make_rng(11)), random_candidate(fresh, reused, make_rng(11))
    assert all(torch.equal(x[i], y[i]) for x, y in zip(a, b) for i in (0, 1))
    d_reused, d_random = key_deviation(fresh, reused, pairs, freq), key_deviation(fresh, a, pairs, freq)
    assert d_reused.shape == (4, 2, 2) and d_reused.max() > 0
    np.testing.assert_allclose(d_random, d_reused, rtol=1e-4, atol=1e-9)
    assert not key_deviation(fresh, fresh, pairs, freq).any()
    assert not torch.equal(a[0][0], reused[0][0])


def test_greedy_decoding_starts_from_the_bare_prompt_and_follows_the_argmax(model):
    tokens = np.array([2, 3, 4, 5, 6])
    cache, at_end = prefill(model, tokens, 2, last_logits=True)
    out = greedy(model, cache, at_end, 6)
    assert out.shape == (6,) and cache.get_seq_length() == len(tokens) + 5
    ids = tokens.tolist()
    for expected in out:
        nxt = int(model(torch.tensor([ids])).logits[0, -1].argmax())
        assert nxt == expected
        ids.append(nxt)


def test_the_unrelated_token_anchor_removes_each_keys_own_position(model):
    receiver = np.array([2, 3, 4, 9, 6, 7, 11, 12])
    pairs = np.column_stack((np.arange(6), np.arange(1, 7)))        # positions 1..6 of the receiver
    freq = model.model.rotary_emb.inv_freq
    fresh = matched(prefill(model, receiver, 4), pairs[:, 1])
    actual = key_deviation(fresh, unrelated_candidate(fresh, pairs, freq), pairs, freq)
    # The same quantity from keys with position removed first: content against the content half a set away.
    zero = np.zeros(len(pairs), dtype=np.int64)
    for layer, (keys, _) in enumerate(fresh):
        content = relocate(keys, pairs[:, 1], zero, freq).double()
        other = content.roll(len(pairs) // 2, dims=-2)
        spread = ((content - content.mean(-2, keepdim=True)) ** 2).sum((-2, -1)) / len(pairs)
        expected = (((other - content) ** 2).sum(-1) / spread[..., None])[0].T
        np.testing.assert_allclose(actual[:, layer], expected.numpy(), rtol=1e-4, atol=1e-8)
    # Rolling the keys without moving them leaves a rotation by the difference of positions: a different number.
    stray = key_deviation(fresh, [(k.roll(len(pairs) // 2, dims=-2), v) for k, v in fresh], pairs, freq)
    assert not np.allclose(stray, actual, rtol=1e-2)


def test_reused_first_layer_keys_equal_the_fresh_ones_once_position_is_removed(model):
    source, receiver = np.array([15, 3, 4, 5, 6, 7]), np.array([2, 9, 3, 4, 6, 7, 8])
    pairs = np.array([[1, 2], [2, 3], [4, 4], [5, 5]])
    freq = model.model.rotary_emb.inv_freq
    fresh = matched(prefill(model, receiver, 2), pairs[:, 1])
    reused = relocated_candidate(matched(prefill(model, source, 2), pairs[:, 0]), pairs, freq)
    deviation = key_deviation(fresh, reused, pairs, freq)
    assert deviation[:, 0].max() < 1e-9 and deviation[:, 1].max() > 1e-6
    wrong = relocated_candidate(matched(prefill(model, source, 2), pairs[:, 0]), pairs, freq * 1.01)
    assert key_deviation(fresh, wrong, pairs, freq)[:, 0].max() > 1e-6     # a wrong schedule shows at layer 0
