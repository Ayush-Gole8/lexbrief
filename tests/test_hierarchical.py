"""Tests for the M3a BiLSTM-CRF tagger and forward-backward marginals."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import torch

from lexbrief.models.hierarchical import HierarchicalTagger, crf_marginals


def _batch(lengths: list[int], dim: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    t = max(lengths)
    emb = torch.randn(len(lengths), t, dim, generator=g)
    mask = torch.zeros(len(lengths), t, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
        emb[i, n:] = 0
    return emb, mask


@pytest.mark.parametrize("use_crf", [True, False])
@pytest.mark.parametrize("use_position", [True, False])
def test_marginals_sum_to_one_and_zero_padding(use_crf: bool, use_position: bool) -> None:
    torch.manual_seed(0)
    model = HierarchicalTagger(
        input_dim=8, hidden=6, num_labels=5, use_crf=use_crf, use_position=use_position
    ).eval()
    emb, mask = _batch([7, 3, 1])
    m = model.marginals(emb, mask)
    assert m.shape == (3, 7, 5)
    sums = m.sum(-1)
    assert torch.allclose(sums[mask], torch.ones(int(mask.sum())), atol=1e-5)
    assert torch.all(sums[~mask] == 0)
    assert torch.all(m >= 0)


def test_crf_marginals_match_brute_force() -> None:
    torch.manual_seed(1)
    n_tags, lengths = 3, [4, 2]
    em = torch.randn(len(lengths), max(lengths), n_tags)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    start, end = torch.randn(n_tags), torch.randn(n_tags)
    trans = torch.randn(n_tags, n_tags)
    post = crf_marginals(em, mask, start, trans, end)
    for b, n in enumerate(lengths):
        expected = torch.zeros(n, n_tags)
        total = torch.tensor(0.0)
        for path in itertools.product(range(n_tags), repeat=n):
            s = start[path[0]] + end[path[-1]] + sum(em[b, t, path[t]] for t in range(n))
            s = s + sum(trans[path[t - 1], path[t]] for t in range(1, n))
            w = torch.exp(s)
            total = total + w
            for t in range(n):
                expected[t, path[t]] += w
        expected /= total
        assert torch.allclose(post[b, :n], expected, atol=1e-5)


def test_fast_crf_loss_matches_torchcrf() -> None:
    torch.manual_seed(3)
    model = HierarchicalTagger(input_dim=8, hidden=6, num_labels=5, dropout=0.0).eval()
    emb, mask = _batch([9, 4, 1, 6])
    labels = torch.randint(0, 5, mask.shape).masked_fill(~mask, -100)
    ours = model.loss(emb, mask, labels)
    em = model.emissions(emb, mask)
    ref = -model.crf(em, labels.clamp(min=0), mask=mask, reduction="mean")
    assert torch.allclose(ours, ref, atol=1e-4)


def test_loss_backward_and_decode_lengths() -> None:
    for use_crf in (True, False):
        model = HierarchicalTagger(input_dim=8, hidden=6, num_labels=5, use_crf=use_crf)
        emb, mask = _batch([5, 2])
        labels = torch.randint(0, 5, mask.shape).masked_fill(~mask, -100)
        loss = model.loss(emb, mask, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.emit.weight.grad is not None
        paths = model.eval().decode(emb, mask)
        assert [len(p) for p in paths] == [5, 2]
        assert all(0 <= t < 5 for p in paths for t in p)


def test_crf_viterbi_agrees_with_marginal_argmax_when_peaked() -> None:
    model = HierarchicalTagger(input_dim=4, hidden=4, num_labels=3, dropout=0.0).eval()
    with torch.no_grad():
        model.crf.transitions.zero_()
        model.crf.start_transitions.zero_()
        model.crf.end_transitions.zero_()
    emb, mask = _batch([6], dim=4)
    paths = model.decode(emb, mask)
    argmax = model.marginals(emb, mask)[0].argmax(-1).tolist()
    assert paths[0] == argmax  # with zero transitions both reduce to per-step argmax


def test_emissions_reject_wrong_embedding_size() -> None:
    model = HierarchicalTagger(input_dim=4, hidden=4, num_labels=3)
    emb, mask = _batch([3], dim=8)
    with pytest.raises(ValueError, match="size 4, got 8"):
        model.emissions(emb, mask)


def test_emissions_deterministic_in_eval() -> None:
    model = HierarchicalTagger(input_dim=8, hidden=6, num_labels=5).eval()
    emb, mask = _batch([6, 2])
    with torch.no_grad():
        assert torch.equal(model.emissions(emb, mask), model.emissions(emb, mask))


def test_config_base_inheritance(tmp_path: Path) -> None:
    from lexbrief.config import load_config

    (tmp_path / "a.yaml").write_text("seed: 1\nmodel:\n  use_crf: true\n  name: a\n", "utf-8")
    (tmp_path / "b.yaml").write_text("base: a.yaml\nmodel:\n  use_crf: false\n", "utf-8")
    cfg = load_config(tmp_path / "b.yaml", ["model.name=b"])
    assert cfg.seed == 1 and cfg.model.use_crf is False and cfg.model.name == "b"
