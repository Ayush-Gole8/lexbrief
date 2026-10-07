"""M3a: document-level BiLSTM-CRF tagger over cached sentence embeddings.

Per document: sentence embeddings ``[n, d]`` (+ optional relative position ``idx/n``) ->
dropout -> BiLSTM -> Linear(2*hidden, 13) emissions -> ``torchcrf.CRF``. With
``use_crf=False`` the emissions are trained with cross-entropy (M3a-noCRF softmax ablation).

:meth:`HierarchicalTagger.marginals` implements the forward-backward algorithm (pytorch-crf
only offers Viterbi), giving per-sentence role probabilities.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchcrf import CRF

from lexbrief.labels import FINE_LABELS

logger = logging.getLogger(__name__)


class HierarchicalTagger(nn.Module):
    """BiLSTM(-CRF) sequence tagger over sentence vectors."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden: int = 256,
        num_labels: int = len(FINE_LABELS),
        dropout: float = 0.3,
        num_layers: int = 1,
        use_crf: bool = True,
        use_position: bool = True,
    ) -> None:
        super().__init__()
        self.use_crf = use_crf
        self.use_position = use_position
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_dim + int(use_position),
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.emit = nn.Linear(2 * hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True) if use_crf else None

    # ------------------------------------------------------------------ core
    def emissions(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Emission scores ``[B, T, L]`` for padded embeddings ``[B, T, D]`` and mask ``[B, T]``."""
        expected = self.lstm.input_size - int(self.use_position)
        if emb.size(-1) != expected:
            # nn.LSTM does not validate the feature size of PackedSequence inputs and silently
            # reads out of bounds, giving non-deterministic garbage; fail loudly instead.
            raise ValueError(f"Expected sentence embeddings of size {expected}, got {emb.size(-1)}")
        x = emb.float()
        lengths = mask.sum(dim=1)
        if self.use_position:
            pos = torch.arange(x.size(1), device=x.device, dtype=x.dtype).unsqueeze(0)
            rel = pos / lengths.clamp(min=1).unsqueeze(1).to(x.dtype)
            x = torch.cat([x, (rel * mask).unsqueeze(-1)], dim=-1)
        x = self.dropout(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.emit(self.dropout(out))

    def loss(self, emb: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood (CRF) or token cross-entropy (noCRF).

        ``labels`` uses -100 for unknown tags; such positions are ignored by CE and, for the
        CRF, treated as tag 0 (training data is fully labelled, so this is a safety net).
        """
        em = self.emissions(emb, mask)
        if self.crf is None:
            tgt = labels.masked_fill(~mask, -100)
            return nn.functional.cross_entropy(
                em.reshape(-1, self.num_labels), tgt.reshape(-1), ignore_index=-100
            )
        tags = labels.clamp(min=0)
        c = self.crf
        nll = crf_log_partition(em, mask, c.start_transitions, c.transitions, c.end_transitions)
        nll = nll - crf_path_score(em, tags, mask, c.start_transitions, c.transitions,
                                   c.end_transitions)
        return nll.mean()  # same value as -torchcrf.CRF(..., reduction="mean"), much faster

    @torch.no_grad()
    def decode(self, emb: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi (CRF) or argmax (noCRF) tag sequences, one list per document."""
        em = self.emissions(emb, mask)
        if self.crf is not None:
            return self.crf.decode(em, mask=mask)
        lengths = mask.sum(dim=1).tolist()
        return [row[:n].tolist() for row, n in zip(em.argmax(-1), lengths, strict=True)]

    @torch.no_grad()
    def marginals(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-sentence label probabilities ``[B, T, L]`` (zeros at padding)."""
        em = self.emissions(emb, mask)
        if self.crf is None:
            return torch.softmax(em, dim=-1) * mask.unsqueeze(-1)
        return crf_marginals(
            em, mask, self.crf.start_transitions, self.crf.transitions, self.crf.end_transitions
        )


_NEG = -1e4  # log(0) stand-in that keeps gradients finite


def _log_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Log-semiring matrix product: ``out[..., i, j] = logsumexp_k a[..., i, k] + b[..., k, j]``."""
    return torch.logsumexp(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2)


def crf_log_partition(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start: torch.Tensor,
    trans: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Log partition function ``log Z`` per sequence ``[B]`` via a parallel (tree) scan.

    The forward recursion ``alpha_t = alpha_{t-1} (x) M_t`` with ``M_t[i, j] = trans[i, j] +
    em_t[j]`` is associative in the log semiring, so the T-1 step matrices are multiplied in a
    balanced tree (O(log T) sequential depth) instead of a Python loop over T. Padded steps
    use the log-identity matrix. Numerically equal to torchcrf's ``_compute_normalizer``.
    """
    em = emissions.float()
    bsz, seq_len, n = em.shape
    alpha0 = start.float() + em[:, 0]  # [B, L]
    eye = torch.full((n, n), _NEG, device=em.device)
    eye.fill_diagonal_(0.0)
    if seq_len > 1:
        steps = trans.float().view(1, 1, n, n) + em[:, 1:].unsqueeze(2)  # [B, T-1, L, L]
        steps = torch.where(mask[:, 1:].bool().view(bsz, -1, 1, 1), steps, eye)
        while steps.size(1) > 1:
            if steps.size(1) % 2:
                steps = torch.cat([steps, eye.expand(bsz, 1, n, n)], dim=1)
            steps = _log_matmul(steps[:, 0::2], steps[:, 1::2])
        alpha = torch.logsumexp(alpha0.unsqueeze(2) + steps[:, 0], dim=1)  # [B, L]
    else:
        alpha = alpha0
    return torch.logsumexp(alpha + end.float(), dim=1)


def crf_path_score(
    emissions: torch.Tensor,
    tags: torch.Tensor,
    mask: torch.Tensor,
    start: torch.Tensor,
    trans: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Unnormalised score of the given tag paths ``[B]`` (vectorised over time)."""
    em = emissions.float()
    m = mask.float()
    emit = em.gather(2, tags.unsqueeze(-1)).squeeze(-1)  # [B, T]
    score = start.float()[tags[:, 0]] + (emit * m).sum(1)
    if tags.size(1) > 1:
        tr = trans.float()[tags[:, :-1], tags[:, 1:]]  # [B, T-1]
        score = score + (tr * m[:, 1:]).sum(1)
    last = mask.long().sum(1) - 1
    return score + end.float()[tags.gather(1, last.unsqueeze(1)).squeeze(1)]


def crf_marginals(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start: torch.Tensor,
    trans: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Forward-backward posterior marginals for a linear-chain CRF.

    Uses pytorch-crf conventions: ``trans[i, j]`` scores moving from tag i to tag j; ``mask`` is a
    left-aligned boolean ``[B, T]`` with ``mask[:, 0]`` all True.

    Returns:
        ``[B, T, L]`` probabilities; rows at valid positions sum to 1, padding rows are 0.
    """
    em = emissions.float()
    start, trans, end = start.float(), trans.float(), end.float()
    bsz, seq_len, _ = em.shape
    mask = mask.bool()
    last = mask.sum(dim=1) - 1  # [B]

    # forward: alpha[t, j] = log sum over paths ending in j at t
    alphas = []
    alpha = start + em[:, 0]
    alphas.append(alpha)
    for t in range(1, seq_len):
        nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + em[:, t]
        alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        alphas.append(alpha)
    alpha_all = torch.stack(alphas, dim=1)  # [B, T, L]
    log_z = torch.logsumexp(alpha + end, dim=1)  # alpha is frozen after each sequence end

    # backward: beta[t, i] = log sum over continuations from i at t (incl. end transition)
    betas: list[torch.Tensor] = [torch.zeros_like(alpha)] * seq_len
    beta = end.expand(bsz, -1).clone()
    t_idx = torch.arange(seq_len, device=em.device)
    for t in range(seq_len - 1, -1, -1):
        if t < seq_len - 1:
            step = torch.logsumexp(
                trans.unsqueeze(0) + (em[:, t + 1] + beta).unsqueeze(1), dim=2
            )
            # positions before the last valid one take the recursion; the last takes `end`
            beta = torch.where((t_idx[t] < last).unsqueeze(1), step, end.expand(bsz, -1))
        betas[t] = beta
    beta_all = torch.stack(betas, dim=1)

    post = torch.exp(alpha_all + beta_all - log_z.view(bsz, 1, 1))
    return post * mask.unsqueeze(-1)
