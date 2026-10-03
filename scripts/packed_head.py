"""Packed ragged-support logprob head with dense weight gradients.

This is a Python/PyTorch prototype, not a fused kernel. It avoids padded
``[N, K, D]`` gathers by operating on the ``E`` valid support edges, but it
still constructs and returns the full dense ``[V, D]`` weight gradient.
There is no entropy output: the objective is the support-restricted action
logprob used by the sparse lane in :mod:`qualify`.

Shape contract
--------------
``h`` is ``[N, D]`` and ``w`` is ``[V, D]``. The source ``ids`` passed to
``pack_support`` is ``[N, K]``; every negative id is padding, and valid ids
may be interspersed with padding. ``positions`` is ``[N]`` and indexes the
chosen action in the original padded ``ids`` row. ``PackedSupport`` stores
the resulting ``E`` valid edges and one chosen edge per row. For evaluation,
all tensors must be on one device, and ``h`` and ``w`` must have the same
floating dtype. Run :func:`validate_support` once before a timed region.
"""

from numbers import Integral, Real
from typing import NamedTuple

import torch


_SIGNED_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
_INDEX_DTYPES = _SIGNED_INTEGER_DTYPES + (torch.uint8,)


class PackedSupport(NamedTuple):
    """Packed edge arrays for an ``N``-row support set.

    ``active_ids[e]`` is the vocabulary row for edge ``e``; ``row_ids[e]``
    identifies its hidden-state row. ``row_offsets`` has length ``N + 1``
    and delimits each row's contiguous edge range. ``action_edges`` has
    length ``N`` and points to the selected action edge for each row.
    All four tensors are one-dimensional int64 tensors on the input ``ids``
    device.
    """

    active_ids: torch.Tensor
    row_ids: torch.Tensor
    row_offsets: torch.Tensor
    action_edges: torch.Tensor


def _check_source_shapes(ids: torch.Tensor, positions: torch.Tensor) -> None:
    if not isinstance(ids, torch.Tensor) or not isinstance(positions, torch.Tensor):
        raise TypeError("ids and positions must be torch.Tensor objects")
    if ids.ndim != 2:
        raise ValueError("ids must have shape [N, K]")
    if positions.ndim != 1 or positions.shape[0] != ids.shape[0]:
        raise ValueError("positions must have shape [N], matching ids")
    if ids.shape[0] == 0 or ids.shape[1] == 0:
        raise ValueError("ids must have at least one row and one padded column")
    if ids.dtype not in _SIGNED_INTEGER_DTYPES:
        raise TypeError("ids must use a signed integer dtype; negative values are padding")
    if positions.dtype not in _INDEX_DTYPES:
        raise TypeError("positions must use an integer dtype")
    if ids.device != positions.device:
        raise ValueError("ids and positions must be on the same device")


def validate_support(
    ids: torch.Tensor,
    positions: torch.Tensor,
    vocab_size: int,
) -> None:
    """Validate a padded support and its chosen action outside timed code.

    This intentionally performs device-to-host checks. Call it once when
    preparing a trace, then use ``pack_support(..., validate=False)`` inside
    any timed conversion lane. Any negative id is accepted as padding. Valid
    ids must be in ``[0, vocab_size)``, unique within each row, and every row
    must contain at least one valid id. Each action position must be within
    its padded row and point to a valid id.
    """
    _check_source_shapes(ids, positions)
    if not isinstance(vocab_size, Integral) or isinstance(vocab_size, bool):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    valid = ids >= 0
    empty_rows = ~valid.any(dim=1)
    bad_positions = (positions < 0) | (positions >= ids.shape[1])
    safe_positions = positions.to(torch.long).clamp(0, ids.shape[1] - 1)
    chosen_ids = ids.gather(1, safe_positions[:, None]).squeeze(1)
    bad_actions = bad_positions | (chosen_ids < 0)
    bad_vocab_ids = valid & (ids >= vocab_size)

    sorted_ids = ids.sort(dim=1).values
    duplicate_ids = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0)
    invalid = (
        empty_rows.any()
        | bad_actions.any()
        | bad_vocab_ids.any()
        | duplicate_ids.any()
    )
    if bool(invalid.item()):
        if bool(empty_rows.any().item()):
            raise ValueError("every row must have a nonempty support")
        if bool(bad_positions.any().item()):
            raise ValueError("an action position is outside its padded ids row")
        if bool((chosen_ids < 0).any().item()):
            raise ValueError("an action position points to padding")
        if bool(bad_vocab_ids.any().item()):
            raise ValueError("a valid support id is outside the vocabulary")
        raise ValueError("valid support ids must be unique within each row")


def pack_support(
    ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    validate: bool = False,
    vocab_size: int | None = None,
) -> PackedSupport:
    """Convert padded ``ids[N, K]`` and action positions into packed edges.

    Negative ids are omitted wherever they occur, including interspersed
    padding. Edge order within each row follows the original padded order.
    By default validation uses only shape metadata. Boolean indexing has a
    dynamic output size and can synchronize CUDA internally; the conversion
    lane includes that cost. Call :func:`validate_support` separately before
    timing. Set ``validate=True`` to validate here and supply ``vocab_size``.
    """
    _check_source_shapes(ids, positions)
    if validate:
        if vocab_size is None:
            raise ValueError("vocab_size is required when validate=True")
        validate_support(ids, positions, vocab_size)

    valid = ids >= 0
    lengths = valid.sum(dim=1, dtype=torch.int64)
    zero = torch.zeros((1,), device=ids.device, dtype=torch.int64)
    row_offsets = torch.cat((zero, lengths.cumsum(dim=0)))
    active_ids = ids[valid].to(torch.long)
    row_grid = torch.arange(ids.shape[0], device=ids.device, dtype=torch.long)
    row_ids = row_grid[:, None].expand_as(ids)[valid]

    positions_long = positions.to(torch.long)
    valid_prefix = valid.to(torch.int64).cumsum(dim=1)
    action_in_row = valid_prefix.gather(1, positions_long[:, None]).squeeze(1) - 1
    action_edges = row_offsets[:-1] + action_in_row
    return PackedSupport(active_ids, row_ids, row_offsets, action_edges)


def _edge_logits(
    h: torch.Tensor,
    w: torch.Tensor,
    active_ids: torch.Tensor,
    row_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute only the E selected dot products with qualify.py BMM precision."""
    selected_w = w.index_select(0, active_ids)
    selected_h = h.index_select(0, row_ids)
    # This mirrors qualify.selected_logits: the BMM result is rounded to the
    # input dtype before conversion to FP32 for temperature and logsumexp.
    dots = torch.bmm(selected_w.unsqueeze(1), selected_h.unsqueeze(2)).reshape(-1)
    return dots.float() / temperature


def _segment_logsumexp(
    edge_logits: torch.Tensor,
    row_ids: torch.Tensor,
    row_offsets: torch.Tensor,
) -> torch.Tensor:
    lengths = row_offsets[1:] - row_offsets[:-1]
    row_max = torch.segment_reduce(edge_logits, reduce="max", lengths=lengths, axis=0)
    shifted = edge_logits - row_max.index_select(0, row_ids)
    row_sum = torch.segment_reduce(shifted.exp(), reduce="sum", lengths=lengths, axis=0)
    return row_max + row_sum.log()


class _PackedSupportHead(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        h: torch.Tensor,
        w: torch.Tensor,
        active_ids: torch.Tensor,
        row_ids: torch.Tensor,
        row_offsets: torch.Tensor,
        action_edges: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        edge_logits = _edge_logits(h, w, active_ids, row_ids, temperature)
        row_logz = _segment_logsumexp(edge_logits, row_ids, row_offsets)
        logprob = edge_logits.index_select(0, action_edges) - row_logz
        ctx.save_for_backward(h, w, active_ids, row_ids, action_edges, row_logz)
        ctx.temperature = temperature
        ctx.set_materialize_grads(False)
        return logprob

    @staticmethod
    def backward(ctx, grad_logprob: torch.Tensor | None):
        if grad_logprob is None:
            return None, None, None, None, None, None, None

        h, w, active_ids, row_ids, action_edges, row_logz = ctx.saved_tensors
        temperature = ctx.temperature
        edge_logits = _edge_logits(h, w, active_ids, row_ids, temperature)
        edge_grad = -(
            edge_logits - row_logz.index_select(0, row_ids)
        ).exp() * grad_logprob.index_select(0, row_ids)
        edge_grad = edge_grad.index_add(
            0, action_edges, grad_logprob
        )
        # Match qualify.Head.backward: cast the temperature-scaled coefficient
        # to h.dtype before using it in the FP32 gradient accumulations.
        edge_grad = (edge_grad / temperature).to(h.dtype).float()

        selected_w = w.index_select(0, active_ids)
        selected_h = h.index_select(0, row_ids)
        dh_fp32 = torch.zeros(h.shape, device=h.device, dtype=torch.float32)
        dh_fp32.index_add_(0, row_ids, edge_grad[:, None] * selected_w.float())

        # The operator preserves qualify.py's full dense dW contract. This is
        # intentionally not a sparse optimizer or a sparse gradient return.
        dw_fp32 = torch.zeros(w.shape, device=w.device, dtype=torch.float32)
        dw_fp32.index_add_(0, active_ids, edge_grad[:, None] * selected_h.float())
        return dh_fp32.to(h.dtype), dw_fp32.to(w.dtype), None, None, None, None, None


def _check_evaluate_shapes(
    h: torch.Tensor,
    w: torch.Tensor,
    packed: PackedSupport,
    temperature: Real,
    coeff: torch.Tensor,
) -> None:
    if h.ndim != 2 or w.ndim != 2 or h.shape[1] != w.shape[1]:
        raise ValueError("h and w must have shapes [N, D] and [V, D]")
    n = h.shape[0]
    if n == 0 or h.shape[1] == 0 or w.shape[0] == 0:
        raise ValueError("h and w dimensions must be nonempty")
    if not isinstance(packed, PackedSupport):
        raise TypeError("packed must be a PackedSupport returned by pack_support")
    if packed.active_ids.ndim != 1 or packed.active_ids.numel() == 0:
        raise ValueError("packed support must contain at least one edge")
    if packed.row_ids.shape != packed.active_ids.shape:
        raise ValueError("active_ids and row_ids must have the same shape")
    if packed.row_offsets.shape != (n + 1,) or packed.action_edges.shape != (n,):
        raise ValueError("packed offsets/actions do not match the hidden-state row count")
    packed_tensors = (
        packed.active_ids,
        packed.row_ids,
        packed.row_offsets,
        packed.action_edges,
    )
    if any(t.dtype != torch.int64 for t in packed_tensors):
        raise TypeError("all PackedSupport tensors must use int64")
    if any(t.device != h.device for t in packed_tensors) or w.device != h.device:
        raise ValueError("h, w, and PackedSupport tensors must be on one device")
    if h.dtype != w.dtype or not h.is_floating_point():
        raise TypeError("h and w must have the same floating dtype")
    if coeff.shape != (n,) or coeff.device != h.device or not coeff.is_floating_point():
        raise ValueError("coeff must be a floating tensor of shape [N] on h's device")
    if not isinstance(temperature, Real) or isinstance(temperature, bool):
        raise TypeError("temperature must be a positive Python real scalar")
    if temperature <= 0:
        raise ValueError("temperature must be positive")


def evaluate_packed(
    h: torch.Tensor,
    w: torch.Tensor,
    packed: PackedSupport,
    temperature: float,
    coeff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return support logprobs and their hidden/weight gradients.

    ``lp`` is FP32 with shape ``[N]``. ``dh`` and ``dw`` match the input
    dtypes and have shapes ``[N, D]`` and full dense ``[V, D]``. Gradient
    computation uses custom autograd with FP32 accumulation. The caller must
    validate source IDs and positions once during setup; this timed function
    performs only shape/metadata checks and no tensor ``item()`` reads.
    """
    _check_evaluate_shapes(h, w, packed, temperature, coeff)
    x = h.detach().requires_grad_(True)
    weight = w.detach().requires_grad_(True)
    lp = _PackedSupportHead.apply(
        x,
        weight,
        packed.active_ids,
        packed.row_ids,
        packed.row_offsets,
        packed.action_edges,
        float(temperature),
    )
    dh, dw = torch.autograd.grad(lp, (x, weight), coeff)
    return lp.detach(), dh, dw
