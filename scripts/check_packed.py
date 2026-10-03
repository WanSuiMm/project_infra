"""Small independent dense-autograd check; this is not a real-model gate."""
import argparse
import json
from pathlib import Path

import torch

from packed_head import evaluate_packed, pack_support, validate_support


def error(candidate, reference):
    delta = candidate.float() - reference.float()
    return {"max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / reference.float().norm().clamp_min(1e-12)).item()}


def check(device="cpu", dtype=torch.float32):
    generator = torch.Generator(device=device).manual_seed(17)
    h = torch.randn(5, 16, device=device, generator=generator, dtype=dtype, requires_grad=True)
    w = torch.randn(43, 16, device=device, generator=generator, dtype=dtype, requires_grad=True)
    # Holes, vocabulary zero, cross-row repeated IDs and a singleton support.
    ids = torch.tensor([[0, 2, -1, 7, -1, 9], [3, -1, 5, 11, -1, -1],
                        [0, 4, 8, 13, 14, -1], [-1, 12, -1, -1, -1, -1],
                        [2, 7, 15, -1, 20, -1]], device=device)
    positions = torch.tensor([3, 2, 0, 1, 4], device=device)
    coeff = torch.randn(5, device=device, generator=generator)
    validate_support(ids, positions, len(w))
    packed = pack_support(ids, positions)
    assert packed.active_ids.numel() == 17
    assert packed.row_offsets.tolist() == [0, 4, 7, 12, 13, 17]
    assert packed.action_edges.tolist() == [2, 5, 7, 12, 16]
    # The reference computes the entire vocabulary with native autograd.
    logits = (h @ w.T).float() / 0.7
    mask = torch.zeros_like(logits, dtype=torch.int32)
    mask.scatter_add_(1, ids.clamp_min(0), (ids >= 0).to(torch.int32))
    target = ids.gather(1, positions[:, None])
    lp = logits.masked_fill(mask == 0, -torch.inf).log_softmax(-1).gather(1, target).squeeze(1)
    dh, dw = torch.autograd.grad(lp, (h, w), coeff)
    result = evaluate_packed(h, w, packed, 0.7, coeff)
    report = {name: error(a, b) for name, a, b in zip(("logprob", "dh", "dw"), result, (lp, dh, dw))}
    for name, value in report.items():
        if dtype == torch.float32:
            assert value["max_abs"] < 5e-5, (name, value)
        elif name == "logprob":
            assert value["max_abs"] <= 0.05, (name, value)
        else:
            assert value["relative_l2"] <= 0.02, (name, value)
    assert result[2].shape == w.shape and result[2].dtype == w.dtype
    assert torch.count_nonzero(result[2][mask.sum(0) == 0]) == 0
    invalid = []
    empty = ids.clone(); empty[0] = -1
    invalid.append((empty, positions, len(w)))
    duplicate = ids.clone(); duplicate[0, 1] = 0
    invalid.append((duplicate, positions, len(w)))
    bad_action = positions.clone(); bad_action[0] = 2
    invalid.append((ids, bad_action, len(w)))
    outside = ids.clone(); outside[0, 0] = len(w)
    invalid.append((outside, positions, len(w)))
    for bad_ids, bad_positions, vocab in invalid:
        try:
            validate_support(bad_ids, bad_positions, vocab)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("Invalid support accepted")
    return {"status": "PASS", "scope": "toy numerical/input-contract check only",
            "device": device, "dtype": str(dtype), "torch": torch.__version__,
            "active_pairs": 17, "padded_pairs": ids.numel(), "errors": report,
            "invalid_cases_rejected": len(invalid), "dense_dw_shape": list(result[2].shape)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = check(args.device, getattr(torch, args.dtype))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
