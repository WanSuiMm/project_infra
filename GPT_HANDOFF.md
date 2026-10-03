# Incremental review: packed v02 preparation

- Review base: `29d58b69f4310880cecd45de879addf81755f1e6` (published v01 evidence).
- Evidence/code head: `0d8cf814d2f1468134dab046a6e3a112c0621ff0`.
- A later metadata-only commit adds this handoff; use the stable head above for the implementation and toy evidence.

## Changed and unchanged claims

**Unchanged:** v01 source, protocol and canonical result bytes; padded cross-configuration `PROTOTYPE_NO_GO`; hybrid/128 `NUMERICAL_UNQUALIFIED`; no RL, optimizer, full-model backward or end-to-end speedup evidence.

**Added:** real packed edge processing with full dense dW; independent CPU FP32 and CUDA BF16 toy checks PASS; a separately frozen Gate A replay plan. The real-model v02 numerical and performance endpoints remain **NOT_RUN**.

**Clarified:** the original bounded512/128 trace has 3,464 valid versus 65,536 padded slots. This motivates removing a measured quantity of redundant work, while its causal contribution to runtime remains unresolved. A numerical gate pass is empirical tolerance-based agreement, and the 2.129x result is a narrow descriptive positive.

## Minimal reading order

1. [PACKED_V02.md](PACKED_V02.md): operator, costs, controlled comparison and stopping rule.
2. [packed_head.py](scripts/packed_head.py), then [check_packed.py](scripts/check_packed.py).
3. [Gate A protocol](protocols/qwen15b_v02_gateA.json), then [qualify_packed.py](scripts/qualify_packed.py).
4. [Toy checks](evidence/packed_v02_toy/checks.json). Run `python scripts/verify_public.py` for source/evidence integrity.

Read [RESULTS.md](RESULTS.md) only for the revised interpretation. The original large trace tensors and logs are unnecessary for this code review and remain excluded from Git. The v01 evidence files need rereading only when checking a disputed original number.

## Reviewer questions and next decision

1. Does edge packing preserve support/action mapping and the complete dense dW contract, including tied-weight scope?
2. Is conversion-inclusive wall time a fair primary metric, with prepacked time and the v01 adapter's unused entropy-placeholder cost clearly labeled?
3. Does the planned bounded512 endpoint isolate a useful implementation decision without promoting a marginal shared-GPU >1.0x result into a general claim?

The next empirical decision is the fixed 1.5B Gate A replay. Failure stops this packed PyTorch recipe. Optimizer integration and hybrid repair require separate questions and protocols.
