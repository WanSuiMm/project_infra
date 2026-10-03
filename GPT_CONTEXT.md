# Review context

## Task and claim boundary

The upstream sampling support is fixed before the training-head computation. For each position t, only logits in S_t enter its restricted softmax normalizer. This pilot tests an equivalent operator implementation, not a new sampling policy, an approximation to full-vocabulary cross entropy, or a claim of novelty for sampling-mask replay.

For z_j = h W_j^T / temperature:

`logprob = z_action - logsumexp(z_support)`.

The output-head gradient outside the fixed support is zero for this loss term. Tied embedding contributions, optimizer state and weight decay are outside this operator test; zero head-branch gradient does not imply no parameter update.

## Formal status

- Execution: COMPLETE, exit code 0.
- FP32 small algebra check: PASS for six modes.
- Support-only real-model numerical gate: PASS in all four arms.
- Support-only frozen performance gate: PROTOTYPE_NO_GO.
- Hybrid 32-position arms: numerical PASS, measured slowdown.
- Hybrid 128-position arms: NUMERICAL_UNQUALIFIED; not timed.
- Chunked 128-position references: NUMERICAL_UNQUALIFIED.
- Packed v02: IMPLEMENTED; toy CPU FP32/CUDA BF16 checks PASS; 1.5B replay and Gate A NOT_RUN.
- Full-model backward, optimizer updates, RL quality, production trace validation and end-to-end throughput: NOT_RUN.
- Independent scientific units: eight authored diagnostic prompt trajectories; timing repetitions are technical repeats. No statistical generalization claim.

## Variants and code map

The v01 implementation symbols below are in the unchanged `scripts/qualify.py`.

| Symbol/mode | Meaning |
|---|---|
| `collect` | Offline local model, eight prompts, two fixed sampling settings, 32 generation steps; excludes post-EOS positions |
| `Head.forward`, `dense_lp` / `chunked_lp` | Full-vocabulary projection, selected-logit normalization, no full entropy diagnostic |
| `Head.forward`, `dense` / `chunked` | Full-vocabulary projection plus exact full-vocabulary entropy diagnostic |
| `Head.forward`, `sparse` | Gather/BMM only at padded support positions; no full entropy output |
| `Head.forward`, `hybrid` | Chunked full forward/entropy, support-sparse backward |
| `Head.backward` | Full dense gradient output for every mode; sparse modes include gather temporaries and FP32 index-add accumulation |
| `algebra_check` | Separate small FP32 dense-mask autograd reference for logprob, dH and dW |
| `benchmark` | CUDA-event F+B timing, two warmups, five shuffled rounds of three iterations; measures peak allocated memory |

Support-only is compared with the faster numerically qualified `dense_lp`/`chunked_lp` baseline. Hybrid is compared with qualified `dense`/`chunked`. Full entropy is never silently removed from a matched task. At 128 positions the chunked baselines failed the gate, leaving the dense baseline.

Numerical tolerance is fixed in the protocol (max logprob absolute error 0.05; relative gradient L2 error 0.02). Passing is not a general proof or a bitwise equivalence guarantee.

Packed v02 uses `scripts/packed_head.py`: `pack_support` creates valid edge IDs, row IDs, row offsets and action-edge positions; `_PackedSupportHead` computes selected BMM logits, segmented normalization and FP32 accumulation into full dense dW. `scripts/check_packed.py` compares against native dense autograd, including interspersed padding, repeated IDs across rows and singleton support. `scripts/qualify_packed.py` replays hash-bound v01 inputs and remeasures qualified dense, padded and packed variants. Conversion-inclusive synchronized wall time is primary; prepacked and CUDA-event time are secondary. Details: `PACKED_V02.md`.

## Evidence route

Read `RESULTS.md`, then `evidence/qwen15b_v01/summary.json`. `provenance.json` binds the exported runner, protocol and original result summary by SHA256 and records non-identifying software/hardware metadata. `sanity.json` is the algebra check from the formal run. `model.json` records the verified model revision and authored prompts. Do not start with generated trace tensors: they are not included.

## Questions for the reviewer

1. Are the baseline and candidate tasks matched, especially entropy and gradient-buffer costs?
2. Is the hybrid forward/backward numerical inconsistency adequately isolated before any performance claim?
3. Does the conversion-inclusive packed implementation remove enough work to pass its separately specified Gate A, while preserving full dense dW and the v01 verdict?
4. What stronger optimized baseline is needed before extrapolating the narrow small-support positive result?

The initial v01 evidence remains frozen. An implementation follow-up and toy checks were added after review; the real-model rerun is pending. Use `GPT_HANDOFF.md` for the incremental reading route.
