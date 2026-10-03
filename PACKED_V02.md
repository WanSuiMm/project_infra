# Packed support: prepared Gate A

The v01 result evaluates one padded PyTorch implementation. Its failure leaves a specific unresolved question: can removing padded projection/gradient work make bounded512 profitable with the same full dense dW output? The packed implementation and toy checks are complete. **The 1.5B replay has not run.**

## Operator and code map

For E valid support edges, `active_ids[E]` names vocabulary rows, `row_ids[E]` names hidden rows, `row_offsets[N+1]` delimits each support, and `action_edges[N]` selects the recorded action. `h[N,D]` and `W[V,D]` retain their original dtypes. Packing preserves each row's support and action even with interspersed padding.

`z_e = dot(h[row_ids[e]], W[active_ids[e]]) / temperature`

`lp_i = z_action(i) - logsumexp(z_edges(i))`

For the incoming logprob gradient c_i, `r_e = c_i (1[e=action(i)] - softmax(z)_e) / temperature`. Then `dH_i = sum_e r_e W_j` and `dW_j = sum_e r_e h_i`, accumulated over the relevant edges. The implementation rounds the scaled r coefficients to the input dtype and accumulates gradients in FP32, matching the v01 sparse convention; numerical agreement is tolerance-based.

| File / symbol | Responsibility |
|---|---|
| `scripts/packed_head.py::validate_support` | Off-timer checks: nonempty set, valid action, vocabulary bounds, no duplicate within a row |
| `pack_support` / `PackedSupport` | Pack valid edges, row offsets and selected action mapping |
| `_edge_logits` / `_segment_logsumexp` | E selected BMM dot products and per-row segmented normalization |
| `_PackedSupportHead.backward` | Recompute E logits; use the saved forward normalizer; FP32 accumulation into dH and full dense dW |
| `scripts/check_packed.py::check` | Independent dense native-autograd reference on a small fixture |
| `scripts/qualify_packed.py::replay` / `timed` | Frozen-trace replay, numerical exclusions, shuffled technical repeats and Gate A verdict |

The edge tensors are proportional to E, while dW remains `[V,D]`. Full FP32 dW zeroing and conversion to the original dtype are included in every candidate evaluation. Tied input-embedding gradients, optimizer state and Adam updates remain outside this operator task. This is a PyTorch prototype; removing padding does not remove dense gradient-buffer costs.

## Controlled comparison and stop rule

Use only the first 128 valid rows of each original v01 trace at the pinned model revision. The protocol binds their exact file hashes and the original model asset receipt. The runner checks asset metadata/JSON digests and streams SHA256 over weight files to verify the original HF LFS blob identities before loading. Sampling, temperature, hidden states, support, actions and coefficient vector are identical across implementations within each arm. Coefficients are deterministically seeded for v02; timing comparisons use remeasured controls, not historical v01 milliseconds. `dense_lp` inherits its numerical-reference qualification from the hash-matched v01 source and evidence; `chunked_lp` must pass the current numerical check.

Compare `dense_lp`, qualified `chunked_lp`, unchanged v01 `padded`, `packed_from_padded` and `packed_prepacked`. All compute fixed-support logprob F+B and return full dW; none includes full-vocabulary entropy. Numerically failing candidates are excluded from timing. Gates retain logprob max absolute error <=0.05 and gradient relative L2 error <=0.02.

The primary lane starts from the GPU padded input and includes packing plus F+B. Boolean indexing can synchronize CUDA because its output size is dynamic; synchronized wall time includes that cost. The prepacked lane starts from metadata prepared outside the timer and is a separate diagnostic. Both report CUDA-event time and total/incremental peak allocated memory. Input transfer, model loading, weight verification, trace collection and off-timer validation are excluded equally.

This compares whole PyTorch implementations. The frozen v01 adapter allocates an unused entropy placeholder, while the packed API returns only logprob and gradients. That small interface cost and different reduction kernels also enter the measured difference. A marginal packed-over-padded result cannot isolate padding as its sole cause.

Gate A requires conversion-inclusive packed speedup **>1.0x versus the faster qualified dense baseline and >1.0x versus padded**, at bounded512/128. Numerical failure yields `NUMERICAL_UNQUALIFIED`; performance failure yields `PACKED_RECIPE_NO_GO`; a pass yields `PROVISIONAL_GATE_A_PASS`. Default/128 is a diagnostic. This mechanism screen is a new gate and does not replace v01's frozen >=1.10x requirement across both configurations. A shared-GPU value just above 1 is descriptive, with no statistical generalization claim.

On failure, stop this recipe. On a provisional pass, inspect whether full dense dW dominates the remaining cost before proposing a separate optimizer experiment. Sparse gradient/optimizer integration, hybrid numerical repair and kernel tuning are outside this gate.

## Validation and replay

Run from the repository root; each output directory must be new:

```bash
python scripts/verify_public.py
python scripts/check_packed.py
python scripts/check_packed.py --device cuda --dtype bfloat16
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python scripts/qualify_packed.py \
  --protocol protocols/qwen15b_v02_gateA.json \
  --trace-dir runs/qwen15b_qualification_v01 \
  --model-dir /your/model/cache/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output runs/qwen15b_packed_gateA_v02
```

The frozen traces and original `model_assets.json` are private local experiment artifacts and are excluded from Git. Exact replay requires those files. For a fresh reproduction, collect traces with the unchanged v01 runner, then freeze their hashes and model asset receipt hash in a new protocol before using the replay runner. Do not change this protocol after seeing timing results. The recorded toy check used PyTorch 2.5.1; the planned real-model replay uses the registered v01 environment, PyTorch 2.11.0+cu128 / Transformers 5.8.1. Private host/PID/device/command receipts stay under the ignored output directory.
