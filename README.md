# Support-sparse LM head: a bounded qualification

Can an already-required rollout sampling support eliminate output-layer work while preserving the fixed-support logprob objective? The completed Qwen2.5-1.5B-Instruct **padded v01** passed all four numerical checks but failed its frozen performance gate: **2.129x** / **0.805x** at 128 positions for the small/broader support configurations. A **packed v02** now processes only valid support entries and retains full dense dW. Its toy checks pass; its real-model performance gate is **NOT_RUN**. Hybrid preserving full-vocabulary entropy was numerically unqualified at 128 positions. No RL training or end-to-end speedup was measured.

Start here:

1. [RESULTS.md](RESULTS.md): measured results, negative outcomes and limitations.
2. [GPT_HANDOFF.md](GPT_HANDOFF.md): incremental changes since the published v01 evidence.
3. [GPT_CONTEXT.md](GPT_CONTEXT.md) and [PACKED_V02.md](PACKED_V02.md): task, code map and prepared Gate A.
4. [Canonical numerical evidence](evidence/qwen15b_v01/summary.json), only when checking individual errors or timings. No large logs or model files are required for review.

## Reproduce

The recorded environment used Python 3.12, PyTorch 2.11.0+cu128, Transformers 5.8.1 and an RTX 5090. Timings were taken on a shared GPU. Results are hardware- and workload-specific.

Install the appropriate CUDA build of PyTorch, then the packages in [requirements.txt](requirements.txt). Obtain `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` separately; model weights are not included. The runner is offline-only and requires the local snapshot directory's basename to match that revision.

```bash
python scripts/verify_public.py
CUDA_VISIBLE_DEVICES=0 python scripts/qualify.py \
  --protocol protocols/qwen15b_v01.json \
  --output runs/new_smoke --smoke-only
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python scripts/qualify.py \
  --protocol protocols/qwen15b_v01.json \
  --model-dir /your/model/cache/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output runs/new_qualification
```

Each output directory must be new. Run outputs contain local provenance and are ignored by Git. The public evidence includes sanitized provenance and compact canonical results; private launch receipts, generated trace tensors and model caches are excluded.

The prototype uses PyTorch gather/BMM/index-add, not a fused sparse kernel. Its failure does not establish that all support-sparse implementations are unprofitable. Follow-up work must preserve this run and use a new protocol/run directory.

## Packed v02

Run the small independent numerical checks without model weights:

```bash
python scripts/check_packed.py
python scripts/check_packed.py --device cuda --dtype bfloat16
```

The recorded toy checks used PyTorch 2.5.1 and an RTX 4060 Laptop GPU; they do not qualify the 1.5B workload. [Gate A](protocols/qwen15b_v02_gateA.json) prepares a replay of the exact saved v01 traces, with packing included in the primary timing and prepacked timing reported separately. See [PACKED_V02.md](PACKED_V02.md) for the command and stop criteria. Frozen trace tensors are retained locally and excluded from this repository; exact replay requires those files. A fresh reproduction must freeze its own trace hashes in a separate protocol before timing.
