# Support-sparse LM head: a bounded qualification

Can an already-required rollout sampling support eliminate output-layer work while preserving the fixed-support logprob objective? The completed Qwen2.5-1.5B-Instruct **padded v01** passed all four numerical checks but failed its frozen performance gate: **2.129x** / **0.805x** at 128 positions for the small/broader support configurations. **Packed v02** also passed numerical checks but failed Gate A: at bounded512/128, conversion-inclusive and prepacked speedups were **0.482x** and **0.557x** versus dense. Both retain full dense dW. These are shared-GPU operator measurements. Hybrid preserving full-vocabulary entropy was numerically unqualified at 128 positions. No RL training or end-to-end speedup was measured.

Start here:

1. [RESULTS.md](RESULTS.md): measured results, negative outcomes and limitations.
2. [GPT_HANDOFF.md](GPT_HANDOFF.md): incremental changes since the published v01 evidence.
3. [GPT_CONTEXT.md](GPT_CONTEXT.md) and [PACKED_V02.md](PACKED_V02.md): task, code map and completed Gate A.
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

The initial toy checks used PyTorch 2.5.1 and an RTX 4060 Laptop GPU. The completed [Gate A results](evidence/qwen15b_packed_gateA_v02/RESULTS.md) replay the exact saved v01 traces on the original RTX 5090 environment, with packing included in the primary timing and prepacked timing reported separately. The frozen verdict is `PACKED_RECIPE_NO_GO`; this recipe stops here. See [PACKED_V02.md](PACKED_V02.md) for the command and scope. Frozen trace tensors are retained locally and excluded from this repository; exact replay requires those files. A fresh reproduction must freeze its own trace hashes and asset receipt in a separate protocol before timing.
