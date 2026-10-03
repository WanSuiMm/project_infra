# Support-sparse LM head: a bounded qualification

Can an already-required rollout sampling support eliminate output-layer work while preserving the fixed-support logprob objective? This repository contains one completed Qwen2.5-1.5B-Instruct operator pilot. The support-only implementation passed all four numerical checks, but failed the preregistered performance gate: at 128 positions it achieved **2.129x** for the small-support configuration and **0.805x** for the broader configuration. The hybrid implementation preserving full-vocabulary entropy was numerically unqualified at 128 positions. No RL training or end-to-end training speedup was measured.

Start here:

1. [RESULTS.md](RESULTS.md): measured results, negative outcomes and limitations.
2. [GPT_CONTEXT.md](GPT_CONTEXT.md): exact task, source symbols and review questions.
3. [Frozen protocol](protocols/qwen15b_v01.json) and [runner](scripts/qualify.py).
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
