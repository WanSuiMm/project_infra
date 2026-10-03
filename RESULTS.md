# Qwen2.5-1.5B output-head qualification

Completed run: `qwen15b_qualification_v01`; exit code 0. Canonical public source: [summary.json](evidence/qwen15b_v01/summary.json). Its bytes match the original saved summary; private run outputs are preserved locally.

| Setting | Rows | Mean support | Dense logprob F+B ms | Sparse F+B ms | Speedup | Sparse numerical gate |
|---|---:|---:|---:|---:|---:|---|
| model_default | 32 | 3.03 | 2.682 | 1.945 | 1.379x | PASS |
| model_default | 128 | 3.99 | 4.201 | 1.973 | 2.129x | PASS |
| bounded512 | 32 | 13.59 | 2.721 | 2.360 | 1.153x | PASS |
| bounded512 | 128 | 27.06 | 3.655 | 4.540 | 0.805x | PASS |

## Interpretation

- Support-only: all four numerical arms pass. The frozen gate requires >=1.10x at 128 rows for both configurations; 2.129x and 0.805x fail that gate. This is a prototype-level no-go, with a narrower small-support positive result.
- Hybrid: 32-row arms pass numerical checks but are slower (0.783x and 0.697x). Both 128-row arms fail numerical checks, so they were not timed. Treat those endpoints as NUMERICAL_UNQUALIFIED, not measured zero speedup. The original generated run table uses 0.000x for missing timing; that placeholder is not a measurement.
- Chunked references also fail the 128-row numerical gate: max logprob differences 0.171 and 0.240. Hybrid uses chunked GEMM scores in forward and selected BMM scores in backward; the observed numerical inconsistency must be resolved before a same-interface performance claim. No repair or rerun was performed.
- Sparse prototype computes padded slots too: the 128-row default arm retains 3.99 of 20 slots on average; bounded512 retains 27.06 of 512. Packing/segmented processing is a plausible follow-up, not a tested explanation or automatic rescue of this frozen result.
- Sparse peak allocated memory is higher: at 128 rows, default dense/sparse 1540.6/1823.8 MiB; bounded512 1541.7/2377.0 MiB. Dense weight-gradient output, FP32 index accumulation and gather temporaries are included.
- Timings are technical repeats on a shared GPU, from 8 diagnostic prompt trajectories and two prescribed sampling configurations. No full-model backward, optimizer update, RL training, production workload, or end-to-end speedup was measured.

Source SHA256: `47316f665069507b473aa9071e01fa25be1a532beec6b0ceec207bca77423cbc`.
