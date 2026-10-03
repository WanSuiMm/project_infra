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
- Sparse v01 computes padded slots too. Direct inspection of the retained traces gives 511 valid / 2,560 padded slots for default/128 and 3,464 valid / 65,536 padded slots for bounded512/128. The latter has 94.7% padding; 18.92x is the slot-count ratio. Padding is a concrete implementation cost, but its contribution to the observed slowdown has not yet been measured.
- Sparse peak allocated memory is higher: at 128 rows, default dense/sparse 1540.6/1823.8 MiB; bounded512 1541.7/2377.0 MiB. Dense weight-gradient output, FP32 index accumulation and gather temporaries are included.
- Timings are technical repeats on a shared GPU, from 8 diagnostic prompt trajectories and two prescribed sampling configurations. No full-model backward, optimizer update, RL training, production workload, or end-to-end speedup was measured.

Source SHA256: `47316f665069507b473aa9071e01fa25be1a532beec6b0ceec207bca77423cbc`.

## Packed v02 follow-up

The current reading is: **v01 numerical agreement PASS; small-support descriptive positive; padded cross-configuration gate FAIL; packed v02 numerical agreement PASS and Gate A FAIL**. Both frozen negative verdicts are preserved.

The [packed head](scripts/packed_head.py) processes only valid edges and still returns full dense dW. The [completed real-model result table](evidence/qwen15b_packed_gateA_v02/RESULTS.md) is generated from the saved canonical summary. At bounded512/128, dense/padded/conversion-inclusive packed/prepacked medians were 4.792/6.667/9.934/8.607 ms. Packed speedups versus dense are 0.482x and 0.557x; the primary lane also loses to padded (0.671x). Numerical checks pass for both packed lanes at both settings. Chunked references fail and are excluded from timing.

Bounded512 peak allocation falls from padded 2376.1 to packed 1820.9 MiB, while dense is 1542.5 MiB. This is a descriptive allocation reduction, and does not satisfy the performance gate. The exact saved inputs, model weight identities, source hashes and successful exit were verified. Measurements were on a shared GPU; they are an implementation comparison on the same diagnostic trajectories, not new independent samples or end-to-end evidence.

The frozen [Gate A](protocols/qwen15b_v02_gateA.json) verdict is `PACKED_RECIPE_NO_GO`. Stop this PyTorch recipe without a hidden sweep. The initial [toy checks](evidence/packed_v02_toy/checks.json) remain preserved as preparation evidence. [PACKED_V02.md](PACKED_V02.md) gives scope, code map and commands. Sparse optimizer integration and hybrid repair remain separate, untested questions.
