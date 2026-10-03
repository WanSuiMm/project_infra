# Packed v02 Gate A result

Completed, exit code 0; numerical agreement passed for both packed lanes in both arms. Frozen verdict: **PACKED_RECIPE_NO_GO**.

| Setting | Dense ms | Padded ms | Packed with conversion ms | Prepacked ms | Packed / dense speedup | Prepacked / dense speedup |
|---|---:|---:|---:|---:|---:|---:|
| model_default | 6.136 | 3.961 | 10.413 | 6.684 | 0.589x | 0.918x |
| bounded512 | 4.792 | 6.667 | 9.934 | 8.607 | 0.482x | 0.557x |

All arms use 128 rows. Times are synchronized wall-clock F+B, measured on a shared RTX 5090. Chunked references failed numerical gates and were excluded. Packing is charged in the primary packed lane; prepacked is secondary. Full dense dW remains included.

The bounded512 endpoint is slower than both the dense and padded controls; stop this packed PyTorch recipe without a hidden tuning sweep. A lower allocated-memory peak does not pass the frozen performance gate. This result does not establish a no-go for fused kernels or the general computational-exhaust theme.

Canonical numbers: [summary.json](summary.json). Provenance and source bindings: [provenance.json](provenance.json).
