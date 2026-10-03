# Incremental review: completed packed v02 Gate A

- Review base: `e416680ab43fcfe70ea42cddf4db85ca98133fef` (published packed preparation).
- Evidence head: `e6b164aa29c0049d9f927181b41d439f64ed5ad7`.
- Runtime source commit: `85e4c38f8084ca5b93c60f2540b332826c898624`; operator and protocol bytes match the preparation code.
- A later metadata-only commit updates this handoff; keep the evidence head above stable.

## Changed and unchanged claims

**Unchanged:** v01 source, protocol and canonical result bytes; padded cross-configuration `PROTOTYPE_NO_GO`; hybrid/128 `NUMERICAL_UNQUALIFIED`; packed operator/protocol and initial toy evidence; no RL, optimizer, full-model backward or end-to-end evidence.

**Changed:** v02 execution moves from NOT_RUN to COMPLETE/exit 0. Both packed lanes pass numerical checks in both settings. Frozen Gate A verdict is **PACKED_RECIPE_NO_GO**: bounded512/128 conversion-inclusive packed is 0.482x versus dense and 0.671x versus padded; prepacked is 0.557x versus dense. Stop this PyTorch recipe.

**New evidence:** exact saved inputs and model weight identities were verified; shared-GPU controls were remeasured. Chunked references fail and are excluded. Packed peak allocation is about 555 MiB below padded at bounded512, but 278 MiB above dense. These are descriptive operator measurements on the same eight diagnostic trajectories.

## Minimal reading order

1. [New result table](evidence/qwen15b_packed_gateA_v02/RESULTS.md) and [provenance](evidence/qwen15b_packed_gateA_v02/provenance.json).
2. [New canonical summary](evidence/qwen15b_packed_gateA_v02/summary.json): numerical gates, qualified timings, raw technical repeats and allocation peaks.
3. [RESULTS.md](RESULTS.md): revised interpretation and preserved v01 result.
4. [PACKED_V02.md](PACKED_V02.md) only for scope and stopping criteria. Run `python scripts/verify_public.py` for source/evidence integrity.

The packed implementation and protocol are unchanged; they need rereading only for a specific disputed mechanism. Large trace tensors, private launch receipts and logs remain excluded. The v01 and toy evidence need rereading only when checking a disputed original number.

## Reviewer questions and next decision

1. Are numerical exclusions, speedup arithmetic and the frozen Gate A verdict consistent with the saved artifacts?
2. Does the result justify stopping this PyTorch recipe with full dense dW and conversion charged in the primary lane?
3. Are shared-GPU timing and reused-trajectory limitations sufficient to keep the negative conclusion within this implementation's scope?

No additional run or hidden tuning sweep follows this failure. Optimizer integration, hybrid repair and a fused kernel would require separate questions and protocols.
