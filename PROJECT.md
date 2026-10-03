# Support-sparse output head

Question: can an already-required, fixed rollout support remove vocabulary projection work without changing the support-conditioned logprob objective?

Status: Qwen2.5-1.5B-Instruct operator qualification completed, exit code 0. Support-only numerical checks passed all four arms; 128-row speedups were 2.129x / 0.805x for the two fixed sampling configurations, failing the frozen cross-configuration performance gate. Hybrid failed the 128-row numerical gate and is numerically unqualified there, not an established negative performance result. No RL training. Protocol: `protocols/qwen15b_v01.json`. Entry: `scripts/qualify.py`. New reproduction outputs belong under `runs/`.

Public v01 evidence: `evidence/qwen15b_v01/`. See `RESULTS.md` for interpretation and `evidence/qwen15b_v01/summary.json` for canonical numbers. Private launch receipts and generated traces are retained locally and excluded from Git.

Follow-up status: v02 Gate A COMPLETE, exit code 0, process gone. Both packed lanes pass numerical checks at both 128-row settings. Gate A verdict: `PACKED_RECIPE_NO_GO`; bounded512 conversion-inclusive/prepacked speedups are 0.482x/0.557x versus dense. Stop this packed PyTorch recipe. Canonical evidence: `evidence/qwen15b_packed_gateA_v02/`; private full receipts: `runs/qwen15b_packed_gateA_v02*`. Entry: `scripts/qualify_packed.py`; protocol: `protocols/qwen15b_v02_gateA.json`; design and commands: `PACKED_V02.md`. The frozen v01 and initial toy evidence remain unchanged. Runtime source commit: `85e4c38f8084ca5b93c60f2540b332826c898624`.

Compare full-vocabulary dense/chunked projection followed by masking against support-only projection with full dense weight-gradient output. Measure forward plus backward, including gradient buffer initialization and index accumulation. Separately preserve full-vocabulary entropy in a forward-dense/backward-sparse lane. These are matched tasks, not interchangeable diagnostics.

BF16 numerical agreement is tolerance-based, not bitwise. A small FP32 algebra check precedes the real-model qualification. Real traces are generated under two preregistered sampling configurations; neither is claimed to be an existing RL workload. GPU timings on a shared device are descriptive. Failure of this prototype does not prove all sparse kernels impossible.

No model downloads, environment upgrades, optimizer updates, multi-GPU work, or ongoing monitor are authorized by this protocol.
