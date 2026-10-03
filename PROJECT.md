# Support-sparse output head

Question: can an already-required, fixed rollout support remove vocabulary projection work without changing the support-conditioned logprob objective?

Status: Qwen2.5-1.5B-Instruct operator qualification completed, exit code 0. Support-only numerical checks passed all four arms; 128-row speedups were 2.129x / 0.805x for the two fixed sampling configurations, failing the frozen cross-configuration performance gate. Hybrid failed the 128-row numerical gate and is numerically unqualified there, not an established negative performance result. No RL training. Protocol: `protocols/qwen15b_v01.json`. Entry: `scripts/qualify.py`. New reproduction outputs belong under `runs/`.

Public evidence: `evidence/qwen15b_v01/`. See `RESULTS.md` for interpretation and `evidence/qwen15b_v01/summary.json` for canonical numbers. Private launch receipts and generated traces are retained locally and excluded from Git. No repair or rerun has been launched.

Compare full-vocabulary dense/chunked projection followed by masking against support-only projection with full dense weight-gradient output. Measure forward plus backward, including gradient buffer initialization and index accumulation. Separately preserve full-vocabulary entropy in a forward-dense/backward-sparse lane. These are matched tasks, not interchangeable diagnostics.

BF16 numerical agreement is tolerance-based, not bitwise. A small FP32 algebra check precedes the real-model qualification. Real traces are generated under two preregistered sampling configurations; neither is claimed to be an existing RL workload. GPU timings on a shared device are descriptive. Failure of this prototype does not prove all sparse kernels impossible.

No model downloads, environment upgrades, optimizer updates, multi-GPU work, or ongoing monitor are authorized by this protocol.
