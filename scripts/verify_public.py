"""CPU-only integrity checks for the compact public evidence."""
import ast
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
evidence = root / "evidence/qwen15b_v01"
provenance = json.loads((evidence / "provenance.json").read_text())
for relative, expected in provenance["sha256"].items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    assert actual == expected, f"Artifact changed: {relative}"
for script in (root / "scripts").glob("*.py"):
    ast.parse(script.read_text(encoding="utf-8"))
s = json.loads((evidence / "summary.json").read_text())
assert len(s["arms"]) == 4
for arm in s["arms"]:
    assert arm["correctness"]["sparse"]["pass"]
    times = arm["timings"]
    base = min(times[m]["median_ms"] for m in ("dense_lp", "chunked_lp") if m in times)
    assert math.isclose(arm["speedups"]["sparse"], base / times["sparse"]["median_ms"], rel_tol=1e-12)
    for mode in ("sparse", "hybrid"):
        if not arm["correctness"][mode]["pass"]:
            assert mode not in times and mode not in arm["speedups"]
endpoints = [a for a in s["arms"] if a["rows"] == 128]
assert len(endpoints) == 2
assert not all(a["speedups"]["sparse"] >= 1.10 for a in endpoints)
assert s["verdict"]["sparse"] == "PROTOTYPE_NO_GO"
assert all(not a["correctness"]["hybrid"]["pass"] for a in endpoints)
for arm in json.loads((evidence / "sanity.json").read_text()).values():
    assert all(item["max_abs"] < 5e-5 for item in arm.values())
packed = json.loads((root / "evidence/packed_v02_toy/checks.json").read_text())
assert packed["real_model_gate"] == "NOT_RUN"
for relative, expected in packed["source_sha256"].items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
for name, check in packed["checks"].items():
    assert check["status"] == "PASS" and check["active_pairs"] == 17
    assert check["dense_dw_shape"] == [43, 16] and check["invalid_cases_rejected"] == 4
    for metric, value in check["errors"].items():
        if name == "cpu_fp32":
            assert value["max_abs"] < 5e-5
        elif metric == "logprob":
            assert value["max_abs"] <= 0.05
        else:
            assert value["relative_l2"] <= 0.02
v02_path = root / "evidence/qwen15b_packed_gateA_v02"
v02_provenance = json.loads((v02_path / "provenance.json").read_text())
assert v02_provenance["execution"] == "COMPLETE" and v02_provenance["exit_code"] == 0
for relative, expected in v02_provenance["sha256"].items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
v02 = json.loads((v02_path / "summary.json").read_text())
assert len(v02["arms"]) == 2
for arm in v02["arms"]:
    assert arm["rows"] == 128
    assert all(arm["correctness"][mode]["pass"] for mode in ("padded", "packed_from_padded", "packed_prepacked"))
    times = arm["timings"]
    base = min(times[m]["median_wall_ms"] for m in ("dense_lp", "chunked_lp") if m in times)
    for mode, ratio in arm["speedups"].items():
        assert math.isclose(ratio, base / times[mode]["median_wall_ms"], rel_tol=1e-12)
    for mode, check in arm["correctness"].items():
        if not check["pass"]:
            assert mode not in times
endpoint = next(a for a in v02["arms"] if a["setting"] == "bounded512")
assert endpoint["speedups"]["packed_from_padded"] < 1 and endpoint["packed_over_padded"] < 1
assert v02["verdict"] == "PACKED_RECIPE_NO_GO"
print("PASS: frozen v01 and v02 integrity, numerical exclusions, speedups and both no-go verdicts")
