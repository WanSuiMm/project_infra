"""Gate A replay of frozen traces. No generation, downloads or optimizer steps."""
import argparse
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from packed_head import evaluate_packed, pack_support, validate_support
from qualify import errors, evaluate, save_json


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def verify_model_assets(model_path, trace_dir, config, out):
    receipt = Path(trace_dir) / "model_assets.json"
    if digest(receipt) != config["v01_reference"]["model_assets_sha256"]:
        raise ValueError("Frozen v01 model asset receipt mismatch")
    expected = json.loads(receipt.read_text(encoding="utf-8"))
    current = {}
    weight_digests = {}
    for file in sorted(model_path.iterdir()):
        if file.is_file() and (file.suffix == ".json" or file.name.endswith(".safetensors")):
            record = {"bytes": file.stat().st_size, "resolved_blob": file.resolve().name}
            if file.suffix == ".json":
                record["sha256"] = digest(file)
            else:
                # Frozen HF LFS blob names bind the weight bytes, not just sizes.
                expected_id = expected["assets"].get(file.name, {}).get("resolved_blob", "")
                if len(expected_id) != 64 or any(c not in "0123456789abcdef" for c in expected_id):
                    raise ValueError("Expected a SHA256-named frozen HF weight blob")
                weight_digests[file.name] = digest(file)
                if weight_digests[file.name] != expected_id:
                    raise ValueError("Model weight content differs from frozen HF blob identity")
            current[file.name] = record
    if not weight_digests or current != expected["assets"] or expected["verified_snapshot"] != model_path.name:
        raise ValueError("Model assets differ from the v01 snapshot receipt")
    save_json(out / "model_assets.json", {"verified_snapshot": model_path.name, "assets": current,
                                         "weight_sha256": weight_digests})


def invoke(mode, h, w, ids, positions, packed, coeff, temp, chunk):
    if mode.startswith("packed"):
        metadata = pack_support(ids, positions, validate=False) if mode == "packed_from_padded" else packed
        return evaluate_packed(h, w, metadata, temp, coeff)
    result = evaluate(h, w, ids, positions, coeff, temp,
                      "sparse" if mode == "padded" else mode, chunk)
    return result[0], result[2], result[3]


def timed(calls, config):
    timer = config["timing"]
    stats = {mode: {"wall_round_ms": [], "cuda_round_ms": [],
                    "peak_allocated_mib": [], "incremental_peak_mib": []} for mode in calls}
    for call in calls.values():
        for _ in range(timer["warmup"]):
            result = call()
            del result
    rng = random.Random(config["seed"])
    for _ in range(timer["paired_rounds"]):
        order = list(calls)
        rng.shuffle(order)
        for mode in order:
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            for _ in range(timer["iterations_per_round"]):
                result = calls[mode]()
                del result
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000 / timer["iterations_per_round"]
            peak = torch.cuda.max_memory_allocated()
            record = stats[mode]
            record["wall_round_ms"].append(wall_ms)
            record["cuda_round_ms"].append(start.elapsed_time(end) / timer["iterations_per_round"])
            record["peak_allocated_mib"].append(peak / 2**20)
            record["incremental_peak_mib"].append((peak - base) / 2**20)
    for record in stats.values():
        record["median_wall_ms"] = statistics.median(record["wall_round_ms"])
        record["median_cuda_ms"] = statistics.median(record["cuda_round_ms"])
    return stats


def replay(args, config, out):
    model_path = Path(args.model_dir)
    if model_path.name != config["revision"]:
        raise ValueError("Model snapshot basename must match the protocol revision")
    source = Path(__file__).parent
    frozen = config["v01_reference"]
    if digest(source / "qualify.py") != frozen["runner_sha256"]:
        raise ValueError("Inherited v01 reference source changed")
    evidence = source.parent / "evidence/qwen15b_v01/summary.json"
    if digest(evidence) != frozen["summary_sha256"]:
        raise ValueError("Inherited v01 reference evidence changed")
    v01 = json.loads(evidence.read_text(encoding="utf-8"))
    if not all(a["correctness"]["dense_lp"]["pass"] for a in v01["arms"]):
        raise ValueError("dense_lp reference lacks inherited qualification")
    verify_model_assets(model_path, args.trace_dir, config, out)
    traces = []
    for spec in config["traces"]:
        path = Path(args.trace_dir) / spec["file"]
        if digest(path) != spec["sha256"]:
            raise ValueError(f"Frozen trace digest mismatch: {spec['file']}")
        trace = torch.load(path, map_location="cpu", weights_only=True)
        if trace["setting"]["name"] != spec["name"] or trace["setting"]["temperature"] != spec["temperature"]:
            raise ValueError("Trace setting does not match protocol")
        traces.append((spec, trace))
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    w = model.get_output_embeddings().weight.detach().to("cuda").clone()
    if w.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 output-head weights")
    model_meta = {"revision": model_path.name, "vocab": len(w), "hidden": w.shape[1],
                  "tied_embeddings": bool(model.config.tie_word_embeddings)}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    summary = {"model": model_meta, "arms": [], "reference": "dense_lp eligibility inherited from source/summary-hash-matched v01 numerical qualification",
               "claim": "Gate A saved-trace operator comparison; full dense dW retained."}
    for index, (spec, trace) in enumerate(traces):
        for n in config["rows"]:
            if len(trace["h"]) < n or not bool(trace["pre_eos"][:n].all()):
                raise ValueError("Insufficient valid pre-EOS trace rows")
            ids_cpu, pos_cpu = trace["ids"][:n], trace["positions"][:n]
            validate_support(ids_cpu, pos_cpu, len(w))
            h, ids, pos = trace["h"][:n].to("cuda").clone(), ids_cpu.to("cuda"), pos_cpu.to("cuda")
            if h.dtype != torch.bfloat16 or h.shape[1] != w.shape[1]:
                raise ValueError("Trace hidden input dtype/shape mismatch")
            packed = pack_support(ids, pos, validate=False)
            generator = torch.Generator(device="cuda").manual_seed(config["seed"] + index * 1000 + n)
            coeff = torch.randn(n, device="cuda", generator=generator) / n
            temp, chunk = spec["temperature"], config["vocab_chunk"]
            modes = ["dense_lp", "chunked_lp", "padded", "packed_from_padded", "packed_prepacked"]
            calls = {m: (lambda mode=m: invoke(mode, h, w, ids, pos, packed, coeff, temp, chunk)) for m in modes}
            ref = calls["dense_lp"]()
            checks = {"dense_lp": {"pass": True, "role": "v01-qualified numerical reference (inherited)"}}
            for mode in modes[1:]:
                result = calls[mode]()
                e = {name: errors(a, b) for name, a, b in zip(("logprob", "dh", "dw"), result, ref)}
                gate = config["correctness"]
                passed = e["logprob"]["max_abs"] <= gate["logprob_max_abs"] and all(
                    e[k]["relative_l2"] <= gate["gradient_relative_l2"] for k in ("dh", "dw"))
                checks[mode] = {"pass": bool(passed), "errors": e}
                del result
            del ref
            timings = timed({m: call for m, call in calls.items() if checks[m]["pass"]}, config)
            base_ms = min(timings[m]["median_wall_ms"] for m in ("dense_lp", "chunked_lp") if m in timings)
            active = int((ids_cpu >= 0).sum())
            arm = {"setting": spec["name"], "rows": n, "active_pairs": active,
                   "padded_pairs": ids_cpu.numel(), "padding_fraction": 1 - active / ids_cpu.numel(),
                   "correctness": checks, "timings": timings,
                   "speedups": {m: base_ms / timings[m]["median_wall_ms"] for m in
                                ("padded", "packed_from_padded", "packed_prepacked") if m in timings}}
            if "packed_from_padded" in timings and "padded" in timings:
                arm["packed_over_padded"] = timings["padded"]["median_wall_ms"] / timings["packed_from_padded"]["median_wall_ms"]
            summary["arms"].append(arm)
            save_json(out / "summary.json", summary)
            print(json.dumps(arm), flush=True)
            del h, ids, pos, packed, coeff, calls
    gate = config["gate"]
    endpoint = next(a for a in summary["arms"] if a["setting"] == gate["setting"] and a["rows"] == gate["rows"])
    if not endpoint["correctness"]["packed_from_padded"]["pass"] or not endpoint["correctness"]["padded"]["pass"]:
        verdict = "NUMERICAL_UNQUALIFIED"
    elif (endpoint["speedups"]["packed_from_padded"] > gate["speedup_strictly_greater_than"]
          and endpoint["packed_over_padded"] > gate["packed_over_padded_strictly_greater_than"]):
        verdict = "PROVISIONAL_GATE_A_PASS"
    else:
        verdict = "PACKED_RECIPE_NO_GO"
    summary["verdict"] = verdict
    save_json(out / "summary.json", summary)
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path(__file__).parent
    save_json(out / "manifest.json", {"protocol_sha256": digest(args.protocol), "protocol": config,
        "source_sha256": {name: digest(source / name) for name in ("qualify.py", "qualify_packed.py", "packed_head.py")},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "start_time": time.time(),
        "pid": os.getpid(), "host": platform.node(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": [sys.executable, *sys.argv]})
    save_json(out / "status.json", {"state": "RUNNING", "phase": "saved_trace_replay", "pid": os.getpid()})
    try:
        from check_packed import check
        save_json(out / "sanity.json", check("cuda"))
        verdict = replay(args, config, out)
        save_json(out / "status.json", {"state": "COMPLETE", "exit_code": 0, "verdict": verdict, "end_time": time.time()})
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        save_json(out / "status.json", {"state": "FAILED", "exit_code": 1, "end_time": time.time()})
        raise


if __name__ == "__main__":
    main()
