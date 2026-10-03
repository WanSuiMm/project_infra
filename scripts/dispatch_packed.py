"""One-shot preflight and durable dispatch; machine paths are CLI arguments."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root, model, traces = Path(args.root), Path(args.model_dir), Path(args.trace_dir)
    if not all(p.is_absolute() for p in (root, model, traces)):
        raise ValueError("Use absolute remote paths")
    if Path(args.run_id).name != args.run_id or args.run_id in ("", ".", ".."):
        raise ValueError("run-id must be a directory basename")
    out = root / "runs" / args.run_id
    log = root / "runs" / (args.run_id + ".log")
    receipt_path = root / "runs" / (args.run_id + "_launch.json")
    if out.exists() or log.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to overwrite an existing run or receipt")
    deployment = json.loads((root / "deployment.json").read_text(encoding="utf-8"))
    for relative, expected in deployment["sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Deployment hash mismatch: {relative}")
    protocol_path = root / "protocols/qwen15b_v02_gateA.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if model.name != protocol["revision"] or not model.is_dir():
        raise ValueError("Frozen model snapshot missing")
    if sha256(traces / "model_assets.json") != protocol["v01_reference"]["model_assets_sha256"]:
        raise ValueError("Model receipt changed")
    for spec in protocol["traces"]:
        if sha256(traces / spec["file"]) != spec["sha256"]:
            raise ValueError("Frozen input trace changed")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    sys.path.insert(0, str(root / "scripts"))
    import torch
    import transformers
    if torch.__version__ != "2.11.0+cu128" or transformers.__version__ != "5.8.1":
        raise RuntimeError("Registered v01 runtime versions changed")
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 2**30 or shutil.disk_usage(root).free < 2**30:
        raise RuntimeError("Insufficient memory or disk headroom")
    command = [sys.executable, "-u", str(root / "scripts/qualify_packed.py"),
               "--protocol", str(protocol_path), "--trace-dir", str(traces),
               "--model-dir", str(model), "--output", str(out)]
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        from check_packed import check
        import qualify_packed
        checks = {"fp32": check("cuda"), "bf16": check("cuda", torch.bfloat16)}
        record = {"state": "PREFLIGHT_PASS", "checks": checks,
                  "gpu_index": args.gpu, "gpu": torch.cuda.get_device_name(),
                  "free_mib": free / 2**20, "total_mib": total / 2**20,
                  "source_commit": deployment["source_commit"], "command": command,
                  "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        path = out.parent / (args.run_id + "_preflight.json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)
        return
    preflight_path = out.parent / (args.run_id + "_preflight.json")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight["state"] != "PREFLIGHT_PASS" or preflight["command"] != command:
        raise RuntimeError("Matching preflight required before launch")
    with log.open("xb") as handle:
        process = subprocess.Popen(command, cwd=root, env=os.environ.copy(),
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True)
    record = {"state": "DISPATCHED", "host": socket.gethostname(), "pid": process.pid,
              "gpu_index": args.gpu, "gpu": torch.cuda.get_device_name(),
              "launch_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "command": command, "output": str(out), "log": str(log),
              "alive_at_dispatch": process.poll() is None, "free_mib_at_dispatch": free / 2**20,
              "deployment": deployment, "expected_window": "minutes; shared-GPU operator replay"}
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
