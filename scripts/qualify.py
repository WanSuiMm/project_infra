"""Bounded real-model qualification. No downloads and no optimizer updates."""
import argparse
import gc
import hashlib
import json
import os
import platform
import random
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "Solve and explain: a train travels 180 km in 3 hours. What is its average speed?",
    "Find all real roots of x squared minus five x plus six equals zero.",
    "Write a Python function that removes duplicates while preserving order.",
    "Explain why binary search needs a sorted input, with a short example.",
    "Explain the difference between evaporation and boiling in plain language.",
    "Translate into Chinese: Reliable experiments preserve negative results.",
    "A shop discounts a 240 dollar item by 15 percent. Calculate its new price.",
    "Give three practical steps to diagnose a program that unexpectedly uses too much memory.",
]


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def selected_logits(h, w, ids):
    # BMM avoids a full vocabulary projection; the temporary gather is measured.
    gathered = w[ids]
    return torch.bmm(gathered, h.unsqueeze(2)).squeeze(2).float()


def summarize(z, temperature):
    z = z.float() / temperature
    lp = z.log_softmax(-1)
    return lp, -(lp.exp() * lp).sum(-1)


class Head(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, ids, positions, upstream_temp, mode, chunk):
        temp = upstream_temp
        entropy = None
        if mode == "sparse":
            chosen = selected_logits(h, w, ids)
        else:
            chosen = torch.empty(ids.shape, device=h.device, dtype=torch.float32)
            m = torch.full((len(h),), -torch.inf, device=h.device)
            s = torch.zeros_like(m)
            t = torch.zeros_like(m)
            full_entropy = mode not in ("dense_lp", "chunked_lp")
            block = len(w) if mode in ("dense", "dense_lp") else chunk
            for lo in range(0, len(w), block):
                hi = min(lo + block, len(w))
                z = (h @ w[lo:hi].T).float()
                inside = (ids >= lo) & (ids < hi)
                local = (ids - lo).clamp(0, hi - lo - 1)
                chosen = torch.where(inside, z.gather(1, local), chosen)
                if full_entropy:
                    z = z / temp
                    new_m = torch.maximum(m, z.amax(-1))
                    factor = (m - new_m).exp()
                    exp = (z - new_m[:, None]).exp()
                    s = s * factor + exp.sum(-1)
                    t = t * factor + (exp * z).sum(-1)
                    m = new_m
            if full_entropy:
                entropy = m + s.log() - t / s
        valid = ids >= 0
        z = (chosen / temp).masked_fill(~valid, -torch.inf)
        logz = z.logsumexp(-1)
        result = z.gather(1, positions[:, None]).squeeze(1) - logz
        ctx.save_for_backward(h, w, ids, positions, logz)
        ctx.temp, ctx.mode, ctx.chunk = temp, mode, chunk
        ctx.set_materialize_grads(False)
        # masked entropy is intentionally not returned as full-vocabulary entropy.
        if entropy is None:
            entropy = torch.zeros_like(result)
        return result, entropy

    @staticmethod
    def backward(ctx, grad, grad_entropy):
        assert grad_entropy is None, "Full entropy is a diagnostic, not a differentiable loss."
        h, w, ids, positions, logz = ctx.saved_tensors
        temp = ctx.temp
        if ctx.mode in ("sparse", "hybrid"):
            safe = ids.clamp_min(0)
            z = selected_logits(h, w, safe) / temp
            probs = (z - logz[:, None]).exp().masked_fill(ids < 0, 0)
            r = -probs * grad[:, None]
            r.scatter_add_(1, positions[:, None], grad[:, None])
            r = (r / temp).to(h.dtype)
            # FP32 reduction and accumulation; dense output dW is retained.
            gathered = w[safe]
            dh = (r.float()[:, :, None] * gathered.float()).sum(1).to(h.dtype)
            dw = torch.zeros(w.shape, device=w.device, dtype=torch.float32)
            edge_grad = r.float()[:, :, None] * h.float()[:, None, :]
            dw.index_add_(0, safe.reshape(-1), edge_grad.reshape(-1, w.shape[1]))
            dw = dw.to(w.dtype)
        else:
            dh = torch.zeros_like(h)
            dw = torch.zeros_like(w)
            block = len(w) if ctx.mode in ("dense", "dense_lp") else ctx.chunk
            target = ids.gather(1, positions[:, None]).squeeze(1)
            for lo in range(0, len(w), block):
                hi = min(lo + block, len(w))
                z = (h @ w[lo:hi].T).float() / temp
                inside = (ids >= lo) & (ids < hi)
                local = (ids - lo).clamp(0, hi - lo - 1)
                mask = torch.zeros(z.shape, device=z.device, dtype=torch.int32)
                mask.scatter_add_(1, local, inside.to(torch.int32))
                z.masked_fill_(mask == 0, -torch.inf)
                r = -(z - logz[:, None]).exp() * grad[:, None]
                target_in = (target >= lo) & (target < hi)
                loc = (target - lo).clamp(0, hi - lo - 1)
                r.scatter_add_(1, loc[:, None], (grad * target_in)[:, None])
                r = (r / temp).to(h.dtype)
                dh.add_(r @ w[lo:hi])
                dw[lo:hi] = r.T @ h
        return dh, dw, None, None, None, None, None


def evaluate(h, w, ids, positions, coeff, temp, mode, chunk):
    x = h.detach().requires_grad_()
    weight = w.detach().requires_grad_()
    lp, entropy = Head.apply(x, weight, ids, positions, temp, mode, chunk)
    dh, dw = torch.autograd.grad(lp, (x, weight), coeff)
    return lp.detach(), entropy.detach(), dh, dw


def errors(candidate, reference):
    diff = candidate.float() - reference.float()
    return {"max_abs": diff.abs().max().item(),
            "relative_l2": (diff.norm() / reference.float().norm().clamp_min(1e-12)).item()}


def algebra_check():
    gen = torch.Generator(device="cuda").manual_seed(17)
    h = torch.randn(5, 16, generator=gen, device="cuda", requires_grad=True)
    w = torch.randn(43, 16, generator=gen, device="cuda", requires_grad=True)
    ids = torch.stack([torch.randperm(43, generator=gen, device="cuda")[:7] for _ in range(5)])
    ids[0, -2:] = -1
    pos = torch.arange(5, device="cuda") % 5
    coeff = torch.randn(5, generator=gen, device="cuda")
    logits = h @ w.T / 0.7
    indicator = torch.zeros(logits.shape, dtype=torch.int32, device="cuda")
    indicator.scatter_add_(1, ids.clamp_min(0), (ids >= 0).to(torch.int32))
    keep = indicator > 0
    target = ids.gather(1, pos[:, None]).squeeze(1)
    lp = logits.masked_fill(~keep, -torch.inf).log_softmax(-1).gather(1, target[:, None]).squeeze(1)
    dh, dw = torch.autograd.grad(lp, (h, w), coeff)
    report = {}
    for mode in ("dense", "chunked", "dense_lp", "chunked_lp", "sparse", "hybrid"):
        result = evaluate(h, w, ids, pos, coeff, 0.7, mode, 11)
        report[mode] = {"logprob": errors(result[0], lp), "dh": errors(result[2], dh), "dw": errors(result[3], dw)}
        assert all(v["max_abs"] < 5e-5 for v in report[mode].values()), report
    return report


@torch.inference_mode()
def collect(model_dir, config, out):
    model_path = Path(model_dir)
    assert model_path.name == config["revision"], "Model snapshot does not match frozen revision."
    assets = {}
    for file in sorted(model_path.iterdir()):
        if file.suffix == ".json" or file.name.endswith(".safetensors"):
            if file.is_file():
                record = {"bytes": file.stat().st_size, "resolved_blob": file.resolve().name}
                if file.suffix == ".json":
                    record["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
                assets[file.name] = record
    save_json(out / "model_assets.json", {"verified_snapshot": model_path.name, "assets": assets})
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    weight = model.get_output_embeddings().weight.detach().clone()
    traces = []
    for setting_index, setting in enumerate(config["sampling"]):
        torch.manual_seed(config["seed"] + setting_index)
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                 add_generation_prompt=True) for p in PROMPTS]
        batch = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
        tokens, attention = batch.input_ids, batch.attention_mask
        cache, hs, supports, positions, counts, active_rows = None, [], [], [], [], []
        alive = torch.ones(len(PROMPTS), dtype=torch.bool, device="cuda")
        for step in range(config["trace_steps"]):
            # Use the backbone directly: its final state is exactly the LM-head input.
            kwargs = model.prepare_inputs_for_generation(tokens, attention_mask=attention,
                                                        past_key_values=cache, use_cache=True)
            kwargs.pop("logits_to_keep", None)
            kwargs.pop("num_logits_to_keep", None)
            result = model.model(**kwargs)
            h = result.last_hidden_state[:, -1].contiguous()
            z = (h @ weight.T).float() / setting["temperature"]
            values, ids = z.topk(setting["top_k"], dim=-1)
            probs = values.softmax(-1)
            remove = probs.cumsum(-1) > setting["top_p"]
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            filtered = values.masked_fill(remove, -torch.inf)
            chosen_pos = torch.multinomial(filtered.softmax(-1), 1).squeeze(1)
            next_token = ids.gather(1, chosen_pos[:, None])
            hs.append(h.cpu()); supports.append(ids.masked_fill(remove, -1).cpu())
            positions.append(chosen_pos.cpu()); counts.append((~remove).sum(-1).cpu())
            active_rows.append(alive.cpu())
            alive &= next_token.squeeze(1) != tokenizer.eos_token_id
            tokens = torch.cat((tokens, next_token), dim=1)
            attention = torch.cat((attention, torch.ones_like(next_token)), dim=1)
            cache = result.past_key_values
        trace = {"setting": setting, "h": torch.cat(hs), "ids": torch.cat(supports),
                 "positions": torch.cat(positions), "counts": torch.cat(counts),
                 "pre_eos": torch.cat(active_rows), "token_ids": tokens.cpu()}
        valid_rows = trace["pre_eos"]
        for field in ("h", "ids", "positions", "counts", "pre_eos"):
            trace[field] = trace[field][valid_rows]
        assert len(trace["h"]) >= max(config["rows"]), "Insufficient pre-EOS positions; do not substitute post-EOS rows."
        torch.save(trace, out / (setting["name"] + "_trace.pt"))
        traces.append(trace)
        del cache, result, tokens
    meta = {"verified_revision": model_path.name, "model_type": model.config.model_type, "vocab": len(weight), "hidden": weight.shape[1],
            "tie_word_embeddings": model.config.tie_word_embeddings, "prompts": PROMPTS,
            "post_eos": "Post-EOS hidden/support positions excluded from benchmark; full diagnostic token sequences retained. Not a production RL rollout bank."}
    del model
    gc.collect(); torch.cuda.empty_cache()
    return weight, traces, meta


def benchmark(h, w, ids, positions, coeff, temp, modes, config):
    stats = {mode: {"round_ms": [], "peak_allocated_mib": []} for mode in modes}
    timer = config["timing"]
    chunk = config["vocab_chunk"]
    for mode in modes:
        for _ in range(timer["warmup"]):
            result = evaluate(h, w, ids, positions, coeff, temp, mode, chunk)
            del result
    rng = random.Random(config["seed"])
    for _ in range(timer["paired_rounds"]):
        order = modes.copy(); rng.shuffle(order)
        for mode in order:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(timer["iterations_per_round"]):
                result = evaluate(h, w, ids, positions, coeff, temp, mode, chunk)
                del result
            end.record(); end.synchronize()
            stats[mode]["round_ms"].append(start.elapsed_time(end) / timer["iterations_per_round"])
            stats[mode]["peak_allocated_mib"].append(torch.cuda.max_memory_allocated() / 2**20)
    import statistics
    for v in stats.values():
        v["median_ms"] = statistics.median(v["round_ms"])
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir")
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = {"pid": os.getpid(), "host": platform.node(), "start_time": time.time(),
                "gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch": torch.__version__, "protocol": config,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(out / "manifest.json", manifest)
    save_json(out / "status.json", {"state": "RUNNING", "phase": "algebra_check", "pid": os.getpid()})
    try:
        sanity = algebra_check(); save_json(out / "sanity.json", sanity)
        if args.smoke_only:
            save_json(out / "status.json", {"state": "COMPLETE", "smoke_only": True, "exit_code": 0})
            print("SMOKE_PASS", flush=True); return
        save_json(out / "status.json", {"state": "RUNNING", "phase": "trace_collection", "pid": os.getpid()})
        w, traces, meta = collect(args.model_dir, config, out)
        save_json(out / "model.json", meta)
        summary = {"arms": [], "claim": "Real-model operator pilot; no RL update or end-to-end throughput measured."}
        for trace in traces:
            for n in config["rows"]:
                h, ids = trace["h"][:n].to("cuda"), trace["ids"][:n].to("cuda")
                pos = trace["positions"][:n].to("cuda")
                # Inference tensors need a normal clone before autograd saves them.
                h, w, ids, pos = h.clone(), w.clone(), ids.clone(), pos.clone()
                coeff = torch.randn(n, device="cuda") / n
                temp = trace["setting"]["temperature"]
                ref = evaluate(h, w, ids, pos, coeff, temp, "dense", config["vocab_chunk"])
                comparisons, passing = {}, []
                for mode in ["chunked", "dense_lp", "chunked_lp", "sparse", "hybrid"]:
                    result = evaluate(h, w, ids, pos, coeff, temp, mode, config["vocab_chunk"])
                    e = {"logprob": errors(result[0], ref[0]), "dh": errors(result[2], ref[2]), "dw": errors(result[3], ref[3])}
                    if mode in ("chunked", "hybrid"): e["entropy"] = errors(result[1], ref[1])
                    gate = config["correctness"]
                    ok = e["logprob"]["max_abs"] <= gate["logprob_max_abs"] and all(e[k]["relative_l2"] <= gate["gradient_relative_l2"] for k in ("dh", "dw"))
                    if "entropy" in e: ok &= e["entropy"]["max_abs"] <= gate["entropy_max_abs"]
                    comparisons[mode] = {"pass": bool(ok), "errors": e}
                    if ok: passing.append(mode)
                    del result
                del ref; torch.cuda.synchronize()
                save_json(out / "status.json", {"state": "RUNNING", "phase": "timing", "setting": trace["setting"]["name"], "rows": n, "pid": os.getpid()})
                timings = benchmark(h, w, ids, pos, coeff, temp, ["dense"] + passing, config)
                baseline_full = min(timings[m]["median_ms"] for m in ["dense", "chunked"] if m in timings)
                baseline_lp = min(timings[m]["median_ms"] for m in ["dense_lp", "chunked_lp"] if m in timings) if any(m in timings for m in ["dense_lp", "chunked_lp"]) else None
                counts = trace["counts"][:n].float()
                arm = {"setting": trace["setting"], "rows": n, "support_mean": counts.mean().item(), "support_min": counts.min().item(), "support_max": counts.max().item(),
                       "pre_eos_fraction": trace["pre_eos"][:n].float().mean().item(), "correctness": comparisons, "timings": timings,
                       "speedups": {m: (baseline_lp if m == "sparse" else baseline_full) / timings[m]["median_ms"] for m in ("sparse", "hybrid") if m in timings and (m != "sparse" or baseline_lp is not None)}}
                summary["arms"].append(arm)
                save_json(out / "summary.json", summary)
                print(json.dumps(arm), flush=True)
        endpoints = [a for a in summary["arms"] if a["rows"] == max(config["rows"])]
        summary["verdict"] = {mode: ("PROVISIONAL_OPERATOR_PASS" if all(a["speedups"].get(mode, 0) >= 1.10 for a in endpoints) else "PROTOTYPE_NO_GO") for mode in ("sparse", "hybrid")}
        save_json(out / "summary.json", summary)
        lines = ["# Qwen 1.5B qualification", "", summary["claim"], "", "| Setting | Rows | Mean support | Sparse speedup | Hybrid speedup |", "|---|---:|---:|---:|---:|"]
        for a in summary["arms"]:
            lines.append(f"| {a['setting']['name']} | {a['rows']} | {a['support_mean']:.1f} | {a['speedups'].get('sparse', 0):.3f}x | {a['speedups'].get('hybrid', 0):.3f}x |")
        lines += ["", json.dumps(summary["verdict"]), "", "Shared-GPU timings, two fixed support configurations, no optimizer or full-model backward. FP32 gradient accumulation differs from BF16 dense reduction order within the frozen tolerance."]
        (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
        save_json(out / "status.json", {"state": "COMPLETE", "exit_code": 0, "end_time": time.time(), "verdict": summary["verdict"]})
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        save_json(out / "status.json", {"state": "FAILED", "exit_code": 1, "end_time": time.time()})
        raise


if __name__ == "__main__":
    main()
