#!/usr/bin/env python3
"""Aggregate linear-probe results across runs into a comparison table.

Scans a linprobe output directory tree (each run's subdirectory holds
run_meta.json + log.txt, written by main_linprobe_siamjepa.py /
util/experiment_tracking.py) and, for each run, looks up the pretrain
run it evaluated (parsed from run_meta.json's args.finetune path, which
points into a pretrain output directory that has its own run_meta.json).
Prints one row per linprobe run: the pretrain hyperparameters, which
checkpoint epoch was evaluated, and the final/best top-1 accuracy from
that linprobe run's own training curve (90 epochs by default).

Usage:
    python3 collect_results.py
    python3 collect_results.py --linprobe_dir output_dir_linprobe --csv results.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def read_log_lines(path):
    lines = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return lines


def find_pretrain_meta(finetune_path):
    """Given a checkpoint path like .../output_dir_siamjepa/run.../checkpoint-100.pth,
    return (epoch, pretrain_run_meta_or_None)."""
    p = Path(finetune_path)
    m = re.search(r"checkpoint-(\d+)\.pth$", p.name)
    epoch = int(m.group(1)) if m else None
    meta = read_json(p.parent / "run_meta.json")
    return epoch, meta


def pretrain_tag(meta):
    if meta is None:
        return "unknown (pretrain run_meta.json not found)"
    pargs = meta.get("args", {})
    git = meta.get("git", {})
    masking = "ijepa" if pargs.get("ijepa_masking") else "scattered"
    return (f"kl={pargs.get('kl_scale')} wd={pargs.get('weight_decay')} "
            f"blr={pargs.get('blr')} mask={masking} "
            f"commit={git.get('short_commit')}")


def collect(linprobe_dir):
    rows = []
    base = Path(linprobe_dir)
    for run_dir in sorted(base.glob("run*")):
        meta = read_json(run_dir / "run_meta.json")
        if meta is None:
            continue
        largs = meta.get("args", {})
        finetune = largs.get("finetune")
        epoch, pmeta = find_pretrain_meta(finetune) if finetune else (None, None)
        log_lines = read_log_lines(run_dir / "log.txt")
        accs = [l["test_acc1"] for l in log_lines if "test_acc1" in l]
        final_acc1 = accs[-1] if accs else None
        best_acc1 = max(accs) if accs else None
        rows.append({
            "pretrain_config": pretrain_tag(pmeta),
            "epoch": epoch,
            "final_top1": round(final_acc1, 2) if final_acc1 is not None else None,
            "best_top1": round(best_acc1, 2) if best_acc1 is not None else None,
            "linprobe_epochs_done": len(log_lines),
            "run_dir": str(run_dir),
        })
    rows.sort(key=lambda r: (r["pretrain_config"], r["epoch"] if r["epoch"] is not None else -1))
    return rows


def print_table(rows):
    headers = ["pretrain_config", "epoch", "final_top1", "best_top1", "linprobe_epochs_done", "run_dir"]
    if not rows:
        print("(no linprobe runs found)")
        return
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    def fmt_row(vals):
        return " | ".join(str(v).ljust(widths[h]) for h, v in zip(headers, vals))

    print(fmt_row(headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(fmt_row([r[h] for h in headers]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linprobe_dir", default="output_dir_linprobe")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = collect(args.linprobe_dir)
    print_table(rows)

    if args.csv:
        headers = ["pretrain_config", "epoch", "final_top1", "best_top1", "linprobe_epochs_done", "run_dir"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
