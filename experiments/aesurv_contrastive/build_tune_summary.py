#!/usr/bin/env python3
"""Build tune_summary.json + best_config.json from finished grid run folders."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.aesurv_contrastive.run_tune_grid import CONTRAST_SPECS, _load_meta, _summarize_runs


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=str, default="runs/aesurv_contrastive_tune")
    args = p.parse_args()
    out_root = Path(args.out_root)
    runs = []
    for sp in CONTRAST_SPECS:
        d = out_root / sp["id"]
        if (d / "aesurv_aux_contrastive_run_meta.json").exists():
            runs.append({"id": sp["id"], "out_dir": str(d), "phase": "contrast", **sp, **_load_meta(d)})
    if not runs:
        raise SystemExit(f"No completed runs under {out_root}")
    summary = _summarize_runs(out_root, runs)
    print(json.dumps(summary.get("best_fhs_val"), indent=2))


if __name__ == "__main__":
    main()
