from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

from pm25stgnn.config import load_config


def _tee_run(cmd: list[str], log_file: Path, cwd: Path | None = None) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_build", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--dataset", default="data/processed/tensor_dataset.npz")
    args = ap.parse_args()

    cfg = load_config(args.config)

    run_name = args.run_name.strip()
    if not run_name:
        run_name = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    out_root = Path(cfg.outputs_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    pipeline_log = out_root / "pipeline.log"

    python = sys.executable

    if not args.skip_download:
        _tee_run(
            [
                python,
                "-m",
                "pm25stgnn.scripts.download",
                "--config",
                args.config,
                "--start",
                args.start,
                "--end",
                args.end,
            ],
            pipeline_log,
        )

    if not args.skip_build:
        _tee_run(
            [
                python,
                "-m",
                "pm25stgnn.scripts.build_dataset",
                "--config",
                args.config,
                "--start",
                args.start,
                "--end",
                args.end,
                "--out",
                args.dataset,
            ],
            pipeline_log,
        )

    if not args.skip_train:
        for horizon in [24, 48]:
            for model in ["stgnn", "gru"]:
                _tee_run(
                    [
                        python,
                        "-m",
                        "pm25stgnn.train",
                        "--config",
                        args.config,
                        "--horizon",
                        str(horizon),
                        "--model",
                        model,
                        "--dataset",
                        args.dataset,
                        "--run_name",
                        run_name,
                    ],
                    pipeline_log,
                )

    if not args.skip_eval:
        for horizon in [24, 48]:
            for model in ["stgnn", "gru"]:
                ckpt = Path(cfg.outputs_dir) / run_name / f"h{horizon}_{model}" / "best.pt"
                if ckpt.exists():
                    _tee_run(
                        [
                            python,
                            "-m",
                            "pm25stgnn.eval",
                            "--config",
                            args.config,
                            "--horizon",
                            str(horizon),
                            "--model",
                            model,
                            "--dataset",
                            args.dataset,
                            "--checkpoint",
                            str(ckpt),
                        ],
                        pipeline_log,
                    )

    print(f"Done. Logs: {pipeline_log}")


if __name__ == "__main__":
    main()
