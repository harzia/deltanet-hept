"""Structured logging shared by humans and coding agents (Claude, Codex, ...).

Every run gets its own directory under ``logs/`` containing:

* ``run.log``      -- human-readable, timestamped console mirror.
* ``metrics.jsonl``-- one JSON object per logged event (machine-readable).
* ``config.yaml``  -- the exact resolved config used for the run.
* ``best_model.pt``-- best checkpoint by the main metric (written by the runner).

The JSONL stream is the contract other agents should parse: each line has a
``kind`` field ("meta" | "epoch" | "final" | "event") plus a flat set of scalar
fields, so it is trivial to load with ``pandas.read_json(path, lines=True)``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


class RunLogger:
    def __init__(self, log_dir: Path, config: dict | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "run.log"
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self._log_fh = open(self.log_path, "a", buffering=1)
        self._jsonl_fh = open(self.jsonl_path, "a", buffering=1)
        if config is not None:
            with open(self.log_dir / "config.yaml", "w") as f:
                yaml.safe_dump(config, f, sort_keys=False)

    # -- human-readable line (console + run.log) ----------------------------
    def info(self, *args):
        msg = " ".join(str(a) for a in args)
        line = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        self._log_fh.write(line + "\n")

    # -- machine-readable record (metrics.jsonl) ----------------------------
    def event(self, kind: str, **fields):
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "kind": kind}
        rec.update(fields)
        self._jsonl_fh.write(json.dumps(rec) + "\n")
        return rec

    def epoch(self, epoch, phase, metrics: dict, lr=None):
        """Log an epoch's metrics to both streams."""
        flat = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
        self.event("epoch", epoch=epoch, phase=phase, lr=lr, **flat)
        pretty = ", ".join(f"{k}: {v:.4f}" for k, v in flat.items() if isinstance(v, float))
        self.info(f"[epoch {epoch}] {phase}: {pretty}")

    def close(self):
        for fh in (self._log_fh, self._jsonl_fh):
            try:
                fh.close()
            except Exception:
                pass


def make_log_dir(root: Path, model_name: str, dataset_name: str, note: str, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
    name = f"{stamp}_{model_name}_{dataset_name}_{seed}_{note}"
    return Path(root) / name
