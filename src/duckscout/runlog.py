"""Persist run settings and timestamped records without interpreting the task."""

import hashlib
import json
import platform
import subprocess
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TextIO


def run_metadata(arguments: dict) -> dict:
    """Identify the actual code and inputs, including uncommitted Python edits."""
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = tomllib.loads((root / "assets/sources.toml").read_text())
    status = git("status", "--porcelain")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()
        },
        "asset_revision": manifest["source"]["revision"],
        "policy_reference_revision": manifest["policy_source"]["revision"],
        "code_revision": git("rev-parse", "HEAD"),
        "code_dirty": None if status is None else bool(status),
        "python_sources_sha256": {
            str(p.relative_to(root)): digest(p)
            for p in sorted((root / "src/duckscout").glob("*.py"))
        },
        "inputs_sha256": {
            str(p): digest(p)
            for p in (
                Path(arguments["scene"]),
                *(
                    [Path(arguments["policy"])]
                    if arguments["policy"] is not None
                    else []
                ),
                root / "uv.lock",
                root / "assets/sources.toml",
            )
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "mujoco",
                "numpy",
                "onnxruntime",
                "better-actuator-models",
                "opencv-python-headless",
            )
        },
        "random_seed": None,
        "randomness": "No randomized placements or perturbations are used.",
    }


@dataclass(kw_only=True)
class RunLog:
    """渡された設定・データ・イベントを保存する。状態名の意味や成否は判断しない。"""

    output: Path
    metadata: dict

    stream: TextIO | None = field(init=False, default=None, repr=False)
    # 呼び出し元の時計は持たない。終了時には最後に記録された時刻だけを添える。
    last_time: float | None = field(init=False, default=None)

    def __enter__(self):
        # 同じ名前の実行記録を誤って消さないよう、既存ディレクトリは拒否する。
        self.output.mkdir(parents=True, exist_ok=False)
        self.stream = (self.output / "records.jsonl").open(
            "x", encoding="utf-8", buffering=1
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            reason = "closed"
            if exc_type is not None:
                reason = (
                    "interrupted"
                    if issubclass(exc_type, KeyboardInterrupt)
                    else "error"
                )
            self.write(
                "event",
                None,
                event="exit",
                last_record_time_s=self.last_time,
                reason=reason,
                error=None if exc is None else str(exc),
            )
        finally:
            self.stream.close()
        # エラーは記録したうえで呼び出し元にも伝える。
        return False

    def start(self, config: dict) -> None:
        (self.output / "config.json").write_text(
            json.dumps(
                {**self.metadata, **config, "schema_version": 1},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Recording: {self.output}")

    def write(self, kind: str, time_s: float | None, **values) -> None:
        # 1行1レコードで逐次保存する。リセットによる時刻の巻き戻りもそのまま残す。
        # 行バッファリングにより、終了まで全データをメモリに溜めない。
        line = json.dumps(
            {"kind": kind, "time_s": time_s, **values},
            ensure_ascii=False,
            allow_nan=False,
        )
        self.stream.write(line + "\n")
        self.last_time = time_s
