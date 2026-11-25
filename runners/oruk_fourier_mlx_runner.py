#!/usr/bin/env python3
"""Persistent JSONL MLX ASR runner for Oruk Fourier 2B (mlx/8bit) & Qwen3-ASR.

Follows `runners/PROTOCOL.md`:
- Reads JSONL requests (`initialize`, `transcribe`, `shutdown`) on stdin.
- Emits JSONL responses (`ready`, `result`, `error`) on stdout only.
- Never downloads weights during `initialize`; resolves pre-cached local paths
  (or `--model-dir` / `ORUK_FOURIER_MODEL_DIR`) and returns a structured error
  if weights are not present locally.

Pre-download once (fetching repo-root `config.json` + `mlx/8bit/*` for
canonical Hugging Face Hub attribution):
    hf download oruk/oruk-fourier-2b --include "config.json" --include "mlx/8bit/*"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _find_cached_model_dir(model_id: str, subfolder: str | None) -> Path | None:
    env_override = os.environ.get("ORUK_FOURIER_MODEL_DIR")
    if env_override:
        p = Path(env_override).expanduser()
        if p.exists():
            return p
    direct = Path(model_id).expanduser()
    if direct.exists():
        if subfolder and (direct / subfolder).exists():
            return direct / subfolder
        return direct
    try:
        from huggingface_hub import snapshot_download

        patterns = ["config.json", f"{subfolder}/*"] if subfolder else None
        snap = Path(
            snapshot_download(
                model_id,
                allow_patterns=patterns,
                local_files_only=True,
            )
        )
        return (snap / subfolder) if subfolder else snap
    except Exception:
        return None


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None, help="Pre-downloaded local MLX model directory")
    parser.add_argument("--subfolder", default="mlx/8bit", help="Subfolder inside canonical HF repo")
    parser.add_argument("--language", default="English", help="Target language hint")
    args = parser.parse_args()

    model = None
    runtime_meta: dict[str, Any] = {}

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"type": "error", "error": f"invalid json: {exc}"})
            continue

        rtype = req.get("type")
        if rtype == "shutdown":
            _emit({"type": "shutdown", "status": "ok"})
            break

        if rtype == "initialize":
            model_id = str(req.get("model") or "oruk/oruk-fourier-2b")
            model_dir = (
                Path(args.model_dir).expanduser()
                if args.model_dir
                else _find_cached_model_dir(model_id, args.subfolder or None)
            )
            if not model_dir or not model_dir.exists():
                _emit(
                    {
                        "type": "error",
                        "error": (
                            f"local weights not found for {model_id!r}; pre-download with: "
                            f'hf download {model_id} --include "config.json" --include "{args.subfolder}/*"'
                        ),
                    }
                )
                continue
            try:
                import mlx.core as mx
                from mlx_audio.stt.utils import load_model

                t0 = time.monotonic()
                model = load_model(str(model_dir), model_type="qwen3_asr")
                load_s = round(time.monotonic() - t0, 4)
                runtime_meta = {
                    "backend": "mlx-audio",
                    "device": str(mx.default_device()),
                    "decoding": "greedy",
                    "quantization": "8-bit",
                    "subfolder": args.subfolder,
                    "load_seconds": load_s,
                }
                _emit({"type": "ready", "schema_version": 1, "runtime_metadata": runtime_meta})
            except Exception as exc:  # noqa: BLE001
                _emit({"type": "error", "error": f"failed to initialize model: {exc}"})
            continue

        if rtype == "transcribe":
            if model is None:
                _emit({"type": "error", "error": "runner not initialized"})
                continue
            audio_path = req.get("audio")
            if not audio_path or not Path(audio_path).exists():
                _emit({"type": "error", "error": f"audio file not found: {audio_path}"})
                continue
            try:
                import mlx.core as mx

                res = model.generate(str(audio_path), language=args.language, verbose=False)
                mx.synchronize()
                text = str(getattr(res, "text", res) or "").strip()
                peak_mb = round(mx.get_peak_memory() / (1024**2), 2) if hasattr(mx, "get_peak_memory") else None
                meta = dict(runtime_meta)
                if peak_mb is not None:
                    meta["mlx_peak_memory_mb"] = peak_mb
                _emit({"type": "result", "transcript": text, "runtime_metadata": meta})
            except Exception as exc:  # noqa: BLE001
                _emit({"type": "error", "error": str(exc)})
            continue

        _emit({"type": "error", "error": f"unsupported request type: {rtype}"})


if __name__ == "__main__":
    main()
