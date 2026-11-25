#!/usr/bin/env python3
"""Minimal plain-text adapter wrapping transcribe-cli for stt-benchmark."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--audio", required=True, help="Path to 16 kHz mono WAV")
    parser.add_argument("--language", default="en")
    parser.add_argument("--backend", default="metal")
    args = parser.parse_args()

    cli = os.environ.get("TRANSCRIBE_CLI", "transcribe-cli")
    with tempfile.TemporaryDirectory() as tmp:
        out_txt = Path(tmp) / "transcript.txt"
        cmd = [
            cli,
            "-q",
            "--backend",
            args.backend,
            "--language",
            args.language,
            "-m",
            args.model,
            "-o",
            str(out_txt),
            args.audio,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr or proc.stdout)
            sys.exit(proc.returncode)
        text = out_txt.read_text(encoding="utf-8").strip() if out_txt.exists() else ""
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
