"""Dataset, machine-profile, execution, and reporting commands for STT runs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


DATASETS = {
    "librispeech-clean": {
        "dataset": "openslr/librispeech_asr",
        "config": "clean",
        "split": "test",
        "audio": "audio",
        "text": "text",
    },
    "earnings22": {
        "dataset": "distil-whisper/earnings22",
        "config": "chunked",
        "split": "test",
        "audio": "audio",
        "text": "transcription",
    },
}

PUNCTUATION = re.compile(r"[^\w\s']", flags=re.UNICODE)
EDGE_APOSTROPHES = re.compile(r"(?<!\w)'|'(?!\w)")
WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Use one deliberately simple, shared normalization for ref and hypothesis."""
    text = EDGE_APOSTROPHES.sub(" ", PUNCTUATION.sub(" ", text.lower()))
    return WHITESPACE.sub(" ", text).strip()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_audio_path(manifest: Path, entry: dict[str, Any]) -> Path:
    """Resolve both manifest-relative local clips and repository-relative corpora."""
    relative = Path(entry["audio"])
    local = (manifest.parent / relative).resolve()
    if local.exists():
        return local
    return (manifest.parent.parent.parent / relative).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def machine_profile() -> dict[str, Any]:
    memory_bytes = command_output(["sysctl", "-n", "hw.memsize"])
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "chip": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "cpu_cores": command_output(["sysctl", "-n", "hw.ncpu"]),
        "memory_bytes": int(memory_bytes) if memory_bytes and memory_bytes.isdigit() else None,
        "battery": command_output(["pmset", "-g", "batt"]),
    }


def parquet_urls(spec: dict[str, str]) -> list[str]:
    url = "https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}".format(**spec)
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def mono_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate != 16_000:
        output_length = round(len(samples) * 16_000 / sample_rate)
        samples = np.interp(
            np.linspace(0.0, 1.0, output_length, endpoint=False),
            np.linspace(0.0, 1.0, len(samples), endpoint=False),
            samples,
        )
    return samples.astype(np.float32)


def fetch_dataset(root: Path, name: str, count: int, minimum: float, maximum: float, force: bool) -> None:
    spec = DATASETS[name]
    dataset_dir = root / name
    clips = dataset_dir / "clips"
    manifest = dataset_dir / "refs.jsonl"
    if manifest.exists() and not force:
        print(f"{name}: manifest exists; use --force to rebuild", file=sys.stderr)
        return
    clips.mkdir(parents=True, exist_ok=True)
    shard = root / f"{name}.parquet"
    if force or not shard.exists():
        urls = parquet_urls(spec)
        print(f"{name}: downloading first of {len(urls)} parquet shard(s)", file=sys.stderr)
        urllib.request.urlretrieve(urls[0], shard)
    rows = pq.read_table(shard, columns=[spec["audio"], spec["text"]]).to_pylist()
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if len(records) == count:
            break
        audio, text = row[spec["audio"]], (row[spec["text"]] or "").strip()
        if not text or not audio or not audio.get("bytes"):
            continue
        try:
            samples, rate = sf.read(io.BytesIO(audio["bytes"]))
        except RuntimeError:
            continue
        samples = mono_16k(samples, rate)
        duration = len(samples) / 16_000
        if not minimum <= duration <= maximum:
            continue
        clip_id = f"{name}-{row_index:05d}"
        wav = clips / f"{clip_id}.wav"
        sf.write(wav, samples, 16_000, subtype="PCM_16")
        records.append({
            "id": clip_id,
            "audio": str(wav.relative_to(root.parent)),
            "reference": text,
            "duration_seconds": round(duration, 3),
            "source_row": row_index,
        })
    manifest.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    write_json(dataset_dir / "manifest.json", {
        "dataset": name, "clips": len(records), "duration_range_seconds": [minimum, maximum],
        "manifest": str(manifest.relative_to(root.parent)), "manifest_sha256": digest,
    })
    print(f"{name}: wrote {len(records)} clips; manifest sha256={digest}")


def parse_transcript(stdout: str, result_format: str) -> str:
    if result_format == "muesli-cli-json":
        return json.loads(stdout)["data"]["transcript"]
    if result_format == "plain-text":
        return stdout.strip()
    raise ValueError(f"unsupported result format: {result_format}")


def run_workload(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest).resolve()
    adapter = json.loads(Path(args.adapter).read_text())
    machine = json.loads(Path(args.machine).read_text()) if args.machine else machine_profile()
    template = adapter["command"]
    if "{audio}" not in template or "{model}" not in template:
        raise ValueError("adapter command must contain {audio} and {model}")
    started_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for entry in jsonl(manifest):
        # Manifest paths are repository-relative (for example,
        # `data/librispeech-clean/clips/...`), while the manifest itself lives
        # under `data/<workload>/`.
        audio = resolve_audio_path(manifest, entry)
        command = shlex.split(template.format(audio=audio, model=args.model))
        started = time.monotonic()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        wall_seconds = time.monotonic() - started
        record = {
            "type": "clip", "id": entry["id"], "audio": entry["audio"],
            "duration_seconds": entry["duration_seconds"], "reference": entry["reference"],
            "wall_seconds": round(wall_seconds, 4),
            "rtf": round(wall_seconds / entry["duration_seconds"], 5),
        }
        if completed.returncode:
            record.update({"status": "failed", "error": completed.stderr.strip()[:500]})
        else:
            try:
                hypothesis = parse_transcript(completed.stdout, adapter["result_format"])
                record.update({"status": "ok", "hypothesis": hypothesis, "wer": round(jiwer.wer(normalize(entry["reference"]), normalize(hypothesis)), 6)})
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                record.update({"status": "failed", "error": f"unparseable output: {error}"})
        rows.append(record)
        print(f"{record['id']}: {record['status']} {record['wall_seconds']:.2f}s")
    successful = [row for row in rows if row["status"] == "ok"]
    # Failed attempts are empty hypotheses in the primary quality metric. This
    # makes unsupported input lengths and runtime failures visible instead of
    # letting a model look more accurate by silently dropping them.
    all_refs = [normalize(row["reference"]) for row in rows]
    all_hyps = [normalize(row.get("hypothesis", "")) for row in rows]
    all_measures = jiwer.process_words(all_refs, all_hyps) if rows else None
    refs = [normalize(row["reference"]) for row in successful]
    hyps = [normalize(row["hypothesis"]) for row in successful]
    successful_measures = jiwer.process_words(refs, hyps) if refs else None
    summary = {
        "type": "summary", "started_at": started_at, "finished_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "adapter": adapter["name"], "model": args.model, "runtime": args.runtime, "machine": machine,
        "clips": len(rows), "successful_clips": len(successful), "failed_clips": len(rows) - len(successful),
        "wer_all_attempts": round(all_measures.wer, 6) if all_measures else None,
        "wer_successful_only": round(successful_measures.wer, 6) if successful_measures else None,
        "substitutions": all_measures.substitutions if all_measures else None,
        "deletions": all_measures.deletions if all_measures else None,
        "insertions": all_measures.insertions if all_measures else None,
        "audio_seconds": round(sum(row["duration_seconds"] for row in successful), 3),
        "wall_seconds": round(sum(row["wall_seconds"] for row in rows), 3),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in [*rows, summary]))
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_runner_session(args: argparse.Namespace) -> None:
    """Run one model in a persistent process via the benchmark JSONL protocol."""
    manifest = Path(args.manifest).resolve()
    machine = json.loads(Path(args.machine).read_text()) if args.machine else machine_profile()
    command = shlex.split(args.runner.format(model=args.model, runtime=args.runtime))
    started_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    if not process.stdin or not process.stdout:
        process.kill()
        raise RuntimeError("could not open runner stdin/stdout")

    def request(payload: dict[str, Any]) -> dict[str, Any]:
        process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
        process.stdin.flush()
        response = process.stdout.readline()
        if not response:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"runner exited unexpectedly: {stderr[-500:]}")
        return json.loads(response)

    try:
        ready = request({"type": "initialize", "schema_version": 1, "model": args.model, "runtime": args.runtime})
        if ready.get("type") != "ready":
            raise RuntimeError(f"runner did not become ready: {ready}")
        for entry in jsonl(manifest):
            audio = resolve_audio_path(manifest, entry)
            for pass_index in range(args.warm_runs + 1):
                started = time.monotonic()
                response = request({
                    "type": "transcribe", "id": entry["id"], "audio": str(audio),
                    "pass": pass_index, "reference": entry["reference"],
                })
                wall_seconds = time.monotonic() - started
                hypothesis = str(response.get("transcript", "")).strip()
                status = "ok" if response.get("type") == "result" else "failed"
                row = {
                    "type": "clip", "id": entry["id"], "audio": entry["audio"],
                    "duration_seconds": entry["duration_seconds"], "reference": entry["reference"],
                    "model": args.model, "runtime": args.runtime, "pass": pass_index,
                    "state": "cold" if not rows else "warm",
                    "wall_seconds": round(wall_seconds, 4),
                    "rtf": round(wall_seconds / entry["duration_seconds"], 5),
                    "status": status, "hypothesis": hypothesis,
                    "error": response.get("error") if status == "failed" else None,
                    "wer": round(jiwer.wer(normalize(entry["reference"]), normalize(hypothesis)), 6),
                }
                rows.append(row)
                print(f"{entry['id']} pass={pass_index}: {status} {wall_seconds:.2f}s")
    finally:
        try:
            request({"type": "shutdown"})
        except (BrokenPipeError, RuntimeError, json.JSONDecodeError):
            pass
        process.terminate()
        process.wait(timeout=10)

    successful = [row for row in rows if row["status"] == "ok"]
    all_measures = jiwer.process_words(
        [normalize(row["reference"]) for row in rows],
        [normalize(row["hypothesis"]) for row in rows],
    )
    successful_measures = jiwer.process_words(
        [normalize(row["reference"]) for row in successful],
        [normalize(row["hypothesis"]) for row in successful],
    ) if successful else None
    summary = {
        "type": "summary", "started_at": started_at, "finished_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "runner": command, "model": args.model, "runtime": args.runtime, "machine": machine,
        "warm_runs": args.warm_runs, "attempts": len(rows), "successful_attempts": len(successful),
        "failed_attempts": len(rows) - len(successful),
        "wer_all_attempts": round(all_measures.wer, 6),
        "wer_successful_only": round(successful_measures.wer, 6) if successful_measures else None,
        "wall_seconds": round(sum(row["wall_seconds"] for row in rows), 3),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in [*rows, summary]))
    print(json.dumps(summary, indent=2, sort_keys=True))


def report(path: Path) -> None:
    records = jsonl(path)
    summary = next((record for record in records if record.get("type") == "summary"), None)
    if not summary:
        raise ValueError("result file has no summary record")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(prog="stt-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download fixed public workload samples")
    fetch.add_argument("--data-dir", default="data", type=Path)
    fetch.add_argument("--sets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    fetch.add_argument("--count", type=int, default=40)
    fetch.add_argument("--min-seconds", type=float, default=2.0)
    fetch.add_argument("--max-seconds", type=float, default=40.0)
    fetch.add_argument("--force", action="store_true")
    machine = commands.add_parser("machine", help="capture a machine profile")
    machine.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="run a manifest through an adapter")
    run.add_argument("--manifest", required=True)
    run.add_argument("--adapter", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--runtime", required=True)
    run.add_argument("--machine")
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--output", required=True)
    session = commands.add_parser("session", help="run a persistent JSONL runtime runner")
    session.add_argument("--manifest", required=True)
    session.add_argument("--runner", required=True, help="runner command; may contain {model} and {runtime}")
    session.add_argument("--model", required=True)
    session.add_argument("--runtime", required=True, choices=["coreml", "litert", "gguf", "mlx", "apple-speech"])
    session.add_argument("--machine")
    session.add_argument("--warm-runs", type=int, default=3)
    session.add_argument("--output", required=True)
    report_command = commands.add_parser("report", help="print a result summary")
    report_command.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        for name in args.sets:
            fetch_dataset(args.data_dir, name, args.count, args.min_seconds, args.max_seconds, args.force)
    elif args.command == "machine":
        write_json(args.output, machine_profile())
    elif args.command == "run":
        run_workload(args)
    elif args.command == "session":
        run_runner_session(args)
    else:
        report(args.result)


if __name__ == "__main__":
    main()
