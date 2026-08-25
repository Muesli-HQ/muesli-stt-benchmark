#!/bin/zsh
# Run one Muesli Core ML ASR model per process. Process exit releases model
# managers; the idle interval then separates consecutive-model measurements.
set -euo pipefail

usage() {
  print "Usage: $0 --muesli-cli PATH --manifest PATH --output-dir DIR [--cooldown-seconds N] [--warm-runs N] [--models a,b,c]"
}

cli=""
manifest=""
output_dir=""
cooldown_seconds=60
warm_runs=3
models="parakeet-unified,parakeet-v3,parakeet-v2,parakeet-eou-320ms,sensevoice,qwen3-asr,nemotron35"

while (( $# > 0 )); do
  case "$1" in
    --muesli-cli) cli="$2"; shift 2 ;;
    --manifest) manifest="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --cooldown-seconds) cooldown_seconds="$2"; shift 2 ;;
    --warm-runs) warm_runs="$2"; shift 2 ;;
    --models) models="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) print -u2 "Unknown option: $1"; usage; exit 2 ;;
  esac
done

[[ -n "$cli" && -x "$cli" ]] || { print -u2 "--muesli-cli must be an executable"; exit 2; }
[[ -n "$manifest" && -f "$manifest" ]] || { print -u2 "--manifest must be a file"; exit 2; }
[[ -n "$output_dir" ]] || { print -u2 "--output-dir is required"; exit 2; }
[[ "$cooldown_seconds" == <-> ]] || { print -u2 "--cooldown-seconds must be a non-negative integer"; exit 2; }
[[ "$warm_runs" == <-> ]] || { print -u2 "--warm-runs must be a non-negative integer"; exit 2; }

mkdir -p "$output_dir"
typeset -a model_list
model_list=( ${(s:,:)models} )

for index in {1..${#model_list}}; do
  model="${model_list[$index]}"
  print "==> Core ML benchmark: $model"
  "$cli" benchmark \
    --manifest "$manifest" \
    --model "$model" \
    --warm-runs "$warm_runs" \
    --output "$output_dir/constitution--$model--coreml.jsonl"

  if (( index < ${#model_list} && cooldown_seconds > 0 )); then
    print "==> Releasing $model; idling ${cooldown_seconds}s before ${model_list[$((index + 1))]}"
    sleep "$cooldown_seconds"
  fi
done
