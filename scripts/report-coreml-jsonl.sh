#!/bin/zsh
# Summarize the JSONL emitted by `muesli-cli benchmark` without hiding failed
# attempts. A model's cold value is the first inference in its fresh process;
# warm RTF is aggregated across every later successful invocation.
set -euo pipefail

if (( $# == 0 )); then
  print -u2 "Usage: $0 RESULT.jsonl [RESULT.jsonl ...]"
  exit 2
fi

print '| Model | Cold total (s) | Warm RTF | India WER, all attempts | US WER, all attempts | Successful / total |'
print '| --- | ---: | ---: | ---: | ---: | ---: |'

jq -s -r '
  map(select(.type == "clip"))
  | sort_by(.model)
  | group_by(.model)[]
  | .[0].model as $model
  | [ .[] | select(.state == "cold") ][0] as $cold
  | [ .[] | select(.state == "warm" and .error == null) ] as $warm
  | [ .[] | select(.id == "india-constitution-preamble") | .wer ] as $india_wers
  | [ .[] | select(.id == "us-constitution-preamble") | .wer ] as $us_wers
  | [ .[] | select(.error == null) ] as $successful
  | [
      $model,
      ($cold.totalSeconds * 1000 | round / 1000 | tostring),
      (($warm | map(.transcriptionSeconds) | add) / ($warm | map(.audioSeconds) | add) * 10000 | round / 10000 | tostring),
      (($india_wers | add / length * 10000 | round / 100) | tostring + "%"),
      (($us_wers | add / length * 10000 | round / 100) | tostring + "%"),
      "\($successful | length)/\(length)"
    ]
  | "| " + join(" | ") + " |"
' "$@" | sort
