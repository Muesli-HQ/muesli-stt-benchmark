# Datasets

This directory stores dataset definitions, not audio. Downloaded clips and
references go in the ignored `data/` directory.

The initial public workloads are deliberately small, fixed samples:

- `librispeech-clean`: `openslr/librispeech_asr`, `clean/test`; clean read
  English and the usual quality baseline.
- `earnings22`: `distil-whisper/earnings22`, `chunked/test`; real earnings-call
  audio with compression, accents, and disfluencies.

The fetcher takes the first usable clips within a declared duration range. A
published run records the resulting manifest hash, so a future sampler can be
added without making historical results ambiguous.

Do not commit source audio, raw dataset shards, or third-party model weights.
