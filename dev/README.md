# Dev

Libraries whose design is still converging, built and tested with the rest
of raven.

- A dev library belongs to no package and is never installed. No library
  under `packages/` or `contrib/` depends on one.
- Its interface changes without a `CHANGES.md` entry.
- A library leaves `dev/` for `contrib/` or `packages/` once its design is
  settled.

| Library | What it does |
|---|---|
| [whisper](whisper/) | OpenAI's Whisper: its checkpoints, log-mel features and decoding on any device, with a parity tool against upstream |
