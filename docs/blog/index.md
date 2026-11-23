# Updates

Progress reports on Raven's development, published a few times a year

- **[November 2026 — gpt-oss-20b in OCaml, level with vLLM](2026-11-19-gpt-oss-in-ocaml.md)**
  raven runs OpenAI's gpt-oss-20b on one workstation GPU at 123 tokens a second, as fast as vLLM, in under 15 GB, from one OCaml executable with no kernel written by hand.

- **[July 2026 — A Compiler, GPUs, and the Road to Beta](2026-07-08-raven-on-gpus.md)**
  Raven runs on GPUs: the tinygrad port is complete, a GPT-2 training step runs at 110 ms on an H100 (tinygrad: 90 ms), Rune and Kaun were rewritten on typed parameter trees, and three new libraries landed (Munin, Vega, Norn). The beta 1 scope is complete.

- **[November 2025 — Alpha Complete, Toward the JIT](2025-11-alpha-complete.md)**
  Alpha 1 shipped with three new libraries (Fehu, Saga, Talon), FFT and a full linear algebra suite in Nx, workshops at FunOCaml, two Outreachy internships kicked off — and the decision that set up everything since: porting tinygrad to OCaml.
