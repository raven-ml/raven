# gpt-oss

OpenAI's gpt-oss (2025) on raven: a decoder whose blocks pair grouped-query
attention, with sinks and YaRN positions, with a mixture of experts. The
weights load from the HuggingFace checkpoint as they are stored, the experts
packed as MXFP4; the chat goes through the harmony format; the decode step
runs eagerly on the CPU, or compiled on GPUs, expert-parallel over several.

```bash
dune exec apps/gpt-oss/main.exe -- --prompt "What is the capital of France?"
dune exec apps/gpt-oss/main.exe -- --repo openai/gpt-oss-20b --devices CUDA \
  --prompt "What is the capital of France?" --show-analysis
```

The default checkpoint, `tiny-random/gpt-oss-mxfp4`, is 14 MB of random
weights: it runs every path, and what it says is noise. gpt-oss-20b is 13.8 GB.

| File | What it holds |
|---|---|
| `gpt_oss.mli` | the model: configuration, parameters, placement, forward passes, loading |
| `moe.mli` | the experts: routing, the clamped SwiGLU, the routed products |
| `layer_loop.mli` | the model compiled one block program at a time, called once per layer |
| `harmony.mli` | the chat format: rendering a conversation, parsing what the model writes |
| `main.ml` | the chat: the decode loop, streaming the answer |
| `test/` | fast tests, and the parity programs with their fixtures and recorders |
| `bench/` | gpt-oss-20b's decode step and prefills on a GPU, random weights |

## Tests

`dune build @apps/gpt-oss/runtest` runs the fast tests: the parameters'
structures, loading and tensor-parallel loading, expert and row parallelism
of the experts within a float32 error bound, and the compiled layer loop's
placement and copies, all on small weights made in place.

Parity with the reference implementations runs outside it, since each reads a
downloaded checkpoint. From `apps/gpt-oss/test`:

```bash
dune exec ./validate.exe          # the tiny checkpoints against transformers
dune exec ./validate_text.exe     # harmony against openai-harmony
dune exec ./validate_stream.exe -- fixtures/gpt-oss-20b-stream.json
```

`validate.exe` and `validate_stream.exe` take `--dtype bfloat16`, and
`--devices` or `--device` to compile what they check for a GPU. The fixtures
are recorded by the scripts in `test/gen`, with `uv run`.

### Last results

On fudge (Intel Core Ultra 5 235, two cores, under load), on the CPU,
at 650f321432, 2026-11-18:

| Program | Checks | Worst |
|---|---|---|
| `validate.exe` | 93 passed | relative error 3.7e-07 on the whole model, 2.5e-05 on the rotary tables at position 4095 |
| `validate.exe --dtype bfloat16` | 42 passed | relative error 8.1e-03 against the float32 reference |
| `validate_text.exe` | 88 passed | rendered ids and parsed messages equal |
| `validate_stream.exe fixtures/gpt-oss-20b-stream.json` | 228 passed | streams 8.3e-05 of their rms, logits 3.4e-06; every expert choice and argmax equal; 284 s |

## Benchmarks

`dune build @apps/gpt-oss/bench/bench` times gpt-oss-20b's decode step and
prefills on a CUDA, NV or AMD GPU against the baselines in
`bench/gpt_oss.thumper`. Without such a GPU it measures nothing.
