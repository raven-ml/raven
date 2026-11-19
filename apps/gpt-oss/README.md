# gpt-oss

OpenAI's gpt-oss on raven. Each block pairs grouped-query attention, with
sinks and YaRN positions, with a mixture of experts. The weights load from
the HuggingFace checkpoint as stored, experts packed as MXFP4. The chat goes
through the harmony format. The decode step runs compiled over the devices
`--devices` names, the host by default, expert-parallel over several, or
eagerly on the host with `--eager`.

```bash
dune exec apps/gpt-oss/bin/main.exe -- --prompt "What is the capital of France?"
dune exec apps/gpt-oss/bin/main.exe -- --repo openai/gpt-oss-20b --devices CUDA \
  --prompt "What is the capital of France?" --show-analysis
```

The default checkpoint, `tiny-random/gpt-oss-mxfp4`, is 14 MB of random
weights: it runs every path and says nothing. gpt-oss-20b is 13.8 GB.

| File | |
|---|---|
| `lib/gpt_oss.mli` | the model: configuration, parameters, forward passes, loading |
| `lib/moe.mli` | the experts: routing, the clamped SwiGLU, the routed products |
| `lib/layer_loop.mli` | the model compiled one block program at a time |
| `lib/harmony.mli` | the chat format: rendering a conversation, parsing the reply |
| `bin/main.ml` | the chat |

## Tests

`dune build @apps/gpt-oss/runtest` runs the tests, offline. `test_parity`,
`test_harmony` and `test_stream` compare with transformers and
openai-harmony on the small checkpoints and trimmed tokenizer in
`test/checkpoints`, from fixtures recorded there; the scripts in `test/gen`
write both, with `uv run`.

`test_stream` also compares gpt-oss-20b, block by block, when its weights are
in raven's cache at the revision the fixture records. `runtest` uses a cache
of its own, so it skips that case. To fetch the weights and run it, with
about 17 GB of memory:

```bash
r=6cee5e81ee83917806bbde320786a8fb61efebee
d=~/.cache/raven/huggingface/openai-gpt-oss-20b/$r && mkdir -p $d
for f in config.json model.safetensors.index.json model-0000{0,1,2}-of-00002.safetensors; do
  curl -L -o $d/$f https://huggingface.co/openai/gpt-oss-20b/resolve/$r/$f; done
dune build apps/gpt-oss/test/
cd apps/gpt-oss/test && ../../../_build/default/apps/gpt-oss/test/test_stream.exe
```

## Benchmarks

```bash
dune build @apps/gpt-oss/bench/bench
```

times gpt-oss-20b at its own shapes on random weights, on the first CUDA,
NV or AMD GPU that opens, after compiling every kernel into tolk's cache
(`--warm`). Rows are seconds per call: decode tokens per second is one over
`decode-step-cache-1024`, prefill tokens per second 512 over
`prefill-512`.

| Row | What it times |
|---|---|
| `decode-step-cache-1024` | one decode step at position 512 of a 1,024-token cache, token read back |
| `prefill-512`, `prefill-12` | a prompt of 512 or 12 tokens |
| `…-Q8_0` | the same with the attention projections and head as Q8_0 |
| `first-call-12` (warm, cold, cold serial) | a new process's first 12-token call: time to the first token |
| `experts-at-prefill-N` | one block's experts alone, at N tokens: a component of the step, for tolk's kernels |

`bench/gpt_oss.thumper` holds the baselines: nonnormal (AMD Radeon AI PRO
R9700) for wall time, fudge (NVIDIA RTX 5000 Ada) for allocations. The
comparison with llama.cpp at the same shapes is in
[docs/performance.md](../../docs/performance.md).
