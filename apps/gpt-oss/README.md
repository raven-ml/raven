# gpt-oss

OpenAI's gpt-oss on raven. Each block pairs grouped-query attention, with
sinks and YaRN positions, with a mixture of experts. The weights load from
the HuggingFace checkpoint as stored, experts packed as MXFP4. The chat goes
through the harmony format. The decode step runs eagerly on the host, or
compiled over the devices `--devices` names, expert-parallel over several.

```bash
dune exec apps/gpt-oss/main.exe -- --prompt "What is the capital of France?"
dune exec apps/gpt-oss/main.exe -- --repo openai/gpt-oss-20b --devices CUDA \
  --prompt "What is the capital of France?" --show-analysis
```

The default checkpoint, `tiny-random/gpt-oss-mxfp4`, is 14 MB of random
weights: it runs every path and says nothing. gpt-oss-20b is 13.8 GB.

| File | |
|---|---|
| `gpt_oss.mli` | the model: configuration, parameters, forward passes, loading |
| `moe.mli` | the experts: routing, the clamped SwiGLU, the routed products |
| `layer_loop.mli` | the model compiled one block program at a time |
| `harmony.mli` | the chat format: rendering a conversation, parsing the reply |
| `main.ml` | the chat |

## Tests

`dune build @apps/gpt-oss/runtest` runs the tests. Some run on small weights
made in place; `test_parity`, `test_harmony` and `test_stream` compare with
transformers and openai-harmony, from fixtures their recorders in `test/gen`
wrote, and download the tiny checkpoints they were recorded from.

`test_stream` also compares gpt-oss-20b, block by block, when its weights are
in raven's cache at the revision the fixture records, as
`Kaun_hf.load_safetensors ~revision` downloads them. `runtest` uses a cache
of its own, so it skips that case. To run it, with about 17 GB of memory:

```bash
dune build apps/gpt-oss/test/test_stream.exe
cd apps/gpt-oss/test && ../../../_build/default/apps/gpt-oss/test/test_stream.exe
```

## Benchmarks

`dune build @apps/gpt-oss/bench/bench` times gpt-oss-20b's decode step and
prefills on a CUDA, NV or AMD GPU, on random weights, against
`bench/gpt_oss.thumper`.
