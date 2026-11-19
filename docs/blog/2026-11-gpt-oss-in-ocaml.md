# gpt-oss-20b in OCaml, level with vLLM

raven runs OpenAI's open-weight mixture of experts on one workstation GPU at 123 tokens a second, as fast as vLLM, in under 15 GB, and prints its first token two seconds after the process starts.

<!-- TODO: paste the transcript from fudge (hero.txt) verbatim, HTML-escaped. -->
<figure class="code">
<figcaption><span>main.exe, gpt-oss-20b on an RTX 5000 Ada</span></figcaption>
<pre tabindex="0" role="region" aria-label="A chat with gpt-oss-20b in a terminal"><mark class="todo">transcript</mark></pre>
<figcaption>The analysis channel, then the answer, as the terminal printed them.</figcaption>
</figure>

<div class="kpis">
<div><b>123<small>tok/s</small></b><span>decode, against vLLM’s 122</span></div>
<div><b>2.0<small>s</small></b><span>to the first token, against vLLM’s 19.7</span></div>
<div><b>14.4<small>GB</small></b><span>of device memory, against vLLM’s 32.4</span></div>
<div><b>463<small>lines</small></b><span>of OCaml, no kernel written by hand</span></div>
</div>

gpt-oss-20b is OpenAI's open-weight model for a single machine: 21 billion parameters, of which a mixture of experts activates 3.6 billion for each token. The experts ship in MXFP4, four bits a weight, so the model fits on one workstation GPU. Running it fast has meant one of two stacks: llama.cpp, a C++ engine with CUDA kernels written for each operation, or vLLM, a Python serving system on PyTorch.

raven runs it from an OCaml program, at the same speed.

## Decoding as fast as vLLM

On an NVIDIA RTX 5000 Ada, raven decodes **123 tokens a second**, against vLLM's 122. With the attention weights packed as llama.cpp packs them, it reaches 163, within 3% of llama.cpp's 169, and level with it when raven drives the card through NVIDIA's kernel driver directly, without CUDA.

<figure class="chart">
<div class="legend" aria-hidden="true"><span><i style="background: var(--raven)"></i>raven</span><span><i style="background: var(--ref)"></i>other stacks</span><span>tokens a second, one sequence · longer is better</span></div>
<ul class="bars">
<li><div class="bars-head"><span>Decode, weights as the checkpoint stores them</span><span class="caption">level with vLLM</span></div>
<div class="bar"><span class="who">raven</span><span class="track"><span class="fill" style="width: 100.0%; background: var(--raven)"></span></span><span class="v">123</span></div>
<div class="bar"><span class="who">vLLM</span><span class="track"><span class="fill" style="width: 99.3%; background: var(--ref)"></span></span><span class="v">122</span></div>
<div class="bar"><span class="who">transformers</span><span class="track"><span class="fill" style="width: 29.1%; background: var(--ref)"></span></span><span class="v">35.8</span></div>
</li>
<li><div class="bars-head"><span>Decode, attention packed as Q8_0</span><span class="caption">3% behind llama.cpp</span></div>
<div class="bar"><span class="who">raven, CUDA</span><span class="track"><span class="fill" style="width: 96.6%; background: var(--raven)"></span></span><span class="v">163</span></div>
<div class="bar"><span class="who">raven, NV</span><span class="track"><span class="fill" style="width: 100.0%; background: var(--raven)"></span></span><span class="v">169</span></div>
<div class="bar"><span class="who">llama.cpp</span><span class="track"><span class="fill" style="width: 99.6%; background: var(--ref)"></span></span><span class="v">169</span></div>
</li>
<li><div class="bars-head"><span>Prefill, a 512-token prompt</span><span class="caption">5% behind vLLM</span></div>
<div class="bar"><span class="who">raven</span><span class="track"><span class="fill" style="width: 94.7%; background: var(--raven)"></span></span><span class="v">12,140</span></div>
<div class="bar"><span class="who">vLLM</span><span class="track"><span class="fill" style="width: 100.0%; background: var(--ref)"></span></span><span class="v">12,820</span></div>
<div class="bar"><span class="who">llama.cpp</span><span class="track"><span class="fill" style="width: 75.0%; background: var(--ref)"></span></span><span class="v">9,620</span></div>
<div class="bar"><span class="who">transformers</span><span class="track"><span class="fill" style="width: 42.7%; background: var(--ref)"></span></span><span class="v">5,480</span></div>
</li>
</ul>
<figcaption>gpt-oss-20b on an NVIDIA RTX 5000 Ada, batch 1, greedy; the median of three rounds.</figcaption>
</figure>

Packing is one argument at load time. `Gpt_oss.of_hf ~pack:Q8_0` quantizes the attention projections and the head with `Nx_quant.quantize_q8_0` as the checkpoint loads, and the experts stay in MXFP4, as the checkpoint stores them. There is no conversion step and no second file.

## Ready in two seconds

A new raven process loads the 13.8 GB checkpoint and prints its first token in **2.0 seconds**; llama.cpp takes 2.1. vLLM needs 19.7 seconds once its compilation cache is warm, and 110 the first time on a machine. raven compiles its own kernels too, and caches them: an empty cache adds 1.9 seconds, once.

It also stays small. Decoding takes **14.4 GB** of the card's memory, the CUDA context included, where vLLM reserves 32.4 GB, 90% of the card, by default.

## An ordinary program

None of this is a separate engine. The model is **463 lines of OCaml** over arrays, written the way the paper draws it, and Rune.jit compiles each block into kernels for the card it runs on. The kernels come from tolk, raven's compiler, a port of tinygrad to OCaml. Nobody wrote a kernel for gpt-oss.

<!-- TODO: confirm on fudge which kernels vLLM 0.34 picks on Ada. -->
The implementations that run gpt-oss fast get there with kernels written for its formats: llama.cpp's CUDA, and in vLLM, Marlin's kernels for the MXFP4 experts and Triton kernels for attention. raven matches them with the model's code alone.

<figure class="code">
<figcaption><span>apps/gpt-oss/lib/gpt_oss.ml</span><a href="../../apps/gpt-oss/lib/gpt_oss.ml" class="note u">source ↗</a></figcaption>
<pre tabindex="0" role="region" aria-label="gpt_oss.ml excerpt"><!-- slot: excerpt apps/gpt-oss/lib/gpt_oss.ml block --></pre>
</figure>

A whole transformer block. The only line about devices is `close`, which sums partial results when the experts are split across GPUs: the same code runs eagerly on a laptop's CPU, compiled on one GPU, or split by expert across several.

<!-- TODO: run test_stream's gpt-oss-20b case on fudge and see it pass before publishing this sentence. -->
It gives the same answers as the reference. On three prompts at float32, raven picks the same experts as transformers for every token in every block, and the same next token at every position.

## The tokenizer is OCaml too

Text reaches the model through [brot](../../packages/brot/README.md), raven's tokenizer. It reads the tokenizer.json that Hugging Face publishes, gives the same ids as Hugging Face's tokenizers on gpt-oss's o200k vocabulary and eleven others, and decodes one token at a time, so the chat prints each word as the model writes it, in the harmony format's channels.

Instead of running the pre-tokenizer's regular expression, brot recognises the o200k, cl100k and GPT-2 patterns and walks the bytes with a scanner written for each, fused with the merges that follow. On OpenWebText, on one core, it tokenizes faster than gigatoken, a Rust tokenizer built for throughput, on two of three vocabularies, and **23 times faster** than Hugging Face's on BERT's.

<figure class="chart">
<div class="legend" aria-hidden="true"><span><i style="background: var(--raven)"></i>raven</span><span><i style="background: var(--ref)"></i>other tokenizers</span><span>megabytes of text a second, one thread · longer is better</span></div>
<ul class="bars">
<li><div class="bars-head"><span>LLaMA</span><span class="caption">2.7× gigatoken</span></div>
<div class="bar"><span class="who">brot</span><span class="track"><span class="fill" style="width: 100.0%; background: var(--raven)"></span></span><span class="v">190</span></div>
<div class="bar"><span class="who">gigatoken</span><span class="track"><span class="fill" style="width: 37.6%; background: var(--ref)"></span></span><span class="v">71.4</span></div>
</li>
<li><div class="bars-head"><span>Mistral</span><span class="caption">1.4× gigatoken</span></div>
<div class="bar"><span class="who">brot</span><span class="track"><span class="fill" style="width: 100.0%; background: var(--raven)"></span></span><span class="v">204</span></div>
<div class="bar"><span class="who">gigatoken</span><span class="track"><span class="fill" style="width: 71.4%; background: var(--ref)"></span></span><span class="v">145</span></div>
</li>
<li><div class="bars-head"><span>GPT-2</span><span class="caption">gigatoken 1.7× faster</span></div>
<div class="bar"><span class="who">brot</span><span class="track"><span class="fill" style="width: 58.1%; background: var(--raven)"></span></span><span class="v">474</span></div>
<div class="bar"><span class="who">gigatoken</span><span class="track"><span class="fill" style="width: 100.0%; background: var(--ref)"></span></span><span class="v">817</span></div>
</li>
<li><div class="bars-head"><span>BERT-base</span><span class="caption">23× HF tokenizers</span></div>
<div class="bar"><span class="who">brot</span><span class="track"><span class="fill" style="width: 100.0%; background: var(--raven)"></span></span><span class="v">62.7</span></div>
<div class="bar"><span class="who">HF tokenizers</span><span class="track"><span class="fill" style="width: 4.3%; background: var(--ref)"></span></span><span class="v">2.7</span></div>
</li>
</ul>
<figcaption>OpenWebText, 100 MB, the same ids from both. gigatoken refuses WordPiece, so BERT-base runs against Hugging Face’s tokenizers.</figcaption>
</figure>

<!-- TODO: brot against tiktoken on o200k, gpt-oss's own vocabulary. -->

## One executable, down to the driver

vLLM is a Python program on PyTorch. To run gpt-oss-20b it installs 9.32 GB, most of it PyTorch and NVIDIA's libraries, at versions that must agree with each other and with the driver. All of it has to be installed wherever the model runs. Hugging Face's transformers installs 5.74 GB, and llama.cpp, a C++ program, 1.15 GB with NVIDIA's cuBLAS.

raven's gpt-oss is one OCaml executable: 32 MB, stripped, linking only the C library. It loads NVRTC, 134 MB, only to compile a kernel that isn't in tolk's cache. Under it is [rig](../../packages/rig/README.md), raven's GPU runtime, which does the work of NVIDIA's closed libcuda and of AMD's runtime itself, in OCaml and C, and talks to the kernel driver directly. Through NVIDIA's kernel driver, without CUDA, gpt-oss needs no libcuda and decodes at **128 tokens a second**, 4% faster than through CUDA. On Linux, rig opens an AMD GPU with no kernel driver at all.

<!-- TODO: the number of Python packages in vLLM's environment (uv pip list), and each layer's size if measured. -->
<figure class="versus">
<div><p class="label">vLLM</p><ol><li><span>Python 3.12</span><span></span></li><li><span>PyTorch 2.14</span><span></span></li><li><span>CUDA 13’s libraries</span><span></span></li><li><span>vLLM 0.34</span><span></span></li><li><span><mark class="todo">N</mark> other Python packages</span><span></span></li><li><span>NVIDIA’s libcuda</span><span>driver</span></li></ol><p class="total"><span>to install</span><b>9.32 GB</b></p></div>
<div><p class="label">raven</p><ol><li><span>main.exe</span><span>32 MB</span></li><li><span>NVRTC, only for kernels not yet cached</span><span>134 MB</span></li><li><span>NVIDIA’s libcuda</span><span>optional</span></li></ol><p class="total"><span>to install</span><b>166 MB</b></p></div>
<figcaption>What runs gpt-oss-20b on the GPU, the driver and the 13.8 GB of weights aside. raven needs no libcuda: without it, rig drives the card through NVIDIA’s kernel driver itself, and it uses libcuda only where it is installed. raven’s main.exe, stripped, and CUDA’s NVRTC; vLLM 0.34.0 in a fresh uv environment with its bench pins, torch’s CUDA libraries and Python 3.12. RTX 5000 Ada, 19 November 2026.</figcaption>
</figure>

## What's next

gpt-oss-20b is a goal post on [our road to 1.0](../roadmap.md): the model, jit-compiled, at least as fast as llama.cpp, vLLM, tinygrad and JAX on NVIDIA, through CUDA and through NVIDIA's kernel driver, and on AMD. Decoding level with vLLM and llama.cpp is the first stretch of it. The rest is still ahead: prefill is 5% behind vLLM, and 31% slower through NVIDIA's kernel driver than through CUDA; llama.cpp decodes 3% faster through CUDA; and the AMD card is not measured yet.

It is also a stepping stone to the goal post after it: serving open-weight models behind an OpenAI-compatible API, from one static binary with nothing to install beyond the GPU's kernel driver, at 90% or better of vLLM's throughput. The executable here is the start of that binary. Ahead-of-time compilation will embed its kernels, which removes NVRTC, its last dependency; an inference engine will add the batching that serving needs, which this post doesn't measure; and the models grow to Llama 3.1 70B and the DeepSeek-V4 family.

## Try it

```sh
dune exec apps/gpt-oss/bin/main.exe -- --repo openai/gpt-oss-20b \
  --devices CUDA --prompt "What is the capital of France?"
```

It downloads the 13.8 GB checkpoint on the first run, and without `--devices` runs on the CPU.

## Thanks

Thank you to [Ahrefs](https://ahrefs.com), whose sponsorship supports the development of Raven. Work like this, a compiler, a GPU runtime and a tokenizer brought level with the fastest stacks there are, takes long stretches of focused time, and their backing is what makes it possible.

## How we measured

On 19 November 2026, on an Intel Core Ultra 5 235 with an NVIDIA RTX 5000 Ada (32 GB), driver 615.71.09, CUDA 13.4. raven at 3c1daa79d5, llama.cpp b12541, vLLM 0.34.0 and transformers 5.23.0, all on openai/gpt-oss-20b at 6cee5e81, every one keeping the experts in MXFP4. Each figure is the median of three rounds, run in a different order each time. [Every number, the scripts, and how to reproduce them](../../apps/gpt-oss/PERFORMANCE.md).
