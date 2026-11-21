# AION-1's performance

On an NVIDIA RTX 5000 Ada, raven embeds objects with aion-base 2.0 times as
fast as AION-1's own PyTorch code and computes redshift posteriors 1.6 times
as fast, ahead of PyTorch under `torch.compile` too, and draws DESI spectra
at 0.92 times its speed. As a whole command, `main.exe embed` takes 1.8 s for
64 real objects against 8.0 s for upstream's code. On two CPU cores, raven
embeds 1.4 times as fast as PyTorch eagerly (`--eager`), and at half its
speed compiled, the default.

## The model on a GPU

Objects a second for the model alone: a batch of 16 objects, `main.exe`'s
default, from their tokens on the host to the result read back, in float32
on both sides. Each cell is the median of three rounds and their range.

| | raven, CUDA | raven, NV | PyTorch | PyTorch, `torch.compile` (one run) |
|---|---|---|---|---|
| aion-base: embed | 125 (122–125) | 122 (120–124) | 62.2 (62.0–62.4) | 96.0 |
| aion-base: redshift posterior | 163 (160–164) | 161 (160–161) | 100 (99.4–101) | 137 |
| aion-base: DESI spectrum from Gaia XP, 12 steps | 20.8 (20.5–20.9) | 20.5 (20.5–20.5) | 22.6 (22.2–22.7) | 25.8 |
| aion-large: embed | 36.5 (36.1–36.7) | 36.1 (36.0–36.2) | 21.5 (21.4–21.5) | 31.2 |
| aion-large: redshift posterior | 46.2 (46.2–46.6) | 46.2 (46.1–46.2) | 33.0 (33.0–33.1) | 43.7 |
| aion-large: DESI spectrum from Gaia XP, 12 steps | 5.32 (5.32–5.33) | 5.26 (5.26–5.28) | 7.13 (7.10–7.26) | 8.04 |

`embed` encodes the tutorial file's inputs, a Legacy Survey image, nine
Legacy Survey values and a DESI spectrum (858 tokens), and averages the
encoder's output; the posterior is redshift's 1025 classes from the image
and the g, r, i and z fluxes (581 tokens); a spectrum is 273 tokens drawn in
upstream's 12 linear steps from the 110 tokens of Gaia's XP coefficients.
NV is the same card through NVIDIA's kernel driver, without CUDA.
`torch.compile` of the transformer's encoder and decoder narrows the gap:
raven still embeds 1.3 times as fast and computes posteriors 1.2 times as
fast as compiled PyTorch, which draws spectra 1.24 times as fast as raven;
upstream ships it eager.

At 64 objects a call, the tutorial's batch, raven embeds 117 objects a
second against PyTorch's 62, computes 151 posteriors against 100, and draws
17.4 spectra against 23.5 (one run each).

One object at a time, in milliseconds a call:

| aion-base | raven, CUDA | raven, NV | PyTorch |
|---|---|---|---|
| embed | 10.7 (10.1–11.0) | 10.9 (10.8–10.9) | 13.8 (13.8–13.9) |
| redshift posterior | 10.1 (9.98–10.2) | 10.4 (10.4–10.5) | 11.0 (10.9–11.2) |
| DESI spectrum, 12 steps | 87.7 (86.9–87.8) | 90.8 (90.8–91.5) | 100 (99.8–100) |

## The model on the CPU

Objects a second on two cores of an Intel Core Ultra 5 235: embed and the
posterior at 4 objects a call, a spectrum one at a time.

| aion-base | raven, compiled (default) | raven, eager (`--eager`) | PyTorch |
|---|---|---|---|
| embed | 0.442 (0.440–0.467) | 1.22 (1.20–1.29) | 0.890 (0.851–0.890) |
| redshift posterior | 0.842 (0.840–0.895) | 1.67 (1.65–1.75) | 1.39 (1.36–1.45) |
| DESI spectrum, 12 steps | 0.044 (0.044–0.047) | 0.212 (0.210–0.222) | 0.347 (0.344–0.362) |

## The codecs

Objects a second, from data on the host to tokens read back, or tokens to a
spectrum on the codec's grid: 16 objects a call on the GPU, 4 on the CPU.

| | raven, CUDA | PyTorch, CUDA | raven, CPU compiled (default) | raven, CPU eager (`--eager`) | PyTorch, CPU |
|---|---|---|---|---|---|
| Image encode, 4 × 160 × 160 | 219 (219–220) | 727 (725–733) | 2.19 (2.17–2.30) | 3.54 (3.51–3.70) | 4.22 (4.12–4.43) |
| Spectrum encode, 7781 pixels | 797 (796–806) | 402 (402–403) | 9.60 (9.18–10.1) | 6.02 (5.86–6.24) | 8.63 (8.34–9.09) |
| Spectrum decode, 273 tokens | 2,309 (2,273–2,356) | 859 (858–860) | 34.7 (33.4–38.2) | 17.4 (16.9–18.3) | 27.4 (27.0–28.7) |

## Where the time goes

Device time of one call at 16 objects with aion-base on CUDA, by kernel:
raven's from tolk's `DEBUG=2` timing of each kernel, PyTorch's from
`torch.profiler`.

| | raven, embed | PyTorch, embed | raven, spectrum | PyTorch, spectrum |
|---|---|---|---|---|
| Products | 88 ms, 27.4 TFLOP/s | 77 ms, 30.3 TFLOP/s | 659 ms, 24.6 TFLOP/s | 396 ms, 25.0 TFLOP/s |
| Attention | 27 ms | 59 ms | 53 ms | 77 ms |
| Norms, elementwise, small reductions | 12 ms | 120 ms | 81 ms | 206 ms |
| In all | 128 ms | 257 ms | 793 ms | 679 ms |

- **Embedding and the posterior.** raven's products run at nine tenths of
  cuBLAS's rate; it fuses the norms and elementwise work into a tenth of
  PyTorch's time, and its attention takes half of PyTorch's, which upstream
  computes as two batched products and a softmax.
- **Spectra.** raven's draw encodes all 383 tokens at each of the 12 steps,
  the target's absent ones included; upstream's encodes the present ones
  only, 110 at the first step and 361 at the last, 236 on average. raven's
  encoder computes 1.6 times the rows, at cuBLAS's rate, which makes 1.7
  times PyTorch's product time. aion-large's products run at 21.4 TFLOP/s
  against cuBLAS's 25.2.
- **The codecs.** raven's image codec runs its convolutions in float32, 67
  ms of kernels for 16 images; PyTorch's run on cuDNN with TF32, cuDNN's
  default, in 20 ms, so that part of upstream is not IEEE float32.
- **The CPU.** Eager raven draws spectra slower than PyTorch for the same 1.6
  times the encoder's rows. Compiled, embed spends 85% of its kernel time
  in products and 14% in attention (`DEBUG=2` at dee7042d32, 4 objects),
  where eager raven runs nx's own kernels 2.8 times as fast. Other users'
  jobs shared the host's cores at times, which the ranges show.

## Whole commands, starting and memory

Seconds for a new process from its start to its exit: `main.exe` and
`bench/upstream.py`, the tutorial's code, each loading aion-base's released
weights from a warm page cache, reading real rows of the Multimodal Universe
with h5py or `Survey`, and writing an npz file. 64 objects of the tutorial's
file and 64 stars of Gaia's healpix 1631 on the GPU, 16 objects and 4 stars
on the CPU, in batches of 16; raven's kernels come from its disk cache.

| | raven, CUDA | PyTorch, CUDA | raven, CPU compiled (default) | raven, CPU eager (`--eager`) | PyTorch, CPU |
|---|---|---|---|---|---|
| `embed`, image, spectrum and 9 values | 1.80 (1.80–1.82) | 7.95 (7.88–7.98) | 43.3 (43.3–43.3) | 24.6 (23.7–28.2) | 32.6 (32.4–33.3) |
| `posterior` of redshift, image and 4 fluxes | 1.75 (1.72–1.88) | 5.99 (5.96–6.02) | 24.1 (24.0–24.2) | 18.9 (17.5–44.3) | 20.6 (20.5–20.7) |
| `generate` of a DESI spectrum from XP, 12 steps | 5.08 (5.04–5.12) | 7.66 (7.66–7.66) | 41.8 (41.8–41.9) | 22.7 (22.1–71.8) | 17.5 (17.3–17.6) |
| `embed` over an empty kernel cache | 9.88 (9.73–9.88) | | 48.3 (48.2–48.4) | nothing to compile | |

On the GPU, besides the model's time at the tables' rates, `main.exe`'s
`embed` takes 1.2 s to start, load the weights, read the rows and run the
codecs, which it runs on the GPU as PyTorch does; PyTorch's process takes
6.9 s. `generate` draws spectra slower (The model on a GPU) but starts
faster, so it finishes first. Compiling raven's kernels over an empty cache
adds 8 s to `embed`, 5 s on the CPU; PyTorch ships its kernels compiled.
On the CPU, compiled, `main.exe` takes 1.2 to 2.4 times PyTorch's time, as
its compiled model rows do (The model on the CPU); eager, the commands
follow the eager rows. Other users' jobs ran on the timing cores during the
eager commands' third round, at a load of 44, which the tops of their
ranges show.

Device memory while embedding 16 objects, the most each side held,
allocator caches included: raven 1.1 GiB with aion-base and 3.1 GiB with
aion-large, PyTorch 3.3 and 6.1 GiB. raven's bench transformer leaves out
the tables of the 25 modalities the rows do not use, 0.29 GiB of aion-base,
which PyTorch's holds.

## How it was measured

- **Machine.** fudge: Intel Core Ultra 5 235, NVIDIA RTX 5000 Ada (sm_89,
  32 GB), driver 615.71.09, CUDA 13.4, Debian 13. Every timed process ran on
  two cores (`taskset`), torch with two threads, the CPU rows with the GPU
  hidden. Other users' jobs shared the host, at loads from 1 to 44, never the
  GPU: its process list was empty before and after every GPU run.
- **Versions.** raven at e4e0b64ea7 for the GPU's rows and their
  breakdowns; at eb04a315af for the CPU's rows and the compiled host's whole
  commands; at 50d61418f0, after `main.exe` moved its codecs to the device
  and compiled by default, for the other whole commands. AION-1
  (`polymathic-aion`) at db7fba6d on torch 2.15.0+cu132, Python 3.12.15,
  cuDNN 9.26. The whole
  commands load polymathic-ai/aion-base at 40541618, the Hub's current
  commit, which `Hub` pins.
- **Statistic.** Three rounds on 2026-11-20 and 21, each running raven and
  PyTorch in turn on one device, in alternate orders. raven's cells are
  thumper's medians (5 samples on the GPU, 3 on the host, after 3 warmup
  calls), the others `pytorch.py`'s (5 calls on the GPU, 3 on the host,
  after 1 or 2); the tables give the median of the three rounds and their
  range. `torch.compile`, beside eba27a14be's rounds, and the batch of 64
  ran once. The whole commands ran three rounds of `whole.sh` each,
  PyTorch's CPU cells in the compiled commands' rounds. The breakdowns are
  one profiled call each, outside the timed rounds.
- **Same work.** The rows run at random weights of the released
  architectures, since the time does not depend on the values; the whole
  commands load the released weights. Both sides compute in float32, and
  torch's products at full precision, its default; cuDNN's convolutions,
  in PyTorch's codecs, use TF32, cuDNN's default. Upstream's embedding
  averages all of an object's tokens and raven's its present ones, which
  here are all of them. Upstream draws one order of the spectrum's
  positions for the batch and raven one per object, and upstream's encoder
  reads only the present tokens (Where the time goes).
- **Not measured.** aion-large on the CPU and in whole commands; the whole
  commands through NV; a breakdown of eager raven on the CPU, whose nx
  kernels report no times, so the CPU codecs' gaps are not attributed;
  bfloat16, for which upstream ships no path; `torch.compile` on the CPU;
  AMD and Apple GPUs; aion-xlarge, not released.

```bash
AION_DEVICE=cuda dune build @apps/aion/bench/bench    # nv for NV
uv run apps/aion/bench/pytorch.py cuda                # TORCH_COMPILE=1 to compile
AION_BATCH=4 dune build @apps/aion/bench/bench    # AION_EAGER=1 for --eager
AION_BATCH=4 uv run apps/aion/bench/pytorch.py cpu
uv run apps/aion/bench/fetch.py DATA
apps/aion/bench/whole.sh DATA DIR cuda                # cpu on the host
DEBUG=2 AION_DEVICE=cuda _build/default/apps/aion/bench/bench_aion.exe \
  --profile embed/base/16 2>&1 | python3 apps/aion/bench/roles.py
uv run apps/aion/bench/pytorch.py cuda --profile embed/base/16
```

`AION_BATCH=64` runs the batch of 64. On the CPU, run the commands under
`taskset -c 2,3`.
