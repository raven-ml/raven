# AION-1's performance

On an NVIDIA RTX 5000 Ada, raven embeds objects with aion-base 1.6 times as
fast as AION-1's own PyTorch code and computes redshift posteriors 1.3 times
as fast (level with PyTorch under `torch.compile`), and draws DESI spectra at
0.74 times its speed. As a whole command, `main.exe embed` takes 1.8 s for
64 real objects against 8.0 s for upstream's code. On two CPU cores, raven
embeds 1.4 times as fast as PyTorch eagerly (`--eager`), and at a third of
its speed compiled, the default.

## The model on a GPU

Objects a second for the model alone: a batch of 16 objects, `main.exe`'s
default, from their tokens on the host to the result read back, in float32
on both sides. Each cell is the median of three rounds and their range.

| | raven, CUDA | raven, NV | PyTorch | PyTorch, `torch.compile` (one run) |
|---|---|---|---|---|
| aion-base: embed | 101 (99.2–102) | 99.9 (99.4–100) | 61.6 (61.4–62.0) | 96.0 |
| aion-base: redshift posterior | 133 (132–135) | 132 (131–132) | 99.1 (98.6–99.9) | 137 |
| aion-base: DESI spectrum from Gaia XP, 12 steps | 16.4 (16.3–16.5) | 16.2 (16.2–16.3) | 22.2 (22.0–22.2) | 25.8 |
| aion-large: embed | 29.2 (29.1–29.5) | 29.0 (28.9–29.1) | 21.3 (21.2–21.4) | 31.2 |
| aion-large: redshift posterior | 38.1 (38.0–38.3) | 38.0 (37.8–38.0) | 32.7 (32.7–32.9) | 43.7 |
| aion-large: DESI spectrum from Gaia XP, 12 steps | 4.27 (4.26–4.28) | 3.64 (3.64–3.65) | 7.03 (6.98–7.07) | 8.04 |

`embed` encodes the tutorial file's inputs, a Legacy Survey image, nine
Legacy Survey values and a DESI spectrum (858 tokens), and averages the
encoder's output; the posterior is redshift's 1025 classes from the image
and the g, r, i and z fluxes (581 tokens); a spectrum is 273 tokens drawn in
upstream's 12 linear steps from the 110 tokens of Gaia's XP coefficients.
NV is the same card through NVIDIA's kernel driver, without CUDA.
`torch.compile` of the transformer's encoder and decoder brings PyTorch
level with raven on embed (raven 5% faster) and the posterior (raven 3%
slower), and draws spectra 1.6 times as fast as raven; upstream ships it
eager.

At 64 objects a call, the tutorial's batch, raven embeds 98 objects a second
against PyTorch's 62, computes 128 posteriors against 99, and draws 14.4
spectra against 23.1 (one run each).

One object at a time, in milliseconds a call:

| aion-base | raven, CUDA | raven, NV | PyTorch |
|---|---|---|---|
| embed | 30.9 (30.7–31.1) | 34.5 (34.5–34.6) | 14.0 (14.0–14.0) |
| redshift posterior | 17.8 (17.7–17.9) | 24.0 (23.9–24.1) | 11.1 (11.1–11.2) |
| DESI spectrum, 12 steps | 115 (114–116) | 120 (120–121) | 98.6 (94.9–103) |

## The model on the CPU

Objects a second on two cores of an Intel Core Ultra 5 235: embed and the
posterior at 4 objects a call, a spectrum one at a time.

| aion-base | raven, compiled (default) | raven, eager (`--eager`) | PyTorch |
|---|---|---|---|
| embed | 0.281 (0.277–0.292) | 1.20 (1.19–1.25) | 0.834 (0.702–0.851) |
| redshift posterior | 0.553 (0.549–0.561) | 1.65 (1.64–1.69) | 1.35 (1.32–1.43) |
| DESI spectrum, 12 steps | 0.039 (one round) | 0.211 (0.207–0.215) | 0.340 (0.337–0.353) |

## The codecs

Objects a second, from data on the host to tokens read back, or tokens to a
spectrum on the codec's grid: 16 objects a call on the GPU, 4 on the CPU.

| | raven, CUDA | PyTorch, CUDA | raven, CPU compiled (default) | raven, CPU eager (`--eager`) | PyTorch, CPU |
|---|---|---|---|---|---|
| Image encode, 4 × 160 × 160 | 205 (205–208) | 726 (726–727) | 2.15 (2.14–2.15) | 3.42 (3.36–3.60) | 4.13 (4.11–4.43) |
| Spectrum encode, 7781 pixels | 785 (785–804) | 398 (397–398) | 8.67 (8.65–8.90) | 5.80 (5.28–6.06) | 8.33 (8.29–8.68) |
| Spectrum decode, 273 tokens | 2,286 (2,257–2,315) | 852 (851–855) | 32.8 (32.7–32.9) | 16.1 (7.77–17.0) | 26.4 (26.0–26.9) |

## Where the time goes

Device time of one call at 16 objects with aion-base on CUDA, by kernel:
raven's from tolk's `DEBUG=2` timing of each kernel, at dee7042d32, whose
kernels take eba27a14be's time within 1%; PyTorch's from `torch.profiler`.

| | raven, embed | PyTorch, embed | raven, spectrum | PyTorch, spectrum |
|---|---|---|---|---|
| Products | 118 ms, 20.5 TFLOP/s | 77 ms, 30.3 TFLOP/s | 840 ms, 19.0 TFLOP/s | 396 ms, 25.0 TFLOP/s |
| Attention | 26 ms | 59 ms | 51 ms | 77 ms |
| Norms, elementwise, small reductions | 12 ms | 120 ms | 79 ms | 206 ms |
| In all | 156 ms | 257 ms | 970 ms | 679 ms |

- **Embedding and the posterior.** raven's products run at two thirds of
  cuBLAS's rate, but it fuses the norms and elementwise work into a tenth of
  PyTorch's time, and its attention takes half of PyTorch's, which upstream
  computes as two batched products and a softmax.
- **Spectra.** raven's draw encodes all 383 tokens at each of the 12 steps,
  the target's absent ones included; upstream's encodes the present ones
  only, 110 at the first step and 361 at the last, 236 on average. raven's
  encoder computes 1.6 times the rows, which with its products' rate makes
  2.1 times PyTorch's product time. aion-large widens the gap: its products
  run at 17.0 TFLOP/s against cuBLAS's 25.2.
- **One object.** A single object's 858 rows tile poorly in raven's products
  (5.3 TFLOP/s against cuBLAS's 23.0), which take 29 of embed's 32 ms of
  kernels.
- **NV.** Where NV is slower than CUDA, its products are: aion-large's draws
  run them at 14.2 TFLOP/s against 17.0 through CUDA, and one object's
  posterior at 6.4 against 9.2.
- **The codecs.** raven's image codec runs its convolutions at 22 TFLOP/s in
  float32; PyTorch's run on cuDNN with TF32, cuDNN's default, in 20 ms
  against raven's 71, so that part of upstream is not IEEE float32.
- **The CPU.** Eager raven draws spectra slower than PyTorch for the same 1.6
  times the encoder's rows. Compiled, embed spends 59% of its kernel time
  in products and 41% in attention (`DEBUG=2`, 4 objects, at eba27a14be),
  where eager raven runs nx's own kernels 4.3 times as fast. Other users'
  jobs shared the host's cores at times, which the lower ends of the CPU
  ranges show.

## Whole commands, starting and memory

Seconds for a new process from its start to its exit: `main.exe` and
`bench/upstream.py`, the tutorial's code, each loading aion-base's released
weights from a warm page cache, reading real rows of the Multimodal Universe
with h5py or `Survey`, and writing an npz file. 64 objects of the tutorial's
file and 64 stars of Gaia's healpix 1631 on the GPU, 16 objects and 4 stars
on the CPU, in batches of 16; raven's kernels come from its disk cache.

| | raven, CUDA | PyTorch, CUDA | raven, CPU compiled (default) | raven, CPU eager (`--eager`) | PyTorch, CPU |
|---|---|---|---|---|---|
| `embed`, image, spectrum and 9 values | 1.80 (1.80–1.82) | 7.95 (7.88–7.98) | 64.1 (63.9–64.3) | 24.6 (23.7–28.2) | 32.5 (32.3–34.0) |
| `posterior` of redshift, image and 4 fluxes | 1.75 (1.72–1.88) | 5.99 (5.96–6.02) | 34.2 (34.2–34.2) | 18.9 (17.5–44.3) | 21.3 (20.7–21.8) |
| `generate` of a DESI spectrum from XP, 12 steps | 5.08 (5.04–5.12) | 7.66 (7.66–7.66) | 50.6 (50.4–50.6) | 22.7 (22.1–71.8) | 17.7 (17.2–17.7) |
| `embed` over an empty kernel cache | 9.88 (9.73–9.88) | | 69.3 (68.9–69.4) | nothing to compile | |

On the GPU, besides the model's time at the tables' rates, `main.exe`'s
`embed` takes 1.2 s to start, load the weights, read the rows and run the
codecs, which it runs on the GPU as PyTorch does; PyTorch's process takes
6.9 s. `generate` draws spectra slower (The model on a GPU) but starts
faster, so it finishes first. Compiling raven's kernels over an empty cache
adds 8 s to `embed`, 5 s on the CPU; PyTorch ships its kernels compiled.
On the CPU, compiled, `main.exe` takes 1.6 to 2.9 times PyTorch's time, as
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
  hidden. Other users' jobs shared the host, at a load of 6 to 21, never the
  GPU: its process list was empty before and after every GPU run.
- **Versions.** raven at eba27a14be for the model's and the codecs' rows,
  and at 50d61418f0 for the whole commands, after `main.exe` moved its
  codecs to the device and compiled by default. AION-1 (`polymathic-aion`)
  at db7fba6d on torch 2.15.0+cu132, Python 3.12.15, cuDNN 9.26. The whole
  commands load polymathic-ai/aion-base at 40541618, the Hub's current
  commit, which `Hub` pins.
- **Statistic.** Three rounds on 2026-11-20, each running raven and PyTorch
  in turn on one device, in alternate orders. raven's cells are thumper's
  medians (5 samples on the GPU, 3 on the host, after 3 warmup calls), the
  others `pytorch.py`'s (5 calls on the GPU, 3 on the host, after 1 or 2);
  the tables give the median of the three rounds and their range. The
  compiled host's spectrum row, at 25 s a call, ran in the first round
  only; `torch.compile` and the batch of 64 ran once. The whole commands
  ran three rounds of `whole.sh` each, PyTorch's CPU cells in the compiled
  commands' rounds. The breakdowns are one profiled call each, outside the
  timed rounds.
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
