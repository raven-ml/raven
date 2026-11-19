# aion

AION-1, Polymathic AI's multimodal foundation model for astronomy
(github.com/PolymathicAI/AION, weights `polymathic-ai/aion-base` on Hugging
Face), on raven. `apps/AGENTS.md`'s rules hold; these are the app's own.

## Boundaries

- Upstream at db7fba6df53b and the Hub files at the revision `Hub.revision`
  names are the oracle for what the model computes; the paper is context.
- Where the app computes something other than upstream, `DIVERGENCES.md`
  says what and why. What raven lacked goes to `TODO.md`, with the library
  that owns the fix.

## Decisions

- Upstream bugs on paths the weights never trained through are fixed;
  behaviour the weights were trained with stays, however odd. Each fix is
  listed in `DIVERGENCES.md` with its evidence.
- Scalar codecs are one concept: a transform (identity, a clamped `log`,
  `asinh` twice) and one codebook of 1024 stored edges per channel. A value
  tokenizes to the number of edges below it, among all but the last, and a
  token decodes to its edge, the upper end of its bin. Upstream's four codec
  classes and their quantizers reduce to this.
- A modality is a variant, `Modality.t`, with the scalar ones a variant of
  their own inside it, so a scalar codec takes only a scalar modality.
- A codec lies on a device set like an nx value; its `place` moves it,
  and nothing moves it implicitly.
- The spectrum codec composes kaun's layers (channels last) with the port's
  own GRN, resampling and LFQ. Its networks are lists of layers read from the
  weights' names, so a smaller network of the same layout loads too, and a
  weight no layer reads is refused.
- A spectrum modality is `Modality.Spectrum Desi | Sdss`, and
  `Spectrum.encode` takes it and pads to its length, as upstream's survey
  classes do: the weights know only padded spectra. The unpadded encode of
  upstream's base `Spectrum`, which its goldens use, is for tests.
- Parity through a deep network is checked by a rounding ensemble. The
  arbiter is upstream's code run in float64. Its ensemble is n more float64
  runs (32 here), each moving every rounding operation's result by a normal
  draw at the operation's local scale: the order model, nx's per-operation
  rounding with summation orders unrelated to the values (Higham and Mary,
  SIAM J. Sci. Comput. 41(5), 2019). A value's spread ŝ is the root mean
  square of the runs' distances from the reference. A value passes within
  τ·ŝ, τ = √(n((2N/α)^(2/n) − 1)) over the N values of an output, α = 1e-3:
  to first order a computation at that rounding scale then fails with
  probability at most α. A decision (an LFQ bit, the normalisation's token, a
  pixel's mask) may differ from the reference's only where the reference lies
  within τ·ŝ of its boundary, and only to the outcome across it. The rule is
  `test/support/spread.mli`; the ensembles are recorded by
  `test/gen/spectrum_ensemble.py` and `image_ensemble.py` (AION-1's weights,
  the parity programs') and `spectrum.py` and `image.py` (the runtest
  fixtures'), all moving operations by `perturb.py`.
  - Two checks run on real weights: the float32 codec against the ensemble at
    float32's scale, and the codec cast to float64 against the ensemble at
    float64's, where a wrong constant or structure shows by orders of
    magnitude. The contract scale (nx.mli's bounds as Hoeffding proxies) and
    PyTorch's measured float32 error are reported beside them and gate
    nothing.
  - A constant matters on real weights below float32's noise, so each gets a
    fixture built so that it matters: the spectrum's pre-quantizer norm's
    lane variance near its eps, a GRN whose channels' mean norm is near its
    eps, the image encoder's final norm's lane variance near its eps and its
    squeeze-excite gates open.
  - The preparation, a few operations, keeps worst-case running error bounds
    from nx.mli (`test/support/shadow.mli`).
- The image codec takes cutouts as the surveys store them, `[n; c; h; w]`,
  and runs its network channels last, as `Nx.conv` does. Its weights load
  from upstream's state dict, PyTorch's `[out; in; kh; kw]` moved to
  `[out; kh; kw; in]` once at load.
- A pixel's unit is its band's (`Image.unit`): HSC flux on AB zero point 27,
  Legacy Survey nanomaggies, converted by `Ymir.Quantity.value`.
- The image codec computes as AION-1 trained, on CUDA. There PyTorch divides
  a tensor by a Python number as `a * inv_b`, `inv_b` the float32 quotient
  of 1 by the number rounded to float32
  (`aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu`, `is_cpu_scalar(2)`).
  That form is the same in v2.4.0 (upstream's floor, `torch>=2.4.0`),
  v2.6.0, v2.7.0, v2.8.0 and v2.9.0; it changed to a double reciprocal only
  in b5e5ee907e (PR #169507, 2025-12-04), after the codecs' weights were on
  the Hub (June 2025) and the paper (arXiv:2510.17960, 2025-10-20). So
  `x / 0.01` trained as `x * 100`, `y / 10` as `y * 0.1f`, and the HSC
  rescale as a product with float32 10^-1.8, which `Ymir.Quantity.value`
  gives. The projection's `(9 / c) ** 0.5` is a float32 quotient, then a
  square root (`cuda/PowKernel.cu` takes the exponent 0.5 as `sqrt`). The
  CPU reference divides; the difference is within one rounding.
- The transformer reads dense rows: each object's present tokens first, in
  the inputs' order (a stable `Nx.argsort`), the absent ones zeroed after
  them, and padding stated once, as each object's length, to every
  attention over the encoder's row (`Nx.attention ~lengths`). The decoder's
  self-attention mask is `equal ids_q ids_k`: for whole targets that is the
  training mask, and for a target at chosen positions (`decode`'s
  `positions`) a random-order step's, whose decoder holds those positions
  only. Slicing a whole target's logits is a different computation, 2·10⁴
  times the narrow fixture's τ ŝ away. Training's partial targets add
  `k < cumsum(counts)[q]`, with a test that the clause
  changes something.
- Generation (`Roar.sample`) takes the schedule as counts per step, with no
  default (the maintainer's decision), one key from which every draw
  derives by `fold_in`, and an order per object. A step's draw is
  `Nx.Rng.categorical` of the logits less their row's largest over the
  temperature, so a small temperature never overflows. Draws range over the
  whole vocabulary: redshift's class 1024, which no codec decodes, is drawn,
  and `Scalar.decode` refuses it by name.
- Logits come from a fixed slice of the decoder's rows per target. Training
  heads (one `Nx.matmul_segments` per vocabulary size) wait for training.

## Traps

- Seven HSC codebooks hold NaN edges (the training sample held NaN
  magnitudes and shapes, and sorted them last). `torch.bucketize`'s binary
  search goes right at a NaN edge, so values above the last finite edge it
  probes, and NaN, take token 1023. `Nx.searchsorted` orders NaN last and
  disagrees there, so `Scalar.encode` runs upstream's search itself.
- The `log` clamp: 36 % of `FLUX_W3`'s and 48 % of `FLUX_W4`'s training
  sample sit on the edge `log(float32(e⁻³)) = -3.0`. A `log` one ulp off at
  that one input moves half their tokens; a test pins it.
- `Z`'s stored grid spans [0, 6], where upstream's constructor builds
  [0, 1]: the stored buffer wins, as it does upstream.
- Encoding follows its input's dtype, as upstream does: upstream never
  casts, and `torch.bucketize` compares a float64 input with its float32
  edges in float64 (torch 2.14.1: 1 + 1e-12 takes the bin above the edge 1,
  its float32 rounding the bin below). Multimodal Universe's Parquet
  columns are all float32, but its HDF5 files hold float64 Gaia fluxes,
  parallax, ra and dec and DESI redshift, and the tutorial's file float64
  `Z_HP`.
  Rounding those to float32 first would change tokens near edges.
  `Survey` does round them, once, as it reads: the
  dtype MMU's builders declare and training's loaders produced. On the
  parity subset's 1,599 float64 values it changes no token.
- The spectrum's pad wavelength 99999 is far above the latent grid, so the
  grid beyond a row's last pixel interpolates toward the pad's normalised
  flux, -0.2. The weights were trained on it; it stays.
- LFQ's value 0 is the most significant bit of its token, against the
  MAGVIT-v2 paper's order; the tokens are upstream's.
- Multimodal Universe pads SDSS wavelengths with -1 within a file. `Survey`
  gives each row's length, the pixels before them, and `Spectrum.encode
  ~lengths` reads only those; without lengths it refuses the unsorted
  row.
- aion-base's file stores each modality embedding once, under
  `decoder_embeddings.<key>.mod_emb`; its metadata aliases the encoder's
  name to it. `Nx_io.load_safetensors` does not expose the metadata, so the
  loader reads the decoder's name for both, and a file holding the
  encoder's name too is refused as holding a tensor never read.
- Every layer norm's shift is a stored zero buffer (146 of them). The loader
  refuses a nonzero one, since `Kaun.Layer_norm` without a shift would not
  compute it. The positional tables are stored too, in the encoder and the
  decoder, and the loader reads both and refuses them if they differ: they
  are float32 values torch computed, which a formula need not reproduce bit
  for bit.
- The decoder's token tables (40) are in the file and never read: every
  decoder input but the catalog's is the mask token.
- The narrow fixture (`test/upstream/transformer*.safetensors`) makes each
  constant matter: embeddings and query-key projections near 1e-3 put the
  first layer norms and the query-key norms near their eps. Dropping any of
  eps, the query-key norm, a mask, the lengths, the context projection's bias
  or the shared modality embedding moves the result by 250 to 450,000 times
  the ensemble's τ ŝ. The test reports that comparison and does not gate it
  yet.
