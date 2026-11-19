# aion: departures from upstream

Each place where aion computes something other than upstream
(PolymathicAI/AION at db7fba6df53b), with its reason and evidence. Upstream
bugs on paths the weights never trained through are fixed; behaviour the
weights were trained with stays.

## Scalar codecs

- **A negative token is refused.** `Scalar.decode` raises
  `Invalid_argument` for a token outside `[0, n)`. Upstream indexes its
  codebook with the token (`aion/codecs/quantizers/scalar.py:169-182`,
  `aion/codecs/quantizers/__init__.py:353-355`), so a token of `n` or more
  raises `IndexError` there too, while a negative one wraps around and
  decodes to an edge from the top. No encode makes a negative token, so the
  weights never saw one decoded.

## Spectrum codec

- **Decoding outside the latent grid gives a bad pixel of flux 0.** Upstream
  resamples the decoder's flux and mask logit onto the requested wavelengths
  with the fill 0 outside the grid's 3500 to 10462.4 Å
  (`aion/codecs/modules/spectrum.py:48-49`), then computes the flux
  `(0 + 1) · s`, the normalisation, and the mask `round (sigmoid 0) = 0`, a
  good pixel (`aion/codecs/spectrum.py:163-176`). So it reports the
  normalisation as a measured flux wherever it knows nothing. The resampling
  onto requested wavelengths follows the decoder, so no weight was trained
  through it.
- **`encode` refuses a row longer than its modality's length.** Upstream's
  `F.pad` with a negative width cuts the row's red end without a word
  (`aion/codecs/preprocessing/spectrum.py:18-20`). No training spectrum was
  cut: Multimodal Universe's DESI rows hold 7781 pixels against 7808, and its
  SDSS, BOSS and eBOSS rows at most 4668 against 4800.
- **`encode` refuses wavelengths that decrease or are NaN.** Upstream's
  interpolation binary-searches each row as if sorted
  (`aion/codecs/modules/spectrum.py:25`), so on a row ending in Multimodal
  Universe's -1 pads it reads intervals that hold no pixel near the
  wavelength; on the 64 SDSS rows of the parity subset this moves 3 to 44 of
  273 tokens a row. `Survey` gives each row's length before those pads and
  `Spectrum.encode ~lengths` reads only those pixels (the maintainer's
  decision, until a redshift measurement settles whether the weights adapted
  to the pads), and the codec refuses what remains unsorted. Parity on SDSS is checked
  against upstream fed the trimmed rows.
- **A zero-width interval has slope 0.** Upstream divides by the interval's
  width (`aion/codecs/modules/spectrum.py:40`), 0 between repeated
  wavelengths. The interval is read only by a grid wavelength equal to a
  row's first where the second repeats it, and there upstream's value is
  NaN, which reaches every token of the row through the encoder's norms. A
  NaN there would have stopped training, so the weights never met it; the
  port reads the first pixel's value.
- **`decode` refuses a token outside `[0, 1024)`.** Upstream indexes the
  normalisation's codebook with its token, where a negative one wraps around
  (`aion/codecs/quantizers/__init__.py:355`), and reads an LFQ token's ten
  low bits alone (`lookup_free_quantization.py:236`), so 1024 decodes as 0 and
  -1 as 1023. No encode makes such a token.

## Finite scalar quantization

- **A code outside the codebook is refused.** `Fsq.decode` raises
  `Invalid_argument` for a code outside `[0, size)`. Upstream's
  `FiniteScalarQuantizer.decode` (`aion/codecs/quantizers/__init__.py:149-165`)
  takes digits by floor division and remainder, which wrap any integer onto
  some code: 4375 decodes as 0 under the image levels. No encode makes such a
  code.
- **NaN is refused.** `Fsq.encode` raises `Invalid_argument` for a NaN latent.
  Upstream casts the rounded NaN to int32 (`:131-147`), a conversion C leaves
  undefined, so its code is arbitrary.

## Image codec

- **Each band keeps its own survey's unit.** `Image.prepare` converts each
  band from its survey's unit to nanomaggies, and `Image.decode` converts
  each back. Upstream takes the survey of the first band for every band
  (`aion/codecs/image.py:77-79`, `:103-104`, `:146`). The two agree on a
  list of one survey's bands, the only lists AION-1 was trained with. On a
  mixed list upstream leaves HSC bands unscaled or scales Legacy Survey ones;
  and `decode` with `bands=None`, which asks for all nine bands with HSC-G
  first, multiplies the four Legacy Survey bands by 10^1.8 ≈ 63.1. Tests:
  `test_image.ml`, "each band is prepared by its own survey" and "each decoded
  band is in its own unit", both of which fail under upstream's rule.
- **Divisions by constants are CUDA's.** AION-1 trained on CUDA, where
  PyTorch up to 2.9 divides a tensor by a Python number as a product with
  the float32 reciprocal (`AGENTS.md`). `Image` computes so: `x / 0.01` as
  `x * 100`, `y / 10` as `y * 0.1f`, and the HSC rescale as `Quantity.value`'s
  product with float32 10^-1.8. Upstream's CPU code, which made the goldens
  and the reference runs, divides (`aion/codecs/image.py:83`, `:90`, `:92`,
  `aion/codecs/preprocessing/image.py:101`); the two differ by at most one
  rounding. The projection's scale `(9 / c) ** 0.5`
  (`aion/codecs/modules/subsampler.py:35`) is a float32 quotient then its
  square root on both devices.
- **Encoding never writes its input.** Upstream clamps and rescales the
  caller's `flux` in place (`aion/codecs/preprocessing/image.py:81`, `:101`),
  so a second encode of one tensor sees the first's writes.
- **A band twice or an image under 96 pixels is refused.** Upstream assigns a
  repeated band's channel twice, the last write winning
  (`aion/codecs/preprocessing/image.py:36`), and crops a smaller image from
  a negative start, which slices from its end (`:66-70`).

## Transformer

- **The decoder attends with the mask it was trained with.** `AION.forward`
  gives the decoder an all-zero compressed attention mask
  (`aion/model.py:98-99`), so `adapt_decoder_attention_mask` hides every
  pair (`aion/fourm/fm.py:613-617`) and each decoder token's
  self-attention, filled with `-finfo.max`, is the uniform mean of every
  decoder token's value, across targets. Training put each target's token
  count at its first token (4M's sampler, ml-4m
  `fourm/data/masking.py:262-264`), so a token saw the tokens of its own
  target. `Transformer.forward` computes that. For a one-token target the
  two agree bit for bit (float64);
  for a DESI spectrum from an image they differ by up to 17.6 in a logit,
  and argmax differs at 1,928 of 2,184 tokens.
- **Targets keep their order.** Upstream shuffles the target modalities
  with Python's `random.sample` (`aion/fourm/fm.py:360-363`), which changes
  the order of sums, so the bits of several targets' logits vary from call
  to call. `Transformer.forward` keeps the order it is given; the values
  agree in exact arithmetic.
- **`encode` keeps every present token.** `AION.encode` keeps the first
  `num_encoder_tokens` of the row, 256 by default (`aion/model.py:153`), so
  an input of more than 256 present tokens loses its tail. `Transformer.encode`
  keeps them all, as `AION.forward` does.
- **An object with no present token is refused.** Upstream's every query
  then sees only hidden keys and takes the uniform mean of the zeroed
  padding's values. Raven's attention gives such a row zero.
  `Transformer.encode` raises `Invalid_argument` instead of choosing either.
- **An absent token's value is not read.** Upstream embeds every token
  before masking (`aion/fourm/encoder_embeddings.py:234`), so an absent
  token outside the vocabulary raises `IndexError`. `Transformer.encode`
  reads present tokens only, and refuses a present one outside the
  vocabulary.
- **The catalog is left out.** Its model tokens (vocabulary 3,468, offset
  204, `aion/fourm/modality_info.py:560-583`) have no published map from
  its codec's 100 tokens, and `AION.forward` cannot predict it (its
  sequence shift leaves 49 rows for 50 tokens). The transformer does not
  read its tensors.

## Generation

- **Each object draws its own order.** Upstream's ROAR step draws one
  `torch.rand(N)` for the batch (`aion/fourm/generate.py:634-643`), so every
  object of a batch reveals the same positions at every step, and draws batched
  from one object are independent only given that order. AION-1 calls repeated
  runs i.i.d. draws (arXiv:2510.17960, §7.1). `Roar.sample` draws an order per
  object. Each object's own law is unchanged, so a statistic over one draw per
  object is too; what changes is the spread of draws batched from one object.
  No effect was measurable for one star at linear 8: token agreement within a
  run differs by -0.0007 ± 0.0013.
  Upstream also redraws its order among the unknown positions at each step,
  where `Roar.sample` fixes each object's order up front; the first `k` of a
  uniform order are a uniform `k`-subset and the rest a uniform order of the
  remainder, so the two have the same law.
  Upstream also seeds step `t` of a run with `seed + t` (`generate.py:1531`), so
  runs with consecutive seeds repeat each other's uniforms one step apart; seeds
  7 and 8 hold the same token 1.76 times as often as independent runs. Keys
  derived by `fold_in` repeat nothing.
- **Objects with different unknown counts are not reachable.** Upstream's step
  decodes as many positions as the first object has unknown
  (`generate.py:629-631`). Where another object has fewer, its decoder rows
  include known positions, `forward_logits` keeps the target's rows only
  (`aion/fourm/fm.py:698-699`), and `reshape(B, N, -1)` (`generate.py:1053`)
  cuts the remaining logits across objects: in a run where one of two objects
  knew 250 of 273 DESI tokens, 162 of them were overwritten, and other counts
  raise. `Roar.sample` starts
  every target with no token revealed and reveals equal counts, so it never
  takes that path. Where no object has fewer unknown tokens than the step
  decodes, upstream's unmasked decoder equals training's mask bit for bit, and
  so `Transformer.decode` at the step's positions.
- **Redshift's class 1024 is drawn and refused at decode.** `tok_z`'s
  vocabulary has 1025 classes (`aion/fourm/modality_info.py:547`) and its codec
  1024, whose encode clamps to 1023 (`aion/codecs/quantizers/__init__.py:347-349`).
  Upstream's decode of 1024 raises `IndexError`; `Scalar.decode` raises
  `Invalid_argument` naming the class. `Roar.sample` draws over all 1025, as the
  model defines them: the class holds up to 16 % of a row's mass on SDSS
  spectra, and at z in [3, 5) its median mass is 2,400 times that below z = 1,
  so leaving it out of the draws
  would move that mass onto the grid. What it means is open; it may be the
  redshifts past the grid's last edge, from a tokenization without the clamp.

## Inputs

- **A non-finite value makes its modality absent.** `Codecs.encode` marks
  a scalar modality absent for an object one of whose values is NaN or
  infinite where its codebook holds no NaN edge, every token of an XP
  spectrum with it. Upstream encodes the value: NaN and +∞ take the last
  token, -∞ the first. Where the codebook holds no NaN edge the training
  sample held no NaN, so that token claims a measurement at the
  distribution's edge: Gaia's parallax is NaN in 53 of healpix 1631's
  10,870 rows, and its token 1023 decodes to 69.6 mas, a star 14 pc away,
  where the median token decodes to 0.77 mas.
  Seven HSC codebooks end in NaN edges, whose training sample held NaN, so
  the weights learned NaN as a value there: those keep upstream's encoding
  of every value: trained behaviour stays.
- **Values are float32 when read.** `Survey` casts every value of
  Multimodal Universe's HDF5 files to float32 once, as it reads it: the
  dtype MMU's dataset builders declare for every float column, which the
  Hub datasets AION-1's card names as its training data deliver, and the
  tutorial's `astype("float32")`. Upstream's codecs take a float64 input as
  it is, and compare it with their float32 edges in float64. The files
  hold Gaia's fluxes, parallax, `ra` and `dec` and DESI's `Z` as float64;
  on 1,599 such values of the parity subset, upstream's tokens of the
  float32 roundings equal those of the float64 values. Positions for
  cross-matching stay float64.
- **An embedding is the mean over the object's present tokens.**
  `Transformer.embedding`, which `main.exe embed` writes, averages each
  object's context over the rows before its length,
  AION-1's Eq. 7 (arXiv:2510.17960, §7.2) over the object's tokens.
  Upstream's tutorial pools `model.encode(...).mean(axis=1)`, over every
  row of the batch's inputs, absent tokens included; with an absent input
  an object then moves with its batch: aion-base on an object whose XP
  spectrum is absent gives an embedding at cosine 0.71 to the same object
  encoded alone without it, where the mean over present rows agrees to
  1.1e-6. Every published
  evaluation and tutorial has all tokens present, and there the two means
  are one.
