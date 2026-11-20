embed, posterior and generate read objects from the Multimodal Universe's
files and run the transformer of --model, here the narrow one cache.exe stores as
aion-base's weights, over Flux_g, Xp_bp, Xp_rp and Z.

  $ ./cache.exe ../upstream/transformer.safetensors hub/huggingface
  $ export RAVEN_CACHE_ROOT=$PWD/hub
  $ aion () { ../../bin/main.exe "$@"; }
  $ ls=../mmu/legacysurvey.h5 gaia=../mmu/gaia.h5

An object is a row of the first source with a reciprocal partner within 1″
in every other: Legacy Survey rows 0 and 1 pair with Gaia rows 1 and 0, and
row 2's nearest Gaia row is 1.5″ away. rows holds each object's row in each
source.

  $ aion embed --legacysurvey $ls --gaia $gaia --out e.npz
  embed: 2 objects; of rows 0:3, 1 without a partner in every source, 0 without a present input
  $ ./npz.exe entries e.npz
  embedding float32 [2; 16]
  rows int64 [2; 2]
  $ ./npz.exe ints e.npz rows
  0 1
  1 0

The inputs are every modality the sources hold and the model reads: here
Flux_g from the Legacy Survey, and Gaia's XP spectra. Legacy Survey row 1's
FLUX_G is infinite, so its object has Flux_g absent, and its embedding, the
mean over its present tokens, is the one it has without Flux_g, alone.

  $ aion embed --legacysurvey $ls --gaia $gaia --rows 1:2 --inputs tok_xp_bp,tok_xp_rp --out alone.npz
  embed: 1 objects; of rows 1:2, 0 without a partner in every source, 0 without a present input
  $ ./npz.exe same embedding e.npz 1 alone.npz 0
  equal within 1e-5

An embedding does not depend on the objects batched with it.

  $ aion embed --legacysurvey $ls --gaia $gaia --batch 1 --out one.npz > /dev/null
  $ ./npz.exe same embedding e.npz 0 one.npz 0
  equal within 1e-5
  $ ./npz.exe same embedding e.npz 1 one.npz 1
  equal within 1e-5

--model takes a file of aion-base's layout too.

  $ aion embed --gaia $gaia --model ../upstream/transformer.safetensors --out file.npz > /dev/null
  $ aion embed --gaia $gaia --out base.npz > /dev/null
  $ ./npz.exe same embedding file.npz 3 base.npz 3
  equal within 1e-5

posterior writes the decoder's distribution over a scalar target's classes:
p over the classes that decode to a value, upper each class's value, the
upper edge of its bin, and for redshift p_1024, the probability of its class
1024, which decodes to none.

  $ aion posterior --legacysurvey $ls --gaia $gaia --target tok_z --out z.npz
  posterior: 2 objects; of rows 0:3, 1 without a partner in every source, 0 without a present input
  $ ./npz.exe entries z.npz
  p float32 [2; 1024]
  p_1024 float32 [2]
  rows int64 [2; 2]
  upper float32 [1024]
  $ ./npz.exe sums z.npz
  each object's probabilities sum to 1 within 1e-6
  $ ./npz.exe upper z.npz hub/huggingface tok_z
  upper is each codebook's edges, decoded, within 1e-6

p is the softmax of the decoder's logits, which reference.exe computes from
the library for one object, p_1024 the mass past the classes with a value.

  $ aion posterior --gaia $gaia --target tok_z --inputs tok_xp_bp --out zg.npz > /dev/null
  $ ./reference.exe $gaia 0 tok_xp_bp tok_z ref.npz
  $ ./npz.exe posterior zg.npz 0 ref.npz
  p is the softmax of the decoder's logits within 1e-6

An XP target has a distribution per coefficient. Gaia row 2 has a NaN BP
coefficient, so its only input is absent and it is left out.

  $ aion posterior --gaia $gaia --target tok_xp_rp --inputs tok_xp_bp --out xp.npz
  posterior: 3 objects; of rows 0:4, 0 without a partner in every source, 1 without a present input
  $ ./npz.exe entries xp.npz
  p float32 [3; 55; 1024]
  rows int64 [3; 1]
  upper float32 [55; 1024]
  $ ./npz.exe sums xp.npz
  each object's probabilities sum to 1 within 1e-6
  $ ./npz.exe ints xp.npz rows
  0
  1
  3

Each XP coefficient has its own codebook, and its own row of upper.

  $ ./npz.exe upper xp.npz hub/huggingface tok_xp_rp
  upper is each codebook's edges, decoded, within 1e-6
  $ ./reference.exe $gaia 3 tok_xp_bp tok_xp_rp ref.npz
  $ ./npz.exe posterior xp.npz 2 ref.npz
  p is the softmax of the decoder's logits within 1e-6

generate draws a target's tokens given the inputs, by random-order
generation in --steps steps, and decodes them with aion-base's codec: for an
XP target, values holds each draw's 55 coefficients.

  $ aion generate --gaia $gaia --target tok_xp_rp --inputs tok_xp_bp --steps 4 --draws 2 --batch 2 --seed 4 --out x.npz
  generate: 3 objects; of rows 0:4, 0 without a partner in every source, 1 without a present input
  $ ./npz.exe entries x.npz
  rows int64 [3; 1]
  tokens int64 [3; 2; 55]
  values float32 [3; 2; 55]
  $ ./npz.exe values x.npz hub/huggingface tok_xp_rp
  values are the tokens' edges, decoded, within 1e-6, NaN past the edges

Draw j of batch b is Roar.sample's with the key fold_in (fold_in (key seed) b)
j, and the steps reveal upstream's linear counts, the larger first: 14, 14,
14 and 13 of the 55 tokens. reference.exe draws them from the library. Here
batch 1 holds Gaia row 3 alone, since row 2 has no present input.

  $ ./reference.exe generate $gaia tok_xp_bp tok_xp_rp 4 14,14,14,13 2 2 ref.npz
  $ ./npz.exe equal x.npz ref.npz tokens
  tokens equal

So a seed gives the same draws, and another seed others.

  $ aion generate --gaia $gaia --rows 0:2 --target tok_z --steps 1 --draws 3 --seed 7 --out g.npz
  generate: 2 objects; of rows 0:2, 0 without a partner in every source, 0 without a present input
  $ ./npz.exe ints g.npz tokens
  978
  85
  154
  510
  217
  966
  $ aion generate --gaia $gaia --rows 0:2 --target tok_z --steps 1 --draws 3 --seed 8 --out g8.npz > /dev/null
  $ ./npz.exe equal g.npz g8.npz tokens
  tokens differ in 6 of 6 values

--device cpu-jit runs the transformer and the codecs compiled on the host,
each batch padded to --batch rows. Compiled sums may add their terms in
another order: test_transformer holds the compiled transformer within 2τŝ
of the eager one by the narrow network's rounding ensemble, which is at most
1.5e-5 of an object's largest encoding or logit there. So embeddings, means
of the encoding, and probabilities, whose softmax takes the logits'
differences, agree within 2e-5 of their largest value, and these draws
exactly, none of their uniforms falling that close to a boundary between
two tokens. Row 2 of each run has no partner, or no present input, so a
batch keeps fewer objects than it reads.

  $ aion embed --legacysurvey $ls --gaia $gaia --device cpu-jit --out ej.npz
  embed: 2 objects; of rows 0:3, 1 without a partner in every source, 0 without a present input
  $ ./npz.exe same embedding e.npz 0 ej.npz 0 2e-5
  equal within 2e-5
  $ ./npz.exe same embedding e.npz 1 ej.npz 1 2e-5
  equal within 2e-5
  $ aion posterior --gaia $gaia --target tok_z --inputs tok_xp_bp --device cpu-jit --out zj.npz > /dev/null
  $ ./npz.exe same p zg.npz 0 zj.npz 0 2e-5
  equal within 2e-5
  $ ./npz.exe same p zg.npz 2 zj.npz 2 2e-5
  equal within 2e-5
  $ aion posterior --gaia $gaia --target tok_z --inputs tok_xp_bp --device cpu-jit --batch 2 --out z2.npz > /dev/null
  $ ./npz.exe same p zg.npz 2 z2.npz 2 2e-5
  equal within 2e-5
  $ aion generate --gaia $gaia --rows 0:2 --target tok_z --steps 1 --draws 3 --seed 7 --device cpu-jit --out gj.npz > /dev/null
  $ ./npz.exe equal g.npz gj.npz tokens
  tokens equal
  $ aion embed --gaia $gaia --device vulkan --out x.npz
  aion: --device vulkan: "vulkan" names no device: a name is KIND or KIND:N, where KIND is one of CPU, METAL, CUDA, NV, AMD, NV-PCI, AMD-PCI, in any case, and N a number with no leading zero
  [2]

A one-token target drawn in one step follows the posterior: the frequencies
of 2000 draws of each object's redshift token agree with posterior's p, by
Pearson's test at its 0.999 quantile. Redshift's class 1024 is drawn, and
its value is NaN.

  $ aion generate --gaia $gaia --target tok_z --inputs tok_xp_bp --steps 1 --draws 2000 --seed 1 --out z.npz
  generate: 3 objects; of rows 0:4, 0 without a partner in every source, 1 without a present input
  generate: 5 draws of tok_z's class 1024, which decodes to no value
  $ aion posterior --gaia $gaia --target tok_z --inputs tok_xp_bp --out p.npz > /dev/null
  $ ./npz.exe chi2 z.npz p.npz 1
  the draws' frequencies agree with p: chi2 below its 0.999 quantile
  $ ./npz.exe values z.npz hub/huggingface tok_z
  values are the tokens' edges, decoded, within 1e-6, NaN past the edges

At temperature 0.5 the draws follow p squared, normalised, which the test
tells from p.

  $ aion generate --gaia $gaia --target tok_z --inputs tok_xp_bp --steps 1 --draws 2000 --seed 1 --temperature 0.5 --out z5.npz > /dev/null
  $ ./npz.exe chi2 z5.npz p.npz 0.5
  the draws' frequencies agree with p: chi2 below its 0.999 quantile
  $ ./npz.exe chi2 z5.npz p.npz 1
  the draws' frequencies differ from p: chi2 above its 0.999 quantile

A path to an existing file is the file, even where it holds :/.

  $ mkdir x: && cp $gaia x:/gaia.h5
  $ aion embed --gaia x:/gaia.h5 --out x.npz
  embed: 4 objects; of rows 0:4, 0 without a partner in every source, 0 without a present input

Refusals.

  $ aion posterior --gaia $gaia --target tok_z --inputs tok_z --out x.npz
  aion: --inputs tok_z: the target is no input
  [2]
  $ aion posterior --gaia $gaia --target tok_xp_rp --inputs tok_flux_g --out x.npz
  aion: --inputs tok_flux_g: no source holds it
  [2]
  $ aion posterior --gaia $gaia --target tok_image --out x.npz
  aion: --target tok_image: not a scalar modality
  [2]

A target is refused before any file is read.

  $ aion posterior --gaia missing.h5 --target tok_image --model missing --out x.npz
  aion: --target tok_image: not a scalar modality
  [2]
  $ aion posterior --gaia $gaia --target tok_nope --out x.npz
  aion: --target tok_nope: names no modality
  [2]
  $ aion posterior --gaia $gaia --target tok_flux_r --out x.npz
  aion: --target tok_flux_r: the model predicts no tok_flux_r
  [2]
  $ aion embed --gaia $gaia --model missing.safetensors --out x.npz
  aion: --model missing.safetensors: missing.safetensors: No such file or directory
  [2]
  $ aion embed --gaia $gaia --rows 2:9 --out x.npz
  aion: --rows 2:9: the first source has rows 0:4
  [2]
  $ aion embed --out x.npz
  aion: no source: give one of --legacysurvey, --hsc, --desi, --sdss or --gaia
  [2]

generate's options are refused before any file is read.

  $ aion generate --gaia missing.h5 --model missing --target tok_z --out x.npz
  aion: --steps is required
  [2]
  $ aion generate --gaia missing.h5 --model missing --target tok_xp_rp --steps 56 --out x.npz
  aion: --steps 56: not in [1, 55], tok_xp_rp's number of tokens
  [2]
  $ aion generate --gaia missing.h5 --model missing --target tok_segmap --steps 2 --out x.npz
  aion: --target tok_segmap: no codec decodes it
  [2]
  $ aion generate --gaia missing.h5 --model missing --target tok_z --steps 1 --draws 0 --out x.npz
  aion: --draws 0: at least 1
  [2]
  $ aion generate --gaia missing.h5 --model missing --target tok_z --steps 1 --temperature -1 --out x.npz
  aion: --temperature -1: not in [0, inf)
  [2]
