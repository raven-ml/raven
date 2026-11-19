# whisper

Whisper, OpenAI's speech recognition models (github.com/openai/whisper,
the paper arXiv 2212.04356), implemented on raven: its tokenizers, its
checkpoints, its encoder and decoder. It lives in `dev/`, raven's sandbox,
and is never published (no opam package, no CHANGES entry).

## Boundaries

- A clean OCaml implementation designed by `docs/dev/design.md`, not a
  translation of the upstream Python. Upstream at `86098128` run on the CPU
  in float32 is the oracle for what the model computes; the paper is
  context.
- Raven's rules hold here: no dependencies, public interfaces of raven only,
  `AGENTS.md`'s code style. A capability raven lacks is built in the library
  that owns it, never worked around here.
- `lib/` is the library `whisper`: `Audio` (a WAV or FLAC file's samples,
  mixed to one channel at 16 kHz, the 30 s window, and their log-mel spectrogram of any
  length, floored over the whole call, composed from omr),
  `Tokenizer` (the rank-file encodings and
  Whisper's special tokens, numbered after the ranks) and `Model` (the
  dimensions, the parameters as kaun layers and their structure, the
  checkpoints, the releases' alignment heads, the encoder, and the decoder
  over kaun's key-value cache with the audio's keys and values computed
  once), `Decoding` (options, initial tokens, suppressed ids, the logit
  filter over lanes, and greedy, sampled and beam decoding as one
  `Rune.iterate` on any device, ranked on the host, the languages'
  probabilities and the compression ratio), `English` (the text
  normaliser of Whisper's evaluation, with upstream's `english.json`
  spelling map embedded at build time) and `Wer` (jiwer's word counts and
  corpus word error rate).
- `test/` holds the suites, on trimmed rank tables and small models with
  goldens from upstream's own code. Each output is compared as
  `test/support/tolerance.mli` states: at float64, as one function with
  upstream's, and at float32, as accurate as upstream's, eager and under
  `Rune.jit`. `test/check/check.exe` compares the port with the oracle on
  the full rank files (`tokenizer`) and the released checkpoints (`model`:
  encoder, first-step logits, first greedy tokens; `greedy`: whole greedy
  decodes of the oracle's utterances, their sums, scores and WER; `beam`:
  beam searches of them, their finished candidates in order, live beams,
  sums and WER; `e2e`: `greedy` from each utterance's FLAC file through
  `Audio`, with its samples checked against ffmpeg's and its features
  against the float64 reference and upstream's; `excuse`: the excuse rule
  at a differing step), which the repository does not hold, so no test
  runs it.
- No Python, no weights and no data in the repository, but small goldens
  under `test/upstream/`, each cut outside the repository by a generator its
  `dune` stanza names.

## Decisions

- Attention scores are `q·k/√d`, upstream's default path (SDPA), not its
  manual path's `d^-¼` on each of `q` and `k`, which only word timestamps
  run. The two are one function rounded differently, and at float32 the
  port is closer to the default on tiny and base.en.

- `Tokenizer.encode` never matches a special token's text, where upstream's
  `encode` raises on it (tiktoken's `disallowed_special="all"`). The text of
  a prompt is text; refusing some texts protects nothing when no text can
  become a special token.

- A decoding samples at most `n_text_ctx / 2` tokens, upstream's default
  `sample_len`, which `transcribe` never changes, and the option is left
  out. Upstream cuts the prefix to its last `n_text_ctx / 2 - sample_len`
  ids, which at that default is Python's `prefix[-0:]`, the whole prefix:
  the prefix is kept whole.

- The specials no transcript holds (`<|transcribe|>`, `<|startoftranscript|>`,
  ...) are suppressed whatever ids are given to suppress. Upstream suppresses
  none of them when given none, which lets a step sample them.

- `English.normalize` reads upstream's Python regular expressions as Python's
  `re` does, with the character classes (letters, numbers, decimal digits,
  marks, symbols, punctuation) of brot's regular expressions, Unicode
  18.0.0, where Python 3.13 has Unicode 15.1. raven has no other public
  table of Unicode properties.

- `max_initial_timestamp` counts timestamp steps of 0.02 s, an `int
  option`: `Some 0` allows only `<|0.00|>`. Upstream takes seconds, turns
  them into an index through the model's `n_audio_ctx`, which a token of
  fixed meaning does not depend on, and reads `0.0` as no limit.

- Ties take nx's specified orders: a beam's proposals the lowest id first
  among equal log probabilities (`Nx.top_k`), its candidates the earlier
  beam then the earlier proposal first among equal sums, and the live beams
  that fill a short list of candidates the lower lane first among equal
  sums (`Nx.sort`, stable). Upstream's `torch.topk` leaves its order of
  equal values unspecified, and its reversed `np.argsort` depends on the
  CPU's sorting kernel. nx also orders `+0.` above `-0.`, which Python
  compares equal.

- The timestamp rule's comparison (the timestamps' log-sum-exp against the
  greatest text log probability, both under a log-softmax) has the bits of
  nx's kernels, within the bound they state, not torch's. A step whose two sides lie within that
  bound may decide otherwise; the parity check excuses a differing token
  only there or where upstream's two greatest logits lie within the logits'
  distance.

- A candidate of no token, the end of text sampled first, scores `-∞`
  without a length penalty. Upstream divides its sum by its length of 0,
  which raises `ZeroDivisionError`.

- A beam states its `patience`, `1.` being upstream's default, and a beam
  whose `size * patience`, rounded half to even as Python's `round`, is
  below 1 raises, `patience = 0.` included. Upstream reads a patience of
  `0.0` as `1.0`, Python's truthiness.

- Audios decode together under every search, each from its own initial
  tokens, each lane stopping at its own length limit, and a beam's audio
  stops once it holds its finished candidates. Upstream decodes one audio
  at a time with beams (its cross-attention's broadcast raises for more)
  and stops every lane at the batch's shared limit. The tests hold the law
  that a batch decodes as each audio alone.

- `Audio.load` takes each sample at its dtype's full scale, rounded once to
  float32, and mixes channels and resamples another rate by libswresample's
  computation at float64. Upstream pipes any file through ffmpeg (`-ac 1 -ar
  16000 -f s16le`) to 16-bit samples: ffmpeg floors deeper integer samples to
  16 bits (a 24-bit 384 gives 1, -129 gives -1), clips float samples to full
  scale (1.5 gives 32767), and rounds mixed or resampled ones to the nearest
  16-bit value, clipped. On a 16 kHz mono 16-bit file both give the same
  samples. Here a deeper file keeps its low bits, and a float file its peaks
  above 1, and so does a mixed or resampled one: a full-scale 16-bit square
  wave at 44.1 kHz rings to 1.186 in `load`, 6099 steps past ffmpeg's
  clipped samples, at half the samples. Such files give features past the
  parity gates.
  - ffmpeg mixes and resamples in float32 (swresample.c's `swr_init` picks
    float32 between filters unless both formats are 16-bit at one rate or
    their bytes sum to at most 3), then rounds. C4.4 measured its float32
    within 1.3e-7 of the float64 computation against PyAV. On 20 s files at
    8, 22.05, 32, 43.65, 44.1, 44.101, 47.53, 48, 96 and 96.001 kHz and 2 s
    at 1000.003 kHz, of 1, 2 and 6 channels and 16 and 24 bits, every 16-bit
    sample of ffmpeg's lies within 0.5022 of a step of `load`'s times 32768,
    so it is `load`'s rounded but for 0.013% to 0.031% of samples, off by one
    step where `load`'s lies within 0.0022 of a step of a half.
  - ffmpeg works on 16-bit integers with 15-bit weights where `swr_init` picks
    them. Two channels at 16 kHz give `⌊(l + r + 1) / 2⌋`, where `load` gives
    the mean; a 16-bit 5.1 file at 16 kHz differs from `load`'s by up to 1.28
    steps (6467 of 32000 samples rounded otherwise); an 8-bit file at another
    rate or of several channels by up to 2.77 steps (15634 of 32000 at 44.1
    kHz).
  - Past 1024 phases ffmpeg's x86 kernels read the sample after the last its
    filter weighs for row 1024 of its bank, which at the end of a packet of
    input holds a sample ffmpeg has not yet read: at 44.101 kHz one output in
    320000, at a packet boundary, lies 1.1e-6 from `load`'s. On aarch64, and
    on x86 where `T` is a multiple of 8 (AVX) or of 4 (SSE alone), the
    kernels read `T` taps as the C template does, and row 1024 lacks row 0's
    last tap or holds it `T` samples back: about 4e-6 of full scale from
    `load`'s at phase 1023.
  - For 8-byte samples (float64) `swr_init` picks float64 between filters,
    so ffmpeg computes as `load` does before its rounding to 16 bits.
  - `load` gives `⌈16000 n / r⌉` samples; ffmpeg's flush gives that many or
    one fewer, and fewer on a file shorter than the filter (none below 62
    samples at 44.1 kHz).
  - ffmpeg refuses to mix a layout without a front speaker or with half of a
    pair (front, side, back, left and right of centre, top front); `load`
    mixes it by the same weights, and refuses a layout whose weights are all
    0. ffmpeg guesses the speakers of 10, 12, 14, 16 and 24 channels a file
    does not place, and refuses other counts past 8; `load` refuses every
    count past 8. A FLAC file whose channel-mask comment does not number its
    channels takes FLAC's order in ffmpeg and the guess for a WAV file here,
    which differ at 3 and 4 channels.

- `Audio.log_mel` mirrors a signal of 200 samples or fewer as often as its
  frames need, so 160 to 200 samples give one frame and fewer none.
  Upstream's centred transform raises there (torch's reflect padding needs
  more samples than it pads).

- Sampling draws from raven's key, folded with the step; its parity with
  upstream is in distribution only.
