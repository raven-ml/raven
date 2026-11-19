# whisper

Whisper, OpenAI's speech recognition models (github.com/openai/whisper,
the paper arXiv 2212.04356), implemented on raven: its tokenizers, its
checkpoints, its encoder and decoder. It lives in `dev/`, raven's sandbox,
and is never published (no opam package, no CHANGES entry).

## Boundaries

- A clean OCaml implementation designed by `doc/dev/design.md`, not a
  translation of the upstream Python. Upstream at `86098128` run on the CPU
  in float32 is the oracle for what the model computes; the paper is
  context.
- Raven's rules hold here: no dependencies, public interfaces of raven only,
  `AGENTS.md`'s code style. A capability raven lacks is built in the library
  that owns it, never worked around here.
- `lib/` is the library `whisper`: `Audio` (a 16 kHz mono WAV or FLAC
  file's samples, the 30 s window, and their log-mel spectrogram of any
  length, floored over the whole call, composed from omr),
  `Tokenizer` (the rank-file encodings and
  Whisper's special tokens, numbered after the ranks) and `Model` (the
  dimensions, the parameters as kaun layers and their structure, the
  checkpoints, the releases' alignment heads, the encoder, and the decoder
  over kaun's key-value cache with the audio's keys and values computed
  once), `Decoding` (options, initial tokens, suppressed ids, the logit
  filter over lanes, and greedy, sampled and beam decoding as one
  `Rune.iterate` on any device, ranked on the host), `English` (the text
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

- `Audio.load` reads 16 kHz files of one channel and takes each sample at
  its dtype's full scale, rounded once to float32. Upstream pipes any file
  through ffmpeg to 16-bit samples, resampled to 16 kHz and mixed down to
  one channel: ffmpeg floors deeper integer samples to 16 bits (a 24-bit 384
  gives 1, -129 gives -1) and clips float samples to full scale (1.5 gives
  32767). On a 16 kHz mono 16-bit file both give the same samples. Here a
  deeper file keeps its low bits and a float file its peaks above 1, so a
  float file that peaks above 1 gives features past the parity gates; another
  rate or channel count raises.

- `Audio.log_mel` mirrors a signal of 200 samples or fewer as often as its
  frames need, so 160 to 200 samples give one frame and fewer none.
  Upstream's centred transform raises there (torch's reflect padding needs
  more samples than it pads).

- Sampling draws from raven's key, folded with the step; its parity with
  upstream is in distribution only.
