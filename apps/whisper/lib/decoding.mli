(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Decoding a transcript from a window's audio features, one token at a time.

    The decoder ({!Model.decode} and {!Model.logits}) reads a sequence of token
    ids and gives, at each position, the logits of the token that follows. A
    transcript is decoded by feeding it the {e initial tokens}, then each token
    it samples, until it samples the end of text or reaches a length limit. The
    initial tokens are, in order:
    + with a {e prompt}, [<|startofprev|>] and the prompt's last
      [n_text_ctx / 2 - 1] ids: text that came before the window, usually the
      transcript of the previous windows, which the decoder reads as context;
    + the start sequence ({!Tokenizer.sot_sequence}), and [<|notimestamps|>]
      when the transcript has no timestamps;
    + the {e prefix}: ids the transcript starts with, as if sampled.

    The tokens sampled after them are the transcript. Before a step samples, its
    logits go through the {e filter} ({!filter}), which makes some tokens
    impossible. The {e suppressed} tokens are impossible at every step: ids that
    annotate rather than transcribe speech ({!Tokenizer.non_speech}, by default)
    and the special tokens no transcript holds. With timestamps, the timestamp
    rules make the timestamps come in pairs around each segment's text, in
    increasing order.

    {!val-decode} decodes several audios at once, each in a group of {e lanes},
    and its {e search} says how a step chooses: the likeliest token, a draw, or
    a beam search over the group's lanes. It is Nx and Rune operations on the
    model's set, so it runs on any device and compiles under [Rune.jit]; its
    {e outcome} holds every candidate, and {!results} ranks them on the host. *)

(** {1:options Options} *)

type options = {
  timestamps : bool;
      (** Whether the transcript marks where its segments start and end with
          timestamps. *)
  suppress : int array;
      (** Ids suppressed at every step, besides the specials no transcript
          holds. *)
  suppress_blank : bool;
      (** Whether the first sampled token may be neither a space nor the end of
          text, so that a transcript is never blank. *)
  max_initial_timestamp : int option;
      (** With timestamps, the greatest [i] whose [Timestamp i] may be the first
          sampled token; [None] for no limit. *)
  length_penalty : float option;
      (** How {!results} ranks candidates of different lengths: [None] by their
          sum of log probabilities over their length, [Some a], [0 <= a <= 1],
          over [((5 + length) / 6) ** a] (Wu et al. 2016, §7). *)
}
(** The type for the options of a decoding: what its filter and its ranking
    read. They hold for every window of a recording. *)

val options : Tokenizer.t -> options
(** [options t] is: with timestamps, {!Tokenizer.non_speech} suppressed, a blank
    transcript suppressed, the first timestamp at most [Timestamp 50], 1 s into
    the window, and candidates ranked by their sum over their length. *)

(** {1:start Initial tokens} *)

type start = {
  tokens : int array;
      (** The initial tokens. The sampled tokens start at [Array.length tokens].
      *)
  sot : int;  (** The position of [<|startoftranscript|>] in [tokens]. *)
}
(** The type for the initial tokens of a decoding. *)

val start :
  ?prompt:int array ->
  ?prefix:int array ->
  Tokenizer.t ->
  Model.dims ->
  options ->
  (string * Tokenizer.task) option ->
  start
(** [start ~prompt ~prefix t dims o language] is the initial tokens of a
    decoding of [language] with [o] by a model of [dims] over [t]: the [prompt]
    cut to its last [dims.n_text_ctx / 2 - 1] ids, the [prefix] whole, each
    [[||]] by default. [language] is the language's code and the task of a
    multilingual model, [None] for an English-only one: what
    {!Tokenizer.sot_sequence} takes. A text [s] gives the prompt
    {!prompt}[ t s].

    Raises [Invalid_argument] if [language] is [None] and the model is
    multilingual ({!Model.multilingual}), or [Some] and it is not, if an id of
    the prompt or the prefix is not one of [t]'s, if {!Tokenizer.sot_sequence}
    does, or if the initial tokens are more than [dims.n_text_ctx]. *)

val prompt : Tokenizer.t -> string -> int array
(** [prompt t s] is the ids of the text [s] as a prompt: [s] without the white
    space at its ends, as Python's [str.strip ()] removes it (Unicode's, U+3000
    included), after a space. *)

val suppressed : Tokenizer.t -> options -> int array
(** [suppressed t o] is the ids, in increasing order, that every step
    suppresses: [o.suppress] and the specials [<|transcribe|>], [<|translate|>],
    [<|startoftranscript|>], [<|startofprev|>], [<|startoflm|>] and
    [<|nospeech|>]. The end of text, the languages and, without timestamps, the
    timestamps are not suppressed.

    Raises [Invalid_argument] if an id of [o.suppress] is not one of [t]'s. *)

(** {1:filters Logit filter} *)

val filter :
  Tokenizer.t ->
  options ->
  sampled:'d Nx.int64_t ->
  'd Nx.float32_t ->
  'd Nx.float32_t
(** [filter t o ~sampled logits] is [logits], of shape
    [[| lanes; vocab_size |]], with every token lane [k] may not sample next at
    [-∞] and every other entry as it was, bit for bit. Row [k] of [sampled], of
    shape [[| lanes; w |]], holds the tokens lane [k] sampled after its initial
    tokens, in order, then [-1]. The rules, in order:
    + at a lane's first step, the ids of [Tokenizer.encode t " "] and the end of
      text, where [o.suppress_blank];
    + {!suppressed} at every step;
    + where [o.timestamps]: [<|notimestamps|>]; every timestamp after two
      timestamps, and every token below the end of text after a timestamp that
      follows text; every timestamp before the last one sampled, and that one
      too unless it follows text; at the first step every token but the
      timestamps, and every timestamp past [Timestamp m] for
      [o.max_initial_timestamp = Some m];
    + last, where [o.timestamps], every token below [Timestamp 0] where the
      log-sum-exp of the timestamps' log probabilities is greater than the
      greatest such token's, both under the log-softmax at float32 of the logits
      so filtered. These are two reductions whose bits are those of the kernels'
      library, within the bound it states: a lane whose two sides lie within it
      may decide otherwise on another set or under [Rune.jit].

    It is Nx operations on [logits]' set alone, so it runs on any device and
    under [Rune.jit].

    Raises [Invalid_argument] if [logits] is not [vocab_size] wide, the two
    shapes disagree on [lanes], or as {!suppressed} does. *)

(** {1:search Search} *)

(** The type for how a step chooses each lane's next token. *)
type 'd search =
  | Greedy
      (** One lane per audio; each step takes the greatest filtered logit, the
          first of equal ones. *)
  | Sample of {
      best_of : int;
      temperature : 'd Nx.float32_t;
      key : 'd Nx.Rng.key;
    }
      (** [best_of] lanes per audio, each step drawing from the softmax of the
          filtered logits over [temperature], a positive scalar, with [key]
          folded with the step ({!Nx.Rng.fold_in_tensor}). *)
  | Beam of { size : int; patience : float }
      (** [size] beams per audio. Each beam proposes its [size + 1] likeliest
          next tokens, the lowest id first among equal ones, and the candidates
          rank by their sum of log probabilities, the earlier beam then the
          earlier proposal first among equal sums. The first [size] that do not
          end the text continue; those that end it, ranked above the [size]-th
          continuing one, finish while the audio holds fewer than [m], [m] being
          [size * patience] rounded half to even. An audio's search ends once it
          holds [m] finished. *)

val walk_search :
  ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd search -> 'e search
(** [walk_search c s] walks [s] at [c]'s path: its case, its integers, and
    under [Sample] its [temperature] and [key]. A compiled {!val-decode} keeps
    a program per case. *)

(** {1:decode Decoding} *)

type 'd starts = {
  tokens : 'd Nx.int64_t;
      (** [[| n; w |]]: audio [i]'s initial tokens ({!val-start}) at the end of
          row [i], [-1] before them. *)
  sot : 'd Nx.int64_t;
      (** [[| n |]]: the column of [<|startoftranscript|>] in each row. *)
}
(** The type for the initial tokens of [n] audios. *)

val starts : ?width:int -> start array -> Nx.host starts
(** [starts s] places each start of [s] right-aligned in rows of [width], by
    default the least power of two at or above the longest, so that a compiled
    {!val-decode} serves every start up to [width].

    Raises [Invalid_argument] if [s] is empty or a start is longer than [width].
*)

val walk_starts :
  ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd starts -> 'e starts
(** [walk_starts c s] walks [s] at [c]'s path: [tokens] and [sot]. *)

type 'd outcome = {
  tokens : 'd Nx.int64_t;
      (** [[| n; g; n_text_ctx / 2 |]]: each lane's sampled tokens, the end of
          text included, then [-1]; [g] is [1], [best_of] or [size]. *)
  sums : 'd Nx.float32_t;
      (** [[| n; g |]]: each lane's sum, at float32, of its sampled tokens' log
          probabilities under the softmax of the filtered logits each was
          sampled from, the end of text's included. *)
  finished : 'd Nx.int64_t;
      (** [[| n; m; n_text_ctx / 2 |]]: a beam's finished candidates in the
          order they finished, each ending in the end of text, then [-1]; [m] is
          [0] but under [Beam]. *)
  scores : 'd Nx.float32_t;  (** [[| n; m |]]: their sums. *)
  count : 'd Nx.int64_t;  (** [[| n |]]: the finished candidates held. *)
  no_speech : 'd Nx.float32_t;
      (** [[| n |]]: the probability of [<|nospeech|>] as the token after
          [<|startoftranscript|>], under the softmax of its unfiltered logits:
          the decoder's belief that the window holds no speech. *)
}
(** The type for what a decoding gives. *)

val walk_outcome :
  ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd outcome -> 'e outcome
(** [walk_outcome c o] walks [o] at [c]'s path, its fields in order. *)

val decode :
  ((float, 'b, 'd) Nx.t, 'd) Model.t ->
  Tokenizer.t ->
  options ->
  'd search ->
  'd starts ->
  (float, 'b, 'd) Nx.t ->
  'd outcome
(** [decode m t o search starts features] decodes each audio of the audio
    features [features] ({!Model.encode}), of shape
    [[| n; n_audio_ctx; n_audio_state |]], after its initial tokens, every
    step's logits passed through {!filter} with [o], in [g] lanes as [search]
    says. A token's log probability is under the softmax, at float32, of the
    filtered logits it was sampled from. A lane stops at the end of text, once
    its initial and sampled tokens number more than [n_text_ctx], the last of
    them never read by the decoder, or after [n_text_ctx / 2] sampled tokens; a
    beam's audio stops with its search, or as its lanes do.

    It is Nx and Rune operations on [m]'s set that read no value inside a step:
    applied eagerly it reads one boolean a step, whether every lane has stopped;
    under [Rune.jit] it is one program per dimensions, search case, [n] and [w],
    which reads nothing until it returns.

    Raises [Invalid_argument] as {!filter} and {!Model.val-audio} do, if
    [starts] and [features] disagree on [n], if a [best_of] or [size] is not
    positive, if a beam's [m] is below [1], or if [o.length_penalty] is outside
    [[0, 1]]. *)

type result = {
  tokens : int array;
      (** The sampled tokens, up to the end of text and without it. *)
  text : string;
      (** The text of [tokens] ({!Tokenizer.decode}) without its leading and
          trailing white space. *)
  avg_logprob : float;
      (** The candidate's sum of log probabilities, the end of text's included,
          over [Array.length tokens + 1]. *)
  no_speech_prob : float;  (** The audio's [no_speech]. *)
}
(** The type for the result of decoding one audio. *)

val results : Tokenizer.t -> options -> Nx.host outcome -> result array
(** [results t o out] is each audio's chosen candidate. A beam's candidates are
    its finished ones and, while they number fewer than [size], its lanes by
    decreasing sum, the lower lane first among equal sums; otherwise an audio's
    lanes in order. Each scores, at float64, its sum over its length, its tokens
    before the end of text, or over [((5 + length) / 6) ** a] for
    [o.length_penalty = Some a]; a candidate of no token scores [-∞] without a
    penalty. The first of the greatest scores is chosen.

    Raises [Invalid_argument] if [o.length_penalty] is outside [[0, 1]]. *)

(** {1:languages Languages} *)

val languages :
  ((float, 'b, 'd) Nx.t, 'd) Model.t ->
  Tokenizer.t ->
  (float, 'b, 'd) Nx.t ->
  'd Nx.float32_t
(** [languages m t features] is, for each audio of the audio features [features]
    ({!Model.encode}), of shape [[| n; n_audio_ctx; n_audio_state |]], the
    probability of each of {!Tokenizer.languages}[ t], in that order, as the
    token after [<|startoftranscript|>] read alone: the softmax, at float32, of
    the decoder's logits there over the language tokens. Shape
    [[| n; Array.length (Tokenizer.languages t) |]]. It is Nx operations on
    [m]'s set, so it runs under [Rune.jit].

    Raises [Invalid_argument] if [m] is English-only ({!Model.multilingual}), or
    as {!Model.val-audio} does. *)

(** {1:ratio Compression ratio} *)

val compression_ratio : string -> float
(** [compression_ratio s] measures how much [s] repeats itself: the length of
    [s] in bytes over the length of its zlib stream at level 6,
    {!Lz.Zlib.encode_string}[ ~level:6 s]. A text that loops on a phrase
    compresses well and scores high, and [compression_ratio ""] is [0.]. The
    stream is lz's: another zlib encoder at level 6 may write one of another
    length, so a ratio near a threshold may fall on the other side of it. *)
