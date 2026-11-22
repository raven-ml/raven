(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transcribing a recording of any length, 30 seconds at a time.

    The encoder reads a {e window} of 3000 log-mel frames, 30 s ({!Audio}). A
    recording of [n] samples is transcribed over one log-mel spectrogram,
    {!Audio.log_mel} of its samples followed by {!Audio.length} zeros, floored
    over the whole recording. Its first [n / 160] frames, the {e content}, are
    the recording's own; the next 3000 read the zeros. Frames are 10 ms apart,
    and every time in a transcript is a frame; frame [f] is [f / 100] s into the
    recording. The {e seek} is the frame where the next window starts.

    A {e clip} is a stretch [(start, stop)] of the content, its stop cut to the
    content's end. Each clip in turn is transcribed from its start while the
    seek lies before its stop. Each step:
    + takes the window, the [size] frames from the seek, [size] the least of
      3000 and the frames left in the clip, followed by frames of [0.] up to
      3000;
    + decodes it ({!Decoding.val-decode}) by each rung of the ladder in turn,
      one {e attempt} each, until an attempt passes ({!section-options});
    + if the chosen attempt is {e silent}, its no-speech probability above
      [o.no_speech_threshold] and its average log probability not above
      [o.logprob_threshold] (any, for [None]), advances the seek by [size] and
      changes nothing else;
    + otherwise cuts the attempt's tokens into segments
      ({!section-transcripts}), advances the seek and appends the segments'
      tokens to the {e context}; then empties the context if [o.condition] is
      false or the rung's temperature is above [0.5].

    A window's prompt ({!Decoding.val-start}) is the context, of which the
    decoder reads the last [n_text_ctx / 2 - 1] ids. The context starts as
    [o.prompt]. With [o.carry_prompt] it starts empty, and each window's prompt
    is [o.prompt]'s last [n_text_ctx / 2 - 1] ids followed by as many of the
    context's last ids as fit beside them.

    This transcribes a file with a model [m] over the tokenizer [t], detecting
    its language, and prints each segment once its window is decided:
    {[
    let steps = Whisper.Transcript.steps t (Whisper.Decoding.options t) in
    let each (w : Whisper.Transcript.window) =
      List.iter
        (fun (s : Whisper.Transcript.segment) ->
          Printf.printf "[%d --> %d]%s\n%!" s.start s.stop s.text)
        w.segments
    in
    let r =
      Whisper.Transcript.transcribe ~each steps m Whisper.Transcript.default
        (Whisper.Audio.load "talk.flac")
    in
    print_endline r.text
    ]} *)

(** {1:steps Steps} *)

type compile = {
  compile : 'a 'r 'd. ('a -> 'r, 'd) Nx.Ptree.fn -> ('a -> 'r) -> 'a -> 'r;
}
(** The type for how a step is computed: [compile s f] is the function called
    for [f], whose signature is [s], as [Rune.jit s f] is. *)

type ('b, 'd) steps
(** The type for a window's computations on a model's set over payloads
    [(float, 'b, 'd) Nx.t]: its audio features ({!Model.encode}), its languages'
    probabilities ({!Decoding.val-languages}) and its decoding
    ({!Decoding.val-decode}), with the tokenizer and the decoding options they
    use. A value keeps what its [compile] made, compiled programs included, for
    its life. *)

val steps :
  ?compile:compile -> Tokenizer.t -> Decoding.options -> ('b, 'd) steps
(** [steps ~compile t o] is the steps over [t] with [o], [compile] applied once
    to each. [compile] defaults to calling each function as it is: Nx operations
    applied eagerly, on a set with kernels.
    [{ compile = fun s f -> Rune.jit s f }] compiles them, on any set, a GPU's
    included: a program for the features, one for the languages and one for
    decoding per kind of rung and its integers, for each model's dimensions.
    Every window is 3000 frames, and keys and temperatures are arrays; given
    [compile], every start is placed in rows of the least power of two at or
    above [n_text_ctx / 2 + 4], the longest start ({!Decoding.val-starts}), so
    later windows and recordings compile nothing. Without it each start takes
    its own rows, so a window's first step computes none past its tokens. Make
    the value once and pass it to each transcription.

    Raises [Invalid_argument] as {!Decoding.suppressed} does. *)

(** {1:options Options}

    An attempt {e fails} if the compression ratio of its text
    ({!Decoding.compression_ratio}) is above [o.compression_ratio_threshold], or
    if its average log probability is below [o.logprob_threshold]. A failed
    attempt still passes if its no-speech probability is above
    [o.no_speech_threshold] while its average log probability is below
    [o.logprob_threshold], neither [None]. Such a window holds no speech, and a
    hotter decode would invent some. A threshold of [None] leaves its test out,
    and with [o.no_speech_threshold = None] no window is silent. The last rung's
    attempt is chosen whatever it is.

    The attempt at rung [i], counting from [0], of the window at seek [s] draws
    from [Nx.Rng.fold_in (Nx.Rng.fold_in key s) i]. Seeks increase through a
    transcription, so no two attempts share a key, and an attempt draws the same
    tokens whenever its key, window and prompt are the same. Another set, or
    compiled steps against eager ones, may round the logits otherwise and draw
    another token. *)

(** The type for a rung of the ladder: how an attempt decodes. *)
type rung =
  | Greedy  (** The likeliest token at each step, at temperature [0.]. *)
  | Beam of { size : int; patience : float }
      (** A beam search ({!Decoding.Beam}), at temperature [0.]. *)
  | Sample of { best_of : int; temperature : float }
      (** [best_of] draws at [temperature], positive and finite
          ({!Decoding.Sample}), the best chosen by {!Decoding.results}. *)

type options = {
  language : string option;
      (** The code of the spoken language ({!Tokenizer.languages}), or [None] to
          detect it on a multilingual model: the first of the likeliest under
          {!Decoding.val-languages} of the spectrogram's first 3000 frames. An
          English-only model takes [None] or [Some "en"]. *)
  task : Tokenizer.task;
      (** {!Tokenizer.Translate} needs a multilingual model. *)
  ladder : rung list;  (** The rungs tried in order, at least one. *)
  compression_ratio_threshold : float option;
  logprob_threshold : float option;
  no_speech_threshold : float option;
      (** The tests of the fallback and of silence; [None] leaves one out. *)
  condition : bool;  (** Whether a window's prompt holds the context. *)
  prompt : int array;
      (** Ids of text before the recording, [[||]] for none. A text [s] gives
          {!Decoding.prompt}[ t s]. *)
  carry_prompt : bool;
      (** Whether every window's prompt starts with [prompt]. *)
  clips : (int * int) list;
      (** The clips, in frames, increasing and disjoint; a stop past the
          content, [max_int] included, is the content's end. *)
}
(** The type for the options of a transcription. *)

val default : options
(** [default] detects the language and transcribes; climbs [Greedy] then
    [Sample { best_of = 1; temperature }] at [0.2], [0.4], [0.6], [0.8] and
    [1.]; has the thresholds [Some 2.4], [Some (-1.)] and [Some 0.6]; conditions
    on the context; has no prompt; and transcribes the whole recording,
    [[ (0, max_int) ]]. *)

(** {1:transcripts Transcripts}

    With timestamps ([timestamps] of {!Decoding.type-options}), an attempt's
    tokens are cut as follows, timestamps included and the end of text excluded;
    [Timestamp i] is [2 i] frames after the seek. Where two timestamps follow
    each other, the first closes a segment and the second opens the next. A
    segment runs from its opening timestamp to its closing one. If the tokens
    end in text and then one timestamp, the speech is finished; that last
    stretch is a segment too, and the seek advances by [size]. Otherwise the
    tokens after the last closing timestamp are dropped, to be decoded again in
    the next window, and the seek advances to that timestamp. Tokens with no two
    timestamps in a row are one segment from the seek to their last timestamp,
    or of [size] frames if they hold none or the last is [Timestamp 0], and the
    seek advances by [size].

    Without timestamps the window is one segment of [size] frames, whatever
    timestamps the decoding sampled, and the seek advances by [size].

    A segment's text is that of its tokens below the end of text
    ({!Tokenizer.decode}). A segment that stops where it starts, or whose text
    is blank, keeps its frames and has no tokens and no text. *)

type segment = {
  start : int;  (** Its first frame. *)
  stop : int;  (** The frame where it ends. *)
  tokens : int array;  (** Its tokens, timestamps included. *)
  text : string;  (** The text of [tokens] below the end of text. *)
}
(** The type for segments: a stretch of speech between two timestamps of a
    window, or a whole window. *)

type attempt = {
  rung : rung;
  result : Decoding.result;
  compression_ratio : float;  (** Of [result]'s text. *)
}
(** The type for an attempt at a window. *)

type window = {
  seek : int;  (** Its first frame. *)
  size : int;  (** The content's frames it reads. *)
  choice : attempt;  (** The attempt that passed, or the last rung's. *)
  rejected : attempt list;  (** The attempts that failed before, in order. *)
  silent : bool;  (** Whether [choice] is silent. *)
  segments : segment list;  (** In order; [[]] when [silent]. *)
}
(** The type for a decided window. *)

type t = {
  language : string;  (** The language's code, ["en"] for English-only. *)
  text : string;  (** {!Tokenizer.decode} of every segment's tokens. *)
  windows : window list;  (** In order. *)
}
(** The type for transcripts. *)

val transcribe :
  ?key:Nx.host Nx.Rng.key ->
  ?each:(window -> unit) ->
  ('b, 'd) steps ->
  ((float, 'b, 'd) Nx.t, 'd) Model.t ->
  options ->
  Nx.host Nx.float32_t ->
  t
(** [transcribe ~key ~each steps m o x] is the transcript of the samples [x], of
    shape [[| n |]] at {!Audio.rate}, by [m] with [o], computed by [steps]. The
    spectrogram is computed on the host. Each window is cast to [m]'s dtype,
    placed on [m]'s set and encoded once; each attempt is one decoding, read on
    the host. [each] receives each window once it is decided, in order; an
    exception it raises ends the transcription and passes on. [key] defaults to
    {!Nx.Rng.next_key}[ ()], taken at the call whatever the ladder.

    Raises [Invalid_argument] if [x] has another number of axes than one, if
    [o.ladder] is empty or holds a rung {!Decoding.val-decode} refuses or a
    temperature that is not positive and finite, if a threshold is NaN, if a
    clip starts below [0] or at or past its stop or before the stop of the clip
    before it, if [o.language] or [o.task] does not fit [m], and as
    {!Decoding.val-start} does. *)

(**/**)

type request = { seek : int; size : int; start : Decoding.start; rung : int }

val loop :
  Tokenizer.t ->
  Decoding.options ->
  Model.dims ->
  options ->
  language:string ->
  frames:int ->
  (request -> Decoding.result) ->
  window list
(** [loop t d dims o ~language ~frames decode] is the windows of a recording of
    [frames] content frames, each attempt's result given by [decode]: the seek
    loop without a model, for the suites and check.exe's replay. *)

(**/**)
