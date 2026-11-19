(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Whisper's tokenizers: text to token ids and back.

    Whisper reads and writes text as the ids of a tiktoken encoding
    ({!Brot.section-tiktoken}): the ranks of a rank file, the English-only
    models' [gpt2.tiktoken] or the others' [multilingual.tiktoken], cut into
    pretokens by Whisper's pattern. After the ranks come Whisper's
    {e special tokens}, numbered from the number of ranks on in this order:
    [<|endoftext|>], [<|startoftranscript|>], one [<|code|>] per language for
    the first {e languages} codes of {!languages}' order, [<|translate|>],
    [<|transcribe|>], [<|startoflm|>], [<|startofprev|>], [<|nospeech|>],
    [<|notimestamps|>], then the 1501 {e timestamps} [<|0.00|>] to [<|30.00|>],
    one every 0.02 s. The English-only models have 99 languages, the
    multilingual ones 99 or 100 (large-v3 added Cantonese, [yue]).

    The decoder starts each transcript with a {e start sequence}: the start of
    transcript, then, for a multilingual model, the token of the spoken language
    and the token of the task. It then reads the text's ids, the timestamp ids
    that mark where segments start and end, and the end of text. {!decode} drops
    the timestamps; {!decode_with_timestamps} writes them as their text.

    Encoding never matches a special token's text: [encode t "<|en|>"] is the
    ids of the six characters. *)

type t
(** The type for Whisper's tokenizers. *)

(** {1:constructors Constructors} *)

val of_ranks : languages:int -> string -> (t, string) result
(** [of_ranks ~languages ranks] is the tokenizer of the rank file [ranks], its
    text read as {!Brot.of_tiktoken} reads it, with the special tokens of
    [languages] languages.

    Errors as {!Brot.of_tiktoken} does. Raises [Invalid_argument] if [languages]
    is outside \[[0], [100]\]. *)

val vocab_size : t -> int
(** [vocab_size t] is the number of [t]'s tokens, the ranks' and the specials':
    a model over [t] has as many logits. *)

(** {1:specials Special tokens} *)

(** The type for tasks: transcribe the speech in its language, or translate it
    into English. *)
type task = Transcribe | Translate

(** The type for special tokens. *)
type special =
  | End_of_text  (** [<|endoftext|>]. *)
  | Start_of_transcript  (** [<|startoftranscript|>]. *)
  | Language of string
      (** The language of this code: [Language "en"] is [<|en|>]. *)
  | Task of task  (** [<|transcribe|>] or [<|translate|>]. *)
  | Start_of_lm  (** [<|startoflm|>]. *)
  | Start_of_prev  (** [<|startofprev|>], which opens a prompt. *)
  | No_speech  (** [<|nospeech|>]. *)
  | No_timestamps  (** [<|notimestamps|>]. *)
  | Timestamp of int
      (** [Timestamp i] is the time [i * 0.02] s, [<|0.00|>] to [<|30.00|>] for
          [i] from [0] to [1500]. *)

val languages : t -> string array
(** [languages t] is the codes of [t]'s languages in their tokens' order:
    ["en"; "zh"; "de"; ...], [yue] last when [t] has 100. *)

val special : t -> special -> int
(** [special t s] is the id of [s] in [t].

    Raises [Invalid_argument] if [s] is a [Language] whose code [t] has no token
    for, or [Timestamp i] with [i] outside \[[0], [1500]\]. *)

val sot_sequence : t -> (string * task) option -> int array
(** [sot_sequence t prompt] is the start sequence of a transcript: the start of
    transcript, then, with [Some (code, task)], the tokens of [Language code]
    and [Task task]. An English-only model takes [None]. A transcript without
    timestamps appends {!No_timestamps}.

    Raises [Invalid_argument] as {!special} does. *)

val non_speech : t -> int array
(** [non_speech t] is the ids, in increasing order, that a decoder suppresses so
    as not to write speaker tags or annotations of sounds, such as [♪♪♪],
    [( SPEAKING FOREIGN LANGUAGE )] or [[DAVID] Hey]. They are the first id of
    [encode t " -"] and of [encode t " '"], every [encode t s] and
    [encode t (" " ^ s)] that is one id for [s] among the symbols
    {v
        " # ( ) * + / : ; < = > @ [ \ ] ^ _ ` { | } ~ 「 」 『 』
    << >> <<< >>> -- --- -( -[ (' (" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪
    v}
    and the first id of both encodings of each of [♩ ♪ ♫ ♬ ♭ ♮ ♯]. A hyphen or
    an apostrophe within a word stays allowed. *)

(** {1:text Text} *)

val encode : t -> string -> int array
(** [encode t s] is the ids of the text [s] under [t]'s ranks and pattern. The
    text of a special token is text: it never encodes to that token. *)

val decode : t -> int array -> string
(** [decode t ids] is the text of [ids] with every timestamp dropped. Each other
    special token is its text. Bytes that are no UTF-8 decode to U+FFFD as
    tiktoken's [errors="replace"] does, and an id that names no token gives no
    text. *)

val decode_with_timestamps : t -> int array -> string
(** [decode_with_timestamps t ids] is {!decode} with each timestamp written as
    its text, as in [<|0.00|> Hello world.<|1.08|>]. *)
