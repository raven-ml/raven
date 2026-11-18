(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The harmony chat format of gpt-oss.

    gpt-oss reads and writes conversations as a sequence of messages, each
    opened by the token [<|start|>], a header naming who speaks, the token
    [<|message|>], the text, and a closing token. What the assistant writes goes
    to one of three channels that its header names after the token
    [<|channel|>]: its reasoning to [analysis], asides to [commentary] and the
    answer meant for the user to [final].

    {!render} builds the tokens of a prompt up to the point where the
    assistant speaks, and a {!type-parser} reads the tokens the model then
    generates. Tools and function calls are not covered: a message the model
    addresses to a tool is read as text of its channel. *)

(** {1:encodings Encodings} *)

type t
(** The type for harmony encodings: the o200k_harmony tokenizer and the
    identifiers of the tokens that delimit messages. *)

val of_file : string -> t
(** [of_file path] is the encoding of the [tokenizer.json] at [path], the file
    gpt-oss checkpoints ship.

    Raises [Failure] if the file cannot be read, does not hold a tokenizer or
    lacks one of the delimiting tokens. *)

(** {1:prompts Prompts} *)

(** The type for the assistant's channels. *)
type channel = Analysis | Commentary | Final

val render : t -> string -> int array
(** [render t prompt] is the token ids of a conversation in which the user
    says [prompt], ready for the model to continue as the assistant: a system
    message, the user's message, then [<|start|>assistant] left open. The
    system message states the model's identity, its knowledge cutoff, a
    medium reasoning effort and the channels.

    [prompt] is encoded as plain text: a user who types [<|end|>] gets the
    characters, never the token, so the prompt cannot close itself or open
    another message. *)

(** {1:parsing Parsing} *)

type parser
(** The type for parsers of the assistant's tokens. A parser is a value: feeding
    it gives a new one. *)

val parser : t -> parser
(** [parser t] reads what the model generates after the tokens of {!render},
    starting inside the header that [<|start|>assistant] opened. *)

val feed : parser -> int -> parser * (channel * string) option
(** [feed p id] is [p] after the token [id], and the text that [id] completes,
    with the channel it belongs to.

    Headers and delimiting tokens give no text. Neither does a token that ends
    in the middle of a character: its bytes are held until the tokens that
    complete the character arrive, so the concatenation of the texts is always
    valid UTF-8, and is the text of the messages. Bytes still held when a
    message closes are given then, invalid ones as U+FFFD.

    A header that names no known channel is read as {!Final}, so that malformed
    output is shown rather than lost. Feeding a stopped parser gives it back. *)

val stopped : parser -> bool
(** [stopped p] is [true] iff [p] was fed a token that ends the assistant's
    turn: [<|return|>], which closes the final answer, [<|call|>], which hands
    over to a tool, or [<|endoftext|>]. The token [<|end|>] closes a message but
    not the turn: another message follows. *)
