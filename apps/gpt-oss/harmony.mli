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

    {!render} builds the tokens of a conversation up to the point where the
    assistant speaks, and a {!type-parser} reads the tokens the model then
    generates. Tools and function calls are not covered: nothing renders their
    definitions, and a message the model addresses to a tool is read as text of
    its channel. *)

(** {1:encodings Encodings} *)

type t
(** The type for harmony encodings: the o200k_harmony tokenizer and the
    identifiers of the tokens that delimit messages. *)

val of_file : string -> t
(** [of_file path] is the encoding of the [tokenizer.json] at [path], the file
    gpt-oss checkpoints ship.

    Raises [Failure] if the file does not hold a tokenizer or lacks one of the
    delimiting tokens, and [Sys_error] if it cannot be read. *)

(** {1:conversations Conversations} *)

(** The type for reasoning efforts: how much the system message asks the model
    to write on the analysis channel before it answers. *)
type effort = Low | Medium | High

(** The type for the assistant's channels. *)
type channel = Analysis | Commentary | Final

(** The type for the turns of a conversation: what the user said, or one message
    of the assistant on a channel. *)
type turn = User of string | Assistant of channel * string

val render :
  t ->
  ?date:string ->
  ?instructions:string ->
  effort:effort ->
  turn list ->
  int array
(** [render t ~effort turns] is the token ids of the conversation [turns], ready
    for the model to continue as the assistant: a system message, a developer
    message if there are [instructions], the [turns] in order, then
    [<|start|>assistant] left open.

    The system message states the model's identity, its knowledge cutoff, the
    current [date] if given (as [YYYY-MM-DD]), the reasoning [effort] and the
    channels. The developer message carries [instructions], what a chat
    interface calls the system prompt.

    The text of a turn is encoded as plain text: a user who types [<|end|>] gets
    the characters, never the token, so no turn can close itself or open
    another.

    Earlier reasoning is not shown again: when the last assistant message of
    [turns] is on {!Final}, the {!Analysis} messages that come before the first
    {!Final} one are left out, as the model was trained to expect. *)

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
