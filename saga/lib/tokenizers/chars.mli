(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Character-level tokenization algorithm.

    This module implements character-level tokenization where each character
    maps directly to its Unicode code point. No vocabulary is needed; all
    possible Unicode characters are supported.

    {1 Algorithm Overview}

    Character-level tokenization is trivial: 1. Decode UTF-8 input into Unicode
    code points 2. Each code point becomes a token with ID = code point value 3.
    Preserve character offsets for alignment

    {1 Use Cases}

    - Language modeling at character level
    - Handling any text without vocabulary limitations
    - Fine-grained text generation
    - Debugging and analysis
    - Languages with very large character sets (e.g., Chinese, Japanese)

    {1 Performance Characteristics}

    - Tokenization: O(n) where n = number of characters
    - Memory: O(1) (no vocabulary storage)
    - Token count: Much higher than subword methods for same text

    {1 Limitations}

    - Long sequences: Character-level tokens create much longer sequences than
      subword methods
    - No linguistic structure: Treats all characters equally, ignoring word or
      morpheme boundaries
    - Model capacity: Requires larger models to capture same linguistic patterns
*)

type t
(** Character-level tokenization model.

    Contains no vocabulary; stateless. *)

val create : unit -> t
(** [create ()] constructs character-level tokenizer.

    Model is stateless and requires no configuration. *)

val tokenize : t -> string -> (int * string * (int * int)) list
(** [tokenize model text] encodes text as Unicode code points.

    Returns list of (code_point, char_string, (start_offset, end_offset))
    tuples. Code point is the Unicode scalar value (0 to 0x10FFFF). *)

val token_to_id : t -> string -> int option
(** [token_to_id model char] converts single character to code point.

    Returns [None] if input is not exactly one character. *)

val id_to_token : t -> int -> string option
(** [id_to_token model code_point] converts code point back to character.

    Returns [None] if code point is invalid Unicode. *)

val get_vocab : t -> (string * int) list
(** [get_vocab model] returns empty list (no explicit vocabulary). *)

val get_vocab_size : t -> int
(** [get_vocab_size model] returns theoretical vocabulary size (1114112 for all
    Unicode code points). *)

val save : t -> folder:string -> unit -> string list
(** [save model ~folder ()] no-op for character-level.

    Returns empty list (no files to save). *)
