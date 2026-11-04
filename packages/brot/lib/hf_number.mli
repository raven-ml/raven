(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** JSON numbers as HuggingFace's tokenizers reads them.

    {b Internal module.} tokenizers parses the numbers of a [tokenizer.json] as
    serde_json does without its [float_roundtrip] feature. The digits are read
    as an unsigned 64-bit integer: in the integer part, a digit that would
    overflow it and the ones after it count only as tens; in the fraction, they
    are dropped. The integer is rounded to a double, and one multiplication or
    division by the double nearest a power of ten scales it, after as many
    divisions by 10{^ 308} as an exponent below -308 needs.
    Where the integer passes 2{^ 53} the two roundings can land one unit in the
    last place from the double the decimal names: [-12.130167007446289], T5's
    score for [---], reads as [-12.130167007446287]. A Unigram segmentation
    compares sums of scores, so that unit decides ties.

    Every step is integer or IEEE arithmetic, so the result is the same on
    every system. *)

val read : string -> first:int -> last:int -> float
(** [read text ~first ~last] is the number whose JSON text is
    [text.\[first..last\]], both ends included, as tokenizers reads it. The text
    must be a JSON number; it is not checked.

    Raises [Failure] when tokenizers refuses the number as out of range: one
    whose value overflows a double. *)
