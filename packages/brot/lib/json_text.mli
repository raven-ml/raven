(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** JSON text with numbers as HuggingFace's tokenizers writes them.

    {b Internal module.} tokenizers writes a [tokenizer.json] through
    serde_json, which writes an integer field as its digits and a float field
    as the shortest decimal that reads back as the field's value in the field's
    own type, [f64] or [f32]. A {!Jsont.json} tree holds every number as a
    double, so a float field is marked where the tree is built, by building its
    number with {!f64} or {!f32}. The mark lives in the number's meta and is
    lost if the meta is replaced. The digits come from {!Float_text}, so the
    text is the same on every system. *)

val f64 : float -> Jsont.json
(** [f64 x] is the number [x], written as tokenizers writes an [f64]. *)

val f32 : float -> Jsont.json
(** [f32 x] is the number nearest [x] in single precision, written as
    tokenizers writes an [f32]. *)

val to_string : Jsont.json -> string
(** [to_string j] is [j]'s text, as jsont's encoder writes it with
    [~format:Jsont.Minify] except for numbers. A number built by {!f64} or
    {!f32} is written as tokenizers writes it, as in [0.0],
    [-12.130167007446289], [0.00001], [1e+16] and [1.5e-7]. Any other number
    that is an integer of magnitude below 2{^ 53} is written as its digits
    without a point, its sign included, as in [100] and [-0]; any other
    number is written as by {!f64}. A NaN or an infinity is written [null]. *)
