(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Late memory coalescing and image selection.

    This module mirrors tinygrad [codegen/late/coalese.py]. It owns
    coalesced memory access construction, buffer-to-image selection, and the
    small post-image simplifications that keep image memory float-typed.

    Final [read_imagef]/[write_imagef] emission is rendered by
    {!Tolk.Renderer.Cstyle}, matching tinygrad's [renderer/cstyle.py]
    ownership of backend syntax. *)

val image_valid_dims :
  ?osx:bool ->
  image_pitch_alignment:int option ->
  base:Tolk_uop.Dtype.Val.t ->
  size:int ->
  unit ->
  (int * int) list
(** [image_valid_dims ~image_pitch_alignment ~base ~size] returns
    [(height, width)] candidates for lowering a flat half/float buffer of
    [size] scalar elements to an image whose pitch satisfies
    [image_pitch_alignment]. The optional [osx] flag selects tinygrad's
    macOS byte-alignment exception for one-row images. *)

val drop_valid_stmts :
  Tolk_uop.Uop.t -> Tolk_uop.Uop.t -> int -> int -> Tolk_uop.Uop.t list
(** [drop_valid_stmts valid idx height width] returns validity clauses
    that are redundant for image coordinates [idx] with bounds [height]
    and [width]. *)

val indexing_simplify : Tolk_uop.Upat.Pattern_matcher.t
(** [indexing_simplify] simplifies invalid-gated memory indexes under their
    validity predicate. For image pointers, it also drops valid clauses that
    are already implied by out-of-bounds image coordinates, matching tinygrad's
    [late.coalese.indexing_simplify]. *)

val pm_simplify_add_image :
  Renderer.t -> Tolk_uop.Upat.Pattern_matcher.t
(** [pm_simplify_add_image ren] selects eligible buffer indexes for image
    storage and applies the image-specific cleanup rules from tinygrad
    [pm_simplify_add_image]. It is a matcher so it can run in the same
    graph-rewrite slot as tinygrad, immediately after {!memory_coalesing}. *)

val memory_coalesing : Renderer.t -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [memory_coalesing ren root] folds adjacent scalar loads and stores into
    vector accesses over {!Tolk_uop.Ops.Shrink}, using renderer vector-width
    capabilities and tinygrad's DSP/image special cases. [DMC] disables the
    pass, matching tinygrad. *)
