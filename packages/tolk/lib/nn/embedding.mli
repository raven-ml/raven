(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Token embedding: a lookup table from integer indices to dense vectors. *)

open Tolk_frontend

type t = { weight : Tensor.t  (** Shape [(vocab_size, embed_size)]. *) }
(** An embedding layer. *)

val create : int -> int -> t
(** [create vocab_size embed_size] is an embedding table of [vocab_size] rows
    of [embed_size] features, initialised with {!Rand.glorot_uniform}. Load
    trained values with {!State.load_state_dict}. *)

val apply : t -> Tensor.t -> Tensor.t
(** [apply e idx] looks up the row of the table for every element of the
    integer tensor [idx], appending an axis of size [embed_size] to its shape.

    @raise Invalid_argument if [idx] is not an integer tensor. *)
