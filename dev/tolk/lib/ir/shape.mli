(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Tensor shapes.

    A shape is an ordered sequence of dimensions. Each dimension is either a
    static integer or a bounded symbolic variable. *)

(** {1:types Types} *)

(** The type for individual dimensions. *)
type dim =
  | Static of int  (** A concrete dimension size. *)
  | Symbol of { name : string; lo : int; hi : int }
      (** A symbolic dimension bounded by \[[lo];[hi]\]. *)

type t
(** The type for shapes. *)

(** {1:constructors Constructors} *)

val scalar : t
(** [scalar] is the empty shape (rank 0). *)

val of_dims : int list -> t
(** [of_dims ns] is a shape where every dimension is {!Static}. *)

val of_dim_list : dim list -> t
(** [of_dim_list ds] is a shape from an explicit list of dimensions. *)

(** {1:access Accessors} *)

val dims : t -> dim list
(** [dims s] is the dimension list of [s]. *)

val rank : t -> int
(** [rank s] is [List.length (dims s)]. *)

val static_dims : t -> int list option
(** [static_dims s] is [Some ns] if every dimension of [s] is {!Static}, where
    [ns] are the sizes. Returns [None] if any dimension is symbolic. *)

(** {1:fmt Formatting} *)

val pp_dim : Format.formatter -> dim -> unit
(** [pp_dim] formats a dimension. Static dimensions are formatted as integers;
    symbolic dimensions as [name[lo..hi]]. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a shape as a bracketed, comma-separated dimension list (e.g.
    [[3, 4, 5]]). *)
