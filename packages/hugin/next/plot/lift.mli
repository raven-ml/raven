(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Lifts evaluated where their data lives: each row's quantity or category, and
    the rows that are missing. *)

module Scale := Hugin_next_kit.Scale

type miss = {
  rows : Nx.bool_t option;
      (** The missing rows, broadcastable to the mark's shape, [None] if none
          is: a NaN or an infinity, a [false] in [valid], a value missing for
          the scale, a code outside the labels, or a category outside a domain
          the scale sets. *)
  counts : (Nx.bool_t Lazy.t * (int -> string)) list;
      (** Rows to count, each with the warning of its count. *)
}

(** A lift's rows, indexed by its kind. *)
type _ t =
  | Quantities : { values : Nx.float64_t; miss : miss } -> float t
  | Categories : {
      lift : string Channel.lift;
      codes : Nx.int64_t;
          (** The code, the axis index, or the string's index in the array. *)
      miss : miss;
    }
      -> string t

val eval : int array -> role:string -> 'd Channel.lift -> 'd Scale.t -> 'd t
(** [eval shape ~role l s] is [l] in a mark of shape [shape], read through the
    specification or fitted scale [s]. The counted problems name [role]. *)

val miss : 'd t -> miss
(** [miss l] is the missing rows of [l]. *)

val values : float t -> Nx.float64_t
(** [values q] is each row's quantity, [nan] where it is missing. *)

val positions : string Scale.t -> string t -> Nx.int64_t
(** [positions s c] is each row's index in the domain of the fitted band scale
    [s], [-1] where it is missing or [s] has no such category. *)

val labelled : string Channel.lift -> bool
(** [labelled l] is [true] iff [l] identifies its categories by label, and
    [false] if by integer. *)

val label : string Channel.lift -> int -> string
(** [label l k] is the text that shows the category of code [k] of [l]: its
    label, or its integer in decimal. *)
