(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Definitions shared by every stage. *)

val err : string -> ('a, Format.formatter, unit, 'b) format4 -> 'a
(** [err fn fmt] raises [Invalid_argument] with the message [fmt], prefixed by
    the qualified name of the function [fn]. *)

val is_pos : float -> bool
(** [is_pos x] is [true] iff [x] is finite and positive. *)

val pp_shape : Format.formatter -> int array -> unit
(** [pp_shape] formats a shape as [[2; 3]]. *)

(** {1:ids Ids and warnings} *)

type id = Nx.Ptree.Path.t
type warning = id * string

val pp_id : Format.formatter -> id -> unit
(** [pp_id] formats an id, the root as [root]. *)

val pp_warning : Format.formatter -> warning -> unit
val compare_id : id -> id -> int

(** {1:tensors Tensors} *)

val equal_tensor : ('a, 'b) Nx.t -> ('c, 'd) Nx.t -> bool
(** [equal_tensor x y] is [true] iff [x] and [y] are physically equal. *)

val equal_strings : string array -> string array -> bool
