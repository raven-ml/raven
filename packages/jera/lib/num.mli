(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Facts about float dtypes and the checks every module shares. *)

val eps : (float, 'b) Nx.dtype -> float
(** [eps dtype] is the distance from [1] to the next float of [dtype]. *)

val bits : (float, 'b) Nx.dtype -> int
(** [bits dtype] is the width of [dtype]'s floats in bits, which bounds the
    bisections of their ordered-integer images. *)

val constant : (float, 'b) Nx.dtype -> float array -> (float, 'b) Nx.t
(** [constant dtype a] is [a] as a 1-D tensor of [dtype], each element computed
    on the host in float64 and rounded once. *)

type map = { f : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }
(** The type for maps of float tensors of any dtype. *)

val on_float : string -> map -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t
(** [on_float fn m x] is [m.f x] for a float [x].

    Raises [Invalid_argument] naming [fn] if [x] is not a float tensor. *)

val shape : int array -> string
(** [shape s] formats [s] as ["[2,3]"]. *)

val check_increasing : string -> string -> (float, 'b) Nx.t -> unit
(** [check_increasing fn what x] checks that the 1-D [x] is strictly increasing,
    raising [Invalid_argument] through {!Nx.check} naming [fn], [what], the
    first index that is not above its predecessor and both values. *)

val ordered_midpoint : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [ordered_midpoint a b] is the float halfway between [a <= b] in the order of
    the floats themselves: the midpoint of their ordered-integer images, so a
    bisection by it ends within the dtype's width of steps. *)

val adjacent : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t
(** [adjacent a b] is [true] where no float lies strictly between [a <= b]. *)
