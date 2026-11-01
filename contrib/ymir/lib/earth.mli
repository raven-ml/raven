(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Earth's orientation and motion: the implementation of {!Ymir.Earth},
    with the rotations {!Frame} builds the frames of date from. *)

type 'd tensor = (float, Nx.float64_elt, 'd) Nx.t

type 'd t = {
  tt : (Time.tt, 'd) Time.t;
  ut1_day : (int64, Nx.int64_elt, 'd) Nx.t;
  ut1_second : 'd tensor;
}
(** {!Ymir.Earth.t}: TT, and UT1 as a day and the seconds since its 0h in
    \[0, 86400), all of one shape. *)

val of_utc : (Time.utc, 'd) Time.t -> dut1:'d tensor Ymir_units.Quantity.t -> 'd t
val of_tt : (Time.tt, 'd) Time.t -> delta_t:'d tensor Ymir_units.Quantity.t -> 'd t
val tt : 'd t -> (Time.tt, 'd) Time.t
val era : 'd t -> 'd tensor Ymir_units.Quantity.t
val gmst : 'd t -> 'd tensor Ymir_units.Quantity.t
val gst : 'd t -> 'd tensor Ymir_units.Quantity.t

val heliocentric :
  (Time.tdb, 'd) Time.t ->
  'd tensor Ymir_units.Quantity.t * 'd tensor Ymir_units.Quantity.t

val barycentric :
  (Time.tdb, 'd) Time.t ->
  'd tensor Ymir_units.Quantity.t * 'd tensor Ymir_units.Quantity.t

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd t -> 'e t
val ptree : unit -> ('d t, 'd) Nx.Ptree.t

(** {1:frames For the frames of date} *)

val era_radians : 'd t -> 'd tensor
(** [era_radians e] is {!era}'s value in radians. *)

val intermediate : 'd t -> 'd tensor * 'd tensor
(** [intermediate e] is [(c2i, eo)]: the matrix from GCRS to CIRS,
    [[...; 3; 3]], and the equation of the origins in radians, from one
    evaluation of the precession-nutation's series. *)

val rz : 'd tensor -> 'd tensor
(** [rz a] is the rotation of the axes by [a] radians about z, [[...; 3; 3]],
    [v' = rz a · v]: SOFA's eraRz. *)

val entry : 'd tensor -> int -> int -> 'd tensor
(** [entry m i j] is the entry [(i, j)] of each matrix of [m], [[...; 3; 3]]. *)

val shape : 'd t -> int array
(** [shape e] is the batch shape of [e]'s instants. *)

val beside : 'd t -> 'd Nx.Placement.t
(** [beside e] is the placement of [e]'s tensors. *)
