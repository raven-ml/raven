(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Directions: the implementation of {!Ymir.Direction}, with the vector
    arithmetic {!Transform} and {!Grid} share.

    {!Transform} builds and reads the record {!t} directly. Vectors as
    triples are their components 0, 1 and 2 ({!components}), each of the
    batch shape. *)

type 'f t = { frame : 'f Frame.t; xyz : (float, Nx.float64_elt) Nx.t }
(** {!Ymir.Direction.t}: [xyz] is [[...; 3]]. *)

val lonlat :
  'f Frame.t ->
  lon:(float, Nx.float64_elt) Nx.t Ymir_units.Quantity.t ->
  lat:(float, Nx.float64_elt) Nx.t Ymir_units.Quantity.t ->
  'f t

val of_xyz : 'f Frame.t -> (float, Nx.float64_elt) Nx.t -> 'f t
val frame : 'f t -> 'f Frame.t
val xyz : 'f t -> (float, Nx.float64_elt) Nx.t
val lon : 'f t -> (float, Nx.float64_elt) Nx.t Ymir_units.Quantity.t
val lat : 'f t -> (float, Nx.float64_elt) Nx.t Ymir_units.Quantity.t
val rotate : ('b Frame.fixed as 'g) Frame.t -> 'a Frame.fixed t -> 'g t

val separation :
  'f t -> 'f t -> (float, Nx.float64_elt) Nx.t Ymir_units.Quantity.t

val position_angle :
  'f t -> 'f t -> (float, Nx.float64_elt) Nx.t Ymir_units.Quantity.t

val ptree : unit -> 'f t Nx.Ptree.t

(** {1:vectors Vectors} *)

val rows :
  string -> (float, Nx.float64_elt) Nx.t -> (float, Nx.float64_elt) Nx.t
(** [rows what v] is [v], [[...; 3]], with each row times the power of two
    that brings its largest component into \[2, 4). The scaling is exact and
    carries no derivative, and squares and cross products of a scaled finite
    row stay in range. A zero row stays zero.

    Raises [Invalid_argument] through {!Nx.check} where a row has an infinite
    component, naming [what], as in
    ["Direction.lon: the vector at [3] is infinite"]. *)

val components : ('a, 'b) Nx.t -> ('a, 'b) Nx.t * ('a, 'b) Nx.t * ('a, 'b) Nx.t
(** [components v] is [v]'s components 0, 1 and 2 on its last axis. *)

val sub3 :
  ('a, 'b) Nx.t * ('c, 'd) Nx.t * ('e, 'f) Nx.t ->
  ('a, 'b) Nx.t * ('c, 'd) Nx.t * ('e, 'f) Nx.t ->
  ('a, 'b) Nx.t * ('c, 'd) Nx.t * ('e, 'f) Nx.t
(** [sub3 a b] is [a - b]. *)

val dot :
  ('a, 'b) Nx.t * ('a, 'b) Nx.t * ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t * ('a, 'b) Nx.t * ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t
(** [dot a b] is [a · b]. *)

val cross :
  ('a, 'b) Nx.t * ('a, 'b) Nx.t * ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t * ('a, 'b) Nx.t * ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t * ('a, 'b) Nx.t * ('a, 'b) Nx.t
(** [cross a b] is [a × b]. *)

val at : int array -> string
(** [at i] is the text that places a message at the batch index [i], as
    [" at [3; 1]"], and [""] for the index [[||]] of a single vector. *)
