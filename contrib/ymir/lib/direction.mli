(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Directions: the implementation of {!Ymir.Direction}, with the vector
    arithmetic {!Transform} and {!Grid} share.

    {!Transform} builds and reads the record {!t} directly. Vectors as
    triples are their components 0, 1 and 2 ({!components}), each of the
    batch shape. *)

type ('f, 'd) t = { frame : 'f Frame.t; xyz : (float, Nx.float64_elt, 'd) Nx.t }
(** {!Ymir.Direction.t}: [xyz] is [[...; 3]]. *)

val lonlat :
  'f Frame.t ->
  lon:(float, Nx.float64_elt, 'd) Nx.t Ymir_units.Quantity.t ->
  lat:(float, Nx.float64_elt, 'd) Nx.t Ymir_units.Quantity.t ->
  ('f, 'd) t

val of_xyz : 'f Frame.t -> (float, Nx.float64_elt, 'd) Nx.t -> ('f, 'd) t
val frame : ('f, 'd) t -> 'f Frame.t
val xyz : ('f, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t
val lon : ('f, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t Ymir_units.Quantity.t
val lat : ('f, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t Ymir_units.Quantity.t
val rotate : ('b Frame.fixed as 'g) Frame.t -> ('a Frame.fixed, 'd) t -> ('g, 'd) t

val separation :
  ('f, 'd) t -> ('f, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t Ymir_units.Quantity.t

val position_angle :
  ('f, 'd) t -> ('f, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t Ymir_units.Quantity.t

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('f, 'd) t -> ('f, 'e) t
val ptree : unit -> (('f, 'd) t, 'd) Nx.Ptree.t

(** {1:vectors Vectors} *)

val rows :
  string -> (float, Nx.float64_elt, 'd) Nx.t -> (float, Nx.float64_elt, 'd) Nx.t
(** [rows what v] is [v], [[...; 3]], with each row times the power of two
    that brings its largest component into \[2, 4). The scaling is exact and
    carries no derivative, and squares and cross products of a scaled finite
    row stay in range. A zero row stays zero.

    Raises [Invalid_argument] through {!Nx.check} where a row has an infinite
    component, naming [what], as in
    ["Direction.lon: the vector at [3] is infinite"]. *)

val components : ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
(** [components v] is [v]'s components 0, 1 and 2 on its last axis. *)

val sub3 :
  ('a, 'b, 's) Nx.t * ('c, 'd, 's) Nx.t * ('e, 'f, 's) Nx.t ->
  ('a, 'b, 's) Nx.t * ('c, 'd, 's) Nx.t * ('e, 'f, 's) Nx.t ->
  ('a, 'b, 's) Nx.t * ('c, 'd, 's) Nx.t * ('e, 'f, 's) Nx.t
(** [sub3 a b] is [a - b]. *)

val dot :
  ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t ->
  ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t ->
  ('a, 'b, 'd) Nx.t
(** [dot a b] is [a · b]. *)

val cross :
  ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t ->
  ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t ->
  ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
(** [cross a b] is [a × b]. *)

val at : int array -> string
(** [at i] is the text that places a message at the batch index [i], as
    [" at [3; 1]"], and [""] for the index [[||]] of a single vector. *)
