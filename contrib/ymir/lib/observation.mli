(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Observations: the implementation of {!Ymir.Observation}. *)

type ('w, 'e, 'd) t

val v :
  ?variance:(float, 'e, 'd) Nx.t Ymir_units.Quantity.t ->
  ?valid:'d Nx.bit_t ->
  ?area:(float, 'e, 'd) Nx.t Ymir_units.Quantity.t ->
  ('w, 'e, 'd) Grid.t ->
  (float, 'e, 'd) Nx.t Ymir_units.Quantity.t ->
  ('w, 'e, 'd) t

val grid : ('w, 'e, 'd) t -> ('w, 'e, 'd) Grid.t
val data : ('w, 'e, 'd) t -> (float, 'e, 'd) Nx.t Ymir_units.Quantity.t
val variance : ('w, 'e, 'd) t -> (float, 'e, 'd) Nx.t Ymir_units.Quantity.t option
val valid : ('w, 'e, 'd) t -> 'd Nx.bit_t option
val area : ('w, 'e, 'd) t -> (float, 'e, 'd) Nx.t Ymir_units.Quantity.t option
val restrict : 'd Nx.bit_t -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
val map_world : ('w, 'v, 'd) Transform.t -> ('w, 'e, 'd) t -> ('v, 'e, 'd) t
val add : ('w, 'e, 'd) t -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
val sub : ('w, 'e, 'd) t -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
val scale : (float, 'e, 'd) Nx.t Ymir_units.Quantity.t -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
val window : start:'d Nx.int64_t -> shape:int array -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
val around : 'w -> shape:int array -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t

type ('e, 'd) integral = {
  value : (float, 'e, 'd) Nx.t Ymir_units.Quantity.t;
  variance : (float, 'e, 'd) Nx.t Ymir_units.Quantity.t option;
  area : (float, 'e, 'd) Nx.t Ymir_units.Quantity.t;
  coverage : (float, 'e, 'd) Nx.t;
}
(** {!Ymir.Observation.integral}. *)

val integrate : ('w, 'e, 'd) Region.t -> ('w, 'e, 'd) t -> ('e, 'd) integral
val ptree : unit -> (('w, 'e, 'd) t, 'd) Nx.Ptree.t
