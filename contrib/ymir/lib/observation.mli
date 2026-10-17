(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Observations: the implementation of {!Ymir.Observation}. *)

type ('w, 'e) t

val v :
  ?variance:(float, 'e) Nx.t Ymir_units.Quantity.t ->
  ?valid:Nx.bit_t ->
  ?area:(float, 'e) Nx.t Ymir_units.Quantity.t ->
  ('w, 'e) Grid.t ->
  (float, 'e) Nx.t Ymir_units.Quantity.t ->
  ('w, 'e) t

val grid : ('w, 'e) t -> ('w, 'e) Grid.t
val data : ('w, 'e) t -> (float, 'e) Nx.t Ymir_units.Quantity.t
val variance : ('w, 'e) t -> (float, 'e) Nx.t Ymir_units.Quantity.t option
val valid : ('w, 'e) t -> Nx.bit_t option
val area : ('w, 'e) t -> (float, 'e) Nx.t Ymir_units.Quantity.t option
val restrict : Nx.bit_t -> ('w, 'e) t -> ('w, 'e) t
val map_world : ('w, 'v) Transform.t -> ('w, 'e) t -> ('v, 'e) t
val add : ('w, 'e) t -> ('w, 'e) t -> ('w, 'e) t
val sub : ('w, 'e) t -> ('w, 'e) t -> ('w, 'e) t
val scale : (float, 'e) Nx.t Ymir_units.Quantity.t -> ('w, 'e) t -> ('w, 'e) t
val window : start:Nx.int64_t -> shape:int array -> ('w, 'e) t -> ('w, 'e) t
val around : 'w -> shape:int array -> ('w, 'e) t -> ('w, 'e) t

type 'e integral = {
  value : (float, 'e) Nx.t Ymir_units.Quantity.t;
  variance : (float, 'e) Nx.t Ymir_units.Quantity.t option;
  area : (float, 'e) Nx.t Ymir_units.Quantity.t;
  coverage : (float, 'e) Nx.t;
}
(** {!Ymir.Observation.integral}. *)

val integrate : ('w, 'e) Region.t -> ('w, 'e) t -> 'e integral
val ptree : unit -> ('w, 'e) t Nx.Ptree.t
