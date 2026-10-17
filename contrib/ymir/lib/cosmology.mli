(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Cosmologies: the implementation of {!Ymir.Cosmology}. *)

open Ymir_units

(** {!Ymir.Cosmology.t}. *)
type 'p t = {
  codata : Codata.t;
  h0 : 'p Quantity.t;
  omega_cb : 'p;
  omega_k : 'p;
  w0 : 'p;
  wa : 'p;
  t_cmb : 'p Quantity.t;
  n_eff : 'p;
  m_nu : 'p Quantity.t;
}

val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
val pp : Format.formatter -> (float, 'b) Nx.t t -> unit

(** {1:realisations Realisations} *)

val planck2018 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t t
val planck2015 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t t
val planck2013 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t t
val wmap9 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t t
val wmap7 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t t
val wmap5 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t t
val wmap3 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t t
val wmap1 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t t

(** {1:expansion Expansion} *)

val hubble :
  (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t Quantity.t

(** {!Ymir.Cosmology.component}. *)
type component = Cold_matter | Photons | Neutrinos | Dark_energy | Curvature

val density_parameter :
  (float, 'b) Nx.t t -> component -> (float, 'b) Nx.t -> (float, 'b) Nx.t

val critical_density :
  (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t Quantity.t

(** {1:distances Distances} *)

val comoving_distance :
  (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t Quantity.t

val transverse :
  (float, 'b) Nx.t t ->
  (float, 'b) Nx.t Quantity.t ->
  (float, 'b) Nx.t Quantity.t

val angular_diameter_distance :
  (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t Quantity.t

val luminosity_distance :
  (float, 'b) Nx.t t ->
  ?observed:(float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t Quantity.t

val distance_modulus :
  (float, 'b) Nx.t t ->
  ?observed:(float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t

(** {1:volumes Volumes} *)

val comoving_volume :
  (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t Quantity.t

val comoving_volume_element :
  (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t Quantity.t

(** {1:times Times} *)

val lookback_time :
  (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t Quantity.t

val age : (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t Quantity.t
