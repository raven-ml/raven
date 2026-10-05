(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** CODATA releases, as {!Ymir_units.Codata} documents them. *)

type t

val v2018 : t
val v2022 : t
val year : t -> int
val constants : t -> Constant.t list
val newtonian_gravitation : t -> Constant.t
val fine_structure : t -> Constant.t
val vacuum_permeability : t -> Constant.t
val vacuum_permittivity : t -> Constant.t
val dalton : t -> Constant.t
val electron_mass : t -> Constant.t
val muon_mass : t -> Constant.t
val tau_mass : t -> Constant.t
val proton_mass : t -> Constant.t
val neutron_mass : t -> Constant.t
val deuteron_mass : t -> Constant.t
val triton_mass : t -> Constant.t
val helion_mass : t -> Constant.t
val alpha_particle_mass : t -> Constant.t
val rydberg : t -> Constant.t
val bohr_radius : t -> Constant.t
val classical_electron_radius : t -> Constant.t
val compton_wavelength : t -> Constant.t
val thomson_cross_section : t -> Constant.t
val hartree_energy : t -> Constant.t
val bohr_magneton : t -> Constant.t
val nuclear_magneton : t -> Constant.t
val electron_magnetic_moment : t -> Constant.t
val proton_magnetic_moment : t -> Constant.t
val electron_g_factor : t -> Constant.t
