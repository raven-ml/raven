(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { year : int; constants : Constant.t list }

let release year rows =
  let constant (name, text, unit) = Constant.v ~name text unit in
  { year; constants = List.map constant rows }

let v2018 = release 2018 Codata_tables.v2018
let v2022 = release 2022 Codata_tables.v2022
let year r = r.year
let constants r = r.constants

(* [find fn name r] is the constant NIST names [name] in [r]. *)
let find fn name r =
  let named k = String.equal (Constant.name k) name in
  match List.find_opt named r.constants with
  | Some k -> k
  | None ->
      invalid_arg
        (Printf.sprintf "Codata.%s: CODATA %d has no %s" fn r.year name)

let newtonian_gravitation =
  find "newtonian_gravitation" "Newtonian constant of gravitation"

let fine_structure = find "fine_structure" "fine-structure constant"
let vacuum_permeability = find "vacuum_permeability" "vacuum mag. permeability"

let vacuum_permittivity =
  find "vacuum_permittivity" "vacuum electric permittivity"

let dalton = find "dalton" "atomic mass constant"
let electron_mass = find "electron_mass" "electron mass"
let muon_mass = find "muon_mass" "muon mass"
let tau_mass = find "tau_mass" "tau mass"
let proton_mass = find "proton_mass" "proton mass"
let neutron_mass = find "neutron_mass" "neutron mass"
let deuteron_mass = find "deuteron_mass" "deuteron mass"
let triton_mass = find "triton_mass" "triton mass"
let helion_mass = find "helion_mass" "helion mass"
let alpha_particle_mass = find "alpha_particle_mass" "alpha particle mass"
let rydberg = find "rydberg" "Rydberg constant"
let bohr_radius = find "bohr_radius" "Bohr radius"

let classical_electron_radius =
  find "classical_electron_radius" "classical electron radius"

let compton_wavelength = find "compton_wavelength" "Compton wavelength"
let thomson_cross_section = find "thomson_cross_section" "Thomson cross section"
let hartree_energy = find "hartree_energy" "Hartree energy"
let bohr_magneton = find "bohr_magneton" "Bohr magneton"
let nuclear_magneton = find "nuclear_magneton" "nuclear magneton"

let electron_magnetic_moment =
  find "electron_magnetic_moment" "electron mag. mom."

let proton_magnetic_moment = find "proton_magnetic_moment" "proton mag. mom."
let electron_g_factor = find "electron_g_factor" "electron g factor"
