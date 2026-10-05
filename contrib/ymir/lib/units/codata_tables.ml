(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Written by test/units/gen/codata.py from NIST's tables of the CODATA
   adjustments; do not edit. Each row is a constant's name, value with
   uncertainty, and unit. A comment gives NIST's unit where the row's
   unit holds the radian, which NIST writes as 1. Each release names its
   source and the SHA-256 of the text read from it. *)

[@@@ocamlformat "disable"]

(* CODATA 2018: 354 rows, 81 exact, 79 restating another, 194 kept.
   Source: https://physics.nist.gov/cuu/Constants/ArchiveASCII/allascii_2018.txt
   SHA-256: 8c47c05db62c4d314a5244db51a47b4831616e55a8d357ced373a8620ff43be1 *)
let v2018 =
  [
    ("alpha particle-electron mass ratio", "7294.29954142(24)", Unit.one);
    ("alpha particle mass", "6.6446573357(20)e-27", Unit.kilogram);
    ( "alpha particle molar mass",
      "4.0015061777(12)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("alpha particle-proton mass ratio", "3.97259969009(22)", Unit.one);
    ("alpha particle relative atomic mass", "4.001506179127(63)", Unit.one);
    ("Angstrom star", "1.00001495(90)e-10", Unit.metre);
    ("atomic mass constant", "1.66053906660(50)e-27", Unit.kilogram);
    ( "atomic unit of 1st hyperpolarizability",
      "3.2063613061(15)e-53",
      Unit.((coulomb ** 3) * (metre ** 3) * (joule ** -2)) );
    ( "atomic unit of 2nd hyperpolarizability",
      "6.2353799905(38)e-65",
      Unit.((coulomb ** 4) * (metre ** 4) * (joule ** -3)) );
    ( "atomic unit of charge density",
      "1.08120238457(49)e12",
      Unit.(coulomb * (metre ** -3)) );
    ("atomic unit of current", "6.623618237510(13)e-3", Unit.ampere);
    ( "atomic unit of electric dipole mom.",
      "8.4783536255(13)e-30",
      Unit.(coulomb * metre) );
    ( "atomic unit of electric field",
      "5.14220674763(78)e11",
      Unit.(volt * (metre ** -1)) );
    ( "atomic unit of electric field gradient",
      "9.7173624292(29)e21",
      Unit.(volt * (metre ** -2)) );
    ( "atomic unit of electric polarizability",
      "1.64877727436(50)e-41",
      Unit.((coulomb ** 2) * (metre ** 2) * (joule ** -1)) );
    ("atomic unit of electric potential", "27.211386245988(53)", Unit.volt);
    ( "atomic unit of electric quadrupole mom.",
      "4.4865515246(14)e-40",
      Unit.(coulomb * (metre ** 2)) );
    ("atomic unit of energy", "4.3597447222071(85)e-18", Unit.joule);
    ("atomic unit of force", "8.2387234983(12)e-8", Unit.newton);
    ("atomic unit of length", "5.29177210903(80)e-11", Unit.metre);
    ( "atomic unit of mag. dipole mom.",
      "1.85480201566(56)e-23",
      Unit.(joule * (tesla ** -1)) );
    ("atomic unit of mag. flux density", "2.35051756758(71)e5", Unit.tesla);
    ( "atomic unit of magnetizability",
      "7.8910366008(48)e-29",
      Unit.(joule * (tesla ** -2)) );
    ("atomic unit of mass", "9.1093837015(28)e-31", Unit.kilogram);
    ( "atomic unit of momentum",
      "1.99285191410(30)e-24",
      Unit.(kilogram * metre * (second ** -1)) );
    ( "atomic unit of permittivity",
      "1.11265005545(17)e-10",
      Unit.(farad * (metre ** -1)) );
    ("atomic unit of time", "2.4188843265857(47)e-17", Unit.second);
    ( "atomic unit of velocity",
      "2.18769126364(33)e6",
      Unit.(metre * (second ** -1)) );
    ("Bohr magneton", "9.2740100783(28)e-24", Unit.(joule * (tesla ** -1)));
    ("Bohr radius", "5.29177210903(80)e-11", Unit.metre);
    ("characteristic impedance of vacuum", "376.730313668(57)", Unit.ohm);
    ("classical electron radius", "2.8179403262(13)e-15", Unit.metre);
    ("Compton wavelength", "2.42631023867(73)e-12", Unit.metre);
    ("Copper x unit", "1.00207697(28)e-13", Unit.metre);
    ("deuteron-electron mag. mom. ratio", "-4.664345551(12)e-4", Unit.one);
    ("deuteron-electron mass ratio", "3670.48296788(13)", Unit.one);
    ("deuteron g factor", "0.8574382338(22)", Unit.one);
    ("deuteron mag. mom.", "4.330735094(11)e-27", Unit.(joule * (tesla ** -1)));
    ( "deuteron mag. mom. to Bohr magneton ratio",
      "4.669754570(12)e-4",
      Unit.one );
    ( "deuteron mag. mom. to nuclear magneton ratio",
      "0.8574382338(22)",
      Unit.one );
    ("deuteron mass", "3.3435837724(10)e-27", Unit.kilogram);
    ( "deuteron molar mass",
      "2.01355321205(61)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("deuteron-neutron mag. mom. ratio", "-0.44820653(11)", Unit.one);
    ("deuteron-proton mag. mom. ratio", "0.30701220939(79)", Unit.one);
    ("deuteron-proton mass ratio", "1.99900750139(11)", Unit.one);
    ("deuteron relative atomic mass", "2.013553212745(40)", Unit.one);
    ("deuteron rms charge radius", "2.12799(74)e-15", Unit.metre);
    ( "electron charge to mass quotient",
      "-1.75882001076(53)e11",
      Unit.(coulomb * (kilogram ** -1)) );
    ("electron-deuteron mag. mom. ratio", "-2143.9234915(56)", Unit.one);
    ("electron-deuteron mass ratio", "2.724437107462(96)e-4", Unit.one);
    ("electron g factor", "-2.00231930436256(35)", Unit.one);
    (* NIST: s^-1 T^-1 *)
    ( "electron gyromag. ratio",
      "1.76085963023(53)e11",
      Unit.(radian * (second ** -1) * (tesla ** -1)) );
    ("electron-helion mass ratio", "1.819543074573(79)e-4", Unit.one);
    ( "electron mag. mom.",
      "-9.2847647043(28)e-24",
      Unit.(joule * (tesla ** -1)) );
    ("electron mag. mom. anomaly", "1.15965218128(18)e-3", Unit.one);
    ( "electron mag. mom. to Bohr magneton ratio",
      "-1.00115965218128(18)",
      Unit.one );
    ( "electron mag. mom. to nuclear magneton ratio",
      "-1838.28197188(11)",
      Unit.one );
    ("electron mass", "9.1093837015(28)e-31", Unit.kilogram);
    ( "electron molar mass",
      "5.4857990888(17)e-7",
      Unit.(kilogram * (mole ** -1)) );
    ("electron-muon mag. mom. ratio", "206.7669883(46)", Unit.one);
    ("electron-muon mass ratio", "4.83633169(11)e-3", Unit.one);
    ("electron-neutron mag. mom. ratio", "960.92050(23)", Unit.one);
    ("electron-neutron mass ratio", "5.4386734424(26)e-4", Unit.one);
    ("electron-proton mag. mom. ratio", "-658.21068789(20)", Unit.one);
    ("electron-proton mass ratio", "5.44617021487(33)e-4", Unit.one);
    ("electron relative atomic mass", "5.48579909065(16)e-4", Unit.one);
    ("electron-tau mass ratio", "2.87585(19)e-4", Unit.one);
    ( "electron to alpha particle mass ratio",
      "1.370933554787(45)e-4",
      Unit.one );
    ("electron to shielded helion mag. mom. ratio", "864.058257(10)", Unit.one);
    ( "electron to shielded proton mag. mom. ratio",
      "-658.2275971(72)",
      Unit.one );
    ("electron-triton mass ratio", "1.819200062251(90)e-4", Unit.one);
    ( "Fermi coupling constant",
      "1.1663787(6)e-5",
      Unit.((giga electronvolt) ** -2) );
    ("fine-structure constant", "7.2973525693(11)e-3", Unit.one);
    ("Hartree energy", "4.3597447222071(85)e-18", Unit.joule);
    ("helion-electron mass ratio", "5495.88528007(24)", Unit.one);
    ("helion g factor", "-4.255250615(50)", Unit.one);
    ("helion mag. mom.", "-1.074617532(13)e-26", Unit.(joule * (tesla ** -1)));
    ( "helion mag. mom. to Bohr magneton ratio",
      "-1.158740958(14)e-3",
      Unit.one );
    ( "helion mag. mom. to nuclear magneton ratio",
      "-2.127625307(25)",
      Unit.one );
    ("helion mass", "5.0064127796(15)e-27", Unit.kilogram);
    ( "helion molar mass",
      "3.01493224613(91)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("helion-proton mass ratio", "2.99315267167(13)", Unit.one);
    ("helion relative atomic mass", "3.014932247175(97)", Unit.one);
    ("helion shielding shift", "5.996743(10)e-5", Unit.one);
    ("inverse fine-structure constant", "137.035999084(21)", Unit.one);
    ("lattice parameter of silicon", "5.431020511(89)e-10", Unit.metre);
    ("lattice spacing of ideal Si (220)", "1.920155716(32)e-10", Unit.metre);
    ( "molar mass constant",
      "0.99999999965(30)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ( "molar mass of carbon-12",
      "11.9999999958(36)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ( "molar volume of silicon",
      "1.205883199(60)e-5",
      Unit.((metre ** 3) * (mole ** -1)) );
    ("Molybdenum x unit", "1.00209952(53)e-13", Unit.metre);
    ("muon Compton wavelength", "1.173444110(26)e-14", Unit.metre);
    ("muon-electron mass ratio", "206.7682830(46)", Unit.one);
    ("muon g factor", "-2.0023318418(13)", Unit.one);
    ("muon mag. mom.", "-4.49044830(10)e-26", Unit.(joule * (tesla ** -1)));
    ("muon mag. mom. anomaly", "1.16592089(63)e-3", Unit.one);
    ("muon mag. mom. to Bohr magneton ratio", "-4.84197047(11)e-3", Unit.one);
    ("muon mag. mom. to nuclear magneton ratio", "-8.89059703(20)", Unit.one);
    ("muon mass", "1.883531627(42)e-28", Unit.kilogram);
    ("muon molar mass", "1.134289259(25)e-4", Unit.(kilogram * (mole ** -1)));
    ("muon-neutron mass ratio", "0.1124545170(25)", Unit.one);
    ("muon-proton mag. mom. ratio", "-3.183345142(71)", Unit.one);
    ("muon-proton mass ratio", "0.1126095264(25)", Unit.one);
    ("muon-tau mass ratio", "5.94635(40)e-2", Unit.one);
    ("natural unit of energy", "8.1871057769(25)e-14", Unit.joule);
    ("natural unit of length", "3.8615926796(12)e-13", Unit.metre);
    ("natural unit of mass", "9.1093837015(28)e-31", Unit.kilogram);
    ( "natural unit of momentum",
      "2.73092453075(82)e-22",
      Unit.(kilogram * metre * (second ** -1)) );
    ("natural unit of time", "1.28808866819(39)e-21", Unit.second);
    ("neutron Compton wavelength", "1.31959090581(75)e-15", Unit.metre);
    ("neutron-electron mag. mom. ratio", "1.04066882(25)e-3", Unit.one);
    ("neutron-electron mass ratio", "1838.68366173(89)", Unit.one);
    ("neutron g factor", "-3.82608545(90)", Unit.one);
    (* NIST: s^-1 T^-1 *)
    ( "neutron gyromag. ratio",
      "1.83247171(43)e8",
      Unit.(radian * (second ** -1) * (tesla ** -1)) );
    ("neutron mag. mom.", "-9.6623651(23)e-27", Unit.(joule * (tesla ** -1)));
    ( "neutron mag. mom. to Bohr magneton ratio",
      "-1.04187563(25)e-3",
      Unit.one );
    ( "neutron mag. mom. to nuclear magneton ratio",
      "-1.91304273(45)",
      Unit.one );
    ("neutron mass", "1.67492749804(95)e-27", Unit.kilogram);
    ( "neutron molar mass",
      "1.00866491560(57)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("neutron-muon mass ratio", "8.89248406(20)", Unit.one);
    ("neutron-proton mag. mom. ratio", "-0.68497934(16)", Unit.one);
    ("neutron-proton mass difference", "2.30557435(82)e-30", Unit.kilogram);
    ("neutron-proton mass ratio", "1.00137841931(49)", Unit.one);
    ("neutron relative atomic mass", "1.00866491595(49)", Unit.one);
    ("neutron-tau mass ratio", "0.528779(36)", Unit.one);
    ("neutron to shielded proton mag. mom. ratio", "-0.68499694(16)", Unit.one);
    ( "Newtonian constant of gravitation",
      "6.67430(15)e-11",
      Unit.((metre ** 3) * (kilogram ** -1) * (second ** -2)) );
    ("nuclear magneton", "5.0507837461(15)e-27", Unit.(joule * (tesla ** -1)));
    ("Planck length", "1.616255(18)e-35", Unit.metre);
    ("Planck mass", "2.176434(24)e-8", Unit.kilogram);
    ("Planck temperature", "1.416784(16)e32", Unit.kelvin);
    ("Planck time", "5.391247(60)e-44", Unit.second);
    ( "proton charge to mass quotient",
      "9.5788331560(29)e7",
      Unit.(coulomb * (kilogram ** -1)) );
    ("proton Compton wavelength", "1.32140985539(40)e-15", Unit.metre);
    ("proton-electron mass ratio", "1836.15267343(11)", Unit.one);
    ("proton g factor", "5.5856946893(16)", Unit.one);
    (* NIST: s^-1 T^-1 *)
    ( "proton gyromag. ratio",
      "2.6752218744(11)e8",
      Unit.(radian * (second ** -1) * (tesla ** -1)) );
    ("proton mag. mom.", "1.41060679736(60)e-26", Unit.(joule * (tesla ** -1)));
    ( "proton mag. mom. to Bohr magneton ratio",
      "1.52103220230(46)e-3",
      Unit.one );
    ( "proton mag. mom. to nuclear magneton ratio",
      "2.79284734463(82)",
      Unit.one );
    ("proton mag. shielding correction", "2.5689(11)e-5", Unit.one);
    ("proton mass", "1.67262192369(51)e-27", Unit.kilogram);
    ( "proton molar mass",
      "1.00727646627(31)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("proton-muon mass ratio", "8.88024337(20)", Unit.one);
    ("proton-neutron mag. mom. ratio", "-1.45989805(34)", Unit.one);
    ("proton-neutron mass ratio", "0.99862347812(49)", Unit.one);
    ("proton relative atomic mass", "1.007276466621(53)", Unit.one);
    ("proton rms charge radius", "8.414(19)e-16", Unit.metre);
    ("proton-tau mass ratio", "0.528051(36)", Unit.one);
    ( "quantum of circulation",
      "3.6369475516(11)e-4",
      Unit.((metre ** 2) * (second ** -1)) );
    ( "quantum of circulation times 2",
      "7.2738951032(22)e-4",
      Unit.((metre ** 2) * (second ** -1)) );
    (* NIST: m *)
    ( "reduced Compton wavelength",
      "3.8615926796(12)e-13",
      Unit.((radian ** -1) * metre) );
    (* NIST: m *)
    ( "reduced muon Compton wavelength",
      "1.867594306(42)e-15",
      Unit.((radian ** -1) * metre) );
    (* NIST: m *)
    ( "reduced neutron Compton wavelength",
      "2.1001941552(12)e-16",
      Unit.((radian ** -1) * metre) );
    (* NIST: m *)
    ( "reduced proton Compton wavelength",
      "2.10308910336(64)e-16",
      Unit.((radian ** -1) * metre) );
    (* NIST: m *)
    ( "reduced tau Compton wavelength",
      "1.110538(75)e-16",
      Unit.((radian ** -1) * metre) );
    ("Rydberg constant", "10973731.568160(21)", Unit.(metre ** -1));
    ("Sackur-Tetrode constant (1 K, 100 kPa)", "-1.15170753706(45)", Unit.one);
    ( "Sackur-Tetrode constant (1 K, 101.325 kPa)",
      "-1.16487052358(45)",
      Unit.one );
    (* NIST: s^-1 T^-1 *)
    ( "shielded helion gyromag. ratio",
      "2.037894569(24)e8",
      Unit.(radian * (second ** -1) * (tesla ** -1)) );
    ( "shielded helion mag. mom.",
      "-1.074553090(13)e-26",
      Unit.(joule * (tesla ** -1)) );
    ( "shielded helion mag. mom. to Bohr magneton ratio",
      "-1.158671471(14)e-3",
      Unit.one );
    ( "shielded helion mag. mom. to nuclear magneton ratio",
      "-2.127497719(25)",
      Unit.one );
    ( "shielded helion to proton mag. mom. ratio",
      "-0.7617665618(89)",
      Unit.one );
    ( "shielded helion to shielded proton mag. mom. ratio",
      "-0.7617861313(33)",
      Unit.one );
    (* NIST: s^-1 T^-1 *)
    ( "shielded proton gyromag. ratio",
      "2.675153151(29)e8",
      Unit.(radian * (second ** -1) * (tesla ** -1)) );
    ( "shielded proton mag. mom.",
      "1.410570560(15)e-26",
      Unit.(joule * (tesla ** -1)) );
    ( "shielded proton mag. mom. to Bohr magneton ratio",
      "1.520993128(17)e-3",
      Unit.one );
    ( "shielded proton mag. mom. to nuclear magneton ratio",
      "2.792775599(30)",
      Unit.one );
    ("shielding difference of d and p in HD", "2.0200(20)e-8", Unit.one);
    ("shielding difference of t and p in HT", "2.4140(20)e-8", Unit.one);
    ("tau Compton wavelength", "6.97771(47)e-16", Unit.metre);
    ("tau-electron mass ratio", "3477.23(23)", Unit.one);
    ("tau mass", "3.16754(21)e-27", Unit.kilogram);
    ("tau molar mass", "1.90754(13)e-3", Unit.(kilogram * (mole ** -1)));
    ("tau-muon mass ratio", "16.8170(11)", Unit.one);
    ("tau-neutron mass ratio", "1.89115(13)", Unit.one);
    ("tau-proton mass ratio", "1.89376(13)", Unit.one);
    ("Thomson cross section", "6.6524587321(60)e-29", Unit.(metre ** 2));
    ("triton-electron mass ratio", "5496.92153573(27)", Unit.one);
    ("triton g factor", "5.957924931(12)", Unit.one);
    ("triton mag. mom.", "1.5046095202(30)e-26", Unit.(joule * (tesla ** -1)));
    ( "triton mag. mom. to Bohr magneton ratio",
      "1.6223936651(32)e-3",
      Unit.one );
    ( "triton mag. mom. to nuclear magneton ratio",
      "2.9789624656(59)",
      Unit.one );
    ("triton mass", "5.0073567446(15)e-27", Unit.kilogram);
    ( "triton molar mass",
      "3.01550071517(92)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("triton-proton mass ratio", "2.99371703414(15)", Unit.one);
    ("triton relative atomic mass", "3.01550071621(12)", Unit.one);
    ("triton to proton mag. mom. ratio", "1.0666399191(21)", Unit.one);
    ("unified atomic mass unit", "1.66053906660(50)e-27", Unit.kilogram);
    ( "vacuum electric permittivity",
      "8.8541878128(13)e-12",
      Unit.(farad * (metre ** -1)) );
    ( "vacuum mag. permeability",
      "1.25663706212(19)e-6",
      Unit.(newton * (ampere ** -2)) );
    ("weak mixing angle", "0.22290(30)", Unit.one);
    ("W to Z mass ratio", "0.88153(17)", Unit.one);
  ]

(* CODATA 2022: 355 rows, 81 exact, 79 restating another, 195 kept.
   Source: https://physics.nist.gov/cuu/Constants/Table/allascii.txt
   SHA-256: 77fb90e66c40db3e6eb16630bc9c88e4c7c8beddbe5e71be406f2f26e3f67e67 *)
let v2022 =
  [
    ("alpha particle-electron mass ratio", "7294.29954171(17)", Unit.one);
    ("alpha particle mass", "6.6446573450(21)e-27", Unit.kilogram);
    ( "alpha particle molar mass",
      "4.0015061833(12)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("alpha particle-proton mass ratio", "3.972599690252(70)", Unit.one);
    ("alpha particle relative atomic mass", "4.001506179129(62)", Unit.one);
    ("alpha particle rms charge radius", "1.6785(21)e-15", Unit.metre);
    ("Angstrom star", "1.00001495(90)e-10", Unit.metre);
    ("atomic mass constant", "1.66053906892(52)e-27", Unit.kilogram);
    ( "atomic unit of 1st hyperpolarizability",
      "3.2063612996(15)e-53",
      Unit.((coulomb ** 3) * (metre ** 3) * (joule ** -2)) );
    ( "atomic unit of 2nd hyperpolarizability",
      "6.2353799735(39)e-65",
      Unit.((coulomb ** 4) * (metre ** 4) * (joule ** -3)) );
    ( "atomic unit of charge density",
      "1.08120238677(51)e12",
      Unit.(coulomb * (metre ** -3)) );
    ("atomic unit of current", "6.6236182375082(72)e-3", Unit.ampere);
    ( "atomic unit of electric dipole mom.",
      "8.4783536198(13)e-30",
      Unit.(coulomb * metre) );
    ( "atomic unit of electric field",
      "5.14220675112(80)e11",
      Unit.(volt * (metre ** -1)) );
    ( "atomic unit of electric field gradient",
      "9.7173624424(30)e21",
      Unit.(volt * (metre ** -2)) );
    ( "atomic unit of electric polarizability",
      "1.64877727212(51)e-41",
      Unit.((coulomb ** 2) * (metre ** 2) * (joule ** -1)) );
    ("atomic unit of electric potential", "27.211386245981(30)", Unit.volt);
    ( "atomic unit of electric quadrupole mom.",
      "4.4865515185(14)e-40",
      Unit.(coulomb * (metre ** 2)) );
    ("atomic unit of energy", "4.3597447222060(48)e-18", Unit.joule);
    ("atomic unit of force", "8.2387235038(13)e-8", Unit.newton);
    ("atomic unit of length", "5.29177210544(82)e-11", Unit.metre);
    ( "atomic unit of mag. dipole mom.",
      "1.85480201315(58)e-23",
      Unit.(joule * (tesla ** -1)) );
    ("atomic unit of mag. flux density", "2.35051757077(73)e5", Unit.tesla);
    ( "atomic unit of magnetizability",
      "7.8910365794(49)e-29",
      Unit.(joule * (tesla ** -2)) );
    ("atomic unit of mass", "9.1093837139(28)e-31", Unit.kilogram);
    ( "atomic unit of momentum",
      "1.99285191545(31)e-24",
      Unit.(kilogram * metre * (second ** -1)) );
    ( "atomic unit of permittivity",
      "1.11265005620(17)e-10",
      Unit.(farad * (metre ** -1)) );
    ("atomic unit of time", "2.4188843265864(26)e-17", Unit.second);
    ( "atomic unit of velocity",
      "2.18769126216(34)e6",
      Unit.(metre * (second ** -1)) );
    ("Bohr magneton", "9.2740100657(29)e-24", Unit.(joule * (tesla ** -1)));
    ("Bohr radius", "5.29177210544(82)e-11", Unit.metre);
    ("characteristic impedance of vacuum", "376.730313412(59)", Unit.ohm);
    ("classical electron radius", "2.8179403205(13)e-15", Unit.metre);
    ("Compton wavelength", "2.42631023538(76)e-12", Unit.metre);
    ("Copper x unit", "1.00207697(28)e-13", Unit.metre);
    ("deuteron-electron mag. mom. ratio", "-4.664345550(12)e-4", Unit.one);
    ("deuteron-electron mass ratio", "3670.482967655(63)", Unit.one);
    ("deuteron g factor", "0.8574382335(22)", Unit.one);
    ("deuteron mag. mom.", "4.330735087(11)e-27", Unit.(joule * (tesla ** -1)));
    ( "deuteron mag. mom. to Bohr magneton ratio",
      "4.669754568(12)e-4",
      Unit.one );
    ( "deuteron mag. mom. to nuclear magneton ratio",
      "0.8574382335(22)",
      Unit.one );
    ("deuteron mass", "3.3435837768(10)e-27", Unit.kilogram);
    ( "deuteron molar mass",
      "2.01355321466(63)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("deuteron-neutron mag. mom. ratio", "-0.44820652(11)", Unit.one);
    ("deuteron-proton mag. mom. ratio", "0.30701220930(79)", Unit.one);
    ("deuteron-proton mass ratio", "1.9990075012699(84)", Unit.one);
    ("deuteron relative atomic mass", "2.013553212544(15)", Unit.one);
    ("deuteron rms charge radius", "2.12778(27)e-15", Unit.metre);
    ( "electron charge to mass quotient",
      "-1.75882000838(55)e11",
      Unit.(coulomb * (kilogram ** -1)) );
    ("electron-deuteron mag. mom. ratio", "-2143.9234921(56)", Unit.one);
    ("electron-deuteron mass ratio", "2.724437107629(47)e-4", Unit.one);
    ("electron g factor", "-2.00231930436092(36)", Unit.one);
    (* NIST: s^-1 T^-1 *)
    ( "electron gyromag. ratio",
      "1.76085962784(55)e11",
      Unit.(radian * (second ** -1) * (tesla ** -1)) );
    ("electron-helion mass ratio", "1.819543074649(53)e-4", Unit.one);
    ( "electron mag. mom.",
      "-9.2847646917(29)e-24",
      Unit.(joule * (tesla ** -1)) );
    ("electron mag. mom. anomaly", "1.15965218046(18)e-3", Unit.one);
    ( "electron mag. mom. to Bohr magneton ratio",
      "-1.00115965218046(18)",
      Unit.one );
    ( "electron mag. mom. to nuclear magneton ratio",
      "-1838.281971877(32)",
      Unit.one );
    ("electron mass", "9.1093837139(28)e-31", Unit.kilogram);
    ( "electron molar mass",
      "5.4857990962(17)e-7",
      Unit.(kilogram * (mole ** -1)) );
    ("electron-muon mag. mom. ratio", "206.7669881(46)", Unit.one);
    ("electron-muon mass ratio", "4.83633170(11)e-3", Unit.one);
    ("electron-neutron mag. mom. ratio", "960.92048(23)", Unit.one);
    ("electron-neutron mass ratio", "5.4386734416(22)e-4", Unit.one);
    ("electron-proton mag. mom. ratio", "-658.21068789(19)", Unit.one);
    ("electron-proton mass ratio", "5.446170214889(94)e-4", Unit.one);
    ("electron relative atomic mass", "5.485799090441(97)e-4", Unit.one);
    ("electron-tau mass ratio", "2.87585(19)e-4", Unit.one);
    ( "electron to alpha particle mass ratio",
      "1.370933554733(32)e-4",
      Unit.one );
    ( "electron to shielded helion mag. mom. ratio",
      "864.05823986(70)",
      Unit.one );
    ( "electron to shielded proton mag. mom. ratio",
      "-658.2275856(27)",
      Unit.one );
    ("electron-triton mass ratio", "1.819200062327(68)e-4", Unit.one);
    ( "Fermi coupling constant",
      "1.1663787(6)e-5",
      Unit.((giga electronvolt) ** -2) );
    ("fine-structure constant", "7.2973525643(11)e-3", Unit.one);
    ("Hartree energy", "4.3597447222060(48)e-18", Unit.joule);
    ("helion-electron mass ratio", "5495.88527984(16)", Unit.one);
    ("helion g factor", "-4.2552506995(34)", Unit.one);
    ( "helion mag. mom.",
      "-1.07461755198(93)e-26",
      Unit.(joule * (tesla ** -1)) );
    ( "helion mag. mom. to Bohr magneton ratio",
      "-1.15874098083(94)e-3",
      Unit.one );
    ( "helion mag. mom. to nuclear magneton ratio",
      "-2.1276253498(17)",
      Unit.one );
    ("helion mass", "5.0064127862(16)e-27", Unit.kilogram);
    ( "helion molar mass",
      "3.01493225010(94)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("helion-proton mass ratio", "2.993152671552(70)", Unit.one);
    ("helion relative atomic mass", "3.014932246932(74)", Unit.one);
    ("helion shielding shift", "5.9967029(23)e-5", Unit.one);
    ("inverse fine-structure constant", "137.035999177(21)", Unit.one);
    ("lattice parameter of silicon", "5.431020511(89)e-10", Unit.metre);
    ("lattice spacing of ideal Si (220)", "1.920155716(32)e-10", Unit.metre);
    ( "molar mass constant",
      "1.00000000105(31)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ( "molar mass of carbon-12",
      "12.0000000126(37)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ( "molar volume of silicon",
      "1.205883199(60)e-5",
      Unit.((metre ** 3) * (mole ** -1)) );
    ("Molybdenum x unit", "1.00209952(53)e-13", Unit.metre);
    ("muon Compton wavelength", "1.173444110(26)e-14", Unit.metre);
    ("muon-electron mass ratio", "206.7682827(46)", Unit.one);
    ("muon g factor", "-2.00233184123(82)", Unit.one);
    ("muon mag. mom.", "-4.49044830(10)e-26", Unit.(joule * (tesla ** -1)));
    ("muon mag. mom. anomaly", "1.16592062(41)e-3", Unit.one);
    ("muon mag. mom. to Bohr magneton ratio", "-4.84197048(11)e-3", Unit.one);
    ("muon mag. mom. to nuclear magneton ratio", "-8.89059704(20)", Unit.one);
    ("muon mass", "1.883531627(42)e-28", Unit.kilogram);
    ("muon molar mass", "1.134289258(25)e-4", Unit.(kilogram * (mole ** -1)));
    ("muon-neutron mass ratio", "0.1124545168(25)", Unit.one);
    ("muon-proton mag. mom. ratio", "-3.183345146(71)", Unit.one);
    ("muon-proton mass ratio", "0.1126095262(25)", Unit.one);
    ("muon-tau mass ratio", "5.94635(40)e-2", Unit.one);
    ("natural unit of energy", "8.1871057880(26)e-14", Unit.joule);
    ("natural unit of length", "3.8615926744(12)e-13", Unit.metre);
    ("natural unit of mass", "9.1093837139(28)e-31", Unit.kilogram);
    ( "natural unit of momentum",
      "2.73092453446(85)e-22",
      Unit.(kilogram * metre * (second ** -1)) );
    ("natural unit of time", "1.28808866644(40)e-21", Unit.second);
    ("neutron Compton wavelength", "1.31959090382(67)e-15", Unit.metre);
    ("neutron-electron mag. mom. ratio", "1.04066884(24)e-3", Unit.one);
    ("neutron-electron mass ratio", "1838.68366200(74)", Unit.one);
    ("neutron g factor", "-3.82608552(90)", Unit.one);
    (* NIST: s^-1 T^-1 *)
    ( "neutron gyromag. ratio",
      "1.83247174(43)e8",
      Unit.(radian * (second ** -1) * (tesla ** -1)) );
    ("neutron mag. mom.", "-9.6623653(23)e-27", Unit.(joule * (tesla ** -1)));
    ( "neutron mag. mom. to Bohr magneton ratio",
      "-1.04187565(25)e-3",
      Unit.one );
    ( "neutron mag. mom. to nuclear magneton ratio",
      "-1.91304276(45)",
      Unit.one );
    ("neutron mass", "1.67492750056(85)e-27", Unit.kilogram);
    ( "neutron molar mass",
      "1.00866491712(51)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("neutron-muon mass ratio", "8.89248408(20)", Unit.one);
    ("neutron-proton mag. mom. ratio", "-0.68497935(16)", Unit.one);
    ("neutron-proton mass difference", "2.30557461(67)e-30", Unit.kilogram);
    ("neutron-proton mass ratio", "1.00137841946(40)", Unit.one);
    ("neutron relative atomic mass", "1.00866491606(40)", Unit.one);
    ("neutron-tau mass ratio", "0.528779(36)", Unit.one);
    ("neutron to shielded proton mag. mom. ratio", "-0.68499694(16)", Unit.one);
    ( "Newtonian constant of gravitation",
      "6.67430(15)e-11",
      Unit.((metre ** 3) * (kilogram ** -1) * (second ** -2)) );
    ("nuclear magneton", "5.0507837393(16)e-27", Unit.(joule * (tesla ** -1)));
    ("Planck length", "1.616255(18)e-35", Unit.metre);
    ("Planck mass", "2.176434(24)e-8", Unit.kilogram);
    ("Planck temperature", "1.416784(16)e32", Unit.kelvin);
    ("Planck time", "5.391247(60)e-44", Unit.second);
    ( "proton charge to mass quotient",
      "9.5788331430(30)e7",
      Unit.(coulomb * (kilogram ** -1)) );
    ("proton Compton wavelength", "1.32140985360(41)e-15", Unit.metre);
    ("proton-electron mass ratio", "1836.152673426(32)", Unit.one);
    ("proton g factor", "5.5856946893(16)", Unit.one);
    (* NIST: s^-1 T^-1 *)
    ( "proton gyromag. ratio",
      "2.6752218708(11)e8",
      Unit.(radian * (second ** -1) * (tesla ** -1)) );
    ("proton mag. mom.", "1.41060679545(60)e-26", Unit.(joule * (tesla ** -1)));
    ( "proton mag. mom. to Bohr magneton ratio",
      "1.52103220230(45)e-3",
      Unit.one );
    ( "proton mag. mom. to nuclear magneton ratio",
      "2.79284734463(82)",
      Unit.one );
    ("proton mag. shielding correction", "2.56715(41)e-5", Unit.one);
    ("proton mass", "1.67262192595(52)e-27", Unit.kilogram);
    ( "proton molar mass",
      "1.00727646764(31)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("proton-muon mass ratio", "8.88024338(20)", Unit.one);
    ("proton-neutron mag. mom. ratio", "-1.45989802(34)", Unit.one);
    ("proton-neutron mass ratio", "0.99862347797(40)", Unit.one);
    ("proton relative atomic mass", "1.0072764665789(83)", Unit.one);
    ("proton rms charge radius", "8.4075(64)e-16", Unit.metre);
    ("proton-tau mass ratio", "0.528051(36)", Unit.one);
    ( "quantum of circulation",
      "3.6369475467(11)e-4",
      Unit.((metre ** 2) * (second ** -1)) );
    ( "quantum of circulation times 2",
      "7.2738950934(23)e-4",
      Unit.((metre ** 2) * (second ** -1)) );
    (* NIST: m *)
    ( "reduced Compton wavelength",
      "3.8615926744(12)e-13",
      Unit.((radian ** -1) * metre) );
    (* NIST: m *)
    ( "reduced muon Compton wavelength",
      "1.867594306(42)e-15",
      Unit.((radian ** -1) * metre) );
    (* NIST: m *)
    ( "reduced neutron Compton wavelength",
      "2.1001941520(11)e-16",
      Unit.((radian ** -1) * metre) );
    (* NIST: m *)
    ( "reduced proton Compton wavelength",
      "2.10308910051(66)e-16",
      Unit.((radian ** -1) * metre) );
    (* NIST: m *)
    ( "reduced tau Compton wavelength",
      "1.110538(75)e-16",
      Unit.((radian ** -1) * metre) );
    ("Rydberg constant", "10973731.568157(12)", Unit.(metre ** -1));
    ("Sackur-Tetrode constant (1 K, 100 kPa)", "-1.15170753496(47)", Unit.one);
    ( "Sackur-Tetrode constant (1 K, 101.325 kPa)",
      "-1.16487052149(47)",
      Unit.one );
    (* NIST: s^-1 T^-1 *)
    ( "shielded helion gyromag. ratio",
      "2.0378946078(18)e8",
      Unit.(radian * (second ** -1) * (tesla ** -1)) );
    ( "shielded helion mag. mom.",
      "-1.07455311035(93)e-26",
      Unit.(joule * (tesla ** -1)) );
    ( "shielded helion mag. mom. to Bohr magneton ratio",
      "-1.15867149457(94)e-3",
      Unit.one );
    ( "shielded helion mag. mom. to nuclear magneton ratio",
      "-2.1274977624(17)",
      Unit.one );
    ( "shielded helion to proton mag. mom. ratio",
      "-0.76176657721(66)",
      Unit.one );
    ( "shielded helion to shielded proton mag. mom. ratio",
      "-0.7617861334(31)",
      Unit.one );
    (* NIST: s^-1 T^-1 *)
    ( "shielded proton gyromag. ratio",
      "2.675153194(11)e8",
      Unit.(radian * (second ** -1) * (tesla ** -1)) );
    ( "shielded proton mag. mom.",
      "1.4105705830(58)e-26",
      Unit.(joule * (tesla ** -1)) );
    ( "shielded proton mag. mom. to Bohr magneton ratio",
      "1.5209931551(62)e-3",
      Unit.one );
    ( "shielded proton mag. mom. to nuclear magneton ratio",
      "2.792775648(11)",
      Unit.one );
    ("shielding difference of d and p in HD", "1.98770(10)e-8", Unit.one);
    ("shielding difference of t and p in HT", "2.39450(20)e-8", Unit.one);
    ("tau Compton wavelength", "6.97771(47)e-16", Unit.metre);
    ("tau-electron mass ratio", "3477.23(23)", Unit.one);
    ("tau mass", "3.16754(21)e-27", Unit.kilogram);
    ("tau molar mass", "1.90754(13)e-3", Unit.(kilogram * (mole ** -1)));
    ("tau-muon mass ratio", "16.8170(11)", Unit.one);
    ("tau-neutron mass ratio", "1.89115(13)", Unit.one);
    ("tau-proton mass ratio", "1.89376(13)", Unit.one);
    ("Thomson cross section", "6.6524587051(62)e-29", Unit.(metre ** 2));
    ("triton-electron mass ratio", "5496.92153551(21)", Unit.one);
    ("triton g factor", "5.957924930(12)", Unit.one);
    ("triton mag. mom.", "1.5046095178(30)e-26", Unit.(joule * (tesla ** -1)));
    ( "triton mag. mom. to Bohr magneton ratio",
      "1.6223936648(32)e-3",
      Unit.one );
    ( "triton mag. mom. to nuclear magneton ratio",
      "2.9789624650(59)",
      Unit.one );
    ("triton mass", "5.0073567512(16)e-27", Unit.kilogram);
    ( "triton molar mass",
      "3.01550071913(94)e-3",
      Unit.(kilogram * (mole ** -1)) );
    ("triton-proton mass ratio", "2.99371703403(10)", Unit.one);
    ("triton relative atomic mass", "3.01550071597(10)", Unit.one);
    ("triton to proton mag. mom. ratio", "1.0666399189(21)", Unit.one);
    ("unified atomic mass unit", "1.66053906892(52)e-27", Unit.kilogram);
    ( "vacuum electric permittivity",
      "8.8541878188(14)e-12",
      Unit.(farad * (metre ** -1)) );
    ( "vacuum mag. permeability",
      "1.25663706127(20)e-6",
      Unit.(newton * (ampere ** -2)) );
    ("weak mixing angle", "0.22305(23)", Unit.one);
    ("W to Z mass ratio", "0.88145(13)", Unit.one);
  ]
