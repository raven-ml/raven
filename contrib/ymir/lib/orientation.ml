(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Written by test/gen/frames.py; do not edit. Each value is a fixed frame's
   orientation R from ICRS, v_frame = R . v_icrs, in row-major order: its
   definition evaluated at 60 digits and rounded once to float64. *)

[@@@ocamlformat "disable"]

(* r5h^T, r5h the rotation by (-19.9, -9.1, +22.9) mas: FK5's
   orientation from Hipparcos at J2000.0 (Mignard and Froeschle 2000;
   eraFk5hip). *)
let fk5_j2000 = [|
  0x1.fffffffffffc0p-1; -0x1.dcd657fd3c8f3p-24; -0x1.7af89e61cb788p-25;
  0x1.dcd6592ff199bp-24; 0x1.fffffffffff9fp-1; 0x1.9e5e984f985f0p-24;
  0x1.7af8985a25caap-25; -0x1.9e5e99b08a26ep-24; 0x1.fffffffffffcdp-1;
|]

(* R3(-32.93192) R1(90 - 27.12825) R3(90 + 192.85948), in degrees: the
   Hipparcos Catalogue's definition on ICRS (ESA 1997, Vol. 1 1.5.3;
   eraIcrs2g). *)
let galactic = [|
  -0x1.c18a642acb4d6p-5; -0x1.bf332573582edp-1; -0x1.ef727241c3f7cp-2;
  0x1.f9f7d2657bcdep-2; -0x1.c7816b23e14f8p-2; 0x1.7e7474ed9dd07p-1;
  -0x1.bc3ebccbc409ep-1; -0x1.95a910cffa753p-3; 0x1.d2ed6938b6c7cp-2;
|]

(* R1(eps0) B, eps0 = 84381.406", B the IAU 2006 frame bias at J2000
   from the Fukushima-Williams angles gamma = -0.052928",
   phi = 84381.412819", psi = -0.041775" (eraEcm06 at J2000 TT). *)
let ecliptic_j2000 = [|
  0x1.fffffffffffccp-1; -0x1.30037d6267513p-24; 0x1.5a03026ad0b89p-24;
  0x1.1a9546e9bfe11p-25; 0x1.d5c037bd4c600p-1; 0x1.9752da8eda502p-2;
  -0x1.b663a4c38f2b3p-24; -0x1.9752da8eda4a9p-2; 0x1.d5c037bd4c5e6p-1;
|]

(* R3(90) R2(90 - 6.32) R3(47.37) R_galactic, in degrees: the north
   supergalactic pole at Galactic (47.37, +6.32), the origin at
   l = 137.37 (de Vaucouleurs et al. 1976; Lahav et al. 2000). *)
let supergalactic = [|
  0x1.80040eb2b69d7p-2; 0x1.5d8d3425023aap-2; 0x1.b9485c01c6acep-1;
  -0x1.cbf0a8a466cb9p-1; -0x1.88192eac0df82p-4; 0x1.b71371ddc875fp-2;
  0x1.d4bc660a11819p-3; -0x1.debe4dca298bbp-1; 0x1.153fa3c392c72p-2;
|]
