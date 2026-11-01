(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Written by test/gen/pantheon.py; do not edit. *)

[@@@ocamlformat "disable"]

(* scipy's maximum-likelihood parameters, their Fisher errors and chi^2. *)
type fit = { theta : float array; sigma : float array; chi2 : float }

let distances = "Pantheon+SH0ES.dat"
let distances_url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
let distances_blake2b = "f8c368550776f1b5d5df9ff1add0ce2f28b93f9015013fb1e914e7e8de7fed2d"
let covariance = "Pantheon+SH0ES_STAT+SYS.cov"
let covariance_url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
let covariance_blake2b = "69a334aff9f2a0ad51e9c5c74cd05856283cef018b7878b906e09f7fc2b0b14d"
let n_sn = 1590
let n_shoes = 1657

let sn =
  { theta = [| 0x1.2e7bec68079ebp-2; 0x1.39da7c3b099c0p-1; -0x1.36bff25c220f8p+4 |];
    sigma = [| 0x1.bc499b8ef072cp-5; 0x1.4bb212a8d5d9bp-4; 0x1.140bbf316b726p-7 |];
    chi2 = 0x1.5e9a709d50891p+10 }

let sn_flat =
  { theta = [| 0x1.538899b97c5cap-2; -0x1.36cdaaca70b2ap+4 |];
    sigma = [| 0x1.28fc3df54e07cp-6; 0x1.c5cb74a26ba2fp-8 |];
    chi2 = 0x1.5ebad2ff585dfp+10 }

let shoes =
  { theta = [| 0x1.36e5a4f0f036ap-2; 0x1.4007c9f4b8723p-1; 0x1.25d75e3dbb762p+6; -0x1.33e5bec6cf0fdp+4 |];
    sigma = [| 0x1.b8fdc91cc706bp-5; 0x1.47aa6e0949726p-4; 0x1.063ef974c88a8p+0; 0x1.e448b780b0b9cp-6 |];
    chi2 = 0x1.6aed12f5c867cp+10 }

let shoes_flat =
  { theta = [| 0x1.53c68e85c7751p-2; 0x1.262192603453cp+6; -0x1.33e7a8c7ce391p+4 |];
    sigma = [| 0x1.26b070cf8d5c6p-6; 0x1.04569cd244f6bp+0; 0x1.e4171925a17bdp-6 |];
    chi2 = 0x1.6b01055d7ed36p+10 }
