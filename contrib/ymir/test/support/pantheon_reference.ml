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
  { theta = [| 0x1.2e7bec1f4d0b6p-2; 0x1.39da7c530c7bep-1; -0x1.36bff25c6eff8p+4 |];
    sigma = [| 0x1.bc499b64600cdp-5; 0x1.4bb2128453c43p-4; 0x1.140bbf0afd418p-7 |];
    chi2 = 0x1.5e9a709d508a4p+10 }

let sn_flat =
  { theta = [| 0x1.538899b168fa3p-2; -0x1.36cdaaca7c2f6p+4 |];
    sigma = [| 0x1.28fc3dead1371p-6; 0x1.c5cb7492192e5p-8 |];
    chi2 = 0x1.5ebad2ff5867cp+10 }

let shoes =
  { theta = [| 0x1.36e5a4e0f88adp-2; 0x1.4007c9f9a3acbp-1; 0x1.25d75e3e423d0p+6; -0x1.33e5bec6d0303p+4 |];
    sigma = [| 0x1.b8fdc93f6640fp-5; 0x1.47aa6e4515c8ap-4; 0x1.063ef97825bc0p+0; 0x1.e448b780c2e6ap-6 |];
    chi2 = 0x1.6aed12f5c86e6p+10 }

let shoes_flat =
  { theta = [| 0x1.53c68e851e469p-2; 0x1.262192604d934p+6; -0x1.33e7a8c7cc20ap+4 |];
    sigma = [| 0x1.26b070db57c46p-6; 0x1.04569cd2ff56dp+0; 0x1.e417192595f21p-6 |];
    chi2 = 0x1.6b01055d7eea5p+10 }
