(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.Rng's draws compiled by Rune.jit on the host give nx's eager draws from
   the same key, bit for bit, but for dirichlet, whose row sum the compiler adds
   in its own order (Draws), at every float dtype and shape Nx_rng_support
   draws. *)

open Windtrap
open Rune_support
module R = Nx_rng_support

let law k =
  prop ~count:4
    (R.name k ^ " compiles to eager's draws")
    (R.gen k)
    (Draws.law
       (module Nx.Host)
       ~dtypes:R.[ Float16; Bfloat16; Float32; Float64 ]
       ~shapes:R.[ Zero; One; Block; Row ]
       k)

let () =
  exit (run "Rune.jit draws" [ group "draws" (List.map law R.samplers) ])
