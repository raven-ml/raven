(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx on a set whose kernels are nx.amd's, on the machine's AMD GPU 0: every
   draw of Nx.Rng has the host's bits (Nx_rng_support). A draw is a program
   of kinds, which nx.amd computes by nx_kinds.h's functions, and of integer
   arithmetic, a dirichlet row's sum among it. Every test skips where rig
   sees no AMD GPU that nx.amd computes on. *)

open Windtrap

let set =
  lazy
    (match Nx_kernels_support.amd () with
    | None -> None
    | Some b -> Some (Nx.devices ~kernels:b.kernels [ b.device ]))

let draws s =
  prop ~count:10
    (Nx_rng_support.name s ^ " has the host's bits")
    (Nx_rng_support.gen s)
    (fun c ->
      match Lazy.force set with
      | None -> skip ~reason:"no AMD GPU that nx.amd computes on" ()
      | Some set ->
          Nx_rng_support.(
            law (placed set)
              ~dtypes:[ Float16; Bfloat16; Float32; Float64 ]
              ~shapes:[ Zero; One; Block; Row ] s c))

let () =
  exit
    (run "Nx.Rng on nx.amd"
       [ group "draws" (List.map draws Nx_rng_support.samplers) ])
