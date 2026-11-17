(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.Rng's draws compiled by Rune.jit on the host give nx's eager draws from
   the same key, bit for bit (Draws), at every float dtype and shape
   Nx_rng_support draws. dirichlet's compile to them too with its rows cut
   along their last axis over two devices, and over a 2 by 2 mesh that cuts
   them along one mesh axis and copies them along the other; mapped by
   Rune.vmap, eager and compiled, they are each lane's draw. *)

open Windtrap
open Rune_support
module R = Nx_rng_support

let m = Nx_support.memory

module Two = (val Nx.devices [ m 0; m 1 ])
module Four = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let mesh = Nx.Mesh.v Four.v [ ("x", 2); ("y", 2) ]
let dtypes = R.[ Float16; Bfloat16; Float32; Float64 ]

let law k =
  prop ~count:4
    (R.name k ^ " compiles to eager's draws")
    (R.gen k)
    (Draws.law (module Nx.Host) ~dtypes ~shapes:R.[ Zero; One; Block; Row ] k)

(* [cut r] places an argument of rank [r] cut along its last axis. *)
let placed name whole cut k =
  prop ~count:4
    (R.name k ^ " compiles to eager's draws, its rows " ^ name)
    (R.gen k)
    (R.law (Draws.at whole cut) ~dtypes ~shapes:R.[ One; Block; Row ] k)

let dirichlet = List.find (fun k -> R.name k = "dirichlet") R.samplers

(* A case draws, at float32 and at float64, three lanes of two rows of 1,
   of 8 and of 1025 concentrations each, log-uniform in [e^-9, e^3]. *)

let bits x = Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 x))

let lanes_at (type s) (dt : (float, s) Nx.dtype) seed len =
  let st = Random.State.make [| seed; len |] in
  let n = 3 * 2 * len in
  let a =
    Nx.create dt [| 3; 2; len |]
      (Array.init n (fun _ -> Float.exp (Random.State.float st 12. -. 9.)))
  in
  let key = Nx.Rng.key seed in
  let draw a = Nx.Rng.dirichlet ~key a in
  let each = Nx.stack (List.init 3 (fun i -> draw (Nx.get [ i ] a))) in
  let msg = Printf.sprintf "%s rows of %d" (Nx.Dtype.name dt) len in
  equal ~msg:(msg ^ ", vmap") (array int64) (bits each)
    (bits (Rune.vmap' draw a));
  let compiled =
    Rune.jit Nx.Ptree.(tensor @-> returns tensor) (Rune.vmap' draw) a
  in
  equal ~msg:(msg ^ ", jit of vmap") (array int64) (bits each) (bits compiled)

let lanes_law seed =
  List.iter
    (fun len ->
      lanes_at Nx.float32 seed len;
      lanes_at Nx.float64 seed len)
    [ 1; 8; 1025 ]

let () =
  exit
    (run "Rune.jit draws"
       [
         group "draws" (List.map law R.samplers);
         group "placed"
           [
             placed "cut over two devices" Two.on
               (fun r -> Two.split ~axis:(r - 1))
               dirichlet;
             placed "cut over a mesh axis and copied along the other" Four.on
               (fun r -> Nx.Placement.mesh mesh [ (r - 1, [ "x" ]) ])
               dirichlet;
           ];
         group "lanes"
           [
             prop ~count:10 "dirichlet mapped by vmap is each lane's draw"
               Gen.int lanes_law;
           ];
       ])
