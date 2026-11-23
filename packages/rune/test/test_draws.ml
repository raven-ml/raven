(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.Rng's draws, eager and compiled by Rune.jit on the host, against nx's
   eager draws from the same key, bit for bit, in rune.mli's order. dirichlet
   mapped by Rune.vmap is each lane's draw, eager and compiled. Its rows cut
   along their last axis over two devices, and over a 2 by 2 mesh that cuts them
   along one mesh axis and copies them along the other, compile to eager's
   draws. Every sampler compiles to them whole (Draws), at every float dtype and
   shape Nx_rng_support draws. Each law compiles a draw once per dtype and
   shape, and its cases run that program. *)

open Windtrap
open Rune_support
module R = Nx_rng_support

let m = Nx_support.memory

module Two = (val Nx.devices [ m 0; m 1 ])
module Four = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let dirichlet = List.find (fun k -> R.name k = "dirichlet") R.samplers

(* Lanes

   A case draws, at float32 and at float64, three lanes of two rows of 1, of 8
   and of 1025 concentrations each, log-uniform in [e^-9, e^3], from one key
   every lane shares. *)

let bits x = Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 x))
let lanes key a = Rune.vmap' (fun a -> Nx.Rng.dirichlet ~key a) a
let lengths = [ 1; 8; 1025 ]

let lanes_at (type s) (dt : (float, s) Nx.dtype) lanes seed len =
  let st = Random.State.make [| seed; len |] in
  let a =
    Nx.create dt [| 3; 2; len |]
      (Array.init
         (3 * 2 * len)
         (fun _ -> Float.exp (Random.State.float st 12. -. 9.)))
  in
  let key = Nx.Rng.key seed in
  let each =
    Nx.stack (List.init 3 (fun i -> Nx.Rng.dirichlet ~key (Nx.get [ i ] a)))
  in
  let msg = Printf.sprintf "%s rows of %d" (Nx.Dtype.name dt) len in
  equal ~msg (array int64) (bits each) (bits (lanes key a))

let lanes_law l32 l64 seed =
  List.iter
    (fun len ->
      lanes_at Nx.float32 l32 seed len;
      lanes_at Nx.float64 l64 seed len)
    lengths

(* One compiled function per dtype, which traces at its first call: each case
   runs the program of each row length. *)
let compiled_lanes =
  let spec () = Nx.Ptree.(Nx.Rng.ptree @-> tensor @-> returns tensor) in
  let checked f k a =
    let before = Rune.evaluations () in
    let y = f k a in
    Compiled.here ();
    equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
    y
  in
  let l32 : Nx.host Nx.Rng.key -> (float, Nx.float32_elt, Nx.host) Nx.t -> _ =
    Rune.jit (spec ()) lanes
  and l64 : Nx.host Nx.Rng.key -> (float, Nx.float64_elt, Nx.host) Nx.t -> _ =
    Rune.jit (spec ()) lanes
  in
  lanes_law (checked l32) (checked l64)

let dtypes = R.[ Float16; Bfloat16; Float32; Float64 ]

(* Placed

   [cut r] places an argument of rank [r] cut along its last axis. *)
let placed name whole cut =
  prop ~count:4
    ("dirichlet compiles to eager's draws, its rows " ^ name)
    (R.gen dirichlet)
    (R.law (Draws.at whole cut) ~dtypes ~shapes:R.[ One; Block; Row ] dirichlet)

let mesh = Nx.Mesh.v Four.v [ ("x", 2); ("y", 2) ]

(* Draws

   Each dtype and shape is a program of its own, compiled once for the cases.
   binomial's, of 48 inversion terms and 18 rejection rounds, takes about 3 s to
   compile, so it draws at its two compute dtypes alone, float32 and float64, at
   no element and at the block, which holds its cases' bounds and rejections: a
   narrower float's program is float32's after a cast of [p], and one element's
   the block's at another extent, which the other samplers hold at every
   dtype. *)
let law k =
  let dtypes, shapes =
    if R.name k = "binomial" then (R.[ Float32; Float64 ], R.[ Zero; Block ])
    else (dtypes, R.[ Zero; One; Block; Row ])
  in
  prop ~count:4
    (R.name k ^ " compiles to eager's draws")
    (R.gen k)
    (Draws.law (module Nx.Host) ~dtypes ~shapes k)

let () =
  exit
    (run "Rune.jit draws"
       [
         group "lanes"
           [
             prop ~count:10 "dirichlet mapped by vmap is each lane's draw"
               Gen.int (lanes_law lanes lanes);
             prop ~count:10
               "dirichlet mapped by vmap and compiled is each lane's draw"
               Gen.int compiled_lanes;
           ];
         group "placed"
           [
             placed "cut over two devices" Two.on (fun r ->
                 Two.split ~axis:(r - 1));
             placed "cut over a mesh axis and copied along the other" Four.on
               (fun r -> Nx.Placement.mesh mesh [ (r - 1, [ "x" ]) ]);
           ];
         group "draws" (List.map law R.samplers);
       ])
