(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Programs over specialised metallibs, loaded by an agent on its machine's
   Metal GPU: the image's bytes go from the controller through the wire to the
   agent's Rig_metal.image, which specialises them there. *)

open Windtrap
open Remote_job
module G = Rig_program

(* rig_metal.mli's Code section: "RMSC", a count, that many little-endian
   32-bit words, then a metallib. *)
let specialised ws =
  let lib = Rig_metal_support.fixture ~dir:"../metal/fixtures" "constants" in
  let b = Buffer.create (8 + (4 * List.length ws) + String.length lib) in
  Buffer.add_string b "RMSC";
  Buffer.add_int32_le b (Int32.of_int (List.length ws));
  List.iter (fun w -> Buffer.add_int32_le b (Int32.of_int w)) ws;
  Buffer.add_string b lib;
  Buffer.contents b

let fixed n = { G.add = n; terms = [||] }

(* constants.metal's [pair], once over one thread on device 0, writing its
   constants [a] and [b] into input 0. *)
let pair arch ws =
  let launch =
    G.Launch
      {
        image = 0;
        name = "pair";
        label = "pair";
        params = String.make 8 '\000';
        scalars = [||];
        refs = [| { G.slot = 0; at = 0; use = Write } |];
        groups = (fixed 1, fixed 1, fixed 1);
        threads = (fixed 1, fixed 1, fixed 1);
        shared = fixed 0;
      }
  in
  {
    G.devices = [| arch |];
    ints = 0;
    memory = [||];
    images = [| { G.device = 0; binary = specialised ws } |];
    inputs = [| { G.device = 0; bytes = 8 } |];
    steps =
      [| G.Submit { device = 0; slots = [| G.Input 0 |]; parts = [| launch |] } |];
  }

(* [f] with the agent's Metal GPU. *)
let with_metal f =
  if not (Rig_metal_support.present ()) then
    skip ~reason:"the machine has no METAL GPU rig drives" ();
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  match Rig_remote.devices h "METAL" with
  | Ok [ d ] -> f d
  | Ok ds -> failf "the agent has %d METAL devices" (List.length ds)
  | Error why -> failf "devices: %s" why

let words s = List.init 2 (fun i -> Int32.to_int (String.get_int32_le s (4 * i)))

let runs () =
  with_metal @@ fun d ->
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load (pair (Rig.arch d) [ 31; 32 ]) [| d |])
  in
  let out = far_of_string d (String.make 8 '\000') in
  ignore (G.run p { inputs = [| out |]; ints = [||] } : Rig.Point.t array);
  equal (list int) [ 31; 32 ] (words (read out))

let refused () =
  with_metal @@ fun d ->
  let why = require_error (G.load (pair (Rig.arch d) [ 31 ]) [| d |]) in
  contains ~sub:"its required constant b (1) is past the image's 1 value" why

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.metal"
       [
         test ~timeout:60.
           "a program over a specialised metallib runs with its values on an \
            agent's GPU"
           runs;
         test ~timeout:60.
           "an agent's load refuses a specialised metallib that lacks a \
            required value"
           refused;
       ])
