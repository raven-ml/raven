(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The program the suites of rig.program run on an agent's devices and here, and
   the laws of its times. *)

open Windtrap
module G = Rig_program

let le64 v =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int v);
  Bytes.to_string b

let words n f = String.concat "" (List.init n (fun i -> le64 (f i)))
let size = 64
let archs ds = Iarray.of_array (Array.map Rig.arch ds)

(* [ds] and their machine's host. *)
let with_host ds = Array.append ds [| Rig.host_of ds.(0) |]

let polled h =
  match Rig_remote.devices h "POLLED" with
  | Ok ds -> ds
  | Error why -> failf "devices: %s" why

(* test/fixtures' host code [f], an image on device [device]. *)
let host_image device f =
  { G.device; binary = Rig_host_support.fixture ~dir:"../fixtures" f }

(* The value [n], and the value of word [w] of the run's ints. *)
let fixed n = { G.add = n; terms = [||] }
let int_at w = { G.add = 0; terms = [| (w, 1) |] }

let launch ?(scalars : _ iarray = [||]) ~image ~groups name params refs =
  G.Launch
    {
      image;
      name;
      label = name;
      params;
      scalars;
      refs;
      groups = (groups, fixed 1, fixed 1);
      threads = (fixed 1, fixed 1, fixed 1);
      shared = fixed 0;
    }

let ref_ use at slot = { G.slot; at; use }

(* Device 0 fills int 0 words of memory 0, which starts as zeros, from
   [100 + 10 * int 1]; device 1 copies all of it into input 0 through its
   borrow; the host, device 2, adds one to input 1's first word int 1 times
   with test/fixtures' [affine]. *)
let description archs =
  {
    G.devices = archs;
    memory =
      [|
        G.Alloc
          {
            device = 0;
            kind = Rig.Buffer.Device;
            bytes = size;
            init = String.make size '\000';
          };
      |];
    images =
      [|
        { G.device = 0; binary = "functions" };
        { G.device = 1; binary = "functions" };
        host_image 2 "affine";
      |];
    inputs = [| { G.device = 1; bytes = size }; { G.device = 0; bytes = 8 } |];
    ints = 2;
    steps =
      [|
        G.Submit
          {
            device = 0;
            slots = [| G.Memory { memory = 0; offset = 0; length = size } |];
            parts =
              [|
                launch ~image:0 ~groups:(int_at 0) "fill"
                  ~scalars:
                    [|
                      {
                        G.at = 8;
                        width = W64;
                        value = { add = 100; terms = [| (1, 10) |] };
                      };
                    |]
                  (le64 0 ^ le64 0)
                  [| ref_ Write 0 0 |];
              |];
          };
        G.Submit
          {
            device = 1;
            slots =
              [| G.Memory { memory = 0; offset = 0; length = size }; G.Input 0 |];
            parts =
              [|
                launch ~image:1 ~groups:(fixed 1) "copy"
                  (le64 0 ^ le64 0 ^ le64 size)
                  [| ref_ Read 0 0; ref_ Write 8 1 |];
              |];
          };
        G.Loop
          {
            trips = int_at 1;
            trip = None;
            flag = None;
            body =
              [|
                G.Submit
                  {
                    device = 2;
                    slots = [| G.Input 1 |];
                    parts =
                      [|
                        launch ~image:2 ~groups:(fixed 1) "affine"
                          (le64 0 ^ le64 0 ^ le64 1 ^ le64 1 ^ le64 1)
                          [| ref_ Write 0 0; ref_ Read 8 0 |];
                      |];
                  };
              |];
          };
      |];
  }

(* The most groups of a time's fill, and the most groups a core runs in a
   nanosecond: Polled calls its [fill] once per group, and no core makes more
   than one call a cycle at 8 GHz. A fill of [large] groups thus takes at least
   [large / per_ns] ns. *)
let large = 1 lsl 14
let per_ns = 8

(* [description] without its loop, whose body is a step of its own, then a
   submission of no parts on device 0: a program [time] times. Its memory 0
   holds [large] words, all of which its fill may write. *)
let unlooped archs =
  let t = description archs in
  let words = G.Memory { memory = 0; offset = 0; length = 8 * large } in
  let memory =
    Iarray.map
      (function G.Alloc a -> G.Alloc { a with bytes = 8 * large } | m -> m)
      t.memory
  in
  let fill =
    match Iarray.get t.steps 0 with
    | G.Submit s -> G.Submit { s with slots = [| words |] }
    | step -> step
  in
  let body =
    match Iarray.get t.steps 2 with
    | G.Loop { body; _ } -> Iarray.get body 0
    | step -> step
  in
  let none =
    G.Submit { device = 0; slots = [||]; parts = [||] }
  in
  { t with memory; steps = [| fill; Iarray.get t.steps 1; body; none |] }

let expected = (words 8 (fun i -> 100 + i), le64 1)

(* A time's durations against the host's time [window] around its call: at
   least 0 for each of the three steps of parts, each on a device of its own,
   and at most [window], at least [least] for the fill, and 0 for the
   submission of no parts. *)
let window_law ?(least = 0) times window =
  equal ~msg:"one per step" int 4 (Array.length times);
  for i = 0 to 2 do
    let msg = Printf.sprintf "step %d's time" i in
    at_least ~msg int ~than:0 times.(i);
    at_most ~msg int ~than:window times.(i)
  done;
  at_least ~msg:"the fill's time" int ~than:least times.(0);
  equal ~msg:"the submission of no parts" int 0 times.(3)
