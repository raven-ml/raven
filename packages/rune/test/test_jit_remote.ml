(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit on another machine's host: its programs compile for that machine
   and run there, giving eager computation's bits on this process's host. The
   other machine is rig's test agent, a real process on the loopback
   (Remote_job). *)

open Windtrap

(* The host of a job over one agent, which the run shares. *)
let remote =
  fixture
    ~teardown:(fun (j, a) ->
      Rig_remote.close j;
      Remote_job.kill a)
    (fun () ->
      let a = Remote_job.with_key_file Remote_job.start in
      (Remote_job.connect [ a ], a))

let host () = List.hd (Rig_remote.hosts (fst (remote ())))

(* An exact function of float32 arrays, on any set. *)
type fn = {
  f :
    'd.
    (float, Bigarray.float32_elt, 'd) Nx.t ->
    (float, Bigarray.float32_elt, 'd) Nx.t;
}

(* The bits of [x]'s elements, every NaN as one: a rounding promises some
   NaN. *)
let bits x =
  Array.map
    (fun v -> if Float.is_nan v then 0x7fc00000l else Int32.bits_of_float v)
    (Nx.to_array x)

(* No element, one, and whole and partial vectors: each length compiles a
   program of its own. *)
let lengths = Gen.one_of (List.map Gen.constant [ 0; 1; 8; 33 ])

(* The elements, and how many the argument's memory holds before them. *)
let values = Gen.(pair (array ~size:lengths any_float) (int_range 0 7))

let edges =
  [ ([| -0.; 0.; nan; infinity; neg_infinity; -2.5; 2.5; 1e-45 |], 1) ]

(* [xs] in memory of [h] that starts [k] elements into its buffer, so that
   the run starts [4 k] bytes past 16 where [k mod 4 <> 0]. *)
let placed (type d) (module R : Nx.Devices with type d = d) h k xs =
  let n = Array.length xs in
  let b = Rig.Buffer.create h ((k + n) * 4) in
  let run = Rig.Buffer.view b ~first:(k * 4) ~length:(n * 4) in
  let x = Nx.create Nx.float32 [| n |] xs in
  if n > 0 then
    Rig.Buffer.copy ~src:(Nx.Repr.read x Nx_array.buffer) ~dst:run;
  Nx.Repr.of_array R.v
    (Nx_array.v Nx.float32
       (Nx_array.Layout.contiguous (Iarray.of_list [ n ]))
       run)

let agrees name { f } =
  prop ~examples:edges name values @@ fun (xs, k) ->
  let n = Array.length xs in
  cover ~share:0.1 "no element" (n = 0);
  cover ~share:0.1 "a NaN or an infinity"
    (Array.exists (fun v -> not (Float.is_finite v)) xs);
  cover ~share:0.3 "a run off 16 bytes" (n > 0 && k mod 4 <> 0);
  let h = host () in
  let module R = (val Nx.devices [ h ]) in
  let x = Nx.create Nx.float32 [| n |] xs in
  let far = Rune.jit' f (placed (module R) h k xs) in
  equal (array int32) (bits (f x)) (bits (Nx.place Nx.Host.on far))

(* The processor another machine's host compiles for, by its instruction
   set. *)
let baselines = [ ("x86_64", "x86-64"); ("arm64", "generic") ]

let compiles_for_baseline =
  cases ~id:fst "another machine's host compiles for its baseline" baselines
  @@ fun (machine, cpu) ->
  let h = host () in
  if Rig.arch h <> machine then skip ~reason:"another instruction set" ();
  let t = Tolk_engine.target h in
  equal string (machine ^ "," ^ cpu) t.arch;
  equal string "CPU" t.device;
  equal string "CLANG" t.renderer

let () =
  Watchdog.start ();
  exit
    (run "Rune.jit on another machine's host"
       [
         compiles_for_baseline;
         group "gives eager's bits"
           [ agrees "neg" { f = Nx.neg }; agrees "floor" { f = Nx.floor } ];
       ])
