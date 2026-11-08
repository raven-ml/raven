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

(* Another machine's host compiles for its arch: on x86-64 its level, which
   Clang takes as its processor, on arm64 the base architecture, untuned. *)
let compiles_for_its_arch () =
  let h = host () in
  let t = Tolk_engine.target h in
  let cpu =
    match Rig.instruction_set h with
    | "x86_64" -> Rig.arch h
    | "arm64" -> "generic"
    | isa -> fail ("another instruction set: " ^ isa)
  in
  equal string (Rig.instruction_set h ^ "," ^ cpu) t.arch;
  equal string "CPU" t.device;
  equal string "CLANG" t.renderer

(* Half casts *)

let set () = Nx.devices [ host () ]

(* The bits of half [x], every NaN as one. *)
let half_bits x =
  Array.map
    (fun b -> if b land 0x7c00 = 0x7c00 && b land 0x3ff <> 0 then 0x7e00 else b)
    (Nx.to_array (Nx.bitcast Nx.uint16 x))

(* Every half, by its bits. *)
let halves () = Nx.bitcast Nx.float16 (Nx.arange Nx.uint16 0 65536 1)

(* Widening keeps every bit a NaN has, with its quiet bit set, as hardware
   conversions do: the results' bits are compared, NaNs included. *)
let widens_to_float32 () =
  let module R = (val set ()) in
  let x = halves () in
  let far = Rune.jit' (Nx.cast Nx.float32) (Nx.place R.on x) in
  let raw y = Nx.to_array (Nx.bitcast Nx.int32 y) in
  equal (array int32)
    (raw (Nx.cast Nx.float32 x))
    (raw (Nx.place Nx.Host.on far))

let widens_to_float64 () =
  let module R = (val set ()) in
  let x = halves () in
  let far = Rune.jit' (Nx.cast Nx.float64) (Nx.place R.on x) in
  let raw y = Nx.to_array (Nx.bitcast Nx.int64 y) in
  equal (array int64)
    (raw (Nx.cast Nx.float64 x))
    (raw (Nx.place Nx.Host.on far))

(* Around each power of two of the halves, from below the least subnormal
   past the greatest finite half: the power, its neighbours a half's and a
   quarter's step away, ties among them, and past 65504 the tie 65520 to an
   infinity. *)
let half_edges =
  let around e =
    let p = Float.ldexp 1. e in
    let ulp = Float.ldexp 1. (Int.max (e - 10) (-24)) in
    List.concat_map
      (fun d -> [ p +. d; p -. d ])
      [ 0.; ulp /. 4.; ulp /. 2.; 3. *. ulp /. 4.; ulp; 3. *. ulp /. 2. ]
  in
  let finite = List.concat_map around (List.init 42 (fun i -> i - 26)) in
  let special =
    [ 0.; -0.; nan; infinity; neg_infinity; 65504.; 65519.; 65520.; 65536.;
      1e5; 1e-8; 1023. *. 0x1p-24; 1.5 *. 0x1p-24; 0x1p-25; 0x1.0000002p-25 ]
  in
  let all = finite @ special in
  Array.of_list (all @ List.map Float.neg all)

let narrows dt () =
  let module R = (val set ()) in
  let x = Nx.create dt [| Array.length half_edges |] half_edges in
  let far = Rune.jit' (Nx.cast Nx.float16) (Nx.place R.on x) in
  equal (array int)
    (half_bits (Nx.cast Nx.float16 x))
    (half_bits (Nx.place Nx.Host.on far))

(* Doubles whose float32 rounding lands on a half's tie: a cast that rounds
   twice rounds them the wrong way. *)
let double_ties =
  Array.of_list
    (List.concat_map
       (fun d -> [ 1. +. 0x1p-11 +. d; 1. +. 0x1p-11 -. d; 0x1p-25 +. d ])
       [ 0x1p-40; 0x1p-50; 0x1p-60 ])

let narrows_from_float64 () =
  let module R = (val set ()) in
  let x = Nx.create Nx.float64 [| Array.length double_ties |] double_ties in
  let far = Rune.jit' (Nx.cast Nx.float16) (Nx.place R.on x) in
  equal (array int)
    (half_bits (Nx.cast Nx.float16 x))
    (half_bits (Nx.place Nx.Host.on far))

(* Floats from uniform bit patterns, every exponent equally likely. *)
let float32s =
  Gen.(array ~size:(constant 64) (map Int32.float_of_bits int32))

let narrows_any =
  prop "any float32 narrows as eager" float32s @@ fun xs ->
  cover ~share:0.5 "a half subnormal or zero"
    (Array.exists (fun v -> Float.abs v < 0x1p-14) xs);
  let module R = (val set ()) in
  let x = Nx.create Nx.float32 [| Array.length xs |] xs in
  let far = Rune.jit' (Nx.cast Nx.float16) (Nx.place R.on x) in
  equal (array int)
    (half_bits (Nx.cast Nx.float16 x))
    (half_bits (Nx.place Nx.Host.on far))

(* Machines

   A compiled program runs on one machine: a set of this machine's host and
   another's, or a place onto another machine's set, raises at the call, naming
   the program's devices. *)

(* The message over this machine's host and another's, [h]. *)
let devices_named h =
  "Rune.jit: the program computes on devices of several machines, [CPU; "
  ^ Rig.name h ^ "]"

let spanning () =
  let h = host () in
  let module R = (val Nx.devices [ Rig.host; h ]) in
  let x = Nx.place R.on (Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |]) in
  raises_match
    (Exn.invalid_arg ~sub:(devices_named h)) (fun () -> Rune.jit' Nx.neg x)

let place_elsewhere () =
  let h = host () in
  let module R = (val Nx.devices [ h ]) in
  let x = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let f x = Nx.place Nx.Host.on (Nx.place R.on x) in
  raises_match
    (Exn.invalid_arg ~sub:(devices_named h)) (fun () -> Rune.jit' f x)

(* Values on one machine's device of such a set compile. *)
let on_one_machine () =
  let h = host () in
  let module R = (val Nx.devices [ Rig.host; h ]) in
  let xs = [| 1.; -2.; 3.; -4. |] in
  let x =
    Nx.place (Nx.Placement.device R.v h) (Nx.create Nx.float32 [| 4 |] xs)
  in
  let y = Rune.jit' Nx.neg x in
  equal (array int32)
    (bits (Nx.neg (Nx.create Nx.float32 [| 4 |] xs)))
    (bits (Nx.place Nx.Host.on y))

let () =
  Watchdog.start ();
  exit
    (run "Rune.jit on another machine's host"
       [
         test "another machine's host compiles for its arch"
           compiles_for_its_arch;
         test "a set spanning machines raises at the call, naming it"
           spanning;
         test "a place onto another machine's set raises at the call"
           place_elsewhere;
         test "values on one machine of a set spanning machines compile"
           on_one_machine;
         group "gives eager's bits"
           [ agrees "neg" { f = Nx.neg }; agrees "floor" { f = Nx.floor } ];
         group "half casts give eager's bits"
           [
             test "every half widens to float32" widens_to_float32;
             test "every half widens to float64" widens_to_float64;
             test "float32 edges narrow" (narrows Nx.float32);
             test "float64 edges narrow" (narrows Nx.float64);
             test "float64 near ties narrows once" narrows_from_float64;
             narrows_any;
           ];
       ])
