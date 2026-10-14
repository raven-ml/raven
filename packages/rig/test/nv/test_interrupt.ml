(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of a GPU through NVIDIA's kernel driver, driven by the
   driver alone. A call on an image of forty functions, each with local
   memory of its own size, is made once for each allocation it makes, with
   every allocation from that one on raising until the call returns, as an
   asynchronous exception does. The image's one unload then frees every
   launch the image made, the interrupted call's included, so the C heap
   holds no more than its allocator's rounding. *)

open Windtrap
module N = Rig_nv
module P = Rig_nv_nvidia

let strf = Printf.sprintf
let timeout = 60.
let functions = List.init 40 (fun k -> strf "local%d" k)
let made = List.filteri (fun k _ -> k < 32) functions
let last = List.nth functions 32

(* The C heap's bytes, once the collector freed what it holds there. *)
let heap () =
  Gc.full_major ();
  Gc.full_major ();
  Option.get (Rig_support.heap_bytes ())

(* The bytes the C heap may hold after a walk: its allocator's rounding. *)
let slack = 256

(* The entries of [fs] on the image [i]. *)
let entries i fs =
  List.iter
    (fun f ->
      match N.entry i f with
      | Entry _ -> ()
      | Absent -> failwith (f ^ " is absent")
      | No_memory n -> failwith (strf "no memory for %s's %d bytes" f n))
    fs

(* [loaded g bin f] is [f i] for a new image [i] of [bin] on [g], which it
   unloads and frees after. *)
let loaded g bin f =
  let i, r, _ = Rig_nv_support.image g bin in
  Fun.protect
    ~finally:(fun () ->
      N.unload g i;
      N.free g r)
    (fun () -> f i)

(* [opened f] is [f g bin] for a device [g] of the first GPU and the fixture
   [bin], once the local memory serves each of [bin]'s functions. *)
let opened f =
  if P.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  let bin = Rig_nv_support.fixture "locals_sm89.cubin" in
  let g = Result.get_ok (Rig_gpu_support.opened (P.gpu 0)) in
  Fun.protect ~finally:(fun () -> N.stop g ~fault:None) @@ fun () ->
  loaded g bin (fun i -> entries i functions);
  f g bin

(* [walk attempt] checks [attempt n], the C heap's growth over an attempt
   interrupted from the [n]-th allocation and whether it was reached, from an
   uninterrupted attempt, which makes what the walk keeps for good, such as
   the memory profile's own, then from the first allocation on. *)
let walk attempt =
  ignore (attempt max_int);
  let rec from n =
    let kept, reached = attempt n in
    at_most
      ~msg:(strf "C heap kept, interrupted from allocation %d" n)
      int ~than:slack kept;
    if reached then from (n + 1)
  in
  from 1

(* An image's 33rd entry, where a table of 32 entries would grow. *)
let test_entries () =
  opened @@ fun g bin ->
  walk @@ fun n ->
  let before = heap () in
  let reached =
    loaded g bin (fun i ->
        entries i made;
        Rig_support.interrupted n (fun () -> entries i [ last ]))
  in
  (heap () - before, reached)

(* The unload of an image with an entry of each function. *)
let test_unload () =
  opened @@ fun g bin ->
  walk @@ fun n ->
  let before = heap () in
  let i, r, _ = Rig_nv_support.image g bin in
  entries i functions;
  let reached = Rig_support.interrupted n (fun () -> N.unload g i) in
  N.free g r;
  (heap () - before, reached)

let () =
  exit
    (run "rig_nv interrupt"
       [
         group ~timeout "an exception at any allocation"
           [
             test
               "of an image's entry leaves its unload to free every launch"
               test_entries;
             test "of an image's unload leaves no launch held" test_unload;
           ];
       ])
