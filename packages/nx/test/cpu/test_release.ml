(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's promise: a call keeps other domains waiting to collect for no
   longer than about one unloaded core's copy of 1 MiB at a time, however
   many operands, reductions or program nodes it has, so a call whose work
   takes longer runs it with the domain lock released; a released call
   whose work takes several threads waits for any other such job to end;
   and several domains may call at once, each result a function of its
   operands' values alone. The runtime's hooks on a blocking section show
   when the calling thread releases the lock and how long it held it. One
   call per family of kernels, at a size whose work lies past the copy,
   whether it runs inline on the calling thread or as a job, and one per
   input of nx.cpu's estimate of the work that no family's call takes: a
   batch of matrices each under the copy, a padded load, and a map of no
   operand. The reduction's call measures its longest held stretch too. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog

(* nx.cpu, whose one-node maps the tests call. *)
let cpu = (module Nx_cpu : Nx_kernel.S)

(* [f ()] releases the domain lock. *)
let releases f =
  let ~releases, ~held:_, ~cpu:_ = Nx_release_support.watch f in
  greater int ~msg:"releases" ~than:0 releases

(* Runs of a call, each watched: a loaded host's interrupts and page faults
   lengthen a stretch in some runs, and the least of the runs' longest
   stretches is the call's own. *)
let runs = 3

(* [f ()] releases the domain lock, and holds it for no stretch longer
   than the copy allows, in the least of [runs] runs. *)
let holds_within f =
  (* An uncounted run first, so that the pool's threads, their kept scratch
     and the operands' pages exist; then the setup's garbage collected, so
     that no collection runs within the calls. *)
  f ();
  Gc.full_major ();
  let least =
    List.fold_left
      (fun least _ ->
        let ~releases, ~held, ~cpu:_ = Nx_release_support.watch f in
        greater int ~msg:"releases" ~than:0 releases;
        min least held)
      max_int (List.init runs Fun.id)
  in
  less int
    ~msg:"the least of the runs' longest stretches held, in ticks of CPU time"
    ~than:(Nx_release_support.bound ()) least

let done_ = function
  | A.Done -> ()
  | r -> failf "answered %a" Nx_array_support.pp_answer r

let f32 s = A.create Rig.host D.Float32 s
let f64 s = A.create Rig.host D.Float64 s
let c64 s = A.create Rig.host D.Complex64 s
let ones n = A.of_array D.Float32 [| n |] (Array.make n 1.)
let identity = P.v ~ins:[| D.Any D.Float32 |] [| P.In 0 |] ~outs:[| 0 |]

let reduce rs ~dsts x =
  let shape = Iarray.to_array (Nx_array.Layout.shape (A.layout x)) in
  done_
    (Nx_cpu.reduce identity ~shape ~axes:[| 0 |] rs ~dsts
       [| Nx_kernel.Plain x |])

let plain x : Nx_kernel.operand = { array = A.Any x; scale = None }

let contract ~dst a b =
  done_
    (Nx_cpu.contract ~batch:[||] ~contracting:[| (1, 0) |] ~segments:None
       ~acc:(D.Any D.Float32) ~dst ~init:None (plain a) (plain b))

(* x to the powers y then 1 / y, [k] times over: x^y, then (x^y)^(1/y),
   and so on, each power as costly as the first. *)
let powers k =
  P.v
    ~ins:[| D.Any D.Float64; D.Any D.Float64; D.Any D.Float64 |]
    (Iarray.init (k + 3) (fun i ->
         if i < 3 then P.In i
         else P.Op2 (Binary Pow, (if i = 3 then 0 else i - 1), 1 + (i mod 2))))
    ~outs:[| k + 2 |]

(* [x], of [m] elements, then y and 1 / y broadcast to them: the loads of
   [powers]. *)
let exponents m x =
  let along v =
    let one = A.of_array D.Float64 [| 1 |] [| v |] in
    let layout = Nx_array.Layout.v ~strides:[| 0 |] [| m |] in
    Nx_kernel.Plain (A.v D.Float64 layout (A.buffer one))
  in
  [| Nx_kernel.Plain x; along 1.5; along (1. /. 1.5) |]

let families =
  group "a long call lets another domain collect"
    [
      test "a reduction, held for no longer than the copy at a time"
        (fun () ->
          let n = 1 lsl 22 in
          let x = ones n and sum = f32 [||] in
          holds_within (fun () ->
              reduce [| (Monoid Sum, 0) |] ~dsts:[| A.Any sum |] x));
      test "many reductions of a program" (fun () ->
          let m = 4096 and k = 2048 in
          let x = ones m in
          let dsts = Array.init k (fun _ -> A.Any (f32 [||])) in
          let rs = Array.make k (Nx_kernel.Monoid Sum, 0) in
          releases (fun () -> reduce rs ~dsts x));
      test "a scan on one thread" (fun () ->
          let n = 1 lsl 21 in
          let x = ones n and dst = f32 [| n |] in
          releases (fun () ->
              Nx_cpu.one_thread (fun () ->
                  done_
                    (Nx_cpu.scan identity ~axis:0 (Monoid Sum, 0)
                       ~dsts:[| A.Any dst |]
                       [| Nx_kernel.Plain x |]))));
      test "a transform" (fun () ->
          let n = 1 lsl 19 in
          let x = c64 [| n |] and y = c64 [| n |] in
          releases (fun () ->
              done_ (Nx_cpu.c2c Forward ~axes:[| 0 |] ~dst:y x)));
      test "a real transform of no point, its bins written" (fun () ->
          let n = 1 lsl 18 in
          let x = f32 [| n; 0 |] and y = c64 [| n; 1 |] in
          releases (fun () -> done_ (Nx_cpu.r2c ~axes:[| 1 |] ~dst:y x)));
      test "a contraction" (fun () ->
          let n = 512 in
          let a = f32 [| n; n |] and b = f32 [| n; n |] in
          let dst = f32 [| n; n |] in
          releases (fun () -> contract ~dst a b));
      test "a gather" (fun () ->
          let m = 1 lsl 21 and k = 1 lsl 20 in
          let x = f32 [| k |] in
          let at =
            A.of_array D.Int64 [| m |]
              (Array.init m (fun i -> Int64.of_int (i * 7919 mod k)))
          in
          let dst = f32 [| m |] in
          releases (fun () -> done_ (Nx_cpu.gather ~axis:0 ~dst at x)));
      test "a scatter" (fun () ->
          let m = 1 lsl 21 and k = 1 lsl 20 in
          let into = f32 [| k |] and dst = f32 [| k |] in
          let at =
            A.of_array D.Int64 [| m |]
              (Array.init m (fun i -> Int64.of_int (i * 7919 mod k)))
          in
          let u = ones m in
          releases (fun () ->
              done_ (Nx_cpu.scatter Add ~unique:false ~axis:0 ~dst ~into at u)));
      test "a sort" (fun () ->
          let m = 1 lsl 20 in
          let x =
            A.of_array D.Float64 [| m |]
              (Array.init m (fun i -> float_of_int (i * 7919 mod m)))
          in
          let values = f64 [| m |] in
          let positions = A.create Rig.host D.Int64 [| m |] in
          releases (fun () ->
              done_
                (Nx_cpu.sort ~axis:0 ~descending:false ~values ~positions x)));
      test "an assembly of many pieces" (fun () ->
          let np = 6144 in
          let dst = f32 [| np |] in
          let piece i =
            let range = { Nx_array.Move.start = i; count = 1; step = 1 } in
            (([| range |] : _ iarray), f32 [| 1 |])
          in
          let pieces = Array.init np piece in
          releases (fun () -> done_ (Nx_cpu.assemble ~fill:0. ~dst pieces)));
      test "a fold of windows" (fun () ->
          let n = 1 lsl 16 in
          let window =
            { Nx_array.Move.axis = 0; size = 64; step = 1; dilation = 1 }
          in
          let pad =
            {
              Nx_kernel.lo = [| 0 |];
              hi = [| 0 |];
              interior = [| 0 |];
              windows = [| window |];
            }
          in
          let shape = Result.get_ok (Nx_kernel.Shape.padded pad [| n |]) in
          let x = f32 (Iarray.of_array shape) and dst = f32 [| n |] in
          releases (fun () -> done_ (Nx_cpu.fold pad ~dst x)));
      test "a routine" (fun () ->
          let k = 512 in
          let a =
            A.of_array D.Float64 [| k; k |]
              (Array.init (k * k) (fun i -> float_of_int (i * 7919 mod 1009)))
          in
          let lu = f64 [| k; k |] in
          let pivots = A.create Rig.host D.Int64 [| k |] in
          let perm = A.create Rig.host D.Int64 [| k |] in
          releases (fun () -> done_ (Nx_cpu.lu ~lu ~pivots ~perm a)));
      test "a map" (fun () ->
          let n = 1 lsl 22 in
          let x = ones n and dst = f32 [| n |] in
          releases (fun () ->
              done_ (Nx_kernels_support.map2 cpu (Binary Pow) ~dst x x)));
      test "a map of a long program" (fun () ->
          let m = 64 in
          let x = A.of_array D.Float64 [| m |] (Array.make m 1.5) in
          let dst = f64 [| m |] in
          let prog = powers 20000 and loads = exponents m x in
          releases (fun () ->
              done_ (Nx_cpu.map prog ~dsts:[| A.Any dst |] loads)));
    ]

(* The estimate's inputs: a batch of matrices whose work is each under the
   copy and past it summed, a padded load, whose copy is the work, a map of
   no operand, and a copy of rows too short to stride along whose
   sub-byte elements the walk cannot reorder, each at the least size past
   the copy. *)
let estimates =
  group "a call past the copy by each input of the estimate"
    [
      test "a batch of many 1 × 1 LUs" (fun () ->
          let b = 1 lsl 20 in
          let one = A.of_array D.Float32 [| 1 |] [| 3. |] in
          let a =
            A.v D.Float32
              (Nx_array.Layout.v ~strides:[| 0; 0; 0 |] [| b; 1; 1 |])
              (A.buffer one)
          in
          let lu = f32 [| b; 1; 1 |] in
          let pivots = A.create Rig.host D.Int64 [| b; 1 |] in
          let perm = A.create Rig.host D.Int64 [| b; 1 |] in
          releases (fun () -> done_ (Nx_cpu.lu ~lu ~pivots ~perm a)));
      test "a map of a padded load" (fun () ->
          let n = 1 lsl 18 in
          let x = A.of_array D.Float64 [| n |] (Array.make n 1.5) in
          let dst = f64 [| n + 2 |] in
          let pad =
            {
              Nx_kernel.lo = [| 1 |];
              hi = [| 1 |];
              interior = [| 0 |];
              windows = [||];
            }
          in
          let prog =
            P.v ~ins:[| D.Any D.Float64 |] [| P.In 0 |] ~outs:[| 0 |]
          in
          releases (fun () ->
              done_
                (Nx_cpu.map prog ~dsts:[| A.Any dst |]
                   [| Nx_kernel.Padded { x; fill = 1.5; pad } |])));
      test "a copy of short rows of sub-byte elements" (fun () ->
          let n = 21000 in
          let base = A.create Rig.host D.Int4 [| n; 4 |] in
          let layout = Nx_array.Layout.v ~strides:[| 4; 1 |] [| n; 3 |] in
          let x = A.v D.Int4 layout (A.buffer base) in
          let dst = A.create Rig.host D.Int4 [| n; 3 |] in
          let prog = P.v ~ins:[| D.Any D.Int4 |] [| P.In 0 |] ~outs:[| 0 |] in
          releases (fun () ->
              done_
                (Nx_cpu.map prog ~dsts:[| A.Any dst |] [| Nx_kernel.Plain x |])));
      test "an iota of a strided destination" (fun () ->
          let n = 1 lsl 18 in
          let base = A.create Rig.host D.Int64 [| n; 2 |] in
          let layout = Nx_array.Layout.v ~strides:[| 2; 1 |] [| n; 1 |] in
          let dst = A.v D.Int64 layout (A.buffer base) in
          releases (fun () ->
              done_ (Nx_kernels_support.map0 cpu (Iota 0) ~dst)));
    ]

(* Domains *)

(* The bits of a float32 or float64 result. *)
let bits (A.Any y) =
  match A.dtype y with
  | D.Float32 -> Array.map Int64.bits_of_float (A.to_array y)
  | D.Float64 -> Array.map Int64.bits_of_float (A.to_array y)
  | _ -> invalid_arg "bits"

(* A reduction, a map and a contraction of arrays drawn from [seed]: the
   bits of their results. *)
let work seed () =
  let r = Random.State.make [| seed |] in
  let draw n = Array.init n (fun _ -> Random.State.float r 2. -. 1.) in
  let n = 1 lsl 19 and k = 128 in
  let x = A.of_array D.Float32 [| n |] (draw n) in
  let a = A.of_array D.Float32 [| k; k |] (draw (k * k)) in
  let b = A.of_array D.Float32 [| k; k |] (draw (k * k)) in
  let sum = f32 [||] and exp = f32 [| n |] and ab = f32 [| k; k |] in
  reduce [| (Monoid Sum, 0) |] ~dsts:[| A.Any sum |] x;
  done_ (Nx_kernels_support.map1 cpu (Unary Exp) ~dst:exp x);
  contract ~dst:ab a b;
  List.map bits [ A.Any sum; A.Any exp; A.Any ab ]

(* Two domains each run [work] of a seed of their own at once, from a common
   start: every result has the bits one thread gives. *)
let law_concurrent (s1, s2) =
  let want = List.map (fun s -> Nx_cpu.one_thread (work s)) [ s1; s2 ] in
  let ready = Atomic.make 0 in
  let run s =
    Domain.spawn (fun () ->
        Atomic.incr ready;
        while Atomic.get ready < 2 do
          Domain.cpu_relax ()
        done;
        work s ())
  in
  let ds = List.map run [ s1; s2 ] in
  List.iter2
    (fun want d -> equal (list (array int64)) want (Domain.join d))
    want ds

let domains =
  group "domains"
    [
      prop ~count:20 "several domains may call concurrently"
        Gen.(pair (int_range 0 1_000_000) (int_range 0 1_000_000))
        law_concurrent;
      test "a call waits for another domain's job released" (fun () ->
          (* Another domain occupies the pool with a job that lasts until
             this thread releases the lock. The call adds 16384 float64
             elements, whose job takes two threads and so waits for the
             other's to end: a call that waited holding the lock would
             never release it, and the other job would end at its
             deadline instead. *)
          let k = 16384 in
          let a = A.of_array D.Float64 [| k |] (Array.make k 1.) in
          let sum = f64 [| k |] in
          let small () =
            done_ (Nx_kernels_support.map2 cpu (Binary Add) ~dst:sum a a)
          in
          small ();
          let other = Domain.spawn Nx_release_support.occupy in
          while not (Nx_release_support.occupied ()) do
            Domain.cpu_relax ()
          done;
          let ~releases, ~held:_, ~cpu:_ = Nx_release_support.watch small in
          equal ~msg:"the other job ended on the call's release" bool true
            (Domain.join other);
          greater int ~msg:"releases" ~than:0 releases);
    ]

let () =
  exit (Windtrap.run "nx_cpu.release" [ families; estimates; domains ])
