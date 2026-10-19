(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's promise: a call whose work takes longer than a copy of about
   1 MiB on one core runs it with the domain lock released, and holds the
   lock for no longer than about that copy, so that it keeps no other
   domain waiting to collect. The runtime's hooks on a blocking section
   show when the calling thread releases the lock and reacquires it, and
   the thread's CPU time how long it held it, which a loaded host's
   scheduler does not lengthen. Each family's calls are ones whose work
   runs inline, on the calling thread, at any count of cores, or as jobs,
   whose threads hold the lock wherever their call does. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog

(* The probe *)

(* What the bound allows a held stretch: NX_CPU_HOLD_BYTES (cpu.h), 1 MiB,
   of one core's memcpy, in nanoseconds. *)
let hold = lazy (Nx_release_support.copy ())

(* Runs of a call, each watched: a loaded host's interrupts and page faults
   lengthen a stretch in some runs, and the least of the runs' longest
   stretches is the call's own, as [hold] takes the least of its copies. *)
let runs = 3

(* [f ()]'s work takes at most what the threshold allows, or it releases
   the domain lock and holds it for no stretch longer than that, on a
   loaded host. *)
let releases f =
  (* An uncounted run first, so that the pool's threads, their kept scratch
     and the operands' pages exist. *)
  f ();
  (* The setup's garbage collected first, so that no collection runs within
     the calls. *)
  Gc.full_major ();
  let hold = Lazy.force hold in
  let watched = List.init runs (fun _ -> Nx_release_support.watch f) in
  let long = ref true and least = ref max_int in
  List.iter
    (fun (~releases, ~held, ~cpu) ->
      long := !long && cpu > hold;
      least := min !least held;
      if cpu > hold then
        greater int
          ~msg:(Printf.sprintf "releases within %d ns of CPU time" cpu)
          ~than:0 releases)
    watched;
  if !long then
    less int
      ~msg:"the least of the runs' longest stretches held, in ns of CPU time"
      ~than:(Nx_release_support.bound ()) !least

let done_ = function
  | A.Done -> ()
  | r -> failf "answered %a" Nx_array_support.pp_answer r

let f32 s = A.create Rig.host D.Float32 s

(* Reductions *)

let identity = P.v ~ins:[| D.Any D.Float32 |] [| P.In 0 |] ~outs:[| 0 |]

let reduce r axes ~dst x =
  let shape = Nx_array.Layout.shape (A.layout x) in
  done_
    (Nx_cpu.reduce identity ~shape ~axes
       [| (r, 0) |]
       ~dsts:[| A.Any dst |]
       [| Nx_kernel.Plain x |])

let scan ~dsts r x =
  Nx_cpu.one_thread (fun () ->
      done_
        (Nx_cpu.scan identity ~axis:0 (r, 0) ~dsts [| Nx_kernel.Plain x |]))

let n = 1 lsl 25
let ones = lazy (A.of_array D.Float32 [| n |] (Array.make n 1.))

let reductions =
  group "reductions"
    [
      test "two reductions of a program on one thread" (fun () ->
          let x = Lazy.force ones and lse = f32 [||] and sum = f32 [||] in
          let exp =
            P.v ~ins:[| D.Any D.Float32 |]
              [| P.In 0; P.Op1 (Unary Exp, D.Any D.Float32, 0) |]
              ~outs:[| 1 |]
          in
          releases (fun () ->
              Nx_cpu.one_thread (fun () ->
                  done_
                    (Nx_cpu.reduce exp ~shape:[| n |] ~axes:[| 0 |]
                       [| (Monoid Logsumexp, 0); (Monoid Sum, 0) |]
                       ~dsts:[| A.Any lse; A.Any sum |]
                       [| Nx_kernel.Plain x |]))));
      test "a logsumexp on one thread" (fun () ->
          let x = Lazy.force ones and dst = f32 [||] in
          releases (fun () ->
              Nx_cpu.one_thread (fun () ->
                  reduce (Monoid Logsumexp) [| 0 |] ~dst x)));
      test "a scan on one thread" (fun () ->
          let x = Lazy.force ones and dst = f32 [| n |] in
          releases (fun () -> scan (Monoid Sum) ~dsts:[| A.Any dst |] x));
      test "a moments scan on one thread" (fun () ->
          let m = n / 2 in
          let x = A.of_array D.Float32 [| m |] (Array.make m 1.) in
          let mean = f32 [| m |] and var = f32 [| m |] in
          releases (fun () ->
              scan Moments ~dsts:[| A.Any mean; A.Any var |] x));
    ]

(* Transforms *)

let c64 s = A.create Rig.host D.Complex64 s

let transforms =
  let n = 1 lsl 22 in
  group "transforms"
    [
      test "a transform on one thread" (fun () ->
          let x = c64 [| n |] and y = c64 [| n |] in
          releases (fun () ->
              Nx_cpu.one_thread (fun () ->
                  done_ (Nx_cpu.c2c Forward ~axes:[| 0 |] ~dst:y x))));
      test "a real transform's inverse on one thread" (fun () ->
          let n = 2 * n in
          let x = c64 [| (n / 2) + 1 |] and y = f32 [| n |] in
          releases (fun () ->
              Nx_cpu.one_thread (fun () ->
                  done_ (Nx_cpu.c2r ~axes:[| 0 |] ~dst:y x))));
      test "a real transform of no point" (fun () ->
          let n = 2 * n in
          let x = f32 [| n; 0 |] and y = c64 [| n; 1 |] in
          releases (fun () -> done_ (Nx_cpu.r2c ~axes:[| 1 |] ~dst:y x)));
    ]

(* Contractions *)

let contract ~acc ~dst a b =
  done_
    (Nx_cpu.contract ~order:Chain ~batch:[||] ~contracting:[| (1, 0) |] ~acc
       ~dst ~init:None a b)

let contractions =
  group "contractions"
    [
      test "a batch of many 1 × 1 products" (fun () ->
          let b = 1 lsl 22 in
          let a = f32 [| b; 1; 1 |] and x = f32 [| b; 1; 1 |] in
          let dst = f32 [| b; 1; 1 |] in
          releases (fun () ->
              done_
                (Nx_cpu.contract ~order:Chain ~batch:[| (0, 0) |]
                   ~contracting:[| (2, 1) |] ~acc:(D.Any D.Float32) ~dst
                   ~init:None a x)));
      test "a complex one whose operand is converted" (fun () ->
          let m = 2048 and k = 4096 in
          let a = c64 [| m; k |] and b = c64 [| k; 1 |] in
          let dst = A.create Rig.host D.Complex128 [| m; 1 |] in
          releases (fun () -> contract ~acc:(D.Any D.Complex128) ~dst a b));
      test "an 8-bit one whose rows are copied into words" (fun () ->
          let m = 32768 and k = 4097 in
          let a = A.create Rig.host D.Int8 [| m; k |] in
          let b = A.create Rig.host D.Int8 [| k; 1 |] in
          let dst = A.create Rig.host D.Int32 [| m; 1 |] in
          releases (fun () -> contract ~acc:(D.Any D.Int32) ~dst a b));
    ]

(* Gathers, scatters, sorts and iotas *)

let indexing =
  let n = 1 lsl 24 in
  group "gathers, scatters, sorts and iotas"
    [
      test "a gather at scattered positions" (fun () ->
          (* Each of 96 rows gathers the same 2^20 positions of the same
             2^26 bytes, both broadcast along the rows. *)
          let rows = 96 and m = 1 lsl 20 and k = 1 lsl 26 in
          let row v =
            A.v D.Uint8
              (Nx_array.Layout.v ~strides:[| 0; 1 |] [| rows; k |])
              (A.buffer v)
          in
          let x = row (A.create Rig.host D.Uint8 [| k |]) in
          let b = Bigarray.(Array1.create int64 c_layout m) in
          for i = 0 to m - 1 do
            b.{i} <- Int64.of_int (i * 7919 mod k)
          done;
          let at = A.of_bigarray D.Int64 (Bigarray.genarray_of_array1 b) in
          let at =
            A.v D.Int64
              (Nx_array.Layout.v ~strides:[| 0; 1 |] [| rows; m |])
              (A.buffer at)
          in
          let dst = A.create Rig.host D.Uint8 [| rows; m |] in
          releases (fun () -> done_ (Nx_cpu.gather ~axis:1 ~dst at x)));
      test "a narrow float sum's scatter" (fun () ->
          let n = 2 * n in
          let dst = A.create Rig.host D.Float16 [| n |] in
          let idx = A.of_array D.Int64 [| 1 |] [| 7L |] in
          let u = A.create Rig.host D.Float16 [| 1 |] in
          releases (fun () ->
              done_
                (Nx_cpu.scatter Add ~unique:false ~axis:0 ~dst ~into:dst idx
                   u)));
      test "a sort of 2^22 float64 elements" (fun () ->
          let m = 1 lsl 22 in
          let x =
            A.of_array D.Float64 [| m |]
              (Array.init m (fun i -> float_of_int (i * 7919 mod m)))
          in
          let values = A.create Rig.host D.Float64 [| m |] in
          let positions = A.create Rig.host D.Int64 [| m |] in
          releases (fun () ->
              done_
                (Nx_cpu.sort ~axis:0 ~descending:false ~values ~positions x)));
      test "an iota of a strided destination" (fun () ->
          let base = A.create Rig.host D.Uint8 [| n; 2 |] in
          let layout = Nx_array.Layout.v ~strides:[| 2; 1 |] [| n; 1 |] in
          let dst = A.v D.Uint8 layout (A.buffer base) in
          releases (fun () -> done_ (Nx_cpu.apply0 (P.Iota 0) ~dst)));
    ]

(* Assemblies and folds *)

let assemblies =
  group "assemblies and folds"
    [
      test "an assembly of many pieces" (fun () ->
          let np = 6144 in
          let dst = f32 [| np |] in
          let piece i =
            let range = { Nx_array.Move.start = i; count = 1; step = 1 } in
            (([| range |] : _ iarray), f32 [| 1 |])
          in
          let pieces = Array.init np piece in
          releases (fun () -> done_ (Nx_cpu.assemble ~fill:0. ~dst pieces)));
      test "an assembly of many strided pieces" (fun () ->
          (* Pieces of two elements two apart: no pair of them is
             compared, so the call's work grows as the pieces do. *)
          let np = 32768 in
          let dst = f32 [| 4 * np |] in
          let piece i =
            let range = { Nx_array.Move.start = 4 * i; count = 2; step = 2 } in
            (([| range |] : _ iarray), f32 [| 2 |])
          in
          let pieces = Array.init np piece in
          releases (fun () -> done_ (Nx_cpu.assemble ~fill:0. ~dst pieces)));
      test "a fold of windows" (fun () ->
          let n = 1 lsl 16 in
          let window =
            { Nx_array.Move.axis = 0; size = 64; step = 1; dilation = 1 }
          in
          let pad =
            { Nx_kernel.lo = [| 0 |]; hi = [| 0 |]; interior = [| 0 |];
              windows = [| window |] }
          in
          let shape = Result.get_ok (Nx_kernel.Shape.padded pad [| n |]) in
          let x = f32 shape and dst = f32 [| n |] in
          releases (fun () -> done_ (Nx_cpu.fold pad ~dst x)));
    ]

(* Matrices *)

let f64 s = A.create Rig.host D.Float64 s

let matrices =
  group "matrices"
    [
      test "a batch of many 1 × 1 LUs" (fun () ->
          let b = 1 lsl 22 in
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
      test "an LU of 2560 × 2560" (fun () ->
          let k = 2560 in
          let a =
            A.of_array D.Float64 [| k; k |]
              (Array.init (k * k) (fun i -> float_of_int (i * 7919 mod 1009)))
          in
          let lu = f64 [| k; k |] in
          let pivots = A.create Rig.host D.Int64 [| k |] in
          let perm = A.create Rig.host D.Int64 [| k |] in
          releases (fun () -> done_ (Nx_cpu.lu ~lu ~pivots ~perm a)));
      test "a batch of 56 × 56 eigenvalues" (fun () ->
          let b = 1 lsl 12 and k = 56 in
          let a =
            A.of_array D.Float64 [| b; k; k |]
              (Array.init (b * k * k) (fun i ->
                   float_of_int (i * 7919 mod 1009)))
          in
          let w = f64 [| b; k |] in
          releases (fun () -> done_ (Nx_cpu.eigh ~w a)));
    ]

(* Elementwise calls and maps *)

(* x to the powers y then 1 / y, [k] times over: x^y, then (x^y)^(1/y),
   and so on, each power as costly as the first. *)
let powers k =
  P.v
    ~ins:[| D.Any D.Float64; D.Any D.Float64; D.Any D.Float64 |]
    (Iarray.init (k + 3) (fun i ->
         if i < 3 then P.In i
         else P.Op2 (Binary Pow, (if i = 3 then 0 else i - 1), 1 + (i mod 2))))
    ~outs:[| k + 2 |]

let halves n = A.of_array D.Float64 [| n |] (Array.make n 1.5)

(* [x], of [m] elements once loaded, then y and 1 / y broadcast to them:
   the loads of [powers]. *)
let exponents m x =
  let along v =
    let one = A.of_array D.Float64 [| 1 |] [| v |] in
    let layout = Nx_array.Layout.v ~strides:[| 0 |] [| m |] in
    Nx_kernel.Plain (A.v D.Float64 layout (A.buffer one))
  in
  [| x; along 1.5; along (1. /. 1.5) |]

let elementwise =
  let n = 1 lsl 22 in
  group "elementwise calls and maps"
    [
      test "a power of float8 elements" (fun () ->
          let m = 1 lsl 26 in
          let one = A.of_array D.Float8_e4m3fn [| 1 |] [| 1.5 |] in
          let x =
            A.v D.Float8_e4m3fn
              (Nx_array.Layout.v ~strides:[| 0 |] [| m |])
              (A.buffer one)
          in
          let dst = A.create Rig.host D.Float8_e4m3fn [| m |] in
          releases (fun () -> done_ (Nx_cpu.apply2 (Binary Pow) ~dst x x)));
      test "a map of powers" (fun () ->
          let x = halves n and dst = f64 [| n |] in
          let prog = powers 16 and loads = exponents n (Plain x) in
          releases (fun () ->
              done_ (Nx_cpu.map prog ~dsts:[| A.Any dst |] loads)));
      test "a map of powers of a padded load" (fun () ->
          let x = halves n and dst = f64 [| n + 2 |] in
          let pad =
            { Nx_kernel.lo = [| 1 |]; hi = [| 1 |]; interior = [| 0 |];
              windows = [||] }
          in
          let padded = Nx_kernel.Padded { x; fill = 1.5; pad } in
          let prog = powers 16 and loads = exponents (n + 2) padded in
          releases (fun () ->
              done_ (Nx_cpu.map prog ~dsts:[| A.Any dst |] loads)));
    ]

(* Domains *)

let domains =
  group "domains"
    [
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
          let small () = done_ (Nx_cpu.apply2 (Binary Add) ~dst:sum a a) in
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
  exit
    (Windtrap.run "nx_cpu.release"
       [
         group "a long call lets another domain collect"
           [
             reductions;
             transforms;
             contractions;
             indexing;
             assemblies;
             matrices;
             elementwise;
             domains;
           ];
       ])
