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
  let shape = Nx_array.Layout.shape (A.layout x) in
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
  Iarray.of_list [ Nx_kernel.Plain x; along 1.5; along (1. /. 1.5) ]

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
          let dsts = Iarray.init k (fun _ -> A.Any (f32 [||])) in
          let rs = Iarray.init k (fun _ -> (Nx_kernel.Monoid Sum, 0)) in
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
          let pieces = Iarray.init np piece in
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

(* The bits of a float32 result, as unboxed integers: as an int64 array,
   half a million elements were as many words allocated, and the minor
   collections they forced each stopped both domains. *)
let bits (A.Any y) =
  match A.dtype y with
  | D.Float32 ->
      Array.map (fun x -> Int32.to_int (Int32.bits_of_float x)) (A.to_array y)
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
    (fun want d -> equal (list (array int)) want (Domain.join d))
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

(* Operands over memory that something other than the host holds: the host's
   borrow of an io device's pages, as a load maps a file, and a bigarray. *)
let mapped (type v s) (a : (v, s) A.t) : (v, s) A.t =
  let io = Nx_array_support.mapped_read_only a in
  let pages = Option.get (Rig.Buffer.borrow Rig.host (A.buffer io)) in
  A.v (A.dtype a) (A.layout a) pages

let bigarray (type v s) (a : (v, s) A.t) : (v, s) A.t =
  let n = Rig.Buffer.length (A.buffer a) in
  let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  let b = Rig.Buffer.of_bigarray ba in
  Rig.Buffer.copy ~src:(A.buffer a) ~dst:b;
  A.v (A.dtype a) (A.layout a) b

type over = { over : 'v 's. ('v, 's) A.t -> ('v, 's) A.t }

(* A gather of eight elements, a map of 64 and a sum of 64, each held far
   under the copy over the host's own memory, released over borrowed
   memory, whatever the operand it reads there: their count of releases. *)
let small_calls { over } =
  let x = A.of_array D.Float32 [| 64 |] (Array.init 64 float_of_int) in
  let at =
    A.of_array D.Int64 [| 8 |] (Array.init 8 (fun i -> Int64.of_int (i * 7)))
  in
  let calls =
    [
      ( "a gather",
        fun () ->
          done_ (Nx_cpu.gather ~axis:0 ~dst:(f32 [| 8 |]) (over at) (over x)) );
      ( "a map",
        fun () ->
          done_
            (Nx_kernels_support.map2 cpu (Binary Add) ~dst:(f32 [| 64 |])
               (over x) x) );
      ( "a sum",
        fun () ->
          reduce [| (Monoid Sum, 0) |] ~dsts:[| A.Any (f32 [||]) |] (over x) );
    ]
  in
  List.map
    (fun (what, f) ->
      f ();
      let ~releases, ~held:_, ~cpu:_ = Nx_release_support.watch f in
      (what, releases))
    calls

let borrowed =
  group "a call that reads borrowed memory releases, however small"
    [
      cases "over borrowed memory"
        ~id:(fun (what, _) -> what)
        [
          ("a host borrow of an io device's pages", { over = mapped });
          ("a bigarray", { over = bigarray });
        ]
        (fun (_, over) ->
          List.iter
            (fun (what, releases) ->
              greater ~msg:(what ^ ": releases") int ~than:0 releases)
            (small_calls over));
      test
        "nx.array's host copies release over borrowed memory and past \
         256 KiB, and hold below"
        (fun () ->
          let releases f =
            f ();
            let ~releases, ~held:_, ~cpu:_ = Nx_release_support.watch f in
            releases
          in
          let small = A.of_array D.Float32 [| 64 |] (Array.make 64 1.) in
          let ints = A.of_array D.Int64 [| 64 |] (Array.make 64 1L) in
          let int8s = A.of_array D.Int8 [| 64 |] (Array.make 64 1) in
          let n = 1 lsl 16 in
          let large = f32 [| n |] and few = Array.make 64 1. in
          (* Borrows made outside the calls, whose mapping may release. *)
          let mapped_small = mapped small and bigarray_small = bigarray small in
          let mapped_ints = mapped ints and mapped_int8s = mapped int8s in
          equal
            (list (pair string bool))
            [
              ("a copy of a borrow", true);
              ("a copy of a bigarray", true);
              ("int64s out of a borrow", true);
              ("floats out of a borrow", true);
              ("int8s out of a borrow", true);
              ("a copy of 256 KiB", true);
              ("a copy of 256 B", false);
              ("floats out of 256 B", false);
              ("floats into 256 B", false);
            ]
            (List.map
               (fun (what, f) -> (what, releases f > 0))
               [
                 ("a copy of a borrow", fun () -> ignore (A.copy mapped_small));
                 ( "a copy of a bigarray",
                   fun () -> ignore (A.copy bigarray_small) );
                 ( "int64s out of a borrow",
                   fun () -> ignore (A.to_array mapped_ints) );
                 ( "floats out of a borrow",
                   fun () -> ignore (A.to_array mapped_small) );
                 ( "int8s out of a borrow",
                   fun () -> ignore (A.to_array mapped_int8s) );
                 ("a copy of 256 KiB", fun () -> ignore (A.copy large));
                 ("a copy of 256 B", fun () -> ignore (A.copy small));
                 ("floats out of 256 B", fun () -> ignore (A.to_array small));
                 ( "floats into 256 B",
                   fun () -> ignore (A.of_array D.Float32 [| 64 |] few) );
               ]));
      test
        "to_array of a borrow gives its elements while a compaction runs in \
         its release"
        (fun () ->
          (* A compaction moves young blocks: the OCaml arrays of a small
             borrow's elements are young, and a borrow's call releases once,
             so the compaction lands in its walk. *)
          let floats k = Array.init k (fun i -> float_of_int (i - 7)) in
          let ints k = Array.init k (fun i -> (i mod 255) - 127) in
          let once f =
            let y = ref [||] in
            let ~releases, ~held:_, ~cpu:_ =
              Nx_release_support.watch (fun () -> y := f ())
            in
            (!y, releases)
          in
          let x = mapped (A.of_array D.Float32 [| 64 |] (floats 64)) in
          equal ~msg:"floats out of a borrow"
            (pair (pair (array float_exact) int) bool)
            ((floats 64, 1), true)
            (Nx_release_support.compacting (fun () ->
                 once (fun () -> A.to_array x)));
          let x = mapped (A.of_array D.Int8 [| 64 |] (ints 64)) in
          equal ~msg:"int8s out of a borrow"
            (pair (pair (array int) int) bool)
            ((ints 64, 1), true)
            (Nx_release_support.compacting (fun () ->
                 once (fun () -> A.to_array x))));
      test
        "to_array of strided, reversed and sub-byte borrows gives their \
         elements in order, released"
        (fun () ->
          let released f =
            let y = ref [||] in
            let ~releases, ~held:_, ~cpu:_ =
              Nx_release_support.watch (fun () -> y := f ())
            in
            equal ~msg:"releases" int 1 releases;
            !y
          in
          let reversed (type v s) (x : (v, s) A.t) n =
            Option.get
              (A.move (Slice [| { start = n - 1; count = n; step = -1 } |]) x)
          in
          (* A transpose of 9 x 13 float32s, of shape [13; 9]. *)
          let r = 13 and c = 9 in
          let base =
            A.of_array D.Float32 [| c; r |]
              (Array.init (r * c) (fun k -> float_of_int k))
          in
          let t = mapped (Option.get (A.move (Permute [| 1; 0 |]) base)) in
          equal ~msg:"a transpose" (array float_exact)
            (Array.init (r * c) (fun k ->
                 float_of_int ((k mod c * r) + (k / c))))
            (released (fun () -> A.to_array t));
          (* A permutation of 4 x 5 x 6 float32s, of rank 3. *)
          let base =
            A.of_array D.Float32 [| 4; 5; 6 |] (Array.init 120 float_of_int)
          in
          let p = mapped (Option.get (A.move (Permute [| 2; 0; 1 |]) base)) in
          equal ~msg:"a permutation of rank 3" (array float_exact)
            (Array.init 120 (fun m ->
                 let i = m / 20 and j = m / 5 mod 4 and k = m mod 5 in
                 float_of_int ((((j * 5) + k) * 6) + i)))
            (released (fun () -> A.to_array p));
          (* 301 int16s reversed: immediates, a negative step. *)
          let n = 301 in
          let ints = Array.init n (fun k -> (k * 211 mod 65536) - 32768) in
          let rev = mapped (reversed (A.of_array D.Int16 [| n |] ints) n) in
          equal ~msg:"int16s reversed" (array int)
            (Array.init n (fun k -> ints.(n - 1 - k)))
            (released (fun () -> A.to_array rev));
          (* 301 float16s, contiguous: the run of whole float16s. *)
          let halves = Array.init n (fun k -> float_of_int (k - 150)) in
          let x = mapped (A.of_array D.Float16 [| n |] halves) in
          equal ~msg:"float16s" (array float_exact) halves
            (released (fun () -> A.to_array x));
          (* 301 int4s reversed: elements share bytes. *)
          let nibbles = Array.init n (fun k -> (k mod 16) - 8) in
          let rev = mapped (reversed (A.of_array D.Int4 [| n |] nibbles) n) in
          equal ~msg:"int4s reversed" (array int)
            (Array.init n (fun k -> nibbles.(n - 1 - k)))
            (released (fun () -> A.to_array rev)));
      test "the same calls over the host's own memory hold" (fun () ->
          equal
            (list (pair string int))
            [ ("a gather", 0); ("a map", 0); ("a sum", 0) ]
            (small_calls { over = Fun.id }));
    ]

let () =
  exit
    (Windtrap.run "nx_cpu.release" [ families; estimates; borrowed; domains ])
