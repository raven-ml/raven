(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's copies, casts and maps, as nx_cpu.mli states them: each kind of
   one or two operands has nx_kinds.h's bits, its maps, and its refusals. *)

open Windtrap
open Nx_array_gen
open Kernels_cases
module A = Nx_array
module D = Nx_array.Dtype
module B = Rig.Buffer
module Support = Nx_kernels_support

(* The kernels a law calls. *)
type kernels = (module Nx_kernel.S)

let copy (k : kernels) ~dst a = Support.map1 k Nx_kernel.Prog.Copy ~dst a

let cast (k : kernels) ~dst a = Support.map1 k Nx_kernel.Prog.Cast ~dst a

(* nx.cpu's refusals *)

(* Each answers the refusal nx_cpu.mli states, before any write. *)
let refuses r dst call =
  let before = bits_of dst in
  equal answer r (call ());
  equal (array int) before (bits_of dst)

let test_refusals k () =
  let module K = (val k : Nx_kernel.S) in
  let f32 = D.Float32 in
  (* A program's copy of a float32, over a load or into a destination of
     another dtype. *)
  let copy_f32 ~dst x =
    let f = D.Any f32 in
    K.map
      (Nx_kernel.Prog.of_node ~ins:[| f |] (Op1 (Copy, f, 0)))
      ~dsts:[| A.Any dst |] [| Nx_kernel.Plain x |]
  in
  let x = A.of_array f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let i32 = A.of_array D.Int32 [| 2; 3 |] (Array.make 6 7l) in
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) x) in
  let d = A.of_array f32 [| 2; 3 |] (Array.make 6 9.) in
  refuses A.Wrong_dtype d (fun () -> copy_f32 ~dst:d i32);
  refuses A.Shape_mismatch d (fun () -> copy k ~dst:d t);
  refuses A.Shape_mismatch d (fun () -> cast k ~dst:d t);
  let d = A.of_array D.Int32 [| 2; 3 |] (Array.make 6 9l) in
  refuses A.Wrong_dtype d (fun () -> copy_f32 ~dst:d x);
  (* Into bytes the source reads, and into a broadcast. *)
  let flat = A.of_array f32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let part start =
    Option.get (A.move (M.Slice [| { M.start; count = 3; step = 1 } |]) flat)
  in
  refuses A.Overlapping (part 0) (fun () -> copy k ~dst:(part 0) (part 1));
  refuses A.Overlapping (part 0) (fun () -> cast k ~dst:(part 0) (part 2));
  let b =
    Option.get
      (A.move
         (M.Broadcast [| 2; 3 |])
         (A.of_array f32 [| 3 |] [| 0.; 0.; 0. |]))
  in
  refuses A.Repeated_elements b (fun () -> cast k ~dst:b i32)

(* Kinds of no, two and three operands *)

(* Each of large_views against a contiguous operand: the 32 drawn cases of
   a law over large pairs miss one view in some runs. *)
let large_view_pairs =
  List.map
    (fun (Case x) -> Pair (x, seeded (A.dtype x) (L.shape (A.layout x)) 2))
    large_views

(* A contiguous bfloat16 pair: a kind that stages every operand, on several
   threads. *)
let large_bfloat16_pair =
  let (Case x) = large_at D.Bfloat16 257 65 0 1 in
  Pair (x, seeded (A.dtype x) (L.shape (A.layout x)) 2)

(* Values the kinds' documentation states, through nx.cpu. *)
let test_kind_values () =
  let k = (module Nx_cpu : Nx_kernel.S) in
  let map0 op ~dst = Support.map0 k op ~dst in
  let map1 op ~dst x = Support.map1 k op ~dst x in
  let map2 op ~dst x y = Support.map2 k op ~dst x y in
  let map3 op ~dst c x y = Support.map3 k op ~dst c x y in
  let done_ ~msg a = equal ~msg answer A.Done a in
  let f32 xs = A.of_array D.Float32 [| Array.length xs |] xs in
  let d = A.create Rig.host D.Float32 [| 2 |] in
  done_ ~msg:"add" (map2 (Binary Add) ~dst:d (f32 [| 1.; 2. |]) (f32 [| 3.; 0.5 |]));
  equal ~msg:"add" (array float_exact) [| 4.; 2.5 |] (A.to_array d);
  let i8 xs = A.of_array D.Int8 [| Array.length xs |] xs in
  let d = A.create Rig.host D.Int8 [| 3 |] in
  done_ ~msg:"idiv"
    (map2 (Binary Idiv) ~dst:d (i8 [| -128; 7; 5 |]) (i8 [| -1; 0; -2 |]));
  equal ~msg:"idiv: the least value by -1, by zero, toward zero" (array int)
    [| -128; 0; -2 |] (A.to_array d);
  let u8 xs = A.of_array D.Uint8 [| Array.length xs |] xs in
  let b = A.create Rig.host D.Bool [| 2 |] in
  done_ ~msg:"less" (map2 (Compare Less) ~dst:b (u8 [| 200; 1 |]) (u8 [| 100; 2 |]));
  equal ~msg:"less: unsigned order" (array bool) [| false; true |] (A.to_array b);
  let b2 = A.create Rig.host D.Bool [| 1 |] in
  done_ ~msg:"less"
    (map2 (Compare Less) ~dst:b2 (f32 [| -0. |]) (f32 [| 0. |]));
  equal ~msg:"less: -0 equals +0" (array bool) [| false |] (A.to_array b2);
  let m = A.create Rig.host D.Float32 [| 1 |] in
  done_ ~msg:"maximum"
    (map2 (Binary Maximum) ~dst:m (f32 [| -0. |]) (f32 [| 0. |]));
  equal ~msg:"maximum: -0 orders below +0" bool false
    (Float.sign_bit (A.get m [| 0 |]));
  let d = A.create Rig.host D.Float32 [| 2 |] in
  done_ ~msg:"where" (map3 Where ~dst:d b (f32 [| 1.; 2. |]) (f32 [| 3.; 4. |]));
  equal ~msg:"where" (array float_exact) [| 3.; 2. |] (A.to_array d);
  let d = A.create Rig.host D.Int64 [| 2; 3 |] in
  done_ ~msg:"iota" (map0 (Iota 1) ~dst:d);
  equal ~msg:"iota along axis 1" (array int64) [| 0L; 1L; 2L; 0L; 1L; 2L |]
    (A.to_array d);
  equal ~msg:"an iota is an int64" answer A.Wrong_dtype
    (map0 (Iota 0) ~dst:(A.create Rig.host D.Int32 [| 2 |]));
  (* Large enough for a job of several units, each starting inside a run. *)
  let big = A.create Rig.host D.Int64 [| 3; 70001 |] in
  List.iter
    (fun axis ->
      done_ ~msg:"iota" (map0 (Iota axis) ~dst:big);
      let want =
        Array.init (3 * 70001) (fun p ->
            Int64.of_int (if axis = 0 then p / 70001 else p mod 70001))
      in
      equal ~msg:(strf "iota along axis %d of 3x70001" axis) (array int64) want
        (A.to_array big))
    [ 0; 1 ];
  let d = A.create Rig.host D.Int32 [| 2; 3 |] in
  done_ ~msg:"fill" (map0 (Fill (P.bits D.Int32 7l)) ~dst:d);
  equal ~msg:"fill" (array int32) (Array.make 6 7l) (A.to_array d);
  (* A signalling NaN with a payload, stored as its bits. *)
  let nan = Bytes.create 4 in
  Bytes.set_int32_le nan 0 0x7f800021l;
  let d = A.create Rig.host D.Float32 [| 3 |] in
  done_ ~msg:"fill" (map0 (Fill (Bytes.to_string nan)) ~dst:d);
  equal ~msg:"fill: a NaN's bits" (array int32) (Array.make 3 0x7f800021l)
    (A.to_array (Option.get (A.bitcast D.Int32 d)));
  (* A boolean byte of 2, true, stored as it is. *)
  let d = A.create Rig.host D.Bool [| 3 |] in
  done_ ~msg:"fill" (map0 (Fill "\002") ~dst:d);
  equal ~msg:"fill: a boolean's byte" (array int) [| 2; 2; 2 |]
    (A.to_array (Option.get (A.bitcast D.Uint8 d)));
  let i4 xs = A.of_array D.Int4 [| Array.length xs |] xs in
  let d = A.create Rig.host D.Int4 [| 2 |] in
  done_ ~msg:"add" (map2 (Binary Add) ~dst:d (i4 [| 7; -8 |]) (i4 [| 1; -1 |]));
  equal ~msg:"add: int4 wraps" (array int) [| -8; 7 |] (A.to_array d);
  let bf16 xs = A.of_array D.Bfloat16 [| Array.length xs |] xs in
  let d = A.create Rig.host D.Bfloat16 [| 1 |] in
  done_ ~msg:"add"
    (map2 (Binary Add) ~dst:d (bf16 [| 256. |]) (bf16 [| 1. |]));
  equal ~msg:"add: bfloat16 rounds once, ties to even" (array float_exact)
    [| 256. |] (A.to_array d);
  let unary ~msg u xs want =
    let d = A.create Rig.host D.Float32 [| Array.length xs |] in
    done_ ~msg (map1 (Unary u) ~dst:d (f32 xs));
    equal ~msg (array float_exact) want (A.to_array d)
  in
  unary ~msg:"round: half away from zero" Round [| 2.5; -2.5; 0.49999997 |]
    [| 3.; -3.; 0. |];
  unary ~msg:"exp2: exact at integers" Exp2 [| 3.; -1.; 10.; -149. |]
    [| 8.; 0.5; 1024.; 0x1p-149 |];
  unary ~msg:"log2: exact at powers of two" Log2 [| 8.; 0.5; 0x1p-149 |]
    [| 3.; -1.; -149. |];
  let d = A.create Rig.host D.Float32 [| 1 |] in
  done_ ~msg:"sign" (map1 (Unary Sign) ~dst:d (f32 [| Float.nan |]));
  equal ~msg:"sign: NaN for a NaN" bool true (Float.is_nan (A.get d [| 0 |]));
  let d = A.create Rig.host D.Int8 [| 4 |] in
  done_ ~msg:"recip" (map1 (Unary Recip) ~dst:d (i8 [| 1; -1; 2; 0 |]));
  equal ~msg:"recip: x for 1 and -1, 0 otherwise" (array int) [| 1; -1; 0; 0 |]
    (A.to_array d);
  let d = A.create Rig.host D.Uint32 [| 1 |] in
  done_ ~msg:"bitcast" (map1 Bitcast ~dst:d (f32 [| 1. |]));
  equal ~msg:"bitcast: 1.0's bits" (array int32) [| 0x3f800000l |]
    (A.to_array d);
  (* A kind off its domain makes no program. *)
  raises_match ~msg:"bitcast across widths" Exn.invalid_arg (fun () ->
      map1 Bitcast ~dst:(A.create Rig.host D.Float64 [| 1 |]) (f32 [| 1. |]));
  raises_match ~msg:"exp on int32" Exn.invalid_arg (fun () ->
      map1 (Unary Exp)
        ~dst:(A.create Rig.host D.Int32 [| 1 |])
        (A.of_array D.Int32 [| 1 |] [| 1l |]));
  raises_match ~msg:"idiv on float32" Exn.invalid_arg (fun () ->
      map2 (Binary Idiv)
        ~dst:(A.create Rig.host D.Float32 [| 2 |])
        (f32 [| 1.; 2. |]) (f32 [| 1.; 2. |]));
  let module K = (val k) in
  let f32' = D.Any D.Float32 in
  equal ~msg:"a destination of another dtype is refused" answer A.Wrong_dtype
    (K.map
       (P.of_node ~ins:[| f32' |] (Op1 (Unary Neg, f32', 0)))
       ~dsts:[| A.Any (A.create Rig.host D.Float64 [| 1 |]) |]
       [| Plain (f32 [| 1. |]) |])

(* A map of constants alone holds no value in a slot: it runs at its
   destination's shape all the same. *)
let test_constant_map () =
  let f32 = D.Any D.Float32 in
  let p = P.v ~ins:[||] [| P.Const (f32, P.bits D.Float32 2.5) |] ~outs:[| 0 |] in
  let y = A.create Rig.host D.Float32 [| 3; 5 |] in
  match Nx_cpu.map p ~dsts:[| A.Any y |] [||] with
  | A.Done -> equal (array float_exact) (Array.make 15 2.5) (A.to_array y)
  | r -> failf "map answered %a" Nx_array_support.pp_answer r

(* A loop is computed whatever its number of loads and outputs together,
   the arrays the door reads in runs past the most a block holds: a map, a
   reduction of two reductions and a scan, load 0 padded and the coordinate
   read. The program weighs each load by its own odd weight, so a load read
   in another's place changes the result; every value is exact. *)
let test_loop_widths () =
  let f64 = D.Any D.Float64 in
  let weight j = (2 * j) + 3 in
  let fill = 7. in
  let operand j =
    if j = 0 then A.of_array D.Float64 [| 2 |] [| 1.; 2. |]
    else
      A.of_array D.Float64 [| 3 |]
        (Array.init 3 (fun i -> Float.of_int ((j * 3) + i)))
  in
  let pad : Nx_kernel.pad =
    { lo = [| 1 |]; hi = [| 0 |]; interior = [| 0 |]; windows = [||] }
  in
  let loads n =
    Array.init n (fun j ->
        if j = 0 then Nx_kernel.Padded { x = operand 0; fill; pad }
        else Nx_kernel.Plain (operand j))
  in
  (* Load [j]'s element [i], as the loop reads it. *)
  let read j i =
    if j = 0 then (if i = 0 then fill else A.get (operand 0) [| i - 1 |])
    else A.get (operand j) [| i |]
  in
  let prog n ~both =
    let x j = P.E.input j D.Float64 in
    let weighed w = List.init n (fun j -> P.E.(const D.Float64 (w j) * x j)) in
    let total ts = List.fold_left P.E.add (List.hd ts) (List.tl ts) in
    let first =
      P.E.(total (weighed (fun j -> Float.of_int (weight j)))
           + cast D.Float64 (coord 0))
    and second = total (weighed Float.of_int) in
    P.of_exprs
      ~ins:(Iarray.init n (fun _ -> f64))
      (if both then [| P.E.any first; P.E.any second |]
       else [| P.E.any first |])
  in
  let first n i =
    Float.of_int i
    +. List.fold_left ( +. ) 0.
         (List.init n (fun j -> Float.of_int (weight j) *. read j i))
  and second n i =
    List.fold_left ( +. ) 0. (List.init n (fun j -> Float.of_int j *. read j i))
  in
  let widths w =
    let n = w - 2 and msg k = Printf.sprintf "%s, %d arrays" k w in
    let fresh s = A.create Rig.host D.Float64 s in
    let s = fresh [| 3 |] and q = fresh [| 3 |] in
    equal ~msg:(msg "map") answer A.Done
      (Nx_cpu.map (prog n ~both:true) ~dsts:[| A.Any s; A.Any q |] (loads n));
    equal ~msg:(msg "map's first") (array float_exact) (Array.init 3 (first n))
      (A.to_array s);
    equal ~msg:(msg "map's second") (array float_exact)
      (Array.init 3 (second n)) (A.to_array q);
    let s = fresh [||] and q = fresh [||] in
    equal ~msg:(msg "reduction") answer A.Done
      (Nx_cpu.reduce (prog n ~both:true) ~shape:[| 3 |] ~axes:[| 0 |]
         [| (Monoid Sum, 0); (Monoid Max, 1) |]
         ~dsts:[| A.Any s; A.Any q |] (loads n));
    equal ~msg:(msg "reduction's sum") float_exact
      (Array.fold_left ( +. ) 0. (Array.init 3 (first n))) (A.get s [||]);
    equal ~msg:(msg "reduction's max") float_exact
      (Array.fold_left Float.max Float.neg_infinity (Array.init 3 (second n)))
      (A.get q [||]);
    let s = fresh [| 3 |] in
    equal ~msg:(msg "scan") answer A.Done
      (Nx_cpu.scan (prog (n + 1) ~both:false) ~axis:0 (Monoid Sum, 0)
         ~dsts:[| A.Any s |] (loads (n + 1)));
    let terms = Array.init 3 (first (n + 1)) in
    let running =
      Array.mapi
        (fun i _ -> Array.fold_left ( +. ) 0. (Array.sub terms 0 (i + 1)))
        terms
    in
    equal ~msg:(msg "scan's sums") (array float_exact) running (A.to_array s)
  in
  List.iter widths [ 15; 17; 18; 19; 32; 100 ]

(* A boolean constant of any byte is true iff the byte is not zero, inside a
   program of several nodes as alone: a [Where] it selects by and its [Cast]
   to int32, two outputs, which nx.cpu's general map computes. *)
let test_bool_bytes () =
  let f32 = D.Any D.Float32 in
  let xs = [| 1.; 2. |] and ys = [| -1.; -2. |] in
  for b = 0 to 255 do
    let flag = P.Const (D.Any D.Bool, String.make 1 (Char.chr b)) in
    let p =
      P.v ~ins:[| f32; f32 |]
        [|
          In 0; In 1; flag; Op3 (Where, 2, 0, 1); Op1 (Cast, D.Any D.Int32, 2);
        |]
        ~outs:[| 3; 4 |]
    in
    let x = A.of_array D.Float32 [| 2 |] xs
    and y = A.of_array D.Float32 [| 2 |] ys in
    let sel = A.create Rig.host D.Float32 [| 2 |]
    and int = A.create Rig.host D.Int32 [| 2 |] in
    let msg = Printf.sprintf "byte %d" b in
    equal ~msg answer A.Done
      (Nx_cpu.map p ~dsts:[| A.Any sel; A.Any int |]
         [| Nx_kernel.Plain x; Plain y |]);
    equal ~msg (array float_exact) (if b <> 0 then xs else ys) (A.to_array sel);
    equal ~msg (array int32)
      (Array.make 2 (if b <> 0 then 1l else 0l))
      (A.to_array int)
  done

(* A map whose coordinate names an axis past its rank answers
   [Shape_mismatch] and writes nothing. *)
let test_coord_past_rank () =
  let p =
    P.v ~ins:[||] [| P.Coord 1; Op1 (Cast, D.Any D.Float32, 0) |] ~outs:[| 1 |]
  in
  let y = A.of_array D.Float32 [| 3 |] [| 9.; 9.; 9. |] in
  equal ~msg:"answer" answer A.Shape_mismatch
    (Nx_cpu.map p ~dsts:[| A.Any y |] [||]);
  equal ~msg:"untouched" (array float_exact) [| 9.; 9.; 9. |] (A.to_array y)

(* A map whose padded load has more elements than a layout admits, or
   another shape than its destinations, answers [Shape_mismatch] before it
   copies the load: padded to 2^50 elements, more than any host's memory,
   and writes nothing. *)
let test_padded_misfits () =
  let x = A.of_array D.Float32 [| 2 |] [| 1.; 2. |] in
  let padded lo : Nx_kernel.load =
    Padded
      {
        x;
        fill = 0.;
        pad = { lo = [| lo |]; hi = [| 0 |]; interior = [| 0 |]; windows = [||] };
      }
  in
  let f32 = D.Any D.Float32 in
  List.iter
    (fun (name, p) ->
      List.iter
        (fun (what, lo) ->
          let y = A.of_array D.Float32 [| 3 |] [| 9.; 9.; 9. |] in
          let msg = name ^ ", " ^ what in
          equal ~msg answer A.Shape_mismatch
            (Nx_cpu.map p ~dsts:[| A.Any y |] [| padded lo |]);
          equal ~msg:(msg ^ ": untouched") (array float_exact) [| 9.; 9.; 9. |]
            (A.to_array y))
        [ ("past max_numel", L.max_numel - 1); ("another shape", 1 lsl 50) ])
    [
      ("identity", P.v ~ins:[| f32 |] [| P.In 0 |] ~outs:[| 0 |]);
      ( "a program",
        P.v ~ins:[| f32 |] [| P.In 0; Op2 (Binary Add, 0, 0) |] ~outs:[| 1 |] );
    ]

(* A padded load of a dead operand answers [Dead_buffer] and writes
   nothing, also where none of the operand's elements lands in the loop
   and every element read is the fill: float32 [2] cropped by 2 and padded
   by 3, read by a map, a reduction and a scan. *)
let test_padded_dead () =
  let x = A.of_array D.Float32 [| 2 |] [| 1.; 2. |] in
  equal ~msg:"freed" bool true (B.free (A.buffer x));
  let pad =
    {
      Nx_kernel.lo = [| -2 |];
      hi = [| 3 |];
      interior = [| 0 |];
      windows = [||];
    }
  in
  let load = Nx_kernel.Padded { x; fill = 0.; pad } in
  let id = P.v ~ins:[| D.Any D.Float32 |] [| P.In 0 |] ~outs:[| 0 |] in
  let refused msg s call =
    let n = Array.fold_left ( * ) 1 s in
    let y = A.of_array D.Float32 (Iarray.of_array s) (Array.make n 9.) in
    equal ~msg answer A.Dead_buffer (call (A.Any y));
    equal ~msg:(msg ^ ": untouched") (array float_exact) (Array.make n 9.)
      (A.to_array y)
  in
  refused "a map" [| 3 |] (fun y -> Nx_cpu.map id ~dsts:[| y |] [| load |]);
  refused "a reduction" [||] (fun y ->
      Nx_cpu.reduce id ~shape:[| 3 |] ~axes:[| 0 |]
        [| (Monoid Sum, 0) |]
        ~dsts:[| y |] [| load |]);
  refused "a scan" [| 3 |] (fun y ->
      Nx_cpu.scan id ~axis:0 (Monoid Sum, 0) ~dsts:[| y |] [| load |])

(* A map that holds many complex128 values gets blocks of few elements; over
   a transposed float32 operand the walk tiles rows, as many as fit the
   block. Each output is [k] times [2x], cast back to float32. *)
let test_tiled_short_blocks () =
  let n = 64 and k = 140 in
  let f32 = D.Any D.Float32 and c128 = D.Any D.Complex128 in
  let cast = P.Op1 (Cast, c128, 0) in
  let twice = Array.init k (fun _ -> P.Op2 (Binary Add, 1, 1)) in
  let sums =
    Array.init (k - 1) (fun j ->
        if j = 0 then P.Op2 (Binary Add, 2, 3)
        else P.Op2 (Binary Add, 2 + k + j - 1, 2 + j + 1))
  in
  let nodes = Array.concat [ [| P.In 0; cast |]; twice; sums ] in
  let last = Array.length nodes - 1 in
  let nodes = Array.append nodes [| P.Op1 (Cast, f32, last) |] in
  let p = P.v ~ins:[| f32 |] (Iarray.of_array nodes) ~outs:[| last + 1 |] in
  let data = Array.init (n * n) Float.of_int in
  let x =
    Option.get
      (A.move (M.Permute [| 1; 0 |]) (A.of_array D.Float32 [| n; n |] data))
  in
  let y = A.create Rig.host D.Float32 [| n; n |] in
  match
    Nx_cpu.map p ~dsts:[| A.Any y |] [| Nx_kernel.Plain x |]
  with
  | A.Done ->
      let want =
        Array.init (n * n) (fun q ->
            let i = q / n and j = q mod n in
            Float.of_int (2 * k) *. data.((j * n) + i))
      in
      equal (array float_exact) want (A.to_array y)
  | r -> failf "map answered %a" Nx_array_support.pp_answer r

(* Programs *)

(* A program of sixteen loads, reading the coordinate along every axis, as
   its nodes run one by one. *)
let law_map_full (b : Support.backend) (Case x) =
  let module K = (val b.kernels) in
  let shape = L.shape (A.layout x) in
  let rank = Iarray.length shape and nins = 16 in
  let i64 = D.Any D.Int64 in
  let coord a = [ P.Coord a; P.Op2 (Binary Add, nins + (2 * a), nins + (2 * a) + 1) ] in
  let nodes =
    List.init nins (fun k -> P.In k)
    @ (P.Op1 (Cast, i64, 0) :: List.concat_map coord (List.init rank Fun.id))
  in
  let p =
    P.v
      ~ins:(Iarray.init nins (fun _ -> D.Any (A.dtype x)))
      (Iarray.of_list nodes)
      ~outs:[| List.length nodes - 1 |]
  in
  let types = Array.init (List.length nodes) (P.dtype p) in
  let x = on b x in
  let ops = Array.make nins (A.Any x) in
  cover "coordinates" (rank > 0);
  match split b p types shape ops with
  | exception Skip -> fail "a node declined"
  | values ->
      let dst = A.create b.device D.Int64 shape in
      equal ~msg:"map" answer A.Done
        (K.map p ~dsts:[| A.Any dst |] (Array.map (fun (A.Any x) -> Nx_kernel.Plain x) ops));
      let (A.Any want) = values.(List.length nodes - 1) in
      equal (array int) (bits_of (host want)) (bits_of (host dst))

(* The widths of the maps a law draws: the loads and outputs together, one
   below a block's operands to many past them. *)
let widths = Gen.of_list ~pp:Format.pp_print_int [ 17; 18; 19; 32; 100 ]

(* Other dtypes a wide map loads, each staged or held its own way. *)
let wide_dtypes = D.[ Any Int32; Any Float16; Any Uint4; Any Float64 ]

(* A map of [w] arrays, two outputs and [w - 2] loads, has the bits of its
   nodes run one by one. The loads are [x], [y] and contiguous arrays of
   [wide_dtypes] in turn, the last a float64 that destination 0 is, every
   other seed, identical to; the program casts each load to float64 and
   sums them with the coordinate along the last axis into output 0, and
   casts that sum to [x]'s dtype into output 1. *)
let law_map_widths (b : Support.backend) ((Pair (x, y), seed), w) =
  let module K = (val b.kernels) in
  let f64 = D.Any D.Float64 and shape = L.shape (A.layout x) in
  let rank = Iarray.length shape and nins = w - 2 in
  let dt = D.Any (A.dtype x) in
  let other k = List.nth wide_dtypes (k mod List.length wide_dtypes) in
  let load k =
    if k = nins - 1 then A.Any (on b (seeded D.Float64 shape seed))
    else
      match k mod 3 with
      | 0 -> A.Any (on b x)
      | 1 -> A.Any (on b y)
      | _ ->
          let (D.Any d) = other k in
          A.Any (on b (seeded d shape (seed + k)))
  in
  let ops = Array.init nins load in
  let ins = Array.map (fun (A.Any a) -> D.Any (A.dtype a)) ops in
  let nodes = ref [] in
  let add n =
    nodes := n :: !nodes;
    List.length !nodes - 1
  in
  let loads = List.init nins (fun k -> add (P.In k)) in
  let casts = List.map (fun k -> add (P.Op1 (Cast, f64, k))) loads in
  let terms =
    if rank = 0 then casts
    else casts @ [ add (P.Op1 (Cast, f64, add (P.Coord (rank - 1)))) ]
  in
  let sum =
    List.fold_left (fun acc t -> add (P.Op2 (Binary Add, acc, t)))
      (List.hd terms) (List.tl terms)
  in
  let back = add (P.Op1 (Cast, dt, sum)) in
  let p =
    P.v ~ins:(Iarray.of_array ins)
      (Iarray.of_list (List.rev !nodes))
      ~outs:[| sum; back |]
  in
  let types = Array.init (List.length !nodes) (P.dtype p) in
  (* The reference reads a copy of the load the destination may be. *)
  let (A.Any last) = ops.(nins - 1) in
  let kept = Array.copy ops in
  kept.(nins - 1) <- A.Any (on b (A.copy (host last)));
  let alias = seed mod 2 = 0 in
  cover "a destination identical to a load" alias;
  cover "no element" (L.numel (A.layout x) = 0);
  cover "a broadcast or strided load"
    (not (L.is_contiguous (A.layout x) && L.is_contiguous (A.layout y)));
  cover "past a block's operands" (w > A.max_operands);
  match split b p types shape kept with
  | exception Skip -> fail "a node declined"
  | values ->
      let s =
        if alias then A.expect D.Float64 ops.(nins - 1)
        else A.create b.device D.Float64 shape
      in
      let (D.Any d) = dt in
      let q = A.create b.device d shape in
      let msg = strf "%d arrays" w in
      equal ~msg answer A.Done
        (K.map p ~dsts:[| A.Any s; A.Any q |]
           (Array.map (fun (A.Any a) -> Nx_kernel.Plain a) ops));
      let check name (A.Any got) i =
        let (A.Any want) = values.(i) in
        equal ~msg:(strf "%s, %s" msg name)
          (same_values (D.Any (A.dtype want)))
          (bits_of (host want)) (bits_of (host got))
      in
      check "the sum" (A.Any s) sum;
      check "its cast" (A.Any q) back

(* A program that holds hundreds of values at once, over a padded load and
   the coordinates, as its nodes run one by one: [m] products of the operand,
   cast to int64, by a constant or a coordinate, all summed at the end. *)
let law_map_wide (b : Support.backend) ((Case x), m) =
  let module K = (val b.kernels) in
  let shape = L.shape (A.layout x) in
  let rank = Iarray.length shape and i64 = D.Any D.Int64 in
  let nodes = ref [ P.Op1 (Cast, i64, 0); P.In 0 ] in
  let add n =
    nodes := n :: !nodes;
    List.length !nodes - 1
  in
  let coords = List.init rank (fun a -> add (P.Coord a)) in
  let c = 1 in
  let factor i =
    match coords with
    | [] -> add (P.Const (i64, P.bits D.Int64 (Int64.of_int i)))
    | _ -> List.nth coords (i mod rank)
  in
  let values = List.init m (fun i -> add (P.Op2 (Binary Mul, c, factor i))) in
  let sum =
    List.fold_left (fun acc v -> add (P.Op2 (Binary Add, acc, v)))
      (List.hd values) (List.tl values)
  in
  let nodes = Array.of_list (List.rev !nodes) in
  let p =
    P.v ~ins:(Iarray.init 1 (fun _ -> D.Any (A.dtype x)))
      (Iarray.of_list (Array.to_list nodes)) ~outs:[| sum |]
  in
  let types = Array.init (Array.length nodes) (P.dtype p) in
  let x = on b x in
  cover "more than 256 values held" (held p > 256);
  cover "coordinates" (rank > 0);
  match split b p types shape [| A.Any x |] with
  | exception Skip -> fail "a node declined"
  | v ->
      let dst = A.create b.device D.Int64 shape in
      equal ~msg:"map" answer A.Done
        (K.map p ~dsts:[| A.Any dst |] [| unpadded x |]);
      let (A.Any want) = v.(sum) in
      equal (array int) (bits_of (host want)) (bits_of (host dst))

(* Guard pages *)

(* What an elementwise kernel computes, with its result's dtype. *)
type elementwise =
  | K0 of Support.op0
  | K1 of P.op1 * D.any
  | K2 of P.op2
  | K3 of P.op3

(* A kernel of [kind] at [dt] over [ops], each drawn through a view, into a
   result of shape [shape]. *)
type guard = {
  kind : elementwise;
  dt : D.any;
  shape : int array;
  ops : A.any array;
  views : string list;
}

let kind_name = function
  | K0 (Fill _) -> "fill"
  | K0 (Iota _) -> "iota"
  | K1 (Copy, _) -> "copy"
  | K1 (Cast, _) -> "cast"
  | K1 (Bitcast, _) -> "bitcast"
  | K1 (Unary u, _) -> List.assoc u unaries
  | K2 k -> name2 k
  | K3 Where -> "where"
  | K3 Fma -> "fma"

let pp_guard ppf g =
  Format.fprintf ppf "%s at %a into %a of %a (%s)" (kind_name g.kind) pp_dtype
    g.dt pp_ints g.shape
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (A.Any x) ->
         Format.fprintf ppf "%a %a" D.pp (A.dtype x) L.pp (A.layout x)))
    (Array.to_list g.ops)
    (String.concat ", " g.views)

(* Shapes whose last extent runs from one element to a few vectors past the
   widest, sub-byte ones' bits included. *)
let guard_shape =
  let open Gen in
  let* n =
    frequency
      [
        (6, int_range 1 72);
        (2, int_range 73 300);
        ( 1,
          of_list ~pp:Format.pp_print_int [ 511; 512; 513; 1023; 1024; 1025 ]
        );
      ]
  in
  let+ outer = array ~size:(int_range 0 2) (int_range 1 3) in
  Array.append outer [| n |]

(* An operand of [d] and shape [s] through a drawn view, broadcast along
   its first axis one time in four, its elements from [rs]. *)
let guard_operand d s rs =
  let open Gen in
  let r = Array.length s in
  let* lead = frequency [ (3, constant false); (1, constant (r >= 2)) ] in
  let+ v = Elements.view_of ~broadcast:(if lead then [ 0 ] else []) r in
  ( Elements.operand d s v (fun _ -> Elements.element d rs),
    Elements.view_names v )

let guard_case =
  Gen.with_pp pp_guard
    (let open Gen in
     let* (D.Any dt as d) = dtypes in
     let* shape = guard_shape in
     let* to_ = dtypes in
     let* fill = int in
     let width = List.filter (fun (D.Any w) -> D.bits w = D.bits dt) D.all in
     let* same = of_list ~pp:pp_dtype width in
     let* cond = of_list ~pp:pp_dtype D.[ Any Bool; Any Bit ] in
     let ones =
       List.filter_map
         (fun (u, _) -> if P.accepts1 (Unary u) dt dt then Some u else None)
         unaries
     in
     let twos = List.filter (fun k -> P.accepts2 k dt) op2s in
     let* axis = int_range 0 (Array.length shape - 1) in
     let rs = Random.State.make [| fill |] in
     let accepts = function
       | K0 (Fill _) -> true
       | K0 (Iota _) -> D.equal dt D.Int64
       | K1 (k, D.Any y) -> P.accepts1 k dt y
       | K2 k -> P.accepts2 k dt
       | K3 Where ->
           let (D.Any c) = cond in
           P.accepts3 Where c dt
       | K3 Fma -> P.accepts3 Fma dt dt
     in
     let some =
       List.filter_map (fun (w, k) ->
           if accepts k then Some (w, constant k) else None)
     in
     let unary =
       if ones = [] then []
       else [ (4, map (fun u -> K1 (Unary u, d)) (of_list ones)) ]
     in
     let binary =
       if twos = [] then [] else [ (4, map (fun k -> K2 k) (of_list twos)) ]
     in
     let* kind =
       frequency
         (some
            [
              (* Sub-byte fills, whose end bytes are shared, come more
                 often. *)
              ( (if D.bits dt < 8 then 6 else 1),
                K0 (Fill (Elements.element d rs)) );
              (1, K0 (Iota axis));
              (1, K1 (Copy, d));
              (2, K1 (Cast, to_));
              (1, K1 (Bitcast, same));
              (1, K3 Where);
              (1, K3 Fma);
            ]
         @ unary @ binary)
     in
     let ds =
       match kind with
       | K0 _ -> []
       | K1 _ -> [ d ]
       | K2 _ -> [ d; d ]
       | K3 Where -> [ cond; d; d ]
       | K3 Fma -> [ d; d; d ]
     in
     let rec operands = function
       | [] -> constant []
       | d :: rest ->
           let* x = guard_operand d shape rs in
           let+ xs = operands rest in
           x :: xs
     in
     let+ ops = operands ds in
     {
       kind;
       dt = d;
       shape;
       ops = Array.of_list (List.map fst ops);
       views = List.concat_map snd ops;
     })

(* The result's dtype. *)
let result_dtype g =
  match g.kind with
  | K1 (_, d) -> d
  | K2 (Compare _) -> D.Any D.Bool
  | K0 _ | K2 (Binary _) | K3 _ -> g.dt

(* The same answer and bits with the operands, through their views, and the
   result against an inaccessible page. *)
let law_guards (b : Support.backend) (g, side) =
  let module K = (val b.kernels) in
  let after = side = Support.After in
  collect (kind_name g.kind);
  cover "of no operand" (match g.kind with K0 _ -> true | _ -> false);
  cover "a sub-byte fill"
    (match (g.kind, g.dt) with
    | K0 (Fill _), D.Any dt -> D.bits dt < 8
    | _ -> false);
  cover "a cast" (match g.kind with K1 (Cast, _) -> true | _ -> false);
  cover "a bitcast" (match g.kind with K1 (Bitcast, _) -> true | _ -> false);
  cover "a unary kind" (match g.kind with K1 (Unary _, _) -> true | _ -> false);
  cover "a binary kind" (match g.kind with K2 (Binary _) -> true | _ -> false);
  cover "a comparison" (match g.kind with K2 (Compare _) -> true | _ -> false);
  cover "of three operands" (match g.kind with K3 _ -> true | _ -> false);
  cover "sub-byte" (let (D.Any dt) = g.dt in D.bits dt < 8);
  cover "a stepped operand at the page before"
    ((not after) && List.mem "stepped" g.views);
  cover "a reversed operand at the page after"
    (after && List.mem "reversed" g.views);
  cover "a broadcast operand" (List.mem "broadcast" g.views);
  let call xs ds =
    let (A.Any dst) = ds.(0) in
    match g.kind with
    | K0 k -> Support.map0 (module K) k ~dst
    | K1 (k, _) ->
        let (A.Any x) = xs.(0) in
        Support.map1 (module K) k ~dst x
    | K2 k ->
        let (A.Any x) = xs.(0) in
        Support.map2 (module K) k ~dst x (A.expect (A.dtype x) xs.(1))
    | K3 k ->
        let (A.Any c) = xs.(0) in
        let (A.Any x) = xs.(1) in
        Support.map3 (module K) k ~dst:(A.expect (A.dtype x) ds.(0)) c x
          (A.expect (A.dtype x) xs.(2))
  in
  match
    Elements.same_guarded ~any_nan:true side ~call g.ops
      [| (result_dtype g, g.shape) |]
  with
  | A.Done -> cover "computed" true
  | A.Declined -> cover "declined" true
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a

(* Large copies and casts, staged through blocks and threads, with the
   operand and the result against the page. *)
let law_large_guards (b : Support.backend) ((Case a as c), d, side) =
  let module K = (val b.kernels) in
  covers_large c;
  List.iter
    (fun (k, d) ->
      let call xs ds =
        let (A.Any x) = xs.(0) in
        let (A.Any dst) = ds.(0) in
        Support.map1 (module K) k ~dst x
      in
      equal answer A.Done
        (Elements.same_guarded side ~call [| A.Any a |]
           [| (d, Iarray.to_array (L.shape (A.layout a))) |]))
    [ (P.Copy, D.Any (A.dtype a)); (Cast, d) ]

(* A map's outputs with its loads, plain or padded, through their views,
   and its destinations against the page. *)
let law_map_guards (b : Support.backend) ((Pair (x, y), seed), side) =
  let module K = (val b.kernels) in
  let st = Random.State.make [| seed |] in
  let dt = D.Any (A.dtype x) and shape = L.shape (A.layout x) in
  let ins, ops =
    if Random.State.bool st then ([| dt; dt |], [| A.Any x; A.Any y |])
    else ([| dt |], [| A.Any x |])
  in
  let padded = Random.State.bool st in
  cover "padded loads" padded;
  cover "plain loads" (not padded);
  let p, types = program st ins ~room:16 ~rank:(Iarray.length shape) in
  let s = Iarray.to_array shape in
  let dsts = Array.map (fun o -> (types.(o), s)) (Iarray.to_array p.outs) in
  let load (A.Any x) = if padded then unpadded x else Nx_kernel.Plain x in
  let call xs ds = K.map p ~dsts:ds (Array.map load xs) in
  equal answer A.Done (Elements.same_guarded side ~call ops dsts)

(* Erf's ranges and nested jobs *)

(* Erf at float64 is nx_kinds.h's at each index over rows whose pieces mix
   its two ranges, either one the most, into a fresh destination and into
   the operand itself: a row computes the range most of a piece's lanes are
   in and redoes the others. *)
let law_erf_ranges (near, seed) =
  let n = 1000 in
  let st = Random.State.make [| seed |] in
  let draw () =
    let a = Random.State.float st 1. in
    let v =
      if Random.State.float st 1. < near then a *. 2.5
      else 2.5 +. Float.succ 0. +. (a *. 5.)
    in
    if Random.State.bool st then -.v else v
  in
  let xs = Array.init n (fun _ -> draw ()) in
  let want = Array.map (fun v -> Int64.bits_of_float (Nx_kinds_support.f64 "erf" [| v |])) xs in
  let x = A.of_array D.Float64 [| n |] xs in
  let fresh = A.create Rig.host D.Float64 [| n |] in
  equal answer A.Done (Support.map1 (module Nx_cpu) (Unary Erf) ~dst:fresh x);
  equal ~msg:"fresh" (array int64) want (Array.map Int64.bits_of_float (A.to_array fresh));
  equal answer A.Done (Support.map1 (module Nx_cpu) (Unary Erf) ~dst:x x);
  equal ~msg:"in place" (array int64) want (Array.map Int64.bits_of_float (A.to_array x))

(* A job begun from a job's body runs to its end and adds what it is given,
   whether the outer job runs on the pool's threads or on the calling thread:
   64 units of 64 MiB each, and 1 unit of 2 MiB. Each of the n outer units
   adds 0 to n - 1. *)
let test_nested_jobs () =
  List.iter
    (fun (n, cost) ->
      equal
        ~msg:(Printf.sprintf "%d units of %d bytes" n cost)
        int
        (n * (n * (n - 1) / 2))
        (Support.nested n cost))
    [ (64, 64 * 1024 * 1024); (1, 2 * 1024 * 1024); (3, 0) ]

(* Programs that hold hundreds to thousands of values at once *)

(* How a held program's values are made: each straight from the operand, so
   that no slot is reused before the end, or each after a few steps, some
   folded into the sum as they go, so that slots are reused. *)
type pattern = Never | Reuse

let pp_pattern ppf p =
  Format.pp_print_string ppf (match p with Never -> "never" | Reuse -> "reuse")

let pp_side ppf = function
  | None -> Format.pp_print_string ppf "unguarded"
  | Some Support.Before -> Format.pp_print_string ppf "before"
  | Some Support.After -> Format.pp_print_string ppf "after"

(* A program over one operand of [xt]: its cast to [dt], then [k] values of
   [dt] made from it by kinds of one operand and of two with a constant,
   held as [pattern] says and summed by kinds of two; its outputs the sum
   and, sometimes, the cast. Raises [Skip] where [dt] takes no cast from
   [xt] or no kind of two operands. *)
let held_program st (D.Any xt) (D.Any dt) ~k ~pattern =
  let t = D.Any dt in
  let rev = ref [] and tys = ref [] and n = ref 0 in
  let add node ty =
    rev := node :: !rev;
    tys := ty :: !tys;
    incr n;
    !n - 1
  in
  let x = add (P.In 0) (D.Any xt) in
  if not (P.accepts1 Cast xt dt) then raise Skip;
  let base = add (P.Op1 (Cast, t, x)) t in
  let bins =
    List.filter
      (fun k -> match k with P.Binary _ -> P.accepts2 k dt | _ -> false)
      op2s
  in
  let uns = List.filter (fun (u, _) -> P.accepts1 (Unary u) dt dt) unaries in
  if bins = [] then raise Skip;
  let pick l = List.nth l (Random.State.int st (List.length l)) in
  let const () =
    let e = bits_of (seeded dt [| 1 |] (Random.State.bits st)) in
    let e = if D.is D.Boolean dt then [| e.(0) land 1 |] else e in
    add (P.Const (t, String.init (Array.length e) (fun i -> Char.chr e.(i)))) t
  in
  let step prev =
    if uns <> [] && Random.State.bool st then
      add (P.Op1 (Unary (fst (pick uns)), t, prev)) t
    else
      let c = const () in
      add (P.Op2 (pick bins, prev, c)) t
  in
  let acc = ref None and pending = Queue.create () in
  let fold v =
    acc :=
      Some
        (match !acc with
        | None -> v
        | Some a -> add (P.Op2 (pick bins, a, v)) t)
  in
  for _ = 1 to k do
    let r = match pattern with Never -> 0 | Reuse -> Random.State.int st 5 in
    let prev = ref base in
    for _ = 1 to r do
      prev := step !prev
    done;
    Queue.push (step !prev) pending;
    if pattern = Reuse && Random.State.int st 4 = 0 then
      fold (Queue.pop pending)
  done;
  Queue.iter fold pending;
  let last = Option.get !acc in
  let nodes = Array.of_list (List.rev !rev) in
  let outs = if Random.State.bool st then [| last |] else [| last; base |] in
  ( P.v ~ins:(Iarray.of_list [ D.Any xt ]) (Iarray.of_array nodes)
      ~outs:(Iarray.of_array outs),
    Array.of_list (List.rev !tys) )

(* A map of a held program gives the bits of its nodes run one by one, any
   NaN for a NaN, at every dtype, its operand and destinations in memory of
   their own beside a guard page or not. *)
let law_map_held ?(small = false) (b : Support.backend)
    (Case x, D.Any dt, k, pattern, side, seed) =
  let module K = (val b.kernels) in
  let st = Random.State.make [| seed |] in
  let shape = L.shape (A.layout x) in
  match held_program st (D.Any (A.dtype x)) (D.Any dt) ~k ~pattern with
  | exception Skip -> cover "cast declined" true
  | p, types -> (
      let h = held p in
      cover "more than 256 held" (h > 256);
      cover "more than 1040 held" (h > 1040);
      cover "reuse" (pattern = Reuse);
      cover "never" (pattern = Never);
      cover "guarded" (side <> None);
      let numel = Iarray.fold_left ( * ) 1 shape in
      if small then cover "a few elements" (numel <= 4)
      else cover "a thousand elements or more" (numel >= 1000);
      match split b p types shape [| A.Any x |] with
      | exception Skip -> fail "a node declined"
      | values ->
          let frees = ref [] in
          let guard (type v s) (dt : (v, s) D.t) l : (v, s) A.t =
            match side with
            | None -> A.create b.device dt (L.shape l)
            | Some s ->
                let g, free = Support.guarded s dt l in
                frees := free :: !frees;
                g
          in
          Fun.protect
            ~finally:(fun () -> List.iter (fun f -> f ()) !frees)
            (fun () ->
              let x' =
                match side with
                | None -> x
                | Some _ ->
                    let g = guard (A.dtype x) (L.contiguous shape) in
                    equal ~msg:"copy" answer A.Done
                      (Support.map1 (module Nx_cpu) P.Copy ~dst:g x);
                    g
              in
              let dsts =
                Array.map
                  (fun o ->
                    let (D.Any d) = types.(o) in
                    A.Any (guard d (L.contiguous shape)))
                  (Iarray.to_array p.outs)
              in
              equal ~msg:"map" answer A.Done
                (K.map p ~dsts [| Nx_kernel.Plain x' |]);
              Array.iteri
                (fun i o ->
                  let (A.Any want) = values.(o) in
                  let (A.Any got) = dsts.(i) in
                  equal
                    ~msg:
                      (strf "output %d, %d nodes, %d held, %s from %s" i
                         (Iarray.length p.nodes) h (D.name dt)
                         (D.name (A.dtype x)))
                    (same_values (D.Any (A.dtype want)))
                    (bits_of (host want)) (bits_of (host got)))
                (Iarray.to_array p.outs)))

(* Arrays of a few hundred to two thousand elements through each view of
   [large_at]: past a block's 16 elements, so a held program's walk runs
   many blocks on the job's threads, its arena on the heap past 1040
   values. *)
let held_views =
  Gen.with_pp pp_case
    (let open Gen in
     let* (D.Any dt) = dtypes in
     let* h = of_list ~pp:Format.pp_print_int [ 17; 33 ] in
     let* w = of_list ~pp:Format.pp_print_int [ 7; 65 ] in
     let* view = of_list ~pp:Format.pp_print_int [ 0; 1; 2; 3 ] in
     let+ seed = int in
     large_at dt h w view seed)

(* Arenas past the stack's at both ends of the widths: over a thousand
   int8 values, and three hundred complex128 ones, each slot of 16
   elements 256 bytes. *)
let heap_arenas =
  [
    (large_at D.Int8 33 65 1 1, D.Any D.Int8, 1200, Never, None, 1);
    ( large_at D.Complex128 33 65 2 2,
      D.Any D.Complex128,
      300,
      Never,
      Some Support.After,
      2 );
  ]

let held_cases cases =
  let open Gen in
  let* c = cases in
  let* dt = dtypes in
  let* k = frequency [ (3, int_range 257 400); (1, int_range 1000 1600) ] in
  let* pattern = of_list ~pp:pp_pattern [ Never; Reuse ] in
  let* side =
    of_list ~pp:pp_side [ None; Some Support.Before; Some Support.After ]
  in
  let+ seed = int in
  (c, dt, k, pattern, side, seed)

(* [n] elements of [dt] drawn from its specials alone, by [seed]: NaNs of
   both signs, signalling and quiet, with payloads, infinities, zeros, and
   numbers that some kinds make NaN from. *)
let specials_only (type v s) (dt : (v, s) D.t) n seed : (v, s) A.t =
  let sp = Array.of_list (specials dt) in
  let b = bytes_of_seed seed n in
  let pick i = sp.(Char.code b.[i] * 7 mod Array.length sp) in
  let data = String.concat "" (List.init n pick) in
  A.v dt (L.contiguous [| n |]) (B.of_string data)

(* A float kind's results at special values are of one class into a
   destination that is its operand and into a new one: a NaN where one is a
   NaN, else the same bits. *)
let law_nan_class (D.Any dt, seed) =
  let n = 300 in
  let x = specials_only dt n seed and y = specials_only dt n (seed + 1) in
  let z = specials_only dt n (seed + 2) in
  let unary (u, name) =
    if not (P.accepts1 (Unary u) dt dt) then None
    else
      Some (name, fun (module K : Nx_kernel.S) dst x ->
          Support.map1 (module K) (Unary u) ~dst x)
  in
  let binary = function
    | P.Binary b as k when P.accepts2 k dt ->
        Some (name2 k, fun (module K : Nx_kernel.S) dst x ->
            Support.map2 (module K) (Binary b) ~dst x y)
    | _ -> None
  in
  let fma (module K : Nx_kernel.S) dst x =
    Support.map3 (module K) Fma ~dst x y z
  in
  let kinds =
    List.filter_map unary unaries @ List.filter_map binary op2s
    @ [ ("fma", fma) ]
  in
  let check (name, run) =
    let under (b : Support.backend) ~alias =
      b.around (fun () ->
          let x' = A.copy x in
          let dst = if alias then x' else A.create Rig.host dt [| n |] in
          equal ~msg:name answer A.Done (run b.kernels dst x');
          bits_of dst)
    in
    let b = Support.cpu in
    equal ~msg:(strf "%s in place on %s" name b.name)
      (same_values (D.Any dt))
      (under b ~alias:false) (under b ~alias:true)
  in
  List.iter check kinds

(* The suite *)

(* nx.cpu under the table the host runs best: what nx_cpu.mli promises beyond
   Nx_kernel.S. *)
let cpu =
  let never _ _ = false in
  let maps (b : Support.backend) =
    [
      prop
        (strf "%s's kinds of one operand are nx_kinds.h's at each index" b.name)
        case
        (fun c ->
          b.around (fun () ->
              covers c;
              law_map1 b c));
      prop ~count:8
        (strf "%s's kinds of one operand over large views" b.name)
        large
        (fun c ->
          b.around (fun () ->
              law_map1
                ~kinds:P.[ (Exp, "exp"); (Sin, "sin"); (Neg, "neg") ]
                b c));
      prop
        (strf "%s's kinds of two operands are nx_kinds.h's at each index"
           b.name)
        pairs
        (fun (Pair (x, y) as p) ->
          b.around (fun () ->
              (* Operands whose loop is one run of steps of one or none, in
                 a dtype the kinds compute in, as nx.cpu takes in one row:
                 neither sub-byte nor a float narrower than float32. *)
              let dt = A.dtype x and lx = A.layout x in
              let c =
                L.coalesce [| L.contiguous (L.shape lx); lx; A.layout y |]
              in
              let one_run =
                L.rank c.(0) = 1
                && Array.for_all (fun l -> abs (L.stride l 0) <= 1) c
                && Array.for_all (fun l -> L.stride l 0 >= 0) c
                && D.bits dt >= 8
                && not (D.is D.Float dt && D.bits dt < 32)
              in
              cover "operands of one run in their own dtype" one_run;
              cover "operands of one run, of one element"
                (one_run && L.numel lx = 1);
              law_map2 b p));
      prop ~count:8 ~examples:[ large_bfloat16_pair ]
        (strf "%s's kinds of two operands over large views" b.name)
        large_pairs
        (fun c ->
          b.around (fun () ->
              law_map2 ~kinds:P.[ Binary Add; Compare Less ] b c));
      prop ~count:300
        (strf "%s computes every map its loop holds" b.name)
        (Gen.pair pairs Gen.int)
        (fun c -> b.around (fun () -> law_map_split ~declines:never b c));
      prop ~count:300
        (strf "%s computes held programs of a few elements" b.name)
        (held_cases case)
        (fun c -> b.around (fun () -> law_map_held ~small:true b c));
      prop ~count:12 ~examples:heap_arenas
        (strf "%s computes held programs over views" b.name)
        (held_cases held_views)
        (fun c -> b.around (fun () -> law_map_held b c));
      prop ~count:20
        (strf "%s computes a map of hundreds of values held at once" b.name)
        (Gen.pair case (Gen.int_range 257 1000))
        (fun c -> b.around (fun () -> law_map_wide b c));
      prop ~count:100
        (strf "%s computes programs of hundreds of nodes over padded loads"
           b.name)
        (Gen.pair pairs Gen.int)
        (fun c ->
          b.around (fun () ->
              law_map_split ~length:400 ~labels:false
                ~padded:true ~declines:never
                b c));
      prop
        (strf "%s computes a map of every operand with coordinates" b.name)
        case
        (fun c -> b.around (fun () -> law_map_full b c));
      prop ~count:200
        (strf "%s computes maps of any number of arrays" b.name)
        (Gen.pair (Gen.pair pairs Gen.int) widths)
        (fun c -> b.around (fun () -> law_map_widths b c));
      prop ~count:1000
        (strf
           "%s's elementwise kernels read and write no byte past their \
            operands and dst, whatever pages lie beside them"
           b.name)
        (Gen.pair guard_case Elements.sides)
        (fun c -> b.around (fun () -> law_guards b c));
      prop ~count:300
        (strf
           "%s's maps read and write no byte past their loads and dsts, \
            whatever pages lie beside them"
           b.name)
        (Gen.pair (Gen.pair pairs Gen.int) Elements.sides)
        (fun c -> b.around (fun () -> law_map_guards b c));
      prop ~count:32
        ~examples:
          (List.concat_map
             (fun c ->
               [
                 (c, D.Any D.Float64, Support.Before);
                 (c, D.Any D.Int8, Support.After);
               ])
             large_views)
        (strf
           "%s's large copies and casts read and write no byte past their \
            operand and dst"
           b.name)
        (Gen.triple large dtypes Elements.sides)
        (fun c -> b.around (fun () -> law_large_guards b c));
      prop ~count:32
        ~examples:(List.map (fun p -> (p, 1)) large_view_pairs)
        (strf "%s computes maps over large views" b.name)
        (Gen.pair large_pairs Gen.int)
        (fun ((Pair (x, _), _) as c) ->
          covers_large (Case x);
          b.around (fun () ->
              law_map_split ~declines:never ~labels:false b c));
    ]
  in
  group "nx.cpu"
    (maps Support.cpu
    @ [
      prop ~count:8
        "kinds at special values give one class in place"
        (Gen.pair
           (Gen.of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ])
           Gen.int)
        law_nan_class;
      test "refuses before any write"
        (test_refusals (module Nx_cpu : Nx_kernel.S));
      test "kinds give the values their documentation states" test_kind_values;
      test "a job begun from a job's body runs to its end" test_nested_jobs;
      test "a map of constants alone fills its destination" test_constant_map;
      test "loops of any number of loads and outputs are computed"
        test_loop_widths;
      test "a boolean constant is true iff its byte is not zero"
        test_bool_bytes;
      test "a map's coordinate past its rank is refused" test_coord_past_rank;
      test "a map refuses a padded load that does not fit before copying it"
        test_padded_misfits;
      test "a padded load of a dead operand is refused, fill alone or not"
        test_padded_dead;
      test "a map of short blocks over a transposed operand tiles them"
        test_tiled_short_blocks;
      prop ~count:20 "erf at float64 is nx_kinds.h's over pieces of both ranges"
        (Gen.pair (Gen.of_list ~pp:Format.pp_print_float [ 0.02; 0.4; 0.6; 0.98 ]) Gen.int)
        law_erf_ranges;
    ])

let () = exit (run "nx_cpu.kernels" [ cpu ])
