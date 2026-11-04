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
  (* A contiguous loop runs as one row, along which the coordinate of every
     axis but the first wraps, and repeats once its period has passed. *)
  cover "a coordinate repeating along a row"
    (L.is_contiguous (A.layout x)
    && Iarray.for_all (fun d -> d >= 1) shape
    && rank >= 2
    && Iarray.get shape (rank - 1) >= 2
    && Iarray.get shape (rank - 2) >= 2);
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

(* An operand of [d] and shape [s] through a drawn view, its elements
   from [rs]: one element repeated along each row where [along], else
   broadcast along its first axis one time in five, and one time in five
   along its last. *)
let guard_operand ~along d s rs =
  let open Gen in
  let r = Array.length s in
  let* broadcast =
    if along then constant [ r - 1 ]
    else
      frequency
        [
          (3, constant []);
          (1, constant (if r >= 2 then [ 0 ] else []));
          (1, constant [ r - 1 ]);
        ]
  in
  let+ v = Elements.view_of ~broadcast r in
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
     (* Every operand repeated along each row one time in six: a row of
        such operands computes its one value once. *)
     let* along = frequency [ (1, constant true); (5, constant false) ] in
     let rec operands = function
       | [] -> constant []
       | d :: rest ->
           let* x = guard_operand ~along d shape rs in
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
let law_kind_guards (b : Support.backend) (g, side) =
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
  let along (A.Any x) =
    let l = A.layout x in
    let r = L.rank l in
    r > 0 && L.stride l (r - 1) = 0 && Iarray.get (L.shape l) (r - 1) > 1
  in
  cover "every operand one element repeated along each row"
    (Array.length g.ops >= 2 && Array.for_all along g.ops);
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
   and its destinations against the page, a NaN for a NaN: a kind's NaN bits
   follow the operands' addresses. *)
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
  equal answer A.Done (Elements.same_guarded ~any_nan:true side ~call ops dsts);
  (* The last node cast to the loads' dtype, written over a C-contiguous
     load against the page, as a donated one is. *)
  let over = not padded in
  cover "written over a load" over;
  if over then begin
    let (D.Any d) = dt in
    let n = Iarray.length p.nodes in
    let q =
      P.v ~ins:(Iarray.of_array ins)
        (Iarray.of_list (Iarray.to_list p.nodes @ [ P.Op1 (Cast, dt, n - 1) ]))
        ~outs:[| n |]
    in
    let k = Random.State.int st (Array.length ops) in
    let fresh = A.create Rig.host d shape in
    equal answer A.Done (K.map q ~dsts:[| A.Any fresh |] (Array.map load ops));
    Elements.with_guards side (fun guard _ ->
        let (A.Any g) = guard (A.Any (A.copy (A.expect d ops.(k)))) in
        let ops = Array.copy ops in
        ops.(k) <- A.Any g;
        equal answer A.Done (K.map q ~dsts:[| A.Any g |] (Array.map load ops));
        equal (array (Elements.same dt)) (Elements.elements (A.Any fresh))
          (Elements.elements (A.Any g)))
  end

(* The kernels guarded: one of a kind over drawn views, a large copy and
   cast, or a program's map. *)
type guarded =
  | Kind of guard
  | Large of case * D.any
  | Program of pair * int

let pp_guarded ppf = function
  | Kind g -> pp_guard ppf g
  | Large (c, d) -> Format.fprintf ppf "%a cast to %a" pp_case c pp_dtype d
  | Program (p, seed) -> Format.fprintf ppf "%a program %d" pp_pair p seed

let guarded_cases =
  Gen.with_pp pp_guarded
    (Gen.frequency
       [
         (6, Gen.map (fun g -> Kind g) guard_case);
         (1, Gen.map (fun (c, d) -> Large (c, d)) (Gen.pair large dtypes));
         (3, Gen.map (fun (p, s) -> Program (p, s)) (Gen.pair pairs Gen.int));
       ])

let guarded_examples =
  List.concat_map
    (fun c ->
      [
        (Large (c, D.Any D.Float64), Support.Before);
        (Large (c, D.Any D.Int8), Support.After);
      ])
    large_views

let law_guards b (g, side) =
  match g with
  | Kind g -> law_kind_guards b (g, side)
  | Large (c, d) -> law_large_guards b (c, d, side)
  | Program (p, seed) -> law_map_guards b ((p, seed), side)

(* Kinds over repeated operands *)

(* A kind of two or three operands of [dtype] and [extents], each through
   its view, a Where's condition of [cond] through the first: a view
   repeats one element along each row of the last axis, or one element in
   all, or is drawn; [all] where every one repeats along each row. *)
type repeated = {
  dtype : D.any;
  cond : D.any;
  extents : int array;
  layouts : Elements.view array;
  all : bool;
  seed : int;
}

let pp_repeated ppf c =
  Format.fprintf ppf "%a, condition %a, %a (%s), seed %d" pp_dtype c.dtype
    pp_dtype c.cond pp_ints c.extents
    (String.concat " | "
       (Array.to_list
          (Array.map
             (fun (v : Elements.view) ->
               String.concat " "
                 (List.map string_of_int v.broadcast
                 @ Elements.view_names v))
             c.layouts)))
    c.seed

let repeated_cases =
  Gen.with_pp pp_repeated
    (let open Gen in
     let* dtype = dtypes in
     let* cond = of_list ~pp:pp_dtype D.[ Any Bool; Any Bit ] in
     let* n =
       frequency
         [
           (6, int_range 2 72);
           (1, int_range 73 300);
           (1, of_list ~pp:Format.pp_print_int [ 1; 511; 512; 513 ]);
         ]
     in
     let* outer = array ~size:(int_range 0 2) (int_range 1 3) in
     let extents = Array.append outer [| n |] in
     let r = Array.length extents in
     let along = Elements.view_of ~broadcast:[ r - 1 ] r in
     let layout =
       frequency
         [
           (3, along);
           (1, Elements.view_of ~broadcast:(List.init r Fun.id) r);
           (1, Elements.view_of r);
           (1, constant (Elements.plain r));
         ]
     in
     let* all = frequency [ (1, constant true); (2, constant false) ] in
     let one = if all then along else layout in
     let* a = one in
     let* b = one in
     let* c = one in
     let+ seed = int in
     { dtype; cond; extents; layouts = [| a; b; c |]; all; seed })

(* Each kind of two operands, and Where and Fma, over operands that repeat
   one element along each row gives nx_kinds.h's bits at each index: a
   row computes such an operand's element once, and a row of operands
   that all repeat one, its one value once. *)
let law_repeated (b : Support.backend) c =
  let module K = (val b.kernels) in
  let st = Random.State.make [| c.seed |] in
  let r = Array.length c.extents in
  let make d v =
    Elements.operand d c.extents v (fun _ -> Elements.element d st)
  in
  let along (v : Elements.view) =
    List.mem (r - 1) v.broadcast && c.extents.(r - 1) > 1
  in
  let rows = Elements.total (Array.sub c.extents 0 (r - 1)) > 1 in
  let a = along c.layouts.(0) and x = along c.layouts.(1)
  and y = along c.layouts.(2) in
  cover "both operands repeated along each row" (x && y);
  cover "both repeated along each row, rows apart" (x && y && rows);
  cover "one repeated along each row, the other not" (x <> y);
  cover "the condition repeated along each row" a;
  cover "both values repeated along each row, the condition not"
    (x && y && not a);
  cover "all three repeated along each row" (c.all && c.extents.(r - 1) > 1);
  cover "beside a stepped, permuted or reversed operand"
    (Array.exists
       (fun v -> List.exists (( <> ) "broadcast") (Elements.view_names v))
       c.layouts);
  let (D.Any dt) = c.dtype in
  let x = A.expect dt (make c.dtype c.layouts.(1)) in
  let y = A.expect dt (make c.dtype c.layouts.(2)) in
  let z = A.expect dt (make c.dtype c.layouts.(0)) in
  let shape = Iarray.of_array c.extents in
  law_map2 b (Pair (x, y));
  let w = max 1 (D.bits dt / 8) in
  let where (type s) (cnd : (bool, s) A.t) =
    answers b (K3 Where) c.dtype c.dtype shape
      ~accepted:(P.accepts3 Where (A.dtype cnd) dt)
      ~want:(fun () ->
        let cs = A.to_array cnd and xs = bits_of x and ys = bits_of y in
        Array.concat
          (List.mapi
             (fun i k -> Array.sub (if k then xs else ys) (i * w) w)
             (Array.to_list cs)))
      (fun dst ->
        Support.map3 (module K) Where ~dst:(A.expect dt (A.Any dst))
          (on b cnd) (on b x) (on b y))
  in
  (match make c.cond c.layouts.(0) with
  | A.Any k -> (
      match A.dtype k with D.Bool -> where k | D.Bit -> where k | _ -> ()));
  answers b (K3 Fma) c.dtype c.dtype shape ~accepted:(P.accepts3 Fma dt dt)
    ~want:(fun () -> expected "fma" dt ~compare:false [| z; x; y |])
    (fun dst ->
      Support.map3 (module K) Fma ~dst:(A.expect dt (A.Any dst)) (on b z)
        (on b x) (on b y))

(* Broadcast loads *)

(* The floats narrower than float32 of a byte or more, which a map holds
   as their codes and decodes where it reads them. *)
let narrow_floats =
  D.[ Any Float16; Any Bfloat16; Any Float8_e4m3fn; Any Float8_e5m2 ]

let is_narrow (D.Any d) =
  List.exists (fun (D.Any n) -> D.equal d n) narrow_floats

(* What a map over broadcast loads computes: a drawn program over its
   loads, or the cast of its one load into a dtype. *)
type broadcast_prog = Drawn | Cast_to of D.any

(* A map over one to three loads of [dt] and [shape], the first
   broadcast along some axes, each through its view; the elements and a
   drawn program from [seed]. *)
type broadcast = {
  dt : D.any;
  shape : int array;
  views : Elements.view array;
  prog : broadcast_prog;
  seed : int;
}

let pp_broadcast ppf c =
  let axes (v : Elements.view) =
    "[" ^ String.concat ";" (List.map string_of_int v.broadcast) ^ "]"
  in
  let prog =
    match c.prog with
    | Drawn -> "a drawn program"
    | Cast_to (D.Any t) -> "a cast to " ^ D.name t
  in
  Format.fprintf ppf "%a %a, loads broadcast along %s (%s), %s, seed %d"
    pp_dtype c.dt pp_ints c.shape
    (String.concat " " (Array.to_list (Array.map axes c.views)))
    (String.concat ", "
       (List.concat_map Elements.view_names (Array.to_list c.views)))
    prog c.seed

(* Axes of a rank [r] to broadcast along, the last more often than not; at
   least one where [some]. *)
let broadcast_axes ~some r =
  let open Gen in
  let* last = frequency [ (2, constant true); (1, bool) ] in
  let+ outer = list ~size:(constant (r - 1)) bool in
  let chosen a = if a = r - 1 then last else List.nth outer a in
  let axes = List.filter chosen (List.init r Fun.id) in
  if axes = [] && some then [ r - 1 ] else axes

(* How the first load is broadcast: one element along each row, rows
   apart, over outer axes of two elements or more; one row repeated along
   the first axis; or along drawn axes. *)
type spread = Apart | Row | Any_axes

let broadcast_cases =
  Gen.with_pp pp_broadcast
    (let open Gen in
     (* A narrow float a third of the time: a map decodes such a load
        where it reads it. *)
     let* (D.Any d as dt) =
       frequency [ (1, of_list ~pp:pp_dtype narrow_floats); (2, dtypes) ]
     in
     let* n =
       frequency
         [
           (6, int_range 1 72);
           (2, int_range 73 300);
           (1, of_list ~pp:Format.pp_print_int [ 511; 512; 513; 1025 ]);
         ]
     in
     let* spread =
       frequency
         [ (2, constant Apart); (1, constant Row); (1, constant Any_axes) ]
     in
     let* outer =
       match spread with
       | Apart | Row -> array ~size:(int_range 1 2) (int_range 2 5)
       | Any_axes -> array ~size:(int_range 0 2) (int_range 1 5)
     in
     let n = if spread = Apart then max n 2 else n in
     let shape = Array.append outer [| n |] in
     let r = Array.length shape in
     (* A load is C-contiguous where its view is plain, else permuted,
        stepped or reversed as drawn. *)
     let view broadcast =
       frequency
         [
           (1, constant { (Elements.plain r) with broadcast });
           (2, Elements.view_of ~broadcast r);
         ]
     in
     let* first =
       match spread with
       | Apart -> view [ r - 1 ]
       | Row -> view [ 0 ]
       | Any_axes ->
           let* axes = broadcast_axes ~some:true r in
           view axes
     in
     (* A cast a quarter of the time: its operand's rows are made
        contiguous before it converts them. *)
     let targets =
       List.filter
         (fun (D.Any t) -> P.accepts1 Cast d t && not (D.equal d t))
         D.all
     in
     let* prog =
       if targets = [] then constant Drawn
       else
         frequency
           [
             (3, constant Drawn);
             (1, map (fun t -> Cast_to t) (of_list ~pp:pp_dtype targets));
           ]
     in
     let* others =
       match prog with
       | Cast_to _ -> constant []
       | Drawn ->
           let other =
             let* axes = broadcast_axes ~some:false r in
             view axes
           in
           list ~size:(int_range 0 2) other
     in
     let+ seed = int in
     { dt; shape; views = Array.of_list (first :: others); prog; seed })

(* float32 rows of 2048 elements, longer than a block, beside a contiguous
   load: one element repeated along each row, or one row repeated. *)
let broadcast_examples =
  let case axes side =
    ( {
        dt = D.Any D.Float32;
        shape = [| 16; 2048 |];
        views =
          [| { (Elements.plain 2) with broadcast = axes }; Elements.plain 2 |];
        prog = Drawn;
        seed = 1;
      },
      side )
  in
  List.concat_map
    (fun axes -> [ case axes Support.Before; case axes Support.After ])
    [ [ 1 ]; [ 0 ] ]

(* A map reads a broadcast load as it reads the load's copy into a
   C-contiguous array, and computes what its nodes run one by one do:
   the same outputs, bit for bit, any NaN for a NaN, with the loads and
   destinations against an inaccessible page. A map whose loads all repeat
   one element along each row holds a value of them one element a row. *)
let law_map_broadcast (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  let st = Random.State.make [| c.seed |] in
  let r = Array.length c.shape in
  let ops =
    Array.map
      (fun v ->
        Elements.operand c.dt c.shape v (fun _ -> Elements.element c.dt st))
      c.views
  in
  let p, types =
    match c.prog with
    | Drawn ->
        let ins = Array.map (fun _ -> c.dt) ops in
        program st ins ~room:16 ~rank:r
    | Cast_to t ->
        ( P.v
            ~ins:(Iarray.of_list [ c.dt ])
            (Iarray.of_list [ P.In 0; P.Op1 (Cast, t, 0) ])
            ~outs:(Iarray.of_list [ 1 ]),
          [| c.dt; t |] )
  in
  let dsts =
    Array.map (fun o -> (types.(o), c.shape)) (Iarray.to_array p.outs)
  in
  (* The first load's axes of more than one element it does not
     broadcast, and whether its view reverses one or orders two apart
     from the result's order: made axis [m] holds the result's axis
     [inv.(m)]. *)
  let first = c.views.(0) and axes = List.init r Fun.id in
  let along a = List.mem a first.broadcast and wide a = c.shape.(a) > 1 in
  let kept = List.filter (fun a -> wide a && not (along a)) axes in
  let inv = Elements.inverse first.perm in
  let reversed =
    match first.reversed with Some m -> List.mem inv.(m) kept | None -> false
  in
  let permuted =
    List.exists
      (fun a ->
        List.exists (fun b -> a < b && first.perm.(a) > first.perm.(b)) kept)
      kept
  in
  let apart = along (r - 1) && wide (r - 1) && kept <> [] in
  let cast = match c.prog with Cast_to _ -> true | Drawn -> false in
  let narrow = is_narrow c.dt in
  cover "one element repeated along each row, rows apart" apart;
  cover "rows apart, the first load reversed" (apart && reversed);
  cover "rows apart, the first load permuted" (apart && permuted);
  cover "rows apart, a narrow float" (apart && narrow);
  cover "rows apart, cast" (apart && cast);
  cover "rows apart, a narrow float cast" (apart && narrow && cast);
  cover "a row repeated"
    ((not (along (r - 1))) && List.exists (fun a -> wide a && along a) axes);
  cover "one element repeated"
    (List.for_all along axes && Elements.total c.shape > 1);
  cover "beside another load stepped, permuted or reversed"
    (Array.exists
       (fun v -> List.exists (( <> ) "broadcast") (Elements.view_names v))
       (Array.sub c.views 1 (Array.length c.views - 1)));
  cover "every load repeated along each row, rows apart"
    (wide (r - 1) && kept <> []
    && Array.for_all (fun (v : Elements.view) -> List.mem (r - 1) v.broadcast)
         c.views);
  let call xs ds =
    K.map p ~dsts:ds (Array.map (fun (A.Any x) -> Nx_kernel.Plain x) xs)
  in
  let fresh =
    Array.map
      (fun (D.Any dt, s) -> A.Any (A.create Rig.host dt (Iarray.of_array s)))
      dsts
  in
  let copies = Array.map (fun (A.Any x) -> A.Any (A.copy x)) ops in
  equal ~msg:"copied" answer A.Done (call copies fresh);
  (match split b p types (Iarray.of_array c.shape) ops with
  | exception Skip -> fail "a node declined"
  | values ->
      Array.iteri
        (fun i o ->
          equal
            ~msg:(strf "output %d of %s, node by node" i (describe p))
            (array (Elements.same (fst dsts.(i))))
            (Elements.elements fresh.(i))
            (Elements.elements values.(o)))
        (Iarray.to_array p.outs));
  Elements.with_guards side (fun guard place ->
      let ds = Array.map (fun (d, s) -> place d s) dsts in
      equal ~msg:"broadcast" answer A.Done (call (Array.map guard ops) ds);
      Array.iteri
        (fun i d ->
          equal
            ~msg:(strf "output %d of %s" i (describe p))
            (array (Elements.same (fst dsts.(i))))
            (Elements.elements fresh.(i))
            (Elements.elements d))
        ds)

(* Values held one element a row late in a program *)

(* How a late program's second output is its sum: itself, or a Copy, a
   Cast or a Bitcast of it into a dtype of its width, which a map writes in
   place as it writes the sum. *)
type wrap = Bare | Copied | Cast_same | Bitcast_f64

(* A program of [negs] Negs of an int64 load of [outer] rows of [n]
   elements, repeated along each row, then a coordinate along the outer
   axis or along the rows, and the sum of the two. *)
type late = {
  negs : int;
  outer : int;
  n : int;
  along_rows : bool;
  wrap : wrap;
  seed : int;
}

let wrap_name = function
  | Bare -> "the sum"
  | Copied -> "a copy of the sum"
  | Cast_same -> "a cast of the sum into int64"
  | Bitcast_f64 -> "a bitcast of the sum into float64"

let pp_late ppf c =
  Format.fprintf ppf "%d negs, [%d; %d], coordinate along %s, %s, seed %d"
    c.negs c.outer c.n
    (if c.along_rows then "the rows" else "the outer axis")
    (wrap_name c.wrap) c.seed

let late_cases =
  Gen.with_pp pp_late
    (let open Gen in
     let* negs =
       frequency [ (3, int_range 505 700); (1, int_range 1 40) ]
     in
     let* outer = int_range 2 4 in
     let* n = int_range 2 100 in
     let* along_rows = frequency [ (3, constant false); (1, constant true) ] in
     let* wrap =
       of_list
         ~pp:(fun ppf w -> Format.pp_print_string ppf (wrap_name w))
         [ Bare; Copied; Cast_same; Bitcast_f64 ]
     in
     let+ seed = int in
     { negs; outer; n; along_rows; wrap; seed })

(* A map holds a value one element a row wherever it lies in its program,
   past its 512th node too, and writes an output that is a Copy, Cast or
   Bitcast of a kind in place: its outputs, the last Neg and the sum, are
   what its nodes run one by one give. *)
let law_map_late (b : Support.backend) c =
  let module K = (val b.kernels) in
  let st = Random.State.make [| c.seed |] in
  let shape = [| c.outer; c.n |] in
  let i64 = D.Any D.Int64 in
  let repeated = { (Elements.plain 2) with broadcast = [ 1 ] } in
  let x =
    Elements.operand i64 shape repeated (fun _ -> Elements.element i64 st)
  in
  let negs = List.init c.negs (fun k -> P.Op1 (Unary Neg, i64, k)) in
  let coord = c.negs + 1 and sum = c.negs + 2 in
  let wrap =
    match c.wrap with
    | Bare -> []
    | Copied -> [ P.Op1 (Copy, i64, sum) ]
    | Cast_same -> [ P.Op1 (Cast, i64, sum) ]
    | Bitcast_f64 -> [ P.Op1 (Bitcast, D.Any D.Float64, sum) ]
  in
  let nodes =
    (P.In 0 :: negs)
    @ [
        P.Coord (if c.along_rows then 1 else 0);
        P.Op2 (Binary Add, coord, c.negs);
      ]
    @ wrap
  in
  let last = List.length nodes - 1 in
  let p =
    P.v
      ~ins:(Iarray.of_list [ i64 ])
      (Iarray.of_list nodes)
      ~outs:(Iarray.of_list [ c.negs; last ])
  in
  let late = coord >= 512 in
  cover "a Neg of a repeated load past node 512" (c.negs >= 512);
  cover "a coordinate held one element a row past node 512"
    (late && not c.along_rows);
  cover "a coordinate along the rows past node 512" (late && c.along_rows);
  cover "a short program" (not late);
  cover "an output that is a Copy of a kind, written in place"
    (c.wrap = Copied);
  cover "an output that is a Cast of a kind, written in place"
    (c.wrap = Cast_same);
  cover "an output that is a Bitcast of a kind, written in place"
    (c.wrap = Bitcast_f64);
  let types = Array.init (List.length nodes) (P.dtype p) in
  match split b p types (Iarray.of_array shape) [| x |] with
  | exception Skip -> fail "a node declined"
  | values ->
      let outs = Iarray.to_array p.outs in
      let dsts =
        Array.map
          (fun o ->
            let (D.Any d) = types.(o) in
            A.Any (A.create b.device d (Iarray.of_array shape)))
          outs
      in
      equal ~msg:"map" answer A.Done
        (K.map p ~dsts
           (Array.map (fun (A.Any x) -> Nx_kernel.Plain x) [| x |]));
      Array.iteri
        (fun i o ->
          equal
            ~msg:(strf "output %d" i)
            (array (Elements.same types.(o)))
            (Elements.elements values.(o))
            (Elements.elements dsts.(i)))
        outs

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
let law_map_held (b : Support.backend)
    (Case x, D.Any dt, k, pattern, side, seed) =
  let module K = (val b.kernels) in
  let st = Random.State.make [| seed |] in
  let shape = L.shape (A.layout x) in
  match held_program st (D.Any (A.dtype x)) (D.Any dt) ~k ~pattern with
  | exception Skip -> cover "cast declined" true
  | p, types -> (
      let h = held p in
      cover "more than 256 held" (h > 256);
      cover "more than a thousand values held" (h > 1000);
      cover "reuse" (pattern = Reuse);
      cover "never" (pattern = Never);
      cover "guarded" (side <> None);
      let numel = Iarray.fold_left ( * ) 1 shape in
      cover "a few elements" (numel <= 4);
      cover "a thousand elements or more" (numel >= 1000);
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
   [large_at]: a held program's walk over many blocks on the job's
   threads. *)
let held_views =
  Gen.with_pp pp_case
    (let open Gen in
     let* (D.Any dt) = dtypes in
     let* h = of_list ~pp:Format.pp_print_int [ 17; 33 ] in
     let* w = of_list ~pp:Format.pp_print_int [ 7; 65 ] in
     let* view = of_list ~pp:Format.pp_print_int [ 0; 1; 2; 3 ] in
     let+ seed = int in
     large_at dt h w view seed)

(* Over a thousand int8 values held, and three hundred complex128 ones,
   over views of a few thousand elements. *)
let held_examples =
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

(* The suite *)

(* nx.cpu under the table the host runs best: what nx_cpu.mli promises beyond
   Nx_kernel.S. *)
let cpu =
  let never _ _ = false in
  let maps (b : Support.backend) =
    [
      prop ~examples:[ erf_case ]
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
      (* "a where" and "a load of one repeated element" come in one case
         in twenty-five: 400 cases reach each in every run. *)
      prop ~count:400
        (strf "%s computes every map its loop holds" b.name)
        split_cases
        (fun c -> b.around (fun () -> law_split ~declines:never b c));
      prop ~count:200 ~examples:held_examples
        (strf "%s computes held programs" b.name)
        (held_cases (Gen.frequency [ (3, case); (1, held_views) ]))
        (fun c -> b.around (fun () -> law_map_held b c));
      prop ~count:20
        (strf "%s computes a map of hundreds of values held at once" b.name)
        (Gen.pair case (Gen.int_range 257 1000))
        (fun c -> b.around (fun () -> law_map_wide b c));
      (* A contiguous int32 3x4 reaches "a coordinate repeating along a
         row" in every run; [case] draws it in about one run in a hundred. *)
      prop ~examples:[ large_at D.Int32 3 4 0 1 ]
        (strf "%s computes a map of every operand with coordinates" b.name)
        case
        (fun c -> b.around (fun () -> law_map_full b c));
      prop ~count:200
        (strf "%s computes maps of any number of arrays" b.name)
        (Gen.pair (Gen.pair pairs Gen.int) widths)
        (fun c -> b.around (fun () -> law_map_widths b c));
      prop ~count:200
        (strf "%s's kinds read operands repeated along their rows" b.name)
        repeated_cases
        (fun c -> b.around (fun () -> law_repeated b c));
      prop ~count:100
        (strf "%s holds values one element a row late in a program" b.name)
        late_cases
        (fun c -> b.around (fun () -> law_map_late b c));
      prop ~count:300 ~examples:broadcast_examples
        (strf "%s reads a broadcast load as its copy" b.name)
        (Gen.pair broadcast_cases Elements.sides)
        (fun c -> b.around (fun () -> law_map_broadcast b c));
      prop ~count:600 ~examples:guarded_examples
        (strf
           "%s reads and writes no byte past its operands, loads and \
            destinations, whatever pages lie beside them"
           b.name)
        (Gen.pair guarded_cases Elements.sides)
        (fun c -> b.around (fun () -> law_guards b c));
    ]
  in
  group "nx.cpu"
    (maps Support.cpu
    @ [
      test "refuses before any write"
        (test_refusals (module Nx_cpu : Nx_kernel.S));
      test "kinds give the values their documentation states" test_kind_values;
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
    ])

let () = exit (run "nx_cpu.kernels" [ cpu ])
