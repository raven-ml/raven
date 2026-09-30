(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reductions, scans, arg-reductions and sorts, lowered: each is checked against
   nx's eager result by its class. Integer sums and products, extremes,
   positions and sorts give eager's bits over the values that break them: both
   zeros, NaN, the infinities, ties, the integers' ends, and empty axes and axes
   of one element. A float sum is within the error of some association of its
   terms, and exact where every association is, as it is over small integers and
   powers of two. The graph is evaluated by tolk's reference interpreter. *)

open Windtrap
open Nx_test
open Traces

let pp_float ppf x = Format.fprintf ppf "%h" x

let traced f =
  let s, y = trace f in
  value s y

let agrees f = exact (f ()) (traced f)

(* Operands *)

type float_dtype = F : string * (float, 'b) Nx.dtype -> float_dtype

let float_dtypes =
  [
    F ("float32", Nx.float32);
    F ("float64", Nx.float64);
    F ("float16", Nx.float16);
    F ("bfloat16", Nx.bfloat16);
  ]

(* Shapes of at least one axis. *)
let ranked = Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range 0 4)

(* Values whose sums and products are exact in every association and every float
   dtype: small integers, halves and the special values. *)
let exact_float =
  Gen.of_list ~pp:pp_float
    [
      0.;
      -0.;
      1.;
      -1.;
      2.;
      -2.;
      0.5;
      3.;
      Float.infinity;
      Float.neg_infinity;
      Float.nan;
    ]

let small_float = Gen.map float_of_int (Gen.int_range (-8) 8)

(* The axes of [x], those that hold elements when [filled]. *)
let axes_of ~filled x =
  List.filter
    (fun a -> (not filled) || Nx.dim a x > 0)
    (List.init (Nx.ndim x) Fun.id)

(* A value of [dtype] under some layout, and some of its axes. *)
let with_axes ?(filled = false) ~pp dtype value =
  let open Gen in
  let* x = viewed ~pp dtype value in
  let+ axes = subsequence (axes_of ~filled x) in
  (x, axes)

(* A value of [dtype] under some layout, and one of its axes. *)
let with_axis ?(filled = false) ~pp dtype value =
  let open Gen in
  let* x =
    such_that
      (fun x -> axes_of ~filled x <> [])
      (viewed ~shape:ranked ~pp dtype value)
  in
  let+ axis = of_list (axes_of ~filled x) in
  (x, axis)

let pp_int ppf _ = Format.pp_print_string ppf "_"

type float_law = { on_float : 'b. (float, 'b) Nx.dtype -> test list }
type int_law = { on_int : 'a 'b. ('a, 'b) Nx.dtype -> 'a Gen.t -> test list }

(* Tests of each float dtype. *)
let per_float name { on_float } =
  group name
    (List.map (fun (F (dname, dt)) -> group dname (on_float dt)) float_dtypes)

(* Tests of each integer dtype, over the values that break arithmetic. *)
let per_int name { on_int } =
  group name
    (List.map
       (fun (Int_dtype { name; dtype; bits; signed; of_i64; _ }) ->
         group name (on_int dtype (Gen.map of_i64 (int_value ~bits ~signed))))
       int_dtypes)

(* Sums and products *)

type reduction = { r : 'a 'b. ?axes:int list -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let sum = { r = (fun ?axes x -> Nx.sum ?axes x) }
let prod = { r = (fun ?axes x -> Nx.prod ?axes x) }
let max = { r = (fun ?axes x -> Nx.max ?axes x) }
let min = { r = (fun ?axes x -> Nx.min ?axes x) }

let reduced name ?filled ~pp dtype value { r } =
  prop name (with_axes ?filled ~pp dtype value) (fun (x, axes) ->
      agrees (fun () -> r ~axes x))

(* [|a - e| <= 2 (n - 1) u sum |x|], [n] the terms of each output: [a] and [e]
   are each within half of it of the exact sum. *)
let rounded_sum (type b) (dt : (float, b) Nx.dtype) (x, axes) =
  let expected = Nx.sum ~axes x
  and actual = traced (fun () -> Nx.sum ~axes x) in
  let magnitude = Nx.to_array (Nx.sum ~axes (Nx.abs x)) in
  let terms = List.fold_left (fun n a -> n * (Nx.shape x).(a)) 1 axes in
  let u = if Nx_dtype.itemsize dt = 8 then 0x1p-53 else 0x1p-24 in
  Array.iteri
    (fun i e ->
      let a = (Nx.to_array actual).(i) in
      let bound = 2. *. float_of_int (terms - 1) *. u *. magnitude.(i) in
      if not (Float.abs (a -. e) <= bound) then
        failf "sum %d: %h, eager %h, beyond %h" i a e bound)
    (Nx.to_array expected)

let sums =
  group "sums and products"
    [
      per_int "integer sums wrap"
        { on_int = (fun dt v -> [ reduced "sum" ~pp:pp_int dt v sum ]) };
      per_int "integer products wrap"
        { on_int = (fun dt v -> [ reduced "prod" ~pp:pp_int dt v prod ]) };
      per_float "float sums of small integers are exact"
        {
          on_float =
            (fun dt -> [ reduced "sum" ~pp:pp_float dt small_float sum ]);
        };
      per_float "float sums of special values are exact"
        {
          on_float =
            (fun dt -> [ reduced "sum" ~pp:pp_float dt exact_float sum ]);
        };
      per_float "float products of special values are exact"
        {
          on_float =
            (fun dt -> [ reduced "prod" ~pp:pp_float dt exact_float prod ]);
        };
      group "a float sum is within the bound of an association"
        [
          prop "float32"
            (with_axes ~pp:pp_float Nx.float32 (Gen.float_range (-1e3) 1e3))
            (rounded_sum Nx.float32);
          prop "float64"
            (with_axes ~pp:pp_float Nx.float64 (Gen.float_range (-1e3) 1e3))
            (rounded_sum Nx.float64);
        ];
      group "a zero sum is +0."
        [
          test "over one term" (fun () ->
              let x = Nx.create Nx.float32 [| 1 |] [| -0. |] in
              agrees (fun () -> Nx.sum x));
          test "over an axis of one element" (fun () ->
              let x = Nx.full Nx.float32 [| 3; 1 |] (-0.) in
              agrees (fun () -> Nx.sum ~axes:[ 1 ] x));
          test "over an axis a kernel unrolls" (fun () ->
              let x = Nx.full Nx.float64 [| 2; 4 |] (-0.) in
              agrees (fun () -> Nx.sum ~axes:[ 0 ] x));
          test "in float16, accumulated in float32" (fun () ->
              let x = Nx.create Nx.float16 [| 2 |] [| -0.; -0. |] in
              agrees (fun () -> Nx.sum x));
          test "over no element" (fun () ->
              let x = Nx.zeros Nx.float32 [| 2; 0 |] in
              agrees (fun () -> Nx.sum ~axes:[ 1 ] x));
        ];
      test "a product over no element is 1" (fun () ->
          agrees (fun () -> Nx.prod ~axes:[ 1 ] (Nx.zeros Nx.int8 [| 2; 0 |]));
          agrees (fun () ->
              Nx.prod ~axes:[ 1 ] (Nx.zeros Nx.float32 [| 2; 0 |])));
      test "narrow floats round once" (fun () ->
          (* In float16, 2048 + 1 + 1 is 2048 term by term, and 2050 once. *)
          let x = Nx.create Nx.float16 [| 3 |] [| 2048.; 1.; 1. |] in
          agrees (fun () -> Nx.sum x);
          let y = Nx.create Nx.bfloat16 [| 3 |] [| 3.; 3.; 3. |] in
          agrees (fun () -> Nx.prod y));
    ]

(* Extremes *)

let extremes =
  group "extremes"
    [
      per_float "float extremes are exact"
        {
          on_float =
            (fun dt ->
              [
                reduced "max" ~filled:true ~pp:pp_float dt Gen.any_float max;
                reduced "min" ~filled:true ~pp:pp_float dt Gen.any_float min;
              ]);
        };
      per_int "integer extremes are exact"
        {
          on_int =
            (fun dt v ->
              [
                reduced "max" ~filled:true ~pp:pp_int dt v max;
                reduced "min" ~filled:true ~pp:pp_int dt v min;
              ]);
        };
      test "the maximum of both zeros is +0. and the minimum -0." (fun () ->
          List.iter
            (fun zeros ->
              let x = Nx.create Nx.float32 [| 2 |] zeros in
              agrees (fun () -> Nx.max x);
              agrees (fun () -> Nx.min x))
            [ [| -0.; 0. |]; [| 0.; -0. |] ]);
      test "NaN propagates wherever it lies" (fun () ->
          let x =
            Nx.create Nx.float32 [| 3; 3 |]
              [|
                Float.nan;
                1.;
                2.;
                1.;
                Float.nan;
                Float.infinity;
                1.;
                2.;
                Float.nan;
              |]
          in
          agrees (fun () -> Nx.max ~axes:[ 1 ] x);
          agrees (fun () -> Nx.min ~axes:[ 0 ] x));
      test "subnormals keep their order" (fun () ->
          let tiny = Float.ldexp 1. (-140) in
          let x = Nx.create Nx.float32 [| 3 |] [| -.tiny; 0.; tiny |] in
          agrees (fun () -> Nx.max x);
          agrees (fun () -> Nx.min x));
      test "booleans" (fun () ->
          let b =
            Nx.create Nx.bool [| 2; 2 |] [| true; false; false; false |]
          in
          agrees (fun () -> Nx.max ~axes:[ 1 ] b);
          agrees (fun () -> Nx.min ~axes:[ 1 ] b));
    ]

(* Scans *)

type scanning = { c : 'a 'b. axis:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let cummax = { c = (fun ~axis x -> Nx.cummax ~axis x) }
let cummin = { c = (fun ~axis x -> Nx.cummin ~axis x) }

let scans_of =
  [
    ("cumsum", { c = (fun ~axis x -> Nx.cumsum ~axis x) });
    ("cumprod", { c = (fun ~axis x -> Nx.cumprod ~axis x) });
    ("cummax", cummax);
    ("cummin", cummin);
  ]

let scanned name ~pp dtype value { c } =
  prop name (with_axis ~pp dtype value) (fun (x, axis) ->
      agrees (fun () -> c ~axis x))

let scans =
  group "scans"
    [
      per_int "integer scans are exact"
        {
          on_int =
            (fun dt v ->
              List.map (fun (op, c) -> scanned op ~pp:pp_int dt v c) scans_of);
        };
      per_float "float scans of special values are exact"
        {
          on_float =
            (fun dt ->
              List.map
                (fun (op, c) -> scanned op ~pp:pp_float dt exact_float c)
                scans_of);
        };
      per_float "float running extremes are exact"
        {
          on_float =
            (fun dt ->
              [
                scanned "cummax" ~pp:pp_float dt Gen.any_float cummax;
                scanned "cummin" ~pp:pp_float dt Gen.any_float cummin;
              ]);
        };
      test "a running sum starts from +0." (fun () ->
          let x = Nx.create Nx.float32 [| 3 |] [| -0.; -0.; 1. |] in
          agrees (fun () -> Nx.cumsum x));
      test "a running extreme is NaN from the first NaN on" (fun () ->
          let x =
            Nx.create Nx.float32 [| 5 |] [| 1.; -0.; Float.nan; 3.; 0. |]
          in
          agrees (fun () -> Nx.cummax x);
          agrees (fun () -> Nx.cummin x));
      test "an empty axis" (fun () ->
          agrees (fun () -> Nx.cumsum ~axis:1 (Nx.zeros Nx.float32 [| 2; 0 |])));
      test "a long axis runs in two stages" (fun () ->
          let n = 600 in
          let x =
            Nx.create Nx.int32 [| 2; n |]
              (Array.init (2 * n) (fun i ->
                   Int32.of_int ((i * 7919 mod 97) - 48)))
          in
          agrees (fun () -> Nx.cumsum ~axis:1 x);
          agrees (fun () -> Nx.cummax ~axis:1 x);
          agrees (fun () -> Nx.cummin ~axis:1 x);
          let y =
            Nx.create Nx.float32 [| n |]
              (Array.init n (fun i ->
                   if i = 400 then Float.nan else float_of_int (i * 31 mod 17)))
          in
          agrees (fun () -> Nx.cummax y));
    ]

(* Arg-reductions *)

type positioning = {
  p : 'a 'b. axis:int -> ('a, 'b) Nx.t -> (int32, Nx.int32_elt) Nx.t;
}

let positions_of =
  [
    ("argmax", { p = (fun ~axis x -> Nx.argmax ~axis x) });
    ("argmin", { p = (fun ~axis x -> Nx.argmin ~axis x) });
  ]

let positioned name ~pp dtype value { p } =
  prop name (with_axis ~filled:true ~pp dtype value) (fun (x, axis) ->
      agrees (fun () -> p ~axis x))

let arg_reductions =
  group "arg-reductions"
    [
      per_float "of floats"
        {
          on_float =
            (fun dt ->
              List.map
                (fun (op, p) -> positioned op ~pp:pp_float dt Gen.any_float p)
                positions_of);
        };
      per_int "of integers"
        {
          on_int =
            (fun dt v ->
              List.map
                (fun (op, p) -> positioned op ~pp:pp_int dt v p)
                positions_of);
        };
      test "the first of equal extremes" (fun () ->
          let x = Nx.create Nx.int8 [| 5 |] [| 1; 3; -2; 3; -2 |] in
          agrees (fun () -> Nx.argmax x);
          agrees (fun () -> Nx.argmin x));
      test "-0. is below +0." (fun () ->
          List.iter
            (fun zeros ->
              let x = Nx.create Nx.float32 [| 2 |] zeros in
              agrees (fun () -> Nx.argmax x);
              agrees (fun () -> Nx.argmin x))
            [ [| -0.; 0. |]; [| 0.; -0. |] ]);
      test "the first NaN is the extreme" (fun () ->
          let x =
            Nx.create Nx.float64 [| 5 |]
              [| Float.infinity; Float.nan; Float.neg_infinity; Float.nan; 0. |]
          in
          agrees (fun () -> Nx.argmax x);
          agrees (fun () -> Nx.argmin x));
      test "an axis of one element" (fun () ->
          agrees (fun () -> Nx.argmax ~axis:1 (Nx.zeros Nx.float32 [| 3; 1 |])));
    ]

(* Sorts *)

let sorted ~descending ~axis x =
  let values, positions = Nx.sort ~descending ~axis x in
  (values, positions)

let sort_agrees ~descending ~axis x =
  let values, positions = sorted ~descending ~axis x in
  exact values (traced (fun () -> fst (sorted ~descending ~axis x)));
  exact positions (traced (fun () -> Nx.argsort ~descending ~axis x))

let sort_props ~pp dtype value =
  List.map
    (fun descending ->
      prop
        (if descending then "descending" else "ascending")
        (with_axis ~pp dtype value)
        (fun (x, axis) -> sort_agrees ~descending ~axis x))
    [ false; true ]

(* Many ties, both zeros and NaN of both signs: whether the sort is stable. *)
let tied = Gen.of_list ~pp:pp_float [ 1.; -0.; 0.; Float.nan; -.Float.nan; -1. ]

let sorts =
  group "sorts"
    [
      per_float "of floats"
        { on_float = (fun dt -> sort_props ~pp:pp_float dt Gen.any_float) };
      per_float "of ties"
        {
          on_float =
            (fun dt ->
              sort_props ~pp:pp_float dt tied
              @ [
                  prop ~count:20 "along a long axis"
                    (Gen.map
                       (fun xs -> Nx.create dt [| 13 |] xs)
                       (Gen.array ~size:(Gen.constant 13) tied))
                    (fun x ->
                      sort_agrees ~descending:false ~axis:0 x;
                      sort_agrees ~descending:true ~axis:0 x);
                ]);
        };
      per_int "of integers"
        { on_int = (fun dt v -> sort_props ~pp:pp_int dt v) };
      test "-0. sorts before +0." (fun () ->
          let x = Nx.create Nx.float32 [| 4 |] [| 0.; -0.; 0.; -0. |] in
          sort_agrees ~descending:false ~axis:0 x;
          sort_agrees ~descending:true ~axis:0 x);
      test "a NaN keeps its bits" (fun () ->
          let bits = [| 0x7fc00001l; 0xffc00002l; 0x3f800000l; 0x7fc0beefl |] in
          let x = Nx.bitcast Nx.float32 (Nx.create Nx.int32 [| 4 |] bits) in
          let as_bits y = Nx.bitcast Nx.int32 y in
          agrees (fun () -> as_bits (fst (Nx.sort x)));
          agrees (fun () -> as_bits (fst (Nx.sort ~descending:true x))));
      test "64-bit keys equal in their high half" (fun () ->
          let x =
            Nx.create Nx.int64 [| 6 |]
              [|
                0x1_0000_0002L;
                -1L;
                0x1_0000_0001L;
                Int64.min_int;
                -0x1_0000_0000L;
                0x1_0000_0002L;
              |]
          in
          sort_agrees ~descending:false ~axis:0 x;
          sort_agrees ~descending:true ~axis:0 x;
          let u =
            Nx.create Nx.uint64 [| 4 |] [| -1L; 0x1_0000_0000L; 1L; -2L |]
          in
          sort_agrees ~descending:false ~axis:0 u);
      test "an axis of one element or none" (fun () ->
          sort_agrees ~descending:false ~axis:1 (Nx.zeros Nx.float32 [| 3; 1 |]);
          sort_agrees ~descending:false ~axis:1 (Nx.zeros Nx.float32 [| 3; 0 |]));
      test "booleans" (fun () ->
          let b =
            Nx.create Nx.bool [| 5 |] [| true; false; true; false; false |]
          in
          sort_agrees ~descending:false ~axis:0 b;
          sort_agrees ~descending:true ~axis:0 b);
    ]

(* Compiled for the host

   What a kernel computes where the interpreter cannot tell: the sign of a zero
   sum over a loop the compiler removes, and integer accumulators that overflow,
   which C leaves undefined for signed ones. *)

let compiled f =
  let s, y = trace f in
  exact (f ()) (Programs.compiled s y)

let on_the_host =
  group ~tags:[ "slow" ] "compiled for the host"
    [
      group "a zero sum is +0."
        [
          test "over one term" (fun () ->
              let x = Nx.full Nx.float32 [| 3; 1 |] (-0.) in
              compiled (fun () -> Nx.sum ~axes:[ 1 ] x));
          test "over an unrolled axis" (fun () ->
              let x = Nx.full Nx.float32 [| 4; 3 |] (-0.) in
              compiled (fun () -> Nx.sum ~axes:[ 0 ] x));
          test "running" (fun () ->
              let x = Nx.full Nx.float32 [| 4 |] (-0.) in
              compiled (fun () -> Nx.cumsum x));
        ];
      test "integer sums and products wrap" (fun () ->
          let x =
            Nx.create Nx.int32 [| 2; 3 |]
              [| Int32.max_int; Int32.max_int; 7l; Int32.min_int; -1l; 3l |]
          in
          compiled (fun () -> Nx.sum ~axes:[ 1 ] x);
          compiled (fun () -> Nx.prod ~axes:[ 1 ] x);
          let y = Nx.create Nx.int8 [| 3 |] [| 127; 127; -128 |] in
          compiled (fun () -> Nx.sum y);
          compiled (fun () -> Nx.prod y));
      test "extremes of NaN and both zeros" (fun () ->
          let x =
            Nx.create Nx.float32 [| 3; 3 |]
              [|
                -0.;
                0.;
                -1.;
                0.;
                -0.;
                Float.nan;
                Float.neg_infinity;
                0x1p-149;
                -0x1p-149;
              |]
          in
          compiled (fun () -> Nx.max ~axes:[ 1 ] x);
          compiled (fun () -> Nx.min ~axes:[ 1 ] x));
    ]

(* Graph parity: the kernels tinygrad schedules for the same program. *)

let parity =
  let x () = Nx.zeros Nx.float32 [| 4; 4 |] in
  let i () = Nx.zeros Nx.int32 [| 4; 4 |] in
  let u ?(shape = [| 4; 4 |]) () = Nx.zeros Nx.uint32 shape in
  let case file f =
    Golden.graph (file ^ ".golden") (fun () ->
        let args = f () in
        Programs.kernels (snd (trace (fun () -> args ()))))
  in
  group "graph parity"
    [
      case "sum_axis" (fun () ->
          let a = x () in
          fun () -> Nx.sum ~axes:[ 1 ] a);
      case "sum_all" (fun () ->
          let a = x () in
          fun () -> Nx.sum a);
      case "sum_uint" (fun () ->
          let a = u () in
          fun () -> Nx.sum ~axes:[ 1 ] a);
      case "prod_axis" (fun () ->
          let a = x () in
          fun () -> Nx.prod ~axes:[ 1 ] a);
      case "max_int" (fun () ->
          let a = i () in
          fun () -> Nx.max ~axes:[ 1 ] a);
      case "min_int" (fun () ->
          let a = i () in
          fun () -> Nx.min ~axes:[ 1 ] a);
      case "cumsum" (fun () ->
          let a = x () in
          fun () -> Nx.cumsum ~axis:1 a);
      case "cumsum_uint" (fun () ->
          let a = u () in
          fun () -> Nx.cumsum ~axis:1 a);
      case "cumsum_long" (fun () ->
          let a = u ~shape:[| 2; 600 |] () in
          fun () -> Nx.cumsum ~axis:1 a);
      case "cumprod" (fun () ->
          let a = x () in
          fun () -> Nx.cumprod ~axis:1 a);
      case "cummax_int" (fun () ->
          let a = i () in
          fun () -> Nx.cummax ~axis:1 a);
      case "cummin_int" (fun () ->
          let a = i () in
          fun () -> Nx.cummin ~axis:1 a);
      case "argmax_int" (fun () ->
          let a = i () in
          fun () -> Nx.argmax ~axis:1 a);
      case "argmin_int" (fun () ->
          let a = i () in
          fun () -> Nx.argmin ~axis:1 a);
      case "argsort_int" (fun () ->
          let a = i () in
          fun () -> Nx.argsort ~axis:1 a);
      case "argsort_int_descending" (fun () ->
          let a = i () in
          fun () -> Nx.argsort ~descending:true ~axis:1 a);
    ]

let () =
  exit
  @@ run "lower_reduce"
       [ sums; extremes; scans; arg_reductions; sorts; on_the_host; parity ]
