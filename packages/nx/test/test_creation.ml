(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Nx_test

let ints = Ref.witness int32
let floats = Ref.witness (float_rel ~rel:1e-12 ~abs:1e-12)
let shape = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4)
let small = Gen.int_range (-6) 6
let value = Gen.map Int32.of_int small
let index_value s idx = Int32.of_int (Ref.ravel s idx)
let one_if c = if c then 1l else 0l

let row_major s =
  Array.init (Ref.numel s) (fun i -> index_value s (Ref.unravel s i))

let filled =
  group "filled tensors"
    [
      prop "create lays its data out in row-major order" shape (fun s ->
          equal ints
            (Ref.init s (index_value s))
            (Ref.of_nx (Nx.create Nx.int32 s (row_major s))));
      prop "init puts f idx at idx" shape (fun s ->
          equal ints
            (Ref.init s (index_value s))
            (Ref.of_nx (Nx.init Nx.int32 s (index_value s))));
      prop "full puts its value everywhere" (Gen.pair shape value)
        (fun (s, v) ->
          equal ints
            (Ref.init s (fun _ -> v))
            (Ref.of_nx (Nx.full Nx.int32 s v)));
      prop "zeros, ones and full_like are full" (Gen.pair shape value)
        (fun (s, v) ->
          let t = Nx.create Nx.int32 s (row_major s) in
          equal ints
            (Ref.init s (fun _ -> 0l))
            (Ref.of_nx (Nx.zeros Nx.int32 s));
          equal ints (Ref.init s (fun _ -> 1l)) (Ref.of_nx (Nx.ones Nx.int32 s));
          equal ints (Ref.init s (fun _ -> v)) (Ref.of_nx (Nx.full_like t v));
          equal ints (Ref.init s (fun _ -> 0l)) (Ref.of_nx (Nx.zeros_like t));
          equal ints (Ref.init s (fun _ -> 1l)) (Ref.of_nx (Nx.ones_like t)));
      prop "scalar and scalar_like have shape [||]" value (fun v ->
          equal ints (Ref.create [||] [| v |])
            (Ref.of_nx (Nx.scalar Nx.int32 v));
          equal ints (Ref.create [||] [| v |])
            (Ref.of_nx (Nx.scalar_like (Nx.zeros Nx.int32 [| 2 |]) v)));
      test "create refuses data of the wrong length" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.create Nx.int32 [| 2; 3 |] [| 1l; 2l; 3l |]));
      test "create refuses a negative dimension" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.create Nx.int32 [| 2; -3 |] [| 1l; 2l |]));
      test
        "create, zeros and eye refuse negative dimensions whose product is the \
         element count (nx.mli is silent)" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.create Nx.int32 [| -2; -3 |] (Array.make 6 0l));
          raises_invalid_arg (fun () -> Nx.zeros Nx.int32 [| -2; -3 |]);
          raises_invalid_arg (fun () -> Nx.eye ~m:(-2) Nx.int32 (-3)));
    ]

let dims = Gen.int_range 0 5

let diagonals =
  group "diagonals"
    [
      prop "eye is one exactly where column - row = k"
        (Gen.triple dims dims (Gen.int_range (-7) 7))
        (fun (n, m, k) ->
          equal ints
            (Ref.init [| n; m |] (fun i -> one_if (i.(1) - i.(0) = k)))
            (Ref.of_nx (Nx.eye ~m ~k Nx.int32 n)));
      prop "diag of a vector puts it on the k-th diagonal" (Gen.pair dims small)
        (fun (n, k) ->
          let v = Array.init n (fun i -> Int32.of_int (i + 1)) in
          let size = n + abs k in
          equal ints
            (Ref.init [| size; size |] (fun i ->
                 if i.(1) - i.(0) = k then v.(Int.min i.(0) i.(1)) else 0l))
            (Ref.of_nx (Nx.diag ~k (Nx.create Nx.int32 [| n |] v))));
      prop "diag of a matrix reads the diagonal that diag of a vector wrote"
        (Gen.pair dims small) (fun (n, k) ->
          Law.round_trip (tensor int32) (tensor int32) (Nx.diag ~k) (Nx.diag ~k)
            (Nx.arange Nx.int32 1 (n + 1) 1));
      prop "tril k and triu (k + 1) partition each matrix of a batch"
        (Gen.quad (Gen.int_range 0 2) dims dims small)
        (fun (b, n, m, k) ->
          let s = if b = 0 then [| n; m |] else [| b; n; m |] in
          let x = Nx.create Nx.int32 s (row_major s) in
          equal (tensor int32) x (Nx.add (Nx.tril ~k x) (Nx.triu ~k:(k + 1) x));
          equal ints
            (Ref.init s (fun i ->
                 let r = i.(Array.length s - 2)
                 and c = i.(Array.length s - 1) in
                 if c - r <= k then index_value s i else 0l))
            (Ref.of_nx (Nx.tril ~k x)));
      test "diag refuses a scalar and a rank-3 tensor" (fun () ->
          raises_invalid_arg (fun () -> Nx.diag (Nx.scalar Nx.int32 1l));
          raises_invalid_arg (fun () ->
              Nx.diag (Nx.zeros Nx.int32 [| 1; 1; 1 |])));
      test "tril and triu refuse a vector" (fun () ->
          raises_invalid_arg (fun () -> Nx.tril (Nx.zeros Nx.int32 [| 3 |]));
          raises_invalid_arg (fun () -> Nx.triu (Nx.zeros Nx.int32 [| 3 |])));
    ]

let arithmetic_progression start stop step =
  let rec go i =
    if (step > 0 && i < stop) || (step < 0 && i > stop) then i :: go (i + step)
    else []
  in
  go start

let linear_points ~endpoint start stop n =
  let span = if endpoint then n - 1 else n in
  let step = if span = 0 then 0. else (stop -. start) /. float_of_int span in
  Array.init n (fun i -> start +. (float_of_int i *. step))

type dtype = Dtype : ('a, 'b) Nx.dtype -> dtype

(* Each dtype with the least and greatest integers it holds, from arange's
   contract: an integer dtype's range, 0 and 1 for bool, and the magnitudes up
   to a float dtype's largest finite value, clipped to OCaml's ints. *)
let held =
  [
    (Dtype Nx.bool, 0, 1);
    (Dtype Nx.int4, -8, 7);
    (Dtype Nx.uint4, 0, 15);
    (Dtype Nx.int8, -128, 127);
    (Dtype Nx.uint8, 0, 255);
    (Dtype Nx.int16, -32768, 32767);
    (Dtype Nx.uint16, 0, 65535);
    (Dtype Nx.int32, -(1 lsl 31), (1 lsl 31) - 1);
    (Dtype Nx.uint32, 0, (1 lsl 32) - 1);
    (Dtype Nx.int64, min_int, max_int);
    (Dtype Nx.uint64, 0, max_int);
    (Dtype Nx.float16, -65504, 65504);
    (Dtype Nx.bfloat16, min_int, max_int);
    (Dtype Nx.float32, min_int, max_int);
    (Dtype Nx.float64, min_int, max_int);
    (Dtype Nx.float8_e4m3, -448, 448);
    (Dtype Nx.float8_e5m2, -57344, 57344);
    (Dtype Nx.complex64, min_int, max_int);
    (Dtype Nx.complex128, min_int, max_int);
  ]

let pp_held ppf (Dtype d, _, _) = Nx.pp_dtype ppf d

let int64s l =
  let a = Array.of_list (List.map Int64.of_int l) in
  Nx.create Nx.int64 [| Array.length a |] a

(* Tensors of one dtype, equal in shape and elements. *)
let same () =
  Testable.make ~pp:Nx.pp ~equal:(fun a b ->
      Nx.shape a = Nx.shape b && Nx.to_array a = Nx.to_array b)

(* Each bound a dtype holds short of OCaml's ints, with the direction that
   leaves the dtype: +1 past the greatest, -1 past the least. *)
let held_bounds =
  List.concat_map
    (fun ((_, least, greatest) as d) ->
      (if least > min_int then [ (d, least, -1) ] else [])
      @ if greatest < max_int then [ (d, greatest, 1) ] else [])
    held

(* The bound [b] is reached from inside, toward and away from it, and the values
   one past it are refused from both directions. *)
let held_bound ((Dtype dtype, _, _), b, out) =
  let holds start stop step expected =
    equal (same ())
      (Nx.cast dtype (int64s expected))
      (Nx.arange dtype start stop step)
  in
  holds (b - out) (b + out) out [ b - out; b ];
  holds b (b - (2 * out)) (-out) [ b; b - out ];
  raises_invalid_arg (fun () -> Nx.arange dtype (b - out) (b + (2 * out)) out);
  raises_invalid_arg (fun () ->
      Nx.arange dtype (b + out) (b - (2 * out)) (-out))

let bound = Gen.float_range (-100.) 100.

type float_dtype = F : (float, 'b) Nx.dtype -> float_dtype

let pp_float_dtype ppf (F d) = Nx.pp_dtype ppf d

let float_dtypes =
  Gen.of_list ~pp:pp_float_dtype
    [
      F Nx.float16;
      F Nx.float32;
      F Nx.float64;
      F Nx.bfloat16;
      F Nx.float8_e4m3;
      F Nx.float8_e5m2;
    ]

(* Bounds of every magnitude and both zeros, clamped to [d]'s finite values. *)
let linspace_bound (F d) =
  let hi = Nx_dtype.max_finite d in
  Gen.map
    (fun x -> Float.min hi (Float.max (-.hi) x))
    (Gen.frequency
       [
         (3, bound);
         (2, Gen.float);
         ( 1,
           Gen.of_list ~pp:Format.pp_print_float
             [ 0.; -0.; 0.1; -0.1; 1e300; -1e300; hi; -.hi ] );
       ])

let pp_linspace ppf (d, endpoint, start, stop, n) =
  Format.fprintf ppf "linspace ~endpoint:%b %a %h %h %d" endpoint pp_float_dtype
    d start stop n

let linspace_case =
  Gen.with_pp pp_linspace
  @@
  let open Gen in
  let* d = float_dtypes in
  let+ start = linspace_bound d
  and+ stop = linspace_bound d
  and+ n = frequency [ (1, int_range 1 2); (2, int_range 3 40) ]
  and+ endpoint = bool in
  (d, endpoint, start, stop, n)

(* [x] stored in [d]: what an element given [x] holds. *)
let stored d x = Nx.item [] (Nx.scalar d x)

let linspace_bounds (F d, endpoint, start, stop, n) =
  let narrow = Nx_dtype.itemsize d < 4 in
  cover "one point" (n = 1);
  cover "two points" (n = 2);
  cover "a reversed range" (stop < start);
  cover "a narrow float" narrow;
  cover "a range wider than the largest finite value"
    (Float.abs (stop -. start) > Nx_dtype.max_finite d);
  let xs = Nx.to_array (Nx.linspace ~endpoint d start stop n) in
  let start = stored d start and stop = stored d stop in
  equal ~msg:"the first point" float_exact start xs.(0);
  if endpoint && n >= 2 then
    equal ~msg:"the last point" float_exact stop xs.(n - 1);
  (* Between them numerically: a zero between zeros may have either sign. *)
  let numeric = Testable.with_compare Stdlib.compare float_exact in
  let lo = Float.min start stop and hi = Float.max start stop in
  Array.iteri
    (fun i x ->
      let msg = Printf.sprintf "point %d" i in
      at_least ~msg numeric ~than:lo x;
      at_most ~msg numeric ~than:hi x)
    xs

let ranges =
  group "ranges"
    [
      prop "arange is start + i * step in every dtype that holds its values"
        (Gen.quad
           (Gen.of_list ~pp:pp_held held)
           (Gen.int_range (-20) 20) (Gen.int_range (-20) 20)
           (Gen.one_of [ Gen.int_range (-7) (-1); Gen.int_range 1 7 ]))
        (fun ((Dtype dtype, least, greatest), start, stop, step) ->
          let values = arithmetic_progression start stop step in
          let fits =
            List.for_all (fun v -> least <= v && v <= greatest) values
          in
          cover "every value fits" (fits && values <> []);
          cover "a value does not fit" (not fits);
          if fits then
            equal (same ())
              (Nx.cast dtype (int64s values))
              (Nx.arange dtype start stop step)
          else raises_invalid_arg (fun () -> Nx.arange dtype start stop step));
      prop "a long arange is start + i * step, up to the ends of int64"
        (let open Gen in
         let* n = int_range 1025 5000 in
         let* step = int_range 1 (max_int / n) in
         let+ step = of_list [ step; -step ]
         and+ slack = int_range 0 1000
         and+ low = bool in
         (* A start at either end of OCaml's ints that the values, and the
            stop one step past them, stay within. *)
         let span = n * abs step in
         let start =
           match (low, step > 0) with
           | true, true -> min_int + slack
           | true, false -> min_int + span + slack
           | false, true -> max_int - span - slack
           | false, false -> max_int - slack
         in
         (n, start, step))
        (fun (n, start, step) ->
          let stop = start + (n * step) in
          equal (same ())
            (int64s (List.init n (fun i -> start + (i * step))))
            (Nx.arange Nx.int64 start stop step));
      prop "arange_f counts from start by step while short of stop"
        (Gen.triple (Gen.int_range (-20) 20) (Gen.int_range (-20) 20)
           (Gen.one_of [ Gen.int_range (-7) (-1); Gen.int_range 1 7 ]))
        (fun (start, stop, step) ->
          (* Quarters are exact in binary, so the count is exact too. *)
          let quarter i = float_of_int i /. 4. in
          let l =
            Array.of_list
              (List.map quarter (arithmetic_progression start stop step))
          in
          equal (Ref.witness float_exact)
            (Ref.create [| Array.length l |] l)
            (Ref.of_nx
               (Nx.arange_f Nx.float64 (quarter start) (quarter stop)
                  (quarter step))));
      prop "linspace spaces n points evenly from start to stop"
        (Gen.quad Gen.bool bound bound (Gen.int_range 0 12))
        (fun (endpoint, start, stop, n) ->
          equal floats
            (Ref.create [| n |] (linear_points ~endpoint start stop n))
            (Ref.of_nx (Nx.linspace ~endpoint Nx.float64 start stop n)));
      prop
        "linspace starts on start and, with its endpoint, ends on stop, bit \
         for bit, its points between them"
        ~examples:
          [
            (F Nx.float64, true, -.Float.max_float, Float.max_float, 3);
            (F Nx.float16, true, 1., 2., 1);
            (F Nx.bfloat16, true, 3., -1., 2);
          ]
        linspace_case linspace_bounds;
      prop "logspace is base to the power of linspace"
        (Gen.quad Gen.bool (Gen.float_range (-3.) 3.) (Gen.float_range (-3.) 3.)
           (Gen.int_range 0 12))
        (fun (endpoint, start, stop, n) ->
          equal floats
            (Ref.create [| n |]
               (Array.map
                  (fun x -> 2. ** x)
                  (linear_points ~endpoint start stop n)))
            (Ref.of_nx (Nx.logspace ~endpoint ~base:2. Nx.float64 start stop n)));
      prop "geomspace is linspace in log space"
        (Gen.quad Gen.bool
           (Gen.float_range 0.01 100.)
           (Gen.float_range 0.01 100.)
           (Gen.int_range 0 12))
        (fun (endpoint, start, stop, n) ->
          equal
            (Ref.witness (float_rel ~rel:1e-9 ~abs:1e-12))
            (Ref.create [| n |]
               (Array.map exp
                  (linear_points ~endpoint (log start) (log stop) n)))
            (Ref.of_nx (Nx.geomspace ~endpoint Nx.float64 start stop n)));
      cases "logspace raises its base, 10 by default, to each point"
        ~name:(fun (name, _, _) -> name)
        [
          ("base 10", [| 1.; 10.; 100. |], Nx.logspace Nx.float64 0. 2. 3);
          ( "base e",
            [| 1.; exp 1.; exp 2. |],
            Nx.logspace ~base:(exp 1.) Nx.float64 0. 2. 3 );
        ]
        (fun (_, expected, t) ->
          equal floats (Ref.create [| 3 |] expected) (Ref.of_nx t));
      test "arange refuses a zero step" (fun () ->
          raises_invalid_arg (fun () -> Nx.arange Nx.int32 0 3 0);
          raises_invalid_arg (fun () -> Nx.arange_f Nx.float64 0. 3. 0.));
      cases "arange holds the last value of its dtype and refuses the next"
        ~name:(fun (d, b, _) -> Format.asprintf "%a %d" pp_held d b)
        held_bounds held_bound;
      test "arange holds 0 and 1 in bool and refuses 2 and -1" (fun () ->
          equal (same ())
            (Nx.create Nx.bool [| 2 |] [| false; true |])
            (Nx.arange Nx.bool 0 2 1);
          equal (same ())
            (Nx.create Nx.bool [| 1 |] [| true |])
            (Nx.arange Nx.bool 1 2 1);
          raises_invalid_arg (fun () -> Nx.arange Nx.bool 0 3 1);
          raises_invalid_arg (fun () -> Nx.arange Nx.bool (-1) 1 1));
      test "an empty arange raises nothing, whatever its bounds" (fun () ->
          equal (array int) [| 0 |] (Nx.shape (Nx.arange Nx.int8 1000 0 1));
          equal (array int) [| 0 |] (Nx.shape (Nx.arange Nx.uint8 (-5) (-10) 1));
          equal (array int) [| 0 |] (Nx.shape (Nx.arange Nx.int4 100 50 1)));
      test "arange spans OCaml's whole int range without overflow" (fun () ->
          let p61 = 1 lsl 61 in
          equal (same ())
            (int64s [ min_int; -p61; 0; p61 ])
            (Nx.arange Nx.int64 min_int max_int p61);
          equal (same ())
            (int64s [ max_int; max_int - p61; max_int - (2 * p61); -p61 - 1 ])
            (Nx.arange Nx.int64 max_int min_int (-p61));
          (* The last partial sum, 3 * step, leaves int64. *)
          equal (same ())
            (int64s [ min_int; -1; max_int - 1 ])
            (Nx.arange Nx.int64 min_int max_int max_int);
          equal (same ())
            (int64s [ max_int; 0; -max_int ])
            (Nx.arange Nx.int64 max_int min_int (-max_int)));
      test "a float arange rounds as cast does, to nearest even" (fun () ->
          equal (same ())
            (Nx.create Nx.float16 [| 8 |]
               [| 2048.; 2048.; 2050.; 2052.; 2052.; 2052.; 2054.; 2056. |])
            (Nx.arange Nx.float16 2048 2056 1));
      test "a complex arange has a zero imaginary part" (fun () ->
          let c re = { Complex.re; im = 0. } in
          equal (same ())
            (Nx.create Nx.complex64 [| 4 |] [| c (-2.); c (-1.); c 0.; c 1. |])
            (Nx.arange Nx.complex64 (-2) 2 1));
      test "linspace and logspace refuse a negative count" (fun () ->
          raises_invalid_arg (fun () -> Nx.linspace Nx.float64 0. 1. (-1));
          raises_invalid_arg (fun () -> Nx.logspace Nx.float64 0. 1. (-1)));
      test "geomspace refuses a bound that is not positive" (fun () ->
          raises_invalid_arg (fun () -> Nx.geomspace Nx.float64 0. 1. 3);
          raises_invalid_arg (fun () -> Nx.geomspace Nx.float64 1. (-1.) 3));
    ]

let grids =
  group "grids"
    [
      prop "meshgrid repeats x along rows and y along columns"
        (Gen.pair dims dims) (fun (n, m) ->
          let x = Nx.arange Nx.int32 0 n 1
          and y = Nx.arange Nx.int32 10 (10 + m) 1 in
          let gx, gy = Nx.meshgrid x y in
          equal ints
            (Ref.init [| m; n |] (fun i -> Int32.of_int i.(1)))
            (Ref.of_nx gx);
          equal ints
            (Ref.init [| m; n |] (fun i -> Int32.of_int (10 + i.(0))))
            (Ref.of_nx gy);
          let gx, gy = Nx.meshgrid ~indexing:`ij x y in
          equal ints
            (Ref.init [| n; m |] (fun i -> Int32.of_int i.(0)))
            (Ref.of_nx gx);
          equal ints
            (Ref.init [| n; m |] (fun i -> Int32.of_int (10 + i.(1))))
            (Ref.of_nx gy));
      prop "one_hot sets the class of each index, and none out of range"
        (Gen.pair (Gen.int_range 1 5)
           (Gen.array ~size:(Gen.int_range 0 6) (Gen.int_range (-2) 6)))
        (fun (num_classes, labels) ->
          let n = Array.length labels in
          equal (Ref.witness int)
            (Ref.init [| n; num_classes |] (fun i ->
                 if labels.(i.(0)) = i.(1) then 1 else 0))
            (Ref.of_nx
               (Nx.one_hot ~num_classes
                  (Nx.create Nx.int32 [| n |] (Array.map Int32.of_int labels)))));
      test "meshgrid refuses a matrix" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.meshgrid
                (Nx.zeros Nx.int32 [| 2; 2 |])
                (Nx.zeros Nx.int32 [| 2 |]));
          raises_invalid_arg (fun () ->
              Nx.meshgrid
                (Nx.zeros Nx.int32 [| 2 |])
                (Nx.zeros Nx.int32 [| 2; 2 |])));
      cases
        "one_hot marks one class per index in range and none out of range, \
         whatever the index dtype"
        ~name:(fun (name, _, _, _) -> name)
        [
          ( "uint8 index 5 of 300 classes",
            300,
            Some 5,
            fun () ->
              Nx.one_hot ~num_classes:300 (Nx.create Nx.uint8 [| 1 |] [| 5 |])
          );
          ( "int8 index -100 of 200 classes",
            200,
            None,
            fun () ->
              Nx.one_hot ~num_classes:200 (Nx.create Nx.int8 [| 1 |] [| -100 |])
          );
        ]
        (fun (_, n, hot, one_hot) ->
          equal (Ref.witness int)
            (Ref.init [| 1; n |] (fun i -> if Some i.(1) = hot then 1 else 0))
            (Ref.of_nx (one_hot ())));
      test "one_hot refuses zero classes and float indices" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.one_hot ~num_classes:0 (Nx.zeros Nx.int32 [| 2 |]));
          raises_invalid_arg (fun () ->
              Nx.one_hot ~num_classes:2 (Nx.zeros Nx.float32 [| 2 |])));
    ]

let () = exit (run "nx creation" [ filled; diagonals; ranges; grids ])
