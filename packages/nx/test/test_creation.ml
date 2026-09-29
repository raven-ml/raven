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

let bound = Gen.float_range (-100.) 100.

let ranges =
  group "ranges"
    [
      prop "arange counts from start by step while short of stop"
        (Gen.triple (Gen.int_range (-20) 20) (Gen.int_range (-20) 20)
           (Gen.one_of [ Gen.int_range (-7) (-1); Gen.int_range 1 7 ]))
        (fun (start, stop, step) ->
          let l =
            Array.of_list
              (List.map Int32.of_int (arithmetic_progression start stop step))
          in
          equal ints
            (Ref.create [| Array.length l |] l)
            (Ref.of_nx (Nx.arange Nx.int32 start stop step)));
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
