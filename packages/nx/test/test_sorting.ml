(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sorting and selection, against a stable sort of each lane. *)

open Windtrap
open Nx_test

(* A dtype with the order of its elements: NaN after every number, and -0 before
   +0. *)
type sortable =
  | S : {
      name : string;
      dtype : ('a, 'b) Nx.dtype;
      value : 'a Gen.t;
      compare : 'a -> 'a -> int;
      is_nan : 'a -> bool;
      exact : 'a testable;
      pp : Format.formatter -> 'a -> unit;
    }
      -> sortable

let pp_float ppf x = Format.fprintf ppf "%.17g" x

(* The order of two numbers, -0 before +0. *)
let compare_float a b =
  if a < b then -1
  else if a > b then 1
  else Bool.compare (Float.sign_bit b) (Float.sign_bit a)

let floats name dtype =
  S
    {
      name;
      dtype;
      value =
        Gen.frequency
          [
            (6, Gen.map float_of_int (Gen.int_range (-3) 3));
            (2, Gen.float_range (-1e3) 1e3);
            ( 1,
              Gen.of_list ~pp:pp_float
                [ Float.nan; -0.; 0.; infinity; neg_infinity ] );
          ];
      compare = compare_float;
      is_nan = Float.is_nan;
      exact = float_exact;
      pp = pp_float;
    }

let integers name dtype lo hi =
  S
    {
      name;
      dtype;
      value =
        Gen.frequency
          [ (3, Gen.int_range (Int.max lo (-3)) 3); (1, Gen.int_range lo hi) ];
      compare = Int.compare;
      is_nan = (fun _ -> false);
      exact = int;
      pp = Format.pp_print_int;
    }

let sortables =
  [
    floats "float64" Nx.float64;
    floats "float32" Nx.float32;
    floats "float16" Nx.float16;
    floats "bfloat16" Nx.bfloat16;
    integers "int8" Nx.int8 (-128) 127;
    integers "uint16" Nx.uint16 0 65535;
    S
      {
        name = "int32";
        dtype = Nx.int32;
        value = Gen.map Int32.of_int (Gen.int_range (-3) 3);
        compare = Int32.compare;
        is_nan = (fun _ -> false);
        exact = int32;
        pp = (fun ppf v -> Format.fprintf ppf "%ld" v);
      };
    S
      {
        name = "uint32";
        dtype = Nx.uint32;
        value =
          Gen.frequency
            [ (3, Gen.map Int32.of_int (Gen.int_range 0 3)); (1, Gen.int32) ];
        compare = Int32.unsigned_compare;
        is_nan = (fun _ -> false);
        exact = int32;
        pp = (fun ppf v -> Format.fprintf ppf "%lu" v);
      };
    (* Complex numbers order by real part, then imaginary part, each with -0
       before +0; NaN in either part sorts last. *)
    S
      {
        name = "complex128";
        dtype = Nx.complex128;
        value =
          (let part =
             Gen.frequency
               [
                 (6, Gen.map float_of_int (Gen.int_range (-2) 2));
                 (1, Gen.of_list ~pp:pp_float [ Float.nan; -0. ]);
               ]
           in
           Gen.(
             let+ re = part and+ im = part in
             Complex.{ re; im }));
        compare =
          (fun (a : Complex.t) (b : Complex.t) ->
            match compare_float a.re b.re with
            | 0 -> compare_float a.im b.im
            | k -> k);
        is_nan = (fun (z : Complex.t) -> Float.is_nan z.re || Float.is_nan z.im);
        exact =
          Testable.contramap
            (fun (z : Complex.t) -> (z.re, z.im))
            (pair float_exact float_exact);
        pp =
          (fun ppf (z : Complex.t) -> Format.fprintf ppf "(%g, %g)" z.re z.im);
      };
    S
      {
        name = "uint64";
        dtype = Nx.uint64;
        value =
          Gen.frequency
            [ (3, Gen.map Int64.of_int (Gen.int_range 0 3)); (1, Gen.int64) ];
        compare = Int64.unsigned_compare;
        is_nan = (fun _ -> false);
        exact = int64;
        pp = (fun ppf v -> Format.fprintf ppf "%Lu" v);
      };
  ]

(* The positions of a stable sort of [lane]: NaN last in either direction. *)
let stable_order ~descending compare is_nan lane =
  let key i j =
    match (is_nan lane.(i), is_nan lane.(j)) with
    | true, true -> 0
    | true, false -> 1
    | false, true -> -1
    | false, false ->
        if descending then compare lane.(j) lane.(i)
        else compare lane.(i) lane.(j)
  in
  Array.of_list (List.stable_sort key (List.init (Array.length lane) Fun.id))

let shape = Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range 0 5)

(* Lanes on both sides of the length past which a sort takes one pass per byte
   of the key instead of comparisons: 16 entries per byte, 128 for float64. *)
let long_lanes =
  Gen.map
    (fun (rows, length) -> [| rows; length |])
    (Gen.pair (Gen.int_range 1 3)
       (Gen.of_list ~pp:Format.pp_print_int
          [ 15; 16; 17; 63; 64; 65; 127; 128; 129; 1000 ]))

let sorts_as_a_stable_sort (S s) ~shape =
  let drawn =
    let open Gen in
    let* t = viewed ~shape ~pp:s.pp s.dtype s.value in
    let+ axis = int_range (-Nx.ndim t) (Nx.ndim t - 1) and+ descending = bool in
    (t, axis, descending)
  in
  fun name ->
    prop name drawn (fun (t, axis, descending) ->
        let r = Ref.of_nx t in
        let a = Ref.axis r axis in
        let n = r.shape.(a) in
        let order =
          Ref.along ~axis:a ~length:n
            (stable_order ~descending s.compare s.is_nan)
            r
        in
        let values =
          Ref.along ~axis:a ~length:n
            (fun lane ->
              Array.map
                (fun i -> lane.(i))
                (stable_order ~descending s.compare s.is_nan lane))
            r
        in
        let v, i = Nx.sort ~descending ~axis t in
        equal (Ref.witness s.exact) values (Ref.of_nx v);
        equal (Ref.witness int64) (Ref.map Int64.of_int order) (Ref.of_nx i);
        equal (tensor int64) i (Nx.argsort ~descending ~axis t))

let sorts =
  group "sort"
    (List.concat_map
       (fun (S s as sortable) ->
         [
           sorts_as_a_stable_sort sortable ~shape
             (s.name
            ^ " sorts each lane stably, NaN last, and returns the positions");
           sorts_as_a_stable_sort sortable ~shape:long_lanes
             (s.name ^ " sorts long lanes as a stable sort does");
         ])
       sortables
    @ [
        test "-0 sorts before +0, and after it descending" (fun () ->
            let t = Nx.create Nx.float32 [| 4 |] [| 0.; -0.; 0.; -0. |] in
            let check ~descending values indices =
              let v, i = Nx.sort ~descending t in
              equal ~msg:"values" (array float_exact) values (Nx.to_array v);
              equal ~msg:"indices" (array int64) indices (Nx.to_array i)
            in
            check ~descending:false [| -0.; -0.; 0.; 0. |] [| 1L; 3L; 0L; 2L |];
            check ~descending:true [| 0.; 0.; -0.; -0. |] [| 0L; 2L; 1L; 3L |]);
        test "sort refuses an axis out of bounds" (fun () ->
            raises_invalid_arg (fun () ->
                Nx.sort ~axis:1 (Nx.zeros Nx.float32 [| 3 |])));
        test "sorting 4096 rows of 128 float64 values sorts each row" (fun () ->
            let rows = 4096 and cols = 128 in
            let xs = Array.init (rows * cols) (fun _ -> Random.float 1.) in
            let t = Nx.create Nx.float64 [| rows; cols |] xs in
            let v, i = Nx.sort t in
            let expected =
              Array.concat
                (List.init rows (fun r ->
                     let row = Array.sub xs (r * cols) cols in
                     Array.stable_sort Float.compare row;
                     row))
            in
            equal (array float_exact) expected (Nx.to_array v);
            equal (tensor float_exact) v
              (Nx.take_along_axis ~axis:1 ~indices:i t);
            equal (tensor int64) i (Nx.argsort t));
      ])

(* Lanes on both sides of the lengths where top_k changes method: 8 passes, a
   sort up to 2048 entries, a radix select past it. *)
let top_ks =
  let lane_length =
    Gen.frequency
      [
        (4, Gen.int_range 1 20);
        (2, Gen.int_range 2040 2060);
        (1, Gen.int_range 3000 5000);
      ]
  in
  group "top_k"
    (List.map
       (fun (S s) ->
         let drawn =
           let open Gen in
           let* n = lane_length in
           let* rows = int_range 1 3 in
           let* t =
             viewed
               ~shape:(constant [| rows; n |])
               ~layout:(constant ~pp:pp_layout [])
               ~pp:s.pp s.dtype s.value
           in
           let+ k = int_range 1 (Int.min n 40) in
           (t, k)
         in
         prop (s.name ^ " top_k is the first k of a descending sort") drawn
           (fun (t, k) ->
             cover "a radix select" (Nx.dim 1 t > 2048);
             cover "more than eight" (k > 8);
             let v, i = Nx.sort ~descending:true ~axis:1 t in
             let tv, ti = Nx.top_k ~k ~axis:1 t in
             equal ~msg:"values" (tensor s.exact)
               (Nx.slice [ A; R (0, k) ] v)
               tv;
             equal ~msg:"indices" (tensor int64) (Nx.slice [ A; R (0, k) ] i) ti))
       (List.filter
          (fun (S s) -> List.mem s.name [ "float32"; "int32"; "uint16" ])
          sortables)
    @ [
        test "top_k puts +0 before -0 and agrees with argmax, on each path"
          (fun () ->
            List.iter
              (fun (k, n) ->
                let xs = Array.make n (-1.) in
                xs.(1) <- -0.;
                xs.(3) <- 0.;
                let t = Nx.create Nx.float32 [| n |] xs in
                let msg = Printf.sprintf "k = %d of %d" k n in
                let v, i = Nx.top_k ~k t in
                equal ~msg (array float_exact) [| 0.; -0. |]
                  (Array.sub (Nx.to_array v) 0 2);
                equal ~msg (array int64) [| 3L; 1L |]
                  (Array.sub (Nx.to_array i) 0 2);
                equal ~msg int64
                  (Nx.item [] (Nx.argmax t))
                  (Nx.item [ 0 ] (snd (Nx.top_k ~k:1 t))))
              [ (2, 5); (9, 100); (9, 3000) ]);
        test
          "top_k refuses a scalar, an axis out of bounds and a k past the axis"
          (fun () ->
            raises_invalid_arg (fun () ->
                Nx.top_k ~k:1 (Nx.scalar Nx.float32 1.));
            raises_invalid_arg (fun () ->
                Nx.top_k ~k:1 ~axis:1 (Nx.zeros Nx.float32 [| 3 |]));
            raises_invalid_arg (fun () ->
                Nx.top_k ~k:4 (Nx.zeros Nx.float32 [| 3 |])));
      ])

let gathers =
  group "gathering by position"
    [
      prop "take_along_axis of the positions gives the sorted values"
        (viewed ~shape ~pp:pp_float Nx.float64 (Gen.float_range (-10.) 10.))
        (fun t ->
          let a = Nx.ndim t - 1 in
          let v, i = Nx.sort ~axis:a t in
          equal (tensor float_exact) v (Nx.take_along_axis ~axis:a ~indices:i t));
    ]

let () = exit (run "nx sorting" [ sorts; top_ks; gathers ])
