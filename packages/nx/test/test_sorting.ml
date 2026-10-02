(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sorting and selection, against a stable sort of each lane in the sort order:
   NaN above every number, and -0 below +0; descending is its exact reverse. *)

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
      numeric : 'a -> 'a -> int;
          (** The order of two numbers as [less] takes it: -0 equals +0. *)
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

(* Floats with ties, signed zeros, infinities and NaN. *)
let float_values =
  Gen.frequency
    [
      (6, Gen.map float_of_int (Gen.int_range (-3) 3));
      (2, Gen.float_range (-1e3) 1e3);
      ( 1,
        Gen.of_list ~pp:pp_float [ Float.nan; -0.; 0.; infinity; neg_infinity ]
      );
    ]

let floats name dtype =
  S
    {
      name;
      dtype;
      value = float_values;
      compare = compare_float;
      numeric = Float.compare;
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
      numeric = Int.compare;
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
    integers "int4" Nx.int4 (-8) 7;
    integers "uint4" Nx.uint4 0 15;
    S
      {
        name = "bool";
        dtype = Nx.bool;
        value = Gen.bool;
        compare = Bool.compare;
        numeric = Bool.compare;
        is_nan = (fun _ -> false);
        exact = bool;
        pp = Format.pp_print_bool;
      };
    integers "int8" Nx.int8 (-128) 127;
    integers "uint16" Nx.uint16 0 65535;
    S
      {
        name = "int32";
        dtype = Nx.int32;
        value = Gen.map Int32.of_int (Gen.int_range (-3) 3);
        compare = Int32.compare;
        numeric = Int32.compare;
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
        numeric = Int32.unsigned_compare;
        is_nan = (fun _ -> false);
        exact = int32;
        pp = (fun ppf v -> Format.fprintf ppf "%lu" v);
      };
    (* Complex numbers order by real part, then imaginary part, each with -0
       before +0; one with a NaN part ranks as a NaN. *)
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
        numeric = (fun _ _ -> invalid_arg "complex numbers are not ordered");
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
        numeric = Int64.unsigned_compare;
        is_nan = (fun _ -> false);
        exact = int64;
        pp = (fun ppf v -> Format.fprintf ppf "%Lu" v);
      };
  ]

(* The order of [compare] with every NaN equal to every other and above every
   number. *)
let order compare is_nan a b =
  match (is_nan a, is_nan b) with
  | true, true -> 0
  | true, false -> 1
  | false, true -> -1
  | false, false -> compare a b

(* The positions of a stable sort of [lane] in the sort order, every NaN equal
   to every other and above every number, or in its reverse. *)
let stable_order ~descending compare is_nan lane =
  let order = order compare is_nan in
  let key i j =
    if descending then order lane.(j) lane.(i) else order lane.(i) lane.(j)
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

(* A tensor of [dtype] and one of its axes. *)
let with_axis ~pp dtype value =
  let open Gen in
  let* t = viewed ~shape ~pp dtype value in
  let+ axis = int_range 0 (Nx.ndim t - 1) in
  (t, axis)

(* A descending sort is the exact reverse of the ascending one: the values
   reversed, NaN payloads aside (two NaNs tie, and each direction keeps them in
   input order). Complex numbers are left out: two with NaN in different parts
   tie but differ as values. *)
let reverses (S s) =
  prop (s.name ^ " a descending sort is the ascending sort reversed")
    (with_axis ~pp:s.pp s.dtype s.value) (fun (t, axis) ->
      equal (tensor s.exact)
        (Nx.flip ~axes:[ axis ] (fst (Nx.sort ~axis t)))
        (fst (Nx.sort ~descending:true ~axis t)))

(* A descending integer argsort is the ascending argsort of the complement,
   which reverses the order of an integer of either signedness. *)
let complements (S s) =
  prop
    (s.name ^ " a descending argsort is the ascending argsort of the complement")
    (with_axis ~pp:s.pp s.dtype s.value) (fun (t, axis) ->
      equal (tensor int64)
        (Nx.argsort ~axis (Nx.bitwise_not t))
        (Nx.argsort ~descending:true ~axis t))

let sorts =
  group "sort"
    (List.concat_map
       (fun (S s as sortable) ->
         [
           sorts_as_a_stable_sort sortable ~shape
             (s.name
            ^ " sorts each lane stably in the sort order or its reverse, and \
               returns the positions");
           sorts_as_a_stable_sort sortable ~shape:long_lanes
             (s.name ^ " sorts long lanes as a stable sort does");
         ])
       sortables
    @ List.filter_map
        (fun (S s as sortable) ->
          if s.name = "complex128" then None else Some (reverses sortable))
        sortables
    @ List.filter_map
        (fun (S s as sortable) ->
          if List.mem s.name [ "int8"; "uint16"; "int32"; "uint32"; "uint64" ]
          then Some (complements sortable)
          else None)
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
        test
          "NaN sorts last ascending and first descending, keeping its input \
           order and bits" (fun () ->
            let t =
              Nx.bitcast Nx.float32
                (Nx.create Nx.int32 [| 5 |]
                   [|
                     Int32.bits_of_float 1.;
                     0x7fc00001l;
                     Int32.bits_of_float (-0.);
                     0xffc00002l;
                     0l;
                   |])
            in
            let check ~descending values indices =
              let v, i = Nx.sort ~descending t in
              equal ~msg:"bits" (array int32) values
                (Nx.to_array (Nx.bitcast Nx.int32 v));
              equal ~msg:"indices" (array int64) indices (Nx.to_array i)
            in
            let one = Int32.bits_of_float 1. and neg_zero = Int32.min_int in
            check ~descending:false
              [| neg_zero; 0l; one; 0x7fc00001l; 0xffc00002l |]
              [| 2L; 4L; 0L; 1L; 3L |];
            check ~descending:true
              [| 0x7fc00001l; 0xffc00002l; one; 0l; neg_zero |]
              [| 1L; 3L; 0L; 4L; 2L |]);
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

(* Lanes on both sides of the lengths where top_k changes method: counting ranks
   up to 32 entries, then 8 passes or a sort up to 2048 entries, a radix select
   past it. *)
let top_ks =
  let lane_length =
    Gen.frequency
      [
        (4, Gen.int_range 1 40);
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
             cover "ranks counted" (Nx.dim 1 t <= 32);
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
                equal ~msg int64 (Nx.item [] (Nx.argmax t)) (Nx.item [ 0 ] i))
              [ (2, 5); (5, 5); (2, 100); (9, 100); (9, 3000) ]);
        test "top_k puts NaN first and agrees with argmax, on each path"
          (fun () ->
            List.iter
              (fun (k, n) ->
                let xs = Array.make n (-1.) in
                xs.(1) <- Float.nan;
                xs.(2) <- 0.;
                xs.(3) <- Float.nan;
                let t = Nx.create Nx.float32 [| n |] xs in
                let msg = Printf.sprintf "k = %d of %d" k n in
                let v, i = Nx.top_k ~k t in
                equal ~msg (array float_exact)
                  [| Float.nan; Float.nan; 0. |]
                  (Array.sub (Nx.to_array v) 0 3);
                equal ~msg (array int64) [| 1L; 3L; 2L |]
                  (Array.sub (Nx.to_array i) 0 3);
                equal ~msg int64 (Nx.item [] (Nx.argmax t)) (Nx.item [ 0 ] i))
              [ (3, 5); (5, 5); (3, 100); (9, 100); (9, 3000) ]);
        test "top_k takes tied entries in their positions' order, on each path"
          (fun () ->
            List.iter
              (fun (k, n) ->
                let t =
                  Nx.init Nx.float32 [| 2; n |] (fun i ->
                      Float.of_int (i.(1) * 7 mod 3))
                in
                let msg = Printf.sprintf "k = %d of %d" k n in
                let v, i = Nx.sort ~descending:true ~axis:1 t in
                let tv, ti = Nx.top_k ~k ~axis:1 t in
                equal ~msg (tensor float_exact) (Nx.slice [ A; R (0, k) ] v) tv;
                equal ~msg (tensor int64) (Nx.slice [ A; R (0, k) ] i) ti)
              [ (1, 6); (3, 6); (6, 6); (1, 100); (4, 100); (9, 100) ]);
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

(* Keys *)

let ordered = List.filter (fun (S s) -> s.name <> "complex128") sortables
let sign n = Int.compare n 0

(* Layouts that keep a tensor's rank. *)
let same_rank =
  let keeps l =
    List.mem l.name
      [
        "transposed";
        "flipped";
        "every other row";
        "without its first row";
        "without its last row";
        "every other column";
      ]
  in
  Gen.with_pp pp_layout
    (Gen.list ~size:(Gen.int_range 0 2)
       (Gen.of_list (List.filter keeps layout_steps)))

(* Layouts that keep a tensor's columns. *)
let same_columns =
  Gen.with_pp pp_layout
    (Gen.list ~size:(Gen.int_range 0 2)
       (Gen.of_list
          (List.filter
             (fun l ->
               List.mem l.name
                 [
                   "flipped";
                   "every other row";
                   "without its first row";
                   "without its last row";
                 ])
             layout_steps)))

(* [keys_of ~pp dtype value n w] draws a vector of [dtype], or a matrix when [w]
   is [Some _], of about [n] rows and [w] columns, laid out by [layout]. *)
let keys_of ?(layout = same_rank) ~pp dtype value n w =
  let shape = match w with None -> [| n |] | Some w -> [| n; w |] in
  viewed ~shape:(Gen.constant shape) ~layout ~pp dtype value

(* The rows of [keys], one element each for a vector. *)
let rows_of keys =
  let r = Ref.of_nx keys in
  let n = r.shape.(0) in
  let w = if Array.length r.shape = 1 then 1 else r.shape.(1) in
  Array.init n (fun i -> Array.sub r.data (i * w) w)

let lexicographic cmp a b =
  let rec go j =
    if j = Array.length a then 0
    else match cmp a.(j) b.(j) with 0 -> go (j + 1) | c -> c
  in
  go 0

(* The unsigned dtypes keys take. *)
type width = W : ('a, 'b) Nx.dtype -> width

let widths = [ W Nx.uint8; W Nx.uint16; W Nx.uint32; W Nx.uint64 ]

let order_keys =
  let sorts_as_elements (S s) (W kd) =
    prop
      (Printf.sprintf "%s order keys of %s compare as their elements do"
         (Nx_dtype.to_string kd) s.name)
      (Gen.bind (Gen.int_range 0 8) (fun n ->
           keys_of ~pp:s.pp s.dtype s.value n None))
      (fun t ->
        let x = Nx.to_array t in
        let k = Nx.to_array (Nx.cast Nx.uint64 (Nx.order_key kd t)) in
        let n = Array.length x in
        for i = 0 to n - 1 do
          for j = 0 to n - 1 do
            equal
              ~msg:(Printf.sprintf "elements %d and %d" i j)
              int
              (sign (order s.compare s.is_nan x.(i) x.(j)))
              (sign (Int64.unsigned_compare k.(i) k.(j)))
          done
        done)
  in
  let at_every_width (S s) =
    List.filter_map
      (fun (W kd) ->
        if Nx_dtype.itemsize kd >= Nx_dtype.itemsize s.dtype then
          Some (sorts_as_elements (S s) (W kd))
        else None)
      widths
  in
  (* An exact cast within a family, to a dtype no wider than the key. *)
  let keeps (type a b c d e f) name (narrow : (a, b) Nx.dtype)
      (wide : (c, d) Nx.dtype) (kd : (e, f) Nx.dtype) values =
    prop
      (Printf.sprintf "an exact cast from %s to %s keeps the %s key" name
         (Nx_dtype.to_string wide) (Nx_dtype.to_string kd))
      (Gen.array ~size:(Gen.int_range 0 8) (Gen.of_list ~pp:pp_float values))
      (fun v ->
        let t = Nx.cast narrow (Nx.create Nx.float64 [| Array.length v |] v) in
        let key t = Nx.cast Nx.uint64 (Nx.order_key kd t) in
        equal (tensor int64) (key t) (key (Nx.cast wide t)))
  in
  let signed = [ -128.; -8.; -1.; 0.; 1.; 7.; 127. ]
  and unsigned = [ 0.; 1.; 15.; 255. ]
  and floats =
    [ neg_infinity; -448.; -1.5; -0.; 0.; 0.25; 448.; infinity; Float.nan ]
  in
  let keys_of dtype bits =
    Nx.to_array
      (Nx.order_key Nx.uint64 (Nx.create dtype [| Array.length bits |] bits))
  in
  let narrow_keys kd dtype bits =
    Nx.to_array (Nx.order_key kd (Nx.create dtype [| Array.length bits |] bits))
  in
  group "order keys"
    (List.concat_map at_every_width ordered
    @ [
        keeps "int4" Nx.int4 Nx.int64 Nx.uint64 [ -8.; -1.; 0.; 7. ];
        keeps "int8" Nx.int8 Nx.int64 Nx.uint64 signed;
        keeps "int16" Nx.int16 Nx.int64 Nx.uint64 (-32768. :: signed);
        keeps "int32" Nx.int32 Nx.int64 Nx.uint64
          (-2147483648. :: 2147483647. :: signed);
        keeps "uint4" Nx.uint4 Nx.uint64 Nx.uint64 [ 0.; 1.; 15. ];
        keeps "uint8" Nx.uint8 Nx.uint64 Nx.uint64 unsigned;
        keeps "uint16" Nx.uint16 Nx.uint64 Nx.uint64 (65535. :: unsigned);
        keeps "uint32" Nx.uint32 Nx.uint64 Nx.uint64 (4294967295. :: unsigned);
        keeps "float8_e4m3" Nx.float8_e4m3 Nx.float64 Nx.uint64 floats;
        keeps "float8_e5m2" Nx.float8_e5m2 Nx.float64 Nx.uint64 floats;
        keeps "float16" Nx.float16 Nx.float64 Nx.uint64 floats;
        keeps "bfloat16" Nx.bfloat16 Nx.float64 Nx.uint64 floats;
        keeps "float32" Nx.float32 Nx.float64 Nx.uint64 floats;
        keeps "int4" Nx.int4 Nx.int8 Nx.uint8 [ -8.; -1.; 0.; 7. ];
        keeps "int8" Nx.int8 Nx.int16 Nx.uint16 signed;
        keeps "int16" Nx.int16 Nx.int32 Nx.uint32 (-32768. :: signed);
        keeps "uint8" Nx.uint8 Nx.uint16 Nx.uint16 unsigned;
        keeps "float8_e4m3" Nx.float8_e4m3 Nx.float16 Nx.uint16 floats;
        keeps "float8_e5m2" Nx.float8_e5m2 Nx.float16 Nx.uint16 floats;
        keeps "float16" Nx.float16 Nx.float32 Nx.uint32 floats;
        keeps "bfloat16" Nx.bfloat16 Nx.float32 Nx.uint32 floats;
        test
          "the order key of a float flips its bits by its sign, and of a NaN \
           is all ones" (fun () ->
            let nan_bits = Int64.bits_of_float Float.nan in
            let x =
              Nx.bitcast Nx.float64
                (Nx.create Nx.int64 [| 9 |]
                   (Array.append
                      (Array.map Int64.bits_of_float
                         [| neg_infinity; -1.; -0.; 0.; 1.; infinity |])
                      [|
                        nan_bits;
                        Int64.logor nan_bits Int64.min_int;
                        0x7ff0000000000001L;
                      |]))
            in
            equal (array string)
              [|
                "000fffffffffffff";
                "400fffffffffffff";
                "7fffffffffffffff";
                "8000000000000000";
                "bff0000000000000";
                "fff0000000000000";
                "ffffffffffffffff";
                "ffffffffffffffff";
                "ffffffffffffffff";
              |]
              (Array.map (Printf.sprintf "%016Lx")
                 (Nx.to_array (Nx.order_key Nx.uint64 x))));
        test
          "the order key of an integer is its value, signed ones with the sign \
           bit flipped" (fun () ->
            equal (array int64)
              [| 0L; -1L; Int64.min_int |]
              (keys_of Nx.int64 [| Int64.min_int; Int64.max_int; 0L |]);
            equal (array int64) [| -1L; 5L |] (keys_of Nx.uint64 [| -1L; 5L |]);
            equal (array int64) [| 0L; 1L |] (keys_of Nx.bool [| false; true |]);
            equal (array int64)
              [| 0x7ffffffffffffff8L; 0x8000000000000007L |]
              (keys_of Nx.int4 [| -8; 7 |]));
        test "a key at an element's own width applies the rule at that width"
          (fun () ->
            equal (array int) [| 0; 0x7f; 0x80; 0xff |]
              (narrow_keys Nx.uint8 Nx.int8 [| -128; -1; 0; 127 |]);
            equal (array int) [| 0x78; 0x80; 0x87 |]
              (narrow_keys Nx.uint8 Nx.int4 [| -8; 0; 7 |]);
            equal (array int)
              [| 0x03ff; 0x7fff; 0x8000; 0xfc00; 0xffff |]
              (narrow_keys Nx.uint16 Nx.float16
                 [| neg_infinity; -0.; 0.; infinity; nan |]);
            equal (array int) [| 0x7f; 0x80; 0xff |]
              (narrow_keys Nx.uint8 Nx.float8_e5m2 [| -0.; 0.; nan |]));
        test "a float8's 8-bit keys order every bit pattern as its 64-bit keys"
          (fun () ->
            let every (type b) (fp8 : (float, b) Nx.dtype) =
              let x = Nx.bitcast fp8 (Nx.arange Nx.uint8 0 256 1) in
              let k8 = Nx.to_array (Nx.order_key Nx.uint8 x) in
              let k64 = Nx.to_array (Nx.order_key Nx.uint64 x) in
              for i = 0 to 255 do
                for j = 0 to 255 do
                  equal
                    ~msg:
                      (Printf.sprintf "%s bits 0x%02x and 0x%02x"
                         (Nx_dtype.to_string fp8) i j)
                    int
                    (sign (Int.compare k8.(i) k8.(j)))
                    (sign (Int64.unsigned_compare k64.(i) k64.(j)))
                done
              done
            in
            every Nx.float8_e4m3;
            every Nx.float8_e5m2);
        test "order_key refuses complex numbers" (fun () ->
            raises_invalid_arg (fun () ->
                Nx.order_key Nx.uint64 (Nx.zeros Nx.complex64 [| 1 |])));
        test
          "order_key refuses a key narrower than the elements, or not unsigned"
          (fun () ->
            raises_invalid_arg (fun () ->
                Nx.order_key Nx.uint8 (Nx.zeros Nx.int16 [| 1 |]));
            raises_invalid_arg (fun () ->
                Nx.order_key Nx.uint32 (Nx.zeros Nx.float64 [| 1 |]));
            raises_invalid_arg (fun () ->
                Nx.order_key Nx.int64 (Nx.zeros Nx.int8 [| 1 |]));
            raises_invalid_arg (fun () ->
                Nx.order_key Nx.uint4 (Nx.zeros Nx.bool [| 1 |])));
      ])

let lexsorts =
  let stable (S s) =
    prop
      (s.name ^ " lexsort is a stable sort of the rows")
      (let open Gen in
       let* n = int_range 0 9 in
       let* w = option (int_range 0 3) in
       keys_of ~pp:s.pp s.dtype s.value n w)
      (fun keys ->
        let rows = rows_of keys in
        let expected =
          List.stable_sort
            (fun i j ->
              lexicographic (order s.compare s.is_nan) rows.(i) rows.(j))
            (List.init (Array.length rows) Fun.id)
        in
        equal (array int64)
          (Array.of_list (List.map Int64.of_int expected))
          (Nx.to_array (Nx.lexsort keys)))
  in
  let descending (S s) =
    prop
      (s.name
     ^ " a descending argsort is the lexsort of the complemented order keys")
      (Gen.bind (Gen.int_range 0 9) (fun n ->
           keys_of ~pp:s.pp s.dtype s.value n None))
      (fun t ->
        equal (tensor int64)
          (Nx.argsort ~descending:true t)
          (Nx.lexsort (Nx.bitwise_not (Nx.order_key Nx.uint64 t))))
  in
  let two_keys =
    let open Gen in
    let* n = int_range 0 12 in
    let+ a = array ~size:(constant n) (int_range (-2) 2)
    and+ b =
      array ~size:(constant n)
        (of_list ~pp:pp_float [ -1.; -0.; 0.; 2.; Float.nan; Float.nan ])
    and+ down = bool in
    (a, b, down)
  in
  group "lexsort"
    (List.map stable ordered
    @ List.map descending ordered
    @ [
        prop
          "rows of order keys sort as their columns do, a complemented one \
           descending"
          two_keys (fun (a, b, down) ->
            let n = Array.length a in
            let kb = Nx.order_key Nx.uint64 (Nx.create Nx.float64 [| n |] b) in
            let keys =
              Nx.stack ~axis:1
                [
                  Nx.order_key Nx.uint64
                    (Nx.create Nx.int64 [| n |] (Array.map Int64.of_int a));
                  (if down then Nx.bitwise_not kb else kb);
                ]
            in
            let by_b i j =
              let c = order compare_float Float.is_nan b.(i) b.(j) in
              if down then -c else c
            in
            let expected =
              List.stable_sort
                (fun i j ->
                  match Int.compare a.(i) a.(j) with 0 -> by_b i j | c -> c)
                (List.init n Fun.id)
            in
            equal (array int64)
              (Array.of_list (List.map Int64.of_int expected))
              (Nx.to_array (Nx.lexsort keys)));
        test "lexsort refuses a scalar, a 3-D tensor and complex numbers"
          (fun () ->
            raises_invalid_arg (fun () -> Nx.lexsort (Nx.scalar Nx.int32 1l));
            raises_invalid_arg (fun () ->
                Nx.lexsort (Nx.zeros Nx.int32 [| 1; 1; 1 |]));
            raises_invalid_arg (fun () ->
                Nx.lexsort (Nx.zeros Nx.complex64 [| 2 |])));
      ])

(* The number of keys of [s] before [q] ([`Left]) or at or before it, by
   [cmp]. *)
let counted cmp side s q =
  Array.fold_left
    (fun n k ->
      let c = cmp k q in
      if c < 0 || (c = 0 && side = `Right) then n + 1 else n)
    0 s

let searchsorts =
  let side =
    Gen.of_list
      ~pp:(fun ppf s ->
        Format.pp_print_string ppf
          (match s with `Left -> "Left" | `Right -> "Right"))
      [ `Left; `Right ]
  in
  let counts (S s) =
    prop
      (s.name ^ " searchsorted counts the keys before each query")
      (let open Gen in
       let* m = int_range 0 20 in
       let* rank = int_range 0 2 in
       let* shape = array ~size:(constant rank) (int_range 0 3) in
       let+ s = keys_of ~pp:s.pp s.dtype s.value m None
       and+ v = viewed ~shape:(constant shape) ~pp:s.pp s.dtype s.value
       and+ side = side in
       (fst (Nx.sort s), v, side))
      (fun (sorted, v, side) ->
        let cmp = order s.numeric s.is_nan in
        let keys = Nx.to_array sorted in
        equal (tensor int64)
          (Nx.create Nx.int64 (Nx.shape v)
             (Array.map
                (fun q -> Int64.of_int (counted cmp side keys q))
                (Nx.to_array v)))
          (Nx.searchsorted ~side sorted v))
  in
  let rows (S s) =
    prop
      (s.name ^ " searchsorted counts the rows before each row")
      (let open Gen in
       let* m = int_range 0 12 in
       let* w = int_range 0 3 in
       let layout = same_columns in
       let* keys = keys_of ~layout ~pp:s.pp s.dtype s.value m (Some w) in
       let* fresh = keys_of ~layout ~pp:s.pp s.dtype s.value 4 (Some w) in
       let* copied =
         array ~size:(int_range 0 4) (int_range 0 (Int.max 0 (m - 1)))
       in
       let+ side = side in
       (keys, fresh, copied, side))
      (fun (keys, fresh, copied, side) ->
        let cmp = lexicographic (order s.numeric s.is_nan) in
        (* In order as searchsorted compares rows: -0 equal to 0. *)
        let sorted =
          let rows = rows_of keys in
          let perm =
            List.stable_sort
              (fun i j -> cmp rows.(i) rows.(j))
              (List.init (Array.length rows) Fun.id)
          in
          Nx.take ~axis:0
            ~indices:
              (Nx.create Nx.int64
                 [| List.length perm |]
                 (Array.of_list (List.map Int64.of_int perm)))
            keys
        in
        let copied = if Nx.dim 0 keys = 0 then [||] else copied in
        let v =
          Nx.concatenate ~axis:0
            [
              fresh;
              Nx.take ~axis:0
                ~indices:
                  (Nx.create Nx.int64
                     [| Array.length copied |]
                     (Array.map Int64.of_int copied))
                sorted;
            ]
        in
        let table = rows_of sorted in
        equal (tensor int64)
          (Nx.create Nx.int64
             [| Nx.dim 0 v |]
             (Array.map
                (fun q -> Int64.of_int (counted cmp side table q))
                (rows_of v)))
          (Nx.searchsorted ~side sorted v))
  in
  let nan = Float.nan in
  let knots = Nx.create Nx.float64 [| 4 |] [| 0.; 1.; 1.; 2. |] in
  group "searchsorted"
    (List.map counts ordered
    @ List.filter_map
        (fun (S s as sortable) ->
          if List.mem s.name [ "float64"; "int32"; "uint64" ] then
            Some (rows sortable)
          else None)
        ordered
    @ [
        prop "on an unsorted s each result is a position in [0, m]"
          (let open Gen in
           let* m = int_range 0 9 in
           let+ s = array ~size:(constant m) (int_range (-3) 3)
           and+ v = array ~size:(int_range 0 6) (int_range (-4) 4)
           and+ side = side in
           (s, v, side))
          (fun (s, v, side) ->
            let r =
              Nx.searchsorted ~side
                (Nx.create Nx.int32
                   [| Array.length s |]
                   (Array.map Int32.of_int s))
                (Nx.create Nx.int32
                   [| Array.length v |]
                   (Array.map Int32.of_int v))
            in
            Array.iter
              (fun p ->
                at_least int64 ~than:0L p;
                at_most int64 ~than:(Int64.of_int (Array.length s)) p)
              (Nx.to_array r));
        test "-0 and 0 are one number, and a NaN follows every number"
          (fun () ->
            let x = Nx.create Nx.float64 [| 4 |] [| -0.; 1.; 1.5; nan |] in
            equal (array int64) [| 0L; 1L; 3L; 4L |]
              (Nx.to_array (Nx.searchsorted ~side:`Left knots x));
            equal (array int64) [| 1L; 3L; 3L; 4L |]
              (Nx.to_array (Nx.searchsorted ~side:`Right knots x)));
        test
          "a query of all-ones keys counts every key under Right, past none of \
           the padding" (fun () ->
            List.iter
              (fun m ->
                let s = Nx.full Nx.uint64 [| m |] (-1L)
                and q = Nx.full Nx.uint64 [| 1 |] (-1L) in
                equal
                  ~msg:(Printf.sprintf "m = %d" m)
                  (array int64)
                  [| Int64.of_int m |]
                  (Nx.to_array (Nx.searchsorted ~side:`Right s q));
                equal
                  ~msg:(Printf.sprintf "m = %d" m)
                  (array int64) [| 0L |]
                  (Nx.to_array (Nx.searchsorted ~side:`Left s q)))
              [ 1; 2; 3; 4; 7; 8; 9 ]);
        test "rows of order keys compare as given, -0's key below 0's"
          (fun () ->
            let key x =
              Nx.reshape [| 1; 1 |]
                (Nx.order_key Nx.uint64 (Nx.create Nx.float64 [| 1 |] [| x |]))
            in
            equal (array int64) [| 0L |]
              (Nx.to_array (Nx.searchsorted ~side:`Left (key 0.) (key (-0.))));
            equal (array int64) [| 0L |]
              (Nx.to_array (Nx.searchsorted ~side:`Left (key 0.) (key 0.)));
            equal (array int64) [| 1L |]
              (Nx.to_array (Nx.searchsorted ~side:`Right (key 0.) (key 0.))));
        test "rows without a column all equal each query" (fun () ->
            let s = Nx.zeros Nx.int32 [| 3; 0 |]
            and v = Nx.zeros Nx.int32 [| 2; 0 |] in
            equal (array int64) [| 0L; 0L |]
              (Nx.to_array (Nx.searchsorted ~side:`Left s v));
            equal (array int64) [| 3L; 3L |]
              (Nx.to_array (Nx.searchsorted ~side:`Right s v)));
        test "searchsorted refuses mismatched rows, a 3-D s and complex numbers"
          (fun () ->
            raises_invalid_arg (fun () ->
                Nx.searchsorted ~side:`Left
                  (Nx.zeros Nx.int32 [| 2; 2 |])
                  (Nx.zeros Nx.int32 [| 2; 3 |]));
            raises_invalid_arg (fun () ->
                Nx.searchsorted ~side:`Left
                  (Nx.zeros Nx.int32 [| 2; 2 |])
                  (Nx.zeros Nx.int32 [| 2 |]));
            raises_invalid_arg (fun () ->
                Nx.searchsorted ~side:`Left
                  (Nx.zeros Nx.int32 [| 1; 1; 1 |])
                  (Nx.zeros Nx.int32 [| 1 |]));
            raises_invalid_arg (fun () ->
                Nx.searchsorted ~side:`Left
                  (Nx.zeros Nx.complex64 [| 2 |])
                  (Nx.zeros Nx.complex64 [| 1 |])));
      ])

(* Grouping *)

let group_rows x = Nx.Op.eval (Group { by = "test_sorting"; x })

(* [rows] numbered in order of first appearance. *)
let first_appearance rows =
  let seen = Hashtbl.create 16 in
  Array.map
    (fun r ->
      match Hashtbl.find_opt seen r with
      | Some id -> id
      | None ->
          let id = Int64.of_int (Hashtbl.length seen) in
          Hashtbl.add seen r id;
          id)
    rows

let pp_word ppf w = Format.fprintf ppf "%Lu" w

(* Words among a few, so that rows repeat: the extremes and their neighbours. *)
let tied_words = Gen.of_list ~pp:pp_word [ 0L; 1L; 2L; -1L; Int64.min_int ]

(* Rows past one block of 2^16 rows, so that the blocks' groups merge: [n] rows
   of [w] words drawn by [seed] among [distinct] rows, then laid out. *)
type many = {
  n : int;
  w : int;
  distinct : int;
  layout : [ `Contiguous | `Flipped | `Every_other_row ];
  seed : int;
}

let pp_many ppf m =
  Format.fprintf ppf "%d rows of %d words among %d, %s, seed %d" m.n m.w
    m.distinct
    (match m.layout with
    | `Contiguous -> "contiguous"
    | `Flipped -> "flipped"
    | `Every_other_row -> "every other row")
    m.seed

let many =
  let open Gen in
  let+ n = int_range 65_537 200_000
  and+ w = int_range 1 3
  and+ distinct = of_list [ 1; 7; 5_000; 1_000_000 ]
  and+ layout = of_list [ `Contiguous; `Flipped; `Every_other_row ]
  and+ seed = int_range 0 0x3fff_ffff in
  { n; w; distinct; layout; seed }

(* [m]'s rows of [value k j], word [j] of the [k]th distinct row. *)
let many_of dtype value m =
  let st = Random.State.make [| m.seed |] in
  let rows = match m.layout with `Every_other_row -> 2 * m.n | _ -> m.n in
  let x = Array.make (rows * m.w) (Nx_dtype.zero dtype) in
  for i = 0 to rows - 1 do
    let k = Random.State.int st m.distinct in
    for j = 0 to m.w - 1 do
      x.((i * m.w) + j) <- value k j
    done
  done;
  let x = Nx.create dtype [| rows; m.w |] x in
  match m.layout with
  | `Contiguous -> x
  | `Flipped -> Nx.flip ~axes:[ 0 ] x
  | `Every_other_row ->
      Nx.squeeze ~axes:[ 2 ] (Nx.sliding_window ~axis:0 ~window:1 ~step:2 x)

(* Words that spread a row's number over all 64 bits. *)
let many_rows =
  many_of Nx.uint64 (fun k j ->
      Int64.mul (Int64.of_int (k + j)) 0x9E3779B97F4A7C15L)

(* Floats with NaNs of two payloads and both zeros among them, one column for
   rows of one word. *)
let many_floats m =
  let specials =
    [|
      Float.nan; Int64.float_of_bits 0xfff8_0000_0000_0002L; -0.; 0.; infinity;
    |]
  in
  let x =
    many_of Nx.float64
      (fun k j ->
        let k = k + j in
        if k < Array.length specials then specials.(k) else float_of_int k)
      m
  in
  if m.w = 1 then Nx.reshape [| Nx.dim 0 x |] x else x

(* unique's reference composition: a stable sort of the keys' order keys, then
   each run of equal keys and its first row in input order. *)
let unique_by_sorting keys : Nx.groups =
  let n = Nx.dim 0 keys in
  if n = 0 then
    let none = Nx.zeros Nx.int64 [| 0 |] in
    { ids = none; first = none; counts = none }
  else
    let k = Nx.order_key Nx.uint64 keys in
    let perm = Nx.lexsort k in
    let sorted = Nx.take ~axis:0 ~indices:perm k in
    let starts =
      let differs =
        Nx.not_equal
          (Nx.slice [ R (1, n) ] sorted)
          (Nx.slice [ R (0, n - 1) ] sorted)
      in
      let differs =
        if Nx.ndim k = 2 then Nx.any ~axes:[ 1 ] differs else differs
      in
      Nx.pad [| (1, 0) |] true differs
    in
    let iota = Nx.arange Nx.int64 0 n 1 in
    let run = Nx.cummax (Nx.where starts iota (Nx.zeros_like iota)) in
    let inverse =
      Nx.scatter ~unique_indices:true ~axis:0 ~indices:perm ~values:iota
        (Nx.zeros Nx.int64 [| n |])
    in
    (* The sort is stable, so a run's first row in input order is the first
       occurrence of its key. *)
    let firsts = Nx.take ~indices:inverse starts in
    let earlier =
      let f = Nx.cast Nx.int64 firsts in
      Nx.sub (Nx.cumsum f) f
    in
    let ids =
      Nx.take ~indices:inverse
        (Nx.take ~indices:(Nx.take ~indices:run perm) earlier)
    in
    let first = Nx.positions firsts in
    let counts =
      Nx.reduce_segments `Add ~segments:(Nx.dim 0 first) ids
        (Nx.ones Nx.int64 [| n |])
    in
    { ids; first; counts }

let uniques =
  let groups (S s) =
    prop
      (s.name ^ " unique numbers equal keys in order of first appearance")
      (let open Gen in
       let* n = int_range 0 12 in
       let* w = option (int_range 0 2) in
       keys_of ~pp:s.pp s.dtype s.value n w)
      (fun keys ->
        let rows = rows_of keys in
        let n = Array.length rows in
        let same a b = lexicographic (order s.compare s.is_nan) a b = 0 in
        let firsts = ref [] in
        let ids =
          Array.init n (fun i ->
              let rec find j = function
                | [] ->
                    firsts := !firsts @ [ i ];
                    j
                | f :: rest ->
                    if same rows.(f) rows.(i) then j else find (j + 1) rest
              in
              find 0 !firsts)
        in
        let k = List.length !firsts in
        let counts = Array.make k 0 in
        Array.iter (fun g -> counts.(g) <- counts.(g) + 1) ids;
        let g = Nx.unique keys in
        let int64s a = Array.map Int64.of_int a in
        equal ~msg:"ids" (array int64) (int64s ids) (Nx.to_array g.ids);
        equal ~msg:"first" (array int64)
          (int64s (Array.of_list !firsts))
          (Nx.to_array g.first);
        equal ~msg:"counts" (array int64) (int64s counts) (Nx.to_array g.counts))
  in
  let to_arrays (g : Nx.groups) =
    (Nx.to_array g.ids, Nx.to_array g.first, Nx.to_array g.counts)
  in
  let arrays = triple (array int64) (array int64) (array int64) in
  group "unique"
    (List.map groups ordered
    @ [
        prop "Group numbers rows of words in order of first appearance"
          (let open Gen in
           let* n = int_range 0 12 in
           let* w = int_range 0 3 in
           keys_of ~pp:pp_word Nx.uint64 tied_words n (Some w))
          (fun x ->
            equal (array int64)
              (first_appearance (rows_of x))
              (Nx.to_array (group_rows x)));
        prop ~count:12 "Group numbers rows across blocks"
          (Gen.with_pp pp_many many) (fun m ->
            let x = many_rows m in
            equal (array int64)
              (first_appearance (rows_of x))
              (Nx.to_array (group_rows x)));
        prop ~count:12 "unique groups as the sort composition does"
          (Gen.with_pp pp_many many) (fun m ->
            let keys = many_floats m in
            equal arrays
              (to_arrays (unique_by_sorting keys))
              (to_arrays (Nx.unique keys)));
      ]
    @ [
        test "NaNs of every payload form one group, and -0 and 0 two" (fun () ->
            let x =
              Nx.bitcast Nx.float64
                (Nx.create Nx.int64 [| 5 |]
                   [|
                     0x7ff8000000000001L;
                     Int64.bits_of_float (-0.);
                     0xfff8000000000002L;
                     0L;
                     0x7ff0000000000003L;
                   |])
            in
            equal arrays
              ([| 0L; 1L; 0L; 2L; 0L |], [| 0L; 1L; 3L |], [| 3L; 1L; 1L |])
              (to_arrays (Nx.unique x)));
        test "unique of nothing, of one key, and of rows without a column"
          (fun () ->
            equal arrays ([||], [||], [||])
              (to_arrays (Nx.unique (Nx.zeros Nx.int32 [| 0 |])));
            equal arrays
              ([| 0L |], [| 0L |], [| 1L |])
              (to_arrays (Nx.unique (Nx.zeros Nx.int32 [| 1 |])));
            equal arrays
              ([| 0L; 0L; 0L |], [| 0L |], [| 3L |])
              (to_arrays (Nx.unique (Nx.zeros Nx.int32 [| 3; 0 |]))));
        test "unique refuses complex numbers, a scalar and a 3-D tensor"
          (fun () ->
            raises_invalid_arg (fun () ->
                Nx.unique (Nx.zeros Nx.complex64 [| 2 |]));
            raises_invalid_arg (fun () -> Nx.unique (Nx.scalar Nx.int32 1l));
            raises_invalid_arg (fun () ->
                Nx.unique (Nx.zeros Nx.int32 [| 1; 1; 1 |])));
      ])

(* Quantiles *)

let r32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* The quantiles [qs] of [lane] by the definition: linear between the order
   statistics around [q * (n - 1)] of the sort order, exactly the lower one at
   an integer position or between equal ones, each operation rounded by
   [round]. *)
let quantiles ~round qs lane =
  let n = Array.length lane in
  let sorted =
    Array.map
      (fun i -> lane.(i))
      (stable_order ~descending:false compare_float Float.is_nan lane)
  in
  Array.map
    (fun q ->
      let h = q *. float_of_int (n - 1) in
      let lo = int_of_float (Float.floor h) in
      let a = sorted.(lo) and b = sorted.(Int.min (lo + 1) (n - 1)) in
      let f = h -. Float.floor h in
      if f = 0. || a = b then a
      else round (a +. round (round f *. round (b -. a))))
    qs

let probabilities =
  Gen.array ~size:(Gen.int_range 0 4)
    (Gen.frequency
       [
         (3, Gen.float_range 0. 1.);
         (1, Gen.of_list ~pp:pp_float [ 0.; 0.25; 0.5; 1.; 1. /. 3. ]);
       ])

let quantiled ~round name dtype =
  let drawn =
    let open Gen in
    let* t =
      viewed
        ~shape:(Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range 1 5))
        ~pp:pp_float dtype float_values
    in
    let+ axis = option (int_range (-Nx.ndim t) (Nx.ndim t - 1))
    and+ qs = probabilities in
    (t, axis, qs)
  in
  prop name drawn (fun (t, axis, qs) ->
      assume (Nx.numel t > 0);
      let k = Array.length qs in
      let r = Ref.of_nx t in
      match axis with
      | None ->
          let expected = quantiles ~round qs r.data in
          equal (array float_exact) expected (Nx.to_array (Nx.quantile qs t))
      | Some a ->
          let a = Ref.axis r a in
          let expected = Ref.along ~axis:a ~length:k (quantiles ~round qs) r in
          equal (Ref.witness float_exact) expected
            (Ref.of_nx (Nx.moveaxis 0 a (Nx.quantile ~axis:a qs t))))

let quantiles_group =
  let x = Nx.create Nx.float64 [| 4 |] [| 4.; 1.; 3.; 2. |] in
  group "quantile"
    [
      quantiled ~round:Fun.id "float64 quantiles are the definition's"
        Nx.float64;
      quantiled ~round:r32
        "float32 quantiles are the definition's, rounded at each operation"
        Nx.float32;
      prop "float16 and bfloat16 interpolate in float32 and round once"
        (Gen.pair
           (Gen.array ~size:(Gen.int_range 1 9) float_values)
           probabilities)
        (fun (xs, qs) ->
          let narrow (type b) (dt : (float, b) Nx.dtype) =
            let t =
              Nx.cast dt (Nx.create Nx.float32 [| Array.length xs |] xs)
            in
            equal (tensor float_exact)
              (Nx.cast dt (Nx.quantile qs (Nx.cast Nx.float32 t)))
              (Nx.quantile qs t)
          in
          narrow Nx.float16;
          narrow Nx.bfloat16);
      cases "quantile interpolates between order statistics" ~name:fst
        [
          ("the median of an even count", ([| 0.5 |], [| 2.5 |]));
          ("the extremes", ([| 0.; 1. |], [| 1.; 4. |]));
          ("an integer position", ([| 1. /. 3. |], [| 2. |]));
          ("a quarter", ([| 0.25 |], [| 1.75 |]));
        ]
        (fun (_, (qs, expected)) ->
          equal (array float_exact) expected (Nx.to_array (Nx.quantile qs x)));
      test "NaN sorts last and reaches the top quantile" (fun () ->
          let t = Nx.create Nx.float64 [| 3 |] [| nan; 1.; 2. |] in
          equal (array float_exact) [| 1.; 2.; nan |]
            (Nx.to_array (Nx.quantile [| 0.; 0.5; 1. |] t)));
      test "between equal infinities the quantile is that infinity" (fun () ->
          let t = Nx.create Nx.float64 [| 2 |] [| infinity; infinity |] in
          equal (array float_exact) [| infinity |]
            (Nx.to_array (Nx.quantile [| 0.5 |] t)));
      test "the probabilities lead the result's shape" (fun () ->
          let t = Nx.zeros Nx.float32 [| 2; 3; 4 |] in
          equal (array int) [| 5; 2; 4 |]
            (Nx.shape (Nx.quantile ~axis:1 [| 0.; 0.1; 0.2; 0.3; 1. |] t));
          equal (array int) [| 0 |] (Nx.shape (Nx.quantile [||] t)));
      cases "quantile refuses what has no quantile" ~name:fst
        [
          ("a probability below 0", fun () -> Nx.quantile [| -0.1 |] x);
          ("a probability above 1", fun () -> Nx.quantile [| 1.5 |] x);
          ("a NaN probability", fun () -> Nx.quantile [| nan |] x);
          ( "an empty axis",
            fun () ->
              Nx.quantile ~axis:1 [| 0.5 |] (Nx.zeros Nx.float64 [| 2; 0 |]) );
          ( "an empty tensor",
            fun () -> Nx.quantile [| 0.5 |] (Nx.zeros Nx.float64 [| 0 |]) );
          ("an axis out of bounds", fun () -> Nx.quantile ~axis:1 [| 0.5 |] x);
        ]
        (fun (_, f) -> raises_invalid_arg f);
    ]

let () =
  exit
    (run "nx sorting"
       [
         sorts;
         top_ks;
         gathers;
         order_keys;
         lexsorts;
         searchsorts;
         uniques;
         quantiles_group;
       ])
