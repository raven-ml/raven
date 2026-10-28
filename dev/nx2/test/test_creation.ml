(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Leaves, host values made from numbers, each against the elements its doc
   states, at every dtype; their [_at] forms at their placements; and the
   creations beside a reference, at its placement. *)

open Windtrap
open Nx_support
module D = Nx_array.Dtype

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ m 2 ])

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

(* The value an int64 [i] stores as in [dt], by the cast rule. *)
let of_int64 (type v s) (dt : (v, s) D.t) i : v =
  (Nx.to_array (Nx.cast dt (Nx.create Nx.int64 [||] [| Int64.of_int i |]))).(0)

let of_float64 (type v s) (dt : (v, s) D.t) f : v =
  (Nx.to_array (Nx.cast dt (Nx.create Nx.float64 [||] [| f |]))).(0)

let dtype = Gen.of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all

let float_dtype =
  Gen.of_list
    ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt)
    (List.filter (fun (D.Any dt) -> D.is D.Float dt) D.all)

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency
       [ (1, Gen.constant 0); (1, Gen.constant 1); (4, Gen.int_range 2 4) ])
  |> Gen.with_pp pp_ints

let numel s = Array.fold_left ( * ) 1 s

let contains s w =
  let n = String.length w in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = w || at (i + 1))
  in
  at 0

(* A refusal in nx's words: its message starts with the function the user
   called and names no function of the layers below. *)
let named ~by f =
  match f () with
  | () -> failf "%s raised nothing" by
  | exception Invalid_argument m ->
      let n = Int.min (String.length by + 2) (String.length m) in
      let start = String.sub m 0 n in
      let below = [ "Nx_array"; "Nx_kernel"; "Layout."; "Move."; "Prog." ] in
      equal ~msg:m (pair string (list string)) (by ^ ": ", [])
        (start, List.filter (contains m) below)

(* Each refusal raises naming its function; cases are numbered to name them. *)
let refusals l =
  cases "refusals"
    ~name:(fun (n, _, _) -> n)
    (List.mapi (fun i (by, f) -> (Printf.sprintf "%d %s" i by, by, f)) l)
    (fun (_, by, f) -> named ~by f)

(* Past the array layer's limits: 33 axes, and extents other than 0 whose
   product is 2^54. *)
let ranked = Array.make 33 1
let huge = [| 0; 1 lsl 27; 1 lsl 27 |]

(* Fills *)

let law_ones (D.Any dt, s) =
  cover "no element" (numel s = 0);
  cover "rank 0" (s = [||]);
  let x = Nx.ones dt s in
  equal (array int) s (Nx.shape x);
  equal_elements dt (Array.make (numel s) (D.one dt)) (Nx.to_array x)

(* One value of [dt] from drawn bytes: any of the format's, NaN, signed zeros,
   infinities and extremes among them. *)
let drawn (type v s) (dt : (v, s) D.t) : v Gen.t =
  let open Gen in
  let+ data = string_of ~size:(constant (D.bytes dt 1)) char in
  let data =
    match dt with
    | D.Bool | D.Bit -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | D.Int4 | D.Uint4 | D.Float4_e2m1fn ->
        String.map (fun c -> Char.chr (Char.code c land 15)) data
    | _ -> data
  in
  Nx_array.get
    (Nx_array.v dt (Nx_array.Layout.contiguous [||]) (Rig.Buffer.of_string data))
    [||]

type filled = Filled : ('v, 's) D.t * int array * 'v -> filled

let filled =
  Gen.with_pp
    (fun ppf (Filled (dt, s, v)) ->
      Format.fprintf ppf "%a %a %a" D.pp dt pp_ints s (D.pp_value dt) v)
    (let open Gen in
     let* (D.Any dt) = dtype in
     let* s = shape in
     let+ v = drawn dt in
     Filled (dt, s, v))

let law_full (Filled (dt, s, v)) =
  cover "no element" (numel s = 0);
  equal_elements dt (Array.make (numel s) v) (Nx.to_array (Nx.full dt s v))

let test_full_stores () =
  let x = Nx.full Nx.float16 [| 2 |] 0.1 in
  equal (array float_exact) (Array.make 2 (D.of_float D.Float16 0.1)) (Nx.to_array x);
  equal (array float_exact) [| 448.; 448. |]
    (Nx.to_array (Nx.full Nx.float8_e4m3fn [| 2 |] 1e6))

(* float8_e5m2's stores about its largest finite value, 57344: the value past
   it is 65536, so 61440 is the tie, which rounds to the even code, infinity. *)
let e5m2_overflow =
  [
    (57344., 57344.);
    (61439., 57344.);
    (61440., infinity);
    (65536., infinity);
    (1e6, infinity);
    (infinity, infinity);
  ]
  |> List.concat_map (fun (x, v) -> [ (x, v); (-.x, -.v) ])

(* Each way a float becomes an element of float8_e5m2 stores [x] as [v]. *)
let e5m2_stores =
  let e5m2 = Nx.float8_e5m2 in
  let cast dt x = Nx.cast e5m2 (Nx.create dt [| 1 |] [| x |]) in
  [
    ("full", fun x -> Nx.full e5m2 [| 1 |] x);
    ("scalar", fun x -> Nx.reshape [| 1 |] (Nx.scalar e5m2 x));
    ("create", fun x -> Nx.create e5m2 [| 1 |] [| x |]);
    ("a cast from float32", cast Nx.float32);
    ("a cast from float64", cast Nx.float64);
  ]

let test_e5m2_store (_, store) =
  List.iter
    (fun (x, v) ->
      equal ~msg:(Printf.sprintf "%h" x) (array float_exact) [| v |]
        (Nx.to_array (store x)))
    e5m2_overflow

let fills =
  group "fills"
    [
      cases ~name:fst "float8_e5m2 overflows to infinity" e5m2_stores
        test_e5m2_store;
      prop "ones is one at every element" (Gen.pair dtype shape) law_ones;
      prop "full is its value at every element" filled law_full;
      test "full stores a float by the conversion rule" test_full_stores;
      refusals
        [
          ("Nx.ones", fun () -> ignore (Nx.ones Nx.float32 [| 2; -1 |]));
          ("Nx.full", fun () -> ignore (Nx.full Nx.uint8 [| 2 |] 256));
          ("Nx.full", fun () -> ignore (Nx.full Nx.int4 [| 2 |] (-9)));
          ("Nx.zeros", fun () -> ignore (Nx.zeros Nx.float32 [| -1 |]));
          ("Nx.zeros", fun () -> ignore (Nx.zeros Nx.float32 ranked));
          ("Nx.zeros", fun () -> ignore (Nx.zeros Nx.float32 huge));
          ("Nx.full", fun () -> ignore (Nx.full Nx.uint8 ranked 0));
          ("Nx.full", fun () -> ignore (Nx.full Nx.uint8 huge 0));
          ("Nx.scalar", fun () -> ignore (Nx.scalar Nx.uint8 256));
          ("Nx.create", fun () -> ignore (Nx.create Nx.float32 [| -1 |] [||]));
          ("Nx.create", fun () -> ignore (Nx.create Nx.float32 ranked [| 0. |]));
          ("Nx.create", fun () -> ignore (Nx.create Nx.float32 huge [||]));
          ("Nx.create", fun () -> ignore (Nx.create Nx.float32 [| 2 |] [| 0. |]));
          ("Nx.create", fun () -> ignore (Nx.create Nx.uint8 [| 1 |] [| 300 |]));
          ("Nx.create", fun () -> ignore (Nx.create Nx.int4 [| 2 |] [| 0; -9 |]));
          ("Nx.init", fun () -> ignore (Nx.init Nx.float32 ranked (fun _ -> 0.)));
          ("Nx.init", fun () -> ignore (Nx.init Nx.float32 huge (fun _ -> 0.)));
          ("Nx.init", fun () -> ignore (Nx.init Nx.uint8 [| 1 |] (fun _ -> 300)));
          ( "Nx.map_item",
            fun () -> ignore (Nx.map_item (fun _ -> 300) (Nx.zeros Nx.uint8 [| 1 |])) );
          ("Nx.add_s", fun () -> ignore (Nx.add_s (Nx.zeros Nx.uint8 [| 1 |]) 300));
          ("Nx.clamp", fun () -> ignore (Nx.clamp ~max:300 (Nx.zeros Nx.uint8 [| 1 |])));
          ("Nx.Rng.uniform", fun () -> ignore (Nx.Rng.uniform Nx.float32 [| -1 |]));
          ("Nx.Rng.uniform", fun () -> ignore (Nx.Rng.uniform Nx.float32 ranked));
          ("Nx.Rng.uniform", fun () -> ignore (Nx.Rng.uniform Nx.float32 huge));
        ]

    ]

(* Ranges *)

type ranged = Ranged : ('v, 's) D.t * int * int * int -> ranged

(* Bounds a dtype holds, small enough for every integer dtype: start and stop
   in [0, 7], steps from -3 to 3 but 0. A quarter of the cases are booleans,
   one dtype among many, so that every run reaches a range a boolean can't
   hold. *)
let ranged =
  Gen.with_pp
    (fun ppf (Ranged (dt, a, b, c)) ->
      Format.fprintf ppf "%a %d %d %d" D.pp dt a b c)
    (let open Gen in
     let* (D.Any dt) =
       frequency [ (3, dtype); (1, constant (D.Any D.Bool)) ]
     in
     let* start = int_range 0 7 in
     let* stop = int_range 0 7 in
     let+ step = one_of [ int_range 1 3; int_range (-3) (-1) ] in
     Ranged (dt, start, stop, step))

let rec values start stop step =
  if (step > 0 && start >= stop) || (step < 0 && start <= stop) then []
  else start :: values (start + step) stop step

let law_arange (Ranged (dt, start, stop, step)) =
  let vs = values start stop step in
  cover "empty" (vs = []);
  cover "one value" (List.length vs = 1);
  cover "descending" (step < 0 && vs <> []);
  let in_range =
    match D.kind dt with
    | D.Boolean -> List.for_all (fun v -> v <= 1) vs
    | _ -> true
  in
  cover "out of a boolean's range" (not in_range);
  if not in_range then invalid ~by:"Nx.arange" (fun () -> Nx.arange dt start stop step)
  else
    equal_elements dt
      (Array.of_list (List.map (of_int64 dt) vs))
      (Nx.to_array (Nx.arange dt start stop step))

let test_arange_bounds () =
  equal (array int) [| 253; 254; 255 |] (Nx.to_array (Nx.arange Nx.uint8 253 256 1));
  equal (array int) [| -8; 7 |] (Nx.to_array (Nx.arange Nx.int4 (-8) 8 15));
  equal (array int64)
    [| Int64.of_int (max_int - 1) |]
    (Nx.to_array (Nx.arange Nx.int64 (max_int - 1) max_int 1));
  equal (array int64) [| Int64.of_int max_int |]
    (Nx.to_array (Nx.arange Nx.int64 max_int (max_int - 1) (-1)));
  (* Across the whole range, where [stop - start] overflows [int], in either
     direction. *)
  equal (array int64)
    (Array.map Int64.of_int [| min_int; -(1 lsl 61); 0; 1 lsl 61 |])
    (Nx.to_array (Nx.arange Nx.int64 min_int max_int (1 lsl 61)));
  equal (array int64)
    (Array.map Int64.of_int [| max_int; max_int - (1 lsl 62) |])
    (Nx.to_array (Nx.arange Nx.int64 max_int min_int (-(1 lsl 62))))

let law_arange_f (D.Any dt, (start, (stop, step))) =
  let start = Float.of_int start /. 4. and stop = Float.of_int stop /. 4. in
  let step = Float.of_int step /. 8. in
  assume (step <> 0.);
  let n = Stdlib.max 0 (Float.to_int (Float.ceil ((stop -. start) /. step))) in
  cover "empty" (n = 0);
  let expected =
    Array.init n (fun i -> of_float64 dt (Float.fma (Float.of_int i) step start))
  in
  match D.kind dt with
  | D.Float -> equal_elements dt expected (Nx.to_array (Nx.arange_f dt start stop step))
  | _ -> ()

(* With [endpoint] and two values or more, the ends are [start] and [stop] as
   stored, and every value lies between them. *)
let law_linspace (D.Any dt, (n, (start, (stop, endpoint)))) =
  let start = Float.of_int start /. 3. and stop = Float.of_int stop /. 7. in
  match D.kind dt with
  | D.Float ->
      let x = Nx.to_array (Nx.linspace dt ~endpoint start stop n) in
      equal int n (Array.length x);
      cover "one value" (n = 1);
      if n >= 1 then equal_elements dt [| D.of_float dt start |] [| x.(0) |];
      if endpoint && n >= 2 then
        equal_elements dt [| D.of_float dt stop |] [| x.(n - 1) |];
      let lo = D.of_float dt (Float.min start stop)
      and hi = D.of_float dt (Float.max start stop) in
      Array.iter (fun v -> equal bool true (v >= lo && v <= hi)) x;
      let reference i =
        let div = if endpoint then n - 1 else n in
        let step =
          if div > 0 then (stop -. start) /. Float.of_int div else 0.
        in
        if endpoint && n >= 2 && i = n - 1 then stop
        else Float.fma (Float.of_int i) step start
      in
      equal_elements dt (Array.init n (fun i -> of_float64 dt (reference i))) x
  | _ -> ()

let test_linspace_ints () =
  equal (array int) [| 0; 2; 5; 7; 10 |] (Nx.to_array (Nx.linspace Nx.int32 0. 10. 5 |> Nx.cast Nx.int16));
  equal (array float_exact) [| 0.; 2.; 4.; 6.; 8. |]
    (Nx.to_array (Nx.linspace Nx.float64 ~endpoint:false 0. 10. 5));
  equal (array float_exact) [| 3. |] (Nx.to_array (Nx.linspace Nx.float64 3. 10. 1));
  equal (array float_exact) [||] (Nx.to_array (Nx.linspace Nx.float64 3. 10. 0))

let test_logspace () =
  equal (array float_exact) [| 1.; 10.; 100. |]
    (Nx.to_array (Nx.logspace Nx.float32 0. 2. 3));
  equal (array float_exact) [| 1.; 2.; 4.; 8. |]
    (Nx.to_array (Nx.logspace Nx.float64 ~base:2. 0. 3. 4));
  equal (array float_exact) [| 1.; 2.; 4. |]
    (Nx.to_array (Nx.logspace Nx.bfloat16 ~endpoint:false ~base:2. 0. 3. 3))

(* The ends are [start] and [stop] as stored, and each value is
   [start (stop / start)^(i / div)] stored once: within one step of [dt] of
   that value at float64 stored, or 2{^-40} of it at float64. Far ends put
   [stop / start] past float64's range. *)
let law_geomspace
    (D.Any dt, (n, ((far, flip), ((m1, m2), ((a, b), endpoint))))) =
  match D.kind dt with
  | D.Float ->
      let ea, eb =
        if far then (-(600 + a), 600 + b) else ((a mod 41) - 20, (b mod 41) - 20)
      in
      let start = Float.ldexp (Float.of_int m1) ea
      and stop = Float.ldexp (Float.of_int m2) eb in
      let start, stop = if flip then (stop, start) else (start, stop) in
      cover "a ratio past float64's range" far;
      cover "one value" (n = 1);
      let x = Nx.to_array (Nx.geomspace dt ~endpoint start stop n) in
      equal int n (Array.length x);
      if n >= 1 then equal_elements dt [| D.of_float dt start |] [| x.(0) |];
      if endpoint && n >= 2 then
        equal_elements dt [| D.of_float dt stop |] [| x.(n - 1) |];
      let div = if endpoint then n - 1 else n in
      let f = D.float_format dt in
      let rel = Float.max f.epsilon (Float.ldexp 1. (-40)) in
      let lo = Float.log start in
      Array.iteri
        (fun i v ->
          let e = if div > 0 then Float.of_int i /. Float.of_int div else 0. in
          let r = D.of_float dt (Float.exp (lo +. (e *. (Float.log stop -. lo)))) in
          if not (Float.equal v r) then
            at_most ~msg:(Printf.sprintf "value %d of %g" i r) float_exact
              ~than:((rel *. Float.abs r) +. (f.min_normal *. f.epsilon))
              (Float.abs (v -. r)))
        x
  | _ -> ()

let test_geomspace () =
  equal (array float_exact) [| 1.; 10.; 100.; 1000. |]
    (Nx.to_array (Nx.geomspace Nx.float32 1. 1000. 4));
  equal (array float_exact) [| 1.; 2.; 4.; 8. |]
    (Nx.to_array (Nx.geomspace Nx.float32 ~endpoint:false 1. 16. 4));
  equal (array float_exact) [| 3. |] (Nx.to_array (Nx.geomspace Nx.float64 3. 10. 1));
  equal (array float_exact) [||] (Nx.to_array (Nx.geomspace Nx.float64 3. 10. 0))

let ranges =
  group "ranges"
    [
      prop "arange is start, start + step, ... before stop" ranged law_arange;
      test "arange reaches its dtype's bounds" test_arange_bounds;
      prop "arange_f is start + i step rounded once"
        (Gen.pair float_dtype
           (Gen.pair (Gen.int_range (-8) 8)
              (Gen.pair (Gen.int_range (-8) 8) (Gen.int_range (-5) 5))))
        law_arange_f;
      prop "linspace keeps its ends and lies between them"
        (Gen.pair float_dtype
           (Gen.pair (Gen.int_range 0 7)
              (Gen.pair (Gen.int_range (-20) 20)
                 (Gen.pair (Gen.int_range (-20) 20) Gen.bool))))
        law_linspace;
      test "linspace into integers, without its end, of one and of none"
        test_linspace_ints;
      test "logspace is exact at exact powers" test_logspace;
      prop "geomspace keeps its ends and its ratio"
        (Gen.pair float_dtype
           (Gen.pair (Gen.int_range 0 7)
              (Gen.pair (Gen.pair Gen.bool Gen.bool)
                 (Gen.pair
                    (Gen.pair (Gen.int_range 1 7) (Gen.int_range 1 7))
                    (Gen.pair
                       (Gen.pair (Gen.int_range 0 400) (Gen.int_range 0 400))
                       Gen.bool)))))
        law_geomspace;
      test "geomspace at exact powers, of one and of none" test_geomspace;
      refusals
        [
          ("Nx.geomspace", fun () -> ignore (Nx.geomspace Nx.float32 0. 1. 3));
          ("Nx.geomspace", fun () -> ignore (Nx.geomspace Nx.float32 (-1.) (-2.) 3));
          ("Nx.geomspace", fun () -> ignore (Nx.geomspace Nx.float32 1. infinity 3));
          ("Nx.geomspace", fun () -> ignore (Nx.geomspace Nx.float32 1. 2. (-1)));
          ("Nx.arange", fun () -> ignore (Nx.arange Nx.int32 0 4 0));
          ("Nx.arange", fun () -> ignore (Nx.arange Nx.uint8 0 300 1));
          ("Nx.arange", fun () -> ignore (Nx.arange Nx.uint8 2 (-2) (-1)));
          ("Nx.arange", fun () -> ignore (Nx.arange Nx.bool 0 3 1));
          ("Nx.arange", fun () -> ignore (Nx.arange Nx.int64 min_int max_int 1));
          ("Nx.arange", fun () -> ignore (Nx.arange Nx.int64 max_int min_int (-1)));
          ("Nx.arange_f", fun () -> ignore (Nx.arange_f Nx.float32 0. 1. 0.));
          ("Nx.arange_f", fun () -> ignore (Nx.arange_f Nx.float32 0. infinity 1.));
          ("Nx.arange_f", fun () -> ignore (Nx.arange_f Nx.float32 nan 1. 1.));
          ("Nx.linspace", fun () -> ignore (Nx.linspace Nx.float32 0. 1. (-1)));
          ("Nx.logspace", fun () -> ignore (Nx.logspace Nx.float32 0. 1. (-1)));
          ("Nx.eye", fun () -> ignore (Nx.eye Nx.float32 (-1)));
          ("Nx.eye", fun () -> ignore (Nx.eye ~m:(-2) Nx.float32 2));
          ("Nx.eye", fun () -> ignore (Nx.eye Nx.float32 (1 lsl 27)));
          ("Nx.arange", fun () -> ignore (Nx.arange Nx.int64 0 (1 lsl 54) 1));
          ("Nx.arange_f", fun () -> ignore (Nx.arange_f Nx.float64 0. 0x1p54 1.));
          ("Nx.linspace", fun () -> ignore (Nx.linspace Nx.float32 0. 1. (1 lsl 54)));
          ("Nx.logspace", fun () -> ignore (Nx.logspace Nx.float32 0. 1. (1 lsl 54)));
        ]

    ]

(* Eye *)

let law_eye (D.Any dt, (n, (m, k))) =
  let x = Nx.to_array (Nx.eye ~m ~k dt n) in
  cover "no element" (n * m = 0);
  cover "off the matrix" (k >= m || -k >= n);
  equal_elements dt
    (Array.init (n * m) (fun j ->
         if (j mod m) - (j / m) = k then D.one dt else D.zero dt))
    x

(* Whole on both devices, on one, and split along each axis. *)
let eye_placements =
  [ S2.on; Nx.Placement.device S2.v (m 1); S2.split ~axis:0; S2.split ~axis:1 ]

(* [eye_at] at each placement lies there and holds the host value's elements,
   where a split cuts an odd extent and where the diagonal leaves the matrix
   too. *)
let law_eye_at (D.Any dt, (n, (m, k))) =
  let host = Array.to_list (Nx.to_array (Nx.eye ~m ~k dt n)) in
  cover "no element" (n * m = 0);
  cover "off the matrix" (k >= m || -k >= n);
  cover "an odd extent" (n mod 2 = 1 || m mod 2 = 1);
  List.iter
    (fun p ->
      let y = Nx.eye_at ~m ~k p dt n in
      let msg = Format.asprintf "%a" Nx.Placement.pp p in
      equal ~msg (placement ()) p (Nx.placement y);
      equal ~msg (elements dt) host (Array.to_list (Nx.to_array y)))
    eye_placements

let eye =
  group "eye"
    [
      prop "eye is one where column - row = k"
        (Gen.pair dtype
           (Gen.pair (Gen.int_range 0 4)
              (Gen.pair (Gen.int_range 0 4) (Gen.int_range (-5) 5))))
        law_eye;
      prop "eye_at holds eye's elements wherever it lies"
        (Gen.pair dtype
           (Gen.pair (Gen.int_range 0 5)
              (Gen.pair (Gen.int_range 0 5) (Gen.int_range (-7) 7))))
        law_eye_at;
      test "eye defaults to the square main diagonal" (fun () ->
          equal (array int32) [| 1l; 0l; 0l; 1l |]
            (Nx.to_array (Nx.eye Nx.int32 2)));
    ]

(* Placements *)

(* A creation's host form and its [_at] form: [lies p] is where the [_at]
   form at [p] lies, [p] itself but for a 0-d value. *)
type pair =
  | Pair : {
      name : string;
      dtype : ('v, 's) D.t;
      host : unit -> ('v, 's, Nx.host) Nx.t;
      at : 'd. 'd Nx.Placement.t -> ('v, 's, 'd) Nx.t;
      lies : 'd. 'd Nx.Placement.t -> 'd Nx.Placement.t;
    }
      -> pair

(* An [_at] form, polymorphic in its brand. *)
type ('v, 's) at = { f : 'd. 'd Nx.Placement.t -> ('v, 's, 'd) Nx.t }

let pair name dtype host at =
  Pair { name; dtype; host; at = at.f; lies = Fun.id }

let pairs =
  let f32 = Nx.float32 in
  [
    pair "zeros" f32
      (fun () -> Nx.zeros f32 [| 4; 3 |])
      { f = (fun p -> Nx.zeros_at p f32 [| 4; 3 |]) };
    pair "ones" Nx.int8
      (fun () -> Nx.ones Nx.int8 [| 4; 3 |])
      { f = (fun p -> Nx.ones_at p Nx.int8 [| 4; 3 |]) };
    pair "full" f32
      (fun () -> Nx.full f32 [| 4; 3 |] 2.5)
      { f = (fun p -> Nx.full_at p f32 [| 4; 3 |] 2.5) };
    Pair
      {
        name = "scalar";
        dtype = Nx.int32;
        host = (fun () -> Nx.scalar Nx.int32 7l);
        at = (fun p -> Nx.scalar_at p Nx.int32 7l);
        lies = Nx.Placement.uncut;
      };
    pair "arange" Nx.int64
      (fun () -> Nx.arange Nx.int64 3 19 2)
      { f = (fun p -> Nx.arange_at p Nx.int64 3 19 2) };
    pair "arange_f" f32
      (fun () -> Nx.arange_f f32 0.5 4.5 0.5)
      { f = (fun p -> Nx.arange_f_at p f32 0.5 4.5 0.5) };
    pair "linspace" Nx.float64
      (fun () -> Nx.linspace Nx.float64 (-1.) 2. 7)
      { f = (fun p -> Nx.linspace_at p Nx.float64 (-1.) 2. 7) };
    pair "logspace" f32
      (fun () -> Nx.logspace f32 ~base:2. 0. 5. 6)
      { f = (fun p -> Nx.logspace_at p f32 ~base:2. 0. 5. 6) };
    pair "geomspace" Nx.float64
      (fun () -> Nx.geomspace Nx.float64 1. 1000. 5)
      { f = (fun p -> Nx.geomspace_at p Nx.float64 1. 1000. 5) };
    pair "eye" Nx.uint8
      (fun () -> Nx.eye ~m:5 ~k:1 Nx.uint8 4)
      { f = (fun p -> Nx.eye_at ~m:5 ~k:1 p Nx.uint8 4) };
    pair "fftfreq" Nx.float64
      (fun () -> Nx.fftfreq Nx.float64 ~d:0.5 9)
      { f = (fun p -> Nx.fftfreq_at p Nx.float64 ~d:0.5 9) };
    pair "rfftfreq" f32
      (fun () -> Nx.rfftfreq f32 8)
      { f = (fun p -> Nx.rfftfreq_at p f32 8) };
    pair "hann" Nx.float64
      (fun () -> Nx.hann Nx.float64 6)
      { f = (fun p -> Nx.hann_at p Nx.float64 6) };
  ]

(* The host form lies on the host; the [_at] form at a split lies there,
   each device holding its window, and holds the host form's elements. *)
let test_pair (Pair { dtype; host; at; lies; _ }) =
  let h = host () in
  equal ~msg:"host form" (placement ()) Nx.Host.on (Nx.placement h);
  let p = S2.split ~axis:0 in
  let y = at p in
  equal ~msg:"_at form" (placement ()) (lies p) (Nx.placement y);
  (* A split of [n] elements over two devices gives the first [n / 2] rounded
     up; a 0-d value lies whole on each. *)
  let window j =
    let s = Array.copy (Nx.shape y) in
    if Array.length s > 0 then begin
      let n = s.(0) in
      s.(0) <- (if j < n mod 2 then (n / 2) + 1 else n / 2)
    end;
    s
  in
  Iarray.iteri
    (fun j a ->
      equal ~msg:(Printf.sprintf "device %d's window" j) (array int) (window j)
        (Iarray.to_array (Nx_array.Layout.shape (Nx_array.layout a))))
    (Nx.Repr.shards y);
  equal_elements dtype (Nx.to_array h) (Nx.to_array y)

(* Placements over two devices, and shapes of up to three axes: an [_at]
   form lies at its placement where the shape has every axis it cuts, and
   raises naming itself elsewhere. *)
let placement_gen =
  Gen.of_list ~pp:Nx.Placement.pp
    [
      S2.on;
      S2.split ~axis:0;
      S2.split ~axis:1;
      S2.split ~axis:2;
      Nx.Placement.device S2.v (m 1);
    ]

let law_at (p, (s, v)) =
  let cut = not (Nx.Placement.equal (Nx.Placement.uncut p) p) in
  let lacks =
    List.exists
      (fun a -> Nx.Placement.equal p (S2.split ~axis:a) && a >= Array.length s)
      [ 0; 1; 2 ]
  in
  cover "cuts an axis the shape lacks" lacks;
  cover "cuts an axis the shape has" (cut && not lacks);
  if lacks then invalid ~by:"Nx.full_at" (fun () -> Nx.full_at p Nx.float32 s v)
  else
    let y = Nx.full_at p Nx.float32 s v in
    equal (placement ()) p (Nx.placement y);
    let stored = D.of_float D.Float32 v in
    equal_elements D.Float32 (Array.make (numel s) stored) (Nx.to_array y)

let test_scalar_at_uncut () =
  let mesh = Nx.Mesh.v S2.v [ ("x", 2) ] in
  List.iter
    (fun p ->
      let y = Nx.scalar_at p Nx.float32 1.5 in
      equal (placement ()) S2.on (Nx.placement y);
      equal (array float_exact) [| 1.5 |] (Nx.to_array y))
    [
      S2.split ~axis:0;
      S2.split ~axis:3;
      Nx.Placement.mesh mesh [ (1, [ "x" ]) ];
    ]

(* An operation over host creations, and a movement of one, lie on the host. *)
let test_host_computed () =
  let c = Nx.zeros Nx.float32 [| 2 |] in
  equal ~msg:"an operation" (placement ()) Nx.Host.on
    (Nx.placement (Nx.add c c));
  equal ~msg:"a movement" (placement ()) Nx.Host.on
    (Nx.placement (Nx.reshape [| 1; 2 |] c))

let placements =
  group "placements"
    [
      test "what host creations compute lies on the host" test_host_computed;
      cases "each creation's host form and _at form"
        ~name:(fun (Pair { name; _ }) -> name)
        pairs test_pair;
      prop "full_at lies at its placement"
        (Gen.pair placement_gen (Gen.pair shape (Gen.float_range (-4.) 4.)))
        law_at;
      test "scalar_at at a cut placement lies whole on each device"
        test_scalar_at_uncut;
      refusals
        [
          ( "Nx.zeros_at",
            fun () ->
              ignore (Nx.zeros_at (S2.split ~axis:1) Nx.float32 [| 4 |]) );
          ( "Nx.arange_at",
            fun () -> ignore (Nx.arange_at (S2.split ~axis:1) Nx.int32 0 4 1) );
          ( "Nx.eye_at",
            fun () -> ignore (Nx.eye_at (S2.split ~axis:2) Nx.float32 3) );
          ( "Nx.hann_at",
            fun () -> ignore (Nx.hann_at (S2.split ~axis:1) Nx.float32 8) );
          ( "Nx.ones_at",
            fun () -> ignore (Nx.ones_at S2.on Nx.float32 [| -1 |]) );
          ( "Nx.full_at",
            fun () -> ignore (Nx.full_at S2.on Nx.uint8 [| 2 |] 256) );
        ];
    ]

(* Beside a reference *)

let test_like_placement () =
  let s = [| 4; 2 |] in
  let x = Nx.place (S2.split ~axis:0) (Nx.zeros Nx.float32 s) in
  let at y = Nx.placement y in
  equal (placement ()) (S2.split ~axis:0) (at (Nx.zeros_like x));
  equal (placement ()) (S2.split ~axis:0) (at (Nx.ones_like x));
  equal (placement ()) (S2.split ~axis:0) (at (Nx.full_like x 2.));
  equal (placement ()) S2.on (at (Nx.scalar_like x 2.));
  equal (array float_exact) (Array.make 8 0.) (Nx.to_array (Nx.zeros_like x));
  equal (array float_exact) (Array.make 8 2.) (Nx.to_array (Nx.full_like x 2.));
  equal (array float_exact) (Array.make 8 1.) (Nx.to_array (Nx.ones_like x));
  equal (array float_exact) [| 2. |] (Nx.to_array (Nx.scalar_like x 2.))

(* [beside x] is [x]'s placement uncut, and a value of another shape made
   there combines with [x] however [x] is cut. *)
let law_beside p =
  let cut = not (Nx.Placement.equal (Nx.Placement.uncut p) p) in
  cover "a cut placement" cut;
  cover "an uncut placement" (not cut);
  let x = Nx.place p (Nx.arange_f Nx.float32 0. 24. 1. |> Nx.reshape [| 2; 3; 4 |]) in
  equal (placement ()) (Nx.Placement.uncut p) (Nx.beside x);
  let k = Nx.arange_f_at (Nx.beside x) Nx.float32 0. 4. 1. in
  let want = Array.init 24 (fun i -> Float.of_int (i + (i mod 4))) in
  equal (array float_exact) want (Nx.to_array (Nx.add x k))

(* A sub-byte fill is one kernel call on a device of nx.cpu's kernels. *)
let test_like_sub_byte () =
  let check (type v s) (dt : (v, s) D.t) (v : v) =
    let x = Nx.place Count.on (Nx.zeros dt [| 13 |]) in
    C.reset ();
    let y = Nx.full_like x v in
    let got = Nx.to_array y in
    equal ~msg:(D.name dt ^ " in one kernel call") int 1 (C.calls ());
    equal_elements dt (Array.make 13 v) got
  in
  check D.Int4 (-7);
  check D.Uint4 11;
  check D.Float4_e2m1fn (-3.);
  check D.Bit true

let test_like_host () =
  let x = Nx.create Nx.bool [| 2 |] [| false; false |] in
  equal bool true
    (Nx.Placement.equal Nx.Host.on (Nx.placement (Nx.ones_like x)));
  equal (array bool) [| true; true |] (Nx.to_array (Nx.ones_like x))

let like =
  group "beside a reference"
    [
      test "a split reference: a like is split, scalar_like whole on each"
        test_like_placement;
      test "a host reference gives host values" test_like_host;
      prop "beside is the uncut placement, and combines" placement_gen
        law_beside;
      test "a sub-byte reference fills in one kernel call" test_like_sub_byte;
      refusals
        [
          ("Nx.full_like", fun () -> ignore (Nx.full_like (Nx.zeros Nx.uint4 [| 1 |]) 16));
          ("Nx.scalar_like", fun () -> ignore (Nx.scalar_like (Nx.zeros Nx.int8 [| 1 |]) 128));
          ( "Nx.add",
            fun () ->
              let x = Nx.place (S2.split ~axis:0) (Nx.zeros Nx.float32 [| 4; 3 |]) in
              ignore (Nx.add x (Nx.arange_at (Nx.placement x) Nx.float32 0 3 1)) );
        ]

    ]

(* Without kernels *)

(* A set no kernel library computes on, as a GPU set is outside a compiler:
   a memory device and an io device, which does not run on the host. *)
let io = Nx_array_support.io_device ()

module Bare = (val Nx.devices [ m 3; io ])

(* [Bare]'s placements: whole on each device, cut along an axis, and each
   device alone. *)
let bare_placements =
  [
    ("on", Bare.on);
    ("split ~axis:0", Bare.split ~axis:0);
    ("memory device", Nx.Placement.device Bare.v (m 3));
    ("io device", Nx.Placement.device Bare.v io);
  ]

(* Each device's window of [x], as its array's shape. *)
let windows x =
  List.map
    (fun a -> Iarray.to_array (Nx_array.Layout.shape (Nx_array.layout a)))
    (Iarray.to_list (Nx.Repr.shards x))

(* [y] lies where [want] lies, holding [want]'s windows and elements. *)
let same_as ~msg dt want y =
  equal ~msg:(msg ^ ": placement") (placement ()) (Nx.placement want)
    (Nx.placement y);
  equal ~msg:(msg ^ ": windows") (list (array int)) (windows want) (windows y);
  equal ~msg:(msg ^ ": elements") (elements dt)
    (Array.to_list (Nx.to_array want))
    (Array.to_list (Nx.to_array y))

(* Each [_at] form on [Bare] is its host form placed there. *)
let test_bare (Pair { dtype; host; at; lies; _ }) =
  let h = host () in
  List.iter
    (fun (msg, p) -> same_as ~msg dtype (Nx.place (lies p) h) (at p))
    bare_placements

let bare_gen =
  Gen.of_list
    ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
    (("split ~axis:1", Bare.split ~axis:1) :: bare_placements)

(* The fills and the forms beside a reference, at every dtype and at
   [Bare]'s placements, are their host values placed there: on no element,
   on one, and over windows a cut leaves uneven or empty. *)
let law_bare ((name, p), (D.Any dt, s)) =
  let axis = if name = "split ~axis:0" then 0 else 1 in
  let cut = String.starts_with ~prefix:"split" name in
  let lacks = cut && Array.length s <= axis in
  cover "cuts an axis the shape lacks" lacks;
  cover "no element" ((not lacks) && numel s = 0);
  cover "one element" ((not lacks) && numel s = 1);
  cover "uneven windows" (cut && (not lacks) && s.(axis) mod 2 = 1);
  cover "an empty window" (cut && (not lacks) && s.(axis) < 2);
  if lacks then invalid ~by:"Nx.zeros_at" (fun () -> Nx.zeros_at p dt s)
  else begin
    let one = D.one dt in
    let zeros = Nx.place p (Nx.zeros dt s) in
    let ones = Nx.place p (Nx.ones dt s) in
    let scalar = Nx.place (Nx.Placement.uncut p) (Nx.scalar dt one) in
    same_as ~msg:"zeros_at" dt zeros (Nx.zeros_at p dt s);
    same_as ~msg:"ones_at" dt ones (Nx.ones_at p dt s);
    same_as ~msg:"full_at" dt ones (Nx.full_at p dt s one);
    same_as ~msg:"scalar_at" dt scalar (Nx.scalar_at p dt one);
    same_as ~msg:"zeros_like" dt zeros (Nx.zeros_like ones);
    same_as ~msg:"ones_like" dt ones (Nx.ones_like zeros);
    same_as ~msg:"full_like" dt ones (Nx.full_like zeros one);
    same_as ~msg:"scalar_like" dt scalar (Nx.scalar_like zeros one)
  end

(* The operations an extent receives while [f] runs, each as its name and
   its count of operands: one of no operands computes, and the first that
   reads an operand ends [f]. *)
exception Reads

let staged f =
  let seen = ref [] in
  let rule (type r) _ ~by (op : r Nx.Prim.t) : r =
    let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
    seen := (Nx.Prim.name op, List.length xs) :: !seen;
    match xs with [] -> Nx.Prim.eval ~by op | _ :: _ -> raise Reads
  in
  (match Nx.Prim.interpret ~name:"test.staged" Extent rule (fun _ -> f ()) with
  | () -> ()
  | exception Reads -> ());
  List.rev !seen

let test_bare_computes () =
  let x = Nx.zeros_at Bare.on Nx.float32 [| 2 |] in
  raises_match
    (Exn.invalid_arg ~substring:"has no kernels to compute Map")
    (fun () -> Nx.add x x);
  named ~by:"Nx.add" (fun () -> ignore (Nx.add x x))

(* On [Bare], an extent that stages eye_at stages no operation that reads an
   operand there. *)
let test_bare_staged () =
  let ops = staged (fun () -> ignore (Nx.eye_at Bare.on Nx.float32 3)) in
  equal
    (list (Windtrap.pair string int))
    []
    (List.filter (fun (_, n) -> n > 0) ops)

(* [f ()] under an interpretation of [reach] whose rule computes what it
   receives now, as a staged program evaluated at once does. *)
let under reach f =
  let rule (type r) _ ~by (op : r Nx.Prim.t) : r = Nx.Prim.eval ~by op in
  Nx.Prim.interpret ~name:"test.now" reach rule (fun _ -> f ())

(* Each [_at] form on [Bare] is its host form placed there under an
   interpretation that reaches its values alone, and under an extent that
   computes what it receives. *)
let test_bare_interpreted (Pair { dtype; host; at; lies; _ }) =
  let p = Bare.on in
  let want = Nx.place (lies p) (host ()) in
  same_as ~msg:"Values" dtype want (under Values (fun () -> at p));
  same_as ~msg:"Extent" dtype want (under Extent (fun () -> at p))

(* An operation whose result needs no element of its operands, as [f] of
   values placed at a placement. *)
type shortcut =
  | Shortcut : {
      name : string;
      dtype : ('w, 'r) D.t;
      f : 'd. 'd Nx.Placement.t -> ('w, 'r, 'd) Nx.t;
    }
      -> shortcut

let shortcut name dtype (f : ('w, 'r) at) = Shortcut { name; dtype; f = f.f }

let shortcuts =
  let f64 = Nx.float64 in
  let ints p =
    Nx.place p (Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |])
  in
  let zeros p s = Nx.place p (Nx.zeros f64 s) in
  [
    shortcut "isnan of int32" Nx.bool { f = (fun p -> Nx.isnan (ints p)) };
    shortcut "isinf of int32" Nx.bool { f = (fun p -> Nx.isinf (ints p)) };
    shortcut "isfinite of bool" Nx.bool
      { f = (fun p -> Nx.isfinite (Nx.place p (Nx.ones Nx.bool [| 3 |]))) };
    shortcut "pad of no element" f64
      { f = (fun p -> Nx.pad [| (1, 2) |] 5. (zeros p [| 0 |])) };
    shortcut "matrix_power a 0" f64
      { f = (fun p -> Nx.matrix_power (zeros p [| 2; 3; 3 |]) 0) };
    shortcut "pinv of an empty matrix" f64
      { f = (fun p -> Nx.pinv (zeros p [| 0; 3 |])) };
    shortcut "matrix_rank of empty matrices" Nx.int64
      { f = (fun p -> Nx.matrix_rank (zeros p [| 2; 3; 0 |])) };
    shortcut "lstsq of an empty matrix" f64
      {
        f =
          (fun p ->
            let a = zeros p [| 3; 0 |] and b = zeros p [| 3 |] in
            let x, _, _, _ = Nx.lstsq f64 a b in
            x);
      };
    shortcut "conv with no tap" f64
      {
        f =
          (fun p ->
            Nx.conv (zeros p [| 1; 4; 3 |]) (zeros p [| 2; 0; 3 |]));
      };
    shortcut "lexsort of rows of no column" Nx.int64
      { f = (fun p -> Nx.lexsort (zeros p [| 3; 0 |])) };
    shortcut "array_equal of two shapes" Nx.bool
      { f = (fun p -> Nx.array_equal (zeros p [| 2 |]) (zeros p [| 3 |])) };
    shortcut "searchsorted among no key" Nx.int64
      {
        f =
          (fun p ->
            Nx.searchsorted ~side:`Left (zeros p [| 0 |]) (zeros p [| 3 |]));
      };
  ]

(* Each raises on [Bare], naming itself and the missing kernels, and on a set
   with kernels is its host result placed there. *)
let test_shortcut (Shortcut { name; dtype; f }) =
  let by = "Nx." ^ List.hd (String.split_on_char ' ' name) in
  raises_match
    (Exn.invalid_arg ~substring:"has no kernels to compute")
    (fun () -> f Bare.on);
  named ~by (fun () -> ignore (f Bare.on));
  same_as ~msg:"with kernels" dtype (Nx.place S2.on (f Nx.Host.on)) (f S2.on)

(* Over operands cut along an axis or read through strided and broadcast
   views, on a set with kernels, each holds its host result's elements and
   lies where the routing rule puts it: a cut of an axis the result keeps
   cuts it alike, and one the result drops closes whole on each device. *)
let test_shortcut_views () =
  let f64 = Nx.float64 in
  let split a x = Nx.place (S2.split ~axis:a) x in
  let whole x = Nx.place S2.on x in
  let cut x = split 0 x in
  let check msg dt at host y =
    equal ~msg:(msg ^ ": placement") (placement ()) at (Nx.placement y);
    equal ~msg (elements dt) (Array.to_list (Nx.to_array host))
      (Array.to_list (Nx.to_array y))
  in
  let z s = Nx.zeros f64 s in
  let a = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let row = Nx.create f64 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let ints = Nx.create Nx.int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let rows = S2.split ~axis:0 in
  check "array_equal, a split operand" Nx.bool S2.on
    (Nx.array_equal (z [| 4 |]) (z [| 3 |]))
    (Nx.array_equal (cut (z [| 4 |])) (cut (z [| 3 |])));
  check "array_equal, uneven windows" Nx.bool S2.on
    (Nx.array_equal (z [| 3 |]) (z [| 2 |]))
    (Nx.array_equal (cut (z [| 3 |])) (cut (z [| 2 |])));
  check "matrix_rank, a split batch" Nx.int64 rows
    (Nx.matrix_rank (z [| 2; 3; 0 |]))
    (Nx.matrix_rank (cut (z [| 2; 3; 0 |])));
  check "pinv, a split batch" f64 rows (Nx.pinv (z [| 2; 0; 3 |]))
    (Nx.pinv (cut (z [| 2; 0; 3 |])));
  check "lexsort, split rows of no column" Nx.int64 rows
    (Nx.lexsort (z [| 4; 0 |]))
    (Nx.lexsort (cut (z [| 4; 0 |])));
  let widths = [| (0, 0); (1, 1); (0, 0) |] in
  check "pad of no element, a cut of a kept axis" f64 (S2.split ~axis:2)
    (Nx.pad widths 5. (z [| 2; 0; 5 |]))
    (Nx.pad widths 5. (split 2 (z [| 2; 0; 5 |])));
  (* The channels, kept after a spatial axis the result changes. *)
  let x = z [| 1; 4; 3 |] and w = z [| 3; 0; 3 |] in
  check "conv with no tap, cut channels" f64 (S2.split ~axis:2) (Nx.conv x w)
    (Nx.conv (split 2 x) (whole w));
  check "array_equal, a transposed operand" Nx.bool S2.on
    (Nx.array_equal (Nx.transpose a) (z [| 3; 3 |]))
    (Nx.array_equal (Nx.transpose (whole a)) (whole (z [| 3; 3 |])));
  check "array_equal, a broadcast operand" Nx.bool S2.on
    (Nx.array_equal (Nx.broadcast_to [| 4; 5 |] row) row)
    (Nx.array_equal (Nx.broadcast_to [| 4; 5 |] (whole row)) (whole row));
  check "isinf, a split transposed operand" Nx.bool (S2.split ~axis:1)
    (Nx.isinf (Nx.transpose ints))
    (Nx.isinf (Nx.transpose (cut ints)))

(* A shortcut whose result's shape is not its operand's loops over one
   position per lane it keeps, however large the operand: here a broadcast
   view of 10^8 elements. *)
let test_shortcut_cost () =
  let row = Nx.place Count.on (Nx.ones Nx.float32 [| 10_000 |]) in
  let big = Nx.broadcast_to [| 10_000; 10_000 |] row in
  C.reset ();
  let y = Nx.array_equal big row in
  less ~msg:"positions" int ~than:10 (C.positions ());
  equal (array bool) [| false |] (Nx.to_array y)

(* Under an extent, a shortcut over a concrete operand on [Bare] reaches
   the interpretation with its operand. *)
let test_shortcut_received () =
  let a = Nx.zeros_at Bare.on Nx.float64 [| 3; 3 |] in
  equal
    (list (Windtrap.pair string int))
    [ ("Map", 1) ]
    (staged (fun () -> ignore (Nx.matrix_power a 0)))

(* A donated value another operation consumed is dead: a predicate of an
   integer value, which reads no element of it, still raises, naming
   itself. *)
let test_predicate_dead () =
  let dead () =
    let x = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |] in
    let _ = Nx.neg (Nx.donate x) in
    x
  in
  invalid ~by:"Nx.isinf" (fun () -> Nx.isinf (dead ()));
  invalid ~by:"Nx.isfinite" (fun () -> Nx.isfinite (dead ()))

let bare =
  group "without kernels"
    [
      cases "each _at form is its host form placed"
        ~name:(fun (Pair { name; _ }) -> name)
        pairs test_bare;
      prop "fills and forms beside a reference are host values placed"
        ~examples:
          [
            (("split ~axis:1", Bare.split ~axis:1), (D.Any D.Float32, [| 3 |]));
            (("on", Bare.on), (D.Any D.Int4, [| 2; 0 |]));
            (("io device", Nx.Placement.device Bare.v io), (D.Any D.Bit, [||]));
            (("split ~axis:0", Bare.split ~axis:0), (D.Any D.Uint8, [| 3 |]));
            (("split ~axis:0", Bare.split ~axis:0), (D.Any D.Bool, [| 1; 2 |]));
          ]
        (Gen.pair bare_gen (Gen.pair dtype shape))
        law_bare;
      cases "an operation needing no element raises"
        ~name:(fun (Shortcut { name; _ }) -> name)
        shortcuts test_shortcut;
      test "a shortcut over cut and strided operands holds its elements"
        test_shortcut_views;
      test "a shortcut's work does not grow with its operand"
        test_shortcut_cost;
      test "under an extent, a shortcut is received with its operand"
        test_shortcut_received;
      test "a predicate of a dead integer value raises" test_predicate_dead;
      test "a computation raises naming the set's missing kernels"
        test_bare_computes;
      test "under an extent, eye_at stages no operation reading an operand"
        test_bare_staged;
      cases "each _at form under an interpretation is its host form placed"
        ~name:(fun (Pair { name; _ }) -> name)
        pairs test_bare_interpreted;
    ]

let () =
  exit (run "nx creation" [ fills; ranges; eye; placements; like; bare ])
