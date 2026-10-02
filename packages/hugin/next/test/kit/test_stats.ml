(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_kit

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f
let floats = array float_exact

(* Floats compared as numbers: [-0.] equals [0.], and [nan] equals [nan]. *)
let number =
  Testable.make ~pp:(fun ppf x -> Format.fprintf ppf "%h" x) ~equal:Float.equal

let ints = array int
let host t = Nx.to_array t
let f64 shape a = Nx.create Nx.float64 shape a

(* Data *)

(* The dtypes the data is drawn in: every float dtype and some integers. *)
type dtype = F16 | Bf16 | F32 | F64 | F8_e4m3 | F8_e5m2 | I8 | U8 | I32 | I64

let dtypes = [ F16; Bf16; F32; F64; F8_e4m3; F8_e5m2; I8; U8; I32; I64 ]

let dtype_name = function
  | F16 -> "float16"
  | Bf16 -> "bfloat16"
  | F32 -> "float32"
  | F64 -> "float64"
  | F8_e4m3 -> "float8_e4m3"
  | F8_e5m2 -> "float8_e5m2"
  | I8 -> "int8"
  | U8 -> "uint8"
  | I32 -> "int32"
  | I64 -> "int64"

(* [histogram ?bins d v] is the histogram of [v] cast to [d], with the values it
   holds read back as floats. *)
let histogram ?bins d v =
  let run t = (Stats.histogram ?bins t, host (Nx.cast Nx.float64 t)) in
  match d with
  | F16 -> run (Nx.cast Nx.float16 v)
  | Bf16 -> run (Nx.cast Nx.bfloat16 v)
  | F32 -> run (Nx.cast Nx.float32 v)
  | F64 -> run v
  | F8_e4m3 -> run (Nx.cast Nx.float8_e4m3 v)
  | F8_e5m2 -> run (Nx.cast Nx.float8_e5m2 v)
  | I8 -> run (Nx.cast Nx.int8 v)
  | U8 -> run (Nx.cast Nx.uint8 v)
  | I32 -> run (Nx.cast Nx.int32 v)
  | I64 -> run (Nx.cast Nx.int64 v)

type case = {
  dtype : dtype;
  lead : int array; (* The axes before the last. *)
  len : int; (* The length of the last axis. *)
  values : float array;
  bins : int option;
}

let pp_case ppf c =
  Format.fprintf ppf
    "@[<2>{ dtype = %s;@ shape = [%a];@ bins = %s;@ values = [%a] }@]"
    (dtype_name c.dtype)
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    (Array.to_list c.lead @ [ c.len ])
    (match c.bins with None -> "default" | Some n -> string_of_int n)
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf x -> Format.fprintf ppf "%h" x))
    (Array.to_list c.values)

(* Values on a small range, small integers that fall on the edges of bins that
   divide their span, the signed zeros and the non-finite values. *)
let gen_value =
  Gen.frequency
    [
      (6, Gen.float_range (-100.) 100.);
      (3, Gen.map Float.of_int (Gen.int_range (-4) 8));
      (1, Gen.of_list [ 0.; -0. ]);
      (1, Gen.of_list [ nan; infinity; neg_infinity ]);
    ]

let gen_case =
  let open Gen in
  let* lead =
    frequency
      [
        (3, constant [||]);
        (2, map (fun a -> [| a |]) (int_range 0 4));
        ( 1,
          map (fun (a, b) -> [| a; b |]) (pair (int_range 1 3) (int_range 0 3))
        );
      ]
  in
  let* len = frequency [ (1, int_range 0 2); (4, int_range 3 40) ] in
  let groups = Array.fold_left ( * ) 1 lead in
  let+ dtype = of_list dtypes
  and+ values = array ~size:(constant (groups * len)) gen_value
  and+ bins = option (int_range 1 12) in
  { dtype; lead; len; values; bins }

let gen_case = Gen.with_pp pp_case gen_case

(* The groups of [values]: one array per index of the axes before the last. *)
let groups len values =
  if len = 0 then [||]
  else
    Array.init
      (Array.length values / len)
      (fun g -> Array.sub values (g * len) len)

let finite a = List.filter Float.is_finite (Array.to_list a)

(* The reference: the number of the finite [values] in each bin, by the bin's
   edges, the last bin closed. *)
let count_ref x x2 values =
  let n = Array.length x in
  Array.init n (fun j ->
      List.length
        (List.filter
           (fun v -> x.(j) <= v && (v < x2.(j) || (j = n - 1 && v = x2.(j))))
           (finite values)))

let run_case c =
  let shape = Array.append c.lead [| c.len |] in
  let h, values = histogram ?bins:c.bins c.dtype (f64 shape c.values) in
  (h, values)

(* Laws *)

let law_edges c =
  let h, values = run_case c in
  let x = host h.x and x2 = host h.x2 in
  let n = Array.length x in
  let expected_n =
    match c.bins with
    | Some n -> n
    | None ->
        if c.len <= 1 then 1
        else 1 + int_of_float (Float.ceil (Float.log2 (Float.of_int c.len)))
  in
  equal int ~msg:"bins" expected_n n;
  equal floats ~msg:"x2 follows x"
    (Array.sub x 1 (n - 1))
    (Array.sub x2 0 (n - 1));
  let fs = finite values in
  let lo, hi =
    match fs with
    | [] -> (0., 1.)
    | f :: fs ->
        let lo = List.fold_left Float.min f fs
        and hi = List.fold_left Float.max f fs in
        if lo < hi then (lo, hi)
        else
          let h = Float.max 0.5 (Float.abs lo *. epsilon_float) in
          (Float.max (lo -. h) (-.max_float), Float.min (hi +. h) max_float)
  in
  cover "no finite value" (fs = []);
  cover "one finite value" (match fs with [ _ ] -> true | _ -> false);
  cover "all equal"
    (fs <> [] && List.for_all (fun v -> v = List.hd fs) fs && List.length fs > 1);
  equal number ~msg:"first edge" lo x.(0);
  equal number ~msg:"last edge" hi x2.(n - 1);
  let width = (hi -. lo) /. Float.of_int n in
  let eps = 4. *. epsilon_float *. Float.max (Float.abs lo) (Float.abs hi) in
  Array.iteri
    (fun j a ->
      equal (float eps)
        ~msg:(Printf.sprintf "width of bin %d" j)
        width
        (x2.(j) -. a))
    x

let law_counts c =
  let h, values = run_case c in
  let x = host h.x and x2 = host h.x2 in
  let n = Array.length x in
  equal ints ~msg:"shape" (Array.append c.lead [| n |]) (Nx.shape h.count);
  let count = groups n (host h.count) in
  Array.iteri
    (fun g group ->
      cover "a non-finite value"
        (Array.exists (fun v -> not (Float.is_finite v)) group);
      cover "a value on an inner edge"
        (Array.exists
           (fun v -> Array.exists (fun e -> e = v) (Array.sub x 1 (n - 1)))
           group);
      let msg = Printf.sprintf "group %d" g in
      equal floats ~msg
        (Array.map Float.of_int (count_ref x x2 group))
        count.(g);
      equal float_exact ~msg:(msg ^ ", total")
        (Float.of_int (List.length (finite group)))
        (Array.fold_left ( +. ) 0. count.(g)))
    (groups c.len values)

let law_density c =
  let h, values = run_case c in
  let x = host h.x and x2 = host h.x2 in
  let n = Array.length x in
  equal ints ~msg:"shape" (Array.append c.lead [| n |]) (Nx.shape h.density);
  let density = groups n (host h.density) and count = groups n (host h.count) in
  Array.iteri
    (fun g group ->
      let msg = Printf.sprintf "group %d" g in
      let total = Float.of_int (List.length (finite group)) in
      if total = 0. then (
        cover "a group without a finite value" true;
        equal floats ~msg (Array.make n nan) density.(g))
      else (
        Array.iteri
          (fun j d ->
            equal
              (float_rel ~rel:1e-12 ~abs:0.)
              ~msg:(Printf.sprintf "%s, bin %d" msg j)
              (count.(g).(j) /. total /. (x2.(j) -. x.(j)))
              d)
          density.(g);
        let integral = ref 0. in
        Array.iteri
          (fun j d -> integral := !integral +. (d *. (x2.(j) -. x.(j))))
          density.(g);
        equal (float 1e-12) ~msg:(msg ^ ", integral") 1. !integral))
    (groups c.len values)

(* Each group's counts are those of [Nx.histogram] over its values alone, with
   the edges the histogram shares. *)
let law_batch c =
  let h, values = run_case c in
  let n = Nx.numel h.x in
  let e = Nx.concatenate ~axis:0 [ h.x; Nx.slice [ Nx.R (n - 1, n) ] h.x2 ] in
  let count = groups n (host h.count) in
  Array.iteri
    (fun g group ->
      let row = Nx.histogram [ (e, f64 [| c.len |] group) ] in
      equal floats ~msg:(Printf.sprintf "group %d" g) (host row) count.(g))
    (groups c.len values)

let laws =
  group "laws"
    [
      prop "edges span the finite values in bins of equal width" gen_case
        (fun c ->
          List.iter (fun d -> cover (dtype_name d) (c.dtype = d)) dtypes;
          law_edges c);
      prop "each bin counts the finite values between its edges" gen_case
        law_counts;
      prop "densities times widths sum to one" gen_case law_density;
      prop "groups count as Nx.histogram row by row" gen_case law_batch;
    ]

(* Cases *)

let span name values (lo, hi) =
  test name (fun () ->
      let h = Stats.histogram ~bins:4 (f64 [| Array.length values |] values) in
      let x = host h.x and x2 = host h.x2 in
      equal float_exact ~msg:"first edge" lo x.(0);
      equal float_exact ~msg:"last edge" hi x2.(3))

let spans =
  let big = Float.ldexp 1. 60 in
  let below_max = max_float -. (max_float *. epsilon_float) in
  group "spans"
    [
      span "spans the least to the greatest finite value"
        [| 3.; nan; -2.; infinity; 7.; neg_infinity |]
        (-2., 7.);
      span "equal values span a half either side" [| 3.; 3.; nan |] (2.5, 3.5);
      span "one value spans a half either side" [| -1. |] (-1.5, -0.5);
      span "signed zeros are equal values" [| -0.; 0. |] (-0.5, 0.5);
      span "a large value spans the gap to its neighbours" [| big; big |]
        (big -. 256., big +. 256.);
      span "the largest float spans within the finite floats" [| max_float |]
        (below_max, max_float);
      span "no finite value spans zero to one" [| nan; infinity |] (0., 1.);
      span "no value spans zero to one" [||] (0., 1.);
    ]

let sturges =
  cases
    ~name:(fun (len, n) -> Printf.sprintf "%d values make %d bins" len n)
    "default bins"
    [
      (0, 1);
      (1, 1);
      (2, 2);
      (3, 3);
      (4, 3);
      (5, 4);
      (8, 4);
      (9, 5);
      (1024, 11);
      (1025, 12);
    ]
    (fun (len, n) ->
      let h = Stats.histogram (Nx.zeros Nx.float64 [| 2; len |]) in
      equal int n (Nx.numel h.x))

let shapes =
  group "shapes"
    [
      test "an empty axis gives empty histograms with density nan" (fun () ->
          let h = Stats.histogram (Nx.zeros Nx.float64 [| 2; 0 |]) in
          equal floats [| 0.; 0. |] (host h.count);
          equal floats [| nan; nan |] (host h.density);
          equal floats [| 0. |] (host h.x);
          equal floats [| 1. |] (host h.x2));
      test "no group gives histograms of no row" (fun () ->
          let h = Stats.histogram ~bins:3 (Nx.zeros Nx.float64 [| 0; 4; 5 |]) in
          equal ints [| 0; 4; 3 |] (Nx.shape h.count);
          equal ints [| 0; 4; 3 |] (Nx.shape h.density);
          equal floats [| 0.; 1. /. 3.; 2. /. 3. |] (host h.x));
      test "the last axis becomes the bins" (fun () ->
          let h = Stats.histogram ~bins:5 (Nx.zeros Nx.float32 [| 2; 3; 7 |]) in
          equal ints [| 2; 3; 5 |] (Nx.shape h.count);
          equal ints [| 5 |] (Nx.shape h.x);
          equal ints [| 5 |] (Nx.shape h.x2));
      test "a bin wider than max_float has density zero" (fun () ->
          let h =
            Stats.histogram ~bins:1 (f64 [| 2 |] [| -.max_float; max_float |])
          in
          equal floats [| 2. |] (host h.count);
          equal floats [| 0. |] (host h.density));
      test "two bins over the whole float range have finite widths" (fun () ->
          let h =
            Stats.histogram ~bins:2 (f64 [| 2 |] [| -.max_float; max_float |])
          in
          equal floats [| -.max_float; 0. |] (host h.x);
          equal floats [| 1.; 1. |] (host h.count);
          equal floats [| 0.5 /. max_float; 0.5 /. max_float |] (host h.density));
      test "integers count by value" (fun () ->
          let v = Nx.create Nx.int32 [| 6 |] [| 0l; 1l; 1l; 2l; 3l; 4l |] in
          let h = Stats.histogram ~bins:4 v in
          equal floats [| 0.; 1.; 2.; 3. |] (host h.x);
          equal floats [| 1.; 2.; 1.; 2. |] (host h.count));
    ]

let errors =
  group "errors"
    [
      test "raises on zero bins" (fun () ->
          invalid (fun () ->
              Stats.histogram ~bins:0 (Nx.zeros Nx.float64 [| 3 |])));
      test "raises on negative bins" (fun () ->
          invalid (fun () ->
              Stats.histogram ~bins:(-1) (Nx.zeros Nx.float64 [| 3 |])));
      test "raises on a scalar" (fun () ->
          invalid (fun () -> Stats.histogram (Nx.scalar Nx.float64 1.)));
      test "raises on complex values" (fun () ->
          invalid (fun () -> Stats.histogram (Nx.zeros Nx.complex64 [| 3 |])));
      test "raises on booleans" (fun () ->
          invalid (fun () -> Stats.histogram (Nx.zeros Nx.bool [| 3 |])));
    ]

let () = exit (run "Stats" [ laws; spans; sturges; shapes; errors ])
