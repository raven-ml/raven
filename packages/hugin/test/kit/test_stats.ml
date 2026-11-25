(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_kit

let floats = array float_exact
let ints = array int
let host t = Nx.to_array t
let f64 shape a = Nx.create Nx.float64 shape a
let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

(* Floats compared as numbers: [-0.] equals [0.], and [nan] equals [nan]. *)
let number =
  Testable.make ~pp:(fun ppf x -> Format.fprintf ppf "%h" x) ~equal:Float.equal

(* A set of one software device, computed by nx.cpu as the host is. *)
let memory =
  fixture (fun () -> Nx.devices [ Result.get_ok (Rig.software "m0") ])

(* Two software devices, over which data is cut along its first axis. *)
let two =
  fixture (fun () ->
      Nx.devices
        [ Result.get_ok (Rig.software "m1"); Result.get_ok (Rig.software "m2") ])

(* Data *)

type dtype = D : ('a, 'b) Nx.dtype -> dtype

let name (D d) = Nx.Dtype.name d

(* Every float dtype and some integers. *)
let dtypes =
  [
    D Nx.float16;
    D Nx.bfloat16;
    D Nx.float32;
    D Nx.float64;
    D Nx.float8_e4m3fn;
    D Nx.float8_e5m2;
    D Nx.float4_e2m1fn;
    D Nx.int8;
    D Nx.uint8;
    D Nx.int32;
    D Nx.int64;
  ]

type case = {
  dtype : dtype;
  lead : int array; (* The axes before the last. *)
  len : int; (* The length of the last axis. *)
  values : float array;
  bins : int option;
}

let pp_case ppf c =
  let list pp =
    Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ") pp
  in
  Format.fprintf ppf
    "@[<2>{ dtype = %s;@ shape = [%a];@ bins = %s;@ values = [%a] }@]"
    (name c.dtype) (list Format.pp_print_int)
    (Array.to_list c.lead @ [ c.len ])
    (match c.bins with None -> "default" | Some n -> string_of_int n)
    (list (fun ppf x -> Format.fprintf ppf "%h" x))
    (Array.to_list c.values)

let f64_case ?bins values =
  { dtype = D Nx.float64; lead = [||]; len = Array.length values; values; bins }

(* [histogram c] is the histogram of [c]'s values cast to its dtype, and those
   values read back as floats. *)
let histogram c =
  match c.dtype with
  | D d ->
      let v = Nx.cast d (f64 (Array.append c.lead [| c.len |]) c.values) in
      (Stats.histogram ?bins:c.bins v, host (Nx.cast Nx.float64 v))

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

(* [size] floats at most three spacings above a base at [0.], among the
   subnormals, at [2^-1000], where four spacings a bin are narrower than
   [min_float], at [1.], at [2^60] or [-2^60], or at either end of the finite
   floats. *)
let gen_cluster size =
  let open Gen in
  let rec succ k a = if k = 0 then a else succ (k - 1) (Float.succ a) in
  let* base =
    of_list
      [
        0.;
        Float.ldexp 1. (-1000);
        1.;
        Float.ldexp 1. 60;
        -.Float.ldexp 1. 60;
        -.max_float;
        Float.pred (Float.pred max_float);
      ]
  in
  array ~size:(constant size) (map (fun k -> succ k base) (int_range 0 3))

let gen_non_finite = Gen.of_list [ nan; infinity; neg_infinity ]

(* [size] non-finite values but one finite, at any place, if [size] is not
   zero. *)
let gen_lone size =
  let open Gen in
  if size = 0 then constant [||]
  else
    let+ values = array ~size:(constant size) gen_non_finite
    and+ at = int_range 0 (size - 1)
    and+ v = float_range (-100.) 100. in
    Array.mapi (fun i x -> if i = at then v else x) values

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
  let size = Array.fold_left ( * ) 1 lead * len in
  (* Independent draws almost never give samples whose finite values are all
     equal or span too little for their bins, and give one finite value, or a
     group without one, in about one case in twenty-five. A repeated value draws
     the first, a cluster of floats a few spacings apart, kept in float64, the
     second, one finite value among non-finite ones the third, and non-finite
     values alone the fourth. *)
  let+ dtype, values =
    frequency
      [
        (4, pair (of_list dtypes) (array ~size:(constant size) gen_value));
        (1, pair (of_list dtypes) (map (Array.make size) gen_value));
        (1, pair (of_list dtypes) (gen_lone size));
        (1, pair (of_list dtypes) (array ~size:(constant size) gen_non_finite));
        (1, map (fun v -> (D Nx.float64, v)) (gen_cluster size));
      ]
  and+ bins = option (int_range 1 12) in
  { dtype; lead; len; values; bins }

let gen_case = Gen.with_pp pp_case gen_case

(* The rarest label of a law, a dtype, marks about one case in twelve: 300 cases
   miss it with odds under one in a billion. *)
let law_cases = 300

(* The groups of [values], whose last axis is [len] long: one array per index of
   the axes [lead] before the last, empty when [len] is [0]. *)
let groups lead len values =
  Array.init (Array.fold_left ( * ) 1 lead) (fun g ->
      Array.sub values (g * len) len)

let finite a = List.filter Float.is_finite (Array.to_list a)

(* The least and greatest finite values, or [0.] and [1.] if there are none. *)
let ends values =
  match finite values with
  | [] -> (0., 1.)
  | f :: fs -> (List.fold_left Float.min f fs, List.fold_left Float.max f fs)

(* The width [w] that a span of [n] bins from [lo] to [hi] widens to. *)
let least_width n (lo, hi) =
  let m = Float.max (Float.abs lo) (Float.abs hi) in
  let w = Float.of_int n *. Float.max (m *. (4. *. epsilon_float)) min_float in
  if lo = hi then Float.max w 1. else w

let sturges len =
  if len <= 1 then 1
  else 1 + int_of_float (Float.ceil (Float.log2 (Float.of_int len)))

(* The number of the finite [values] in each bin, by the bin's edges, the last
   bin closed. *)
let count_ref x x2 values =
  let n = Array.length x in
  Array.init n (fun j ->
      List.length
        (List.filter
           (fun v -> x.(j) <= v && (v < x2.(j) || (j = n - 1 && v = x2.(j))))
           (finite values)))

(* Results *)

let fields (h : _ Stats.bins) (m : _ Stats.bins) =
  [
    ("x", host h.x, host m.x);
    ("x2", host h.x2, host m.x2);
    ("count", host h.count, host m.count);
    ("density", host h.density, host m.density);
  ]

let on_set c =
  let (module M : Nx.Devices) = memory () in
  let v = f64 (Array.append c.lead [| c.len |]) c.values in
  let m = Stats.histogram ?bins:c.bins (Nx.place M.on v) in
  List.iter
    (fun (msg, p) -> equal (placement ()) ~msg M.on p)
    [
      ("x", Nx.placement m.x);
      ("x2", Nx.placement m.x2);
      ("count", Nx.placement m.count);
      ("density", Nx.placement m.density);
    ];
  List.iter
    (fun (msg, a, b) -> equal floats ~msg a b)
    (fields (Stats.histogram ?bins:c.bins v) m)

(* The edges are whole on each device, as the constants made beside the data
   are. *)
let on_two () =
  let (module M : Nx.Devices) = two () in
  let v = f64 [| 2; 5 |] [| 3.; -1.; 0.5; nan; 2.; 7.; 7.; 1.; -4.; 0. |] in
  let m = Stats.histogram ~bins:3 (Nx.place (M.split ~axis:0) v) in
  equal (placement ()) ~msg:"x" M.on (Nx.placement m.x);
  equal (placement ()) ~msg:"x2" M.on (Nx.placement m.x2);
  List.iter
    (fun (msg, a, b) -> equal floats ~msg a b)
    (fields (Stats.histogram ~bins:3 v) m)

let results =
  group "results"
    [
      prop ~count:law_cases "lie on the data's set, equal to the host's"
        gen_case on_set;
      test "of data cut over two devices lie on both, equal to the host's"
        on_two;
    ]

(* Bins *)

let edge_shapes c =
  let h, _ = histogram c in
  let n = Option.value c.bins ~default:(sturges c.len) in
  equal ints ~msg:"x" [| n |] (Nx.shape h.x);
  equal ints ~msg:"x2" [| n |] (Nx.shape h.x2)

let x2_follows c =
  let h, _ = histogram c in
  let x = host h.x and x2 = host h.x2 in
  let n = Array.length x in
  equal floats (Array.sub x 1 (n - 1)) (Array.sub x2 0 (n - 1))

let value_shapes c =
  let h, _ = histogram c in
  cover "no group" (Array.mem 0 c.lead);
  cover "two axes before the last" (Array.length c.lead = 2);
  let shape = Array.append c.lead [| Nx.numel h.x |] in
  equal ints ~msg:"count" shape (Nx.shape h.count);
  equal ints ~msg:"density" shape (Nx.shape h.density)

let bins =
  group "bins"
    [
      prop ~count:law_cases "x and x2 have the shape [|n|]" gen_case edge_shapes;
      prop ~count:law_cases "x2.{j} is x.{j+1}" gen_case x2_follows;
      prop ~count:law_cases
        "count and density have the data's shape, its last axis the bins"
        gen_case value_shapes;
    ]

(* Histogram *)

let spans_finite c =
  let h, values = histogram c in
  let x = host h.x and x2 = host h.x2 in
  let n = Array.length x in
  let lo, hi = ends values in
  let kept = hi -. lo >= least_width n (lo, hi) in
  cover "no finite value" (finite values = []);
  cover "a span kept" (kept && finite values <> []);
  if kept then equal (pair number number) (lo, hi) (x.(0), x2.(n - 1))

(* The edges of four bins, derived from the rule by hand. *)
let four_bins =
  let big = Float.ldexp 1. 60 in
  (* [max_float] widens by [4 * 4 max_float epsilon], all of it below. *)
  let below_max = max_float -. (max_float *. (16. *. epsilon_float)) in
  [
    ( "least to greatest finite value",
      [| 3.; nan; -2.; infinity; 7.; neg_infinity |],
      (-2., 7.) );
    ("equal values: a half either side", [| 3.; 3.; nan |], (2.5, 3.5));
    ("one value: a half either side", [| -1. |], (-1.5, -0.5));
    ("signed zeros: equal values", [| -0.; 0. |], (-0.5, 0.5));
    ( "2^60 twice: four spacings a bin",
      [| big; big |],
      (big -. 2048., big +. 2048.) );
    ("max_float: shifted below it", [| max_float |], (below_max, max_float));
    ("no finite value: 0 to 1", [| nan; infinity |], (0., 1.));
    ("no value: 0 to 1", [||], (0., 1.));
  ]

let spans_four (_, values, span) =
  let h = Stats.histogram ~bins:4 (f64 [| Array.length values |] values) in
  equal (pair float_exact float_exact) span ((host h.x).(0), (host h.x2).(3))

(* The code makes three rounded operations ([w - (hi - lo)], [lo - d], [hi + d])
   and the law one ([last - first]), each on values below [2 max (m, 1)], so
   each errs by at most [max (m, 1) epsilon]. *)
let widens c =
  let h, values = histogram c in
  let x = host h.x and x2 = host h.x2 in
  let n = Array.length x in
  let lo, hi = ends values in
  let w = least_width n (lo, hi) in
  let first = x.(0) and last = x2.(n - 1) in
  let narrow = hi -. lo < w in
  let inside = first > -.max_float && last < max_float in
  cover "equal ends" (narrow && lo = hi);
  cover "ends apart" (narrow && lo < hi);
  cover "shifted within the finite floats" (narrow && not inside);
  if narrow then begin
    let m = Float.max (Float.abs lo) (Float.abs hi) in
    let within = float (4. *. Float.max m 1. *. epsilon_float) in
    at_most float_exact ~msg:"first edge" ~than:lo first;
    at_least float_exact ~msg:"last edge" ~than:hi last;
    at_least float_exact ~msg:"first edge" ~than:(-.max_float) first;
    at_most float_exact ~msg:"last edge" ~than:max_float last;
    equal within ~msg:"span" w (last -. first);
    if inside then equal within ~msg:"widened equally" (lo -. first) (last -. hi)
  end

(* An edge, a few rounded operations on values within [M], the greater magnitude
   of the two ends, is within 1.5 spacings of [M]: a bin's width is within 3.5
   once subtracted, and the law's width [last / n - first / n], which cannot
   overflow, within 1.5. Spacings are [min_float epsilon] apart below
   [min_float]. *)
let equal_widths c =
  let h, _ = histogram c in
  let x = host h.x and x2 = host h.x2 in
  let n = Float.of_int (Array.length x) in
  let first = x.(0) and last = x2.(Array.length x - 1) in
  let width = (last /. n) -. (first /. n) in
  let spacing =
    epsilon_float
    *. Float.max min_float (Float.max (Float.abs first) (Float.abs last))
  in
  Array.iteri
    (fun j a ->
      equal
        ~msg:(Printf.sprintf "bin %d" j)
        (float (5. *. spacing))
        width
        (x2.(j) -. a))
    x

let positive_widths x x2 =
  Array.iteri
    (fun j a ->
      greater float_exact ~msg:(Printf.sprintf "bin %d" j) ~than:a x2.(j))
    x

let finite_densities density =
  Array.iteri
    (fun j d ->
      satisfies ~claim:"a finite density"
        ~msg:(Printf.sprintf "bin %d" j)
        float_exact Float.is_finite d)
    density

let positive c =
  let h, values = histogram c in
  let x = host h.x and x2 = host h.x2 in
  positive_widths x x2;
  let density = groups c.lead (Array.length x) (host h.density) in
  Array.iteri
    (fun g group -> if finite group <> [] then finite_densities density.(g))
    (groups c.lead c.len values)

(* Inputs whose bins had a width of zero or an infinite density. *)
let few_spacings =
  let big = Float.ldexp 1. 60 and tiny = Float.ldexp 1. (-1000) in
  [
    ( "three int64 at max_int",
      fun () ->
        Stats.histogram
          (Nx.create Nx.int64 [| 3 |] (Array.make 3 Int64.max_int)) );
    ( "2^60 and its successor in 12 bins",
      fun () -> Stats.histogram ~bins:12 (f64 [| 2 |] [| big; Float.succ big |])
    );
    ( "2^-1024, 0 and 0",
      fun () ->
        Stats.histogram (f64 [| 3 |] [| Float.ldexp 1. (-1024); 0.; 0. |]) );
    ( "the least subnormal, 0 and 0",
      fun () ->
        Stats.histogram (f64 [| 3 |] [| Float.ldexp 1. (-1074); 0.; 0. |]) );
    ( "2^-1000 and its successor in 2 bins",
      fun () ->
        Stats.histogram ~bins:2 (f64 [| 2 |] [| tiny; Float.succ tiny |]) );
  ]

let few_spacings_hold (_, histogram) =
  let (h : _ Stats.bins) = histogram () in
  positive_widths (host h.x) (host h.x2);
  finite_densities (host h.density)

let counts c =
  let h, values = histogram c in
  let x = host h.x and x2 = host h.x2 in
  let n = Array.length x in
  List.iter (fun d -> cover (name d) (name c.dtype = name d)) dtypes;
  cover "an empty group" (c.len = 0 && not (Array.mem 0 c.lead));
  cover "a non-finite value"
    (Array.exists (fun v -> not (Float.is_finite v)) values);
  cover "a value on an inner edge"
    (Array.exists
       (fun v -> Array.exists (fun e -> e = v) (Array.sub x 1 (n - 1)))
       values);
  let count = groups c.lead n (host h.count) in
  Array.iteri
    (fun g group ->
      equal floats
        ~msg:(Printf.sprintf "group %d" g)
        (Array.map Float.of_int (count_ref x x2 group))
        count.(g))
    (groups c.lead c.len values)

(* Integers on the inner edges of four bins from 0 to 4. *)
let on_edges =
  {
    dtype = D Nx.int32;
    lead = [||];
    len = 6;
    values = [| 0.; 1.; 1.; 2.; 3.; 4. |];
    bins = Some 4;
  }

(* Two groups of no value, in one default bin. *)
let empty_groups =
  { dtype = D Nx.float64; lead = [| 2 |]; len = 0; values = [||]; bins = None }

(* Two bins over the whole float range, each [max_float] wide. *)
let whole_range = f64_case ~bins:2 [| -.max_float; max_float |]

(* One bin over the whole float range, wider than [max_float]. *)
let wider_than_max = f64_case ~bins:1 [| -.max_float; max_float |]

(* Either side, [count / total / width] or [count / (total * width)], rounds
   twice, so the two agree within [2 epsilon], or within a subnormal spacing
   where densities are that small. *)
let quotient =
  float_rel ~rel:(2. *. epsilon_float) ~abs:(Float.ldexp 1. (-1074))

let density_is c =
  let h, values = histogram c in
  let x = host h.x and x2 = host h.x2 in
  let n = Array.length x in
  let count = groups c.lead n (host h.count)
  and density = groups c.lead n (host h.density) in
  Array.iteri
    (fun g group ->
      let total = Float.of_int (List.length (finite group)) in
      cover "a group with a finite value" (total > 0.);
      if total > 0. then
        Array.iteri
          (fun j d ->
            equal quotient
              ~msg:(Printf.sprintf "group %d, bin %d" g j)
              (count.(g).(j) /. total /. (x2.(j) -. x.(j)))
              d)
          density.(g))
    (groups c.lead c.len values)

(* Each of the [n] terms [d * w] rounds three times on its way, [1.5 epsilon] of
   the term, and where [d] is subnormal errs by at most half a subnormal spacing
   times a width below [2^1024], [2 epsilon]. Each addition adds half an
   [epsilon]. So the sum is within [4 n epsilon] of one. *)
let sum_to_one c =
  let h, values = histogram c in
  let x = host h.x and x2 = host h.x2 in
  let n = Array.length x in
  let widths = Array.init n (fun j -> x2.(j) -. x.(j)) in
  let wide = Array.exists (fun w -> w = infinity) widths in
  cover "a bin wider than max_float" wide;
  let density = groups c.lead n (host h.density) in
  Array.iteri
    (fun g group ->
      let msg = Printf.sprintf "group %d" g in
      if finite group = [] then ()
      else if wide then
        Array.iteri
          (fun j d -> if widths.(j) = infinity then equal ~msg float_exact 0. d)
          density.(g)
      else
        let sum = ref 0. in
        Array.iteri (fun j d -> sum := !sum +. (d *. widths.(j))) density.(g);
        equal ~msg (float (4. *. Float.of_int n *. epsilon_float)) 1. !sum)
    (groups c.lead c.len values)

let no_finite_nan c =
  let h, values = histogram c in
  let n = Nx.numel h.x in
  let density = groups c.lead n (host h.density) in
  Array.iteri
    (fun g group ->
      cover "a group without a finite value" (finite group = []);
      cover "an empty group" (group = [||]);
      if finite group = [] then
        equal
          ~msg:(Printf.sprintf "group %d" g)
          floats (Array.make n nan) density.(g))
    (groups c.lead c.len values)

let sturges_rows =
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

let histograms =
  group "histogram"
    [
      prop ~count:law_cases
        "the bins span the least to the greatest finite value, or 0 to 1"
        gen_case spans_finite;
      cases "four bins span as the rule states"
        ~id:(fun (n, _, _) -> n)
        four_bins spans_four;
      prop ~count:law_cases
        ~examples:[ f64_case ~bins:4 [| max_float |] ]
        "a narrow span widens to w equally on either side, within the floats"
        gen_case widens;
      prop ~count:law_cases ~examples:[ whole_range ]
        "the bins have equal widths" gen_case equal_widths;
      prop ~count:law_cases
        "every bin has a positive width and a finite density" gen_case positive;
      cases "a span a few spacings wide keeps bins wide and densities finite"
        ~id:fst few_spacings few_spacings_hold;
      prop ~count:law_cases ~examples:[ on_edges; empty_groups ]
        "a bin counts the finite values from its left edge, the last closed"
        gen_case counts;
      prop ~count:law_cases ~examples:[ whole_range ]
        "the density is the count over the total and the bin's width" gen_case
        density_is;
      prop ~count:law_cases
        ~examples:[ whole_range; wider_than_max ]
        "densities times widths sum to one, a bin wider than max_float at 0"
        gen_case sum_to_one;
      prop ~count:law_cases ~examples:[ empty_groups ]
        "a histogram with no finite value has the density nan" gen_case
        no_finite_nan;
      cases "bins defaults to Sturges' rule"
        ~id:(fun (len, n) -> Printf.sprintf "%d values make %d bins" len n)
        sturges_rows
        (fun (len, n) ->
          let h = Stats.histogram (Nx.zeros Nx.float64 [| 2; len |]) in
          equal int n (Nx.numel h.x));
      cases "raises if bins < 1" ~id:string_of_int [ 0; -1; min_int ]
        (fun bins ->
          raises_match Exn.invalid_arg (fun () ->
              Stats.histogram ~bins (Nx.zeros Nx.float64 [| 3 |])));
      test "raises on a scalar" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Stats.histogram (Nx.scalar Nx.float64 1.)));
      test "raises on complex values" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Stats.histogram (Nx.zeros Nx.complex64 [| 3 |])));
      test "raises on booleans" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Stats.histogram (Nx.zeros Nx.bool [| 3 |])));
    ]

let () = exit (run "Stats" [ results; bins; histograms ])
