(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Fn = Kaun.Fn

let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let ninf = Float.neg_infinity

(* The unit roundoff of float64. *)
let u = Float.ldexp 1. (-53)
let sigmoid x = 1. /. (1. +. Float.exp (-.x))
let gauss_cdf x = 0.5 *. (1. +. Float.erf (x /. Float.sqrt 2.))
let gauss_pdf x = Float.exp (-.x *. x /. 2.) /. Float.sqrt (2. *. Float.pi)
let sqrt_2_over_pi = Float.sqrt (2. /. Float.pi)
let gelu_ref x = x *. gauss_cdf x

let gelu_approx_ref x =
  0.5 *. x
  *. (1. +. Float.tanh (sqrt_2_over_pi *. (x +. (0.044715 *. x *. x *. x))))

(* [f] applied to [xs] at float64 equals [reference] at each element within
   [bound x]. *)
let agrees ~bound reference f xs =
  let got = Nx.to_array (f (vec xs)) in
  Array.iteri
    (fun i x ->
      equal
        ~msg:(Printf.sprintf "at x = %h" x)
        (float (bound x))
        (reference x) got.(i))
    xs

let bitwise reference f xs =
  equal (array float_exact) (Array.map reference xs) (Nx.to_array (f (vec xs)))

let bitwise_nx reference f xs =
  equal (array float_exact)
    (Nx.to_array (reference (vec xs)))
    (Nx.to_array (f (vec xs)))

let any_floats = Gen.array Gen.any_float
let moderate = Gen.array ~size:(Gen.int_range 1 16) (Gen.float_range (-10.) 10.)

(* Activations *)

let preserve_shape_and_dtype shape =
  let check (type b) (dt : (float, b) Nx.dtype) =
    let x = Nx.zeros dt shape in
    List.iter
      (fun (name, (f : (float, b, Nx.host) Nx.t -> (float, b, Nx.host) Nx.t)) ->
        let y = f x in
        let dtype_name dt = Format.asprintf "%a" Nx.Dtype.pp dt in
        equal ~msg:name string (dtype_name dt) (dtype_name (Nx.dtype y));
        equal ~msg:name (array int) shape (Nx.shape y))
      [
        ("relu", Fn.relu);
        ("leaky_relu", fun x -> Fn.leaky_relu x);
        ("elu", fun x -> Fn.elu x);
        ("gelu", Fn.gelu);
        ("gelu_approx", Fn.gelu_approx);
        ("silu", Fn.silu);
        ("softplus", Fn.softplus);
      ]
  in
  check Nx.float64;
  check Nx.float32;
  check Nx.float16;
  check Nx.bfloat16;
  check Nx.float8_e4m3fn;
  check Nx.float8_e5m2;
  check Nx.float4_e2m1fn

(* Each activation with its derivative, the definition differentiated, where
   fn.mli states it: leaky_relu's at 0 is not stated. *)
let derivatives =
  [
    ("relu", Fn.relu, fun x -> Some (if x > 0. then 1. else 0.));
    ( "leaky_relu",
      (fun x -> Fn.leaky_relu x),
      fun x -> if x = 0. then None else Some (if x > 0. then 1. else 0.01) );
    ( "elu",
      (fun x -> Fn.elu x),
      fun x -> Some (if x > 0. then 1. else Float.exp x) );
    ("gelu", Fn.gelu, fun x -> Some (gauss_cdf x +. (x *. gauss_pdf x)));
    ( "gelu_approx",
      Fn.gelu_approx,
      fun x ->
        let g = sqrt_2_over_pi *. (x +. (0.044715 *. x *. x *. x)) in
        let g' = sqrt_2_over_pi *. (1. +. (3. *. 0.044715 *. x *. x)) in
        let t = Float.tanh g in
        Some ((0.5 *. (1. +. t)) +. (0.5 *. x *. (1. -. (t *. t)) *. g')) );
    ( "silu",
      Fn.silu,
      fun x -> Some (sigmoid x *. (1. +. (x *. (1. -. sigmoid x)))) );
    ("softplus", Fn.softplus, fun x -> Some (sigmoid x));
  ]

(* Each side evaluates a derivative as a sum of at most three terms. At the
   drawn [|x| <= 4] each term is at most [1 + |x|] in magnitude (gelu_approx's
   [x (1 - t^2) g' / 2] is the largest, 5 at [|x| = 4]), and at [|x| = 1000]
   every term but one vanishes. Each term rounds a few products, and exp, erf
   and tanh within 2 ulps and Nx.sigmoid within 3 (nx.mli): within [16u (1 +
   |x|)] of its exact value, so the two sides differ by less than [64u (1 +
   |x|)]. *)
let derivative_bound x = 64. *. u *. (1. +. Float.abs x)

let reverse_mode xs =
  List.iter
    (fun (name, f, f') ->
      let got = Nx.to_array (Rune.grad' (fun x -> Nx.sum (f x)) (vec xs)) in
      Array.iteri
        (fun i x ->
          match f' x with
          | None -> ()
          | Some d ->
              equal
                ~msg:(Printf.sprintf "%s' at x = %h" name x)
                (float (derivative_bound x))
                d got.(i))
        xs)
    derivatives

let forward_points = [| -3.; -0.75; -0.; 0.; 0.5; 2.; 1000.; -1000. |]
let tangent = [| 1.5; -2.; 0.25; 3.; -1.; 0.5; 2.; 1. |]

let forward_mode (name, f, f') =
  let _, got = Rune.jvp' f (vec forward_points) (vec tangent) in
  let got = Nx.to_array got in
  Array.iteri
    (fun i x ->
      let v = tangent.(i) in
      match f' x with
      | None -> ()
      | Some d ->
          equal
            ~msg:(Printf.sprintf "%s' at x = %h along %h" name x v)
            (float (derivative_bound x *. Float.abs v))
            (d *. v) got.(i))
    forward_points

(* Element-wise activations *)

let relu_ref x = if x <= 0. then 0. else x

let leaky_relu_law (slope, xs) =
  let s = Option.value slope ~default:0.01 in
  bitwise
    (fun x -> if x > 0. then x else s *. x)
    (Fn.leaky_relu ?negative_slope:slope)
    xs

(* Where x is finite and not 0, at which fn.mli states no derivative. A zero
   slope's derivative may be either zero: [float]'s smallest tolerance makes
   [-0.] equal [0.] and tells every other pair of floats apart. *)
let leaky_relu_derivative (slope, xs) =
  let s = Option.value slope ~default:0.01 in
  let xs =
    Array.of_list
      (List.filter (fun x -> Float.is_finite x && x <> 0.) (Array.to_list xs))
  in
  equal
    (array (float (Float.ldexp 1. (-1074))))
    (Array.map (fun x -> if x > 0. then 1. else s) xs)
    (Nx.to_array
       (Rune.grad'
          (fun x -> Nx.sum (Fn.leaky_relu ?negative_slope:slope x))
          (vec xs)))

(* ELU against its definition, at one dtype: the value against [x] or [alpha
   (exp x - 1)] computed in float64, and the derivative against float64 finite
   differences of that definition. Both read [x] and [alpha] rounded to the
   dtype, as [Fn.elu] does. The differences stay on [x]'s side of [0], where the
   derivative jumps from [alpha] to [1], and difference [alpha exp x], which has
   the definition's slope below [0] without the cancellation against
   [-alpha]. *)
let law_elu (type b) (dt : (float, b) Nx.dtype) ~rel ~abs (x, alpha) =
  let round v = (Nx.to_array (Nx.create dt [| 1 |] [| v |])).(0) in
  let x = round x and a = round alpha in
  let f v = if v > 0. then v else a *. Float.expm1 v in
  cover "x > 0" (x > 0.);
  cover "x <= 0" (x <= 0.);
  cover "|x| < 1e-3" (Float.abs x < 1e-3);
  cover "exp x overflows float16" (x > 11.1);
  let g v = if x > 0. then v else a *. Float.exp v in
  let h = 1e-5 *. Float.max 1. (Float.abs x) in
  let slope =
    if Float.abs x > 2. *. h then (g (x +. h) -. g (x -. h)) /. (2. *. h)
    else
      (* Second-order one-sided differences. *)
      let s = if x > 0. then 1. else -1. in
      s
      *. ((4. *. g (x +. (s *. h))) -. g (x +. (2. *. s *. h)) -. (3. *. g x))
      /. (2. *. h)
  in
  let t = Nx.create dt [| 1 |] [| x |] in
  let value = Nx.to_array (Fn.elu ~alpha t) in
  let grad = Nx.to_array (Rune.grad' (fun t -> Nx.sum (Fn.elu ~alpha t)) t) in
  equal ~msg:"value" (float_rel ~rel ~abs) (f x) value.(0);
  equal ~msg:"derivative" (float_rel ~rel:(rel +. 1e-6) ~abs) slope grad.(0)

let elu_inputs =
  Gen.pair
    (Gen.frequency
       [ (3, Gen.float_range (-100.) 100.); (1, Gen.float_range (-1e-3) 1e-3) ])
    (Gen.float_range 0.1 3.)

(* [alpha (exp x - 1)] is [-alpha] to the precision of every float dtype at
   [-1e3], and [1e3] is past [exp]'s range in all of them. *)
let elu_examples = [ (0., 1.); (1e3, 1.); (-1e3, 1.); (0., 2.); (-1e3, 2.) ]

let elu_props =
  (* Four units in the last place of a [p]-bit significand, relative, and four
     of its smallest subnormal, absolute: the exponential, the product by
     [alpha] and the rounding of a narrow format's float32 result each round
     once. *)
  let at name dt ~p ~tiny =
    let rel = 4. *. Float.ldexp 1. (-p) and abs = 4. *. tiny in
    prop ~examples:elu_examples
      ("is its definition and derivative at " ^ name)
      elu_inputs (law_elu dt ~rel ~abs)
  in
  [
    at "float64" Nx.float64 ~p:53 ~tiny:(Float.ldexp 1. (-1074));
    at "float32" Nx.float32 ~p:24 ~tiny:(Float.ldexp 1. (-149));
    at "bfloat16" Nx.bfloat16 ~p:8 ~tiny:(Float.ldexp 1. (-133));
    at "float16" Nx.float16 ~p:11 ~tiny:(Float.ldexp 1. (-24));
  ]

let elu_edges alpha =
  let x = vec [| -0.; ninf; Float.nan |] in
  equal (array float_exact)
    [| -0.; -.alpha; Float.nan |]
    (Nx.to_array (Fn.elu ~alpha x))

(* The derivative is [1] above [0], where [exp x] overflows at [1e3] and
   [infinity], and [alpha exp x] at and below it. *)
let elu_derivative_edges alpha =
  let x = vec [| 0.; -0.; 1e3; infinity; -1e3; ninf; Float.nan |] in
  equal (array float_exact)
    [| alpha; alpha; 1.; 1.; 0.; 0.; Float.nan |]
    (Nx.to_array (Rune.grad' (fun x -> Nx.sum (Fn.elu ~alpha x)) x))

(* Each side of a comparison rounds erf or tanh (within 2 ulps), its argument
   (whose error |z| erf'(z) <= 0.49 or |g| tanh'(g) <= 0.45 damps), the sum with
   1 and the products once each: within about 5.6u |x| of the exact value, so
   less than 16u max(1, |x|) from each other. *)
let gelu_bound x = 16. *. u *. Float.max 1. (Float.abs x)

(* The largest difference is 4.73e-4, near x = -2.6989. *)
let gelu_approx_is_close xs =
  let gap =
    Nx.to_array (Nx.abs (Nx.sub (Fn.gelu (vec xs)) (Fn.gelu_approx (vec xs))))
  in
  Array.iteri
    (fun i g ->
      at_most ~msg:(Printf.sprintf "at x = %h" xs.(i)) float_exact ~than:1e-3 g)
    gap

(* The derivative is rune's rule for Nx.softplus; at 0 both kinks of an earlier
   [relu x + log (1 + exp (-|x|))] form gave -1/2. *)
let softplus_at_zero () =
  equal (array float_exact) [| 0.5; 0.5 |]
    (Nx.to_array
       (Rune.grad' (fun x -> Nx.sum (Fn.softplus x)) (vec [| 0.; -0. |])))

(* Sampling masks *)

(* Rows of logits: [shape] is the logits' shape, [param] one entry for every row
   when [per_row], else one scalar. *)
type 'p rows = {
  shape : int array;
  logits : float array;
  param : 'p array;
  per_row : bool;
}

let pp_rows pp_p ppf r =
  let pp_array pp ppf a =
    Format.fprintf ppf "[|%a|]"
      (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") pp)
      (Array.to_list a)
  in
  Format.fprintf ppf "{ shape = %a; logits = %a; param = %a; per_row = %b }"
    (pp_array Format.pp_print_int)
    r.shape
    (pp_array (fun ppf v -> Format.fprintf ppf "%g" v))
    r.logits (pp_array pp_p) r.param r.per_row

let vocab r = r.shape.(Array.length r.shape - 1)
let lead r = Array.sub r.shape 0 (Array.length r.shape - 1)
let row_count r = Array.length r.logits / vocab r
let row r i = Array.sub r.logits (i * vocab r) (vocab r)
let param_of r i = r.param.(if r.per_row then i else 0)

let rows_gen ?(neg_infinity = false) param =
  let open Gen in
  let entry =
    let ints = map float_of_int (int_range (-3) 3) in
    if neg_infinity then frequency [ (9, ints); (1, constant ninf) ] else ints
  in
  let* lead = array ~size:(int_range 0 2) (int_range 1 3) in
  let* vocab = int_range 1 6 in
  let n = Array.fold_left ( * ) 1 lead in
  let* logits = array ~size:(constant (n * vocab)) entry in
  let* per_row = bool in
  let+ param =
    array ~size:(constant (if per_row then n else 1)) (param vocab)
  in
  { shape = Array.append lead [| vocab |]; logits; param; per_row }

let logits_tensor r = Nx.create Nx.float64 r.shape r.logits

let param_tensor dt r =
  if r.per_row then Nx.create dt (lead r) r.param else Nx.scalar dt r.param.(0)

let sorted_desc row =
  let s = Array.copy row in
  Array.sort (fun a b -> Float.compare b a) s;
  s

let keep_from threshold row =
  Array.map (fun v -> if v >= threshold then v else ninf) row

let kept row = Array.fold_left (fun n v -> if v > ninf then n + 1 else n) 0 row

(* keep_top_k *)

(* [k] clamped to [1, vocab] in int64, where [k - 1] would wrap at
   [Int64.min_int]. *)
let clamped k n = Int64.to_int (Int64.max 1L (Int64.min (Int64.of_int n) k))

let top_k_row k row =
  keep_from (sorted_desc row).(clamped k (Array.length row) - 1) row

let top_k_law r =
  let ks = r.param in
  let k i = param_of r i in
  let rows = List.init (row_count r) Fun.id in
  let expected =
    Array.concat (List.map (fun i -> top_k_row (k i) (row r i)) rows)
  in
  cover "k <= 1" (Array.exists (fun k -> k <= 1L) ks);
  cover "k >= vocab" (Array.exists (fun k -> k >= Int64.of_int (vocab r)) ks);
  cover "a tie at the k-th largest keeps more than k"
    (List.exists
       (fun i -> kept (top_k_row (k i) (row r i)) > clamped (k i) (vocab r))
       rows);
  cover "one k per row of several" (r.per_row && row_count r > 1);
  cover "logits of rank 3" (Array.length r.shape = 3);
  let got = Fn.keep_top_k ~k:(param_tensor Nx.int64 r) (logits_tensor r) in
  equal ~msg:"shape" (array int) r.shape (Nx.shape got);
  equal (array float_exact) expected (Nx.to_array got)

let top_k_gen =
  rows_gen ~neg_infinity:true (fun vocab ->
      Gen.frequency
        [
          (8, Gen.map Int64.of_int (Gen.int_range (-1) (vocab + 1)));
          (1, Gen.int64);
        ])
  |> Gen.with_pp (pp_rows (fun ppf k -> Format.fprintf ppf "%LdL" k))

(* The int64 extremes, where [k - 1] wraps. *)
let top_k_examples =
  List.map
    (fun k ->
      {
        shape = [| 4 |];
        logits = [| 1.; 3.; 2.; 3. |];
        param = [| k |];
        per_row = false;
      })
    [ Int64.min_int; Int64.add Int64.min_int 1L; Int64.max_int ]

(* keep_top_p *)

(* The softmax probabilities of [row], in descending order. *)
let sorted_probs row =
  let m = Array.fold_left Float.max ninf row in
  let e = Array.map (fun v -> Float.exp (v -. m)) (sorted_desc row) in
  let z = Array.fold_left ( +. ) 0. e in
  Array.map (fun v -> v /. z) e

let partial_sums probs =
  let acc = ref 0. in
  Array.map
    (fun p ->
      acc := !acc +. p;
      !acc)
    probs

(* The fewest entries whose probabilities sum to at least [p], and every entry
   tied with the last of them. *)
let top_p_row p row =
  let n = Array.length row in
  let sums = partial_sums (sorted_probs row) in
  let rec fewest i =
    if i = n - 1 || sums.(i) >= p then i + 1 else fewest (i + 1)
  in
  let count = if p >= 1. then n else fewest 0 in
  keep_from (sorted_desc row).(count - 1) row

(* A partial sum within 1e-12 of [p] could round to either side of it; the draws
   are continuous, so this discards almost no case. *)
let p_is_clear r =
  List.for_all
    (fun i ->
      let p = param_of r i in
      p >= 1.
      || Array.for_all
           (fun s -> Float.abs (s -. p) > 1e-12)
           (partial_sums (sorted_probs (row r i))))
    (List.init (row_count r) Fun.id)

let top_p_law r =
  assume (p_is_clear r);
  let rows = List.init (row_count r) Fun.id in
  let expected =
    Array.concat (List.map (fun i -> top_p_row (param_of r i) (row r i)) rows)
  in
  cover "p <= 0" (Array.exists (fun p -> p <= 0.) r.param);
  cover "p >= 1" (Array.exists (fun p -> p >= 1.) r.param);
  cover "a row keeps some entries but not all"
    (List.exists
       (fun i ->
         let n = kept (top_p_row (param_of r i) (row r i)) in
         n > 1 && n < vocab r)
       rows);
  cover "one p per row of several" (r.per_row && row_count r > 1);
  let got = Fn.keep_top_p ~p:(param_tensor Nx.float64 r) (logits_tensor r) in
  equal ~msg:"shape" (array int) r.shape (Nx.shape got);
  equal (array float_exact) expected (Nx.to_array got)

let pp_float ppf v = Format.fprintf ppf "%h" v

let top_p_gen =
  rows_gen (fun _ -> Gen.float_range (-0.25) 1.25)
  |> Gen.with_pp (pp_rows pp_float)

let top_p_examples =
  [
    (* An all-equal row: every entry ties with the most likely. *)
    {
      shape = [| 3 |];
      logits = [| 1.; 1.; 1. |];
      param = [| 0.1 |];
      per_row = false;
    };
    (* Entries already removed stay removed. *)
    {
      shape = [| 3 |];
      logits = [| ninf; 2.; 1. |];
      param = [| 0.5 |];
      per_row = false;
    };
    {
      shape = [| 2; 1; 3 |];
      logits = [| 0.; 1.; 2.; 0.; 1.; 2. |];
      param = [| 0.6; 0.95 |];
      per_row = true;
    };
  ]

(* At float32 the cumulative sum of a confident row rounds to one before its
   tail, which a test of the mass alone would drop. *)
let top_p_keeps_confident_tail () =
  let logits = Nx.create Nx.float32 [| 1; 4 |] [| 20.; 0.; 0.; 0. |] in
  equal (array float_exact) [| 20.; 0.; 0.; 0. |]
    (Nx.to_array (Fn.keep_top_p ~p:(Nx.scalar Nx.float32 1.) logits))

(* One program, with the logits and both parameters as its inputs. *)
let masks =
  lazy
    (Rune.jit
       Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
       (fun logits k p -> Fn.keep_top_p ~p (Fn.keep_top_k ~k logits)))

let masks_case =
  let open Gen in
  let small = map float_of_int (int_range (-3) 3) in
  triple
    (array ~size:(constant 10) small)
    (array ~size:(constant 2) (int_range 0 6))
    (array ~size:(constant 2) (float_range 0. 1.))

(* The compiled softmax may add in another order: a probability sum within 1e-5
   of [p] (float32's precision over five terms) could round either way. *)
let compiled_masks (logits, ks, ps) =
  Compiled.here ();
  let r = { shape = [| 2; 5 |]; logits; param = ps; per_row = true } in
  let p_clear i =
    let masked = top_k_row (Int64.of_int ks.(i)) (row r i) in
    ps.(i) >= 1.
    || Array.for_all
         (fun s -> Float.abs (s -. ps.(i)) > 1e-5)
         (partial_sums (sorted_probs masked))
  in
  assume (p_clear 0 && p_clear 1);
  let logits = Nx.create Nx.float32 [| 2; 5 |] logits in
  let k = Nx.create Nx.int64 [| 2 |] (Array.map Int64.of_int ks) in
  let p = Nx.create Nx.float32 [| 2 |] ps in
  equal (array float_exact)
    (Nx.to_array (Fn.keep_top_p ~p (Fn.keep_top_k ~k logits)))
    (Nx.to_array ((Lazy.force masks) logits k p))

let refuses (_, f) = raises_match (fun e -> Exn.invalid_arg e) f

let () =
  let k_of n = Nx.scalar Nx.int64 (Int64.of_int n) in
  let p_of v = Nx.scalar Nx.float64 v in
  exit
    (run "kaun fn"
       [
         group "activations"
           [
             cases "preserve their argument's shape and dtype"
               ~id:(fun s ->
                 "[|"
                 ^ String.concat "; "
                     (Array.to_list (Array.map string_of_int s))
                 ^ "|]")
               [ [||]; [| 0 |]; [| 2; 3 |] ]
               preserve_shape_and_dtype;
             prop "differentiate in reverse mode to their derivatives"
               ~examples:[ [| 0.; -0.; 1000.; -1000. |] ]
               (Gen.array ~size:(Gen.int_range 1 16) (Gen.float_range (-4.) 4.))
               reverse_mode;
             cases "differentiate in forward mode to their derivatives"
               ~id:(fun (name, _, _) -> name)
               derivatives forward_mode;
           ];
         group "relu"
           [
             prop "is zero where x <= 0 and x elsewhere"
               ~examples:[ [| -0.; 0.; Float.nan; infinity; ninf |] ]
               any_floats (bitwise relu_ref Fn.relu);
             test "has derivative 0 at 0" (fun () ->
                 equal (array float_exact) [| 0.; 0. |]
                   (Nx.to_array
                      (Rune.grad'
                         (fun x -> Nx.sum (Fn.relu x))
                         (vec [| 0.; -0. |]))));
           ];
         group "leaky_relu"
           [
             prop "is x where x > 0 and negative_slope * x elsewhere"
               ~examples:[ (None, [| -0.; 0.; Float.nan; ninf |]) ]
               Gen.(pair (option (float_range (-2.) 2.)) any_floats)
               leaky_relu_law;
             prop "has derivative 1 above 0 and negative_slope below"
               ~examples:[ (Some (-0.), [| -1. |]) ]
               Gen.(pair (option (float_range (-2.) 2.)) any_floats)
               leaky_relu_derivative;
           ];
         group "elu"
           (elu_props
           @ [
               cases "is -0. at -0., -alpha at negative infinity and NaN at NaN"
                 ~id:string_of_float [ 1.; 2. ] elu_edges;
               cases "has derivative alpha at 0 and is NaN only at NaN"
                 ~id:string_of_float [ 1.; 2. ] elu_derivative_edges;
               cases "raises on an alpha that is not finite and positive"
                 ~id:string_of_float [ -1.; 0.; -0.; infinity; Float.nan ]
                 (fun alpha ->
                   raises_match
                     (fun e -> Exn.invalid_arg e)
                     (fun () -> Fn.elu ~alpha (vec [| 1. |])));
             ]);
         group "gelu"
           [
             prop "is 0.5 x (1 + erf (x / sqrt 2))"
               ~examples:[ [| 0.; -0.; 1000.; -1000. |] ]
               moderate
               (agrees ~bound:gelu_bound gelu_ref Fn.gelu);
           ];
         group "gelu_approx"
           [
             prop "is the tanh approximation of gelu"
               ~examples:[ [| 0.; -0.; 1000.; -1000. |] ]
               moderate
               (agrees ~bound:gelu_bound gelu_approx_ref Fn.gelu_approx);
             prop "agrees with gelu within 1e-3" ~examples:[ [| -2.6989 |] ]
               (Gen.array ~size:(Gen.int_range 1 16)
                  (Gen.float_range (-20.) 20.))
               gelu_approx_is_close;
           ];
         group "silu"
           [
             prop "is x * Nx.sigmoid x"
               ~examples:[ [| -0.; Float.nan; infinity; ninf; 1000.; -1000. |] ]
               any_floats
               (bitwise_nx (fun x -> Nx.mul x (Nx.sigmoid x)) Fn.silu);
           ];
         group "softplus"
           [
             prop "is Nx.softplus"
               ~examples:[ [| -0.; 1000.; -1000.; Float.nan; infinity; ninf |] ]
               any_floats
               (bitwise_nx Nx.softplus Fn.softplus);
             test "has derivative 1/2 at 0 and -0." softplus_at_zero;
           ];
         group "sampling masks"
           [
             prop "compiled with their parameters as inputs, equal eager masks"
               masks_case compiled_masks;
           ];
         group "keep_top_k"
           [
             prop
               "keeps each row's entries at or above its k-th largest, in its \
                shape"
               ~examples:top_k_examples top_k_gen top_k_law;
             cases "raises on scalar logits or a k of another shape" ~id:fst
               [
                 ( "scalar logits",
                   fun () -> ignore (Fn.keep_top_k ~k:(k_of 1) (p_of 1.)) );
                 ( "k of shape [|3|] for logits of shape [|2; 4|]",
                   fun () ->
                     ignore
                       (Fn.keep_top_k
                          ~k:(Nx.create Nx.int64 [| 3 |] [| 1L; 1L; 1L |])
                          (Nx.zeros Nx.float64 [| 2; 4 |])) );
               ]
               refuses;
           ];
         group "keep_top_p"
           [
             prop
               "keeps the fewest entries reaching p and their ties, in its \
                shape"
               ~examples:top_p_examples top_p_gen top_p_law;
             test "p >= 1 keeps a confident row's tail at float32"
               top_p_keeps_confident_tail;
             cases "raises on scalar logits or a p of another shape" ~id:fst
               [
                 ( "scalar logits",
                   fun () -> ignore (Fn.keep_top_p ~p:(p_of 0.5) (p_of 1.)) );
                 ( "p of shape [|3|] for logits of shape [|2; 4|]",
                   fun () ->
                     ignore
                       (Fn.keep_top_p
                          ~p:(Nx.create Nx.float64 [| 3 |] [| 0.5; 0.5; 0.5 |])
                          (Nx.zeros Nx.float64 [| 2; 4 |])) );
               ]
               refuses;
           ];
       ])
