(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's special functions, in nx.mli's order: each against correctly rounded
   references (golden/special/, written by gen/special.py) at the bound nx.mli
   states per function and dtype, at every narrow float by the rule nx.mli
   states for them, at the edges it states, and by the laws it keeps. An ulp is
   counted as nx.mli counts it (Nx_support.ulps). *)

open Windtrap
module D = Nx_array.Dtype

type 's v = (float, 's, Nx.host) Nx.t

let values = Nx.to_array
let nan = Float.nan
let inf = Float.infinity
let eps = epsilon_float

type f = { f : 's. 's v array -> 's v }
type u = { u : 's. 's v -> 's v }
type b = { b : 's. 's v -> 's v -> 's v }
type c = { c : 's. 's v -> 's v -> 's v -> 's v }

let one { u } = { f = (fun a -> u a.(0)) }
let two { b } = { f = (fun a -> b a.(0) a.(1)) }
let three { c } = { f = (fun a -> c a.(0) a.(1) a.(2)) }

(* A float dtype: float32 or float64 for the goldens and the edges, the narrow
   ones for their sweeps. *)
type dtype = Dtype : (float, 's) Nx.dtype -> dtype

let dtype_id (Dtype dt) = D.name dt
let wide = [ Dtype Nx.float32; Dtype Nx.float64 ]

let narrows =
  [
    Dtype Nx.float16;
    Dtype Nx.bfloat16;
    Dtype Nx.float8_e4m3fn;
    Dtype Nx.float8_e5m2;
    Dtype Nx.float4_e2m1fn;
  ]

let pp_args ppf args =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
    (fun ppf x -> Format.fprintf ppf "%h" x)
    ppf (Array.to_list args)

(* Bounds *)

(* The forms of a bound nx.mli states: [k] ulps; [2k (1 + |log f|) + 4] ulps,
   the exponential of a logarithm within [k] ulps; [k] ulps, absolute near
   zeros, or [|f̂ - f| <= a ε] where [|f| < 1]; [k] ulps or [|f̂ - f| <= k ε
   scale]; an inverse's [k] ulps plus its condition number times its forward
   function's bound. *)
type bound =
  | Ulps of int
  | Log_ulps of int
  | Near_zeros of int * int
  | Scaled of int
  | Inverse of int * int

let pp_bound ppf = function
  | Ulps k -> Format.fprintf ppf "%d ulps" k
  | Near_zeros (k, a) ->
      Format.fprintf ppf "%d ulps, or %d eps where below 1" k a
  | Scaled k -> Format.fprintf ppf "%d ulps, or %d eps times the scale" k k
  | Inverse (k, f) -> Format.fprintf ppf "%d + kappa * %d ulps" k f
  | Log_ulps k -> Format.fprintf ppf "%d (1 + |log f|) + 4 ulps" (2 * k)

(* [Log_ulps k]'s bound at [f]. *)
let log_ulps k f =
  (2. *. Float.of_int k *. (1. +. Float.abs (Stdlib.log (Float.abs f)))) +. 4.

let everywhere b _ = b

(* A bound for [x > 0] and another below. *)
let by_sign ~positive ~negative args =
  if args.(0) < 0. then negative else positive

(* Goldens *)

(* Whether [got] meets [bound] against the correctly rounded [row]: exactly
   where the value is an exact infinity or NaN. An overflow, the infinity one
   float past the greatest, and a zero, beside the least subnormal of its sign,
   are held by rank to the bound in ulps, as nx.mli counts them. A value that
   rounds to zero is held as the least subnormal is: its logarithm bounds
   [Log_ulps]. *)
let within (type s) (dt : (float, s) Nx.dtype) bound (row : Nx_support.row) got
    =
  let f32 = D.bits dt = 32 in
  let eps = if f32 then 0x1p-23 else 0x1p-52 in
  let r = row.value in
  let d = Nx_support.ulps dt r got in
  let error = Float.abs (got -. r) in
  let log_ulps k r =
    log_ulps k (if r = 0. then if f32 then 0x1p-149 else 0x1p-1074 else r)
  in
  let in_ulps =
    match bound with
    | Ulps k | Near_zeros (k, _) | Scaled k -> Float.of_int k
    | Log_ulps k -> log_ulps k r
    | Inverse (k, forward) ->
        Float.of_int k +. (row.extra *. Float.of_int forward)
  in
  if (not (Float.is_finite r)) && not row.overflow then d = 0.
  else if row.overflow then d <= in_ulps
  else
    d <= in_ulps
    ||
    match bound with
    | Near_zeros (_, a) -> Float.abs r < 1. && error <= Float.of_int a *. eps
    | Scaled k -> error <= Float.of_int k *. eps *. row.extra
    | Ulps _ | Log_ulps _ | Inverse _ -> false

(* [golden ?keep ~bound name f (Dtype dt)] holds [f] to [bound args] at every
   row of [name]'s golden at [dt] that [keep] keeps, by column in one call, and
   fails listing the rows that miss, worst first. *)
let golden ?(keep = fun _ rows -> rows) ~bound name { f } (Dtype dt) =
  let rows = keep (Dtype dt) (Nx_support.golden name dt) in
  let n = List.length rows in
  let arity = Array.length (List.hd rows).args in
  let column i =
    Nx.create dt [| n |]
      (Array.of_list (List.map (fun r -> r.Nx_support.args.(i)) rows))
  in
  let got = values (Nx.cast Nx.float64 (f (Array.init arity column))) in
  let misses =
    List.concat
      (List.mapi
         (fun i (row : Nx_support.row) ->
           let b = bound row.args in
           if within dt b row got.(i) then []
           else [ (Nx_support.ulps dt row.value got.(i), row, got.(i), b) ])
         rows)
  in
  let misses =
    List.sort (fun (a, _, _, _) (b, _, _, _) -> compare b a) misses
  in
  if misses <> [] then
    let pp_miss ppf (d, (row : Nx_support.row), got, b) =
      Format.fprintf ppf "at (%a): %h, not %h (%g ulps), bound %a" pp_args
        row.args got row.value d pp_bound b;
      if Float.is_finite row.extra then
        Format.fprintf ppf " (scale %g)" row.extra
    in
    failf "%d of %d rows miss their bound:@\n%a" (List.length misses) n
      (Format.pp_print_list pp_miss)
      (List.filteri (fun i _ -> i < 8) misses)

(* Narrow floats

   A narrow float computes at float32 and rounds once. Its reference is the
   float64 result, whose own bound the goldens hold: rounded to the narrow dtype
   it is the correctly rounded value, or one of its neighbours where the float64
   result lies within its bound of a narrow midpoint. *)

(* An inverse's condition number at its arguments and its float64 result. *)
type kappa = { k : 's. 's v array -> 's v -> 's v }

(* Every float of a narrow dtype, and the ranks of a value of it: [+0] at 0,
   [-0] at -1. *)
let every (type s) (dt : (float, s) Nx.dtype) : s v =
  let every =
    match D.bits dt with
    | 4 -> Nx.bitcast dt (Nx.arange Nx.uint4 0 16 1)
    | 8 -> Nx.bitcast dt (Nx.arange Nx.uint8 0 256 1)
    | _ -> Nx.bitcast dt (Nx.arange Nx.uint16 0 65536 1)
  in
  Nx.place Nx.Host.on every

let ranks (type s) (x : s v) =
  let width = D.bits (Nx.dtype x) in
  let codes =
    match width with
    | 4 -> values (Nx.bitcast Nx.uint4 x)
    | 8 -> values (Nx.bitcast Nx.uint8 x)
    | _ -> values (Nx.bitcast Nx.uint16 x)
  in
  let sign = 1 lsl (width - 1) in
  Array.map
    (fun b -> if b land sign = 0 then b else -(b land (sign - 1)) - 1)
    codes

(* [hold dt ~bound ?kappa f args n] holds [f] at [args], [n] floats of the
   narrow dtype [dt] each: within one ulp of the reference where the bound is in
   ulps, within the float32 bound plus half an ulp of [dt] at the result of the
   reference rounded to float32 where it is absolute, a zero with the
   reference's sign, and a non-finite value exactly. *)
let hold (type s) (dt : (float, s) Nx.dtype) ~bound ?kappa { f }
    (args : s v array) n =
  let got = f args in
  let args64 = Array.map (Nx.cast Nx.float64) args in
  let v = f args64 in
  let xs = Array.map values args64 in
  let at i = Array.map (fun a -> a.(i)) xs in
  let scales =
    match kappa with
    | Some { k } -> values (k args64 v)
    | None -> Array.make n Float.nan
  in
  let vs = values v in
  (* The float64 bound, in ulps, at each input. *)
  let bounds =
    Array.init n (fun i ->
        match bound (at i) with
        | Ulps k | Near_zeros (k, _) | Scaled k -> Float.of_int k
        | Log_ulps k -> log_ulps k vs.(i)
        | Inverse (k, fw) -> Float.of_int k +. (scales.(i) *. Float.of_int fw))
  in
  let slack =
    Nx.create Nx.float64 [| n |]
      (Array.map
         (fun b -> if Float.is_finite b then b *. 0x1p-52 else 0.)
         bounds)
  in
  let rounded k = Nx.cast dt (Nx.mul v (Nx.add_s (Nx.mul_s slack k) 1.)) in
  let candidates = List.map (fun k -> ranks (rounded k)) [ 0.; -1.; 1. ] in
  let got_ranks = ranks got in
  let gs = values (Nx.cast Nx.float64 got) in
  (* What the float64 result is in the narrow dtype: a dtype with no infinities
     has no other value for one. *)
  let cs = values (Nx.cast Nx.float64 (Nx.cast dt v)) in
  let misses = ref [] in
  for i = 0 to n - 1 do
    let g = gs.(i) and v = vs.(i) in
    let d =
      if Float.is_nan v || Float.is_nan g then
        if Float.is_nan v && Float.is_nan g then 0 else max_int
      else
        List.fold_left
          (fun d c -> Int.min d (abs (got_ranks.(i) - c.(i))))
          max_int candidates
    in
    (* An absolute bound holds at float32, from the value rounded there. *)
    let r = Int32.float_of_bits (Int32.bits_of_float v) in
    let h = 0.5 *. Nx_support.ulp dt g in
    let absolute =
      match bound (at i) with
      | Ulps _ | Log_ulps _ | Inverse _ -> false
      | Near_zeros (_, a) ->
          Float.abs r < 1.
          && Float.abs (g -. r) <= (Float.of_int a *. 0x1p-23) +. h
      | Scaled k ->
          Float.abs (g -. r) <= (Float.of_int k *. 0x1p-23 *. scales.(i)) +. h
    in
    let ok =
      if not (Float.is_finite v) then
        (Float.is_nan g && Float.is_nan cs.(i)) || g = cs.(i)
      else
        d <= 1
        && (g <> 0. || cs.(i) <> 0. || Float.sign_bit g = Float.sign_bit cs.(i))
        || absolute
    in
    if not ok then misses := (at i, g, v) :: !misses
  done;
  match !misses with
  | [] -> ()
  | misses ->
      failf "%d of %d inputs miss:@\n%a" (List.length misses) n
        (Format.pp_print_list (fun ppf (x, g, v) ->
             Format.fprintf ppf "at (%a): %h, float64 %h" pp_args x g v))
        (List.filteri (fun i _ -> i < 8) (List.rev misses))

(* [narrow1 ~bound ?kappa f] holds [f] of one argument at every float of a
   narrow dtype. *)
let narrow1 ~bound ?kappa f (Dtype dt) =
  let x = every dt in
  hold dt ~bound ?kappa f [| x |] (Nx.numel x)

(* [narrow2 ?keep ~bound ?kappa ~firsts ~seconds f] holds [f] of two arguments
   at every pair of floats of a dtype of at most 8 bits, and of a 16-bit one at
   every float as either argument against each of [firsts] or [seconds] as the
   other, rounded to the dtype; at the pairs [keep] keeps. *)
let narrow2 ?(keep = fun _ -> true) ~bound ?kappa ~firsts ~seconds f (Dtype dt)
    =
  let all = Array.to_list (values (Nx.cast Nx.float64 (every dt))) in
  let pairs =
    if D.bits dt <= 8 then
      List.concat_map (fun a -> List.map (fun x -> (a, x)) all) all
    else
      List.concat_map (fun a -> List.map (fun x -> (a, x)) all) firsts
      @ List.concat_map (fun x -> List.map (fun a -> (a, x)) all) seconds
  in
  let pairs = List.filter (fun (a, x) -> keep [| a; x |]) pairs in
  let n = List.length pairs in
  let column sel =
    Nx.cast dt
      (Nx.create Nx.float64 [| n |] (Array.of_list (List.map sel pairs)))
  in
  hold dt ~bound ?kappa f [| column fst; column snd |] n

(* [narrow3 ~bound ?kappa ~settings f] holds [f] of three arguments at every
   float of a narrow dtype as the argument a setting names, the others the
   setting's, rounded to the dtype; at the triples whose parameters are at most
   [2^20], where the bound holds. *)
let narrow3 ~bound ?kappa ~settings f (Dtype dt) =
  let all = values (Nx.cast Nx.float64 (every dt)) in
  let at slot others v =
    let t = Array.copy others in
    t.(slot) <- v;
    t
  in
  let triples =
    List.concat_map
      (fun (slot, others) -> Array.to_list (Array.map (at slot others) all))
      settings
    |> List.filter (fun t -> t.(0) <= 0x1p20 && t.(1) <= 0x1p20)
  in
  let n = List.length triples in
  let column i =
    Nx.cast dt
      (Nx.create Nx.float64 [| n |]
         (Array.of_list (List.map (fun t -> t.(i)) triples)))
  in
  hold dt ~bound ?kappa f [| column 0; column 1; column 2 |] n

(* Edges

   The values nx.mli states at the ends of each domain, a NaN in any argument
   included, exactly. *)

(* [edges f table (Dtype dt)]: [f]'s results at the points of [table], each a
   row of arguments, are its values, at [dt]. *)
let edges { f } table (Dtype dt) =
  let points = List.map fst table in
  let n = List.length points in
  let k = Array.length (List.hd points) in
  let column i =
    Nx.create dt [| n |] (Array.of_list (List.map (fun p -> p.(i)) points))
  in
  equal (array float_exact)
    (Array.of_list (List.map snd table))
    (values (Nx.cast Nx.float64 (f (Array.init k column))))

let golden_cases ?keep ~bound name f =
  cases "within its bound of the golden" ~id:dtype_id wide
    (golden ?keep ~bound name f)

let edge_cases f table =
  cases "is nx.mli's value at its edges" ~id:dtype_id wide (edges f table)

(* Laws

   Identities the functions keep, at float64 over drawn batches, each within the
   sum of its terms' bounds. Each case is a batch of [size] points, each drawn
   alone: most of a call's time on a batch this small is its fixed cost, so 25
   cases of 128 draw the 3,200 points that 100 cases of 32 did in about a
   quarter of the time. *)

let size = 128
let prop name gen law = prop ~count:25 name gen law

let batch lo hi =
  Gen.with_pp Nx.pp
    (Gen.map
       (fun xs -> Nx.create Nx.float64 [| Array.length xs |] xs)
       (Gen.array ~size:(Gen.constant size) (Gen.float_range lo hi)))

(* Floats of [dt] of either sign, their magnitudes [2^e] for [e] drawn in [emin,
   emax], with zeros, the least subnormal and the largest float among them. *)
let signed_floats (type s) (dt : (float, s) Nx.dtype) ~emin ~emax =
  let edges =
    [ 0.; -0.; Float.ldexp 1. emin; Float.ldexp 1. (emin + 1); Float.max_float ]
  in
  let one =
    Gen.frequency
      [
        (1, Gen.of_list ~pp:Format.pp_print_float edges);
        ( 8,
          Gen.map
            (fun (e, neg) ->
              let v = Float.pow 2. e in
              if neg then -.v else v)
            (Gen.pair
               (Gen.float_range (float_of_int emin) (float_of_int emax))
               Gen.bool) );
      ]
  in
  Gen.with_pp Nx.pp
    (Gen.map
       (fun xs -> Nx.create dt [| Array.length xs |] xs)
       (Gen.array ~size:(Gen.constant size) one))

(* Values log-uniform in [lo, hi]. *)
let log_batch lo hi =
  Gen.with_pp Nx.pp
    (Gen.map
       (fun xs -> Nx.exp (Nx.create Nx.float64 [| Array.length xs |] xs))
       (Gen.array ~size:(Gen.constant size)
          (Gen.float_range (Stdlib.log lo) (Stdlib.log hi))))

(* [close tol expected actual]: elementwise, within [tol] absolutely. *)
let close_all tol expected actual =
  let e = values expected and a = values actual and t = values tol in
  Array.iteri
    (fun i e ->
      at_most
        ~msg:(Printf.sprintf "the error at element %d" i)
        float_exact ~than:t.(i)
        (Nx_support.gap a.(i) e))
    e

(* [x] on a grid of [2^-20], where [x + 1] and [1 - x] are exact. *)
let grid x = Nx.div_s (Nx.round (Nx.mul_s x 0x1p20)) 0x1p20

(* [sin (pi x)] on (0, 1), reduced exactly to (0, 1/2]. *)
let sinpi x =
  let m = Nx.where (Nx.greater_s x 0.5) (Nx.rsub_s 1. x) x in
  Nx.sin (Nx.mul_s m Float.pi)

let bits_equal (type s) (dt : (float, s) Nx.dtype) a b =
  match D.bits dt with
  | 64 ->
      equal (array int64)
        (values (Nx.bitcast Nx.int64 a))
        (values (Nx.bitcast Nx.int64 b))
  | _ ->
      equal (array int32)
        (values (Nx.bitcast Nx.int32 a))
        (values (Nx.bitcast Nx.int32 b))

(* The absolute error [k] ulps allow at float64 at [v]: an ulp is at most [eps
   |v|], and the least subnormal below the normal range. *)
let in_ulps k v = k *. ((eps *. Float.abs v) +. 0x1p-1074)

(* How far [a] lies above [b], infinite where either is NaN: windtrap orders NaN
   below every float, so a bound on a NaN [a] itself would pass. *)
let above a b =
  if Float.is_nan a || Float.is_nan b then Float.infinity
  else if a = b then 0.
  else a -. b

(* [f x <= f y] where [x <= y], for an increasing function: within the errors
   [error] bounds at each, its results may cross by their sum. *)
let increasing ~error f x y =
  Array.iter2
    (fun a b -> at_most float_exact ~than:(error a +. error b) (above a b))
    (values (f x))
    (values (f y))

(* The linear bound in ulps of [f], as a tolerance. *)
let linear f =
  if f = 0. then 0. else log_ulps 16 f *. (eps /. 2.) *. Float.abs f

let elementwise f a b c =
  let a = values a and b = values b and c = values c in
  Array.iteri (fun i a -> f i a b.(i) c.(i)) a

let close ~than i got want =
  at_most
    ~msg:(Printf.sprintf "the error at element %d" i)
    float_exact ~than (Nx_support.gap got want)

let pairs = Gen.pair (log_batch 1e-3 1e4) (log_batch 1e-3 1e4)

(* Error function and the standard normal *)

(* [κ = |p √π e^(x²) / (2x)|] at [x = erfinv p], [1] at [0]. *)
let erfinv_kappa =
  {
    k =
      (fun a x ->
        let p = a.(0) in
        let zero = Nx.equal_s p 0. in
        let k =
          Nx.div
            (Nx.mul_s (Nx.mul p (Nx.exp (Nx.square x))) (Float.sqrt Float.pi))
            (Nx.mul_s (Nx.where zero (Nx.ones_like x) x) 2.)
        in
        Nx.where zero (Nx.ones_like p) (Nx.abs k));
  }

(* [κ = |p / (x Φ'(x))|] at [x = ndtri p]. *)
let ndtri_kappa =
  {
    k =
      (fun a x ->
        let slope =
          Nx.mul_s
            (Nx.exp (Nx.mul_s (Nx.square x) (-0.5)))
            (1. /. Float.sqrt (2. *. Float.pi))
        in
        Nx.abs (Nx.div a.(0) (Nx.mul x slope)));
  }

let erfc =
  let f = one { u = Nx.erfc } in
  group "erfc"
    [
      golden_cases ~bound:(everywhere (Ulps 8)) "erfc" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow1 ~bound:(everywhere (Ulps 8)) f);
      edge_cases f
        [
          ([| nan |], nan);
          ([| -.inf |], 2.);
          ([| inf |], 0.);
          ([| 0. |], 1.);
          ([| -0. |], 1.);
        ];
      prop "erfc (-x) is 2 - erfc x" (batch 0. 30.) (fun x ->
          let open Nx in
          let tol = mul_s (add (erfc (neg x)) (erfc x)) (17. *. eps) in
          close_all tol (rsub_s 2. (erfc x)) (erfc (neg x)));
    ]

let erfinv =
  let f = one { u = Nx.erfinv } in
  let bound = everywhere (Inverse (4, 8)) in
  group "erfinv"
    [
      golden_cases ~bound "erfinv" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow1 ~bound ~kappa:erfinv_kappa f);
      edge_cases f
        [
          ([| nan |], nan);
          ([| -1. |], -.inf);
          ([| 1. |], inf);
          ([| 0. |], 0.);
          ([| -0. |], -0.);
          ([| 1.5 |], nan);
          ([| -1.5 |], nan);
          ([| inf |], nan);
          ([| -.inf |], nan);
        ];
      prop "erfinv inverts erf" (batch (-5.) 5.) (fun x ->
          (* [erf] within 2 ulps: [erfinv p] is within [4 + 8κ] ulps of [x], and
             [p]'s error moves it by [2κ] more. *)
          let open Nx in
          let p = erf x in
          let k = erfinv_kappa.k [| p |] x in
          let tol = mul_s (mul (abs x) (add_s (mul_s k 10.) 4.)) eps in
          let tol =
            where (less_s (abs p) 1.) tol (full_like x Float.infinity)
          in
          close_all tol x (erfinv p));
    ]

let ndtr =
  let f = one { u = Nx.ndtr } in
  group "ndtr"
    [
      golden_cases ~bound:(everywhere (Ulps 16)) "ndtr" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow1 ~bound:(everywhere (Ulps 16)) f);
      edge_cases f
        [
          ([| nan |], nan);
          ([| -.inf |], 0.);
          ([| inf |], 1.);
          ([| 0. |], 0.5);
          ([| -0. |], 0.5);
        ];
      prop "ndtr x + ndtr (-x) is 1" (batch (-40.) 40.) (fun x ->
          let open Nx in
          close_all
            (full_like x (33. *. eps))
            (ones_like x)
            (add (ndtr x) (ndtr (neg x))));
      prop "increases" (batch (-50.) 50.) (fun x ->
          increasing ~error:(in_ulps 16.) Nx.ndtr x (Nx.add_s x 1e-3));
    ]

let log_ndtr =
  let f = one { u = Nx.log_ndtr } in
  group "log_ndtr"
    [
      golden_cases ~bound:(everywhere (Ulps 32)) "log_ndtr" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow1 ~bound:(everywhere (Ulps 32)) f);
      edge_cases f [ ([| nan |], nan); ([| -.inf |], -.inf); ([| inf |], -0.) ];
      prop "is log ndtr where ndtr is normal" (batch (-37.) 9.) (fun x ->
          let open Nx in
          let l = log_ndtr x in
          close_all (mul_s (add_s (abs l) 1.) (48. *. eps)) (log (ndtr x)) l);
      prop "increases" (batch (-50.) 50.) (fun x ->
          increasing ~error:(in_ulps 32.) Nx.log_ndtr x (Nx.add_s x 1e-3));
    ]

let ndtri =
  let f = one { u = Nx.ndtri } in
  let bound = everywhere (Inverse (4, 16)) in
  group "ndtri"
    [
      golden_cases ~bound "ndtri" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow1 ~bound ~kappa:ndtri_kappa f);
      edge_cases f
        [
          ([| nan |], nan);
          ([| 0. |], -.inf);
          ([| -0. |], -.inf);
          ([| 1. |], inf);
          ([| 0.5 |], 0.);
          ([| -1e-30 |], nan);
          ([| 1.5 |], nan);
          ([| inf |], nan);
          ([| -.inf |], nan);
        ];
      prop "ndtri inverts ndtr" (batch (-8.) 8.) (fun x ->
          let open Nx in
          let p = ndtr x in
          let slope = mul x (exp (mul_s (square x) (-0.5))) in
          let zero = equal_s x 0. in
          let kappa = abs (div p (where zero (ones_like x) slope)) in
          let kappa =
            where zero (ones_like x) (mul_s kappa (Float.sqrt (2. *. Float.pi)))
          in
          let tol = mul_s (mul (abs x) (add_s (mul_s kappa 32.) 8.)) eps in
          close_all (add_s tol (8. *. eps)) x (ndtri p));
    ]

(* Gamma *)

let lgamma_scale =
  { k = (fun a _ -> Nx.add_s (Nx.abs (Nx.lgamma (Nx.rsub_s 1. a.(0)))) 1.) }

let digamma_scale =
  {
    k =
      (fun a _ ->
        let px = Nx.mul_s a.(0) Float.pi in
        let cot = Nx.div (Nx.cos px) (Nx.sin px) in
        Nx.add_s (Nx.abs (Nx.mul_s cot Float.pi)) 1.);
  }

let lgamma =
  let f = one { u = Nx.lgamma } in
  let bound = by_sign ~positive:(Near_zeros (16, 16)) ~negative:(Scaled 16) in
  group "lgamma"
    [
      golden_cases ~bound "lgamma" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow1 ~bound ~kappa:lgamma_scale f);
      edge_cases f
        [
          ([| nan |], nan);
          ([| 0. |], inf);
          ([| -0. |], inf);
          ([| -1. |], inf);
          ([| -2. |], inf);
          ([| inf |], inf);
          ([| -.inf |], nan);
          ([| 1. |], 0.);
          ([| 2. |], 0.);
        ];
      prop "lgamma (x + 1) is lgamma x + log x" (batch 1e-3 50.) (fun x ->
          let open Nx in
          let x = grid x in
          let l = lgamma x and lx = log x in
          let tol = mul_s (add_s (add (abs l) (abs lx)) 1.) (40. *. eps) in
          close_all tol (add l lx) (lgamma (add_s x 1.)));
      prop "lgamma x + lgamma (1 - x) is log (pi / sin (pi x)) on (0, 1)"
        (batch 1e-3 0.999) (fun x ->
          let open Nx in
          let x = grid x in
          let r = rsub_s (Stdlib.log Float.pi) (log (sinpi x)) in
          let tol = mul_s (add_s (abs r) 1.) (40. *. eps) in
          close_all tol r (add (lgamma x) (lgamma (rsub_s 1. x))));
    ]

let digamma =
  let f = one { u = Nx.digamma } in
  let bound = by_sign ~positive:(Near_zeros (16, 16)) ~negative:(Scaled 16) in
  group "digamma"
    [
      golden_cases ~bound "digamma" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow1 ~bound ~kappa:digamma_scale f);
      edge_cases f
        [
          ([| nan |], nan);
          ([| 0. |], -.inf);
          ([| -0. |], inf);
          ([| -1. |], nan);
          ([| -2. |], nan);
          ([| inf |], inf);
          ([| -.inf |], nan);
        ];
      prop "digamma (x + 1) is digamma x + 1/x" (batch 1e-3 50.) (fun x ->
          let open Nx in
          let x = grid x in
          let d = digamma x and r = recip x in
          let tol = mul_s (add_s (add (abs d) (abs r)) 1.) (40. *. eps) in
          close_all tol (add d r) (digamma (add_s x 1.)));
      prop "digamma (1 - x) - digamma x is pi cot (pi x) on (0, 1)"
        (batch 1e-3 0.999) (fun x ->
          let open Nx in
          let x = grid x in
          let cospi = sin (mul_s (rsub_s 0.5 x) Float.pi) in
          let r = mul_s (div cospi (sinpi x)) Float.pi in
          let d1 = digamma (rsub_s 1. x) and d0 = digamma x in
          let size = add (add (abs d1) (abs d0)) (abs r) in
          close_all (mul_s (add_s size 1.) (32. *. eps)) r (sub d1 d0));
    ]

let lbeta =
  let f = two { b = Nx.lbeta } in
  let bound = everywhere (Near_zeros (256, 512)) in
  group "lbeta"
    [
      golden_cases ~bound "lbeta" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow2 ~bound ~firsts:[ 0.5; 10. ] ~seconds:[ 1.; 20. ] f);
      edge_cases f
        [
          ([| nan; 1. |], nan);
          ([| 1.; nan |], nan);
          ([| 0.; 1. |], inf);
          ([| 1.; 0. |], inf);
          ([| 0.; 0. |], inf);
          ([| inf; 1. |], -.inf);
          ([| 1.; inf |], -.inf);
          ([| 0.; inf |], nan);
          ([| -1.; 1. |], nan);
          ([| 1.; -1. |], nan);
        ];
      prop "is symmetric"
        (Gen.pair (batch 1e-3 1e3) (batch 1e-3 1e3))
        (fun (a, b) ->
          (* Each within 256 ulps, or 512 eps where below 1. *)
          let l = Nx.lbeta a b in
          close_all
            (Nx.mul_s (Nx.add_s (Nx.abs l) 2.) (512. *. eps))
            l (Nx.lbeta b a));
      prop "lbeta a 1 is -log a" (batch 1e-3 1e6) (fun a ->
          let open Nx in
          let r = neg (log a) in
          let tol = mul_s (add_s (abs r) 1.) (512. *. eps) in
          close_all tol r (lbeta a (ones_like a)));
    ]

(* The incomplete gamma family. Its bounds hold for [a <= 2^20]; above, the
   goldens' few rows are edges. *)

let in_domain _ rows =
  List.filter
    (fun (r : Nx_support.row) ->
      let a = r.args.(0) in
      not (Float.is_finite a && a > 0x1p20))
    rows

(* The partners of every 16-bit float: two of the parameter and two of the
   variable, one and two for an inverse, whose program is three times as
   long. *)
let parameters = [ 0.5; 10. ]
let variables = [ 1.; 20. ]

(* [narrow2] at the pairs whose parameter is at most [2^20]. *)
let gamma_narrow ~bound ?kappa ?(firsts = parameters) ?(seconds = variables) f =
  cases "within one ulp at every narrow float" ~id:dtype_id narrows
    (narrow2
       ~keep:(fun args -> args.(0) <= 0x1p20)
       ~bound ?kappa ~firsts ~seconds f)

(* The incomplete gamma functions at [(a, x)]: [P], with [Q = 1 - P], [log P]
   and [log Q] from it. *)
let gamma_table =
  let row a x p = ([| a; x |], p) in
  [
    row 1. 0. 0.;
    row 0.5 0. 0.;
    row 1. (-0.) 0.;
    row 0.5 (-0.) 0.;
    row 1. inf 1.;
    row 0. inf 1.;
    row inf 0. 0.;
    row inf inf nan;
    row 0. 0. nan;
    row (-1.) 1. nan;
    row 1. (-1.) nan;
    row nan 1. nan;
    row 1. nan nan;
    row nan 0. nan;
  ]

let gamma_tail f = List.map (fun (args, p) -> (args, f p)) gamma_table

(* As [a] goes to [+inf] for finite [x > 0], [P] tends to 0 and [Q] to 1 from
   below; at [a = 0], [P = 1 - a E1(x) + O(a^2)] for [x > 0] tends to 1 from
   below. *)
let at_a_inf v = List.map (fun x -> ([| inf; x |], v)) [ 0x1p-149; 1.; 1e30 ]

let at_a_zero v =
  List.concat_map
    (fun x -> [ ([| 0.; x |], v); ([| -0.; x |], v) ])
    [ 0x1p-149; 1e-30; 0.5; 1.1; 3.; 30.; 1e30 ]

(* An inverse's condition number [|m / (x ∂ₓF)|] at its result [x], [F] the
   tail, [m = min (p, 1 - p)] the smaller tail and [∂ₓP = x^(a - 1) e^-x /
   Γ(a)]. *)
let quantile_kappa =
  {
    k =
      (fun args x ->
        let a = args.(0) and p = args.(1) in
        let open Nx in
        let density = sub (sub (mul a (log x)) x) (lgamma a) in
        let m = minimum p (rsub_s 1. p) in
        exp (sub (log m) density));
  }

(* The forward bound at [f], in ulps, rounded down. *)
let forward f = int_of_float (log_ulps 16 f)

(* An inverse's bound at its probability [args.(1)], read in the smaller
   tail. *)
let inverse_bound args =
  let p = args.(1) in
  Inverse (4, forward (Float.min p (1. -. p)))

(* [upper_tail ~tail ~inverse a x] holds [inverse]'s bound above [1/2], where it
   inverts the other tail. [t = tail a x], within [B(t)] ulps, is read as [p = 1
   - t] rounded, whose smaller tail [m = 1 - p] is exact by Sterbenz's lemma and
   differs from [t] by [m - t], also exact. The quantile at [p] is [x] moved by
   [κ (B(t) + |log (m / t)| / ε)] ulps, for [κ] the slope of [log x] in [log m],
   and [|log (m / t)|] is at most [|m - t| / min (m, t)]. [inverse a p] is
   within [4 + κ B(m)] ulps of it. *)
let upper_tail ~tail ~(inverse : b) a x =
  let open Nx in
  let t = tail a x in
  let p = rsub_s 1. t in
  let back = inverse.b a p in
  let kappa = values (quantile_kappa.k [| a; p |] back) in
  let t = values t and p = values p and back = values back in
  Array.iteri
    (fun i x ->
      let t = t.(i) and m = 1. -. p.(i) in
      if t >= 0x1p-1022 && t < 0.5 && p.(i) < 1. then
        let moved =
          kappa.(i)
          *. (log_ulps 16 t +. (Float.abs (m -. t) /. (Float.min m t *. eps)))
        in
        let own = 4. +. (kappa.(i) *. log_ulps 16 m) in
        close i ~than:((own +. moved) *. eps *. x) back.(i) x)
    (values x)

(* [p] within [B(p)] ulps moves the quantile by [κ_p B(p)], for [κ_p = κ p / m];
   the inverse adds [4 + κ B(m)]. *)
let inverted ~forward ~(inverse : b) (a, x) =
  let p = forward a x in
  let back = inverse.b a p in
  let kappa = values (quantile_kappa.k [| a; p |] back) in
  elementwise
    (fun i x p back ->
      if p >= 0x1p-1022 && p < 1. then
        let m = Float.min p (1. -. p) in
        let moved = kappa.(i) *. p /. m *. log_ulps 16 p in
        let own = 4. +. (kappa.(i) *. log_ulps 16 m) in
        close i ~than:((own +. moved) *. eps *. x) back x)
    x p back

let gammainc =
  let f = two { b = Nx.gammainc } in
  let bound = everywhere (Log_ulps 16) in
  group "gammainc"
    [
      golden_cases ~keep:in_domain ~bound "gammainc" f;
      gamma_narrow ~bound f;
      edge_cases f (gamma_table @ at_a_inf 0. @ at_a_zero 1.);
      prop "gammainc + gammaincc is 1" pairs (fun (a, x) ->
          elementwise
            (fun i p q _ ->
              close i ~than:(linear p +. linear q +. eps) (p +. q) 1.)
            (Nx.gammainc a x) (Nx.gammaincc a x) x);
      prop "increases in x and decreases in a"
        (Gen.pair pairs (log_batch 1e-6 1.))
        (fun ((a, x), d) ->
          (* Within their bounds, two results may cross by their sum. *)
          let error v = in_ulps (log_ulps 16 v) v in
          let up = Nx.mul x (Nx.add_s d 1.)
          and more = Nx.mul a (Nx.add_s d 1.) in
          elementwise
            (fun i p q r ->
              at_most
                ~msg:(Printf.sprintf "P(a, x) at element %d" i)
                float_exact
                ~than:(error p +. error q)
                (above p q);
              at_most
                ~msg:(Printf.sprintf "P(a', x) at element %d" i)
                float_exact
                ~than:(error p +. error r)
                (above r p))
            (Nx.gammainc a x) (Nx.gammainc a up) (Nx.gammainc more x));
      prop "gammainc 1/2 x is erf (sqrt x)" (log_batch 1e-6 30.) (fun x ->
          elementwise
            (fun i p e _ -> close i ~than:(linear e +. (4. *. eps *. e)) p e)
            (Nx.gammainc (Nx.full_like x 0.5) x)
            (Nx.erf (Nx.sqrt x))
            x);
    ]

let gammaincc =
  let f = two { b = Nx.gammaincc } in
  let bound = everywhere (Log_ulps 16) in
  group "gammaincc"
    [
      golden_cases ~keep:in_domain ~bound "gammaincc" f;
      gamma_narrow ~bound f;
      edge_cases f (gamma_tail (fun p -> 1. -. p) @ at_a_inf 1. @ at_a_zero 0.);
      cases "a Poisson tail gammaincc (k + 1) lam is 0 at k = -1" ~id:dtype_id
        wide
        (edges
           { f = (fun a -> Nx.gammaincc (Nx.add_s a.(0) 1.) a.(1)) }
           [ ([| -1.; 2.5 |], 0.) ]);
      prop "gammaincc 1 x is exp (-x)" (log_batch 1e-300 700.) (fun x ->
          elementwise
            (fun i q e _ -> close i ~than:(linear e +. (2. *. eps *. e)) q e)
            (Nx.gammaincc (Nx.ones_like x) x)
            (Nx.exp (Nx.neg x))
            x);
    ]

let log_gammainc =
  let f = two { b = Nx.log_gammainc } in
  let bound = everywhere (Near_zeros (16, 16)) in
  group "log_gammainc"
    [
      golden_cases ~keep:in_domain ~bound "log_gammainc" f;
      gamma_narrow ~bound f;
      edge_cases f (gamma_tail Stdlib.log @ at_a_inf (-.inf) @ at_a_zero (-0.));
      prop "is log1p (-gammaincc) where gammaincc < 1/2" pairs (fun (a, x) ->
          elementwise
            (fun i lp q _ ->
              if q < 0.5 then
                close i
                  ~than:
                    ((16. *. eps *. Float.max 1. (Float.abs lp))
                    +. (2. *. linear q))
                  lp (Stdlib.log1p (-.q)))
            (Nx.log_gammainc a x) (Nx.gammaincc a x) x);
      prop "gammainc is exp log_gammainc" pairs (fun (a, x) ->
          (* [log P] within 16 ulps, or 16 eps where below 1, moves [exp] by
             that much relatively; [exp] rounds once. *)
          let lp = Nx.log_gammainc a x in
          elementwise
            (fun i p lp e ->
              let moved =
                ((16. *. Float.max 1. (Float.abs lp)) +. 1.) *. 1.01
              in
              close i ~than:(in_ulps (log_ulps 16 p) p +. in_ulps moved e) p e)
            (Nx.gammainc a x) lp (Nx.exp lp));
      prop "tends to a log x - lgamma (a + 1) as x goes to 0"
        (Gen.pair (log_batch 1e-3 1e3) (log_batch 1e-300 1e-200))
        (fun (a, x) ->
          let r = Nx.sub (Nx.mul a (Nx.log x)) (Nx.lgamma (Nx.add_s a 1.)) in
          elementwise
            (fun i l r _ -> close i ~than:(32. *. eps *. Float.abs r) l r)
            (Nx.log_gammainc a x) r x);
    ]

let log_gammaincc =
  let f = two { b = Nx.log_gammaincc } in
  let bound = everywhere (Near_zeros (16, 16)) in
  group "log_gammaincc"
    [
      golden_cases ~keep:in_domain ~bound "log_gammaincc" f;
      gamma_narrow ~bound f;
      edge_cases f
        (gamma_tail (fun p -> Stdlib.log (1. -. p))
        @ at_a_inf (-0.) @ at_a_zero (-.inf));
      prop "is log1p (-gammainc) where gammainc < 1/2" pairs (fun (a, x) ->
          elementwise
            (fun i lq p _ ->
              if p < 0.5 then
                close i
                  ~than:
                    ((16. *. eps *. Float.max 1. (Float.abs lq))
                    +. (2. *. linear p))
                  lq (Stdlib.log1p (-.p)))
            (Nx.log_gammaincc a x) (Nx.gammainc a x) x);
    ]

let gammaincinv =
  let inverse = { b = Nx.gammaincinv } in
  let f = two inverse in
  group "gammaincinv"
    [
      golden_cases ~keep:in_domain ~bound:inverse_bound "gammaincinv" f;
      gamma_narrow ~bound:inverse_bound ~kappa:quantile_kappa ~firsts:[ 2.5 ]
        ~seconds:[ 0.1; 0.9 ] f;
      edge_cases f
        [
          ([| 1.; 0. |], 0.);
          ([| 1.; 1. |], inf);
          ([| inf; 0.5 |], inf);
          ([| inf; 0. |], 0.);
          ([| 0.; 0.5 |], 0.);
          ([| 0.; 0. |], 0.);
          ([| 0.; 1e-30 |], 0.);
          ([| 0.; 1. |], inf);
          ([| -1.; 0.5 |], nan);
          ([| 1.; 1.5 |], nan);
          ([| 1.; -0.5 |], nan);
          ([| nan; 0.5 |], nan);
          ([| 1.; nan |], nan);
        ];
      prop "inverts gammainc" pairs (inverted ~forward:Nx.gammainc ~inverse);
      prop "holds its bound above p = 1/2" pairs (fun (a, x) ->
          upper_tail ~tail:Nx.gammaincc ~inverse a x);
    ]

let gammainccinv =
  let inverse = { b = Nx.gammainccinv } in
  let f = two inverse in
  group "gammainccinv"
    [
      golden_cases ~keep:in_domain ~bound:inverse_bound "gammainccinv" f;
      gamma_narrow ~bound:inverse_bound ~kappa:quantile_kappa ~firsts:[ 2.5 ]
        ~seconds:[ 0.1; 0.9 ] f;
      edge_cases f
        [
          ([| 1.; 1. |], 0.);
          ([| 1.; 0. |], inf);
          ([| inf; 0.5 |], inf);
          ([| inf; 1. |], 0.);
          ([| 0.; 0.5 |], 0.);
          ([| 0.; 1. |], 0.);
          ([| 0.; 0. |], inf);
          ([| -1.; 0.5 |], nan);
          ([| 1.; 1.5 |], nan);
          ([| 1.; -0.5 |], nan);
          ([| nan; 0.5 |], nan);
          ([| 1.; nan |], nan);
        ];
      prop "inverts gammaincc" pairs (inverted ~forward:Nx.gammaincc ~inverse);
      prop "holds its bound above q = 1/2" pairs (fun (a, x) ->
          upper_tail ~tail:Nx.gammainc ~inverse a x);
    ]

(* The incomplete beta family *)

(* Every float as [x] beside [a = 2.5, b = 10.], as [a] beside [b = 0.5, x =
   0.3], and as [b] beside [a = 10., x = 0.7]. *)
let beta_settings =
  [ (2, [| 2.5; 10.; 0. |]); (0, [| 0.; 0.5; 0.3 |]); (1, [| 10.; 0.; 0.7 |]) ]

(* Every float as [p] beside [a = 2.5, b = 10.], and as [a] beside [b = 0.5, p =
   0.3]: an inverse's program is three times as long. *)
let beta_inverse_settings = [ (2, [| 2.5; 10.; 0. |]); (0, [| 0.; 0.5; 0.3 |]) ]

(* [log (x ∂ₓI)] for [∂ₓI = x^(a - 1) (1 - x)^(b - 1) / B(a, b)] the beta's
   density. *)
let beta_density a b x =
  let open Nx in
  sub (add (mul a (log x)) (mul (sub_s b 1.) (log1p (neg x)))) (lbeta a b)

(* The beta's quantile's condition number [|m / (x ∂ₓI)|] at its result [x], [m
   = min (p, 1 - p)] the smaller tail. *)
let beta_kappa =
  {
    k =
      (fun args x ->
        let a = args.(0) and b = args.(1) and p = args.(2) in
        let open Nx in
        let m = minimum p (rsub_s 1. p) in
        exp (sub (log m) (beta_density a b x)));
  }

(* The beta's quantile's bound at its probability [args.(2)], read in the
   smaller tail. *)
let beta_inverse_bound args =
  let p = args.(2) in
  Inverse (4, int_of_float (log_ulps 32 (Float.min p (1. -. p))))

(* nx.mli's carve-out: where the bound at [x = 1/2], [κ_h (64 (1 + |log m|) + 4)
   ε] for [κ_h = m / (f(1/2) / 2)], reaches 1, the tail is flat between masses
   near 0 and 1, and a [p] within [64 (1 + |log m|) + 4] ulps of [m] of the tail
   at [1/2] may give any [x] between them. [half] is the tail at [1/2] and
   [log_f] is [log f(1/2)]. *)
let beta_flat ~f32 ~half ~log_f p =
  let eps = if f32 then 0x1p-23 else 0x1p-52 in
  let m = Float.min p (1. -. p) in
  let kappa = m /. (0.5 *. Float.exp log_f) in
  let bound = log_ulps 32 m *. eps in
  kappa *. bound >= 1. && Float.abs (p -. half) <= bound *. m

(* The tail and [log f] at [1/2], as float arrays, for the shapes [a] and
   [b]. *)
let beta_at_half ~upper a b =
  let open Nx in
  let tail = if upper then betaincc else betainc in
  let half = tail a b (full_like a 0.5) in
  let log_f = sub (mul_s (sub_s (add a b) 2.) (-.Stdlib.log 2.)) (lbeta a b) in
  (values half, values log_f)

let beta_plateau ~f32 ~upper a b p =
  let s v = Nx.scalar Nx.float64 v in
  let half, log_f = beta_at_half ~upper (s a) (s b) in
  beta_flat ~f32 ~half:half.(0) ~log_f:log_f.(0) p

(* The rows whose parameters are at most [2^20], where the bound holds, and that
   are not on the plateau. An incomplete beta costs milliseconds a call and
   microseconds an element: the rows share one call. *)
let beta_off_plateau ~upper (Dtype dt) rows =
  let rows =
    List.filter
      (fun (r : Nx_support.row) ->
        let a = r.args.(0) and b = r.args.(1) in
        not
          ((Float.is_finite a && a > 0x1p20)
          || (Float.is_finite b && b > 0x1p20)))
      rows
  in
  let column i =
    let values =
      Array.of_list (List.map (fun r -> r.Nx_support.args.(i)) rows)
    in
    Nx.create Nx.float64 [| Array.length values |] values
  in
  let half, log_f = beta_at_half ~upper (column 0) (column 1) in
  let f32 = D.bits dt = 32 in
  List.filteri
    (fun i (row : Nx_support.row) ->
      not (beta_flat ~f32 ~half:half.(i) ~log_f:log_f.(i) row.args.(2)))
    rows

(* [beta_inverted ~upper a b x p back] holds [back], drawn from [p] at the
   normal [x], to the equation that defines it, read at the smaller tail [m =
   min (p, 1 - p)], exact by Sterbenz's lemma where it is [1 - p]. Within [4 + κ
   B(m)] ulps of the quantile, the tail of [m] at the four floats on each side
   of [back] brackets [m], the ends within their own [B(t)] ulps and [κ B(m)]
   ulps moving that tail by [B(m) m ε], since [κ x ∂ₓI = m]. [x] only draws [p]:
   [|back - x|] has no such bound, since [p] holds [x] only to its own rounding,
   and where an ulp of [p] is most of [m], a tail going as [x^a] carries [x] to
   [(1 + δ)^(1/a) x], which a slope at [x] understates by any factor. On the
   plateau [back] may be anywhere in [[0, 1]]. *)
let beta_inverted ~upper a b x p back =
  let open Nx in
  let n = numel back in
  let doubled v = Array.append (values v) (values v) in
  let column v = create Nx.float64 [| 2 * n |] v in
  (* Four floats aside, within [[0, 1]]: [n] below [back], then [n] above, in
     one call: a call costs far more than its elements. *)
  let aside step =
    let far v = Float.min 1. (Float.max 0. (step (step (step (step v))))) in
    Array.map far (values back)
  in
  let points = column (Array.append (aside Float.pred) (aside Float.succ)) in
  let a2 = column (doubled a) and b2 = column (doubled b) in
  let lo = values (betainc a2 b2 points) in
  let hi = values (betaincc a2 b2 points) in
  let av = values a and bv = values b and back = values back in
  let p = values p in
  Array.iteri
    (fun i x ->
      let p = p.(i) in
      if p >= 0x1p-1022 && p < 1. && x >= 0x1p-1022 then begin
        let m = Float.min p (1. -. p) in
        let tail = if upper = (p <= 0.5) then hi else lo in
        let t1 = tail.(i) and t2 = tail.(n + i) in
        let t_lo = Float.min t1 t2 and t_hi = Float.max t1 t2 in
        let own = log_ulps 32 m *. m in
        let slack t = if t = 0. then own else (log_ulps 32 t *. t) +. own in
        let floor = t_lo -. (slack t_lo *. eps) in
        let ceiling = t_hi +. (slack t_hi *. eps) in
        if
          (m < floor || m > ceiling)
          && beta_plateau ~f32:false ~upper av.(i) bv.(i) p
        then
          at_most
            ~msg:(Printf.sprintf "element %d, on the plateau, from 1/2" i)
            float_exact ~than:0.5
            (Nx_support.gap back.(i) 0.5)
        else begin
          let msg = Printf.sprintf "the tail bracketing element %d" i in
          at_least ~msg float_exact ~than:floor m;
          at_most ~msg float_exact ~than:ceiling m
        end
      end)
    (values x)

(* The incomplete beta functions at [(a, b, x)]: [I], [1 - I], [log I] and [log
   (1 - I)]. A tail of 1 at [x] inside (0, 1) is a limit reached from below, and
   its logarithm is [-0.]. *)
let beta_table =
  let row a b x i ic l lc = ([| a; b; x |], (i, ic, l, lc)) in
  let invalid a b x = row a b x nan nan nan nan in
  [
    row 2. 3. 0. 0. 1. (-.inf) 0.;
    row 2. 3. 1. 1. 0. 0. (-.inf);
    row 0. 3. 0.5 1. 0. (-0.) (-.inf);
    row 2. 0. 0.5 0. 1. (-.inf) (-0.);
    row inf 3. 0.5 0. 1. (-.inf) (-0.);
    row 2. inf 0.5 1. 0. (-0.) (-.inf);
    invalid 0. 0. 0.5;
    invalid inf inf 0.5;
    invalid 0. 3. 0.;
    invalid 2. inf 0.;
    invalid 2. 0. 1.;
    invalid inf 3. 1.;
    invalid (-1.) 3. 0.5;
    invalid 2. (-1.) 0.5;
    invalid 2. 3. (-0.1);
    invalid 2. 3. 1.5;
    invalid nan 3. 0.5;
    invalid 2. nan 0.5;
    invalid 2. 3. nan;
  ]

let beta_tail sel = List.map (fun (args, v) -> (args, sel v)) beta_table
let shapes = log_batch 1e-3 1e4

(* The inverses' shapes reach down to where [log (a B(a, b))] is O(a). *)
let inverse_shapes = log_batch 1e-30 1e5
let large_shapes = log_batch 0x1p6 0x1p20

let beta_narrow ~bound f =
  cases "within one ulp at every narrow float" ~id:dtype_id narrows
    (narrow3 ~bound ~settings:beta_settings f)

let betainc =
  let f = three { c = Nx.betainc } in
  let bound = everywhere (Log_ulps 32) in
  group "betainc"
    [
      golden_cases ~bound "betainc" f;
      beta_narrow ~bound f;
      edge_cases f (beta_tail (fun (i, _, _, _) -> i));
      prop "betainc and betaincc sum to 1"
        (Gen.triple shapes shapes (batch 0. 1.))
        (fun (a, b, x) ->
          let open Nx in
          let p = betainc a b x and q = betaincc a b x in
          let tol v =
            let b = mul_s (add_s (abs (log v)) 1.) (66. *. eps) in
            where (equal_s v 0.) v (mul b v)
          in
          close_all (add (tol p) (tol q)) (ones_like p) (add p q));
      prop "increases in x"
        (Gen.triple shapes shapes (batch 0. 0.999))
        (fun (a, b, x) ->
          let error v = in_ulps (log_ulps 32 v) v in
          increasing ~error (Nx.betainc a b) x (Nx.add_s x 1e-3));
      prop "betainc a 1 x is x^a and betainc 1 b x is 1 - (1 - x)^b"
        (Gen.pair (log_batch 1e-2 1e2) (batch 1e-3 0.999))
        (fun (a, x) ->
          let open Nx in
          let one = ones_like a in
          let tol v =
            let b = mul_s (add_s (abs (log v)) 1.) (70. *. eps) in
            where (equal_s v 0.) v (mul b v)
          in
          let p = pow x a in
          close_all (tol p) p (betainc a one x);
          let q = neg (expm1 (mul a (log1p (neg x)))) in
          close_all (tol q) q (betainc one a x));
    ]

let betaincc =
  let f = three { c = Nx.betaincc } in
  let bound = everywhere (Log_ulps 32) in
  group "betaincc"
    [
      golden_cases ~bound "betaincc" f;
      beta_narrow ~bound f;
      edge_cases f (beta_tail (fun (_, ic, _, _) -> ic));
      prop "betaincc a b x is betainc b a (1 - x) bit for bit where x >= 1/2"
        (Gen.triple shapes shapes (batch 0.5 1.))
        (fun (a, b, x) ->
          (* Every fourth [x] is 1/2 and every fifth [b] is [a]: the ties. *)
          let open Nx in
          let every k = init Nx.bool [| numel x |] (fun i -> i.(0) mod k = 0) in
          let x = where (every 4) (full_like x 0.5) x in
          let b = where (every 5) a b in
          bits_equal Nx.float64 (betaincc a b x) (betainc b a (rsub_s 1. x)));
    ]

let betaincinv =
  let inverse = { c = Nx.betaincinv } in
  let f = three inverse in
  group "betaincinv"
    [
      golden_cases
        ~keep:(beta_off_plateau ~upper:false)
        ~bound:beta_inverse_bound "betaincinv" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow3 ~bound:beta_inverse_bound ~kappa:beta_kappa
           ~settings:beta_inverse_settings f);
      edge_cases f
        [
          ([| 2.; 3.; 0. |], 0.);
          ([| 2.; 3.; 1. |], 1.);
          ([| 0.; 3.; 0.5 |], 0.);
          ([| 2.; inf; 0.5 |], 0.);
          ([| 2.; 0.; 0.5 |], 1.);
          ([| inf; 3.; 0.5 |], 1.);
          ([| 0.; 3.; 1. |], 1.);
          ([| 2.; 0.; 0. |], 0.);
          ([| 0.; inf; 0.5 |], 0.);
          ([| inf; 0.; 0.5 |], 1.);
          ([| 0.; 0.; 0.5 |], nan);
          ([| inf; inf; 0.5 |], nan);
          ([| -1.; 3.; 0.5 |], nan);
          ([| 2.; -1.; 0.5 |], nan);
          ([| 2.; 3.; -0.5 |], nan);
          ([| 2.; 3.; 1.5 |], nan);
          ([| nan; 3.; 0.5 |], nan);
          ([| 2.; nan; 0.5 |], nan);
          ([| 2.; 3.; nan |], nan);
        ];
      prop "inverts betainc"
        (Gen.triple inverse_shapes inverse_shapes (batch 0. 1.))
        (fun (a, b, x) ->
          let p = Nx.betainc a b x in
          beta_inverted ~upper:false a b x p (Nx.betaincinv a b p));
      prop "float32 keeps its bound at large shapes near p = 1/2"
        (Gen.triple large_shapes large_shapes (batch 0.4 0.6))
        (fun (a, b, p) ->
          (* Every fourth [p] is 1/2 and every fourth after it the float above
             it, where the guess's equation has [eta] near 0. The float64
             inverse, within its own much smaller bound, is the reference. *)
          let open Nx in
          let at r v p =
            where
              (init Nx.bool (shape p) (fun i -> i.(0) mod 4 = r))
              (full_like p v) p
          in
          let p = at 1 0x1.000002p-1 (at 0 0.5 (cast Nx.float32 p)) in
          let a = cast Nx.float32 a and b = cast Nx.float32 b in
          let wide v = cast Nx.float64 v in
          let a64 = wide a and b64 = wide b and p64 = wide p in
          let x = betaincinv a64 b64 p64 in
          let kappa = values (beta_kappa.k [| a64; b64; p64 |] x) in
          let bv = values b64 and pv = values p64 in
          let got = wide (betaincinv a b p) in
          elementwise
            (fun i a x got ->
              if a <= 0x1p20 && bv.(i) <= 0x1p20 then
                let m = Float.min pv.(i) (1. -. pv.(i)) in
                let bound = 5. +. (kappa.(i) *. log_ulps 32 m) in
                close i ~than:(bound *. 0x1p-23 *. x) got x)
            a64 x got);
      test "bfloat16 at a large shape" (fun () ->
          (* The exact quantile, from mpmath: a float32 result rounds once,
             within an ulp of bfloat16, 2^-12 at 0.06. *)
          let args =
            Array.map (Nx.scalar Nx.bfloat16) [| 67584.; 0x1p20; 0.5 |]
          in
          let got = Nx.item [] (Nx.betaincinv args.(0) args.(1) args.(2)) in
          close 0 ~than:0x1p-12 got 0x1.f006f71390e46p-5);
      cases "is the closed forms" ~id:fst
        [
          ("a b 1/2 is 1/2 for a = b", (3.5, 3.5, 0.5, 0.5));
          ( "1/2 1/2 p is sin^2 (pi p / 2)",
            (0.5, 0.5, 0.3, Float.pow (Stdlib.sin (Float.pi *. 0.15)) 2.) );
          ("a 1 p is p^(1/a)", (4., 1., 0.2, Float.pow 0.2 0.25));
          ("1 b p is 1 - (1 - p)^(1/b)", (1., 4., 0.2, 1. -. Float.pow 0.8 0.25));
          ("1 1 p is p", (1., 1., 0.3, 0.3));
        ]
        (fun (_, (a, b, p, x)) ->
          (* Within the bound, and the closed form's own two roundings. *)
          let args = Array.map (Nx.scalar Nx.float64) [| a; b; p |] in
          let got = Nx.betaincinv args.(0) args.(1) args.(2) in
          (* At the exact quantile, which a wrong result cannot move. *)
          let kappa = Nx.item [] (beta_kappa.k args (Nx.scalar Nx.float64 x)) in
          let m = Float.min p (1. -. p) in
          let bound = 6. +. (kappa *. log_ulps 32 m) in
          close 0 ~than:(bound *. eps *. x) (Nx.item [] got) x);
    ]

let betainccinv =
  let inverse = { c = Nx.betainccinv } in
  let f = three inverse in
  group "betainccinv"
    [
      golden_cases
        ~keep:(beta_off_plateau ~upper:true)
        ~bound:beta_inverse_bound "betainccinv" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow3 ~bound:beta_inverse_bound ~kappa:beta_kappa
           ~settings:beta_inverse_settings f);
      edge_cases f
        [
          ([| 2.; 3.; 1. |], 0.);
          ([| 2.; 3.; 0. |], 1.);
          ([| 0.; 3.; 0.5 |], 0.);
          ([| 2.; inf; 0.5 |], 0.);
          ([| 2.; 0.; 0.5 |], 1.);
          ([| inf; 3.; 0.5 |], 1.);
          ([| 0.; 3.; 0. |], 1.);
          ([| 2.; 0.; 1. |], 0.);
          ([| 0.; 0.; 0.5 |], nan);
          ([| inf; inf; 0.5 |], nan);
          ([| -1.; 3.; 0.5 |], nan);
          ([| 2.; 3.; 1.5 |], nan);
          ([| 2.; 3.; nan |], nan);
        ];
      prop "inverts betaincc"
        (Gen.triple inverse_shapes inverse_shapes (batch 0. 1.))
        (fun (a, b, x) ->
          let q = Nx.betaincc a b x in
          beta_inverted ~upper:true a b x q (Nx.betainccinv a b q));
      prop "betainccinv a b q is 1 - betaincinv b a q"
        (Gen.triple inverse_shapes inverse_shapes (batch 0. 1.))
        (fun (a, b, q) ->
          (* [1 - x] at [x = betaincinv b a q] has [x]'s error: within its bound
             in [x]'s ulps, and the subtraction rounds once. *)
          let x = Nx.betaincinv b a q in
          let kappa = values (beta_kappa.k [| b; a; q |] x) in
          elementwise
            (fun i x q c ->
              if q >= 0x1p-1022 && q < 1. && x >= 0x1p-1022 && x < 1. then
                let m = Float.min q (1. -. q) in
                let l = log_ulps 32 m in
                let own = 4. +. (kappa.(i) *. l) in
                let mirror = 4. +. (kappa.(i) *. x /. (1. -. x) *. l) in
                close i
                  ~than:(((own *. x) +. (mirror *. c) +. c) *. eps)
                  c (1. -. x))
            x q (Nx.betainccinv a b q));
      test "where one ulp of q is most of its tail" (fun () ->
          (* [q = 1 - 3 2^-53], whose tail [m] is [3 2^-53], and the exact
             quantile at it, from mpmath; a [q] one float away moves the
             quantile across the unit interval. *)
          let args =
            Array.map (Nx.scalar Nx.float64)
              [|
                0.017384192644148398;
                5.842454388186878e-18;
                1. -. (3. *. 0x1p-53);
              |]
          in
          let x = 0x1.807d2f4ecc787p-2 in
          let got = Nx.betainccinv args.(0) args.(1) args.(2) in
          let kappa = Nx.item [] (beta_kappa.k args (Nx.scalar Nx.float64 x)) in
          let bound = 5. +. (kappa *. log_ulps 32 (3. *. 0x1p-53)) in
          close 0 ~than:(bound *. eps *. x) (Nx.item [] got) x);
      test "stays in [0, 1] on a plateau" (fun () ->
          (* Shapes far below 1, at a [q] of the flat tail, where the last
             step's series is read far outside its convergence. *)
          let args =
            Array.map (Nx.scalar Nx.float64)
              [|
                2.320895026900713e-17; 1.5764313287958e-17; 0x1.30e6a0610437ap-1;
              |]
          in
          let got = Nx.item [] (Nx.betainccinv args.(0) args.(1) args.(2)) in
          at_least float_exact ~than:0. got;
          at_most float_exact ~than:1. got);
    ]

let log_betainc =
  let f = three { c = Nx.log_betainc } in
  let bound = everywhere (Near_zeros (32, 32)) in
  group "log_betainc"
    [
      golden_cases ~bound "log_betainc" f;
      beta_narrow ~bound f;
      edge_cases f (beta_tail (fun (_, _, l, _) -> l));
      prop "exp log_betainc is betainc"
        (Gen.triple shapes shapes (batch 0. 1.))
        (fun (a, b, x) ->
          (* [log I] within 32 ulps, or 32 eps where below 1, moves [exp] by
             that much relatively; [exp] rounds once. *)
          let l = Nx.log_betainc a b x in
          let i = values (Nx.betainc a b x) and e = values (Nx.exp l) in
          Array.iteri
            (fun k l ->
              let moved = ((32. *. Float.max 1. (Float.abs l)) +. 1.) *. 1.01 in
              close k
                ~than:(in_ulps (log_ulps 32 i.(k)) i.(k) +. in_ulps moved e.(k))
                i.(k) e.(k))
            (values l));
    ]

let log_betaincc =
  let f = three { c = Nx.log_betaincc } in
  let bound = everywhere (Near_zeros (32, 32)) in
  group "log_betaincc"
    [
      golden_cases ~bound "log_betaincc" f;
      beta_narrow ~bound f;
      edge_cases f (beta_tail (fun (_, _, _, lc) -> lc));
    ]

(* Modified Bessel *)

let i0e =
  let f = one { u = Nx.i0e } in
  group "i0e"
    [
      golden_cases ~bound:(everywhere (Ulps 8)) "i0e" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow1 ~bound:(everywhere (Ulps 8)) f);
      edge_cases f
        [
          ([| nan |], nan);
          ([| 0. |], 1.);
          ([| -0. |], 1.);
          ([| inf |], 0.);
          ([| -.inf |], 0.);
        ];
      prop "is even, bit for bit, at float64"
        (signed_floats Nx.float64 ~emin:(-1074) ~emax:1023) (fun x ->
          bits_equal Nx.float64 (Nx.i0e x) (Nx.i0e (Nx.neg x)));
      prop "is even, bit for bit, at float32"
        (signed_floats Nx.float32 ~emin:(-149) ~emax:127) (fun x ->
          bits_equal Nx.float32 (Nx.i0e x) (Nx.i0e (Nx.neg x)));
      prop "decreases on (0, inf)" (log_batch 1e-3 1e3) (fun x ->
          increasing ~error:(in_ulps 8.) Nx.i0e (Nx.add_s x 1e-3) x);
    ]

let i1e =
  let f = one { u = Nx.i1e } in
  group "i1e"
    [
      golden_cases ~bound:(everywhere (Ulps 8)) "i1e" f;
      cases "within one ulp at every narrow float" ~id:dtype_id narrows
        (narrow1 ~bound:(everywhere (Ulps 8)) f);
      edge_cases f
        [
          ([| nan |], nan);
          ([| 0. |], 0.);
          ([| -0. |], -0.);
          ([| inf |], 0.);
          ([| -.inf |], -0.);
        ];
      prop "is odd, bit for bit, at float64"
        (signed_floats Nx.float64 ~emin:(-1074) ~emax:1023) (fun x ->
          bits_equal Nx.float64 (Nx.neg (Nx.i1e x)) (Nx.i1e (Nx.neg x)));
      prop "is odd, bit for bit, at float32"
        (signed_floats Nx.float32 ~emin:(-149) ~emax:127) (fun x ->
          bits_equal Nx.float32 (Nx.neg (Nx.i1e x)) (Nx.i1e (Nx.neg x)));
      prop "lies in (0, i0e) on (0, inf)" (log_batch 1e-3 1e3) (fun x ->
          Array.iter2
            (fun i1 i0 ->
              Windtrap.greater float_exact ~than:0. i1;
              Windtrap.less float_exact ~than:i0 i1)
            (values (Nx.i1e x))
            (values (Nx.i0e x)));
    ]

(* The section *)

(* A result reaching one of the functions' logarithm constants has the same bits
   on every system: the constants are written out, never computed by the C
   library. The bits are a baseline recorded on one system. *)
let platforms () =
  let lgamma = one { u = Nx.lgamma } in
  let log_ndtr = one { u = Nx.log_ndtr } in
  let betainc = three { c = Nx.betainc } in
  let show (type s) (dt : (float, s) Nx.dtype) name { f } points =
    let k = Array.length (List.hd points) and n = List.length points in
    let column i =
      Nx.create dt [| n |] (Array.of_list (List.map (fun p -> p.(i)) points))
    in
    Array.iter
      (Printf.printf "%s %s %h\n" (D.name dt) name)
      (values (Nx.cast Nx.float64 (f (Array.init k column))))
  in
  let all dt =
    show dt "lgamma" lgamma [ [| -2.5 |]; [| -0.5 |] ];
    show dt "log_ndtr" log_ndtr [ [| -40. |] ];
    show dt "betainc" betainc [ [| 0.5; 0.5; 0.95 |] ]
  in
  all Nx.float64;
  all Nx.float32;
  expect (output ())
  @@ __POS_OF__
       {|
        float64 lgamma -0x1.ccbf9f5ed0f2p-5
        float64 lgamma 0x1.43f89a3f0edd6p+0
        float64 log_ndtr -0x1.924de16d8b4aep+9
        float64 betainc 0x1.b67e7a9468945p-1
        float32 lgamma -0x1.ccbfcp-5
        float32 lgamma 0x1.43f89ap+0
        float32 log_ndtr -0x1.924de2p+9
        float32 betainc 0x1.b67e7ap-1
        |}

let section =
  let f64 = Nx.float64 in
  group "the section"
    [
      test "results through a logarithm constant keep their bits" platforms;
      test "a zero-size argument gives a zero-size result" (fun () ->
          equal (array int) [| 0 |] (Nx.shape (Nx.erfc (Nx.zeros f64 [| 0 |])));
          equal (array int) [| 2; 0 |]
            (Nx.shape
               (Nx.gammainc (Nx.zeros f64 [| 2; 0 |]) (Nx.ones f64 [| 1 |])));
          equal (array int) [| 0; 3 |]
            (Nx.shape
               (Nx.betainc
                  (Nx.zeros f64 [| 0; 3 |])
                  (Nx.ones f64 [| 3 |]) (Nx.full f64 [| 1 |] 0.5))));
      test "arguments broadcast, each element its arguments' value" (fun () ->
          let a = [| 0.5; 2.; 30. |] and b = [| 1.5; 4. |] in
          let x = [| 0.1; 0.5; 0.9; 1. |] in
          let r =
            Nx.betainc
              (Nx.create f64 [| 3; 1; 1 |] a)
              (Nx.create f64 [| 1; 2; 1 |] b)
              (Nx.create f64 [| 4 |] x)
          in
          equal (array int) [| 3; 2; 4 |] (Nx.shape r);
          let one i j k =
            let s v = Nx.scalar f64 v in
            Nx.item [] (Nx.betainc (s a.(i)) (s b.(j)) (s x.(k)))
          in
          equal (array float_exact)
            (Array.init 24 (fun n -> one (n / 8) (n / 4 mod 2) (n mod 4)))
            (values r));
    ]

let () =
  exit
    (run "nx special"
       [
         erfc;
         erfinv;
         ndtr;
         log_ndtr;
         ndtri;
         lgamma;
         digamma;
         lbeta;
         gammainc;
         gammaincc;
         log_gammainc;
         log_gammaincc;
         gammaincinv;
         gammainccinv;
         betainc;
         betaincc;
         betaincinv;
         betainccinv;
         log_betainc;
         log_betaincc;
         i0e;
         i1e;
         section;
       ])
