(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's special functions against correctly rounded references (nx's
   golden/special/, written by its gen/special.py), at the bounds nx.mli
   states per function and dtype; the narrow floats at the rule nx.mli states
   for them; the laws the functions keep; their edges. *)

open Windtrap
module D = Nx_array.Dtype

type 's v = (float, 's, Nx.host) Nx.t

let values = Nx.to_array

(* Goldens

   A golden is a table of hexadecimal floats, one row per point and dtype:
   the arguments, the value correctly rounded to the dtype, and for some a
   scale, an inverse's condition number or an absolute bound's factor. *)

(* [overflow] when [value] is an infinity a finite exact value rounds to past
   the dtype's range. *)
type row = { args : float array; value : float; overflow : bool; scale : float }

let goldens = "../../../packages/nx/test/golden/special"

(* The rows of the golden of [name] at the dtype named [short] ("f32",
   "f64"). *)
let rows name short =
  let dir = Filename.concat (Filename.dirname Sys.executable_name) goldens in
  let path = Filename.concat dir (name ^ ".golden") in
  let lines = In_channel.with_open_bin path In_channel.input_lines in
  match List.filter (fun l -> not (String.starts_with ~prefix:"#" l)) lines with
  | [] -> failwith (path ^ ": no header")
  | header :: lines ->
      (* The columns are the dtype, the arguments, [f] and maybe a scale. *)
      let columns = String.split_on_char '\t' header in
      let scaled = List.nth columns (List.length columns - 1) <> "f" in
      let n = List.length columns - if scaled then 3 else 2 in
      let float = function
        | "overflow" -> Float.infinity
        | "-overflow" -> Float.neg_infinity
        | cell -> float_of_string cell
      in
      List.filter_map
        (fun line ->
          match String.split_on_char '\t' line with
          | dtype :: cells when dtype = short ->
              let overflow =
                match List.nth cells n with
                | "overflow" | "-overflow" -> true
                | _ -> false
              in
              let cells = Array.of_list (List.map float cells) in
              Some
                {
                  args = Array.sub cells 0 n;
                  value = cells.(n);
                  overflow;
                  scale = (if scaled then cells.(n + 1) else Float.nan);
                }
          | _ -> None)
        lines

(* The forms of a bound: [k] ulps; [2k (1 + |log f|) + 4] ulps, the
   exponential of a logarithm within [k] ulps, absolute near zeros; [k] ulps
   or [|f̂ - f| <= a ε] where [|f| < 1]; [k] ulps or [|f̂ - f| <= k ε scale];
   an inverse's [k] ulps plus its condition number times its forward
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

(* The position of the [w]-bit float whose bits are [b] among its format's
   values in increasing order: [+0] at 0, [-0] at -1. *)
let rank ~width b =
  let sign = Int64.shift_left 1L (width - 1) in
  if Int64.logand b sign = 0L then b
  else Int64.(sub (neg (logand b (sub sign 1L))) 1L)

let rank_of ~f32 x =
  if f32 then
    rank ~width:32
      (Int64.logand (Int64.of_int32 (Int32.bits_of_float x)) 0xffff_ffffL)
  else rank ~width:64 (Int64.bits_of_float x)

(* The distance between [a] and [b] in floats, as a float so that it reaches
   past [max_int]: zero between two NaNs, infinite between a NaN and a
   number. *)
let ulps ~f32 a b =
  if Float.is_nan a || Float.is_nan b then
    if Float.is_nan a && Float.is_nan b then 0. else Float.infinity
  else Int64.to_float (Int64.abs (Int64.sub (rank_of ~f32 a) (rank_of ~f32 b)))

(* [Log_ulps k]'s bound at [f]. *)
let log_ulps k f =
  (2. *. Float.of_int k *. (1. +. Float.abs (Stdlib.log (Float.abs f)))) +. 4.

(* Whether [got] meets [bound] against the correctly rounded [row]: exactly
   where the value is an exact infinity or NaN, and with its sign where it is
   zero. An overflow, read as the infinity one rank past the largest float, is
   held by rank to the bound in ulps, since its absolute error is infinite. A
   value that rounds to zero is held as the least subnormal is: its logarithm
   bounds [Log_ulps]. *)
let within ~f32 bound row got =
  let eps = if f32 then 0x1p-23 else 0x1p-52 in
  let r = row.value in
  let log_ulps k r =
    log_ulps k (if r = 0. then if f32 then 0x1p-149 else 0x1p-1074 else r)
  in
  let d = ulps ~f32 r got in
  let error = Float.abs (got -. r) in
  let in_ulps =
    match bound with
    | Ulps k | Near_zeros (k, _) | Scaled k -> Float.of_int k
    | Log_ulps k -> log_ulps k r
    | Inverse (k, forward) ->
        Float.of_int k +. (row.scale *. Float.of_int forward)
  in
  if (not (Float.is_finite r)) && not row.overflow then d = 0.
  else if r = 0. && Float.sign_bit got <> Float.sign_bit r then false
  else if row.overflow then d <= in_ulps
  else
    d <= in_ulps
    ||
    match bound with
    | Near_zeros (_, a) -> Float.abs r < 1. && error <= Float.of_int a *. eps
    | Scaled k -> error <= Float.of_int k *. eps *. row.scale
    | Ulps _ | Log_ulps _ | Inverse _ -> false

type f = { f : 's. 's v array -> 's v }

let eval (type s) (dt : (float, s) Nx.dtype) { f } rows =
  let n = List.length rows in
  let arity = match rows with [] -> 0 | r :: _ -> Array.length r.args in
  let args =
    Array.init arity (fun i ->
        let column = List.map (fun r -> r.args.(i)) rows in
        Nx.create dt [| n |] (Array.of_list column))
  in
  values (Nx.cast Nx.float64 (f args))

let pp_args ppf args =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
    (fun ppf x -> Format.fprintf ppf "%h" x)
    ppf (Array.to_list args)

(* [check ~bound name f] holds [f] to [bound args] at every row of [name]'s
   golden that [keep] keeps, at float32 and float64, and fails listing the
   rows that miss it, worst first. *)
let check ?(keep = fun _ -> true) ~bound name f =
  List.map
    (fun (short, f32) ->
      test short (fun () ->
          let rows = List.filter keep (rows name short) in
          let got =
            if f32 then eval Nx.float32 f rows else eval Nx.float64 f rows
          in
          let misses =
            List.concat
              (List.mapi
                 (fun i row ->
                   let b = bound row.args in
                   if within ~f32 b row got.(i) then []
                   else [ (ulps ~f32 row.value got.(i), row, got.(i), b) ])
                 rows)
          in
          let misses =
            List.sort (fun (a, _, _, _) (b, _, _, _) -> compare b a) misses
          in
          if misses <> [] then
            let pp_miss ppf (d, row, got, b) =
              Format.fprintf ppf "at (%a): %h, not %h (%g ulps), bound %a"
                pp_args row.args got row.value d pp_bound b;
              if Float.is_finite row.scale then
                Format.fprintf ppf " (scale %g)" row.scale
            in
            failf "%d of %d rows miss their bound:@\n%a" (List.length misses)
              (List.length rows)
              (Format.pp_print_list pp_miss)
              (List.filteri (fun i _ -> i < 8) misses)))
    [ ("f32", true); ("f64", false) ]

(* Narrow floats

   A narrow float computes at float32 and rounds once. Its reference is the
   float64 result, whose own bound the goldens hold: rounded to the narrow
   dtype it is the correctly rounded value, or one of its neighbours where
   the float64 result lies within its bound of a narrow midpoint. *)

type narrow =
  | Narrow : (float, 's) Nx.dtype * int * int -> narrow
      (** A dtype, its precision and its least exponent. *)

let narrows =
  [
    Narrow (Nx.float16, 11, -14);
    Narrow (Nx.bfloat16, 8, -126);
    Narrow (Nx.float8_e4m3fn, 4, -6);
    Narrow (Nx.float8_e5m2, 3, -14);
    Narrow (Nx.float4_e2m1fn, 2, 0);
  ]

(* The unit in the last place of [x] in a format of precision [p] and least
   exponent [emin]: the least subnormal at zero. *)
let ulp ~p ~emin x =
  let _, e = Float.frexp x in
  let e = if x = 0. then emin else Int.max (e - 1) emin in
  Float.ldexp 1. (e - (p - 1))

(* Every float of a narrow dtype, and the ranks of a value of it. *)
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
  Array.map (fun b -> Int64.to_int (rank ~width (Int64.of_int b))) codes

(* [hold dt ~bound ?scale f args n] holds [f] at [args], [n] floats of the
   narrow dtype [dt] each: within one ulp of the reference where the bound is
   in ulps, within the float32 bound plus half an ulp of [dt] where it is
   absolute, a zero with the reference's sign, and a non-finite value
   exactly. *)
let hold (type s) (dt : (float, s) Nx.dtype) p emin ~bound ?scale { f }
    (args : s v array) n =
  let got = f args in
  let args64 = Array.map (Nx.cast Nx.float64) args in
  let v = f args64 in
  let xs = Array.map values args64 in
  let at i = Array.map (fun a -> a.(i)) xs in
  let scales =
    match scale with
    | Some { f } -> values (f args64)
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
  (* What the float64 result is in the narrow dtype: a dtype with no
     infinities has no other value for one. *)
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
    let h = 0.5 *. ulp ~p ~emin v in
    let absolute =
      match bound (at i) with
      | Ulps _ | Log_ulps _ | Inverse _ -> false
      | Near_zeros (_, a) ->
          Float.abs v < 1.
          && Float.abs (g -. v) <= (Float.of_int a *. 0x1p-23) +. h
      | Scaled k ->
          Float.abs (g -. v) <= (Float.of_int k *. 0x1p-23 *. scales.(i)) +. h
    in
    let ok =
      if not (Float.is_finite v) then
        (Float.is_nan g && Float.is_nan cs.(i)) || g = cs.(i)
      else
        (d <= 1
        && (g <> 0. || cs.(i) <> 0.
           || Float.sign_bit g = Float.sign_bit cs.(i)))
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

(* [narrow ~bound ?scale f] holds [f] at every float of each narrow dtype. *)
let narrow ~bound ?scale f =
  List.map
    (fun (Narrow (dt, p, emin)) ->
      test (D.name dt) (fun () ->
          let x = every dt in
          hold dt p emin ~bound ?scale f [| x |] (Nx.numel x)))
    narrows

(* [narrow2 ?keep ~bound ?scale ~firsts ~seconds f] holds [f] of two
   arguments as [narrow] holds one: at every pair of floats of a dtype of at
   most 8 bits, and of a 16-bit one at every float as either argument against
   each of [firsts] or [seconds] as the other, rounded to the dtype; at the
   pairs [keep] keeps. *)
let narrow2 ?(keep = fun _ -> true) ~bound ?scale ~firsts ~seconds f =
  List.map
    (fun (Narrow (dt, p, emin)) ->
      test (D.name dt) (fun () ->
          let all = Array.to_list (values (Nx.cast Nx.float64 (every dt))) in
          let pairs =
            if D.bits dt <= 8 then
              List.concat_map (fun a -> List.map (fun x -> (a, x)) all) all
            else
              List.concat_map (fun a -> List.map (fun x -> (a, x)) all) firsts
              @ List.concat_map
                  (fun x -> List.map (fun a -> (a, x)) all)
                  seconds
          in
          let pairs = List.filter (fun (a, x) -> keep [| a; x |]) pairs in
          let n = List.length pairs in
          let column sel =
            Nx.cast dt
              (Nx.create Nx.float64 [| n |]
                 (Array.of_list (List.map sel pairs)))
          in
          hold dt p emin ~bound ?scale f [| column fst; column snd |] n))
    narrows

(* Suites against the goldens *)

type u = { u : 's. 's v -> 's v }

let unary name ~bound ?scale { u } =
  let f = { f = (fun a -> u a.(0)) } in
  let scale = Option.map (fun { u } -> { f = (fun a -> u a.(0)) }) scale in
  group name
    [
      group "against the goldens" (check ~bound name f);
      group "at every narrow float" (narrow ~bound ?scale f);
    ]

let everywhere b _ = b

(* A bound for [x > 0] and another below. *)
let by_sign ~positive ~negative args =
  if args.(0) < 0. then negative else positive

(* An inverse's condition number [|p / (x F'(x))|] at [x], its result. *)
let erfinv_kappa =
  {
    u =
      (fun p ->
        let x = Nx.erfinv p in
        let zero = Nx.equal_s p 0. in
        let k =
          Nx.div
            (Nx.mul_s (Nx.mul p (Nx.exp (Nx.square x))) (Float.sqrt Float.pi))
            (Nx.mul_s (Nx.where zero (Nx.ones_like x) x) 2.)
        in
        Nx.where zero (Nx.ones_like p) (Nx.abs k));
  }

let ndtri_kappa =
  {
    u =
      (fun p ->
        let x = Nx.ndtri p in
        let slope =
          Nx.mul_s
            (Nx.exp (Nx.mul_s (Nx.square x) (-0.5)))
            (1. /. Float.sqrt (2. *. Float.pi))
        in
        Nx.abs (Nx.div p (Nx.mul x slope)));
  }

let lgamma_scale =
  { u = (fun x -> Nx.add_s (Nx.abs (Nx.lgamma (Nx.rsub_s 1. x))) 1.) }

let digamma_scale =
  {
    u =
      (fun x ->
        let px = Nx.mul_s x Float.pi in
        let cot = Nx.div (Nx.cos px) (Nx.sin px) in
        Nx.add_s (Nx.abs (Nx.mul_s cot Float.pi)) 1.);
  }

let error_function =
  group "error function"
    [
      unary "erfc" ~bound:(everywhere (Ulps 8)) { u = Nx.erfc };
      unary "erfinv"
        ~bound:(everywhere (Inverse (4, 8)))
        ~scale:erfinv_kappa { u = Nx.erfinv };
    ]

let normal =
  group "standard normal"
    [
      unary "ndtr" ~bound:(everywhere (Ulps 16)) { u = Nx.ndtr };
      unary "log_ndtr" ~bound:(everywhere (Ulps 32)) { u = Nx.log_ndtr };
      unary "ndtri"
        ~bound:(everywhere (Inverse (4, 16)))
        ~scale:ndtri_kappa { u = Nx.ndtri };
    ]

let gamma =
  group "gamma"
    [
      unary "lgamma"
        ~bound:(by_sign ~positive:(Near_zeros (16, 16)) ~negative:(Scaled 16))
        ~scale:lgamma_scale { u = Nx.lgamma };
      unary "digamma"
        ~bound:(by_sign ~positive:(Near_zeros (16, 16)) ~negative:(Scaled 16))
        ~scale:digamma_scale { u = Nx.digamma };
      (let bound = everywhere (Near_zeros (256, 512)) in
       let f = { f = (fun a -> Nx.lbeta a.(0) a.(1)) } in
       group "lbeta"
         [
           group "against the goldens" (check ~bound "lbeta" f);
           group "at every narrow float"
             (narrow2 ~bound ~firsts:[ 0.5; 10. ] ~seconds:[ 1.; 20. ] f);
         ]);
    ]

let bessel =
  group "modified Bessel"
    [
      unary "i0e" ~bound:(everywhere (Ulps 8)) { u = Nx.i0e };
      unary "i1e" ~bound:(everywhere (Ulps 8)) { u = Nx.i1e };
    ]

(* The incomplete gamma family. Its bounds hold for [a <= 2^20]; above, the
   goldens' few rows are edges. *)

let in_domain row =
  let a = row.args.(0) in
  not (Float.is_finite a && a > 0x1p20)

type b = { b : 's. 's v -> 's v -> 's v }

let binary name ~bound ?scale ~firsts ~seconds { b } =
  let f = { f = (fun a -> b a.(0) a.(1)) } in
  let scale =
    Option.map (fun { b } -> { f = (fun a -> b a.(0) a.(1)) }) scale
  in
  group name
    [
      group "against the goldens" (check ~keep:in_domain ~bound name f);
      group "at every narrow float"
        (narrow2
           ~keep:(fun args -> args.(0) <= 0x1p20)
           ~bound ?scale ~firsts ~seconds f);
    ]

(* An inverse's condition number [|p / (x ∂ₓF)|] at its result [x], [F] the
   tail, [∂ₓP = x^(a - 1) e^-x / Γ(a)]. *)
let quantile_kappa (inverse : b) =
  {
    b =
      (fun a p ->
        let x = inverse.b a p in
        let open Nx in
        let density = sub (sub (mul a (log x)) x) (lgamma a) in
        exp (sub (log p) density));
  }

(* The forward bound at [f], in ulps, rounded down. *)
let forward f = int_of_float (log_ulps 16 f)
let inverse_bound args = Inverse (4, forward args.(1))

(* The partners of every 16-bit float: two of the parameter and two of the
   variable, one and two for an inverse, whose program is three times as
   long. *)
let parameters = [ 0.5; 10. ]
let variables = [ 1.; 20. ]

let incomplete_gamma =
  group "incomplete gamma"
    [
      binary "gammainc" ~bound:(everywhere (Log_ulps 16)) ~firsts:parameters
        ~seconds:variables { b = Nx.gammainc };
      binary "gammaincc" ~bound:(everywhere (Log_ulps 16)) ~firsts:parameters
        ~seconds:variables { b = Nx.gammaincc };
      binary "log_gammainc"
        ~bound:(everywhere (Near_zeros (16, 16)))
        ~firsts:parameters ~seconds:variables { b = Nx.log_gammainc };
      binary "log_gammaincc"
        ~bound:(everywhere (Near_zeros (16, 16)))
        ~firsts:parameters ~seconds:variables { b = Nx.log_gammaincc };
      binary "gammaincinv" ~bound:inverse_bound
        ~scale:(quantile_kappa { b = Nx.gammaincinv })
        ~firsts:[ 2.5 ] ~seconds:[ 0.1; 0.9 ] { b = Nx.gammaincinv };
      binary "gammainccinv" ~bound:inverse_bound
        ~scale:(quantile_kappa { b = Nx.gammainccinv })
        ~firsts:[ 2.5 ] ~seconds:[ 0.1; 0.9 ] { b = Nx.gammainccinv };
    ]

let incomplete_beta =
  let ternary name ~bound f =
    group name (check ~bound:(everywhere bound) name f)
  in
  group "incomplete beta"
    [
      ternary "betainc" ~bound:(Log_ulps 32)
        { f = (fun a -> Nx.betainc a.(0) a.(1) a.(2)) };
      ternary "betaincc" ~bound:(Log_ulps 32)
        { f = (fun a -> Nx.betaincc a.(0) a.(1) a.(2)) };
      ternary "log_betainc"
        ~bound:(Near_zeros (32, 32))
        { f = (fun a -> Nx.log_betainc a.(0) a.(1) a.(2)) };
      ternary "log_betaincc"
        ~bound:(Near_zeros (32, 32))
        { f = (fun a -> Nx.log_betaincc a.(0) a.(1) a.(2)) };
    ]

(* Laws

   Identities the functions keep, at float64 over drawn batches, each within
   the sum of its terms' bounds. *)

let eps = epsilon_float

let batch lo hi =
  Gen.with_pp Nx.pp
    (Gen.map
       (fun xs -> Nx.create Nx.float64 [| Array.length xs |] xs)
       (Gen.array ~size:(Gen.constant 32) (Gen.float_range lo hi)))

(* Floats of [dt] of either sign, their magnitudes [2^e] for [e] drawn in
   [emin, emax], with zeros, the least subnormal and the largest float among
   them. *)
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
       (Gen.array ~size:(Gen.constant 32) one))

(* Values log-uniform in [lo, hi]. *)
let log_batch lo hi =
  Gen.with_pp Nx.pp
    (Gen.map
       (fun xs -> Nx.exp (Nx.create Nx.float64 [| Array.length xs |] xs))
       (Gen.array ~size:(Gen.constant 32)
          (Gen.float_range (Stdlib.log lo) (Stdlib.log hi))))

(* [within tol expected actual]: elementwise, within [tol] absolutely. *)
let within tol expected actual =
  let e = values expected and a = values actual and t = values tol in
  Array.iteri
    (fun i e ->
      at_most
        ~msg:(Printf.sprintf "the error at element %d" i)
        float_exact ~than:t.(i)
        (Float.abs (a.(i) -. e)))
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
      equal (array int64) (values (Nx.bitcast Nx.int64 a))
        (values (Nx.bitcast Nx.int64 b))
  | _ ->
      equal (array int32) (values (Nx.bitcast Nx.int32 a))
        (values (Nx.bitcast Nx.int32 b))

let increasing f x y =
  Array.iter2
    (fun a b -> at_most float_exact ~than:b a)
    (values (f x)) (values (f y))

let laws =
  let open Nx in
  group "laws"
    [
      prop "erfc (-x) is 2 - erfc x" (batch 0. 30.) (fun x ->
          let tol = mul_s (add (erfc (neg x)) (erfc x)) (17. *. eps) in
          within tol (rsub_s 2. (erfc x)) (erfc (neg x)));
      prop "erfinv inverts erf" (batch (-5.) 5.) (fun x ->
          (* [κ = |p √π e^(x²) / (2x)|] at [p = erf x], [1] at [0], and [erf]
             within 2 ulps: [erfinv p] is within [4 + 8κ] ulps of [x], and
             [p]'s error moves it by [2κ] more. *)
          let p = erf x in
          let zero = equal_s x 0. in
          let k =
            div
              (mul_s (mul p (exp (square x))) (Float.sqrt Float.pi))
              (mul_s (where zero (ones_like x) x) 2.)
          in
          let k = where zero (ones_like x) (abs k) in
          let tol = mul_s (mul (abs x) (add_s (mul_s k 10.) 4.)) eps in
          let tol =
            where (less_s (abs p) 1.) tol (full_like x Float.infinity)
          in
          within tol x (erfinv p));
      prop "ndtr x + ndtr (-x) is 1" (batch (-40.) 40.) (fun x ->
          within
            (full_like x (33. *. eps))
            (ones_like x)
            (add (ndtr x) (ndtr (neg x))));
      prop "log_ndtr is log ndtr where ndtr is normal" (batch (-37.) 9.)
        (fun x ->
          let l = log_ndtr x in
          within (mul_s (add_s (abs l) 1.) (48. *. eps)) (log (ndtr x)) l);
      prop "ndtri inverts ndtr" (batch (-8.) 8.) (fun x ->
          let p = ndtr x in
          let slope = mul x (exp (mul_s (square x) (-0.5))) in
          let zero = equal_s x 0. in
          let kappa = abs (div p (where zero (ones_like x) slope)) in
          let kappa =
            where zero (ones_like x) (mul_s kappa (Float.sqrt (2. *. Float.pi)))
          in
          let tol = mul_s (mul (abs x) (add_s (mul_s kappa 32.) 8.)) eps in
          within (add_s tol (8. *. eps)) x (ndtri p));
      prop "ndtr and log_ndtr increase" (batch (-50.) 50.) (fun x ->
          let y = add_s x 1e-3 in
          increasing ndtr x y;
          increasing log_ndtr x y);
      prop "lgamma (x + 1) is lgamma x + log x" (batch 1e-3 50.) (fun x ->
          let x = grid x in
          let l = lgamma x and lx = log x in
          let tol = mul_s (add_s (add (abs l) (abs lx)) 1.) (40. *. eps) in
          within tol (add l lx) (lgamma (add_s x 1.)));
      prop "lgamma x + lgamma (1 - x) is log (pi / sin (pi x)) on (0, 1)"
        (batch 1e-3 0.999) (fun x ->
          let x = grid x in
          let r = rsub_s (Stdlib.log Float.pi) (log (sinpi x)) in
          let tol = mul_s (add_s (abs r) 1.) (40. *. eps) in
          within tol r (add (lgamma x) (lgamma (rsub_s 1. x))));
      prop "digamma (x + 1) is digamma x + 1/x" (batch 1e-3 50.) (fun x ->
          let x = grid x in
          let d = digamma x and r = recip x in
          let tol = mul_s (add_s (add (abs d) (abs r)) 1.) (40. *. eps) in
          within tol (add d r) (digamma (add_s x 1.)));
      prop "digamma (1 - x) - digamma x is pi cot (pi x) on (0, 1)"
        (batch 1e-3 0.999) (fun x ->
          let x = grid x in
          let cospi = sin (mul_s (rsub_s 0.5 x) Float.pi) in
          let r = mul_s (div cospi (sinpi x)) Float.pi in
          let d1 = digamma (rsub_s 1. x) and d0 = digamma x in
          let size = add (add (abs d1) (abs d0)) (abs r) in
          within (mul_s (add_s size 1.) (32. *. eps)) r (sub d1 d0));
      prop "lbeta is symmetric"
        (Gen.pair (batch 1e-3 1e3) (batch 1e-3 1e3))
        (fun (a, b) -> within (zeros_like a) (lbeta a b) (lbeta b a));
      prop "lbeta a 1 is -log a" (batch 1e-3 1e6) (fun a ->
          let r = neg (log a) in
          let tol = mul_s (add_s (abs r) 1.) (512. *. eps) in
          within tol r (lbeta a (ones_like a)));
      prop "i0e is even and i1e odd, bit for bit, at float64"
        (signed_floats Nx.float64 ~emin:(-1074) ~emax:1023) (fun x ->
          bits_equal Nx.float64 (i0e x) (i0e (neg x));
          bits_equal Nx.float64 (neg (i1e x)) (i1e (neg x)));
      prop "i0e is even and i1e odd, bit for bit, at float32"
        (signed_floats Nx.float32 ~emin:(-149) ~emax:127) (fun x ->
          bits_equal Nx.float32 (i0e x) (i0e (neg x));
          bits_equal Nx.float32 (neg (i1e x)) (i1e (neg x)));
      prop "i0e decreases and 0 < i1e < i0e on (0, inf)" (log_batch 1e-3 1e3)
        (fun x ->
          increasing i0e (add_s x 1e-3) x;
          Array.iter2
            (fun i1 i0 ->
              Windtrap.greater float_exact ~than:0. i1;
              Windtrap.less float_exact ~than:i0 i1)
            (values (i1e x)) (values (i0e x)));
    ]

(* The incomplete functions' laws, at float64, each within the sum of its
   terms' bounds. *)

let pairs = Gen.pair (log_batch 1e-3 1e4) (log_batch 1e-3 1e4)

(* The linear bound in ulps of [f], as a tolerance: [u] for an ulp. *)
let linear f =
  if f = 0. then 0. else log_ulps 16 f *. (eps /. 2.) *. Float.abs f

let elementwise f a b c =
  let a = values a and b = values b and c = values c in
  Array.iteri (fun i a -> f i a b.(i) c.(i)) a

let close ~than i error =
  at_most
    ~msg:(Printf.sprintf "the error at element %d" i)
    float_exact ~than error

let gamma_laws =
  let open Nx in
  group "incomplete gamma laws"
    [
      prop "gammainc + gammaincc is 1" pairs (fun (a, x) ->
          elementwise
            (fun i p q _ ->
              close i
                ~than:(linear p +. linear q +. eps)
                (Float.abs (p +. q -. 1.)))
            (gammainc a x) (gammaincc a x) x);
      prop "log_gammainc is log1p (-gammaincc) where gammaincc < 1/2" pairs
        (fun (a, x) ->
          elementwise
            (fun i lp q _ ->
              if q < 0.5 then
                close i
                  ~than:
                    ((16. *. eps *. Float.max 1. (Float.abs lp))
                    +. (2. *. linear q))
                  (Float.abs (lp -. Stdlib.log1p (-.q))))
            (log_gammainc a x) (gammaincc a x) x);
      prop "log_gammaincc is log1p (-gammainc) where gammainc < 1/2" pairs
        (fun (a, x) ->
          elementwise
            (fun i lq p _ ->
              if p < 0.5 then
                close i
                  ~than:
                    ((16. *. eps *. Float.max 1. (Float.abs lq))
                    +. (2. *. linear p))
                  (Float.abs (lq -. Stdlib.log1p (-.p))))
            (log_gammaincc a x) (gammainc a x) x);
      prop "gammainc is exp log_gammainc" pairs (fun (a, x) ->
          bits_equal Nx.float64 (exp (log_gammainc a x)) (gammainc a x));
      prop "gammaincinv inverts gammainc" pairs (fun (a, x) ->
          let p = gammainc a x in
          let back = gammaincinv a p in
          let kappa = values ((quantile_kappa { b = gammaincinv }).b a p) in
          elementwise
            (fun i x p back ->
              if p >= 0x1p-1022 && p < 1. then
                close i
                  ~than:((4. +. (2. *. kappa.(i) *. log_ulps 16 p)) *. eps *. x)
                  (Float.abs (back -. x)))
            x p back);
      prop "gammainccinv inverts gammaincc" pairs (fun (a, x) ->
          let q = gammaincc a x in
          let back = gammainccinv a q in
          let kappa = values ((quantile_kappa { b = gammainccinv }).b a q) in
          elementwise
            (fun i x q back ->
              if q >= 0x1p-1022 && q < 1. then
                close i
                  ~than:((4. +. (2. *. kappa.(i) *. log_ulps 16 q)) *. eps *. x)
                  (Float.abs (back -. x)))
            x q back);
      prop "gammainc increases in x and decreases in a"
        (Gen.pair pairs (log_batch 1e-6 1.))
        (fun ((a, x), d) ->
          let up = mul x (add_s d 1.) and more = mul a (add_s d 1.) in
          elementwise
            (fun i p q r ->
              at_most
                ~msg:(Printf.sprintf "P(a, x) at element %d" i)
                float_exact ~than:q p;
              at_most
                ~msg:(Printf.sprintf "P(a', x) at element %d" i)
                float_exact ~than:p r)
            (gammainc a x) (gammainc a up) (gammainc more x));
      prop "gammaincc 1 x is exp (-x)" (log_batch 1e-300 700.) (fun x ->
          elementwise
            (fun i q e _ ->
              close i ~than:(linear e +. (2. *. eps *. e)) (Float.abs (q -. e)))
            (gammaincc (ones_like x) x) (exp (neg x)) x);
      prop "gammainc 1/2 x is erf (sqrt x)" (log_batch 1e-6 30.) (fun x ->
          elementwise
            (fun i p e _ ->
              close i ~than:(linear e +. (4. *. eps *. e)) (Float.abs (p -. e)))
            (gammainc (full_like x 0.5) x) (erf (sqrt x)) x);
      prop "log_gammainc a x tends to a log x - lgamma (a + 1) as x goes to 0"
        (Gen.pair (log_batch 1e-3 1e3) (log_batch 1e-300 1e-200))
        (fun (a, x) ->
          let r = sub (mul a (log x)) (lgamma (add_s a 1.)) in
          elementwise
            (fun i l r _ ->
              close i ~than:(32. *. eps *. Float.abs r) (Float.abs (l -. r)))
            (log_gammainc a x) r x);
    ]

let beta_laws =
  let open Nx in
  let shapes = log_batch 1e-3 1e4 in
  group "incomplete beta laws"
    [
      prop "betaincc a b x is betainc b a (1 - x) bit for bit where x >= 1/2"
        (Gen.triple shapes shapes (batch 0.5 1.))
        (fun (a, b, x) ->
          (* Every fourth [x] is 1/2 and every fifth [b] is [a]: the ties. *)
          let every k = init Nx.bool [| numel x |] (fun i -> i.(0) mod k = 0) in
          let x = where (every 4) (full_like x 0.5) x in
          let b = where (every 5) a b in
          bits_equal Nx.float64 (betaincc a b x) (betainc b a (rsub_s 1. x)));
      prop "betainc and betaincc sum to 1"
        (Gen.triple shapes shapes (batch 0. 1.))
        (fun (a, b, x) ->
          let p = betainc a b x and q = betaincc a b x in
          let tol v =
            let b = mul_s (add_s (abs (log v)) 1.) (66. *. eps) in
            where (equal_s v 0.) v (mul b v)
          in
          within (add (tol p) (tol q)) (ones_like p) (add p q));
      prop "betainc increases in x"
        (Gen.triple shapes shapes (batch 0. 0.999))
        (fun (a, b, x) -> increasing (betainc a b) x (add_s x 1e-3));
      prop "betainc a 1 x is x^a and betainc 1 b x is 1 - (1 - x)^b"
        (Gen.pair (log_batch 1e-2 1e2) (batch 1e-3 0.999))
        (fun (a, x) ->
          let one = ones_like a in
          let tol v =
            let b = mul_s (add_s (abs (log v)) 1.) (70. *. eps) in
            where (equal_s v 0.) v (mul b v)
          in
          let p = pow x a in
          within (tol p) p (betainc a one x);
          let q = neg (expm1 (mul a (log1p (neg x)))) in
          within (tol q) q (betainc one a x));
      prop "exp log_betainc is betainc" (Gen.triple shapes shapes (batch 0. 1.))
        (fun (a, b, x) ->
          bits_equal Nx.float64 (exp (log_betainc a b x)) (betainc a b x));
    ]
(* Edges

   The values nx.mli states at the ends of each domain, a NaN in any argument
   included, exactly, at float64 and float32. *)

let nan = Float.nan
let inf = Float.infinity

(* [f]'s results at [points], each a row of arguments, at [dt]. *)
let at (type s) (dt : (float, s) Nx.dtype) { f } points =
  let n = List.length points in
  let k = match points with [] -> 0 | p :: _ -> Array.length p in
  let column i =
    Nx.create dt [| n |] (Array.of_list (List.map (fun p -> p.(i)) points))
  in
  values (Nx.cast Nx.float64 (f (Array.init k column)))

let edges name f table =
  let expected = Array.of_list (List.map snd table) in
  let points = List.map fst table in
  group name
    [
      test "f64" (fun () ->
          equal (array float_exact) expected (at Nx.float64 f points));
      test "f32" (fun () ->
          equal (array float_exact) expected (at Nx.float32 f points));
    ]

let ones =
  group "edges of one argument"
    [
      edges "erfc"
        { f = (fun a -> Nx.erfc a.(0)) }
        [ ([| nan |], nan); ([| -.inf |], 2.); ([| inf |], 0.); ([| 0. |], 1.);
          ([| -0. |], 1.) ];
      edges "erfinv"
        { f = (fun a -> Nx.erfinv a.(0)) }
        [ ([| nan |], nan); ([| -1. |], -.inf); ([| 1. |], inf); ([| 0. |], 0.);
          ([| -0. |], -0.); ([| 1.5 |], nan); ([| -1.5 |], nan);
          ([| inf |], nan);
          ([| -.inf |], nan) ];
      edges "ndtr"
        { f = (fun a -> Nx.ndtr a.(0)) }
        [ ([| nan |], nan); ([| -.inf |], 0.); ([| inf |], 1.); ([| 0. |], 0.5);
          ([| -0. |], 0.5) ];
      edges "log_ndtr"
        { f = (fun a -> Nx.log_ndtr a.(0)) }
        [ ([| nan |], nan); ([| -.inf |], -.inf); ([| inf |], -0.) ];
      edges "ndtri"
        { f = (fun a -> Nx.ndtri a.(0)) }
        [ ([| nan |], nan); ([| 0. |], -.inf); ([| -0. |], -.inf);
          ([| 1. |], inf); ([| 0.5 |], 0.); ([| -1e-30 |], nan);
          ([| 1.5 |], nan); ([| inf |], nan);
          ([| -.inf |], nan) ];
      edges "lgamma"
        { f = (fun a -> Nx.lgamma a.(0)) }
        [ ([| nan |], nan); ([| 0. |], inf); ([| -0. |], inf); ([| -1. |], inf);
          ([| -2. |], inf); ([| inf |], inf); ([| -.inf |], nan);
          ([| 1. |], 0.);
          ([| 2. |], 0.) ];
      edges "digamma"
        { f = (fun a -> Nx.digamma a.(0)) }
        [ ([| nan |], nan); ([| 0. |], -.inf); ([| -0. |], inf);
          ([| -1. |], nan);
          ([| -2. |], nan); ([| inf |], inf); ([| -.inf |], nan) ];
      edges "i0e"
        { f = (fun a -> Nx.i0e a.(0)) }
        [ ([| nan |], nan); ([| 0. |], 1.); ([| -0. |], 1.); ([| inf |], 0.);
          ([| -.inf |], 0.) ];
      edges "i1e"
        { f = (fun a -> Nx.i1e a.(0)) }
        [ ([| nan |], nan); ([| 0. |], 0.); ([| -0. |], -0.); ([| inf |], 0.);
          ([| -.inf |], -0.) ];
      edges "lbeta"
        { f = (fun a -> Nx.lbeta a.(0) a.(1)) }
        [ ([| nan; 1. |], nan); ([| 1.; nan |], nan); ([| 0.; 1. |], inf);
          ([| 1.; 0. |], inf); ([| 0.; 0. |], inf); ([| inf; 1. |], -.inf);
          ([| 1.; inf |], -.inf); ([| 0.; inf |], nan); ([| -1.; 1. |], nan);
          ([| 1.; -1. |], nan) ];
    ]

(* The incomplete gamma functions at [(a, x)]: [P], with [Q = 1 - P], [log P]
   and [log Q] from it, but where the table gives them. *)
let gamma_edges =
  let row a x p = ([| a; x |], p) in
  let table =
    [ row 1. 0. 0.; row 0.5 0. 0.; row 1. (-0.) 0.; row 0.5 (-0.) 0.;
      row 1. inf 1.; row 0. inf 1.; row inf 0. 0.;
      row inf inf nan; row 0. 0. nan; row (-1.) 1. nan; row 1. (-1.) nan;
      row nan 1. nan; row 1. nan nan; row nan 0. nan ]
  in
  let tail f = List.map (fun (args, p) -> (args, f p)) table in
  group "edges of the incomplete gamma"
    [
      edges "gammainc" { f = (fun a -> Nx.gammainc a.(0) a.(1)) } table;
      edges "gammaincc"
        { f = (fun a -> Nx.gammaincc a.(0) a.(1)) }
        (tail (fun p -> 1. -. p));
      edges "log_gammainc"
        { f = (fun a -> Nx.log_gammainc a.(0) a.(1)) }
        (tail Stdlib.log);
      edges "log_gammaincc"
        { f = (fun a -> Nx.log_gammaincc a.(0) a.(1)) }
        (tail (fun p -> Stdlib.log (1. -. p)));
      (* As [a] goes to [+inf], [Q] tends to 1 from below for finite [x > 0]:
         [log Q] reaches 0 from below. *)
      group "the limits at a = +inf"
        (List.map
           (fun (name, g, v) ->
             edges name { f = (fun a -> g.b a.(0) a.(1)) }
               (List.map (fun x -> ([| inf; x |], v)) [ 0x1p-149; 1.; 1e30 ]))
           [
             ("gammainc", { b = Nx.gammainc }, 0.);
             ("gammaincc", { b = Nx.gammaincc }, 1.);
             ("log_gammainc", { b = Nx.log_gammainc }, -.inf);
             ("log_gammaincc", { b = Nx.log_gammaincc }, -0.);
           ]);
      (* At [a = 0] each is its limit, [P = 1 - a E1(x) + O(a^2)] for [x > 0]:
         [log P] reaches 0 from below. *)
      group "the limits at a = 0"
        (List.map
           (fun (name, g, v) ->
             edges name { f = (fun a -> g.b a.(0) a.(1)) }
               (List.concat_map
                  (fun x -> [ ([| 0.; x |], v); ([| -0.; x |], v) ])
                  [ 0x1p-149; 1e-30; 0.5; 1.1; 3.; 30.; 1e30 ]))
           [
             ("gammainc", { b = Nx.gammainc }, 1.);
             ("gammaincc", { b = Nx.gammaincc }, 0.);
             ("log_gammainc", { b = Nx.log_gammainc }, -0.);
             ("log_gammaincc", { b = Nx.log_gammaincc }, -.inf);
           ]);
      edges "a Poisson tail gammaincc (k + 1) lam is 0 at k = -1"
        { f = (fun a -> Nx.gammaincc (Nx.add_s a.(0) 1.) a.(1)) }
        [ ([| -1.; 2.5 |], 0.) ];
      edges "gammaincinv"
        { f = (fun a -> Nx.gammaincinv a.(0) a.(1)) }
        [ ([| 1.; 0. |], 0.); ([| 1.; 1. |], inf); ([| inf; 0.5 |], inf);
          ([| inf; 0. |], 0.); ([| 0.; 0.5 |], 0.); ([| 0.; 0. |], 0.);
          ([| 0.; 1e-30 |], 0.); ([| 0.; 1. |], inf); ([| -1.; 0.5 |], nan);
          ([| 1.; 1.5 |], nan); ([| 1.; -0.5 |], nan); ([| nan; 0.5 |], nan);
          ([| 1.; nan |], nan) ];
      edges "gammainccinv"
        { f = (fun a -> Nx.gammainccinv a.(0) a.(1)) }
        [ ([| 1.; 1. |], 0.); ([| 1.; 0. |], inf); ([| inf; 0.5 |], inf);
          ([| inf; 1. |], 0.); ([| 0.; 0.5 |], 0.); ([| 0.; 1. |], 0.);
          ([| 0.; 0. |], inf); ([| -1.; 0.5 |], nan); ([| 1.; 1.5 |], nan);
          ([| 1.; -0.5 |], nan); ([| nan; 0.5 |], nan); ([| 1.; nan |], nan) ];
    ]

(* The incomplete beta functions at [(a, b, x)]: [I], [1 - I], [log I] and
   [log (1 - I)]. A tail of 1 at [x] inside (0, 1) is a limit reached from
   below, and its logarithm is [-0.]. *)
let beta_edges =
  let row a b x i ic l lc = ([| a; b; x |], (i, ic, l, lc)) in
  let invalid a b x = row a b x nan nan nan nan in
  let table =
    [ row 2. 3. 0. 0. 1. (-.inf) 0.; row 2. 3. 1. 1. 0. 0. (-.inf);
      row 0. 3. 0.5 1. 0. (-0.) (-.inf); row 2. 0. 0.5 0. 1. (-.inf) (-0.);
      row inf 3. 0.5 0. 1. (-.inf) (-0.); row 2. inf 0.5 1. 0. (-0.) (-.inf);
      invalid 0. 0. 0.5; invalid inf inf 0.5; invalid 0. 3. 0.;
      invalid 2. inf 0.; invalid 2. 0. 1.; invalid inf 3. 1.;
      invalid (-1.) 3. 0.5; invalid 2. (-1.) 0.5; invalid 2. 3. (-0.1);
      invalid 2. 3. 1.5; invalid nan 3. 0.5; invalid 2. nan 0.5;
      invalid 2. 3. nan ]
  in
  let tail sel = List.map (fun (args, v) -> (args, sel v)) table in
  group "edges of the incomplete beta"
    [
      edges "betainc"
        { f = (fun a -> Nx.betainc a.(0) a.(1) a.(2)) }
        (tail (fun (i, _, _, _) -> i));
      edges "betaincc"
        { f = (fun a -> Nx.betaincc a.(0) a.(1) a.(2)) }
        (tail (fun (_, ic, _, _) -> ic));
      edges "log_betainc"
        { f = (fun a -> Nx.log_betainc a.(0) a.(1) a.(2)) }
        (tail (fun (_, _, l, _) -> l));
      edges "log_betaincc"
        { f = (fun a -> Nx.log_betaincc a.(0) a.(1) a.(2)) }
        (tail (fun (_, _, _, lc) -> lc));
    ]

(* Shapes *)

let shapes =
  let f64 = Nx.float64 in
  group "shapes"
    [
      test "a zero-size argument gives a zero-size result" (fun () ->
          equal (array int) [| 0 |] (Nx.shape (Nx.erfc (Nx.zeros f64 [| 0 |])));
          equal (array int) [| 2; 0 |]
            (Nx.shape
               (Nx.gammainc (Nx.zeros f64 [| 2; 0 |]) (Nx.ones f64 [| 1 |])));
          equal (array int) [| 0; 3 |]
            (Nx.shape
               (Nx.betainc (Nx.zeros f64 [| 0; 3 |]) (Nx.ones f64 [| 3 |])
                  (Nx.full f64 [| 1 |] 0.5))));
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
         error_function; normal; gamma; bessel; incomplete_gamma;
         incomplete_beta; laws; gamma_laws; beta_laws; ones; gamma_edges;
         beta_edges; shapes;
       ])
