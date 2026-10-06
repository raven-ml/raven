(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's special functions against correctly rounded references (golden/special/,
   written by gen/special.py), at the bounds nx.mli states per function and
   dtype; a zero result is held to its sign. *)

open Windtrap
open Nx_test.Special

let golden name = "golden/special/" ^ name ^ ".golden"

type u = { u : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }

let unary name ~bound ?scale { u } =
  let f = { f = (fun a -> u a.(0)) } in
  let scale = Option.map (fun { u } -> { f = (fun a -> u a.(0)) }) scale in
  group name
    [
      group "against the goldens" (check ~bound (golden name) f);
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
        Nx.add_s
          (Nx.abs (Nx.mul_s (Nx.div (Nx.cos px) (Nx.sin px)) Float.pi))
          1.);
  }

let error_function =
  group "error function"
    [
      unary "erf" ~bound:(everywhere (Ulps 2)) { u = Nx.erf };
      unary "erfinv"
        ~bound:(everywhere (Inverse (4, 8)))
        ~scale:erfinv_kappa { u = Nx.erfinv };
      unary "erfc" ~bound:(everywhere (Ulps 8)) { u = Nx.erfc };
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
      group "lbeta"
        [
          group "against the goldens"
            (check
               ~bound:(everywhere (Near_zeros (256, 512)))
               (golden "lbeta")
               { f = (fun a -> Nx.lbeta a.(0) a.(1)) });
        ];
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

type b = { b : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t }

let binary name ~bound ?scale ~firsts ~seconds { b } =
  let f = { f = (fun a -> b a.(0) a.(1)) } in
  let scale =
    Option.map (fun { b } -> { f = (fun a -> b a.(0) a.(1)) }) scale
  in
  group name
    [
      group "against the goldens" (check ~keep:in_domain ~bound (golden name) f);
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

(* The partners of every narrow float: two of the parameter and two of the
   variable, three for an inverse, whose program is three times as long. *)
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
    group name (check ~bound:(everywhere bound) (golden name) f)
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

   Identities the functions keep, at float64 over drawn batches, each within the
   sum of its terms' bounds. *)

let eps = epsilon_float

let batch lo hi =
  Gen.with_pp Nx.pp
    (Gen.map
       (fun xs -> Nx.create Nx.float64 [| Array.length xs |] xs)
       (Gen.array ~size:(Gen.constant 32) (Gen.float_range lo hi)))

let values t = Nx.to_array t

(* Floats of [dt] of either sign, their magnitudes [2^e] for [e] drawn in [emin,
   emax], with zeros, the least subnormal and the largest float among them. *)
let signed_floats (type b) (dt : (float, b) Nx.dtype) ~emin ~emax =
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

(* Shapes log-uniform in [lo, hi]. *)
let shapes lo hi =
  Gen.with_pp Nx.pp (Gen.map Nx.exp (batch (Stdlib.log lo) (Stdlib.log hi)))

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

let laws =
  let open Nx in
  group "laws"
    [
      prop "erfc (-x) is 2 - erfc x" (batch 0. 30.) (fun x ->
          let tol = mul_s (add (erfc (neg x)) (erfc x)) (17. *. eps) in
          within tol (rsub_s 2. (erfc x)) (erfc (neg x)));
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
          let slope = mul x (Nx.exp (mul_s (square x) (-0.5))) in
          let zero = equal_s x 0. in
          let kappa = abs (div p (where zero (ones_like x) slope)) in
          let kappa =
            where zero (ones_like x) (mul_s kappa (Float.sqrt (2. *. Float.pi)))
          in
          let tol = mul_s (mul (abs x) (add_s (mul_s kappa 32.) 8.)) eps in
          within (add_s tol (8. *. eps)) x (ndtri p));
      prop "ndtr and log_ndtr increase" (batch (-50.) 50.) (fun x ->
          let y = add_s x 1e-3 in
          let increasing f =
            Array.iter2
              (fun a b -> Windtrap.at_most Windtrap.float_exact ~than:b a)
              (values (f x))
              (values (f y))
          in
          increasing ndtr;
          increasing log_ndtr);
      prop "lgamma (x + 1) is lgamma x + log x" (batch 1e-3 50.) (fun x ->
          (* x on a grid of 2^-20, so that x + 1 is exact *)
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
          within
            (mul_s (add_s (abs r) 1.) (512. *. eps))
            r
            (lbeta a (ones_like a)));
      prop "i0e is even and i1e odd, bit for bit, at float64"
        (signed_floats Nx.float64 ~emin:(-1074) ~emax:1023) (fun x ->
          let bits t = values (bitcast Nx.int64 t) in
          Windtrap.equal
            (Windtrap.array Windtrap.int64)
            (bits (i0e x))
            (bits (i0e (neg x)));
          Windtrap.equal
            (Windtrap.array Windtrap.int64)
            (bits (neg (i1e x)))
            (bits (i1e (neg x))));
      prop "i0e is even and i1e odd, bit for bit, at float32"
        (signed_floats Nx.float32 ~emin:(-149) ~emax:127) (fun x ->
          let bits t = values (bitcast Nx.int32 t) in
          Windtrap.equal
            (Windtrap.array Windtrap.int32)
            (bits (i0e x))
            (bits (i0e (neg x)));
          Windtrap.equal
            (Windtrap.array Windtrap.int32)
            (bits (neg (i1e x)))
            (bits (i1e (neg x))));
      prop "betaincc a b x is betainc b a (1 - x) bit for bit where x >= 1/2"
        (Gen.triple (shapes 1e-3 1e4) (shapes 1e-3 1e4) (batch 0.5 1.))
        (fun (a, b, x) ->
          (* Every fourth [x] is 1/2 and every fifth [b] is [a]: the ties. *)
          let n = numel x in
          let every k = init Nx.bool [| n |] (fun i -> i.(0) mod k = 0) in
          let x = where (every 4) (full_like x 0.5) x in
          let b = where (every 5) a b in
          within (zeros_like x) (betaincc a b x) (betainc b a (rsub_s 1. x)));
      prop "betainc and betaincc sum to 1"
        (Gen.triple (shapes 1e-3 1e4) (shapes 1e-3 1e4) (batch 0. 1.))
        (fun (a, b, x) ->
          let p = betainc a b x and q = betaincc a b x in
          let tol v =
            let b = mul (mul_s (add_s (abs (log v)) 1.) (66. *. eps)) v in
            where (equal_s v 0.) v b
          in
          within (add (tol p) (tol q)) (ones_like p) (add p q));
      prop "betainc increases in x"
        (Gen.triple (shapes 1e-3 1e4) (shapes 1e-3 1e4) (batch 0. 0.999))
        (fun (a, b, x) ->
          Array.iter2
            (fun lo hi -> Windtrap.at_most float_exact ~than:hi lo)
            (values (betainc a b x))
            (values (betainc a b (add_s x 1e-3))));
      prop "betainc a 1 x is x^a and betainc 1 b x is 1 - (1 - x)^b"
        (Gen.pair (shapes 1e-2 1e2) (batch 1e-3 0.999))
        (fun (a, x) ->
          let one = ones_like a in
          let tol v =
            let b = mul (mul_s (add_s (abs (log v)) 1.) (70. *. eps)) v in
            where (equal_s v 0.) v b
          in
          let p = pow x a in
          within (tol p) p (betainc a one x);
          let q = neg (expm1 (mul a (log1p (neg x)))) in
          within (tol q) q (betainc one a x));
      prop "i0e is even and i1e odd, bit for bit" (batch (-1e3) 1e3) (fun x ->
          within (zeros_like x) (i0e x) (i0e (neg x));
          within (zeros_like x) (neg (i1e x)) (i1e (neg x)));
      prop "i0e decreases and 0 < i1e < i0e on (0, inf)" (batch 1e-3 1e3)
        (fun x ->
          let y = add_s x 1e-3 in
          Array.iter2
            (fun a b -> Windtrap.at_most float_exact ~than:a b)
            (values (i0e x))
            (values (i0e y));
          Array.iter2
            (fun i1 i0 ->
              Windtrap.greater float_exact ~than:0. i1;
              Windtrap.less float_exact ~than:i0 i1)
            (values (i1e x))
            (values (i0e x)));
    ]

(* The incomplete gamma's laws, at float64, each within the sum of its terms'
   bounds. *)

let log_batch lo hi =
  Gen.with_pp Nx.pp
    (Gen.map
       (fun xs -> Nx.exp (Nx.create Nx.float64 [| Array.length xs |] xs))
       (Gen.array ~size:(Gen.constant 32)
          (Gen.float_range (Stdlib.log lo) (Stdlib.log hi))))

let pairs = Gen.pair (log_batch 1e-3 1e4) (log_batch 1e-3 1e4)

(* The linear bound in ulps of [f], as a tolerance: [u] for an ulp. *)
let linear f =
  if f = 0. then 0. else log_ulps 16 f *. (eps /. 2.) *. Float.abs f

let gamma_laws =
  let open Nx in
  let elementwise f a b c =
    let a = values a and b = values b and c = values c in
    Array.iteri (fun i a -> f i a b.(i) c.(i)) a
  in
  group "incomplete gamma laws"
    [
      prop "gammainc + gammaincc is 1" pairs (fun (a, x) ->
          elementwise
            (fun i p q _ ->
              at_most
                ~msg:(Printf.sprintf "the error at element %d" i)
                float_exact
                ~than:(linear p +. linear q +. eps)
                (Float.abs (p +. q -. 1.)))
            (gammainc a x) (gammaincc a x) x);
      prop "log_gammainc is log1p (-gammaincc) where gammaincc < 1/2" pairs
        (fun (a, x) ->
          elementwise
            (fun i lp q _ ->
              if q < 0.5 then
                at_most
                  ~msg:(Printf.sprintf "the error at element %d" i)
                  float_exact
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
                at_most
                  ~msg:(Printf.sprintf "the error at element %d" i)
                  float_exact
                  ~than:
                    ((16. *. eps *. Float.max 1. (Float.abs lq))
                    +. (2. *. linear p))
                  (Float.abs (lq -. Stdlib.log1p (-.p))))
            (log_gammaincc a x) (gammainc a x) x);
      prop "gammainc is exp log_gammainc" pairs (fun (a, x) ->
          Windtrap.equal
            (Windtrap.array float_exact)
            (values (exp (log_gammainc a x)))
            (values (gammainc a x)));
      prop "gammaincinv inverts gammainc" pairs (fun (a, x) ->
          let p = gammainc a x in
          let back = gammaincinv a p in
          let kappa = values ((quantile_kappa { b = gammaincinv }).b a p) in
          elementwise
            (fun i x p back ->
              if p >= 0x1p-1022 && p < 1. then
                at_most
                  ~msg:(Printf.sprintf "the error at element %d" i)
                  float_exact
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
                at_most
                  ~msg:(Printf.sprintf "the error at element %d" i)
                  float_exact
                  ~than:((4. +. (2. *. kappa.(i) *. log_ulps 16 q)) *. eps *. x)
                  (Float.abs (back -. x)))
            x q back);
      prop "gammainc increases in x and decreases in a"
        (Gen.pair pairs (log_batch 1e-6 1.))
        (fun ((a, x), d) ->
          let up = mul x (add_s d 1.) in
          let more = mul a (add_s d 1.) in
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
          let q = gammaincc (ones_like x) x in
          let e = Nx.exp (neg x) in
          elementwise
            (fun i q e _ ->
              at_most
                ~msg:(Printf.sprintf "the error at element %d" i)
                float_exact
                ~than:(linear e +. (2. *. eps *. e))
                (Float.abs (q -. e)))
            q e x);
      prop "gammainc 1/2 x is erf (sqrt x)" (log_batch 1e-6 30.) (fun x ->
          let p = gammainc (full_like x 0.5) x in
          let e = erf (sqrt x) in
          elementwise
            (fun i p e _ ->
              at_most
                ~msg:(Printf.sprintf "the error at element %d" i)
                float_exact
                ~than:(linear e +. (4. *. eps *. e))
                (Float.abs (p -. e)))
            p e x);
      prop "log_gammainc a x tends to a log x - lgamma (a + 1) as x goes to 0"
        (Gen.pair (log_batch 1e-3 1e3) (log_batch 1e-300 1e-200))
        (fun (a, x) ->
          let r = sub (mul a (log x)) (lgamma (add_s a 1.)) in
          elementwise
            (fun i l r _ ->
              at_most
                ~msg:(Printf.sprintf "the error at element %d" i)
                float_exact
                ~than:(32. *. eps *. Float.abs r)
                (Float.abs (l -. r)))
            (log_gammainc a x) r x);
      prop "gammainc a x tends to 1 as a goes to 0" (log_batch 1e-30 1e3)
        (fun x ->
          Windtrap.equal
            (Windtrap.array float_exact)
            (Array.make 32 1.)
            (values (gammainc (full_like x 1e-300) x)));
      cases
        ~name:(fun (a, x, _, _) -> Printf.sprintf "at a = %g, x = %g" a x)
        "edges"
        [
          (1., 0., 0., 1.);
          (0.5, 0., 0., 1.);
          (1., Float.infinity, 1., 0.);
          (0., Float.infinity, 1., 0.);
          (Float.infinity, 0., 0., 1.);
          (Float.infinity, Float.infinity, Float.nan, Float.nan);
          (0., 0., Float.nan, Float.nan);
          (-1., 1., Float.nan, Float.nan);
          (1., -1., Float.nan, Float.nan);
          (Float.nan, 1., Float.nan, Float.nan);
          (1., Float.nan, Float.nan, Float.nan);
          (Float.nan, 0., Float.nan, Float.nan);
        ]
        (fun (a, x, p, q) ->
          let a = scalar float64 a and x = scalar float64 x in
          Windtrap.equal float_exact p (item [] (gammainc a x));
          Windtrap.equal float_exact q (item [] (gammaincc a x));
          Windtrap.equal float_exact (Stdlib.log p) (item [] (log_gammainc a x));
          Windtrap.equal float_exact (Stdlib.log q)
            (item [] (log_gammaincc a x)));
      (* As [a] goes to [+inf], [Q] tends to 1 from below for finite [x > 0]:
         [log Q] reaches 0 from below. *)
      cases
        ~name:(fun x -> Printf.sprintf "at a = +inf, x = %g" x)
        "the limit at a = +inf" [ 0x1p-1074; 1.; 1e300 ]
        (fun x ->
          let a = scalar float64 Float.infinity and x = scalar float64 x in
          Windtrap.equal float_exact 0. (item [] (gammainc a x));
          Windtrap.equal float_exact 1. (item [] (gammaincc a x));
          Windtrap.equal float_exact Float.neg_infinity
            (item [] (log_gammainc a x));
          Windtrap.equal float_exact (-0.) (item [] (log_gammaincc a x)));
      (* At [a = 0] each is its limit, [P = 1 - a E1(x) + O(a^2)] for [x > 0]:
         [log P] reaches 0 from below. *)
      cases
        ~name:(fun x -> Printf.sprintf "at a = ±0, x = %g" x)
        "the limit at a = 0"
        [ 0x1p-1074; 1e-300; 0.5; 1.1; 3.; 30.; 1e300 ]
        (fun x ->
          List.iter
            (fun a ->
              let a = scalar float64 a and x = scalar float64 x in
              Windtrap.equal float_exact 1. (item [] (gammainc a x));
              Windtrap.equal float_exact 0. (item [] (gammaincc a x));
              Windtrap.equal float_exact (-0.) (item [] (log_gammainc a x));
              Windtrap.equal float_exact Float.neg_infinity
                (item [] (log_gammaincc a x)))
            [ 0.; -0. ]);
      test "a Poisson tail gammaincc (k + 1) lam is 0 at k = -1" (fun () ->
          let k = create float64 [| 2 |] [| -1.; 0. |] in
          let lam = full_like k 2.5 in
          let q = values (gammaincc (add_s k 1.) lam) in
          Windtrap.equal float_exact 0. q.(0);
          Windtrap.equal (Windtrap.float 1e-15) (Stdlib.exp (-2.5)) q.(1));
      cases
        ~name:(fun (a, p, _) -> Printf.sprintf "gammaincinv %g %h" a p)
        "quantile edges"
        [
          (1., 0., 0.);
          (1., 1., Float.infinity);
          (Float.infinity, 0.5, Float.infinity);
          (Float.infinity, 0., 0.);
          (0., 0.5, 0.);
          (0., 0., 0.);
          (0., 1e-300, 0.);
          (0., 1. -. 0x1p-53, 0.);
          (0., 1., Float.infinity);
          (-1., 0.5, Float.nan);
          (1., 1.5, Float.nan);
          (1., -0.5, Float.nan);
          (Float.nan, 0.5, Float.nan);
          (1., Float.nan, Float.nan);
        ]
        (fun (a, p, x) ->
          let a = scalar float64 a and p = scalar float64 p in
          Windtrap.equal float_exact x (item [] (gammaincinv a p));
          Windtrap.equal float_exact x (item [] (gammainccinv a (rsub_s 1. p))));
    ]

let () =
  exit
    (run "nx special"
       [
         error_function;
         normal;
         gamma;
         bessel;
         incomplete_gamma;
         incomplete_beta;
         laws;
         gamma_laws;
       ])
