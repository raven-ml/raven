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

let () = exit (run "nx special" [ error_function; normal; gamma; bessel; laws ])
