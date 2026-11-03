(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The derivatives of nx's incomplete beta's quantiles, against goldens of
   their exact values. *)

open Windtrap
open Special

(* The beta's quantiles' partials, compiled as one program per dtype: in [p],
   [64 (1 + |log (x ∂ₓI)|) + 16κ (64 (1 + |log m|) + 4)] ulps, [m] the
   smaller tail and [x ∂ₓI = m / κ] at the quantile, whose logarithm the
   derivative exponentiates; in [a] and [b], [2^-40 (1 + |log m|)] relative at
   float64 and [2^-16 (1 + |log m|)] at float32. *)
let beta_inverse_derivatives =
  let in_p args =
    let p = args.(2) in
    let m = Float.min p (1. -. p) in
    Inverse_density
      { k = 64; m; forward = 16 * int_of_float (log_ulps 32 m) }
  in
  let in_shape = everywhere (Relative_scaled (0x1p-40, 0x1p-16)) in
  let inverse name { c } =
    let g v =
      let s = Nx.Ptree.(pair (pair tensor tensor) tensor) in
      let (ga, gb), gp =
        Rune.jit
          Nx.Ptree.(s @-> returns s)
          (Rune.grad s (fun ((a, b), p) -> Nx.sum (c a b p)))
          ((v.(0), v.(1)), v.(2))
      in
      [| ga; gb; gp |]
    in
    let names = [ name ^ "_a"; name ^ "_b"; name ^ "_p" ] in
    let compiled = joint names { g } in
    let eager i =
      let d a b p =
        match i with
        | 0 -> Rune.grad' (fun a -> Nx.sum (c a b p)) a
        | 1 -> Rune.grad' (fun b -> Nx.sum (c a b p)) b
        | _ -> Rune.grad' (fun p -> Nx.sum (c a b p)) p
      in
      { f = (fun v -> d v.(0) v.(1) v.(2)) }
    in
    List.mapi
      (fun i (suffix, bound) ->
        let name = name ^ suffix in
        partial name ~bound ~eager:(eager i) (compiled.golden name))
      [ ("_a", in_shape); ("_b", in_shape); ("_p", in_p) ]
  in
  group "incomplete beta inverse derivatives"
    (inverse "betaincinv" { c = Nx.betaincinv }
    @ inverse "betainccinv" { c = Nx.betainccinv })

(* The beta's quantile's second derivative in [p] is [-I''(x) / I'(x)^3 =
   -((a - 1) / x - (b - 1) / (1 - x)) / f(x)^2], [f] the density, and in [q]
   the same. rune.mli holds it to within about [√ε] of its terms'
   magnitudes, [|a - 1| / x] and [|b - 1| / (1 - x)] over [f(x)^2] and the
   loss it states, [1 / x] and [1 / (1 - x)] over [f(x)^2], which [p]
   entering through [log m] cancels: here within [√ε] of them at float64,
   eagerly. *)
let beta_inverse_second =
  Windtrap.cases "the beta's quantile's second derivative is -I''/I'^3"
    ~id:(fun (a, b, p) -> Printf.sprintf "(%g, %g, %g)" a b p)
    [
      (1., 1., 0.3); (2., 3., 0.3); (0.5, 0.5, 0.2); (5., 1., 0.4);
      (30., 10., 0.7); (2.5, 4., 0.01); (0.3, 20., 0.9);
    ]
    (fun (a, b, p) ->
      List.iter
        (fun inverse ->
          let s v = Nx.scalar Nx.float64 v in
          let d2 =
            Rune.grad'
              (fun p -> Rune.grad' (fun p -> inverse (s a) (s b) p) p)
              (s p)
          in
          let x = Nx.item [] (inverse (s a) (s b) (s p)) in
          let lf =
            ((a -. 1.) *. Stdlib.log x)
            +. ((b -. 1.) *. Stdlib.log1p (-.x))
            -. Nx.item [] (Nx.lbeta (s a) (s b))
          in
          let f2 = Float.exp (2. *. lf) in
          let t1 = (a -. 1.) /. x and t2 = (b -. 1.) /. (1. -. x) in
          let exact = -.(t1 -. t2) /. f2 in
          let scale =
            (Float.abs t1 +. Float.abs t2 +. (1. /. x) +. (1. /. (1. -. x))) /. f2
          in
          let tolerance = Float.sqrt epsilon_float in
          at_most float_exact
            ~than:(tolerance *. Float.max scale (Float.abs exact))
            (Float.abs (Nx.item [] d2 -. exact)))
        [
          Nx.betaincinv;
          (fun a b p -> Nx.betainccinv a b (Nx.rsub_s 1. p));
        ])

let () =
  exit
    (run "rune special beta inverse"
       [ beta_inverse_derivatives; beta_inverse_second ])
