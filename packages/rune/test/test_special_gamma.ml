(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The derivatives of nx's incomplete gamma and its quantiles, against goldens
   of their exact values. *)

open Windtrap
open Special

(* The derivative of [b] in its first argument, or its second. *)
let d_first { b } = { b = (fun a x -> Rune.grad' (fun a -> Nx.sum (b a x)) a) }
let d_second { b } = { b = (fun a x -> Rune.grad' (fun x -> Nx.sum (b a x)) x) }

(* [b]'s partials [first] and [second], each with its bound, as [compiled]
   reads them. *)
let partials2 (first, bound1) (second, bound2) { b } compiled =
  let eager d = { f = (fun a -> d.b a.(0) a.(1)) } in
  [
    partial first ~bound:bound1
      ~eager:(eager (d_first { b }))
      (compiled.golden first);
    partial second ~bound:bound2
      ~eager:(eager (d_second { b }))
      (compiled.golden second);
  ]

(* In [x], 16 times the function's own bound. *)
let linear_in_x = everywhere (Inverse (64, 512))

(* The gradients of the incomplete gamma [value] and of its logarithm [log],
   compiled as one program, as [tail_grads] compiles the beta's. The partials
   are [value]'s, named [value_name ^ "_a"] and [value_name ^ "_x"], then
   [log]'s, named likewise. *)
let gamma_tail_grads (value_name, { b = value }) (log_name, { b = log }) =
  let g v =
    let s = Nx.Ptree.(pair tensor tensor) in
    let grad b = Rune.grad s (fun (a, x) -> Nx.sum (b a x)) in
    let (ga, gx), (la, lx) =
      Rune.jit
        Nx.Ptree.(s @-> returns (pair s s))
        (fun p -> (grad value p, grad log p))
        (v.(0), v.(1))
    in
    [| ga; gx; la; lx |]
  in
  let names name = [ name ^ "_a"; name ^ "_x" ] in
  joint (names value_name @ names log_name) { g }

let lower_gamma_grads =
  gamma_tail_grads
    ("gammainc", { b = Nx.gammainc })
    ("log_gammainc", { b = Nx.log_gammainc })

let upper_gamma_grads =
  gamma_tail_grads
    ("gammaincc", { b = Nx.gammaincc })
    ("log_gammaincc", { b = Nx.log_gammaincc })

(* The gamma quantile [inverse]'s partials in [a] and [p], named [name ^ "_a"]
   and [name ^ "_p"], and its second derivative in [p], named [second],
   compiled as one program, which computes the quantile once for the three. *)
let quantile_grads name second { b = inverse } =
  let g v =
    let s = Nx.Ptree.(pair tensor tensor) in
    let d2 (a, p) =
      Rune.grad'
        (fun p -> Nx.sum (Rune.grad' (fun p -> Nx.sum (inverse a p)) p))
        p
    in
    let (ga, gp), gpp =
      Rune.jit
        Nx.Ptree.(s @-> returns (pair s tensor))
        (fun v -> (Rune.grad s (fun (a, p) -> Nx.sum (inverse a p)) v, d2 v))
        (v.(0), v.(1))
    in
    [| ga; gp; gpp |]
  in
  joint [ name ^ "_a"; name ^ "_p"; second ] { g }

let lower_quantile_grads =
  quantile_grads "gammaincinv" "gammaincinv_pp" { b = Nx.gammaincinv }

let upper_quantile_grads =
  quantile_grads "gammainccinv" "gammainccinv_qq" { b = Nx.gammainccinv }

let incomplete_gamma_derivatives =
  (* Read in the smaller tail [m = min (p, 1 - p)], as the scale's [κ] is. *)
  let gammaincinv_p args =
    let p = args.(1) in
    Inverse (64, 16 * int_of_float (log_ulps 16 (Float.min p (1. -. p))))
  in
  let inverse_a = everywhere (Relative (0x1p-40, 0x1p-16)) in
  let near_zeros = everywhere (Near_zeros (256, 256)) in
  group "incomplete gamma derivatives"
    (List.concat
       [
         partials2 ("gammainc_a", in_a)
           ("gammainc_x", linear_in_x)
           { b = Nx.gammainc } lower_gamma_grads;
         partials2 ("gammaincc_a", in_a)
           ("gammaincc_x", linear_in_x)
           { b = Nx.gammaincc } upper_gamma_grads;
         partials2 ("log_gammainc_a", in_a)
           ("log_gammainc_x", near_zeros)
           { b = Nx.log_gammainc } lower_gamma_grads;
         partials2 ("log_gammaincc_a", in_a)
           ("log_gammaincc_x", near_zeros)
           { b = Nx.log_gammaincc } upper_gamma_grads;
         partials2
           ("gammaincinv_a", inverse_a)
           ("gammaincinv_p", gammaincinv_p)
           { b = Nx.gammaincinv } lower_quantile_grads;
         partials2
           ("gammainccinv_a", inverse_a)
           ("gammainccinv_p", gammaincinv_p)
           { b = Nx.gammainccinv } upper_quantile_grads;
       ])

(* The gamma's quantile's second derivative in [p] is [-f'(x) / f(x)^3 =
   -((a - 1) / x - 1) / f(x)^2], [f] the density, and in [q] the same.
   rune.mli holds it to within about [√ε] of its terms' magnitudes, [|a -
   1| / x] and [1] over [f(x)^2] and the loss it states, [1 / x] over
   [f(x)^2], which [p] entering through [log m] cancels: here within [√ε] of
   them at both dtypes, eagerly and compiled, at shapes up to [2^20] and
   tails down to [1e-300], compiled by the run of the quantile's partials. *)
let gamma_inverse_second =
  let bound =
    everywhere (Relative_scaled (Float.sqrt 0x1p-52, Float.sqrt 0x1p-23))
  in
  let second name { b = inverse } compiled =
    let eager a p =
      Rune.grad'
        (fun p -> Nx.sum (Rune.grad' (fun p -> Nx.sum (inverse a p)) p))
        p
    in
    partial name ~bound
      ~eager:{ f = (fun v -> eager v.(0) v.(1)) }
      (compiled.golden name)
  in
  group "the gamma's quantile's second derivative is -f'/f^3"
    [
      second "gammaincinv_pp" { b = Nx.gammaincinv } lower_quantile_grads;
      second "gammainccinv_qq" { b = Nx.gammainccinv } upper_quantile_grads;
    ]

let () =
  exit
    (run "rune special gamma"
       [ incomplete_gamma_derivatives; gamma_inverse_second ])
