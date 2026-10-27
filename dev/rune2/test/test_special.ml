(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The derivatives of nx's special functions, through the operations nx computes
   them with, against goldens of their exact values. *)

open Windtrap
open Special
module Rune = Rune2

type u = { u : 'b. (float, 'b, Nx.host) Nx.t -> (float, 'b, Nx.host) Nx.t }

type b = {
  b :
    'b.
    (float, 'b, Nx.host) Nx.t ->
    (float, 'b, Nx.host) Nx.t ->
    (float, 'b, Nx.host) Nx.t;
}

type c = {
  c :
    'b.
    (float, 'b, Nx.host) Nx.t ->
    (float, 'b, Nx.host) Nx.t ->
    (float, 'b, Nx.host) Nx.t ->
    (float, 'b, Nx.host) Nx.t;
}

let everywhere b _ = b

let by_sign ~positive ~negative args =
  if args.(0) < 0. then negative else positive

(* [log_ulps k f] is [2k (1 + |log f|) + 4], an inverse's forward bound. *)
let log_ulps k f =
  (2. *. Float.of_int k *. (1. +. Float.abs (Stdlib.log (Float.abs f)))) +. 4.

(* Derivatives

   Each derivative is held to golden/special_grad/ (written by
   gen/special_grad.py) eagerly and compiled, at the budgets rune2's derivatives
   of special functions promise: first derivatives within 16 times the
   function's bound, second derivatives within [2^-40] relative at float64 and
   [2^-16] at float32. *)

let grad_golden name = "golden/special_grad/" ^ name ^ ".golden"

(* [d] held eagerly and compiled, at both dtypes. *)
let derivative name ~bound (d : u) =
  let eager = { f = (fun a -> d.u a.(0)) } in
  let compiled = { f = (fun a -> Rune.jit' d.u a.(0)) } in
  group name
    [
      group "eagerly" (check ~bound (grad_golden name) eager);
      group "compiled" (check ~bound (grad_golden name) compiled);
    ]

let derivative2 name ~bound (d : b) =
  let eager = { f = (fun a -> d.b a.(0) a.(1)) } in
  let compiled =
    {
      f =
        (fun a ->
          Rune.jit
            Nx.Ptree.(tensor @-> tensor @-> returns tensor)
            d.b a.(0) a.(1));
    }
  in
  group name
    [
      group "eagerly" (check ~bound (grad_golden name) eager);
      group "compiled" (check ~bound (grad_golden name) compiled);
    ]

let d { u } = { u = (fun x -> Rune.grad' (fun x -> Nx.sum (u x)) x) }

(* [f]'s derivative in its [i]th argument, held to [grad_golden name] eagerly
   and compiled. *)
let derivative3 name ~bound i { c } =
  let d a b x =
    match i with
    | 0 -> Rune.grad' (fun a -> Nx.sum (c a b x)) a
    | 1 -> Rune.grad' (fun b -> Nx.sum (c a b x)) b
    | _ -> Rune.grad' (fun x -> Nx.sum (c a b x)) x
  in
  let eager = { f = (fun v -> d v.(0) v.(1) v.(2)) } in
  let compiled =
    {
      f =
        (fun v ->
          Rune.jit
            Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
            d v.(0) v.(1) v.(2));
    }
  in
  group name
    [
      group "eagerly" (check ~bound (grad_golden name) eager);
      group "compiled" (check ~bound (grad_golden name) compiled);
    ]

let second = Relative (0x1p-40, 0x1p-16)

(* [digamma]'s derivative, [1/x^2] to leading order, overflows to [+inf] as [x]
   nears 0, from either side. *)
let digamma_near_zero =
  let at (type b) (dt : (float, b) Nx.dtype) xs =
    let x = Nx.create dt [| Array.length xs |] xs in
    let g = d { u = Nx.digamma } in
    let infinite = Array.map (fun _ -> Float.infinity) xs in
    equal ~msg:"eagerly" (array float_exact) infinite
      (Nx.to_array (Nx.cast Nx.float64 (g.u x)));
    equal ~msg:"compiled" (array float_exact) infinite
      (Nx.to_array (Nx.cast Nx.float64 (Rune.jit' g.u x)))
  in
  test "digamma's derivative is +inf beside 0" (fun () ->
      at Nx.float64 [| -1e-300; -1e-160; 1e-300 |];
      at Nx.float32 [| -1e-20; -1e-40; 1e-20 |])

(* [lbeta]'s derivative in [first], then in [second]. *)
let lbeta_d2 first second a b =
  let d1 a b =
    match first with
    | `A -> Rune.grad' (fun a -> Nx.sum (Nx.lbeta a b)) a
    | `B -> Rune.grad' (fun b -> Nx.sum (Nx.lbeta a b)) b
  in
  match second with
  | `A -> Rune.grad' (fun a -> Nx.sum (d1 a b)) a
  | `B -> Rune.grad' (fun b -> Nx.sum (d1 a b)) b

(* [log_ndtr x] is [-ndtr (-x)] to leading order for large [x], so its
   derivatives vanish there, also where [x^2] passes the largest float. *)
let log_ndtr_far =
  let at (type b) (dt : (float, b) Nx.dtype) xs =
    let x = Nx.create dt [| Array.length xs |] xs in
    let g = d (d { u = Nx.log_ndtr }) in
    let zeros = Array.map (fun _ -> 0.) xs in
    equal ~msg:"eagerly" (array float_exact) zeros
      (Nx.to_array (Nx.cast Nx.float64 (Nx.abs (g.u x))));
    equal ~msg:"compiled" (array float_exact) zeros
      (Nx.to_array (Nx.cast Nx.float64 (Nx.abs (Rune.jit' g.u x))))
  in
  test "log_ndtr's second derivative is 0 at large x" (fun () ->
      at Nx.float64 [| 1e100; 1e200; Float.max_float |];
      at Nx.float32 [| 1e10; 1e30; 0x1.fffffep127 |])

(* At [-0], [i0e]'s and [i1e]'s derivatives are their right-hand ones, those at
   [+0]. *)
let bessel_at_zero =
  let at (type b) (dt : (float, b) Nx.dtype) =
    let x = Nx.create dt [| 2 |] [| 0.; -0. |] in
    let check name { u } =
      let both y =
        match Nx.to_array (Nx.cast Nx.float64 y) with
        | [| p; n |] -> (p, n)
        | _ -> assert false
      in
      let p, n = both (u x) in
      equal ~msg:(name ^ " eagerly") float_exact p n;
      let p, n = both (Rune.jit' u x) in
      equal ~msg:(name ^ " compiled") float_exact p n
    in
    check "i0e" (d { u = Nx.i0e });
    check "i1e" (d { u = Nx.i1e })
  in
  test "i0e's and i1e's derivatives at -0 are those at +0" (fun () ->
      at Nx.float64;
      at Nx.float32)

(* At [a = 0] the upper tail vanishes as [a g], [g = ∫ₓ¹ (1 - t)^(b-1) / t dt],
   and at [b = 0] the lower tail as [b h], [h = ∫₀ˣ t^(a-1) / (1 - t) dt]: at [b
   = 1], [g = -log x], and at [a = 1], [h = -log (1 - x)]. The derivatives there
   are held to [2^-40] relative at float64 and [2^-16] at float32, eagerly, and
   one on each edge compiled at float32, each compile taking seconds. *)
type run = Eager | Eager_and_compiled

let beta_at_zero =
  let xs = [| 0.1; 0.5; 0.9 |] in
  let at (type b) (dt : (float, b) Nx.dtype) tol jit =
    let x = Nx.create dt [| 3 |] xs and c v = Nx.full dt [| 3 |] v in
    let check ?(run = Eager) name expected u v =
      let read y = Nx.to_array (Nx.cast Nx.float64 y) in
      let compare how got =
        Array.iteri
          (fun i x ->
            let e = expected x in
            at_most
              ~msg:(Printf.sprintf "%s %s at x = %g" name how x)
              float_exact
              ~than:(tol *. Float.abs e)
              (Float.abs (got.(i) -. e)))
          xs
      in
      compare "eagerly" (read (u v));
      match run with
      | Eager -> ()
      | Eager_and_compiled -> compare "compiled" (read (Rune.jit' u v))
    in
    let grad f v = Rune.grad' (fun v -> Nx.sum (f v)) v in
    let g x = -.log x and h x = -.log1p (-.x) in
    check ~run:jit "betaincc in a" g
      (grad (fun a -> Nx.betaincc a (c 1.) x))
      (c 0.);
    check "betainc in a"
      (fun x -> -.g x)
      (grad (fun a -> Nx.betainc a (c 1.) x))
      (c 0.);
    check "log_betainc in a"
      (fun x -> -.g x)
      (grad (fun a -> Nx.log_betainc a (c 1.) x))
      (c 0.);
    check "log_betaincc in x"
      (fun x -> -1. /. (x *. g x))
      (grad (fun x -> Nx.log_betaincc (c 0.) (c 1.) x))
      x;
    check ~run:jit "betainc in b" h
      (grad (fun b -> Nx.betainc (c 1.) b x))
      (c 0.);
    check "betaincc in b"
      (fun x -> -.h x)
      (grad (fun b -> Nx.betaincc (c 1.) b x))
      (c 0.);
    check "log_betaincc in b"
      (fun x -> -.h x)
      (grad (fun b -> Nx.log_betaincc (c 1.) b x))
      (c 0.);
    check "log_betainc in x"
      (fun x -> 1. /. ((1. -. x) *. h x))
      (grad (fun x -> Nx.log_betainc (c 1.) (c 0.) x))
      x
  in
  test "the incomplete beta's derivatives at a = 0 and b = 0 are its slopes"
    (fun () ->
      at Nx.float64 0x1p-40 Eager;
      at Nx.float32 0x1p-16 Eager_and_compiled)

(* The derivative of [b] in its first argument, or its second. *)
let d_first { b } = { b = (fun a x -> Rune.grad' (fun a -> Nx.sum (b a x)) a) }
let d_second { b } = { b = (fun a x -> Rune.grad' (fun x -> Nx.sum (b a x)) x) }

(* In [a], [2^-40 (1 + |log f|)] relative at float64 and [2^-16 (1 + |log f|)]
   at float32, the scale holding [1 + |log f|]; in [x], 16 times the function's
   own bound. *)
let in_a = everywhere (Relative_scaled (0x1p-40, 0x1p-16))
let linear_in_x = everywhere (Inverse (64, 512))

let incomplete_gamma_derivatives =
  group "incomplete gamma derivatives"
    [
      derivative2 "gammainc_a" ~bound:in_a (d_first { b = Nx.gammainc });
      derivative2 "gammainc_x" ~bound:linear_in_x (d_second { b = Nx.gammainc });
      derivative2 "gammaincc_a" ~bound:in_a (d_first { b = Nx.gammaincc });
      derivative2 "gammaincc_x" ~bound:linear_in_x
        (d_second { b = Nx.gammaincc });
      derivative2 "log_gammainc_a" ~bound:in_a (d_first { b = Nx.log_gammainc });
      derivative2 "log_gammainc_x"
        ~bound:(everywhere (Near_zeros (256, 256)))
        (d_second { b = Nx.log_gammainc });
      derivative2 "log_gammaincc_a" ~bound:in_a
        (d_first { b = Nx.log_gammaincc });
      derivative2 "log_gammaincc_x"
        ~bound:(everywhere (Near_zeros (256, 256)))
        (d_second { b = Nx.log_gammaincc });
      derivative2 "gammaincinv_a"
        ~bound:(everywhere (Relative (0x1p-40, 0x1p-16)))
        (d_first { b = Nx.gammaincinv });
      derivative2 "gammaincinv_p"
        ~bound:(fun args ->
          Inverse (64, 16 * int_of_float (log_ulps 16 args.(1))))
        (d_second { b = Nx.gammaincinv });
      derivative2 "gammainccinv_a"
        ~bound:(everywhere (Relative (0x1p-40, 0x1p-16)))
        (d_first { b = Nx.gammainccinv });
      derivative2 "gammainccinv_p"
        ~bound:(fun args ->
          Inverse (64, 16 * int_of_float (log_ulps 16 args.(1))))
        (d_second { b = Nx.gammainccinv });
    ]

let derivatives =
  group "derivatives"
    [
      digamma_near_zero;
      log_ndtr_far;
      bessel_at_zero;
      beta_at_zero;
      derivative "i0e" ~bound:(everywhere (Ulps 128)) (d { u = Nx.i0e });
      group "incomplete beta"
        (List.concat_map
           (fun (name, in_x, c) ->
             [
               derivative3 (name ^ "_x") ~bound:(everywhere in_x) 2 c;
               derivative3 (name ^ "_a") ~bound:in_a 0 c;
               derivative3 (name ^ "_b") ~bound:in_a 1 c;
             ])
           [
             ("betainc", Inverse (64, 1024), { c = Nx.betainc });
             ("betaincc", Inverse (64, 1024), { c = Nx.betaincc });
             ("log_betainc", Near_zeros (512, 512), { c = Nx.log_betainc });
             ("log_betaincc", Near_zeros (512, 512), { c = Nx.log_betaincc });
           ]);
      derivative "i1e"
        ~bound:(fun args ->
          let x = Float.abs args.(0) in
          if 1. <= x && x <= 2. then Near_zeros (128, 128) else Ulps 128)
        (d { u = Nx.i1e });
      derivative "erfc" ~bound:(everywhere (Ulps 128)) (d { u = Nx.erfc });
      derivative "ndtr" ~bound:(everywhere (Ulps 256)) (d { u = Nx.ndtr });
      derivative "log_ndtr" ~bound:(everywhere (Ulps 512))
        (d { u = Nx.log_ndtr });
      derivative "erfinv" ~bound:(everywhere (Ulps 64)) (d { u = Nx.erfinv });
      derivative "ndtri" ~bound:(everywhere (Ulps 64)) (d { u = Nx.ndtri });
      derivative "lgamma"
        ~bound:
          (by_sign ~positive:(Near_zeros (256, 256)) ~negative:(Scaled 256))
        (d { u = Nx.lgamma });
      derivative "digamma"
        ~bound:(by_sign ~positive:(Ulps 256) ~negative:(Scaled 256))
        (d { u = Nx.digamma });
      derivative2 "lbeta_a"
        ~bound:(everywhere (Near_zeros (4096, 8192)))
        { b = (fun a b -> Rune.grad' (fun a -> Nx.sum (Nx.lbeta a b)) a) };
      derivative2 "lbeta_b"
        ~bound:(everywhere (Near_zeros (4096, 8192)))
        { b = (fun a b -> Rune.grad' (fun b -> Nx.sum (Nx.lbeta a b)) b) };
      derivative "lgamma_2" ~bound:(everywhere second) (d (d { u = Nx.lgamma }));
      derivative2 "lbeta_aa" ~bound:(everywhere second)
        { b = (fun a b -> lbeta_d2 `A `A a b) };
      derivative2 "lbeta_bb" ~bound:(everywhere second)
        { b = (fun a b -> lbeta_d2 `B `B a b) };
      derivative2 "lbeta_ab" ~bound:(everywhere second)
        { b = (fun a b -> lbeta_d2 `A `B a b) };
    ]

let () =
  exit (run "rune2 special" [ derivatives; incomplete_gamma_derivatives ])
