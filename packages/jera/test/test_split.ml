(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Split: palindromic compositions of exact kicks and drifts. The trusted
   side is the pendulum integrated in OCaml floats by a fine Runge–Kutta march,
   and the laws a palindromic consistent scheme keeps. *)

open Windtrap
open Jera

let f64 = Nx.float64
let scalar x = Nx.scalar f64 x
let vec a = Nx.create f64 [| Array.length a |] a
let state = Nx.Ptree.(pair tensor tensor)
let close ?(rel = 1e-12) () = Oracle.structure ~rel ~abs:1e-14 state

(* The pendulum [H = p²/2 − cos q], whose kick and drift are exact. *)
let kick h (q, p) = (q, Nx.sub p (Nx.mul h (Nx.sin q)))
let drift h (q, p) = (Nx.add q (Nx.mul h p), p)

(* q(t) of the pendulum from (q0, 0), by 2¹⁴ classical Runge–Kutta steps in
   OCaml floats: an error near 1e-16 at t = 2. *)
let reference q0 t =
  let n = 1 lsl 14 in
  let h = t /. Float.of_int n in
  let f (q, p) = (p, -.Float.sin q) in
  let ( +^ ) (a, b) (c, d) = (a +. c, b +. d)
  and ( *^ ) s (a, b) = (s *. a, s *. b) in
  let rec go i y =
    if i = n then fst y
    else
      let k1 = f y in
      let k2 = f (y +^ (h /. 2. *^ k1)) in
      let k3 = f (y +^ (h /. 2. *^ k2)) in
      let k4 = f (y +^ (h *^ k3)) in
      go (i + 1) (y +^ (h /. 6. *^ (k1 +^ (2. *^ k2) +^ (2. *^ k3) +^ k4)))
  in
  go 0 (q0, 0.)

let schemes =
  [
    ("leapfrog", Split.leapfrog, 2, 64);
    ("mclachlan", Split.mclachlan, 2, 64);
    ("yoshida4", Split.yoshida4, 4, 16);
    ("yoshida6", Split.yoshida6, 6, 8);
    ("yoshida8", Split.yoshida8, 8, 4);
  ]

let name (n, _, _, _) = n

(* The error at t = 2 of [n] steps from q = 1. *)
let error m n =
  let at = vec [| 0.; 2. |] in
  let q, _ =
    Split.march state m ~steps:n ~kick ~drift ~at (scalar 1., scalar 0.)
  in
  Float.abs (Nx.item [ 1 ] q -. reference 1. 2.)

let order_tests =
  [
    cases ~name "each scheme reaches its order on the pendulum" schemes
      (fun (_, m, p, n) ->
        let order = Oracle.slope (error m n) (error m (2 * n)) in
        at_least (float 1e-9) ~than:(Float.of_int p -. 0.3) order;
        at_most (float 1e-9) ~than:(Float.of_int p +. 0.6) order);
    cases ~name "each scheme steps back to where it started" schemes
      (fun (_, m, _, _) ->
        let s = (vec [| 1.; -0.3; 2.5 |], vec [| 0.; 0.7; -1.1 |]) in
        let h = scalar 0.37 in
        let back =
          Split.step m ~kick ~drift (Nx.neg h) (Split.step m ~kick ~drift h s)
        in
        equal (close ()) s back);
  ]

(* Chains of the pendulum, positions and momenta of shape [[c; 2]], and one
   duration per chain, of shape [[c; 1]]. *)
let chains =
  Gen.(
    let* c = int_range 1 5 in
    let floats lo hi = array ~size:(constant c) (float_range lo hi) in
    let+ q = floats (-2.) 2.
    and+ p = floats (-1.) 1.
    and+ h = floats (-0.5) 0.5 in
    (c, q, p, h))
  |> Gen.with_pp (fun ppf (c, _, _, h) ->
      Format.fprintf ppf "%d chains, h [%s]" c
        (String.concat "; " (Array.to_list (Array.map string_of_float h))))

let chain_tests =
  [
    prop "a step of one duration per chain steps each chain alone" chains
      (fun (c, q, p, h) ->
        let leaves a =
          Nx.create f64 [| c; 2 |]
            (Array.concat
               (Array.to_list (Array.map (fun x -> [| x; 0.5 *. x |]) a)))
        in
        let s = (leaves q, leaves p) in
        let hs = Nx.create f64 [| c; 1 |] h in
        let q', p' = Split.step Split.yoshida4 ~kick ~drift hs s in
        let alone i =
          let row t = Nx.slice [ Nx.R (i, i + 1) ] t in
          Split.step Split.yoshida4 ~kick ~drift
            (scalar h.(i))
            (row (fst s), row (snd s))
        in
        let rows = List.init c alone in
        let stacked =
          ( Nx.concatenate ~axis:0 (List.map fst rows),
            Nx.concatenate ~axis:0 (List.map snd rows) )
        in
        equal (Oracle.structure state) stacked (q', p'));
  ]

let march_tests =
  [
    test "a march stacks the state at each time, the start first" (fun () ->
        let at = vec [| 0.; 0.5; 1.5 |] in
        let q, p =
          Split.march state Split.leapfrog ~steps:3 ~kick ~drift ~at
            (vec [| 1.; 2. |], vec [| 0.; 0. |])
        in
        equal
          (list (array int))
          [ [| 3; 2 |]; [| 3; 2 |] ]
          [ Nx.shape q; Nx.shape p ];
        equal (Oracle.tensor ()) (vec [| 1.; 2. |]) (Nx.get [ 0 ] q));
    test "an interval of n steps is n steps with merged kicks" (fun () ->
        let s0 = (vec [| 1.; -0.4 |], vec [| 0.2; 0.9 |]) in
        let at = vec [| 0.; 0.75 |] in
        let h = scalar (0.75 /. 5.) in
        let stepped =
          List.fold_left
            (fun s _ -> Split.step Split.yoshida4 ~kick ~drift h s)
            s0 [ 1; 2; 3; 4; 5 ]
        in
        let q, p =
          Split.march state Split.yoshida4 ~steps:5 ~kick ~drift ~at s0
        in
        equal (close ()) stepped (Nx.get [ 1 ] q, Nx.get [ 1 ] p));
    test "one time is the start alone" (fun () ->
        let q, _ =
          Split.march state Split.leapfrog ~steps:2 ~kick ~drift
            ~at:(vec [| 3. |])
            (scalar 1., scalar 0.)
        in
        equal (Oracle.tensor ()) (vec [| 1. |]) q);
    test "decreasing times march backward" (fun () ->
        let s0 = (scalar 1., scalar 0.) in
        let q, p =
          Split.march state Split.leapfrog ~steps:4 ~kick ~drift
            ~at:(vec [| 0.; 1. |])
            s0
        in
        let q', _ =
          Split.march state Split.leapfrog ~steps:4 ~kick ~drift
            ~at:(vec [| 1.; 0. |])
            (Nx.get [ 1 ] q, Nx.get [ 1 ] p)
        in
        equal (Oracle.tensor ~rel:1e-13 ()) (scalar 1.) (Nx.get [ 1 ] q'));
  ]

(* The position at t = 1 as a function of the initial position and the end time,
   both through the march. *)
let final q0 =
  let q, _ =
    Split.march state Split.yoshida4 ~steps:8 ~kick ~drift
      ~at:(vec [| 0.; 0.5; 1. |])
      (q0, Nx.zeros_like q0)
  in
  Nx.sum (Nx.get [ 2 ] q)

let final_at t1 =
  let at = Nx.concatenate ~axis:0 [ vec [| 0. |]; Nx.reshape [| 1 |] t1 ] in
  let q, _ =
    Split.march state Split.leapfrog ~steps:16 ~kick ~drift ~at
      (scalar 1., scalar 0.)
  in
  Nx.sum (Nx.get [ 1 ] q)

let transformation_tests =
  [
    test "grad in the initial state is the finite difference" (fun () ->
        let q0 = vec [| 0.3; 1.; -0.8 |] in
        let v = vec [| 1.; -0.5; 0.25 |] in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 final q0 v))
          (scalar (Oracle.dot (Rune.grad' final q0) v)));
    test "grad in the end time is the finite difference" (fun () ->
        let t1 = scalar 1.3 in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (Oracle.central ~eps:1e-6 final_at t1 (scalar 1.))
          (Rune.grad' final_at t1));
    test "compiled equals eager" (fun () ->
        let q0 = vec [| 0.3; 1.; -0.8 |] in
        equal (Oracle.tensor ~rel:1e-14 ()) (final q0) (Rune.jit' final q0));
    test "vmap is each lane's march" (fun () ->
        let q0 = vec [| 0.3; 1.; -0.8 |] in
        let one q = final (Nx.reshape [| 1 |] q) in
        equal
          (Oracle.tensor ~rel:1e-15 ())
          (Nx.stack (List.init 3 (fun i -> one (Nx.get [ i ] q0))))
          (Rune.vmap' one q0));
  ]

let error_tests =
  let raises_with sub f = raises_match (Exn.invalid_arg ~substring:sub) f in
  let march ?(steps = 2) at () =
    Split.march state Split.leapfrog ~steps ~kick ~drift ~at
      (scalar 1., scalar 0.)
  in
  [
    test "v rejects a sequence that is not palindromic" (fun () ->
        raises_with "not palindromic" (fun () ->
            Split.v ~kick:[| 0.25; 0.75 |] ~drift:[| 1. |]));
    test "v rejects kicks that do not sum to one" (fun () ->
        raises_with "kick does not sum to 1" (fun () ->
            Split.v ~kick:[| 0.5; 0.6 |] ~drift:[| 1. |]));
    test "v rejects a kick count that is not one more" (fun () ->
        raises_with "kick needs one more" (fun () ->
            Split.v ~kick:[| 0.5; 0.5 |] ~drift:[| 0.5; 0.5 |]));
    test "v rejects a non-finite coefficient" (fun () ->
        raises_with "not finite" (fun () ->
            Split.v ~kick:[| nan; nan |] ~drift:[| 1. |]));
    test "v accepts the leapfrog's coefficients" (fun () ->
        ignore (Split.v ~kick:[| 0.5; 0.5 |] ~drift:[| 1. |]));
    test "march rejects zero steps" (fun () ->
        raises_with "steps = 0" (march ~steps:0 (vec [| 0.; 1. |])));
    test "march rejects no time" (fun () ->
        raises_with "at must hold" (march (vec [||])));
    test "march rejects times that turn back" (fun () ->
        raises_with "not strictly monotone at [2]: 0.5 after 1"
          (march (vec [| 0.; 1.; 0.5 |])));
    test "march rejects a repeated time" (fun () ->
        raises_with "not strictly monotone at [1]" (march (vec [| 0.; 0. |])));
    test "march rejects a NaN time" (fun () ->
        raises_with "not strictly monotone" (march (vec [| 0.; nan |])));
  ]

let () =
  exit
    (run "Jera.Split"
       [
         group "order" order_tests;
         group "march" march_tests;
         group "chains" chain_tests;
         group "transformations" transformation_tests;
         group "errors" error_tests;
       ])
