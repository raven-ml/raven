(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Split: palindromic compositions of exact kicks and drifts. The trusted
   side is the pendulum integrated in OCaml floats by a fine Runge–Kutta march,
   the coefficients jera.mli and Yoshida (1990) state, and the laws a
   palindromic composition of exact flows keeps. *)

open Windtrap
open Jera

let f64 = Nx.float64
let scalar x = Nx.scalar f64 x
let vec a = Nx.create f64 [| Array.length a |] a
let state = Nx.Ptree.(pair tensor tensor)

(* jera's Conventions promise [Invalid_argument] naming the function. *)
let raises_in fn f = raises_match (Exn.invalid_arg ~sub:fn) f

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

(* Each scheme with its leapfrogs: jera.mli's drifts per step. *)
let schemes =
  [
    ("leapfrog", Split.leapfrog, 1);
    ("mclachlan", Split.mclachlan, 2);
    ("yoshida4", Split.yoshida4, 3);
    ("yoshida6", Split.yoshida6, 7);
    ("yoshida8", Split.yoshida8, 15);
  ]

let scheme_name (name, _, _) = name

let scheme =
  Gen.of_list
    ~pp:(fun ppf s -> Format.pp_print_string ppf (scheme_name s))
    schemes

(* A demand per scheme, so that every run draws each. *)
let cover_schemes (name, _, _) =
  List.iter (fun s -> cover (scheme_name s) (scheme_name s = name)) schemes

(* Schemes *)

(* The error at t = 2 of [n] steps from q = 1. *)
let error m n =
  let q, _ =
    Split.march state m ~steps:n ~kick ~drift
      ~at:(vec [| 0.; 2. |])
      (scalar 1., scalar 0.)
  in
  Float.abs (Nx.item [ 1 ] q -. reference 1. 2.)

(* Leapfrogs of durations [w h] in sequence. *)
let leapfrogs ws h s =
  Array.fold_left
    (fun s w -> Split.step Split.leapfrog ~kick ~drift (Nx.mul_s h w) s)
    s ws

let scheme_tests =
  [
    cases "each scheme reaches its order on the pendulum"
      ~id:(fun (name, _, _, _) -> name)
      [
        ("leapfrog", Split.leapfrog, 2, 64);
        ("mclachlan", Split.mclachlan, 2, 64);
        ("yoshida4", Split.yoshida4, 4, 16);
        ("yoshida6", Split.yoshida6, 6, 8);
        ("yoshida8", Split.yoshida8, 8, 4);
      ]
      (fun (_, m, p, n) ->
        (* The slope between n and 2n steps, each error far above the
           reference's 1e-16. Its upper bound tells a scheme from the next
           order's. *)
        let order = Oracle.slope (error m n) (error m (2 * n)) in
        at_least (float 1e-9) ~than:(Float.of_int p -. 0.3) order;
        at_most (float 1e-9) ~than:(Float.of_int p +. 0.6) order);
    cases "leapfrog and mclachlan step as v of the coefficients jera.mli states"
      ~id:(fun (name, _, _, _) -> name)
      [
        ("leapfrog", Split.leapfrog, [| 0.5; 0.5 |], [| 1. |]);
        ( "mclachlan",
          Split.mclachlan,
          (let l = 0.1931833275037836 in
           [| l; 1. -. (2. *. l); l |]),
          [| 0.5; 0.5 |] );
      ]
      (fun (_, m, kicks, drifts) ->
        let s = (vec [| 1.; -0.4 |], vec [| 0.2; 0.9 |]) and h = scalar 0.3 in
        equal (Oracle.structure state)
          (Split.step (Split.v ~kick:kicks ~drift:drifts) ~kick ~drift h s)
          (Split.step m ~kick ~drift h s));
    cases "each Yoshida scheme steps as its leapfrogs of Yoshida's weights"
      ~id:(fun (name, _, _) -> name)
      [
        (* Yoshida (1990), (4.6). *)
        ("yoshida4", Split.yoshida4, [| 1. /. (2. -. Float.cbrt 2.) |]);
        (* Table 1, solution A. *)
        ( "yoshida6",
          Split.yoshida6,
          [| -1.17767998417887; 0.235573213359357; 0.784513610477560 |] );
        (* Table 2, solution D. *)
        ( "yoshida8",
          Split.yoshida8,
          [|
            0.102799849391985;
            -1.96061023297549;
            1.93813913762276;
            -0.158240635368243;
            -1.44485223686048;
            0.253693336566229;
            0.914844246229740;
          |] );
      ]
      (fun (_, m, ws) ->
        (* The leapfrogs of w_m … w_1 w_0 w_1 … w_m, w_0 = 1 − 2 Σ w_i (Yoshida,
           1990, (5.1) and (5.18)). The scheme merges adjacent half kicks, one
           rounding each; at h = 0.05 the flows grow an error at most e^(0.05 Σ
           |c|) ≤ 5-fold, so 31 roundings stay within 1e-13. *)
        let w0 = 1. -. (2. *. Array.fold_left ( +. ) 0. ws) in
        let rev = Array.of_list (List.rev (Array.to_list ws)) in
        let weights = Array.concat [ rev; [| w0 |]; ws ] in
        let s = (vec [| 1.; -0.4 |], vec [| 0.2; 0.9 |]) and h = scalar 0.05 in
        equal
          (Oracle.structure ~rel:1e-13 ~abs:1e-15 state)
          (leapfrogs weights h s)
          (Split.step m ~kick ~drift h s));
  ]

(* v *)

let v_tests =
  [
    cases "v rejects coefficients that break one of its conditions" ~id:fst
      [
        (* An infinite sum passes the sum's test and the palindrome's. *)
        ("an infinite coefficient", ([| infinity; infinity |], [| 1. |]));
        ("an empty drift", ([| 1. |], [||]));
        ("a kick count that is not one more", ([| 0.5; 0.5 |], [| 0.5; 0.5 |]));
        ("kicks that do not sum to one", ([| 0.6; 0.6 |], [| 1. |]));
        ( "drifts that do not sum to one",
          ([| 0.25; 0.5; 0.25 |], [| 0.4; 0.4 |]) );
        ("kicks that are not palindromic", ([| 0.25; 0.75 |], [| 1. |]));
        ( "drifts that are not palindromic",
          ([| 1. /. 3.; 1. /. 3.; 1. /. 3. |], [| 0.25; 0.75 |]) );
      ]
      (fun (_, (kicks, drifts)) ->
        raises_in "Jera.Split.v" (fun () -> Split.v ~kick:kicks ~drift:drifts));
    test "v accepts kicks that sum to one only within rounding" (fun () ->
        (* 0.1 + 0.2 + 0.4 + 0.2 + 0.1 is 1.0000000000000002 in floats. A unit
           flow of each part counts its coefficients over a step of 1. *)
        let kicks = [| 0.1; 0.2; 0.4; 0.2; 0.1 |] in
        let m = Split.v ~kick:kicks ~drift:[| 0.25; 0.25; 0.25; 0.25 |] in
        let q, p =
          Split.step m
            ~kick:(fun h (q, p) -> (q, Nx.add p h))
            ~drift:(fun h (q, p) -> (Nx.add q h, p))
            (scalar 1.)
            (scalar 0., scalar 0.)
        in
        equal
          (pair float_exact float_exact)
          (1., Array.fold_left ( +. ) 0. kicks)
          (Nx.item [] q, Nx.item [] p));
  ]

(* step *)

(* A step applies at most 62 flows (yoshida8), each rounding once and each
   growing an error at most by 1 + |c h|: by e^(0.1 × 30.5) ≤ 21 in all for a
   step of at most 0.1, so its rounding stays within 62 × 21 × 2.2e-16 =
   3e-13. *)
let close = Oracle.structure ~rel:1e-12 ~abs:1e-14 state

(* A scheme, a state of one to three pendulums and a step of at most 0.1. *)
let steps =
  Gen.(
    let* n = int_range 1 3 in
    let+ m = scheme
    and+ q = array ~size:(constant n) (float_range (-2.) 2.)
    and+ p = array ~size:(constant n) (float_range (-1.) 1.)
    and+ h = float_range (-0.1) 0.1 in
    (m, q, p, h))
  |> Gen.with_pp (fun ppf (m, q, p, h) ->
      let floats a =
        String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%h") a))
      in
      Format.fprintf ppf "%s, q [%s], p [%s], h %h" (scheme_name m) (floats q)
        (floats p) h)

let steps_back (((_, m, _) as s), q, p, h) =
  cover_schemes s;
  Law.round_trip close close
    (Split.step m ~kick ~drift (scalar h))
    (Split.step m ~kick ~drift (scalar (-.h)))
    (vec q, vec p)

(* A scheme, one pendulum and a step of at most 0.1. *)
let pendulums =
  Gen.(
    quad scheme (float_range (-2.) 2.) (float_range (-1.) 1.)
      (float_range (-0.1) 0.1))
  |> Gen.with_pp (fun ppf (m, q, p, h) ->
      Format.fprintf ppf "%s, q %h, p %h, h %h" (scheme_name m) q p h)

let preserves_area (((_, m, _) as s), q, p, h) =
  (* For one degree of freedom a symplectic map preserves area: its Jacobian's
     determinant is 1. Its entries are at most 21 (the growth above), so the
     determinant rounds within 2 × 21² × 62 × 2.2e-16 = 1.2e-11. *)
  cover_schemes s;
  let out i (q, p) =
    let q', p' = Split.step m ~kick ~drift (scalar h) (q, p) in
    if i = 0 then q' else p'
  in
  let d i j =
    let f x = out i (if j = 0 then (x, scalar p) else (scalar q, x)) in
    Nx.item [] (Rune.grad' f (scalar (if j = 0 then q else p)))
  in
  equal (float 2e-11) 1. ((d 0 0 *. d 1 1) -. (d 0 1 *. d 1 0))

(* Chains of the pendulum, positions and momenta of shape [[c; 2]], and one
   duration per chain, of shape [[c; 1]]. *)
let chains =
  Gen.(
    let* c = int_range 1 5 in
    let floats lo hi = array ~size:(constant c) (float_range lo hi) in
    let+ m = scheme
    and+ q = floats (-2.) 2.
    and+ p = floats (-1.) 1.
    and+ h = floats (-0.5) 0.5 in
    (m, c, q, p, h))
  |> Gen.with_pp (fun ppf (m, c, _, _, h) ->
      Format.fprintf ppf "%s, %d chains, h [%s]" (scheme_name m) c
        (String.concat "; " (Array.to_list (Array.map string_of_float h))))

let each_chain (((_, m, _) as s), c, q, p, h) =
  cover_schemes s;
  cover "one chain" (c = 1);
  let leaves a =
    Nx.create f64 [| c; 2 |]
      (Array.concat (Array.to_list (Array.map (fun x -> [| x; 0.5 *. x |]) a)))
  in
  let s = (leaves q, leaves p) in
  (* The flows receive the durations in h's shape. *)
  let shaped flow h s =
    equal (array int) [| c; 1 |] (Nx.shape h);
    flow h s
  in
  let stepped =
    Split.step m ~kick:(shaped kick) ~drift:(shaped drift)
      (Nx.create f64 [| c; 1 |] h)
      s
  in
  let alone i =
    let row t = Nx.slice [ Nx.R (i, i + 1) ] t in
    Split.step m ~kick ~drift (scalar h.(i)) (row (fst s), row (snd s))
  in
  let rows = List.init c alone in
  let expected =
    ( Nx.concatenate ~axis:0 (List.map fst rows),
      Nx.concatenate ~axis:0 (List.map snd rows) )
  in
  equal (Oracle.structure state) expected stepped

let step_tests =
  [
    prop "a step back undoes it" steps ~count:200 steps_back;
    prop "a step preserves area" pendulums ~count:150 preserves_area;
    prop "a step of one duration per chain steps each chain alone" chains
      ~count:150 each_chain;
    cases "a step makes leapfrogs + 1 kicks" ~id:scheme_name schemes
      (fun (_, m, leapfrogs) ->
        (* A kick counts itself in the second leaf. *)
        let _, n =
          Split.step m
            ~kick:(fun _ (q, n) -> (q, Nx.add_s n 1.))
            ~drift:(fun h (q, n) -> (Nx.add q h, n))
            (scalar 1.)
            (scalar 0., scalar 0.)
        in
        equal float_exact (Float.of_int (leapfrogs + 1)) (Nx.item [] n));
  ]

(* march *)

(* A scheme, steps per interval and two or three times of [0, 0.1], increasing
   or decreasing. *)
let marches =
  Gen.(
    let+ m = scheme
    and+ steps = int_range 1 6
    and+ ts = list ~size:(int_range 2 3) (float_range 0. 0.1)
    and+ up = bool in
    let ts = List.sort_uniq compare ts in
    (m, steps, Array.of_list (if up then ts else List.rev ts)))
  |> Gen.with_pp (fun ppf (m, steps, ts) ->
      Format.fprintf ppf "%s, %d steps, at [%s]" (scheme_name m) steps
        (String.concat "; "
           (Array.to_list (Array.map (Printf.sprintf "%h") ts))))

(* The state at each time of [at], stepped with jera.mli's grid: h = (b − a) /
   steps from a, each start held at b where it rounds past, the last step from
   its start to b. *)
let stepped m ~steps at s =
  let s = ref s in
  let states = ref [ !s ] in
  for i = 1 to Array.length at - 1 do
    let a = at.(i - 1) and b = at.(i) in
    let h = (b -. a) /. Float.of_int steps in
    for _ = 1 to steps - 1 do
      s := Split.step m ~kick ~drift (scalar h) !s
    done;
    (* The product rounds before the sum, as the march's tensors compute it:
       ocamlopt for arm64 fuses [a +. (k *. h)] into one multiply-add. *)
    let start = a +. Sys.opaque_identity (Float.of_int (steps - 1) *. h) in
    let start =
      if (b > a && start > b) || (b < a && start < b) then b else start
    in
    s := Split.step m ~kick ~drift (scalar (b -. start)) !s;
    states := !s :: !states
  done;
  let states = List.rev !states in
  (Nx.stack (List.map fst states), Nx.stack (List.map snd states))

let is_its_steps (((_, m, _) as s), steps, ts) =
  (* A march merges each pair of adjacent kicks into one, a rounding of the sum
     of their durations per step; over at most 0.1 the flows grow an error at
     most 21-fold (see step's), so 6 steps × 62 flows × 21 × 2.2e-16 = 2e-12. *)
  assume (Array.length ts >= 2);
  cover_schemes s;
  cover "one step" (steps = 1);
  cover "two steps" (steps = 2);
  cover "a scanned interval" (steps >= 4);
  cover "decreasing times" (ts.(0) > ts.(1));
  cover "three times" (Array.length ts = 3);
  let s0 = (vec [| 1.; -0.4 |], vec [| 0.2; 0.9 |]) in
  equal
    (Oracle.structure ~rel:1e-11 ~abs:1e-14 state)
    (stepped m ~steps ts s0)
    (Split.march state m ~steps ~kick ~drift ~at:(vec ts) s0)

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

(* The largest |H − H₀| of the pendulum over states [lo] to [hi]. *)
let energy_error (q, p) lo hi =
  let q = Nx.to_array q and p = Nx.to_array p in
  let h i = (p.(i) *. p.(i) /. 2.) -. Float.cos q.(i) in
  let m = ref 0. in
  for i = lo to hi do
    m := Float.max !m (Float.abs (h i -. h 0))
  done;
  !m

let leapfrog_march ~steps at =
  Split.march state Split.leapfrog ~steps ~kick ~drift ~at (scalar 1., scalar 0.)

let march_tests =
  [
    test "a march stacks the state at each time, the start first" (fun () ->
        let q, p =
          Split.march state Split.leapfrog ~steps:3 ~kick ~drift
            ~at:(vec [| 0.; 0.5; 1.5 |])
            (vec [| 1.; 2. |], vec [| 0.; 0. |])
        in
        equal
          (list (array int))
          [ [| 3; 2 |]; [| 3; 2 |] ]
          [ Nx.shape q; Nx.shape p ];
        equal (Oracle.tensor ()) (vec [| 1.; 2. |]) (Nx.get [ 0 ] q));
    test "one time is the start alone" (fun () ->
        let q, _ = leapfrog_march ~steps:2 (vec [| 3. |]) in
        equal (Oracle.tensor ()) (vec [| 1. |]) q);
    prop "a march is its steps, adjacent kicks merged" marches ~count:150
      ~examples:[ (List.nth schemes 2, 5, [| 0.; 0.75 |]) ]
      is_its_steps;
    test "a march over reversed times returns to its start" (fun () ->
        (* Each step undoes one: 8 steps of leapfrog over 1 round within
           1e-13. *)
        let q, p = leapfrog_march ~steps:4 (vec [| 0.; 1. |]) in
        let q', _ =
          Split.march state Split.leapfrog ~steps:4 ~kick ~drift
            ~at:(vec [| 1.; 0. |])
            (Nx.get [ 1 ] q, Nx.get [ 1 ] p)
        in
        equal (Oracle.tensor ~rel:1e-13 ()) (scalar 1.) (Nx.get [ 1 ] q'));
    cases "an interval of n steps makes n × leapfrogs + 1 kicks"
      ~id:(fun (s, n) -> Printf.sprintf "%s, %d steps" (scheme_name s) n)
      (List.concat_map (fun s -> [ (s, 1); (s, 4) ]) schemes)
      (fun ((_, m, leapfrogs), n) ->
        (* A kick counts itself in the second leaf. *)
        let _, counts =
          Split.march state m ~steps:n
            ~kick:(fun _ (q, n) -> (q, Nx.add_s n 1.))
            ~drift:(fun h (q, n) -> (Nx.add q h, n))
            ~at:(vec [| 0.; 1. |])
            (scalar 0., scalar 0.)
        in
        equal (Oracle.tensor ())
          (vec [| 0.; Float.of_int ((n * leapfrogs) + 1) |])
          counts);
    cases "steps have size h, the last ending at its interval's end"
      ~id:(fun (at, steps) ->
        Printf.sprintf "[%s] in %d"
          (String.concat "; "
             (Array.to_list (Array.map (Printf.sprintf "%h") at)))
          steps)
      [
        ([| 0.; 0.3; 0.9 |], 10);
        ([| 0.9; 0.3; 0. |], 7);
        ([| 0.; 1.5e-323 |], 5);
        ([| 1.; 1. +. Float.epsilon |], 4);
      ]
      (fun (at, steps) ->
        (* A unit drift moves q by each step's size: h = (b - a) / steps for all
           but the last, which runs from a + (steps - 1) h, held at b where that
           rounds past it, to b. *)
        let expected = Array.copy at in
        let q = ref at.(0) in
        for i = 1 to Array.length at - 1 do
          let a = at.(i - 1) and b = at.(i) in
          let h = (b -. a) /. Float.of_int steps in
          for _ = 1 to steps - 1 do
            q := !q +. h
          done;
          (* The product rounds before the sum, as the march's tensors compute
             it: ocamlopt for arm64 fuses [a +. (k *. h)] into one multiply-add,
             which rounds once. *)
          let start =
            a +. Sys.opaque_identity (Float.of_int (steps - 1) *. h)
          in
          let start =
            if (b > a && start > b) || (b < a && start < b) then b else start
          in
          q := !q +. (b -. start);
          expected.(i) <- !q
        done;
        let q, _ =
          Split.march state Split.leapfrog ~steps
            ~kick:(fun _ s -> s)
            ~drift:(fun h (q, p) -> (Nx.add q (Nx.mul h p), p))
            ~at:(vec at)
            (scalar at.(0), scalar 1.)
        in
        equal (Oracle.tensor ()) (vec expected) q);
    cases "times whose gaps multiply to below the floats are monotone"
      ~id:(fun a -> Printf.sprintf "[%g; %g; %g]" a.(0) a.(1) a.(2))
      [ [| 0.; 1e-162; 3e-162 |]; [| 3e-162; 1e-162; 0. |] ]
      (fun at ->
        let q, _ = leapfrog_march ~steps:2 (vec at) in
        equal (array int) [| 3 |] (Nx.shape q));
    test "leapfrog's energy error does not grow over long times" (fun () ->
        (* From q = 1 at rest, steps of 0.1 to t = 200, some thirty periods.
           Energy that drifted would show at the end a multiple of its error
           over the first twenty; a bounded error shows the same oscillation
           there. Twice is the bound. *)
        let s = leapfrog_march ~steps:1 (Nx.linspace f64 0. 200. 2001) in
        at_most float_exact
          ~than:(2. *. energy_error s 0 200)
          (energy_error s 1800 2000));
    test "grad in the initial state is the finite difference" (fun () ->
        (* Central differences at 1e-6: truncation near 1e-12, rounding near eps
           / 1e-6 = 2e-10, so 1e-7 holds. *)
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
        (* Compiled code may fuse and reorder the flows' operations, each
           rounding once: some hundred of them within 1e-14. *)
        Compiled.here ();
        let q0 = vec [| 0.3; 1.; -0.8 |] in
        equal (Oracle.tensor ~rel:1e-14 ()) (final q0) (Rune.jit' final q0));
    test "vmap is each lane's march" (fun () ->
        let q0 = vec [| 0.3; 1.; -0.8 |] in
        let one q = final (Nx.reshape [| 1 |] q) in
        equal
          (Oracle.tensor ~rel:1e-15 ())
          (Nx.stack (List.init 3 (fun i -> one (Nx.get [ i ] q0))))
          (Rune.vmap' one q0));
    test "march rejects zero steps" (fun () ->
        raises_in "Jera.Split.march" (fun () ->
            leapfrog_march ~steps:0 (vec [| 0.; 1. |])));
    cases "march rejects at that is not a vector of times"
      ~id:(fun at ->
        Printf.sprintf "shape [%s]"
          (String.concat "; "
             (Array.to_list (Array.map string_of_int (Nx.shape at)))))
      [ vec [||]; Nx.create f64 [| 1; 2 |] [| 0.; 1. |] ]
      (fun at ->
        raises_in "Jera.Split.march" (fun () -> leapfrog_march ~steps:2 at));
    cases "march rejects times that are not strictly monotone"
      ~id:(fun a ->
        String.concat "; " (Array.to_list (Array.map string_of_float a)))
      [ [| 0.; 1.; 0.5 |]; [| 0.; 0. |]; [| 0.; Float.nan |] ]
      (fun at ->
        raises_in "Jera.Split.march" (fun () ->
            leapfrog_march ~steps:2 (vec at)));
  ]

let () =
  exit
    (run "Jera.Split"
       [
         group "schemes" scheme_tests;
         group "v" v_tests;
         group "step" step_tests;
         group "march" march_tests;
       ])
