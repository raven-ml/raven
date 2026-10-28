(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.scan and Rune.iterate: traced at the call, eagerly; a scan is the OCaml
   fold of its step bit for bit, every NaN read as one, zero rows included;
   reads and keyless draws in a step raise naming the loop. *)

open Windtrap
open Compose
module Rune = Rune2

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let f32 = Nx.float32
let v xs = Nx.create f32 [| Array.length xs |] xs
let t = Nx.Ptree.tensor

(* The fold of [step] over the rows of [xs], each output broadcast to [s]. *)
let fold step ~init xs =
  let n = (Nx.shape xs).(0) in
  let rec go i c ys =
    if i = n then (c, List.rev ys)
    else
      let c, y = step c (Nx.get [ i ] xs) in
      go (i + 1) c (y :: ys)
  in
  go 0 init []

type case =
  | Case :
      (float, 's) Nx.dtype * int array * int * operand * operand * expr * expr
      -> case

let cases =
  let open Gen in
  (let* s = shape in
   let* n = of_list [ 0; 1; 2; 5 ] in
   let rank = Array.length s in
   let* g = expr ~rank and+ h = expr ~rank in
   let* c = operand s and+ xs = operand (Array.append [| n |] s) in
   let+ (Nx.Dtype.Any dt) =
     of_list [ Nx.Dtype.Any Nx.float32; Any Nx.float64 ]
   in
   match Nx.Dtype.kind dt with
   | Float -> Case (dt, s, n, c, xs, g, h)
   | _ -> assert false)
  |> with_pp (fun ppf (Case (dt, s, n, c, xs, g, h)) ->
      Format.fprintf ppf "%a %a, %d rows, carry %a, xs %a: carry %a, output %a"
        Nx.pp_dtype dt pp_shape s n pp_view c.view pp_view xs.view pp g pp h)

let law (Case (dt, s, n, co, xso, g, h)) =
  let init = make dt s co and xs = make dt (Array.append [| n |] s) xso in
  cover "no row" (n = 0);
  cover "several rows" (n > 1);
  cover "a zero-size row" (numel s = 0);
  cover "a view operand" (co.view <> Plain || xso.view <> Plain);
  let step c x =
    (Nx.broadcast_to s (eval c x g), Nx.broadcast_to s (eval c x h))
  in
  (* The step is traced once at the call, as it is applied to the first row:
     with no row its errors still raise. *)
  let expected =
    outcome (fun () ->
        if n = 0 then ignore (step init (Nx.zeros dt s));
        let c, ys = fold step ~init xs in
        (bits c, List.map bits ys))
  in
  let scanned =
    outcome (fun () ->
        let c, ys = Rune.scan' ~f:step ~init xs in
        equal (array int) (Array.append [| n |] s) (Nx.shape ys);
        (bits c, List.init n (fun i -> bits (Nx.get [ i ] ys))))
  in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  equal (outcome_t (pair bits_t (list bits_t))) expected scanned

(* Every dtype, through the functions every dtype has. *)
type any_case =
  | Any_case :
      ('v, 's) Nx.dtype * int array * int * operand * operand * expr * expr
      -> any_case

let any_cases =
  let open Gen in
  (let* s = shape in
   let* n = of_list [ 0; 1; 3 ] in
   let rank = Array.length s in
   let* g = expr ~rank and+ h = expr ~rank in
   let* c = operand s and+ xs = operand (Array.append [| n |] s) in
   let+ (Nx.Dtype.Any dt) = of_list Nx.Dtype.all in
   Any_case (dt, s, n, c, xs, generic g, generic h))
  |> with_pp (fun ppf (Any_case (dt, s, n, _, _, g, h)) ->
      Format.fprintf ppf "%a %a, %d rows: carry %a, output %a" Nx.pp_dtype dt
        pp_shape s n pp g pp h)

let law_any (Any_case (dt, s, n, co, xso, g, h)) =
  let init = make_any dt s co
  and xs = make_any dt (Array.append [| n |] s) xso in
  cover "no row" (n = 0);
  cover "a complex dtype"
    (match Nx.Dtype.kind dt with Complex -> true | _ -> false);
  cover "an integer dtype"
    (match Nx.Dtype.kind dt with Signed | Unsigned -> true | _ -> false);
  let step c x =
    (Nx.broadcast_to s (eval_any c x g), Nx.broadcast_to s (eval_any c x h))
  in
  let expected =
    outcome (fun () ->
        if n = 0 then ignore (step init (Nx.zeros dt s));
        let c, ys = fold step ~init xs in
        (codes c, List.map codes ys))
  in
  let scanned =
    outcome (fun () ->
        let c, ys = Rune.scan' ~f:step ~init xs in
        (codes c, List.init n (fun i -> codes (Nx.get [ i ] ys))))
  in
  equal (outcome_t (pair codes_t (list codes_t))) expected scanned

let xs3 () = Nx.create f32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]

let test_runs_once () =
  List.iter
    (fun n ->
      let calls = ref 0 in
      let step c x =
        incr calls;
        (Nx.add c x, Nx.mul c x)
      in
      ignore
        (Rune.scan' ~f:step ~init:(v [| 0.; 0. |]) (Nx.zeros f32 [| n; 2 |]));
      equal ~msg:(Printf.sprintf "%d rows" n) int 1 !calls)
    [ 0; 1; 7 ]

let test_read () =
  invalid ~sub:"Nx.item: the value is traced by Rune.scan" (fun () ->
      Rune.scan t t Nx.Ptree.unit
        ~f:(fun c x ->
          if Nx.item [ 0 ] x > 0. then (Nx.add c x, ()) else (c, ()))
        ~init:(v [| 0.; 0. |])
        (xs3 ()));
  invalid ~sub:"Nx.item: the value is traced by Rune.iterate" (fun () ->
      Rune.iterate' ~max:3
        ~until:(fun x -> Nx.less x (Nx.scalar f32 (Nx.item [ 0 ] x)))
        ~f:Nx.neg (v [| 1. |]))

let test_keyless () =
  invalid ~sub:"no key inside Rune.scan" (fun () ->
      Rune.scan'
        ~f:(fun c x -> (Nx.add c (Nx.Rng.uniform f32 [| 2 |]), x))
        ~init:(v [| 0.; 0. |])
        (xs3 ()));
  (* Even with no row, the step is traced. *)
  invalid ~sub:"no key inside Rune.scan" (fun () ->
      Rune.scan'
        ~f:(fun c x -> (Nx.add c (Nx.Rng.uniform f32 [| 2 |]), x))
        ~init:(v [| 0.; 0. |])
        (Nx.zeros f32 [| 0; 2 |]));
  invalid ~sub:"no key inside Rune.iterate" (fun () ->
      Rune.iterate' ~max:3
        ~until:(fun x -> Nx.less (Nx.Rng.uniform f32 [||]) x)
        ~f:Nx.neg (v [| 1. |]))

(* A step draws from a key in its rows: the draws are the fold's. *)
let test_keyed () =
  let keys = Nx.Rng.split_batch ~n:4 (Nx.Rng.key 3) in
  let step c k =
    let d = Nx.Rng.normal ~key:(Nx.Rng.of_tensor k) f32 [| 2 |] in
    (Nx.add c d, d)
  in
  let init = v [| 0.; 0. |] in
  let c, ys = Rune.scan' ~f:step ~init (Nx.Rng.to_tensor keys) in
  let c', ys' = fold step ~init (Nx.Rng.to_tensor keys) in
  equal bits_t (bits c') (bits c);
  equal bits_t (bits (Nx.stack ys')) (bits ys)

let test_carry_refused () =
  invalid
    ~sub:
      "Rune.scan': the root: shape [1] in the carry the step returned, [2] in \
       the carry it received" (fun () ->
      Rune.scan'
        ~f:(fun c x -> (Nx.sum ~keepdims:true c, x))
        ~init:(v [| 0.; 0. |])
        (xs3 ()));
  invalid
    ~sub:
      "Rune.scan: the root: length 2 in the carry the step returned, length 1 \
       in the carry it received" (fun () ->
      Rune.scan (Nx.Ptree.list t) t Nx.Ptree.unit
        ~f:(fun c x -> (x :: c, ()))
        ~init:[ v [| 0.; 0. |] ]
        (xs3 ()))

let test_xs_refused () =
  invalid ~sub:"Rune.scan: xs has no leaf" (fun () ->
      Rune.scan t Nx.Ptree.unit Nx.Ptree.unit
        ~f:(fun c () -> (c, ()))
        ~init:(v [| 0. |]) ());
  invalid ~sub:"Rune.scan': xs: the root is 0-d" (fun () ->
      Rune.scan' ~f:(fun c x -> (c, x)) ~init:(v [| 0. |]) (Nx.scalar f32 1.));
  invalid ~sub:"Rune.scan: xs: 1: 3 rows, 0: 2" (fun () ->
      Rune.scan t
        Nx.Ptree.(pair tensor tensor)
        Nx.Ptree.unit
        ~f:(fun c _ -> (c, ()))
        ~init:(v [| 0. |])
        (Nx.zeros f32 [| 2 |], Nx.zeros f32 [| 3 |]))

(* A step captures an array, and a scan nested in a step captures the outer
   step's carry: both are the folds'. *)
let test_nested () =
  let w = v [| 0.5; -2. |] in
  let inner c x = (Nx.add (Nx.mul c w) x, c) in
  let outer c x =
    let c', ys =
      Rune.scan'
        ~f:(fun d y -> inner (Nx.add d c) y)
        ~init:c
        (Nx.stack [ x; x ])
    in
    (Nx.sub c' (Nx.sum ~axes:[ 0 ] ys), c')
  in
  let outer_fold c x =
    let c', ys =
      fold (fun d y -> inner (Nx.add d c) y) ~init:c (Nx.stack [ x; x ])
    in
    (Nx.sub c' (Nx.sum ~axes:[ 0 ] (Nx.stack ys)), c')
  in
  let init = v [| 1.; -0. |] in
  let c, ys = Rune.scan' ~f:outer ~init (xs3 ()) in
  let c', ys' = fold outer_fold ~init (xs3 ()) in
  equal bits_t (bits c') (bits c);
  equal bits_t (bits (Nx.stack ys')) (bits ys)

(* A scan under jit, and a jit inside a step, are the eager scan's. *)
let test_jit () =
  let step c x = (Nx.sin (Nx.add c x), Nx.cos c) in
  let init = v [| 0.25; -1. |] in
  let c, ys = Rune.scan' ~f:step ~init (xs3 ()) in
  let c', ys' =
    Rune.jit
      Nx.Ptree.(tensor @-> returns (pair tensor tensor))
      (fun xs -> Rune.scan' ~f:step ~init xs)
      (xs3 ())
  in
  equal bits_t (bits c) (bits c');
  equal bits_t (bits ys) (bits ys');
  let c'', ys'' =
    Rune.scan'
      ~f:(fun c x ->
        Rune.jit
          Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
          step c x)
      ~init (xs3 ())
  in
  equal bits_t (bits c) (bits c'');
  equal bits_t (bits ys) (bits ys'')

(* iterate *)

let halve x = Nx.mul x (Nx.scalar f32 0.5)
let below_one x = Nx.less (Nx.max x) (Nx.scalar f32 1.)

let test_iterate () =
  let rec reference x =
    if Nx.item [] (below_one x) then x else reference (halve x)
  in
  let x = v [| 20.; 3. |] in
  equal bits_t
    (bits (reference x))
    (bits (Rune.iterate' ~max:10 ~until:below_one ~f:halve x));
  (* An init that holds is returned unchanged. *)
  let small = v [| 0.25; -0. |] in
  equal bits_t (bits small)
    (bits (Rune.iterate' ~max:0 ~until:below_one ~f:halve small));
  (* Steps counted in the carry. *)
  let s = Nx.Ptree.(pair tensor tensor) in
  let x, k =
    Rune.iterate s ~max:10
      ~until:(fun (x, _) -> below_one x)
      ~f:(fun (x, k) -> (halve x, Nx.add k (Nx.scalar Nx.int32 1l)))
      (v [| 20.; 3. |], Nx.scalar Nx.int32 0l)
  in
  equal int32 5l (Nx.item [] k);
  equal bits_t (bits (v [| 0.625; 0.09375 |])) (bits x)

(* An iterate whose stop counts trips in its carry, against the OCaml while loop
   of its step, at every dtype; the step is traced at the call, so its errors
   raise when no step is taken too. *)
type iterate_case =
  | Iterate_case :
      ('v, 's) Nx.dtype * int array * int * int * operand * expr
      -> iterate_case

let iterate_cases =
  let open Gen in
  (let* s = shape in
   let* trips = int_range 0 3 in
   let* max = of_list [ trips; trips + 2; trips - 1 ] in
   let* g = expr ~rank:(Array.length s) in
   let* c = operand s in
   let+ (Nx.Dtype.Any dt) = of_list Nx.Dtype.all in
   Iterate_case (dt, s, trips, max, c, generic g))
  |> with_pp (fun ppf (Iterate_case (dt, s, trips, max, _, g)) ->
      Format.fprintf ppf "%a %a, %d trips, max %d: %a" Nx.pp_dtype dt pp_shape s
        trips max pp g)

let law_iterate (Iterate_case (dt, s, trips, max, co, g)) =
  cover "no trip" (trips = 0);
  cover "max reached" (max < trips);
  cover "a negative max" (max < 0);
  let init = make_any dt s co in
  let step (x, k) =
    (Nx.broadcast_to s (eval_any x x g), Nx.add k (Nx.scalar Nx.int32 1l))
  in
  let until (_, k) =
    Nx.greater_equal k (Nx.scalar Nx.int32 (Int32.of_int trips))
  in
  let k0 = Nx.scalar Nx.int32 0l in
  let expected =
    outcome (fun () ->
        if max < 0 then
          invalid_arg (Printf.sprintf "Rune.iterate: max = %d is negative" max);
        ignore (step (init, k0));
        let rec go j c =
          if j = trips then c
          else if j = max then
            invalid_arg
              (Printf.sprintf
                 "Rune.iterate: until is still false after max = %d steps" max)
          else go (j + 1) (step c)
        in
        codes (fst (go 0 (init, k0))))
  in
  let iterated =
    outcome (fun () ->
        codes
          (fst
             (Rune.iterate
                Nx.Ptree.(pair tensor tensor)
                ~max ~until ~f:step (init, k0))))
  in
  equal (outcome_t codes_t) expected iterated

let test_iterate_refused () =
  invalid ~sub:"Rune.iterate': until is still false after max = 2 steps"
    (fun () -> Rune.iterate' ~max:2 ~until:below_one ~f:halve (v [| 20. |]));
  invalid ~sub:"Rune.iterate': max = -1 is negative" (fun () ->
      Rune.iterate' ~max:(-1) ~until:below_one ~f:halve (v [| 20. |]));
  invalid ~sub:"Rune.iterate': until must return one boolean, got shape [2]"
    (fun () ->
      Rune.iterate' ~max:2
        ~until:(fun x -> Nx.less x x)
        ~f:halve
        (v [| 20.; 1. |]));
  invalid
    ~sub:"Rune.iterate': the root: shape [1] in the carry the step returned"
    (fun () ->
      Rune.iterate' ~max:2 ~until:below_one ~f:(Nx.sum ~keepdims:true)
        (v [| 20.; 1. |]))

(* Constructs in steps *)

module Total = Rune.Total

let test_detach () =
  let x = v [| 1.; -0. |] in
  let c, _ =
    Rune.scan' ~f:(fun c x -> (Nx.add (Rune.detach c) x, c)) ~init:x (xs3 ())
  in
  equal bits_t (bits (v [| 10.; 12. |])) (bits c)

(* With no map around, lanes and lane_index raise in a step, whatever it
   computes and with no row. *)
let test_no_map () =
  let a = Rune.axis () in
  let no_lanes = "Rune.lanes: no map named by the axis is around the call" in
  let no_index =
    "Rune.lane_index: no map named by the axis is around the call"
  in
  List.iter
    (fun n ->
      invalid ~sub:no_lanes (fun () ->
          Rune.scan'
            ~f:(fun c x -> (c, Rune.lanes a x))
            ~init:(v [| 0.; 0. |])
            (Nx.zeros f32 [| n; 2 |]));
      invalid ~sub:no_index (fun () ->
          Rune.scan'
            ~f:(fun c x -> (Nx.add c (Nx.cast f32 (Rune.lane_index a c)), x))
            ~init:(v [| 0.; 0. |])
            (Nx.zeros f32 [| n; 2 |])))
    [ 0; 2 ]

let zero () = Nx.scalar f32 (-0.)

(* Each addition a step makes counts once per step taken, a constant's too,
   summed in step order from -0 and added once. *)
let test_in_loops () =
  let t = Total.make () in
  let xs = xs3 () in
  let step c x =
    Total.add t (Nx.sum x);
    Total.add t (Nx.scalar f32 1.);
    (Nx.add c x, ())
  in
  let scanned () =
    Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit ~f:step
      ~init:(v [| 0.; 0. |])
  in
  let folded xs () = ignore (fold step ~init:(v [| 0.; 0. |]) xs) in
  let total f = snd (Total.collect t ~zero:(zero ()) f) in
  equal bits_t
    (bits (total (folded xs)))
    (bits (total (fun () -> scanned () xs)));
  (* No row: the total keeps its zero's bits. *)
  equal bits_t
    (bits (zero ()))
    (bits (total (fun () -> scanned () (Nx.zeros f32 [| 0; 2 |]))));
  (* Two additions of a step to one total chain in the order they are made: the
     sum is not the sum of each one's sum. *)
  let big c _ =
    Total.add t (Nx.scalar f32 12.);
    Total.add t (Nx.scalar f32 1e8);
    (c, ())
  in
  let three = Nx.zeros f32 [| 3; 1 |] in
  equal bits_t
    (bits (Nx.scalar f32 300000064.))
    (bits
       (total (fun () ->
            Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit ~f:big
              ~init:(v [| 0. |]) three)));
  (* The chain starts from the collect's running total: after 1e8, three 12s are
     added in turn, each rounded, as the fold adds them. *)
  let after =
    total (fun () ->
        Total.add t (Nx.scalar f32 1e8);
        Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
          ~f:(fun c _ ->
            Total.add t (Nx.scalar f32 12.);
            (c, ()))
          ~init:(v [| 0. |]) three)
  in
  let twelve = Nx.scalar f32 12. in
  equal bits_t
    (bits
       Nx.(
         add (add (add (add (zero ()) (scalar f32 1e8)) twelve) twelve) twelve))
    (bits after);
  (* Per trip of an iterate, and per test of its stop. *)
  let trips =
    total (fun () ->
        Rune.iterate' ~max:10
          ~until:(fun x ->
            Total.add t (Nx.scalar f32 100.);
            Nx.less (Nx.max x) (Nx.scalar f32 1.))
          ~f:(fun x ->
            Total.add t (Nx.scalar f32 1.);
            Nx.mul x (Nx.scalar f32 0.5))
          (v [| 5. |]))
  in
  equal bits_t (bits (Nx.scalar f32 403.)) (bits trips)

(* The additions of a scan's steps, to two totals, collected around the scan and
   around the OCaml fold: the same totals, where the zero is 0 or -0. *)
let total_cases =
  let open Gen in
  let value = map (fun i -> pool.(i)) (int_range 0 (Array.length pool - 1)) in
  let adds = list ~size:(int_range 0 3) (pair bool value) in
  triple (of_list [ 0; 1; 3 ]) (of_list [ 0.; -0. ]) adds
  |> with_pp (fun ppf (n, z, adds) ->
      Format.fprintf ppf "%d rows, zero %h: %s" n z
        (String.concat " "
           (List.map
              (fun (u, c) -> Printf.sprintf "%s%h" (if u then "u" else "t") c)
              adds)))

let law_totals (n, z, adds) =
  cover "no row" (n = 0);
  cover "two additions to one total" (List.length adds >= 2);
  cover "an addition before the loop" (List.exists fst adds);
  let t = Total.make () and u = Total.make () in
  let step c x =
    List.iter
      (fun (to_u, k) ->
        Total.add (if to_u then u else t) (Nx.mul (Nx.sum x) (Nx.scalar f32 k)))
      adds;
    (Nx.add c x, ())
  in
  let xs =
    Nx.create f32 [| n; 2 |] (Array.init (2 * n) (fun i -> pool.(i mod 15)))
  in
  let init = v [| 0.; 0. |] in
  let totals f =
    let ((), a), b =
      Total.collect t ~zero:(Nx.scalar f32 z) (fun () ->
          Total.collect u ~zero:(Nx.scalar f32 z) f)
    in
    (bits a, bits b)
  in
  (* [u]'s additions are made once before the loop too, so its chain starts from
     a running total that is not its zero. *)
  let before () =
    List.iter
      (fun (to_u, k) -> if to_u then Total.add u (Nx.scalar f32 (k *. 1e7)))
      adds
  in
  let eager =
    totals (fun () ->
        before ();
        ignore (fold step ~init xs))
  in
  let scanned =
    totals (fun () ->
        before ();
        ignore
          (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit ~f:step ~init
             xs))
  in
  equal (pair bits_t bits_t) eager scanned

(* A scope inside a step sums the step's own additions. *)
let test_scopes_inside () =
  let t = Total.make () in
  let step c x =
    let (), s =
      Total.collect t ~zero:(Nx.scalar f32 0.) (fun () ->
          Total.add t (Nx.sum x))
    in
    (Nx.add c s, s)
  in
  let c, ys =
    Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.tensor ~f:step
      ~init:(Nx.scalar f32 0.) (xs3 ())
  in
  equal bits_t (bits (Nx.scalar f32 21.)) (bits c);
  equal bits_t (bits (v [| 3.; 7.; 11. |])) (bits ys)

(* Carries, rows and additions that creations make: a value is read back at
   the form it was traced at, and a step is traced once. A step that returns
   a carry on another set than its init's does not type. *)
let placement =
  Testable.make ~pp:Nx.Placement.pp ~equal:(fun p q -> Nx.Placement.equal p q)

let host x = Nx.place Nx.Host.on x
let three = Nx.scalar f32 3.

let test_constant_carry () =
  let one = host (Nx.scalar f32 1.) in
  let c =
    Rune.iterate t ~max:10
      ~until:(fun c -> Nx.greater_equal c three)
      ~f:(fun c -> Nx.add c one)
      (Nx.scalar f32 0.)
  in
  equal bits_t (bits three) (bits c);
  equal ~msg:"where the step returns it" placement Nx.Host.on (Nx.placement c);
  let c, _ =
    Rune.scan t t t
      ~f:(fun c x -> (Nx.add c x, x))
      ~init:(Nx.scalar f32 0.)
      (host (v [| 1.; 2. |]))
  in
  equal bits_t (bits three) (bits c)

(* One leaf computed by its step, the other a creation of it. *)
let test_crossing_carry () =
  let one = host (Nx.scalar f32 1.) in
  let a, b =
    Rune.iterate (Nx.Ptree.pair t t) ~max:10
      ~until:(fun (a, _) -> Nx.greater_equal a three)
      ~f:(fun (a, _) -> (Nx.add a one, Nx.zeros f32 [||]))
      (Nx.scalar f32 0., host (Nx.scalar f32 5.))
  in
  equal bits_t (bits three) (bits a);
  equal bits_t (bits (Nx.scalar f32 0.)) (bits b)

(* A carry of two leaves, one computed from the other: one trace. *)
let test_chain () =
  let one = host (Nx.scalar f32 1.) and traces = ref 0 in
  let a, b =
    Rune.iterate (Nx.Ptree.pair t t) ~max:10
      ~until:(fun (a, _) -> Nx.greater_equal a three)
      ~f:(fun (a, _) ->
        incr traces;
        (Nx.add a one, a))
      (Nx.scalar f32 0., Nx.scalar f32 0.)
  in
  equal ~msg:"traces" int 1 !traces;
  equal bits_t (bits three) (bits a);
  equal bits_t (bits (Nx.scalar f32 2.)) (bits b)

(* Row 0 of the outputs is the constant [init], the others are placed. *)
let test_constant_row () =
  let c, ys =
    Rune.scan t t t
      ~f:(fun c x -> (Nx.add c x, c))
      ~init:(Nx.scalar f32 0.)
      (host (v [| 1.; 2.; 3. |]))
  in
  equal bits_t (bits (Nx.scalar f32 6.)) (bits c);
  equal bits_t (bits (v [| 0.; 1.; 3. |])) (bits ys)

(* A constant addition, then a placed one, to one total in a step. *)
let test_constant_addition () =
  let u = Total.make () in
  let (), total =
    Total.collect u ~zero:(Nx.scalar f32 0.) (fun () ->
        ignore
          (Rune.scan t t Nx.Ptree.unit
             ~f:(fun c x ->
               Total.add u (Nx.scalar f32 1.);
               Total.add u (Nx.sum x);
               (c, ()))
             ~init:(Nx.scalar f32 0.)
             (host (v [| 1.; 2.; 3. |]))))
  in
  equal bits_t (bits (Nx.scalar f32 9.)) (bits total)

(* Two devices of one set: placements that differ on one set. *)
module Two = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])

let on k x = Nx.place (Nx.Placement.device Two.v (Nx_support.memory k)) x

(* Leaves that swap devices of one set bind as traced: one trace. *)
let test_swap () =
  let traces = ref 0 in
  let (a, b), _ =
    Rune.scan (Nx.Ptree.pair t t) t t
      ~f:(fun (a, b) x ->
        incr traces;
        ((b, a), x))
      ~init:(on 0 (Nx.scalar f32 1.), on 1 (Nx.scalar f32 2.))
      (on 0 (v [| 0. |]))
  in
  equal ~msg:"traces" int 1 !traces;
  equal bits_t (bits (Nx.scalar f32 2.)) (bits a);
  equal bits_t (bits (Nx.scalar f32 1.)) (bits b)

(* An iterate whose step returns a constant and whose [init] is placed: no step
   or one, under jit and inside a scan's step. *)
let test_placed_init () =
  let zero = Nx.scalar f32 0. in
  let loop a =
    Rune.iterate' ~max:5
      ~until:(fun c -> Nx.greater_equal c zero)
      ~f:(fun _ -> Nx.zeros f32 [||])
      a
  in
  equal ~msg:"no step" bits_t
    (bits (Nx.scalar f32 1.))
    (bits (Rune.jit' loop (host (Nx.scalar f32 1.))));
  equal ~msg:"one step" bits_t (bits zero)
    (bits (Rune.jit' loop (host (Nx.scalar f32 (-1.)))));
  let _, ys =
    Rune.scan' ~f:(fun c x -> (c, loop x)) ~init:zero (host (v [| 1.; -1. |]))
  in
  equal ~msg:"in a scan" bits_t (bits (v [| 1.; 0. |])) (bits ys)

(* A jit around an iterate whose constant carry the jit's argument places. *)
let test_constant_carry_in_jit () =
  let y =
    Rune.jit'
      (fun a ->
        Rune.iterate' ~max:10
          ~until:(fun c -> Nx.greater_equal c three)
          ~f:(fun c -> Nx.add c a)
          (Nx.scalar f32 0.))
      (host (Nx.scalar f32 1.))
  in
  equal bits_t (bits three) (bits y)

let () =
  exit
    (run "rune2 loops"
       [
         group "scan"
           [
             prop "scan gives the fold's bits" cases law;
             prop "scan gives the fold's bits at every dtype" any_cases law_any;
             test "the step runs once per call" test_runs_once;
             test "a read raises naming the loop" test_read;
             test "a keyless draw raises naming the loop" test_keyless;
             test "a keyed draw is the fold's" test_keyed;
             test "a carry that changes raises" test_carry_refused;
             test "rows without a leading axis raise" test_xs_refused;
             test "captures and nested scans" test_nested;
             test "under jit and around jit" test_jit;
           ];
         group "constructs in steps"
           [
             test "detach in a step" test_detach;
             test "lanes and lane_index raise with no map" test_no_map;
             test "a loop counts each step's additions" test_in_loops;
             prop "a scan adds what its fold adds" total_cases law_totals;
             test "a scope inside a step" test_scopes_inside;
           ];
         group "iterate"
           [
             test "iterate is the while loop" test_iterate;
             prop "iterate gives the while loop's bits" iterate_cases
               law_iterate;
             test "misuse raises" test_iterate_refused;
             test "a carry lies where its step returns it" test_constant_carry;
             test "a step that returns a creation" test_crossing_carry;
             test "a step is traced once" test_chain;
             test "a constant first row" test_constant_row;
             test "a constant addition before a placed one"
               test_constant_addition;
             test "a constant carry under jit" test_constant_carry_in_jit;
             test "leaves that swap devices of one set" test_swap;
             test "a placed init and a constant step" test_placed_init;
           ];
       ])
