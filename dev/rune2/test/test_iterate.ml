(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.iterate under forward mode, reverse mode, maps and jit: against the
   OCaml while loop of its step, run one lane at a time, bit for bit but for the
   zeros a loop holds where the steps written out have none (Compose.lenient).
   Each lane stops on its own. *)

open Windtrap
open Compose
module Rune = Rune2
module T = Nx.Ptree
module D = Nx.Dtype
module Total = Rune.Total

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs

(* The loop written out: the step applied while [until] is false, as many times
   as [max] allows, the step traced once even where it takes none, as iterate's
   is. *)
let looped ~max ~until ~f init =
  ignore (f init);
  let rec go j c =
    if Nx.item [] (until c) then c
    else if j = max then
      invalid_arg
        (Printf.sprintf
           "Rune.iterate: until is still false after max = %d steps" max)
    else go (j + 1) (f c)
  in
  go 0 init

(* A carry's sums kept in step order: [e] with each sum along an axis replaced
   by its extreme, so that a map's lanes compute what one lane does whatever the
   order its sums associate in. *)
let rec unsummed = function
  | (X | Y | K _) as e -> e
  | Un ((Sum a | Cumsum a | Mean a), e) -> Un (Max a, unsummed e)
  | Un (u, e) -> Un (u, unsummed e)
  | Bin (b, e, e') -> Bin (b, unsummed e, unsummed e')

(* Bits within a few units in the last place, every value that is not finite
   alike: the lanes' terms of a captured value's gradient are summed in another
   order than one lane at a time sums them, and a held lane adds a zero times
   its donor's partial derivative, NaN where that is infinite. *)
let near (type s) (dt : (float, s) Nx.dtype) =
  let tol = if D.equal dt Nx.float32 then 1e-5 else 1e-12 in
  Testable.make ~pp:(Testable.pp bits_t) ~equal:(fun (s, a) (s', b) ->
      s = s'
      && Array.length a = Array.length b
      && Array.for_all2
           (fun x y ->
             x = y
             ||
             let x = Int64.float_of_bits x and y = Int64.float_of_bits y in
             ((not (Float.is_finite x)) && not (Float.is_finite y))
             || Float.abs (x -. y) <= tol *. Float.max 1. (Float.abs x))
           a b)

(* Derivatives *)

(* A step [c -> g(c, w)] and a counter, the loop taking [trips] steps, [max] at
   most. *)
type case =
  | Case : {
      dt : (float, 's) Nx.dtype;
      s : int array;
      trips : int;
      max : int;
      g : expr;
      c : operand;
      w : operand;
      tc : operand;
      tw : operand;
    }
      -> case

let cases =
  let open Gen in
  (let* s = shape in
   let* trips = int_range 0 3 in
   let* max = of_list [ trips; trips + 2; trips - 1 ] in
   let* g = expr ~rank:(Array.length s) in
   let* c = operand s and+ w = operand s in
   let* tc = operand s and+ tw = operand s in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Case { dt; s; trips; max = Int.max 0 max; g; c; w; tc; tw }
   | _ -> assert false)
  |> with_pp (fun ppf (Case c) ->
      Format.fprintf ppf "%a %a, %d trips, max %d: %a" Nx.pp_dtype c.dt pp_shape
        c.s c.trips c.max pp c.g)

let law_derivatives (Case c) =
  cover "no step" (c.trips = 0);
  cover "several steps" (c.trips > 1);
  cover "max reached" (c.max < c.trips);
  let x = make c.dt c.s c.c and w = make c.dt c.s c.w in
  let tx = make c.dt c.s c.tc and tw = make c.dt c.s c.tw in
  let k0 = Nx.scalar Nx.int32 0l in
  (* The capture is read beside the carry, so that the step computes nothing
     from it alone, once at the call. *)
  let step w (x, k) =
    ( Nx.broadcast_to c.s (eval x (Nx.mul x w) c.g),
      Nx.add k (Nx.scalar Nx.int32 1l) )
  in
  let until (_, k) =
    Nx.greater_equal k (Nx.scalar Nx.int32 (Int32.of_int c.trips))
  in
  let iterated (x, w) =
    fst
      (Rune.iterate
         T.(pair tensor tensor)
         ~max:c.max ~until ~f:(step w) (x, k0))
  in
  let written (x, w) = fst (looped ~max:c.max ~until ~f:(step w) (x, k0)) in
  let p = T.(pair tensor tensor) in
  let l = lenient ~nan:(constant c.g) in
  let jvp f =
    outcome (fun () -> bits (snd (Rune.jvp p T.tensor f (x, w) (tx, tw))))
  in
  equal ~msg:"jvp" (outcome_t l) (jvp written) (jvp iterated);
  let grad f =
    outcome (fun () ->
        let gx, gw = Rune.grad p (fun xw -> Nx.sum (f xw)) (x, w) in
        (bits gx, bits gw))
  in
  let expected = grad written in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  let pair = outcome_t (Testable.pair l l) in
  equal ~msg:"grad" pair expected (grad iterated);
  (* Compiled, reverse mode keeps room for [max] steps, and a value the step
     captures receives a zero where the loop takes none, NaN where it meets an
     infinite partial derivative. *)
  let lj = lenient ~nan:(constant c.g || c.trips = 0) in
  equal ~msg:"jit of grad"
    (outcome_t (Testable.pair lj lj))
    expected
    (outcome (fun () ->
         let gx, gw =
           Rune.jit
             T.(p @-> returns p)
             (Rune.grad p (fun xw -> Nx.sum (iterated xw)))
             (x, w)
         in
         (bits gx, bits gw)))

(* An iterate's derivative is the steps taken, a stop with none. *)
let test_trips_taken () =
  let x = vec [| 3. |] in
  let f x =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.less (Nx.max x) (Nx.scalar f64 1.))
      ~f:(fun x -> Nx.mul_s x 0.5)
      x
  in
  (* Two halvings: 3, 1.5, 0.75. *)
  equal ~msg:"grad" bits_t
    (bits (vec [| 0.25 |]))
    (bits (Rune.grad' (fun x -> Nx.sum (f x)) x));
  equal ~msg:"jvp" bits_t
    (bits (vec [| 0.25 |]))
    (bits (snd (Rune.jvp' f x (vec [| 1. |]))));
  (* A tolerance the stop reads has no derivative. *)
  let tol = vec [| 1. |] in
  let g tol =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.less (Nx.max x) (Nx.max tol))
      ~f:(fun x -> Nx.mul_s x 0.5)
      x
  in
  equal ~msg:"a stop's tracked value" bits_t
    (bits (vec [| 0. |]))
    (bits (Rune.grad' (fun t -> Nx.sum (g t)) tol))

(* Maps *)

(* Lanes of [x] and a trip count each, the lanes stopping apart. *)
type lanes =
  | Lanes : {
      dt : (float, 's) Nx.dtype;
      s : int array;
      trips : int array;
      max : int;
      g : expr;
      x : operand;
      w : operand;
    }
      -> lanes

let lane_cases =
  let open Gen in
  (let* s = shape in
   let* n = int_range 0 3 in
   let* trips = array ~size:(constant n) (int_range 0 3) in
   let most = Array.fold_left Int.max 0 trips in
   let* max = of_list [ most; most + 1; most - 1 ] in
   let* g = expr ~rank:(Array.length s) in
   let* x = operand (Array.append [| n |] s) and+ w = operand s in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Lanes { dt; s; trips; max = Int.max 0 max; g = unsummed g; x; w }
   | _ -> assert false)
  |> with_pp (fun ppf (Lanes c) ->
      Format.fprintf ppf "%a %a, trips [%s], max %d: %a" Nx.pp_dtype c.dt
        pp_shape c.s
        (String.concat "; " (Array.to_list (Array.map string_of_int c.trips)))
        c.max pp c.g)

let one_by_one f n (x, t) =
  if n > 0 then
    Nx.stack ~axis:0 (List.init n (fun k -> f (Nx.get [ k ] x, Nx.get [ k ] t)))
  else
    let row = Array.sub (Nx.shape x) 1 (Nx.ndim x - 1) in
    let y = f (Nx.ones (Nx.dtype x) row, Nx.scalar Nx.int32 0l) in
    Nx.zeros (Nx.dtype y) (Array.append [| 0 |] (Nx.shape y))

let law_lanes (Lanes c) =
  let n = Array.length c.trips in
  cover "lanes stopping apart"
    (n > 1 && Array.exists (fun t -> t <> c.trips.(0)) c.trips);
  cover "a lane with no step" (Array.exists (( = ) 0) c.trips);
  cover "max reached" (c.max < Array.fold_left Int.max 0 c.trips);
  cover "no lane" (n = 0);
  let x = make c.dt (Array.append [| n |] c.s) c.x and w = make c.dt c.s c.w in
  let t = Nx.create Nx.int32 [| n |] (Array.map Int32.of_int c.trips) in
  let one = Nx.scalar Nx.int32 1l in
  let step w (x, k) =
    (Nx.broadcast_to c.s (eval x (Nx.mul x w) c.g), Nx.add k one)
  in
  let iterated w (x, t) =
    fst
      (Rune.iterate
         T.(pair tensor tensor)
         ~max:c.max
         ~until:(fun (_, k) -> Nx.greater_equal k t)
         ~f:(step w)
         (x, Nx.scalar Nx.int32 0l))
  in
  let written w (x, t) =
    fst
      (looped ~max:c.max
         ~until:(fun (_, k) -> Nx.greater_equal k t)
         ~f:(step w)
         (x, Nx.scalar Nx.int32 0l))
  in
  let s = T.(pair tensor tensor @-> returns tensor) in
  (* A lane that runs out of steps is named, the first one. *)
  let first_short =
    let rec go k =
      if k = n then None else if c.trips.(k) > c.max then Some k else go (k + 1)
    in
    go 0
  in
  let expected =
    outcome (fun () ->
        (* The step is traced before any lane runs out of steps. *)
        ignore (written w (Nx.ones c.dt c.s, Nx.scalar Nx.int32 0l));
        Option.iter
          (fun k ->
            invalid_arg
              (Printf.sprintf
                 "Rune.iterate: until is still false after max = %d steps, in \
                  lane %d"
                 c.max k))
          first_short;
        codes (one_by_one (written w) n (x, t)))
  in
  let map f = outcome (fun () -> codes (Rune.vmap s f (x, t))) in
  equal ~msg:"vmap" (outcome_t codes_t) expected (map (iterated w));
  equal ~msg:"vmap of jit" (outcome_t codes_t) expected
    (map (Rune.jit s (iterated w)));
  (* Each lane's gradient is its own loop's; a stopped lane's is finite where
     its own steps are. *)
  if Option.is_none first_short then begin
    let l = lenient ~nan:(constant c.g) in
    let grad f xt = Rune.grad' (fun x -> Nx.sum (f (x, snd xt))) (fst xt) in
    let per_lane =
      outcome (fun () -> bits (one_by_one (grad (written w)) n (x, t)))
    in
    equal ~msg:"vmap of grad" (outcome_t l) per_lane
      (outcome (fun () -> bits (Rune.vmap s (grad (iterated w)) (x, t))));
    equal ~msg:"grad of vmap" (outcome_t l) per_lane
      (outcome (fun () ->
           bits
             (Rune.grad' (fun x -> Nx.sum (Rune.vmap s (iterated w) (x, t))) x)));
    (* A captured parameter's gradient sums every lane's. *)
    let gw f =
      outcome (fun () -> bits (Rune.grad' (fun w -> Nx.sum (f w)) w))
    in
    let lanes_of f w =
      one_by_one (fun xt -> Nx.reshape [| 1 |] (Nx.sum (f w xt))) n (x, t)
    in
    equal ~msg:"a capture's gradient"
      (outcome_t (near c.dt))
      (gw (lanes_of written))
      (gw (fun w -> Rune.vmap s (iterated w) (x, t)))
  end

(* A stopped lane never computes at a point no lane reached: it computes its
   donor's step, so a check the step makes there passes. *)
let test_donors () =
  let x = vec [| 1.; 60. |] in
  let f =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.greater_equal (Nx.max x) (Nx.scalar f64 10.))
      ~f:(fun x ->
        Nx.check T.unit (Nx.less x (Nx.scalar f64 55.)) () (fun _ () -> Exit);
        Nx.mul_s x 2.)
  in
  equal bits_t (bits (vec [| 16.; 60. |])) (bits (Rune.vmap' f x));
  (* A lane whose own step is NaN but that stops at its init has the gradient of
     no step. *)
  let g =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.greater_equal (Nx.max x) (Nx.scalar f64 10.))
      ~f:(fun x -> Nx.add (Nx.log (Nx.sub_s x 0.5)) (Nx.mul_s x 2.))
  in
  let x = vec [| 1.; 20. |] in
  let per_lane =
    Nx.stack [ Rune.grad' (fun x -> Nx.sum (g x)) (vec [| 1. |]); vec [| 1. |] ]
  in
  equal bits_t
    (bits (Nx.reshape [| 2; 1 |] per_lane))
    (bits
       (Rune.vmap'
          (fun x -> Rune.grad' (fun x -> Nx.sum (g x)) x)
          (Nx.reshape [| 2; 1 |] x)))

(* Lanes held keep a NaN or infinite carry bit for bit. *)
let test_nonfinite () =
  let f =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.logical_not (Nx.less (Nx.max x) (Nx.scalar f64 10.)))
      ~f:(fun x -> Nx.add_s x 3.)
  in
  let x = vec [| Float.nan; Float.infinity; 2.; -0. |] in
  let x = Nx.reshape [| 4; 1 |] x in
  equal codes_t
    (codes (one_by_one (fun (x, _) -> f x) 4 (x, Nx.zeros Nx.int32 [| 4 |])))
    (codes (Rune.vmap' f x))

(* Inside a step whose lanes stop apart, the map's lanes raise; a stop every
   lane shares stops them together, and lanes may meet in its step. *)
let test_lanes_in_step () =
  let a = Rune.axis () in
  let x = vec [| 1.; 4. |] in
  invalid ~sub:"Rune.lanes: the lanes of the map stand at different steps"
    (fun () ->
      Rune.vmap' ~axis:a
        (Rune.iterate' ~max:10
           ~until:(fun x -> Nx.greater_s (Nx.sum x) 5.)
           ~f:(fun x -> Nx.add x (Nx.sum (Rune.lanes a x))))
        (Nx.reshape [| 2; 1 |] x));
  let y =
    Rune.vmap' ~axis:a
      (fun x ->
        fst
          (Rune.iterate
             T.(pair tensor tensor)
             ~max:10
             ~until:(fun (_, k) -> Nx.greater_equal_s k 2.)
             ~f:(fun (x, k) ->
               (Nx.add x (Nx.sum (Rune.lanes a x)), Nx.add_s k 1.))
             (x, Nx.scalar f64 0.)))
      (Nx.reshape [| 2; 1 |] x)
  in
  (* [1; 4] -> [6; 9] -> [21; 24]. *)
  equal bits_t (bits (Nx.create f64 [| 2; 1 |] [| 21.; 24. |])) (bits y)

(* An iterate in a step whose lanes stop apart: an outer lane that has stopped
   starts the inner loop stopped. *)
let test_nested () =
  let inner x =
    Rune.iterate' ~max:10
      ~until:(fun y -> Nx.greater_equal (Nx.max y) (Nx.max x))
      ~f:(fun y -> Nx.add_s y 1.)
      (Nx.zeros_like x)
  in
  let outer x =
    Rune.iterate' ~max:10
      ~until:(fun y -> Nx.greater_equal (Nx.max y) (Nx.scalar f64 6.))
      ~f:(fun y -> Nx.add (Nx.add_s y 1.) (inner y))
      x
  in
  let x = Nx.create f64 [| 3; 1 |] [| 0.5; 2.; 7. |] in
  let n = 3 in
  equal codes_t
    (codes
       (one_by_one (fun (x, _) -> outer x) n (x, Nx.zeros Nx.int32 [| n |])))
    (codes (Rune.vmap' outer x))

(* Totals: each lane adds once per step it takes, and per test of its stop. *)
let test_totals () =
  let t = Total.make () in
  let f x =
    Rune.iterate' ~max:10
      ~until:(fun x ->
        Total.add t (Nx.scalar f64 100.);
        Nx.greater_equal (Nx.max x) (Nx.scalar f64 3.))
      ~f:(fun x ->
        Total.add t (Nx.scalar f64 1.);
        Nx.add_s x 1.)
      x
  in
  let x = Nx.create f64 [| 3; 1 |] [| 0.; 2.; 5. |] in
  (* The lanes' counts depend on their values, on the host, where the totals
     lie. *)
  let zero () = Nx.place Nx.Host.on (Nx.scalar f64 0.) in
  let total f = snd (Total.collect t ~zero:(zero ()) f) in
  (* Three, one and no steps: 7 tests, 4 steps. *)
  equal ~msg:"around the map" bits_t
    (bits (Nx.scalar f64 704.))
    (bits (total (fun () -> ignore (Rune.vmap' f x))));
  equal ~msg:"inside the map" bits_t
    (bits (vec [| 403.; 201.; 100. |]))
    (bits
       (Rune.vmap'
          (fun x -> Nx.reshape [| 1 |] (total (fun () -> ignore (f x))))
          x
       |> Nx.reshape [| 3 |]));
  equal ~msg:"around grad of the map" bits_t
    (bits (Nx.scalar f64 704.))
    (bits
       (total (fun () ->
            ignore (Rune.grad' (fun x -> Nx.sum (Rune.vmap' f x)) x))))

(* Carries of other dtypes and structures. *)
let test_carries () =
  (* An integer carry has no derivative. *)
  let f (x, n) =
    Rune.iterate
      T.(pair tensor tensor)
      ~max:10
      ~until:(fun (_, n) -> Nx.greater_equal_s n 3l)
      ~f:(fun (x, n) -> (Nx.mul x (Nx.cast f64 n), Nx.add_s n 1l))
      (x, n)
  in
  let x = vec [| 2. |] and n = Nx.create Nx.int32 [| 1 |] [| 1l |] in
  let gx, gn =
    Rune.grad T.(pair tensor tensor) (fun xn -> Nx.sum (fst (f xn))) (x, n)
  in
  equal ~msg:"the float's" bits_t (bits (vec [| 2. |])) (bits gx);
  equal ~msg:"the integer's" (array int32) [| 0l |] (Nx.to_array gn);
  (* A complex carry, under a map. *)
  let c =
    Nx.complex Nx.complex128
      ~re:(Nx.create f64 [| 2; 1 |] [| 0.5; 2. |])
      ~im:(Nx.create f64 [| 2; 1 |] [| 1.; -1. |])
  in
  let g z =
    Rune.iterate' ~max:10
      ~until:(fun z -> Nx.greater_equal_s (Nx.max (Nx.real f64 z)) 2.)
      ~f:(fun z ->
        Nx.mul z
          (Nx.complex Nx.complex128 ~re:(Nx.scalar f64 1.5)
             ~im:(Nx.scalar f64 0.)))
      z
  in
  let grad z =
    Rune.grad' (fun z -> Nx.sum (Nx.real f64 (Nx.mul (g z) (g z)))) z
  in
  let parts z = Array.map (fun z -> (z.Complex.re, z.im)) (Nx.to_array z) in
  equal ~msg:"a complex carry"
    (array (pair float_exact float_exact))
    (parts (Nx.stack [ grad (Nx.get [ 0 ] c); grad (Nx.get [ 1 ] c) ]))
    (parts (Rune.vmap' grad c));
  (* A map of one lane is the loop. *)
  let h x =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.greater_s (Nx.max x) 4.)
      ~f:(fun x -> Nx.mul_s x 3.)
      x
  in
  equal ~msg:"one lane" bits_t
    (bits (Nx.reshape [| 1; 1 |] (h (vec [| 1. |]))))
    (bits (Rune.vmap' h (Nx.reshape [| 1; 1 |] (vec [| 1. |]))));
  (* vmap of grad of vmap: lanes of lanes. *)
  let xs = Nx.create f64 [| 2; 2; 1 |] [| 1.; 5.; 0.5; 3. |] in
  let gg =
    Rune.vmap' (fun row -> Rune.grad' (fun r -> Nx.sum (Rune.vmap' h r)) row) xs
  in
  let expected =
    Nx.reshape [| 2; 2; 1 |]
      (Nx.stack
         (List.concat_map
            (fun i ->
              List.map
                (fun j ->
                  Rune.grad' (fun x -> Nx.sum (h x)) (Nx.get [ i; j ] xs))
                [ 0; 1 ])
            [ 0; 1 ]))
  in
  equal ~msg:"lanes of lanes" bits_t (bits expected) (bits gg)

(* A read in a stop raises, as in a step. *)
let test_reads () =
  invalid ~sub:"Nx.item: the value is traced by Rune.iterate" (fun () ->
      Rune.iterate' ~max:3
        ~until:(fun x -> Nx.scalar Nx.bool (Nx.item [ 0 ] x > 1.))
        ~f:(fun x -> Nx.mul_s x 2.)
        (vec [| 1. |]))

(* Memory *)

let device = Nx_support.memory 1

module Mem = (val Nx.devices [ device ])

let settled () =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create device 0);
    Rig.free_cache device
  done;
  Rig.allocated device

(* Eager reverse mode keeps a carry per step taken, never [max] of them. *)
let test_memory () =
  let n = 1024 in
  let x = Nx.place Mem.on (Nx.full Nx.float32 [| n |] 0.25) in
  let kept = ref [] in
  let held trips =
    let run () =
      kept :=
        Rune.vjp'
          (fun x ->
            fst
              (Rune.iterate
                 T.(pair tensor tensor)
                 ~max:50
                 ~until:(fun (_, k) ->
                   Nx.greater_equal_s k (Int32.of_int trips))
                 ~f:(fun (x, k) -> (Nx.sin x, Nx.add_s k 1l))
                 (x, Nx.scalar_at (Nx.Placement.uncut Mem.on) Nx.int32 0l)))
          x
        :: !kept
    in
    run ();
    let before = settled () in
    run ();
    let held = settled () - before in
    ignore (Sys.opaque_identity !kept);
    kept := [];
    held
  in
  (* The result and each step's carry, which the sine's tangent reads; with no
     step, the result is the argument. *)
  equal ~msg:"three steps" int (4 * n * 4) (held 3);
  equal ~msg:"no step" int 0 (held 0);
  ignore (Sys.opaque_identity x)

let () =
  exit
    (run "rune2 iterate"
       [
         group "derivatives"
           [
             prop "iterate's derivatives are its loop's" cases law_derivatives;
             test "the steps taken" test_trips_taken;
           ];
         group "maps"
           [
             prop "each lane stops on its own" lane_cases law_lanes;
             test "a stopped lane computes its donor's step" test_donors;
             test "a stopped lane keeps a non-finite carry" test_nonfinite;
             test "lanes in a step" test_lanes_in_step;
             test "an iterate in a step" test_nested;
             test "totals count each lane's steps" test_totals;
             test "carries" test_carries;
           ];
         test "a read in a stop raises" test_reads;
         group "memory"
           [ test "reverse mode keeps the steps taken" test_memory ];
       ])
