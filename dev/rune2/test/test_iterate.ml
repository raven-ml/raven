(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.iterate: the OCaml while loop of its step, bit for bit, every NaN read
   as one, eagerly, under forward mode, reverse mode and maps, and compiled
   where the step's functions are exact; each lane of a map stops on its own. *)

open Windtrap
open Rune2_support
open Witness
open Expr
module Rune = Rune2
module T = Nx.Ptree
module D = Nx.Dtype
module Total = Rune.Total

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let f32 = Nx.float32
let f64 = Nx.float64
let c64 = Nx.complex64
let c128 = Nx.complex128
let v xs = Nx.create f32 [| Array.length xs |] xs
let vec xs = Nx.create f64 [| Array.length xs |] xs
let t = T.tensor
let p2 () = T.(pair tensor tensor)
let k0 () = Nx.scalar Nx.int32 0l
let tick k = Nx.add k (Nx.scalar Nx.int32 1l)

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

(* Bits, but for zeros' signs, and with [nan] NaNs the second value has: a
   loop's derivative holds a tangent or cotangent of a carry at every step, a
   zero where a step gives none, whose products are zeros of either sign, and
   NaNs with an infinite partial derivative, where the steps written out have no
   term. *)
let lenient ~nan =
  let zero x = x = 0L || x = Int64.min_int in
  Testable.make ~pp:(Testable.pp bits_t) ~equal:(fun (s, a) (s', b) ->
      s = s'
      && Array.length a = Array.length b
      && Array.for_all2
           (fun x y ->
             x = y || (nan && y = Int64.minus_one) || (zero x && zero y))
           a b)

(* [f ()], covering that a call of a loop compiled: no call it made was
   evaluated. *)
let compiled f =
  let before = Rune.evaluations () in
  let r = f () in
  cover "compiled"
    ((match r with Raised _ -> false | _ -> true)
    && Rune.evaluations () = before);
  r

(* Eagerly *)

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
  let x, k =
    Rune.iterate (p2 ()) ~max:10
      ~until:(fun (x, _) -> below_one x)
      ~f:(fun (x, k) -> (halve x, tick k))
      (v [| 20.; 3. |], k0 ())
  in
  equal int32 5l (Nx.item [] k);
  equal bits_t (bits (v [| 0.625; 0.09375 |])) (bits x)

(* An iterate whose stop counts trips in its carry, against the while loop of
   its step, at every dtype, and compiled where the step is exact; the step is
   traced at the call, so its errors raise when no step is taken too. *)
type any_case =
  | Any_case :
      ('v, 's) Nx.dtype * int array * int * int * Operand.t * Expr.t
      -> any_case

let any_cases =
  let open Gen in
  (let* s = Operand.shape in
   let* trips = int_range 0 3 in
   let* max = of_list [ trips; trips + 2; trips - 1 ] in
   let* g = gen ~rank:(Array.length s) in
   let* c = Operand.draw s in
   let+ (D.Any dt) = of_list D.all in
   Any_case (dt, s, trips, max, c, generic g))
  |> with_pp (fun ppf (Any_case (dt, s, trips, max, _, g)) ->
      Format.fprintf ppf "%a %a, %d trips, max %d: %a" Nx.pp_dtype dt
        Operand.pp_shape s trips max pp g)

let law_any (Any_case (dt, s, trips, max, co, g)) =
  cover "no trip" (trips = 0);
  cover "max reached" (max < trips);
  cover "a negative max" (max < 0);
  let init = Operand.make dt s co in
  let step g (x, k) = (Nx.broadcast_to s (eval_any x x g), tick k) in
  let until (_, k) =
    Nx.greater_equal k (Nx.scalar Nx.int32 (Int32.of_int trips))
  in
  let expected g =
    outcome (fun () ->
        if max < 0 then
          invalid_arg (Printf.sprintf "Rune.iterate: max = %d is negative" max);
        codes (fst (looped ~max ~until ~f:(step g) (init, k0 ()))))
  in
  let iterate g (init, k) =
    fst (Rune.iterate (p2 ()) ~max ~until ~f:(step g) (init, k))
  in
  equal (outcome_t codes_t) (expected g)
    (outcome (fun () -> codes (iterate g (init, k0 ()))));
  let g = exact_at dt g in
  let jitted =
    Rune.jit
      T.(tensor @-> tensor @-> returns tensor)
      (fun init k -> iterate g (init, k))
  in
  equal ~msg:"compiled" (outcome_t codes_t) (expected g)
    (compiled (fun () -> outcome (fun () -> codes (jitted init (k0 ())))))

let test_refused () =
  let halving ~max ~until x = Rune.iterate t ~max ~until ~f:halve x in
  invalid ~sub:"Rune.iterate: until is still false after max = 2 steps"
    (fun () -> halving ~max:2 ~until:below_one (v [| 20. |]));
  invalid ~sub:"Rune.iterate: max = -1 is negative" (fun () ->
      halving ~max:(-1) ~until:below_one (v [| 20. |]));
  invalid ~sub:"Rune.iterate: until must return one boolean, got shape [2]"
    (fun () -> halving ~max:2 ~until:(fun x -> Nx.less x x) (v [| 20.; 1. |]));
  invalid
    ~sub:"Rune.iterate: the root: shape [1] in the carry the step returned"
    (fun () ->
      Rune.iterate t ~max:2 ~until:below_one ~f:(Nx.sum ~keepdims:true)
        (v [| 20.; 1. |]));
  (* A message names the entry point applied. *)
  invalid ~sub:"Rune.iterate': max = -1 is negative" (fun () ->
      Rune.iterate' ~max:(-1) ~until:below_one ~f:halve (v [| 20. |]))

exception Until
exception Step

(* [until] is tested before each step: a carry that fails a check in both raises
   the test's failure, eagerly and compiled, and an init that satisfies it is
   never stepped. *)
let test_until_first () =
  let checked e x =
    Nx.check T.unit (Nx.less_s x 4.) () (fun _ () -> e);
    x
  in
  let loop x =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.greater_s (Nx.max (checked Until x)) 100.)
      ~f:(fun x -> Nx.add_s (checked Step x) 2.)
      x
  in
  let x = Nx.place Nx.Host.on (v [| 0. |]) in
  raises Until (fun () -> loop x);
  raises ~msg:"compiled" Until (fun () -> Rune.jit' loop x);
  let stopped x =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.greater_s (Nx.max x) 1.)
      ~f:(checked Step) x
  in
  equal bits_t (bits (v [| 9. |])) (bits (stopped (v [| 9. |])));
  equal ~msg:"compiled" bits_t
    (bits (v [| 9. |]))
    (bits (Rune.jit' stopped (v [| 9. |])))

(* Carries and creations *)

let placement =
  Testable.make ~pp:Nx.Placement.pp ~equal:(fun p q -> Nx.Placement.equal p q)

let host x = Nx.place Nx.Host.on x
let three = Nx.scalar f32 3.

(* A constant init whose step returns a placed carry: the carry lies where the
   step returns it. *)
let test_constant_carry () =
  let one = host (Nx.scalar f32 1.) in
  let c =
    Rune.iterate t ~max:10
      ~until:(fun c -> Nx.greater_equal c three)
      ~f:(fun c -> Nx.add c one)
      (Nx.scalar f32 0.)
  in
  equal bits_t (bits three) (bits c);
  equal ~msg:"where the step returns it" placement Nx.Host.on (Nx.placement c)

(* One leaf computed by its step, the other a creation of it. *)
let test_crossing_carry () =
  let one = host (Nx.scalar f32 1.) in
  let a, b =
    Rune.iterate (p2 ()) ~max:10
      ~until:(fun (a, _) -> Nx.greater_equal a three)
      ~f:(fun (a, _) -> (Nx.add a one, Nx.zeros f32 [||]))
      (Nx.scalar f32 0., host (Nx.scalar f32 5.))
  in
  equal bits_t (bits three) (bits a);
  equal bits_t (bits (Nx.scalar f32 0.)) (bits b);
  equal ~msg:"a's placement" placement Nx.Host.on (Nx.placement a);
  equal ~msg:"b's placement" placement
    (Nx.placement (Nx.zeros f32 [||]))
    (Nx.placement b)

(* A carry of two leaves, one computed from the other: one trace. *)
let test_chain () =
  let one = host (Nx.scalar f32 1.) and traces = ref 0 in
  let a, b =
    Rune.iterate (p2 ()) ~max:10
      ~until:(fun (a, _) -> Nx.greater_equal a three)
      ~f:(fun (a, _) ->
        incr traces;
        (Nx.add a one, a))
      (Nx.scalar f32 0., Nx.scalar f32 0.)
  in
  equal ~msg:"traces" int 1 !traces;
  equal bits_t (bits three) (bits a);
  equal bits_t (bits (Nx.scalar f32 2.)) (bits b)

(* A step that returns a constant over a placed init: no step or one, under jit
   and inside a scan's step. *)
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

(* A constant carry under jit, whose step captures the jit's argument. *)
let test_captured_argument () =
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

(* Compiled, the step is traced once and the loop is one loop of the program,
   which runs no eager kernel, whatever its number of trips. *)
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])

let test_staged () =
  let traces = ref 0 in
  let until x =
    incr traces;
    let s = Nx.sum x in
    Nx.less s (Nx.scalar_like s 0.01)
  in
  let halve x = Nx.mul_s x 0.5 in
  let g = Rune.jit' (fun x -> Rune.iterate' ~max:50 ~until ~f:halve x) in
  let x = Nx.place Count.on (v [| 1.; 2.; 3. |]) in
  ignore (g x);
  equal ~msg:"one trace" int 1 !traces;
  traces := 0;
  C.reset ();
  let y = g x in
  equal ~msg:"no retrace" int 0 !traces;
  equal ~msg:"no eager kernel" int 0 (C.calls ());
  equal bits_t (bits (Rune.iterate' ~max:50 ~until ~f:halve x)) (bits y);
  invalid ~sub:"Rune.iterate': until is still false after max = 3 steps"
    (fun () -> Rune.jit' (fun x -> Rune.iterate' ~max:3 ~until ~f:halve x) x)

(* A compiled call in a step stays a call, which computes at each step the bits
   it computes outside the loop, over carries in every view and a capture. *)
type call =
  | Call : {
      dt : (float, 's) Nx.dtype;
      sine : (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t;
      s : int array;
      trips : int;
      c : Operand.t;
      w : Operand.t;
    }
      -> call

let sine32 = Rune.jit' (Nx.sin : (float, Nx.float32_elt, Nx.host) Nx.t -> _)
let sine64 = Rune.jit' (Nx.sin : (float, Nx.float64_elt, Nx.host) Nx.t -> _)

let call_cases =
  let open Gen in
  (let* s = Operand.shape in
   let* trips = int_range 1 3 in
   let* c = Operand.draw s and+ w = Operand.draw s in
   let+ wide = bool in
   if wide then Call { dt = f64; sine = sine64; s; trips; c; w }
   else Call { dt = f32; sine = sine32; s; trips; c; w })
  |> with_pp (fun ppf (Call c) ->
      Format.fprintf ppf "%a %a, %d trips, carry %a" Nx.pp_dtype c.dt
        Operand.pp_shape c.s c.trips Operand.pp_view c.c.view)

let law_call (Call c) =
  cover "a view" (c.c.view <> Operand.Plain);
  cover "a zero-size carry" (Operand.numel c.s = 0);
  let init = Operand.make c.dt c.s c.c and w = Operand.make c.dt c.s c.w in
  (* Each call reads the carry, so that it computes in the step. *)
  let step sine (x, k) = (Nx.add (sine x) (sine (Nx.maximum w x)), tick k) in
  let until (_, k) = Nx.greater_equal_s k (Int32.of_int c.trips) in
  let written sine = fst (looped ~max:3 ~until ~f:(step sine) (init, k0 ())) in
  cover "a step the call computes apart from eager"
    (bits (written c.sine) <> bits (written Nx.sin));
  equal bits_t
    (bits (written c.sine))
    (bits
       (fst (Rune.iterate (p2 ()) ~max:3 ~until ~f:(step c.sine) (init, k0 ()))))

(* Derivatives *)

(* A step [c -> g(c, c w)] over host leaves in every view, and a counter, the
   loop taking [trips] steps, [max] at most. *)
type case =
  | Case : {
      dt : (float, 's) Nx.dtype;
      s : int array;
      trips : int;
      max : int;
      g : Expr.t;
      o : Operand.t array;
    }
      -> case

let cases =
  let open Gen in
  (let* s = Operand.shape in
   let* trips = int_range 0 3 in
   let* max = of_list [ trips; trips + 2; trips - 1 ] in
   let* g = gen ~rank:(Array.length s) in
   let* c = Operand.draw s and+ w = Operand.draw s in
   let* tc = Operand.draw s and+ tw = Operand.draw s in
   let+ (D.Any dt) = of_list [ D.Any f32; Any f64 ] in
   match D.kind dt with
   | Float ->
       Case { dt; s; trips; max = Int.max 0 max; g; o = [| c; w; tc; tw |] }
   | _ -> assert false)
  |> with_pp (fun ppf (Case c) ->
      Format.fprintf ppf "%a %a, %d trips, max %d, carry %a, capture %a: %a"
        Nx.pp_dtype c.dt Operand.pp_shape c.s c.trips c.max Operand.pp_view
        c.o.(0).view Operand.pp_view c.o.(1).view pp c.g)

let law_derivatives (Case c) =
  cover "no step" (c.trips = 0);
  cover "several steps" (c.trips > 1);
  cover "max reached" (c.max < c.trips);
  cover "room past the steps" (c.trips > 0 && c.max > c.trips);
  cover "a view operand"
    (c.o.(0).view <> Operand.Plain || c.o.(1).view <> Operand.Plain);
  let make k = Operand.make c.dt c.s c.o.(k) in
  let xw = (make 0, make 1) and txw = (make 2, make 3) in
  (* The capture is read beside the carry, so that the step computes nothing
     from it alone, once at the call. *)
  let step g w (x, k) = (Nx.broadcast_to c.s (eval x (Nx.mul x w) g), tick k) in
  let until (_, k) =
    Nx.greater_equal k (Nx.scalar Nx.int32 (Int32.of_int c.trips))
  in
  let iterated g (x, w) =
    fst (Rune.iterate (p2 ()) ~max:c.max ~until ~f:(step g w) (x, k0 ()))
  in
  let written g (x, w) =
    fst (looped ~max:c.max ~until ~f:(step g w) (x, k0 ()))
  in
  (* A carry whose step reads nothing keeps a zero tangent and cotangent where
     the loop written out has none. *)
  let l = if constant c.g then lenient ~nan:true else bits_t in
  let jvp f =
    outcome (fun () ->
        let y, t = Rune.jvp (p2 ()) t f xw txw in
        (bits y, bits t))
  in
  let expected = jvp (written c.g) in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  let tj = outcome_t (Testable.pair bits_t l) in
  equal ~msg:"jvp" tj expected (jvp (iterated c.g));
  (* A linearization's record gives jvp's tangent, as often as it is applied. *)
  let linearized =
    outcome (fun () ->
        let y, at = Rune.linearize (p2 ()) t (iterated c.g) xw in
        let t = at txw in
        (bits y, bits t, bits (at txw)))
  in
  let first = function
    | Value (y, t, _) -> Value (y, t)
    | Raised m -> Raised m
  in
  equal ~msg:"linearize"
    (outcome_t (Testable.pair bits_t bits_t))
    (jvp (iterated c.g))
    (first linearized);
  (match linearized with
  | Value (_, t, t') -> equal ~msg:"linearize applied again" bits_t t t'
  | Raised _ -> ());
  let grad f =
    outcome (fun () ->
        let gx, gw = Rune.grad (p2 ()) (fun xw -> Nx.sum (f xw)) xw in
        (bits gx, bits gw))
  in
  let expected = grad (written c.g) in
  let lp = outcome_t (Testable.pair l l) in
  equal ~msg:"grad" lp expected (grad (iterated c.g));
  (* Compiled, reverse mode keeps room for [max] steps, and where the loop takes
     fewer the capture receives a zero cotangent, NaN where it meets an infinite
     partial derivative. Its bits are the written loop's where the step's
     functions are exact and sum no terms along an axis. *)
  let capture = if c.max > c.trips then lenient ~nan:true else l in
  let room = outcome_t (Testable.pair l capture) in
  let ge = unsummed (exact c.g) in
  let le = if constant ge then lenient ~nan:true else bits_t in
  let le_capture = if c.max > c.trips then lenient ~nan:true else le in
  let calls = Rune.evaluations () in
  equal ~msg:"jit of grad"
    (outcome_t (Testable.pair le le_capture))
    (grad (written ge))
    (compiled (fun () ->
         outcome (fun () ->
             let gx, gw =
               Rune.jit
                 T.(p2 () @-> returns (p2 ()))
                 (Rune.grad (p2 ()) (fun xw -> Nx.sum (iterated ge xw)))
                 xw
             in
             (bits gx, bits gw))));
  equal ~msg:"grad compiled" int 0 (Rune.evaluations () - calls);
  (* A linearization's record compiles as forward mode's steps. *)
  let tangent = function Value (_, t) -> Value t | Raised m -> Raised m in
  let calls = Rune.evaluations () in
  equal ~msg:"jit of a linearization" (outcome_t le)
    (tangent (jvp (written ge)))
    (compiled (fun () ->
         outcome (fun () ->
             let at xw txw =
               snd (Rune.linearize (p2 ()) t (iterated ge) xw) txw
             in
             bits (Rune.jit T.(p2 () @-> p2 () @-> returns tensor) at xw txw))));
  equal ~msg:"linearization compiled" int 0 (Rune.evaluations () - calls);
  (* Inside a remat reverse mode keeps room for [max] steps, as compiled. *)
  let remat f = Rune.remat T.(p2 () @-> returns tensor) f in
  equal ~msg:"grad of a remat" room expected
    (grad (fun xw -> remat (iterated c.g) xw));
  let jvp_grad f =
    outcome (fun () ->
        let _, (tx, tw) =
          Rune.jvp (p2 ()) (p2 ())
            (Rune.grad (p2 ()) (fun xw -> Nx.sum (f xw)))
            xw txw
        in
        (bits tx, bits tw))
  in
  (* With forward mode around it too. The transposed loop's carry, the carry's
     cotangent, holds a tangent at every step: a zero where no step gave it one
     yet, which a step's product turns into a zero of either sign, or NaN, in
     the carry's and the capture's tangents alike (Sums). *)
  equal ~msg:"jvp of grad"
    (outcome_t (Testable.pair (lenient ~nan:true) (lenient ~nan:true)))
    (jvp_grad (written c.g))
    (jvp_grad (iterated c.g));
  (* Forward mode compiles, where the step's functions are exact and sum no
     terms along an axis. *)
  let g = unsummed (exact c.g) in
  let staged =
    Rune.jit
      T.(p2 () @-> p2 () @-> returns (p2 ()))
      (fun xw txw -> Rune.jvp (p2 ()) t (iterated g) xw txw)
  in
  equal ~msg:"jit of jvp"
    (outcome_t
       (Testable.pair bits_t (if constant g then lenient ~nan:true else bits_t)))
    (jvp (written g))
    (compiled (fun () ->
         outcome (fun () ->
             let y, t = staged xw txw in
             (bits y, bits t))))

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

(* An iterate of a constant computes at the call, so the function it scales
   stays linear. *)
let test_transpose () =
  let doubled x =
    fst
      (Rune.iterate (p2 ()) ~max:3
         ~until:(fun (_, k) -> Nx.greater_equal_s k 2l)
         ~f:(fun (x, k) -> (Nx.mul_s x 2., tick k))
         (x, k0 ()))
  in
  let scaled x = Nx.mul x (doubled (vec [| 1.; -1. |])) in
  equal bits_t
    (bits (vec [| 4.; -8. |]))
    (bits
       (Rune.linear_transpose t t scaled (vec [| 1.; -2. |]) (vec [| 1.; 2. |])));
  (* An iterate over the argument whose stop counts its steps is the steps
     taken, transposed. *)
  let counted x =
    fst
      (Rune.iterate (p2 ()) ~max:3
         ~until:(fun (_, k) -> Nx.greater_equal_s k 2l)
         ~f:(fun (x, k) -> (Nx.mul_s x 2., tick k))
         (x, k0 ()))
  in
  equal bits_t
    (bits (vec [| 4.; -8. |]))
    (bits
       (Rune.linear_transpose t t counted
          (vec [| 0.; 0. |])
          (vec [| 1.; -2. |])));
  (* One whose stop reads the argument has no transpose, and raises at the
     call. *)
  let stopped x =
    Rune.iterate' ~max:3
      ~until:(fun x -> Nx.less (Nx.max x) (Nx.scalar f64 1.))
      ~f:(fun x -> Nx.mul_s x 0.5)
      x
  in
  let refused f =
    raises
      (Invalid_argument
         "Rune.linear_transpose: an iterate's stop reads the function's \
          argument") (fun () ->
        let (_ : _ -> _) = Rune.linear_transpose t t f (vec [| 0.; 0. |]) in
        ())
  in
  refused stopped;
  (* So does one whose stop captures the argument, or reads a carry the steps
     compute from it. *)
  refused (fun x ->
      Rune.iterate' ~max:3
        ~until:(fun c -> Nx.less (Nx.max (Nx.add c x)) (Nx.scalar f64 1.))
        ~f:(fun c -> Nx.mul_s c 0.5)
        (vec [| 4.; 4. |]));
  refused (fun x ->
      fst
        (Rune.iterate (p2 ()) ~max:3
           ~until:(fun (_, s) -> Nx.greater (Nx.max s) (Nx.scalar f64 1.))
           ~f:(fun (y, s) -> (Nx.mul_s y 2., Nx.add s x))
           (x, vec [| 0.; 0. |])))

(* Maps *)

(* [e] with each sum along an axis replaced by its extreme, so that a map's
   lanes compute what one lane does whatever the order its sums associate in. *)
let rec no_sums = function
  | (X | Y | K _) as e -> e
  | Un ((Sum a | Cumsum a | Mean a), e) -> Un (Max a, no_sums e)
  | Un (u, e) -> Un (u, no_sums e)
  | Bin (b, e, e') -> Bin (b, no_sums e, no_sums e')

(* [y] is the sum of [terms], arrays of one shape, in some order, beside zeros
   that may meet an infinite partial derivative, giving NaN. Each element is
   NaN, or else: non-finite where a term is, NaN never where one is; within [2 k
   u Σ|term|] of the sum of its [k] terms added in order where it is finite, [u]
   the dtype's unit roundoff, as each sum is within [(k - 1) u Σ|term|] of the
   exact one; infinite only where [Σ|term|] passes the dtype's greatest float,
   so that a partial sum may overflow. An addition whose result is subnormal is
   exact, so the bound holds there too. *)
let summed (type s) (dt : (float, s) Nx.dtype) terms y =
  let f32 = D.equal dt f32 in
  let u = if f32 then 0x1p-24 else 0x1p-53 in
  let greatest = if f32 then 0x1.fffffep127 else Float.max_float in
  let k = Float.of_int (List.length terms) in
  let terms = List.map Nx.to_array terms and y = Nx.to_array y in
  Array.iteri
    (fun i y ->
      let at f = List.fold_left (fun a t -> a +. f t.(i)) 0. terms in
      let any p = List.exists (fun t -> p t.(i)) terms in
      let total = at Fun.id and bound = at Float.abs in
      let msg = Printf.sprintf "element %d: %h from %h" i y total in
      cover "a capture's sum bounded" (Float.is_finite y && bound <= greatest);
      cover "a lane's steps cancel"
        (Float.is_finite bound && Float.abs total < bound);
      if Float.is_nan y then ()
      else if any Float.is_nan then failf "%s, a term NaN" msg
      else if any (fun t -> Float.abs t = infinity) then
        if Float.is_finite y then failf "%s, a term infinite" msg else ()
      else if Float.is_finite y then
        at_most ~msg float_exact
          ~than:(2. *. k *. u *. bound)
          (Float.abs (y -. total))
      else if bound <= greatest then
        failf "%s, the terms' magnitudes summing to %h" msg bound)
    y

(* Lanes of [x] and a trip count each, the lanes stopping apart, at a float or
   complex dtype. *)
type lanes =
  | Lanes : {
      dt : ('v, 's) Nx.dtype;
      s : int array;
      trips : int array;
      max : int;
      g : Expr.t;
      x : Operand.t;
      w : Operand.t;
    }
      -> lanes

(* [re] whose drawn floats take [im]'s as imaginary parts, in storage order and
   repeated: the two views may hold different numbers of elements. *)
let paired (re : Operand.t) (im : Operand.t) =
  let n = Array.length im.values in
  let pair k (a : Operand.value) : Operand.value =
    match (a, im.values.(k mod n)) with F a, F b -> C (a, b) | a, _ -> a
  in
  if n = 0 then re else { re with values = Array.mapi pair re.values }

let lane_cases =
  let open Gen in
  (let* s = Operand.shape in
   let* n = int_range 0 3 in
   let* trips = array ~size:(constant n) (int_range 0 3) in
   let most = Array.fold_left Int.max 0 trips in
   let* max = of_list [ most; most + 1; most - 1 ] in
   let* g = gen ~rank:(Array.length s) in
   let xs = Array.append [| n |] s in
   let* x = Operand.draw xs and+ w = Operand.draw s in
   let* xi = Operand.draw xs and+ wi = Operand.draw s in
   let+ (D.Any dt) = of_list [ D.Any f32; Any f64; Any c64; Any c128 ] in
   let max = Int.max 0 max in
   match D.kind dt with
   | Float -> Lanes { dt; s; trips; max; g = no_sums g; x; w }
   | _ ->
       let x = paired x xi and w = paired w wi in
       Lanes { dt; s; trips; max; g = exact_at dt (generic g); x; w })
  |> with_pp (fun ppf (Lanes c) ->
      Format.fprintf ppf "%a %a, trips [%s], max %d: %a" Nx.pp_dtype c.dt
        Operand.pp_shape c.s
        (String.concat "; " (Array.to_list (Array.map string_of_int c.trips)))
        c.max pp c.g)

let one_by_one f n (x, t) =
  if n > 0 then
    Nx.stack ~axis:0 (List.init n (fun k -> f (Nx.get [ k ] x, Nx.get [ k ] t)))
  else
    let row = Array.sub (Nx.shape x) 1 (Nx.ndim x - 1) in
    let y = f (Nx.ones (Nx.dtype x) row, k0 ()) in
    Nx.zeros (Nx.dtype y) (Array.append [| 0 |] (Nx.shape y))

(* [e] at [x] and [y], by the functions [dt] has. *)
let eval_at (type v s d) (dt : (v, s) Nx.dtype) (x : (v, s, d) Nx.t)
    (y : (v, s, d) Nx.t) e : (v, s, d) Nx.t =
  match D.kind dt with Float -> eval x y e | _ -> eval_any x y e

(* A float or complex array's bits, a complex one's real parts then its
   imaginary ones. *)
let bits_at (type v s d) (dt : (v, s) Nx.dtype) (x : (v, s, d) Nx.t) =
  match D.kind dt with
  | Float -> bits x
  | Complex ->
      let parts at = bits (Nx.stack [ Nx.real at x; Nx.imag at x ]) in
      if D.equal dt c64 then parts f32 else parts f64
  | _ -> assert false

(* The gradient of the sum of [f x], of its real parts on a complex one. *)
let grad_at (type v s d) (dt : (v, s) Nx.dtype)
    (f : (v, s, d) Nx.t -> (v, s, d) Nx.t) (x : (v, s, d) Nx.t) =
  match D.kind dt with
  | Complex when D.equal dt c64 ->
      Rune.grad' (fun x -> Nx.sum (Nx.real f32 (f x))) x
  | Complex -> Rune.grad' (fun x -> Nx.sum (Nx.real f64 (f x))) x
  | _ -> Rune.grad' (fun x -> Nx.sum (f x)) x

(* [summed] at [dt], on each part of a complex array: a complex sum adds each
   part as a real one. *)
let summed_at (type v s d) (dt : (v, s) Nx.dtype) (terms : (v, s, d) Nx.t list)
    (y : (v, s, d) Nx.t) =
  match D.kind dt with
  | Float -> summed dt terms y
  | Complex ->
      let parts : type r. (float, r) Nx.dtype -> unit =
       fun at ->
        let re x = Nx.real at x and im x = Nx.imag at x in
        summed at (List.map re terms) (re y);
        summed at (List.map im terms) (im y)
      in
      if D.equal dt c64 then parts f32 else parts f64
  | _ -> assert false

let law_lanes (Lanes c) =
  let n = Array.length c.trips in
  cover "lanes stopping apart"
    (n > 1 && Array.exists (fun t -> t <> c.trips.(0)) c.trips);
  cover "a lane with no step" (Array.exists (( = ) 0) c.trips);
  cover "max reached" (c.max < Array.fold_left Int.max 0 c.trips);
  cover "room past a lane's steps"
    (Array.exists (fun t -> t > 0 && t < c.max) c.trips);
  cover "no lane" (n = 0);
  cover "a complex dtype" (D.is Complex c.dt);
  cover "an imaginary part not zero"
    (Array.exists
       (function Operand.C (_, im) -> im <> 0. | _ -> false)
       c.x.values);
  let x = Operand.make c.dt (Array.append [| n |] c.s) c.x in
  let w = Operand.make c.dt c.s c.w in
  let t = Nx.create Nx.int32 [| n |] (Array.map Int32.of_int c.trips) in
  let step ?(g = c.g) w (x, k) =
    (Nx.broadcast_to c.s (eval_at c.dt x (Nx.mul x w) g), tick k)
  in
  let until t (_, k) = Nx.greater_equal k t in
  let iterated ?g w (x, t) =
    fst
      (Rune.iterate (p2 ()) ~max:c.max ~until:(until t) ~f:(step ?g w)
         (x, k0 ()))
  in
  let written ?g w (x, t) =
    fst (looped ~max:c.max ~until:(until t) ~f:(step ?g w) (x, k0 ()))
  in
  (* Compiled, the step's functions are exact. *)
  let exact = Expr.exact c.g in
  let s = T.(p2 () @-> returns tensor) in
  (* A lane that runs out of steps is named, the first one. *)
  let first_short =
    let rec go k =
      if k = n then None else if c.trips.(k) > c.max then Some k else go (k + 1)
    in
    go 0
  in
  let expected g =
    outcome (fun () ->
        (* The step is traced before any lane runs out of steps. *)
        ignore (written ~g w (Nx.ones c.dt c.s, k0 ()));
        Option.iter
          (fun k ->
            invalid_arg
              (Printf.sprintf
                 "Rune.iterate: until is still false after max = %d steps, in \
                  lane %d"
                 c.max k))
          first_short;
        codes (one_by_one (written ~g w) n (x, t)))
  in
  let map f = outcome (fun () -> codes (Rune.vmap s f (x, t))) in
  let ct = outcome_t codes_t in
  equal ~msg:"vmap" ct (expected c.g) (map (iterated w));
  let expected = expected exact in
  equal ~msg:"vmap of jit" ct expected (map (Rune.jit s (iterated ~g:exact w)));
  equal ~msg:"jit of vmap" ct expected
    (outcome (fun () ->
         codes (Rune.jit s (Rune.vmap s (iterated ~g:exact w)) (x, t))));
  (* Each lane's gradient is its own loop's; a stopped lane's is finite where
     its own steps are. *)
  if Option.is_none first_short then begin
    let l = outcome_t (if constant c.g then lenient ~nan:true else bits_t) in
    let bits = bits_at c.dt in
    let grad f xt = grad_at c.dt (fun x -> f (x, snd xt)) (fst xt) in
    let per_lane =
      outcome (fun () -> bits (one_by_one (grad (written w)) n (x, t)))
    in
    (* A lane whose gradient is -0 where a held lane's selection would add a
       +0. *)
    cover "a lane's gradient holds -0"
      (match per_lane with
      | Value (_, a) -> Array.exists (Int64.equal Int64.min_int) a
      | Raised _ -> false);
    equal ~msg:"vmap of grad" l per_lane
      (outcome (fun () -> bits (Rune.vmap s (grad (iterated w)) (x, t))));
    equal ~msg:"grad of vmap" l per_lane
      (outcome (fun () ->
           bits (grad_at c.dt (fun x -> Rune.vmap s (iterated w) (x, t)) x)));
    (* Compiled, the bits are each lane's loop's where the step's derivative
       also sums no terms along an axis. *)
    let g = unsummed exact in
    let exact_lanes =
      outcome (fun () -> bits (one_by_one (grad (written ~g w)) n (x, t)))
    in
    let calls = Rune.evaluations () in
    equal ~msg:"jit of vmap of grad" l exact_lanes
      (outcome (fun () ->
           let f = Rune.vmap s (grad (iterated ~g w)) in
           bits (Rune.jit s f (x, t))));
    equal ~msg:"jit of grad of vmap" l exact_lanes
      (outcome (fun () ->
           let f = grad_at c.dt (fun x -> Rune.vmap s (iterated ~g w) (x, t)) in
           bits (Rune.jit' f x)));
    equal ~msg:"grad compiled" int 0 (Rune.evaluations () - calls);
    (* Forward mode, inside and around the map. *)
    let jvp f xt =
      snd (Rune.jvp' (fun x -> f (x, snd xt)) (fst xt) (Nx.ones_like (fst xt)))
    in
    let per_lane =
      outcome (fun () -> bits (one_by_one (jvp (written w)) n (x, t)))
    in
    equal ~msg:"vmap of jvp" l per_lane
      (outcome (fun () -> bits (Rune.vmap s (jvp (iterated w)) (x, t))));
    equal ~msg:"jvp of vmap" l per_lane
      (outcome (fun () ->
           bits
             (snd
                (Rune.jvp'
                   (fun x -> Rune.vmap s (iterated w) (x, t))
                   x (Nx.ones_like x)))));
    let calls = Rune.evaluations () in
    equal ~msg:"jit of jvp of vmap" l
      (outcome (fun () -> bits (one_by_one (jvp (written ~g:exact w)) n (x, t))))
      (outcome (fun () ->
           let f x = Rune.vmap s (iterated ~g:exact w) (x, t) in
           bits (Rune.jit' (fun x -> snd (Rune.jvp' f x (Nx.ones_like x))) x)));
    equal ~msg:"jvp compiled" int 0 (Rune.evaluations () - calls);
    (* A captured parameter's gradient sums each step's of every lane, in an
       order the map chooses: step [j]'s is the gradient at row [j] of a stack
       of the parameter, of the lane's loop whose step [j] reads that row. *)
    let gw f = grad_at c.dt f w in
    let rows = Nx.broadcast_to (Array.append [| Int.max 1 c.max |] c.s) w in
    let at ws (x, k) =
      step (Nx.get [ Int32.to_int (Nx.item [] k) ] ws) (x, k)
    in
    let term k =
      let lane ws =
        fst
          (looped ~max:c.max
             ~until:(until (Nx.get [ k ] t))
             ~f:(at ws)
             (Nx.get [ k ] x, k0 ()))
      in
      let g = grad_at c.dt lane (Nx.copy rows) in
      List.init c.max (fun j -> Nx.get [ j ] g)
    in
    (* The step is traced, so a refused one raises with no lane too. *)
    let terms =
      outcome (fun () ->
          ignore (gw (fun w -> written w (Nx.ones c.dt c.s, k0 ())));
          List.concat (List.init n term))
    in
    let mapped =
      outcome (fun () -> gw (fun w -> Rune.vmap s (iterated w) (x, t)))
    in
    match (terms, mapped) with
    | Value terms, Value y -> summed_at c.dt terms y
    | _ ->
        let unit = function Value _ -> Value () | Raised m -> Raised m in
        equal ~msg:"a capture's gradient" (outcome_t Testable.unit) (unit terms)
          (unit mapped)
  end

(* One lane whose steps' terms of the capture's gradient cancel: from [x = 2],
   [w = 1], the step [x w - x²] gives 130, -26 and -6. One lane whose gradient,
   [w = -0], is -0. *)
let lane_examples =
  let one v = { Operand.view = Plain; values = [| F v |] } in
  [
    Lanes
      {
        dt = f32;
        s = [| 1 |];
        trips = [| 1 |];
        max = 1;
        g = Y;
        x = one 2.;
        w = one (-0.);
      };
    Lanes
      {
        dt = f64;
        s = [| 1 |];
        trips = [| 3 |];
        max = 3;
        g = Bin (Sub, Y, Un (Square, X));
        x = one 2.;
        w = one 1.;
      };
  ]

(* Lanes of [x] at any dtype, along one map or two, each with its trip count:
   compiled, the lanes stopping apart or together, the map gives eager's bits
   and its failure, with no call evaluated. *)
type apart =
  | Apart :
      ('v, 's) Nx.dtype
      * int array
      * int array
      * int array
      * int
      * Expr.t
      * Operand.t
      -> apart

let apart_cases =
  let open Gen in
  (let* s = Operand.shape in
   let* lanes = of_list [ [| 0 |]; [| 1 |]; [| 3 |]; [| 2; 2 |]; [| 1; 3 |] ] in
   let n = Array.fold_left ( * ) 1 lanes in
   let* pattern = int_range 0 2 in
   let* trips =
     match pattern with
     | 0 -> array ~size:(constant n) (int_range 0 3)
     | 1 ->
         let+ t = int_range 1 3 in
         Array.make n t
     | _ -> constant (Array.make n 0)
   in
   let most = Array.fold_left Int.max 0 trips in
   let* max = of_list [ most; most + 1; most - 1 ] in
   let* g = gen ~rank:(Array.length s) in
   let* x = Operand.draw (Array.append lanes s) in
   let+ (D.Any dt) = of_list D.all in
   Apart (dt, s, lanes, trips, Int.max 0 max, exact_at dt (generic g), x))
  |> with_pp (fun ppf (Apart (dt, s, lanes, trips, max, g, _)) ->
      let ints a =
        String.concat "; " (Array.to_list (Array.map string_of_int a))
      in
      Format.fprintf ppf "%a %a, lanes [%s], trips [%s], max %d: %a" Nx.pp_dtype
        dt Operand.pp_shape s (ints lanes) (ints trips) max pp g)

let law_apart (Apart (dt, s, lanes, trips, max, g, co)) =
  let n = Array.length trips in
  let apart = n > 1 && Array.exists (( <> ) trips.(0)) trips in
  cover "lanes stopping apart" apart;
  cover "lanes stopping together" (n > 1 && (not apart) && trips.(0) > 0);
  cover "no lane steps" (n > 0 && Array.for_all (( = ) 0) trips);
  cover "one lane" (n = 1);
  cover "lanes of lanes" (Array.length lanes = 2);
  cover "max reached" (max < Array.fold_left Int.max 0 trips);
  cover "no lane" (n = 0);
  cover "max = 0" (max = 0);
  let x = Operand.make dt (Array.append lanes s) co in
  let t = Nx.create Nx.int32 lanes (Array.map Int32.of_int trips) in
  let step (x, k) = (Nx.broadcast_to s (eval_any x x g), tick k) in
  let iterated (x, t) =
    fst
      (Rune.iterate (p2 ()) ~max
         ~until:(fun (_, k) -> Nx.greater_equal k t)
         ~f:step
         (x, k0 ()))
  in
  let sg = T.(p2 () @-> returns tensor) in
  let mapped =
    if Array.length lanes = 1 then Rune.vmap sg iterated
    else Rune.vmap sg (Rune.vmap sg iterated)
  in
  let expected = outcome (fun () -> codes (mapped (x, t))) in
  let calls = Rune.evaluations () in
  equal ~msg:"compiled" (outcome_t codes_t) expected
    (outcome (fun () -> codes (Rune.jit sg mapped (x, t))));
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - calls)

(* A key in the carry, folded with the step count at each step: eagerly,
   compiled, per lane with lanes stopping apart, and under reverse mode, the
   draws are the written loop's, bit for bit. *)
let keyed_max = 3

let keyed_cases =
  let open Gen in
  (let* n = int_range 1 3 in
   let* trips = array ~size:(constant n) (int_range 0 keyed_max) in
   let+ seed = int_range 0 1000 in
   (seed, trips))
  |> with_pp (fun ppf (seed, trips) ->
      Format.fprintf ppf "seed %d, trips [%s]" seed
        (String.concat "; " (Array.to_list (Array.map string_of_int trips))))

let keyed_step (x, (key, k)) =
  let key' = Nx.Rng.fold_in_tensor (Nx.Rng.of_tensor key) k in
  (Nx.mul x (Nx.Rng.uniform ~key:key' f64 [| 2 |]), (key, tick k))

let keyed ~loop trips (x, key) =
  let until (_, (_, k)) = Nx.greater_equal k trips in
  fst (loop ~until (x, (key, k0 ())))

let keyed_iterate =
  keyed ~loop:(fun ~until ->
      Rune.iterate
        T.(pair tensor (pair tensor tensor))
        ~max:keyed_max ~until ~f:keyed_step)

let keyed_written = keyed ~loop:(looped ~max:keyed_max ~f:keyed_step)
let keyed_s = T.(tensor @-> p2 () @-> returns tensor)
let keyed_jit = Rune.jit keyed_s keyed_iterate

let keyed_grad f trips (x, key) =
  Rune.grad' (fun x -> Nx.sum (f trips (x, key))) x

let law_keyed (seed, trips) =
  let n = Array.length trips in
  cover "no trip" (Array.exists (( = ) 0) trips);
  cover "lanes apart" (Array.exists (( <> ) trips.(0)) trips);
  let keys = Nx.Rng.to_tensor (Nx.Rng.split_batch ~n (Nx.Rng.key seed)) in
  let xs = Nx.full f64 [| n; 2 |] 1.5 in
  let tr = Nx.create Nx.int32 [| n |] (Array.map Int32.of_int trips) in
  let lane f k = f (Nx.get [ k ] tr) (Nx.get [ k ] xs, Nx.get [ k ] keys) in
  let per_lane f = bits (Nx.stack (List.init n (lane f))) in
  let expected = per_lane keyed_written in
  equal ~msg:"eagerly" bits_t expected (per_lane keyed_iterate);
  equal ~msg:"compiled" bits_t expected (per_lane keyed_jit);
  equal ~msg:"vmap" bits_t expected
    (bits (Rune.vmap keyed_s keyed_iterate tr (xs, keys)));
  let expected = per_lane (keyed_grad keyed_written) in
  equal ~msg:"grad" bits_t expected (per_lane (keyed_grad keyed_iterate));
  equal ~msg:"vmap of grad" bits_t expected
    (bits (Rune.vmap keyed_s (keyed_grad keyed_iterate) tr (xs, keys)))

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
  (* Lane 1 stops at its init, where its own step is NaN, [log (15 - 20)]: its
     gradient is that of no step. *)
  let g =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.greater_equal (Nx.max x) (Nx.scalar f64 10.))
      ~f:(fun x ->
        Nx.add (Nx.log (Nx.sub (Nx.scalar f64 15.) x)) (Nx.mul_s x 2.))
  in
  let grad x = Rune.grad' (fun x -> Nx.sum (g x)) x in
  let per_lane = Nx.stack [ grad (vec [| 1. |]); vec [| 1. |] ] in
  equal bits_t (bits per_lane)
    (bits (Rune.vmap' grad (Nx.reshape [| 2; 1 |] (vec [| 1.; 20. |]))))

(* A constant init whose stop reads the lanes through a capture: each lane stops
   on its own. *)
let test_constant_init () =
  let f t =
    Rune.iterate' ~max:10
      ~until:(fun y -> Nx.greater_equal (Nx.max y) (Nx.max t))
      ~f:(fun y -> Nx.add_s y 1.)
      (Nx.zeros f64 [| 1 |])
  in
  equal bits_t
    (bits (Nx.create f64 [| 2; 1 |] [| 1.; 3. |]))
    (bits (Rune.vmap' f (Nx.create f64 [| 2; 1 |] [| 1.; 3. |])));
  (* Its forward derivative, the tangents beside the primals. *)
  let g x =
    Rune.iterate' ~max:10
      ~until:(fun y -> Nx.greater_equal (Nx.max y) (Nx.scalar f64 3.))
      ~f:(fun y -> Nx.add y x)
      (Nx.zeros f64 [| 1 |])
  in
  let x = Nx.create f64 [| 2; 1 |] [| 1.; 3. |] in
  equal bits_t
    (bits (Nx.create f64 [| 2; 1 |] [| 3.; 1. |]))
    (bits (Rune.vmap' (fun x -> snd (Rune.jvp' g x (Nx.ones_like x))) x))

(* Lanes held keep a NaN, its payload, or an infinite carry bit for bit. *)
let test_nonfinite () =
  let f =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.logical_not (Nx.less (Nx.max x) (Nx.scalar f64 10.)))
      ~f:(fun x -> Nx.add_s x 3.)
  in
  let nan = Int64.float_of_bits 0x7ff8_0000_0000_0123L in
  let x = Nx.reshape [| 4; 1 |] (vec [| nan; Float.infinity; 2.; -0. |]) in
  let raw x = Array.map Int64.bits_of_float (Nx.to_array x) in
  equal (array int64)
    (raw (vec [| nan; Float.infinity; 11.; 12. |]))
    (raw (Rune.vmap' f x))

(* Inside a step whose lanes stop apart, the map's lanes raise; a stop every
   lane shares stops them together, and lanes may meet in its step. *)
let test_lanes_in_step () =
  let a = Rune.axis () in
  let x = vec [| 1.; 4. |] in
  invalid ~sub:"Rune.lanes: the lanes of the map are held apart" (fun () ->
      Rune.vmap' ~axis:a
        (Rune.iterate' ~max:10
           ~until:(fun x -> Nx.greater_s (Nx.sum x) 5.)
           ~f:(fun x -> Nx.add x (Nx.sum (Rune.lanes a x))))
        (Nx.reshape [| 2; 1 |] x));
  let y =
    Rune.vmap' ~axis:a
      (fun x ->
        fst
          (Rune.iterate (p2 ()) ~max:10
             ~until:(fun (_, k) -> Nx.greater_equal_s k 2.)
             ~f:(fun (x, k) ->
               (Nx.add x (Nx.sum (Rune.lanes a x)), Nx.add_s k 1.))
             (x, Nx.scalar f64 0.)))
      (Nx.reshape [| 2; 1 |] x)
  in
  (* [1; 4] -> [6; 9] -> [21; 24]. *)
  equal bits_t (bits (Nx.create f64 [| 2; 1 |] [| 21.; 24. |])) (bits y);
  (* A call in the step: its lanes raise, and a held lane's index is its
     donor's. Lane 2 starts stopped, so its own index, outside the switch's
     branches, is never computed. *)
  invalid ~sub:"Rune.lanes: the lanes of the map are held apart" (fun () ->
      Rune.vmap' ~axis:a
        (Rune.iterate' ~max:10
           ~until:(fun x -> Nx.greater_s (Nx.sum x) 5.)
           ~f:(Rune.jit' (fun x -> Nx.add x (Nx.sum (Rune.lanes a x)))))
        (Nx.reshape [| 2; 1 |] x));
  let by_index x =
    Rune.switch t (Rune.lane_index a x)
      [ (fun () -> Nx.add_s x 1.); (fun () -> Nx.add_s x 2.) ]
  in
  let stepped f =
    Rune.vmap' ~axis:a
      (Rune.iterate' ~max:10
         ~until:(fun x -> Nx.greater_equal_s (Nx.sum x) 3.)
         ~f)
      (Nx.reshape [| 3; 1 |] (vec [| 0.; 0.; 5. |]))
  in
  equal ~msg:"eager" bits_t
    (bits (Nx.create f64 [| 3; 1 |] [| 3.; 4.; 5. |]))
    (bits (stepped by_index));
  let called = Rune.jit' by_index in
  equal ~msg:"a call" bits_t
    (bits (Nx.create f64 [| 3; 1 |] [| 3.; 4.; 5. |]))
    (bits (stepped called));
  (* The same call under lanes that stop together, then held again: each reads
     the lanes it runs under. *)
  equal ~msg:"a call, lanes together" bits_t
    (bits (Nx.create f64 [| 2; 1 |] [| 1.; 2. |]))
    (bits
       (Rune.vmap' ~axis:a called (Nx.reshape [| 2; 1 |] (vec [| 0.; 0. |]))));
  equal ~msg:"a call, held again" bits_t
    (bits (Nx.create f64 [| 3; 1 |] [| 3.; 4.; 5. |]))
    (bits (stepped called));
  (* Each lane's gradient through a called step, x -> 1.5 x + 1 until 3: lanes
     from 0.5, 2 and 4 take two, one and no steps. *)
  let affine = Rune.jit' (fun x -> Nx.add_s (Nx.mul_s x 1.5) 1.) in
  let f x =
    Rune.iterate' ~max:10 ~until:(fun x -> Nx.greater_equal_s x 3.) ~f:affine x
  in
  equal ~msg:"a call's gradient" bits_t
    (bits (vec [| 2.25; 1.5; 1. |]))
    (bits (Rune.vmap' (Rune.grad' f) (vec [| 0.5; 2.; 4. |])))

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
  let expected =
    codes (one_by_one (fun (x, _) -> outer x) 3 (x, Nx.zeros Nx.int32 [| 3 |]))
  in
  equal codes_t expected (codes (Rune.vmap' outer x));
  (* Compiled, the inner loop's lanes are held inside the outer's. *)
  let calls = Rune.evaluations () in
  equal ~msg:"compiled" codes_t expected
    (codes (Rune.jit' (Rune.vmap' outer) x));
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - calls)

(* A map of the derivative of an iterate in a loop's step: the step's tape
   stacks every step the loop may take, and counts those each lane takes. *)
let test_nested_derivative () =
  let halve c =
    Rune.iterate' ~max:50
      ~until:(fun y -> Nx.less_s (Nx.abs y) 0.1)
      ~f:(fun y -> Nx.mul_s y 0.5)
      c
  in
  let f x =
    fst (Rune.scan' ~init:x ~f:(fun c _ -> (halve c, c)) (Nx.zeros f64 [| 2 |]))
  in
  (* Each row halves [x] [k] times, the [k] that brings it below 0.1: a power of
     two, exact. *)
  let points = [| 0.05; 0.7; -3. |] in
  let rec slope x d =
    if Float.abs x < 0.1 then d else slope (x *. 0.5) (d *. 0.5)
  in
  equal bits_t
    (bits (vec (Array.map (fun x -> slope x 1.) points)))
    (bits (Rune.vmap' (Rune.grad' f) (vec points)))

(* Totals: each lane adds once per step it takes, and per test of its stop. *)
let test_totals () =
  let u = Total.make () in
  let f x =
    Rune.iterate' ~max:10
      ~until:(fun x ->
        Total.add u (Nx.scalar f64 100.);
        Nx.greater_equal (Nx.max x) (Nx.scalar f64 3.))
      ~f:(fun x ->
        Total.add u (Nx.scalar f64 1.);
        Nx.add_s x 1.)
      x
  in
  let x = Nx.create f64 [| 3; 1 |] [| 0.; 2.; 5. |] in
  let zero () = Nx.place Nx.Host.on (Nx.scalar f64 0.) in
  let total f = snd (Total.collect u ~zero:(zero ()) f) in
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
  equal ~msg:"around nested maps" bits_t
    (bits (Nx.scalar f64 704.))
    (bits
       (total (fun () ->
            ignore (Rune.vmap' (Rune.vmap' f) (Nx.reshape [| 1; 3; 1 |] x)))));
  (* A call's additions in a step whose lanes stop apart: a held lane's are
     dropped. *)
  let g =
    Rune.jit' (fun x ->
        Total.add u (Nx.scalar f64 1.);
        Nx.add_s x 1.)
  in
  let called x =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.greater_equal (Nx.max x) (Nx.scalar f64 3.))
      ~f:g x
  in
  equal ~msg:"a call's, around the map" bits_t
    (bits (Nx.scalar f64 4.))
    (bits (total (fun () -> ignore (Rune.vmap' called x))));
  equal ~msg:"around grad of the map" bits_t
    (bits (Nx.scalar f64 704.))
    (bits
       (total (fun () ->
            ignore (Rune.grad' (fun x -> Nx.sum (Rune.vmap' f x)) x))));
  (* With no collect open, the additions do nothing. *)
  ignore (Rune.vmap' f x);
  ignore (Rune.vmap' (Rune.vmap' f) (Nx.reshape [| 1; 3; 1 |] x));
  ignore (Rune.jit' (Rune.vmap' f) x);
  ignore
    (Rune.scan' ~init:(Nx.scalar f64 0.)
       ~f:(fun c r ->
         Total.add u (Nx.scalar f64 1.);
         (Nx.add c (Nx.sum (Rune.vmap' f (Nx.add_s r 0.))), c))
       (Nx.reshape [| 1; 3; 1 |] x));
  (* A jit around the map with no collect holds the counts where they lie, so a
     later call under a collect adds them. *)
  let g = Rune.jit' (Rune.vmap' f) in
  ignore (g x);
  equal ~msg:"a jit's later call" bits_t
    (bits (Nx.scalar f64 704.))
    (bits (total (fun () -> ignore (g x))))

(* An integer carry has no derivative. *)
let test_integer_carry () =
  let f (x, n) =
    Rune.iterate (p2 ()) ~max:10
      ~until:(fun (_, n) -> Nx.greater_equal_s n 3l)
      ~f:(fun (x, n) -> (Nx.mul x (Nx.cast f64 n), Nx.add_s n 1l))
      (x, n)
  in
  let x = vec [| 2. |] and n = Nx.create Nx.int32 [| 1 |] [| 1l |] in
  let gx, gn = Rune.grad (p2 ()) (fun xn -> Nx.sum (fst (f xn))) (x, n) in
  equal ~msg:"the float's" bits_t (bits (vec [| 2. |])) (bits gx);
  equal ~msg:"the integer's" (array int32) [| 0l |] (Nx.to_array gn)

(* A complex carry's gradient under a map is each lane's. *)
let test_complex_carry () =
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
  let per_lane = Nx.stack [ grad (Nx.get [ 0 ] c); grad (Nx.get [ 1 ] c) ] in
  equal codes_t (codes per_lane) (codes (Rune.vmap' grad c));
  (* Compiled, its stacked steps and the backward loop's rows are complex. *)
  let calls = Rune.evaluations () in
  equal ~msg:"jit of grad" codes_t
    (codes (grad (Nx.get [ 0 ] c)))
    (codes (Rune.jit' grad (Nx.get [ 0 ] c)));
  equal ~msg:"jit of vmap of grad" codes_t (codes per_lane)
    (codes (Rune.jit' (Rune.vmap' grad) c));
  equal ~msg:"compiled" int 0 (Rune.evaluations () - calls)

(* A map of one lane is the loop, and lanes of lanes are each lane's loop. *)
let test_one_lane () =
  let h x =
    Rune.iterate' ~max:10
      ~until:(fun x -> Nx.greater_s (Nx.max x) 4.)
      ~f:(fun x -> Nx.mul_s x 3.)
      x
  in
  equal ~msg:"one lane" bits_t
    (bits (Nx.reshape [| 1; 1 |] (h (vec [| 1. |]))))
    (bits (Rune.vmap' h (Nx.reshape [| 1; 1 |] (vec [| 1. |]))));
  let xs = Nx.create f64 [| 2; 2; 1 |] [| 1.; 5.; 0.5; 3. |] in
  let gg =
    Rune.vmap' (fun row -> Rune.grad' (fun r -> Nx.sum (Rune.vmap' h r)) row) xs
  in
  let lane i j = Rune.grad' (fun x -> Nx.sum (h x)) (Nx.get [ i; j ] xs) in
  equal ~msg:"lanes of lanes" bits_t
    (bits
       (Nx.reshape [| 2; 2; 1 |]
          (Nx.stack [ lane 0 0; lane 0 1; lane 1 0; lane 1 1 ])))
    (bits gg)

(* Memory *)

(* Eager reverse mode keeps a carry per step taken, never [max] of them. *)
let test_memory () =
  let n = 1024 in
  let x = Nx.place Memory.On.on (Nx.full f32 [| n |] 0.25) in
  let kept = ref [] in
  let held trips =
    let held =
      Memory.held (fun () ->
          kept :=
            Rune.vjp'
              (fun x ->
                fst
                  (Rune.iterate (p2 ()) ~max:50
                     ~until:(fun (_, k) ->
                       Nx.greater_equal_s k (Int32.of_int trips))
                     ~f:(fun (x, k) -> (Nx.sin x, Nx.add_s k 1l))
                     ( x,
                       Nx.scalar_at
                         (Nx.Placement.uncut Memory.On.on)
                         Nx.int32 0l )))
              x
            :: !kept)
    in
    ignore (Sys.opaque_identity !kept);
    kept := [];
    held
  in
  (* Each step taken keeps its carry, which the sine's tangent reads; with no
     step, the result is the argument. *)
  let carry = n * 4 in
  equal ~msg:"a third step" int carry (held 3 - held 2);
  equal ~msg:"a fourth step" int carry (held 4 - held 3);
  at_most ~msg:"three steps of max = 50, and the result" int ~than:(4 * carry)
    (held 3);
  equal ~msg:"no step" int 0 (held 0);
  ignore (Sys.opaque_identity x)

let () =
  exit
    (run "rune2 iterate"
       [
         group "eagerly"
           [
             test "iterate is the while loop" test_iterate;
             prop "iterate is the while loop at every dtype" any_cases law_any;
             test "misuse raises" test_refused;
             test "until is tested before each step" test_until_first;
             test "a carry lies where its step returns it" test_constant_carry;
             test "a step that returns a creation" test_crossing_carry;
             test "a step is traced once" test_chain;
             test "a placed init and a constant step" test_placed_init;
             test "a step that captures a jit's argument" test_captured_argument;
             test "compiled, traced once, no eager kernel" test_staged;
             prop "a compiled call in a step gives its own bits" call_cases
               law_call;
           ];
         group "derivatives"
           [
             prop "iterate's derivatives are its loop's" cases law_derivatives;
             test "the steps taken" test_trips_taken;
             test "linear_transpose over an iterate" test_transpose;
             test "an integer carry has none" test_integer_carry;
           ];
         group "maps"
           [
             prop ~examples:lane_examples "each lane stops on its own"
               lane_cases law_lanes;
             prop "compiled, lanes stop apart at every dtype" apart_cases
               law_apart;
             prop "a key in the carry draws as the loop" keyed_cases law_keyed;
             test "a stopped lane computes its donor's step" test_donors;
             test "a stopped lane keeps a non-finite carry" test_nonfinite;
             test "a constant init" test_constant_init;
             test "lanes in a step" test_lanes_in_step;
             test "an iterate in a step" test_nested;
             test "an iterate's derivative in a loop's step"
               test_nested_derivative;
             test "totals count each lane's steps" test_totals;
             test "a complex carry's gradient" test_complex_carry;
             test "one lane, and lanes of lanes" test_one_lane;
           ];
         group "memory"
           [ test "reverse mode keeps the steps taken" test_memory ];
       ])
