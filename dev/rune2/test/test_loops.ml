(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.iterate: traced at the call, eagerly; an iterate is the while loop of
   its step, bit for bit, every NaN read as one. *)

open Windtrap
open Compose
module Rune = Rune2

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let f32 = Nx.float32
let v xs = Nx.create f32 [| Array.length xs |] xs
let t = Nx.Ptree.tensor
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
  let iterate step (init, k0) =
    fst
      (Rune.iterate
         Nx.Ptree.(pair tensor tensor)
         ~max ~until ~f:step (init, k0))
  in
  let iterated = outcome (fun () -> codes (iterate step (init, k0))) in
  equal (outcome_t codes_t) expected iterated;
  (* Compiled, the loop gives eager's bits where its step is exact. *)
  let g' = exact_any dt g in
  let exact_step (x, k) =
    (Nx.broadcast_to s (eval_any x x g'), Nx.add k (Nx.scalar Nx.int32 1l))
  in
  let jitted =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      (fun init k0 -> iterate exact_step (init, k0))
  in
  let before = Rune.evaluations () in
  let compiled = outcome (fun () -> codes (jitted init k0)) in
  cover "the compiled call compiled"
    ((match compiled with Raised _ -> false | _ -> true)
    && Rune.evaluations () = before);
  equal ~msg:"compiled" (outcome_t codes_t)
    (outcome (fun () -> codes (iterate exact_step (init, k0))))
    compiled

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

(* Carries, rows and additions that creations make: a value is read back at the
   form it was traced at, and a step is traced once. A step that returns a carry
   on another set than its init's does not type. *)
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
