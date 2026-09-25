(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [S.eval] reads a schedule at a host counter; the values are float32, so
   expectations carry float32 tolerances. *)

open Windtrap
module S = Vega.Schedule

let test_constant () =
  let sched = S.constant 0.1 in
  equal (float 1e-6) 0.1 (S.eval sched 0);
  equal (float 1e-6) 0.1 (S.eval sched 1000)

let test_exponential_decay () =
  let sched =
    S.exponential_decay ~init_value:0.5 ~decay_rate:0.1 ~decay_steps:100
  in
  equal (float 1e-6) 0.5 (S.eval sched 0);
  equal (float 1e-6) 0.05 (S.eval sched 100);
  equal (float 1e-6) 0.005 (S.eval sched 200)

let test_cosine_decay () =
  (* alpha = 0.1 makes the final value alpha * init_value = 0.01. *)
  let sched = S.cosine_decay ~init_value:0.1 ~decay_steps:100 ~alpha:0.1 () in
  equal (float 1e-6) 0.1 (S.eval sched 0);
  equal (float 1e-6) 0.055 (S.eval sched 50);
  equal (float 1e-6) 0.01 (S.eval sched 100);
  equal ~msg:"stays at final past steps" (float 1e-6) 0.01 (S.eval sched 250)

let test_warmup_cosine () =
  let sched =
    S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:10
      ~decay_steps:100 ()
  in
  equal (float 1e-6) 0.0 (S.eval sched 0);
  equal (float 1e-6) 0.5 (S.eval sched 5);
  equal (float 1e-6) 1.0 (S.eval sched 10);
  equal ~msg:"cosine midpoint" (float 1e-6) 0.5 (S.eval sched 60);
  equal (float 1e-6) 0.0 (S.eval sched 110)

let test_schedule_validation () =
  raises
    (Invalid_argument "Schedule.exponential_decay: decay_steps must be positive")
    (fun () ->
      ignore
        (S.exponential_decay ~init_value:1.0 ~decay_rate:0.5 ~decay_steps:0
          : S.t));
  raises
    (Invalid_argument "Schedule.cosine_decay: decay_steps must be positive")
    (fun () ->
      ignore (S.cosine_decay ~init_value:1.0 ~decay_steps:(-1) () : S.t));
  raises
    (Invalid_argument
       "Schedule.warmup_cosine_decay: warmup_steps must be positive") (fun () ->
      ignore
        (S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:0
           ~decay_steps:10 ()
          : S.t));
  raises
    (Invalid_argument
       "Schedule.warmup_cosine_decay: decay_steps must be positive") (fun () ->
      ignore
        (S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:10
           ~decay_steps:0 ()
          : S.t))

let test_polynomial_decay () =
  let s =
    S.polynomial_decay ~init_value:1.0 ~end_value:0.0 ~decay_steps:100 ()
  in
  equal ~msg:"step 0" (float 1e-6) 1.0 (S.eval s 0);
  equal ~msg:"step 50 (power=1, linear)" (float 1e-6) 0.5 (S.eval s 50);
  equal ~msg:"step 100" (float 1e-6) 0.0 (S.eval s 100);
  equal ~msg:"clamps past end" (float 1e-6) 0.0 (S.eval s 200);
  let s2 =
    S.polynomial_decay ~init_value:1.0 ~end_value:0.0 ~decay_steps:100
      ~power:2.0 ()
  in
  equal ~msg:"power=2 at midpoint" (float 1e-6) 0.25 (S.eval s2 50)

let test_warmup_cosine_decay () =
  let s =
    S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:10
      ~decay_steps:90 ()
  in
  equal ~msg:"step 0" (float 1e-6) 0.0 (S.eval s 0);
  equal ~msg:"step 5 (warmup midpoint)" (float 1e-6) 0.5 (S.eval s 5);
  equal ~msg:"step 10 (peak)" (float 1e-6) 1.0 (S.eval s 10);
  equal ~msg:"step 100 (fully decayed)" (float 1e-6) 0.0 (S.eval s 100);
  equal ~msg:"past end" (float 1e-6) 0.0 (S.eval s 200)

let test_piecewise_constant () =
  let s =
    S.piecewise_constant ~boundaries:[ 10; 20 ] ~values:[ 1.0; 0.1; 0.01 ]
  in
  equal ~msg:"segment 1" (float 1e-6) 1.0 (S.eval s 5);
  equal ~msg:"boundary" (float 1e-6) 1.0 (S.eval s 10);
  equal ~msg:"segment 2" (float 1e-6) 0.1 (S.eval s 15);
  equal ~msg:"segment 3" (float 1e-6) 0.01 (S.eval s 25)

let test_piecewise_constant_validation () =
  raises_match Exn.invalid_arg (fun () ->
      ignore (S.piecewise_constant ~boundaries:[ 10 ] ~values:[ 1.0 ] : S.t));
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (S.piecewise_constant ~boundaries:[ 20; 10 ] ~values:[ 1.; 2.; 3. ]
          : S.t))

let test_join () =
  let s =
    S.join [ (10, S.constant 1.0); (10, S.constant 2.0); (10, S.constant 3.0) ]
  in
  equal ~msg:"segment 1" (float 1e-6) 1.0 (S.eval s 5);
  equal ~msg:"segment 2" (float 1e-6) 2.0 (S.eval s 15);
  equal ~msg:"segment 3" (float 1e-6) 3.0 (S.eval s 25);
  equal ~msg:"past end extends last" (float 1e-6) 3.0 (S.eval s 100)

let test_join_step_reset () =
  let calls = ref [] in
  let spy name =
    S.join
      [
        ( 5,
          fun step ->
            calls := (name, Int32.to_int (Nx.item [] step)) :: !calls;
            Nx.scalar Nx.float32 0. );
      ]
  in
  let s = spy "a" in
  ignore (S.eval s 3);
  equal ~msg:"step passed to inner schedule"
    (list (pair string int))
    [ ("a", 3) ]
    (List.rev !calls)

let test_join_validation () =
  raises_match Exn.invalid_arg (fun () -> ignore (S.join [] : S.t));
  raises_match Exn.invalid_arg (fun () ->
      ignore (S.join [ (0, S.constant 1.0) ] : S.t))

let test_cosine_decay_restarts () =
  let s = S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:100 () in
  equal ~msg:"step 0 (peak)" (float 1e-6) 1.0 (S.eval s 0);
  equal ~msg:"step 100 (restart)" (float 1e-6) 1.0 (S.eval s 100);
  equal ~msg:"step 200 (second restart)" (float 1e-6) 1.0 (S.eval s 200);
  equal ~msg:"step 50 (midpoint)" (float 1e-6) 0.5 (S.eval s 50)

let test_cosine_decay_restarts_t_mul () =
  let s =
    S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:10 ~t_mul:2.0 ()
  in
  (* First cycle: 10 steps. Second: 20 steps. *)
  equal ~msg:"step 0 (start)" (float 1e-6) 1.0 (S.eval s 0);
  equal ~msg:"step 10 (second cycle start)" (float 1e-6) 1.0 (S.eval s 10);
  equal ~msg:"step 30 (third cycle start)" (float 1e-6) 1.0 (S.eval s 30)

let test_cosine_decay_restarts_m_mul () =
  let s =
    S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:100 ~m_mul:0.5 ()
  in
  equal ~msg:"cycle 0 peak" (float 1e-6) 1.0 (S.eval s 0);
  equal ~msg:"cycle 1 peak" (float 1e-6) 0.5 (S.eval s 100);
  equal ~msg:"cycle 2 peak" (float 1e-6) 0.25 (S.eval s 200)

let test_one_cycle () =
  let s = S.one_cycle ~max_value:1.0 ~total_steps:100 () in
  (* warmup: 30 steps (pct_start=0.3), init=1/25=0.04, peak=1.0 *)
  equal ~msg:"step 0" (float 1e-6) 0.04 (S.eval s 0);
  equal ~msg:"step 30 (peak)" (float 1e-6) 1.0 (S.eval s 30);
  (* decay: 70 steps, from 1.0 to 1/10000=0.0001 *)
  let end_val = 1.0 /. 10000.0 in
  equal ~msg:"step 100 (end)" (float 1e-6) end_val (S.eval s 100)

let test_step_count_validation () =
  raises_match Exn.invalid_arg (fun () ->
      ignore (S.cosine_decay_restarts ~init_value:1. ~decay_steps:0 () : S.t));
  raises_match Exn.invalid_arg (fun () ->
      ignore (S.one_cycle ~max_value:1. ~total_steps:0 () : S.t))

let () =
  run "vega schedules"
    [
      group "schedules"
        [
          test "constant is constant" test_constant;
          test "exponential decay is geometric in steps" test_exponential_decay;
          test "cosine decay spans init to final" test_cosine_decay;
          test "warmup cosine ramps then decays" test_warmup_cosine;
          test "constructors reject bad step counts" test_schedule_validation;
          test "polynomial_decay" test_polynomial_decay;
          test "warmup_cosine_decay" test_warmup_cosine_decay;
          test "piecewise_constant" test_piecewise_constant;
          test "piecewise_constant validation"
            test_piecewise_constant_validation;
          test "join" test_join;
          test "join step reset" test_join_step_reset;
          test "join validation" test_join_validation;
          test "cosine_decay_restarts" test_cosine_decay_restarts;
          test "cosine_decay_restarts t_mul" test_cosine_decay_restarts_t_mul;
          test "cosine_decay_restarts m_mul" test_cosine_decay_restarts_m_mul;
          test "one_cycle" test_one_cycle;
          prop "constant is constant at any step"
            Gen.(pair float nat)
            (fun (v, step) ->
              let s = S.constant v in
              equal float_exact (S.eval s 0) (S.eval s step));
          prop "cosine_decay bounded" Gen.nat (fun step ->
              let s = S.cosine_decay ~init_value:1.0 ~decay_steps:100 () in
              let v = S.eval s step in
              is_true ~msg:">=0" (v >= 0.0);
              is_true ~msg:"<=1" (v <= 1.0 +. 1e-6));
          prop "one_cycle bounded" Gen.nat (fun step ->
              let s = S.one_cycle ~max_value:1.0 ~total_steps:100 () in
              let v = S.eval s step in
              is_true ~msg:">=0" (v >= 0.0);
              is_true ~msg:"<=max" (v <= 1.0 +. 1e-6));
          prop "cosine_decay_restarts periodic" Gen.nat (fun step ->
              let period = 50 in
              let s =
                S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:period ()
              in
              let v1 = S.eval s step in
              let v2 = S.eval s (step + period) in
              equal ~msg:"periodic" (float 1e-5) v1 v2);
          test "restarts and one-cycle reject bad step counts"
            test_step_count_validation;
        ];
    ]
