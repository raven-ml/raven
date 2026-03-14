(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Vega.Schedule

(* Helpers *)

let f32 = Nx.float32
let vec xs = Nx.create f32 [| Array.length xs |] xs
let mat r c xs = Nx.create f32 [| r; c |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] t)
let eps = float 1e-5
let lr01 = S.constant 0.1

let converges ~msg ~tol tx =
  let param = ref (vec [| 5.0; -3.0 |]) in
  let st = ref (Vega.init tx !param) in
  for _ = 1 to 200 do
    let p, s = Vega.step !st ~grad:!param ~param:!param in
    param := p;
    st := s
  done;
  let v = to_arr !param in
  equal ~msg:(msg ^ "[0]") (float tol) 0.0 v.(0);
  equal ~msg:(msg ^ "[1]") (float tol) 0.0 v.(1)

let raises_invalid_arg f =
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    f

(* Schedules *)

let test_polynomial_decay () =
  let s =
    S.polynomial_decay ~init_value:1.0 ~end_value:0.0 ~decay_steps:100 ()
  in
  equal ~msg:"step 0" (float 1e-10) 1.0 (s 0);
  equal ~msg:"step 50 (power=1, linear)" (float 1e-6) 0.5 (s 50);
  equal ~msg:"step 100" (float 1e-10) 0.0 (s 100);
  equal ~msg:"clamps past end" (float 1e-10) 0.0 (s 200);
  let s2 =
    S.polynomial_decay ~init_value:1.0 ~end_value:0.0 ~decay_steps:100
      ~power:2.0 ()
  in
  equal ~msg:"power=2 at midpoint" (float 1e-6) 0.25 (s2 50)

let test_warmup_cosine_decay () =
  let s =
    S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:10
      ~decay_steps:90 ()
  in
  equal ~msg:"step 0" (float 1e-10) 0.0 (s 0);
  equal ~msg:"step 5 (warmup midpoint)" (float 1e-6) 0.5 (s 5);
  equal ~msg:"step 10 (peak)" (float 1e-6) 1.0 (s 10);
  equal ~msg:"step 100 (fully decayed)" (float 1e-10) 0.0 (s 100);
  equal ~msg:"past end" (float 1e-10) 0.0 (s 200)

let test_piecewise_constant () =
  let s =
    S.piecewise_constant ~boundaries:[ 10; 20 ] ~values:[ 1.0; 0.1; 0.01 ]
  in
  equal ~msg:"segment 1" (float 1e-10) 1.0 (s 5);
  equal ~msg:"boundary" (float 1e-10) 1.0 (s 10);
  equal ~msg:"segment 2" (float 1e-10) 0.1 (s 15);
  equal ~msg:"segment 3" (float 1e-10) 0.01 (s 25)

let test_piecewise_constant_validation () =
  raises_invalid_arg (fun () ->
      ignore (S.piecewise_constant ~boundaries:[ 10 ] ~values:[ 1.0 ] 0));
  raises_invalid_arg (fun () ->
      ignore
        (S.piecewise_constant ~boundaries:[ 20; 10 ] ~values:[ 1.; 2.; 3. ] 0))

let test_join () =
  let s =
    S.join [ (10, S.constant 1.0); (10, S.constant 2.0); (10, S.constant 3.0) ]
  in
  equal ~msg:"segment 1" (float 1e-10) 1.0 (s 5);
  equal ~msg:"segment 2" (float 1e-10) 2.0 (s 15);
  equal ~msg:"segment 3" (float 1e-10) 3.0 (s 25);
  equal ~msg:"past end extends last" (float 1e-10) 3.0 (s 100)

let test_join_step_reset () =
  let calls = ref [] in
  let spy name =
    S.join
      [
        ( 5,
          fun step ->
            calls := (name, step) :: !calls;
            0. );
      ]
  in
  let s = spy "a" in
  ignore (s 3);
  equal ~msg:"step passed to inner schedule"
    (list (pair string int))
    [ ("a", 3) ]
    (List.rev !calls)

let test_join_validation () =
  raises_invalid_arg (fun () -> ignore (S.join [] 0));
  raises_invalid_arg (fun () -> ignore (S.join [ (0, S.constant 1.0) ] 0))

let test_cosine_decay_restarts () =
  let s = S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:100 () in
  equal ~msg:"step 0 (peak)" (float 1e-10) 1.0 (s 0);
  equal ~msg:"step 100 (restart)" (float 1e-6) 1.0 (s 100);
  equal ~msg:"step 200 (second restart)" (float 1e-6) 1.0 (s 200);
  equal ~msg:"step 50 (midpoint)" (float 1e-6) 0.5 (s 50)

let test_cosine_decay_restarts_t_mul () =
  let s =
    S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:10 ~t_mul:2.0 ()
  in
  (* First cycle: 10 steps. Second: 20 steps. *)
  equal ~msg:"step 0 (start)" (float 1e-6) 1.0 (s 0);
  equal ~msg:"step 10 (second cycle start)" (float 1e-6) 1.0 (s 10);
  equal ~msg:"step 30 (third cycle start)" (float 1e-6) 1.0 (s 30)

let test_cosine_decay_restarts_m_mul () =
  let s =
    S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:100 ~m_mul:0.5 ()
  in
  equal ~msg:"cycle 0 peak" (float 1e-6) 1.0 (s 0);
  equal ~msg:"cycle 1 peak" (float 1e-6) 0.5 (s 100);
  equal ~msg:"cycle 2 peak" (float 1e-6) 0.25 (s 200)

let test_one_cycle () =
  let s = S.one_cycle ~max_value:1.0 ~total_steps:100 () in
  (* warmup: 30 steps (pct_start=0.3), init=1/25=0.04, peak=1.0 *)
  equal ~msg:"step 0" (float 1e-6) 0.04 (s 0);
  equal ~msg:"step 30 (peak)" (float 1e-6) 1.0 (s 30);
  (* decay: 70 steps, from 1.0 to 1/10000=0.0001 *)
  let end_val = 1.0 /. 10000.0 in
  equal ~msg:"step 100 (end)" (float 1e-6) end_val (s 100)

(* Schedule property tests — these are `test` values, placed directly in the
   group list below. *)

(* Primitives *)

let test_scale () =
  let tx = Vega.scale 2.0 in
  let grad = vec [| 1.0; -0.5 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"scaled" (array eps) [| 2.0; -1.0 |] (to_arr upd)

let test_scale_by_schedule () =
  let tx = Vega.scale_by_schedule (S.constant 3.0) in
  let grad = vec [| 1.0; 2.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"scheduled" (array eps) [| 3.0; 6.0 |] (to_arr upd)

let test_scale_by_learning_rate () =
  let tx = Vega.scale_by_learning_rate (S.constant 0.1) in
  let grad = vec [| 10.0; -5.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* negated: updates = grad * (-0.1) *)
  equal ~msg:"negated lr" (array eps) [| -1.0; 0.5 |] (to_arr upd)

let test_trace () =
  let tx = Vega.trace ~decay:0.9 () in
  let grad = vec [| 1.0; 2.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* step 1: vel = 0.9*0 + grad = grad; output = vel *)
  equal ~msg:"step 1" (array eps) [| 1.0; 2.0 |] (to_arr upd)

let test_trace_nesterov () =
  let tx = Vega.trace ~decay:0.9 ~nesterov:true () in
  let grad = vec [| 1.0; 2.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* vel = grad; nesterov output = grad + 0.9 * vel = grad + 0.9*grad =
     1.9*grad *)
  equal ~msg:"nesterov" (array eps) [| 1.9; 3.8 |] (to_arr upd)

let test_add_decayed_weights () =
  let tx = Vega.add_decayed_weights ~rate:(S.constant 0.1) () in
  let grad = vec [| 1.0; 0.0 |] in
  let param = vec [| 10.0; -5.0 |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* updates + 0.1 * param *)
  equal ~msg:"wd" (array eps) [| 2.0; -0.5 |] (to_arr upd)

let test_add_decayed_weights_scheduled () =
  let rate step = 0.01 *. float_of_int step in
  let tx = Vega.add_decayed_weights ~rate () in
  let grad = vec [| 0.0 |] in
  let param = vec [| 10.0 |] in
  let st = Vega.init tx param in
  let upd1, st = Vega.update st ~grad ~param in
  (* step 1: rate=0.01, updates = 0 + 0.01*10 = 0.1 *)
  equal ~msg:"step 1" (array eps) [| 0.1 |] (to_arr upd1);
  let upd2, _ = Vega.update st ~grad ~param in
  (* step 2: rate=0.02, updates = 0 + 0.02*10 = 0.2 *)
  equal ~msg:"step 2" (array eps) [| 0.2 |] (to_arr upd2)

let test_clip () =
  let tx = Vega.clip_by_value 1.0 in
  let grad = vec [| 5.0; -0.5; -3.0 |] in
  let param = vec [| 0.; 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"clipped" (array eps) [| 1.0; -0.5; -1.0 |] (to_arr upd)

let test_clip_by_norm () =
  (* norm of [3, 4] = 5, clip to 2.5 → scale by 0.5 *)
  let tx = Vega.clip_by_norm 2.5 in
  let grad = vec [| 3.0; 4.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"rescaled" (array eps) [| 1.5; 2.0 |] (to_arr upd)

let test_clip_by_norm_no_op () =
  let tx = Vega.clip_by_norm 10.0 in
  let grad = vec [| 1.0; 1.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"unchanged" (array eps) [| 1.0; 1.0 |] (to_arr upd)

let test_trust_ratio () =
  let tx = Vega.scale_by_trust_ratio () in
  let grad = vec [| 1.0; 0.0 |] in
  let param = vec [| 3.0; 4.0 |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* ||param|| = 5, ||grad|| = 1, ratio = 5/1 = 5 *)
  equal ~msg:"ratio" (array (float 1e-4)) [| 5.0; 0.0 |] (to_arr upd)

let test_trust_ratio_zero_param () =
  let tx = Vega.scale_by_trust_ratio () in
  let grad = vec [| 1.0 |] in
  let param = vec [| 0.0 |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* zero param norm → ratio = 1 *)
  equal ~msg:"fallback" (array eps) [| 1.0 |] (to_arr upd)

(* Gradient processing *)

let test_centralize_2d () =
  let tx = Vega.centralize in
  (* 2x3 matrix: row 0 = [1,2,3] mean=2, row 1 = [4,5,6] mean=5 *)
  let grad = mat 2 3 [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let param = mat 2 3 [| 0.; 0.; 0.; 0.; 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"centralized" (array eps)
    [| -1.; 0.; 1.; -1.; 0.; 1. |]
    (to_arr upd)

let test_centralize_1d () =
  let tx = Vega.centralize in
  let grad = vec [| 1.; 2.; 3. |] in
  let param = vec [| 0.; 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"1d unchanged" (array eps) [| 1.; 2.; 3. |] (to_arr upd)

let test_add_noise () =
  let tx = Vega.add_noise ~eta:(S.constant 1.0) () in
  let grad = vec [| 0.; 0. |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  let v = to_arr upd in
  (* With zero grad, output is pure noise — should be non-zero with high prob *)
  is_true ~msg:"noise injected"
    (Float.abs v.(0) > 1e-10 || Float.abs v.(1) > 1e-10)

(* Adam variants *)

let test_scale_by_adam_step1 () =
  let tx = Vega.scale_by_adam ~b1:0.9 ~b2:0.999 ~eps:1e-8 () in
  let grad = vec [| 2.0 |] in
  let param = vec [| 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* mu = 0.1*2 = 0.2, nu = 0.001*4 = 0.004 bc1 = 0.1, bc2 = 0.001 m_hat =
     0.2/0.1 = 2.0, v_hat = 0.004/0.001 = 4.0 out = 2 / (sqrt(4) + 1e-8) = 2/2 =
     1.0 *)
  equal ~msg:"adam step 1" (array (float 1e-4)) [| 1.0 |] (to_arr upd)

let test_amsgrad () =
  let tx = Vega.scale_by_adam ~amsgrad:true () in
  let param = vec [| 0. |] in
  let st = Vega.init tx param in
  (* Step 1: large gradient → large v *)
  let _, st = Vega.update st ~grad:(vec [| 10.0 |]) ~param in
  (* Step 2: small gradient → v decreases, but v_max holds *)
  let _, st = Vega.update st ~grad:(vec [| 0.01 |]) ~param in
  let _, tensors = Vega.state_to_tensors st in
  let nu = to_arr tensors.(1) in
  let v_max = to_arr tensors.(2) in
  is_true ~msg:"v_max >= nu" (v_max.(0) >= nu.(0))

let test_nesterov_differs () =
  let tx_std = Vega.scale_by_adam () in
  let tx_nes = Vega.scale_by_adam ~nesterov:true () in
  let grad = vec [| 3.0; -1.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd_std, _ = Vega.update (Vega.init tx_std param) ~grad ~param in
  let upd_nes, _ = Vega.update (Vega.init tx_nes param) ~grad ~param in
  let a = to_arr upd_std and b = to_arr upd_nes in
  is_true ~msg:"nesterov differs from standard"
    (Float.abs (a.(0) -. b.(0)) > 1e-6)

(* Optimizer convergence *)

let test_lion_converges () = converges ~msg:"lion" ~tol:1.0 (Vega.lion lr01)
let test_radam_converges () = converges ~msg:"radam" ~tol:0.5 (Vega.radam lr01)

let test_adan_converges () =
  converges ~msg:"adan" ~tol:1.0 (Vega.adan (S.constant 0.05))

let test_lamb_converges () = converges ~msg:"lamb" ~tol:0.5 (Vega.lamb lr01)

let test_lars_converges () =
  converges ~msg:"lars" ~tol:0.5 (Vega.lars (S.constant 0.05))

let test_adafactor_converges () =
  (* Adafactor includes its own LR (eps_scale/sqrt(step) ≈ 0.001/sqrt(t)).
     Cumulative displacement after N steps is ~0.002*sqrt(N), so use small
     initial values and 2D shape to exercise the factored path. *)
  let tx = Vega.adafactor () in
  let param = ref (mat 2 2 [| 0.1; -0.05; 0.08; -0.03 |]) in
  let st = ref (Vega.init tx !param) in
  for _ = 1 to 5000 do
    let p, s = Vega.step !st ~grad:!param ~param:!param in
    param := p;
    st := s
  done;
  let v = to_arr !param in
  Array.iter
    (fun x -> is_true ~msg:"adafactor converges" (Float.abs x < 0.05))
    v

let test_adam_amsgrad_converges () =
  converges ~msg:"adam+amsgrad" ~tol:0.5
    (Vega.adam ~b1:0.9 ~b2:0.999 ~eps:1e-8 lr01)

(* Chain composition *)

let test_chain_associativity () =
  let a = Vega.scale 2.0 in
  let b = Vega.clip_by_value 5.0 in
  let c = Vega.scale 0.5 in
  let tx1 = Vega.chain [ Vega.chain [ a; b ]; c ] in
  let tx2 = Vega.chain [ a; b; c ] in
  let grad = vec [| 3.0; -4.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd1, _ = Vega.update (Vega.init tx1 param) ~grad ~param in
  let upd2, _ = Vega.update (Vega.init tx2 param) ~grad ~param in
  equal ~msg:"associative" (array eps) (to_arr upd1) (to_arr upd2)

let test_chain_identity () =
  let tx = Vega.scale_by_adam () in
  let tx_wrapped = Vega.chain [ tx ] in
  let grad = vec [| 1.0; -2.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd1, _ = Vega.update (Vega.init tx param) ~grad ~param in
  let upd2, _ = Vega.update (Vega.init tx_wrapped param) ~grad ~param in
  equal ~msg:"identity" (array eps) (to_arr upd1) (to_arr upd2)

let test_chain_ordering_matters () =
  let tx1 = Vega.chain [ Vega.clip_by_value 1.0; Vega.scale 10.0 ] in
  let tx2 = Vega.chain [ Vega.scale 10.0; Vega.clip_by_value 1.0 ] in
  let grad = vec [| 0.5 |] in
  let param = vec [| 0. |] in
  let upd1, _ = Vega.update (Vega.init tx1 param) ~grad ~param in
  let upd2, _ = Vega.update (Vega.init tx2 param) ~grad ~param in
  (* clip then scale: 0.5 → 0.5 → 5.0 ; scale then clip: 0.5 → 5.0 → 1.0 *)
  equal ~msg:"clip then scale" (array eps) [| 5.0 |] (to_arr upd1);
  equal ~msg:"scale then clip" (array eps) [| 1.0 |] (to_arr upd2)

(* apply_if_finite *)

let test_finite_passes_through () =
  let inner = Vega.scale 2.0 in
  let tx = Vega.apply_if_finite inner in
  let grad = vec [| 1.0; -0.5 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"pass-through" (array eps) [| 2.0; -1.0 |] (to_arr upd)

let test_nan_skipped () =
  let inner = Vega.scale 1.0 in
  let tx = Vega.apply_if_finite inner in
  let param = vec [| 0.; 0. |] in
  let grad = vec [| Float.nan; 1.0 |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  let v = to_arr upd in
  equal ~msg:"nan → zero[0]" (float 1e-10) 0.0 v.(0);
  equal ~msg:"nan → zero[1]" (float 1e-10) 0.0 v.(1)

let test_inf_skipped () =
  let inner = Vega.scale 1.0 in
  let tx = Vega.apply_if_finite inner in
  let param = vec [| 0.; 0. |] in
  let grad = vec [| Float.infinity; 1.0 |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  let v = to_arr upd in
  equal ~msg:"inf → zero" (float 1e-10) 0.0 v.(0)

let test_nonfinite_counter () =
  let inner = Vega.scale 1.0 in
  let tx = Vega.apply_if_finite inner in
  let param = vec [| 0. |] in
  let nan_grad = vec [| Float.nan |] in
  let st = Vega.init tx param in
  let _, st = Vega.update st ~grad:nan_grad ~param in
  let _, st = Vega.update st ~grad:nan_grad ~param in
  let _, tensors = Vega.state_to_tensors st in
  (* Last tensor is the counter *)
  let counter = Nx.item [] tensors.(Array.length tensors - 1) in
  equal ~msg:"2 consecutive non-finite" (float 1e-10) 2.0 counter

(* Serialization *)

let test_n_tensors () =
  equal ~msg:"sgd" int 0 (Vega.n_tensors (Vega.sgd lr01));
  equal ~msg:"sgd+momentum" int 1 (Vega.n_tensors (Vega.sgd ~momentum:0.9 lr01));
  equal ~msg:"adam" int 2 (Vega.n_tensors (Vega.adam lr01));
  equal ~msg:"adam+amsgrad" int 3
    (Vega.n_tensors
       (Vega.chain
          [
            Vega.scale_by_adam ~amsgrad:true ();
            Vega.scale_by_learning_rate lr01;
          ]));
  equal ~msg:"lion" int 1 (Vega.n_tensors (Vega.lion lr01));
  equal ~msg:"adan" int 4 (Vega.n_tensors (Vega.adan lr01));
  equal ~msg:"adafactor" int 2 (Vega.n_tensors (Vega.adafactor ()))

let test_serialization_round_trip () =
  let optimizers =
    [
      ("adam", Vega.adam lr01);
      ("adamw", Vega.adamw lr01);
      ("lion", Vega.lion lr01);
      ("radam", Vega.radam lr01);
    ]
  in
  List.iter
    (fun (name, tx) ->
      let param = vec [| 3.0; -2.0 |] in
      let grad = vec [| 1.0; -1.0 |] in
      (* Step once *)
      let st = Vega.init tx param in
      let _, st = Vega.update st ~grad ~param in
      (* Serialize and deserialize *)
      let count, tensors = Vega.state_to_tensors st in
      let st2 = Vega.state_of_tensors tx ~count tensors in
      (* Step again from both *)
      let upd1, _ = Vega.update st ~grad ~param in
      let upd2, _ = Vega.update st2 ~grad ~param in
      equal ~msg:(name ^ " round-trip") (array eps) (to_arr upd1) (to_arr upd2))
    optimizers

let test_wrong_tensor_count () =
  let tx = Vega.adam lr01 in
  raises_invalid_arg (fun () ->
      ignore (Vega.state_of_tensors tx ~count:1 [| vec [| 0. |] |]))

(* Validation *)

let test_validation () =
  raises_invalid_arg (fun () -> ignore (Vega.scale_by_lion ~b1:1.0 ()));
  raises_invalid_arg (fun () -> ignore (Vega.scale_by_lion ~b2:(-0.1) ()));
  raises_invalid_arg (fun () -> ignore (Vega.scale_by_adan ~b1:1.0 ()));
  raises_invalid_arg (fun () -> ignore (Vega.scale_by_adan ~b2:(-0.1) ()));
  raises_invalid_arg (fun () -> ignore (Vega.scale_by_adan ~b3:1.0 ()));
  raises_invalid_arg (fun () -> ignore (Vega.adan ~weight_decay:(-1.) lr01));
  raises_invalid_arg (fun () ->
      ignore (S.cosine_decay_restarts ~init_value:1. ~decay_steps:0 () 0));
  raises_invalid_arg (fun () ->
      ignore (S.one_cycle ~max_value:1. ~total_steps:0 () 0))

(* Entry point *)

let () =
  run "Vega"
    [
      group "schedule"
        [
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
          prop2 "constant is constant" (float 0.) nat (fun v step ->
              S.constant v step = v);
          prop' "cosine_decay bounded" nat (fun step ->
              let s = S.cosine_decay ~init_value:1.0 ~decay_steps:100 () in
              let v = s step in
              is_true ~msg:">=0" (v >= 0.0);
              is_true ~msg:"<=1" (v <= 1.0 +. 1e-10));
          prop' "one_cycle bounded" nat (fun step ->
              let s = S.one_cycle ~max_value:1.0 ~total_steps:100 () in
              let v = s step in
              is_true ~msg:">=0" (v >= 0.0);
              is_true ~msg:"<=max" (v <= 1.0 +. 1e-10));
          prop' "cosine_decay_restarts periodic" nat (fun step ->
              let period = 50 in
              let s =
                S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:period ()
              in
              let v1 = s step in
              let v2 = s (step + period) in
              equal ~msg:"periodic" (float 1e-10) v1 v2);
        ];
      group "primitives"
        [
          test "scale" test_scale;
          test "scale_by_schedule" test_scale_by_schedule;
          test "scale_by_learning_rate" test_scale_by_learning_rate;
          test "trace" test_trace;
          test "trace nesterov" test_trace_nesterov;
          test "add_decayed_weights" test_add_decayed_weights;
          test "add_decayed_weights scheduled"
            test_add_decayed_weights_scheduled;
          test "clip" test_clip;
          test "clip_by_norm" test_clip_by_norm;
          test "clip_by_norm no-op" test_clip_by_norm_no_op;
          test "trust_ratio" test_trust_ratio;
          test "trust_ratio zero param" test_trust_ratio_zero_param;
          test "centralize 2d" test_centralize_2d;
          test "centralize 1d" test_centralize_1d;
          test "add_noise" test_add_noise;
        ];
      group "adam"
        [
          test "step 1 exact" test_scale_by_adam_step1;
          test "amsgrad holds max" test_amsgrad;
          test "nesterov differs" test_nesterov_differs;
        ];
      group "optimizers"
        [
          test "lion converges" test_lion_converges;
          test "radam converges" test_radam_converges;
          test "adan converges" test_adan_converges;
          test "lamb converges" test_lamb_converges;
          test "lars converges" test_lars_converges;
          test "adafactor converges" test_adafactor_converges;
          test "adam+amsgrad converges" test_adam_amsgrad_converges;
        ];
      group "chain"
        [
          test "associativity" test_chain_associativity;
          test "identity" test_chain_identity;
          test "ordering matters" test_chain_ordering_matters;
        ];
      group "apply_if_finite"
        [
          test "finite passes through" test_finite_passes_through;
          test "nan skipped" test_nan_skipped;
          test "inf skipped" test_inf_skipped;
          test "counter tracks failures" test_nonfinite_counter;
        ];
      group "serialization"
        [
          test "n_tensors" test_n_tensors;
          test "round-trip" test_serialization_round_trip;
          test "wrong count raises" test_wrong_tensor_count;
        ];
      group "validation" [ test "invalid hyperparameters" test_validation ];
    ]
