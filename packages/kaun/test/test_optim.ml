(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Optim = Kaun.Optim
module Ptree = Kaun.Ptree
module Grad = Kaun.Grad

(* Schedules *)

let test_constant_schedule () =
  let s = Vega.Schedule.constant 0.01 in
  equal ~msg:"step 1" (float 1e-10) 0.01 (s 1);
  equal ~msg:"step 100" (float 1e-10) 0.01 (s 100);
  equal ~msg:"step 0" (float 1e-10) 0.01 (s 0)

let test_cosine_decay () =
  let s = Vega.Schedule.cosine_decay ~init_value:0.1 ~decay_steps:100 () in
  equal ~msg:"step 0" (float 1e-10) 0.1 (s 0);
  equal ~msg:"step 100 (fully decayed)" (float 1e-10) 0.0 (s 100);
  equal ~msg:"step 200 (past decay)" (float 1e-10) 0.0 (s 200);
  let mid = s 50 in
  equal ~msg:"step 50 (midpoint)" (float 1e-6) 0.05 mid

let test_cosine_decay_alpha () =
  let s =
    Vega.Schedule.cosine_decay ~init_value:0.1 ~decay_steps:100 ~alpha:0.1 ()
  in
  equal ~msg:"step 100 (alpha floor)" (float 1e-10) 0.01 (s 100)

let test_warmup_cosine () =
  let s =
    Vega.Schedule.warmup_cosine ~init_value:0.0 ~peak_value:0.01
      ~warmup_steps:100
  in
  equal ~msg:"step 0" (float 1e-10) 0.0 (s 0);
  equal ~msg:"step 100 (peak)" (float 1e-10) 0.01 (s 100);
  equal ~msg:"step 200 (past warmup)" (float 1e-10) 0.01 (s 200)

let test_warmup_linear () =
  let s = Vega.Schedule.linear ~init_value:0.0 ~end_value:0.1 ~steps:10 in
  equal ~msg:"step 0" (float 1e-10) 0.0 (s 0);
  equal ~msg:"step 5 (midpoint)" (float 1e-10) 0.05 (s 5);
  equal ~msg:"step 10 (peak)" (float 1e-10) 0.1 (s 10);
  equal ~msg:"step 20 (past warmup)" (float 1e-10) 0.1 (s 20)

let test_exponential_decay () =
  let s =
    Vega.Schedule.exponential_decay ~init_value:1.0 ~decay_rate:0.5
      ~decay_steps:10
  in
  equal ~msg:"step 0" (float 1e-10) 1.0 (s 0);
  equal ~msg:"step 10" (float 1e-6) 0.5 (s 10);
  equal ~msg:"step 20" (float 1e-6) 0.25 (s 20)

(* Helpers *)

let quadratic_loss params =
  (* f(x) = 0.5 * sum(x^2), grad = x *)
  let (Ptree.P t) = Ptree.as_tensor_exn params in
  let t = Ptree.Tensor.to_typed_exn Nx.float32 (Ptree.P t) in
  Nx.mul (Nx.scalar Nx.float32 0.5) (Nx.sum (Nx.mul t t))

let make_params values =
  Ptree.tensor (Nx.create Nx.float32 [| Array.length values |] values)

let get_values params =
  let (Ptree.P t) = Ptree.as_tensor_exn params in
  let t = Ptree.Tensor.to_typed_exn Nx.float32 (Ptree.P t) in
  Nx.to_array (Nx.reshape [| -1 |] t)

let train_steps algo params ~steps =
  let state = Optim.init algo params in
  let p = ref params in
  let s = ref state in
  for _ = 1 to steps do
    let _loss, grads = Grad.value_and_grad quadratic_loss !p in
    let new_params, state' = Optim.step !s !p grads in
    p := new_params;
    s := state'
  done;
  !p

(* SGD *)

let test_sgd_basic () =
  let lr = Vega.Schedule.constant 0.1 in
  let algo = Vega.sgd lr in
  let params = make_params [| 4.0; -3.0 |] in
  let result = train_steps algo params ~steps:1 in
  let v = get_values result in
  (* After 1 step: x - lr * x = x * (1 - lr) = x * 0.9 *)
  equal ~msg:"sgd[0] after 1 step" (float 1e-5) 3.6 v.(0);
  equal ~msg:"sgd[1] after 1 step" (float 1e-5) (-2.7) v.(1)

let test_sgd_converges () =
  let lr = Vega.Schedule.constant 0.1 in
  let algo = Vega.sgd lr in
  let params = make_params [| 10.0; -8.0 |] in
  let result = train_steps algo params ~steps:100 in
  let v = get_values result in
  equal ~msg:"sgd converges[0]" (float 1e-3) 0.0 v.(0);
  equal ~msg:"sgd converges[1]" (float 1e-3) 0.0 v.(1)

let test_sgd_momentum () =
  let lr = Vega.Schedule.constant 0.01 in
  let algo = Vega.sgd ~momentum:0.9 lr in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:100 in
  let v = get_values result in
  equal ~msg:"sgd+momentum converges[0]" (float 0.1) 0.0 v.(0);
  equal ~msg:"sgd+momentum converges[1]" (float 0.1) 0.0 v.(1)

let test_sgd_nesterov () =
  let lr = Vega.Schedule.constant 0.01 in
  let algo = Vega.sgd ~momentum:0.9 ~nesterov:true lr in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:100 in
  let v = get_values result in
  equal ~msg:"sgd+nesterov converges[0]" (float 1e-2) 0.0 v.(0);
  equal ~msg:"sgd+nesterov converges[1]" (float 1e-2) 0.0 v.(1)

(* Adam *)

let test_adam_converges () =
  let lr = Vega.Schedule.constant 0.1 in
  let algo = Vega.adam lr in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:100 in
  let v = get_values result in
  equal ~msg:"adam converges[0]" (float 0.5) 0.0 v.(0);
  equal ~msg:"adam converges[1]" (float 0.5) 0.0 v.(1)

(* AdamW *)

let test_adamw_converges () =
  let lr = Vega.Schedule.constant 0.1 in
  let algo = Vega.adamw lr in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:100 in
  let v = get_values result in
  equal ~msg:"adamw converges[0]" (float 0.5) 0.0 v.(0);
  equal ~msg:"adamw converges[1]" (float 0.5) 0.0 v.(1)

(* RMSprop *)

let test_rmsprop_converges () =
  let lr = Vega.Schedule.constant 0.1 in
  let algo = Vega.rmsprop lr in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:100 in
  let v = get_values result in
  equal ~msg:"rmsprop converges[0]" (float 0.5) 0.0 v.(0);
  equal ~msg:"rmsprop converges[1]" (float 0.5) 0.0 v.(1)

let test_rmsprop_momentum () =
  let lr = Vega.Schedule.constant 0.01 in
  let algo = Vega.rmsprop ~momentum:0.9 lr in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:100 in
  let v = get_values result in
  equal ~msg:"rmsprop+momentum converges[0]" (float 0.5) 0.0 v.(0);
  equal ~msg:"rmsprop+momentum converges[1]" (float 0.5) 0.0 v.(1)

(* Adagrad *)

let test_adagrad_converges () =
  let lr = Vega.Schedule.constant 0.5 in
  let algo = Vega.adagrad lr in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:100 in
  let v = get_values result in
  equal ~msg:"adagrad converges[0]" (float 0.5) 0.0 v.(0);
  equal ~msg:"adagrad converges[1]" (float 0.5) 0.0 v.(1)

let test_invalid_hyperparameters () =
  let lr = Vega.Schedule.constant 0.1 in
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Vega.sgd ~momentum:1.0 lr));
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Vega.adam ~eps:0.0 lr));
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Vega.adamw ~weight_decay:(-0.1) lr));
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Vega.rmsprop ~decay:1.0 lr));
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Vega.adagrad ~eps:0.0 lr))

(* Gradient utilities *)

let test_global_norm () =
  let t =
    Ptree.dict
      [
        ("a", Ptree.tensor (Nx.create Nx.float32 [| 2 |] [| 3.0; 4.0 |]));
        ("b", Ptree.tensor (Nx.create Nx.float32 [| 1 |] [| 0.0 |]));
      ]
  in
  (* sqrt(9 + 16 + 0) = 5 *)
  equal ~msg:"global_norm" (float 1e-5) 5.0 (Optim.global_norm t)

let test_clip_by_global_norm () =
  let t = Ptree.tensor (Nx.create Nx.float32 [| 2 |] [| 3.0; 4.0 |]) in
  (* norm = 5, clip to 2.5 → scale by 0.5 *)
  let clipped = Optim.clip_by_global_norm 2.5 t in
  let v = get_values clipped in
  equal ~msg:"clipped[0]" (float 1e-5) 1.5 v.(0);
  equal ~msg:"clipped[1]" (float 1e-5) 2.0 v.(1)

let test_clip_no_op () =
  let t = Ptree.tensor (Nx.create Nx.float32 [| 2 |] [| 1.0; 1.0 |]) in
  (* norm = sqrt(2) ~ 1.41, max_norm = 5.0 → no clipping *)
  let clipped = Optim.clip_by_global_norm 5.0 t in
  let v = get_values clipped in
  equal ~msg:"no clip[0]" (float 1e-5) 1.0 v.(0);
  equal ~msg:"no clip[1]" (float 1e-5) 1.0 v.(1)

(* Multi-parameter tree *)

let test_multi_param_tree () =
  let lr = Vega.Schedule.constant 0.1 in
  let algo = Vega.sgd lr in
  let params =
    Ptree.dict
      [
        ("w", Ptree.tensor (Nx.create Nx.float32 [| 2 |] [| 4.0; -2.0 |]));
        ("b", Ptree.tensor (Nx.create Nx.float32 [| 1 |] [| 1.0 |]));
      ]
  in
  let f p =
    let fields = Ptree.Dict.fields_exn p in
    let w = Ptree.Dict.get_tensor_exn fields ~name:"w" Nx.float32 in
    let b = Ptree.Dict.get_tensor_exn fields ~name:"b" Nx.float32 in
    Nx.add
      (Nx.mul (Nx.scalar Nx.float32 0.5) (Nx.sum (Nx.mul w w)))
      (Nx.mul (Nx.scalar Nx.float32 0.5) (Nx.sum (Nx.mul b b)))
  in
  let state = Optim.init algo params in
  let _loss, grads = Grad.value_and_grad f params in
  let result, _state' = Optim.step state params grads in
  let fields = Ptree.Dict.fields_exn result in
  let w = Ptree.Dict.get_tensor_exn fields ~name:"w" Nx.float32 in
  let b = Ptree.Dict.get_tensor_exn fields ~name:"b" Nx.float32 in
  (* w_new = w - lr * w = w * 0.9 *)
  equal ~msg:"w[0]" (float 1e-5) 3.6 (Nx.item [ 0 ] w);
  equal ~msg:"w[1]" (float 1e-5) (-1.8) (Nx.item [ 1 ] w);
  equal ~msg:"b[0]" (float 1e-5) 0.9 (Nx.item [ 0 ] b)

let () =
  run "Kaun.Optim"
    [
      group "schedules"
        [
          test "constant" test_constant_schedule;
          test "cosine decay" test_cosine_decay;
          test "cosine decay alpha" test_cosine_decay_alpha;
          test "warmup cosine" test_warmup_cosine;
          test "warmup linear" test_warmup_linear;
          test "exponential decay" test_exponential_decay;
        ];
      group "sgd"
        [
          test "basic step" test_sgd_basic;
          test "converges" test_sgd_converges;
          test "momentum" test_sgd_momentum;
          test "nesterov" test_sgd_nesterov;
        ];
      group "adam" [ test "converges" test_adam_converges ];
      group "adamw" [ test "converges" test_adamw_converges ];
      group "rmsprop"
        [
          test "converges" test_rmsprop_converges;
          test "momentum" test_rmsprop_momentum;
        ];
      group "adagrad" [ test "converges" test_adagrad_converges ];
      group "validation"
        [ test "invalid hyperparameters" test_invalid_hyperparameters ];
      group "gradient utilities"
        [
          test "global_norm" test_global_norm;
          test "clip_by_global_norm" test_clip_by_global_norm;
          test "clip no-op" test_clip_no_op;
        ];
      group "multi-param" [ test "tree optimizer step" test_multi_param_tree ];
    ]
