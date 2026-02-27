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
  let s = Optim.Schedule.constant 0.01 in
  equal ~msg:"step 1" (float 1e-10) 0.01 (s 1);
  equal ~msg:"step 100" (float 1e-10) 0.01 (s 100);
  equal ~msg:"step 0" (float 1e-10) 0.01 (s 0)

let test_cosine_decay () =
  let s = Optim.Schedule.cosine_decay ~init_value:0.1 ~decay_steps:100 () in
  equal ~msg:"step 0" (float 1e-10) 0.1 (s 0);
  equal ~msg:"step 100 (fully decayed)" (float 1e-10) 0.0 (s 100);
  equal ~msg:"step 200 (past decay)" (float 1e-10) 0.0 (s 200);
  let mid = s 50 in
  equal ~msg:"step 50 (midpoint)" (float 1e-6) 0.05 mid

let test_cosine_decay_alpha () =
  let s =
    Optim.Schedule.cosine_decay ~init_value:0.1 ~decay_steps:100 ~alpha:0.1 ()
  in
  equal ~msg:"step 100 (alpha floor)" (float 1e-10) 0.01 (s 100)

let test_warmup_cosine () =
  let s =
    Optim.Schedule.warmup_cosine ~init_value:0.0 ~peak_value:0.01
      ~warmup_steps:100
  in
  equal ~msg:"step 0" (float 1e-10) 0.0 (s 0);
  equal ~msg:"step 100 (peak)" (float 1e-10) 0.01 (s 100);
  equal ~msg:"step 200 (past warmup)" (float 1e-10) 0.01 (s 200)

let test_warmup_linear () =
  let s =
    Optim.Schedule.warmup_linear ~init_value:0.0 ~peak_value:0.1
      ~warmup_steps:10
  in
  equal ~msg:"step 0" (float 1e-10) 0.0 (s 0);
  equal ~msg:"step 5 (midpoint)" (float 1e-10) 0.05 (s 5);
  equal ~msg:"step 10 (peak)" (float 1e-10) 0.1 (s 10);
  equal ~msg:"step 20 (past warmup)" (float 1e-10) 0.1 (s 20)

let test_exponential_decay () =
  let s =
    Optim.Schedule.exponential_decay ~init_value:1.0 ~decay_rate:0.5
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
    let updates, state' = Optim.step algo !s !p grads in
    p := Optim.apply_updates !p updates;
    s := state'
  done;
  !p

(* SGD *)

let test_sgd_basic () =
  let lr = Optim.Schedule.constant 0.1 in
  let algo = Optim.sgd ~lr () in
  let params = make_params [| 4.0; -3.0 |] in
  let result = train_steps algo params ~steps:1 in
  let v = get_values result in
  (* After 1 step: x - lr * x = x * (1 - lr) = x * 0.9 *)
  equal ~msg:"sgd[0] after 1 step" (float 1e-5) 3.6 v.(0);
  equal ~msg:"sgd[1] after 1 step" (float 1e-5) (-2.7) v.(1)

let test_sgd_converges () =
  let lr = Optim.Schedule.constant 0.1 in
  let algo = Optim.sgd ~lr () in
  let params = make_params [| 10.0; -8.0 |] in
  let result = train_steps algo params ~steps:100 in
  let v = get_values result in
  equal ~msg:"sgd converges[0]" (float 1e-3) 0.0 v.(0);
  equal ~msg:"sgd converges[1]" (float 1e-3) 0.0 v.(1)

let test_sgd_momentum () =
  let lr = Optim.Schedule.constant 0.01 in
  let algo = Optim.sgd ~lr ~momentum:0.9 () in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:200 in
  let v = get_values result in
  equal ~msg:"sgd+momentum converges[0]" (float 1e-2) 0.0 v.(0);
  equal ~msg:"sgd+momentum converges[1]" (float 1e-2) 0.0 v.(1)

let test_sgd_nesterov () =
  let lr = Optim.Schedule.constant 0.01 in
  let algo = Optim.sgd ~lr ~momentum:0.9 ~nesterov:true () in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:100 in
  let v = get_values result in
  equal ~msg:"sgd+nesterov converges[0]" (float 1e-2) 0.0 v.(0);
  equal ~msg:"sgd+nesterov converges[1]" (float 1e-2) 0.0 v.(1)

(* Adam *)

let test_adam_converges () =
  let lr = Optim.Schedule.constant 0.1 in
  let algo = Optim.adam ~lr () in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:200 in
  let v = get_values result in
  equal ~msg:"adam converges[0]" (float 0.1) 0.0 v.(0);
  equal ~msg:"adam converges[1]" (float 0.1) 0.0 v.(1)

(* AdamW *)

let test_adamw_converges () =
  let lr = Optim.Schedule.constant 0.1 in
  let algo = Optim.adamw ~lr () in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:200 in
  let v = get_values result in
  equal ~msg:"adamw converges[0]" (float 0.1) 0.0 v.(0);
  equal ~msg:"adamw converges[1]" (float 0.1) 0.0 v.(1)

(* RMSprop *)

let test_rmsprop_converges () =
  let lr = Optim.Schedule.constant 0.01 in
  let algo = Optim.rmsprop ~lr () in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:550 in
  let v = get_values result in
  equal ~msg:"rmsprop converges[0]" (float 0.1) 0.0 v.(0);
  equal ~msg:"rmsprop converges[1]" (float 0.1) 0.0 v.(1)

let test_rmsprop_momentum () =
  let lr = Optim.Schedule.constant 0.01 in
  let algo = Optim.rmsprop ~lr ~momentum:0.9 () in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:200 in
  let v = get_values result in
  equal ~msg:"rmsprop+momentum converges[0]" (float 0.1) 0.0 v.(0);
  equal ~msg:"rmsprop+momentum converges[1]" (float 0.1) 0.0 v.(1)

(* Adagrad *)

let test_adagrad_converges () =
  let lr = Optim.Schedule.constant 0.5 in
  let algo = Optim.adagrad ~lr () in
  let params = make_params [| 5.0; -3.0 |] in
  let result = train_steps algo params ~steps:200 in
  let v = get_values result in
  equal ~msg:"adagrad converges[0]" (float 0.2) 0.0 v.(0);
  equal ~msg:"adagrad converges[1]" (float 0.2) 0.0 v.(1)

let test_invalid_hyperparameters () =
  let lr = Optim.Schedule.constant 0.1 in
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Optim.sgd ~lr ~momentum:1.0 ()));
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Optim.adam ~lr ~eps:0.0 ()));
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Optim.adamw ~lr ~weight_decay:(-0.1) ()));
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Optim.rmsprop ~lr ~decay:1.0 ()));
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Optim.adagrad ~lr ~eps:0.0 ()))

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

(* apply_updates *)

let test_apply_updates () =
  let params = make_params [| 1.0; 2.0; 3.0 |] in
  let updates = make_params [| -0.1; 0.2; -0.3 |] in
  let result = Optim.apply_updates params updates in
  let v = get_values result in
  equal ~msg:"apply[0]" (float 1e-5) 0.9 v.(0);
  equal ~msg:"apply[1]" (float 1e-5) 2.2 v.(1);
  equal ~msg:"apply[2]" (float 1e-5) 2.7 v.(2)

(* Multi-parameter tree *)

let test_multi_param_tree () =
  let lr = Optim.Schedule.constant 0.1 in
  let algo = Optim.sgd ~lr () in
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
  let updates, _state' = Optim.step algo state params grads in
  let result = Optim.apply_updates params updates in
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
      group "apply_updates" [ test "element-wise add" test_apply_updates ];
      group "multi-param" [ test "tree optimizer step" test_multi_param_tree ];
    ]
