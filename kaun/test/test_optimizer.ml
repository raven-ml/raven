open Kaun

let eps = 1e-3

(* Simple quadratic function for testing: f(x) = 0.5 * x^2 *)
let quadratic_loss params =
  match params with
  | Ptree.Tensor x ->
      let dt = Rune.dtype x in
      Rune.(mul (scalar dt 0.5) (sum (mul x x)))
  | _ -> failwith "Expected tensor parameter"

(* Test that optimizer reduces loss *)
let test_optimizer_reduces_loss optimizer_fn name () =
  (* Initialize parameter *)
  let x = Rune.create Rune.float32 [| 2 |] [| 10.; -5. |] in
  let params = Ptree.tensor x in

  (* Create optimizer *)
  let optimizer : 'a Optimizer.gradient_transformation = optimizer_fn () in
  let opt_state = ref (optimizer.Optimizer.init params) in

  (* Initial loss *)
  let initial_loss = quadratic_loss params |> fun x -> Rune.item [] x in

  (* Training loop *)
  for _ = 1 to 200 do
    let loss, grads = value_and_grad quadratic_loss params in
    let updates, new_state =
      optimizer.Optimizer.update !opt_state params grads
    in
    opt_state := new_state;
    Optimizer.apply_updates_inplace params updates;
    ignore loss
  done;

  (* Final loss *)
  let final_loss = quadratic_loss params |> fun x -> Rune.item [] x in

  (* Check that loss decreased *)
  Alcotest.(check bool)
    (Printf.sprintf "%s reduces loss" name)
    true
    (final_loss < initial_loss *. 0.1);

  (* Loss should reduce by at least 90% *)

  (* Check that we're close to optimum (x = 0) *)
  let final_x =
    match params with
    | Ptree.Tensor x -> Rune.to_bigarray x
    | _ -> failwith "Expected tensor"
  in
  let x_norm =
    let sum = ref 0. in
    for i = 0 to 1 do
      let v = Bigarray.Genarray.get final_x [| i |] in
      sum := !sum +. (v *. v)
    done;
    sqrt !sum
  in
  Alcotest.(check (float 0.2)) (* More lenient for convergence *)
    (Printf.sprintf "%s converges to optimum" name)
    0.0 x_norm

(* Test XOR problem convergence *)
let test_xor_convergence () =
  let rngs = Rune.Rng.key 42 in

  (* Define MLP model for XOR *)
  let model =
    Layer.sequential
      [
        Layer.linear ~in_features:2 ~out_features:8 ();
        Layer.relu ();
        Layer.linear ~in_features:8 ~out_features:1 ();
        Layer.sigmoid ();
      ]
  in

  (* XOR dataset *)
  let x =
    Rune.create Rune.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]
  in
  let y = Rune.create Rune.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in

  (* Initialize model parameters *)
  let params = Kaun.init model ~rngs ~dtype:Rune.float32 in

  (* Create optimizer *)
  let optimizer = Optimizer.adam ~lr:0.01 () in
  let opt_state = ref (optimizer.Optimizer.init params) in

  (* Training loop *)
  let initial_loss = ref 0. in
  let final_loss = ref 0. in

  for epoch = 1 to 500 do
    (* Forward and backward pass *)
    let loss, grads =
      value_and_grad
        (fun params ->
          let predictions = Kaun.apply model params ~training:true x in
          Loss.binary_cross_entropy predictions y)
        params
    in

    if epoch = 1 then initial_loss := Rune.item [] loss;
    if epoch = 500 then final_loss := Rune.item [] loss;

    (* Update weights *)
    let updates, new_state =
      optimizer.Optimizer.update !opt_state params grads
    in
    opt_state := new_state;
    Optimizer.apply_updates_inplace params updates
  done;

  (* Check that loss decreased significantly *)
  Alcotest.(check bool)
    "XOR loss decreases" true
    (!final_loss < !initial_loss *. 0.8);

  (* More lenient - 20% reduction *)

  (* Check predictions *)
  let predictions = Kaun.apply model params ~training:false x in
  let pred_array = Rune.to_bigarray predictions in

  (* Check each prediction *)
  let expected = [| 0.; 1.; 1.; 0. |] in
  for i = 0 to 3 do
    let pred = Bigarray.Genarray.get pred_array [| i; 0 |] in
    let exp = expected.(i) in
    let error = abs_float (pred -. exp) in
    Alcotest.(check bool)
      (Printf.sprintf "XOR prediction %d (pred=%.3f, expected=%.1f)" i pred exp)
      true (error < 0.3)
    (* Allow some error margin *)
  done

(* Test that different optimizers produce different trajectories *)
let test_optimizer_differences () =
  let test_optimizer opt_fn =
    let x = Rune.create Rune.float32 [| 2 |] [| 10.; -5. |] in
    let params = Ptree.tensor x in
    let optimizer = opt_fn () in
    let opt_state = ref (optimizer.Optimizer.init params) in

    (* Take one step *)
    let _, grads = value_and_grad quadratic_loss params in
    let updates, new_state =
      optimizer.Optimizer.update !opt_state params grads
    in
    opt_state := new_state;
    Optimizer.apply_updates_inplace params updates;

    (* Return parameter values *)
    match params with
    | Ptree.Tensor x ->
        let arr = Rune.to_bigarray x in
        [|
          Bigarray.Genarray.get arr [| 0 |]; Bigarray.Genarray.get arr [| 1 |];
        |]
    | _ -> failwith "Expected tensor"
  in

  let sgd_params = test_optimizer (fun () -> Optimizer.sgd ~lr:0.1 ()) in
  let adam_params = test_optimizer (fun () -> Optimizer.adam ~lr:0.1 ()) in

  (* Check that they produce different results (they should due to different
     update rules) *)
  let diff =
    abs_float (sgd_params.(0) -. adam_params.(0))
    +. abs_float (sgd_params.(1) -. adam_params.(1))
  in

  Alcotest.(check bool)
    "SGD and Adam produce different updates" true (diff > 0.01)

(* Test optimizer state persistence *)
let test_optimizer_state_persistence () =
  let x = Rune.create Rune.float32 [| 2 |] [| 10.; -5. |] in
  let params = Ptree.tensor x in

  (* Create Adam optimizer (which has internal state) *)
  let optimizer = Optimizer.adam ~lr:0.1 () in
  let opt_state = ref (optimizer.Optimizer.init params) in

  (* Take multiple steps and check that momentum is building up *)
  let first_update_norm = ref 0. in
  let fifth_update_norm = ref 0. in

  for i = 1 to 5 do
    let _, grads = value_and_grad quadratic_loss params in
    let updates, new_state =
      optimizer.Optimizer.update !opt_state params grads
    in
    opt_state := new_state;

    (* Calculate update norm *)
    let update_norm =
      match updates with
      | Ptree.Tensor u ->
          let arr = Rune.to_bigarray u in
          let n = ref 0. in
          for j = 0 to 1 do
            let v = Bigarray.Genarray.get arr [| j |] in
            n := !n +. (v *. v)
          done;
          sqrt !n
      | _ -> failwith "Expected tensor"
    in

    if i = 1 then first_update_norm := update_norm;
    if i = 5 then fifth_update_norm := update_norm;

    Optimizer.apply_updates_inplace params updates
  done;

  (* For Adam with momentum, later updates should be different from first due to
     accumulated momentum *)
  Alcotest.(check bool)
    "Adam maintains state across updates" true
    (abs_float (!fifth_update_norm -. !first_update_norm) > 0.0001)

(* Test learning rate scheduling *)
let test_learning_rate_schedule () =
  let x = Rune.create Rune.float32 [| 2 |] [| 10.; -5. |] in
  let params = Ptree.tensor x in

  (* Create scheduler that decreases learning rate *)
  let schedule step = if step <= 5 then 0.1 else 0.01 in

  let optimizer =
    Optimizer.chain
      [
        Optimizer.scale_by_adam ();
        Optimizer.scale_by_schedule schedule;
        Optimizer.scale_by_neg_one ();
      ]
  in

  let opt_state = ref (optimizer.Optimizer.init params) in

  (* Collect update magnitudes *)
  let early_update_norm = ref 0. in
  let late_update_norm = ref 0. in

  for i = 1 to 10 do
    let _, grads = value_and_grad quadratic_loss params in
    let updates, new_state =
      optimizer.Optimizer.update !opt_state params grads
    in
    opt_state := new_state;

    let update_norm =
      match updates with
      | Ptree.Tensor u ->
          let arr = Rune.to_bigarray u in
          abs_float (Bigarray.Genarray.get arr [| 0 |])
      | _ -> failwith "Expected tensor"
    in

    if i = 3 then early_update_norm := update_norm;
    if i = 8 then late_update_norm := update_norm;

    Optimizer.apply_updates_inplace params updates
  done;

  (* Late updates should be smaller due to reduced learning rate *)
  Alcotest.(check bool)
    "Learning rate schedule reduces update magnitude" true
    (!late_update_norm < !early_update_norm *. 0.5)

let test_clip_by_global_norm_zero_gradients () =
  let zero_tensor = Rune.zeros Rune.float32 [| 2 |] in
  let params = Ptree.tensor zero_tensor in
  let grads = Ptree.tensor (Rune.zeros_like zero_tensor) in
  let transform = Optimizer.clip_by_global_norm 1.0 in
  let state = transform.Optimizer.init params in
  let updates, _ = transform.Optimizer.update state params grads in
  match updates with
  | Ptree.Tensor t ->
      let arr = Rune.to_bigarray t in
      for i = 0 to 1 do
        let value = Bigarray.Genarray.get arr [| i |] in
        Alcotest.(check bool)
          "value is not NaN" false (Float.is_nan value);
        Alcotest.(check bool)
          "value remains zero" true (abs_float value < 1e-9)
      done
  | _ -> Alcotest.fail "Expected tensor updates"

let test_clip_by_global_norm_empty_tree () =
  let params = Ptree.list_of [] in
  let grads = Ptree.list_of [] in
  let transform = Optimizer.clip_by_global_norm 1.0 in
  let state = transform.Optimizer.init params in
  let updates, _ = transform.Optimizer.update state params grads in
  match updates with
  | Ptree.List [] -> ()
  | _ -> Alcotest.fail "Expected empty list updates"

(* Test gradient clipping *)
let test_gradient_clipping () =
  let x = Rune.create Rune.float32 [| 2 |] [| 100.; -50. |] in
  let params = Ptree.tensor x in

  (* Optimizer with gradient clipping *)
  let optimizer =
    Optimizer.chain
      [
        Optimizer.clip 1.0;
        (* Clip gradients to [-1, 1] *)
        Optimizer.scale_by_neg_one ();
        Optimizer.scale 0.1;
      ]
  in

  let opt_state = ref (optimizer.Optimizer.init params) in
  let _, grads = value_and_grad quadratic_loss params in

  (* Gradients should be [100, -50] before clipping *)
  let updates, _ = optimizer.Optimizer.update !opt_state params grads in

  match updates with
  | Ptree.Tensor u ->
      let arr = Rune.to_bigarray u in
      let update0 = Bigarray.Genarray.get arr [| 0 |] in
      let update1 = Bigarray.Genarray.get arr [| 1 |] in

      (* Updates should be clipped: -0.1 * clip(100, 1) = -0.1 and -0.1 *
         clip(-50, 1) = 0.1 *)
      Alcotest.(check (float eps)) "First update clipped" (-0.1) update0;
      Alcotest.(check (float eps)) "Second update clipped" 0.1 update1
  | _ -> failwith "Expected tensor"

let () =
  let open Alcotest in
  run "Optimizer tests"
    [
      ( "Basic optimization",
        [
          test_case "SGD reduces loss" `Quick
            (test_optimizer_reduces_loss
               (fun () -> Optimizer.sgd ~lr:0.1 ())
               "SGD");
          test_case "Adam reduces loss" `Quick
            (test_optimizer_reduces_loss
               (fun () -> Optimizer.adam ~lr:0.1 ())
               "Adam");
          test_case "AdamW reduces loss" `Quick
            (test_optimizer_reduces_loss
               (fun () -> Optimizer.adamw ~lr:0.1 ())
               "AdamW");
          test_case "RMSprop reduces loss" `Quick
            (test_optimizer_reduces_loss
               (fun () -> Optimizer.rmsprop ~lr:0.1 ())
               "RMSprop");
          test_case "Adagrad reduces loss" `Quick
            (test_optimizer_reduces_loss
               (fun () -> Optimizer.adagrad ~lr:1.0 ())
               "Adagrad");
        ] );
      ( "Complex problems",
        [ test_case "XOR convergence" `Slow test_xor_convergence ] );
      ( "Optimizer behavior",
        [
          test_case "Different optimizers produce different updates" `Quick
            test_optimizer_differences;
          test_case "Optimizer state persistence" `Quick
            test_optimizer_state_persistence;
          test_case "Learning rate scheduling" `Quick
            test_learning_rate_schedule;
          test_case "Global norm clipping handles zero gradients" `Quick
            test_clip_by_global_norm_zero_gradients;
          test_case "Global norm clipping handles empty parameter tree" `Quick
            test_clip_by_global_norm_empty_tree;
          test_case "Gradient clipping" `Quick test_gradient_clipping;
        ] );
    ]
