open Kaun

(* Define the MLP structure *)
type ('layout, 'dev) mlp = {
  l1 : ('layout, 'dev) Linear.params; (* Input to hidden *)
  l2 : ('layout, 'dev) Linear.params; (* Hidden to output *)
}

let mlp_to_params mlp = Linear.params mlp.l1 @ Linear.params mlp.l2

let params_to_mlp params =
  match params with
  | [ w1; b1; w2; b2 ] ->
      let l1 = Linear.{ w = w1; b = Some b1 } in
      let l2 = Linear.{ w = w2; b = Some b2 } in
      { l1; l2 }
  | _ ->
      failwith
        "Expected 4 parameters (2 weights and 2 biases), but got a different \
         number."

(* Forward pass *)
let forward mlp input =
  let h = Linear.forward mlp.l1 input |> Activation.tanh in
  Linear.forward mlp.l2 h (* Returns logits *)

(* Loss function: mean sigmoid binary cross-entropy *)
let loss mlp input target =
  let logits = forward mlp input in
  let per_sample_loss = Loss.sigmoid_binary_cross_entropy logits target in
  Rune.mean per_sample_loss

(* Training function *)
let train_mlp input y_true learning_rate epochs =
  (* Initialize random number generator *)
  let rng = Rng.create ~seed:42 () in
  let dtype = Rune.float32 in
  let device = Rune.cpu in

  (* Initialize linear layers *)
  let l1 = Linear.init ~rng ~dtype ~device 2 4 in
  (* 2 inputs -> 2 hidden *)
  let l2 = Linear.init ~rng ~dtype ~device 4 1 in
  (* 2 hidden -> 1 output *)
  let mlp = { l1; l2 } in

  (* Use a reference for the MLP parameters *)
  let mlp_ref = ref mlp in

  (* Training loop *)
  for epoch = 1 to epochs do
    let mlp = !mlp_ref in
    (* Compute loss and gradients *)
    let loss_func params = loss (params_to_mlp params) input y_true in
    let loss_value, grads =
      Rune.value_and_grads loss_func (mlp_to_params mlp)
    in
    let grad_mlp = params_to_mlp grads in
    (* Update parameters using Linear.update *)
    let updated_l1 = Linear.update ~lr:learning_rate mlp.l1 grad_mlp.l1 in
    let updated_l2 = Linear.update ~lr:learning_rate mlp.l2 grad_mlp.l2 in
    mlp_ref := { l1 = updated_l1; l2 = updated_l2 };
    (* Print loss every 100 epochs *)
    if epoch mod 100 = 0 then
      print_endline
        (Printf.sprintf "Epoch %d: Loss = %f" epoch (Rune.get [||] loss_value))
  done;
  !mlp_ref

(* Main execution *)
let () =
  (* XOR input data: [4 samples, 2 features] *)
  let input =
    Rune.create Rune.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]
  in
  (* XOR target data: [4 samples, 1 output] *)
  let y_true = Rune.create Rune.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in
  let learning_rate = 0.1 in
  let epochs = 2000 in

  (* Train the MLP *)
  let trained_mlp = train_mlp input y_true learning_rate epochs in

  (* Make predictions *)
  let logits = forward trained_mlp input in
  let probs = Activation.sigmoid logits in
  print_endline "Predictions after training (should be close to [0; 1; 1; 0]):";
  Rune.print probs
