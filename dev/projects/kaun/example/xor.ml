open Kaun

module Mlp = struct
  type ('layout, 'dev) t = {
    l1 : ('layout, 'dev) Linear.t; (* Input to hidden *)
    l2 : ('layout, 'dev) Linear.t; (* Hidden to output *)
  }

  let init ~rng ~dtype ~device input_size hidden_size output_size =
    let l1 = Linear.init ~rng ~dtype ~device input_size hidden_size in
    let l2 = Linear.init ~rng ~dtype ~device hidden_size output_size in
    { l1; l2 }

  let forward mlp input =
    input |> Linear.forward mlp.l1 |> Activation.tanh |> Linear.forward mlp.l2

  let params { l1; l2 } =
    Record [ ("l1", Linear.params l1); ("l2", Linear.params l2) ]

  let of_ptree = function
    | Record [ ("l1", p1); ("l2", p2) ] ->
        { l1 = Linear.of_ptree p1; l2 = Linear.of_ptree p2 }
    | _ -> invalid_arg "Mlp.of_ptree"

  let lens = { to_ptree = params; of_ptree }
end

let train_step model optimizer input target =
  let loss_fn model =
    let logits = Mlp.forward model input in
    let per_sample_loss = Loss.sigmoid_binary_cross_entropy logits target in
    Rune.mean per_sample_loss
  in
  let loss_value, grad = value_and_grad ~lens:Mlp.lens loss_fn model in
  Optimizer.update optimizer grad;
  loss_value

(* Training function *)
let train_mlp input y_true learning_rate epochs =
  (* Initialize random number generator *)
  let rng = Rng.create ~seed:42 () in
  let dtype = Rune.float32 in
  let device = Rune.cpu in

  (* Create MLP model *)
  let mlp = Mlp.init ~rng ~dtype ~device 2 4 1 in

  (* Create optimizer *)
  let sgd = Optimizer.adam ~lr:learning_rate () in
  (* Initialize optimizer with the model parameters *)
  let optimizer = Optimizer.init ~lens:Mlp.lens mlp sgd in

  (* Training loop *)
  for epoch = 1 to epochs do
    let loss = train_step mlp optimizer input y_true in
    (* Print loss every 100 epochs *)
    if epoch mod 100 = 0 then
      print_endline
        (Printf.sprintf "Epoch %d: Loss = %f" epoch (Rune.get [||] loss))
  done;

  (* Return the trained model *)
  mlp

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
  let logits = Mlp.forward trained_mlp input in
  let probs = Activation.sigmoid logits in
  print_endline "Predictions after training (should be close to [0; 1; 1; 0]):";
  Rune.print probs
