open Kaun

let train_xor () =
  (* Create RNG *)
  let rngs = Rune.Rng.key 42 in

  (* Define MLP model for XOR *)
  let model =
    Layer.sequential
      [
        Layer.linear ~in_features:2 ~out_features:4 ();
        Layer.tanh ();
        Layer.linear ~in_features:4 ~out_features:1 ();
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

  (* Create optimizer - using new Optax-style API *)
  let lr = Optimizer.Schedule.constant 0.1 in
  let optimizer = Optimizer.adam ~lr () in
  let opt_state = ref (Optimizer.init optimizer params) in

  (* Training loop *)
  let epochs = 500 in
  for epoch = 1 to epochs do
    (* Forward and backward pass *)
    let loss, grads =
      value_and_grad
        (fun params ->
          let predictions = Kaun.apply model params ~training:true x in
          Loss.binary_cross_entropy predictions y)
        params
    in

    (* Update weights *)
    let updates, new_state = Optimizer.step optimizer !opt_state params grads in
    opt_state := new_state;
    (* Apply updates to params in place *)
    Optimizer.apply_updates_inplace params updates;

    if epoch mod 500 = 0 then
      Printf.printf "Epoch %d: Loss = %.6f\n" epoch (Rune.item [] loss)
  done;

  (* Final predictions *)
  let predictions = Kaun.apply model params ~training:false x in
  print_endline "\nFinal predictions (should be close to [0; 1; 1; 0]):";
  Rune.print predictions

let () = train_xor ()
