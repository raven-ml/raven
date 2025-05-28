open Kaun

(* Test single Adam update to see if the fix works *)
let () =
  let rng = Rng.create ~seed:42 () in
  let dtype = Rune.float32 in
  let device = Rune.cpu in
  
  (* Simple linear layer *)
  let linear = Linear.init ~rng ~dtype ~device 2 1 in
  
  (* Simple input and target *)
  let input = Rune.create device dtype [| 1; 2 |] [| 1.; 2. |] in
  let target = Rune.create device dtype [| 1; 1 |] [| 1. |] in
  
  (* Create Adam optimizer *)
  let adam = Optimizer.adam ~lr:0.1 () in
  let optimizer = Optimizer.init ~lens:Linear.lens linear adam in
  
  (* Simple loss function *)
  let loss_fn model =
    let output = Linear.forward model input in
    let diff = Rune.sub output target in
    Rune.mean (Rune.mul diff diff)
  in
  
  print_endline "Computing initial loss...";
  let initial_loss = loss_fn linear in
  print_endline (Printf.sprintf "Initial loss: %f" (Rune.unsafe_get [] initial_loss));
  
  print_endline "Computing gradients...";
  let _loss, grad = value_and_grad ~lens:Linear.lens loss_fn linear in
  
  print_endline "Updating with Adam...";
  Optimizer.update optimizer grad;
  
  print_endline "Computing final loss...";
  let final_loss = loss_fn linear in
  print_endline (Printf.sprintf "Final loss: %f" (Rune.unsafe_get [] final_loss));
  
  if Rune.unsafe_get [] final_loss < Rune.unsafe_get [] initial_loss then
    print_endline "SUCCESS: Loss decreased"
  else
    print_endline "FAILURE: Loss did not decrease"