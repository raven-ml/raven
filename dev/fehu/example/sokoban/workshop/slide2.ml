
(*
```ocaml
 *)
include Slide1
open Kaun

(* Numerically stable log_softmax computation *)
let log_softmax ~axis logits =
  let max_logits = Rune.max logits ~axes:[| axis |] ~keepdims:true in
  let shifted = Rune.sub logits max_logits in
  let exp_shifted = Rune.exp shifted in
  let sum_exp = Rune.sum exp_shifted ~axes:[| axis |] ~keepdims:true in
  Rune.sub shifted (Rune.log sum_exp)

(* Define a simple policy network for our grid world *)
let create_policy_network grid_size n_actions =
  (* Flatten grid input and process through MLP *)
  Layer.sequential [
    Layer.flatten ();  (* Flatten 2D grid to 1D *)
    Layer.linear ~in_features:(grid_size * grid_size)
                 ~out_features:32 ();
    Layer.relu ();
    Layer.linear ~in_features:32 ~out_features:16 ();
    Layer.relu ();
    Layer.linear ~in_features:16 ~out_features:n_actions ();
    (* No softmax here - we'll apply it when needed *)
  ]

(* Initialize the policy *)
let initialize_policy () =
  let rng = Rune.Rng.key 42 in  
  (* Create network *)
  let policy_net = create_policy_network 5 4 in  
  (* Initialize parameters *)
  let params =
    Kaun.init policy_net ~rngs:rng ~device ~dtype:Rune.float32 in  
  (policy_net, params) 

(* Sample action from policy *)
let sample_action policy_net params obs _rng =
  (* Get action logits *)
  let logits =
    Kaun.apply policy_net params ~training:false obs in  
  (* Convert to probabilities *)
  let probs = Rune.softmax ~axes:[|-1|] logits in  
  (* Sample from categorical distribution *)
  (* For workshop: we'll use argmax for simplicity *)
  let action_idx = Rune.argmax ~axis:(-1) probs in
  (* Convert to float tensor for environment *)
  let action = Rune.cast Rune.float32 action_idx in
  (* Also return log probability for REINFORCE *)
  let log_probs = Rune.log probs in
  (* Use take_along_axis to get the log prob
     of the selected action *)
  let action_idx_expanded = Rune.reshape [|1; 1|] action_idx in
  let action_log_prob =
    Rune.take_along_axis ~axis:(-1)
      action_idx_expanded log_probs in
  let action_log_prob = Rune.squeeze action_log_prob in
  (action, action_log_prob)  

let main () =
  print_endline "=== Slide 2: Policy Network ===";
  let policy_net, params = initialize_policy () in
  print_endline "Policy network initialized!";
  
  (* Create a test observation *)
  let test_obs = Rune.zeros device Rune.float32 [|5; 5|] in
  Rune.set_item [2; 2] 1.0 test_obs;  (* Agent at center *)
  
  (* Sample an action *)
  let rng = Rune.Rng.key 42 in
  let action, log_prob =
    sample_action policy_net params test_obs rng in
  
  Printf.printf "Sampled action: %.0f\n" (Rune.item [] action);
  Printf.printf "Log probability: %.4f\n" (Rune.item [] log_prob);
  print_endline "Policy network test complete!"

(*
```
 *)