
(*
```ocaml
 *)
include Slide1
open Kaun
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
  (* Initialize parameters with dummy input *)
  let dummy_obs = Rune.zeros device Rune.float32 [|5; 5|] in
  let params = Kaun.init policy_net ~rngs:rng ~device ~dtype:Rune.float32 in  
  (policy_net, params) 
(* Sample action from policy *)
let sample_action policy_net params obs rng =
  (* Get action logits *)
  let logits = Kaun.apply policy_net params ~training:false obs in  
  (* Convert to probabilities *)
  let probs = Rune.softmax ~axes:[|-1|] logits in  
  (* Sample from categorical distribution *)
  (* For workshop: we'll use argmax for simplicity *)
  let action = Rune.argmax ~axes:[|-1|] probs in  
  (* Also return log probability for REINFORCE *)
  let log_probs = Rune.log probs in
  let action_log_prob = Rune.gather log_probs action in  
  (action, action_log_prob)  
(*
```
 *)