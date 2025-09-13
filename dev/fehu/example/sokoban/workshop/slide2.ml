
(*
```ocaml
 *)
open Slide1
open Kaun

(* Numerically stable log_softmax computation *)
let log_softmax ~axis logits =
  let max_logits =
     Rune.max logits ~axes:[| axis |] ~keepdims:true in
  let shifted = Rune.sub logits max_logits in
  let exp_shifted = Rune.exp shifted in
  let sum_exp =
    Rune.sum exp_shifted ~axes:[| axis |] ~keepdims:true in
  Rune.sub shifted (Rune.log sum_exp)

(* Define a simple policy network for our grid world *)
let create_policy_network grid_size n_actions =
  (* Flatten grid input and process through MLP *)
  Layer.sequential [
    (* Custom reshape to handle [1, 5, 5] -> [1, 25] *)
    {
      Kaun.init =
        (fun ~rngs:_ ~device:_ ~dtype:_ -> Kaun.Ptree.List []);
      Kaun.apply =
        (fun _ ~training:_ ?rngs:_ x ->
        (* Flatten from [batch, height, width]
           to [batch, height*width] *)
        let shape = Rune.shape x in
        let batch_size =
          if Array.length shape >= 1 then shape.(0) else 1 in
        Rune.reshape [|batch_size; grid_size * grid_size|] x
      );
    };
    Layer.linear ~in_features:(grid_size * grid_size)
                 ~out_features:32 ();
    Layer.relu ();
    Layer.linear ~in_features:32 ~out_features:16 ();
    Layer.relu ();
    Layer.linear ~in_features:16 ~out_features:n_actions ();
    (* No softmax here - we'll apply it when needed *)
  ]

let initialize_policy ?(grid_size=5) () =
  let rng = Rune.Rng.key 42 in
  let policy_net = create_policy_network grid_size 4 in
  let params =
    Kaun.init policy_net ~rngs:rng ~device ~dtype:Rune.float32 in
  (policy_net, params) 

let sample_action policy_net params obs _rng =
  (* Add batch dimension if needed -- a frequent source of bugs! *)
  let obs_batched =
    if Array.length (Rune.shape obs) = 2 then
      let shape = Rune.shape obs in
      Rune.reshape [|1; shape.(0); shape.(1)|] obs
    else obs in

  let logits =
    Kaun.apply policy_net params ~training:false obs_batched in
  (* Convert to action probabilities *)
  let probs = Rune.softmax ~axes:[|-1|] logits in
  (* Sample from categorical distribution *)
  (* Simple sampling: convert to CPU, sample, convert back;
     OK because the environment runs on CPU and is the bottleneck. *)
  (* Handle both [1,4] and [4] shapes *)
  let probs_flat =
    if Array.length (Rune.shape probs) = 2 then
      Rune.squeeze ~axes:[|0|] probs
    else probs in
  let probs_array = Rune.to_array probs_flat in
  let cumsum = Array.make 4 0.0 in
  cumsum.(0) <- probs_array.(0);
  for i = 1 to 3 do
    cumsum.(i) <- cumsum.(i-1) +. probs_array.(i)
  done;
  let r = Random.float 1.0 in
  let action_int = ref 0 in
  for i = 0 to 3 do
    if r > cumsum.(i) then action_int := i + 1
  done;
  (* Convert to float tensor for environment *)
  let action =
    Rune.scalar device Rune.float32 (float_of_int !action_int) in
  (* Also return log probability for REINFORCE *)
  let log_probs =
    Rune.log (Rune.add probs 
              (Rune.scalar device Rune.float32 1e-8)) in
  (* Get the log prob of selected action *)
  let log_probs_array =
    Rune.to_array (Rune.reshape [|4|] log_probs) in
  let action_log_prob =
    Rune.scalar device Rune.float32
      log_probs_array.(!action_int) in
  (action, action_log_prob)  

let main () =
  print_endline "=== Slide 2: Policy Network ===";
  let policy_net, params = initialize_policy () in
  print_endline "Policy network initialized!";
  
  (* Create a test observation with batch dimension *)
  let test_obs = Rune.zeros device Rune.float32 [|1; 5; 5|] in
  Rune.set_item [0; 2; 2] 1.0 test_obs;  (* Agent at center *)
  
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