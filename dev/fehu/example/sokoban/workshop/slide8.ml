(*
```ocaml
 *)
open Slide1
(* Compute KL divergence between two policies *)
let compute_kl_divergence old_probs new_probs =
  (* KL(old || new) = sum(old * log(old/new)) *)
  let ratio = Rune.div old_probs new_probs in
  let log_ratio = Rune.log ratio in
  let kl_terms = Rune.mul old_probs log_ratio in
  Rune.sum kl_terms ~axes:[|1|]  (* Sum over actions *)
(* Policy loss with KL penalty *)
let compute_policy_loss_with_kl old_probs new_probs
    advantages beta =
  let device = Rune.c in  
  (* Convert advantages to tensor *)
  let adv_tensor = Rune.create device Rune.float32 
    [|Array.length advantages|] advantages in  
  (* Policy gradient objective *)
  let ratios = Rune.div new_probs old_probs in
  let policy_obj = Rune.mul ratios adv_tensor in
  let policy_loss = Rune.neg (Rune.mean policy_obj) in  
  (* KL penalty *)
  let kl_div = compute_kl_divergence old_probs new_probs in
  let kl_penalty =
    Rune.mul (Rune.scalar device Rune.float32 beta) 
             (Rune.mean kl_div) in  
  (* Combined loss *)
  Rune.add policy_loss kl_penalty

(* Main function to demonstrate KL penalty *)
let main () =
  print_endline "=== Slide 8: KL Divergence Penalty ===";
  
  (* Create example probability distributions *)
  let old_probs = Rune.create device Rune.float32 [|3; 4|] 
    [|0.25; 0.25; 0.25; 0.25;  (* Uniform *)
      0.1; 0.6; 0.2; 0.1;      (* Peaked *)
      0.4; 0.3; 0.2; 0.1|] in   (* Skewed *)
  
  let new_probs = Rune.create device Rune.float32 [|3; 4|]
    [|0.2; 0.3; 0.3; 0.2;      (* Slightly different *)
      0.05; 0.8; 0.1; 0.05;     (* More peaked *)
      0.3; 0.3; 0.3; 0.1|] in   (* Less skewed *)
  
  let advantages = [|0.5; -0.3; 0.8|] in
  
  (* Compute loss with KL penalty *)
  let loss =
    compute_policy_loss_with_kl old_probs new_probs
      advantages 0.01 in
  
  Printf.printf "Policy loss with KL penalty: %.4f\n"
      (Rune.item [] loss);
  
  (* Compute KL divergence *)
  let kl_div = compute_kl_divergence old_probs new_probs in
  Printf.printf "KL divergences: [%.4f, %.4f, %.4f]\n" 
    (Rune.item [0] kl_div)
    (Rune.item [1] kl_div) 
    (Rune.item [2] kl_div);
  
  print_endline "KL penalty demonstration complete!"

(*
```
 *)