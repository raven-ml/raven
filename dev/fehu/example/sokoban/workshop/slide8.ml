(*
```ocaml
 *)
include Slide7
(* Compute KL divergence between two policies *)
let compute_kl_divergence old_probs new_probs =
  (* KL(old || new) = sum(old * log(old/new)) *)
  let ratio = Rune.div old_probs new_probs in
  let log_ratio = Rune.log ratio in
  let kl_terms = Rune.mul old_probs log_ratio in
  Rune.sum kl_terms ~axes:[|1|]  (* Sum over actions *)
(* Policy loss with KL penalty *)
let compute_policy_loss_with_kl old_probs new_probs advantages beta =
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
  let kl_penalty = Rune.mul (Rune.scalar device Rune.float32 beta) 
                            (Rune.mean kl_div) in  
  (* Combined loss *)
  Rune.add policy_loss kl_penalty
(*
```
 *)