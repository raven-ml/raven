(*
```ocaml
 *)
open Slide1

let clip_ratio ratio epsilon =
  let min_ratio = 1.0 -. epsilon in
  let max_ratio = 1.0 +. epsilon in
  Rune.clip ratio ~min:min_ratio ~max:max_ratio

let compute_clipped_loss old_log_probs new_log_probs
    advantages epsilon =
  let device = Rune.c in  
  (* Compute policy ratios: exp(new_log_prob - old_log_prob) *)
  let log_ratios = Rune.sub new_log_probs old_log_probs in
  let ratios = Rune.exp log_ratios in
  let clipped_ratios = clip_ratio ratios epsilon in  
  (* Convert advantages to tensor *)
  let adv_tensor = Rune.create device Rune.float32 
    [|Array.length advantages|] advantages in
  let obj1 = Rune.mul ratios adv_tensor in
  let obj2 = Rune.mul clipped_ratios adv_tensor in  
  (* Take minimum (pessimistic bound) *)
  let clipped_obj = Rune.minimum obj1 obj2 in  
  (* Return negative for gradient ascent → descent *)
  Rune.neg (Rune.mean clipped_obj)

let main () =
  print_endline "=== Slide 7: Clipping for Stability ===";
  
  (* Create some example log probs and advantages *)
  let old_log_probs = Rune.create device Rune.float32 [|5|] 
    [|-2.3; -1.5; -0.8; -1.2; -2.0|] in
  let new_log_probs = Rune.create device Rune.float32 [|5|] 
    [|-2.1; -0.5; -0.9; -1.3; -1.8|] in
  let advantages = [|1.5; -0.8; 0.3; -0.2; 1.0|] in

  let loss =
    compute_clipped_loss old_log_probs new_log_probs
      advantages 0.2 in
  
  Printf.printf "Clipped loss: %.4f\n" (Rune.item [] loss);
  print_endline "Clipping demonstration complete!"
(*
```
 *)