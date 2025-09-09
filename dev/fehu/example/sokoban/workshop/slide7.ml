(*
```ocaml
 *)
include Slide6
(* Helper function to implement clipping *)
let clip_ratio ratio epsilon =
  let min_ratio = 1.0 -. epsilon in
  let max_ratio = 1.0 +. epsilon in
  Rune.clip ratio ~min:min_ratio ~max:max_ratio
(* Compute clipped policy loss *)
let compute_clipped_loss old_log_probs new_log_probs advantages epsilon =
  let device = Rune.c in  
  (* Compute policy ratios: exp(new_log_prob - old_log_prob) *)
  let log_ratios = Rune.sub new_log_probs old_log_probs in
  let ratios = Rune.exp log_ratios in  
  (* Clip ratios *)
  let clipped_ratios = clip_ratio ratios epsilon in  
  (* Convert advantages to tensor *)
  let adv_tensor = Rune.create device Rune.float32 
    [|Array.length advantages|] advantages in  
  (* Compute both objectives *)
  let obj1 = Rune.mul ratios adv_tensor in
  let obj2 = Rune.mul clipped_ratios adv_tensor in  
  (* Take minimum (pessimistic bound) *)
  let clipped_obj = Rune.minimum obj1 obj2 in  
  (* Return negative for gradient ascent → descent *)
  Rune.neg (Rune.mean clipped_obj)
(*
```
 *)