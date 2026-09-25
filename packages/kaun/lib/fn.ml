(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every function lowers to Nx operations with reverse- and forward-mode rules
   in Rune (add, mul, div, exp, log, tanh, erf, maximum, where, reduce_max,
   reduce_sum), so they all differentiate. *)

(* A float constant in [x]'s element type, for the [_s] scalar operations. *)
let const x v = Nx_dtype.of_float (Nx.dtype x) v
let relu = Nx.relu
let sigmoid = Nx.sigmoid
let tanh = Nx.tanh

let leaky_relu ?(negative_slope = 0.01) x =
  Nx.where
    (Nx.greater_s x (const x 0.0))
    x
    (Nx.mul_s x (const x negative_slope))

let inv_sqrt2 = 0.7071067811865476
let sqrt_2_over_pi = 0.7978845608028654

let gelu x =
  let gauss_cdf =
    Nx.add_s (Nx.erf (Nx.mul_s x (const x inv_sqrt2))) (const x 1.0)
  in
  Nx.mul_s (Nx.mul x gauss_cdf) (const x 0.5)

let gelu_approx x =
  let x3 = Nx.mul x (Nx.mul x x) in
  let inner =
    Nx.mul_s
      (Nx.add x (Nx.mul_s x3 (const x 0.044715)))
      (const x sqrt_2_over_pi)
  in
  Nx.mul_s (Nx.mul x (Nx.add_s (Nx.tanh inner) (const x 1.0))) (const x 0.5)

let silu x = Nx.mul x (Nx.sigmoid x)

(* softplus(x) = max(x, 0) + log(1 + exp(-|x|)): [exp] sees a non-positive
   argument on both sides of 0, so large inputs cannot overflow. *)
let softplus x =
  Nx.add (Nx.relu x)
    (Nx.log (Nx.add_s (Nx.exp (Nx.neg (Nx.abs x))) (const x 1.0)))

let softmax ?(axis = -1) x = Nx.softmax ~axes:[ axis ] x
let log_softmax ?(axis = -1) x = Nx.log_softmax ~axes:[ axis ] x

(* Sampling masks *)

let last_axis ~fn logits =
  let shape = Nx.shape logits in
  let rank = Array.length shape in
  if rank = 0 then
    Printf.ksprintf invalid_arg "Fn.%s: logits must not be a scalar" fn;
  (Array.sub shape 0 (rank - 1), shape.(rank - 1))

(* A per-row parameter, of shape [||] or the logits' leading shape, as a column
   over the vocabulary axis. *)
let column ~fn ~lead param =
  let ps = Nx.shape param in
  if ps <> [||] && ps <> lead then
    Printf.ksprintf invalid_arg
      "Fn.%s: the parameter must be a scalar or have the logits' leading shape"
      fn;
  let one = Array.append lead [| 1 |] in
  if ps = [||] then
    Nx.broadcast_to one (Nx.reshape (Array.map (fun _ -> 1) one) param)
  else Nx.reshape one (Nx.contiguous param)

(* Entries below the threshold leave the distribution; ties at it stay. *)
let keep_from ~threshold logits =
  Nx.where
    (Nx.greater_equal logits threshold)
    logits
    (Nx.scalar_like logits Float.neg_infinity)

(* Both masks compare against a threshold read from the sorted values. The
   sorting permutation is never demanded: compiled, it takes a second sort over
   keys twice or four times as wide. *)
let sorted_desc logits = fst (Nx.sort ~descending:true ~axis:(-1) logits)

let keep_top_k ~k logits =
  let lead, vocab = last_axis ~fn:"keep_top_k" logits in
  let k = column ~fn:"keep_top_k" ~lead k in
  let at = Nx.clamp ~min:0l ~max:(Int32.of_int (vocab - 1)) (Nx.sub_s k 1l) in
  let sorted = sorted_desc logits in
  keep_from ~threshold:(Nx.take_along_axis ~axis:(-1) ~indices:at sorted) logits

let keep_top_p ~p logits =
  let lead, vocab = last_axis ~fn:"keep_top_p" logits in
  let p = column ~fn:"keep_top_p" ~lead p in
  let sorted = sorted_desc logits in
  let probs = Nx.softmax ~axes:[ -1 ] sorted in
  (* The mass strictly before each entry: an entry stays while that mass is
     below [p], so the fewest entries reaching [p] survive, and the first always
     does. *)
  let before = Nx.sub (Nx.cumsum ~axis:(-1) probs) probs in
  (* [p >= 1] keeps everything: a confident row's cumulative sum rounds to one
     before its tail, which the mass test alone would drop. *)
  let all = Nx.greater_equal p (Nx.ones_like p) in
  let kept = Nx.cast Nx.int32 (Nx.logical_or (Nx.less before p) all) in
  let count = Nx.sum ~axes:[ -1 ] ~keepdims:true kept in
  let at =
    Nx.clamp ~min:0l ~max:(Int32.of_int (vocab - 1)) (Nx.sub_s count 1l)
  in
  keep_from ~threshold:(Nx.take_along_axis ~axis:(-1) ~indices:at sorted) logits
