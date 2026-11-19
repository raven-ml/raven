(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let check_rate ~by rate =
  if not (rate >= 0.0 && rate < 1.0) then
    invalid_argf "%s: rate must be in [0, 1), got %a" by
      (Nx.Dtype.pp_value Nx.float64)
      rate

(* Raises unless [key]'s batch shape broadcasts to [s], aligned on the right
   as Nx.Rng aligns keys with a sampler's parameters: a batch that would
   widen the draw past [s] is refused. *)
let check_key ~by s key =
  let words = Nx.shape (Nx.Rng.to_tensor key) in
  let b = Array.sub words 0 (Array.length words - 1) in
  let fits =
    match Nx.broadcast_shapes [ b; s ] with
    | s' -> s' = s
    | exception Invalid_argument _ -> false
  in
  if not fits then
    invalid_argf "%s: keys of batch shape %a do not broadcast to x's shape %a"
      by Nx.pp_shape b Nx.pp_shape s

(* At float32 whatever [x]'s dtype, so the mask for a key does not depend on
   the input's precision. Where [x] lies, but whole where it is a sum: the
   mask scales each term alike, so the product stays a sum. *)
let draw ~rate ?key x =
  let at = Nx.Placement.closed (Nx.placement x) in
  Nx.Rng.bernoulli ?key (Nx.full_at at Nx.float32 (Nx.shape x) (1.0 -. rate))

let mask ~rate ?key x =
  let by = "Dropout.mask" in
  check_rate ~by rate;
  Option.iter (check_key ~by (Nx.shape x)) key;
  draw ~rate ?key x

let apply ~rate ~training ?key x =
  let by = "Dropout.apply" in
  check_rate ~by rate;
  Option.iter (check_key ~by (Nx.shape x)) key;
  if (not training) || rate = 0.0 then x
  else
    let keep = 1.0 -. rate in
    let m = Nx.cast (Nx.dtype x) (draw ~rate ?key x) in
    (* Inverted dropout: scale the survivors by 1/keep at training time so the
       expectation matches eval mode, which is the identity. *)
    Nx.mul_s (Nx.mul x m) (1.0 /. keep)
