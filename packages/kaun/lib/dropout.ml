(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let apply ~rate ~training ?key x =
  if rate < 0.0 || rate >= 1.0 then
    invalid_arg
      (Format.asprintf "Dropout.apply: rate must be in [0, 1), got %a"
         (Nx.Dtype.pp_value Nx.float64)
         rate);
  if (not training) || rate = 0.0 then x
  else
    let keep = 1.0 -. rate in
    (* At float32 whatever [x]'s dtype, so the mask for a key does not depend on
       the input's precision. *)
    (* Where [x] lies, but whole where it is a sum: the mask scales each
       term alike, so the product stays a sum. *)
    let at = Nx.Placement.closed (Nx.placement x) in
    let p = Nx.full_at at Nx.float32 (Nx.shape x) keep in
    let bits = Nx.Rng.bernoulli ?key p in
    let mask = Nx.cast (Nx.dtype x) bits in
    (* Inverted dropout: scale the survivors by 1/keep at training time so the
       expectation matches eval mode, which is the identity. *)
    Nx.mul_s (Nx.mul x mask) (1.0 /. keep)
