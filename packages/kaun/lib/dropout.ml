(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let apply ~rate ~training x =
  if rate < 0.0 || rate >= 1.0 then
    invalid_arg
      (Printf.sprintf "Dropout.apply: rate must be in [0, 1), got %g" rate);
  if (not training) || rate = 0.0 then x
  else
    let keep = 1.0 -. rate in
    let mask = Nx.cast (Nx.dtype x) (Nx.bernoulli ~p:keep (Nx.shape x)) in
    (* Inverted dropout: scale the survivors by 1/keep at training time so the
       expectation matches eval mode, which is the identity. *)
    Nx.mul_s (Nx.mul x mask) (1.0 /. keep)
