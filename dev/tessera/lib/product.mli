(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Linear layers whose sum and bias round once.

    [Kaun.Linear.apply] rounds the product, then the bias added to it; at
    bfloat16 that is two roundings where PyTorch's autocast linear makes one.
    This is tessera's one copy of the single rounding until kaun's linear layer
    rounds once. *)

val apply :
  ((float, 's, 'd) Nx.t, 'd) Kaun.Linear.t ->
  (float, 'r, 'd) Nx.t ->
  (float, 's, 'd) Nx.t
(** [apply p x] is [x · w + b] at [w]'s dtype over [x]'s last axis, [x] cast to
    that dtype first, each output [round (b + Σ x w)]. *)
