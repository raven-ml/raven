(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Standard normal draws at a tensor's type. *)

val normal_like : ('a, 'b) Nx.t -> Nx.Rng.t -> int array -> ('a, 'b) Nx.t
(** [normal_like x k shape] is standard normal draws of [shape] at [x]'s dtype,
    or zeros if [x] is not a float tensor. *)
