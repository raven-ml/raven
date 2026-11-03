(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Standard normal draws at a tensor's type. *)

val normal_like : ('a, 'b, 'd) Nx.t -> 'd Nx.Rng.key -> int array -> ('a, 'b, 'd) Nx.t
(** [normal_like x k shape] is standard normal draws of [shape] at [x]'s dtype,
    or zeros if [x] is not a float tensor. *)

val normal : 'd Nx.Rng.key -> (float, 'b) Nx.dtype -> int array -> (float, 'b, 'd) Nx.t
val uniform : 'd Nx.Rng.key -> (float, 'b) Nx.dtype -> int array -> (float, 'b, 'd) Nx.t

val exponential :
  'd Nx.Rng.key -> (float, 'b) Nx.dtype -> int array -> (float, 'b, 'd) Nx.t

val gumbel : 'd Nx.Rng.key -> (float, 'b) Nx.dtype -> int array -> (float, 'b, 'd) Nx.t
(** [normal k dt shape] and the others are {!Nx.Rng}'s draws from [k], at
    {!Nx.Rng.placement}[ k]. *)
