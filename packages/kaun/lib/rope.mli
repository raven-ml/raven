(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Rotary position embeddings (Su et al., 2021).

    A rotary embedding encodes a token's position by rotating pairs of features
    of its query and key vectors: pair [i] turns by the angle
    [position * frequency.(i)]. The dot product of a rotated query and a rotated
    key then depends on their positions only through the difference, which is
    what attention needs.

    A value of type {!t} is the schedule: one inverse frequency per feature
    pair, computed once on the host. It has no parameters and never enters a
    parameter tree. Every published schedule is a choice of those frequencies,
    so a new one is a constructor and {!apply} never changes. *)

type t
(** The type for rotary schedules: the inverse frequencies of one attention
    head, [head_dim / 2] of them. *)

val make : ?theta:float -> head_dim:int -> unit -> t
(** [make ~head_dim ()] is the standard schedule: pair [i] has frequency
    [theta ** (-2 i / head_dim)]. [theta] defaults to [10000.].

    Raises [Invalid_argument] if [head_dim] is not positive and even, or [theta]
    is not positive. *)

val llama3 :
  theta:float ->
  head_dim:int ->
  factor:float ->
  low_freq_factor:float ->
  high_freq_factor:float ->
  original_context:int ->
  t
(** [llama3 ...] is the Llama 3.1 long-context schedule: the standard
    frequencies with the low ones divided by [factor], the high ones kept, and a
    linear blend between, the bands being set by the wavelengths
    [original_context / low_freq_factor] and
    [original_context / high_freq_factor]. Llama 3.1 uses
    [~theta:500000. ~factor:8. ~low_freq_factor:1. ~high_freq_factor:4.
     ~original_context:8192]; Llama 3.2 uses [~factor:32.].

    Raises [Invalid_argument] on a non-positive argument, or if
    [high_freq_factor <= low_freq_factor]. *)

val frequencies : t -> float array
(** [frequencies t] is a copy of [t]'s inverse frequencies, in pair order. *)

val apply :
  t -> pos:(int32, Nx.int32_elt) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [apply t ~pos x] rotates [x], of shape [[| batch; heads; seq; head_dim |]],
    by the positions [pos], of shape [[| batch; seq |]] (or [[| 1; seq |]],
    shared by the batch). Feature [i] is paired with feature [i + head_dim / 2]:

    {v
    y.(i)     = x.(i) cos a     - x.(i + h) sin a
    y.(i + h) = x.(i + h) cos a + x.(i) sin a
    v}

    where [h] is [head_dim / 2] and [a] is [pos * frequency.(i)].

    Position [0] is the identity. The angles and their sines and cosines are
    computed at float32 whatever [x]'s dtype and cast to it for the rotation, so
    a float64 [x] is rotated to float32 accuracy. Differentiable through Rune in
    [x].

    Raises [Invalid_argument] if [x] is not of rank 4, its last axis is not
    twice [t]'s pair count, or [pos] has another shape. *)
