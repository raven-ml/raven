(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The narrow transformer ({!Fixture.transformer}) and its draws compiled by
    [Rune.jit] on a device set, against the eager ones on the host. A compiled
    call's sums may combine their terms in another order: eager and compiled are
    float32 computations of one exact value, each within [τ ŝ] of it, so they
    differ by at most [2 τ ŝ], [ŝ] the fixtures' ensemble spread. *)

open Aion

type 'd transformer = {
  encode :
    ('d Nx.float32_t, 'd) Transformer.t ->
    'd Tokens.t list ->
    ('d Nx.float32_t, 'd) Transformer.encoding;
  decode :
    ('d Nx.float32_t, 'd) Transformer.t ->
    ('d Nx.float32_t, 'd) Transformer.encoding ->
    'd Transformer.target list ->
    'd Nx.float32_t list;
  forward :
    ('d Nx.float32_t, 'd) Transformer.t ->
    'd Tokens.t list ->
    'd Transformer.target list ->
    'd Nx.float32_t list;
  embedding : ('d Nx.float32_t, 'd) Transformer.encoding -> 'd Nx.float32_t;
}
(** The type for a transformer's entry points on the device set ['d]. *)

val transformer : unit -> 'd transformer
(** [transformer ()] is the entry points compiled, a program per list of
    modalities, shapes and placement. *)

val close :
  string ->
  spread:Nx.host Nx.float32_t ->
  'd Nx.float32_t ->
  Nx.host Nx.float32_t ->
  unit
(** [close name ~spread x y] checks that each value of [x] lies within [2 τ ŝ]
    of [y]'s, [ŝ] the value's [spread] and [τ] that of transformer_golden's
    ensemble. *)

val agrees :
  'd Nx.Placement.t ->
  'd transformer ->
  rows:int list ->
  step:string ->
  perm:int list ->
  zs:int list ->
  unit
(** [agrees on c ~rows ~step ~perm ~zs] checks [c] on the transformer placed at
    [on] against {!eager} on the host:
    - for the objects [rows] of {!Fixture.inputs}, the lengths exactly, the
      context within [2 τ ŝ] and the embedding within its sums' bound;
    - for the step [step] of those objects, its positions listed in the order
      [perm], a permutation of [0 … 6], the logits of [forward] and of [decode]
      within [2 τ ŝ];
    - redshift's whole target for the objects [zs], each [1] or [2], whose
      spread the fixture holds, within [2 τ ŝ]. *)

val draw :
  ?traces:int ref ->
  'd Nx.Placement.t ->
  keys:Nx.host Nx.Rng.key ->
  ?temperature:float ->
  reveal:int list ->
  Modality.t ->
  Nx.host Tokens.t list ->
  Nx.host Nx.int64_t
(** [draw ~traces on] is a caller's compiled {!Roar.sample} at [on]:
    [Roar.start] compiled for each call, and one [Roar.step] compiled for every
    call, its traces counted in [traces].
    [draw on ~keys ~temperature ~reveal target inputs] places the narrow
    transformer, [keys] and [inputs] at [on], folds the steps over [reveal] and
    gives the tokens on the host. *)
