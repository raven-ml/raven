(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Splitting methods: the implementation of {!Jera.Split}. *)

type t

val leapfrog : t
val mclachlan : t
val yoshida4 : t
val yoshida6 : t
val yoshida8 : t
val v : kick:float array -> drift:float array -> t

type ('s, 'b) flow = (float, 'b) Nx.t -> 's -> 's

val step :
  t ->
  kick:('s, 'b) flow ->
  drift:('s, 'b) flow ->
  (float, 'b) Nx.t ->
  's ->
  's

val march :
  's Nx.Ptree.t ->
  t ->
  steps:int ->
  kick:('s, 'b) flow ->
  drift:('s, 'b) flow ->
  at:(float, 'b) Nx.t ->
  's ->
  's
