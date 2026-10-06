(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the elementwise solves share: their status codes while running and the
    statement of their answer. *)

val running : int32
(** [running] is the status code of an element still searching. *)

val searching : (int32, Nx.int32_elt) Nx.t -> (bool, Nx.bool_elt) Nx.t
(** [searching st] is [true] where [st] is {!running}. *)

val settle :
  (int32, Nx.int32_elt) Nx.t ->
  (bool, Nx.bool_elt) Nx.t ->
  Solution.status ->
  (int32, Nx.int32_elt) Nx.t
(** [settle st cond s] is [st] with [s] where an element still searching meets
    [cond]. *)

val state :
  string ->
  ok:(bool, Nx.bool_elt) Nx.t ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [state fn ~ok r x] is [x] stated as a zero of [where ok (r x) (x − x)],
    elementwise: its derivative is [r]'s implicit one where [ok], zero
    elsewhere. Its linear solves divide by the diagonal and check the quotient
    with one more product, raising [Failure] naming [fn] where it misses: [r]
    read another element. *)

val accepted :
  Tol.t -> e:(float, 'b) Nx.t -> y:(float, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t
(** [accepted tol ~e ~y] is [true] where the error [e] at [y] meets [tol], each
    element its own lane. *)
