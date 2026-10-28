(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the elementwise solves share: their status codes while running and the
    statement of their answer. *)

val running : int32
(** [running] is the status code of an element still searching. *)

val searching : 'd Nx.int32_t -> 'd Nx.bool_t
(** [searching st] is [true] where [st] is {!running}. *)

val settle : 'd Nx.int32_t -> 'd Nx.bool_t -> Solution.status -> 'd Nx.int32_t
(** [settle st cond s] is [st] with [s] where an element still searching meets
    [cond]. *)

val state :
  string ->
  ok:'d Nx.bool_t ->
  ((float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t) ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [state fn ~ok r x] is [x] stated as a zero of [where ok (r x) (x − x)],
    elementwise: its derivative is [r]'s implicit one where [ok], zero
    elsewhere. Its linear solves divide by the diagonal and check the quotient
    with one more product, raising [Invalid_argument] naming [fn] where it
    misses: [r] read another element. *)

val accepted :
  Tol.t -> e:(float, 'b, 'd) Nx.t -> y:(float, 'b, 'd) Nx.t -> 'd Nx.bool_t
(** [accepted tol ~e ~y] is [true] where the error [e] at [y] meets [tol], each
    element its own lane. *)
