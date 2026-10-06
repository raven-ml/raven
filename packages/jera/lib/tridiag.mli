(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Diagonally dominant tridiagonal systems, solved in parallel. *)

val solve :
  sub:(float, 'b) Nx.t ->
  diag:(float, 'b) Nx.t ->
  sup:(float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [solve ~sub ~diag ~sup r] is the [x] with
    [sub.(i) x.(i − 1) + diag.(i) x.(i) + sup.(i) x.(i + 1) = r.(i)], the three
    diagonals of shape [[n]] with [sub.(0) = 0] and [sup.(n − 1) = 0], [r] and
    [x] of shape [[n] @ rest]. The elimination is three associative scans of
    [O(log n)] rounds; it is stable for a diagonally dominant system, where no
    pivoting is needed. *)
