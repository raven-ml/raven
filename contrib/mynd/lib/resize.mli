(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Resizing: the implementation of {!Mynd.Sample.section-resizing}. *)

val resize_weights :
  edge:[< Sample.edge ] ->
  Sample.kernel ->
  int ->
  int ->
  Nx.host Nx.int64_t * Nx.host Nx.float64_t

val resize :
  edge:[< Sample.edge ] ->
  Sample.kernel ->
  int * int ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t
