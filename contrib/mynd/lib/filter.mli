(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Windows, linear filters and the median: the implementation of {!Mynd.Filter}
    but its gradients and morphology, which [Gradient] and [Morphology]
    implement. *)

val windows :
  edge:[< Image.extension ] ->
  int * int ->
  ('v, 'r, 'd) Nx.t ->
  ('v, 'r, 'd) Nx.t

val correlate :
  edge:[< Nx.extension ] ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t

val gaussian :
  edge:[< Nx.extension ] ->
  radius:int ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t

val median :
  edge:[< Image.extension ] ->
  int * int ->
  ('v, 'r, 'd) Nx.t ->
  ('v, 'r, 'd) Nx.t
