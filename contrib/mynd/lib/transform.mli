(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transforms: the implementation of {!Mynd.Transform}. *)

val apply : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t

val project :
  string ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t * (float, 's, 'd) Nx.t * (float, 's, 'd) Nx.t
(** [project fn m p] is the numerators of [x] and [y] and the denominator [d]
    of each position of [p], [[b…; q…; 2]], under {!apply}'s arithmetic, each
    [[b…; q…]]. It raises as {!apply} does, naming [fn]. *)

val translation : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val scaling : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val rotation : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val shear : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
