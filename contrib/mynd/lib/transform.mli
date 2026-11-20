(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transforms: the implementation of {!Mynd.Transform}. *)

val apply : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val translation : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val scaling : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val rotation : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val shear : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
