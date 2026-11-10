(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Points as the last axis of a tensor, [[...; 2]], for the modules that map
    them. *)

val split :
  string -> 'd Nx.float64_t -> int array * 'd Nx.float64_t * 'd Nx.float64_t
(** [split fn p] is [p]'s batch shape (its shape without the last axis) and its
    two coordinates, each flat.

    Raises [Invalid_argument] naming [fn] if [p]'s last axis is not 2. *)

val join : int array -> 'd Nx.float64_t -> 'd Nx.float64_t -> 'd Nx.float64_t
(** [join batch a b] is the points of the flat coordinates [a] and [b], of shape
    [batch @ [2]]. *)
