(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Expansions: an operation's optional case as core operations, applied through
    [apply].

    One case is optional: a map of several nodes. It expands into one-node maps,
    one per node in order, each over the values of the nodes it reads; a
    constant or a coordinate is a creation of the map's shape. Every other
    operation is core. *)

type apply = { apply : 'r. by:string -> 'r Value.prim -> 'r }

val run : apply -> by:string -> 'r Value.prim -> 'r option
(** [run a ~by op] is [Some r], [r] [op]'s expansion applied through [a], or
    [None] for a core case. *)
