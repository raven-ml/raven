(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Interpretations' records: the values they own, the formulas their rules
    returned, and their order of start. {!Eval} starts them. *)

open Value

val add_formula : interpretation -> int -> unit
(** [add_formula i id] records the node [id] as one of a formula [i]'s rule
    returned. *)

val is_formula : interpretation -> int -> bool
(** [is_formula i id] is [true] iff [add_formula i id] was called. It allocates
    nothing. *)

val traced :
  interpretation -> ('v, 's, 'd) form -> ('v, 's, 'd) payload -> ('v, 's, 'd) t
(** [traced i f p] is a value of form [f] that [i] owns, keeping [p]. *)

val payload : interpretation -> ('v, 's, 'd) t -> ('v, 's, 'd) payload option
(** [payload i x] is [Some p] iff [i] owns [x], [p] what it keeps. *)

val owner : ('v, 's, 'd) t -> interpretation option
(** [owner x] is the interpretation that owns [x], if [x] is traced. *)

val later : interpretation -> interpretation -> bool
(** [later a b] is [true] iff [a] started after [b], on any domains. Of two
    interpretations around one operation, the later is inside the other. *)
