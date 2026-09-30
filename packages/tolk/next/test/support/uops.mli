(** Witnesses of {!Tolk_next.Ops} nodes. *)

open Tolk_next

val uop : Ops.t Windtrap.testable
(** [uop] compares nodes by identity, which is structural equality, orders them
    by {!Tolk_next.Ops.compare}, and prints the graph under a node in the graph
    format ({!Graph}). *)
