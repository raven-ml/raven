(** Witnesses of {!Tolk_next.Ops} nodes, and the comparison of graphs that make
    new storage. *)

open Tolk_next

val uop : Ops.t Windtrap.testable
(** [uop] compares nodes by identity, which is structural equality, orders them
    by {!Tolk_next.Ops.compare}, and prints the graph under a node in the graph
    format ({!Graph}). *)

val numbered_like : Ops.t -> Ops.t -> Ops.t
(** [numbered_like like u] is [u] with its call-local storage
    ({!Tolk_next.Op.Alloc}) in the slots of [like]'s, paired in the order
    {!Tolk_next.Ops.toposort} visits them, call bodies included. A pass numbers
    the storage it makes from a counter that the process shares, so the numbers
    are no property of the pass; this compares a graph with a golden up to them.
    It is [u] if the two graphs make different numbers of storage, which a
    comparison then shows. *)
