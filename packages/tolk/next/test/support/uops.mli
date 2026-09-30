(** Witnesses of {!Tolk_next.Ops} nodes, and the comparison of graphs that make
    new storage. *)

open Tolk_next

val uop : Ops.t Windtrap.testable
(** [uop] compares nodes by identity, which is structural equality, orders them
    by {!Tolk_next.Ops.compare}, and prints the graph under a node in the graph
    format ({!Graph}). *)

val numbered_like : Ops.t -> Ops.t -> Ops.t
(** [numbered_like like u] is [u] with its call-local storage
    ({!Tolk_next.Op.Alloc}) and its buffers ({!Tolk_next.Op.Buffer}) in the
    slots of [like]'s of the same kind, paired in the order
    {!Tolk_next.Ops.toposort} visits them, call bodies included. A pass numbers
    the storage it makes from a counter that the process shares, so the numbers
    are no property of the pass; this compares a graph with a golden up to them.
    The storage that the queue data of a call names ({!Tolk_next.Ops.hcq_info})
    is renumbered with it. It is [u] if the two graphs make different numbers of
    storage, which a comparison then shows. *)

val binaries_as_sources : Ops.t -> Ops.t
(** [binaries_as_sources u] is [u] with the binary of each compiled program
    ({!Tolk_next.Op.Program} ending in an {!Tolk_next.Op.Binary}) replaced by
    the bytes of the program's source ({!Tolk_next.Op.Source}), call bodies
    included. The generators record programs compiled that way, so that a
    golden holds no machine code; this compares a compiled graph with such a
    golden. *)

val placeholders_like : Ops.t -> Ops.t -> Ops.t
(** [placeholders_like like u] is [u] with its placeholders (tagged
    {!Tolk_next.Op.Param}s) in the slots of [like]'s, paired
    in the order {!Tolk_next.Ops.toposort} visits them, call bodies included.
    As storage, placeholders are numbered from the counter the process shares,
    so this compares a graph with a golden up to their numbers. It is [u] if the
    two graphs hold different numbers of them. *)

val without_profile_keys : Ops.t -> Ops.t
(** [without_profile_keys u] is [u] without the profile key of each kernel its
    command-queue calls enqueue ({!Tolk_next.Ops.hcq_kernel}). A kernel's key is
    its program's BLAKE2 digest, where tinygrad's is a SHA-256 (DIVERGENCES
    D12), so a graph is compared with a golden without them. *)
