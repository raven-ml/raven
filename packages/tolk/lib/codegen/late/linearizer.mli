(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Ordering a kernel into a list of instructions.

    A kernel is a graph under a sink, and a program runs its nodes one after
    another. A loop is an {!Op.End} or an {!Op.Backedge} closing one range, its
    second source: its body runs inside the range, and whatever follows it runs
    after. Before a kernel is linearized its loops are made explicit, in two
    steps: each {!Op.End} is split to close one range ({!pm_split_ends}), and
    the loops that share an enclosing loop are chained, each range after the
    loop it must follow ({!pm_add_control_flow}). {!linearize} then lists the
    nodes in a topological order, as close as it can to an ideal one. *)

(** {1:order Linearizing} *)

val linearize : Ops.t -> Ops.t list
(** [linearize sink] is the nodes [sink] reaches ({!Ops.toposort}), each after
    its sources, [sink] last.

    Among the topological orders, it takes the one closest to the ideal order,
    which sorts nodes by, in turn:

    - their run count, the product of the sizes of the ranges they run inside
      ({!Ops.ranges}), a range's size being its {!Ops.vmax} plus one: fewer runs
      come first, so that work leaves the loops it does not need;
    - their kind: parameters first, by slot; then buffers outside local memory,
      buffers in local memory, loop ends ({!Op.End} and {!Op.Backedge}), loads,
      the other nodes, stores, and ranges last;
    - their structure ({!Ops.compare_structure}) if {!Setting.tuple_order}
      holds;
    - their position in {!Ops.toposort}.

    The list is built from its end: starting with [sink], each step places the
    node latest in the ideal order among those whose consumers are all placed.

    When the environment variable [DEBUG_LINEARIZE] holds a nonzero integer,
    each node is printed on standard output with its position, operation, ranges
    and priority. *)

(** {1:cfg Chaining loops} *)

type cfg_context
(** The type for the order a kernel's loops run in. *)

val cfg_context : Ops.t -> cfg_context
(** [cfg_context sink] is the order the loops of [sink] run in. A loop is nested
    in the first loop, in {!Ops.toposort}'s order, that depends on it and whose
    range it depends on, or in the sink if there is none. The loops nested in
    the same one are its children, ordered by the number of their siblings each
    depends on, fewest first, then by position in {!Ops.toposort}. Each child
    runs after the one before it; the first child of a loop runs inside the
    loop's range, and the first child of the sink runs first.

    Raises [Invalid_argument] if a range would run after a loop that depends on
    the range. *)

val pm_add_control_flow : (cfg_context, Ops.t) Ops.Pattern_matcher.t
(** [pm_add_control_flow] adds to each range of a context's sink, as its last
    source, the loop or the range it runs after in that context. It is applied
    bottom-up ({!Ops.graph_rewrite}) to the sink the context was computed from.
*)

val pm_split_ends : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_split_ends] rewrites an {!Op.End} of [u] into nested ends of one range
    each around [u]. Its ranges are, each once, those among its sources after
    [u] and the ranges its other sources after [u] run inside ({!Ops.ranges});
    the greatest by argument is innermost. An {!Op.End} closing no range is [u].
*)
