(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Routes: where an operation reads its operands and where its results lie.

    A route is plain data from placements and shapes; it computes nothing. The
    engine takes placements from it alone, for traced results and eager ones, so
    a traced result lies where its eager twin does, and a route that raises
    raises at trace time as it would eagerly. This module lies above {!Grid}
    and {!Devices}; {!Prim} builds each operation's route by it.

    Each operand is read where it lies, and the results lie at the
    {e forward union} of the operands' cuts. A cut of an operand's axis that
    becomes a result axis, is scanned along it, or regroups into one in tiles
    that divide both extents, cuts that result axis over the same devices,
    each device holding the tile it holds of the operand. A cut of an axis
    read whole cuts nothing. A cut of an axis the operation reduces cuts
    nothing in the results, and the devices compute by partials
    ({!Partial}), where every operand that holds that axis is cut alike along
    it; an axis of its blocks is cut alike in tiles that divide both
    extents. Grid axes that cut a reduced axis hold copies of the
    results.

    The cuts join in operand order. The first operand gives its cuts, and
    each next one adds its own where they join the union so far
    ({!Grid.union}), and none where they do not. An operand alone on one
    device beside operands whole on every device of the set puts the
    results on that device. A cut of the union along an axis that an
    operand regroups into in tiles that do not divide both extents is
    undone.

    Each operand is read at the union moved back to its axes. Where that is
    its own placement, it is read there. Where its own placement holds on
    each device the window it is read at, as a value whole on every device
    does, it is read by a view of that window.

    The route raises [Invalid_argument] where an operand is read neither
    way, so that a device of the results needs a window of it the device
    does not hold: an operand cut along an axis read whole, along an axis
    the union cuts over other devices or in tiles that do not hold the
    union's, or along two axes where the union cuts one over the same
    devices, and one alone on another device than the results. It raises
    too where an operand lies on another set. The message names the
    function, the operand, its placement, the placement it is read at, and
    the [Nx.place] to write, writing the set as [s] and the operand as [x],
    [y], [z], then [x3], [x4], …:

    {v
    Nx.matmul: operand 1 lies at split ~axis:1 [CUDA:0; CUDA:1] and is read
    at split ~axis:0 [CUDA:0; CUDA:1]; Nx.place (Placement.split ~axis:0 s) y
    v}

    The expression is [Placement.on s], [Placement.device s] with the
    device's name, [Placement.split ~axis s], or a [Placement.mesh] over
    [Mesh.v s] whose axes, named ["g0"], ["g1"], …, are the read placement's
    grid axes in normal form ({!Grid}). *)

(** The type for what an operation makes of an operand's axis. *)
type axis =
  | Becomes of int
      (** The results' axis [a] of [Becomes a], of its extent: a window of the
          results reads the operand's axis at its range along [a]. *)
  | Whole
      (** Read whole on each device: an axis the operation acts along, as a
          [Moments] or [Logsumexp] scan's, a sort's, a transform's or a
          scatter's axis, one a padded load pads, crops or windows,
          and one an assembly joins or pads along. *)
  | Reduced of int
      (** Gone from the results: a reduction's axes, a contraction's
          contracting pairs, a gather's source axis and [b]'s axis of
          [Rows]' groups ({!Nx_kernel.type-segments}), but one a padded load
          of the reduction pads, crops or windows, which is [Whole] for every
          load. The axes of [Reduced f] for one factor [f] range over one
          index: a reduction's axis [a] is [Reduced a] in each load, a
          gather's [a] is [Reduced a], a contraction's pair [p] is
          [Reduced p] in both operands, and the axis of [b]'s groups, which
          no axis of [a] shares, is the factor after the pairs. Read whole on
          each device, or cut where the operand lies so, each factor alike in
          every operand ({!route}). *)
  | Blocks of { factor : int; extent : int }
      (** Gone from the results as [Reduced factor] is, its index the block
          of the factor's index it holds: a contraction's scale along a
          contracting pair of [extent] elements, a whole multiple of the
          axis's own, both above one, one value per block. A cut of it into
          [t] tiles routes as [Reduced factor] where [t] divides both
          extents: each device then holds the blocks of its tile of the
          factor. Otherwise it is read whole on each device. *)
  | Scanned of int
      (** The results' axis [a] of [Scanned a], along which a scan of a
          [Sum], [Prod], [Max], [Min] or [Arg] without a carry runs, every
          load reading it plain: cut as the operand is, as [Becomes a] routes
          it, but a window of the results depends on the terms before it, so
          the window law does not hold along it ({!Partial}). *)
  | Regroups of { axis : int; extent : int }
      (** The results' [axis], of [extent] elements, another extent than the
          operand's, both above one: a reshape that factors or merges the
          operand's axis keeps its elements in order, in rows of other
          lengths. A cut of it into [t] tiles routes as [Becomes axis] where
          [t] divides both extents: each window then holds whole rows on both
          sides. Otherwise the axis is read whole on each device, and the
          results whole along [axis]. *)

(** The type for how an operation's results depend on its operands: for each
    axis of each operand, what the operation makes of it. A window of the
    results reads each operand's axis that becomes one of theirs at that axis's
    range, one that regroups into one of theirs at that range scaled by the
    ratio of the extents, and every other axis whole. A result axis that no
    operand's axis becomes or regroups into is kept whole: every window the
    engine computes holds all of it.

    This is the sharding rule of Shardy
    ({{:https://openxla.org/shardy/sdy_dialect}OpShardingRuleAttr}): a
    result axis is a factor that the axes becoming it share, its reduction
    factors are the [Reduced] axes, which an operand may be read cut along,
    and its need-replication factors [Whole] ones, one per axis. *)
type rule =
  | Elementwise
      (** Operands and results of one shape, each axis becoming itself. *)
  | Axes of axis array array
      (** Axis [a] of operand [i] is [t.(i).(a)] of [Axes t]. No two axes of
          an operand become one. A loop of no load has one row and no operand
          behind it: the loop's space, which {!window} reads as an operand of
          the loop's shape. {!route}, which takes a row per operand, never
          receives it. *)

val moved : Nx_array.Move.t -> int array -> int -> axis
(** [moved m s a] is what [m] makes of axis [a] of a value of shape [s]:
    - [Becomes a'] where [m] keeps the axis's elements in order along [a']: a
      permutation's, a broadcast's that does not stretch it, a slice's that
      keeps all of it, and a reshape's pair. A reshape pairs the axes of one
      product of the extents before them in order, each with the result's
      next axis of that product and of its extent.
    - [Regroups] for an axis a reshape leaves unpaired, of more than one
      element after a positive product, onto the result's axis of more than one
      element after the same product, where there is one: unit axes are
      skipped.
    - [Whole] for any other. *)

val window :
  rule ->
  operand:int ->
  int array ->
  Nx_array.Move.range array ->
  Nx_array.Move.range array
(** [window r ~operand:i s w] is the window of operand [i], of shape [s], that
    the window [w] of an operation's results reads by [r]: [w]'s range along an
    axis that becomes one of the results', that range scaled by [e / extent]
    along an axis of [e] elements that regroups into one, whole along any
    other. Where an operation has several results, [w] is a window of the
    first.

    The engine computes an operation one window at a time, each device over its
    operands' windows, under the {e window law}: for every window [w] of the
    results, the operation over each operand's [window r ~operand:i s w] equals
    the window [w] of the operation over the whole operands, bitwise. It
    computes every route that reads no operand cut along a [Reduced] or a
    [Scanned] axis. A kernel library declines a window as it declines the
    whole ({!Nx_kernel.section-contract}). It meets the law where an
    element's association and its algorithm depend on the extents the
    element reads whole, the dtypes, the attributes, the library and the
    device model, never on the extents its window cuts, as nx.cpu's do
    ({!Nx_cpu}); on a library that does not state so, a float or complex
    window keeps its library's bounds alone. A float or complex contraction
    keeps the law but for its association, which its library chooses from
    the window's extents: each window is within the bound of
    {!Nx_kernel.S.contract} and the same on every run for a given placement
    and set of devices. *)

val full : int -> Nx_array.Move.range
(** [full d] is the range of all [d] elements of an axis: from [0], by step
    [1]. *)

val extents : Nx_array.Move.range array -> int iarray
(** [extents w] is the shape of the window [w]: each range's count. *)

val empty : Nx_array.Move.range array -> bool
(** [empty w] is [true] iff a range of [w] counts no element. It allocates
    nothing. *)

val contains : Nx_array.Move.range array -> Nx_array.Move.range array -> bool
(** [contains o w] is [true] iff each range of [w] lies within [o]'s along
    its axis, for windows of one rank whose ranges step by [1], as
    {!Grid.window}'s do.

    Raises [Invalid_argument] if [o] and [w] differ in rank. *)

(** The type for how the devices of a route compute its results. *)
type compute =
  | Windows
      (** Each device computes its window of the results over its operands'
          windows ({!window}), under the window law. *)
  | Partials
      (** An operand is read cut along an axis the operation reduces or scans,
          each such axis holding elements: the devices that hold one window of
          the results compute it from partial results over the terms each
          holds ({!Partial}). *)

type 'd t = {
  operands : 'd Devices.placement array;
      (** Where each operand is read, in the operation's operand order: its
          own placement, or one whose window on each device its own holds
          there. *)
  result : 'd Devices.placement;
  compute : compute;
}

val route :
  by:string -> rule -> 'd Devices.placement array -> int array array -> 'd t
(** [route ~by r ps shapes] routes an operation of rule [r] over operands at
    [ps] of [shapes], as the module's preamble states. [ps] and [shapes] are
    of one positive length, and [r] names each axis of each operand. Operands
    all at one placement are read there and their results lie there, where it
    cuts no axis or [r] is [Elementwise]. A route reads an operand at a
    placement other than its own only through views of its arrays. An axis
    that [r] reduces and that has no element is read whole by every operand;
    one it scans keeps its cut, and the devices compute by windows.

    Raises [Invalid_argument] naming [by] if an operand's placement cuts an
    axis its shape lacks ({!Grid.window}), and with the preamble's message
    where an operand cannot be read where it lies. *)

val pp_call :
  by:string -> 'd Devices.t -> Format.formatter -> Grid.t -> unit
(** [pp_call ~by s ppf q] formats the [Nx.Placement] expression that makes
    [q] over [s], as the preamble's message writes it, writing the set as
    [s]. [by] is given to {!Devices.split}, which does not raise for an axis
    [q] cuts. *)
