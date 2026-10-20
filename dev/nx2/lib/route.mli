(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Routes: where an operation reads its operands and where its results lie.

    A route is plain data from placements and shapes; it computes nothing. The
    engine takes placements from it alone, for traced results and eager ones, so
    a traced result lies where its eager twin does.

    The rule: every result lies at one placement, the {e target}. An operand of
    every set has no placement and joins the others: operands that are all of
    every set give no route. Each other operand is first given the axes the
    operation reads whole on each device (an axis it acts along), and moved to
    the result's axes. The target is the one of these that cuts the results
    into the most windows, the last of them where several do; else, where
    some lie on one device, that device, which every such operand must share;
    else the first. A cut of the target along an axis that an operand regroups
    into, in tiles that do not divide both extents, is undone. Each operand is
    then read where it lies if that gives every device of the target the part
    of the operand the device's window of the result depends on, and lies on no
    other device: the same window, or the whole operand on each device of the
    target; otherwise it is read at the target moved back to its axes, which
    the engine places it at.

    An operand may instead be read cut along the axes the operation reduces
    ({!Reduced}), where the first operand lying so gives the grid of the
    others. Each operand with reduced axes is then read where it lies, or at
    the first's placement where it reads its axes as the first does and lies
    whole on each of its devices, or is of every set. Each must:
    - cut each reduced axis's factor as the first does, over the same grid
      axes;
    - give what the target wants of it with its reduced axes uncut;
    - lie along its other axes, moved to the results' axes, as the target
      does; one such operand lies at the whole target.

    Operands with no reduced axes are read as above. The devices that hold
    one window of the results then hold all of its terms between them. Where
    an operand cannot be read so, or an axis the operation reduces or scans
    has no element, every operand is read as above, whole along its reduced
    axes.

    Two rules follow, with JAX as the reference for placements. Operands of one
    set never raise for where they lie: the brand already makes them one set,
    and moving data within a set is the route's, as JAX computes over operands
    sharded differently on one device set. Operands each alone on a different
    device raise, as JAX refuses operands committed to two devices: no
    arrangement holds both, and moving one is a choice the program makes with
    [Nx.place]. *)

(** The type for what an operation makes of an operand's axis. *)
type axis =
  | Becomes of int
      (** The results' axis [a] of [Becomes a], of its extent: a window of the
          results reads the operand's axis at its range along [a]. *)
  | Whole
      (** Read whole on each device: an axis the operation acts along, as a
          [Moments] or [Logsumexp] scan's, a sort's, a transform's or a
          gather's source axis, and one a padded load pads, crops or
          windows. *)
  | Reduced of int
      (** Gone from the results: a reduction's axes and a contraction's
          contracting pairs, but one a padded load of the reduction pads,
          crops or windows, which is [Whole] for every load. The axes of
          [Reduced f] for one factor [f] range over one index: a reduction's
          axis [a] is [Reduced a] in each load, and a contraction's pair [p]
          is [Reduced p] in both operands. Read whole on each device, or cut
          where the operand lies so, each factor alike in every operand
          ({!route}). *)
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
          behind it: the loop's space, which {!keeps} and {!window} read as an
          operand of the loop's shape. {!at} routes it over no operand;
          {!route}, which takes a row per operand, never receives it. *)
  | Into of axis array array
      (** [Axes]' windows, written into the last operand: the target is the last
          operand's placement where it lies on the set, or, where it is of
          every set, the target {!route} gives with the results' axes it
          does not become whole. *)

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
    [Scanned] axis. A kernel library meets it where an element's association,
    its algorithm and whether the library declines depend on the extents the
    element reads whole, the dtypes, the attributes, the library and the
    device's architecture, never on the extents its window cuts. *)

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
      (** Where each operand is read, in the operation's operand order. *)
  result : 'd Devices.placement;
  compute : compute;
}

(** The type for where operands lie together. *)
type 'd common =
  | Every_set  (** Every operand is of every set. *)
  | One of 'd Devices.placement
      (** Every operand not of every set lies at this placement. *)
  | Other  (** Any other arrangement. *)

val common : 'd Devices.placement option array -> 'd common
(** [common ps] is where operands at [ps] lie together, [None] for an operand of
    every set. Operands at [One p], [p] cutting no axis, are read at [p] and
    their results lie there, whatever the rule ({!route}). *)

val route :
  by:string ->
  rule ->
  'd Devices.placement option array ->
  int array array ->
  'd t option
(** [route ~by r ps shapes] routes an operation of rule [r] over operands at
    [ps] of [shapes], [None] for an operand of every set. It is [None] where
    every operand is: the operation is of every set too. An operand of every
    set is read at the target moved back to its axes.

    Raises [Invalid_argument] naming [by] if operands lie on two sets or on two
    devices alone, if a placement an operand is read at cuts an axis its shape
    lacks ({!Grid.window}), and as an internal fault if [ps] and [shapes] differ
    in length, [ps] is empty, or [r] does not name each axis of each operand. *)

val keeps : rule -> int -> bool
(** [keeps r a] is whether the results of an operation of rule [r] keep their
    axis [a] whole: no row's axis becomes it or regroups into it. *)

val at : by:string -> rule -> int array array -> 'd Devices.placement -> 'd t
(** [at ~by r shapes p] routes an operation of rule [r] over operands of
    [shapes] whose results are wanted at [p]: the result lies at [p], with
    each axis [r] keeps whole in the results whole ({!keeps}) and each cut an
    operand regroups across rows undone ({!Regroups}); each operand is read at
    that target moved back to its axes. It computes by partials where an
    operand is then read cut along an axis [r] scans, each such axis holding
    elements, as {!route} decides.

    Raises [Invalid_argument] naming [by] if a placement an operand is read at
    cuts an axis its shape lacks. *)
