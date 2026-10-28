(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Device sets and placements: the devices values live on, the kernels that
    compute on them, and how a value's shape lies over them.

    A set is minted once and never changes: its devices, its kernels, and a
    number unique in the process, so two mints of one list are two sets. A
    placement is a set and a {!Grid.t} over positions in it. A set holds the
    placement of each of its devices alone and of its whole, made at its mint,
    so on the eager path two values on one device have physically equal
    placements. Sets and placements are immutable once {!mint} returns; any
    domain reads them.

    Brands are phantom. Nothing at run time reads ['d], and a value's arrays
    lie on its placement's devices whatever its brand. A brand names one
    set, and a set's module unpacked twice gives it two brand names. A
    placement is a {!place} at every brand, so the engine holds placements
    at [unit] in routes and reads them at an operation's brand, without a
    conversion. {!set} then gives a placement's set at whatever brand its
    caller names. [Nx] keeps its placements' brand abstract, which ties a
    program's placement to its set's brand, and [Nx.Prim.coerce] gives a
    value another brand only where that brand names the value's set.

    This module lies above {!Grid} alone, and is nx's one site that names
    nx.cpu.

    Errors are [Invalid_argument] naming the [by] the caller passes. *)

type host
(** The brand of {!host}. *)

type +'d t
(** The type for sets of brand ['d]. *)

type place
(** The type for placements. *)

type 'd placement = place
(** The type for placements over a set of brand ['d], a {!place} whatever
    ['d] is. *)

type +'d mesh
(** The type for named axes over a set's devices. *)

(** {1:sets Sets} *)

val host : host t
(** [host] is set [0]: {!Rig.host} alone, computed by nx.cpu. *)

val mint : by:string -> ?kernels:(module Nx_kernel.S) -> Rig.t list -> 'd t
(** [mint ~by ~kernels ds] is a new set over [ds], in their order, numbered
    after every earlier mint. Without [kernels], it is nx.cpu's if every device
    of [ds] runs on the host ({!Rig.runs_on_host}), and has none otherwise.

    Raises [Invalid_argument] naming [by] if [ds] is empty or repeats a device,
    or if [kernels] does not compute on one of [ds]. *)

val number : 'd t -> int
val count : 'd t -> int

val rig : 'd t -> int -> Rig.t
(** Raises [Invalid_argument] if the position is not below {!count}. *)

val position : 'd t -> Rig.t -> int option

val kernels : 'd t -> (module Nx_kernel.S) option
(** [kernels s] is who computes on [s] eagerly. It reads a field. *)

val pp : Format.formatter -> 'd t -> unit
(** [pp] formats a set as [set 2 [CUDA:0; CUDA:1]], {!host} as [host]. *)

(** {1:placements Placements} *)

val one : 'd t -> int -> 'd placement
(** [one s k] is [s]'s device [k] alone, made at [s]'s mint.

    Raises [Invalid_argument] if [k] is not below [count s]. *)

val on : 'd t -> 'd placement
(** [on s] is the whole value on every device of [s], made at [s]'s mint:
    [one s 0] for a set of one device. *)

val split : by:string -> axis:int -> 'd t -> 'd placement
(** [split ~by ~axis s] cuts [axis] into one window per device of [s], in
    order, as {!Grid.window} balances them: [one s 0] for a set of one device.
    Raises [Invalid_argument] naming [by] if [axis] is negative or not below
    {!Nx_array.Layout.max_rank}. *)

val mesh : by:string -> 'd mesh -> (int * string list) list -> 'd placement
(** [mesh ~by m cuts] cuts each axis [a] of [(a, names)] over the mesh axes
    [names], major first, into windows as {!Grid.window} balances them; a mesh
    axis no cut names holds copies. Raises
    [Invalid_argument] naming [by] for a name [m] lacks, an axis or a name in
    two cuts, or an axis that is negative or not below
    {!Nx_array.Layout.max_rank}. *)

val v : by:string -> 'd t -> Grid.t -> 'd placement
(** [v ~by s g] is [g] over [s]: [one s k] when [g] is device [k] alone, and
    [on s] when [g] is [s]'s devices in order with no cut. Raises
    [Invalid_argument] naming [by] if a device of [g] is not below [count s]. *)

val set : 'd placement -> 'd t
val grid : 'd placement -> Grid.t

val device : 'd placement -> int option
(** [device p] is [Some k] iff [p] is its set's device [k] alone. *)

val equal : 'd placement -> 'e placement -> bool
(** [equal p q] compares physically, then sets by number and grids. *)

val window :
  by:string -> 'd placement -> int array -> int -> Nx_array.Move.range array
(** [window ~by p shape i] is the window of a value of [shape] that the [i]th
    device of [p] holds, in [Grid.devices (grid p)]'s order, for a position
    [i] of [p]'s devices. Raises [Invalid_argument] naming [by] if [p] cuts an
    axis [shape] lacks. *)

val uncut : 'd placement -> 'd placement
(** [uncut p] is [p] with every axis whole on each of its devices, so that a
    grid axis that cut one holds copies. It is [p] itself if [p] cuts no
    axis. *)

val pp_placement : Format.formatter -> 'd placement -> unit
(** [pp_placement] formats a placement on one device as the device's name,
    and others as the placement that builds them over the set, as
    [on [CUDA:0; CUDA:1]], [split ~axis:0 [CUDA:0; CUDA:1]], or a mesh's
    grid ({!Grid.pp}). *)

(** {1:meshes Meshes} *)

val mesh_v : by:string -> 'd t -> (string * int) list -> 'd mesh
(** [mesh_v ~by s axes] lays [s]'s devices in row-major order over the named
    [axes]. Raises [Invalid_argument] naming [by] unless the extents are
    positive and multiply to [count s] and the names are distinct. *)
