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

    Brands are phantom: nothing at run time reads ['d], and a value's arrays lie
    on its placement's devices whatever its brand. Two values can carry the same
    set at two brands only through {!rebrand}, whose one caller is the engine's
    computation of constants: a constant's operands, and its memo's keys at
    [unit]. This module is nx's one site that names nx.cpu.

    Errors are [Invalid_argument] naming the [by] the caller passes. *)

type host
(** The brand of {!host}. *)

type +'d t
(** The type for sets of brand ['d]. *)

type +'d placement
(** The type for placements over a set of brand ['d]. *)

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
(** [split ~by ~axis s] cuts [axis] into equal windows, one per device of [s],
    in order: [one s 0] for a set of one device. Raises [Invalid_argument]
    naming [by] if [axis] is negative or not below {!Nx_array.Layout.max_rank}.
*)

val mesh : by:string -> 'd mesh -> (int * string list) list -> 'd placement
(** [mesh ~by m cuts] cuts each axis [a] of [(a, names)] over the mesh axes
    [names], major first; a mesh axis no cut names holds copies. Raises
    [Invalid_argument] naming [by] for a name [m] lacks, an axis or a name in
    two cuts, or an axis that is negative or not below
    {!Nx_array.Layout.max_rank}. *)

val v : by:string -> 'd t -> Grid.t -> 'd placement
(** [v ~by s g] is [g] over [s]: [one s k] when [g] is device [k] alone, and
    [on s] when [g] is [s]'s devices in order with no cut. Raises
    [Invalid_argument] naming [by] if a device of [g] is not below [count s]. *)

val anywhere : 'd placement
(** [anywhere] is the placement of a constant: {!host}'s device, at every brand.
    Its {!set} is {!host}'s at every brand and its {!device} is [Some 0]. It is
    physically distinct from [one host 0], so a route tells a constant, which
    joins any placement, from a value on the host. Only the engine's routes and
    its computation of constants read it; [Nx.placement] of a constant answers
    it. *)

val set : 'd placement -> 'd t
val grid : 'd placement -> Grid.t

val device : 'd placement -> int option
(** [device p] is [Some k] iff [p] is its set's device [k] alone. *)

val equal : 'd placement -> 'd placement -> bool
(** [equal p q] compares physically, then sets by number and grids. *)

val window :
  by:string -> 'd placement -> int array -> int -> Nx_array.Move.range array
(** [window ~by p shape i] is the window of a value of [shape] that the [i]th
    device of [p] holds, in [Grid.devices (grid p)]'s order. Raises
    [Invalid_argument] naming [by] as {!Grid.window} answers [Error], and if [i]
    is not a position of [p]'s devices. *)

val with_leading_axis : 'd placement -> 'd placement
(** [with_leading_axis p] is [p] for a value that gains a new axis 0: each cut
    axis moves up by one, and the new axis is whole on every device. *)

val without_leading_axis : 'd placement -> 'd placement
(** [without_leading_axis p] is [p] for a value that loses its axis 0: a grid
    axis that cut it holds copies, and each other cut axis moves down by one. *)

val rebrand : 'd placement -> 'e placement
(** [rebrand p] is [p] at another brand. *)

val pp_placement : Format.formatter -> 'd placement -> unit
(** [pp_placement] formats {!anywhere} as [anywhere], a placement on one device
    as the device's name, and others as the placement that builds them over the
    set: [on [CUDA:0; CUDA:1]], [split ~axis:0 [CUDA:0; CUDA:1]], or a mesh's
    grid ({!Grid.pp}). *)

(** {1:meshes Meshes} *)

val mesh_v : by:string -> 'd t -> (string * int) list -> 'd mesh
(** [mesh_v ~by s axes] lays [s]'s devices in row-major order over the named
    [axes]. Raises [Invalid_argument] naming [by] unless the extents are
    positive and multiply to [count s] and the names are distinct. *)

val mesh_set : 'd mesh -> 'd t
