(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HDF5 files. {!Nx_io.Hdf5}, {!Nx_io.load_hdf5} and {!Nx_io.decode_hdf5}
    document them. A group or a dataset keeps its file's disk buffer or its
    partial file, and its header's messages, which its readers decode. *)

type t

module Group : sig
  type t

  val path : t -> string
  val attributes : t -> string list
  val attribute : t -> string -> Nx.host Nx.packed
end

val read : string -> Rig.Buffer.t -> Group.t list * t list
(** [read path file] is the groups and the datasets of the disk buffer
    [file] of [path].

    Raises [Failure] without the path as {!Nx_io.load_hdf5} states. *)

val decode : Partial.t -> (Group.t list * t list, (int * int) list) result
(** Raises [Failure] starting with the partial file's name as
    {!Nx_io.decode_hdf5} states. *)

val path : t -> string
val shape : t -> int array
val chunk : t -> int array option
val attributes : t -> string list
val attribute : t -> string -> Nx.host Nx.packed
val values : t -> Nx.host Nx.packed

val lookup3 : ?init:int -> string -> int

val window :
  t ->
  start:int array ->
  shape:int array ->
  (Nx.host Nx.packed, (int * int) list) result
(** Raises [Invalid_argument] naming [Nx_io.Hdf5.window], and [Failure]
    starting with the dataset's file, as {!Nx_io.Hdf5.window} states. *)
