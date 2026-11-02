(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Version files and blobs ({!Store_format.section-artifacts},
    {!Store_format.section-blobs}): the representation of {!Artifact.t}, and how
    {!Session} publishes and {!Store} reads, deletes and checks. *)

type t = {
  name : string;
  version : int;
  digest : string;
  path : string;
  size_bytes : int;
  metadata : (string * Value.t) list;
  run : string;
  step : int option;
  created_at : float;
}

val load :
  root:string -> string -> int -> [ `Version of t | `Deleted | `Missing ]
(** [load ~root name n] reads version [n] of [name]. A file that does not parse
    reads [`Missing]. *)

val numbers : root:string -> string -> int list
(** [numbers ~root name] is every number a file of [name] claims, deleted or
    not, sorted. *)

val names : root:string -> string list

val publish :
  root:string ->
  run:string ->
  step:int option ->
  metadata:(string * Value.t) list ->
  name:string ->
  string ->
  t
(** [publish] stores a version under the store's shared lock. *)

val tombstone : root:string -> t -> unit
val blob_dir : string -> string

val with_lock : string -> [ `Exclusive | `Shared ] -> (unit -> 'a) -> 'a
(** [with_lock root kind f] runs [f] holding the store's [lock]. *)

val tree : ?dst:string -> string -> string * int
(** [tree ?dst p] is the hexadecimal digest of the file or directory [p] and its
    size; with [dst], it also copies [p] there and syncs the copy. *)
