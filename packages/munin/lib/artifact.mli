(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Artifact versions.

    An {e artifact} is a name, such as ["mnist-ckpt"], with {e versions}
    numbered from [1]. A run publishes a version with
    {!Session.log_artifact}: a file or a directory, stored once as a blob
    named by its digest, the run that produced it, the step it was
    produced at, and metadata. A version never changes, and a deleted
    version's number names nothing again. An {e alias}, such as ["best"],
    names one version of an artifact and moves when set again
    ({!Store.set_alias}). A run that used a version records it in its own
    log ({!Session.use_artifact}). {!Store_format.section-artifacts} specifies
    the files.

    A value of type {!t} is one version as read from its file. *)

type t = Version.t
(** The type for artifact versions. *)

val name : t -> string
(** [name a] is [a]'s artifact name. *)

val version : t -> int
(** [version a] is [a]'s number, from [1]. *)

val digest : t -> string
(** [digest a] is the digest of [a]'s content, [blake2b-256:] followed by
    64 hexadecimal digits ({!Store_format.section-blobs}). *)

val path : t -> string
(** [path a] is the absolute path of [a]'s blob, a file or a directory.
    Every version with the same content shares it, so the program must
    not change it. *)

val size_bytes : t -> int
(** [size_bytes a] is the size of [a]'s content, in bytes; for a directory,
    the sum of its files' sizes. *)

val metadata : t -> (string * Value.t) list
(** [metadata a] is what the producing run attached to [a], in the order it
    gave. *)

val run : t -> string
(** [run a] is the id of the run that produced [a], which stays even after
    that run is deleted. *)

val step : t -> int option
(** [step a] is the step the producing run gave for [a], if it gave one. *)

val created_at : t -> float
(** [created_at a] is the time [a] was published. *)

