(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The code objects the library carries, packed into one archive that
    the C stubs embed.

    A code object's key is its target and module, as in
    ["gfx12-generic/cast.float32"]. The archive's index and {!targets} are
    read when the module initialises, which raises [Failure] if the archive
    is malformed: shorter than its index, or with a member outside it. *)

type member = { binary : string; kernels : string list }
(** The type for carried code objects: the object's bytes, and the names
    of the kernels it holds. *)

val member : string -> member option
(** [member key] is the code object of [key], if the library carries it. Its
    bytes are copied out of the archive once, so every call for [key]
    answers the same string. Calls from several domains take turns on a
    lock. Raises [Failure] if the bytes do not parse as a code object. *)

val targets : (string * string) list
(** [targets] is each target the library carries code objects for, with
    the bytes of one of them, sorted by target. *)
