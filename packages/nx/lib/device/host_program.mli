(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Host programs: ELF relocatable objects linked into executable memory. *)

val load :
  (binary:string -> entry:string -> (nativeint * (unit -> unit), string) result)
  option
(** [load] is [None] on a machine other than x86_64 and arm64. Otherwise
    [load ~binary ~entry] links the ELF relocatable object [binary] into new
    executable memory and is the address of its symbol [entry], with the
    function that frees that memory.

    [Error why] if [binary] is not an object for the host's machine, has no
    symbol [entry], has writable data, refers to a symbol that no library of the
    process defines, or has a relocation that cannot be applied. *)
