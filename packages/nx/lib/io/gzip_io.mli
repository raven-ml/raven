(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** gzip decompression, the implementation of {!Nx_io.gunzip}.

    Decompression streams through one buffer, so its memory is bounded
    whatever the sizes of the files. *)

val gunzip : src:string -> dst:string -> unit
