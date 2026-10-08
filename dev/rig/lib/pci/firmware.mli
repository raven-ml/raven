(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Firmware images, verified by digest.

    A driver that boots a GPU itself loads the GPU's firmware, exactly the
    images it was validated with: each has a pinned {!digest}, and no file with
    another digest is loaded. The driver says where images are; this module
    reads files and writes nothing. *)

val find : string list -> string -> digest:string -> (string, string) result
(** [find dirs name ~digest] is the contents of the first file [dir/name], for
    [dir] in [dirs] in order, whose {!digest} is [digest]. [name] is a path such
    as ["amdgpu/psp_13_0_0_sos.bin"]. A file with another digest is skipped,
    since systems ship other versions of an image. Files are read as they are: a
    compressed image is another file.

    [Error why] if no directory holds the image, [why] naming it, its digest,
    the directories, the files with another digest and the files that cannot be
    read, with their cause. *)

val digest : string -> string
(** [digest s] is the lowercase hexadecimal BLAKE2b digest of [s], 32 bytes
    long. *)
