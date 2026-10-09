(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Firmware images, verified by digest.

    A driver that boots a GPU itself loads the GPU's firmware, exactly the
    images it was validated with: each has a pinned {!digest}, and no file with
    another digest is loaded. The driver says where images are; this module
    reads files and writes nothing. *)

type image = {
  path : string;
      (** [Filename.concat dir name], [dir] the first directory that holds it.
      *)
  contents : string;  (** Its bytes, whose {!digest} is the one asked for. *)
}
(** The type for firmware images found. *)

val find : string list -> string -> digest:string -> (image, string) result
(** [find dirs name ~digest] is the first file [dir/name], for [dir] in [dirs]
    in order, whose {!digest} is [digest]. [name] is a path such as
    ["amdgpu/psp_13_0_0_sos.bin"]. A file with another digest is skipped, since
    systems ship other versions of an image. Files are read as they are: a
    compressed image is another file.

    An image found is kept for the life of the process, with its file's identity
    when read: its device, inode, size and modification time. A later find of
    the same file and digest gives it unread while the file keeps that identity,
    and reads and verifies it again once the identity changed. A file rewritten
    in place to the same size within its file system's timestamp resolution
    keeps its identity, and is not read again.

    [Error why] if no directory holds the image, [why] naming it, its digest,
    the directories, the files with another digest and the files that cannot be
    read, with their cause. *)

val digest : string -> string
(** [digest s] is the lowercase hexadecimal BLAKE2b digest of [s], 32 bytes
    long. *)
