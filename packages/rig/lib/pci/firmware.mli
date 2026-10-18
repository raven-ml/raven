(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Firmware images, verified by digest.

    A driver that boots a GPU itself loads the GPU's firmware, exactly the
    images the driver was validated with. Each image has a pinned {!digest},
    and no file with another digest is loaded. The driver says where images
    are, rig's own {!directory} among them; this module reads files and writes
    nothing. The [rig firmware] command fetches the pinned images into rig's
    directory. A directory of the caller's must stay apart from
    [/lib/firmware], since the kernel's drivers load their images from there
    and an image replaced there changes what they load.

    An image found is kept for the life of the process, with its file's identity
    when it was read: its device, inode, size and modification time. A later
    find of the same file and digest gives the kept image unread while the file
    keeps that identity, and reads and verifies the file again once the identity
    has changed. A file rewritten in place to the same size within its file
    system's timestamp resolution keeps its identity, and is not read again. Any
    domain may call {!find}. *)

type image = {
  path : string;
      (** [Filename.concat dir name], for the first directory [dir] that holds
          the image. *)
  contents : string;  (** Its bytes, whose {!digest} is the one asked for. *)
}
(** The type for firmware images found. *)

val find : string list -> string -> digest:string -> (image, string) result
(** [find dirs name ~digest] is the first file [dir/name], for [dir] in [dirs]
    in order, whose {!digest} is [digest]. [name] is a path such as
    ["amdgpu/psp_13_0_0_sos.bin"]. A file with another digest is skipped, since
    systems ship other versions of an image. Files are read as they are, so a
    compressed image is another file.

    [Error why] if no directory holds the image. [why] names the image, its
    digest, the directories, the files with another digest, and the files that
    cannot be read with their cause. *)

val digest : string -> string
(** [digest s] is the lowercase hexadecimal BLAKE2b digest of [s], 32 bytes
    long. *)

val directory : unit -> string option
(** [directory ()] is rig's firmware directory, the one [rig firmware] fills
    when given none: [$XDG_DATA_HOME/rig/firmware], or
    [$HOME/.local/share/rig/firmware] where [XDG_DATA_HOME] is unset, empty or
    relative. [None] where [HOME] is too. It is a data directory, since an
    open downloads nothing: an image removed from it stays missing. *)
