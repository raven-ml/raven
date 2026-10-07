(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Firmware images, verified by digest.

    A driver that boots a GPU itself loads the GPU's firmware, exactly the
    images it was validated with: each has a pinned SHA-256 digest, and no file
    with another digest is loaded. Images come from files: those the system
    installs, and those in the user's cache, [$RAVEN_CACHE_ROOT/firmware] if
    [RAVEN_CACHE_ROOT] is set and [$XDG_CACHE_HOME/raven/firmware] otherwise,
    [XDG_CACHE_HOME] defaulting to [$HOME/.cache]. This module reads them and
    writes nothing. *)

val find : ?dir:string -> string -> sha256:string -> (string, string) result
(** [find name ~sha256] is the contents of the image [name], a path such as
    ["amdgpu/psp_13_0_0_sos.bin"] whose lowercase hexadecimal SHA-256 digest is
    [sha256], from the first of:
    - [dir/name], if [dir] is given, plain or compressed as below;
    - [/lib/firmware/name], plain or compressed as [name.zst] or [name.xz] where
      the system has [libzstd] or [liblzma], skipped when its digest differs,
      since distributions ship other versions;
    - the cache's [name], skipped when its digest differs.

    [Error why] if none holds the image, [why] naming it, its digest and the
    places looked in, or if the file of [dir] has another digest, naming the
    file. *)

val sha256 : string -> string
(** [sha256 s] is the lowercase hexadecimal SHA-256 digest of [s]. *)
