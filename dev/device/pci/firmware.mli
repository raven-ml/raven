(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Firmware images, verified by digest.

    A driver that boots a GPU itself loads the GPU's firmware, exactly the
    images it was validated with: each has a pinned SHA-256 digest, and no file
    with another digest is loaded. Images come from local files: those the
    system installs, and those {!fetch} downloaded into the user's cache,
    [$RAVEN_CACHE_ROOT/firmware] if [RAVEN_CACHE_ROOT] is set and
    [$XDG_CACHE_HOME/raven/firmware] otherwise, [XDG_CACHE_HOME] defaulting to
    [$HOME/.cache]. Looking an image up downloads nothing. *)

val find :
  ?dir:string -> string -> sha256:string -> (string option, string) result
(** [find name ~sha256] is the contents of the image [name], a path such as
    ["amdgpu/psp_13_0_0_sos.bin"] whose lowercase hexadecimal SHA-256 digest is
    [sha256], from the first of:
    - [dir/name], if [dir] is given, plain or compressed as below;
    - [/lib/firmware/name], plain or compressed as [name.zst] or [name.xz] where
      the system has [libzstd] or [liblzma], skipped when its digest differs,
      since distributions ship other versions;
    - the cache's [name], skipped when its digest differs.

    [Ok None] if none holds the image. [Error why] names a file of [dir] with
    another digest. *)

val fetch : base_url:string -> string -> sha256:string -> (unit, string) result
(** [fetch ~base_url name ~sha256] makes the image [name] available to {!find}
    without a directory: unless [/lib/firmware] or the cache holds it already,
    it downloads [base_url ^ name] with the system's [libcurl], checks its
    digest and keeps it in the cache, which persists after the process. It needs
    network access and write access to the cache.

    [Error why] names what failed: a download that could not be made, one with
    another digest, or a cache that cannot be written. *)

val sha256 : string -> string
(** [sha256 s] is the lowercase hexadecimal SHA-256 digest of [s]. *)
