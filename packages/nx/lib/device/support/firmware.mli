(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Firmware images, verified by digest.

    A runtime that boots a GPU itself loads the GPU's firmware, exactly the
    images it was validated with: each has a pinned SHA-256 digest, and no file
    with another digest is loaded. Images come from local files when they match,
    and are otherwise downloaded once into the user's cache. *)

val cache : unit -> string
(** [cache ()] is the directory downloaded images are kept in:
    [$RAVEN_CACHE_ROOT/firmware] if [RAVEN_CACHE_ROOT] is set, and
    [$XDG_CACHE_HOME/raven/firmware] otherwise, [XDG_CACHE_HOME] defaulting to
    [$HOME/.cache]. *)

val get :
  ?dir:string ->
  url:string ->
  string ->
  sha256:string ->
  (string, string) result
(** [get ~url name ~sha256] is the contents of the image [name], a path such as
    ["amdgpu/psp_13_0_0_sos.bin"] whose lowercase hexadecimal SHA-256 digest is
    [sha256], from the first of:
    - [dir/name], if [dir] is given, plain or compressed as below;
    - [/lib/firmware/name], plain or compressed as [name.zst] or [name.xz] where
      the system has [libzstd] or [liblzma], skipped when its digest differs,
      since distributions ship other versions;
    - [cache ()/name];
    - [url ^ name], downloaded with the system's [libcurl] and kept in
      [cache ()/name] when the cache can be written.

    [Error why] names the file and says what failed: a file of [dir] or a
    download with another digest, or a download that could not be made. *)

val sha256 : string -> string
(** [sha256 s] is the lowercase hexadecimal SHA-256 digest of [s]. *)
