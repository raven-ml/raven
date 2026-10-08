(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The firmware a GPU boots with: the images its blocks' versions name, each
    read by the digest this library pins, cut into the pieces its security
    processor (PSP) loads. Pure: any domain.

    An image is a header ([amdgpu_ucode.h]) and its payload; the PSP loads each
    piece of a payload as a firmware type ([psp_gfx_if.h]). *)

type t = {
  sos : (int * string) list;
      (** The PSP's own components, by type: its OS (SOS), drivers, key
          database, table of contents. *)
  smu : (int list * string) option;
      (** The power manager's image, which the PSP loads before its trusted
          memory region, on a GPU of GC 11 or later. *)
  pieces : (int list * string) list;
      (** The other pieces, in the order the PSP loads them, each with the
          firmware types it loads as. *)
  starts : (string * int) list;
      (** The start address of each RS64 engine that runs a piece: ["PFP"],
          ["ME"], ["MEC"]. *)
  mec : int;  (** The version of the compute queues' firmware. *)
}
(** The type for a GPU's firmware. *)

val pinned : (string * string) list
(** [pinned] is every firmware image an open may read: its path under a firmware
    directory, such as ["amdgpu/psp_13_0_0_sos.bin"], and its lowercase
    hexadecimal BLAKE2b-256 digest ({!Rig_pci.Firmware.digest}). *)

val names : Discovery.t -> (string list, string) result
(** [names d] is the paths of the images the blocks of [d] name, in the order
    {!load} reads them, such as ["amdgpu/psp_14_0_3_sos.bin"]. [Error msg] if a
    block a boot needs is missing, or its version names no pinned image, naming
    the block and its version. *)

val load :
  (string -> digest:string -> (string, string) result) ->
  Discovery.t ->
  (t, string) result
(** [load find d] reads each image of {!names}[ d] with [find path ~digest],
    [digest] its pinned digest, and cuts it into pieces. [Error msg] as
    {!names}, with [find]'s message, or if an image's header is of a version
    this library does not read, or places a piece outside the image, naming the
    image. *)
