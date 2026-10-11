(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The firmware a GPU boots with, cut into the pieces its security
    processor (PSP) loads.

    The versions of a GPU's blocks name its images. An image is a header
    and its payload, and the PSP loads each piece of a payload as a
    firmware type. Every value is pure, and any domain may call it. *)

type piece = private { source : string; off : int; len : int }
(** The type for a checked byte range [off], [len] of an image [source].
    [off] and [len] are nonnegative and [off + len <= String.length source].
    A piece holds its whole image. *)

type t = {
  sos : (int * piece) list;
      (** The PSP's own components, by type: its OS (SOS), drivers, key
          database and table of contents. *)
  smu : (int list * piece) option;
      (** The power manager's image, on a GPU of GC 11 or later, which the
          PSP loads before it sets up its trusted memory region (TMR). *)
  pieces : (int list * piece) list;
      (** The other pieces, in the order the PSP loads them, each with the
          firmware types it loads as. *)
  starts : (string * int) list;
      (** The start address of each RS64 engine that runs a piece: ["PFP"],
          ["ME"], ["MEC"]. *)
  mec : int;  (** The version of the compute queues' firmware. *)
}
(** The type for a GPU's firmware. *)

val names : Discovery.t -> ((int * string) list, string) result
(** [names d] is the paths of the images the blocks of [d] name, in the order
    {!load} reads them, each with the block that names it, such as
    [(255, "amdgpu/psp_14_0_3_sos.bin")]. The result is [Error msg] if a
    block a boot needs is missing, naming the block. *)

val load :
  (string -> (string, string) result) -> Discovery.t -> (t, string) result
(** [load find d] reads each image of {!names}[ d] with [find path], and cuts
    it into pieces without copying their bytes. The result is [Error msg]:
    {ul
    {- as {!names};}
    {- with [find]'s message;}
    {- if an image's header is of a version this library does not read, or
       a header field or a piece lies outside the image, naming the
       image.}} *)
