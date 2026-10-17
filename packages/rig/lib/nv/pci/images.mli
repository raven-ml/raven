(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The firmware a GPU boots with, read from its three files: the GSP's
    firmware, its bootloader, and the image that starts the GSP, the booter
    (Ampere, Ada) or the FMC (Blackwell). Reading checks the files'
    containers and points into the files, so that the boot copies each
    section once, into the memory the GPU reads. *)

type range = { contents : string; at : int; length : int }
(** The type for bytes of a file: the [length] bytes of the file's [contents]
    from [at]. *)

type bootloader = {
  image : range;  (** The image, the container's data. *)
  code : int;  (** The offset of its monitor code. *)
  data : int;  (** The offset of its monitor data. *)
  manifest : int;  (** The offset of its manifest. *)
}
(** The type for the GSP's bootloader. *)

type booter = {
  image : string;
      (** The image the falcon loads: the container's data, its production
          signature written at the place the container names. *)
  code : int * int;  (** The offset and size of its code in [image]. *)
  data : int * int;  (** The offset and size of its data. *)
  pkc : int;  (** Where in its data its signature is. *)
  engines : int;  (** The engines it runs on, as a mask. *)
  ucode : int;  (** Its ucode ID. *)
}
(** The type for the booter, a heavy-secure ucode SEC2 runs. Its engines and
    ucode ID come from the container's patch metadata. The signature patched
    in is the one the header's patch index names, whatever SEC2's fuse
    version. *)

type fmc = {
  fmc : range;  (** The image the FSP starts. *)
  hash : range;  (** Its SHA-384, which the FSP checks. *)
  signature : range;
  public_key : range;
}
(** The type for the FMC, from its ELF object's sections [image], [hash],
    [signature] and [publickey]. *)

type t = {
  gsp : range;  (** The GSP's image, its section [.fwimage]. *)
  signature : range;
      (** Its signature for the chip's family, its section
          [.fwsignature_<family>x], such as [.fwsignature_ad10x]. *)
  bootloader : bootloader;
  start : [ `Booter of booter | `Fmc of fmc ];
      (** The image that starts the GSP. *)
}
(** The type for a GPU's firmware. *)

val names : Chip.family -> string list
(** [names f] is the paths of the three files of family [f] under a firmware
    directory, GSP first. *)

val gsp_image : file:string * digest:string
(** [gsp_image] is the GSP's firmware, the first of {!names} for every family: its
    path under a firmware directory and its pinned digest. *)

val find : string list -> string -> (Rig_pci.Firmware.image, string) result
(** [find dirs file] is [file] in the first of the directories [dirs] that holds
    it with its pinned digest ({!Rig_pci.Firmware.find}). [file] is one of
    {!names}. *)

val parse :
  Chip.family ->
  gsp:string ->
  bootloader:string ->
  start:string ->
  (t, string) result
(** [parse f ~gsp ~bootloader ~start] is the firmware of family [f] whose three
    files ({!names}) hold [gsp], [bootloader] and [start]. It is [Error] saying
    what a file lacks if one is not laid out as its format says. *)
