(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The firmware a GPU boots with, read from its files.

    Three images per family, each a file of linux-firmware's [nvidia/] pinned by
    digest: the GSP's firmware, an ELF object whose sections hold its image and
    a signature per chip family; the GSP's bootloader; and the image that starts
    the GSP, the booter (Ampere, Ada) or the FMC (Blackwell). The booter and the
    bootloader are NVIDIA's firmware containers (nouveau's [nvfw/fw.h] and
    [hs.h]); the FMC is an ELF object.

    Reading checks the containers and points into the files: a section is a
    range of its file, which the boot copies once into the memory the GPU reads.
*)

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
    ucode ID come from the container's patch metadata, as NVIDIA's RM reads them
    (s_allocateUcodeFromBinArchive). The signature patched in is the one the
    header's patch index names; NVIDIA's RM picks it by SEC2's fuse version
    instead (s_patchBooterUcodeSignature), which this library does not read. *)

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
