(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** FWSEC, the ucode of an Ampere or Ada GPU's VBIOS that sets up its protected
    memory (private).

    The VBIOS is a chain of PCI expansion ROM images. Its BIT table points to
    the falcon data, whose ucode table names the FWSEC image by its application
    ID; a version 3 descriptor gives its layout and signature. FWSEC runs one
    command, which a DMEM mapper in its data names: FRTS, which reads the VBIOS
    into a 1 MiB region of the GPU's memory that the GSP's boot checks.

    Nothing covers the VBIOS with a digest: a ROM laid out otherwise than this
    walk reads it is an [Error]. Pure functions over the ROM's bytes. *)

val window : int
(** [window] is the size of the PROM window {!read} reads, 1 MiB, the most the
    RM reads ([s_getBaseBiosMaxSize_TU102]): no FWSEC image is larger. *)

val read : Chip.t -> string
(** [read c] is the GPU's VBIOS, read through the PROM window of its registers.
*)

type fwsec = {
  image : string;
      (** The image, its code then its data, patched to run FRTS, its production
          signature in place. *)
  imem_pa : int;  (** The falcon's address of its code. *)
  imem_va : int;  (** The virtual address of its code. *)
  imem_size : int;  (** The size of its code. *)
  dmem_pa : int;  (** The falcon's address of its data. *)
  dmem_size : int;  (** The size of its data. *)
  pkc : int;
      (** The offset in its data of its signature, which the falcon's boot ROM
          checks it with. *)
  engines : int;  (** The engines it runs on, as a mask. *)
  ucode : int;  (** Its ucode ID. *)
}
(** The type for FWSEC, ready to run. *)

val fwsec : string -> frts:int -> (fwsec, string) result
(** [fwsec rom ~frts] is the FWSEC of the VBIOS [rom], patched to set up the
    FRTS region at the byte [frts] of the GPU's memory. The walk is the RM's
    ([kernel_gsp_vbios_tu102.c], [kernel_gsp_fwsec.c]): the PCI expansion ROM
    images, by their PCI data structures and NVIDIA's extension of them, the BIT
    table, found by its signature and checksum, its falcon data, and the first
    production FWSEC entry of the ucode table. [Error] names what the ROM lacks:
    valid images, the BIT table, a production FWSEC with a version 3 descriptor,
    or the DMEM mapper, or a structure that points past its end. The signature
    in place is the last of the descriptor's. *)
