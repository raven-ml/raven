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

val read : Chip.t -> string
(** [read c] is the GPU's VBIOS, read through the PROM window of its registers.
*)

type fwsec = {
  image : string;
      (** The image, patched to run FRTS, its production signature in place. *)
  imem_pa : int;  (** The falcon's address of its code. *)
  imem_va : int;  (** The virtual address of its code. *)
  imem_size : int;  (** The size of its code. *)
  dmem_pa : int;  (** The falcon's address of its data. *)
  dmem_size : int;  (** The size of its data. *)
  signature : int;  (** The offset of its signature in [image]. *)
  engines : int;  (** The engines it runs on, as a mask. *)
  ucode : int;  (** Its ucode ID. *)
}
(** The type for FWSEC, ready to run. *)

val fwsec : string -> frts:int -> (fwsec, string) result
(** [fwsec rom ~frts] is the FWSEC of the VBIOS [rom], patched to set up the
    FRTS region at the byte [frts] of the GPU's memory. [Error] names what the
    ROM lacks: an expansion ROM, the BIT table, the falcon data, FWSEC, a
    version 3 descriptor, or the DMEM mapper. *)
