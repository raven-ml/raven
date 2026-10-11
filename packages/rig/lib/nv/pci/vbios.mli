(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** FWSEC, the ucode of an Ampere or Ada GPU's VBIOS that sets up the GPU's
    protected memory, found by walking the VBIOS's bytes. Every function but
    {!read} is pure. *)

val window : int
(** [window] is the size of the PROM window {!read} reads, 1 MiB, which holds
    every FWSEC image. *)

val read : Chip.t -> string
(** [read c] is the GPU's VBIOS, read through the PROM window of its registers.
*)

type fwsec = {
  image : string;
      (** The image, its code then its data, set to run FRTS, its production
          signature in place. *)
  frts_at : int;
      (** The offset in [image] of the FRTS command's arguments, zero until
          {!patch} sets them. *)
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
(** The type for FWSEC. *)

val fwsec : string -> (fwsec, string) result
(** [fwsec rom] is the first production FWSEC of the VBIOS [rom], set to run
    FRTS, the last of its descriptor's signatures in place. [Error] names
    what the ROM lacks: valid images, the BIT table, a production FWSEC with
    a version 3 descriptor, or the DMEM mapper, or a structure that points
    past its end. *)

val patch : fwsec -> frts:int -> fwsec
(** [patch f ~frts] is [f] ready to run: its FRTS command sets up the FRTS
    region at the byte [frts] of the GPU's memory. *)
