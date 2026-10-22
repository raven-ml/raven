(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** FWSEC, the ucode of an Ampere or Ada GPU's VBIOS that sets up the GPU's
    protected memory at a boot and puts back the VBIOS's own applications at
    a stop, found by walking the VBIOS's bytes. Every function but {!read} and
    a walk of what it gives is pure. *)

val window : int
(** [window] is the size of the PROM window, 1 MiB, which holds every FWSEC
    image. *)

type rom
(** The type for a VBIOS's bytes: those of its chain of images, or all of
    them, at most {!window}, where the chain cannot be walked or passes
    them. *)

val of_string : string -> rom
(** [of_string s] is the VBIOS of the ROM's bytes [s]. *)

val read : Chip.t -> rom
(** [read c] is the GPU's VBIOS, read through the PROM window of its registers
    as a walk reaches its bytes, while [c]'s registers stay mapped. *)

type fwsec = {
  image : string;
      (** The image, its code then its data, its production signature in
          place, and its command and the command's arguments zero until
          {!patch} sets them. *)
  cmd_at : int;  (** The offset in [image] of its command. *)
  args_at : int;  (** The offset in [image] of the command's arguments. *)
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

val fwsec : rom -> (fwsec, string) result
(** [fwsec rom] is the first production FWSEC of the VBIOS [rom], the last of
    its descriptor's signatures in place. [Error] names what the ROM lacks:
    valid images, the BIT table, a production FWSEC with a version 3
    descriptor, or the DMEM mapper, or a structure that points past its
    end. *)

(** The type for the commands FWSEC runs. *)
type command =
  | Frts of int
      (** [Frts frts] sets up the protected memory (WPR2), with the FRTS
          region at the byte [frts] of the GPU's memory, at a boot. *)
  | Sb
      (** [Sb] puts back the VBIOS's own applications that a boot stopped, at
          a stop, before the protected memory is taken down. *)

val patch : fwsec -> command -> fwsec
(** [patch f c] is [f] ready to run the command [c] with its arguments, the
    VBIOS read from the GPU's own ROM. *)
