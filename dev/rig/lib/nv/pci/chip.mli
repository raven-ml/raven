(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A GPU's registers and identity, through its register BAR (private).

    Registers and their fields are the constants of [Defs], generated from
    NVIDIA's published register headers, per family where they differ, so a
    register this library writes exists for every family it boots. Accesses go
    through a {!Rig_pci.Window.t}, on this machine or another. The GPU's owner
    serializes calls. *)

(** The type for the GPU families this library boots. *)
type family =
  | Ampere  (** GA10x: the booter starts the GSP after FWSEC. *)
  | Ada  (** AD10x: as Ampere. *)
  | Blackwell  (** GB20x: the FSP starts the GSP from the FMC. *)

type t = private {
  fn : Rig_pci.Function.t;  (** The GPU's function. *)
  regs : Rig_pci.Window.t;  (** BAR 0, uncombined. *)
  family : family;
  implementation : int;
      (** The chip within its family, such as [2] for AD102. *)
}
(** The type for GPUs whose registers the process reaches. *)

val chip : int -> (family * int, string) result
(** [chip boot42] is the family and implementation of the chip whose
    [NV_PMC_BOOT_42] register holds [boot42], its architecture in bits 29:24 and
    its implementation in bits 23:20. It is [Error] naming the chip if it is
    none of GA102, GA103, GA104, GA106, GA107, AD102, AD103, AD104, AD106,
    AD107, GB202, GB203, GB205, GB206 and GB207, the chips whose boot this
    library lays out. *)

val of_function : Rig_pci.Function.t -> (t, string) result
(** [of_function fn] maps [fn]'s register BAR and reads which chip it is
    ({!chip}), changing nothing on the GPU. It is [Error] as {!chip}, or if the
    BAR cannot be mapped. *)

val name : t -> string
(** [name c] is the chip's name, such as ["AD102"]. *)

val memory : t -> (int, string) result
(** [memory c] is the size of the GPU's memory in bytes, as its firmware writes
    it once its boot after a reset ended ({!Falcon.wait_reset}). [Error] if it
    wrote none. *)

val booted : t -> bool
(** [booted c] is [true] iff the GPU's protected region of memory (WPR2) is up:
    something booted its GSP since its last reset. *)

val get : t -> int -> int
(** [get c r] reads the 32-bit register at offset [r] of BAR 0, such as
    [Defs.nv_pmc_boot_42]. *)

val set : t -> int -> int -> unit
(** [set c r x] writes the low 32 bits of [x] to the register at [r]. *)

val field : int * int -> int -> int
(** [field (lo, n) x] is the [n]-bit field of [x] from bit [lo], a field of
    [Defs]. *)

val wait : t -> string -> ms:int -> (unit -> bool) -> (unit, string) result
(** [wait c what ~ms f] waits for [f ()] ({!Rig_pci.Machine.wait}). [Error]
    names [what] if [ms] milliseconds passed, or the function's failure
    ({!Rig_pci.Function.failed}) if it or its machine failed. *)

val delay : t -> int -> unit
(** [delay c ms] waits [ms] milliseconds. *)

val bus_master : Rig_pci.Function.t -> bool -> unit
(** [bus_master fn on] turns the bus mastering of the GPU's function [fn] on or
    off. *)
