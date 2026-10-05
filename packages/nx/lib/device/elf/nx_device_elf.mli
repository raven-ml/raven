(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Loading ELF objects into flat images.

    Host programs, GPU programs and firmware come as 64-bit little-endian ELF
    objects. This module lays an object's sections out in one image: sections
    with an address at it, the others appended in order at their alignment. It
    resolves symbols to image offsets and lists the relocations for the caller
    to apply, since their kinds are the machine's. *)

type section = {
  name : string;  (** Its name. *)
  kind : int;  (** Its type, [sh_type]. *)
  flags : int;  (** Its flags, [sh_flags]. *)
  offset : int;  (** Its offset in the image, for sections in it. *)
  size : int;
      (** Its size in memory, [sh_size], which a section with no bytes in the
          object, such as [.bss], has too. *)
  contents : string;  (** Its bytes in the object. *)
}
(** The type for sections. *)

(** The type for what a relocation refers to. *)
type target =
  | Offset of int  (** The image offset of a symbol the object defines. *)
  | Undefined of string  (** A symbol the object refers to by name alone. *)

type relocation = {
  at : int;  (** The image offset to patch. *)
  target : target;  (** The symbol it refers to. *)
  kind : int;  (** Its type, [ELF64_R_TYPE]. *)
  addend : int;  (** Its addend, [0] for [SHT_REL] entries. *)
}
(** The type for relocations. *)

type t = {
  kind : int;  (** The object's type, [e_type]: [1] for a relocatable one. *)
  machine : int;  (** Its machine, [e_machine]. *)
  image : string;  (** The laid-out image of the [SHT_PROGBITS] sections. *)
  sections : section list;  (** Every section, in order. *)
  symbols : (string * int) list;
      (** The defined symbols and their image offsets. *)
  symtab : (string * int) array;
      (** Every entry of the symbol table, by index: its name, [""] for a
          nameless one, and the index in [sections] of the section it is defined
          in, [0] for one undefined, absolute or common. *)
  relocations : relocation list;  (** In section order. *)
}
(** The type for loaded objects. *)

val load : ?align:int -> string -> t
(** [load obj] lays out the ELF object [obj], aligning appended sections to at
    least [align] (defaults to [1]).

    Raises [Failure] if [obj] is not a 64-bit little-endian ELF object or is
    truncated. *)

val symbol : t -> string -> int option
(** [symbol o name] is the image offset of the symbol [name] of [o]. *)
