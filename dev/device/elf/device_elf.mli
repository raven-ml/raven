(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ELF objects laid out in one image.

    Host programs, GPU programs and firmware come as 64-bit little-endian ELF
    objects. Reading one lays out its {e image}: the contents of its allocated
    program sections ([SHT_PROGBITS] with [SHF_ALLOC]), its code and data. If
    any of these sections has an address, each goes at its address, as in an
    executable. Otherwise each follows the image's end in section order, at its
    alignment, as in a relocatable object. The bytes between sections are zero.
    Other sections, such as [.bss], the symbol and string tables or debugging
    information, keep their contents apart from the image.

    The object's symbols and relocations come out in image offsets, so a loader
    that copies the image to address [base] finds an offset [o] at [base + o].
    Applying a relocation is the loader's: its kinds, the width of the field it
    patches and the formula that fills it are the machine's.

    In this module an {e offset} is an image offset unless said otherwise. *)

(** {1:objects Objects} *)

(** The type for where a symbol is. *)
type place =
  | Undefined
      (** Not in the object: the object refers to it by its name, which another
          object or the loader defines ([SHN_UNDEF]). *)
  | Absolute of int  (** A value the layout does not move ([SHN_ABS]). *)
  | Image of { section : int; offset : int }
      (** In a section the image holds: the section's index in [sections], and
          the symbol's offset. *)
  | Outside of int
      (** Defined where the image holds no bytes of its section: in a section
          the image does not hold, such as [.bss], past the end of its section,
          or at a special index other than [SHN_UNDEF] and [SHN_ABS], such as
          [SHN_COMMON]. The integer is that section index. *)

type symbol = {
  name : string;  (** Its name, [""] for a nameless one. *)
  place : place;  (** Where it is. *)
}
(** The type for the entries of a symbol table. *)

type section = {
  name : string;  (** Its name, [""] in an object without section names. *)
  kind : int;  (** Its type, [sh_type]. *)
  flags : int;  (** Its flags, [sh_flags]. *)
  offset : int option;  (** Its offset, if the image holds it. *)
  size : int;
      (** Its size in memory, [sh_size], which a section with no bytes in the
          object, such as [.bss], has too. *)
  contents : string;
      (** Its bytes in the object, [""] for a section with none. *)
}
(** The type for sections. *)

type relocation = {
  offset : int;  (** The offset of the field it patches. *)
  kind : int;  (** Its type, [ELF64_R_TYPE (r_info)]. *)
  addend : int;
      (** Its addend, [r_addend]. An [SHT_REL] entry has none and holds [0]: its
          addend is held in the place it patches, in the form its [kind] gives.
      *)
  symbol : symbol;
      (** The symbol whose value it uses, from the symbol table its section
          links to. A relocation that names no symbol uses
          [{ name = ""; place = Absolute 0 }]. *)
}
(** The type for relocations. *)

type t = private {
  kind : int;  (** The object's type, [e_type]: [1] for a relocatable one. *)
  machine : int;  (** Its machine, [e_machine]. *)
  os_abi : int;  (** Its operating system's ABI, [EI_OSABI]. *)
  abi_version : int;
      (** The version of that ABI, [EI_ABIVERSION], whose meaning is [os_abi]'s.
      *)
  flags : int;  (** Its machine's flags, [e_flags]. *)
  image : string;  (** Its image. *)
  sections : section iarray;
      (** Every section by index, from the null section at index [0]. *)
  symbols : symbol iarray;
      (** Every entry of its symbol table by index, from the null symbol at
          index [0]: [.symtab], or [.dynsym] when the object has no [.symtab],
          as an executable stripped of it. *)
  relocations : relocation list;
      (** The relocations that patch the image, in the order of their sections
          and then of their entries. Those of sections that do not occupy memory
          ([SHF_ALLOC] clear), such as debugging information's, are left out. *)
}
(** The type for objects laid out in their image.

    Every [offset] of a section, an [Image] place and a relocation lies in
    \[[0];[String.length image]\], and a relocation's is less than it. A section
    the image holds spans its [size] bytes there. An [Image] place's [section]
    is such a section. *)

(** {1:reading Reading} *)

val of_string : ?align:int -> string -> (t, string) result
(** [of_string ~align obj] is the object [obj] laid out in its image. In an
    object whose sections have no addresses, each section goes at a multiple of
    [align] and of its own alignment. [align] defaults to [1].

    The result is [Error msg] if [obj] is not a well-formed 64-bit little-endian
    ELF object: if a part of it lies past its end, if it refers to a section,
    symbol or name it does not have, if a section's alignment is not a power of
    two, if two sections overlap in the image, if its image would be longer than
    [Sys.max_string_length], if a relocation's offset lies past its section's
    end, or if a relocation patches memory the image does not hold. [msg] says
    which. Any other object is [Ok].

    Raises [Invalid_argument] if [align] is not a positive power of two. *)

val symbol : t -> string -> int option
(** [symbol o name] is the offset of the first symbol of [o.symbols] named
    [name] whose place is {!constructor-Image}, if any. A nameless symbol is
    never found. *)
