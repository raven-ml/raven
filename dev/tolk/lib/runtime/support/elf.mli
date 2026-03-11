(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Relocatable ELF object loading.

    Parses 64-bit little-endian ELF relocatable objects ([ET_REL]) and lays out
    their allocatable sections into a contiguous flat image. Section and
    relocation metadata is preserved for backend-specific loaders, but no
    machine-specific relocations are applied. *)

(** {1:types Types} *)

type section = private {
  name : string;
      (** The ELF section name (e.g. [".text"], [".data"], [".bss"]). *)
  addr : int;  (** Byte offset of the section within the flat {!image}. *)
  size : int;  (** Size of the section in bytes. *)
  content : Bytes.t;
      (** Section contents. For [SHT_NOBITS] sections (e.g. [.bss]), [content]
          is a zero-filled buffer of length {!size}. *)
}
(** The type for sections after image layout. *)

type symbol = private {
  name : string;  (** The symbol name from the string table. *)
  shndx : int;
      (** Section header index the symbol belongs to. [0] for undefined symbols.
      *)
  value : int;
      (** Symbol value: byte offset from the start of the symbol's section. *)
}
(** The type for symbols from the object's symbol table. *)

type reloc = private {
  offset : int;
      (** Absolute byte offset within the flat {!image} where the relocation
          applies. *)
  symbol : symbol;  (** The referenced {!type-symbol}. *)
  r_type : int;
      (** Machine-specific relocation type (e.g. [R_AARCH64_CALL26],
          [R_X86_64_PC32]). *)
  addend : int;  (** Relocation addend. [0] for [SHT_REL] entries. *)
}
(** The type for relocations anchored at absolute image offsets. *)

type t
(** The type for a laid-out relocatable ELF object. Holds the flat image,
    resolved section addresses, symbols, and pending relocations. *)

(** {1:loading Loading} *)

val load : ?force_section_align:int -> Bytes.t -> t
(** [load ?force_section_align obj] parses ELF relocatable object [obj] and lays
    out its allocatable sections into a flat image.

    Sections with a fixed address ([sh_addr <> 0]) are placed first. Remaining
    allocatable sections are appended sequentially, each aligned to the maximum
    of the ELF section alignment and [force_section_align] (defaults to [1]).

    Raises [Invalid_argument] if [obj] is not a valid 64-bit little-endian ELF
    relocatable object. *)

(** {1:accessors Accessors} *)

val image : t -> Bytes.t
(** [image t] is the flat image built from allocatable sections. *)

val sections : t -> section array
(** [sections t] is all object sections in section-header order, with
    {!field-addr} set to their final image offsets. *)

val relocs : t -> reloc list
(** [relocs t] is the list of relocations with offsets resolved to absolute
    image positions. *)

(** {1:lookup Lookup} *)

val find_section : t -> string -> section option
(** [find_section t name] is the section named [name], if any. *)

val find_symbol_offset : t -> string -> int
(** [find_symbol_offset t name] is the absolute byte offset in {!image} of the
    defined symbol [name].

    Raises [Invalid_argument] if no symbol named [name] exists or if the symbol
    is undefined. *)
