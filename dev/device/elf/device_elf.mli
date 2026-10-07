(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ELF objects laid out in one image.

    Host programs, GPU programs and firmware come as little-endian ELF objects,
    32- or 64-bit. Reading one with {!of_string} lays out its {e image}: the
    sections a loader's memory holds, by default its code and data
    ({!allocated}). A section's bytes in the image are its bytes in the object,
    or zeros for a section that has none there ([SHT_NOBITS]), such as [.bss].
    If one of these sections has a nonzero address ([sh_addr]), each goes at its
    address less {!field-address}, as in an executable whose image starts with
    its first section. Otherwise each follows the image's end in section order,
    at its alignment, as in a relocatable object. The image ends where its last
    section ends. Other sections, such as the symbol and string tables or
    debugging information, stay out of the image.

    Reading copies no section's bytes. A section's bytes are a range of the
    object, and the image is described by the sections it holds: it is
    {!field-size} bytes, each section [s] with [s.offset = Some off] puts at
    [off] the [s.length] bytes of the object from [s.at], and every other byte
    is zero. A loader writes this into its destination. Into [bytes], for
    instance:
    {[
    let image (o : Device_elf.t) =
      let b = Bytes.make o.size '\000' in
      let put (s : Device_elf.section) =
        match s.offset with
        | Some off -> Bytes.blit_string o.file s.at b off s.length
        | None -> ()
      in
      Iarray.iter put o.sections;
      b
    ]}

    The object's symbols and relocations come out in image offsets, so a loader
    that writes the image at address [base] finds an offset [o] at [base + o].
    The value of a relocation's symbol, for instance, is:
    {[
    let value ~base (r : Device_elf.relocation) =
      match r.symbol.place with
      | Image { offset; _ } -> Some (base + offset)
      | Absolute v -> Some v
      | Undefined | Outside _ -> None
    ]}
    Applying a relocation is the loader's: its kinds, the width of the field it
    patches and the formula that fills it are the machine's. A formula that adds
    the load address to an address of the object, such as [R_X86_64_RELATIVE]'s
    [B + A], takes [B = base - o.address], how far the image moved from its link
    address.

    In this module an {e offset} is an image offset unless said otherwise.

    {b References.}
    - The
      {{:https://www.sco.com/developers/gabi/latest/contents.html}System V ABI},
      chapter 4, Object Files: headers, sections, symbols and relocation
      entries.
    - A machine's processor supplement to it, such as the
      {{:https://gitlab.com/x86-psABIs/x86-64-ABI}x86-64 psABI}: its relocation
      types and their formulas. *)

(** {1:objects Objects} *)

(** The type for where a symbol is. *)
type place =
  | Undefined
      (** Not in the object: the object refers to it by its name, which another
          object or the loader defines ([SHN_UNDEF]). *)
  | Absolute of int  (** A value the layout does not move ([SHN_ABS]). *)
  | Image of { section : int; offset : int }
      (** In a section the image holds: the section's index in
          {!field-sections}, and the symbol's offset. *)
  | Outside of int
      (** Defined, but not in the image: in a section the image does not hold,
          such as debugging information; before the start or past the end of its
          section; or at a special section index other than [SHN_UNDEF] and
          [SHN_ABS], such as [SHN_COMMON]. The integer is the symbol's section
          index. *)

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
      (** The bytes it takes in memory, and in the image if the image holds it,
          [sh_size]. A section with no bytes in the object, such as [.bss], has
          one too. *)
  at : int;
      (** Where its bytes start in {!field-file}, [sh_offset]; [0] for the null
          section and an [SHT_NOBITS] one. *)
  length : int;
      (** The bytes it has in {!field-file}, which a loader copies: [size], or
          [0] for a section with none ([SHT_NOBITS] and the null section), whose
          bytes in the image are zeros. *)
}
(** The type for sections. *)

type relocation = {
  offset : int;  (** The offset of the field it patches. *)
  kind : int;  (** Its type, [ELF64_R_TYPE (r_info)]. *)
  addend : int;
      (** Its addend, [r_addend]. An entry of an [SHT_REL] section has none and
          holds [0]: its addend is in the field it patches, encoded as its
          [kind] says. *)
  symbol : symbol;
      (** The symbol whose value it uses, from the symbol table its relocation
          section links to. A relocation that names no symbol uses
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
  address : int;
      (** The address its image starts at: the lowest address of a section the
          image holds, rounded down to the largest alignment among them, or [0]
          when its sections follow the image's end. An address [a] of the object
          is the offset [a - o.address]. *)
  file : string;  (** The object: [obj] itself, as {!of_string} read it. *)
  size : int;
      (** The length of its image, in bytes, from [0] to [max_int]. A corrupted
          address can make it any of these, whatever the object's length, so a
          loader checks it against what its destination holds before allocating.
      *)
  sections : section iarray;
      (** Every section by index, from the null section at index [0]. *)
  symbols : symbol iarray;
      (** Every entry of its symbol table by index, from the null symbol at
          index [0]: [.symtab], or [.dynsym] when the object has no [.symtab],
          such as a stripped executable. Empty if it has neither. *)
  relocations : relocation list;
      (** The relocations that patch the image, in the order of their sections
          and then of their entries. Relocations that patch a section that does
          not occupy memory ([SHF_ALLOC] clear), such as debugging information,
          are left out. A relocation section that names no section
          ([sh_info = 0]), such as a dynamic one, patches the image at its
          entries' addresses ([r_offset]) less {!field-address}. *)
}
(** The type for objects laid out in their image. For an object [o]:
    - A section [s]'s bytes lie in the object,
      [s.at + s.length <= String.length o.file], and no two sections share a
      byte of it.
    - A section [s] with [s.offset = Some off] lies in the image,
      [off + s.size <= o.size], at a multiple of its alignment, [sh_addralign],
      and has all its bytes in the object, [s.length = s.size], or none,
      [s.length = 0], such as an [SHT_NOBITS] one. No two such sections share a
      byte of the image.
    - A place [Image { section = i; offset }] names a section [s], index [i] of
      [o.sections], with [s.offset = Some off] and
      [off <= offset <= off + s.size].
    - A relocation's [offset] is less than [o.size]. *)

(** {1:reading Reading} *)

val allocated : section -> bool
(** [allocated s] is [true] iff [s] is code or data a loader's memory holds: an
    allocated ([SHF_ALLOC]) program section ([SHT_PROGBITS]) or section without
    bytes ([SHT_NOBITS]), other than a thread-local one without bytes
    ([SHF_TLS], [.tbss]), a template each thread copies, which takes no memory
    of its own. Other allocated sections, such as [.dynamic], [.dynsym], notes
    or [.init_array], hold what a dynamic loader reads to link and start a
    program. *)

val of_string :
  ?align:int -> ?held:(section -> bool) -> string -> (t, string) result
(** [of_string ~align ~held obj] is the object [obj] laid out in its image.

    The image holds the sections [s] with [held s]. [held] defaults to
    {!allocated}; a loader whose memory holds less passes its own, such as one
    for a format that marks memory of another kind allocated. [held] sees each
    section before the layout, its [offset] [None]. A section it does not hold
    stays out of the image, and its symbols are [Outside].

    When its sections follow the image's end, each goes at the first offset at
    or past it that is a multiple of [align] and of its alignment,
    [sh_addralign]. [align] has no effect when its sections go at their
    addresses, and defaults to [1].

    The result is [Error msg], [msg] saying which, if [obj] is not a 32- or
    64-bit little-endian ELF object, or if:
    - a part of it lies past its end, or a 64-bit field holds a value an [int]
      cannot;
    - its section headers are shorter than its class's, 40 or 64 bytes
      ([e_shentsize]);
    - it refers to a section, symbol or name it does not have, or links a
      section to one of the wrong type, such as a symbol table to names that are
      not a string table;
    - a section's alignment is not a power of two, or a section the image holds
      is at an address that is not a multiple of it;
    - two sections share bytes of the object, or of the image;
    - its names together are longer than it, which only names that share bytes
      of their string table can be;
    - its image would be longer than [max_int] bytes;
    - a relocation's offset lies outside the section it patches, or a relocation
      patches allocated memory the image does not hold: a section the image
      lacks, other than a thread-local one without bytes, which takes no memory,
      or an address before its start or past its end.

    Any other object is [Ok]. Reading takes memory linear in [obj]'s length, and
    time linear in it up to sorting its sections, whatever {!field-size} is.

    Raises [Invalid_argument] if [align] is not a positive power of two. *)

val symbol : t -> string -> int option
(** [symbol o name] is [Some offset] for the first symbol of [o.symbols], by
    index, named [name] whose place is [Image { offset; _ }], and [None] if
    there is none. [symbol o ""] is [None]. It takes time linear in the number
    of symbols; a loader that looks up many names walks {!field-symbols} once.
*)
