(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ELF objects laid out in one image.

    Host programs, GPU programs and firmware come as ELF objects. An object
    holds sections and relocations. A section is a run of bytes, such as
    code, data or a symbol table. A relocation names a field that a loader
    patches with a value computed from a symbol's address. To load an
    object, a program needs the bytes each section puts in memory, where
    each goes, and the symbols and relocations that refer to them.

    {!of_string} reads a 32- or 64-bit little-endian object and lays out its
    {e image}: the sections a loader's memory holds, by default its code and
    data ({!allocated}), each at an offset from the image's start. Each
    section the image holds puts its bytes of the object at its offset, and
    the rest of the image is zero. Sections that have addresses, as in an
    executable, keep the distances between their addresses, and those of a
    relocatable object follow each other at their alignments ({!reading}).
    Other sections, such as debugging information, stay out of the image.

    Reading copies no section's bytes, and every position it gives is an
    image offset, so a loader that writes the image at address [base] finds
    the offset [o] at [base + o]. In this module an {e offset} is an image
    offset unless said otherwise. {!of_string} returns [Error] for an object
    it cannot read.

    A load goes like this:
    {ol
    {- {!of_string} reads the object and lays out its image.}
    {- The loader takes {!field-size} bytes of its destination, at an
       address that is a multiple of {!field-align}.}
    {- It writes the image there.}
    {- It looks up its entry points with {!val-symbol}.}
    {- It applies each relocation by its machine's formula ({!loading}).}}

    The loader does the steps after the first, since only it knows its
    destination. For instance, a loader that loads into [bytes] writes the
    image this way:
    {[
    let image (o : Rig_elf.t) =
      let b = Bytes.make o.size '\000' in
      let put (s : Rig_elf.section) =
        match s.offset with
        | Some off -> Bytes.blit_string o.file s.at b off s.length
        | None -> ()
      in
      Iarray.iter put o.sections;
      b
    ]}

    {b References.}
    {ul
    {- The
       {{:https://www.sco.com/developers/gabi/latest/contents.html}System V
        ABI}, chapter 4, Object Files: headers, sections, symbols and
       relocation entries.}
    {- A machine's processor supplement to it, such as the
       {{:https://gitlab.com/x86-psABIs/x86-64-ABI}x86-64 psABI}: its
       relocation types and their formulas.}} *)

(** {1:objects Objects}

    A read object [o] keeps these invariants:
    {ul
    {- A section [s]'s bytes lie in the object,
       [s.at + s.length <= String.length o.file], and no two sections share a
       byte of it.}
    {- A section [s] with [s.offset = Some off] lies in the image,
       [off + s.size <= o.size], at a multiple of its alignment
       ([sh_addralign], where above [1]), which divides [o.align]. It has all
       its bytes in the object, [s.length = s.size], or none,
       [s.length = 0], as an [SHT_NOBITS] one. No two such sections share a
       byte of the image.}
    {- A place [Image { section = i; offset }] names a section [s], index [i]
       of [o.sections], with [s.offset = Some off] and
       [off <= offset <= off + s.size].}
    {- A relocation's [offset] is less than [o.size]. One whose addend is
       [Implicit { at; length }] has [1 <= length] and
       [at + length <= String.length o.file].}} *)

(** The type for where a symbol is. *)
type place =
  | Undefined
      (** Not in the object. The object refers to it by its name, which
          another object or the loader defines ([SHN_UNDEF]). *)
  | Absolute of int  (** A value the layout does not move ([SHN_ABS]). *)
  | Image of { section : int; offset : int }
      (** In a section the image holds: the section's index in
          {!field-sections}, and the symbol's offset. *)
  | Outside of int
      (** Defined, but not in the image. The symbol is in a section the image
          does not hold, or before the start or past the end of its section,
          or at a special section index other than [SHN_UNDEF] and [SHN_ABS],
          as a common symbol is ([SHN_COMMON]). The integer is the symbol's
          section index. *)

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
      (** The bytes it takes in memory, and in the image if the image holds
          it, [sh_size]. A section with no bytes in the object, such as
          [.bss], has a size too. *)
  at : int;
      (** Where its bytes start in {!field-file}, [sh_offset]; [0] for the
          null section and an [SHT_NOBITS] one. *)
  length : int;
      (** The bytes it has in {!field-file}, which a loader copies. It is
          [size], or [0] for a section with none ([SHT_NOBITS] and the null
          section), whose bytes in the image are zeros. *)
}
(** The type for sections. *)

(** The type for a relocation's addend. *)
type addend =
  | Explicit of int  (** In the entry, [r_addend], as [SHT_RELA] ones hold. *)
  | Implicit of { at : int; length : int }
      (** In the field the relocation patches, as [SHT_REL] ones hold. The
          field's width and encoding are its type's, which the machine
          defines. The field starts at [at] in {!field-file}, and the section
          it patches has [length] bytes there from [at] on, so a field of [w]
          bytes lies in it iff [w <= length]. *)

type relocation = {
  offset : int;  (** The offset of the field it patches. *)
  kind : int;  (** Its type, [ELF64_R_TYPE (r_info)]. *)
  addend : addend;  (** Its addend. *)
  symbol : symbol;
      (** The symbol whose value it uses, from the symbol table its relocation
          section links to. A relocation that names no symbol uses
          [{ name = ""; place = Absolute 0 }]. *)
}
(** The type for relocations. *)

type t = private {
  bits : int;
      (** The object's class, [EI_CLASS]: [32] or [64], the width of its
          addresses. *)
  kind : int;  (** Its type, [e_type]: [1] for a relocatable one. *)
  machine : int;  (** Its machine, [e_machine]. *)
  os_abi : int;  (** Its operating system's ABI, [EI_OSABI]. *)
  abi_version : int;
      (** The version of that ABI, [EI_ABIVERSION], whose meaning is
          [os_abi]'s. *)
  flags : int;  (** Its machine's flags, [e_flags]. *)
  address : int;
      (** The address its image starts at. It is the lowest address of a
          section the image holds, rounded down to the largest alignment
          among them, or [0] when no section the image holds has an address.
          An address [a] of the object is the offset [a - o.address]. *)
  align : int;
      (** The alignment its image asks of the address a loader writes it at.
          It is the largest of [1], the alignment ([sh_addralign]) of each
          section the image holds and, when no section the image holds has an
          address, [align]. A power of two, of which {!field-address} is a
          multiple. *)
  file : string;  (** The object: [obj] itself, as {!of_string} read it. *)
  size : int;
      (** The length of its image, in bytes, from [0] to [max_int]. A
          corrupted address can make it any of these, whatever the object's
          length. A loader must check it against its destination before it
          allocates. *)
  sections : section iarray;
      (** Every section by index, from the null section at index [0]. *)
  symbols : symbol iarray;
      (** Every entry of its symbol table by index, from the null symbol at
          index [0]. The table is [.symtab], or [.dynsym] when the object has
          no [.symtab], such as a stripped executable. Empty if it has
          neither. *)
  relocations : relocation list;
      (** The relocations that patch the image, in the order of their
          sections and then of their entries. Relocations that patch a section
          that does not occupy memory ([SHF_ALLOC] clear), such as debugging
          information, are left out. A relocation section that names no
          section ([sh_info = 0]), such as a dynamic one, patches the image at
          its entries' addresses ([r_offset]) less {!field-address}. *)
}
(** The type for objects laid out in their image ({!objects}). *)

(** {1:reading Reading}

    The image holds the sections that {!of_string}'s [held] selects, laid
    out in one of two ways:
    {ul
    {- If one of them has a nonzero address ([sh_addr]), each goes at its
       address less {!field-address}, as in an executable whose image starts
       with its first section.}
    {- Otherwise each follows the image's end in section order, at the first
       offset at or past it that is a multiple of the [align] given to
       {!of_string} and of its alignment ([sh_addralign]), as in a
       relocatable object.}}
    The image ends where its last section ends.

    {!of_string} returns [Error msg] if [obj] is not a 32- or 64-bit
    little-endian ELF object, or if:
    {ul
    {- a part of it lies past its end, or a 64-bit field holds a value an
       [int] cannot;}
    {- its section headers are shorter than its class's, 40 or 64 bytes
       ([e_shentsize]);}
    {- it refers to a section, symbol or name it does not have, or links a
       section to one of the wrong type, as a symbol table linked to names
       that are not a string table is;}
    {- a section's alignment is not a power of two, or a section the image
       holds is at an address that is not a multiple of it;}
    {- two sections share bytes of the object, or of the image;}
    {- its names together are longer than it, which only names that share
       bytes of their string table can be;}
    {- its image would be longer than [max_int] bytes;}
    {- a relocation's offset lies outside the section it patches, or the
       image does not hold that section although it is allocated;}
    {- a relocation that names no section it patches ([sh_info = 0])
       patches an address outside the image, or in an allocated section the
       image does not hold, other than a thread-local one without bytes;}
    {- a relocation without an addend in its entry ([SHT_REL]) patches bytes
       the object does not have, those of a section without bytes or between
       sections.}}
    [msg] names the condition. *)

val allocated : machine:int -> section -> bool
(** [allocated ~machine s] is [true] iff [s], a section of an object for
    [machine] ([e_machine]), is code or data a loader's memory holds. That is
    an allocated ([SHF_ALLOC]) section that is one of:
    {ul
    {- a program section ([SHT_PROGBITS]);}
    {- a section without bytes ([SHT_NOBITS]), other than a thread-local one
       ([SHF_TLS], [.tbss]), which is a template each thread copies and takes
       no memory of its own;}
    {- on x86-64 ([EM_X86_64], [62]), an unwind table ([SHT_X86_64_UNWIND],
       [.eh_frame]).}}
    The sections a dynamic loader reads, such as [.dynamic], are left
    out. *)

val of_string :
  ?align:int -> ?held:(section -> bool) -> string -> (t, string) result
(** [of_string ~align ~held obj] is the object [obj] laid out in its image
    ({!reading}), which holds the sections [s] with [held s]. It is [Error]
    for an object that does not read ({!reading}), and [Ok] for any other.

    [held] defaults to {!allocated} for [obj]'s machine. A loader whose
    memory holds less passes its own, as a cubin's loader leaves out the
    GPU's on-chip shared memory ([.nv.shared.name]). [held] sees each section
    before the layout, with [offset = None]. [align] defaults to [1].

    Reading takes memory linear in [obj]'s length, and time linear in its
    length plus a sort of its sections, whatever {!field-size} is.

    Raises [Invalid_argument] if [align] is not a positive power of two. *)

val symbol : t -> string -> int option
(** [symbol o name] is [Some offset] for the first symbol of [o.symbols], by
    index, named [name] whose place is [Image { offset; _ }], and [None] if
    there is none. [symbol o ""] is [None]. It takes time linear in the number
    of symbols, so a loader that looks up many names walks {!field-symbols}
    once. *)

(** {1:loading Loading}

    Symbols and relocations come out in image offsets. For a loader that
    writes the image at address [base], the value of a relocation's symbol
    is:
    {[
    let value ~base (r : Rig_elf.relocation) =
      match r.symbol.place with
      | Image { offset; _ } -> Some (base + offset)
      | Absolute v -> Some v
      | Undefined | Outside _ -> None
    ]}

    The loader applies each relocation, since its kinds, the width of the
    field it patches and the formula that fills it are the machine's. A
    formula that adds the load address to an address of the object, as
    [R_X86_64_RELATIVE]'s [B + A] does, takes [B = base - o.address], how
    far the image moved from its link address. *)
