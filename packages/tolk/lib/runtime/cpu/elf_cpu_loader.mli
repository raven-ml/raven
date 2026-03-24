(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** CPU ELF relocation loader.

    This module turns an {!Elf.t} relocatable object into executable machine
    code for the CPU JIT runtime. Loading is split into two phases:

    + {!load} parses the ELF bytes, resolves external symbols, and collects
      relocations. No final addresses are needed yet.
    + {!link} patches all relocations against a concrete load base and returns
      ready-to-execute bytes.

    The supported relocation types are the subset of x86-64 and AArch64 used by
    the Tolk JIT backend:

    - x86-64: [R_X86_64_PC32], [R_X86_64_PLT32].
    - AArch64: [R_AARCH64_ADR_PREL_PG_HI21], [R_AARCH64_ADD_ABS_LO12_NC],
      [R_AARCH64_CALL26], [R_AARCH64_JUMP26], [R_AARCH64_LDST16_ABS_LO12_NC],
      [R_AARCH64_LDST32_ABS_LO12_NC], [R_AARCH64_LDST64_ABS_LO12_NC],
      [R_AARCH64_LDST128_ABS_LO12_NC].

    See {!Elf} for the underlying object parser. *)

(** {1:types Types} *)

type t
(** A prepared CPU image awaiting relocation at a concrete load address. Holds
    the flat image bytes, resolved relocation entries, and the entry-point
    offset. *)

(** {1:loading Loading} *)

val load : link_symbol:(string -> nativeint) -> entry:string -> Bytes.t -> t
(** [load ~link_symbol ~entry obj] parses ELF object bytes [obj] and prepares a
    CPU image for later linking.

    [link_symbol name] resolves external (undefined) symbols to their runtime
    addresses. It is called once per undefined symbol reference during loading.
    Defined symbols are resolved from the ELF section layout.

    [entry] names the symbol whose image offset becomes the entry point (see
    {!entry_offset}).

    Raises [Invalid_argument] if [entry] is missing or undefined in the symbol
    table.

    Raises [Invalid_argument] if [obj] is not a valid ELF object (propagated
    from {!Elf.load}). *)

(** {1:querying Querying} *)

val alloc_size : t -> int
(** [alloc_size t] is the number of bytes needed to materialize the final
    executable image. This is at least the flat image size, plus conservative
    slack for AArch64 branch trampolines that {!link} may emit for out-of-range
    [CALL26] / [JUMP26] relocations (16 bytes per such relocation). *)

val entry_offset : t -> int
(** [entry_offset t] is the byte offset of the entry symbol within the image. *)

(** {1:linking Linking} *)

val link : base:nativeint -> t -> Bytes.t
(** [link ~base t] applies all relocations assuming the image will be loaded at
    address [base] and returns the final executable bytes.

    For AArch64 [CALL26] and [JUMP26] relocations whose target is outside the
    +/-128 MiB direct-branch range, a trampoline
    ([LDR X17, #8; BR X17; <8-byte absolute address>]) is appended to the image
    and the original branch is redirected to it.

    The returned {!Bytes.t} has length at most {!alloc_size} [t].

    Raises [Invalid_argument] if any relocation has an unsupported type. *)
