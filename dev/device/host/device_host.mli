(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Host programs: ELF objects linked into executable memory and called.

    A {e host program} is compiled code the process runs on its own cores: a
    relocatable ELF object, linked once into memory the process executes and
    called at its {e entry}, a function of the object, as the C function

    {v void f(void **buffers, const int64_t *values); v}

    given the addresses of its buffers and its values as 64-bit integers. An
    object is made by a C compiler, for instance with
    {v clang -c -O2 -fPIC --target=x86_64-none-unknown-elf f.c v}
    for an x86_64 host, and [--target=aarch64-none-unknown-elf -ffixed-x18] for
    an arm64 one, since macOS and Windows reserve the register [x18].

    A call may split a program's iterations into blocks that the host's cores
    run at once ({!type-split}). The threads are the process's pool of workers,
    which the host's other parallel code shares, so that together they never
    take more threads than the host has cores.

    Addresses are [int]s: a host address fits in 62 bits on the 64-bit hosts the
    library supports. {!link} and {!call} may be called from any domain at once.

    {b References.}
    - The {{:https://gitlab.com/x86-psABIs/x86-64-ABI}x86-64 psABI}, chapter
      4.4: relocation types and their formulas.
    - The
      {{:https://github.com/ARM-software/abi-aa/blob/main/aaelf64/aaelf64.rst}
       ELF for the Arm 64-bit Architecture}, section 5.7: relocation types,
      their formulas and fields, and the registers a veneer may use. *)

(** {1:programs Programs} *)

type t
(** The type for host programs. A program's code stays in memory while the
    program is reachable, and is unmapped once it is not. *)

val link : entry:string -> string -> (t, string) result
(** [link ~entry obj] is the program of the object [obj], linked into new
    executable memory and called at its function [entry]. See
    {{!linking}Linking} for the objects it links.

    The result is [Error msg], [msg] saying which, if:
    - [obj] is not a well-formed ELF object, or is not relocatable, or is for
      another machine than the host's;
    - an allocated section of [obj] is writable and not empty, such as a [.data]
      or [.bss] in use;
    - a section of [obj] asks for an alignment above the system's page;
    - [obj] has relocations without addends ([SHT_REL]) that patch an allocated
      section, or one of a type {{!relocations}Linking} does not list, or one
      whose value does not fit its field;
    - [entry] names no symbol of an executable section of [obj];
    - [obj] refers to a symbol that neither [obj], this library nor the process
      defines ({{!symbols}Symbols});
    - the process cannot map executable memory, with the system's reason;
    - the host's machine is neither x86_64 nor arm64. *)

val address : t -> int
(** [address p] is the address of [p]'s entry, which linked code calls through
    [device_host_call] ({!linking}). It is valid while [p] is alive: code that
    holds it must run only while a value it can reach, such as the record that
    also holds the calling program, holds [p]. *)

(** {1:calling Calling} *)

type split = {
  extent : int;  (** The iterations, [0] to [extent - 1]. *)
  blocks : int;  (** The number of blocks they are cut into. *)
  lo : int;  (** The index of the values that takes a call's first iteration. *)
  hi : int;  (** The index of the values that takes the one after its last. *)
}
(** The type for splits of a program's iterations into blocks that the host's
    cores run at once. For [e = extent] and [b = min blocks extent], block [i]
    holds the iterations [i * e / b] to [(i + 1) * e / b - 1], rounded down: the
    blocks cover the iterations, none is empty, and their sizes differ by at
    most one. *)

val call : ?split:split -> t -> int array -> int array -> unit
(** [call ~split p buffers values] calls [p]'s entry on the addresses [buffers]
    and the [values], each passed as a 64-bit integer, in the calling thread,
    and returns once it returns.

    The OCaml runtime is released while [p] runs, so other threads and domains
    go on, and several may call programs at once. [call] keeps [p] alive until
    it returns, and nothing else: the memory at [buffers] must stay valid until
    then.

    With [split], [p] is called on disjoint ranges of the iterations, each made
    of whole consecutive blocks, that together cover them: with
    [values.(split.lo)] the range's first iteration, [values.(split.hi)] the one
    after its last, and the other values as given. Each call has its own copy of
    the values. The calls run in any order and at once on at most {!workers}
    threads, so a call must not read what another writes. On one thread, one
    call covers every iteration; a split of no iteration calls [p] never. A
    split call waits while another domain's split call, or another job of the
    host's pool, runs.

    Raises [Invalid_argument] if [split.extent < 0], [split.blocks < 1],
    [split.lo] or [split.hi] is not an index of [values], or
    [split.lo = split.hi]. *)

val workers : unit -> int
(** [workers ()] is the most threads a split {!call} runs on: the host's cores
    that run compute-bound work at full speed. It is the process's performance
    cores on a Mac that reports them, and its cores elsewhere.
    [1 <= workers ()]. It is the same for the life of the process. *)

(** {1:linking Linking}

    [link] lays out the object's allocated sections ([SHF_ALLOC]), its code and
    read-only data, in one image, writes it once into memory mapped for it,
    applies its relocations, then makes the memory executable. The memory is
    never writable and executable at once: on arm64 macOS it is mapped with
    [MAP_JIT], whose write protection only the linking thread lifts while it
    writes; elsewhere it is mapped writable and made executable and read-only
    once written. The instruction caches are synchronized with the written code
    before [link] returns, and each thread synchronizes its instruction stream
    before it enters linked code.

    The object is 64-bit and little-endian, relocatable ([ET_REL]), for the
    host's machine ([EM_X86_64] or [EM_AARCH64]). It has no writable data: code
    that needs a variable gets it through its buffers.

    {2:relocations Relocations}

    The relocations are those a compiler emits for position-independent code of
    the small code model, with [-fPIC], that calls functions and reads its own
    data. For a relocation at the place [P] of the image, of the symbol whose
    address is [S], with the addend [A]:

    - x86_64: [R_X86_64_PC32], [S + A - P]; [R_X86_64_PLT32], [S + A - P], a
      call or jump that goes through a stub when [S] is out of its reach.
    - arm64: [R_AARCH64_ADR_PREL_PG_HI21], [Page(S + A) - Page(P)];
      [R_AARCH64_ADD_ABS_LO12_NC], and [R_AARCH64_LDST8_ABS_LO12_NC] to
      [R_AARCH64_LDST128_ABS_LO12_NC] for loads and stores of 8 to 128 bits, the
      low 12 bits of [S + A]; [R_AARCH64_CALL26] and [R_AARCH64_JUMP26],
      [S + A - P], through a stub when out of reach.

    A stub lies after the image for each symbol the object refers to and does
    not define, with a word that holds the symbol's address. An x86_64 stub
    jumps through the word with no register; an arm64 stub loads it into [x17],
    which a veneer may use.

    {2:symbols Symbols}

    A symbol the object refers to is, in this order:
    - its definition in the object;
    - [device_host_call], which this library defines for the code it links;
    - its definition in the libraries the process loaded into its global scope,
      such as the C and math libraries.

    A compiler also calls functions the source does not name, such as
    [__extendhfsf2] where the machine has no instruction for a 16-bit float
    conversion, or [memcpy] for a large copy. The object defines these itself,
    or they come from the process, with the process's convention: on x86_64
    Windows that convention is not the object's, so code for it defines every
    function the compiler calls by itself.

    {3:device_host_call [device_host_call]}

    {v
    void device_host_call(void (*f)(void **, const int64_t *),
                          void **buffers, const int64_t *values, int64_t n,
                          const int64_t *split);
    v}

    calls the program whose entry is at [f] ({!address}) on [buffers] and the
    [n] values [values], as {!call} does: unsplit for a [NULL] [split], and
    otherwise split by the four values at [split], [extent], [blocks], [lo] and
    [hi], in the order of {!type-split}'s fields. It returns once every call of
    [f] has. It assumes what {!call} checks: [0 <= extent], [1 <= blocks], and
    [lo] and [hi] are distinct indices below [n]. It is called from code that
    {!call} runs, or from a block of it, with the OCaml runtime released; called
    from a block of a split, it runs on that block's thread alone. It aborts the
    process if it finds no memory for a copy of more than 1024 values.

    {2:abi Calling conventions}

    Linked code follows its object's convention: System V AMD64 on x86_64,
    Windows included, and AAPCS64 on arm64. [call] calls the entry, and
    [device_host_call] is called, that way. A function of the process's
    libraries follows the platform's convention, so on x86_64 Windows linked
    code declares those it calls [__attribute__((ms_abi))]. Code for an ELF
    target has no stack probes: on Windows, which commits a thread's stack one
    guard page at a time, a frame above 4 KiB may fault.

    {1:platforms Platform support}

    [link] links code on x86_64 and arm64 hosts running Linux, macOS or Windows.
    On macOS, a program built with the hardened runtime needs the
    [com.apple.security.cs.allow-jit] entitlement (arm64) or
    [com.apple.security.cs.allow-unsigned-executable-memory] (x86_64) to map
    executable memory, and links fail with the
    [com.apple.security.cs.single-jit] or
    [com.apple.security.cs.jit-write-allowlist] entitlements. *)
