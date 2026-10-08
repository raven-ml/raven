(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Cubins: the ELF objects NVIDIA's compilers make for a GPU architecture.

    A cubin holds the machine code of one or more kernels. The kernel [name] is
    the code of the section [.text.name]. What a launch of it needs besides, its
    registers, stack, shared memory and constant banks, the cubin records in
    other sections and in the attributes of its [.nv.info] sections.

    A cubin is uploaded as its {e image}: {!size} bytes, which are the image of
    its object {!elf} ({!Device_elf}), zeros up to [size] for the GPU's
    instruction prefetch, which may read past the code, and {!patches} written
    over them. A loader builds the image of an upload at [base] as:
    {[
    let image c ~base =
      let o = Cubin.elf c in
      let b = Bytes.make (Cubin.size c) '\000' in
      let put (s : Device_elf.section) =
        match s.offset with
        | Some off -> Bytes.blit_string o.file s.at b off s.length
        | None -> ()
      in
      Iarray.iter put o.sections;
      let patch (at, p) = Bytes.blit_string p 0 b at (String.length p) in
      List.iter patch (Cubin.patches c ~base);
      b
    ]}
    The image of {!elf} holds what the GPU's memory holds of the cubin: its
    code, constant banks and globals, a global without an initial value
    ([.nv.global], which has no bytes in the object) as zeros. A kernel's shared
    memory ([.nv.shared.name]) stays out: it is on chip, and each block of a
    launch gets the kernel's [shared_bytes] ({!type-kernel}) of it. Reading a
    cubin copies none of its bytes: a loader writes them once, into its
    destination.

    Offsets are image offsets. *)

type t
(** The type for cubins. *)

val of_string : string -> (t, string) result
(** [of_string obj] is the cubin [obj], the sections the GPU's memory holds laid
    out at an alignment of 128 bytes ({!Device_elf.of_string}). The result is
    [Error msg], [msg] saying which, if [obj] is not a well-formed ELF object,
    or if:
    - it is not for NVIDIA GPUs: its [e_machine] is not [EM_CUDA] ([190]);
    - its image ({!size}) would be longer than [2{^49}] bytes, more than the
      49-bit virtual addresses of GPUs before Hopper reach, which only a
      corrupted address makes;
    - a relocation is of a type other than these three: a symbol's 64-bit
      address ([R_CUDA_64], [0x2]), its low 32 bits ([R_CUDA_ABS32_LO_32],
      [0x38]) and its high 32 bits ([R_CUDA_ABS32_HI_32], [0x39]);
    - an attribute of a [.nv.info] section it reads runs past the section's end,
      or one of the attributes it reads ([EIATTR_REGCOUNT],
      [EIATTR_MIN_STACK_SIZE], [EIATTR_PARAM_CBANK]) holds less than its 8
      bytes, a symbol's index and a value;
    - a relocation's symbol is not in the image, or the bytes it patches
      ({!patches}) lie past the end of the image of {!elf}, or, for one without
      an addend in its entry, past the end of their section. *)

val size : t -> int
(** [size c] is the length of [c]'s image: the image of {!elf}, then zeros up to
    the next multiple of 4 KiB and 4 KiB more. *)

val elf : t -> Device_elf.t
(** [elf c] is the object [c] was read from, which lays out its image. *)

val patches : t -> base:int -> (int * string) list
(** [patches c ~base] is what [c]'s relocations write over its image for an
    upload at the address [base], in the order of its relocations: pairs
    [(at, b)], each writing the little-endian bytes [b] at the offset [at]. A
    relocation of type [R_CUDA_64] writes its symbol's address in the 8 bytes at
    its offset; one of type [R_CUDA_ABS32_LO_32] or [R_CUDA_ABS32_HI_32] writes
    the low or high 32 bits of that address in the 4 bytes 4 past its offset. A
    symbol's address is [base] plus its offset, plus the relocation's addend,
    modulo [2{^64}]. A relocation whose entry holds no addend ([SHT_REL], as
    NVIDIA's compilers write) has it in the bytes it patches: their unsigned
    little-endian value in the image of {!elf}.

    Every patch lies in the image of {!elf}:
    [at + String.length b <= (elf c).size]. A later patch overwrites an earlier
    one where they share bytes. *)

(** {1:kernels Kernels} *)

val kernels : t -> string list
(** [kernels c] is the names of the kernels of [c], in the order of their code
    sections. *)

type bank = {
  index : int;  (** The bank's index, from [0]. *)
  offset : int;  (** The offset of its contents. *)
  bytes : int;  (** Its size. *)
}
(** The type for the constant banks a kernel reads. *)

type kernel = {
  code : int;  (** The offset of the kernel's first instruction. *)
  code_bytes : int;  (** The size of its code. *)
  registers : int;  (** The registers each thread uses. *)
  shared_bytes : int;  (** The shared memory its code declares. *)
  stack_bytes : int;  (** The stack each thread needs at least. *)
  params_offset : int;
      (** The offset in constant bank [0] at which the kernel's parameters
          start. *)
  banks : bank list;
      (** The constant banks the cubin holds for it, in the order their indices
          first appear. Bank [0] holds the parameters, written anew for each
          launch. *)
}
(** The type for what a launch of a kernel needs from its cubin. *)

val kernel : t -> string -> kernel option
(** [kernel c name] is what a launch of the kernel [name] of [c] needs, or
    [None] if [c]'s image holds no section [.text.name], its code.

    - [shared_bytes] is the size of the section [.nv.shared.name].
    - [params_offset] is the offset the attribute [EIATTR_PARAM_CBANK] of
      [.nv.info.name] records.
    - [registers] and [stack_bytes] are the values the attributes
      [EIATTR_REGCOUNT] and [EIATTR_MIN_STACK_SIZE] of the section [.nv.info]
      record for its function, the last if several do. Such an attribute names
      its function by a symbol: the kernel whose code section holds the symbol,
      or else the kernel of the symbol's name.
    - [banks] are the sections [.nv.constantI] and [.nv.constantI.name] the
      image holds: bank [I] at the offset and size of the last such section.

    A field is [0], and [banks] empty, without its section or attribute. *)
