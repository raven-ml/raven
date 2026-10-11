(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Cubins: the ELF objects NVIDIA's compilers make for a GPU architecture.

    A cubin holds the machine code of one or more kernels. The kernel [name]
    is the code of the section [.text.name]. What a launch of it needs
    besides, its registers, stack, shared memory and constant banks, the
    cubin records in other sections and in the attributes of its [.nv.info]
    sections. This module reads three attributes, each a symbol's index and
    a value: [EIATTR_REGCOUNT], [EIATTR_MIN_STACK_SIZE] and
    [EIATTR_PARAM_CBANK].

    A loader uploads a cubin as {!size} bytes: the ELF object's laid-out
    sections ({!Rig_elf}), zeros after them, since the GPU's instruction
    prefetch reads past the code, and {!patches} written over both. This
    module calls those bytes the cubin's {e image}, and offsets are image
    offsets. A loader builds the image of an upload at [base] as:
    {[
    let image c ~base =
      let o = Cubin.elf c in
      let b = Bytes.make (Cubin.size c) '\000' in
      let put (s : Rig_elf.section) =
        match s.offset with
        | Some off -> Bytes.blit_string o.file s.at b off s.length
        | None -> ()
      in
      Iarray.iter put o.sections;
      let patch (at, p) = Bytes.blit_string p 0 b at (String.length p) in
      List.iter patch (Cubin.patches c ~base);
      b
    ]}
    The laid-out sections hold what the GPU's memory holds of the cubin:
    its code, constant banks and globals, a global without an initial value
    ([.nv.global]) as zeros. A kernel's
    shared memory ([.nv.shared.name]) stays out, since it is on chip and
    each block of a launch gets the kernel's [shared_bytes] of it. Reading a
    cubin leaves its bytes in place, since a loader writes them once, into
    its destination.

    A relocation writes a symbol's address, which is the upload's [base]
    plus the symbol's offset, plus the relocation's addend, modulo
    [2{^64}]. This module reads three types of relocation:
    - [R_CUDA_64] ([0x2]) writes the address in the 8 bytes at its offset;
    - [R_CUDA_ABS32_LO_32] ([0x38]) and [R_CUDA_ABS32_HI_32] ([0x39]) write
      its low or high 32 bits in the 4 bytes 4 past its offset.

    A relocation of an [SHT_REL] section, as NVIDIA's compilers write, keeps
    its addend in the bytes it patches: their unsigned little-endian value
    in the laid-out sections. *)

type t
(** The type for cubins. *)

val of_string : string -> (t, string) result
(** [of_string obj] is the cubin [obj], the sections the GPU's memory holds
    laid out at an alignment of 128 bytes ({!Rig_elf.of_string}). The result
    is [Error msg], [msg] saying which, if [obj] is not a well-formed ELF
    object, or if:
    - its [e_machine] is not [EM_CUDA] ([190]);
    - its image ({!size}) would be longer than [2{^49}] bytes, the reach of
      the 49-bit virtual addresses of GPUs before Hopper;
    - a relocation is of another type than the three this module reads;
    - an attribute of a [.nv.info] section it reads runs past the section's
      end, or one of the three attributes holds less than its 8 bytes;
    - a relocation's symbol is not in the image, or the bytes it patches lie
      past the end of the laid-out sections or, for an [SHT_REL]
      relocation, past the end of their section. *)

val size : t -> int
(** [size c] is the length of [c]'s image: its laid-out sections, then zeros
    up to the next multiple of 4 KiB and 4 KiB more. *)

val elf : t -> Rig_elf.t
(** [elf c] is the object [c] was read from, which lays out its sections. *)

val patches : t -> base:int -> (int * string) list
(** [patches c ~base] is what [c]'s relocations write over its image for an
    upload at the address [base], in the order of its relocations: pairs
    [(at, b)], each writing the little-endian bytes [b] at the offset [at].
    Every patch lies in the laid-out sections:
    [at + String.length b <= (elf c).size]. A later patch overwrites an
    earlier one where they share bytes. *)

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
  registers : int;
      (** The registers each thread uses, from [EIATTR_REGCOUNT]. *)
  shared_bytes : int;
      (** The shared memory its code declares: the size of the section
          [.nv.shared.name]. *)
  stack_bytes : int;
      (** The stack each thread needs at least, from
          [EIATTR_MIN_STACK_SIZE]. *)
  params_offset : int;
      (** The offset in constant bank [0] at which the kernel's parameters
          start, from [EIATTR_PARAM_CBANK] of [.nv.info.name]. *)
  params_bytes : int;
      (** The size of the kernel's parameters in bytes, from the same
          attribute. *)
  banks : bank list;
      (** The constant banks the cubin holds for it, in the order their
          indices first appear: bank [I] at the offset and size of the last
          section [.nv.constantI] or [.nv.constantI.name] in the image. Bank
          [0] holds the parameters, written anew for each launch. *)
}
(** The type for what a launch of a kernel needs from its cubin. A field is
    [0], and [banks] empty, without its section or attribute. [registers]
    and [stack_bytes] come from the attributes of the section [.nv.info]
    for the kernel's function, the last if several hold one. Such an
    attribute names its function by a symbol: the kernel whose code section
    holds the symbol, or else the kernel of the symbol's name. *)

val kernel : t -> string -> kernel option
(** [kernel c name] is what a launch of the kernel [name] of [c] needs, or
    [None] if [c] lacks the section [.text.name]. *)
