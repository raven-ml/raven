(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Cubins: the ELF objects NVIDIA's compilers make for a GPU architecture.

    A cubin holds the machine code of one or more kernels. The kernel [name] is
    the code of the section [.text.name]. What a launch of it needs besides, its
    registers, stack, shared memory and constant banks, the cubin records in
    other sections and in the attributes of its [.nv.info] sections.

    A cubin is uploaded as its {e image}: its sections laid out in GPU memory
    ({!Device_elf}), then zeros the GPU's instruction prefetch may read past the
    code. Offsets are image offsets. *)

type t
(** The type for cubins. *)

val of_string : string -> (t, string) result
(** [of_string obj] is the cubin [obj], its sections laid out at an alignment of
    128 bytes ({!Device_elf.of_string}). It is [Error] with the reason if [obj]
    is not a well-formed ELF object, or if a relocation uses a symbol outside
    the image, patches bytes past its end, or is of a type other than the 64-bit
    address of a symbol ([R_CUDA_64], [0x2]) and its low ([R_CUDA_ABS32_LO_32],
    [0x38]) or high ([R_CUDA_ABS32_HI_32], [0x39]) 32 bits. *)

val size : t -> int
(** [size c] is the length of [c]'s image: its sections, then zeros up to the
    next multiple of 4 KiB and 4 KiB more. *)

val image : t -> base:int -> string
(** [image c ~base] is [c]'s image, of {!size}[ c] bytes, relocated for an
    upload at the address [base]: the address of a symbol in the 8 bytes at a
    relocation's offset, or its low or high 32 bits in the 4 bytes 4 after it. A
    symbol's address is [base] plus its offset, plus the relocation's addend. *)

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
