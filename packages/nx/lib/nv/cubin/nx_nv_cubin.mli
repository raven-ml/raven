(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Cubins: the ELF objects NVIDIA's compilers make for a GPU architecture.

    A cubin holds the machine code of one or more kernels. The kernel [name] is
    the code of the section [.text.name]; what a launch of it needs besides, its
    registers, stack, shared memory and constant banks, the cubin records in
    other sections and in attributes of its [.nv.info] sections. *)

type t
(** The type for cubins, laid out as the GPU runs them. *)

val of_string : string -> (t, string) result
(** [of_string obj] is the cubin [obj], its sections laid out at an alignment of
    128 bytes ({!Nx_device_elf.load}). It is [Error] with the reason if [obj] is
    no 64-bit little-endian ELF object, refers to a symbol it does not define,
    or has a relocation of a type other than the 64-bit address of a symbol
    ([0x2]) and its low ([0x38]) or high ([0x39]) 32 bits. *)

val image : t -> string
(** [image c] is [c]'s sections as {!of_string} lays them out, then zeros up to
    the next multiple of 4 KiB and 4 KiB more, which the GPU's instruction
    prefetch may read past the code. *)

val relocate : t -> base:int -> string
(** [relocate c ~base] is {!image}[ c] with its relocations applied for an
    upload at the address [base]: the 64-bit address of a symbol at its offset,
    and the low or high 32 bits of one in the four bytes after its offset. *)

(** {1:kernels Kernels} *)

val kernels : t -> string list
(** [kernels c] is the names of the kernels of [c], in the order of their code's
    sections. *)

type bank = {
  index : int;  (** The bank's index, from [0]. *)
  offset : int;  (** The offset of its contents in the {!image}. *)
  bytes : int;  (** Its size. *)
}
(** The type for the constant banks a kernel reads. *)

type kernel = {
  code : int;  (** The offset of the kernel's first instruction in the image. *)
  code_bytes : int;  (** The size of its code. *)
  registers : int;  (** The registers each thread uses. *)
  shared_bytes : int;  (** The shared memory its code declares. *)
  stack_bytes : int;  (** The stack each thread needs at least. *)
  params_offset : int;
      (** The offset in constant bank 0 at which the kernel's parameters start.
      *)
  banks : bank list;
      (** The constant banks the cubin holds, in the order their indices first
          appear. Bank [0] holds the parameters, written for each launch. *)
}
(** The type for what a launch of a kernel needs from its cubin. *)

val kernel : t -> string -> kernel option
(** [kernel c name] is what a launch of the kernel [name] of [c] needs, or
    [None] if [c] has no section [.text.name], its code. Its [shared_bytes] is
    the size of the section [.nv.shared.name], and its [params_offset] the
    offset the attribute [EIATTR_PARAM_CBANK] of [.nv.info.name] records. Its
    [registers] and [stack_bytes] are the values the attributes
    [EIATTR_REGCOUNT] and [EIATTR_MIN_STACK_SIZE] of the section [.nv.info]
    record for its function, the last if several do: an attribute names its
    function by a symbol, which is the kernel of the code section it is in, or
    else the kernel of its name. Each is [0] without one. Its [banks] are the
    sections [.nv.constantI] and [.nv.constantI.name], bank [I] at the offset
    and size of the last such section. *)
