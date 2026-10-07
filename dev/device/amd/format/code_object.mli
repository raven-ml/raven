(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Code objects: the programs AMD GPUs run.

    A code object is the ELF object a compiler links for one processor: a GPU,
    or a generic processor whose code objects every GPU of a generation runs. A
    device loads its {e image} at an address of its memory, and runs its kernels
    from there. Each kernel has a {e kernel descriptor} in the image, which a
    dispatch reads: where the kernel's code starts, the memory it takes, and the
    registers it sets up for its waves.

    Offsets are offsets in the image, in bytes. *)

type t
(** The type for code objects, relocated. *)

val of_string : string -> (t, string) result
(** [of_string obj] is the code object [obj], with its relocations applied as a
    loader applies them: each [R_AMDGPU_REL64] word holds its target's offset
    from itself, so that the image runs at any address.

    The result is [Error msg] if [obj] is not an ELF object for AMD GPUs
    ({!Device_elf.of_string}), if it is compiled for no processor LLVM names, or
    for a generic one in a code object before version 6 or of generic version
    [0], or if one of its relocations is of another kind or uses a symbol whose
    bytes the image does not hold. [msg] says which. *)

val target : t -> string
(** [target co] is the processor [co] is compiled for, as LLVM names it: a GPU,
    such as ["gfx1201"], or a generic processor, such as ["gfx12-generic"]. *)

val runs_on : t -> string -> bool
(** [runs_on co gpu] is [true] iff the GPU named [gpu] ({!Gpu.processor}) runs
    [co]: [co] is compiled for [gpu], or for a generic processor that LLVM lists
    [gpu] under. *)

val image : t -> string
(** [image co] is the bytes a device loads, padded with zeros to whole 32-bit
    words. *)

val kernels : t -> string list
(** [kernels co] is the names of [co]'s kernels, in increasing order: each
    [name] such that [co] defines the symbol [name ^ ".kd"] in its image. *)

type kernel = {
  descriptor : int;  (** The offset of its kernel descriptor. *)
  entry : int;  (** The offset of its first instruction. *)
  group_segment : int;  (** The bytes of LDS a workgroup takes. *)
  private_segment : int;  (** The bytes of scratch a work-item takes. *)
  kernarg_size : int;  (** The bytes of its arguments. *)
  rsrc1 : int;  (** Its [COMPUTE_PGM_RSRC1], as the compiler set it. *)
  rsrc2 : int;  (** Its [COMPUTE_PGM_RSRC2], as the compiler set it. *)
  rsrc3 : int;  (** Its [COMPUTE_PGM_RSRC3], as the compiler set it. *)
  wave32 : bool;  (** [true] iff its waves are 32 lanes wide, else 64. *)
  dispatch_ptr : bool;
      (** [true] iff its waves read the address of their dispatch packet from
          user SGPRs. *)
  private_segment_buffer : bool;
      (** [true] iff its waves read a buffer descriptor of their scratch from
          their first four user SGPRs. *)
}
(** The type for kernels, as their descriptors describe them. *)

val kernel : t -> string -> (kernel, string) result
(** [kernel co name] is the kernel [name] of [co], whose descriptor is at the
    symbol [name ^ ".kd"].

    The result is [Error msg] if [co] has no such symbol in its image, or if the
    descriptor or the instruction it points to lies outside the image. *)
