(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPU code objects.

    A code object is the ELF executable a compiler links for one processor: a
    GPU, or a generic processor, whose code objects every GPU of a generation
    runs. A device loads its image at an address of its memory. Each kernel in
    it has a kernel descriptor, which a dispatch reads: where the kernel's code
    starts, the memory it takes, and the registers it sets up for its waves. *)

type t
(** The type for code objects, relocated. *)

val of_string : string -> (t, string) result
(** [of_string obj] is the code object [obj], with its relocations applied as a
    loader applies them: each [R_AMDGPU_REL64] word holds its target's offset
    from itself, so that the image runs at any address.

    Errors if [obj] is not a 64-bit little-endian ELF object for AMD GPUs, if it
    is compiled for no processor LLVM names, or for a generic one in a code
    object before version 6 or of generic version [0], or if one of its
    relocations is of another kind or refers to a symbol it does not define. *)

val target : t -> string
(** [target co] is the processor [co] is compiled for, as LLVM names it: a GPU,
    such as ["gfx1201"], or a generic processor, such as ["gfx12-generic"]. *)

val runs_on : t -> string -> bool
(** [runs_on co gpu] is [true] iff the GPU [gpu], such as ["gfx1201"], runs
    [co]: [co] is compiled for [gpu], or for a generic processor that LLVM lists
    [gpu] under. *)

val image : t -> string
(** [image co] is the bytes a device loads, padded with zeros to whole 32-bit
    words. *)

val kernels : t -> string list
(** [kernels co] is the names of the kernels of [co], in increasing order: the
    symbols with a kernel descriptor, [name ^ ".kd"]. *)

type kernel = {
  descriptor : int;  (** The image offset of its kernel descriptor. *)
  entry : int;  (** The image offset of its first instruction. *)
  group_segment : int;  (** The bytes of LDS a workgroup takes. *)
  private_segment : int;  (** The bytes of scratch a work-item takes. *)
  kernarg_size : int;  (** The bytes of its arguments. *)
  rsrc1 : int;  (** Its [COMPUTE_PGM_RSRC1], as the compiler set it. *)
  rsrc2 : int;  (** Its [COMPUTE_PGM_RSRC2], as the compiler set it. *)
  rsrc3 : int;  (** Its [COMPUTE_PGM_RSRC3], as the compiler set it. *)
  wave32 : bool;  (** Whether its waves are 32 lanes wide, else 64. *)
  dispatch_ptr : bool;
      (** Whether its waves read the address of their dispatch packet from user
          SGPRs. *)
  private_segment_buffer : bool;
      (** Whether its waves read a buffer descriptor of their scratch from user
          SGPRs, the first four. *)
}
(** The type for kernels, as their descriptors describe them. *)

val kernel : t -> string -> (kernel, string) result
(** [kernel co name] is the kernel [name] of [co], whose descriptor is at the
    symbol [name ^ ".kd"].

    Errors if [co] has no such symbol, or the descriptor or the code it points
    to lies outside the image. *)
