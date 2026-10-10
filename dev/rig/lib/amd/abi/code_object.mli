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

    The image ({!image}) is {!size} bytes: the image of {!elf} ({!Rig_elf}),
    zeros up to {!size}, then each of {!patches} written over them, in order. It
    starts at the object's lowest section the image holds ([(elf co).address]),
    so the headers and tables a linker puts before its code are not loaded. A
    loader writes it into its destination.

    Offsets are offsets in the image, in bytes. *)

type t
(** The type for code objects: an ELF object for AMD GPUs, and what its
    relocations write over its image. *)

val of_string : string -> (t, string) result
(** [of_string obj] is the code object [obj]. It is [Error msg], [msg] saying
    which, if:
    - [obj] is not an ELF object for AMD GPUs ({!Rig_elf.of_string});
    - it is compiled for no processor LLVM names, or for a generic one in a code
      object before version 6 or of generic version [0];
    - one of its relocations is of another kind than [R_AMDGPU_REL64], uses a
      symbol whose bytes the image does not hold, patches bytes past the image's
      end, or has no addend in its entry ([SHT_REL]): LLVM writes those for
      Mesa's and PAL's code objects, never for HSA's (AMDGPUUsage, "Relocation
      Records");
    - a kernel's descriptor, or the instruction it points to, lies outside the
      image, or its group segment is more than 511 * 512 bytes, the most every
      GPU's LDS field holds, or its private segment is more than a lane's share
      of the most scratch a 64-lane wave of its processor takes: 131056 bytes
      before GFX11, 131068 on GFX11 and 1048572 on GFX12;
    - its image is longer than [2{^48}] bytes, which no GPU's virtual addresses
      reach;
    - a note runs past its section, or its metadata note
      ([NT_AMDGPU_METADATA]) is no MessagePack value, bounds a kernel's
      workgroups to fewer than 1 or more than 1024 work-items, or places an
      argument outside the kernel's arguments. *)

val target : t -> string
(** [target co] is the processor [co] is compiled for, as LLVM names it: a GPU,
    such as ["gfx1201"], or a generic processor, such as ["gfx12-generic"]. *)

val runs_on : t -> Gpu.t -> bool
(** [runs_on co g] is [true] iff [g] runs [co]: [co] is compiled for
    {!Gpu.processor}[ g], or for a generic processor that LLVM lists it under.
*)

val size : t -> int
(** [size co] is the length of [co]'s image, in bytes: [(elf co).size] rounded
    up to whole 32-bit words. *)

val elf : t -> Rig_elf.t
(** [elf co] is the object [co] was read from, which lays out its image. *)

val patches : t -> (int * string) list
(** [patches co] is what [co]'s relocations write over its image, in the order
    of the object's relocations: each [(off, p)] puts the bytes [p] at offset
    [off], and [off + String.length p <= size co]. Each [p] is the 8
    little-endian bytes of an [R_AMDGPU_REL64] word, [S + A - P]: its target's
    offset plus the relocation's addend, less the word's own offset. A patch
    holds wherever the image is loaded. *)

val image : t -> string
(** [image co] is [co]'s image: {!size} bytes, laid out as the module's preamble
    says. *)

val kernels : t -> string list
(** [kernels co] is the names of [co]'s kernels, in increasing order: each
    [name] such that [co] defines the symbol [name ^ ".kd"] in its image. *)

(** The type for the axes of a grid. *)
type axis = X | Y | Z

(** The type for the implicit arguments a kernel reads, which its dispatcher
    writes among its arguments (AMDGPUUsage, "Code Object V5 Metadata"): a
    kernel learns its grid's and its workgroups' sizes from them. *)
type hidden =
  | Block_count of axis
      (** Its grid's whole workgroups along the axis, 32 bits. *)
  | Group_size of axis
      (** The work-items of its whole workgroups along the axis, 16 bits. *)
  | Remainder of axis
      (** The work-items of its last workgroup along the axis, if that one is
          partial, else [0], 16 bits. *)
  | Global_offset of axis
      (** The index of its first work-item along the axis, 64 bits. *)
  | Grid_dims  (** The axes of its grid, [1] to [3], 16 bits. *)
  | Dynamic_lds_size
      (** The bytes of LDS its dispatch adds to its group segment, 32 bits. *)
  | Other of string
      (** Another, by its value kind, such as ["hidden_printf_buffer"]: a
          service of its runtime. *)

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
  max_threads : int;
      (** The most work-items its workgroups have: its
          [.max_flat_workgroup_size] in the object's metadata, or 1024, the
          most of any GPU, if the metadata does not bound it. *)
  hidden : (hidden * int) list;
      (** The implicit arguments it reads, each with its offset among its
          arguments, by increasing offset, as the object's metadata lists them;
          [[]] if the metadata lists none. *)
  explicit_size : int;
      (** The bytes of its explicit arguments, which its caller writes: the end
          of the last argument the metadata lists that is no implicit one
          ({!hidden}): [0] where it lists none, and [kernarg_size] for an object
          without metadata. *)
}
(** The type for kernels, as their descriptors describe them. *)

val kernel : t -> string -> kernel option
(** [kernel co name] is the kernel [name] of [co], whose descriptor is at the
    symbol [name ^ ".kd"], if [co] defines it ({!kernels}). *)
