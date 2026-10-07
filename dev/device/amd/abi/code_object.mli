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

    The image is {!size} bytes: the image of {!elf} ({!Device_elf}), zeros up to
    {!size}, then each of {!patches} written over them, in order. It starts at
    the object's lowest section the image holds ([(elf co).address]), so the
    headers and tables a linker puts before its code are not loaded. A loader
    writes it into its destination. Into [bytes], for instance:
    {[
    let image co =
      let o = Code_object.elf co in
      let b = Bytes.make (Code_object.size co) '\000' in
      let put (s : Device_elf.section) =
        match s.offset with
        | Some off -> Bytes.blit_string o.file s.at b off s.length
        | None -> ()
      in
      Iarray.iter put o.sections;
      let patch (off, p) = Bytes.blit_string p 0 b off (String.length p) in
      List.iter patch (Code_object.patches co);
      b
    ]}

    Offsets are offsets in the image, in bytes. *)

type t
(** The type for code objects: an ELF object for AMD GPUs, and what its
    relocations write over its image. *)

val of_string : string -> (t, string) result
(** [of_string obj] is the code object [obj].

    The result is [Error msg] if [obj] is not an ELF object for AMD GPUs
    ({!Device_elf.of_string}), if it is compiled for no processor LLVM names, or
    for a generic one in a code object before version 6 or of generic version
    [0], if one of its relocations is of another kind than [R_AMDGPU_REL64],
    uses a symbol whose bytes the image does not hold, or patches bytes past the
    image's end, if a kernel's descriptor or the instruction it points to lies
    outside the image, or if its image is longer than [2{^48}] bytes, which no
    GPU's virtual addresses reach. [msg] says which. *)

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

val elf : t -> Device_elf.t
(** [elf co] is the object [co] was read from, which lays out its image. *)

val patches : t -> (int * string) list
(** [patches co] is what [co]'s relocations write over its image, in the order
    of the object's relocations: each [(off, p)] puts the bytes [p] at offset
    [off], and [off + String.length p <= size co]. Each [p] is the 8
    little-endian bytes of an [R_AMDGPU_REL64] word, its target's offset from
    the word itself, so the patches hold wherever the image is loaded. *)

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

val kernel : t -> string -> kernel option
(** [kernel co name] is the kernel [name] of [co], whose descriptor is at the
    symbol [name ^ ".kd"], if [co] defines it ({!kernels}). *)
