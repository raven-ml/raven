(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A GPU's discovery table, which its firmware leaves 64 KiB below the end of
   its memory: the version of each of its blocks, where their registers lie, the
   instances fused off, and the shape of its GC. Pure: any domain.

   Blocks are named by their hardware IP number, as amdgpu numbers them
   (GC_HWIP, SDMA0_HWIP, MP0_HWIP, ...). *)

type version = Device_amd_abi.Gpu.version

type gc = {
  engines : int; (* the shader engines of one die *)
  arrays : int; (* the shader arrays of an engine *)
  units : int; (* the compute units of an array *)
  scratch_slots : int; (* the waves a compute unit runs with scratch memory *)
  waves : int; (* the waves a SIMD runs at once *)
  lds : int; (* the local data share of a workgroup, in bytes *)
}

type t = {
  versions : (int * version) list; (* by block *)
  bases : (int * (int * int array) list) list;
      (* by block, then instance: the base of each register segment, in 32-bit
         words *)
  harvested : (int * int list) list; (* by block: the instances fused off *)
  gc : gc;
}

(* [bytes] is the length of the table read, 10 KiB. *)
val bytes : int

(* [offset ~memory] is the table's offset in a GPU of [memory] bytes. *)
val offset : memory:int -> int

(* [of_string s] is the table [s]. [Error msg] if a signature is wrong, a header
   is of a version this library does not read, or an offset points outside [s],
   naming the field and its byte offset. *)
val of_string : string -> (t, string) result

(* [version d b] is the version of block [b], if the GPU has one. *)
val version : t -> int -> version option

(* [live d b] is the instances of block [b] not fused off, with their segment
   bases, in instance order. *)
val live : t -> int -> (int * int array) list

(* [name b] is amdgpu's name of block [b], such as "GC". *)
val name : int -> string

(* [supported d] is [Ok ()] iff every block a boot programs (GC, SDMA, MP0, MP1,
   MMHUB, OSSSYS, NBIO, HDP) has a version this library holds register tables
   and firmware for. [Error msg] names the first other block and its version, as
   "MP0 13.0.5 is a version this library does not boot". *)
val supported : t -> (unit, string) result
