(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A GPU's discovery table, read without touching the GPU.

    The table lists the version of each of the GPU's blocks, where their
    registers lie, the instances fused off, and the shape of its GC. Its
    firmware leaves it {!offset} bytes below the end of the GPU's memory.
    Every value is pure, and any domain may call it.

    Blocks are named by the hardware ID the table gives them, such as 11
    for GC and 42 for SDMA0. *)

type version = Gpu.version
(** The type for block versions, [(major, minor, revision)]. *)

type gc = {
  engines : int;  (** The shader engines of one die. *)
  arrays : int;  (** The shader arrays of an engine. *)
  units : int;  (** The compute units of an array. *)
  scratch_slots : int;
      (** The waves a compute unit runs with scratch memory. *)
  waves : int;  (** The waves a SIMD runs concurrently. *)
  lds : int;  (** The local data share of a workgroup, in bytes. *)
}
(** The type for the shape of a GC. *)

type t = {
  versions : (int * version) list;
      (** By block: the version of its lowest instance. *)
  bases : (int * (int * int array) list) list;
      (** By block, then instance: the base of each register segment, in 32-bit
          words. *)
  harvested : (int * int list) list;  (** By block: the instances fused off. *)
  gc : gc;
}
(** The type for discovery tables. *)

val offset : int
(** [offset] is how far below the end of the GPU's memory the table starts: 64
    KiB. *)

val bytes : int
(** [bytes] is the length of the table read, 10 KiB. *)

val of_string : string -> (t, string) result
(** [of_string s] is the table [s], with the signatures and byte-sum
    checksums of the binary, its IP table, its GC table and its harvest
    table checked. A base address of a table of 64-bit bases
    keeps its low 30 bits, a word address.

    The result is [Error msg], naming the field and its byte offset, if:
    {ul
    {- a signature or a checksum is wrong;}
    {- the binary lists no IP table or no GC table;}
    {- the IP table lists more dies than its header holds, or a die whose
       ID is not its position;}
    {- the GC table is of a version other than 1 or 2;}
    {- a field lies outside [s].}}
    It raises nothing. *)

val version : t -> int -> version option
(** [version d b] is the version of block [b], if the GPU has one. *)

val live : t -> int -> (int * int array) list
(** [live d b] is the instances of block [b] not fused off, with their segment
    bases, in instance order. *)

val aids : t -> int list
(** [aids d] is the live dies of a GPU of several accelerator dies (AIDs),
    in order. They are die [0] and each die whose four SDMA instances are
    all live, or whose first two or last two are. A GPU of one die has
    [[0]]. A caller names no other die to the indirect window of the GPU's
    PCIe registers, since a write to a die fused off stalls the GPU's
    fabric. *)

val name : int -> string
(** [name b] is the name of block [b], such as ["GC"], or [""] for an ID
    this library does not name. *)
