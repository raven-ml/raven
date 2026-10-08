(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A GPU's discovery table, which its firmware leaves {!offset} bytes below the
    end of its memory: the version of each of its blocks, where their registers
    lie, the instances fused off, and the shape of its GC. Pure: any domain.

    Blocks are named by the hardware ID the table gives them, as the Linux
    kernel's [soc15_hw_ip.h] numbers them: 11 for GC, 42 for SDMA0, 255 for MP0.
*)

type version = Device_amd_abi.Gpu.version
(** The type for block versions, [(major, minor, revision)]. *)

type gc = {
  engines : int;  (** The shader engines of one die. *)
  arrays : int;  (** The shader arrays of an engine. *)
  units : int;  (** The compute units of an array. *)
  scratch_slots : int;
      (** The waves a compute unit runs with scratch memory. *)
  waves : int;  (** The waves a SIMD runs at once. *)
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
(** [of_string s] is the table [s], checked as the kernel driver checks it: the
    signatures and byte-sum checksums of the binary, its IP table, its GC table
    and its harvest table. A base address of a table of 64-bit bases keeps its
    low 30 bits, a word address. [Error msg] if a signature or checksum is
    wrong, a die is out of order, a header is of a version this library does not
    read, or a field lies outside [s], naming the field and its byte offset. It
    never raises. *)

val version : t -> int -> version option
(** [version d b] is the version of block [b], if the GPU has one. *)

val live : t -> int -> (int * int array) list
(** [live d b] is the instances of block [b] not fused off, with their segment
    bases, in instance order. *)

val name : int -> string
(** [name b] is the kernel's name of block [b], such as ["GC"], or [""] for an
    ID it does not name. *)
