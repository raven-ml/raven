(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the host suite and bench share. *)

type words = (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for buffers of 64-bit words. *)

val words : int -> words
(** [words n] is [n] zeroed words, at least one. *)

val address : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> int
(** [address b] is the address of [b]'s first element. *)

val machine : string
(** [machine] is the host's machine as the fixtures name it: ["x86_64"] or
    ["aarch64"]. *)

val fixture : dir:string -> string -> string
(** [fixture ~dir f] is the object of the program [f] for the host, read from
    the directory [dir]: [<f>_<machine>.o], or [<f>_x86_64_windows.o] on x86_64
    Windows when it exists. *)

val fresh : string -> string
(** [fresh o] is the object [o] made an object of its own, whose link no
    other load shares: a count from 1 in the padding of its identification
    ([EI_PAD], bytes 9 to 12), which an ELF reader ignores and a fixture
    leaves zero. *)

val executable : int -> bool
(** [executable a] is [true] iff the page holding the address [a] is mapped
    executable in the process. *)

val empty_job : threads:int -> total:int -> chunks:int -> unit
(** [empty_job ~threads ~total ~chunks] runs the pool's job of [total] units cut
    into [chunks] chunks on at most [threads] threads, whose calls do nothing,
    with the runtime released. *)

val workers : unit -> int
(** [workers ()] is the most threads a launch on [Rig.host] runs on: the
    pool's performance cores. *)
