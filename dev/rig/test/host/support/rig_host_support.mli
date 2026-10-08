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

val executable : int -> bool
(** [executable a] is [true] iff the page holding the address [a] is mapped
    executable in the process. *)

val set_address_space : int -> int
(** [set_address_space n] sets this process's soft limit of its address space to
    [n] bytes, so that a mapping past it fails: [0], the errno, or [-1] off
    Linux, where the limit does not bound mappings. *)

val count_job : threads:int -> total:int -> chunks:int -> words -> unit
(** [count_job ~threads ~total ~chunks counts] runs the pool's job of [total]
    units cut into [chunks] chunks on at most [threads] threads, with the
    runtime released, whose calls add 1 to [counts.{i}] for each unit [i] they
    run. *)

val empty_job : threads:int -> total:int -> chunks:int -> unit
(** [empty_job ~threads ~total ~chunks] runs the pool's job of [total] units cut
    into [chunks] chunks on at most [threads] threads, whose calls do nothing,
    with the runtime released. *)
