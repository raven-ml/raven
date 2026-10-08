(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A range of addresses the process hands out (private).

    The GPU's addresses of the process's memory: first fit over the free ranges,
    which a free merges with its neighbours. Any domain may call any function.
*)

type t
(** The type for ranges of addresses. *)

val make : base:int -> int -> t
(** [make ~base n] is the [n] addresses from [base], all free. *)

val alloc : t -> align:int -> int -> int option
(** [alloc r ~align n] is the first of [n] free addresses of [r], a multiple of
    [align], a power of two; [None] if no free range holds them. *)

val free : t -> int -> int -> unit
(** [free r a n] frees the [n] addresses from [a], which {!alloc} gave. *)
