(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** This machine's memory for functions that reach it (private).

    {!Function} checks every argument first: sizes are positive, addresses are
    on a page, and contiguous memory is at most 2 MiB, a huge page that starts
    on 2 MiB. What the system refuses raises {!Fail.Failed}; off Linux every
    function but {!page} does. Any domain may call. *)

val page : int
(** [page] is the system's page size in bytes. *)

val reserve : base:int -> int -> unit
(** [reserve ~base n] reserves [n] addresses from [base] once per range, for
    {!alloc} and {!map} only. *)

val alloc : ?contiguous:bool -> ?va:int -> int -> Window.t * int list
(** [alloc n] is [n] bytes, rounded up to {!page}, of new, zeroed, locked
    memory, with the physical address of each page, pinned until {!free}. At
    [va], or where the system chooses. [~contiguous:true] memory larger than a
    page is one 2 MiB huge page. [va] and the bytes mapped there lie in a range
    {!reserve} reserved, as {!Function.alloc_dma} checks. *)

val map : ?va:int -> int -> Window.t
(** [map n] is {!alloc}'s memory, neither locked nor read for its addresses. *)

val free : Window.t -> unit
(** [free w] releases {!alloc} or {!map} memory, with {!alloc}'s pins. *)

val pin : int -> int -> int list
(** [pin a n] locks the [n] bytes at [a] and is their pages' physical addresses.
    Pins are counted per page across the process. *)

val unpin : int -> int -> unit
(** [unpin a n] drops one pin of each page of the [n] bytes at [a], which {!pin}
    pinned, and unlocks those whose last pin it drops. *)
