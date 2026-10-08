(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Memory records, their stamps, the devices' caches, release lists and drains,
   and the mappings of borrows. Any domain may call any function; each device's
   lock guards its cache and lists, and no driver call runs under it. *)

open Def

val page : int
(* [page] is the host's page size. *)

(* Stamps *)

val stamps_new : unit -> int
val stamps_ref : int -> unit
val stamps_unref : int -> unit
val stamps_absorb : int -> int -> unit
(* [stamps_absorb dst src] raises [dst] with every point of [src]. *)

val iter_points : (int -> unit) -> int -> unit
(* [iter_points f st] is [f] over the points of [st], the last write first. It
   allocates nothing. *)

val iter_write : (int -> unit) -> int -> unit
(* [iter_write f st] is [f] of [st]'s last write, if any. *)

val check_points : int -> unit
(* [check_points st] raises [Lost] if a point of [st] is on a lost device. *)

(* Records *)

type bytes_ba =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val ba_address : ('a, 'b, 'c) Bigarray.Array1.t -> int
val token : int -> released -> int -> int -> int -> token
(* [token list record bytes room live] puts [record] on the release list [list]
   once collected, pacing the collector by [bytes] out of [room] ([live] for
   memory the host shares, [-1] otherwise). *)

val holds_list : int
(* [holds_list] is the release list of holds, which every drain reads. *)

val entry :
  ?region:region ->
  ?io_region:io_region ->
  device ->
  memory_kind ->
  int ->
  int ->
  entry
(* [entry owner kind bytes stamps] is a release record. *)

val no_entry : entry
(* [no_entry] is the entry of host memory no device borrowed. *)

val make :
  ?keep:keep ->
  ?host:int ->
  ?address:int ->
  ?handle:nativeint ->
  ?token:token ->
  device ->
  int ->
  entry ->
  memory

val stamps : memory -> int
(* [stamps m] is the stamps of the memory [m] lies in, below its borrows. *)

val region_info : region -> int * nativeint * int
(* [region_info r] is [r]'s address, handle and host address, [-1] for none. *)

val note : device -> unit
(* [note d] records [d]'s allocated bytes in the profiles being taken. *)

val ensure_entry : memory -> unit
(* [ensure_entry m] gives host memory [m] stamps and a token, which unmaps its
   mappings once it is collected. *)

val proxy : entry -> int
(* [proxy e] is the C proxy that bigarrays over [e]'s memory share, made at the
   first: [e]'s memory is neither reused nor freed while one is reachable. *)

(* Mappings and borrows *)

val map_host_range : device -> int -> int -> region option
val map_peer_region : device -> region -> region option
val borrow : device -> memory -> memory option
(* [borrow d m] is [d]'s borrow of [m]'s memory, or [None] where [d] cannot map
   it. An io memory borrows through its pages. *)

val prefetch : device -> memory -> at:int -> len:int -> unit
(* [prefetch d m ~at ~len] asks the io device of [m] to read [len] bytes of it
   from [at] ahead, if [d] is no host. *)

val of_io : device -> io_region -> int -> memory
(* [of_io d r n] is a memory record over [n] bytes of the region [r] an io
   library gave the io device [d], after draining [d]. *)

(* Allocation and drains *)

val drain : device -> unit
(* [drain d] returns the memory of [d]'s collected buffers, runs the releases
   due on it and on lost devices, and the holds' releases due. *)

val rounds : int
(* [rounds] is the attempts of an allocation the budget or driver refuses. *)

val reclaim : device -> int -> unit
(* [reclaim d round] releases [d]'s cache, drains every other device and, from
   the second round, collects. *)

val room : device -> int
val free_entry : entry -> unit
(* [free_entry e] gives [e]'s region back to its driver, unmapping other
   devices' mappings of it once their work is done. *)

val alloc_entry : device -> memory_kind -> int -> entry
(* [alloc_entry d kind n] allocates [n] bytes of [d]'s memory of [kind] on the
   allocation path: drains, the cache, the budget, the reclaim rounds, then
   [Out_of_memory]. *)

val alloc : device -> memory_kind -> int -> memory
(* [alloc d kind n] is a memory record over [alloc_entry d kind n], with its
   token. *)

val host_memory : int -> memory
val set_budget : device -> int -> unit
val free_cache : device -> unit
