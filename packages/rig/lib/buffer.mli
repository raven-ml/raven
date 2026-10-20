(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Buffers: the implementation of {!Rig.Buffer}, whose {!Rig.Buffer.copy} is
    {!Copy.copy}.

    A buffer is a range of a memory record ({!Def.memory}): the offset of its
    first byte, its length, and the claim generation it was made in. It is
    live while that generation is its claim's, and dead once {!Claim.consume}
    has raised the claim's generation. Every view and borrow of an
    allocation shares its claim, so a consumption kills every buffer over the
    allocation made before it.

    The host's buffers lie in bigarrays of the host's heap
    ({!Memory.host_memory}) or of the caller ({!of_bigarray}), and get an
    entry when a device first borrows them ({!Memory.ensure_entry}). A
    zero-byte buffer on another device has the address [0] and a memory
    record with {!Memory.no_entry}, since work over zero bytes reads
    nothing. Every other buffer lies in an allocation that {!Memory.alloc}
    or {!Memory.of_io} made.

    Every function that reaches a buffer's bytes first checks that it is
    live, and raises [Invalid_argument] naming the public function
    ({!check_live}). {!export} waits for the stamps. {!array},
    {!blit_string}, {!blit_bytes} and {!set_bytes} wait for nothing, and
    their callers take the claim and the wait, as {!Copy.blit_to_bytes} and
    {!Claim.read_bytes} do. *)

open Def

type t = buffer
(** {!Rig.Buffer.t}. *)

(** {!Rig.Buffer.kind}. *)
type kind = Rig_edge.memory = Device | Pinned | Mapped

(** {!Rig.Buffer.access}. *)
type access = Def.access = Read | Read_write

val is_live : t -> bool
(** [is_live b] is [true] iff no consumption of [b]'s memory came after [b]
    was made. *)

val dead : t -> string option
(** [dead b] is {!Rig.Buffer.dead}[ b]. *)

val access : t -> access
(** [access b] is {!Rig.Buffer.access}[ b]. *)

val check_live : string -> t -> unit
(** [check_live fn b] raises [Invalid_argument] if [b] is dead, naming the
    public function [fn], such as ["Buffer.view"]. *)

val refuse_dead : string -> t -> 'a
(** [refuse_dead fn b] raises what {!check_live} raises for the dead [b]. *)

val of_memory : Def.memory -> int -> t
(** [of_memory m n] is a live buffer over the first [n] bytes of [m]. It
    checks nothing: [n <= m.bytes] must hold. *)

val create : ?memory:kind -> device -> int -> t
(** [create ?memory d n] is {!Rig.Buffer.create}[ ?memory d n]. *)

val of_io : device -> 'r Type.Id.t -> 'r -> access:access -> int -> t
(** [of_io d k r ~access n] is {!Rig.Buffer.of_io}[ d k r ~access n]. *)

val io : t -> 'r Type.Id.t -> 'r option
(** [io b k] is {!Rig.Buffer.io}[ b k]. *)

val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t
(** [of_bigarray a] is {!Rig.Buffer.of_bigarray}[ a]. Its memory starts
    outside the claims ({!Memory.outside}). *)

val borrow : device -> t -> t option
(** [borrow d b] is {!Rig.Buffer.borrow}[ d b]. For io memory it also asks
    the io device to read [b]'s bytes ahead ({!Memory.prefetch}). *)

val wait : t -> access -> unit
(** [wait b access] is {!Rig.Buffer.wait}[ b access]. It is the C callback
    ["rig.buffer.wait"], which [rig_buffer_wait] in [rig.h] calls. *)

val reached : t -> access -> bool
(** [reached b access] is {!Rig.Buffer.reached}[ b access]. *)

val wait_points : t -> access -> unit
(** [wait_points b access] is {!wait} without its checks, for a live [b]
    whose memory and device the caller found not lost. *)

val device : t -> device
(** [device b] is {!Rig.Buffer.device}[ b]. *)

val length : t -> int
(** [length b] is {!Rig.Buffer.length}[ b]. *)

val is_borrowed : t -> bool
(** [is_borrowed b] is {!Rig.Buffer.is_borrowed}[ b]. *)

val view : t -> first:int -> length:int -> t
(** [view b ~first ~length] is {!Rig.Buffer.view}[ b ~first ~length]. *)

val spans : t -> bool
(** [spans b] is {!Rig.Buffer.spans}[ b]. *)

val overlaps : t -> t -> bool
(** [overlaps b b'] is {!Rig.Buffer.overlaps}[ b b']. *)

val places : t -> (int * int) list
(** [places b] is where [b]'s bytes lie, as [(space, first)]: the space and
    first byte that [rig_buffer_span] in [rig.h] gives, then, for memory
    outside the host's space whose pages the host maps, the host's space [0]
    and their host address. It is [[]] if [b] has no bytes. [overlaps b b']
    holds iff a place of [b] and one of [b'] share a space and their bytes,
    [length b] and [length b'] of them, meet there. Pages mapped later add
    a place and move none. *)

val space : t -> int
(** [space b] is the space of [b]'s first place ({!places}). It allocates
    nothing. *)

val first : t -> int
(** [first b] is the first byte of [b]'s first place ({!places}). It
    allocates nothing. *)

val alias : t -> int
(** [alias b] is the first byte of [b]'s second place, in the host's space
    [0], or [0] if it has none ({!places}). It allocates nothing. *)

val array :
  string ->
  ('a, 'b) Bigarray.kind ->
  t ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** [array fn k b] is [b]'s bytes as elements of [k], without a wait, a claim
    or an export. An array over memory outside the host's heap and the
    caller's bigarrays shares the allocation's proxy ({!Memory.proxy}), which
    keeps the allocation from reuse while the array is reachable. Raises
    [Invalid_argument], naming [fn], as {!Rig.Buffer.read}. *)

val export :
  t -> ('a, 'b) Bigarray.kind -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** [export b k] is {!Rig.Buffer.export}[ b k]. It takes the memory out of
    the claims for good ({!Memory.mark_outside}). *)

val of_string : string -> t
(** [of_string s] is {!Rig.Buffer.of_string}[ s]. *)

val host_addressed : t -> bool
(** [host_addressed b] is [true] iff this process's host addresses [b]'s
    memory ({!Rig.Buffer.section-host_access}). *)

val check_range : string -> string -> int -> int -> int -> unit
(** [check_range fn what size at n] raises [Invalid_argument], naming [fn]
    and [what], unless the [n] bytes from byte [at] lie inside [size]. *)

val blit_string : string -> int -> t -> int -> int -> unit
(** [blit_string s i b j n] copies the [n] bytes of [s] from [i] into [b]'s
    host memory from [j], keeping [b] reachable until it returns. It checks
    nothing and waits for nothing. *)

val blit_bytes : t -> int -> bytes -> int -> int -> unit
(** [blit_bytes b i s j n] copies the [n] bytes of [b]'s host memory from [i]
    into [s] from [j], as {!blit_string}. *)

val set_bytes : t -> int -> int -> char -> unit
(** [set_bytes b j n c] sets the [n] bytes of [b]'s host memory from [j] to
    [c], as {!blit_string}. *)

val address : string -> t -> int
(** [address fn b] is the address of [b]'s first byte as its device's work
    addresses it. Raises [Invalid_argument], naming [fn], if [b] is dead, an
    io device's, or named by handle only. *)

val aligned : t -> at:int -> int -> bool
(** [aligned b ~at n] is {!Rig.Buffer.aligned}[ b ~at n]. *)

val handle : t -> nativeint
(** [handle b] is {!Rig.Buffer.handle}[ b]. *)

val offset : t -> int
(** [offset b] is {!Rig.Buffer.offset}[ b]. *)
