(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Buffers: the implementation of {!Rig.Buffer}, whose {!Rig.Buffer.copy} is
    {!Copy.copy}.

    A buffer is a range of a memory record ({!Def.memory}): the offset of its
    first byte, its length, and the claim generation it was made in. It is
    live while that generation is its claim's and its borrow's lease has not
    ended ({!Def.lease}), and dead once {!Claim.consume} has raised the
    claim's generation. Every view and borrow of an allocation shares its
    claim, so a consumption kills every buffer over the allocation made
    before it; an unborrow kills only the borrow and its views.

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

(** {!Rig.Buffer.dma}. *)
type dma = Rig_edge.dma =
  | Bus of (int * int) list
  | Dmabuf of { fd : int; offset : int }

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

val offers : device -> kind -> bool
(** [offers d k] is {!Rig.offers}[ d k]. *)

val create : ?memory:kind -> device -> int -> t
(** [create ?memory d n] is {!Rig.Buffer.create}[ ?memory d n]. [Mapped]
    memory of a device whose memory the host addresses is the device's
    [Device] memory. *)

val of_io : device -> 'r im -> 'r -> access:access -> int -> t
(** [of_io d m r ~access n] is {!Rig.Buffer.of_io}[ d m r ~access n]. *)

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

val host_addresses : device -> bool
(** [host_addresses d] is whether its machine's host addresses all of [d]'s
    memory, and [d] has no [Pinned] memory: a machine's host, a memory device,
    and a driver's device that declares [host_addresses] or that the host of
    another machine reaches. *)

val is_borrowed : t -> bool
(** [is_borrowed b] is {!Rig.Buffer.is_borrowed}[ b]. *)

val view : t -> first:int -> length:int -> t
(** [view b ~first ~length] is {!Rig.Buffer.view}[ b ~first ~length]. *)

val spans : t -> bool
(** [spans b] is {!Rig.Buffer.spans}[ b]. *)

val overlaps : t -> t -> bool
(** [overlaps b b'] is {!Rig.Buffer.overlaps}[ b b']: whether they share a
    word. *)

val share_byte : t -> t -> bool
(** [share_byte b b'] is {!Rig.Buffer.share_byte}[ b b']. *)

val words : int -> int -> int * int
(** [words a n] is [(lo, hi)], the [n] bytes from [a] rounded out to whole
    words: [lo] the first word's start and [hi] the last word's end. *)

val places : t -> (int * int) list
(** [places b] is where [b]'s bytes lie, as [(space, first)]: the space and
    first byte that [rig_buffer_span] in [rig.h] gives, then, for memory
    outside the host's space whose pages the host maps, the host's space [0]
    and their host address. It is [[]] if [b] has no bytes. [overlaps b b']
    holds iff a place of [b] and one of [b'] share a space and their bytes,
    [length b] and [length b'] of them, rounded out to whole words of that
    space, meet there. Pages mapped later add a place and move none. *)

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
  window:bool ->
  string ->
  ('a, 'b) Bigarray.kind ->
  t ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** [array ~window fn k b] is [b]'s bytes as elements of [k], without a wait,
    a claim or an export: with [window], the bytes of [b]'s window where [b]
    is {!windowed}, which the caller flushes after its writes. An array over
    memory outside the host's heap and the caller's bigarrays shares the
    allocation's proxy ({!Memory.proxy}), which keeps the allocation from
    reuse while the array is reachable. Raises [Invalid_argument], naming
    [fn], as {!Rig.Buffer.read}. *)

val export :
  t -> ('a, 'b) Bigarray.kind -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** [export b k] is {!Rig.Buffer.export}[ b k]. It takes the memory out of
    the claims for good ({!Memory.mark_outside}). *)

val of_string : string -> t
(** [of_string s] is {!Rig.Buffer.of_string}[ s]. *)

val host_addressed : t -> bool
(** [host_addressed b] is [true] iff this process's host addresses [b]'s
    memory ({!Rig.Buffer.section-host_access}). *)

val windowed : t -> bool
(** [windowed b] is [true] iff [b]'s memory is [Mapped] memory of a device of
    this machine, which the host writes through its window
    ({!Memory.window}) and does not address otherwise. *)

val at_window : t -> t
(** [at_window b] is [b] over its window, for {!blit_string} and
    {!set_bytes} into a {!windowed} [b], which the caller runs with the
    owner's flush as one counted call ({!Memory.stored}). *)

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

val dma : t -> fn:string -> (dma, string) result
(** [dma b ~fn] is {!Rig.Buffer.dma}[ b ~fn]. *)

val handle : t -> nativeint
(** [handle b] is {!Rig.Buffer.handle}[ b]. *)

val offset : t -> int
(** [offset b] is {!Rig.Buffer.offset}[ b]. *)
