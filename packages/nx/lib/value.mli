(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values: the representation of {!Nx.t}, which {!Nx.Repr} exposes.

    A value is a host array, a placed value over runtime buffers, or a traced
    value, which has no bytes. Each placed value and each traced value gets a
    fresh positive id when it is made, and identity tables key by it. The
    views of one placed storage share it, and each reads its bytes as its
    own dtype.

    A compiled call consumes a storage through rig's claims, which kill its
    buffers. Once one has, {!live}, {!reading}, {!shards} and the checked
    constructors raise [Invalid_argument] with the reason the call gave.
    {!storage}, {!placed} and {!buffer_on} check nothing of the kind. *)

open Nx_array

(** {1:values Values} *)

(** The type for values of {!Nx.t}. A [Host] value is at
    {!Placement.host}. *)
type ('a, 'b) t =
  | Host : ('a, 'b) Nx_array.t -> ('a, 'b) t
  | Placed : ('a, 'b) resident -> ('a, 'b) t
  | Traced : ('a, 'b) traced -> ('a, 'b) t

(** The type for placed values ({!Nx.Repr.Placed.t}). *)
and ('a, 'b) resident = {
  r_id : int;
  r_placement : Placement.t;  (** Never {!Placement.host}. *)
  r_dtype : ('a, 'b) Nx_dtype.t;
  r_view : View.t;  (** Each shard's view, the same on every shard. *)
  r_storage : storage;  (** Shared by every view of it. *)
}

(** The type for storage ({!Nx.Repr.Storage.t}), compared by identity. *)
and storage = {
  placement : Placement.t;  (** Where it lives, whichever value views it. *)
  bytes : int;  (** The bytes of each shard. *)
  buffers : Rig.Buffer.t list;
      (** One buffer per device of [placement], in its order. *)
}

(** The type for traced values ({!Nx.Repr.Traced.t}). *)
and ('a, 'b) traced = {
  t_id : int;
  t_placement : Placement.t;  (** Where the value lives. *)
  t_context : Placement.t;  (** Where a value made beside it is made. *)
  t_dtype : ('a, 'b) Nx_dtype.t;
  t_view : View.t;  (** The layout of the value it stands for. *)
  t_node : ('a, 'b) node;  (** The tracer's payload. *)
}

and ('a, 'b) node = ..
(** {!Nx.Repr.node}. *)

type packed = P : ('a, 'b) t -> packed

type context = Placement.t
(** The type for where a creation makes its value. *)

val fresh_id : unit -> int
(** [fresh_id ()] is an id larger than every id given before. Any domain may
    call it. *)

val outside_trace : unit -> 'a
(** [outside_trace ()] raises [Invalid_argument]. A read of a traced value's
    bytes calls it, since a traced value has no bytes. *)

(** {1:lenses Lenses} *)

val view : ('a, 'b) t -> View.t
(** {!Nx.Repr.view}. *)

val dtype : ('a, 'b) t -> ('a, 'b) Nx_dtype.t

val context : ('a, 'b) t -> context
(** {!Nx.Repr.context}. A host value's context is {!Placement.host}
    itself, which {!on_host} tests first. *)

val on_host : context -> bool
(** [on_host p] is [true] iff a creation at [p] makes a host value. *)

val placement : ('a, 'b) t -> Placement.t

val whole_view : ('a, 'b) resident -> View.t
(** [whole_view r] is [r]'s shard view with the shape of the whole value. *)

val global : Placement.t -> int array -> int array
(** [global p shape] is the shape of a value whose tiles at [p] have shape
    [shape]. *)

(** {1:host Host arrays} *)

val alloc : ('a, 'b) Nx_dtype.t -> int array -> ('a, 'b) Nx_array.t
(** [alloc dtype shape] is a fresh host array of [dtype] and [shape],
    C-contiguous from its first element. Its elements are unspecified. *)

val filled : ('a, 'b) Nx_dtype.t -> int array -> 'a -> ('a, 'b) Nx_array.t
(** [filled dtype shape v] is [alloc dtype shape] with every element [v]. *)

(** {1:storage Storage} *)

val storage : placement:Placement.t -> Rig.Buffer.t list -> storage
(** [storage ~placement buffers] is the storage of [buffers], one per device
    of [placement], each of the length of the first. It checks nothing, and
    {!shard_storage} checks. *)

val shard_storage : string -> Placement.t -> Rig.Buffer.t list -> storage
(** [shard_storage what p buffers] is {!Nx.Repr.Storage.v}[ p buffers], whose
    errors name [what]. *)

val live : storage -> Rig.Buffer.t list
(** [live s] is {!Nx.Repr.Storage.buffers}[ s]. *)

val reading : storage -> (unit -> 'a) -> 'a
(** [reading s f] is [f ()] under a read claim on [s]'s memory, so that no
    compiled call lends it to a result meanwhile. The claim raises before
    [f] runs if a buffer is dead, or [Rig.Lost] if a lost device can reach
    the memory, even for no elements. *)

val buffer_on : storage -> 'a list -> Device.t -> 'a
(** [buffer_on s bufs d] is the element of [bufs], one per device of [s]'s
    placement, that lies in [d]'s memory. Any device over that memory finds
    it.

    Raises [Invalid_argument] if [s] has no buffer in [d]'s memory. *)

val aligned : ('a, 'b) Nx_dtype.t -> Rig.Buffer.t -> bool
(** [aligned dtype b] is [true] iff [b]'s first byte is aligned to one of
    [dtype]'s elements, or [b] lies on an io device, whose bytes are read at
    any offset. *)

val check_host : string -> Rig.Buffer.t -> unit
(** [check_host what b] raises [Invalid_argument] naming [what] unless [b] is
    in the host's memory, where nx.cpu reads it by address. *)

(** {1:making Making values} *)

val placed :
  string -> Placement.t -> ('a, 'b) Nx_dtype.t -> View.t -> storage ->
  ('a, 'b) t
(** [placed what p dtype view s] is the value at [p] of [s] under [view]. It
    checks neither [view] nor alignment, and {!placed_value} checks both.

    Raises [Invalid_argument] naming [what] if [p] is the host's or if [s]'s
    devices do not hold [p]'s memories. *)

val placed_value :
  string -> Placement.t -> ('a, 'b) Nx_dtype.t -> View.t -> storage ->
  ('a, 'b) t
(** [placed_value what p dtype view s] is {!Nx.Repr.Placed.v}[ p dtype view s],
    whose errors name [what]. *)

val host_value :
  string -> ('a, 'b) Nx_dtype.t -> View.t -> Rig.Buffer.t -> ('a, 'b) t
(** [host_value what dtype view b] is the host value of [b] under [view].

    Raises [Invalid_argument] naming [what] if [b] is dead or off the host,
    or if [view] reaches outside it. *)

val traced :
  ?view:View.t -> context -> Placement.t -> ('a, 'b) Nx_dtype.t ->
  int array -> ('a, 'b) node -> ('a, 'b) t
(** [traced ?view ctx p dtype shape node] is
    {!Nx.Repr.Traced.v}[ ~context:ctx ?view p dtype shape node]. *)

val of_shards :
  string -> Placement.t -> ('a, 'b) Nx_dtype.t -> View.t ->
  Rig.Buffer.t list -> ('a, 'b) t
(** [of_shards what p dtype view buffers] is {!Nx.of_shards}, whose errors
    name [what]. *)

val of_buffer : ('a, 'b) Nx_dtype.t -> Shape.t -> Rig.Buffer.t -> ('a, 'b) t
val shards : ('a, 'b) t -> Rig.Buffer.t list * View.t
