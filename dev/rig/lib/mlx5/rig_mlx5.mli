(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ConnectX NICs, driven once open.

    A NIC of this library is the port of a ConnectX NIC (the [mlx5] family) that
    a {e path} opened: a library that reaches the NIC one way, such as through
    Linux's RDMA subsystem, and gives this one the kernel's objects as a
    {!type-path} ({!make}). The NIC moves bytes between this process's memory
    and a peer's by RDMA writes and reads, on reliable-connected queue pairs:
    - a {e region} ({!Region}) is memory the NIC may read or write: host memory
      of this process, or a device's memory exported as a dma-buf. Work entries
      name memory of regions, by the keys the NIC gave them;
    - a {e completion queue} ({!Cq}) is a ring the NIC writes completions in,
      which the process polls;
    - a {e queue pair} ({!Qp}) is a send ring the process writes work entries
      in, connected to one queue pair of a peer. Posting is stores to memory and
      a store to the NIC's doorbell register, with no system call. The NIC runs
      a queue pair's entries in order, and its writes land in the peer's memory
      in that order.

    {[
    let nic = Result.get_ok (Rig_mlx5_uverbs.open_ "mlx5_0") in
    let cq = Result.get_ok (Rig_mlx5.Cq.make nic 256) in
    let qp = Result.get_ok (Rig_mlx5.Qp.make nic cq 128) in
    (* send Qp.endpoint qp to the peer, receive its endpoint [peer] *)
    Result.get_ok (Rig_mlx5.Qp.connect qp peer);
    Rig_mlx5.Qp.post qp
      { op = Write { src; dst }; signal = true; fence = false };
    Rig_mlx5.Qp.ring qp
    ]}

    The kernel creates every object; the rings, the doorbell records and the
    NIC's access region are mapped into the process. Objects of a NIC die with
    the process: the kernel destroys them when its file closes.

    {b Domains.} Every function may be called from any domain, at the same time
    as others, but for these: a queue pair's {!Qp.post} and {!Qp.ring} run one
    call at a time, as do a completion queue's {!Cq.poll} and {!Cq.arm}, and
    {!wait}. A queue pair may be posted while its completion queue is polled in
    another domain. Destroying an object runs after every other call on it
    returned, and no call on it follows.

    {b Fork.} A child process uses none of its parent's NICs.

    {b References.}
    - rdma-core's [libibverbs] and [providers/mlx5]: [verbs.c] (the driver data
      of each object), [qp.c] and [cq.c] (posting, polling and arming).
    - The Linux kernel's [drivers/infiniband/hw/mlx5]: [main.c] (contexts and
      access regions), [qp.c] and [cq.c] (the user rings). *)

(** {1:nics NICs} *)

type t
(** The type for open NIC ports, with their protection domain. *)

val name : t -> string
(** [name nic] is the kernel's name of [nic], such as ["mlx5_0"]. *)

val bus : t -> string
(** [bus nic] is the bus address of [nic]'s PCI function, such as
    ["0000:17:00.0"]. *)

val link : t -> [ `Infiniband | `Ethernet ]
(** [link nic] is the link layer of [nic]'s port. On [`Ethernet] (RoCE v2) every
    packet carries a global route header. *)

val close : t -> unit
(** [close nic] destroys every object of [nic] and closes it. The NIC reads and
    writes no memory of this process once [close] returns. Calls on [nic] and
    its objects that follow raise [Invalid_argument]. A later [close] does
    nothing. *)

(** {1:regions Regions} *)

(** Regions: memory the NIC may read or write. *)
module Region : sig
  type nic := t

  type t
  (** The type for registered memory. *)

  (** The type for memory to register. *)
  type memory =
    | Host of { address : int; bytes : int }
        (** [bytes] bytes of this process's memory at [address], whose pages
            stay resident while registered. *)
    | Dmabuf of { fd : int; offset : int; bytes : int }
        (** [bytes] bytes at [offset] in the dma-buf [fd], a device's memory a
            driver exported. *)

  (** The type for what peers may do to a region. Every region is one this
      process's entries read, and one the NIC writes for its reads. *)
  type access =
    | Local  (** Peers neither read nor write it. *)
    | Remote_write  (** Peers write it. Their writes may land in any order. *)
    | Remote
        (** Peers read and write it. Writes land in the order a queue pair
            posted them, after the writes posted before them. *)

  val register : nic -> access -> memory -> (t, string) result
  (** [register nic a m] registers [m] with [nic] for [a]. It is [Error msg] if
      the kernel refuses, such as when the process's locked-memory limit
      ([RLIMIT_MEMLOCK]) cannot hold [m]'s pages.

      Raises [Invalid_argument] if [m]'s bytes are not positive, its address,
      offset or descriptor is negative. *)

  val deregister : t -> unit
  (** [deregister r] gives [r] back. The NIC reads and writes [r] no more once
      [deregister] returns. *)

  val address : t -> int
  (** [address r] is the address the NIC knows [r]'s first byte by: a host
      region's address, a dma-buf region's offset. *)

  val bytes : t -> int
  (** [bytes r] is the length of [r]. *)

  val local : t -> int -> int -> Rig_mlx5_abi.Entry.local
  (** [local r at n] is the [n] bytes at offset [at] of [r], for an entry of
      this process.

      Raises [Invalid_argument] if they are not in [r]. *)

  val remote : t -> int -> Rig_mlx5_abi.Entry.remote
  (** [remote r at] is offset [at] of [r], for a peer's entries, which [r]'s
      {!access} allows.

      Raises [Invalid_argument] if [at] is not in [r] or [r] is [Local]. *)
end

(** {1:cqs Completion queues} *)

(** Completion queues: where the NIC reports entries that completed. *)
module Cq : sig
  type nic := t

  type t
  (** The type for completion queues. *)

  val make : nic -> int -> (t, string) result
  (** [make nic n] is a completion queue of [nic] with room for at least [n]
      completions. Its queue pairs hold at most [n] entries together
      ({!Qp.make}), so that it never overflows.

      Raises [Invalid_argument] if [n] is not in \[[1];[2{^22}]\]. *)

  val poll : t -> Rig_mlx5_abi.Completion.t option
  (** [poll cq] is the next completion of [cq], which it consumes, or [None] if
      the NIC wrote none since the last. *)

  val arm : t -> unit
  (** [arm cq] asks [nic] to raise one {!Completion} event at the next
      completion of [cq] after those [poll] consumed. *)

  val destroy : t -> unit
  (** [destroy cq] destroys [cq].

      Raises [Invalid_argument] if a queue pair of [cq] is not destroyed. *)
end

(** {1:qps Queue pairs} *)

(** Queue pairs: send rings connected to one queue pair of a peer. *)
module Qp : sig
  type nic := t

  type t
  (** The type for reliable-connected queue pairs. *)

  (** The type for the addresses of ports. *)
  type address =
    | Lid of int  (** An InfiniBand port's local identifier. *)
    | Gid of string
        (** An Ethernet port's 16-byte RoCE v2 global identifier, an IPv6
            address or an IPv4 one mapped into IPv6. *)

  type endpoint = {
    qp : int;  (** The queue pair's number. *)
    psn : int;  (** The packet sequence number it starts at. *)
    address : address;  (** Its port's address. *)
    mtu : int;  (** Its port's active MTU, in bytes. *)
  }
  (** The type for what a peer needs to connect to a queue pair: plain data,
      which crosses machines. *)

  val make : nic -> Cq.t -> int -> (t, string) result
  (** [make nic cq n] is a queue pair of [nic] with room for [n] entries, [n]
      rounded up to a power of two, whose completions go to [cq]. It starts at a
      random packet sequence number and connects to nothing. It has no receive
      ring: peers write and read its regions, and send it nothing.

      Raises [Invalid_argument] if [n] is not in \[[1];[2{^15}]\], or the queue
      pairs of [cq] would hold more entries than [cq]. *)

  val number : t -> int
  (** [number qp] is [qp]'s number, which completions name. *)

  val endpoint : t -> endpoint
  (** [endpoint qp] is what a peer's queue pair connects to [qp] with. *)

  val connect : t -> endpoint -> (unit, string) result
  (** [connect qp e] connects [qp] to the peer's queue pair [e], at the smaller
      of both ports' MTUs, and makes it ready to send. A peer's request that
      [qp]'s peer does not acknowledge is retried 7 times, each after the port's
      acknowledgement timeout (1.07 s on InfiniBand, 67 ms on [`Ethernet]); then
      the entry fails with {!Rig_mlx5_abi.Completion.Retry_exceeded}.

      Raises [Invalid_argument] if [qp] was connected, or [e]'s address is not
      of [qp]'s link layer. *)

  val room : t -> int
  (** [room qp] is how many entries [qp] takes before its ring is full: an
      entry's slot is free once a completion of it, or of an entry after it, was
      polled. *)

  val post : t -> Rig_mlx5_abi.Entry.t -> unit
  (** [post qp e] writes [e] in [qp]'s ring. The NIC runs it after the next
      {!ring}.

      Raises [Invalid_argument] if [qp] is not connected or [room qp] is [0]. *)

  val ring : t -> unit
  (** [ring qp] tells the NIC that the entries posted before it wait: it stores
      the producer count in [qp]'s doorbell record after the entries, and the
      last entry's first 8 bytes in [qp]'s doorbell register after the record.
      It does nothing if no entry was posted since the last. *)

  val destroy : t -> unit
  (** [destroy qp] destroys [qp]. The NIC runs none of its entries and reads and
      writes no memory for it once [destroy] returns. Its peer's entries fail
      once their retries end. *)
end

(** {1:events Events} *)

(** The type for events of a NIC. *)
type event =
  | Completion of Cq.t  (** An armed completion queue holds a completion. *)
  | Failure of string
      (** The NIC, a queue pair or a completion queue failed: the message names
          which, and how. *)
  | Port of string
      (** The port's state changed, as the message says. Nothing failed. *)

val wait : t -> ms:int -> event list
(** [wait nic ~ms] is the events of [nic] since the last [wait], once there is
    one, or [[]] after [ms] milliseconds. An armed completion queue raises one
    {!Completion} per {!Cq.arm}.

    Raises [Invalid_argument] if [ms < 0]. *)

(** {1:paths Paths}

    For the libraries that open NICs. A path reaches a NIC one way, makes its
    objects, and gives them to {!make} as a {!type-path}. The driver data, the
    bytes the [mlx5] driver reads and writes beside each object, are this
    library's: the path carries them. *)

(** The type for the kernel's events. *)
type kernel_event =
  | Completed of int  (** An armed completion queue, by its tag. *)
  | Cq_error of int  (** A completion queue overflowed, by its tag. *)
  | Qp_failed of int * string
      (** A queue pair failed, by its tag, and the kernel's kind. *)
  | Nic_failed  (** The NIC failed. *)
  | Port_changed of string  (** The port's state changed: its new state. *)

(** The type for a queue pair's state changes, in the order {!make} takes them.
*)
type transition =
  | Init  (** To initialised: on port 1, peers may read and write. *)
  | Ready_to_receive of {
      peer : Qp.endpoint;
      mtu : int;  (** In bytes. *)
      gid : int option;
          (** The index of the port's global identifier the queue pair sends
              from, on Ethernet. *)
      reads : int;  (** The peer's reads in flight it serves at once. *)
    }
  | Ready_to_send of {
      psn : int;
      timeout : int;  (** The acknowledgement timeout's exponent. *)
      retries : int;
      reads : int;  (** The reads in flight it starts at once. *)
    }

type context = {
  answer : string;  (** The driver's answer. *)
  port : [ `Infiniband of int | `Ethernet of (int * string) list ];
      (** The NIC's port: on InfiniBand, its local identifier; on Ethernet, its
          RoCE v2 global identifiers, each with its index in the port's table.
      *)
  mtu : int;  (** The port's active MTU, in bytes. *)
  reads : int;  (** The reads a queue pair starts at once, at most. *)
  served : int;  (** The peer's reads a queue pair serves at once, at most. *)
}
(** The type for what a path learns once the process's context of a NIC exists.
*)

type path = {
  name : string;  (** The kernel's name of the NIC. *)
  bus : string;  (** The bus address of its PCI function. *)
  page : int;  (** The host's page size, in bytes. *)
  context : string -> int -> (context, string) result;
      (** [context d n] makes the process's context of the NIC and its
          protection domain, giving the driver [d], and reads the NIC's port and
          limits; the driver's answer is at most [n] bytes. [make] calls it
          first, once. *)
  map : int -> int -> (int, string) result;
      (** [map off n] maps the [n] bytes at offset [off] of the NIC's file:
          their host address. *)
  register : Region.access -> Region.memory -> (int * int * int, string) result;
      (** [register a m] registers [m] for [a], at the address {!Region.address}
          names: its handle, local key and remote key. *)
  cq : entries:int -> tag:int -> string -> int -> (int * string, string) result;
      (** [cq ~entries ~tag d n] makes a completion queue of [entries]
          completions, a power of two, whose events carry [tag], giving the
          driver [d]: its handle and the driver's answer, of at most [n] bytes.
      *)
  qp :
    cq:int ->
    entries:int ->
    tag:int ->
    string ->
    int ->
    (int * int * string, string) result;
      (** [qp ~cq ~entries ~tag d n] makes a reliable-connected queue pair with
          [entries] send entries and no receive ring, completing into the queue
          of handle [cq], whose events carry [tag], giving the driver [d]: its
          handle, its number and the driver's answer, of at most [n] bytes. *)
  modify : int -> transition -> (unit, string) result;
      (** [modify h t] changes the state of the queue pair of handle [h]. *)
  destroy : [ `Region | `Cq | `Qp ] -> int -> unit;
      (** [destroy k h] destroys the object of kind [k] and handle [h]. *)
  wait : int -> kernel_event list;
      (** [wait ms] is the kernel's events once there is one, or [[]] after [ms]
          milliseconds. *)
  close : unit -> unit;
      (** [close ()] closes the NIC's file and unmaps what [map] mapped: the
          kernel destroys every object. *)
}
(** The type for what a path gives a NIC. Each function answers the kernel's
    refusal as [Error msg], naming the call and the reason. Any domain may call
    them at the same time, except [wait], which one domain calls at a time, and
    [close], which runs last. *)

val make : path -> (t, string) result
(** [make p] is the NIC [p] reaches: its context, made through [p], and its
    access region, mapped. It is [Error msg] if [p] refuses, with [p]'s message.
    A failed [make] closes [p]. *)
