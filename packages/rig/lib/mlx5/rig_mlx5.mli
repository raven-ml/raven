(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ConnectX NICs of this machine, opened through Linux's RDMA subsystem.

    Here a NIC is one open port of a ConnectX NIC (the [mlx5] family). The
    kernel's [mlx5_ib] driver owns the NIC and shares it with other programs.
    {!open_} opens the NIC through its user verbs file,
    [/dev/infiniband/uverbsN], and has the kernel make the NIC's objects with
    the verbs ioctls. The NIC then moves bytes between this process's memory
    and a peer's by RDMA writes and reads, on reliable-connected queue pairs.

    This library is untested on real ConnectX NICs.

    Three kinds of objects carry the work:
    - A {e region} ({!Region}) is host memory of this process, or a device's
      memory exported as a dma-buf, that the NIC may read or write. Work
      entries name a region's bytes by the keys the NIC gave it.
    - A {e completion queue} ({!Cq}) is a ring the NIC writes completions in,
      which the process polls.
    - A {e queue pair} ({!Qp}) is a send ring the process writes work entries
      in, connected to one queue pair of a peer. Posting is stores to memory
      and one store to the NIC's doorbell register, with no system call. The
      NIC runs a queue pair's entries in order, and their writes land in that
      order in a region registered [In_order] ({!Region.type-order}).

    A transfer between two processes goes like this:
    + Each process opens its NIC ({!open_}), makes a completion queue and a
      queue pair on it, and registers the memory it shares
      ({!Region.register}).
    + The processes exchange their queue pairs' endpoints
      ({!Qp.val-endpoint}) and their regions' remote addresses
      ({!Region.remote}) by other means, and connect ({!Qp.connect}).
    + A process posts entries ({!Qp.post}), rings the doorbell ({!Qp.ring})
      and polls the completion queue ({!Cq.poll}), or arms it ({!Cq.arm}) and
      waits ({!val-wait}).
    + It destroys the queue pair, then the completion queue and the regions,
      or closes the NIC ({!val-close}).

    {[
    let nic = Result.get_ok (Rig_mlx5.open_ "mlx5_0") in
    let cq = Result.get_ok (Rig_mlx5.Cq.make nic 256) in
    let qp = Result.get_ok (Rig_mlx5.Qp.make cq 128) in
    (* send Qp.endpoint qp to the peer, receive its endpoint [peer] *)
    Result.get_ok (Rig_mlx5.Qp.connect qp peer);
    (* [src] is Region.local of a region here, [dst] the peer's
       Region.remote *)
    Rig_mlx5.Qp.post qp
      Rig_mlx5_abi.Entry.{ op = Write { src; dst }; signal = true };
    Rig_mlx5.Qp.ring qp
    ]}

    The kernel makes every object, and the process maps the rings, the
    doorbell records and the NIC's access regions. When the process's file of
    the NIC closes, at {!val-close} or at the process's death, the kernel
    destroys every object of the NIC, its regions included, so nothing of the
    NIC outlives the process.

    {1:domains Domains}

    Any domain may make any call, concurrently with other domains, with these
    exceptions:
    {ul
    {- {!Qp.post} and {!Qp.ring} of one queue pair run one call at a time.}
    {- {!Cq.poll} and {!Cq.arm} of one completion queue run one call at a
       time.}
    {- {!val-wait} of one NIC runs one call at a time.}
    {- [destroy] and [deregister] must run after every other call on their
       object has returned, and {!val-close} after every other call on the NIC
       and its objects has returned.}}
    A queue pair may be posted while another domain polls its completion
    queue.

    {!val-close}, {!Region.register}, {!Region.deregister}, {!Cq.make},
    {!Cq.destroy}, {!Qp.make}, {!Qp.destroy} and {!val-wait}, called from a
    callback inside a lock ([Rig]'s domains section), raise
    [Invalid_argument], having changed nothing
    ({!Rig_edge.section-drivers}).

    {1:errors Errors}

    A request the kernel refuses returns [Error msg], with [msg] naming the call
    and the kernel's reason. A queue pair or completion queue that fails while
    it runs is reported by {!val-wait}, and an entry that fails by its
    completion ({!Rig_mlx5_abi.Completion.type-status}).

    Misuse raises [Invalid_argument]: a call on a destroyed completion queue
    or queue pair, on a deregistered region, or on any object of a closed NIC.
    A second [destroy] or [deregister] on an open NIC calls nothing and returns
    what the first returned, and a second {!val-close} does nothing.

    A child of [fork] must not use its parent's NICs or their objects. Their
    rings are not mapped in the child.

    {1:references References}

    - InfiniBand Trade Association. {e InfiniBand Architecture Specification},
      Volume 1: reliable-connected queue pairs, packet sequence numbers,
      acknowledgement timeouts and retries; Annex A17, RoCE v2.
    - rdma-core's [providers/mlx5/mlx5dv.h] and its [mlx5dv] manual pages:
      the formats the process and the NIC share ({!Rig_mlx5_abi}).
    - The Linux kernel's
      {{:https://docs.kernel.org/infiniband/user_verbs.html}Userspace verbs
       access} documentation: the verbs files and the pinning of registered
      memory. *)

(** {1:nics NICs}

    A NIC is named as the kernel names its RDMA device, such as ["mlx5_0"]. A
    ConnectX function has one port, and this library opens port 1.

    {b The machine's files.} {!names} and {!open_} read this machine's files:
    the RDMA devices under [/sys/class/infiniband] and
    [/sys/class/infiniband_verbs], and the verbs files under
    [/dev/infiniband]. Off Linux this machine has none of these files, so
    [names ()] is [Ok []] and {!open_} returns [Error].

    {b Privileges.} Opening needs read and write access to the NIC's verbs
    file and no other privilege. Registering memory locks its pages, so the
    process's locked-memory limit ([RLIMIT_MEMLOCK]) bounds the bytes it
    registers unless the process has [CAP_IPC_LOCK].

    {b Kernels.} Registering dma-buf memory needs Linux 5.12 or later. *)

type t
(** The type for open NIC ports. Each has one protection domain, which its
    regions and queue pairs share. *)

val names : unit -> (string list, string) result
(** [names ()] is the names of this machine's RDMA devices whose PCI function
    the [mlx5_core] driver holds, sorted as strings, so ["mlx5_10"] comes
    before ["mlx5_2"]. It is [Ok []] where [/sys/class/infiniband] does not
    exist, and [Error msg] if it cannot be read, [msg] naming it and the
    reason. Listing them changes nothing on the machine. *)

val open_ : string -> (t, string) result
(** [open_ name] opens port 1 of the NIC [name] and maps its access regions,
    the NIC's pages through which the process rings doorbells.

    The result is [Error msg] if:
    - no RDMA device [name] exists, or [mlx5_core] does not hold its PCI
      function;
    - it has no verbs file, or its verbs file cannot be opened, [msg] naming
      the file and the reason;
    - the kernel refuses a call, [msg] naming the call and the reason;
    - port 1 is not active, or is an Ethernet port with no RoCE v2 global
      identifier;
    - the kernel gives fewer doorbell registers than one system page holds. *)

val name : t -> string
(** [name nic] is the kernel's name of [nic], such as ["mlx5_0"]. *)

val bus : t -> string
(** [bus nic] is the bus address of [nic]'s PCI function, such as
    ["0000:17:00.0"]. *)

val link : t -> [ `Infiniband | `Ethernet ]
(** [link nic] is the link layer of [nic]'s port. On [`Ethernet] (RoCE v2)
    every packet carries a global route header. *)

val close : t -> unit
(** [close nic] destroys every object of [nic] and closes it. The NIC reads
    and writes no memory of this process once [close] returns. Later calls on
    [nic] and its objects raise [Invalid_argument], except {!val-name},
    {!val-bus} and {!link}, which still return, and [close], which does
    nothing. *)

(** {1:regions Regions} *)

(** A region is memory the NIC may read or write. *)
module Region : sig
  type nic := t

  type t
  (** The type for registered memory. *)

  (** The type for memory to register. *)
  type memory =
    | Host of Rig_mlx5_abi.buffer
        (** The bytes of a bigarray, whose pages stay resident while
            registered. The NIC keeps the bigarray reachable until a
            {!val-deregister} of the region returns [Ok ()] or the NIC
            closes. *)
    | Dmabuf of { fd : int; offset : int; bytes : int }
        (** [bytes] bytes at [offset] in the dma-buf [fd], a device's memory a
            driver exported. The kernel keeps the dma-buf while it is
            registered, so [fd] may be closed once {!val-register} returns. *)

  (** The type for what peers may do to a region. Whatever the access, the
      NIC reads a region for this process's writes and writes it for this
      process's reads. *)
  type access =
    | Local  (** Peers neither read nor write it. *)
    | Remote_write  (** Peers write it. *)
    | Remote  (** Peers read and write it. *)

  (** The type for the order in which the NIC's writes land in a region.
      They are a peer's writes and the bytes of this process's reads. *)
  type order =
    | In_order
        (** The writes of one queue pair land in the order it posted them. *)
    | Any_order
        (** Writes may land in any order. A reader must not take one write's
            bytes as a sign that an earlier write's bytes have landed. *)

  val register : nic -> access -> order -> memory -> (t, string) result
  (** [register nic a o m] registers [m] with [nic] for access [a] and write
      order [o]. It is [Error msg] if the kernel refuses, such as when the
      process's locked-memory limit ([RLIMIT_MEMLOCK]) cannot hold [m]'s
      pages.

      Raises [Invalid_argument] if [m] has no bytes, or its offset or
      descriptor is negative. *)

  val deregister : t -> (unit, string) result
  (** [deregister r] deregisters [r]. The NIC reads and writes [r] no more
      once [deregister] returns [Ok ()].

      It is [Error msg] if the kernel refuses. [r] then counts as
      deregistered, but the NIC may still read and write its memory until it
      closes, and keeps a [Host] bigarray reachable until then.

      {b Warning.} After such an [Error], the program must not reuse that
      memory before {!val-close}. *)

  val local : t -> int -> int -> Rig_mlx5_abi.Entry.local
  (** [local r at n] is the [n] bytes at offset [at] of [r], as this process's
      entries name them.

      Raises [Invalid_argument] if they are not in [r]. *)

  val remote : t -> int -> Rig_mlx5_abi.Entry.remote
  (** [remote r at] is offset [at] of [r], as a peer's entries name it.

      Raises [Invalid_argument] if [at] is not in [r] or [r] is [Local]. *)
end

(** {1:cqs Completion queues} *)

(** A completion queue is a ring where the NIC reports the entries that
    completed. *)
module Cq : sig
  type nic := t

  type t
  (** The type for completion queues. *)

  val make : nic -> int -> (t, string) result
  (** [make nic n] is a completion queue of [nic] with room for [n]
      completions, [n] rounded up to a power of two no smaller than [2]. Its
      queue pairs hold at most that many entries together ({!Qp.make}), so it
      never overflows. It is [Error msg] if the kernel refuses or the process
      cannot allocate the ring.

      Raises [Invalid_argument] if [n] is not in \[[1];[2{^22}]\]. *)

  val poll : t -> Rig_mlx5_abi.Completion.t option
  (** [poll cq] is the next completion of [cq], which it consumes, or [None]
      if the NIC wrote none after those [poll] consumed. *)

  val arm : t -> unit
  (** [arm cq] asks [cq]'s NIC to raise one {!Completion} event at the next
      completion of [cq] after those [poll] consumed. *)

  val destroy : t -> (unit, string) result
  (** [destroy cq] destroys [cq] and frees its ring.

      It is [Error msg] if the kernel refuses, as it does while it keeps a
      queue pair of [cq] that it refused to destroy. [cq] then counts as
      destroyed and its ring is freed all the same.

      Raises [Invalid_argument] if a queue pair of [cq] is not destroyed. *)
end

(** {1:qps Queue pairs} *)

(** A queue pair is a send ring connected to one queue pair of a peer. *)
module Qp : sig
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
  (** The type for what a peer needs to connect to a queue pair. It is plain
      data, which a program sends to the peer by any means. *)

  val make : Cq.t -> int -> (t, string) result
  (** [make cq n] is an unconnected queue pair of [cq]'s NIC with room for [n]
      entries, [n] rounded up to a power of two, whose completions go to [cq].
      It starts at a random packet sequence number. It has no receive ring, so
      peers read and write this process's regions through it and send it no
      messages.

      It is [Error msg] if the kernel refuses or the process cannot allocate
      the ring.

      Raises [Invalid_argument] if [n] is not in \[[1];[2{^15}]\], or the
      queue pairs of [cq], each counted at its rounded size, would hold more
      entries than [cq]. *)

  val number : t -> int
  (** [number qp] is [qp]'s number, which completions name. *)

  val endpoint : t -> endpoint
  (** [endpoint qp] is what a peer's queue pair connects to [qp] with. *)

  val connect : t -> endpoint -> (unit, string) result
  (** [connect qp e] connects [qp] to the peer's queue pair [e], at the
      smaller of both ports' MTUs, and makes it ready to send. A request of
      [qp] that the peer does not acknowledge is retried 7 times, each after
      an acknowledgement timeout of 1.07 s on InfiniBand or 67 ms on Ethernet,
      and its entry then fails with
      {!Rig_mlx5_abi.Completion.Retry_exceeded}.

      It is [Error msg] if the kernel refuses a change of [qp]'s state.

      Raises [Invalid_argument] if [qp] was connected, [e]'s address is not of
      [qp]'s link layer, or a [Gid] is not 16 bytes. *)

  val room : t -> int
  (** [room qp] is how many entries [qp] takes before its ring is full. An
      entry's slot is free once a completion of it, or of an entry after it,
      has been polled. *)

  val post : t -> Rig_mlx5_abi.Entry.t -> unit
  (** [post qp e] writes [e] in [qp]'s ring. The NIC runs it after the next
      {!ring}.

      Raises [Invalid_argument] if [qp] is not connected or [room qp] is
      [0]. *)

  val ring : t -> unit
  (** [ring qp] hands the NIC the entries posted since the last [ring]. It
      does nothing if there are none. *)

  val destroy : t -> (unit, string) result
  (** [destroy qp] destroys [qp] and frees its ring. The NIC runs none of its
      entries and reads and writes no memory for it once [destroy] returns
      [Ok ()]. The entries of its peer's queue pair fail once their retries
      end.

      It is [Error msg] if the kernel refuses. [qp] then counts as destroyed
      and its ring is freed all the same, but the kernel's queue pair may
      still complete its entries, which keep their room in its completion
      queue ({!make}) until the NIC closes. *)
end

(** {1:events Events} *)

(** The type for events of a NIC. *)
type event =
  | Completion of Cq.t  (** An armed completion queue holds a completion. *)
  | Failure of string
      (** The NIC failed, a queue pair failed, or a completion queue
          overflowed, as the message names. *)
  | Port of string  (** The port's state changed, as the message says. *)

val wait : t -> ms:int -> event list
(** [wait nic ~ms] is the events of [nic] since the last [wait], once there
    is one, or [[]] if none came within [ms] milliseconds. An armed completion
    queue raises one {!Completion} per {!Cq.arm}. Other domains run while it
    waits.

    Raises [Invalid_argument] if [ms < 0]. *)

(**/**)

(** {1:paths Paths}

    For tests. A {e path} reaches a NIC one way, as {!open_} does through
    Linux's RDMA subsystem: it has the kernel make the NIC's objects and gives
    them to {!make}. A test wraps a path, such as to make its [destroy] fail,
    or opens a path of another driver, such as Soft-RoCE's. Each object the
    kernel makes carries {e driver data}, bytes the kernel's [mlx5] driver
    reads when it makes the object and bytes it writes back. This library
    writes and reads the driver data, and a path passes it to and from the
    kernel unchanged. *)

(** The type for the kernel's events. *)
type kernel_event =
  | Completed of int  (** An armed completion queue, by its tag. *)
  | Cq_error of int  (** A completion queue overflowed, by its tag. *)
  | Qp_failed of int * string
      (** A queue pair failed, by its tag, and the kernel's kind. *)
  | Nic_failed  (** The NIC failed. *)
  | Port_changed of string  (** The port's state changed, to this state. *)

(** The type for a queue pair's state changes, in the order {!Qp.connect}
    makes them. *)
type transition =
  | Init
      (** To initialised, on port 1, with peers allowed to read and write. *)
  | Ready_to_receive of {
      peer : Qp.endpoint;
      mtu : int;  (** In bytes. *)
      gid : int option;
          (** The index of the port's global identifier the queue pair sends
              from, on Ethernet. *)
      served : int;  (** The peer's reads in flight it serves concurrently. *)
    }
  | Ready_to_send of {
      psn : int;  (** The packet sequence number it sends from. *)
      timeout : int;  (** The acknowledgement timeout's exponent. *)
      retries : int;  (** The times an unacknowledged request is retried. *)
      reads : int;  (** The reads in flight it starts concurrently. *)
    }

type context = {
  answer : string;  (** The driver's answer. *)
  port : [ `Infiniband of int | `Ethernet of (int * string) list ];
      (** The NIC's port: on InfiniBand, its local identifier; on Ethernet,
          its RoCE v2 global identifiers, each with its index in the port's
          table. *)
  mtu : int;  (** The port's active MTU, in bytes. *)
  reads : int;
      (** The most reads in flight a queue pair starts concurrently. *)
  served : int;
      (** The most of the peer's reads in flight a queue pair serves
          concurrently. *)
}
(** The type for what a path reads of a NIC once it has made the process's
    {e context} of it, the kernel object that holds the process's other
    objects of that NIC. *)

type registered = {
  handle : int;  (** The kernel's handle of the region. *)
  local : int;  (** Its local key. *)
  remote : int;  (** Its remote key. *)
}
(** The type for memory a path registered. *)

type path = {
  name : string;  (** The kernel's name of the NIC. *)
  bus : string;  (** The bus address of its PCI function. *)
  page : int;  (** The host's page size, in bytes. *)
  context : string -> int -> (context, string) result;
      (** [context d n] makes the process's context of the NIC and its
          protection domain, giving the driver [d], and reads the NIC's port
          and limits. The driver's answer is at most [n] bytes. [make] calls
          it first, once. *)
  map : int -> int -> (int, string) result;
      (** [map off n] maps the [n] bytes at offset [off] of the NIC's file and
          is their host address. *)
  register :
    Region.access ->
    Region.order ->
    Region.memory ->
    (registered, string) result;
      (** [register a o m] registers [m] for [a] and [o]. The NIC knows the
          first byte of [Host] memory by its host address, and of [Dmabuf]
          memory by its offset. *)
  cq :
    entries:int ->
    tag:int ->
    string ->
    int ->
    (int * (unit -> int) * string, string) result;
      (** [cq ~entries ~tag d n] makes a completion queue of [entries]
          completions, a power of two, whose events carry [tag], giving the
          driver [d]. It is the queue's handle, a function that is the number
          of the queue's events [wait] took from the kernel, and the driver's
          answer, of at most [n] bytes. The path's [wait] is the count's only
          writer, and counts each event once. *)
  qp :
    cq:int ->
    entries:int ->
    tag:int ->
    string ->
    int ->
    (int * int * string, string) result;
      (** [qp ~cq ~entries ~tag d n] makes a reliable-connected queue pair
          with [entries] send entries and no receive ring, completing into the
          queue of handle [cq], whose events carry [tag], giving the driver
          [d]. It is the pair's handle, its number and the driver's answer, of
          at most [n] bytes. *)
  modify : int -> transition -> (unit, string) result;
      (** [modify h t] changes the state of the queue pair of handle [h]. *)
  destroy : [ `Region | `Cq | `Qp ] -> int -> (unit, string) result;
      (** [destroy k h] destroys the object of kind [k] and handle [h]. *)
  wait : int -> kernel_event list;
      (** [wait ms] is the kernel's events once there is one, or [[]] after
          [ms] milliseconds. The events stay the path's until [reported]:
          each [wait] returns them again, with those that came since. *)
  reported : unit -> unit;
      (** [reported ()] drops the events the last [wait] returned, which
          their caller now holds. It allocates nothing and polls nowhere, so
          a caller that calls it as its last step loses no event to an
          exception, and an exception before it leaves them to the next
          [wait]. *)
  close : unit -> unit;
      (** [close ()] unmaps what [map] mapped and closes the NIC's file, and
          the kernel then destroys every object. *)
}
(** The type for what a path gives a NIC. Each function but [wait],
    [reported] and [close] returns the kernel's refusal as [Error msg],
    naming the call and the reason. Any domain may call them concurrently,
    except [wait] and [reported], which one domain calls at a time, and
    [close], which runs last. *)

val make : path -> (t, string) result
(** [make p] is the NIC [p] reaches, with its context made and its access
    regions, the NIC's pages through which the process rings doorbells, mapped
    through [p].

    [Error msg], with [msg] naming the NIC, if [p] refuses, if [p] reports a
    port with no RoCE v2 global identifier, or if the kernel gives fewer
    doorbell registers than one system page holds. A [make] that fails or
    raises closes [p]. *)

val path : ?root:string -> string -> (path, string) result
(** [path ~root name] is a path through the verbs file of the RDMA device
    [name] under the directory [root] (defaults to ["/"]), on port 1, whatever
    driver holds the device. The path passes the driver data it is given to
    the kernel unchanged, and names the driver the kernel lists for [name].
    [open_under ~root name] is {!make} of this path for a NIC that
    [mlx5_core] holds. The caller must close a path it does not make into a
    NIC.

    The result is [Error msg] if no RDMA device [name] exists, it has no verbs
    file, its verbs file cannot be opened ([msg] names the file and the
    reason), or the kernel lists no driver for [name]. *)

val names_under : root:string -> unit -> (string list, string) result
(** [names_under ~root ()] is {!names} of the machine whose files are under
    the directory [root], such as a container's view of its host. *)

val open_under : root:string -> string -> (t, string) result
(** [open_under ~root name] is {!open_} of the machine whose files are under
    the directory [root]. *)
