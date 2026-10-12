(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Broadcom NetXtreme RoCE NICs, driven over PCI with no kernel driver.

    The library takes a NIC's PCI function from Linux, as [Rig_amd_pci] takes a
    GPU's, and does what the kernel's [bnxt_en] and [bnxt_re] drivers would do:
    it resets the NIC, gives its firmware the context memory and command queue
    of its RoCE engine, and gives the NIC its address on the fabric. The NIC
    then moves bytes between memory of its machine and a peer's, by sends and
    receives on reliable-connected queue pairs over RoCE v2. The NICs it drives
    are the BCM57608 ([14e4:1760]).

    {b Hardware status.} The library is written, never run on hardware.

    Two kinds of objects carry the work:
    - A {e region} ({!Region}) is memory the NIC reads and writes, either system
      memory the library allocates or another device's memory, such as a GPU's
      behind its BAR. A byte of a region is named by its offset in it.
    - A {e queue pair} ({!Qp}) is a send ring and a receive ring, connected to
      one queue pair of a peer. A {e send} posts bytes of a region to the peer,
      where they land in the region of the receive the peer posted next. Each
      ring has its {e completions}, one per entry, in the order of the entries.
      Posting is stores to memory and one store to the NIC's doorbell register,
      with no system call.

    A transfer between two machines goes like this:
    + Each machine opens its NIC ({!open_}), makes a queue pair on it and
      allocates the regions it sends from or receives into.
    + The machines exchange their queue pairs' endpoints ({!Qp.val-endpoint}) by
      other means, and connect ({!Qp.connect}).
    + The receiver posts a receive ({!Qp.receive}), the sender a send
      ({!Qp.send}), and each waits for its completion ({!Qp.wait}).
    + Each closes its NIC ({!val-close}).

    {[
    let nic = Result.get_ok (Rig_bnxt.open_ 0) in
    let src, w = Result.get_ok (Rig_bnxt.Region.host nic 4096) in
    let qp = Result.get_ok (Rig_bnxt.Qp.make nic) in
    (* send Qp.endpoint qp to the peer, receive its endpoint [peer] *)
    Result.get_ok (Rig_bnxt.Qp.connect qp peer);
    Rig_pci.Window.write w 0 "hello";
    Rig_bnxt.Qp.send qp src ~at:0 5;
    ignore (Result.get_ok (Rig_bnxt.Qp.wait qp Rig_bnxt.Qp.Send ~us:1_000_000))
    ]}

    {1:fabric The fabric}

    A NIC is known on the fabric by its MAC address and by the IPv4 address
    [10.x.y.z], whose [x.y.z] are the last three bytes of its MAC address. A
    queue pair sends to its peer's MAC address directly, so two NICs reach each
    other on one Ethernet segment, at an MTU of 4096 bytes. The library trusts
    the fabric. A packet with the right queue pair and sequence number lands in
    the memory its receive names.

    {1:domains Domains}

    Any domain may make any call, concurrently with other domains, with these
    exceptions:
    - {!Qp.send}, {!Qp.poll} and {!Qp.wait} of one queue pair's [Send] side run
      one call at a time, and those of its [Receive] side too.
    - {!Region.deregister} must run after every call that names the region has
      returned, and {!val-close} after every other call on the NIC and its
      objects has returned.

    The NIC's requests to its firmware, which the other calls make, run one at a
    time.

    {1:errors Errors}

    A request the firmware or the system refuses returns [Error msg], with [msg]
    naming the request and the reason. A send or receive that fails is reported
    by its completion. An access to the NIC never fails, as
    {!Rig_pci.section-errors} states; a wait for the NIC returns [Error] once
    the NIC's machine has failed or at its bound.

    Misuse raises [Invalid_argument], such as a call on a closed NIC or its
    objects, but for {!val-name}, {!val-bus}, {!Qp.val-endpoint} and {!Qp.room},
    or a post on a deregistered region or an unconnected queue pair. A second
    {!Region.deregister} calls nothing and returns what the first returned, and
    a second {!val-close} does nothing.

    {1:references References}

    - InfiniBand Trade Association. {e InfiniBand Architecture Specification},
      Volume 1: reliable-connected queue pairs, packet sequence numbers and
      retries; Annex A17, RoCE v2. *)

(** {1:nics NICs}

    NIC [i] of a machine is the [i]th of its BCM57608 functions in bus order,
    whichever driver holds each. The NICs are named ["BNXT"], ["BNXT:1"],
    ["BNXT:2"], …. *)

type t
(** The type for NICs the process holds. *)

val buses : ?machine:Rig_pci.Machine.t -> unit -> string list
(** [buses ~machine ()] is the bus addresses of the NICs of [machine] (defaults
    to {!Rig_pci.Machine.this}), in bus order. It is [[]] for a machine that
    lists no PCI functions, such as this one off Linux. *)

val count : ?machine:Rig_pci.Machine.t -> unit -> int
(** [count ~machine ()] is [List.length (buses ~machine ())]. *)

val open_ : ?machine:Rig_pci.Machine.t -> int -> (t, string) result
(** [open_ ~machine i] holds NIC [i] of [machine] (defaults to
    {!Rig_pci.Machine.this}) and boots it. The NIC must be detached ({!detach}),
    and the process needs what {!Rig_pci.section-needs} lists for a function. It
    needs no firmware file, since the NIC boots from the firmware in its flash.
    Every open resets the NIC, whatever ran on it before. The process holds the
    NIC until {!val-close}, or until it exits, which closes it.

    [Error msg], with [msg] starting with the NIC's name, if [machine] has no
    NIC [i], if a process holds it, if its function cannot be taken or reset, or
    if its firmware refuses a request or does not answer in time.

    Raises [Invalid_argument] if [i < 0]. *)

val name : t -> string
(** [name nic] is [nic]'s name, such as ["BNXT:1"]. *)

val bus : t -> string
(** [bus nic] is the bus address of [nic]'s PCI function, such as
    ["0000:41:00.0"]. *)

val close : t -> unit
(** [close nic] unregisters the process from [nic]'s firmware, turns its bus
    mastering off and releases its function, then frees its system memory,
    regions included. The NIC reads and writes no memory of the process once
    [close] returns, and the next open resets it. *)

(** {1:regions Regions} *)

(** A region is memory the NIC reads and writes. *)
module Region : sig
  type nic := t

  type t
  (** The type for regions. *)

  val host : nic -> int -> (t * Rig_pci.Window.t, string) result
  (** [host nic n] is a region over new, zeroed system memory of [nic]'s
      machine, of [n] bytes rounded up to the machine's page
      ({!Rig_pci.Machine.val-page}), with the window through which the process
      reads and writes it. The process must not use the window once the region
      is deregistered, and must read the bytes a receive wrote only after its
      completion.

      [Error msg] if the machine has no memory for it, as
      {!Rig_pci.Function.alloc_dma} states, or the firmware refuses it.

      Raises [Invalid_argument] if [n <= 0]. *)

  val device : nic -> (int * int) list -> (t, string) result
  (** [device nic runs] is a region over another device's memory at [runs], (bus
      address, bytes) pairs in order, such as a GPU's memory behind its BAR. The
      memory must stay where it is until the region is deregistered.

      [Error msg] if [nic] is behind an IOMMU, which maps no other device's
      memory for it, or the firmware refuses it.

      Raises [Invalid_argument] if [runs] is empty, a run's address is negative,
      a run's size is not positive, or either is not a multiple of 4096. *)

  val deregister : t -> (unit, string) result
  (** [deregister r] deregisters [r] and frees the memory the library allocated
      for it, the memory {!host} made included. The NIC reads and writes [r] no
      more once [deregister] returns [Ok ()].

      It is [Error msg] if the firmware refuses. The NIC may then still read and
      write [r]'s memory, which stays allocated until {!val-close}. *)
end

(** {1:qps Queue pairs} *)

(** A queue pair is a send ring and a receive ring connected to one queue pair
    of a peer. Each ring holds 4095 entries. An entry's slot is free once its
    completion has been taken by {!Qp.poll} or {!Qp.wait}, which take the
    completions of a ring in the order of its entries.

    A queue pair {e fails} when one of its entries fails, when the NIC reports
    its failure, or when the NIC's machine fails. The NIC then runs none of its
    entries that have no completion, of either ring, nor those posted later, and
    each completes with the failure's [Error]. A failed queue pair stays failed.
*)
module Qp : sig
  type nic := t

  type t
  (** The type for reliable-connected queue pairs. *)

  type endpoint = {
    qp : int;  (** The queue pair's number. *)
    gid : string;
        (** Its NIC's 16-byte RoCE v2 address: the IPv4 address [10.x.y.z]
            mapped into IPv6 ({!section-fabric}). *)
    mac : string;  (** Its NIC's 6-byte MAC address. *)
  }
  (** The type for what a peer needs to connect to a queue pair. It is plain
      data, which a program sends to the peer by any means. Both queue pairs
      start at packet sequence number [0]. *)

  (** The type for a queue pair's rings. *)
  type side =
    | Send  (** The sends, and their completions. *)
    | Receive  (** The receives, and their completions. *)

  val make : nic -> (t, string) result
  (** [make nic] is an unconnected queue pair of [nic]. Its rings last until the
      NIC closes.

      [Error msg] if the machine has no memory for its rings or the firmware
      refuses it. *)

  val endpoint : t -> endpoint
  (** [endpoint qp] is what a peer's queue pair connects to [qp] with. *)

  val connect : t -> endpoint -> (unit, string) result
  (** [connect qp e] connects [qp] to the peer's queue pair [e] and makes it
      ready to send. A send to the peer is retried while the peer has posted no
      receive, without end, and a send the peer does not acknowledge is retried
      7 times, each after 67 ms, before its completion fails.

      [Error msg] if the firmware refuses a change of [qp]'s state.

      A [connect] the firmware refused part-way, called again, resumes where it
      stopped.

      Raises [Invalid_argument] if [qp] is connected, or [e]'s [gid] is not 16
      bytes or its [mac] not 6. *)

  val room : t -> side -> int
  (** [room qp side] is how many entries [side]'s ring of [qp] takes before it
      is full. *)

  val send : t -> Region.t -> at:int -> int -> unit
  (** [send qp r ~at n] posts the send of the [n] bytes at offset [at] of [r] to
      [qp]'s peer, which lands them in the region of its next receive.

      Raises [Invalid_argument] if [qp] is not connected, [r] is not a region of
      [qp]'s NIC or is deregistered, the bytes do not lie in [r], [n] is over
      [2{^31}], or [room qp Send] is [0]. *)

  val receive : t -> Region.t -> at:int -> int -> unit
  (** [receive qp r ~at n] posts the receive of up to [n] bytes at offset [at]
      of [r], which the next send of [qp]'s peer fills. A send longer than [n]
      fails both completions.

      Raises [Invalid_argument] as {!send} does, for [room qp Receive]. *)

  val poll : t -> side -> (int, string) result option
  (** [poll qp side] takes the completion of the oldest entry of [side] whose
      completion has not been taken, if it has one. It is [Some (Ok n)] if the
      entry completed, with [n] the bytes it moved: a send's bytes, or the
      length of the message a receive took. It is [Some (Error msg)] if the
      entry failed or the queue pair has, with [msg] saying why, and [None] if
      the entry has no completion yet or no entry is due. *)

  val wait : t -> side -> us:int -> (int, string) result
  (** [wait qp side ~us] is {!poll} once the entry has a completion, waiting for
      it at most [us] microseconds. A long wait sleeps between looks.

      [Error msg] also at the bound, or once the NIC's machine has failed.

      Raises [Invalid_argument] if [us < 0] or no entry of [side] is due. *)
end

(** {1:changes Changes to the machine}

    {!detach} and {!attach} change this machine as {!Rig_pci.Gpus} describes for
    a GPU, and their changes persist after the process. Each refuses a NIC a
    process holds, this one included, and needs [CAP_SYS_ADMIN] and write access
    to the files under [/sys/bus/pci] they write. *)

val detach : int -> (Rig_pci.Gpus.progress, string) result
(** [detach i] takes NIC [i] of this machine from its kernel driver as
    {!Rig_pci.Gpus.detach} does. Its network interfaces and RDMA devices go
    until {!attach} or a reboot. It leaves the other functions of the NIC's
    device, such as a card's other ports, and so returns [Error msg], naming
    them and their drivers, while the device has any and the NIC is not bound
    to [vfio-pci]: a NIC taken physically must be alone on its device.

    Raises [Invalid_argument] if [i < 0]. *)

val attach : int -> (unit, string) result
(** [attach i] gives NIC [i] of this machine back to its kernel driver as
    {!Rig_pci.Gpus.attach} does.

    Raises [Invalid_argument] if [i < 0]. *)
