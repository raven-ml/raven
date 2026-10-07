(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** SDMA packets: the commands of a copy queue.

    An SDMA engine runs its packets in order. A packet never wraps around the
    ring's end. A zero word is a one-word no-op, so a writer pads the ring's end
    with zeros and starts the packet at the ring's start. *)

val copy : Gpu.t -> dst:'v -> src:'v -> int -> 'v Packet.t
(** [copy g ~dst ~src n] copies [n] bytes from address [src] to address [dst],
    in linear copies of at most what [g]'s engine moves in one packet: 1 GiB
    from SDMA 4.4.2 below 5 and from 5.2, 4 MiB otherwise. It is [[]] for
    [n = 0].

    Raises [Invalid_argument] if [n < 0]. *)

val poll : 'v -> Packet.comparison -> 'v -> ?mask:int -> unit -> 'v Packet.t
(** [poll addr cmp v ~mask ()] waits until the 32 bits at [addr], masked by
    [mask], compare to the low 32 bits of [v] as [cmp] says. [mask] defaults to
    all 32 bits. *)

val fence : Gpu.t -> 'v -> 'v -> 'v Packet.t
(** [fence g addr v] writes the low 32 bits of [v] to [addr] once the packets
    before it have completed, bypassing the caches on engines that take a memory
    type (SDMA 5 on). *)

val trap : 'v Packet.t
(** [trap] raises an interrupt. *)

val timestamp : 'v -> 'v Packet.t
(** [timestamp addr] writes the GPU's clock ({!Pm4.source}) to [addr]. *)
