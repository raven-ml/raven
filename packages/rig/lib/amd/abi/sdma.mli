(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The SDMA packets a copy queue reads.

    An SDMA engine runs its packets in order. A packet never wraps around
    the ring's end. A zero word is a one-word no-op, so a writer pads the
    ring's end with zeros and starts a packet that does not fit before it at
    the ring's start. *)

val max_copy : Gpu.t -> int
(** [max_copy g] is the most bytes one linear copy of [g]'s engine moves:
    1 GiB from SDMA 4.4.2 below 5 and from 5.2, and 4 MiB otherwise. *)

val copy_linear : dst:'v -> src:'v -> bytes:'v -> 'v Packet.t
(** [copy_linear ~dst ~src ~bytes] copies [bytes] bytes from address [src]
    to address [dst] in one linear copy of the engine. The caller must keep
    [bytes] in \[[1];[max_copy g]\] for the engine's GPU [g]. *)

val poll : 'v -> Packet.comparison -> 'v -> ?mask:int -> unit -> 'v Packet.t
(** [poll addr cmp v ~mask ()] waits until the 32 bits at [addr], masked by
    [mask], compare to the low 32 bits of [v] as [cmp] says. [mask] defaults
    to all 32 bits. *)

val fence : Gpu.t -> 'v -> 'v -> 'v Packet.t
(** [fence g addr v] writes the low 32 bits of [v] to [addr] once the
    packets before it have completed. From SDMA 5 on, the write bypasses the
    caches. *)

val trap : 'v Packet.t
(** [trap] raises an interrupt. *)

val timestamp : 'v -> 'v Packet.t
(** [timestamp addr] writes the GPU's clock ({!Pm4.source}) to [addr]. *)
