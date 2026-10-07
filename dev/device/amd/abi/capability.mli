(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from an AMD device.

    Code compiled for an AMD device encodes its work with this library's
    encoders for the device's GPU, and gives its kernels scratch memory
    ({!Scratch}). The device's driver gives what this needs as a {!t} when it
    opens the device, under {!key}. Times the GPU writes, such as a
    {!Pm4.copy_data} of its clock, count the GPU's clock; the driver converts
    them to nanoseconds of the host clock when it reads them.

    {b Fills.} Work for one of the device's queues can be a C function, a
    {e fill}: [int fill(void *queue, void *arg, uint64_t v)]. [queue] is the
    queue's writer, valid only during the call, and [v] the value the work
    completes on the device's timeline. The fill places its packets with
    {!field-place} and takes memory for its kernels' arguments with
    {!field-segment}, and nothing else: waiting for earlier work and signalling
    [v] are the driver's. It stops at the first call that fails and returns the
    failure that call returned, or returns [0] once every call succeeded.

    A fill whose work declares [r] ring units places at most [r] words
    ({!Packet.size}), and one whose work declares [s] segment bytes takes at
    most [s] bytes: a call beyond them returns a failure. *)

(** The type for the packets a device's compute queue reads. *)
type compute =
  | Pm4
      (** PM4 packets ({!Pm4}), which one die runs. Each dispatch names its
          scratch buffer ({!Pm4.dispatch}). *)
  | Aql of { scratch : address:int -> int -> unit }
      (** AQL packets ({!Aql}), which every die runs. The queue hands every
          kernel it dispatches one scratch buffer: [scratch ~address n] makes it
          the {!Scratch.size}[ g n] bytes at [address], for kernels of up to [n]
          bytes per lane, [g] the device's {!field-gpu} ({!Scratch.descriptor}).
          Work placed before the call may run with it, so [n] is at least the
          bytes per lane of the buffer it replaces, and the caller keeps that
          buffer until the work placed before completes.

          Raises [Invalid_argument] if [n] is less than the replaced buffer's
          bytes per lane or [address] is not a multiple of 256. Any domain may
          call it. *)

type t = {
  gpu : Gpu.t;  (** The device's GPU, as the encoders take it. *)
  compute : compute;  (** What its compute queue reads. *)
  place : nativeint;
      (** [place] is the address of
          [int place(void *queue, const uint32_t *words, size_t n)], which a
          fill calls to place the [n] words at [words] on its queue: whole
          packets of the queue's kind. The writer applies the ring's rules: a
          PM4 ring wraps, an SDMA or AQL [words] goes whole before the ring's
          end, and an AQL packet's first word is stored last ({!Aql}). [n] is a
          multiple of 16 on an AQL queue.

          It returns [0], or a failure that the fill returns as its own: a
          failure if the words would pass the ring units its work declared. Only
          a fill, during its call, may call it. *)
  segment : nativeint;
      (** [segment] is the address of
          [int segment(void *queue, size_t n, void **host, uint64_t *address)],
          which a fill calls to take [n] bytes of the device's memory for its
          kernels' arguments, such as a dispatch packet a kernel reads
          ({!Aql.dispatch}). It stores at [host] where the host writes them and
          at [address] where the GPU reads them, a multiple of 64. They stay
          valid until [v] is reached.

          It returns [0], or a failure that the fill returns as its own: a
          failure if the bytes would pass the segment bytes its work declared.
          Only a fill, during its call, may call it. *)
}
(** The type for what compiled code needs from an AMD device. *)

val key : t Type.Id.t
(** [key] is the key an AMD device's {!t} is found under. *)
