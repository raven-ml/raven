(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ring entries.

    A channel's ring, its GPFIFO, holds 64-bit entries, each naming a segment of
    a pushbuffer. The channel fetches the entries in order and runs each
    segment's words ({!Method}).

    Compiled code hands one of the GPU's queues work as words alone: ring
    entries, each as two 32-bit words, low first. Each entry names a segment
    of the GPU's memory that compiled code wrote and must keep until the
    work completes. The driver places the entries in the queue's ring after
    its wait for earlier work and before its signal of the work's value, so
    compiled code leaves waits and releases to the driver. *)

val max_words : int
(** [max_words] is the most words a segment holds, [2{^21} - 1]. *)

val entry : Template.operand -> offset:int -> words:int -> Template.t
(** [entry addr ~offset ~words] is the entry of the segment of [words] words at
    [addr + offset], which is 4-byte aligned and below [2{^40}]. The entry is
    [Template.u64 n addr], where [n] depends on [offset] and [words] alone: a
    writer encodes the entry once with the address [0], which gives [n], and
    adds each segment's address. Bit 63 of an entry, which would make
    the channel wait for its earlier work, is clear.

    Raises [Invalid_argument] if [words] is outside \[[0];{!max_words}\] or
    [offset] is outside \[[0];[2{^40}-1]\]. *)
