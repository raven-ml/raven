(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ring entries.

    A channel's ring, its GPFIFO, holds 64-bit entries, each naming a segment of
    a pushbuffer. The channel fetches the entries in order and runs each
    segment's words ({!Method}). *)

val max_words : int
(** [max_words] is the most words a segment holds, [2{^21} - 1]. *)

val entry : 'v -> offset:int -> words:int -> 'v Packet.term
(** [entry addr ~offset ~words] is the entry of the segment of [words] words at
    [addr + offset], which is 4-byte aligned and below [2{^40}]. The entry is
    [Add (Value addr, n)], where [n] depends on [offset] and [words] alone: a
    writer computes [n] once and adds each segment's address. Bit 63 of an
    entry, which would make the channel wait for its earlier work, is clear.

    Raises [Invalid_argument] if [words] is outside \[[0];{!max_words}\] or
    [offset] is outside \[[0];[2{^40}-1]\]. *)
