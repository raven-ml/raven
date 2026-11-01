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

(** {1:windows Sub-windows}

    A sub-window copy, [COPY_LINEAR_SUB_WIND], moves rows of [width]
    elements from one side to the other in one packet, each side's rows
    [pitch] elements apart. An element is 1, 2, 4, 8 or 16 bytes. Each side
    is an address that is a multiple of 4 and an offset of whole elements
    from it, less than 4 bytes, and each pitch is a multiple of 4 bytes.
    A side's offset plus the width is at most its pitch: ROCr refuses a
    rect past it.

    The packet is 13 words: the header; the source's address, low word
    first, its offset, its pitch, and its layers' pitch, 0 for one layer;
    the destination's five; the width and the rows; and the layers less
    one, 0. The width, the rows and the pitches are written less one, as
    AMD's ROCr writes them ([BuildCopyRectCommand] in amd_blit_sdma.cpp). *)

type window = {
  header : int;
      (** The header with an element of one byte: the copy's operation and
          the sub-window's sub-operation. *)
  element : int;
      (** The lowest bit, in the header, of the base-2 logarithm of the
          element's bytes. *)
  src_x : int;  (** The lowest bit of the source's offset in its word. *)
  src_pitch : int;  (** The lowest bit of the source's pitch in its word. *)
  dst_x : int;  (** The lowest bit of the destination's offset. *)
  dst_pitch : int;  (** The lowest bit of the destination's pitch. *)
  width : int;  (** The lowest bit of the width in its word. *)
  rows : int;  (** The lowest bit of the rows in the width's word. *)
  most_width : int;  (** The most elements of a row. *)
  most_rows : int;  (** The most rows of one packet. *)
  most_pitch : int;  (** The most elements of a pitch. *)
}
(** The type for the layout of a GPU's sub-window copy. *)

val window : Gpu.t -> window
(** [window g] is the layout of [g]'s sub-window copy. Before GC 12 it
    takes rows of up to 2{^14} elements, up to 2{^14} rows a packet and
    pitches of up to 2{^19} elements (ROCr's [SDMA_PKT_COPY_LINEAR_RECT],
    which Linux's SDMA 4 to 6 headers lay out alike); from GC 12, 2{^16},
    2{^16} and 2{^16}, its pitches in other bits
    ([SDMA_PKT_COPY_LINEAR_RECT_GFX12]). *)

val max_fill : Gpu.t -> int
(** [max_fill g] is the most bytes one {!const_fill} of [g]'s engine sets,
    {!max_copy}[ g]. *)

val const_fill : dst:'v -> word:'v -> bytes:'v -> 'v Packet.t
(** [const_fill ~dst ~word ~bytes] sets the [bytes] bytes from [dst] to the
    32-bit [word] repeated, reading no memory. The caller must keep [dst]
    and [bytes] multiples of 4, and [bytes] in \[[4];[max_fill g]\] for the
    engine's GPU [g]. *)

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
