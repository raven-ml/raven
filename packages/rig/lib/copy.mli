(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Copies: the implementation of {!Rig.Buffer.submit_copy} and
    {!Rig.Buffer.copy}, of {!Rig.Buffer.submit_fill} and {!Rig.Buffer.fill},
    and of the blits between a buffer and the host's values.

    A copy takes the first route that runs it in one transfer. The host
    copies between memory of this machine it addresses, and between such
    memory and an io device's, through the io library's read or write, row
    by row where a box's rows are not one run. Memory an io device owns, a
    borrow of its pages included, moves only through that read or write,
    which see writes through the pages ({!Rig_edge.Io.pages}); toward a
    device's copy engine it goes through staging memory, which the host's
    side of a copy with another machine's device also takes. A device's copy
    engine copies between memory it reaches, its own or a borrow's, and a
    device of another machine between its memory and this process's, which
    its driver carries: one submission of a part per box, or the device's
    cached copy of one run ({!Submission.run_copy}). A borrow on a device
    that runs no copy is copied by its allocation's own device. A host
    transfer waits for the points the copy follows and the buffers' stamps
    first, and a queued one is ordered by them as every submission is.

    Where no route runs in one transfer, the copy goes through staging memory,
    in pieces of at most 32 MiB: whole rows of the boxes, packed one after the
    other, where a row fits, and a range of one row's bytes otherwise. The
    side away from staging keeps its geometry. Staging memory is halves, each
    an allocation of 32 MiB of its own, since stamps order the uses of a whole
    allocation, made at the first copy that needs it, outside every lock. A
    copy holds one half for one piece and two for more, which alternate, so
    one fills while the other drains, and the host waits for a half's last
    drain before it fills it again. A copy gives its halves back once the
    transfers the host makes are done, with legs through them still queued:
    a half's stamps order every later use after them.

    A half's {e cursor} is the byte from which no unfinished work uses it, and
    pieces start at multiples of 256 bytes. A copy of one piece that the host
    writes takes the free half with room after its cursor whose lap started
    last, where a half's lap starts when its cursor is set to its start, and
    writes there without waiting. A piece a device writes is never packed,
    since its write would follow every use of the half, and no piece is
    packed after it: the host's write adds no point to the stamps, so the
    next piece's drain would follow the device's write. Otherwise a copy
    takes a free half whose uses are done, its cursor at its start; for a
    piece the host writes, else the free half whose lap started first, and
    waits for its uses; else, every half being held, a new half, so no copy
    waits for a hold.

    A device's halves are part of its state ([halves] in {!Def.live}), taken
    and given under its lock. The host's halves serve every device that maps
    host memory, and a device of this machine that maps none has halves of
    its own ({!Rig.Buffer.section-copying}). A copy between two such devices
    holds halves of each. After a give, the buffers of every free half that
    none of the last takes took, as many takes as there are halves, return as
    memory returns free halves ([Memory.return_half]), so they return once the
    work that used them is done. Copies that keep halves busy so keep them,
    and the extra halves of a burst go once that many later takes passed them
    by. Memory returns free halves' buffers where it returns the cache
    ([Memory.release_staging]). A forked child keeps the host's halves as the
    parent left them: a half that a thread of the parent held at the fork
    stays held, and its buffer stays, for good. A half whose stamps name
    unfinished work of a lost device is replaced, so a loss reaches no later
    copy.

    The host's values reach memory the host addresses by its own copy, and
    other memory through staging memory, as a copy whose one side the host
    fills or empties; the blits return once their bytes are where they go. A
    fill sets memory the host addresses itself, writes memory an io device
    owns through the io library's writes from host memory that holds the
    byte, 1 MiB at a time, and sets any other memory with a fill part on the
    copy engine of the device a copy of it would run on
    ({!Submission.run_set}), staging nothing.

    A copy, blit or fill takes its read claims ({!Rig.Buffer.section-copying})
    before it checks its buffers and waits. The host moves bytes that no stamp
    orders, so a donation meanwhile would write them, or kill the buffer the
    next leg names. A queued copy or fill returns a point of {!Dev.host}
    ({!Submission.mark}) where the host made its last transfer. *)

val row : ?src_offset:int -> ?dst_offset:int -> int -> Def.box
(** [row ~src_offset ~dst_offset n] is {!Rig.Buffer.row}[ ~src_offset
    ~dst_offset n]. *)

val submit_copy :
  ?waits:int array -> src:Def.buffer -> dst:Def.buffer -> Def.box array -> int
(** [submit_copy ~waits ~src ~dst boxes] is {!Rig.Buffer.submit_copy}[ ~waits
    ~src ~dst boxes]. *)

val copy : src:Def.buffer -> dst:Def.buffer -> unit
(** [copy ~src ~dst] is {!Rig.Buffer.copy}[ ~src ~dst]. *)

val queued : Def.device -> src:Def.buffer -> dst:Def.buffer -> unit
(** [queued d ~src ~dst] copies [src] into [dst] as one submission on [d]'s
    copy engine, and waits for it. [src] and [dst] must be buffers
    on [d], or, for a device of another machine, one may be this process's
    memory ({!Submission.run_copy}). It raises as {!Rig.submit} and {!Dev.wait}
    do, and raises [Invalid_argument] if a buffer's memory is held
    exclusive. *)

val blit_from_string : string -> int -> Def.buffer -> int -> int -> unit
(** [blit_from_string s i b j n] is {!Rig.Buffer.blit_from_string}[ s i b j
    n]. *)

val blit_to_bytes : Def.buffer -> int -> bytes -> int -> int -> unit
(** [blit_to_bytes b i s j n] is {!Rig.Buffer.blit_to_bytes}[ b i s j n]. *)

val submit_fill : ?waits:int array -> Def.buffer -> char -> int
(** [submit_fill ~waits b c] is {!Rig.Buffer.submit_fill}[ ~waits b c]. *)

val fill : Def.buffer -> char -> unit
(** [fill b c] is {!Rig.Buffer.fill}[ b c]. *)
