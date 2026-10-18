(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Copies: the implementation of {!Rig.Buffer.copy}, and of the blits and
    {!Rig.Buffer.fill} between a buffer and the host's values.

    A copy takes the first route that runs it in one transfer. The host
    copies between memory of this machine it addresses, and between such
    memory and an io device's, through the io library's read or write. A
    device's copy engine copies between memory it reaches, its own or a
    borrow's, and a device of another machine between its memory and this
    process's, which its driver carries. A borrow on a device that runs no
    copy is copied by its allocation's own device. A host transfer waits for
    the buffers' stamps first, and a queued one is ordered by them as every
    submission is.

    Where no route runs in one transfer, the copy goes through staging memory,
    in pieces of 32 MiB. Staging memory is slots of two halves. A copy holds a
    slot that no other copy holds until it returns, and makes a new slot when
    every slot is held, so no copy waits for one. After a give, the halves of
    every free slot that none of the last claims took, as many claims as there
    are slots, return as memory returns free halves ([Memory.return_half]).
    Copies that keep slots busy, one after another or in lockstep, so keep
    their halves, and the extra slots of a burst go once that many later
    claims passed them by. Each half is an allocation of its own, since stamps
    order the uses of a whole allocation, and is made at the first piece that
    needs it, outside every lock, so a copy of one piece makes one. A device's
    slots are part of its state ([slots] in {!Def.live}), taken and given
    under its lock. The host's slots serve every device that maps host memory,
    and a device of this machine that maps none has slots of its own
    ({!Rig.Buffer.section-copying}). A copy between two such devices holds a
    slot of each. Memory returns free slots' halves where it returns the cache
    ([Memory.release_staging]). A forked child keeps the host's slots as the
    parent left them: a slot that a thread of the parent held at the fork
    stays held, and its halves stay, for good. The legs into and out of a half
    alternate, so one half fills while the other empties. A leg on a device's
    queue runs without the host waiting for it, and the next leg on its half
    follows it by the half's stamps. A half whose stamps name unfinished work
    of a lost device is replaced, so a loss reaches no later copy. A copy
    returns once its last transfer is done.

    The host's values reach memory the host addresses by its own copy, and
    other memory through staging memory, as a copy whose one side the host
    fills or empties. A fill writes its byte into the halves for the first
    two pieces only, after which both hold it.

    A copy, blit or fill takes its read claims ({!Rig.Buffer.section-copying})
    before it checks its buffers and waits. The host moves bytes that
    no stamp orders, so a donation meanwhile would write them, or kill the
    buffer the next leg names. *)

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

val fill : Def.buffer -> char -> unit
(** [fill b c] is {!Rig.Buffer.fill}[ b c]. *)
