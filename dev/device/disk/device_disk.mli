(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Files as io memory.

    The {e disk} is a device ({!device}) whose memory is files. A buffer of the
    disk is the bytes of one file, which {!of_file} opens and {!create_file}
    creates; its views ({!Device_core.Buffer.view}) are ranges of those bytes,
    at any byte. The disk runs no work and has no timeline: a program reaches
    its bytes through {!Device_core.Buffer.copy} and
    {!Device_core.Buffer.borrow}, in the calling domain, and they wait for
    nothing.

    {b Copies.} A copy from a disk buffer reads the file and a copy into one
    writes it, with positional reads and writes of the file, straight into or
    out of memory the host addresses and through the host's staging memory
    otherwise. A copy returns once its bytes are in the file; {!barrier} orders
    them before later changes to the file system.

    {b Borrows.} A borrow of a disk buffer by the host, or by a device that
    addresses the host's memory, is the file's pages: the disk maps the whole
    file into host memory at the first such borrow of any of its buffers, and
    keeps the mapping while the file's buffers are reachable. The system reads a
    page when the host first touches it; for a device other than the host, the
    disk asks the system to read the borrowed bytes ahead. A borrow reads only
    the pages a program touches; a copy of the same bytes reads them several
    times faster. Borrow to read part of a file where it lies, copy to read all
    of it.

    The pages of a file {!create_file} made are the file: copies see writes
    through a borrow, and a borrow sees theirs. The pages of a file {!of_file}
    opened are copy-on-write: a write through a borrow, by the host or by a
    device, changes the process's copy of the page, never the file, and copies
    do not see it.

    Nothing outside the process may change a file while its pages are borrowed:
    such a change may show through the pages of a file {!of_file} opened that
    the process has not written, and a read of a page a truncation cut off ends
    the process ([SIGBUS]).

    {b Descriptors.} A buffer names the file it opened, whatever its path names
    later. The file keeps its path and changes only through its buffers while
    the process holds them. A change by anything else shows in what later copies
    read or makes them raise [Sys_error] naming the file; a copy never reads
    another file's bytes. The disk keeps at most 64 descriptors open, more only
    while copies use them, and reopens a file by its path when it needs it
    again.

    {b Errors.} The disk is never lost ({!Device_core.Lost}). A read or write of
    a file that fails raises [Sys_error] naming the file from the copy that made
    it, and changes nothing else.

    {b Domains.} Every function may be called from any domain, at the same time
    as others. Copies of disk buffers from several domains run at once.

    {b References.}
    - {{:https://pubs.opengroup.org/onlinepubs/9799919799/functions/mmap.html}
       POSIX [mmap]}: private mappings, and [SIGBUS] past the end of a file.
    - {{:https://pubs.opengroup.org/onlinepubs/9799919799/functions/fsync.html}
       POSIX [fsync]}, and macOS's fsync(2) on [F_BARRIERFSYNC]. *)

(** {1:disk The disk} *)

val device : Device_core.t
(** [device] is this machine's disk, named ["DISK"]. Its {!Device_core.budget}
    is [max_int] and its {!Device_core.arch} is [""]; it neither computes nor
    reaches other devices' memory ({!Device_core.computes},
    {!Device_core.reaches}). {!Device_core.Buffer.create} raises
    [Invalid_argument] on it: its buffers are files. *)

(** {1:files Files} *)

val of_file : string -> (Device_core.Buffer.t, string) result
(** [of_file path] is the bytes of the regular file [path] on {!device}, for
    reading. Its length is the file's size when [of_file] opens it. A copy into
    it raises [Invalid_argument].

    Each call is a memory of its own: two opens of one file do not overlap
    ({!Device_core.Buffer.overlaps}), a write through a borrow of one is not
    seen through the other, and a copy between overlapping bytes of the two
    leaves the overlap unspecified.

    [Error why], [why] starting with [path], if [path] cannot be opened for
    reading or names no regular file, such as a directory or a FIFO, which
    [of_file] does not wait on. *)

val create_file : string -> int -> (Device_core.Buffer.t, string) result
(** [create_file path n] is a new file at [path] of [n] bytes, all zero until a
    copy writes them, as a buffer on {!device} for reading and writing. The file
    is created only where [path] names nothing: never over an existing file, and
    never at the target of a link at [path]. Its permissions are [0o666] less
    the process's umask.

    [Error why], [why] starting with [path], if [path] names something or the
    file cannot be created or sized.

    Raises [Invalid_argument] if [n < 0]. *)

val barrier : Device_core.Buffer.t -> unit
(** [barrier b] returns once the bytes written to [b]'s file before it, by
    copies or through a borrow, are ordered before every later change to the
    file system: after a crash, a change made after [barrier] returns, such as a
    rename of the file, is seen only with those bytes. It does not make them
    durable: a crash may lose both the bytes and the later change. On a file
    system that offers no barrier and no flush of the drive's cache to its
    medium, such as some network file systems on macOS, [barrier] is the
    system's [fsync] and the ordering is the file system's own. A buffer
    {!of_file} opened is never written: [barrier] returns at once.

    Raises [Invalid_argument] if [b] is not a buffer of {!device} or is dead
    ({!Device_core.Claim.consume}), and [Sys_error] naming the file if the
    system cannot order its writes or, as a copy does, if its path no longer
    names it (Descriptors). *)
