(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Files as an io device.

    The {e disk} is this machine's io device whose memory is files
    ({!device}). {!of_file} and {!create_file} each make one allocation, over
    the bytes of one file. Its buffers and their views ({!Rig.Buffer.view})
    are ranges of those bytes, at any offset. The disk runs no work and has
    no timeline. A program reaches a file's bytes through
    {!Rig.Buffer.copy} and {!Rig.Buffer.borrow}, which run in the calling
    domain and return when done.

    The life of a file's buffer goes like this:
    {ol
    {- {!of_file} opens a file for reading, or {!create_file} makes a new
       one for reading and writing.}
    {- Copies read and write its bytes ({!copies}), and borrows map them into
       memory that the host and devices address ({!borrows}).}
    {- {!barrier} orders the bytes written before later changes to the file
       system, such as a rename.}
    {- Once the buffers of its allocation are collected, the disk's next call
       closes the file's descriptor, and the collector unmaps its pages.}}

    For instance, this writes the buffer [b] to [path] so that, after a
    crash, [path] holds either what it held before or all of [b]'s bytes:
    {[
    let save b path =
      let tmp = path ^ ".tmp" in
      match Rig_disk.create_file tmp (Rig.Buffer.length b) with
      | Error e -> failwith e
      | Ok f ->
          Rig.Buffer.copy ~src:b ~dst:f;
          Rig_disk.barrier f;
          Sys.rename tmp path
    ]}

    The disk is never lost ({!Rig.Lost}). A copy or barrier whose access to a
    file fails raises [Sys_error] naming the file, and a borrow raises it or
    is [None] ({!borrows}). Every function may be called from any domain,
    concurrently with others, and copies of disk buffers from several domains
    run in parallel. In a forked child, inherited buffers and new ones read
    and write their files as in the parent, whatever a parent thread was
    doing at the fork.

    {b References.}
    {ul
    {- {{:https://pubs.opengroup.org/onlinepubs/9799919799/functions/mmap.html}
       POSIX [mmap]}: private mappings, and [SIGBUS] past the end of a
       file.}
    {- {{:https://pubs.opengroup.org/onlinepubs/9799919799/functions/fsync.html}
       POSIX [fsync]}, and macOS's fsync(2) on [F_BARRIERFSYNC].}} *)

(** {1:disk The disk} *)

val device : Rig.t
(** [device] is this machine's disk, named ["DISK"]. Its {!Rig.budget} is
    [max_int] and its {!Rig.arch} is [""]. It neither computes nor reaches
    other devices' memory ({!Rig.computes}, {!Rig.reaches}).
    {!Rig.Buffer.create} of one byte or more raises [Invalid_argument] on it,
    since its buffers are files. *)

(** {1:files Files}

    On POSIX systems, a file {!create_file} makes has the permissions
    [0o666] less the process's umask. On Windows it has those its directory
    gives new files. *)

val of_file : string -> (Rig.Buffer.t, string) result
(** [of_file path] is the bytes of the regular file [path] on {!device}, for
    reading only ({!Rig.Buffer.val-access}), as long as the file was when
    opened. Each call makes an allocation of its own, so two opens of one
    file do not overlap ({!Rig.Buffer.overlaps}).

    The result is [Error why] if [path] holds a NUL byte, cannot be opened
    for reading, or names no regular file, such as a directory or a FIFO.
    [why] starts with [path]. [of_file] never blocks opening a FIFO. *)

val create_file : string -> int -> (Rig.Buffer.t, string) result
(** [create_file path n] is a new file at [path] of [n] zero bytes, as a
    buffer on {!device} for reading and writing.
    [create_file] never replaces what [path] names and never follows a link
    at [path].

    The result is [Error why] if [path] holds a NUL byte or names something,
    or if the file cannot be created or sized. [why] starts with [path].

    Raises [Invalid_argument] if [n < 0]. A file the call created is removed
    if it could not be sized or the call raises. *)

(** {1:copies Copies}

    A copy from a disk buffer reads the file and a copy into one writes it,
    with positional reads and writes of the file. The bytes move straight
    into or out of memory the host addresses, and through the host's staging
    memory otherwise ({!Rig.Buffer.copy}). A copy returns once its bytes are
    in the file, where the system may still hold them in memory.

    {!barrier} orders the bytes written before it before every later change
    to the file system. After a crash, a change made after {!barrier}
    returns, such as a rename of the file, is seen only with those bytes. A
    crash may lose both, since {!barrier} does not make the bytes durable.

    Where the file system offers no barrier and no flush of the drive's cache
    to its medium, such as some network file systems on macOS, {!barrier} is
    the system's [fsync] and the ordering is the file system's own.
    {!barrier} returns immediately on a file {!of_file} opened, an empty
    file, or a buffer with no file ({!Rig.Buffer.create} of no bytes), since
    nothing writes them.

    A copy also raises [Sys_error] on a read past the end of a file that
    something outside the process cut short. The bytes a failed copy was to
    write, in the file or in memory, are then unspecified, and nothing else
    changes. *)

val barrier : Rig.Buffer.t -> unit
(** [barrier b] returns once the bytes written to [b]'s file before it, by
    copies or through a borrow, are ordered before every later change to the
    file system ({!copies}). It first waits for the device work that writes
    [b] through a borrow ({!Rig.Buffer.wait}). A barrier on a file
    {!create_file} made and nothing wrote returns without ordering the file's
    creation or size.

    The disk orders a file's bytes by syncing the file, which asks the system
    to store the bytes written to it. It syncs a file at a barrier, and before
    it closes the descriptor of a file copies wrote since its last sync
    ({!descriptors}). Once the system reports that it could not store bytes
    already written to the file, the file's next barrier raises [Sys_error],
    and so does every later one on its buffers. The system reports such a
    failure once, and the bytes it could not store may be lost.

    Raises [Invalid_argument] if [b] is dead or is neither a buffer of
    {!device} nor a borrow of one, [Sys_error] for a failure to store the
    file's bytes, as above, or if the file cannot be reopened
    ({!descriptors}), and {!Rig.Lost} if work it waits for is on a lost
    device. *)

(** {1:borrows Borrows}

    A borrow of a disk buffer on the host, or on a device that addresses the
    host's memory, is the file's pages ({!Rig.Buffer.borrow}). The disk maps
    the whole file into host memory at the first such borrow of its
    allocation, and keeps the mapping while the allocation lives. For a
    device other than the host, the disk asks the system to read the
    borrowed bytes ahead. A borrow is the faster read of a file the system
    holds in memory, since it reads only the pages a program touches.

    The pages of a file {!create_file} made are the file, so copies see
    writes through a borrow, and a borrow sees the copies' writes. A file
    {!of_file} opened admits only reads ({!Rig.Buffer.val-access}).

    Nothing outside the process may change a file while its pages are
    borrowed. A read of a page that a truncation cut off ends the process
    ([SIGBUS]). Another change may show through the pages of a file
    {!of_file} opened that the process has not written.

    A borrow raises [Sys_error] naming the file if the disk cannot map it for
    the moment, such as when the file cannot be reopened ({!descriptors}),
    and a later borrow tries again. It is [None], for that borrow and every
    later one of the same allocation, if the file system can never map the
    file. A device's driver may also refuse some files' pages
    ({!Rig_edge.Io.pages}).

    {1:descriptors Descriptors}

    A buffer names the file it opened, whatever its path names later. If
    anything but the file's buffers changes the file, or moves it from its
    path, while the process holds them, later copies read the change or raise
    [Sys_error] naming the file. A copy never reads or writes another file's
    bytes.

    The disk keeps at most 64 descriptors open, more only while copies or
    barriers use them. It reopens a file by its path when it needs it again,
    once the system confirms that the path still names it: by its file handle
    on Linux, its inode and birth time on macOS's APFS and HFS+, and its file
    ID on Windows. Before it closes the descriptor of a file copies wrote since
    its last sync, it syncs the file ({!barrier}).

    A file whose system gives no such identity keeps its descriptor until its
    buffers are collected, beyond the 64. Such are the files of a Linux file
    system without file handles, of FAT, exFAT or a network file system on
    macOS, and of any other operating system. Holding many of them can
    exhaust the process's descriptors, and then {!of_file} and {!create_file}
    return [Error] for too many open files. *)
