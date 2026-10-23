(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Compiles in processes of their own.

    A worker is a C program, carried by tolk as bytes, that loads a vendor's
    compiler library and compiles: a library that serialises the compiles of a
    process, or crashes on a source, then holds or ends a worker's process
    only. On Linux the program runs from a sealed memory file, which runs where
    the home directory is mounted noexec; elsewhere from a file in
    {!Helpers.cache_dir} named by the program's digest, written aside and
    renamed into place.

    A request is the worker's own fields and source. A reply is [ok N\n] then
    [N] bytes of binary, or [error N\n] then [N] bytes of message, whole if
    and only if it is [N] bytes past its header. What a worker prints goes to
    its diagnostics, a file. A worker either runs one compile ({!run}) or
    serves compiles in turn ({!call}). [compiler_worker.h] holds the worker's
    side. *)

type t
(** The type for workers. *)

val v : name:string -> string -> t
(** [v ~name program] is the worker whose executable is [program], named
    [name] in its files and its errors. Nothing is written until {!run}. *)

val run : t -> string -> string -> (string, string) result
(** [run w library request] runs [w] once, as
    [PROGRAM LIBRARY PARENT REQUEST REPLY], and waits for it to end: [PARENT]
    is the pid of this process, which on Linux the worker dies with, [REQUEST]
    a file holding [request] and [REPLY] the file it writes its reply to. [w]'s
    program is written out by the first run or call of the process.

    It is [Ok binary] or [Error message] as the worker replies, whatever its
    exit status says, which is lost when [SIGCHLD] is ignored; a worker that
    replies has its diagnostics printed on standard error. A worker that ends
    without a whole reply is [Error] saying how it ended, as in
    ["nvrtc worker was killed by SIGSEGV without a reply"], then its
    diagnostics.

    Raises [Failure] naming the worker if its program cannot be written out or
    started. *)

val call : t -> string -> string -> (string, string) result
(** [call w library request] has a worker of [w] serving [library] answer
    [request]: one that is idle, or else one it starts, as [PROGRAM LIBRARY].
    A serving worker reads requests, each [N\n] then [N] bytes, on its
    standard input, replies to each on its standard output once it has read
    it whole, and ends when its input does, which this process's end closes
    when the process exits. A worker is idle between calls, from any domain;
    there is no limit to their number. A process forked from this one starts
    workers of its own: those it inherits are its parent's, and it closes its
    copies of their descriptors. On Windows a worker inherits only its own
    three handles.

    It is [Ok binary] or [Error message] as the worker replies. A worker that
    ends without a whole reply, or sends what is no reply, fails the request,
    with [Error] saying how it ended then what it printed during the request,
    and serves no other: the next call starts another. An idle worker that has
    ended meanwhile is replaced before the request is sent. A worker that
    replies has what it printed during the request printed on standard error.

    A worker that ends never raises [SIGPIPE] here: requests go to a socket
    sent to without the signal, or on Windows, which has none, to a pipe.

    Raises [Failure] naming the worker if its program cannot be written out or
    started. *)
