(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Compiles in processes of their own.

    A worker is a C program, carried by tolk as bytes, that loads a vendor's
    compiler library and compiles: a library that serialises the compiles of a
    process, or crashes on a source, then holds or ends a worker's process
    only. On Linux the program runs from a sealed memory file, which runs where
    the home directory is mounted noexec; elsewhere from a file named by the
    program's digest, written aside and renamed into place, in
    {!Helpers.cache_dir}, or in [Filename.get_temp_dir_name] if the cache
    cannot be written, other users can write it, or the system refuses to run a
    program from it. Such a file runs only if it is a regular file reached
    through no link, this user's, that no other user can write, without macOS's
    quarantine, holding the program's bytes; it is checked at each start and
    written again otherwise. Where another user's file holds the name, as in a
    shared temporary directory, the program runs from the file written aside,
    removed when this process exits.

    A request is the worker's own fields and source. A reply is [ok N\n] then
    [N] bytes of binary, or [error N\n] then [N] bytes of message, whole if
    and only if it is [N] bytes past its header. What a worker prints goes to
    its diagnostics, a file. A worker serves compiles in turn ({!call}).
    [compiler_worker.h] holds the worker's side. *)

type t
(** The type for workers. *)

val v : name:string -> string -> t
(** [v ~name program] is the worker whose executable is [program], named
    [name] in its files and its errors. Nothing is written until {!call}. *)

val call : t -> string -> string -> (string, string) result
(** [call w library request] has a worker of [w] serving [library] answer
    [request]: one that is idle, or else one it starts, as
    [PROGRAM LIBRARY TEMP PARENT], [PARENT] this process's pid. A serving
    worker reads requests, each [N\n] then [N] bytes, on its standard input,
    replies to each on its standard output once it has read it whole, and
    ends when its input does or when this process ends, killed or not: a
    worker in a compile that does not return ends then too, within a
    second. A process
    forked from this one starts workers of its own: those it inherits are its
    parent's, and it closes its copies of their descriptors when it first
    calls with [w].

    [TEMP] is a directory of the worker's own in
    [Filename.get_temp_dir_name], made its temporary directory. The worker
    removes it, with all in it, when it ends as above, but for what a compile
    creates while the directory is being removed, when this process ends
    during that compile; this process removes it when it finds the worker
    ended, so that what a crashed compile leaves goes with it. The directory
    of a worker killed while idle stays until a call of this process for its
    library finds it ended, or for good if this process exits first.

    A worker is idle between calls, from any domain; there is no limit to
    their number. On Windows a worker inherits only its own three handles.

    It is [Ok binary] or [Error message] as the worker replies. A worker that
    ends without a whole reply, or sends what is no reply, fails the request,
    with [Error] saying how it ended then what it printed during the request,
    and serves no other: the next call starts another. An idle worker that has
    ended meanwhile is replaced before the request is sent. A worker that
    replies has what it printed during the request printed on standard error.

    A worker that ends never raises [SIGPIPE] here: requests go to a socket
    sent to without the signal, or on Windows, which has none, to a pipe.

    Raises [Failure] naming the worker if its program can be written to
    neither directory, or cannot be started; the failure names each
    directory's. *)

(**/**)

val before_program_open : (string -> unit) ref
(** [!before_program_open path] runs as the check of a program file at [path]
    opens it, before each start of any worker, so that a test can change the
    file there. It does nothing unless a test sets it. *)
