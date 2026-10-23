(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Compiles in processes of their own.

    A worker is a C program, carried by tolk as bytes, that loads a vendor's
    compiler library and runs one compile: a library that serialises the
    compiles of a process, or crashes on a source, then holds or ends that
    compile's process only. On Linux the program runs from a sealed memory
    file, which runs where the home directory is mounted noexec; elsewhere from
    a file in {!Helpers.cache_dir} named by the program's digest, written aside
    and renamed into place.

    A worker runs as [PROGRAM LIBRARY PARENT REQUEST REPLY]. [LIBRARY] is the
    library it loads; [PARENT] the pid of this process, which on Linux the
    worker dies with; [REQUEST] a file holding the request, whose form is the
    worker's own; [REPLY] the file it writes its reply to, once, at the end:
    [ok N\n] then [N] bytes of binary, or [error N\n] then [N] bytes of message.
    A reply is whole if and only if it is [N] bytes past its header. What the
    worker prints goes to a third file, its diagnostics. Each worker's program
    includes [compiler_worker.h], which holds the worker's side. *)

type t
(** The type for workers. *)

val v : name:string -> string -> t
(** [v ~name program] is the worker whose executable is [program], named
    [name] in its files and its errors. Nothing is written until {!run}. *)

val run : t -> string -> string -> (string, string) result
(** [run w library request] runs [w] on [library] and [request] and waits for
    it to end. [w]'s program is written out by the first run of the process.

    It is [Ok binary] or [Error message] as the worker replies, whatever its
    exit status says, which is lost when [SIGCHLD] is ignored; a worker that
    replies has its diagnostics printed on standard error. A worker that ends
    without a whole reply is [Error] saying how it ended, as in
    ["nvrtc worker was killed by SIGSEGV without a reply"], then its
    diagnostics.

    Raises [Failure] naming the worker if its program cannot be written out or
    started: that is the machine, not the request. *)
