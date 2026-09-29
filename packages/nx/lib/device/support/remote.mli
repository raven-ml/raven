(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Another machine, reached through its server.

    A machine runs a server ({!Remote_server}, the [nx-remote] command) that
    serves its PCI functions, its memory and its host programs to one client. A
    connection is that client: it takes functions and reads and writes their
    configuration and BARs ({!Pci.take} builds on it), allocates the machine's
    memory, and loads and calls programs there. Addresses are the machine's.

    {b Ordering.} One connection carries every command in order. Writes, copies
    and calls are {e posted}: they return once sent, and the server runs them
    before any later command. Other commands wait for the answer.

    {b Failures.} A command the server refuses raises [Failure] with its reason
    and leaves the connection usable. The connection fails for good when the
    server does not answer within its timeout, when the stream breaks, or when a
    posted command fails, which the server reports at the next command and then
    closes: every later command raises that first error at once. The server then
    releases everything the connection held. Every error message starts with the
    machine's {!name}.

    {b Security.} Both ends prove they hold the same key; nothing else is
    protected. The server gives its client the machine, and the stream after the
    handshake is neither encrypted nor authenticated: use it on the machines'
    own network, or through a tunnel that fails when it cannot forward, such as
    [ssh -o ExitOnForwardFailure=yes -L ...]: whoever answers in the server's
    place learns a proof of the key it can test guesses against. Make the key
    random, such as [head -c 32 /dev/urandom > FILE && chmod 600 FILE], and read
    it with {!read_key}. *)

type t
(** The type for connections. Every function may be called from any domain;
    commands run one at a time. *)

val connect : ?timeout_ms:int -> key:string -> string -> int -> t
(** [connect ~key host port] connects to the server at [host] and [port] and
    proves [key] to it, which must prove it back. [timeout_ms] (defaults to
    [30_000]) bounds the connection, the handshake and each later answer. The
    process ignores [SIGPIPE] from then on, so that writing to a connection the
    server closed raises instead of killing it.

    Raises [Invalid_argument] if [key] is shorter than 16 bytes, and [Failure]
    if the server cannot be reached, is busy with another client, speaks another
    version of the protocol, or either end does not know the key. *)

val read_key : string -> string
(** [read_key file] is the key in [file]: its bytes, 16 to 4096 of them, read
    from one open of it. On POSIX systems [file] must be a regular file of this
    process's user that no other user may read or write, so that only this user
    knows the key.

    Raises [Failure] naming [file] if it cannot be opened, is no regular file,
    belongs to another user, may be read or written by others, or holds too few
    or too many bytes. *)

val name : t -> string
(** [name r] is ["HOST:PORT"], as {!connect} was given them. *)

val arch : t -> string
(** [arch r] is the instruction set of the machine's processor, such as
    ["arm64"] or ["x86_64"]. *)

val page : t -> int
(** [page r] is the machine's page size in bytes. *)

val failed : t -> string option
(** [failed r] is the error that failed [r], if it has failed. *)

val close : t -> unit
(** [close r] closes the connection, after which the server releases what [r]
    held. Later commands raise [Failure]. *)

val ping : t -> unit
(** [ping r] returns once the server has run every command sent before. *)

(** {1:pci PCI functions} *)

val scan :
  t -> vendor:int -> ?class_:int -> (int * int list) list -> string list
(** [scan r ~vendor ?class_ ids] is {!Pci.scan} on the machine. *)

val take : t -> lock:string -> string -> int
(** [take r ~lock bus] is {!Pci.take} on the machine: the number that names the
    function taken, until {!release}. *)

val release : t -> int -> unit
(** [release r f] turns [f]'s bus mastering off, then is {!Pci.release} of [f].
    If its DMA cannot be stopped, [f] stays taken. *)

val read_config : t -> int -> int -> int -> int
(** [read_config r f off n] is {!Pci.read_config} of [f]. *)

val write_config : t -> int -> int -> int -> int -> unit
(** [write_config r f off n v] is {!Pci.write_config} of [f]. *)

val bar : t -> int -> int -> int * int
(** [bar r f i] is {!Pci.bar} of [f]. *)

val map_bar : t -> int -> int -> Mmio.t
(** [map_bar r f i] is the whole of [f]'s BAR [i], which the server maps once: a
    range of the machine's addresses, accessed through [r], a word at a time
    where aligned. *)

val resize_bar : t -> int -> int -> unit
(** [resize_bar r f i] is {!Pci.resize_bar} of [f]. *)

val reset : t -> int -> unit
(** [reset r f] is {!Pci.reset} of [f]. *)

(** {1:memory Memory} *)

val reserve : t -> base:int -> int -> unit
(** [reserve r ~base n] is {!Sysmem.reserve} on the machine, until the
    connection ends. *)

val alloc_sysmem : t -> ?contiguous:bool -> ?va:int -> int -> Mmio.t * int list
(** [alloc_sysmem r ?contiguous ?va n] is {!Sysmem.alloc} on the machine,
    accessed through [r]. [va] must lie in a range {!reserve} reserved on [r],
    outside the memory [r] holds there. *)

val free_sysmem : t -> Mmio.t -> unit
(** [free_sysmem r m] is {!Sysmem.free} on the machine. *)

val alloc : t -> int -> nativeint option
(** [alloc r n] is the address of [n > 0] new bytes of the machine's memory,
    starting on a page, not locked, or [None] if the machine has none. *)

val free : t -> nativeint -> unit
(** [free r a] frees the memory {!alloc} returned at [a]. *)

val pin : t -> nativeint -> int -> int list
(** [pin r a n] is {!Sysmem.pin} on the machine, of memory {!alloc} returned. *)

val unpin : t -> nativeint -> int -> unit
(** [unpin r a n] is {!Sysmem.unpin} on the machine. *)

val read : t -> src:nativeint -> dst:nativeint -> int -> unit
(** [read r ~src ~dst n] copies the machine's [n] bytes at [src], of memory the
    connection allocated or mapped, into the process's memory at [dst]. *)

val write : t -> dst:nativeint -> src:nativeint -> int -> unit
(** [write r ~dst ~src n] posts a copy of the process's [n] bytes at [src] to
    the machine's memory at [dst], which the connection allocated or mapped.
    [src] may be reused once it returns. *)

val copy : t -> dst:nativeint -> src:nativeint -> int -> unit
(** [copy r ~dst ~src n] posts a copy of [n] bytes within the machine's memory,
    which the connection allocated. *)

val access : t -> Mmio.access
(** [access r] reads and writes the machine's memory that the connection
    allocated or mapped. *)

(** {1:programs Programs} *)

val load : t -> binary:string -> name:string -> int
(** [load r ~binary ~name] loads the function [name] of the host program
    [binary] on the machine ([Nx_device.Program.load] there), and is the number
    that names it until {!unload}. *)

val unload : t -> int -> unit
(** [unload r p] drops the program [p]. *)

val call : t -> int -> (nativeint * int) array -> int array -> unit
(** [call r p buffers values] posts a call of the program [p] with the machine's
    memory [buffers], each an address and a size, and [values]
    ([Nx_device.Program.call]'s ABI). *)
