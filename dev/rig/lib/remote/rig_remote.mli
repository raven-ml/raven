(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Another machine, reached over a socket.

    A machine runs a {e server} ({!listen}) that serves its PCI functions and
    its host's memory over TCP to one {e client} at a time. A process
    {e connects} to it ({!connect}) and gets two things: the machine
    ({!machine}), whose GPUs a path that drives them over PCI opens as it opens
    this machine's, and the machine's host ({!host}), a device whose memory the
    server holds and the connection reads and writes.

    {v
       this process                               the server's process
      ─────────────────────────────────          ─────────────────────────
       machine c ── Machine.t ── Function.t       functions it took ── GPU
                         │                              │
                     Window.t ─┐                    their windows
                               ├── one stream ──>       │
       host c ──── io device ──┘                    host memory
    v}

    {1:order Order and cost}

    One connection carries every operation, in the order the process makes them.
    An operation that gives nothing back, such as a store to a window or the
    release of a function, is {e posted}: it returns once sent, and the server
    runs it before any later operation. A configuration write is not: it returns
    once it reached the function. Every other operation waits for the server's
    answer. So each domain's accesses reach the machine in the order it makes
    them, and a load that returns has seen every store sent before it. A load
    costs a round trip; a store, a send. A driver whose device is on such a
    machine declares that its submission may block.

    {1:failures Failures}

    The connection {e fails} for good when the server sends no byte for
    [timeout_ms] while the process waits for an answer, when the stream breaks,
    or when a posted operation fails on the server, which then reports why and
    ends the session. From then on:
    - the machine is failed ({!Rig_pci.Machine.failed}), with a reason that
      starts with the machine's name;
    - its accesses raise nothing: a load gives all ones and a store is dropped,
      as on a function that left the bus, so a driver over it owes the checks
      {!Rig_pci} lists;
    - its requests, such as taking a function, answer [Error], and its list of
      functions is empty;
    - the host's next read or write loses it ({!Rig.Lost}).

    A child of [fork] never uses its parent's connection, whose stream it would
    interleave with the parent's: there the connection is failed from the start,
    and the parent's goes on.

    A request the server refuses, such as taking a function another process
    holds, answers [Error] and leaves the connection usable. The server stops
    the DMA of the functions a client took and frees its memory once the
    connection is gone, however it ended.

    {1:security Security}

    A client is root on the server's machine: it programs its functions' DMA and
    reads and writes the memory it allocated there. Both ends prove that they
    hold the same key before any operation, by HMAC over BLAKE2b-256 of fresh
    random nonces of both, and the key never crosses the network. Nothing
    protects the stream after that. Listen only on a network whose every host
    may drive the machine, such as the machines' own, or on the loopback behind
    a tunnel that fails when it cannot forward, such as
    [ssh -o ExitOnForwardFailure=yes -L ...]. The client proves the key first,
    so whoever answers in the server's place learns a proof against which it can
    test guesses of the key: make the key random, with
    [head -c 32 /dev/urandom > FILE && chmod 600 FILE], and read it with
    {!read_key}.

    {1:references References}

    - H. Krawczyk, M. Bellare and R. Canetti.
      {{:https://www.rfc-editor.org/rfc/rfc2104}RFC 2104},
      {e HMAC: Keyed-Hashing for Message Authentication}: the proof, over a hash
      of 128-byte blocks.
    - M-J. Saarinen and J-P. Aumasson.
      {{:https://www.rfc-editor.org/rfc/rfc7693}RFC 7693},
      {e The BLAKE2 Cryptographic Hash and Message Authentication Code}:
      BLAKE2b, its block and digest sizes. *)

(** {1:connections Connections} *)

type t
(** The type for connections to another machine. Every function may be called
    from any domain at once; the connection runs one operation at a time. *)

val connect :
  ?timeout_ms:int -> key:string -> string -> int -> (t, string) result
(** [connect ~key host port] connects to the server listening at [host] and
    [port], proves [key] to it, checks that it proves [key] back, and opens the
    machine's host ({!host}). [timeout_ms] (defaults to [30_000]) bounds the
    connection, the handshake and, afterwards, each wait for the server's next
    byte; an operation the server runs for longer, such as a reset, fails the
    connection.

    [Error why] if the server cannot be reached, is busy with another client or
    speaks another version of the protocol, if either end does not know the key,
    or if the server describes its machine wrongly. [why] starts with
    ["HOST:PORT: "].

    Raises [Invalid_argument] if [key] has fewer than 16 bytes or if
    [timeout_ms <= 0]. *)

val machine : t -> Rig_pci.Machine.t
(** [machine c] is the machine [c] reaches. Its name is ["HOST:PORT"] as
    {!connect} was given them, followed by ["#n"] for the process's [n]th
    connection to that address from the second on: a name identifies one
    connection's machine for the life of the process, so a device opened on it
    is never another connection's. Its functions are those of the server's
    machine; its system memory, reservations and DMA mappings are made in the
    server's process. A function behind an IOMMU reaches memory at device
    addresses the server maps for it. A function has no interrupts:
    {!Rig_pci.Function.interrupt} is [false] at once. *)

val host : t -> Rig.t
(** [host c] is the host of [c]'s machine, an io device ({!Rig.Io})
    named ["CPU@NAME"], [NAME] the machine's name: {!Rig.host_of} of
    every device of the machine. Its memory is memory of the server's process,
    starting on a page, which the machine's functions may pin; it computes
    nothing, and the process reaches its bytes only by copies over [c]. A write
    returns once sent; if it fails on the server, the host's next use raises
    {!Rig.Lost}. *)

val close : t -> unit
(** [close c] ends the connection once the operations in flight in other domains
    returned, and returns once the server released what [c] held, or at once if
    [c] had failed: the machine fails with the reason
    ["NAME: the connection is closed"], [NAME] the machine's name. Closing it
    again does nothing. *)

(** {1:serving Serving} *)

type server
(** The type for servers of this machine. *)

val listen : key:string -> string -> int -> (server, string) result
(** [listen ~key host port] listens at [host] and [port] and serves this machine
    ({!Rig_pci.Machine.this}) and its host's memory, from two domains of its
    own, until {!stop}, to clients that prove [key]. Port [0] lets the system
    choose ({!port}).

    The server serves one client at a time: its functions' memory lives at
    addresses that client chooses in the server's process, where two would
    collide. A second client that proves the key is told the server is busy. A
    connection holds nothing until it proves the key: the server runs up to 64
    handshakes at once, each for at most 10 seconds, and tells the connections
    beyond those that it has too many. A client that answers no keepalive probe
    for about a minute, such as one whose machine lost power, is gone. Nothing a
    connection does stops the server from accepting the next.

    When a client leaves, for whatever reason, the server turns bus mastering
    off on every function it took, so that their DMA stops, then frees its
    memory and releases its functions. If a function's bus mastering cannot be
    turned off, the server keeps the client's memory for as long as it runs,
    refuses to map memory over it again, and says so on its standard error.

    [Error why] if [host] does not resolve or the process cannot listen there.

    Raises [Invalid_argument] if [key] has fewer than 16 bytes. *)

val port : server -> int
(** [port s] is the port [s] listens at: the one {!listen} was given, or the one
    the system chose for [0]. *)

val stop : server -> unit
(** [stop s] stops listening, ends the client's session, cleans up after it as
    {!listen} says, and returns once [s] has stopped. Stopping it again does
    nothing. *)

(** {1:keys Keys} *)

val read_key : string -> (string, string) result
(** [read_key file] is the key in [file]: its bytes, 16 to 4096 of them, read
    through one open of it. On POSIX systems [file] must be a regular file of
    this process's user that no other user may read or write, so that only this
    user knows the key.

    [Error why] naming [file] if it cannot be opened, is no regular file,
    belongs to another user, may be read or written by others, or holds too few
    or too many bytes. *)

(**/**)

val window : t -> int -> (Rig_pci.Window.t, string) result
(* [window c n] is a window through [c] on [n] new bytes of its host's memory,
   never freed: tests reach the connection's window accesses through it, since a
   function of the server's machine needs root. [Error why] if the server has no
   memory or [c] failed. Raises [Invalid_argument] if [n <= 0]. *)
