(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A server of this machine to another's process.

    The server serves this machine's PCI functions, memory and host programs to
    one {!Remote} client at a time: the client drives the functions it takes as
    if they were its own, and its system memory lives at the addresses it
    chooses in this server's address space, so two clients would collide. A
    second client is told the server is busy.

    {b Security.} A client is root here: it programs the functions' DMA, reads
    and writes the memory it allocated, and runs code. A client must prove that
    it holds the server's key before any command, and the server proves it back.
    Nothing protects the stream after that: listen only on a network whose every
    host may drive this machine, such as the machines' own fabric, or on the
    loopback behind a tunnel. The server checks every address it is given
    against what the connection allocated or mapped, so that a client's mistake
    is an error, not a write into the server.

    {b Cleanup.} When the client disconnects, for whatever reason, the server
    turns bus mastering off on every function it took, so that their DMA stops
    first, then frees the client's memory, drops its programs and releases its
    functions. *)

type programs = {
  load : binary:string -> name:string -> int;
      (** [load ~binary ~name] loads a host program, and is a number that names
          it. *)
  call : int -> Mmio.t array -> int array -> unit;
      (** [call p buffers values] calls [p] on [buffers], memory of this
          process, and [values]. *)
  unload : int -> unit;  (** [unload p] drops [p]. *)
}
(** The type for the host programs a server loads and calls. *)

type t
(** The type for servers. *)

val listen : key:string -> ?programs:programs -> Unix.sockaddr -> t
(** [listen ~key addr] listens at [addr] and serves the clients that prove
    [key], one at a time, in another domain, until {!stop}. Without [programs]
    it loads none.

    Raises [Invalid_argument] if [key] is shorter than 16 bytes, and
    [Unix.Unix_error] if it cannot listen at [addr]. *)

val address : t -> Unix.sockaddr
(** [address s] is the address [s] listens at, with the port the system chose if
    [listen] was given port [0]. *)

val stop : t -> unit
(** [stop s] stops listening, disconnects the client, cleans up after it, and
    returns once [s] has stopped. *)

val wait : t -> unit
(** [wait s] returns once [s] has stopped. *)
