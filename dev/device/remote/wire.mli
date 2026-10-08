(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The protocol both ends speak (private).

    The server speaks first. Then each end proves the key over both nonces, the
    client first:

    {v
     server: magic, version (u32), 0 and a nonce (32 bytes)
             or 1 and why: too many connections
     client: its nonce (32 bytes), its proof (32 bytes)
     server: 0, its proof (32 bytes), its page size (u32)
             or 1 and why: the client does not know the key, or another
             client holds the server
    v}

    [why] is a string: its length (u32) and its bytes. After the handshake a
    request is a header of {!request} bytes, a command (u32), four arguments
    (i64) and the length of its payload (u32), then the payload. A posted
    command has no answer; any other is answered by a header of {!answer} bytes,
    a status (u8), two results (i64) and the length of its payload (u32), then
    the payload. Every integer is little-endian. The client's C
    ([device_remote_link.c]) writes requests and reads answers to this layout.
*)

val magic : string
(** [magic] opens the server's greeting. *)

val version : int
(** [version] is the protocol's version, which both ends must speak. *)

(** {1:commands Commands} *)

type cmd =
  | Functions  (** The machine's functions. *)
  | Take  (** Take a function: its handle and addressing. *)
  | Release
  | Config_read
  | Config_write
  | Bar
  | Map  (** Map a BAR: the window's address and length. *)
  | Unmap
  | Reset
  | Alloc_dma  (** Allocate a function's memory: its address and runs. *)
  | Free_dma
  | Pin
  | Unpin
  | Reserve
  | Alloc  (** Allocate host memory: its address, 0 for none. *)
  | Free
  | Read  (** Read the machine's bytes: they are the answer's payload. *)
  | Write  (** Write the machine's bytes: they are the payload. *)
  | End
      (** End the session: answered once the server released what the client
          held. *)

val code : cmd -> int
(** [code c] is [c]'s number on the wire. *)

val cmd : int -> cmd option
(** [cmd n] is the command numbered [n], if any. *)

val posted : cmd -> bool
(** [posted c] is [true] iff [c] has no answer: its failure ends the session,
    with a fatal answer the client reads at its next command. *)

(** {1:frames Frames} *)

val request : int
(** [request] is the bytes of a request's header. *)

val answer : int
(** [answer] is the bytes of an answer's header. *)

type status =
  | Done  (** The command ran. *)
  | Refused  (** The command was refused; the session goes on. *)
  | Fatal  (** The session ended; the payload says why. *)

val status_code : status -> int
(** [status_code s] is [s]'s number on the wire. *)

val status : int -> status option
(** [status n] is the status numbered [n], if any. *)

val max_text : int
(** [max_text] bounds a message or a list of functions: 1 MiB. *)

val max_payload : int
(** [max_payload] bounds a request's payload: 1 GiB. *)

(** {1:sends Sends}

    Both ends send through these, which never raise [SIGPIPE]: a send to a peer
    that left fails with [EPIPE] and changes no signal disposition. *)

val quiet : Unix.file_descr -> unit
(** [quiet fd] keeps sends on the socket [fd] from raising [SIGPIPE] where the
    system needs a socket option for that (macOS). *)

val send : Unix.file_descr -> string -> unit
(** [send fd s] sends all of [s] on [fd], releasing the runtime meanwhile.
    Raises [Unix.Unix_error] if the stream fails. *)

val send_memory :
  Unix.file_descr ->
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int ->
  int ->
  unit
(** [send_memory fd b off n] sends the [n] bytes of [b] from [off] on [fd], as
    {!send}, without a copy. *)

(** {1:keys Keys and proofs} *)

val min_key : int
(** [min_key] is the fewest bytes of a key: 16. *)

val max_key : int
(** [max_key] is the most bytes of a key file: 4096. *)

val nonce : unit -> string
(** [nonce ()] is 32 bytes from the system's random source. *)

val client_proof : string -> server:string -> client:string -> string
(** [client_proof key ~server ~client] is the client's proof of [key] over the
    server's and the client's nonces. *)

val server_proof : string -> server:string -> client:string -> string
(** [server_proof key ~server ~client] is the server's proof, which differs from
    the client's for the same nonces. *)

val same : string -> string -> bool
(** [same a b] is [a = b], in a time independent of where they differ. *)
