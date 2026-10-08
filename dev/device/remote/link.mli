(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A client's end of a connection, in C (private).

   A link owns the socket after the handshake. Every operation of the client
   goes through it: requests from OCaml and, as the machine's transport, window
   accesses from a driver's C, which hold no OCaml runtime. A lock makes each
   request and its answer one exchange on the stream. A link that failed stays
   failed: its requests answer [Error], its reads fail, its writes are dropped,
   and its socket is closed.

   A link is never freed: windows hold its transport's address, which no
   collection can follow. A closed link keeps its name and reason, about 300
   bytes. Every function may be called from any domain; each releases the
   runtime while it waits for the stream. *)

type t
(** The type for links. *)

val make : Unix.file_descr -> name:string -> timeout_ms:int -> t
(** [make fd ~name ~timeout_ms] is the link over the connected socket [fd],
    which it takes, to the machine named [name]. A wait for the server's next
    byte past [timeout_ms] fails it. *)

val name : t -> string
(** [name l] is the machine's name, ["HOST:PORT"]. *)

val failed : t -> string option
(** [failed l] is why [l] failed, starting with its name, if it did. *)

val fail : t -> string -> unit
(** [fail l why] fails [l] with ["NAME: why"], unless it failed already, and
    closes its socket once the exchange in flight returned. *)

val request :
  t ->
  Wire.cmd ->
  int ->
  int ->
  int ->
  int ->
  string ->
  (int * int * string, string) result
(** [request l c a0 a1 a2 a3 payload] sends the command [c] that is not posted
    and is its answer: its two results and its payload. [Error why] if the
    server refused it, [l] keeping on, or if [l] failed, before or meanwhile;
    [why] starts with [l]'s name. A payload longer than {!Wire.max_text}
    announced by an answer fails [l]. *)

val post : t -> Wire.cmd -> int -> int -> int -> int -> string -> unit
(** [post l c a0 a1 a2 a3 payload] sends the posted command [c]. It is dropped
    if [l] failed. *)

val read : t -> int -> dst:int -> int -> bool
(** [read l a ~dst n] copies the machine's [n] bytes at [a] to this process's
    memory at [dst], in requests of at most {!Wire.max_payload} bytes, and is
    [false] if [l] failed before or meanwhile, the bytes at [dst] then all ones.
*)

val write : t -> int -> src:int -> int -> bool
(** [write l a ~src n] posts a copy of this process's [n] bytes at [src] to the
    machine's at [a], in requests of at most {!Wire.max_payload} bytes, and is
    [false] if [l] had failed. [src] may be reused once it returns. *)

val transport : t -> Device_pci.Window.transport
(** [transport l] is [l] as the machine's transport, whose reads and writes are
    {!read} and {!write} and whose [failed] is {!failed}. *)

val close : t -> unit
(** [close l] sends {!Wire.End} and waits for its answer, unless [l] failed,
    then is [fail l "the connection is closed"]. *)
