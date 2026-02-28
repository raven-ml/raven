(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Minimal HTTP/1.1 server with WebSocket support.

    [Httpd] is a thread-per-connection HTTP/1.1 server with keep-alive and
    {{!websocket}WebSocket} support. It depends only on [unix] and
    [threads.posix].

    - {{!types}Types}
    - {{!responses}Response constructors}
    - {{!requests}Request utilities}
    - {{!websocket}WebSocket}
    - {{!server}Server} *)

(** {1:types Types} *)

type meth =
  | GET
  | HEAD
  | POST
  | PUT
  | DELETE  (** The type for HTTP request methods. *)

type request = {
  meth : meth;  (** The request method. *)
  path : string;  (** The percent-decoded request path. *)
  query : (string * string) list;  (** Percent-decoded query parameters. *)
  headers : (string * string) list;  (** Headers in original case. *)
  body : string;  (** The full request body, read before the handler runs. *)
  client_addr : Unix.sockaddr;  (** The client's socket address. *)
}
(** The type for HTTP requests. *)

type response = {
  status : int;  (** The HTTP status code. *)
  headers : (string * string) list;  (** The response headers. *)
  body : string;
      (** The response body. [Content-Length] is computed at send time. *)
}
(** The type for HTTP responses. *)

(** {1:responses Response constructors} *)

val response :
  ?status:int -> ?headers:(string * string) list -> string -> response
(** [response body] is a response with [body]. [status] defaults to [200].
    [headers] defaults to [[]]. *)

val json : ?status:int -> string -> response
(** [json body] is a response with [Content-Type: application/json]. [status]
    defaults to [200]. *)

(** {1:requests Request utilities} *)

val header : string -> request -> string option
(** [header name req] is the value of the first header named [name] in [req],
    matched case-insensitively. *)

(** {1:websocket WebSocket} *)

type ws
(** The type for WebSocket connections. Sends are thread-safe. *)

val ws_send : ws -> string -> unit
(** [ws_send ws msg] sends [msg] as a text frame. Thread-safe. *)

val ws_recv : ws -> string option
(** [ws_recv ws] blocks until a text or binary message arrives. Returns [None]
    when the peer closes the connection or an I/O error occurs. Ping frames are
    answered automatically. *)

val ws_close : ws -> unit
(** [ws_close ws] sends a close frame and shuts down the socket. Idempotent and
    thread-safe. *)

(** {1:server Server} *)

type t
(** The type for HTTP servers. *)

val create : ?addr:string -> ?port:int -> unit -> t
(** [create ()] is a server. [addr] defaults to ["127.0.0.1"], [port] to [8080].
*)

val route : t -> meth -> string -> (request -> response) -> unit
(** [route server meth path handler] registers [handler] for requests matching
    [meth] and [path] exactly. *)

val static :
  t -> prefix:string -> loader:(string -> string option) -> unit -> unit
(** [static server ~prefix ~loader ()] serves assets for [GET] requests whose
    path starts with [prefix]. The relative path (after stripping [prefix]) is
    passed to [loader]; if it returns [Some data] the data is served with an
    appropriate MIME type, otherwise a 404 is returned. *)

val websocket : t -> string -> (request -> ws -> unit) -> unit
(** [websocket server path handler] registers a WebSocket endpoint at [path].
    The handler runs on the connection thread; loop on {!ws_recv} and return
    when done. *)

val run : ?after_start:(unit -> unit) -> t -> unit
(** [run server] starts the accept loop (blocking). [after_start] defaults to
    [ignore] and is called once the socket is bound and listening. Returns when
    {!stop} is called.

    Raises [Unix.Unix_error] if binding or listening fails. *)

val stop : t -> unit
(** [stop server] requests graceful shutdown. The {!run} call returns once the
    accept loop exits. *)
