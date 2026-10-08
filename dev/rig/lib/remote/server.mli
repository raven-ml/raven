(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Servers of this machine ({!Rig_remote.listen}).

    An acceptor domain runs every handshake itself, over non-blocking sockets in
    one select loop, and gives the session to a client once its proof checks and
    no other client holds it. A session domain runs the client's commands one
    after another, over the functions it took, the windows they gave and the
    host memory it allocated, and checks every address it is given against them.
    Whatever ends a session, its cleanup runs. *)

type t
(** The type for servers. *)

val listen : key:string -> string -> int -> (t, string) result
(** {!Rig_remote.listen}. *)

val port : t -> int
(** {!Rig_remote.port}. *)

val stop : t -> unit
(** {!Rig_remote.stop}. *)
