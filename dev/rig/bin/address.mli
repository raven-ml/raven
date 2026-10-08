(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Machines and addresses, written one way: a name or an address, an IPv6
    address in brackets, as URIs write it: [h100-b], [10.0.0.2], [[fd00::2]],
    [[::1]:7000]. *)

val machine : string -> (string, string) result
(** [machine s] is the host [s] names, its brackets taken off. [Error why] if
    [s] has a port or a bare address with colons, or is no machine. *)

val host_port : string -> (string * int, string) result
(** [host_port s] is the host, its brackets taken off, and the port of [s],
    written [HOST:PORT]. [Error why] names [s]. *)

val with_port : string -> int -> string
(** [with_port host port] is [HOST:PORT], [host] in brackets if it has colons.
*)
