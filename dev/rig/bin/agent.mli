(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** [rig agent]: a machine's half of a job, and the agent it starts.

    The half reads the job's key on its standard input, starts the agent
    ([rig agent] again, under [RIG_REMOTE_REPORT]), passes the agent's lines on
    to its standard output, and ends the agent when its input ends. The agent
    takes the machine's lock, listens and serves one job, and ends when its
    input ends. The half runs one thread; the agent adds one that watches its
    input. *)

val half : firmware:string list -> string -> 'a
(** [half ~firmware address] is the half of the agent at [address], [HOST:PORT]
    as written, which passes [firmware] on to its agent. It exits: 0 once the
    job closed, 123 otherwise. *)

val agent : firmware:string list -> string -> int -> 'a
(** [agent ~firmware host port] is the agent listening at [host] and [port]. Its
    driver-less paths read firmware images from the directories [firmware], in
    order, then from [/lib/firmware]. It exits: 0 once the job closed, 123
    otherwise. *)
