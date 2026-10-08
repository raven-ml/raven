(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** [rig agent]: a machine's half of a job, and the agent it starts.

    The half reads the job's key on its standard input, starts the agent
    ([rig agent] again, under [RIG_REMOTE_REPORT]), passes the agent's lines on
    to its standard output, and ends the agent when its input ends. The agent
    takes the machine's lock, listens and serves one job. Each runs in one
    thread of its own. *)

val half : string -> 'a
(** [half address] is the half of the agent at [address], [HOST:PORT] as
    written. It exits: 0 once the job closed, 123 otherwise. *)

val agent : string -> int -> 'a
(** [agent host port] is the agent listening at [host] and [port]. It exits: 0
    once the job closed, 123 otherwise. *)
