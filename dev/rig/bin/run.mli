(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** [rig run]: a job's launcher.

    It starts an agent on each other machine over ssh, runs the program with the
    agents in its environment, and, when the job fails, collects how each of its
    processes ended, names the cause, and starts the job again. Runs in one
    thread. *)

val run :
  misuse:(string -> unit) ->
  firmware:string list ->
  string list ->
  string ->
  string list ->
  'a
(** [run ~misuse ~firmware machines prog args] runs [prog] with [args] as the
    controller of a job on [machines], as written in [--on]: two or more, each a
    valid machine, named once. Each agent gets the directories [firmware], as
    [--firmware] wrote them. It exits with the status the page of [rig run]
    gives. [misuse why] ends the process when the first machine is not this one.
*)
