(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Interleavings of [rig_pool.h]'s protocol, held
    (rig_pool_hooked_probe_stubs.c).

    A second copy of the pool, compiled with its hook points defined, stops the
    caller and chosen workers at named points until the others reach theirs, so
    each scenario runs one timeline, given in its stub's comment. POSIX only.

    Each scenario is [(timed_out, values)]: [timed_out] names the event a hold
    waited for in vain, [""] when the timeline held, and [values] are what the
    scenario observed, in the order its doc lists them. *)

type answer = string * int array
(** The type for a scenario's answer. *)

val cores : unit -> int
(** [cores ()] is the copy's [rig_pool_cores ()]. The scenarios need 3. *)

(** {1:entering Entering and closing} *)

val stale_add : unit -> answer
(** [stale_add ()] holds a worker's add on a closed job until the next job
    opened and closed. Its values: the next job's count at its close, whether
    closed was set then (1), whether the undo began before that job returned
    (1), whether the count was closed after, the chunks of either job not run
    once, the chunks of the next job off worker 0, whether the held worker
    entered a job. *)

val late_entry : int -> answer
(** [late_entry w] holds worker [w], which saw a job of three threads open,
    until the next job of two threads opened unpublished; [w] enters that job.
    Its values: whether [w] entered, the chunks of either job not run once, the
    chunks of the second off workers 0 and 1, those it ran on [w]. *)

(** {1:waiting Waiting and waking} *)

val caller_sleeps : held:bool -> answer
(** [caller_sleeps ~held] runs a job whose worker's chunk outlasts the caller's
    spin, so the caller sleeps until the worker leaves. With [held], the caller
    holds between setting waiting and reading the count until the worker left.
    Its values: whether the caller set waiting, with [held] whether the worker
    left while it held, the chunks not run once. *)

(** The order of worker 1's parking and a job's publication. *)
type wake_order =
  | Bit_first  (** It sets its parked bit, then the job is published. *)
  | Decision_first
      (** It decides to park, the job is published, then it sets its bit. *)
  | Asleep  (** It sleeps, then the job is published. *)

val wake : wake_order -> answer
(** [wake order] parks worker 1 against a job's publication in [order]. Its
    values: whether worker 1 ran a chunk, the chunks not run once. *)

val narrow_burst : unit -> answer
(** [narrow_burst ()] runs jobs of two threads until worker 2 parks, then a job
    of three. Its values: whether worker 2 parked, whether it ran a chunk of the
    job of three, the chunks of the narrow jobs and of that job not run once. *)

(** {1:fork Fork} *)

val fork_parked : unit -> answer
(** [fork_parked ()] forks while worker 1 holds the parking mutex; the child
    runs a job on every core. Its value: the child's exit status, the number of
    its chunks not run once. *)

val fork_running : unit -> answer
(** [fork_running ()] forks while another thread's job runs a body of 50 ms; the
    child runs a job on every core. Its values: whether fork returned after the
    body ended, the child's exit status. *)
