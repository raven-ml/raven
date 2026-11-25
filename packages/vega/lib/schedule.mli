(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Schedules of rates.

    A schedule maps a step counter to a rate, a step's learning rate or its
    weight decay. The counter is a scalar [int32] tensor and the result a
    scalar [float32] tensor, so a schedule is ordinary tensor arithmetic:
    applied to an optimizer state's own [step] leaf inside a compiled training
    step ([Rune.jit]), the rate is derived inside the program and tracks the
    state across calls; applied eagerly, it computes the same value. One
    family serves both worlds — there is no separate host-side variant, only
    {!eval} to read a schedule at a host counter for logging or for loops that
    keep their own count.

    A schedule's numbers, its values and its step counts, are data, not
    constants. A constructor computes each on the host in float64 and rounds
    it once into a 0-d float32 host array, which the schedule captures and
    places beside the counter when applied, so its values are those of the
    numbers written into the arithmetic as literals: the same bits, NaN for
    NaN, since a compiled step folds literals and a NaN folded there can
    take another sign than one computed at run time. Under
    [Rune.jit] the arrays are data of the program, so schedules of one
    constructor that differ only in their numbers compile to one program: a
    new run length or peak rate compiles nothing the last run did not. A
    compiled step on a device copies the numbers there, and an eager
    application places them beside the counter ({!Nx.place}). What shapes
    the schedule is the program's: the number of [boundaries] or [segments],
    and the branches its numbers select at construction ([t_mul = 1.] in
    {!cosine_decay_restarts}, an empty phase of {!one_cycle}).

    Optimizer steps take a schedule as their [~lr], and the steps that decay
    as their [~weight_decay], and apply it to their state's counter:

    {[
      let sched =
        Vega.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:1000 ()
      in
      let step (params, st) =
        ...
        Vega.adamw_step model ~lr:sched st ~params ~grads
    ]} *)

type 'd t = 'd Nx.int32_t -> 'd Nx.float32_t
(** The type for schedules.

    [s step] is the rate at the scalar counter [step], as a scalar [float32]
    tensor at [step]'s placement. Schedules are defined for [step >= 0] and
    constructors are at their initial value at [step = 0]. An optimizer step
    evaluates its schedule at the state's number of completed steps, starting
    at [0]. *)

val eval : Nx.host t -> int -> float
(** [eval s step] is the schedule's value at the host counter [step], as an
    OCaml float. Use it for logging a schedule, plotting one, or driving an
    eager loop from its own [int] counter. *)

(** {1:basic Basic}

    The constant schedule is {!Vega.rate}. *)

val linear : init_value:float -> end_value:float -> steps:int -> 'd t
(** [linear ~init_value ~end_value ~steps] interpolates linearly from
    [init_value] to [end_value] over [steps]:
    [init_value + (end_value - init_value) * min step steps / steps]. Clamps to
    [end_value] after [steps].

    Raises [Invalid_argument] if [steps] is not positive. *)

(** {1:decay Decay} *)

val cosine_decay :
  init_value:float -> decay_steps:int -> ?alpha:float -> unit -> 'd t
(** [cosine_decay ~init_value ~decay_steps ?alpha ()] is cosine decay from
    [init_value] to [alpha * init_value] over [decay_steps]:
    [init_value * ((1 - alpha) * (1 + cos (pi * min step decay_steps /
     decay_steps)) / 2 + alpha)]. It stays at [alpha * init_value] after
    [decay_steps].

    [alpha] defaults to [0.].

    Raises [Invalid_argument] if [decay_steps] is not positive. *)

val exponential_decay :
  init_value:float -> decay_rate:float -> decay_steps:int -> 'd t
(** [exponential_decay ~init_value ~decay_rate ~decay_steps] is
    [init_value * decay_rate{^ (step / decay_steps)}], the exponent a real
    quotient: the rate decays at every step, by [decay_rate] every
    [decay_steps].

    Raises [Invalid_argument] if [decay_steps] or [decay_rate] is not positive.
*)

val polynomial_decay :
  init_value:float ->
  end_value:float ->
  decay_steps:int ->
  ?power:float ->
  unit ->
  'd t
(** [polynomial_decay ~init_value ~end_value ~decay_steps ?power ()] decays from
    [init_value] to [end_value] over [decay_steps] using a polynomial schedule:
    [end_value + (init_value - end_value) * (1 - step/decay_steps)^power].

    [power] defaults to [1.0] (linear decay). Clamps to [end_value] after
    [decay_steps].

    Raises [Invalid_argument] if [decay_steps] is not positive. *)

(** {1:warmup Warmup} *)

val warmup_cosine :
  init_value:float -> peak_value:float -> warmup_steps:int -> 'd t
(** [warmup_cosine ~init_value ~peak_value ~warmup_steps] is cosine warmup from
    [init_value] to [peak_value] over [warmup_steps]:
    [init_value + (peak_value - init_value) * (1 - cos (pi * min step
     warmup_steps / warmup_steps)) / 2]. Clamps to [peak_value] after
    [warmup_steps].

    Raises [Invalid_argument] if [warmup_steps] is not positive. *)

val warmup_cosine_decay :
  init_value:float ->
  peak_value:float ->
  warmup_steps:int ->
  decay_steps:int ->
  ?end_value:float ->
  unit ->
  'd t
(** [warmup_cosine_decay ~init_value ~peak_value ~warmup_steps ~decay_steps
     ?end_value ()] is linear warmup from [init_value] to [peak_value] over
    [warmup_steps], then cosine decay to [end_value] over [decay_steps]: up to
    [warmup_steps] it is {!linear} from [init_value] to [peak_value], and at
    [s = step - warmup_steps] after it
    [end_value + (peak_value - end_value) * (1 + cos (pi * min s decay_steps /
     decay_steps)) / 2], which stays at [end_value] after [decay_steps].

    [end_value] defaults to [0.].

    Raises [Invalid_argument] if [warmup_steps] or [decay_steps] is not
    positive. *)

(** {1:restarts Warm Restarts} *)

val cosine_decay_restarts :
  init_value:float ->
  decay_steps:int ->
  ?t_mul:float ->
  ?m_mul:float ->
  ?alpha:float ->
  unit ->
  'd t
(** [cosine_decay_restarts ~init_value ~decay_steps ?t_mul ?m_mul ?alpha ()] is
    cosine decay that periodically resets to [init_value] (SGDR).

    After each restart the period is multiplied by [t_mul] and the peak
    amplitude by [m_mul]. [alpha] is the minimum fraction of the cycle's peak.
    With [n = decay_steps], cycle [k] starts at the step
    [c_k = n * (t_mul{^ k} - 1) / (t_mul - 1)] ([n * k] for [t_mul = 1.]) and
    lasts [n * t_mul{^ k}] steps; in it the schedule is
    [init_value * m_mul{^ k} * ((1 - alpha) * (1 + cos (pi * (step - c_k) / (n *
     t_mul{^ k}))) / 2 + alpha)]. The starts are computed at float32, so a
    counter within its rounding of a non-integer start may fall in either cycle.

    [t_mul] defaults to [1.0]. [m_mul] defaults to [1.0]. [alpha] defaults to
    [0.0].

    Raises [Invalid_argument] if [decay_steps] is not positive, if [t_mul] is
    not finite and at least [1.0], or if [m_mul] is not positive. *)

val one_cycle :
  max_value:float ->
  total_steps:int ->
  ?div_factor:float ->
  ?final_div_factor:float ->
  ?pct_start:float ->
  unit ->
  'd t
(** [one_cycle ~max_value ~total_steps ?div_factor ?final_div_factor ?pct_start
     ()] is the 1cycle schedule.

    Phase 1 (warmup): linear from [max_value / div_factor] to [max_value] over
    [w = truncate (pct_start * total_steps)] steps. Phase 2 (decay): cosine from
    [max_value] to [max_value / final_div_factor] over the remaining
    [total_steps - w] steps, as {!warmup_cosine_decay}. Where [w] is [0], the
    schedule is phase 2 alone over [total_steps], from [max_value] at [0]; where
    [w] is [total_steps], phase 1 alone, which then stays at [max_value].

    [div_factor] defaults to [25.0]. [final_div_factor] defaults to [10000.0].
    [pct_start] defaults to [0.3].

    Raises [Invalid_argument] if [total_steps] is not positive, if [pct_start]
    is outside \[[0];[1]\], or if [div_factor] or [final_div_factor] is not
    positive. *)

(** {1:composition Composition} *)

val piecewise_constant : boundaries:int list -> values:float list -> 'd t
(** [piecewise_constant ~boundaries ~values] is a step function. [values] has
    one more element than [boundaries]. The schedule returns [values.(i)] for
    steps in the i-th segment, which holds the counters from the boundary before
    it plus one to its own boundary, the first from [0] and the last without
    end: a boundary counter belongs to the segment it ends.

    For example,
    [piecewise_constant ~boundaries:[100; 200] ~values:[0.1; 0.01; 0.001]]
    returns [0.1] for steps 0--100, [0.01] for 101--200, and [0.001] thereafter.

    Raises [Invalid_argument] if
    [List.length values <> List.length boundaries + 1] or if [boundaries] is not
    strictly increasing. *)

val join : (int * 'd t) list -> 'd t
(** [join segments] sequences schedules end-to-end. Each [(n, s)] runs [s] for
    [n] steps, evaluated at the counter relative to the segment's start: a
    segment that starts at the counter [c0] holds the counters [c0] to
    [c0 + n - 1], where it is [s] at [0] to [n - 1]. The last segment's schedule
    is used for all steps beyond the total.

    Raises [Invalid_argument] if [segments] is empty or any [n <= 0]. *)
