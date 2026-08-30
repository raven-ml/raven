(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Learning-rate schedules.

    A schedule maps a step counter to a learning rate. The counter is a scalar
    [int32] tensor and the result a scalar [float32] tensor, so a schedule is
    ordinary tensor arithmetic: applied to an optimizer state's own [step] leaf
    inside a compiled training step ({!Rune.val-jit}), the rate is derived
    inside the program and tracks the state across calls; applied eagerly, it
    computes the same value. One family serves both worlds — there is no
    separate host-side variant, only {!eval} to read a schedule at a host
    counter for logging or for loops that keep their own count.

    Structural steps take the schedule's result as their [~lr]:

    {[
      let sched =
        Vega.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:1000 ()
      in
      let step (params, st) =
        ...
        Vega.adamw_step model ~lr:(sched st.step) st ~params ~grads
    ]}

    This is the single schedule vocabulary for both of Vega's tiers: the
    per-tensor transforms ([Vega.scale_by_schedule],
    [Vega.scale_by_learning_rate], [Vega.add_decayed_weights]) evaluate their
    schedule at the chain's own update count. *)

type t = Nx.int32_t -> Nx.float32_t
(** The type for learning-rate schedules.

    [s step] is the learning rate at the scalar counter [step], as a scalar
    [float32] tensor. Schedules are defined for [step >= 0] and constructors
    are at their initial value at [step = 0]. Per-tensor chains evaluate their
    schedules at the 1-based update count (the first update evaluates at [1]);
    structural loops evaluate at the state's number of completed steps,
    starting at [0]. *)

val eval : t -> int -> float
(** [eval s step] is the schedule's value at the host counter [step], as a
    host float. Use it for logging a schedule, plotting one, or driving an
    eager loop from its own [int] counter. *)

(** {1:basic Basic} *)

val constant : float -> t
(** [constant lr] is the schedule that always returns [lr]. *)

val linear : init_value:float -> end_value:float -> steps:int -> t
(** [linear ~init_value ~end_value ~steps] interpolates linearly from
    [init_value] to [end_value] over [steps]. Clamps to [end_value] after
    [steps].

    Raises [Invalid_argument] if [steps] is not positive. *)

(** {1:decay Decay} *)

val cosine_decay :
  init_value:float -> decay_steps:int -> ?alpha:float -> unit -> t
(** [cosine_decay ~init_value ~decay_steps ?alpha ()] is cosine decay from
    [init_value] to [alpha * init_value] over [decay_steps].

    [alpha] defaults to [0.].

    Raises [Invalid_argument] if [decay_steps] is not positive. *)

val exponential_decay :
  init_value:float -> decay_rate:float -> decay_steps:int -> t
(** [exponential_decay ~init_value ~decay_rate ~decay_steps] is
    [init_value * decay_rate{^ (step / decay_steps)}].

    Raises [Invalid_argument] if [decay_steps] or [decay_rate] is not
    positive. *)

val polynomial_decay :
  init_value:float ->
  end_value:float ->
  decay_steps:int ->
  ?power:float ->
  unit ->
  t
(** [polynomial_decay ~init_value ~end_value ~decay_steps ?power ()] decays from
    [init_value] to [end_value] over [decay_steps] using a polynomial schedule:
    [end_value + (init_value - end_value) * (1 - step/decay_steps)^power].

    [power] defaults to [1.0] (linear decay). Clamps to [end_value] after
    [decay_steps].

    Raises [Invalid_argument] if [decay_steps] is not positive. *)

(** {1:warmup Warmup} *)

val warmup_cosine :
  init_value:float -> peak_value:float -> warmup_steps:int -> t
(** [warmup_cosine ~init_value ~peak_value ~warmup_steps] is cosine warmup from
    [init_value] to [peak_value] over [warmup_steps]. Clamps to [peak_value]
    after [warmup_steps].

    Raises [Invalid_argument] if [warmup_steps] is not positive. *)

val warmup_cosine_decay :
  init_value:float ->
  peak_value:float ->
  warmup_steps:int ->
  decay_steps:int ->
  ?end_value:float ->
  unit ->
  t
(** [warmup_cosine_decay ~init_value ~peak_value ~warmup_steps ~decay_steps
     ?end_value ()] is linear warmup from [init_value] to [peak_value] over
    [warmup_steps], then cosine decay to [end_value] over [decay_steps].

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
  t
(** [cosine_decay_restarts ~init_value ~decay_steps ?t_mul ?m_mul ?alpha ()] is
    cosine decay that periodically resets to [init_value] (SGDR).

    After each restart the period is multiplied by [t_mul] and the peak
    amplitude by [m_mul]. [alpha] is the minimum fraction of [init_value].

    [t_mul] defaults to [1.0]. [m_mul] defaults to [1.0]. [alpha] defaults to
    [0.0].

    Raises [Invalid_argument] if [decay_steps] is not positive, [t_mul < 1.0],
    or [m_mul] is not positive. *)

val one_cycle :
  max_value:float ->
  total_steps:int ->
  ?div_factor:float ->
  ?final_div_factor:float ->
  ?pct_start:float ->
  unit ->
  t
(** [one_cycle ~max_value ~total_steps ?div_factor ?final_div_factor ?pct_start
     ()] is the 1cycle schedule.

    Phase 1 (warmup): linear from [max_value / div_factor] to [max_value] over
    [pct_start * total_steps] steps. Phase 2 (decay): cosine from [max_value] to
    [max_value / final_div_factor] over the remaining steps.

    [div_factor] defaults to [25.0]. [final_div_factor] defaults to [10000.0].
    [pct_start] defaults to [0.3].

    Raises [Invalid_argument] if [total_steps] is not positive. *)

(** {1:composition Composition} *)

val piecewise_constant : boundaries:int list -> values:float list -> t
(** [piecewise_constant ~boundaries ~values] is a step function. [values] has
    one more element than [boundaries]. The schedule returns [values.(i)] for
    steps in the i-th segment.

    For example,
    [piecewise_constant ~boundaries:[100; 200] ~values:[0.1; 0.01; 0.001]]
    returns [0.1] for steps 1--100, [0.01] for 101--200, and [0.001] thereafter.

    Raises [Invalid_argument] if
    [List.length values <> List.length boundaries + 1] or if [boundaries] is not
    strictly increasing. *)

val join : (int * t) list -> t
(** [join segments] sequences schedules end-to-end. Each [(n, s)] runs [s] for
    [n] steps, evaluated at the counter relative to the segment's start. The
    last segment's schedule is used for all steps beyond the total.

    Raises [Invalid_argument] if [segments] is empty or any [n <= 0]. *)
