(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Learning-rate schedules. *)

type t = int -> float
(** The type for learning-rate schedules.

    [s step] is the learning rate for 1-based [step]. *)

(** {1:basic Basic} *)

val constant : float -> t
(** [constant lr] is the schedule that always returns [lr]. *)

val linear : init_value:float -> end_value:float -> steps:int -> t
(** [linear ~init_value ~end_value ~steps] interpolates linearly from
    [init_value] to [end_value] over [steps]. Clamps to [end_value] after
    [steps]. *)

(** {1:decay Decay} *)

val cosine_decay :
  init_value:float -> decay_steps:int -> ?alpha:float -> unit -> t
(** [cosine_decay ~init_value ~decay_steps ?alpha ()] is cosine decay from
    [init_value] to [alpha * init_value] over [decay_steps].

    [alpha] defaults to [0.]. *)

val exponential_decay :
  init_value:float -> decay_rate:float -> decay_steps:int -> t
(** [exponential_decay ~init_value ~decay_rate ~decay_steps] is
    [init_value * decay_rate{^ (step / decay_steps)}]. *)

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
    [decay_steps]. *)

(** {1:warmup Warmup} *)

val warmup_cosine :
  init_value:float -> peak_value:float -> warmup_steps:int -> t
(** [warmup_cosine ~init_value ~peak_value ~warmup_steps] is cosine warmup from
    [init_value] to [peak_value] over [warmup_steps]. Clamps to [peak_value]
    after [warmup_steps]. *)

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

    [end_value] defaults to [0.]. *)

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
    [0.0]. *)

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
    [pct_start] defaults to [0.3]. *)

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
    [n] steps. Step numbers are restarted from 1 within each segment. The last
    segment's schedule is used for all steps beyond the total.

    Raises [Invalid_argument] if [segments] is empty or any [n <= 0]. *)
