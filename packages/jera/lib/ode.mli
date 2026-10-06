(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ordinary differential equations.

    A problem is a field [f t y], the derivative of the state [y] at the time
    [t], and an initial state. A state is any structure of tensors: its float
    leaves are the state's vector and share its steps, and its other leaves are
    carried unchanged. Times are tensors of their own float dtype ['t], which
    may differ from the leaves'. {!Rune.val-vmap} gives each lane its own
    problem.

    A method's tag says which drivers take it: [`Formula] methods take the
    caller's steps in {!march}; [`Embedded] methods estimate their error.

    {[
    (* The pendulum, sampled at 101 times by 10 steps of tsit5 each *)
    let pendulum _t (q, p) = (p, Nx.neg (Nx.sin q))

    let path =
      Ode.march
        Nx.Ptree.(pair tensor tensor)
        Ode.tsit5 ~steps:10 pendulum
        ~at:(Nx.linspace Nx.float64 0. 10. 101)
        (Nx.scalar Nx.float64 1., Nx.scalar Nx.float64 0.)
    ]} *)

(** {1:methods Methods} *)

type (-'k, 'y, 't) t
(** The type for one-step methods with tags ['k] over states ['y] and times of
    dtype ['t]. A method with more tags can be used as one with fewer, by
    coercion: [(Ode.tsit5 :> ([ `Formula ], _, _) Ode.t)]. *)

val euler : ([ `Formula ], 'y, 't) t
(** [euler] is the explicit Euler method: order 1, one field evaluation per
    step. *)

val rk4 : ([ `Formula ], 'y, 't) t
(** [rk4] is the classical Runge–Kutta method: order 4, four evaluations per
    step. *)

val ssprk3 : ([ `Formula ], 'y, 't) t
(** [ssprk3] is the three-stage strong-stability-preserving method of Shu and
    Osher (1988): order 3, three evaluations per step. A step is a convex
    combination of Euler steps, so it keeps any convex bound Euler keeps under
    the same step size, such as a total-variation bound of a hyperbolic
    discretisation. *)

val bs3 : ([ `Formula | `Embedded ], 'y, 't) t
(** [bs3] is Bogacki and Shampine's (1989) method of order 3 with an embedded
    order 2: three evaluations per step, its last stage the next step's first.
*)

val tsit5 : ([ `Formula | `Embedded ], 'y, 't) t
(** [tsit5] is Tsitouras' (2011) method of order 5 with an embedded order 4: six
    evaluations per step, its last stage the next step's first. *)

val dopri5 : ([ `Formula | `Embedded ], 'y, 't) t
(** [dopri5] is Dormand and Prince's (1980) method of order 5 with an embedded
    order 4: six evaluations per step, its last stage the next step's first. *)

val tableau :
  a:float array array ->
  b:float array ->
  c:float array ->
  ([ `Formula ], 'y, 't) t
(** [tableau ~a ~b ~c] is the explicit Runge–Kutta method of Butcher tableau
    [(a, b, c)] with [s] stages: stage [i] evaluates the field at [t + c.(i) h]
    on [y + h Σ_(j<i) a.(i).(j) k_j], and the step is [y + h Σ_i b.(i) k_i]. Its
    coefficients are given in float64 and rounded once to the time's and each
    leaf's dtype.

    Raises [Invalid_argument] unless [s ≥ 1], [b] and [c] have [s] elements, [a]
    has [s] rows of which row [i] has [i] elements, every coefficient is finite,
    and [b] sums to [1] within rounding. *)

(** {1:marches Marches} *)

type 't time = (float, 't) Nx.t
(** The type for times. *)

type ('y, 't) field = 't time -> 'y -> 'y
(** The type for fields: [f t y] is the derivative of [y] at the scalar time
    [t], a value of [y]'s structure, dtypes and shapes. *)

val march :
  'y Nx.Ptree.t ->
  ([> `Formula ], 'y, 't) t ->
  steps:int ->
  ('y, 't) field ->
  at:'t time ->
  'y ->
  'y
(** [march y m ~steps f ~at y0] is the state at each time of [at], stacked on a
    new leading axis of each leaf, [y0] first at [at.(0)]. Each interval of [at]
    takes [steps] equal steps of [m]; [at] may decrease, to march back.

    {b Error.} A method of order [p] has global error [O(h^p)] for a smooth
    field. {b Stability.} An explicit method is stable only while [h] times the
    field's eigenvalues lies in its stability region, so a stiff field needs a
    step far below its accuracy's. {b Cost.} The method's evaluations per step,
    once each step; a method whose last stage is the next step's first evaluates
    it once. {b Derivative.} The composition's: in the initial state, in the
    times of [at], and in every tracked value the field reads. Reverse mode
    keeps one state per time of [at] and recomputes each interval while it
    reverses it.

    Raises [Invalid_argument] if [steps < 1], if [at] is not a non-empty 1-D
    tensor, if it is not strictly monotone, or if [f] returns a value of another
    structure, dtype or shape than its state. *)

(** {1:solves Solves}

    A solve chooses its steps to meet a tolerance: an embedded method estimates
    each step's local error, and the proportional–integral controller of Hairer,
    Nørsett and Wanner (I, §II.4) sizes the next. The search runs on detached
    values and records its accepted steps; the answer takes them again with the
    tracked field, each step a fraction [s] of its interval, [h = (b − a) s], so
    its derivative is the accepted steps' on their grid, through the initial
    state, the times and every tracked value the field reads, and a lane that
    did not converge returns its detached estimate.

    {b Error.} [e] is one step's embedded error, and [y], per component, the
    larger of the step's two states; a step is accepted when [e] meets [tol].
    The solution's error is, per component, the sum of the magnitudes of the
    accepted steps' local estimates: it estimates the error the steps made, not
    the global error, which [tol] does not bound. An attempt with a non-finite
    stage is rejected; a non-finite field at an accepted state ends the lane
    [Not_finite], a step below the time's resolution [Stalled], and [budget]
    attempted steps [Budget_spent]. The last step of an interval lands on its
    end exactly. {b Cost.} Each attempt costs the method's evaluations less one,
    and the answer evaluates the accepted steps again. {b Memory.} Reverse mode
    keeps one state per time of [at] and, while it reverses an interval, its
    carries: compiled, [budget] of them; eagerly, the steps taken. *)

val solve :
  'y Nx.Ptree.t ->
  ([> `Embedded ], 'y, 't) t ->
  tol:Tol.t ->
  budget:int ->
  ('y, 't) field ->
  t0:'t time ->
  t1:'t time ->
  'y ->
  'y Solution.t
(** [solve y m ~tol ~budget f ~t0 ~t1 y0] is the state at [t1] of the solution
    from [y0] at [t0], scalars; [t1] may precede [t0].

    Raises [Invalid_argument] if [budget < 1], if [t0 = t1], or as {!march} does
    for a field of another structure. *)

val sample :
  'y Nx.Ptree.t ->
  ([> `Embedded ], 'y, 't) t ->
  tol:Tol.t ->
  budget:int ->
  ('y, 't) field ->
  at:'t time ->
  'y ->
  'y Solution.t
(** [sample y m ~tol ~budget f ~at y0] is the state at each time of [at],
    stacked on a new leading axis of each leaf, [y0] first at [at.(0)], with
    [budget] attempts across all of them. The steps land on every time of [at],
    so no state is interpolated.

    Raises [Invalid_argument] if [budget < 1], if [at] is not a non-empty 1-D
    tensor, through {!Nx.check} if it is not strictly monotone, and as {!march}
    does for a field of another structure. *)
