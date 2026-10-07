(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Numerical methods.

    Jera computes quantities defined as limits (integrals, solutions of
    differential equations, approximations of functions) over {!Nx} tensors.
    Every program is eager unless it says otherwise and runs unchanged under
    {!Rune.val-jit}, {!Rune.val-vmap} and the derivatives of {!Rune}; jera never
    compiles anything itself. A random source, a key or a Brownian path, is an
    argument of a compiled function: one it captures is a constant, whose draws
    it refuses.

    {1:index Methods by problem}

    {table
      {tr {th Problem } {th Regime } {th Method } }
      {tr {td Linear system } {td small, dense } {td {!Linear.dense} } }
      {tr {td  } {td banded } {td {!Linear.banded} } }
      {tr {td  } {td large, symmetric positive-definite } {td {!Linear.cg} } }
      {tr {td  } {td large } {td {!Linear.gmres} } }
      {tr {td Zero of a function } {td derivative given } {td {!Root.newton} } }
      {tr {td  } {td a bracket } {td {!Root.bracket} } }
      {tr {td Zero of a system } {td derivative given } {td {!System.newton} } }
      {tr {td  } {td no usable derivative } {td {!System.broyden} } }
      {tr {td  } {td fixed point [x = g x] } {td {!System.anderson} } }
      {tr {td Minimum } {td smooth, small } {td {!Minimize.bfgs} } }
      {tr {td  } {td smooth, large } {td {!Minimize.lbfgs} } }
      {tr {td  } {td smooth, ill-conditioned } {td {!Minimize.newton} } }
      {tr {td  } {td sum of squares } {td {!Minimize.levenberg_marquardt} } }
      {tr {td  } {td no useful gradient } {td {!Minimize.nelder_mead} } }
      {tr {td  } {td one variable, a bracket } {td {!Minimize.bracket} } }
      {tr
        {td Integral, one dimension }
        {td smooth }
        {td {!Quad.adaptive}; {!Quad.fixed}, {!Quad.cumulative} }
      }
      {tr
        {td  }
        {td of samples }
        {td {!Piecewise.integral} of an interpolant }
      }
      {tr
        {td  }
        {td endpoint singularity, infinite range }
        {td {!Quad.tanh_sinh} }
      }
      {tr
        {td Integral, several dimensions }
        {td up to ten }
        {td {!Quad.cubature} }
      }
      {tr {td  } {td high } {td {!Quad.qmc} } }
      {tr
        {td Approximation }
        {td samples }
        {td {!Piecewise.linear}, {!Piecewise.cubic}, {!Piecewise.hermite} }
      }
      {tr {td  } {td monotone samples } {td {!Piecewise.steffen} } }
      {tr {td  } {td a function, to a tolerance } {td {!Piecewise.adapt} } }
      {tr
        {td  }
        {td a function, fixed resolution }
        {td {!Piecewise.chebyshev}, {!Grid.chebyshev} }
      }
      {tr {td  } {td samples on a grid } {td {!Grid.linear}, {!Grid.cubic} } }
      {tr
        {td Differential equation }
        {td non-stiff }
        {td
          {!Ode.solve}, {!Ode.sample} with {!Ode.tsit5} and kin; {!Ode.march}
          with an explicit method, {!Ode.euler} to {!Ode.dopri5}
        }
      }
      {tr
        {td  }
        {td stiff; index-1 algebraic constraints }
        {td {!Ode.kvaerno5}, with a mass }
      }
      {tr {td  } {td randomness } {td {!Sde.march} } }
      {tr {td  } {td separable Hamiltonian, long times } {td {!Split} } }
    }

    {1:conventions Conventions}

    - {b Problems are closures.} A problem is an OCaml function and the data it
      is computed over. Every tracked value the function reads, argument or
      capture, reaches the derivative.
    - {b Batching.} Elementwise families ({!Root}, {!Minimize.bracket},
      {!Quad}'s one-dimensional integrals, evaluation) treat every element as
      its own problem; their function must not reduce or mix along any axis.
      {!Quad.cubature} and {!Quad.qmc} solve one problem per lane, their
      integrand reducing only the last, coordinate axis. Structured families
      solve one problem, and a state's tensors share its steps. {!Rune.val-vmap}
      gives each lane its own.
    - {b Dtypes.} The working dtype is the data's, with no default. Times have
      their own dtype. Float leaves of a state are its vector; other leaves are
      carried unchanged.
    - {b Formula or solve.} A function whose answer can miss on tensor data (a
      tolerance, a budget) is a solve and returns a {!Solution.t}, with a status
      per lane; one that cannot is a formula and returns its value, and its
      derivative is the composition's.
    - {b Searches and answers.} A solve searches on detached values, then states
      its answer from the search's decisions: a zero or a minimum by its
      equation through {!Rune.root}, an integral by its rule over the final
      partition, a flow by its accepted steps taken again, a fit by its final
      pieces. The derivative is the answer's: no search is differentiated, and a
      lane that did not converge has a zero derivative.
    - {b Tags.} A method's tag says which drivers take it: [`Formula] runs in a
      formula, [`Embedded] estimates its error and drives a solve. An impossible
      pairing is a type error.
    - {b Budgets} are arguments only where no bound is derivable, with no
      default.
    - {b Derivatives that steer.} A method that needs a derivative only to steer
      its search takes it from the caller ([slope]); it changes the speed only.
    - {b Errors.} A broken precondition on static data raises [Invalid_argument]
      naming the function, at once. A missed tolerance is a status. A broken
      precondition on a formula's tensor data raises [Invalid_argument] through
      {!Nx.check}: at once eagerly, when the compiled call returns under
      {!Rune.val-jit}.
    - {b Devices.} Constants are computed on the host in float64 and rounded
      once to the working dtype. *)

module Tol : sig
  (** Tolerances.

      A solve accepts a lane when its error estimate [e] at its value [y] meets
      the tolerance: the root mean square, over the lane's float components, of
      [e_i / s_i] is at most [1], with [s_i = abs + rel * |y_i|]. A component
      with [e_i = 0] counts [0]; one with [s_i = 0] and [e_i <> 0] is never
      accepted, so an answer that may be zero needs [abs]. Each solve says what
      its [e] and [y] are.

      The components add in a fixed order, pairwise over the flattened
      components, so a decision depends only on the values of the user's
      function, eagerly and compiled. Tolerances are OCaml floats: constants of
      a compiled program. *)

  type t
  (** The type for tolerances. *)

  val v : rel:float -> abs:float -> t
  (** [v ~rel ~abs] is the tolerance [abs + rel * |y|].

      Raises [Invalid_argument] if either is negative or not finite, or if both
      are [0]. *)

  val rel : float -> t
  (** [rel r] is [v ~rel:r ~abs:0.]. *)

  val abs : float -> t
  (** [abs a] is [v ~rel:0. ~abs:a]. *)

  val ulps : float -> t
  (** [ulps k] is [rel (k *. eps)] with [eps] the distance from [1] to the next
      float of the dtype the tolerance meets. It is met only where the problem
      determines its answer to [k] units: a zero of [f] to about [f]'s rounding
      error over [|f'|], a minimum to about the square root of [f]'s rounding
      over its curvature.

      Raises [Invalid_argument] if [k] is not finite and positive. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf t] formats [t] as ["rel 1e-06 abs 1e-10"] or ["ulps 4"]. *)
end

module Solution : sig
  (** Answers of solves.

      A solve returns its answer with a status per lane: one lane per element
      for an elementwise family ({!Root}, {!Minimize.bracket}, {!Quad}'s
      one-dimensional solves), one per box for {!Quad.cubature} and {!Quad.qmc},
      one for a structured family, and one per lane of each {!Rune.val-vmap}
      around it. {!get} reads an answer that converged everywhere and raises
      otherwise; {!best} and {!ok} read every lane.

      {[
      let s = Root.bracket ~tol f ~lo ~hi in
      let ok = Solution.ok s in
      Nx.where ok (Solution.best s) (Nx.zeros_like lo)
      ]} *)

  type 'a t
  (** The type for answers of type ['a]. *)

  (** The type for a lane's outcome. *)
  type status =
    | Converged  (** The error estimate met the tolerance. *)
    | Budget_spent  (** The budget ran out first. *)
    | Not_bracketed  (** The ends of a bracket have one sign. *)
    | Not_finite  (** A value of the problem's function is not finite. *)
    | Stalled
        (** The search stopped moving without meeting the tolerance: its steps
            fell below the resolution of the floats, or the problem has no
            answer the method can reach, such as a pole in a bracket. *)

  val get : 'a t -> 'a
  (** [get s] is the answer if every lane converged.

      Raises [Failure] with the report of the first lane in C order that did not
      converge: the entry point, the lane, its status, the solve's settings, the
      lane's data, the budget it used, what to change, and how many other
      elements of its problem converged. Inside {!Rune.val-jit}, [get] returns
      {!best} and the compiled call raises when it returns. *)

  val best : 'a t -> 'a
  (** [best s] is the estimate in every lane. A lane that did not converge holds
      its last finite estimate, and its derivative is zero. *)

  val ok : 'a t -> (bool, Nx.bool_elt) Nx.t
  (** [ok s] is [is Converged s]. *)

  val is : status -> 'a t -> (bool, Nx.bool_elt) Nx.t
  (** [is st s] is [true] in each lane whose status is [st], of the lanes'
      shape. *)

  val error : 'a t -> 'a
  (** [error s] is each lane's error estimate, of the answer's structure, as its
      solve defines it: an estimate, never a bound. *)

  val evaluations : 'a t -> (int32, Nx.int32_elt) Nx.t
  (** [evaluations s] is each lane's count of the points at which the problem's
      function was evaluated, of the lanes' shape. *)

  val ptree : 'a Nx.Ptree.t -> 'a t Nx.Ptree.t
  (** [ptree s] is the structure of answers of structure [s], so a compiled
      function returns one: the answer at [value], the error at [error], the
      statuses at [status], the counts at [evaluations], the budget used at
      [used] and the report's data under [report]. *)

  val pp : Format.formatter -> 'a t -> unit
  (** [pp ppf s] formats the count of lanes in each status and the report {!get}
      would raise for the first lane that did not converge. It reads the
      statuses, so it runs eagerly. *)
end

module Linear : sig
  (** Linear systems.

      A system is a linear function [a] on values of a structure ['x] and a
      right-hand side [r]: its solution is the [u] with [a u = r]. The float
      tensors of a value are its vector, of one dtype, the solve's; its other
      tensors are carried from [r]. A solver says how [a] is used: {!dense}
      materialises it from its products and factors it, {!banded} does so for a
      matrix with entries near its diagonal only, and {!cg} and {!gmres} iterate
      on its products.

      {[
      (* u with (k I + L) u = r, L a matrix given by its product *)
      let s =
        Linear.solve Nx.Ptree.tensor Linear.dense
          (fun u -> Nx.add (Nx.mul_s u k) (Nx.matmul l u))
          r
      ]}

      {b Check.} Every answer is checked by one more application of [a]: its
      residual [a u − r] must meet the solver's bound, stated with each. A miss,
      or a non-finite [u] from a finite system, ends the lane [Stalled]: [a] is
      not linear, or it is singular or too ill-conditioned for the solver. A
      non-finite [r] or product of [a] ends it [Not_finite]. The answer's error
      is the magnitude of its residual, per component.

      {b Derivative.} The answer is stated as the zero of [a u − r] through
      {!Rune.root}, with the solver as its linear solve: a derivative solves a
      system of [a], or of its transpose in reverse mode, with the same solver,
      so the solver must suit both. *)

  type 'x t
  (** The type for solvers of systems on values of type ['x]. *)

  val dense : 'x t
  (** [dense] materialises [a] as an [n × n] matrix, [n] the size of the vector,
      with one application to each vector of the standard basis, and solves it
      with {!Nx.solve}: [n + 1] applications with the check, and [2n³/3] flops.
      Its bound is [c n ε (‖A‖ ‖u‖ + ‖r‖)], the backward error of an LU
      factorisation with partial pivoting, with [c = 16], [ε] the dtype's
      machine epsilon, [‖A‖] the Frobenius norm of the matrix and the vectors'
      norms Euclidean. It suits systems of up to some thousands of elements. *)

  val banded : width:int -> 'x t
  (** [banded ~width] is for an [a] whose matrix has entries only within [width]
      of its diagonal, such as a finite difference in one dimension. It probes
      [a] with [2 width + 1] applications, each to the sum of the basis vectors
      of one residue modulo [2 width + 1], whose columns share no row, and
      factors the band by LU with partial pivoting, whose upper band grows to
      [2 width]: [2 width + 2] applications with the check, and [O(n width²)]
      flops in loops of [n] steps. Its bound is {!dense}'s, with [‖A‖] the
      band's Frobenius norm. A band too narrow for [a] mixes its columns, and
      the solution misses the check.

      Raises [Invalid_argument] if [width < 0]. *)

  val cg : rel:float -> budget:int -> precondition:('x -> 'x) -> 'x t
  (** [cg ~rel ~budget ~precondition] is conjugate gradients from [u = 0],
      preconditioned by [precondition], for a symmetric positive-definite [a]
      such as a Hessian at a minimum or a damped Gauss–Newton matrix: an
      application of [a] and of [precondition] per iteration, and [O(n)] other
      work on three vectors. [precondition] approximates the inverse of [a] and
      is symmetric positive-definite itself; [Fun.id] is none. It stops when the
      residual it carries meets [‖a u − r‖ ≤ rel ‖r‖], the vectors' norms
      Euclidean, which is also its bound, so [r = 0] gives [u = 0] with the
      check alone. [budget] iterations end the lane [Budget_spent], and a
      direction [p] with [pᵀ a p ≤ 0], which no positive-definite [a] has, ends
      it [Stalled].

      Raises [Invalid_argument] if [rel] is not in (0, 1) or if [budget < 1]. *)

  val gmres :
    restart:int -> rel:float -> budget:int -> precondition:('x -> 'x) -> 'x t
  (** [gmres ~restart ~rel ~budget ~precondition] is restarted GMRES from
      [u = 0], preconditioned on the right by [precondition], for any
      non-singular [a]. A cycle takes [restart] steps of Arnoldi's process, each
      an application of [a] and of [precondition], orthogonalised twice by
      Gram–Schmidt, then the [u] of least residual in their span; it keeps
      [restart + 1] vectors. Between cycles it applies [a] to [u] for the
      residual, and stops when [‖a u − r‖ ≤ rel ‖r‖], the vectors' norms
      Euclidean, which is also its bound, so [r = 0] gives [u = 0] with no
      application. After [budget / restart] cycles, the lane ends
      [Budget_spent]. [precondition] approximates the inverse of [a] and of its
      transpose, as a derivative solves both; [Fun.id] is none.

      Raises [Invalid_argument] if [restart < 1], if [budget < restart] or if
      [rel] is not in (0, 1). *)

  val solve : 'x Nx.Ptree.t -> 'x t -> ('x -> 'x) -> 'x -> 'x Solution.t
  (** [solve x s a r] is the [u] with [a u = r], found by [s].

      Raises [Invalid_argument] if the float tensors of [r] differ in dtype, or
      if [a] returns a value of another structure, dtype or shape than its
      argument. *)
end

module Root : sig
  (** Zeros of functions of one variable.

      Both methods are elementwise: every element of the input is its own
      problem, with its own status, and [f] must compute each element of its
      result from the same element of its argument alone. The answer is stated
      as a zero of [f] ({!Rune.root}), so its derivative is the implicit one,
      [−∂f/∂θ / ∂f/∂x] at a converged element, through every tracked value [f]
      reads, and zero at an element that did not converge. At a converged zero
      where [∂f/∂x] is [0] the derivative does not exist and is not finite.

      The derivative solves elementwise and checks its solution with one more
      product: where it misses, [f] read another element than its own, and the
      derivative raises [Invalid_argument] naming the entry point.

      {[
      (* Kepler's equation M = E − e sin E, for each epoch *)
      let eccentric_anomaly m e =
        Root.newton
          ~tol:(Tol.v ~rel:1e-12 ~abs:1e-15)
          ~budget:8
          ~slope:(fun x -> Nx.rsub_s 1. (Nx.mul e (Nx.cos x)))
          (fun x -> Nx.sub (Nx.sub x (Nx.mul e (Nx.sin x))) m)
          (Nx.add m (Nx.mul e (Nx.sin m)))
        |> Solution.get
      ]} *)

  val bracket :
    tol:Tol.t ->
    ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
    lo:(float, 'b) Nx.t ->
    hi:(float, 'b) Nx.t ->
    (float, 'b) Nx.t Solution.t
  (** [bracket ~tol f ~lo ~hi] is a zero of [f] between [lo] and [hi] in each
      element, given in either order, which broadcast together.

      {b Method.} ITP steps (Oliveira and Takahashi, 2020) interleaved with
      bisections of the floats' ordered-integer images, so an element ends
      within [2b + 2] evaluations for a [b]-bit dtype (130 in float64), whatever
      [f]. {b Error.} [e] is half the final bracket and [y] its midpoint; an
      element also ends when no float lies strictly inside its bracket. It
      converges only if [|f|] at its estimate, the bracket's end of smaller
      [|f|], is at most the smaller [|f|] at the given ends, so a pole ends
      [Stalled]; a bounded jump looks like a steep zero at float resolution and
      converges at the jump, where the derivative does not exist. Ends of one
      sign end [Not_bracketed]; a non-finite value of [f] ends [Not_finite]. *)

  val newton :
    tol:Tol.t ->
    budget:int ->
    slope:((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
    ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t Solution.t
  (** [newton ~tol ~budget ~slope f x0] is a zero of [f] near [x0] in each
      element, by Newton's method: [slope x] is [f]'s derivative, elementwise.
      It steers the steps only, so a wrong slope slows a solve and cannot end it
      early.

      {b Method.} Undamped Newton steps [δ = −f x / slope x], quadratic near a
      simple zero. {b Error.} [e = |δ| q / (1 − q)], with [q] the ratio of the
      last two steps, the distance left when the steps shrink by [q]: unbounded
      while [q ≥ 1] or before two steps; [y] is the estimate. A zero step is a
      zero. A zero or non-finite slope, or a step that no longer moves the
      estimate without meeting [tol], ends the element [Stalled]; a non-finite
      [f] ends it [Not_finite]; [budget] iterations end it [Budget_spent], so a
      budget of 1 converges only on a zero step. {b Cost.} One call of [f] and
      one of [slope] per iteration.

      Raises [Invalid_argument] if [budget < 1]. *)
end

module System : sig
  (** Zeros of systems.

      A system is a function [f] from values of a structure ['x] to values of
      the same structure, dtypes and shapes: its zero is the [x] with [f x = 0].
      The float tensors of a value are its vector, of one dtype; its other
      tensors are carried from the guess. A system is one problem, with one
      status; {!Rune.val-vmap} gives each lane its own.

      {[
      (* x² + y² = 4 and x = y, from (1, 2) *)
      let f v =
        let x = Nx.get [ 0 ] v and y = Nx.get [ 1 ] v in
        Nx.stack
          [ Nx.sub_s (Nx.add (Nx.square x) (Nx.square y)) 4.; Nx.sub x y ]
      in
      System.solve Nx.Ptree.tensor
        (System.newton ~derivative:(fun v dv -> snd (Rune.jvp' f v dv)))
        ~linear:Linear.dense ~tol:(Tol.rel 1e-12) ~budget:20 f
        (Nx.create Nx.float64 [| 2 |] [| 1.; 2. |])
      ]}

      {b Error.} Every method takes undamped steps [δ]: Newton's or Broyden's
      direction before the line search scales it, Anderson's fixed-point
      residual [f x]. [e = |δ| q / (1 − q)] per component, the distance left
      when the steps shrink by [q], with [q] the larger of the contractions of
      the undamped map [N x = x + δ] over the last two steps, each
      [|N x − N x'| / |x − x'|] from the point [x'] its step started at; after a
      step taken in full it is the ratio of the norms of the last two undamped
      steps. [e] is unbounded while [q ≥ 1] or before three steps, and [y] is
      the estimate. A lane that meets [tol] takes its last step in full. A
      shortened or mixed step moves the estimate but its length never enters
      [e], so a search cannot converge by shrinking its steps. A zero step is a
      zero. A step that no longer moves the estimate without meeting [tol], a
      line search that finds no decrease, or a failed linear solve ends the lane
      [Stalled]; a non-finite [f] at the guess or at an estimate ends it
      [Not_finite]; [budget] iterations end it [Budget_spent].

      {b Derivative.} The answer is stated as the zero of [f] through
      {!Rune.root}, so its derivative is the implicit one, [−J⁻¹ ∂f/∂θ] through
      every tracked value [f] reads, with [J] rune's derivative of [f] at the
      answer, solved by [linear]; zero at a lane that did not converge. At a
      zero where [J] is singular the derivative does not exist and is not
      finite. *)

  type 'x t
  (** The type for methods for systems on values of type ['x]. *)

  val newton : derivative:('x -> 'x -> 'x) -> 'x t
  (** [newton ~derivative] is Newton's method: [derivative x dx] is [f]'s
      Jacobian at [x] applied to [dx], such as [snd (Rune.jvp' f x dx)]. It
      steers the steps only, so a wrong one slows a solve and cannot end it
      early.

      {b Method.} Each step solves [J δ = −f x] with [linear], then backtracks
      from [x + δ] to the sufficient decrease of [|f|² / 2], by safeguarded
      quadratic steps. {b Stability.} The line search makes every step decrease
      [|f|], so the search can end at a local minimum of [|f|] that is no zero,
      where it stalls. {b Cost.} Per iteration, one solve of [linear] on
      [derivative] and one evaluation of [f] per trial of the line search;
      quadratic convergence near a simple zero, and with a Krylov solver it is
      Newton–Krylov. *)

  val broyden : 'x t
  (** [broyden] is Broyden's method, for an [f] with no usable derivative.

      {b Method.} The Jacobian's estimate [B] starts as forward differences and
      takes Broyden's rank-one update after each step; each step solves
      [B δ = −f x] with [linear] and backtracks by halving to Li and Fukushima's
      derivative-free decrease test,
      [|f (x + α δ)| ≤ (1 + η_k) |f x| − σ |α δ|²] with [η_k = 1 / (k + 1)²] and
      [σ = 10⁻⁴], which short steps always meet. {b Cost.} [n] evaluations for
      the differences, [n] the size of the vector, then one per trial; it keeps
      the [n × n] matrix [B], so it suits up to some hundreds of unknowns, and
      converges superlinearly near a simple zero. *)

  val anderson : memory:int -> 'x t
  (** [anderson ~memory] is for a fixed point [x = g x] stated as the residual
      [f x = g x − x].

      {b Method.} Iterates [g] with Anderson mixing of the last [memory] steps:
      the mixed point is [g x − ΔG γ], with [ΔF] and [ΔG] the changes of [f] and
      [g] over those steps and [γ] least [|f x − ΔF γ|] by QR, dropping the
      oldest steps while the ratio of [R]'s diagonal entries exceeds [ε^(−1/2)].
      A mixed point that does not reduce [|f|] is replaced by the Picard point
      [g x]. [memory = 0] is Picard iteration. {b Cost.} One evaluation per
      iteration, two when the Picard point replaces the mixed one, and a QR of
      [n × memory].

      Raises [Invalid_argument] if [memory < 0]. *)

  val solve :
    'x Nx.Ptree.t ->
    'x t ->
    linear:'x Linear.t ->
    tol:Tol.t ->
    budget:int ->
    ('x -> 'x) ->
    'x ->
    'x Solution.t
  (** [solve x m ~linear ~tol ~budget f guess] is a zero of [f] near [guess] by
      [m]. [linear] solves the method's linear systems and the derivative's. Its
      evaluations count the calls of [f], [budget] its iterations.

      Raises [Invalid_argument] if [budget < 1], if the float tensors of [guess]
      differ in dtype, or if [f] returns a value of another structure, dtype or
      shape than its argument. *)
end

module Minimize : sig
  (** Minima.

      A minimum of [f] is a zero of its gradient: every gradient method searches
      with rune's gradient of [f] and states its answer as the zero of that
      gradient, so a hand-written gradient never becomes the equation. A known
      gradient is stated on [f] with {!Rune.custom_jvp}. The float tensors of a
      value of ['x] are its vector, of one dtype, the search's; its other
      tensors are carried from the start. A problem is one problem, with one
      status; {!Rune.val-vmap} gives each lane its own.

      {[
      (* Rosenbrock's function from (−1.2, 1) *)
      let rosenbrock v =
        let x = Nx.get [ 0 ] v and y = Nx.get [ 1 ] v in
        Nx.add
          (Nx.mul_s (Nx.square (Nx.sub y (Nx.square x))) 100.)
          (Nx.square (Nx.rsub_s 1. x))
      in
      Minimize.solve Nx.Ptree.tensor
        (Minimize.bfgs ~linear:Linear.dense)
        ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
        ~budget:100 rosenbrock
        (Nx.create Nx.float64 [| 2 |] [| -1.2; 1. |])
      ]}

      {b Error.} The methods take undamped steps [δ], the quasi-Newton or Newton
      direction before the line search scales it, and measure them as {!System}
      does: [e = |δ| q / (1 − q)] per component, with [q] the larger of the
      contractions of the undamped map [x + δ] over the last two steps, and [y]
      the estimate; a lane that meets [tol] takes its last step in full. A step
      that is not downhill, a line search that finds no decrease, or a failed
      linear solve ends the lane [Stalled]; a non-finite [f] or gradient at the
      start ends it [Not_finite]; [budget] iterations end it [Budget_spent].

      {b Derivative.} The answer is stated as the zero of rune's gradient of [f]
      through {!Rune.root}, so its derivative is [−H⁻¹ ∂θ∇f] through every
      tracked value [f] reads, with [H] the Hessian at the answer, solved by the
      method's [linear] on Hessian-vector products; zero at a lane that did not
      converge. Where [H] is singular the derivative does not exist and is not
      finite. *)

  type ('x, 'f) t
  (** The type for methods for objectives of type ['f] over values of type ['x].
  *)

  val bfgs : linear:'x Linear.t -> ('x, 'x -> (float, 'b) Nx.t) t
  (** [bfgs ~linear] is BFGS, for a smooth objective of up to some thousands of
      unknowns.

      {b Method.} Keeps an estimate [H] of the inverse Hessian, scaled to
      [yᵀs / yᵀy] before its first update, and takes Broyden, Fletcher, Goldfarb
      and Shanno's rank-two update after each step [s] that changed the gradient
      by [y]; each step is [−H ∇f], searched to the strong Wolfe conditions by
      bracketing and zoom (Nocedal and Wright, Algorithms 3.5 and 3.6), so every
      update keeps [H] positive-definite. {b Cost.} [n²] numbers and [O(n²)]
      work per iteration, one evaluation of [f] and its gradient per trial;
      superlinear convergence near a minimum. *)

  val lbfgs : memory:int -> linear:'x Linear.t -> ('x, 'x -> (float, 'b) Nx.t) t
  (** [lbfgs ~memory ~linear] is limited-memory BFGS (Liu and Nocedal, 1989),
      for a smooth objective of many unknowns.

      {b Method.} The direction is the two-loop recursion over the last [memory]
      pairs of steps and gradient changes, with the initial inverse Hessian
      [yᵀs / yᵀy] of the newest pair; a pair of non-positive curvature enters
      neither loop. Steps are searched as {!bfgs}'s. {b Cost.} [2 memory n]
      numbers and [O(memory n)] work per iteration; linear convergence.

      Raises [Invalid_argument] if [memory < 1]. *)

  val newton : linear:'x Linear.t -> ('x, 'x -> (float, 'b) Nx.t) t
  (** [newton ~linear] is Newton's method, for a smooth, ill-conditioned
      objective.

      {b Method.} Each step solves [H δ = −∇f] with [linear] on rune's
      Hessian-vector products, truncated by its tolerance with {!Linear.cg},
      then backtracks from [x + δ] to the sufficient decrease of [f]. A Hessian
      that is not positive-definite gives a step that is not downhill, or ends
      {!Linear.cg}, and the lane stalls. {b Cost.} One solve of [linear] per
      iteration; quadratic convergence near a minimum. *)

  val levenberg_marquardt :
    'r Nx.Ptree.t -> linear:'x Linear.t -> ('x, 'x -> 'r) t
  (** [levenberg_marquardt r ~linear] is for [|r x|² / 2] given by its residual
      [r], a value of structure [r] whose float tensors are one vector.

      {b Method.} Each iteration materialises [r]'s Jacobian [J] with one
      Jacobian-vector product per unknown and solves [(JᵀJ + λ D) δ = −Jᵀ r]
      with [linear]; [D] is the running maximum of [JᵀJ]'s diagonal. A step is
      accepted when it achieves more than [10⁻⁴] of the decrease the linear
      model predicts; [λ] then falls, and grows on each rejection (Nielsen,
      1999), so near a small-residual minimum the steps are Gauss–Newton's.
      {b Error.} The undamped step is the Gauss–Newton step at the estimate,
      solved once the damped step has met the test. {b Stability.} The normal
      equations square [J]'s condition: above about [ε^(−1/2)] its steps lose
      accuracy and convergence slows, while the answer's equation is unaffected.
      {b Cost.} [n] products and one solve of [linear] per iteration, and one
      evaluation of [r] per trial. {b Derivative.} The minimum is the zero of
      rune's gradient of [|r|² / 2]; at a rank-deficient minimum the derivative
      does not exist and is not finite. *)

  val nelder_mead : ('x, 'x -> (float, 'b) Nx.t) t
  (** [nelder_mead] is the Nelder–Mead simplex, for an objective with no useful
      gradient: non-smooth or piecewise constant.

      {b Method.} The simplex of [x0] and [x0 + h_i e_i], [h_i] a twentieth of
      [x0_i] or [2.5 · 10⁻⁴] where it is zero, reflects its worst vertex through
      the others' centroid and expands, contracts or shrinks, with Gao and Han's
      (2012) coefficients for [n] unknowns, [1], [1 + 2/n], [3/4 − 1/(2n)] and
      [1 − 1/n], so the simplex keeps its shape as [n] grows. {b Error.} [e] is
      the simplex's diameter per component around its best vertex, [y] that
      vertex. Once [e] meets [tol], a fresh simplex of that diameter around the
      best vertex is evaluated: if no vertex is lower by a sufficient decrease
      the lane converges, and otherwise it continues from it (Kelley, 1999); a
      diameter alone converges on McKinnon's function at a point whose gradient
      is [(0, 1)]. A non-finite value counts as [+∞], one at the start ends the
      lane [Not_finite]. [budget] counts evaluations. {b Cost.} One evaluation
      per trip, two when it expands or contracts, [n + 1] more when it shrinks.
      {b Derivative.} None: its answer is the detached best vertex, since an
      objective with no useful gradient has a minimum with no equation. *)

  val solve :
    'x Nx.Ptree.t ->
    ('x, 'f) t ->
    tol:Tol.t ->
    budget:int ->
    'f ->
    'x ->
    'x Solution.t
  (** [solve x m ~tol ~budget f x0] is a local minimum of [f] from [x0] by [m].
      Its evaluations count the calls of [f], with its gradient for a gradient
      method, and [budget] its iterations, or its evaluations for
      {!nelder_mead}.

      Raises [Invalid_argument] if [budget < 1], if the float tensors of [x0]
      differ in dtype, or if [f] returns other than a scalar. *)

  val iterates : 'x Nx.Ptree.t -> ('x, 'f) t -> steps:int -> 'f -> 'x -> 'x
  (** [iterates x m ~steps f x0] is the first [steps] estimates of [m]'s search
      from [x0], [x0] first, detached, stacked on a new leading axis of every
      tensor. A lane stops at a zero gradient, or when its search finds no
      decrease, and then repeats its estimate. Its tensors that are not floats
      are [x0]'s, repeated.

      Raises [Invalid_argument] if [steps < 1], or as {!solve} does. *)

  val bracket :
    tol:Tol.t ->
    ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
    lo:(float, 'b) Nx.t ->
    hi:(float, 'b) Nx.t ->
    (float, 'b) Nx.t Solution.t
  (** [bracket ~tol f ~lo ~hi] is a minimum of [f] in each [[lo, hi]],
      elementwise: every element is its own problem, the ends broadcast together
      and come in either order, and [f] must compute each element of its result
      from the same element of its argument alone.

      {b Method.} Brent's method: parabolic steps through the three best points,
      forced to a golden-section step whenever the bracket has not shrunk by
      [0.618] over two evaluations, so an element ends within about [3b]
      evaluations for a [b]-bit dtype. On a function with several minima in the
      bracket it finds one of them. {b Error.} [e] is half the final bracket and
      [y] its midpoint; an element also ends when no float lies strictly inside
      its bracket. A non-finite value of [f] ends it [Not_finite].
      {b Derivative.} A minimum inside the bracket is stated as a zero of rune's
      derivative of [f], so its derivative is [−∂²f/∂x∂θ / ∂²f/∂x²]; a minimum
      at an end is that end, with its derivative; a lane that did not converge
      has a zero derivative. *)
end

module Quad : sig
  (** Integrals.

      An integrand is an elementwise function: it receives points with the
      rule's node axes in front and the range's shape behind, and returns one
      value per point. Over a range every element is its own integral, so the
      integrand must not reduce or mix along any axis; over a box every lane is
      its own integral, and the integrand reduces only the last, coordinate
      axis. The axes in front and their sizes are the method's, and may change
      between calls.

      {[
      (* ∫₀¹ x^a dx for each a, by 10-point Gauss–Legendre *)
      let moments a =
        Quad.fixed (Quad.Rule.gauss 10)
          (fun x -> Nx.pow x a)
          (Quad.Range.v (Nx.zeros_like a) (Nx.ones_like a))
      ]}

      An integral of samples is the integral of their interpolant
      ({!Piecewise.integral}). *)

  (** {1:rules Rules} *)

  (** Quadrature rules on [[−1, 1]].

      A rule's tag says which integrals take it: [`Formula] rules sum their
      nodes in {!fixed} and {!cumulative}; [`Embedded] rules also estimate their
      error. *)
  module Rule : sig
    type -'k t
    (** The type for rules with tags ['k]. A rule with more tags can be used as
        one with fewer, by coercion:
        [(Quad.Rule.kronrod 7 :> [ `Formula ] Quad.Rule.t)]. *)

    val gauss : int -> [ `Formula ] t
    (** [gauss n] is the [n]-point Gauss–Legendre rule, exact for polynomials of
        degree [2n − 1]. Its nodes and weights are computed on the host in
        float64 by Newton's method on the Legendre polynomial.

        Raises [Invalid_argument] if [n < 1]. *)

    val kronrod : int -> [ `Formula | `Embedded ] t
    (** [kronrod n] is the [(2n + 1)]-point Gauss–Kronrod rule that extends
        {!gauss}[ n] (Piessens et al., QUADPACK, 1983): exact for polynomials of
        degree [3n + 1], with the [n] Gauss nodes among its own. [n] is [7] or
        [10].

        Raises [Invalid_argument] if [n] is neither. *)

    val nodes :
      _ t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t * (float, 'b) Nx.t
    (** [nodes r dtype] is [(x, w)], [r]'s nodes in increasing order and their
        weights, 1-D tensors of [dtype]: [Σ w f(x)] approximates [∫₋₁¹ f]. A sum
        in log space is [logsumexp (log w + ℓ x)]. *)
  end

  (** {1:ranges Ranges} *)

  (** Ranges of integration, one per element. *)
  module Range : sig
    type 'b t
    (** The type for ranges of dtype ['b]. *)

    val v : (float, 'b) Nx.t -> (float, 'b) Nx.t -> 'b t
    (** [v a b] is [[a, b]], [a] and [b] broadcast together. An integral over
        [[a, b]] with [b < a] is minus the integral over [[b, a]]. *)

    val from : (float, 'b) Nx.t -> 'b t
    (** [from a] is the half-line from [a] to [+∞], at unit scale: a rule's
        nodes spread over distances near [1] from [a]. Scale the variable so the
        integrand's width is near 1. *)

    val line : (float, 'b) Nx.t -> 'b t
    (** [line c] is the whole line, around [c] at unit scale. *)
  end

  (** Boxes of integration, one per lane. *)
  module Box : sig
    type 'b t
    (** The type for boxes of dtype ['b]. *)

    val v : (float, 'b) Nx.t -> (float, 'b) Nx.t -> 'b t
    (** [v lo hi] is the box with corners [lo] and [hi], which broadcast
        together to [lanes @ [d]]: a point's coordinates are the last axis. An
        axis with [hi < lo] flips the integral's sign, as {!Range.v}.

        Raises [Invalid_argument] if [lo] and [hi] do not broadcast or are
        scalars. *)
  end

  type 'b integrand = (float, 'b) Nx.t -> (float, 'b) Nx.t
  (** The type for integrands: [f x] is [f] at each point of [x]: of [x]'s shape
      over a range, and over a box of [x]'s shape without its last, coordinate
      axis. *)

  (** {1:formulas Formulas} *)

  val fixed :
    [> `Formula ] Rule.t -> 'b integrand -> 'b Range.t -> (float, 'b) Nx.t
  (** [fixed r f range] is [r]'s sum for the integral of [f] over each element
      of [range], of [range]'s shape. [f] receives the points of shape
      [[m] @ shape], [m] the rule's nodes. A finite range maps the rule
      linearly; [from a] by [x = a + (1 + u) / (1 − u)] and [line c] by
      [x = c + u / (1 − u²)], so an infinite range needs an integrand that
      decays fast enough for the rule's degree to show.

      {b Error.} An [n]-point Gauss sum on a finite range is exact for
      polynomials of degree [2n − 1], and for a smooth [f] its error falls
      geometrically in [n]. {b Cost.} One call of [f] on [m] points per element.
      {b Derivative.} The sum's: in the integrand's parameters, in the ends and
      in the points [f] reads.

      Raises [Invalid_argument] if [f]'s result has another shape than its
      points. *)

  val cumulative :
    [> `Formula ] Rule.t -> 'b integrand -> (float, 'b) Nx.t -> (float, 'b) Nx.t
  (** [cumulative r f knots] is the integral of [f] from the first knot to each,
      by [r] on each interval between knots: for [knots] of shape [[n] @ shape],
      a result of the same shape whose row [0] is zero and row [i] the sum of
      the first [i] intervals' integrals. [f] receives points of shape
      [[m; n − 1] @ shape].

      Raises [Invalid_argument] if [knots] has no axis or no knot, or as
      {!fixed} does. *)

  (** {1:solves Solves}

      Each solve is elementwise: every element of the range is its own integral,
      with its own status. Its search runs on detached values; the answer of a
      converged element is its rule over its final decisions, tracked, so its
      derivative is that rule's, and an element that did not converge returns
      its detached best estimate. *)

  val adaptive :
    [> `Embedded ] Rule.t ->
    tol:Tol.t ->
    budget:int ->
    'b integrand ->
    'b Range.t ->
    (float, 'b) Nx.t Solution.t
  (** [adaptive r ~tol ~budget f range] is the integral of [f] over each element
      of [range]. [f] receives points of shape [[m; k] @ shape], [m] the rule's
      points and [k] pieces.

      {b Method.} The rule [r] on a partition it refines: it bisects the piece
      of largest error until the error meets [tol]. An infinite range is mapped
      as {!fixed} maps it. {b Error.} [e] is the sum over the pieces of
      [|K − G|], the Kronrod sum's difference from its embedded Gauss sum, and
      [y] the integral. An element whose worst piece is at level 62, or holds no
      float strictly inside, ends [Stalled]; a non-finite sum ends it
      [Not_finite]; [budget] pieces end it [Budget_spent]. A feature narrower
      than the first rule's nodes can be invisible to every estimate, and an
      element can converge without it. {b Cost.} [2n + 1] points per piece, and
      the answer evaluates the final partition again, in chunks of 32 pieces
      under a {!Rune.scan}, so reverse mode keeps one chunk's values.
      {b Derivative.} The final partition's rule's: the pieces are integers
      [(level, index)] whose ends are [a + (b − a) index / 2^level], so it
      reaches the ends.

      Raises [Invalid_argument] if [budget < 1], or as {!fixed} does. *)

  val tanh_sinh :
    tol:Tol.t -> 'b integrand -> 'b Range.t -> (float, 'b) Nx.t Solution.t
  (** [tanh_sinh ~tol f range] is the integral of [f] over each element of
      [range] by a double-exponential rule: tanh-sinh on a finite range,
      exp-sinh on {!Range.from} and sinh-sinh on {!Range.line}. Its nodes crowd
      toward the ends double-exponentially, so it converges for an integrable
      singularity at an end, and on a half-line or the line for an integrand
      that decays. Scale the variable so the integrand's width is near 1. [f]
      receives points of shape [[64] @ shape], a chunk of nodes.

      {b Method.} The trapezoidal rule in [t] after the change of variable, at
      steps [1, 1/2, 1/4, ...]: each level adds the odd multiples of its step,
      in fixed-size chunks, to the finest level the dtype calls for ([2^-7] in
      float64, [2^-6] in float32). The nodes stop where their numbers leave the
      dtype's normal floats; a node whose point rounds to an end is unused. Near
      an end [a = 0] a point is its own distance to the end, so the nodes reach
      the singularity in full precision; another end loses the digits [a]'s
      magnitude rounds away, unless the integrand computes the distance to the
      end from its argument. {b Error.} [e] is the difference of the last two
      levels and [y] the integral. A lane whose terms at the truncation do not
      fall below the tolerance, or whose finest level does not meet it, ends
      [Stalled]; a non-finite sum ends it [Not_finite]. {b Cost.} At most the
      nodes of the finest level, in chunks of 64, per element, and the answer
      evaluates the final level's nodes again. {b Derivative.} That of the sum
      over the final level, through the ends and the integrand's parameters. *)

  val cubature :
    tol:Tol.t ->
    budget:int ->
    'b integrand ->
    'b Box.t ->
    (float, 'b) Nx.t Solution.t
  (** [cubature ~tol ~budget f box] is the integral of [f] over each lane's box,
      of [d] dimensions with [2 ≤ d ≤ 10]. [f] receives points of shape
      [[m; k] @ lanes @ [d]], [m] the rule's points and [k] boxes, and reduces
      only their last, coordinate axis.

      {b Method.} Genz and Malik's (1980) adaptive rule of degree 7 with an
      embedded degree 5: it bisects the box of largest error across the axis of
      largest fourth difference. {b Error.} [e] is the sum over the boxes of the
      two rules' difference, and [y] the integral. A lane whose worst box is at
      level 62 along its axis, or holds no float strictly inside along it, ends
      [Stalled]; a non-finite sum ends it [Not_finite]; [budget] boxes end it
      [Budget_spent]. {b Cost.} [2^d + 2d² + 2d + 1] points per box, [budget]
      boxes at most per lane, and the answer evaluates the final partition
      again. {b Derivative.} The final partition's rule's, through the corners
      and the integrand's parameters.

      Raises [Invalid_argument] if [d] is not in [[2, 10]], if [budget < 1], or
      if [f]'s result is not the points' shape without its last axis. *)

  val qmc :
    Nx.Rng.t ->
    tol:Tol.t ->
    budget:int ->
    'b integrand ->
    'b Box.t ->
    (float, 'b) Nx.t Solution.t
  (** [qmc key ~tol ~budget f box] is the integral of [f] over each lane's box,
      of any dimension [d] up to 1111, by randomised quasi-Monte Carlo. [f]
      receives points of shape [[64; 16] @ lanes @ [d]], a chunk of 64 points
      under each of 16 shifts, and reduces only their last, coordinate axis.

      {b Method.} The mean of [f] over a Sobol sequence (Joe and Kuo's direction
      numbers) under 16 independent random digital shifts drawn from [key]. It
      adds the sequence in chunks of 64 points and tests at each power of two,
      where a Sobol prefix is balanced. Points are [(i + ½) / 2^k] after the
      shift, [k] the bits the dtype holds below 1 (32 in float64), so none lies
      on the box's boundary. {b Error.} [e] is the standard error of the mean
      over the shifts, an estimate of a standard deviation: the test is
      statistical. [y] is the integral. Each estimate at a fixed point count is
      unbiased, and the stopped one to within its standard error. [budget]
      chunks end a lane [Budget_spent]. {b Cost.} 1024 points per chunk, 64
      under each of 16 shifts, [budget] chunks at most, and the answer evaluates
      the final points again. {b Derivative.} The mean's over the final points:
      an estimate of the integral's derivative where the integrand is Lipschitz
      in the parameter.

      Raises [Invalid_argument] if [d] is above 1111, if [budget < 1], or if
      [f]'s result is not the points' shape without its last axis. *)
end

module Piecewise : sig
  (** Piecewise Chebyshev series.

      One value serves every approximation of a function of one variable:
      splines and other interpolants of samples, fits of a function, and their
      derivatives and integrals. On piece [i], between the breaks [x_i] and
      [x_(i+1)], the value is [Σ_k c_k T_k(u)] with
      [u = 2 (x − x_i) / (x_(i+1) − x_i) − 1] in [[−1, 1]]. A value is plain
      tensors, so it is an argument, a result and a carry of every
      transformation, and closed under evaluation, differentiation and
      integration.

      {[
      let spline = Piecewise.cubic `Natural knots samples
      let slope = Piecewise.eval (Piecewise.derivative spline) x
      ]}

      {b Domain.} The domain is the closed interval from the first break to the
      last. A point on a break lies in the last piece of positive width that
      ends there, at [u = 1], and a point on the first break in the first piece.
      NaN evaluates to NaN. An infinite point raises [Invalid_argument] through
      {!Nx.check}, and so does any other point outside the domain unless the
      value is extended ({!extend}).

      {b Cost.} Evaluation is a binary search of the breaks
      ({!Nx.searchsorted}), a gather of [degree + 1] coefficients and Clenshaw's
      recurrence: [O(log pieces + degree)] per point. Every operation here
      compiles to a fixed graph of tensor operations, fits included; {!adapt} is
      a stopping loop, which a compiled call waits on once per trip.

      {b Derivative.} Every function is a composition of tensor operations, so
      it differentiates in the coefficients, the breaks, the samples and the
      points; the piece a point falls in carries no derivative. {!adapt}'s
      derivative is its final partition's interpolant's, and zero when it did
      not converge. *)

  type ('v, 'b) t
  (** The type for piecewise series with values of structure ['v] over breaks of
      dtype ['b]: each float leaf of the coefficients has shape
      [[pieces; degree + 1] @ value], and evaluation at points of shape [q]
      gives it shape [q @ value]. Leaves may differ in degree and dtype.

      It has [pieces + 1] breaks, non-decreasing, the first below the last. A
      piece of zero width is empty: no point lies in it, and the end pieces are
      the first and last of positive width. *)

  val v : 'v Nx.Ptree.t -> breaks:(float, 'b) Nx.t -> 'v -> ('v, 'b) t
  (** [v s ~breaks c] is the series with [pieces + 1] [breaks] and the
      coefficients [c], each leaf of shape [[pieces; degree + 1] @ value].

      Raises [Invalid_argument] if [breaks] is not 1-D with at least two
      elements, if a leaf of [c] is not a float tensor of at least two axes with
      [pieces] rows, or, through {!Nx.check}, if [breaks] decreases or its first
      equals its last. *)

  (** {1:interpolants Interpolants}

      Each takes knots [x] of shape [[n]], [n ≥ 2], strictly increasing, and
      samples of shape [[n] @ value], and passes through every sample. Each
      raises [Invalid_argument] if [x] is not 1-D with at least two knots, if
      the samples do not have [n] rows, or, through {!Nx.check}, if [x] is not
      strictly increasing. *)

  type 'b ends =
    [ `Natural | `Not_a_knot | `Clamped of (float, 'b) Nx.t * (float, 'b) Nx.t ]
  (** The type for a cubic spline's end conditions, which change it near the
      ends:
      - [`Natural]: zero second derivative at both ends;
      - [`Not_a_knot]: a continuous third derivative at the second and the
        second-last knots, so the first two and last two pieces are each one
        cubic; through three knots, the parabola;
      - [`Clamped (s0, s1)]: the slopes [s0] and [s1], of shape [value], at the
        ends. *)

  val linear : (float, 'b) Nx.t -> (float, 'b) Nx.t -> ((float, 'b) Nx.t, 'b) t
  (** [linear x y] is the broken line through the samples: degree 1. *)

  val cubic :
    'b ends -> (float, 'b) Nx.t -> (float, 'b) Nx.t -> ((float, 'b) Nx.t, 'b) t
  (** [cubic ends x y] is the cubic spline through the samples, twice
      continuously differentiable: degree 3. Its system is solved in parallel by
      {!Nx.associative_scan}, stable because it is diagonally dominant.

      Raises [Invalid_argument] also if a clamped slope's shape is not [value].
  *)

  val steffen : (float, 'b) Nx.t -> (float, 'b) Nx.t -> ((float, 'b) Nx.t, 'b) t
  (** [steffen x y] is Steffen's (1990) interpolant, once continuously
      differentiable: degree 3. It has no extremum between knots that the data
      lack, so the interpolant of monotone samples is monotone. Its end slopes
      are the end intervals' secants. *)

  val hermite :
    (float, 'b) Nx.t ->
    values:(float, 'b) Nx.t ->
    slopes:(float, 'b) Nx.t ->
    ((float, 'b) Nx.t, 'b) t
  (** [hermite x ~values ~slopes] is the piecewise cubic with the given values
      and slopes at the knots: degree 3, once continuously differentiable.

      Raises [Invalid_argument] also if [slopes] has another shape than
      [values]. *)

  (** {1:fits Fits} *)

  val chebyshev :
    'v Nx.Ptree.t ->
    degree:int ->
    pieces:int ->
    ((float, 'b) Nx.t -> 'v) ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t ->
    ('v, 'b) t
  (** [chebyshev s ~degree ~pieces f a b] interpolates [f] on [pieces] equal
      pieces of [[a, b]] at each piece's [degree + 1] Chebyshev points of the
      second kind, ends included ([degree = 0] takes the midpoint). [f] receives
      points of shape [[pieces; degree + 1]] and returns each leaf of shape
      [[pieces; degree + 1] @ value]; [a] and [b] are scalars.

      {b Error.} For an [f] analytic in an ellipse around each piece, the error
      falls geometrically in [degree]; for one with [k] continuous derivatives,
      as [degree^(−k)]. {b Cost.} One call of [f] on all [pieces × (degree + 1)]
      points, and a constant matrix per leaf.

      Raises [Invalid_argument] if [degree < 0], [pieces < 1], if [a] or [b] is
      not a scalar, or if [f]'s leaves do not start with the points' shape. *)

  val adapt :
    'v Nx.Ptree.t ->
    degree:int ->
    tol:Tol.t ->
    budget:int ->
    ((float, 'b) Nx.t -> 'v) ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t ->
    ('v, 'b) t Solution.t
  (** [adapt s ~degree ~tol ~budget f a b] matches [f] on [[a, b]] to [tol] by
      series of [degree]. It solves one problem; {!Rune.val-vmap} gives each
      lane its own.

      {b Method.} Bisects the piece whose tail is largest until every piece
      meets [tol]. {b Error.} A piece's [e] is the larger of its last two
      coefficients in magnitude, and [y] its largest coefficient: the series'
      tail against its size. [budget] bounds the pieces, so the answer always
      holds [budget] pieces, the unused ones empty at [b] after the domain. A
      lane whose worst piece is at level 62, or holds no float strictly inside,
      ends [Stalled]; a non-finite coefficient ends it [Not_finite]; [budget]
      pieces end it [Budget_spent]. The error is a series of degree 0 on the
      answer's breaks, each piece's tail. {b Cost.} [degree + 1] evaluations of
      [f] per piece, and the answer evaluates [f] again at every piece's points.
      {b Derivative.} The final partition's interpolant's: its breaks are
      [a + (b − a) index / 2^level], so it reaches the ends.

      Raises [Invalid_argument] if [degree < 2], [budget < 1], if [a] or [b] is
      not a scalar, or if [f]'s leaves do not start with the points' shape. *)

  (** {1:eval Evaluation} *)

  val eval : ('v, 'b) t -> (float, 'b) Nx.t -> 'v
  (** [eval p x] is [p] at each point of [x], of shape [q]: each leaf of shape
      [q @ value].

      Raises [Invalid_argument] through {!Nx.check} if a point is infinite, or
      if a point that is not NaN lies outside the domain of a value that is not
      extended. *)

  val eval_at :
    ('v, 'b) t -> (int64, Nx.int64_elt) Nx.t -> (float, 'b) Nx.t -> 'v
  (** [eval_at p i u] is the series of piece [i] at the local coordinate [u] in
      [[−1, 1]], [i] and [u] of one shape [q]: each leaf of shape [q @ value].
      [u] outside [[−1, 1]] extrapolates the piece's series.

      Raises [Invalid_argument] if [i] and [u] differ in shape, and through
      {!Nx.check} if an index is not a piece's. *)

  (** {1:calculus Calculus}

      Each keeps the breaks and the extension: outside the domain the result
      extends as {!extend} says, from its own end pieces. Outside the domain,
      then, [eval (derivative p)] is not the derivative of [eval p]: the
      derivative of a held series holds its end slopes, and the integral of a
      held series holds its end values. *)

  val derivative : ('v, 'b) t -> ('v, 'b) t
  (** [derivative p] is the derivative of [p] in [x] on each piece, of degree
      one less (and [0] from degree [0]). At a break where [p]'s derivative
      jumps it takes the value of the piece that ends there. *)

  val integral : ('v, 'b) t -> ('v, 'b) t
  (** [integral p] is the antiderivative of [p] that is zero at the first break
      and continuous across breaks, of degree one more. *)

  val extend : [ `Hold | `Polynomial ] -> ('v, 'b) t -> ('v, 'b) t
  (** [extend e p] is [p] defined at every finite point: [`Hold] takes the value
      at the nearest end, and [`Polynomial] continues the end pieces' series. It
      replaces [p]'s extension. An infinite point still raises. *)

  (** {1:access Access} *)

  val breaks : ('v, 'b) t -> (float, 'b) Nx.t
  (** [breaks p] is [p]'s breaks, of shape [[pieces + 1]]. *)

  val coefficients : ('v, 'b) t -> 'v
  (** [coefficients p] is [p]'s coefficients, each leaf of shape
      [[pieces; degree + 1] @ value]. *)

  val ptree : 'v Nx.Ptree.t -> (float, 'b) Nx.dtype -> ('v, 'b) t Nx.Ptree.t
  (** [ptree s dtype] is the structure of series with values of structure [s]
      over breaks of [dtype]: the breaks at [breaks], the coefficients under
      [coefficients], and the extension reported at [extension] as one of the
      cases ["bounded"], ["hold"] and ["polynomial"]. *)
end

module Grid : sig
  (** Tensor-product series over grids.

      A grid value is the tensor product of {!Piecewise}'s representation: along
      each of its [d] axes, breaks and a Chebyshev series per piece, and in
      between, the series of their products. Its values have shape [value] at
      each point; a point is the last axis of a tensor, its [d] coordinates.

      {[
      let table = Grid.cubic `Not_a_knot ~axes:[ temperature; pressure ] density
      let rho = Grid.eval table states (* states of shape [n; 2] *)
      ]}

      The domain is the closed box between each axis's first and last break,
      with {!Piecewise}'s rules along each axis: a point on a break lies in the
      piece that ends there, or the first piece at the first break; NaN
      evaluates to NaN, and any other point outside raises [Invalid_argument]
      through {!Nx.check}.

      {b Cost.} Evaluation searches each axis's breaks, gathers
      [Π (degree_k + 1)] coefficients per point, and runs Clenshaw's recurrence
      along each axis in turn. {b Derivative.} The composition's, in the values,
      the axes and the points. *)

  type 'b t
  (** The type for grid series over breaks of dtype ['b]. The coefficients have
      shape [[pieces_1; …; pieces_d; degree_1 + 1; …; degree_d + 1] @ value]. *)

  (** {1:interpolants Interpolants}

      Each takes [d ≥ 1] axes of knots, each 1-D with at least two strictly
      increasing knots, and values of shape [[n_1; …; n_d] @ value], [n_k] the
      knots of axis [k], and passes through every value. Each raises
      [Invalid_argument] if an axis is not 1-D with two knots, if the values'
      leading axes do not match the axes, or, through {!Nx.check}, if an axis is
      not strictly increasing. *)

  val linear : axes:(float, 'b) Nx.t list -> (float, 'b) Nx.t -> 'b t
  (** [linear ~axes values] is the multilinear interpolant: degree 1 along each
      axis. *)

  val cubic :
    [ `Natural | `Not_a_knot ] ->
    axes:(float, 'b) Nx.t list ->
    (float, 'b) Nx.t ->
    'b t
  (** [cubic ends ~axes values] is the tensor-product cubic spline, with
      {!Piecewise.ends}' [ends] along every axis: degree 3 along each axis. *)

  (** {1:fits Fits} *)

  val chebyshev :
    degree:int ->
    pieces:int ->
    ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
    lo:(float, 'b) Nx.t ->
    hi:(float, 'b) Nx.t ->
    'b t
  (** [chebyshev ~degree ~pieces f ~lo ~hi] interpolates [f] on the box from
      [lo] to [hi], both of shape [[d]], split into [pieces] equal pieces along
      each axis, at the tensor product of each piece's [degree + 1] Chebyshev
      points of the second kind. [f] receives points of shape [q @ [d]] and
      returns values of shape [q @ value]. It costs one call of [f] on
      [(pieces × (degree + 1))^d] points.

      Raises [Invalid_argument] if [degree < 0], [pieces < 1], if [lo] and [hi]
      are not of one shape [[d]] with [d ≥ 1], or if [f]'s result does not start
      with the points' shape without their last axis. *)

  (** {1:eval Evaluation} *)

  val eval : 'b t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
  (** [eval g x] is [g] at the points [x] of shape [q @ [d]]: of shape
      [q @ value].

      Raises [Invalid_argument] if [x]'s last axis is not [d], and through
      {!Nx.check} if a point that is not NaN lies outside the domain. *)

  (** {1:calculus Calculus} *)

  val derivative : axis:int -> 'b t -> 'b t
  (** [derivative ~axis g] is the partial derivative of [g] along [axis], of one
      degree less along it.

      Raises [Invalid_argument] if [axis] is negative or not below [d]. *)

  val integral : axis:int -> 'b t -> 'b t
  (** [integral ~axis g] is the antiderivative of [g] along [axis], zero at that
      axis's first break, of one degree more along it.

      Raises [Invalid_argument] if [axis] is negative or not below [d]. *)

  (** {1:access Access} *)

  val ptree : (float, 'b) Nx.dtype -> 'b t Nx.Ptree.t
  (** [ptree dtype] is the structure of grid values over breaks of [dtype]: the
      breaks of each axis as a list at [breaks], and the coefficients at
      [coefficients]. *)
end

module Ode : sig
  (** Ordinary differential equations.

      A problem is a field [f t y], the derivative of the state [y] at the time
      [t], and an initial state. A state is any structure of tensors: its float
      leaves are the state's vector and share its steps, and its other leaves
      are carried unchanged. Times are tensors of their own float dtype ['t],
      which may differ from the leaves'. {!Rune.val-vmap} gives each lane its
      own problem.

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
  (** [tsit5] is Tsitouras' (2011) method of order 5 with an embedded order 4:
      six evaluations per step, its last stage the next step's first. *)

  val dopri5 : ([ `Formula | `Embedded ], 'y, 't) t
  (** [dopri5] is Dormand and Prince's (1980) method of order 5 with an embedded
      order 4: six evaluations per step, its last stage the next step's first.
  *)

  val tableau :
    a:float array array ->
    b:float array ->
    c:float array ->
    ([ `Formula ], 'y, 't) t
  (** [tableau ~a ~b ~c] is the explicit Runge–Kutta method of Butcher tableau
      [(a, b, c)] with [s] stages: stage [i] evaluates the field at
      [t + c.(i) h] on [y + h Σ_(j<i) a.(i).(j) k_j], and the step is
      [y + h Σ_i b.(i) k_i]. Its coefficients are given in float64 and rounded
      once to the time's and each leaf's dtype.

      Raises [Invalid_argument] unless [s ≥ 1], [b] and [c] have [s] elements,
      [a] has [s] rows of which row [i] has [i] elements, every coefficient is
      finite, and [b] sums to [1] within rounding. *)

  val kvaerno5 :
    ?mass:('y -> 'y) ->
    linear:'y Linear.t ->
    ((float, 't) Nx.t -> 'y -> 'y -> 'y) ->
    ([ `Embedded ], 'y, 't) t
  (** [kvaerno5 ~mass ~linear derivative] is Kværnø's (2004) singly diagonally
      implicit method of order 5 with an embedded order 4, for stiff fields:
      [derivative t y dy] is the field's Jacobian at [(t, y)] applied to [dy],
      such as [snd (Rune.jvp' (f t) y dy)]. [mass] is the linear map [M] of
      [M y' = f t y], the identity by default; a singular [M] makes the problem
      differential-algebraic.

      {b Method.} Seven stages, the first explicit and the others implicit with
      the diagonal [0.26]. Each stage is solved by simplified Newton on
      [M − 0.26 h J], [J] the Jacobian at the step's start, which [linear]
      prepares once per attempted step: {!Linear.dense} materialises and factors
      it once, and every stage and Newton iteration reuses the factors. A stage
      whose corrections do not shrink to [0.03] of [tol] in ten iterations, or
      whose linear solve fails, rejects the attempt. {b Error.} The step is the
      last stage and its error the difference from the sixth, both at [c = 1]:
      the embedded formula is stiffly accurate like the step, so the estimate is
      defined on algebraic components. {b Stability.} L-stable, so the step
      follows the tolerance on a stiff field instead of its fastest eigenvalue.
      A DAE is solved when its index is 1, when the equations in [M]'s left null
      space determine the variables in its null space; a higher index shows as a
      collapsing step and ends [Stalled]. [y0] must be consistent, [f t0 y0] in
      the range of [M]: from an inconsistent [y0] the first steps project onto
      the constraint. {b Cost.} Per attempt, one preparation of [linear] ([n]
      applications of [derivative] and a factorisation for {!Linear.dense}), and
      one evaluation of the field and one linear solve per Newton iteration.
      {b Derivative.} In the answer each accepted stage is stated as a zero of
      its stage equation through {!Rune.root}, solved by [linear], so the
      derivative is the accepted steps' with their stages' implicit derivatives.
  *)

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
  (** [march y m ~steps f ~at y0] is the state at each time of [at], stacked on
      a new leading axis of each leaf, [y0] first at [at.(0)]. Each interval of
      [at] takes [steps] equal steps of [m]; [at] may decrease, to march back.

      {b Error.} A method of order [p] has global error [O(h^p)] for a smooth
      field. {b Stability.} An explicit method is stable only while [h] times
      the field's eigenvalues lies in its stability region, so a stiff field
      needs a step far below its accuracy's. {b Cost.} The method's evaluations
      per step, once each step; a method whose last stage is the next step's
      first evaluates it once. {b Derivative.} The composition's: in the initial
      state, in the times of [at], and in every tracked value the field reads.
      Reverse mode keeps one state per time of [at] and recomputes each interval
      while it reverses it.

      Raises [Invalid_argument] if [steps < 1], if [at] is not a non-empty 1-D
      tensor, if it is not strictly monotone, or if [f] returns a value of
      another structure, dtype or shape than its state. *)

  (** {1:solves Solves}

      A solve chooses its steps to meet a tolerance: an embedded method
      estimates each step's local error, and the proportional–integral
      controller of Hairer, Nørsett and Wanner (I, §II.4) sizes the next. The
      search runs on detached values and records its accepted steps; the answer
      takes them again with the tracked field.

      {b Error.} [e] is one step's embedded error, and [y], per component, the
      larger of the step's two states; a step is accepted when [e] meets [tol].
      The solution's error is, per component, the sum of the magnitudes of the
      accepted steps' local estimates: it estimates the error the steps made,
      not the global error, which [tol] does not bound. An attempt with a
      non-finite stage is rejected; a non-finite field at an accepted state ends
      the lane [Not_finite], a step below the time's resolution [Stalled], and
      [budget] attempted steps [Budget_spent]. The last step of an interval
      lands on its end exactly. {b Stability.} An explicit method's controller
      keeps [h] inside its stability region, so on a stiff field its steps fall
      to that limit and the budget runs out first. {b Cost.} Each attempt costs
      the method's evaluations less one, and the answer evaluates the accepted
      steps again. Reverse mode keeps one state per time of [at] and, while it
      reverses an interval, its carries: compiled, [budget] of them; eagerly,
      the steps taken. {b Derivative.} The accepted steps', each a fraction [s]
      of its interval, [h = (b − a) s]: through the initial state, the times and
      every tracked value the field reads. A lane that did not converge returns
      its detached estimate, with a zero derivative. *)

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
      from [y0] at [t0], scalars; [t1] may equal or precede [t0]. At [t1 = t0]
      it is [y0], converged with a zero error.

      Raises [Invalid_argument] if [budget < 1], if [t0] or [t1] is not a
      scalar, or as {!march} does for a field of another structure. *)

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
      [budget] attempts across all of them. The steps land on every time of
      [at], so no state is interpolated. Its error is stacked like its value: at
      each time, per component, the sum of the magnitudes of the local estimates
      of the accepted steps before it. Times that are not strictly monotone end
      the lane [Stalled] before any step, and its report names the first.

      Raises [Invalid_argument] if [budget < 1], if [at] is not a non-empty 1-D
      tensor, and as {!march} does for a field of another structure. *)

  (** {1:paths Paths}

      Each embedded method has a continuous extension, a polynomial in the
      step's fraction that matches the step at both ends: {!tsit5}'s free
      interpolant of order 4 (Tsitouras, 2011), {!dopri5}'s of order 4 (Hairer,
      Nørsett and Wanner, I, §II.6), for {!bs3} and {!kvaerno5} the cubic
      Hermite interpolant of the step's ends and fields, of order 3, and for a
      {!kvaerno5} with a mass, whose fields are [M y'], the polynomial through
      its stage values at their fractions in [[0, 1]], of degree 4, defined on
      algebraic components too. Values between steps carry that extension's
      error, which [tol] does not control. *)

  val path :
    'y Nx.Ptree.t ->
    ([> `Embedded ], 'y, 't) t ->
    tol:Tol.t ->
    budget:int ->
    ('y, 't) field ->
    t0:'t time ->
    t1:'t time ->
    'y ->
    ('y, 't) Piecewise.t Solution.t
  (** [path y m ~tol ~budget f ~t0 ~t1 y0] is the solution from [y0] at [t0] to
      [t1] as a function of time: a series of [budget] pieces, one per accepted
      step in increasing time, each the step's continuous extension. The pieces
      past the last step are empty at its end, so the path's domain is the span
      between [t0] and [t1], whichever way the solve runs. At a step's end the
      path is that step's state, up to the rounding of its series.

      [t0 = t1] ends the lane [Stalled], since a path needs a piece of positive
      width.

      {b Error.} The error is a series of degree 0 on the same breaks: on each
      piece, per component, the sum of the magnitudes of the local estimates of
      the accepted steps through it. {b Stability.} As {!solve}'s. {b Cost.} A
      solve's search, then [budget] steps in the answer: the accepted ones taken
      again with the tracked field, and one of zero length in each slot past
      them, so a [budget] near the steps the solve takes keeps the answer's cost
      near the search's. Each piece costs [p + 1] combinations of the stages for
      an extension of order [p]. {b Derivative.} The accepted steps' and their
      extensions', through the initial state, [t0], [t1] and every tracked value
      the field reads; the breaks move with [t0] and [t1].

      Raises [Invalid_argument] if [budget < 1], if [t0] or [t1] is not a
      scalar, if a leaf of the state is not a float tensor, or as {!march} does
      for a field of another structure. *)

  (** {1:events Events} *)

  val event :
    'y Nx.Ptree.t ->
    ([> `Embedded ], 'y, 't) t ->
    tol:Tol.t ->
    budget:int ->
    ('y, 't) field ->
    event:('t time -> 'y -> (float, 'e) Nx.t) ->
    t0:'t time ->
    t1:'t time ->
    'y ->
    ('t time * 'y * (int32, Nx.int32_elt) Nx.t) Solution.t
  (** [event y m ~tol ~budget f ~event ~t0 ~t1 y0] is the solve from [y0] at
      [t0] toward [t1] that ends at the first crossing of an event: the time,
      the state there, and the flat index of the component of [event] that
      crossed. Each component of [event t y] is one event, so several events are
      one tensor.

      {b Method.} A component's sign at an accepted step's end is its last
      non-zero one, so a component that touches zero and turns back does not
      cross, and a zero at [t0] has no sign. The first accepted step across
      which a component's sign changes holds the crossing. In it, a bracketing
      search, {!Root.bracket}'s, finds where each such component takes its new
      sign on the step's continuous extension (see {!section-paths}) to the
      resolution of the time's dtype, which [tol] does not set, and the earliest
      wins. The time returned is the end of the final bracket past the crossing,
      where the component has its new sign, so a solve restarted there starts on
      that side, and the component's new sign tells the crossing's direction. A
      crossing that should not stop the solve, or one in a direction to ignore,
      is a restart from the returned time, one solve per crossing. To restart on
      the crossing's old side, as a bounce that reflects a velocity does, set
      the component to zero: a zero at [t0] is no crossing. A lane without a
      crossing after [t0] up to [t1] included converges with [(t1, y t1, −1)]:
      the index tells the two apart, whichever way the solve runs. At [t1 = t0]
      the answer is [(t0, y0, −1)], converged, so a compiled loop of restarts
      holds a finished lane where it stopped. A step holding an even number of
      crossings of a component shows none, so a crossing narrower than the
      field's steps can be missed or reported out of order: an event whose sign
      changes once per crossing, or {!path} and a finer search, resolves it.

      {b Error.} The time's is half the final bracket, about one unit in the
      last place of the time; the state's, per component, the sum of the
      magnitudes of the local estimates of the accepted steps through the one
      that holds the crossing; the index's is [0]. A crossing component whose
      bracket does not converge ends the lane [Stalled]. {b Cost.} A solve's
      search with one evaluation of [event] per attempted step, then the
      crossing's search, at most [2b + 1] iterations for a time of [b] bits,
      each of which evaluates [event] once per component; the answer takes the
      accepted steps again. {b Derivative.} The time is stated as a zero of the
      crossing component [R t = event t (y t)] on the tracked step, so its
      derivative is [−∂R/∂θ / ∂R/∂t], through the initial state, [t0] and every
      tracked value the field and [event] read; the state's is the extension's
      at that time. Without a crossing, the derivative is {!solve}'s.

      A compiled function returns the answer as a value of this structure, [s]
      the state's:

      {[
      Solution.ptree
        Nx.Ptree.(
          iso
            (fun (t, (y, i)) -> (t, y, i))
            (fun (t, y, i) -> (t, (y, i)))
            (pair tensor (pair s tensor)))
      ]}

      Raises [Invalid_argument] if [budget < 1], if [t0] or [t1] is not a
      scalar, if [event] has no component, or as {!march} does for a field of
      another structure. *)

  (** {1:delays Delays} *)

  val delay :
    'y Nx.Ptree.t ->
    ([> `Formula | `Embedded ], 'y, 't) t ->
    tol:Tol.t ->
    budget:int ->
    pieces:int ->
    ('t time -> 'y -> 'y -> 'y) ->
    lags:'t time ->
    history:('t time -> 'y) ->
    at:'t time ->
    'y ->
    'y Solution.t
  (** [delay y m ~tol ~budget ~pieces f ~lags ~history ~at y0] is the state at
      each time of [at], increasing, of the solution of
      [y' t = f t (y t) (y (t − τ))] for the constant lags [τ] of [lags], 1-D,
      from [y0] at [at.(0)], stacked as {!sample} stacks them. [f]'s third
      argument holds the delayed states stacked on a leading axis, one per lag;
      [history s] is the state at the scalar time [s ≤ at.(0)], which [delay]
      calls once per lag and stacks. [y0] may differ from [history at.(0)]: the
      solution then jumps at [at.(0)], and [f] reads the delayed state at
      [at.(0)] from [history] in the step that ends there and as [y0] in the
      step that starts there.

      {b Method.} An explicit embedded method, which the two tags together mark,
      whose steps never exceed the smallest lag, so every delayed state is read
      from an accepted step's continuous extension (see {!section-paths}) or
      from [history]. The steps land on the breakpoints [at.(0) + Σ_j k_j τ_j]
      with [1 ≤ Σ_j k_j ≤ p], [p] the method's order, where the solution's
      derivative of order [Σ_j k_j] can jump, or of order [1 + Σ_j k_j] when
      [y0 = history at.(0)]. The order of a delay solve is [min(p, q + 1)], [q]
      its extension's order (Bellen and Zennaro, 2003). {b Error.} As
      {!sample}'s. A lag that is not positive, or the largest lag reaching back
      past the last [pieces] accepted steps, ends the lane [Stalled], and its
      report names the count that would hold it. Times of [at] that do not
      increase end the lane [Stalled] too. {b Stability.} As {!solve}'s.
      {b Cost.} Each attempt costs the method's evaluations, its first stage
      among them, since the field may jump where one step ends and the next
      starts. Each stage reads [lags] delayed states, each a binary search of
      the [pieces] pieces and a series of the extension's degree; the carry
      holds [pieces] pieces, so a compiled reverse keeps [budget × pieces].
      {b Derivative.} The answer reads its own tracked pieces, so the derivative
      reaches the delayed states, [history], the lags and every tracked value
      [f] reads.

      Raises [Invalid_argument] if [budget < 1], if [pieces < 1], if [at] is not
      a non-empty 1-D tensor, if [lags] is not a non-empty 1-D tensor, if
      [history] returns a value of another structure, dtype or shape than [y0],
      or as {!march} does for a field of another structure. *)
end

module Sde : sig
  (** Stochastic differential equations.

      A problem is a drift [f t y], the deterministic part of the derivative; a
      diffusion, applied to a Brownian increment as [diffusion t y dw] and
      linear in [dw]; and a Brownian path. The method fixes the calculus: Itô or
      Stratonovich. *)

  (** Brownian paths with their space–time Lévy area.

      A path is a virtual tree over [[t0, t1]] (Foster, Lyons and Oberhauser,
      2020; Jelinčič et al., 2024): each query descends [depth] levels of
      bisections, each drawing the midpoint's increments and areas from their
      distribution given the interval's, under a key that is a pure function of
      the path's key and the node. So a path is one function of time, whatever
      the queries: marches with different steps see one path. Steps finer than
      [(t1 − t0) / 2^depth] see, inside each finest interval, the path's mean
      given that interval's increment and area, a quadratic, and lose their
      stated order. *)
  module Brownian : sig
    type 'b t
    (** The type for Brownian paths of dtype ['b]. *)

    val v :
      Nx.Rng.t ->
      (float, 'b) Nx.dtype ->
      shape:int array ->
      t0:float ->
      t1:float ->
      depth:int ->
      'b t
    (** [v key dtype ~shape ~t0 ~t1 ~depth] is a Brownian path of independent
        standard components of shape [shape] over [[t0, t1]], resolved to
        [(t1 − t0) / 2^depth].

        Raises [Invalid_argument] if [t0] or [t1] is not finite, if [t1 <= t0],
        if [depth] is not in [[0, 30]], or if a dimension of [shape] is
        negative. *)

    val increment :
      'b t ->
      (float, 'b) Nx.t ->
      (float, 'b) Nx.t ->
      (float, 'b) Nx.t * (float, 'b) Nx.t
    (** [increment w s t] is [W t − W s] and the space–time Lévy area over
        [[s, t]],
        [H = (1 / (t − s)) ∫_s^t (W r − W s − (r − s) / (t − s) (W t − W s)) dr],
        each of the path's shape; [H] is zero when [s = t]. [s] and [t] are
        scalars. Increments over adjacent intervals compose by Chen's relation.
        A query costs [2 depth] normal draws of the path's shape at each end; a
        derivative in a time through the path has no meaning.

        Raises [Invalid_argument] through {!Nx.check} if [s] or [t] lies outside
        [[t0, t1]]. *)

    val ptree : (float, 'b) Nx.dtype -> 'b t Nx.Ptree.t
    (** [ptree dtype] is the structure of paths of [dtype]: the key at [key],
        the shape's dimensions reported at [shape], the interval's ends at [t0]
        and [t1], and the depth reported at [depth]. A path draws from its key,
        so a compiled function takes it as an argument of this structure; a path
        it captures is a constant, whose draws it refuses ({!Rune.Jit_error}).
    *)
  end

  (** {1:methods Methods} *)

  type t
  (** The type for methods. *)

  val euler_maruyama : t
  (** [euler_maruyama] is the Euler–Maruyama method, Itô, for any noise: strong
      order 1/2. One drift and one diffusion evaluation per step. *)

  val milstein : t
  (** [milstein] is the derivative-free Milstein method (Kloeden and Platen,
      1992, §11.1), Itô: strong order 1 for diagonal noise, and 1/2 otherwise.
      Diagonal noise has the state one tensor of the path's shape, and
      [diffusion t y dw] the product of [dw] with a tensor whose component [i]
      depends on [y_i] only. One drift and three diffusion evaluations per step.
  *)

  val sra1 : t
  (** [sra1] is Rößler's (2010) SRA1, for additive noise, where the diffusion
      does not depend on the state: strong order 3/2, reading the Lévy area; 1/2
      otherwise. Two drift and two diffusion evaluations per step. *)

  val reversible_heun : t
  (** [reversible_heun] is the reversible Heun method (Kidger, Foster, Li and
      Lyons, 2021), Stratonovich: strong order 1/2, and 1 for additive noise. It
      carries a second state, so each step costs one drift and two diffusion
      evaluations. *)

  (** {1:marches Marches} *)

  val march :
    'y Nx.Ptree.t ->
    t ->
    steps:int ->
    drift:((float, 't) Nx.t -> 'y -> 'y) ->
    diffusion:((float, 't) Nx.t -> 'y -> (float, 't) Nx.t -> 'y) ->
    't Brownian.t ->
    at:(float, 't) Nx.t ->
    'y ->
    'y
  (** [march y m ~steps ~drift ~diffusion w ~at y0] is the state at each time of
      [at], stacked on a new leading axis of each leaf, [y0] first, along the
      path [w]. Each interval of [at] takes [steps] equal steps. [drift t y] is
      the deterministic field; [diffusion t y dw] is [g t y] applied to [dw], of
      [w]'s shape and the times' dtype, and must be linear in [dw]; it casts
      [dw] for leaves of another dtype. Reverse mode keeps one state per time of
      [at] and recomputes each interval while it reverses it; the derivative is
      the composition's, in the initial state and every tracked value the drift
      and diffusion read. A compiled function takes [w] as an argument of
      {!Brownian.ptree}'s structure.

      Raises [Invalid_argument] if [steps < 1], if [at] is not a non-empty 1-D
      tensor, through {!Nx.check} if [at] is not strictly increasing or leaves
      [w]'s interval, and if the drift or the diffusion returns a value of
      another structure, dtype or shape than its state. *)
end

module Split : sig
  (** Splitting methods for separable Hamiltonians.

      A Hamiltonian [H (q, p) = T p + V q] is the sum of two parts whose flows
      the caller computes exactly: the {e kick}, which moves momenta by [−∇V]
      over a duration, and the {e drift}, which moves positions by [∇T]. A
      splitting composes them over a step [h] as
      [K(a₁h) D(b₁h) K(a₂h) … D(b_m h) K(a_(m+1) h)]. Every scheme here is
      palindromic, so each step is symmetric and the march time-reversible; with
      exact flows of a Hamiltonian it is symplectic, and its energy error stays
      bounded over long times.

      A flow may be any exact flow of its part: a Wisdom–Holman drift is a
      Kepler step.

      {b Error.} A scheme of order [p] has global error [O(h^p)] in the state.
      {b Cost.} One kick per element of [kick], one fewer once adjacent kicks
      merge in a march. {b Derivative.} A march is a composition of the flows,
      so its derivative is theirs. *)

  type t
  (** The type for splittings: the coefficients [a] of the kicks and [b] of the
      drifts. [a] has one more element than [b], each sums to [1], and the
      sequence is palindromic. *)

  val leapfrog : t
  (** [leapfrog] is [K(h/2) D(h) K(h/2)], Störmer–Verlet: order 2. *)

  val mclachlan : t
  (** [mclachlan] is [K(λh) D(h/2) K((1 − 2λ)h) D(h/2) K(λh)] with
      [λ = 0.1931833275037836], the two-stage scheme of least error (Omelyan,
      Mryglod and Folk, 2002): order 2. *)

  val yoshida4 : t
  (** [yoshida4] is Yoshida's (1990) composition of three leapfrogs: order 4. *)

  val yoshida6 : t
  (** [yoshida6] is Yoshida's (1990) solution A, seven leapfrogs: order 6. *)

  val yoshida8 : t
  (** [yoshida8] is Yoshida's (1990) solution D, fifteen leapfrogs: order 8. *)

  val v : kick:float array -> drift:float array -> t
  (** [v ~kick ~drift] is the splitting [K(kick.(0) h) D(drift.(0) h) …].

      Raises [Invalid_argument] unless every coefficient is finite, [drift] is
      not empty, [kick] has one more element than [drift], each sums to [1]
      within rounding, and the sequence is palindromic: each array equals its
      reverse. *)

  type ('s, 'b) flow = (float, 'b) Nx.t -> 's -> 's
  (** The type for flows: [flow h s] is the state [s] moved by its part over the
      duration [h]: a scalar, or in {!step} a tensor that broadcasts against the
      state's leaves. *)

  val step :
    t ->
    kick:('s, 'b) flow ->
    drift:('s, 'b) flow ->
    (float, 'b) Nx.t ->
    's ->
    's
  (** [step m ~kick ~drift h s] is [s] after one step [h] of [m]. A negative [h]
      steps back: [step m ~kick ~drift (−h) (step m ~kick ~drift h s)] is [s] up
      to rounding.

      [h] may hold one duration per batch of the state: an [h] of shape
      [[chains; 1]] against leaves of shape [[chains; d]] steps each chain by
      its own duration, as stepping each alone with its scalar would. The flows
      receive [h] times each coefficient, of [h]'s shape. *)

  val march :
    's Nx.Ptree.t ->
    t ->
    steps:int ->
    kick:('s, 'b) flow ->
    drift:('s, 'b) flow ->
    at:(float, 'b) Nx.t ->
    's ->
    's
  (** [march s m ~steps ~kick ~drift ~at s0] is the state at each time of [at],
      stacked on a new leading axis of each leaf, [s0] first at [at.(0)]. Each
      interval of [at] takes [steps] equal steps, its adjacent kicks merged into
      one: a {!leapfrog} interval makes [steps + 1] kicks. Reverse mode keeps
      one state per time of [at] and recomputes each interval while it reverses
      it.

      Raises [Invalid_argument] if [steps < 1], if [at] is not a non-empty 1-D
      tensor, or if [at] is not strictly monotone. *)
end
