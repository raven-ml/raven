(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree
module H = Hamiltonian

let invalid_argf = Rows.invalid_argf

type ('u, 'f) state = {
  position : 'u;
  lp : (float, 'f) Nx.t;
  grad : 'u;
  step_size : (float, 'f) Nx.t;
  length : (float, 'f) Nx.t;
  geometry : ('u, 'f) Gaussian.t;
  stats : 'f Stats.t;
  draw : Nx.int32_t;
  accept : (float, 'f) Nx.t;
}

type ('u, 'f) hmc = ('u, 'f) state
type 'f stats = 'f Stats.t

let ptree (type u f) (u : u P.t) : (u, f) state P.t =
  let g = Gaussian.ptree u in
  let module S = struct
    type _ t = (u, f) hmc

    let walk c s =
      let open P.Walk in
      let position = field c "position" (structure u) s.position in
      let lp = field c "lp" tensor s.lp in
      let grad = field c "grad" (structure u) s.grad in
      let step_size = field c "step_size" tensor s.step_size in
      let length = field c "length" tensor s.length in
      let geometry = field c "geometry" (structure g) s.geometry in
      let stats =
        field c "stats" (structure (Stats.ptree (Nx.dtype s.lp))) s.stats
      in
      let draw = field c "draw" tensor s.draw in
      let accept = field c "accept" tensor s.accept in
      { position; lp; grad; step_size; length; geometry; stats; draw; accept }
  end in
  P.nest (module S) P.unit

let stats = Stats.ptree

(* The longest trajectory, in leapfrog steps. *)
let max_steps = 1024

(* The shared geometry

   One Gaussian serves every chain: the maps over chains capture it. *)

let color u g z = Rune.vmap P.(u @-> returns u) (Geometry.color u g) z
let whiten u g x = Rune.vmap P.(u @-> returns u) (Geometry.whiten u g) x

(* [to_whitened u g z gx] is the gradient in whitened coordinates of a gradient
   [gx] at [color z]: [color]'s transpose applied to it; [to_original] is
   [whiten]'s. *)
let to_whitened u g z gx =
  Rune.vmap
    P.(u @-> u @-> returns u)
    (fun z gx -> snd (Rune.vjp u u (Geometry.color u g) z) gx)
    z gx

let to_original u g x gz =
  Rune.vmap
    P.(u @-> u @-> returns u)
    (fun x gz -> snd (Rune.vjp u u (Geometry.whiten u g) x) gz)
    x gz

(* Keys

   Transition [n] of a run has the key [k = fold_in run n]. Chain [i] has row
   [i] of [split_batch (fold_in k 0)]: its momentum draws from [fold_in key 0],
   its acceptance uniform from [fold_in key 1], and a step-size search's trial
   [j] from [fold_in (fold_in key 2) j]. The chains share the jitter of the
   trajectory's length, drawn from [fold_in k 1]. *)

let chain_keys k c = Nx.Rng.split_batch ~n:c (Nx.Rng.fold_in k 0)

(* A trajectory *)

type ('u, 'f) trip = {
  at : ('u, 'f) H.point;
  running : Nx.bool_t;
  diverging : Nx.bool_t;
  steps : Nx.int32_t; (* each chain's *)
  trip : Nx.int32_t; (* the shared count *)
}

type ('u, 'f) trip' = ('u, 'f) trip

let trip_ptree (type u f) (u : u P.t) : (u, f) trip P.t =
  let point = H.point_ptree u in
  let module S = struct
    type _ t = (u, f) trip'

    let walk c (t : (u, f) trip) : (u, f) trip =
      let open P.Walk in
      let at = field c "at" (structure point) t.at in
      let running = field c "running" tensor t.running in
      let diverging = field c "diverging" tensor t.diverging in
      let steps = field c "steps" tensor t.steps in
      let trip = field c "trip" tensor t.trip in
      { at; running; diverging; steps; trip }
  end in
  P.nest (module S) P.unit

(* What a transition tells warmup: each chain's whitened start, its proposal and
   the proposal's momentum, its acceptance probability, and the trajectory's
   duration. *)
type ('u, 'f) move = {
  z0 : 'u;
  z1 : 'u;
  p1 : 'u;
  alpha : (float, 'f) Nx.t;
  time : (float, 'f) Nx.t;
}

(* [transition u lp k s] is one transition of every chain from [s]: [n] leapfrog
   steps of the shared step size, [n] the jittered length over it, then a
   Metropolis choice between the trajectory's end and its start. A chain whose
   step leaves the reals or whose energy error exceeds the limit stops there and
   is rejected. *)
let transition (type f) u lp k (s : (_, f) state) =
  let eps = s.step_size in
  let dt = Nx.dtype eps in
  let c = (Nx.shape s.lp).(0) in
  let i32 v = Nx.scalar Nx.int32 (Int32.of_int v) in
  let g = s.geometry in
  let keys = chain_keys k c in
  let lp_z z = lp (color u g z) in
  let z0 = whiten u g s.position in
  let g0 = to_whitened u g z0 s.grad in
  let p0 = H.momentum u keys z0 in
  let h = Nx.broadcast_to [| c |] eps in
  let h0 = Nx.sub (H.kinetic u h p0) s.lp in
  let jitter = Nx.mul_s (Nx.Rng.uniform (Nx.Rng.fold_in k 1) dt [||]) 2. in
  let n =
    Nx.clamp ~min:(Nx_dtype.of_float dt 1.)
      ~max:(Nx_dtype.of_float dt (float_of_int max_steps))
      (Nx.ceil (Nx.div (Nx.mul jitter s.length) eps))
  in
  let n = Nx.cast Nx.int32 n in
  let no = Nx.zeros Nx.bool [| c |] in
  let energy_error (at : (_, f) H.point) =
    let d = Nx.sub (Nx.sub (H.kinetic u h at.p) at.lp) h0 in
    Nx.where (Nx.isnan d) (Nx.scalar dt Float.infinity) d
  in
  let step t =
    let leaf, finite = H.leapfrog "Norn.Hmc.step" u lp_z t.running h t.at in
    let diverged =
      Nx.logical_and t.running
        (Nx.logical_or (Nx.logical_not finite)
           (Nx.greater (energy_error leaf) (Nx.scalar dt H.max_energy_error)))
    in
    let moved = Nx.logical_and t.running (Nx.logical_not diverged) in
    {
      at = H.choose_point u moved leaf t.at;
      running = moved;
      diverging = Nx.logical_or t.diverging diverged;
      steps = Nx.add t.steps (Nx.cast Nx.int32 t.running);
      trip = Nx.add t.trip (i32 1);
    }
  in
  let start = { H.z = z0; p = p0; g = g0; lp = s.lp } in
  let t =
    Rune.iterate (trip_ptree u) ~max:max_steps
      ~until:(fun t ->
        Nx.logical_or
          (Nx.greater_equal t.trip n)
          (Nx.logical_not (Nx.any t.running)))
      ~f:step
      {
        at = start;
        running = Nx.ones Nx.bool [| c |];
        diverging = no;
        steps = Nx.zeros Nx.int32 [| c |];
        trip = i32 0;
      }
  in
  let alpha =
    Nx.where t.diverging (Nx.zeros dt [| c |])
      (Nx.minimum (Nx.exp (Nx.neg (energy_error t.at))) (Nx.ones dt [| c |]))
  in
  let uniform =
    Rune.vmap
      P.(Nx.Rng.ptree @-> returns tensor)
      (fun k -> Nx.Rng.uniform (Nx.Rng.fold_in k 1) dt [||])
      keys
  in
  let accepted = Nx.less uniform alpha in
  let x1 = color u g t.at.z in
  let g1 = to_original u g x1 t.at.g in
  let position = Rows.choose u accepted x1 s.position in
  let grad = Rows.choose u accepted g1 s.grad in
  let lp1 = Nx.where accepted t.at.lp s.lp in
  let stats =
    Stats.
      {
        lp = lp1;
        acceptance = alpha;
        step_size = h;
        n_steps = t.steps;
        diverging = t.diverging;
        saturated = no;
        energy = h0;
      }
  in
  let move =
    { z0; z1 = t.at.z; p1 = t.at.p; alpha; time = Nx.mul (Nx.cast dt n) eps }
  in
  ( { s with position; lp = lp1; grad; stats; draw = Nx.add s.draw (i32 1) },
    move )

(* Starting *)

let init u ?(accept = 0.8) ?(rank = 2) ?geometry lp position =
  let context = "Norn.Hmc.init" in
  if not (accept > 0. && accept < 1.) then
    invalid_argf "%s: accept = %g is not in (0, 1)" context accept;
  if rank < 0 then invalid_argf "%s: rank = %d is negative" context rank;
  let c = Rows.count context u position in
  let l, g = Rows.evaluate u lp position in
  Rows.check_density context u lp position l;
  let dt = Nx.dtype l in
  (* Orthonormal directions number at most a chain's float elements. *)
  let rank = min rank (Rows.elements u c position) in
  let geometry =
    match geometry with
    | Some g -> g
    | None ->
        (* The chains' mean, and a variance per element that is the inverse of
           the gradient's root mean square over chains. *)
        let mean = P.map u (fun _ x -> Nx.mean ~axes:[ 0 ] x) position in
        let scale =
          P.map u
            (fun _ g ->
              let rms = Nx.sqrt (Nx.mean ~axes:[ 0 ] (Nx.square g)) in
              Nx.sqrt (Adapt.clip (Nx.recip rms)))
            g
        in
        if rank = 0 then Gaussian.diagonal u dt ~mean ~scale
        else
          let directions =
            P.map u
              (fun _ t ->
                Nx.zeros (Nx.dtype t) (Array.append [| rank |] (Nx.shape t)))
              mean
          in
          Gaussian.low_rank u ~mean ~scale ~directions
            ~variances:(Nx.ones dt [| rank |])
  in
  let zero = Nx.zeros dt [| c |] and no = Nx.zeros Nx.bool [| c |] in
  {
    position;
    lp = l;
    grad = g;
    step_size = Nx.scalar dt 1.;
    length = Nx.scalar dt 1.;
    geometry;
    stats =
      Stats.
        {
          lp = l;
          acceptance = zero;
          step_size = Nx.ones dt [| c |];
          n_steps = Nx.zeros Nx.int32 [| c |];
          diverging = no;
          saturated = no;
          energy = zero;
        };
    draw = Nx.scalar Nx.int32 0l;
    accept = Nx.scalar dt accept;
  }

(* Transitions *)

let step u lp k s = fst (transition u lp k s)

let sample u lp k ~draws (s : (_, _) state) =
  if draws < 1 then
    invalid_argf "Norn.Hmc.sample: draws = %d is not positive" draws;
  let dt = Nx.dtype s.lp in
  let sp = ptree u and st = Stats.ptree dt in
  let s, (xs, ss) =
    Rune.scan sp P.tensor
      P.(pair u st)
      ~f:(fun (s : (_, _) state) _ ->
        let s = step u lp (Nx.Rng.fold_in_tensor k s.draw) s in
        (s, (s.position, s.stats)))
      ~init:s
      (Nx.zeros Nx.int32 [| draws |])
  in
  let swap x = Nx.moveaxis 0 1 x in
  let xs = P.map u (fun _ t -> swap t) xs in
  let ss = P.map st (fun _ t -> swap t) ss in
  (s, Draws.v u xs, Draws.v st ss)

(* Warmup

   The step size is dual-averaged toward an acceptance [accept] of the chains'
   harmonic mean, which a chain that never accepts drives to zero, so the step
   shrinks until every chain moves. The length maximises the change in expected
   squared jumped distance (ChEES; Hoffman, Radul and Sountsov 2021): each
   transition estimates the criterion's gradient in log length from the chains,
   in whitened coordinates, weighted by acceptance; Adam steps it, and the run
   keeps the iterates' polynomial average. *)

(* Adam's step and second-moment decay, without momentum. *)
let adam_rate = 0.025
let adam_decay = 0.95
let adam_eps = 1e-8

(* The weight [t^(-κ)] of iterate [t] in the length's average. *)
let average_kappa = 0.75

type 'f chees = {
  log_length : (float, 'f) Nx.t;
  second : (float, 'f) Nx.t; (* Adam's second moment *)
  bar : (float, 'f) Nx.t; (* the averaged log length *)
  count : (float, 'f) Nx.t;
}

type 'f chees' = 'f chees

let chees_ptree (type f) () : f chees P.t =
  let module S = struct
    type _ t = f chees'

    let walk c (a : f chees) : f chees =
      let open P.Walk in
      let log_length = field c "log_length" tensor a.log_length in
      let second = field c "second" tensor a.second in
      let bar = field c "bar" tensor a.bar in
      let count = field c "count" tensor a.count in
      { log_length; second; bar; count }
  end in
  P.nest (module S) P.unit

(* [centre u w x] is [x] less its mean over chains weighted by [w]. *)
let centre u w x =
  P.map u
    (fun _ t ->
      if not (Rows.float_leaf t) then t
      else
        let w = Rows.column w t in
        let mean =
          Nx.div
            (Nx.sum ~axes:[ 0 ] ~keepdims:true (Nx.mul w t))
            (Nx.sum ~axes:[ 0 ] ~keepdims:true w)
        in
        Nx.sub t mean)
    x

(* [gradient u m] is the acceptance-weighted mean over chains of the ChEES
   criterion's derivative in log length, [t d ((z1 - mean z1) · p1)] with [d =
   |z1 - mean z1|² - |z0 - mean z0|²], the means over chains, [z1]'s weighted by
   acceptance; zero when no chain accepts. *)
let gradient u (m : (_, _) move) =
  let w = m.alpha in
  let total = Nx.sum w in
  let none = Nx.equal total (Nx.zeros_like total) in
  let w = Nx.where none (Nx.zeros_like w) w in
  let weights = Nx.where none (Nx.ones_like w) w in
  let d0 = centre u (Nx.ones_like w) m.z0 and d1 = centre u weights m.z1 in
  let d = Nx.sub (Rows.dot u w d1 d1) (Rows.dot u w d0 d0) in
  let per_chain = Nx.mul (Nx.mul m.time d) (Rows.dot u w d1 m.p1) in
  Nx.div
    (Nx.sum (Nx.mul w per_chain))
    (Nx.where none (Nx.ones_like total) total)

(* One Adam step up the criterion, the log length kept where the trajectory fits
   [max_steps] steps of [eps]. *)
let chees_step (a : _ chees) eps grad =
  let count = Nx.add_s a.count 1. in
  let second =
    Nx.add
      (Nx.mul_s a.second adam_decay)
      (Nx.mul_s (Nx.square grad) (1. -. adam_decay))
  in
  let corrected =
    Nx.div second
      (Nx.rsub_s 1. (Nx.pow (Nx.scalar_like count adam_decay) count))
  in
  let step = Nx.div grad (Nx.add_s (Nx.sqrt corrected) adam_eps) in
  let log_length =
    Nx.minimum
      (Nx.add a.log_length (Nx.mul_s step adam_rate))
      (Nx.log (Nx.mul_s eps (float_of_int max_steps)))
  in
  let weight = Nx.pow count (Nx.scalar_like count (-.average_kappa)) in
  let bar =
    Nx.add (Nx.mul (Nx.rsub_s 1. weight) a.bar) (Nx.mul weight log_length)
  in
  { log_length; second; bar; count }

type ('u, 'f) adapt = {
  st : ('u, 'f) state;
  averaging : 'f Adapt.averaging;
  chees : 'f chees;
  window : 'u Adapt.window;
}

type ('u, 'f) adapt' = ('u, 'f) adapt

let adapt_ptree (type u f) (u : u P.t) : (u, f) adapt P.t =
  let sp = ptree u in
  let module S = struct
    type _ t = (u, f) adapt'

    let walk c (a : (u, f) adapt) : (u, f) adapt =
      let open P.Walk in
      let st = field c "st" (structure sp) a.st in
      let averaging =
        field c "averaging" (structure (Adapt.averaging_ptree ())) a.averaging
      in
      let chees = field c "chees" (structure (chees_ptree ())) a.chees in
      let window =
        field c "window" (structure (Adapt.window_ptree u)) a.window
      in
      { st; averaging; chees; window }
  end in
  P.nest (module S) P.unit

(* [log_harmonic_mean d] is the log of the chains' harmonic mean acceptance of
   log acceptance ratios [d]. *)
let log_harmonic_mean d =
  let a = Nx.minimum (Nx.exp d) (Nx.ones_like d) in
  let c = float_of_int (Nx.shape d).(0) in
  Nx.log (Nx.div (Nx.scalar_like d c) (Nx.sum (Nx.recip a)))

let harmonic_mean a =
  let c = float_of_int (Nx.shape a).(0) in
  Nx.div (Nx.scalar_like a c) (Nx.sum (Nx.recip a))

(* [init_step_size u lp k s] is the step size doubled or halved, from
   [s.step_size], until one leapfrog step from every chain's position with a
   fresh momentum crosses a harmonic mean acceptance of 0.8. [k] is the key of
   the transition the search precedes. *)
let init_step_size u lp k (s : (_, _) state) =
  let c = (Nx.shape s.lp).(0) in
  let g = s.geometry in
  let z0 = whiten u g s.position in
  let start = { H.z = z0; p = z0; g = to_whitened u g z0 s.grad; lp = s.lp } in
  H.search "Norn.Hmc.warmup" u
    (fun z -> lp (color u g z))
    ~reduce:log_harmonic_mean (chain_keys k c) start s.step_size

let warmup (type f) u lp k ~steps (s : (_, f) state) =
  if steps < 0 then invalid_argf "Norn.Hmc.warmup: steps = %d is negative" steps;
  let schedule = Adapt.schedule steps in
  if schedule = [] then s
  else
    let dt = Nx.dtype s.lp in
    let rank = Geometry.rank s.geometry in
    let longest = List.fold_left (fun m (n, _) -> max m n) 0 schedule in
    (* A low-rank fit reads the window's draws; a diagonal one their sums. *)
    let buffer = if rank = 0 then 0 else longest in
    let key (st : (_, _) state) = Nx.Rng.fold_in_tensor k st.draw in
    let s0 = { s with step_size = init_step_size u lp (key s) s } in
    let zero = Nx.zeros dt [||] in
    let a0 =
      {
        st = s0;
        averaging = Adapt.restart s0.step_size;
        chees =
          {
            log_length = Nx.log s0.length;
            second = zero;
            bar = Nx.log s0.length;
            count = zero;
          };
        window = Adapt.empty u ~buffer s0.position;
      }
    in
    let one (a : (_, f) adapt) =
      let st, move = transition u lp (key a.st) a.st in
      let averaging, step_size =
        Adapt.average a.averaging ~target:st.accept (harmonic_mean move.alpha)
      in
      let chees = chees_step a.chees step_size (gradient u move) in
      let st = { st with step_size; length = Nx.exp chees.log_length } in
      let window = Adapt.record u a.window st.position st.grad in
      { st; averaging; chees; window }
    in
    let window (a : (_, f) adapt) (length, refit) =
      let a = { a with window = Adapt.reopen u a.window a.st.position } in
      let a =
        Rune.iterate (adapt_ptree u) ~max:longest
          ~until:(fun a -> Nx.greater_equal a.window.n length)
          ~f:one a
      in
      (* At a slow window's end: refit the geometry to every chain's draws, find
         a step size for it and restart the averaging there. *)
      let refit = Nx.not_equal refit (Nx.scalar Nx.int32 0l) in
      let geometry = Adapt.pooled u dt ~rank a.window in
      let gp = Gaussian.ptree u in
      let refitted =
        {
          a.st with
          geometry =
            P.map2 gp (fun _ g o -> Nx.where refit g o) geometry a.st.geometry;
        }
      in
      let eps = init_step_size u lp (key refitted) refitted in
      let st =
        { refitted with step_size = Nx.where refit eps a.st.step_size }
      in
      let averaging =
        P.map2 (Adapt.averaging_ptree ())
          (fun _ r o -> Nx.where refit r o)
          (Adapt.restart st.step_size)
          a.averaging
      in
      ({ a with st; averaging }, ())
    in
    let a, () =
      Rune.scan (adapt_ptree u)
        P.(pair tensor tensor)
        P.unit ~f:window ~init:a0 (Adapt.windows schedule)
    in
    {
      a.st with
      step_size = Adapt.final a.averaging;
      length = Nx.exp a.chees.bar;
    }
