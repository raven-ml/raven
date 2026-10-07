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

(* [transition u lp k s] is one transition of every chain from [s], and what it
   tells warmup. *)
let transition (type f) u lp k (s : (_, f) state) =
  let t =
    H.transition "Norn.Hmc.step" u lp k s.geometry ~step_size:s.step_size
      ~length:s.length s.position s.lp s.grad
  in
  let c = (Nx.shape s.lp).(0) in
  let stats =
    Stats.
      {
        lp = t.lp;
        acceptance = t.alpha;
        step_size = Nx.broadcast_to [| c |] s.step_size;
        n_steps = t.steps;
        diverging = t.diverging;
        saturated = Nx.zeros Nx.bool [| c |];
        energy = t.energy;
      }
  in
  ( {
      s with
      position = t.position;
      lp = t.lp;
      grad = t.grad;
      stats;
      draw = Nx.add s.draw (Nx.scalar Nx.int32 1l);
    },
    t )

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

type ('u, 'f) adapt = {
  st : ('u, 'f) state;
  averaging : 'f Adapt.averaging;
  chees : 'f H.chees;
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
      let chees = field c "chees" (structure (H.chees_ptree ())) a.chees in
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
  let f = Gaussian.flat u s.position s.geometry in
  H.search "Norn.Hmc.warmup" P.tensor
    (H.rows_density u lp f s.position)
    ~reduce:log_harmonic_mean (H.chain_keys k c)
    (H.enter u f s.position s.lp s.grad)
    s.step_size

let warmup (type f) u lp k ~steps (s : (_, f) state) =
  if steps < 0 then invalid_argf "Norn.Hmc.warmup: steps = %d is negative" steps;
  let schedule = Adapt.schedule steps in
  if schedule = [] then s
  else
    let dt = Nx.dtype s.lp in
    let rank = Gaussian.rank s.geometry in
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
      let chees = H.chees_step a.chees step_size (H.chees_gradient move) in
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
