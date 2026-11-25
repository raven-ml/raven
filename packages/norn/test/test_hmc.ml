(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module H = Norn.Hmc

let t = Nx.Ptree.tensor
let dim = 2

(* A correlated, skewed target: a banana. Its exact draws are [a] standard
   normal and [b = 0.3 a² + z / 2]. *)
let banana x =
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
  Nx.mul_s (Nx.add (Nx.square a) (Nx.mul_s (Nx.square b') 4.)) (-0.5)

(* A standard normal over a batch of chains. *)
let normal x = Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.square x)) (-0.5)
let start c = Nx.Rng.normal ~key:(Nx.Rng.key 21) Nx.float64 [| c; dim |]
let floats x = Nx.to_array x
let row0 x = floats (Nx.slice [ Nx.I 0 ] x)

let draws (d : Nx.host Nx.float64_t Norn.Draws.t) =
  floats (d :> Nx.host Nx.float64_t)

(* A diagonal geometry and its color map [m + s z]. One narrower than the
   banana, so that a unit step is short, and one wider, so that trajectories
   diverge and are rejected. *)
type geometry = {
  g : (Nx.host Nx.float64_t, Nx.float64_elt, Nx.host) Norn.Gaussian.t;
  m : Nx.host Nx.float64_t;
  s : Nx.host Nx.float64_t;
}

let geometry_of scale =
  let m = Nx.create Nx.float64 [| dim |] [| 0.1; -0.2 |] in
  let s = Nx.create Nx.float64 [| dim |] scale in
  { g = Norn.Gaussian.diagonal t Nx.float64 ~mean:m ~scale:s; m; s }

let narrow = geometry_of [| 0.4; 0.2 |]
let wide = geometry_of [| 3.; 1.5 |]
let color geometry z = Nx.add geometry.m (Nx.mul geometry.s z)
let whiten geometry x = Nx.div (Nx.sub x geometry.m) geometry.s

(* [rank s] is the number of directions of the chains' geometry. *)
let rank (s : (_, _, _) H.state) =
  Nx.Ptree.fold (Norn.Gaussian.ptree t)
    (fun p x r ->
      if Nx.Ptree.Path.to_string p = "variances" then (Nx.shape x).(0) else r)
    s.geometry (-1)

(* [with_scalar name v s] is [s] with its scalar [name] at [v]. *)
let with_scalar name v s =
  Nx.Ptree.map (H.ptree t)
    (fun p x ->
      if Nx.Ptree.Path.to_string p <> name then x
      else Nx.cast (Nx.dtype x) (Nx.scalar Nx.float64 v))
    s

let with_step_size = with_scalar "step_size"
let with_length = with_scalar "length"

(* init *)

let init =
  group "init"
    [
      test "a density not batched over chains is refused" (fun () ->
          raises
            (Invalid_argument
               "Norn.Hmc.init: the density returned shape [4; 2] for a \
                position of 4 chains; a density returns one log density per \
                chain, shape [4]") (fun () ->
              H.init t (fun x -> Nx.neg (Nx.square x)) (start 4)));
      test "a density whose rows read each other is refused, naming the row"
        (fun () ->
          let coupled x =
            Nx.add (banana x) (Nx.slice [ Nx.L [ 0; 0; 0; 0 ]; Nx.I 0 ] x)
          in
          raises_match
            (Exn.invalid_arg ~sub:"Norn.Hmc.init: the density's row 0 changes")
            (fun () -> H.init t coupled (start 4)));
      test "a start where the density is -inf is refused, naming the chain"
        (fun () ->
          let walled x =
            let a = Nx.slice [ Nx.A; Nx.I 0 ] x in
            Nx.where (Nx.greater_s a 50.)
              (Nx.full_like a Float.neg_infinity)
              (banana x)
          in
          let x =
            Nx.set [ Nx.I 1; Nx.I 0 ] (Nx.scalar Nx.float64 100.) (start 4)
          in
          raises
            (Invalid_argument
               "Norn.Hmc.init: the density is -inf at chain 1; a chain starts \
                where its density is finite") (fun () -> H.init t walled x));
      test "a tensor that is not a float is refused, naming its path" (fun () ->
          let u = Nx.Ptree.(pair tensor tensor) in
          let x = (start 4, Nx.zeros Nx.int32 [| 4 |]) in
          raises_match (Exn.invalid_arg ~sub:"Norn.Hmc.init: 1 is int32")
            (fun () -> H.init u (fun (x, _) -> banana x) x));
      (* The message prints the shortest decimal that reads back, the same on
         every system: the C library's [%g] wrote 1e+06, 1e+006 under
         mingw-w64. *)
      cases "an acceptance outside (0, 1) is refused" ~id:Fun.id
        [ "0"; "1"; "1000000"; "-4.58e-72"; "nan" ] (fun text ->
          raises
            (Invalid_argument
               (Printf.sprintf "Norn.Hmc.init: accept = %s is not in (0, 1)"
                  text))
            (fun () -> H.init t ~accept:(float_of_string text) banana (start 2)));
      test "a negative rank is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Hmc.init: rank") (fun () ->
              H.init t ~rank:(-1) banana (start 2)));
      test "the defaults are an acceptance of 0.8, step and length 1, rank 2"
        (fun () ->
          let s = H.init t banana (start 2) in
          equal float_exact 0.8 (Nx.item [] s.accept);
          equal float_exact 1. (Nx.item [] s.step_size);
          equal float_exact 1. (Nx.item [] s.length);
          equal int 2 (rank s));
      test "a rank above a chain's float elements is that number" (fun () ->
          let s =
            H.init t ~rank:2 normal
              (Nx.create Nx.float64 [| 2; 1 |] [| 1.; 2. |])
          in
          equal int 1 (rank s));
      (* -x²/2 has gradient -x. Over chains (4, 0.5, 0) and (0, 2, 0) the mean
         is (2, 1.25, 0), and the gradient's root mean square sqrt 8, sqrt 2.125
         and 0: variance 1e20 where it is 0. *)
      test
        "the default geometry is centred on the chains' mean, of variance 1 / \
         rms grad clipped" (fun () ->
          let x =
            Nx.create Nx.float64 [| 2; 3 |] [| 4.; 0.5; 0.; 0.; 2.; 0. |]
          in
          let s = H.init t ~rank:0 normal x in
          equal (array float_exact) [| 2.; 1.25; 0. |]
            (floats (Norn.Gaussian.mean t s.geometry));
          equal
            (array (float_rel ~rel:1e-15 ~abs:0.))
            [| 1. /. Float.sqrt 8.; 1. /. Float.sqrt 2.125; 1e20 |]
            (floats (Norn.Gaussian.variance t s.geometry)));
    ]

(* step

   A transition from the .mli's description and its Keys paragraph, in whitened
   coordinates. *)

(* The banana in whitened coordinates and its gradient, at one chain. *)
let eval geometry z =
  let lp z = banana (Nx.unsqueeze ~axes:[ 0 ] (color geometry z)) in
  let l, g =
    Rune.value_and_grad'
      (fun z -> Nx.sum (lp z))
      (Nx.create Nx.float64 [| dim |] z)
  in
  (Nx.item [] l, floats g)

let dot a b =
  let s = ref 0. in
  Array.iteri (fun i x -> s := !s +. (x *. b.(i))) a;
  !s

let axpy h x y = Array.mapi (fun i y -> y +. (h *. x.(i))) y
let uniform key = Nx.item [] (Nx.Rng.uniform ~key Nx.float64 [||])

type reference = {
  z : float array;
  lp : float;
  h0 : float;
  steps : int;
  alpha : float;
  diverged : bool;
  accepted : bool;
}

(* [reference geometry ~eps ~n key z0] is chain [key]'s transition of [n]
   leapfrog steps of [eps] from [z0]. *)
let reference geometry ~eps ~n key z0 =
  let p0 =
    floats
      (Nx.Rng.normal
         ~key:(Nx.Rng.fold_in (Nx.Rng.fold_in key 0) 0)
         Nx.float64 [| dim |])
  in
  let lp0, g0 = eval geometry z0 in
  let h0 = (0.5 *. dot p0 p0) -. lp0 in
  let rec go i z p g lp =
    if i = n then (z, p, lp, i, false)
    else
      let p' = axpy (eps /. 2.) g p in
      let z' = axpy eps p' z in
      if not (Array.for_all Float.is_finite z') then (z, p, lp, i + 1, true)
      else
        let lp', g' = eval geometry z' in
        let p' = axpy (eps /. 2.) g' p' in
        let h = (0.5 *. dot p' p') -. lp' in
        let error = if Float.is_nan h then Float.infinity else h -. h0 in
        if error > 1000. then (z, p, lp, i + 1, true)
        else go (i + 1) z' p' g' lp'
  in
  let z, p, lp, steps, diverged = go 0 z0 p0 g0 lp0 in
  let alpha =
    if diverged then 0.
    else Float.min 1. (Float.exp (h0 -. ((0.5 *. dot p p) -. lp)))
  in
  let accepted = uniform (Nx.Rng.fold_in key 1) < alpha in
  {
    z = (if accepted then z else z0);
    lp = (if accepted then lp else lp0);
    h0;
    steps;
    alpha;
    diverged;
    accepted;
  }

(* The shared count: [ceil (u length / eps)], [u] twice the jitter's uniform,
   between 1 and 1024. *)
let leapfrogs key ~length ~eps =
  let u = 2. *. uniform (Nx.Rng.fold_in key 1) in
  Float.to_int
    (Float.min 1024. (Float.max 1. (Float.ceil (u *. length /. eps))))

let chains = 4

(* The 60 values of (narrow geometry, step size, key) are checked whole. *)
let transition =
  prop ~count:60 "a transition is n leapfrog steps and a Metropolis acceptance"
    Gen.(triple bool (int_range 0 2) (int_range 0 9))
  @@ fun (is_narrow, size, seed) ->
  let geometry = if is_narrow then narrow else wide in
  let eps = [| 0.1; 0.5; 1.5 |].(size) in
  let key = Nx.Rng.key (1000 + seed) in
  let x0 =
    Nx.Rng.normal ~key:(Nx.Rng.fold_in key 7) Nx.float64 [| chains; dim |]
  in
  let s = with_step_size eps (H.init t ~geometry:geometry.g banana x0) in
  let s' = H.step t banana key s in
  let st = s'.stats in
  let n = leapfrogs key ~length:1. ~eps in
  let keys = Nx.Rng.split ~n:chains (Nx.Rng.fold_in key 0) in
  let refs =
    Array.init chains (fun i ->
        reference geometry ~eps ~n keys.(i)
          (floats (whiten geometry (Nx.slice [ Nx.I i ] x0))))
  in
  let some f = Array.exists f refs in
  cover "a chain is rejected"
    (some (fun r -> (not r.accepted) && not r.diverged));
  cover "a chain accepts" (some (fun r -> r.accepted));
  cover "a chain diverges" (some (fun r -> r.diverged));
  cover "no chain diverges" (not (some (fun r -> r.diverged)));
  Array.iteri
    (fun i r ->
      let msg = Printf.sprintf "chain %d" i in
      let at f = Nx.item [ i ] f in
      let x = color geometry (Nx.create Nx.float64 [| dim |] r.z) in
      equal ~msg int r.steps (Int32.to_int (at st.n_steps));
      equal ~msg bool r.diverged (at st.diverging);
      equal ~msg bool false (at st.saturated);
      equal ~msg (float 1e-12) r.alpha (at st.acceptance);
      equal ~msg (float 1e-12) r.h0 (at st.energy);
      equal ~msg (float 1e-10) r.lp (at s'.lp);
      equal ~msg
        (array (float 1e-10))
        (floats x)
        (floats (Nx.slice [ Nx.I i ] s'.position)))
    refs

(* Invariance. After five transitions of every chain from the banana's exact
   draws, with fixed tuning, [a] and [b - 0.3 a²] keep their normal CDFs within
   the Dvoretzky-Kiefer-Wolfowitz band of level [0.01 / 4], four comparisons
   holding the false alarms at 1%. Given the shared jitters, each transition is
   a kernel of fixed length that leaves the banana invariant, and the chains are
   independent, so their draws are. At 4096 chains the band is 0.032 wide. *)
let invariance_chains = 4096

let band =
  Float.sqrt
    (Float.log (2. /. (0.01 /. 4.)) /. (2. *. float_of_int invariance_chains))

(* [max_deviation xs sd] is the largest distance between the ECDF of [xs] and
   the CDF of a normal of standard deviation [sd]. *)
let max_deviation xs sd =
  let xs = Array.copy xs in
  Array.sort Float.compare xs;
  let n = float_of_int (Array.length xs) in
  let worst = ref 0. in
  Array.iteri
    (fun i x ->
      let f = Nx.item [] (Nx.ndtr (Nx.scalar Nx.float64 (x /. sd))) in
      worst :=
        Float.max !worst
          (Float.max
             (Float.abs (f -. (float_of_int i /. n)))
             (Float.abs (f -. (float_of_int (i + 1) /. n)))))
    xs;
  !worst

let invariant geometry =
  let k = Nx.Rng.key 31 in
  let a =
    Nx.Rng.normal ~key:(Nx.Rng.fold_in k 0) Nx.float64 [| invariance_chains |]
  in
  let z =
    Nx.Rng.normal ~key:(Nx.Rng.fold_in k 1) Nx.float64 [| invariance_chains |]
  in
  let b = Nx.add (Nx.mul_s (Nx.square a) 0.3) (Nx.mul_s z 0.5) in
  let s = ref (H.init t ~geometry banana (Nx.stack ~axis:1 [ a; b ])) in
  for i = 0 to 4 do
    s := H.step t banana (Nx.Rng.fold_in k (10 + i)) !s
  done;
  let x = !s.position in
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
  at_most (float 1e-12) ~than:band (max_deviation (floats a) 1.);
  at_most (float 1e-12) ~than:band (max_deviation (floats b') 0.5)

let step =
  group "step"
    [
      transition;
      (* Length / step = 1e6: the count reaches the cap unless the jitter's
         uniform is below 5e-4, not so on this key. *)
      test "a transition takes at most 1024 steps" (fun () ->
          let s =
            H.init t ~geometry:narrow.g banana (start 2) |> with_step_size 1e-6
          in
          let s' = H.step t banana (Nx.Rng.key 3) s in
          equal (array int32) [| 1024l; 1024l |] (floats s'.stats.n_steps));
      test "a diverging chain is rejected and stays where it was" (fun () ->
          (* Beyond |x| = 1 the density is -inf: a long step from the origin
             leaves the support. *)
          let walled x =
            let r = Nx.sum ~axes:[ 1 ] (Nx.square x) in
            Nx.where (Nx.greater_s r 1.)
              (Nx.full Nx.float64 [| (Nx.shape x).(0) |] Float.neg_infinity)
              (Nx.mul_s r (-0.5))
          in
          let g =
            Norn.Gaussian.diagonal t Nx.float64
              ~mean:(Nx.zeros Nx.float64 [| dim |])
              ~scale:(Nx.full Nx.float64 [| dim |] 100.)
          in
          let x = Nx.zeros Nx.float64 [| 3; dim |] in
          let s' =
            H.step t walled (Nx.Rng.key 5) (H.init t ~geometry:g walled x)
          in
          equal (array bool) [| true; true; true |] (floats s'.stats.diverging);
          equal (array float_exact) [| 0.; 0.; 0. |]
            (floats s'.stats.acceptance);
          equal (array float_exact) (floats x) (floats s'.position));
      test "a chain's transition does not depend on the chain count" (fun () ->
          let one =
            H.step t banana (Nx.Rng.key 3)
              (H.init t ~geometry:narrow.g banana
                 (Nx.slice [ Nx.R (0, 1); Nx.R (0, dim) ] (start 4)))
          in
          let four =
            H.step t banana (Nx.Rng.key 3)
              (H.init t ~geometry:narrow.g banana (start 4))
          in
          equal (array float_exact) (row0 one.position) (row0 four.position));
      cases
        "a density NaN or +inf at a finite position is refused, naming the \
         chain"
        ~id:Float.to_string [ Float.nan; Float.infinity ] (fun v ->
          (* Chain 2 starts at x0 = 100. Past x0 = 50 the density is [v] off its
             start; the others never come near. *)
          let x =
            Nx.set [ Nx.I 2; Nx.I 0 ] (Nx.scalar Nx.float64 100.) (start 4)
          in
          let x2 = Nx.slice [ Nx.I 2 ] x in
          let lp x =
            let a = Nx.slice [ Nx.A; Nx.I 0 ] x in
            let at_start =
              Nx.all ~axes:[ 1 ] (Nx.equal x (Nx.unsqueeze ~axes:[ 0 ] x2))
            in
            let off =
              Nx.logical_and (Nx.greater_s a 50.) (Nx.logical_not at_start)
            in
            Nx.where off (Nx.full_like a v) (normal x)
          in
          let s = H.init t ~geometry:narrow.g lp x in
          raises
            (Invalid_argument
               (Printf.sprintf
                  "Norn.Hmc.step: the density is %s at chain 2, a finite \
                   position"
                  (if Float.is_nan v then "nan" else "inf")))
            (fun () -> H.step t lp (Nx.Rng.key 3) s));
      (* norn.mli's Gaussian: the elements of every float tensor taken together
         are one vector. *)
      test "a structure moves as its float elements taken together" (fun () ->
          let x = start 4 in
          let split x =
            (Nx.slice [ Nx.A; Nx.R (0, 1) ] x, Nx.slice [ Nx.A; Nx.R (1, 2) ] x)
          in
          let joined (a, b) = Nx.concatenate ~axis:1 [ a; b ] in
          let u = Nx.Ptree.(pair tensor tensor) in
          let lp p = banana (joined p) in
          let pair = H.step u lp (Nx.Rng.key 3) (H.init u lp (split x)) in
          let one = H.step t banana (Nx.Rng.key 3) (H.init t banana x) in
          equal (array float_exact) (floats one.position)
            (floats (joined pair.position)));
      prop ~count:20 "a compiled transition is the eager one"
        Gen.(pair bool (int_range 0 9))
        (fun (is_narrow, seed) ->
          Compiled.here ();
          let geometry = if is_narrow then narrow else wide in
          let s = H.init t ~geometry:geometry.g banana (start 4) in
          let sp = H.ptree t in
          let step =
            Rune.jit
              Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
              (H.step t banana)
          in
          let k = Nx.Rng.key seed in
          let compiled = step k s in
          let eager = H.step t banana k s in
          (* Fused kernels round otherwise; a decision near its threshold would
             flip, at a rate of about 1e-13 per decision. *)
          equal
            (array (float 1e-9))
            (floats eager.position) (floats compiled.position);
          equal (array int32)
            (Nx.to_array eager.stats.n_steps)
            (Nx.to_array compiled.stats.n_steps));
      test "a narrow geometry's transition leaves the banana invariant"
        (fun () -> invariant narrow.g);
      test "a wide geometry's transition leaves the banana invariant" (fun () ->
          invariant wide.g);
    ]

(* warmup

   An independent normal of standard deviations 10 and 0.1, from a unit
   geometry: ten times too narrow along one element and ten times too wide along
   the other. *)

let scaled x =
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  Nx.mul_s
    (Nx.add (Nx.square (Nx.div_s a 10.)) (Nx.square (Nx.div_s b 0.1)))
    (-0.5)

(* 64 chains warmed for a harmonic mean acceptance of 0.95, away from the
   default 0.8. *)
let warmed =
  lazy
    (let x = Nx.Rng.normal ~key:(Nx.Rng.key 3) Nx.float64 [| 64; dim |] in
     H.warmup t scaled (Nx.Rng.key 4) ~steps:300
       (H.init t ~accept:0.95 scaled x))

(* [harmonic a] is the harmonic mean of the acceptances [a]. *)
let harmonic a =
  float_of_int (Array.length a)
  /. Array.fold_left (fun s x -> s +. (1. /. x)) 0. a

(* ChEES on a standard normal in 10 dimensions. The flow is a rotation, [x' = x
   cos t + p sin t], and the expected squared change of [|x|²] is [4 d sin² t].
   Over times [t = u T], [u] uniform in (0, 2), its mean [1 - sin 4T / 4T] peaks
   at [4T = tan 4T], [T = 1.1234]. Rounding the step count up lengthens a
   trajectory by half a step on average, so the length plus half a step is near
   the optimum. *)
let chees length =
  let d = 10 in
  let x = Nx.Rng.normal ~key:(Nx.Rng.key 3) Nx.float64 [| 64; d |] in
  let s = H.init t ~accept:0.95 normal x |> with_length length in
  let s = H.warmup t normal (Nx.Rng.key 4) ~steps:300 s in
  Nx.item [] s.length +. (Nx.item [] s.step_size /. 2.)

let warmup =
  group "warmup"
    [
      test "a negative number of steps is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Hmc.warmup: steps")
            (fun () ->
              H.warmup t banana (Nx.Rng.key 1) ~steps:(-1)
                (H.init t banana (start 2))));
      test "no steps leave the state" (fun () ->
          let s = H.init t banana (start 2) in
          let s' = H.warmup t banana (Nx.Rng.key 1) ~steps:0 s in
          equal float_exact (Nx.item [] s.step_size) (Nx.item [] s'.step_size);
          equal int32 0l (Nx.item [] s'.draw));
      (* Stan's schedule: no geometry is fitted below 20 steps. *)
      test "fewer than 20 steps keep the geometry, 20 refit it" (fun () ->
          let s = H.init t ~geometry:narrow.g banana (start 8) in
          let v steps =
            floats
              (Norn.Gaussian.variance t
                 (H.warmup t banana (Nx.Rng.key 1) ~steps s).geometry)
          in
          let before = floats (Norn.Gaussian.variance t s.geometry) in
          equal (array float_exact) before (v 19);
          not_equal (array float_exact) before (v 20));
      (* On a Gaussian the score is linear in the draws, [-x / σ²], so the
         Fisher fit [sqrt (sd x / sd score)] is σ for any draws, and what
         remains is rounding. The default geometry is off by the spread of the
         chains' gradients. *)
      test "the geometry reaches the target's scale" (fun () ->
          let s = Lazy.force warmed in
          let sd = Nx.sqrt (Norn.Gaussian.variance t s.geometry) in
          Array.iteri
            (fun i sd ->
              at_most (float 1e-12) ~than:1e-9
                (Float.abs (Float.log (sd /. [| 10.; 0.1 |].(i)))))
            (floats sd));
      (* Dual averaging aims the chains' harmonic mean acceptance at [s.accept];
         0.95 is away from the default 0.8 and from the step-size search's 0.8.
         Over keys 1 to 40 the mean over 100 draws was 0.948 ± 0.0046 (sd), so
         0.03 lies 6 sds out: a rate of about 2e-9. At 0.6 the averaged step
         size of a 300-step warmup accepts 0.81 ± 0.045, which says nothing of
         the target. *)
      test "the step size gives the target harmonic mean acceptance" (fun () ->
          let _, _, st =
            H.sample t scaled (Nx.Rng.key 5) ~draws:100 (Lazy.force warmed)
          in
          let a = (st :> (Nx.float64_elt, Nx.host) Norn.Stats.t).acceptance in
          let per_draw =
            List.init 100 (fun j ->
                harmonic (floats (Nx.slice [ Nx.A; Nx.I j ] a)))
          in
          equal (float 0.03) 0.95 (List.fold_left ( +. ) 0. per_draw /. 100.));
      (* Chain 0 starts in a pocket of radius 0.1 where the density is finite,
         -inf around it: only a short step moves it. *)
      test "a chain that never accepts shrinks the step until every chain moves"
        (fun () ->
          let pocket = Nx.create Nx.float64 [| 1; dim |] [| 5.; 5. |] in
          let lp x =
            let a = Nx.slice [ Nx.A; Nx.I 0 ] x in
            let inside =
              Nx.less_s (Nx.sum ~axes:[ 1 ] (Nx.square (Nx.sub x pocket))) 0.01
            in
            let open_ = Nx.less_s a 2. in
            Nx.where
              (Nx.logical_or inside open_)
              (normal x)
              (Nx.full_like a Float.neg_infinity)
          in
          let x = Nx.set [ Nx.I 0 ] (Nx.reshape [| dim |] pocket) (start 4) in
          let s = H.warmup t lp (Nx.Rng.key 6) ~steps:30 (H.init t lp x) in
          let _, _, st = H.sample t lp (Nx.Rng.key 7) ~draws:20 s in
          let a = (st :> (Nx.float64_elt, Nx.host) Norn.Stats.t).acceptance in
          greater float_exact ~than:0.
            (Nx.item [] (Nx.mean (Nx.slice [ Nx.I 0 ] a))));
      (* The criterion is flat near its optimum: at ±25% it falls about 10%, so
         the iterates wander there. Over keys 1 to 40 the length from 0.1
         reached 1.226 ± 0.042 (sd), 4.1 sds inside the bound: a rate of about
         2e-5. The criterion has further local maxima at the next roots of [4T =
         tan 4T], T = 1.93, 2.73, ..., where a length started above them
         stays. *)
      test "the length rises to ChEES's optimum" (fun () ->
          equal (float_rel ~rel:0.25 ~abs:0.) 1.1234 (chees 0.1));
      (* With one key for warmup and sampling, the first draw takes the key
         after warmup's last: no two transitions share one. *)
      test "warmup takes the run's first keys and sampling continues" (fun () ->
          let k = Nx.Rng.key 1 in
          let s = H.init t scaled (Nx.ones Nx.float64 [| 2; 2 |]) in
          let w = H.warmup t scaled k ~steps:10 s in
          equal int32 10l (Nx.item [] w.draw);
          let _, d, _ = H.sample t scaled k ~draws:1 w in
          let next = H.step t scaled (Nx.Rng.fold_in k 10) w in
          equal (array float_exact) (floats next.position) (draws d));
    ]

(* sample *)

module M = Norn_model
module R = Recovery

let sample =
  group "sample"
    [
      test "fewer than one draw is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Hmc.sample: draws")
            (fun () ->
              H.sample t banana (Nx.Rng.key 1) ~draws:0
                (H.init t banana (start 2))));
      test "a draws then b draws are a + b draws" (fun () ->
          let s = H.init t ~geometry:narrow.g banana (start 2) in
          let _, ab, _ = H.sample t banana (Nx.Rng.key 8) ~draws:5 s in
          let s', a, _ = H.sample t banana (Nx.Rng.key 8) ~draws:2 s in
          let _, b, _ = H.sample t banana (Nx.Rng.key 8) ~draws:3 s' in
          equal (array float_exact) (draws ab) (draws (Norn.Draws.append t a b)));
      (* 64 chains in superchains of eight, 300 warmup transitions, then 50. *)
      test "draws recover the dogs posterior" (fun () ->
          match R.posterior "dogs-dogs" with
          | Posteriordb.Posterior p as post ->
              let m = p.model in
              let u = M.coords m and lp = M.log_density m p.y in
              let s = H.init u lp (M.init m p.y ~chains:64 R.key) in
              let s = H.warmup u lp R.key ~steps:300 s in
              let _, d, stats = H.sample u lp R.key ~draws:50 s in
              R.chains post ~superchains:8 ~stats
                (Norn.Draws.map u Posteriordb.latent (M.constrain m) d));
    ]

let () = exit (run "Norn.Hmc" [ init; step; warmup; sample ])
