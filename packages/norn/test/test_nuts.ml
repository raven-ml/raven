(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module N = Norn.Nuts

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
   banana, so that a unit step is short and trees grow deep, and one wider, so
   that they diverge and turn early. *)
type geometry = {
  g : (Nx.host Nx.float64_t, Nx.float64_elt, Nx.host) Norn.Gaussian.t;
  m : Nx.host Nx.float64_t;
  s : Nx.host Nx.float64_t;
}

let geometry_of scale =
  let m = Nx.create Nx.float64 [| dim |] [| 0.1; -0.2 |] in
  let s = Nx.create Nx.float64 [| dim |] scale in
  { g = Norn.Gaussian.diagonal t Nx.float64 ~mean:m ~scale:s; m; s }

let narrow = geometry_of [| 0.15; 0.08 |]
let wide = geometry_of [| 1.3; 0.6 |]
let color geometry z = Nx.add geometry.m (Nx.mul geometry.s z)
let whiten geometry x = Nx.div (Nx.sub x geometry.m) geometry.s

(* [variances s] is each chain's geometry's variances, [[chain; element]]. *)
let variances (s : (_, _, _) N.state) =
  Rune.vmap
    Nx.Ptree.(Norn.Gaussian.ptree t @-> returns tensor)
    (Norn.Gaussian.variance t) s.geometry

(* [rank s] is the number of directions of the chains' geometry. *)
let rank (s : (_, _, _) N.state) =
  Nx.Ptree.fold (Norn.Gaussian.ptree t)
    (fun p x r ->
      if Nx.Ptree.Path.to_string p = "variances" then (Nx.shape x).(1) else r)
    s.geometry (-1)

(* [with_step_size eps s] is [s] with chain [i]'s step size [eps.(i)]. *)
let with_step_size eps s =
  Nx.Ptree.map (N.ptree t)
    (fun p x ->
      if Nx.Ptree.Path.to_string p <> "step_size" then x
      else
        Nx.cast (Nx.dtype x) (Nx.create Nx.float64 [| Array.length eps |] eps))
    s

(* init *)

let init =
  group "init"
    [
      test "a density not batched over chains is refused" (fun () ->
          raises
            (Invalid_argument
               "Norn.Nuts.init: the density returned shape [4; 2] for a \
                position of 4 chains; a density returns one log density per \
                chain, shape [4]") (fun () ->
              N.init t (fun x -> Nx.neg (Nx.square x)) (start 4)));
      test "a density whose rows read each other is refused, naming the row"
        (fun () ->
          let coupled x =
            Nx.add (banana x) (Nx.slice [ Nx.L [ 0; 0; 0; 0 ]; Nx.I 0 ] x)
          in
          raises_match
            (Exn.invalid_arg ~sub:"Norn.Nuts.init: the density's row 0 changes")
            (fun () -> N.init t coupled (start 4)));
      test "a density NaN at a chain is refused, naming it" (fun () ->
          let nan_at_two x =
            Nx.where
              (Nx.equal (Nx.arange Nx.int32 0 4 1) (Nx.scalar Nx.int32 2l))
              (Nx.full Nx.float64 [| 4 |] Float.nan)
              (banana x)
          in
          raises
            (Invalid_argument
               "Norn.Nuts.init: the density is nan at chain 2, a finite \
                position") (fun () -> N.init t nan_at_two (start 4)));
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
               "Norn.Nuts.init: the density is -inf at chain 1; a chain starts \
                where its density is finite") (fun () -> N.init t walled x));
      test "a tensor that is not a float is refused, naming its path" (fun () ->
          let u = Nx.Ptree.(pair tensor tensor) in
          let x = (start 4, Nx.zeros Nx.int32 [| 4 |]) in
          raises_match (Exn.invalid_arg ~sub:"Norn.Nuts.init: 1 is int32")
            (fun () -> N.init u (fun (x, _) -> banana x) x));
      cases "a maximum depth below 1 is refused" ~id:string_of_int [ 0; -1 ]
        (fun max_depth ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Nuts.init: max_depth")
            (fun () -> N.init t ~max_depth banana (start 2)));
      cases "an acceptance outside (0, 1) is refused" ~id:Float.to_string
        [ 0.; -0.; 1.; Float.nan; 1e6 ] (fun accept ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Nuts.init: accept")
            (fun () -> N.init t ~accept banana (start 2)));
      test "a negative rank is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Nuts.init: rank") (fun () ->
              N.init t ~rank:(-1) banana (start 2)));
      test
        "the defaults are a depth of 10, an acceptance of 0.8, step 1, rank 2"
        (fun () ->
          let s = N.init t banana (start 2) in
          equal int 10 s.max_depth;
          equal float_exact 0.8 (Nx.item [] s.accept);
          equal (array float_exact) [| 1.; 1. |] (floats s.step_size);
          equal int 2 (rank s));
      test "a rank above a chain's float elements is that number" (fun () ->
          let s =
            N.init t ~rank:2 normal
              (Nx.create Nx.float64 [| 2; 1 |] [| 1.; 2. |])
          in
          equal int 1 (rank s));
      (* -x²/2 has gradient -x: variance 1 / |x|, and 1e20 at the mode. *)
      test
        "the default geometry is centred on each start, of variance 1 / |grad| \
         clipped" (fun () ->
          let x = Nx.create Nx.float64 [| 2; 2 |] [| 4.; 0.5; 0.; 2. |] in
          let s = N.init t ~rank:0 normal x in
          let mean =
            Rune.vmap
              Nx.Ptree.(Norn.Gaussian.ptree t @-> returns tensor)
              (Norn.Gaussian.mean t) s.geometry
          in
          equal (array float_exact) (floats x) (floats mean);
          equal
            (array (float_rel ~rel:1e-15 ~abs:0.))
            [| 0.25; 2.; 1e20; 0.5 |]
            (floats (variances s)));
      test "a given geometry seeds every chain" (fun () ->
          let s = N.init t ~geometry:narrow.g banana (start 3) in
          let v = floats (Nx.square narrow.s) in
          equal (array float_exact)
            (Array.concat [ v; v; v ])
            (floats (variances s)));
    ]

(* step

   Stan's transition (base_nuts.hpp), written recursively over one chain in
   whitened coordinates, from the keys the .mli's Keys paragraph states. *)

type pt = { z : float array; p : float array; g : float array; lp : float }

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
let vadd a b = Array.mapi (fun i x -> x +. b.(i)) a
let crit minus plus rho = dot plus rho > 0. && dot minus rho > 0.

let log_sum_exp a b =
  if a = Float.neg_infinity then b
  else if b = Float.neg_infinity then a
  else
    let m = Float.max a b in
    m +. Float.log (Float.exp (a -. m) +. Float.exp (b -. m))

(* What Stan's transition gives a chain, and whether a subtree of it turned. *)
type stan = {
  sample : pt;
  h0 : float;
  n : int;
  accept : float;
  divergent : bool;
  saturated : bool;
  turned : bool;
}

let stan geometry ~max_depth ~eps key (z0 : float array) =
  let eval = eval geometry in
  let uniform j id =
    let k = Nx.Rng.fold_in (Nx.Rng.fold_in (Nx.Rng.fold_in key 1) j) id in
    Nx.item [] (Nx.Rng.uniform ~key:k Nx.float64 [||])
  in
  let p0 =
    floats
      (Nx.Rng.normal
         ~key:(Nx.Rng.fold_in (Nx.Rng.fold_in key 0) 0)
         Nx.float64 [| dim |])
  in
  let lp0, g0 = eval z0 in
  let h0 = (0.5 *. dot p0 p0) -. lp0 in
  let n_leapfrog = ref 0 and sum_metro = ref 0. and divergent = ref false in
  let turned = ref false in
  let leapfrog s sign =
    let h = sign *. eps in
    let p = axpy (h /. 2.) s.g s.p in
    let z = axpy h p s.z in
    let lp, g = eval z in
    let p = axpy (h /. 2.) g p in
    { z; p; g; lp }
  in
  (* [build depth j count] returns the subtree's validity, proposal, log weight,
     rho and first and last momenta; [count] counts the subtree's leaves. *)
  let rec build cur depth j sign count =
    if depth = 0 then begin
      let s = leapfrog !cur sign in
      cur := s;
      incr n_leapfrog;
      incr count;
      let h = (0.5 *. dot s.p s.p) -. s.lp in
      let h = if Float.is_nan h then Float.infinity else h in
      if h -. h0 > 1000. then divergent := true;
      sum_metro := !sum_metro +. Float.min 1. (Float.exp (h0 -. h));
      (not !divergent, s, h0 -. h, s.p, s.p, s.p)
    end
    else
      let ok_i, prop_i, w_i, rho_i, beg_i, end_i =
        build cur (depth - 1) j sign count
      in
      if not ok_i then (false, prop_i, w_i, rho_i, beg_i, end_i)
      else
        let ok_f, prop_f, w_f, rho_f, beg_f, end_f =
          build cur (depth - 1) j sign count
        in
        if not ok_f then (false, prop_f, w_f, rho_f, beg_f, end_f)
        else
          let w = log_sum_exp w_i w_f in
          let node = (depth lsl max_depth) + (!count lsr depth) in
          let prop =
            if w_f > w then prop_f
            else if uniform j node < Float.exp (w_f -. w) then prop_f
            else prop_i
          in
          let rho = vadd rho_i rho_f in
          let ok =
            crit beg_i end_f rho
            && crit beg_i beg_f (vadd rho_i beg_f)
            && crit end_i end_f (vadd rho_f end_i)
          in
          if not ok then turned := true;
          (ok, prop, w, rho, beg_i, end_f)
  in
  let start = { z = z0; p = p0; g = g0; lp = lp0 } in
  let fwd = ref start and bck = ref start and sample = ref start in
  let rho = ref p0 and total = ref 0. and depth = ref 0 in
  let p_fwd_fwd = ref p0
  and p_fwd_bck = ref p0
  and p_bck_fwd = ref p0
  and p_bck_bck = ref p0 in
  let continue = ref true in
  while !continue && !depth < max_depth do
    let j = !depth in
    let forward = uniform j 0 > 0.5 in
    let cur = ref (if forward then !fwd else !bck) in
    let count = ref 0 in
    if forward then p_bck_fwd := !p_fwd_fwd else p_fwd_bck := !p_bck_bck;
    let ok, prop, w, rho_sub, beg, fin =
      build cur j j (if forward then 1. else -1.) count
    in
    if forward then fwd := !cur else bck := !cur;
    if not ok then continue := false
    else begin
      incr depth;
      if w > !total then sample := prop
      else if uniform j ((max_depth + 1) lsl max_depth) < Float.exp (w -. !total)
      then sample := prop;
      total := log_sum_exp !total w;
      let rho_bck, rho_fwd =
        if forward then (!rho, rho_sub) else (rho_sub, !rho)
      in
      if forward then begin
        p_fwd_bck := beg;
        p_fwd_fwd := fin
      end
      else begin
        p_bck_fwd := beg;
        p_bck_bck := fin
      end;
      rho := vadd rho_bck rho_fwd;
      let persist =
        crit !p_bck_bck !p_fwd_fwd !rho
        && crit !p_bck_bck !p_fwd_bck (vadd rho_bck !p_fwd_bck)
        && crit !p_bck_fwd !p_fwd_fwd (vadd rho_fwd !p_bck_fwd)
      in
      if not persist then continue := false
    end
  done;
  {
    sample = !sample;
    h0;
    n = !n_leapfrog;
    accept = !sum_metro /. float_of_int !n_leapfrog;
    divergent = !divergent;
    saturated = !continue && !depth = max_depth;
    turned = !turned;
  }

(* Four chains move in lock step, each with its own step size, building its own
   tree and stopping on its own. The 80 values of (maximum depth, narrow
   geometry, key) are checked whole, each a transition. *)
let chains = 4
let step_sizes = [| 0.6; 0.8; 1.; 1.2 |]

let stan_tree =
  prop ~count:80 "a transition is Stan's recursive tree"
    Gen.(triple (int_range 1 8) bool (int_range 0 4))
  @@ fun (max_depth, is_narrow, seed) ->
  let key = Nx.Rng.key (1000 + seed) in
  let x0 =
    Nx.Rng.normal ~key:(Nx.Rng.fold_in key 7) Nx.float64 [| chains; dim |]
  in
  let geometry = if is_narrow then narrow else wide in
  let s =
    with_step_size step_sizes
      (N.init t ~max_depth ~geometry:geometry.g banana x0)
  in
  let s' = N.step t banana key s in
  let st = s'.stats in
  let keys = Nx.Rng.split ~n:chains key in
  let refs =
    Array.init chains (fun i ->
        let z0 = floats (whiten geometry (Nx.slice [ Nx.I i ] x0)) in
        stan geometry ~max_depth ~eps:step_sizes.(i) keys.(i) z0)
  in
  let some f = Array.exists f refs in
  cover "chains stop at different steps" (some (fun r -> r.n <> refs.(0).n));
  cover "a subtree turns" (some (fun r -> r.turned));
  cover "a chain diverges" (some (fun r -> r.divergent));
  cover "a chain saturates" (some (fun r -> r.saturated));
  Array.iteri
    (fun i r ->
      let msg = Printf.sprintf "chain %d" i in
      let at f = Nx.item [ i ] f in
      let x = color geometry (Nx.create Nx.float64 [| dim |] r.sample.z) in
      (* The gradient in x is the whitened one over the scale. *)
      let g = Nx.div (Nx.create Nx.float64 [| dim |] r.sample.g) geometry.s in
      equal ~msg int r.n (Int32.to_int (at st.n_steps));
      equal ~msg bool r.divergent (at st.diverging);
      equal ~msg bool r.saturated (at st.saturated);
      equal ~msg (float 1e-12) r.accept (at st.acceptance);
      equal ~msg (float 1e-12) r.h0 (at st.energy);
      equal ~msg float_exact step_sizes.(i) (at st.step_size);
      equal ~msg (float 1e-10) r.sample.lp (at s'.lp);
      equal ~msg (float 1e-10) r.sample.lp (at st.lp);
      equal ~msg
        (array (float 1e-10))
        (floats x)
        (floats (Nx.slice [ Nx.I i ] s'.position));
      equal ~msg
        (array (float 1e-9))
        (floats g)
        (floats (Nx.slice [ Nx.I i ] s'.grad)))
    refs

(* Invariance. After five transitions of every chain from the banana's exact
   draws, [a] and [b - 0.3 a²] keep their normal CDFs within the
   Dvoretzky-Kiefer-Wolfowitz band of level [0.01 / 6], six comparisons holding
   the false alarms at 1%. The chains are independent, so their draws are. At
   4096 chains the band is 0.033 wide: a kernel whose stationary CDF strays
   further is caught. *)
let invariance_chains = 4096

let band =
  Float.sqrt
    (Float.log (2. /. (0.01 /. 6.)) /. (2. *. float_of_int invariance_chains))

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

let invariant (type f) (dt : (float, f) Nx.dtype) geometry =
  let k = Nx.Rng.key 31 in
  let a = Nx.Rng.normal ~key:(Nx.Rng.fold_in k 0) dt [| invariance_chains |] in
  let z = Nx.Rng.normal ~key:(Nx.Rng.fold_in k 1) dt [| invariance_chains |] in
  let b = Nx.add (Nx.mul_s (Nx.square a) 0.3) (Nx.mul_s z 0.5) in
  let geometry =
    Norn.Gaussian.diagonal t dt ~mean:(Nx.cast dt geometry.m)
      ~scale:(Nx.cast dt geometry.s)
  in
  let s = ref (N.init t ~geometry banana (Nx.stack ~axis:1 [ a; b ])) in
  for i = 0 to 4 do
    s := N.step t banana (Nx.Rng.fold_in k (10 + i)) !s
  done;
  let x = Nx.cast Nx.float64 !s.position in
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
  at_most (float 1e-12) ~than:band (max_deviation (floats a) 1.);
  at_most (float 1e-12) ~than:band (max_deviation (floats b') 0.5)

let step =
  group "step"
    [
      stan_tree;
      test "a chain's transition does not depend on the chain count" (fun () ->
          let one =
            N.step t banana (Nx.Rng.key 3)
              (N.init t ~geometry:narrow.g banana
                 (Nx.slice [ Nx.R (0, 1); Nx.R (0, dim) ] (start 4)))
          in
          let four =
            N.step t banana (Nx.Rng.key 3)
              (N.init t ~geometry:narrow.g banana (start 4))
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
            let off =
              Nx.logical_and (Nx.greater_s a 50.)
                (Nx.logical_not
                   (Nx.all ~axes:[ 1 ]
                      (Nx.equal x (Nx.unsqueeze ~axes:[ 0 ] x2))))
            in
            Nx.where off (Nx.full_like a v) (normal x)
          in
          let s = N.init t lp x in
          raises
            (Invalid_argument
               (Printf.sprintf
                  "Norn.Nuts.step: the density is %s at chain 2, a finite \
                   position"
                  (if Float.is_nan v then "nan" else "inf")))
            (fun () -> N.step t lp (Nx.Rng.key 3) s));
      (* norn.mli's preamble: a sampler traces its transitions. *)
      test "a read of the density's argument inside a transition raises"
        (fun () ->
          let reads x =
            ignore (Nx.item [ 0; 0 ] x : float);
            (normal x, Nx.neg x)
          in
          let lp = Norn.with_gradient t reads in
          let s = N.init t ~geometry:narrow.g lp (start 2) in
          raises_match (Exn.invalid_arg ~sub:"traced") (fun () ->
              N.step t lp (Nx.Rng.key 3) s));
      (* norn.mli's Gaussian: the elements of every float tensor taken together
         are one vector. *)
      test "a structure moves as its float elements taken together" (fun () ->
          let x = start 4 in
          let split x =
            (Nx.slice [ Nx.A; Nx.R (0, 1) ] x, Nx.slice [ Nx.A; Nx.R (1, 2) ] x)
          in
          let joined (a, b) = Nx.concatenate ~axis:1 [ a; b ] in
          let u = Nx.Ptree.(pair tensor tensor) in
          let pair =
            N.step u
              (fun p -> banana (joined p))
              (Nx.Rng.key 3)
              (N.init u (fun p -> banana (joined p)) (split x))
          in
          let one = N.step t banana (Nx.Rng.key 3) (N.init t banana x) in
          equal (array float_exact) (floats one.position)
            (floats (joined pair.position)));
      prop ~count:20 "a compiled transition is the eager one"
        Gen.(pair bool (int_range 0 9))
        (fun (is_narrow, seed) ->
          Compiled.here ();
          let geometry = if is_narrow then narrow else wide in
          let s = N.init t ~geometry:geometry.g banana (start 4) in
          let sp = N.ptree t in
          let step =
            Rune.jit
              Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
              (N.step t banana)
          in
          let k = Nx.Rng.key seed in
          let compiled = step k s in
          let eager = N.step t banana k s in
          (* Fused kernels round otherwise; a decision near its threshold would
             flip, at a rate of about 1e-13 per decision. *)
          equal
            (array (float 1e-9))
            (floats eager.position) (floats compiled.position);
          equal (array int32)
            (Nx.to_array eager.stats.n_steps)
            (Nx.to_array compiled.stats.n_steps);
          equal (array bool)
            (Nx.to_array eager.stats.diverging)
            (Nx.to_array compiled.stats.diverging);
          equal int eager.max_depth compiled.max_depth);
      test "a narrow geometry's transition leaves the banana invariant"
        (fun () -> invariant Nx.float64 narrow);
      test "a wide geometry's transition leaves the banana invariant" (fun () ->
          invariant Nx.float64 wide);
      test "a float32 transition leaves the banana invariant" (fun () ->
          invariant Nx.float32 wide);
    ]

(* warmup

   An independent normal of standard deviations 10 and 0.1. *)

let scaled x =
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  Nx.mul_s
    (Nx.add (Nx.square (Nx.div_s a 10.)) (Nx.square (Nx.div_s b 0.1)))
    (-0.5)

let scales = [| 10.; 0.1 |]

(* Four chains warmed for an acceptance of 0.95, away from the default 0.8. *)
let warmed =
  lazy
    (let x = Nx.Rng.normal ~key:(Nx.Rng.key 3) Nx.float64 [| 4; dim |] in
     N.warmup t scaled (Nx.Rng.key 4) ~steps:300
       (N.init t ~accept:0.95 scaled x))

let warmup =
  group "warmup"
    [
      test "a negative number of steps is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Nuts.warmup: steps")
            (fun () ->
              N.warmup t banana (Nx.Rng.key 1) ~steps:(-1)
                (N.init t banana (start 2))));
      test "no steps leave the state" (fun () ->
          let s = N.init t banana (start 2) in
          let s' = N.warmup t banana (Nx.Rng.key 1) ~steps:0 s in
          equal (array float_exact) (floats s.step_size) (floats s'.step_size);
          equal int32 0l (Nx.item [] s'.draw));
      (* Stan's schedule: no geometry is fitted below 20 steps. *)
      test "fewer than 20 steps keep the geometry, 20 refit it" (fun () ->
          let s = N.init t ~geometry:narrow.g banana (start 2) in
          let v steps =
            floats (variances (N.warmup t banana (Nx.Rng.key 1) ~steps s))
          in
          let before = floats (variances s) in
          equal (array float_exact) before (v 19);
          not_equal (array float_exact) before (v 20));
      (* On a Gaussian the score is linear in the draws, [-x / σ²], so the
         Fisher fit [sqrt (sd x / sd score)] is σ for any draws, and what
         remains is rounding. The default geometry is off by a factor of [1 /
         sqrt |x|] at each start. *)
      test "each chain's geometry reaches the target's scale" (fun () ->
          let sd = Nx.sqrt (variances (Lazy.force warmed)) in
          let target = Nx.create Nx.float64 [| 1; dim |] scales in
          Array.iter
            (fun r ->
              at_most (float 1e-12) ~than:1e-9 (Float.abs (Float.log r)))
            (floats (Nx.div sd target)));
      (* Dual averaging aims each chain's step size at [s.accept]; 0.95 is away
         from the default 0.8 and from the step-size search's 0.8. Over keys 1
         to 40 the mean acceptance of 200 draws was 0.957 ± 0.0043 (sd), so 0.03
         lies 5.3 sds out: a rate of about 1e-7. At 0.6 the averaged step size
         of a 300-step warmup accepts 0.76 ± 0.03, which says nothing of the
         target. *)
      test "the step sizes give the target acceptance" (fun () ->
          let _, _, st =
            N.sample t scaled (Nx.Rng.key 5) ~draws:200 (Lazy.force warmed)
          in
          let st = (st :> (Nx.float64_elt, Nx.host) Norn.Stats.t) in
          equal (float 0.03) 0.95 (Nx.item [] (Nx.mean st.acceptance)));
      test "a rank-1 warmup fits one direction" (fun () ->
          let x = Nx.Rng.normal ~key:(Nx.Rng.key 3) Nx.float64 [| 2; dim |] in
          let s =
            N.warmup t scaled (Nx.Rng.key 4) ~steps:150
              (N.init t ~rank:1 scaled x)
          in
          equal int 1 (rank s));
      test "a rank above the dimension fits as many directions as it has"
        (fun () ->
          let s = N.init t ~rank:2 normal (Nx.zeros Nx.float64 [| 2; 1 |]) in
          let s = N.warmup t normal (Nx.Rng.key 3) ~steps:30 s in
          equal int 1 (rank s);
          Array.iter
            (fun e -> satisfies ~claim:"finite" (float 1e-12) Float.is_finite e)
            (floats s.step_size));
      (* Chain 3 starts beside a wall where the density falls to -inf, and its
         trees diverge there. *)
      test "a chain's warmup does not depend on the chain count" (fun () ->
          let walled x =
            let a = Nx.slice [ Nx.A; Nx.I 0 ] x in
            Nx.where (Nx.greater_s a 2.5)
              (Nx.full_like a Float.neg_infinity)
              (banana x)
          in
          let x =
            Nx.set [ Nx.I 3; Nx.I 0 ] (Nx.scalar Nx.float64 2.4) (start 4)
          in
          let warm x =
            N.warmup t walled (Nx.Rng.key 6) ~steps:30 (N.init t walled x)
          in
          let one = warm (Nx.slice [ Nx.R (0, 1) ] x) and four = warm x in
          equal (array float_exact) (row0 one.position) (row0 four.position);
          equal float_exact
            (Nx.item [ 0 ] one.step_size)
            (Nx.item [ 0 ] four.step_size));
      (* With one key for warmup and sampling, the first draw takes the key
         after warmup's last: no two transitions share one. *)
      test "warmup takes the run's first keys and sampling continues" (fun () ->
          let k = Nx.Rng.key 1 in
          let s = N.init t scaled (Nx.ones Nx.float64 [| 2; 2 |]) in
          let w = N.warmup t scaled k ~steps:10 s in
          equal int32 10l (Nx.item [] w.draw);
          let _, d, _ = N.sample t scaled k ~draws:1 w in
          let next = N.step t scaled (Nx.Rng.fold_in k 10) w in
          equal (array float_exact) (floats next.position) (draws d));
    ]

(* sample *)

module M = Norn_model
module R = Recovery

(* [recovered ?accept ~draws name] asserts the recovery law on NUTS's draws of
   the posteriordb posterior [name]: 8 chains in superchains of two, 300 warmup
   transitions, then [draws]. *)
let recovered ?accept ~draws name =
  match R.posterior name with
  | Posteriordb.Posterior p as post ->
      let m = p.model in
      let u = M.coords m and lp = M.log_density m p.y in
      let s = N.init u ?accept lp (M.init m p.y ~chains:8 R.key) in
      let s = N.warmup u lp R.key ~steps:300 s in
      let _, d, stats = N.sample u lp R.key ~draws s in
      R.chains post ~superchains:4 ~stats
        (Norn.Draws.map u Posteriordb.latent (M.constrain m) d)

(* Simulation-based calibration of a conjugate normal model: the ranks of
   simulated truths among NUTS draws of their posterior are uniform. mu ~ N(0,
   1), y_i ~ N(mu, 1) for 5 points. The uniformity test rejects below a p-value
   of 0.005, its false-alarm rate. *)
let calibrated () =
  Compiled.here ();
  let f64 = Nx.scalar Nx.float64 and draws = 99 in
  let model =
    M.v Nx.float64 t t (fun () ->
        let mu = M.sample (Norn.Dist.normal ~loc:(f64 0.) ~scale:(f64 1.)) in
        ( mu,
          M.sample
            (Norn.Dist.iid [| 5 |] (Norn.Dist.normal ~loc:mu ~scale:(f64 1.)))
        ))
  in
  let u = M.coords model in
  let fit =
    Rune.jit
      Nx.Ptree.(Nx.Rng.ptree @-> tensor @-> returns (Norn.Draws.ptree u))
      (fun k y ->
        let lp = M.log_density model y in
        let start = M.init model y ~chains:1 (Nx.Rng.fold_in k 0) in
        let s = N.init u lp start in
        let s = N.warmup u lp (Nx.Rng.fold_in k 1) ~steps:200 s in
        let _, d, _ = N.sample u lp (Nx.Rng.fold_in k 2) ~draws:(draws * 3) s in
        Norn.Draws.thin u ~every:3 d)
  in
  let rank r =
    let k = Nx.Rng.fold_in (Nx.Rng.key 52) r in
    let truth, y = M.simulate model (Nx.Rng.fold_in k 0) in
    let d = fit (Nx.Rng.fold_in k 1) y in
    Norn.Diag.rank t
      ~truth:(M.unconstrain model truth :> Nx.host Nx.float64_t)
      (Norn.Draws.v t (d :> Nx.host Nx.float64_t))
  in
  let p = Norn.Diag.rank_uniformity t ~draws (List.init 100 rank) in
  at_least (float 1e-12) ~than:0.005 (Nx.item [] p)

let sample =
  group "sample"
    [
      test "fewer than one draw is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Nuts.sample: draws")
            (fun () ->
              N.sample t banana (Nx.Rng.key 1) ~draws:0
                (N.init t banana (start 2))));
      test "a draws then b draws are a + b draws" (fun () ->
          let s = N.init t ~geometry:narrow.g banana (start 2) in
          let _, ab, _ = N.sample t banana (Nx.Rng.key 8) ~draws:5 s in
          let s', a, _ = N.sample t banana (Nx.Rng.key 8) ~draws:2 s in
          let _, b, _ = N.sample t banana (Nx.Rng.key 8) ~draws:3 s' in
          equal (array float_exact) (draws ab) (draws (Norn.Draws.append t a b)));
      (* R-hat up to 1.020 on keys 1 to 40 at 100 draws. *)
      test "draws recover the dogs posterior" (fun () ->
          recovered ~draws:300 "dogs-dogs");
      (* At the default acceptance a transition or two diverges on 17 of keys 1
         to 40, and at 0.9 on 3. *)
      test "draws recover the eight schools posterior" (fun () ->
          recovered ~accept:0.95 ~draws:300
            "eight_schools-eight_schools_noncentered");
      test "ranks of simulated truths among its draws are uniform" calibrated;
    ]

let () = exit (run "Norn.Nuts" [ init; step; warmup; sample ])
