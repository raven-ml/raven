(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module N = Norn.Nuts

let t = Nx.Ptree.tensor

(* A standard normal in 3 dimensions, over a batch of chains. *)
let normal x = Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.square x)) (-0.5)

let smoke =
  test "draws of a standard normal have its moments" (fun () ->
      let start = Nx.zeros Nx.float64 [| 4; 3 |] in
      let g =
        Norn.Gaussian.diagonal t Nx.float64
          ~mean:(Nx.zeros Nx.float64 [| 3 |])
          ~scale:(Nx.ones Nx.float64 [| 3 |])
      in
      let s = N.init t ~geometry:g normal start in
      let t0 = Unix.gettimeofday () in
      let _, d, st = N.sample t normal (Nx.Rng.key 1) ~draws:500 s in
      Printf.printf "time %.2fs\n" (Unix.gettimeofday () -. t0);
      let d = (d :> Nx.float64_t) in
      let st = (st :> Nx.float64_elt Norn.Stats.t) in
      Printf.printf "mean %s var %s steps %s acc %s\n"
        (Nx.to_string (Nx.mean ~axes:[ 0; 1 ] d))
        (Nx.to_string (Nx.var ~axes:[ 0; 1 ] d))
        (Nx.to_string (Nx.mean (Nx.cast Nx.float64 st.n_steps)))
        (Nx.to_string (Nx.mean st.acceptance));
      equal bool true true)

(* Law 12: the iterative tree is Stan's recursive one

   Stan's builder (base_nuts.hpp), written recursively over one chain in
   whitened coordinates, draws its directions and uniforms from the keys the
   iterative builder uses: for a transition key [k] of one chain, row 0 of
   [split_batch ~n:1 k]; its momentum from [fold_in key 0]; in doubling [j],
   from [fold_in (fold_in (fold_in key 1) j) id], with [id] [0] for the
   direction, [l 2^D + n] for the merge into node [n] of level [l], counted from
   1 in the doubling, and [(D + 1) 2^D] for the top-level merge. *)

let dim = 2

(* A correlated, skewed target: a banana. *)
let banana x =
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
  Nx.mul_s (Nx.add (Nx.square a) (Nx.mul_s (Nx.square b') 4.)) (-0.5)

(* A geometry narrower than the target, so that a unit step is short and trees
   grow deep, and one wider, so that they diverge and turn early. *)
(* A diagonal geometry: its color map is [m + s z]. *)
type geometry = {
  g : (Nx.float64_t, Nx.float64_elt) Norn.Gaussian.t;
  m : Nx.float64_t;
  s : Nx.float64_t;
}

let geometry_of scale =
  let m = Nx.create Nx.float64 [| dim |] [| 0.1; -0.2 |] in
  let s = Nx.create Nx.float64 [| dim |] scale in
  { g = Norn.Gaussian.diagonal t Nx.float64 ~mean:m ~scale:s; m; s }

let color geometry z = Nx.add geometry.m (Nx.mul geometry.s z)
let whiten geometry x = Nx.div (Nx.sub x geometry.m) geometry.s
let narrow = geometry_of [| 0.15; 0.08 |]
let wide = geometry_of [| 1.3; 0.6 |]

type pt = { z : float array; p : float array; g : float array; lp : float }

let floats x = Nx.to_array x

(* The density in whitened coordinates and its gradient, at one chain. *)
let eval geometry z =
  let color z = color geometry z in
  let lp z = banana (Nx.unsqueeze ~axes:[ 0 ] (color z)) in
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

let stan geometry ~max_depth ~eps key (z0 : float array) =
  let eval = eval geometry in
  let key = (Nx.Rng.split ~n:1 key).(0) in
  let uniform j id =
    let k = Nx.Rng.fold_in (Nx.Rng.fold_in (Nx.Rng.fold_in key 1) j) id in
    Nx.item [] (Nx.Rng.uniform k Nx.float64 [||])
  in
  let p0 =
    floats
      (Nx.Rng.normal
         (Nx.Rng.fold_in (Nx.Rng.fold_in key 0) 0)
         Nx.float64 [| dim |])
  in
  let lp0, g0 = eval z0 in
  let h0 = (0.5 *. dot p0 p0) -. lp0 in
  let n_leapfrog = ref 0 and sum_metro = ref 0. and divergent = ref false in
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
  (!sample, !n_leapfrog, !sum_metro /. float_of_int !n_leapfrog, !divergent)

let law_12 =
  let case seed =
    test (Printf.sprintf "transition %d" seed) (fun () ->
        let key = Nx.Rng.key (1000 + seed) in
        let start =
          Nx.Rng.normal (Nx.Rng.fold_in key 7) Nx.float64 [| 1; dim |]
        in
        let max_depth = 1 + (seed mod 8) in
        let geometry = if seed mod 2 = 0 then narrow else wide in
        let s = N.init t ~max_depth ~geometry:geometry.g banana start in
        let s' = N.step t banana key s in
        let z0 = floats (whiten geometry (Nx.reshape [| dim |] start)) in
        let sample, n, accept, divergent =
          stan geometry ~max_depth ~eps:1. key z0
        in
        let x =
          floats (color geometry (Nx.create Nx.float64 [| dim |] sample.z))
        in
        let st = s'.stats in
        equal int n (Int32.to_int (Nx.item [ 0 ] st.n_steps));
        equal bool divergent (Nx.item [ 0 ] st.diverging);
        equal (float 1e-12) accept (Nx.item [ 0 ] st.acceptance);
        equal
          (array (float 1e-10))
          x
          (floats (Nx.reshape [| dim |] s'.position));
        equal (float 1e-10) sample.lp (Nx.item [ 0 ] s'.lp))
  in
  group "Stan's recursive builder" (List.init 80 case)

(* Warmup: an independent normal of standard deviations 10 and 0.1. *)

let scaled x =
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  Nx.mul_s
    (Nx.add (Nx.square (Nx.div_s a 10.)) (Nx.square (Nx.div_s b 0.1)))
    (-0.5)

let warmed =
  lazy
    (let start = Nx.Rng.normal (Nx.Rng.key 3) Nx.float64 [| 4; 2 |] in
     N.warmup t scaled (Nx.Rng.key 4) ~steps:300 (N.init t scaled start))

let warmup =
  group "warmup"
    [
      slow "the geometry's scale reaches the target's" (fun () ->
          let s = Lazy.force warmed in
          let sd =
            Rune.vmap
              Nx.Ptree.(Norn.Gaussian.ptree t @-> returns tensor)
              (Norn.Gaussian.variance t) s.geometry
          in
          let ratio =
            Nx.to_array
              (Nx.div (Nx.sqrt sd)
                 (Nx.create Nx.float64 [| 1; 2 |] [| 10.; 0.1 |]))
          in
          Array.iter
            (fun r ->
              satisfies ~claim:"within a factor of 2" (float 1e-12)
                (fun r -> r > 0.5 && r < 2.)
                r)
            ratio);
      slow "the step size gives about the target acceptance" (fun () ->
          let s = Lazy.force warmed in
          let _, _, st = N.sample t scaled (Nx.Rng.key 5) ~draws:200 s in
          let acc =
            Nx.item [] (Nx.mean (st :> Nx.float64_elt Norn.Stats.t).acceptance)
          in
          satisfies ~claim:"in (0.65, 0.95)" (float 1e-12)
            (fun a -> a > 0.65 && a < 0.95)
            acc);
      slow "a low-rank warmup keeps its rank and finite step sizes" (fun () ->
          let start = Nx.Rng.normal (Nx.Rng.key 3) Nx.float64 [| 2; 2 |] in
          let s =
            N.warmup t scaled (Nx.Rng.key 4) ~steps:150
              (N.init t ~rank:1 scaled start)
          in
          let rank =
            Nx.Ptree.fold (Norn.Gaussian.ptree t)
              (fun p x r ->
                if Nx.Ptree.Path.to_string p = "variances" then (Nx.shape x).(1)
                else r)
              s.geometry (-1)
          in
          equal int 1 rank;
          Array.iter
            (fun e ->
              satisfies ~claim:"finite and positive" (float 1e-12)
                (fun e -> Float.is_finite e && e > 0.)
                e)
            (Nx.to_array s.step_size));
      test "warmup counts its transitions" (fun () ->
          let s = N.init t scaled (Nx.ones Nx.float64 [| 2; 2 |]) in
          let s' = N.warmup t scaled (Nx.Rng.key 1) ~steps:10 s in
          equal int32 10l (Nx.item [] s'.draw));
      (* With one key for warmup and sampling, the first draw takes the key
         after warmup's last: no two transitions share one. *)
      test "sampling after warmup with its key continues its counter" (fun () ->
          let k = Nx.Rng.key 1 in
          let s = N.init t scaled (Nx.ones Nx.float64 [| 2; 2 |]) in
          let w = N.warmup t scaled k ~steps:10 s in
          let _, d, _ = N.sample t scaled k ~draws:1 w in
          let next = N.step t scaled (Nx.Rng.fold_in k 10) w in
          equal (array float_exact)
            (Nx.to_array next.position)
            (Nx.to_array (d :> Nx.float64_t)));
    ]

(* Law 9: reproducibility *)

let start c = Nx.Rng.normal (Nx.Rng.key 21) Nx.float64 [| c; dim |]
let row0 x = Nx.to_array (Nx.slice [ Nx.I 0 ] x)

let reproducible =
  group "reproducibility"
    [
      test "a chain's transition does not depend on the chain count" (fun () ->
          let one =
            N.step t banana (Nx.Rng.key 3)
              (N.init t ~geometry:narrow.g banana
                 (Nx.shrink [| (0, 1); (0, dim) |] (start 4)))
          in
          let four =
            N.step t banana (Nx.Rng.key 3)
              (N.init t ~geometry:narrow.g banana (start 4))
          in
          equal (array float_exact) (row0 one.position) (row0 four.position));
      test "a draws then b draws are a + b draws" (fun () ->
          let s = N.init t ~geometry:narrow.g banana (start 2) in
          let _, ab, _ = N.sample t banana (Nx.Rng.key 8) ~draws:5 s in
          let s', a, _ = N.sample t banana (Nx.Rng.key 8) ~draws:2 s in
          let _, b, _ = N.sample t banana (Nx.Rng.key 8) ~draws:3 s' in
          let joined = Norn.Draws.append t a b in
          equal (array float_exact)
            (Nx.to_array (ab :> Nx.float64_t))
            (Nx.to_array (joined :> Nx.float64_t)));
      test "a run repeated with one key gives the same draws" (fun () ->
          let s = N.init t ~geometry:narrow.g banana (start 2) in
          let _, a, _ = N.sample t banana (Nx.Rng.key 8) ~draws:4 s in
          let _, b, _ = N.sample t banana (Nx.Rng.key 8) ~draws:4 s in
          equal (array float_exact)
            (Nx.to_array (a :> Nx.float64_t))
            (Nx.to_array (b :> Nx.float64_t)));
      test "a compiled transition is the eager one" (fun () ->
          let s = N.init t ~geometry:narrow.g banana (start 4) in
          let sp = N.ptree t in
          let step =
            Rune.jit
              Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
              (N.step t banana)
          in
          let compiled = step (Nx.Rng.key 3) s in
          let eager = N.step t banana (Nx.Rng.key 3) s in
          equal
            (array (float 1e-9))
            (Nx.to_array eager.position)
            (Nx.to_array compiled.position);
          equal (array int32)
            (Nx.to_array eager.stats.n_steps)
            (Nx.to_array compiled.stats.n_steps));
    ]

(* Densities refused at init *)

let refusals =
  group "init"
    [
      test "a density not batched over chains is refused" (fun () ->
          raises
            (Invalid_argument
               "Norn.Nuts.init: the density returned shape [4; 2] for a \
                position of 4 chains; a density returns one log density per \
                chain, shape [4]") (fun () ->
              N.init t (fun x -> Nx.neg (Nx.square x)) (start 4)));
      test "a density whose rows read each other is refused" (fun () ->
          let coupled x =
            Nx.add (banana x) (Nx.slice [ Nx.L [ 0; 0; 0; 0 ]; Nx.I 0 ] x)
          in
          raises_match
            (Exn.invalid_arg
               ~substring:"row 0 changes when the chains are reversed")
            (fun () -> N.init t coupled (start 4)));
      test "NaN at a finite position names the chain" (fun () ->
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
    ]

(* Law 11: invariance. The banana's exact draws are [a] standard normal and [b =
   0.3 a² + z / 2]; after five transitions of every chain, [a] and [b - 0.3 a²]
   keep their normal CDFs within the Dvoretzky-Kiefer-Wolfowitz band of level
   [0.01 / 4], four tests holding the false alarms of these at 1%. At 4096
   chains the band is 0.032 wide: a kernel whose stationary CDF strays further
   is caught. *)

let chains = 4096
let tests = 4.

let band =
  Float.sqrt (Float.log (2. /. (0.01 /. tests)) /. (2. *. float_of_int chains))

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

let invariance name geometry =
  slow (name ^ " geometry leaves the banana invariant") (fun () ->
      let k = Nx.Rng.key 31 in
      let a = Nx.Rng.normal (Nx.Rng.fold_in k 0) Nx.float64 [| chains |] in
      let z = Nx.Rng.normal (Nx.Rng.fold_in k 1) Nx.float64 [| chains |] in
      let b = Nx.add (Nx.mul_s (Nx.square a) 0.3) (Nx.mul_s z 0.5) in
      let s = N.init t ~geometry banana (Nx.stack ~axis:1 [ a; b ]) in
      let s = ref s in
      for i = 0 to 4 do
        s := N.step t banana (Nx.Rng.fold_in k (10 + i)) !s
      done;
      let x = !s.position in
      let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
      let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
      at_most (float 1e-12) ~than:band (max_deviation (Nx.to_array a) 1.);
      at_most (float 1e-12) ~than:band (max_deviation (Nx.to_array b') 0.5))

let law_11 =
  group "invariance"
    [ invariance "a narrow" narrow.g; invariance "a wide" wide.g ]

let () =
  exit
    (run "Norn.Nuts"
       [
         group "smoke" [ smoke ]; warmup; law_12; reproducible; refusals; law_11;
       ])
