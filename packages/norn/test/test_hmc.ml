(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module H = Norn.Hmc

let t = Nx.Ptree.tensor
let dim = 2

(* A correlated, skewed target: a banana. *)
let banana x =
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
  Nx.mul_s (Nx.add (Nx.square a) (Nx.mul_s (Nx.square b') 4.)) (-0.5)

let diagonal scale =
  Norn.Gaussian.diagonal t Nx.float64
    ~mean:(Nx.create Nx.float64 [| dim |] [| 0.1; -0.2 |])
    ~scale:(Nx.create Nx.float64 [| dim |] scale)

(* A geometry narrower than the target, so that a unit step is short, and one
   wider, so that trajectories diverge and are rejected. *)
let narrow = diagonal [| 0.4; 0.2 |]
let wide = diagonal [| 3.; 1.5 |]
let start c = Nx.Rng.normal (Nx.Rng.key 21) Nx.float64 [| c; dim |]
let row0 x = Nx.to_array (Nx.slice [ Nx.I 0 ] x)

(* Law 9: reproducibility *)

let reproducible =
  group "reproducibility"
    [
      test "a chain's transition does not depend on the chain count" (fun () ->
          let one =
            H.step t banana (Nx.Rng.key 3)
              (H.init t ~geometry:narrow banana
                 (Nx.slice [ Nx.R (0, 1); Nx.R (0, dim) ] (start 4)))
          in
          let four =
            H.step t banana (Nx.Rng.key 3)
              (H.init t ~geometry:narrow banana (start 4))
          in
          equal (array float_exact) (row0 one.position) (row0 four.position));
      test "a draws then b draws are a + b draws" (fun () ->
          let s = H.init t ~geometry:narrow banana (start 2) in
          let _, ab, _ = H.sample t banana (Nx.Rng.key 8) ~draws:5 s in
          let s', a, _ = H.sample t banana (Nx.Rng.key 8) ~draws:2 s in
          let _, b, _ = H.sample t banana (Nx.Rng.key 8) ~draws:3 s' in
          let joined = Norn.Draws.append t a b in
          equal (array float_exact)
            (Nx.to_array (ab :> Nx.float64_t))
            (Nx.to_array (joined :> Nx.float64_t)));
      test "a run repeated with one key gives the same draws" (fun () ->
          let s = H.init t ~geometry:narrow banana (start 2) in
          let _, a, _ = H.sample t banana (Nx.Rng.key 8) ~draws:4 s in
          let _, b, _ = H.sample t banana (Nx.Rng.key 8) ~draws:4 s in
          equal (array float_exact)
            (Nx.to_array (a :> Nx.float64_t))
            (Nx.to_array (b :> Nx.float64_t)));
      test "a compiled transition is the eager one" (fun () ->
          let s = H.init t ~geometry:narrow banana (start 4) in
          let sp = H.ptree t in
          let step =
            Rune.jit
              Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
              (H.step t banana)
          in
          let compiled = step (Nx.Rng.key 3) s in
          let eager = H.step t banana (Nx.Rng.key 3) s in
          equal
            (array (float 1e-9))
            (Nx.to_array eager.position)
            (Nx.to_array compiled.position);
          equal (array int32)
            (Nx.to_array eager.stats.n_steps)
            (Nx.to_array compiled.stats.n_steps));
      test "warmup counts its transitions" (fun () ->
          let s = H.init t banana (start 2) in
          let s' = H.warmup t banana (Nx.Rng.key 1) ~steps:10 s in
          equal int32 10l (Nx.item [] s'.draw));
    ]

(* Transitions *)

let transitions =
  group "transitions"
    [
      test "every chain takes the same number of steps" (fun () ->
          let s = H.init t ~geometry:narrow banana (start 8) in
          let s = H.warmup t banana (Nx.Rng.key 2) ~steps:30 s in
          let s' = H.step t banana (Nx.Rng.key 4) s in
          let n = Nx.to_array s'.stats.n_steps in
          equal (array int32) (Array.make 8 n.(0)) n);
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
          equal (array bool) [| true; true; true |]
            (Nx.to_array s'.stats.diverging);
          equal (array float_exact) [| 0.; 0.; 0. |]
            (Nx.to_array s'.stats.acceptance);
          equal (array float_exact) (Nx.to_array x) (Nx.to_array s'.position));
    ]

(* Warmup: an independent normal of standard deviations 10 and 0.1. *)

let scaled x =
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  Nx.mul_s
    (Nx.add (Nx.square (Nx.div_s a 10.)) (Nx.square (Nx.div_s b 0.1)))
    (-0.5)

let warmed =
  lazy
    (let start = Nx.Rng.normal (Nx.Rng.key 3) Nx.float64 [| 64; 2 |] in
     H.warmup t scaled (Nx.Rng.key 4) ~steps:300 (H.init t scaled start))

let warmup =
  group "warmup"
    [
      slow "the geometry's scale reaches the target's" (fun () ->
          let s = Lazy.force warmed in
          let sd = Nx.sqrt (Norn.Gaussian.variance t s.geometry) in
          let ratio =
            Nx.to_array
              (Nx.div sd (Nx.create Nx.float64 [| 2 |] [| 10.; 0.1 |]))
          in
          Array.iter
            (fun r ->
              satisfies ~claim:"within a factor of 2" (float 1e-12)
                (fun r -> r > 0.5 && r < 2.)
                r)
            ratio);
      slow "the step size gives about the target acceptance" (fun () ->
          let s = Lazy.force warmed in
          let _, _, st = H.sample t scaled (Nx.Rng.key 5) ~draws:100 s in
          let acc =
            Nx.item [] (Nx.mean (st :> Nx.float64_elt Norn.Stats.t).acceptance)
          in
          satisfies ~claim:"in (0.65, 0.95)" (float 1e-12)
            (fun a -> a > 0.65 && a < 0.95)
            acc);
      (* On a standard normal the flow is a rotation, [x' = x cos t + p sin t],
         and the expected squared change of [|x|²] is [4 d sin² t]. Over times
         [t = u T], [u] uniform in (0, 2), its mean [1 - sin 4T / 4T] peaks at
         [4T = tan 4T], [T = 1.1234]. Rounding the step count up lengthens a
         trajectory by half a step on average, so the length plus half a step is
         near the optimum when the step is small. *)
      slow "the length reaches the expected jump's optimum" (fun () ->
          let d = 10 in
          let normal x = Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.square x)) (-0.5) in
          let start = Nx.Rng.normal (Nx.Rng.key 3) Nx.float64 [| 64; d |] in
          let s =
            H.warmup t normal (Nx.Rng.key 4) ~steps:300
              (H.init t ~accept:0.95 normal start)
          in
          let reach = Nx.item [] s.length +. (Nx.item [] s.step_size /. 2.) in
          equal (float_rel ~rel:0.25 ~abs:0.) 1.1234 reach);
      test "sampling after warmup with its key continues its counter" (fun () ->
          let k = Nx.Rng.key 1 in
          let s = H.init t scaled (Nx.ones Nx.float64 [| 2; 2 |]) in
          let w = H.warmup t scaled k ~steps:10 s in
          let _, d, _ = H.sample t scaled k ~draws:1 w in
          let next = H.step t scaled (Nx.Rng.fold_in k 10) w in
          equal (array float_exact)
            (Nx.to_array next.position)
            (Nx.to_array (d :> Nx.float64_t)));
    ]

(* Densities refused at init *)

let refusals =
  group "init"
    [
      test "a density not batched over chains is refused" (fun () ->
          raises
            (Invalid_argument
               "Norn.Hmc.init: the density returned shape [4; 2] for a \
                position of 4 chains; a density returns one log density per \
                chain, shape [4]") (fun () ->
              H.init t (fun x -> Nx.neg (Nx.square x)) (start 4)));
      test "a density whose rows read each other is refused" (fun () ->
          let coupled x =
            Nx.add (banana x) (Nx.slice [ Nx.L [ 0; 0; 0; 0 ]; Nx.I 0 ] x)
          in
          raises_match
            (Exn.invalid_arg
               ~substring:"row 0 changes when the chains are reversed")
            (fun () -> H.init t coupled (start 4)));
      test "an acceptance outside (0, 1) is refused" (fun () ->
          raises (Invalid_argument "Norn.Hmc.init: accept = 1 is not in (0, 1)")
            (fun () -> H.init t ~accept:1. banana (start 2)));
    ]

(* Law 11: invariance. The banana's exact draws are [a] standard normal and [b =
   0.3 a² + z / 2]; after five transitions of every chain with fixed tuning, [a]
   and [b - 0.3 a²] keep their normal CDFs within the Dvoretzky-Kiefer-Wolfowitz
   band of level [0.01 / 4], four tests holding the false alarms of these at 1%.
   At 4096 chains the band is 0.032 wide. *)

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
      let s = ref (H.init t ~geometry banana (Nx.stack ~axis:1 [ a; b ])) in
      for i = 0 to 4 do
        s := H.step t banana (Nx.Rng.fold_in k (10 + i)) !s
      done;
      let x = !s.position in
      let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
      let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
      at_most (float 1e-12) ~than:band (max_deviation (Nx.to_array a) 1.);
      at_most (float 1e-12) ~than:band (max_deviation (Nx.to_array b') 0.5))

let law_11 =
  group "invariance" [ invariance "a narrow" narrow; invariance "a wide" wide ]

let () =
  exit (run "Norn.Hmc" [ reproducible; transitions; warmup; refusals; law_11 ])
