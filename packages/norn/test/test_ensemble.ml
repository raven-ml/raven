(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module E = Norn.Ensemble

let t = Nx.Ptree.tensor
let dim = 2

(* A correlated, skewed target: a banana. *)
let banana x =
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
  Nx.mul_s (Nx.add (Nx.square a) (Nx.mul_s (Nx.square b') 4.)) (-0.5)

let start c = Nx.Rng.normal (Nx.Rng.key 21) Nx.float64 [| c; dim |]
let floats x = Nx.to_array x

(* Law 9: reproducibility *)

let reproducible =
  group "reproducibility"
    [
      test "a walker's draws do not depend on the number of ensembles"
        (fun () ->
          let one = E.init t banana (start 8) in
          let two =
            E.init t ~ensembles:2 banana
              (Nx.concatenate ~axis:0 [ start 8; Nx.mul_s (start 8) 2. ])
          in
          let _, a, _ = E.sample t banana (Nx.Rng.key 3) ~draws:3 one in
          let _, b, _ = E.sample t banana (Nx.Rng.key 3) ~draws:3 two in
          equal (array float_exact)
            (floats (a :> Nx.float64_t))
            (floats (Nx.slice [ Nx.R (0, 8) ] (b :> Nx.float64_t))));
      test "a draws then b draws are a + b draws" (fun () ->
          let s = E.init t banana (start 8) in
          let _, ab, _ = E.sample t banana (Nx.Rng.key 8) ~draws:5 s in
          let s', a, _ = E.sample t banana (Nx.Rng.key 8) ~draws:2 s in
          let _, b, _ = E.sample t banana (Nx.Rng.key 8) ~draws:3 s' in
          equal (array float_exact)
            (floats (ab :> Nx.float64_t))
            (floats (Norn.Draws.append t a b :> Nx.float64_t)));
      test "a run repeated with one key gives the same draws" (fun () ->
          let s = E.init t banana (start 8) in
          let _, a, _ = E.sample t banana (Nx.Rng.key 8) ~draws:4 s in
          let _, b, _ = E.sample t banana (Nx.Rng.key 8) ~draws:4 s in
          equal (array float_exact)
            (floats (a :> Nx.float64_t))
            (floats (b :> Nx.float64_t)));
      test "a compiled transition is the eager one" (fun () ->
          let s = E.init t banana (start 8) in
          let sp = E.ptree t in
          let step =
            Rune.jit
              Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
              (E.step t banana)
          in
          let compiled = step (Nx.Rng.key 3) s in
          let eager = E.step t banana (Nx.Rng.key 3) s in
          equal
            (array (float 1e-9))
            (floats eager.position) (floats compiled.position);
          equal (array int32)
            (Nx.to_array eager.stats.evaluations)
            (Nx.to_array compiled.stats.evaluations));
      test "warmup counts its transitions" (fun () ->
          let s =
            E.warmup t banana (Nx.Rng.key 1) ~steps:3
              (E.init t banana (start 8))
          in
          equal int32 3l (Nx.item [] s.draw));
    ]

(* Transitions *)

(* A walker 10^3 sds from a standard normal whose other walkers lie on it moves
   along directions of the normal's scale. The bracket doubles to the slice's
   length in about ten trips; stepping out, at most 63 widths a transition,
   takes 59 transitions on this key. Over 40 keys the walker returns in 11 to
   31. *)
let straggler =
  test "a walker 10^3 sds from the posterior returns" (fun () ->
      let lp x = Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.square x)) (-0.5) in
      let x = Nx.Rng.normal (Nx.Rng.key 5) Nx.float64 [| 8; dim |] in
      let x = Nx.set [ Nx.I 0; Nx.I 0 ] (Nx.scalar Nx.float64 1e3) x in
      let s = E.warmup t lp (Nx.Rng.key 6) ~steps:40 (E.init t lp x) in
      at_most float_exact ~than:6. (Nx.item [] (Nx.max (Nx.abs s.position))))

let transitions =
  group "transitions"
    [
      test "a move evaluates the first bracket's ends and a point in it"
        (fun () ->
          let s = E.step t banana (Nx.Rng.key 4) (E.init t banana (start 8)) in
          Array.iter
            (fun n -> at_least int32 ~than:3l n)
            (Nx.to_array s.stats.evaluations));
      test "the density sees only the moving half of every ensemble" (fun () ->
          let x = Nx.concatenate ~axis:0 [ start 8; start 8 ] in
          let s = E.init t ~ensembles:2 banana x in
          let rows = ref [] in
          let counted x =
            rows := (Nx.shape x).(0) :: !rows;
            banana x
          in
          ignore (E.step t counted (Nx.Rng.key 4) s);
          equal (list int) (List.map (fun _ -> 8) !rows) !rows);
      straggler;
      test "the log density is the density's at the new position" (fun () ->
          let s = E.step t banana (Nx.Rng.key 4) (E.init t banana (start 8)) in
          equal (array (float 1e-12)) (floats (banana s.position)) (floats s.lp));
      test "a walker never leaves the support" (fun () ->
          (* A half-plane: x_0 > 0. *)
          let half x =
            let a = Nx.slice [ Nx.A; Nx.I 0 ] x in
            Nx.where (Nx.greater_s a 0.)
              (Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.square x)) (-0.5))
              (Nx.full_like a Float.neg_infinity)
          in
          let x = Nx.abs (start 8) in
          let _, d, _ =
            E.sample t half (Nx.Rng.key 6) ~draws:20 (E.init t half x)
          in
          let a = Nx.slice [ Nx.A; Nx.A; Nx.I 0 ] (d :> Nx.float64_t) in
          Array.iter (fun a -> greater float_exact ~than:0. a) (floats a));
    ]

(* Refusals *)

let refusals =
  group "init"
    [
      test "an ensemble needs twice as many walkers as coordinates" (fun () ->
          raises
            (Invalid_argument
               "Norn.Ensemble.init: an ensemble of 3 walkers over 2 \
                coordinates; an ensemble needs at least twice as many walkers \
                as coordinates") (fun () -> E.init t banana (start 3)));
      test "walkers split into equal ensembles" (fun () ->
          raises
            (Invalid_argument
               "Norn.Ensemble.init: 10 walkers do not split into 3 ensembles")
            (fun () -> E.init t ~ensembles:3 banana (start 10)));
      test "a density not batched over walkers is refused" (fun () ->
          raises
            (Invalid_argument
               "Norn.Ensemble.init: the density returned shape [8; 2] for a \
                position of 8 chains; a density returns one log density per \
                chain, shape [8]") (fun () ->
              E.init t (fun x -> Nx.neg (Nx.square x)) (start 8)));
    ]

(* Law 11: invariance. From exact draws of a target, five transitions keep each
   tested coordinate within the Dvoretzky-Kiefer-Wolfowitz band of level [0.01 /
   3] of its CDF, three comparisons holding the false alarms at 1%. *)

let chains = 4096

let band =
  Float.sqrt (Float.log (2. /. (0.01 /. 3.)) /. (2. *. float_of_int chains))

(* [max_deviation cdf xs] is the largest distance between [cdf] and the ECDF of
   [xs]. *)
let max_deviation cdf xs =
  let xs = Array.copy xs in
  Array.sort Float.compare xs;
  let n = float_of_int (Array.length xs) in
  let worst = ref 0. in
  Array.iteri
    (fun i x ->
      let f = cdf x in
      worst :=
        Float.max !worst
          (Float.max
             (Float.abs (f -. (float_of_int i /. n)))
             (Float.abs (f -. (float_of_int (i + 1) /. n)))))
    xs;
  !worst

let normal_cdf sd x = Nx.item [] (Nx.ndtr (Nx.scalar Nx.float64 (x /. sd)))

(* The banana's exact draws: [a] standard normal and [b = 0.3 a² + z / 2]. *)
let law_11 =
  slow "five transitions leave the banana invariant" (fun () ->
      let k = Nx.Rng.key 31 in
      let a = Nx.Rng.normal (Nx.Rng.fold_in k 0) Nx.float64 [| chains |] in
      let z = Nx.Rng.normal (Nx.Rng.fold_in k 1) Nx.float64 [| chains |] in
      let b = Nx.add (Nx.mul_s (Nx.square a) 0.3) (Nx.mul_s z 0.5) in
      let s = ref (E.init t ~ensembles:64 banana (Nx.stack ~axis:1 [ a; b ])) in
      for i = 0 to 4 do
        s := E.step t banana (Nx.Rng.fold_in k (10 + i)) !s
      done;
      let x = !s.position in
      let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
      let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
      at_most (float 1e-12) ~than:band
        (max_deviation (normal_cdf 1.) (floats a));
      at_most (float 1e-12) ~than:band
        (max_deviation (normal_cdf 0.5) (floats b')))

(* Two modes of unequal weight and width on a line: [0.8 N(-3, 1) + 0.2 N(3,
   0.1²)]. In ensembles of two walkers the held half is one point, whose
   Gaussian is degenerate, so every bracket doubles dozens of times, and a slice
   through both modes is two intervals: the acceptance check keeps the move
   reversible. Without it the ECDF strays 0.065 from the CDF. *)
let modes =
  slow "five transitions of two-walker ensembles leave two modes invariant"
    (fun () ->
      let mode m sd w x =
        Nx.add_s
          (Nx.mul_s (Nx.square (Nx.sub_s x m)) (-0.5 /. (sd *. sd)))
          (Float.log (w /. sd))
      in
      let lp x =
        let a = Nx.slice [ Nx.A; Nx.I 0 ] x in
        Nx.logsumexp ~axes:[ 0 ]
          (Nx.stack ~axis:0 [ mode (-3.) 1. 0.8 a; mode 3. 0.1 0.2 a ])
      in
      let cdf x =
        (0.8 *. normal_cdf 1. (x +. 3.)) +. (0.2 *. normal_cdf 0.1 (x -. 3.))
      in
      let k = Nx.Rng.key 32 in
      let first =
        Nx.less_s
          (Nx.Rng.uniform (Nx.Rng.fold_in k 0) Nx.float64 [| chains |])
          0.8
      in
      let at a b =
        Nx.where first
          (Nx.full Nx.float64 [| chains |] a)
          (Nx.full Nx.float64 [| chains |] b)
      in
      let z = Nx.Rng.normal (Nx.Rng.fold_in k 1) Nx.float64 [| chains |] in
      let x =
        Nx.reshape [| chains; 1 |] (Nx.add (Nx.mul z (at 1. 0.1)) (at (-3.) 3.))
      in
      let s = ref (E.init t ~ensembles:(chains / 2) lp x) in
      for i = 0 to 4 do
        s := E.step t lp (Nx.Rng.fold_in k (10 + i)) !s
      done;
      at_most (float 1e-12) ~than:band (max_deviation cdf (floats !s.position)))

let () =
  exit
    (run "Norn.Ensemble"
       [
         reproducible;
         transitions;
         refusals;
         group "invariance" [ law_11; modes ];
       ])
