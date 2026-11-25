(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module E = Norn.Ensemble

let t = Nx.Ptree.tensor
let dim = 2

(* A correlated, skewed target: a banana. Its exact draws are [a] standard
   normal and [b = 0.3 a² + z / 2]. *)
let banana x =
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
  Nx.mul_s (Nx.add (Nx.square a) (Nx.mul_s (Nx.square b') 4.)) (-0.5)

(* A standard normal over a batch of walkers. *)
let normal x = Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.square x)) (-0.5)
let start c = Nx.Rng.normal ~key:(Nx.Rng.key 21) Nx.float64 [| c; dim |]
let floats x = Nx.to_array x

let draws (d : Nx.host Nx.float64_t Norn.Draws.t) =
  floats (d :> Nx.host Nx.float64_t)

(* init *)

let init =
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
      test "fewer than one ensemble is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Ensemble.init: ensembles")
            (fun () -> E.init t ~ensembles:0 banana (start 8)));
      test "a density not batched over walkers is refused" (fun () ->
          raises
            (Invalid_argument
               "Norn.Ensemble.init: the density returned shape [8; 2] for a \
                position of 8 chains; a density returns one log density per \
                chain, shape [8]") (fun () ->
              E.init t (fun x -> Nx.neg (Nx.square x)) (start 8)));
      test "a density whose rows read each other is refused, naming the row"
        (fun () ->
          let coupled x =
            Nx.add (banana x)
              (Nx.slice [ Nx.L (List.init 8 (fun _ -> 0)); Nx.I 0 ] x)
          in
          raises_match
            (Exn.invalid_arg
               ~sub:"Norn.Ensemble.init: the density's row 0 changes")
            (fun () -> E.init t coupled (start 8)));
      test "a start where the density is -inf is refused, naming the walker"
        (fun () ->
          let walled x =
            let a = Nx.slice [ Nx.A; Nx.I 0 ] x in
            Nx.where (Nx.greater_s a 50.)
              (Nx.full_like a Float.neg_infinity)
              (banana x)
          in
          let x =
            Nx.set [ Nx.I 1; Nx.I 0 ] (Nx.scalar Nx.float64 100.) (start 8)
          in
          raises
            (Invalid_argument
               "Norn.Ensemble.init: the density is -inf at chain 1; a chain \
                starts where its density is finite") (fun () ->
              E.init t walled x));
      test "a tensor that is not a float is refused, naming its path" (fun () ->
          let u = Nx.Ptree.(pair tensor tensor) in
          let x = (start 8, Nx.zeros Nx.int32 [| 8 |]) in
          raises_match (Exn.invalid_arg ~sub:"Norn.Ensemble.init: 1 is int32")
            (fun () -> E.init u (fun (x, _) -> banana x) x));
    ]

(* step

   The stretch move from the .mli's description and its Keys paragraph, walker
   by walker. *)

let row x i = floats (Nx.slice [ Nx.I i ] x)
let banana_at y = Nx.item [ 0 ] (banana (Nx.create Nx.float64 [| 1; dim |] y))
let uniform key = Nx.item [] (Nx.Rng.uniform ~key Nx.float64 [||])

(* [reference ~ensembles key x] is a transition of the walkers [x] in
   [ensembles] ensembles: their positions, log densities and acceptance
   probabilities, and whether a move was rejected. *)
let reference ~ensembles key x =
  let c = (Nx.shape x).(0) in
  let n = c / ensembles in
  let keys = Nx.Rng.split ~n:c key in
  let pos = Array.init c (row x) in
  let lp = Array.map banana_at pos in
  let alpha = Array.make c 0. and rejected = ref false in
  for e = 0 to ensembles - 1 do
    let first = e * n in
    let order =
      Array.map
        (fun i -> first + Int64.to_int i)
        (floats
           (Nx.argsort
              (Nx.Rng.uniform
                 ~key:(Nx.Rng.fold_in keys.(first) 2)
                 Nx.float32 [| n |])))
    in
    let halves =
      [| Array.sub order 0 (n / 2); Array.sub order (n / 2) (n - (n / 2)) |]
    in
    for h = 0 to 1 do
      let moving = halves.(h) and held = halves.(1 - h) in
      let o = Array.length held in
      let move w =
        let u m = uniform (Nx.Rng.fold_in (Nx.Rng.fold_in keys.(w) h) m) in
        let partner =
          pos.(held.(min (o - 1) (Float.to_int (float_of_int o *. u 0))))
        in
        let z = ((1. +. u 1) ** 2.) /. 2. in
        let y =
          Array.mapi (fun i p -> p +. (z *. (pos.(w).(i) -. p))) partner
        in
        let ly = banana_at y in
        let ratio = (float_of_int (dim - 1) *. Float.log z) +. ly -. lp.(w) in
        (w, y, ly, ratio, Float.log (u 2) < ratio)
      in
      (* Every walker of the half moves from the positions before the half. *)
      let moves = Array.map move moving in
      Array.iter
        (fun (w, y, ly, ratio, accept) ->
          alpha.(w) <- Float.min 1. (Float.exp ratio);
          if accept then begin
            pos.(w) <- y;
            lp.(w) <- ly
          end
          else rejected := true)
        moves
    done
  done;
  (pos, lp, alpha, !rejected)

(* The 40 values of (ensembles, their size, key) are checked whole, each a
   transition. *)
let stretch =
  prop ~count:40 "a transition is the stretch move"
    Gen.(triple (int_range 1 2) (int_range 4 5) (int_range 0 9))
  @@ fun (ensembles, size, seed) ->
  let key = Nx.Rng.key (1000 + seed) in
  let x =
    Nx.Rng.normal ~key:(Nx.Rng.fold_in key 7) Nx.float64
      [| ensembles * size; dim |]
  in
  let s = E.step t banana key (E.init t ~ensembles banana x) in
  let pos, lp, alpha, rejected = reference ~ensembles key x in
  cover "a move is rejected" rejected;
  cover "an ensemble has an odd size" (size mod 2 = 1);
  equal
    (array (float 1e-12))
    (Array.concat (Array.to_list pos))
    (floats s.position);
  equal (array (float 1e-12)) lp (floats s.lp);
  equal (array (float 1e-12)) alpha (floats s.stats.acceptance)

(* Invariance. From the banana's exact draws, five transitions keep [a] and [b -
   0.3 a²] within the Dvoretzky-Kiefer-Wolfowitz band of level [0.01 / 2] of
   their normal CDFs, two comparisons holding the false alarms at 1%. The
   stretch move leaves the product of the target over an ensemble invariant, so
   the walkers stay independent draws. *)

let invariance_walkers = 4096

let band =
  Float.sqrt
    (Float.log (2. /. (0.01 /. 2.)) /. (2. *. float_of_int invariance_walkers))

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

let invariant () =
  let k = Nx.Rng.key 31 in
  let a =
    Nx.Rng.normal ~key:(Nx.Rng.fold_in k 0) Nx.float64 [| invariance_walkers |]
  in
  let z =
    Nx.Rng.normal ~key:(Nx.Rng.fold_in k 1) Nx.float64 [| invariance_walkers |]
  in
  let b = Nx.add (Nx.mul_s (Nx.square a) 0.3) (Nx.mul_s z 0.5) in
  let s = ref (E.init t ~ensembles:64 banana (Nx.stack ~axis:1 [ a; b ])) in
  for i = 0 to 4 do
    s := E.step t banana (Nx.Rng.fold_in k (10 + i)) !s
  done;
  let x = !s.position in
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  let b' = Nx.sub b (Nx.mul_s (Nx.square a) 0.3) in
  at_most (float 1e-12) ~than:band (max_deviation (floats a) 1.);
  at_most (float 1e-12) ~than:band (max_deviation (floats b') 0.5)

(* An affine map [y = A x + b], [A = R diag (s) Rᵀ] with [R] a rotation by
   [theta] and [log s] uniform in [[-2.3, 2.3]]. *)
type affine = { theta : float; log_s : float * float; shift : float * float }

let affine =
  Gen.(
    let+ theta = float_range 0. Float.pi
    and+ log_s = pair (float_range (-2.3) 2.3) (float_range (-2.3) 2.3)
    and+ shift = pair (float_range (-5.) 5.) (float_range (-5.) 5.) in
    { theta; log_s; shift })

(* [matrix ~power f] is [R diag (s^power) Rᵀ]: [A] at [1], [A⁻¹] at [-1]. *)
let matrix ~power f =
  let c = Float.cos f.theta and s = Float.sin f.theta in
  let s1 = Float.exp (power *. fst f.log_s)
  and s2 = Float.exp (power *. snd f.log_s) in
  Nx.create Nx.float64 [| 2; 2 |]
    [|
      (c *. c *. s1) +. (s *. s *. s2);
      c *. s *. (s1 -. s2);
      c *. s *. (s1 -. s2);
      (s *. s *. s1) +. (c *. c *. s2);
    |]

let shift f = Nx.create Nx.float64 [| 1; dim |] [| fst f.shift; snd f.shift |]
let apply f x = Nx.add (Nx.matmul x (matrix ~power:1. f)) (shift f)

(* [three lp x] is the walkers [x] after three transitions on [lp]. *)
let three lp x =
  let s = ref (E.init t lp x) in
  for i = 0 to 2 do
    s := E.step t lp (Nx.Rng.fold_in (Nx.Rng.key 9) i) !s
  done;
  !s.position

let step =
  group "step"
    [
      stretch;
      (* Goodman and Weare: the move is unchanged by an affine map. Three
         transitions on [banana ∘ A⁻¹ (· - b)] from [A x + b] end at [A x_3 +
         b]. Rounding moves a decision only where [log u] lies within about
         1e-13 of its ratio: about 1e-10 per run. The cover's chance is 1/4 a
         case, so 40 cases miss it at 1e-5: the example holds it. *)
      prop ~count:40
        ~examples:[ { theta = 0.; log_s = (-2.3, 2.3); shift = (0., 0.) } ]
        "a move is unchanged by an affine map of the coordinates" affine
        (fun f ->
          cover "one axis is stretched 10 times the other"
            (Float.abs (fst f.log_s -. snd f.log_s) >= Float.log 10.);
          let mapped y =
            banana (Nx.matmul (Nx.sub y (shift f)) (matrix ~power:(-1.) f))
          in
          let x = start 8 in
          let expected = floats (apply f (three banana x)) in
          let scale =
            Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. expected
          in
          equal
            (array (float_rel ~rel:1e-9 ~abs:(1e-9 *. scale)))
            expected
            (floats (three mapped (apply f x))));
      test "a move to a point of zero density is never taken" (fun () ->
          (* A half-plane: x_0 > 0. *)
          let half x =
            let a = Nx.slice [ Nx.A; Nx.I 0 ] x in
            Nx.where (Nx.greater_s a 0.) (normal x)
              (Nx.full_like a Float.neg_infinity)
          in
          let x = Nx.abs (start 8) in
          let _, d, _ =
            E.sample t half (Nx.Rng.key 6) ~draws:20 (E.init t half x)
          in
          let a = Nx.slice [ Nx.A; Nx.A; Nx.I 0 ] (d :> Nx.host Nx.float64_t) in
          Array.iter (fun a -> greater float_exact ~than:0. a) (floats a));
      test "the density is called twice, on each half of every ensemble"
        (fun () ->
          let x = Nx.concatenate ~axis:0 [ start 8; start 8 ] in
          let s = E.init t ~ensembles:2 banana x in
          let rows = ref [] in
          let counted x =
            rows := (Nx.shape x).(0) :: !rows;
            banana x
          in
          ignore (E.step t counted (Nx.Rng.key 4) s);
          equal (list int) [ 8; 8 ] !rows);
      test "a walker's transition does not depend on the number of ensembles"
        (fun () ->
          let one = E.init t banana (start 8) in
          let two =
            E.init t ~ensembles:2 banana
              (Nx.concatenate ~axis:0 [ start 8; Nx.mul_s (start 8) 2. ])
          in
          let _, a, _ = E.sample t banana (Nx.Rng.key 3) ~draws:3 one in
          let _, b, _ = E.sample t banana (Nx.Rng.key 3) ~draws:3 two in
          equal (array float_exact) (draws a)
            (floats (Nx.slice [ Nx.R (0, 8) ] (b :> Nx.host Nx.float64_t))));
      (* Walker 5 starts at x0 = 1e6, the others in [0, 1]. Its moves land past
         5e5 + 1, where the density is [v] off its start, unless its stretch is
         within 1e-6 of 1/2; a move of another walker toward it ends below 5e5 +
         0.5. *)
      cases "a density NaN or +inf at a walker is refused, naming it"
        ~id:Float.to_string [ Float.nan; Float.infinity ] (fun v ->
          let lp x =
            let a = Nx.slice [ Nx.A; Nx.I 0 ] x in
            let off =
              Nx.logical_and (Nx.greater_s a 500_001.) (Nx.not_equal_s a 1e6)
            in
            Nx.where off (Nx.full_like a v) (normal x)
          in
          let x = Nx.abs (start 8) in
          let x = Nx.set [ Nx.I 5; Nx.I 0 ] (Nx.scalar Nx.float64 1e6) x in
          raises
            (Invalid_argument
               (Printf.sprintf
                  "Norn.Ensemble.step: the density is %s at chain 5, a finite \
                   position"
                  (if Float.is_nan v then "nan" else "inf")))
            (fun () -> E.step t lp (Nx.Rng.key 4) (E.init t lp x)));
      (* norn.mli's Ensemble: [d] is a walker's float elements, every float
         tensor taken together. *)
      test "a structure moves as its float elements taken together" (fun () ->
          let x = start 8 in
          let split x =
            (Nx.slice [ Nx.A; Nx.R (0, 1) ] x, Nx.slice [ Nx.A; Nx.R (1, 2) ] x)
          in
          let joined (a, b) = Nx.concatenate ~axis:1 [ a; b ] in
          let u = Nx.Ptree.(pair tensor tensor) in
          let lp p = banana (joined p) in
          let pair = E.step u lp (Nx.Rng.key 3) (E.init u lp (split x)) in
          let one = E.step t banana (Nx.Rng.key 3) (E.init t banana x) in
          equal (array float_exact) (floats one.position)
            (floats (joined pair.position)));
      prop ~count:10 "a compiled transition is the eager one"
        Gen.(int_range 0 9)
        (fun seed ->
          Compiled.here ();
          let s = E.init t banana (start 8) in
          let sp = E.ptree t in
          let step =
            Rune.jit
              Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
              (E.step t banana)
          in
          let k = Nx.Rng.key seed in
          let compiled = step k s in
          let eager = E.step t banana k s in
          (* Fused kernels round otherwise; a decision near its threshold would
             flip, at a rate of about 1e-13 per decision. *)
          equal
            (array (float 1e-9))
            (floats eager.position) (floats compiled.position);
          equal
            (array (float 1e-9))
            (floats eager.stats.acceptance)
            (floats compiled.stats.acceptance);
          equal int eager.ensembles compiled.ensembles);
      test "five transitions leave the banana invariant" invariant;
    ]

(* warmup *)

let warmup =
  group "warmup"
    [
      test "a negative number of steps is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Ensemble.warmup: steps")
            (fun () ->
              E.warmup t banana (Nx.Rng.key 1) ~steps:(-1)
                (E.init t banana (start 8))));
      (* Nothing is tuned: warmup is sampling whose draws are discarded. *)
      test "warmup's transitions are sample's" (fun () ->
          let s = E.init t banana (start 8) in
          let w = E.warmup t banana (Nx.Rng.key 1) ~steps:3 s in
          let s', _, _ = E.sample t banana (Nx.Rng.key 1) ~draws:3 s in
          equal int32 3l (Nx.item [] w.draw);
          equal (array float_exact) (floats s'.position) (floats w.position));
    ]

(* sample *)

module M = Norn_model
module R = Recovery

let sample =
  group "sample"
    [
      test "fewer than one draw is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Ensemble.sample: draws")
            (fun () ->
              E.sample t banana (Nx.Rng.key 1) ~draws:0
                (E.init t banana (start 8))));
      test "a draws then b draws are a + b draws" (fun () ->
          let s = E.init t banana (start 8) in
          let _, ab, _ = E.sample t banana (Nx.Rng.key 8) ~draws:5 s in
          let s', a, _ = E.sample t banana (Nx.Rng.key 8) ~draws:2 s in
          let _, b, _ = E.sample t banana (Nx.Rng.key 8) ~draws:3 s' in
          equal (array float_exact) (draws ab) (draws (Norn.Draws.append t a b)));
      (* Two ensembles of 32 walkers, at least twice the coordinates, each a
         superchain of nested R-hat. Walkers spread over a region of another
         shape than a narrow, correlated posterior come back to it only over
         many transitions, so they start where HMC leaves them after 600
         transitions. R-hat 1.011 on one of keys 1 to 40 at 500 draws. *)
      test "draws recover the dogs posterior" (fun () ->
          match R.posterior "dogs-dogs" with
          | Posteriordb.Posterior p as post ->
              let m = p.model in
              let u = M.coords m and lp = M.log_density m p.y in
              let h = Norn.Hmc.init u lp (M.init m p.y ~chains:64 R.key) in
              let h =
                Norn.Hmc.warmup u lp (Nx.Rng.fold_in R.key 1) ~steps:600 h
              in
              let s = E.init u ~ensembles:2 lp h.position in
              let s = E.warmup u lp R.key ~steps:300 s in
              let _, d, _ = E.sample u lp R.key ~draws:800 s in
              R.chains post ~superchains:2
                (Norn.Draws.map u Posteriordb.latent (M.constrain m) d));
    ]

let () = exit (run "Norn.Ensemble" [ init; step; warmup; sample ])
