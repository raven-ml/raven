(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

let f64 = Nx.float64
let w = Barlow.v1_weights

(* The objective as v1.0's step writes it: every correlation a P × P matrix of
   its two projections, each whitened alone. *)
let reference (w : Barlow.weights) ~alpha ~perm z1 z2 zm =
  let n = Float.of_int (Nx.dim 0 z1) and p = Nx.dim 1 z1 in
  let whiten z =
    let std = Nx.std ~axes:[ 0 ] ~keepdims:true ~ddof:1 z in
    Nx.div
      (Nx.sub z (Nx.mean ~axes:[ 0 ] ~keepdims:true z))
      (Nx.maximum std (Nx.scalar f64 1e-9))
  in
  let c x y =
    Nx.div_s (Nx.matmul (Nx.matrix_transpose (whiten x)) (whiten y)) n
  in
  let z2p = Nx.take ~axis:0 perm z2 in
  let c12 = c z1 z2 in
  let eye = Nx.eye f64 p in
  let on =
    Nx.sum (Nx.square (Nx.sub (Nx.diagonal c12) (Nx.ones f64 [| p |])))
  in
  let off = Nx.sum (Nx.mul (Nx.rsub_s 1. eye) (Nx.square c12)) in
  let mixed a b = Nx.add (Nx.mul alpha a) (Nx.mul (Nx.rsub_s 1. alpha) b) in
  let ca = mixed (c z1 z1) (c z2p z1) and cb = mixed (c z1 z2p) (c z2p z2p) in
  let sq x = Nx.sum (Nx.square x) in
  let mix = Nx.add (sq (Nx.sub (c zm z1) ca)) (sq (Nx.sub (c zm z2p) cb)) in
  Nx.item []
    (Nx.add (Nx.add on (Nx.mul_s off w.off_diagonal)) (Nx.mul_s mix w.mixup))

(* Projections of [b] rows and [p] columns, [constant] of them equal in every
   row, a mixing weight and a permutation. *)
let projections =
  Gen.(
    map
      (fun ((b, p), (constant, seed)) -> (b, p, min constant p, seed))
      (pair (pair (int_range 2 12) (int_range 2 12)) (pair (int_range 0 2) int)))

let draw (b, p, constant, seed) =
  let key = Nx.Rng.key seed in
  let z i =
    let z = Nx.Rng.normal ~key:(Nx.Rng.fold_in key i) f64 [| b; p |] in
    let cols = Nx.reshape [| 1; p |] (Nx.arange Nx.int64 0 p 1) in
    Nx.where
      (Nx.less_s (Nx.broadcast_to [| b; p |] cols) (Int64.of_int constant))
      (Nx.full f64 [| b; p |] 3.)
      z
  in
  let alpha = Nx.Rng.uniform ~key:(Nx.Rng.fold_in key 4) f64 [||] in
  let perm =
    Nx.Rng.permute ~key:(Nx.Rng.fold_in key 5) ~n:b (Nx.arange Nx.int64 0 b 1)
  in
  (alpha, perm, z 1, z 2, z 3)

let relative a b = Float.abs (a -. b) /. Float.max (Float.abs b) 1e-300

let forms =
  prop "both forms of the objective are its definition" ~count:200 projections
    (fun ((b, p, constant, _) as d) ->
      cover "B < P" (b < p);
      cover "B > P" (b > p);
      cover "a constant column" (constant > 0);
      let alpha, perm, z1, z2, zm = draw d in
      let r = reference w ~alpha ~perm z1 z2 zm in
      List.iter
        (fun form ->
          let x =
            Nx.item [] (Barlow.objective_in form w ~alpha ~perm z1 z2 zm)
          in
          at_most float_exact ~than:1e-10 (relative x r))
        [ Barlow.Gram; Barlow.Correlation ])

let refusals =
  test "the objective refuses projections of other shapes" (fun () ->
      let z = Nx.zeros f64 [| 4; 3 |] and alpha = Nx.scalar f64 0.5 in
      let perm = Nx.arange Nx.int64 0 4 1 in
      raises
        (Invalid_argument
           "Tessera.Barlow.objective: projections of different shapes")
        (fun () ->
          Barlow.objective w ~alpha ~perm z (Nx.zeros f64 [| 4; 2 |]) z);
      raises
        (Invalid_argument
           "Tessera.Barlow.objective: a permutation of [4] expected") (fun () ->
          Barlow.objective w ~alpha ~perm:(Nx.arange Nx.int64 0 3 1) z z z);
      let one = Nx.zeros f64 [| 1; 3 |] in
      raises
        (Invalid_argument "Tessera.Barlow.objective: a group of fewer than 2")
        (fun () ->
          Barlow.objective w ~alpha
            ~perm:(Nx.zeros Nx.int64 [| 1 |])
            one one one))

(* Projectors *)

(* Groups of [b] embeddings of [d] values, through [hidden] normalised
   layers. *)
let embeddings =
  Gen.(pair (pair (int_range 2 9) (int_range 1 6)) (pair (int_range 1 3) int))

(* A projector of [hidden] layers of [d] values whose linear layers are the
   identity, so every product is exact and only batch normalisation computes,
   with drawn scales and shifts. *)
let normalising ~key ~d ~hidden =
  let identity () =
    {
      Kaun.Linear.w = Nx.eye Nx.float32 d;
      b = Some (Nx.zeros Nx.float32 [| d |]);
    }
  in
  let bn i =
    let k = Nx.Rng.fold_in key (100 + i) in
    {
      Kaun.Batch_norm.gamma = Nx.Rng.normal ~key:k Nx.float32 [| d |];
      beta = Nx.Rng.normal ~key:(Nx.Rng.fold_in k 1) Nx.float32 [| d |];
    }
  in
  {
    Barlow.hidden = List.init hidden (fun i -> (identity (), bn i));
    out = identity ();
  }

let relu x = Nx.maximum x (Nx.scalar_like x 0.)

let batch_norm =
  prop "the projector normalises each value over the group's batch" ~count:50
    embeddings (fun ((b, d), (hidden, seed)) ->
      let key = Nx.Rng.key seed in
      let p = normalising ~key ~d ~hidden in
      let e = Nx.Rng.normal ~key:(Nx.Rng.fold_in key 7) Nx.float32 [| b; d |] in
      (* Training mode's batch statistics; the running ones are not read. *)
      let stats =
        {
          Kaun.Batch_norm.Stats.mean = Nx.zeros Nx.float32 [| d |];
          var = Nx.ones Nx.float32 [| d |];
        }
      in
      let expected =
        List.fold_left
          (fun x (_, bn) ->
            relu (fst (Kaun.Batch_norm.apply bn stats ~training:true x)))
          e p.hidden
      in
      (* Exact products: the same standardisation of the same values. *)
      equal (array float_exact) (Nx.to_array expected)
        (Nx.to_array (Barlow.project p e)))

(* Nx.standardize's stated bound (nx.mli) on an element of exact value [y] of a
   float32 lane of [n ≤ 1024] elements, after the affine: [|γ| ((F / 2s + 11u/2)
   |y| + (L + D + 1) u A / √s) + u |γ y| + u |γ y + β|], [s] the exact variance
   [V] plus eps, [A] the lane's mean absolute deviation, and [F], [L] and [D]
   nx.cpu's for the moments of the lane's elements (nx_cpu.mli, Moments), with
   [μ] the lane's mean and [M] its mean magnitude:

   [L = ⌈n / 16⌉ + 3], [D = 0], [E = (L + D) u M + u |μ|],

   [F = (L + D + 5) u V + 2 (L + D + 1) u E √V + (3L + 3D + 7) u E²].

   Computed in float64 from the lane's float32 values. *)
let standardize_bound ~eps (bn : _ Kaun.Batch_norm.t) x =
  let u = Float.ldexp 1. (-24) in
  let n = Nx.dim 0 x in
  let l = Float.of_int (((min n 1024 + 15) / 16) + 3) and dd = 0. in
  let x = Nx.cast Nx.float64 x in
  let col f = Nx.mean ~axes:[ 0 ] ~keepdims:true f in
  let mu = col x in
  let dev = Nx.sub x mu in
  let v = col (Nx.square dev) and a = col (Nx.abs dev) in
  let s = Nx.add_s v eps in
  let y = Nx.div dev (Nx.sqrt s) in
  let g = Nx.cast Nx.float64 bn.gamma and bt = Nx.cast Nx.float64 bn.beta in
  let e =
    Nx.add (Nx.mul_s (col (Nx.abs x)) ((l +. dd) *. u)) (Nx.mul_s (Nx.abs mu) u)
  in
  let f =
    Nx.add
      (Nx.mul_s v ((l +. dd +. 5.) *. u))
      (Nx.add
         (Nx.mul_s (Nx.mul e (Nx.sqrt v)) (2. *. (l +. dd +. 1.) *. u))
         (Nx.mul_s (Nx.square e) (((3. *. l) +. (3. *. dd) +. 7.) *. u)))
  in
  let rel = Nx.add_s (Nx.div f (Nx.mul_s s 2.)) (5.5 *. u) in
  let own =
    Nx.add
      (Nx.mul rel (Nx.abs y))
      (Nx.mul_s (Nx.div a (Nx.sqrt s)) ((l +. dd +. 1.) *. u))
  in
  let gy = Nx.mul g y in
  Nx.add
    (Nx.mul (Nx.abs g) own)
    (Nx.mul_s (Nx.add (Nx.abs gy) (Nx.abs (Nx.add gy bt))) u)

let permuted =
  prop "permuting the group permutes its normalised values" ~count:200
    embeddings (fun ((b, d), (_, seed)) ->
      let key = Nx.Rng.key seed in
      (* One layer: its inputs are the group's values, so the two results differ
         by the order of the statistics' sums alone, each within
         Nx.standardize's bound of the exact value. *)
      let p = normalising ~key ~d ~hidden:1 in
      let e = Nx.Rng.normal ~key:(Nx.Rng.fold_in key 7) Nx.float32 [| b; d |] in
      let perm =
        Nx.Rng.permute ~key:(Nx.Rng.fold_in key 8) ~n:b
          (Nx.arange Nx.int64 0 b 1)
      in
      let got = Nx.take ~axis:0 perm (Barlow.project p e) in
      let other = Barlow.project p (Nx.take ~axis:0 perm e) in
      let bound =
        Nx.mul_s (standardize_bound ~eps:1e-5 (snd (List.hd p.hidden)) e) 2.
      in
      let excess =
        Nx.sub
          (Nx.abs (Nx.cast Nx.float64 (Nx.sub got other)))
          (Nx.take ~axis:0 perm bound)
      in
      at_most float_exact ~than:0. (Nx.item [] (Nx.max excess)))

(* Views and steps *)

let arch pre_ln =
  { Model.pre_ln; width = 16; heads = 2; layers = 2; ffn = 24; dims = 8 }

let model pre = Model.init ~key:(Nx.Rng.key (if pre then 2 else 1)) (arch pre)
let key = Nx.Rng.key 9
let length = 3

(* Pixels of one to five observations of each sensor. *)
let pixels n seed =
  Synth.pixels
    {
      Synth.s2 = List.init n (fun i -> 1 + (((i * 7) + seed) mod 5));
      s1 = List.init n (fun i -> 1 + (((i * 3) + seed) mod 5));
      seed;
    }

let tokens (b : Nx.host Batch.t) =
  (Nx.to_array b.s2.tokens, Nx.to_array b.s1.tokens)

let rows n (b : Nx.host Batch.t) ~from =
  let rows (r : Nx.host Batch.rows) =
    Nx.to_array (Nx.slice [ Nx.R (from, from + n) ] r.tokens)
  in
  (rows b.s2, rows b.s1)

let keyed =
  test "a group's draws are functions of the key and the positions" (fun () ->
      let px = pixels 6 3 in
      let whole = Barlow.group ~key ~length ~first:10 px in
      let tail =
        Barlow.group ~key ~length ~first:13 (Pixels.sub px ~pos:3 ~len:3)
      in
      let pair = pair (array float_exact) (array float_exact) in
      (* Pixel j's views and keys are those of its position. *)
      equal pair (rows 3 whole.a1 ~from:3) (tokens tail.a1);
      equal pair (rows 3 whole.a2 ~from:3) (tokens tail.a2);
      equal (array int32)
        (Nx.to_array
           (Nx.slice [ Nx.A; Nx.R (3, 6) ] (Nx.Rng.to_tensor whole.keys)))
        (Nx.to_array (Nx.Rng.to_tensor tail.keys));
      (* The mixing draws are the first position's, whatever the pixels. *)
      let other = Barlow.group ~key ~length ~first:10 (pixels 6 4) in
      equal float_exact (Nx.item [] whole.alpha) (Nx.item [] other.alpha);
      equal (array int64) (Nx.to_array whole.perm) (Nx.to_array other.perm);
      let next = Barlow.group ~key ~length ~first:11 px in
      not_equal float_exact (Nx.item [] whole.alpha) (Nx.item [] next.alpha);
      equal (array int64)
        [| 0L; 1L; 2L; 3L; 4L; 5L |]
        (Array.of_list
           (List.sort compare (Array.to_list (Nx.to_array whole.perm))));
      not_equal pair (tokens whole.a1) (tokens whole.a2))

let mixup =
  prop "a mixup of weight 1 or 0 is one of its views" ~count:10 Gen.bool
    (fun pre ->
      cover "the teacher's form" pre;
      let m = model pre in
      let g = Barlow.group ~key ~length ~first:0 (pixels 5 1) in
      let keys = Nx.Rng.of_tensor (Nx.get [ 0 ] (Nx.Rng.to_tensor g.keys)) in
      let train i =
        Model.train ~dropout:Model.no_dropout ~remat:false ~keys m i
      in
      let i1 = Model.inputs m g.a1 and i2 = Model.inputs m g.a2 in
      let mix a = Model.mixup ~alpha:(Nx.scalar Nx.float32 a) i1 i2 in
      let bits x = Array.map Int32.bits_of_float (Nx.to_array x) in
      equal (array int32) (bits (train i1)) (bits (train (mix 1.)));
      equal (array int32) (bits (train i2)) (bits (train (mix 0.))))

let leaves s x =
  Nx.Ptree.fold s
    (fun _ a acc -> Nx.to_array (Nx.cast Nx.float64 a) :: acc)
    x []

(* [a] and [b] agree to [1e-5] of their largest magnitude. *)
let close a b =
  let worst = ref 0. and scale = ref 0. in
  List.iter2
    (fun a b ->
      Array.iteri
        (fun i x ->
          worst := Float.max !worst (Float.abs (x -. b.(i)));
          scale := Float.max !scale (Float.abs x))
        a)
    a b;
  at_most float_exact ~than:(1e-5 *. !scale) !worst

let grouping =
  prop "a step is the mean of its groups' objectives" ~count:4 Gen.bool
    (fun pre ->
      cover "the teacher's form" pre;
      let m = model pre in
      let pr = Barlow.projector ~key:(Nx.Rng.key 3) [ 8; 12; 10 ] in
      let st = Barlow.init m pr in
      let px = pixels 8 2 in
      let g first =
        Barlow.group ~key ~length ~first (Pixels.sub px ~pos:first ~len:4)
      in
      let half = { Model.layers = 0.5; reducer = 0.5 } in
      let step gs =
        Barlow.step ~compute:Nx.float32 w ~dropout:half ~remat:false
          ~lr:(Vega.rate 1e-3) ~weight_decay:(Vega.rate 0.) ~clip:1e9
          (Barlow.stack gs, st)
      in
      let l12, s12 = step [ g 0; g 4 ] and l21, s21 = step [ g 4; g 0 ] in
      let l1, s1 = step [ g 0 ] and l2, s2 = step [ g 4 ] in
      let p = Nx.Ptree.instantiate (module Barlow.Params) in
      let mu (s : (_, _) Barlow.state) = leaves p s.opt.mu in
      let mean a b = List.map2 (Array.map2 (fun x y -> (x +. y) /. 2.)) a b in
      let item x = [ [| Nx.item [] (Nx.cast Nx.float64 x) |] ] in
      close (item l12) (mean (item l1) (item l2));
      close (mu s12) (mean (mu s1) (mu s2));
      close (item l12) (item l21);
      close (mu s12) (mu s21))

let part =
  test "a state's model reads back alone" (fun () ->
      let m = model true in
      let st =
        Barlow.init m (Barlow.projector ~key:(Nx.Rng.key 3) [ 8; 12; 10 ])
      in
      let file = Filename.temp_file "tessera" ".safetensors" in
      Fun.protect
        ~finally:(fun () -> Sys.remove file)
        (fun () ->
          Checkpoint.save file
            (module Barlow.State)
            st ~key ~position:7 ~run:"part";
          (* Read into another model of the same structure. *)
          let like = Model.init ~key:(Nx.Rng.key 11) (arch true) in
          let back = Checkpoint.part file (module Model) ~path:"model" ~like in
          let s = Nx.Ptree.instantiate (module Model) in
          let bits m =
            Nx.Ptree.fold s
              (fun _ a acc ->
                Array.map Int32.bits_of_float
                  (Nx.to_array (Nx.cast Nx.float32 a))
                :: acc)
              m []
          in
          equal (list (array int32)) (bits m) (bits back);
          not_equal (list (array int32)) (bits like) (bits back)))

let () =
  exit
    (run "tessera.barlow"
       [
         group "objective" [ forms; refusals ];
         group "projector" [ batch_norm; permuted ];
         group "step" [ keyed; mixup; grouping; part ];
       ])
