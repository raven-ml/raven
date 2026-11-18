(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

let f64 = Nx.float64
let w = Barlow.v1_weights

(* The objective's terms and its value as v1.0's step writes them, at the
   weights [w], each beside its absolute form. Every correlation is a P × P
   matrix of two projections, each whitened alone by Barlow's operations, so
   to Barlow's values, and z₂ₚ's are w₂'s rows permuted, as Barlow's are.

   Both forms and the reference compute each term as a polynomial in the
   whitened values whose monomials pass at most [R] roundings ({!roundings}),
   so each is within [γ(R)] times its absolute form of the exact term, the
   same polynomial over [|w|] with every difference a sum (Nx_kernel's
   bounds: a term that passes [r] roundings is within [γ(r)] of itself). The
   Gram form's absolute forms bound the others': its off-diagonal sum,
   [Σ (w₁ w₁ᵀ) ⊙ (w₂ w₂ᵀ) / B²] less the diagonal's squares, has
   [|a|² + Σᵢ aᵢᵢ²], [a = |w₁|ᵀ |w₂| / B], which can exceed [|c|²] without
   bound, and its mixup's, through [d = w_m − alpha w₁ − (1 − alpha) w₂ₚ], has
   [Dᵀ |w| / B] in place of each [c (z_m, z) − c_t], [D = |w_m| + alpha |w₁| +
   (1 − alpha) |w₂ₚ|], the reference's own. *)
let reference (w : Barlow.weights) ~alpha ~perm z1 z2 zm =
  let n = Float.of_int (Nx.dim 0 z1) and p = Nx.dim 1 z1 in
  let whiten z =
    let std = Nx.std ~axes:[ 0 ] ~keepdims:true ~ddof:1 z in
    Nx.div
      (Nx.sub z (Nx.mean ~axes:[ 0 ] ~keepdims:true z))
      (Nx.maximum std (Nx.scalar f64 1e-9))
  in
  let w1 = whiten z1 and w2 = whiten z2 and wm = whiten zm in
  let w2p = Nx.take ~axis:0 perm w2 in
  let c x y = Nx.div_s (Nx.matmul (Nx.matrix_transpose x) y) n in
  let c12 = c w1 w2 in
  let eye = Nx.eye f64 p in
  let on =
    Nx.sum (Nx.square (Nx.sub (Nx.diagonal c12) (Nx.ones f64 [| p |])))
  in
  let off = Nx.sum (Nx.mul (Nx.rsub_s 1. eye) (Nx.square c12)) in
  let mixed a b = Nx.add (Nx.mul alpha a) (Nx.mul (Nx.rsub_s 1. alpha) b) in
  let ca = mixed (c w1 w1) (c w2p w1) and cb = mixed (c w1 w2p) (c w2p w2p) in
  let sq x = Nx.sum (Nx.square x) in
  let mix = Nx.add (sq (Nx.sub (c wm w1) ca)) (sq (Nx.sub (c wm w2p) cb)) in
  let total =
    Nx.add (Nx.add on (Nx.mul_s off w.off_diagonal)) (Nx.mul_s mix w.mixup)
  in
  let a1 = Nx.abs w1 and a2p = Nx.abs w2p in
  let a12 = c a1 (Nx.abs w2) and d = Nx.add (Nx.abs wm) (mixed a1 a2p) in
  let on' = sq (Nx.add_s (Nx.diagonal a12) 1.) in
  let off' = Nx.add (sq a12) (sq (Nx.diagonal a12)) in
  let mix' = Nx.add (sq (c d a1)) (sq (c d a2p)) in
  let total' =
    Nx.add (Nx.add on' (Nx.mul_s off' w.off_diagonal)) (Nx.mul_s mix' w.mixup)
  in
  List.map (fun (x, a) -> (Nx.item [] x, Nx.item [] a))
    [ (on, on'); (off, off'); (mix, mix'); (total, total') ]

(* [R], the roundings a monomial of a term passes in either form or the
   reference, at most. A sum or a product of [k] terms costs [β(k)]
   (Nx_kernel's bounds, [β(k) = ⌈5 min(k, 256) / 4⌉ + ⌈k / 128⌉]), one for
   its products and one into its destination. A Gram monomial passes two
   products of [P] and a sum of [B²] terms, a P × P one two products of [B]
   and a sum of [P²] terms, and either at most 15 elementwise roundings: five
   on each factor of the reference's mixup squares (the division by [B], the
   target's weight, product and sum, the difference), the square, the sum of
   the two norms, and the objective's weight and additions. *)
let roundings b p =
  let k n = ((5 * min n 256) + 3) / 4 + ((n + 127) / 128) + 2 in
  max (k (b * b) + (2 * k p)) (k (p * p) + (2 * k b)) + 15

(* The terms, then the objective. *)
let values (w : Barlow.weights) (t : _ Barlow.terms) =
  List.map (Nx.item [])
    [ t.on_diagonal; t.off_diagonal; t.mixup; Barlow.objective w t ]

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

(* Weights unlike each other, so a law at them sees the two exchanged. *)
let distinct : Barlow.weights = { off_diagonal = 5e-3; mixup = 2e-2 }

let forms =
  prop "both forms of the objective and its terms are their definition"
    ~count:200 projections (fun ((b, p, constant, _) as d) ->
      cover "B < P" (b < p);
      cover "B > P" (b > p);
      cover "a constant column" (constant > 0);
      let alpha, perm, z1, z2, zm = draw d in
      let w = distinct in
      let r = reference w ~alpha ~perm z1 z2 zm in
      let ru = Float.of_int (roundings b p) *. Float.epsilon /. 2. in
      let gamma = ru /. (1. -. ru) in
      List.iter
        (fun form ->
          let x = values w (Barlow.terms_in form ~alpha ~perm z1 z2 zm) in
          (* Each, and the reference, within [γ(R)] times its absolute form
             of the exact term. *)
          List.iter2
            (fun x (r, a) ->
              at_most float_exact ~than:(2. *. gamma *. a)
                (Float.abs (x -. r)))
            x r)
        [ Barlow.Gram; Barlow.Correlation ])

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
      beta = Some (Nx.Rng.normal ~key:(Nx.Rng.fold_in k 1) Nx.float32 [| d |]);
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
  let g = Nx.cast Nx.float64 bn.gamma in
  let bt = Nx.cast Nx.float64 (Option.get bn.beta) in
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
        Model.train ~dropout:Model.no_dropout ~keys m i
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
  prop "a step's terms are the mean of its groups'" ~count:4 Gen.bool
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
        Barlow.step ~compute:Nx.float32 w ~dropout:half
          ~lr:(Vega.rate 1e-3) ~weight_decay:(Vega.rate 0.) ~clip:1e9
          (Barlow.stack gs, st)
      in
      let l12, s12 = step [ g 0; g 4 ] and l21, s21 = step [ g 4; g 0 ] in
      let l1, s1 = step [ g 0 ] and l2, s2 = step [ g 4 ] in
      let p = Nx.Ptree.instantiate (module Barlow.Params) in
      let mu (s : (_, _) Barlow.state) = leaves p s.opt.mu in
      let mean a b = List.map2 (Array.map2 (fun x y -> (x +. y) /. 2.)) a b in
      let item t = List.map (fun x -> [| x |]) (values w t) in
      close (item l12) (mean (item l1) (item l2));
      close (mu s12) (mean (mu s1) (mu s2));
      close (item l12) (item l21);
      close (mu s12) (mu s21))

let torch_layout (p : _ Barlow.projector) =
  let linear i (l : _ Kaun.Linear.t) =
    [
      (Printf.sprintf "net.%d.weight" i, Nx.P (Nx.matrix_transpose l.w));
      (Printf.sprintf "net.%d.bias" i, Nx.P (Option.get l.b));
    ]
  in
  let norm i (n : _ Kaun.Batch_norm.t) =
    [
      (Printf.sprintf "net.%d.weight" i, Nx.P n.gamma);
      (Printf.sprintf "net.%d.bias" i, Nx.P (Option.get n.beta));
    ]
  in
  let hidden k (l, n) = linear (3 * k) l @ norm ((3 * k) + 1) n in
  List.concat (List.mapi hidden p.hidden)
  @ linear (3 * List.length p.hidden) p.out

let with_npz entries f =
  let file = Filename.temp_file "tessera" ".npz" in
  Fun.protect ~finally:(fun () -> Sys.remove file) @@ fun () ->
  Nx_io.save_npz file (Nx_io.Archive.of_list entries);
  f file

let projector_files =
  [
    test "a projector loads from PyTorch's layout, net.10 after net.9"
      (fun () ->
        (* Five linear layers: modules 0 to 12, read in numeric order. *)
        let p = Barlow.projector ~key:(Nx.Rng.key 5) [ 4; 5; 6; 7; 8; 3 ] in
        let entries = torch_layout p in
        with_npz entries @@ fun file ->
        let arrays l =
          List.map
            (fun (n, Nx.P x) ->
              (n, Nx.shape x, Nx.to_array (Nx.cast Nx.float64 x)))
            l
        in
        equal
          (list (triple string (array int) (array float_exact)))
          (arrays entries)
          (arrays (torch_layout (Barlow.load_projector file))));
    test "a projector refuses another sequence of modules, naming its file"
      (fun () ->
        (* A linear layer, then another: its batch norm, net.1, left out. *)
        let p = Barlow.projector ~key:(Nx.Rng.key 5) [ 4; 5; 3 ] in
        let entries =
          List.filter
            (fun (name, _) -> not (String.starts_with ~prefix:"net.1." name))
            (torch_layout p)
        in
        with_npz entries @@ fun file ->
        raises
          (Failure
             ("Tessera.Barlow.load_projector: " ^ file
            ^ ": not a projector of linear and batch-norm layers"))
          (fun () -> Barlow.load_projector file));
  ]

(* Schedules *)

(* v1.0's trainer's rate at step [s] (tessera-util@290feaf
   src/train_multi_gpu_FSDP_QAT.py:179-191, adjust_single_group_lr). *)
let upstream ~lr ~warmup ~steps s =
  let w = Float.to_int (Float.of_int steps *. warmup) in
  if s < w then if w > 0 then lr *. Float.of_int s /. Float.of_int w else lr
  else if steps <= w then lr
  else
    let p = Float.of_int (s - w) /. Float.of_int (steps - w) in
    let p = Float.max 0. (Float.min 1. p) in
    lr *. 0.5 *. (1. +. Float.cos (Float.pi *. p))

let schedules =
  prop "a schedule is v1.0's trainer's rate at every step" ~count:100
    Gen.(
      triple
        (frequency [ (1, constant 0.); (4, float_range 0. 1.2) ])
        (int_range 0 60) (float_range 1e-4 1.))
    (fun (warmup, steps, lr) ->
      let w = Float.to_int (warmup *. Float.of_int steps) in
      cover "no warmup step" (w = 0 && steps > 0);
      let exact = warmup *. Float.of_int steps in
      cover "a warmup that rounds up" (Float.round exact > Float.of_int w);
      cover "a warmup of every step" (w >= steps && steps > 0);
      let sched = Barlow.schedule ~lr ~warmup ~steps in
      for s = 0 to steps + 2 do
        equal
          ~msg:(Printf.sprintf "step %d" s)
          (float_rel ~rel:1e-5 ~abs:1e-6)
          (upstream ~lr ~warmup ~steps s)
          (Vega.Schedule.eval sched s)
      done)

let schedule_cases =
  [
    test "a schedule truncates its warmup, and may have none" (fun () ->
        (* 0.05 · 30 = 1.5: one warmup step, so step 1 is at the peak. *)
        let at ~warmup ~steps s =
          Vega.Schedule.eval (Barlow.schedule ~lr:1. ~warmup ~steps) s
        in
        equal (float 1e-7) 0. (at ~warmup:0.05 ~steps:30 0);
        equal (float 1e-7) 1. (at ~warmup:0.05 ~steps:30 1);
        (* 0.1 · 6 = 0.6: none, so step 0 is at the peak. *)
        equal (float 1e-7) 1. (at ~warmup:0.1 ~steps:6 0));
    test "a warmup's steps are truncated" (fun () ->
        equal (list int) [ 1; 0; 2; 0; 7 ]
          (List.map
             (fun (warmup, steps) -> Barlow.warmup_steps ~warmup ~steps)
             [ (0.05, 30); (0.1, 6); (0.05, 40); (0.3, 0); (1.4, 5) ]));
    cases "a schedule refuses a warmup or run it cannot count"
      ~id:(fun (w, s) -> Printf.sprintf "%g %d" w s)
      [
        (-0.1, 6); (0.1, -1); (Float.nan, 6); (Float.infinity, 6); (1e300, 6);
      ]
      (fun (warmup, steps) ->
        raises_match (Exn.invalid_arg ~sub:"Tessera.Barlow.warmup_steps")
          (fun () ->
            ignore (Barlow.schedule ~lr:1. ~warmup ~steps : _ Vega.Schedule.t)));
  ]

let () =
  exit
    (run "tessera.barlow"
       [
         group "objective" [ forms ];
         group "projector" ([ batch_norm; permuted ] @ projector_files);
         group "step" [ keyed; mixup; grouping ];
         group "schedule" (schedules :: schedule_cases);
       ])
