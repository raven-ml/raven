(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Stateful layers: batch normalization and dropout. The house pattern under
   test: trainable parameters and running statistics are separate records, a
   training forward returns (output, new stats), and the training step threads
   the stats out of the objective through value_and_grad_aux's aux channel. *)

open Windtrap
open Kaun

let t32 shape xs = Nx.create Nx.float32 shape xs
let vec xs = t32 [| Array.length xs |] xs

let check_arr ?(eps = 1e-5) ?msg expected actual =
  equal ?msg (array (float eps)) expected (Nx.to_array actual)

(* Batch norm *)

(* A fixed 4x2 batch with distinct per-feature statistics. *)
let bn_x = lazy (t32 [| 4; 2 |] [| 1.0; -2.0; 3.0; 4.0; -1.0; 10.0; 5.0; 0.0 |])

let test_bn_normalizes_batch () =
  Nx.Rng.run ~seed:1 @@ fun () ->
  let params, stats = Batch_norm.init ~features:5 in
  let x = Nx.randn Nx.float32 [| 64; 5 |] in
  let x = Nx.add_s (Nx.mul_s x 3.0) 7.0 in
  let y, _ = Batch_norm.apply params stats ~training:true x in
  check_arr ~eps:1e-4 ~msg:"per-feature mean is 0" (Array.make 5 0.0)
    (Nx.mean ~axes:[ 0 ] y);
  check_arr ~eps:1e-3 ~msg:"per-feature variance is 1" (Array.make 5 1.0)
    (Nx.var ~axes:[ 0 ] y)

let test_bn_affine_after_normalization () =
  let params, stats = Batch_norm.init ~features:2 in
  let params = { params with Batch_norm.gamma = vec [| 2.0; 0.5 |] } in
  let params = { params with Batch_norm.beta = vec [| 3.0; -1.0 |] } in
  let y, _ = Batch_norm.apply params stats ~training:true (Lazy.force bn_x) in
  check_arr ~eps:1e-4 ~msg:"mean is beta" [| 3.0; -1.0 |]
    (Nx.mean ~axes:[ 0 ] y);
  check_arr ~eps:1e-2 ~msg:"variance is gamma^2" [| 4.0; 0.25 |]
    (Nx.var ~axes:[ 0 ] y)

let test_bn_running_stats_one_step () =
  let params, stats = Batch_norm.init ~features:2 in
  let x = Lazy.force bn_x in
  let _, stats' =
    Batch_norm.apply ~momentum:0.9 params stats ~training:true x
  in
  let expect base batch =
    Array.map2 (fun b s -> (0.9 *. b) +. (0.1 *. s)) base (Nx.to_array batch)
  in
  (* Initial stats are mean 0, var 1. *)
  check_arr ~msg:"running mean"
    (expect [| 0.0; 0.0 |] (Nx.mean ~axes:[ 0 ] x))
    stats'.Batch_norm.Stats.mean;
  check_arr ~msg:"running var"
    (expect [| 1.0; 1.0 |] (Nx.var ~axes:[ 0 ] x))
    stats'.Batch_norm.Stats.var

let test_bn_running_stats_converge () =
  let params, stats = Batch_norm.init ~features:2 in
  let x = Lazy.force bn_x in
  let stats = ref stats in
  for _ = 1 to 200 do
    let _, s = Batch_norm.apply ~momentum:0.9 params !stats ~training:true x in
    stats := s
  done;
  check_arr ~eps:1e-3 ~msg:"running mean converges to batch mean"
    (Nx.to_array (Nx.mean ~axes:[ 0 ] x))
    !stats.Batch_norm.Stats.mean;
  check_arr ~eps:1e-3 ~msg:"running var converges to batch var"
    (Nx.to_array (Nx.var ~axes:[ 0 ] x))
    !stats.Batch_norm.Stats.var

let test_bn_eval_uses_running_stats () =
  let params, _ = Batch_norm.init ~features:2 in
  let params = { params with Batch_norm.gamma = vec [| 2.0; 1.0 |] } in
  let params = { params with Batch_norm.beta = vec [| 0.5; 0.0 |] } in
  let stats =
    { Batch_norm.Stats.mean = vec [| 1.0; 2.0 |]; var = vec [| 4.0; 9.0 |] }
  in
  let x = t32 [| 2; 2 |] [| 3.0; 5.0; 1.0; -1.0 |] in
  let y, stats' = Batch_norm.apply params stats ~training:false x in
  (* y = gamma * (x - mean) / sqrt (var + eps) + beta, per feature. *)
  let expected =
    let f g b m v x = (g *. (x -. m) /. sqrt (v +. 1e-5)) +. b in
    [|
      f 2.0 0.5 1.0 4.0 3.0;
      f 1.0 0.0 2.0 9.0 5.0;
      f 2.0 0.5 1.0 4.0 1.0;
      f 1.0 0.0 2.0 9.0 (-1.0);
    |]
  in
  check_arr ~msg:"analytic eval output" expected (Nx.reshape [| 4 |] y);
  is_true ~msg:"eval returns the stats unchanged" (stats' == stats)

let test_bn_grads_flow_to_params () =
  let params, stats = Batch_norm.init ~features:2 in
  let params =
    { Batch_norm.gamma = vec [| 1.3; 0.7 |]; beta = vec [| 0.2; -0.4 |] }
  in
  let x = Lazy.force bn_x in
  let f p =
    let y, _ = Batch_norm.apply p stats ~training:true x in
    Nx.sum (Nx.mul y y)
  in
  (match Rune.check_grads ~tol:0.05 (module Batch_norm) f params with
  | Ok () -> ()
  | Error msg -> failf "gradient check failed: %s" msg);
  let grads = Rune.grad (module Batch_norm) f params in
  is_true ~msg:"gamma gradient is non-zero"
    (Array.exists
       (fun g -> Float.abs g > 1e-3)
       (Nx.to_array grads.Batch_norm.gamma));
  is_true ~msg:"beta gradient is non-zero"
    (Array.exists
       (fun g -> Float.abs g > 1e-3)
       (Nx.to_array grads.Batch_norm.beta))

let test_bn_stat_update_is_detached () =
  let params, stats = Batch_norm.init ~features:2 in
  let x = Lazy.force bn_x in
  let g =
    Rune.grad'
      (fun x ->
        let _, stats' = Batch_norm.apply params stats ~training:true x in
        Nx.sum (Nx.add stats'.Batch_norm.Stats.mean stats'.Batch_norm.Stats.var))
      x
  in
  check_arr ~msg:"no gradient flows through the running stats"
    (Array.make 8 0.0) (Nx.reshape [| 8 |] g)

let test_bn_stats_checkpoint () =
  let params, stats = Batch_norm.init ~features:2 in
  let stats =
    { Batch_norm.Stats.mean = vec [| 1.0; 2.0 |]; var = vec [| 3.0; 4.0 |] }
  in
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_params (module Batch_norm) ~prefix:"bn" params;
        Checkpoint.of_params (module Batch_norm.Stats) ~prefix:"bn.stats" stats;
      ]
  in
  equal ~msg:"dot-joined names" (list string)
    [ "bn.beta"; "bn.gamma"; "bn.stats.mean"; "bn.stats.var" ]
    (Checkpoint.names ckpt);
  let _, like = Batch_norm.init ~features:2 in
  let stats' =
    Checkpoint.to_params (module Batch_norm.Stats) ~prefix:"bn.stats" ~like ckpt
  in
  check_arr ~msg:"mean round trips" [| 1.0; 2.0 |] stats'.Batch_norm.Stats.mean;
  check_arr ~msg:"var round trips" [| 3.0; 4.0 |] stats'.Batch_norm.Stats.var

let test_bn_init_validates () =
  raises_match ~msg:"features = 0"
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Batch_norm.init ~features:0)

(* The full training-step round trip: a model mixing stateless and stateful
   layers, stats threaded through value_and_grad_aux's aux channel. *)
module Model = struct
  type t = { lin : Linear.t; bn : Batch_norm.t; out : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { lin; bn; out } =
    { lin = Linear.map f lin; bn = Batch_norm.map f bn; out = Linear.map f out }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    {
      lin = Linear.map2 f p.lin q.lin;
      bn = Batch_norm.map2 f p.bn q.bn;
      out = Linear.map2 f p.out q.out;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { lin; bn; out } =
    Linear.iter f lin;
    Batch_norm.iter f bn;
    Linear.iter f out

  let forward p stats ~training x =
    let h = Linear.apply p.lin x in
    let h, stats = Batch_norm.apply p.bn stats ~training h in
    let h = Nx.tanh h in
    (Linear.apply p.out h, stats)
end

let test_bn_train_step_roundtrip () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let x = Nx.randn Nx.float32 [| 16; 2 |] in
  let y = Nx.sum ~axes:[ 1 ] ~keepdims:true (Nx.mul x x) in
  let bn, stats = Batch_norm.init ~features:8 in
  let params =
    {
      Model.lin = Linear.init ~inputs:2 ~outputs:8;
      bn;
      out = Linear.init ~inputs:8 ~outputs:1;
    }
  in
  let step (params, stats, ostate) =
    let objective p =
      let pred, stats' = Model.forward p stats ~training:true x in
      (Loss.mse pred y, stats')
    in
    let loss, grads, stats' =
      Rune.value_and_grad_aux (module Model) objective params
    in
    let params, ostate =
      Vega.adam_step (module Model) ~lr:0.02 ostate ~params ~grads
    in
    ((params, stats', ostate), Nx.item [] loss)
  in
  let state = ref (params, stats, Vega.adam_init (module Model) params) in
  let first = snd (step !state) in
  let last = ref first in
  for _ = 1 to 300 do
    let s, l = step !state in
    state := s;
    last := l
  done;
  is_true ~msg:"training decreases the loss" (!last < first *. 0.2);
  let params, stats', _ = !state in
  is_true ~msg:"running stats moved away from init"
    (Array.exists
       (fun m -> Float.abs m > 1e-3)
       (Nx.to_array stats'.Batch_norm.Stats.mean));
  (* Eval reuses the same forward; the stats come back unchanged. *)
  let pred, stats'' = Model.forward params stats' ~training:false x in
  is_true ~msg:"eval leaves the stats unchanged" (stats'' == stats');
  let eval_loss = Nx.item [] (Loss.mse pred y) in
  is_true ~msg:"eval loss is in the training loss's ballpark" (eval_loss < first)

(* Dropout *)

let test_dropout_eval_identity () =
  let x = vec [| 1.0; -2.0; 3.0; 0.0 |] in
  let y = Dropout.apply ~rate:0.9 ~training:false x in
  check_arr ~eps:0.0 ~msg:"eval mode is the identity" (Nx.to_array x) y

let test_dropout_rate_zero_identity () =
  let x = vec [| 1.0; -2.0; 3.0 |] in
  let y = Dropout.apply ~rate:0.0 ~training:true x in
  check_arr ~eps:0.0 ~msg:"rate 0 is the identity" (Nx.to_array x) y

let test_dropout_train_statistics () =
  Nx.Rng.run ~seed:11 @@ fun () ->
  let n = 10_000 in
  let rate = 0.3 in
  let x = Nx.ones Nx.float32 [| n |] in
  let y = Nx.to_array (Dropout.apply ~rate ~training:true x) in
  let scaled = 1.0 /. (1.0 -. rate) in
  let kept = ref 0 in
  Array.iteri
    (fun i v ->
      if v <> 0.0 then begin
        incr kept;
        equal
          ~msg:(Printf.sprintf "survivor %d is scaled by 1/keep" i)
          (float 1e-5) scaled v
      end)
    y;
  (* Binomial: mean 7000, sigma ~ 46; +/- 250 is over 5 sigma. *)
  is_true ~msg:"about 1 - rate of the elements survive"
    (abs (!kept - 7_000) < 250);
  let mean = Array.fold_left ( +. ) 0.0 y /. float_of_int n in
  is_true ~msg:"inverted scaling preserves the mean"
    (Float.abs (mean -. 1.0) < 0.05)

let test_dropout_validates_rate () =
  let x = vec [| 1.0 |] in
  let is_invalid = function Invalid_argument _ -> true | _ -> false in
  raises_match ~msg:"rate = 1" is_invalid (fun () ->
      Dropout.apply ~rate:1.0 ~training:true x);
  raises_match ~msg:"rate > 1" is_invalid (fun () ->
      Dropout.apply ~rate:1.5 ~training:true x);
  raises_match ~msg:"rate < 0" is_invalid (fun () ->
      Dropout.apply ~rate:(-0.1) ~training:true x);
  raises_match ~msg:"rate validated in eval mode too" is_invalid (fun () ->
      Dropout.apply ~rate:2.0 ~training:false x)

let test_dropout_deterministic_under_seed () =
  let x = vec (Array.init 100 float_of_int) in
  let run () =
    Nx.Rng.run ~seed:5 @@ fun () -> Dropout.apply ~rate:0.5 ~training:true x
  in
  check_arr ~eps:0.0 ~msg:"same seed, same mask" (Nx.to_array (run ())) (run ())

(* Keyed dropout: the mask is a pure function of the key and the shape. *)

let test_dropout_keyed_deterministic () =
  let x = vec (Array.init 100 (fun i -> float_of_int (i + 1))) in
  let apply key = Dropout.apply ~rate:0.5 ~training:true ~key x in
  check_arr ~eps:0.0 ~msg:"same key, same mask"
    (Nx.to_array (apply (Nx.Rng.key 7)))
    (apply (Nx.Rng.key 7));
  is_true ~msg:"different keys, different masks"
    (Nx.to_array (apply (Nx.Rng.key 7))
    <> Nx.to_array (apply (Nx.Rng.key 8)))

let test_dropout_keyed_statistics () =
  let n = 10_000 in
  let rate = 0.3 in
  let x = Nx.ones Nx.float32 [| n |] in
  let y =
    Nx.to_array
      (Dropout.apply ~rate ~training:true ~key:(Nx.Rng.key 11) x)
  in
  let scaled = 1.0 /. (1.0 -. rate) in
  let kept = ref 0 in
  Array.iteri
    (fun i v ->
      if v <> 0.0 then begin
        incr kept;
        equal
          ~msg:(Printf.sprintf "survivor %d is scaled by 1/keep" i)
          (float 1e-5) scaled v
      end)
    y;
  (* Binomial: mean 7000, sigma ~ 46; +/- 250 is over 5 sigma. *)
  is_true ~msg:"about 1 - rate of the elements survive"
    (abs (!kept - 7_000) < 250);
  let mean = Array.fold_left ( +. ) 0.0 y /. float_of_int n in
  is_true ~msg:"inverted scaling preserves the mean"
    (Float.abs (mean -. 1.0) < 0.05)

let test_dropout_keyed_eval_identity () =
  let x = vec [| 1.0; -2.0; 3.0; 0.0 |] in
  let y = Dropout.apply ~rate:0.9 ~training:false ~key:(Nx.Rng.key 0) x in
  check_arr ~eps:0.0 ~msg:"eval mode ignores the key" (Nx.to_array x) y

(* Under jit: the keyed form compiles, the keyless form is refused loudly. *)

let test_dropout_keyless_jit_raises () =
  let f = Rune.jit' (fun x -> Dropout.apply ~rate:0.5 ~training:true x) in
  raises_match ~msg:"implicit-scope dropout raises inside jit"
    (function Rune.Jit_error _ -> true | _ -> false)
    (fun () -> ignore (f (vec [| 1.0; 2.0; 3.0 |])))

module Keyed_x = struct
  type t = { x : Nx.float32_t; key : Nx.Rng.key }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { x = f t.x; key = f t.key }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { x = f a.x b.x; key = f a.key b.key }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    f t.x;
    f t.key
end

let test_dropout_keyed_jit_matches_eager () =
  let x = vec (Array.init 64 (fun i -> float_of_int (i + 1))) in
  let apply { Keyed_x.x; key } =
    Dropout.apply ~rate:0.5 ~training:true ~key x
  in
  let f = Rune.jit (module Keyed_x) apply in
  let k = Nx.Rng.key 21 and k' = Nx.Rng.key 22 in
  check_arr ~eps:0.0 ~msg:"jit == eager, same key"
    (Nx.to_array (apply { Keyed_x.x; key = k }))
    (f { Keyed_x.x; key = k });
  is_true ~msg:"a fresh key through the compiled step gives a fresh mask"
    (Nx.to_array (f { Keyed_x.x; key = k })
    <> Nx.to_array (f { Keyed_x.x; key = k' }))

(* Jitted training steps: a tiny MLP with dropout between its layers, the
   per-step key an input leaf derived by [fold_in] from a root seed. *)

module Mlp = struct
  type t = { w1 : Nx.float32_t; w2 : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { w1 = f t.w1; w2 = f t.w2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { w1 = f a.w1 b.w1; w2 = f a.w2 b.w2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    f t.w1;
    f t.w2
end

module Mlp_in = struct
  type t = { p : Mlp.t; key : Nx.Rng.key }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { p = Mlp.map f t.p; key = f t.key }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { p = Mlp.map2 f a.p b.p; key = f a.key b.key }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    Mlp.iter f t.p;
    f t.key
end

module Mlp_out = struct
  type t = { p : Mlp.t; loss : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { p = Mlp.map f t.p; loss = f t.loss }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { p = Mlp.map2 f a.p b.p; loss = f a.loss b.loss }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    Mlp.iter f t.p;
    f t.loss
end

let grid n = Array.init n (fun i -> 0.25 *. float_of_int ((i mod 13) - 6))
let mlp_x = t32 [| 8; 16 |] (grid 128)
let mlp_y = t32 [| 8; 4 |] (grid 32)

let mlp_init () =
  { Mlp.w1 = t32 [| 16; 32 |] (grid 512); w2 = t32 [| 32; 4 |] (grid 128) }

let mlp_objective ?key rate p =
  let h = Fn.relu (Nx.matmul mlp_x p.Mlp.w1) in
  let h = Dropout.apply ~rate ~training:true ?key h in
  Loss.mse (Nx.matmul h p.Mlp.w2) mlp_y

let mlp_update p grads =
  let upd w g = Nx.sub w (Nx.mul_s g 0.0005) in
  { Mlp.w1 = upd p.Mlp.w1 grads.Mlp.w1; w2 = upd p.Mlp.w2 grads.Mlp.w2 }

let test_dropout_jitted_training_steps () =
  let steps = 4 in
  let trajectory seed =
    let root = Nx.Rng.key seed in
    let f =
      Rune.jit2
        (module Mlp_in)
        (module Mlp_out)
        (fun { Mlp_in.p; key } ->
          let loss, grads =
            Rune.value_and_grad (module Mlp) (mlp_objective ~key 0.5) p
          in
          { Mlp_out.p = mlp_update p grads; loss })
    in
    let p = ref (mlp_init ()) in
    Array.init steps (fun i ->
        let out = f { Mlp_in.p = !p; key = Nx.Rng.fold_in root i } in
        p := out.Mlp_out.p;
        Nx.item [] out.Mlp_out.loss)
  in
  let dropout_free () =
    let f =
      Rune.jit2
        (module Mlp)
        (module Mlp_out)
        (fun p ->
          let loss, grads =
            Rune.value_and_grad (module Mlp) (mlp_objective 0.0) p
          in
          { Mlp_out.p = mlp_update p grads; loss })
    in
    let p = ref (mlp_init ()) in
    Array.init steps (fun _ ->
        let out = f !p in
        p := out.Mlp_out.p;
        Nx.item [] out.Mlp_out.loss)
  in
  let a = trajectory 42 and b = trajectory 42 in
  is_true ~msg:"same root seed, bitwise-identical trajectories"
    (Array.for_all2
       (fun x y -> Int64.bits_of_float x = Int64.bits_of_float y)
       a b);
  is_true ~msg:"dropout changes the loss trajectory" (a <> dropout_free ());
  is_true ~msg:"a different root changes the trajectory" (a <> trajectory 43)

(* Mixed precision: keyed dropout inside the astype sandwich, masks selected at
   the bfloat16 compute dtype, gradients back at float32. *)

module W = struct
  type t = Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let test_dropout_bf16_sandwich () =
  let x = t32 [| 8; 4 |] (grid 32) in
  let w = t32 [| 4; 4 |] (grid 16) in
  let objective w =
    let h = Nx.matmul (Nx.cast Nx.bfloat16 x) (Nx.cast Nx.bfloat16 w) in
    let h =
      Dropout.apply ~rate:0.5 ~training:true ~key:(Nx.Rng.key 3) h
    in
    Nx.mean (Nx.cast Nx.float32 h)
  in
  let loss, grads = Rune.value_and_grad (module W) objective w in
  is_true ~msg:"loss is finite" (Float.is_finite (Nx.item [] loss));
  is_true ~msg:"float32 gradients are finite"
    (Array.for_all Float.is_finite (Nx.to_array grads))

let tests =
  [
    group "batch norm"
      [
        test "training normalizes with batch statistics"
          test_bn_normalizes_batch;
        test "gamma and beta apply after normalization"
          test_bn_affine_after_normalization;
        test "one step blends running stats with momentum"
          test_bn_running_stats_one_step;
        test "running stats converge to the batch statistics"
          test_bn_running_stats_converge;
        test "eval mode normalizes with the running stats"
          test_bn_eval_uses_running_stats;
        test "gradients flow to gamma and beta" test_bn_grads_flow_to_params;
        test "no gradient flows through the stat update"
          test_bn_stat_update_is_detached;
        test "stats checkpoint under their own prefix" test_bn_stats_checkpoint;
        test "init rejects non-positive features" test_bn_init_validates;
        test "train step threads stats through value_and_grad_aux"
          test_bn_train_step_roundtrip;
      ];
    group "dropout"
      [
        test "eval mode is the identity" test_dropout_eval_identity;
        test "rate zero is the identity in training mode"
          test_dropout_rate_zero_identity;
        test "training keeps about 1 - rate with inverted scaling"
          test_dropout_train_statistics;
        test "invalid rates raise" test_dropout_validates_rate;
        test "fixed seed gives a deterministic mask"
          test_dropout_deterministic_under_seed;
        test "keyed masks are pure functions of the key"
          test_dropout_keyed_deterministic;
        test "keyed training keeps about 1 - rate with inverted scaling"
          test_dropout_keyed_statistics;
        test "keyed eval mode is the identity" test_dropout_keyed_eval_identity;
        test "keyless dropout raises inside jit"
          test_dropout_keyless_jit_raises;
        test "keyed dropout compiles and matches eager"
          test_dropout_keyed_jit_matches_eager;
        test "jitted train steps with fold_in keys reproduce from the seed"
          test_dropout_jitted_training_steps;
        test "keyed dropout composes with the bfloat16 astype sandwich"
          test_dropout_bf16_sandwich;
      ];
  ]

let () = run "kaun stateful" tests
