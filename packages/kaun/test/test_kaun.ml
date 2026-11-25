(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The training step of kaun.mli's preamble: a model record with one walk,
   [Rune.value_and_grad] of a sparse cross-entropy and an AdamW update, over
   [Data] minibatches. *)

open Windtrap
open Kaun

(* The unit roundoff of float64. *)
let u = Float.ldexp 1. (-53)

module Model = struct
  type ('a, 'd) t = { hidden : ('a, 'd) Linear.t; out : ('a, 'd) Linear.t }

  let walk c { hidden; out } =
    let open Nx.Ptree.Walk in
    let hidden = field c "hidden" Linear.walk hidden in
    let out = field c "out" Linear.walk out in
    { hidden; out }

  let apply p x = Linear.apply p.out (Nx.tanh (Linear.apply p.hidden x))
end

let model () = Nx.Ptree.instantiate (module Model)
let features = 4
let classes = 3

(* AdamW's rate, and the defaults of its weight decay and moments (vega.mli). *)
let lr = 1e-2
let weight_decay = 0.01
let b1 = 0.9
let b2 = 0.999
let eps = 1e-8

let step model (params, ostate) (x, y) =
  let loss p = Loss.softmax_cross_entropy_sparse (Model.apply p x) y in
  let l, grads = Rune.value_and_grad model loss params in
  let params, ostate =
    Vega.adamw_step model ~lr:(Vega.rate lr) ostate ~params ~grads
  in
  ((params, ostate), l)

let compiled state model =
  Rune.jit
    Nx.Ptree.(state @-> pair tensor tensor @-> returns (pair state tensor))
    (step model)

(* 24 examples in minibatches of 8, and the parameters, all on the host. *)
let dataset () =
  ( Nx.Rng.normal ~key:(Nx.Rng.key 1) Nx.float64 [| 24; features |],
    Nx.create Nx.int64 [| 24 |]
      (Array.init 24 (fun i -> Int64.of_int (i mod classes))) )

let initial () =
  {
    Model.hidden =
      Linear.make ~key:(Nx.Rng.key 2) ~inputs:features ~outputs:8 Nx.float64;
    out = Linear.make ~key:(Nx.Rng.key 3) ~inputs:8 ~outputs:classes Nx.float64;
  }

let two =
  fixture (fun () ->
      Nx.devices
        (List.map
           (fun n -> Result.get_ok (Rig.software n))
           [ "kaun-test-0"; "kaun-test-1" ]))

(* The elements of [p]'s leaves, in walk order. *)
let flat model p =
  Array.concat
    (List.map
       (fun (Nx.P t) -> Nx.to_array (Nx.cast Nx.float64 t))
       (fst (Nx.Ptree.flatten model p)))

(* A host copy of [s], read element by element: it makes no value over [s]'s
   memory, so [s]'s donation proceeds as without it. *)
let to_host state host_state ~like s =
  Nx.Ptree.rebuild host_state ~like
    (List.map
       (fun (Nx.P t) ->
         Nx.P (Nx.create (Nx.dtype t) (Nx.shape t) (Nx.to_array t)))
       (fst (Nx.Ptree.flatten state s)))

(* The batch gradient's elements and the mean of its examples' gradients'
   magnitudes, element by element, from each example's gradient on the host. *)
let gradient_scale model p (x, y) =
  let n = Nx.dim 0 x in
  let example i =
    let at t = Nx.slice [ Nx.R (i, i + 1) ] t in
    flat model
      (Rune.grad model
         (fun p ->
           Loss.softmax_cross_entropy_sparse (Model.apply p (at x)) (at y))
         p)
  in
  let examples = List.init n example in
  let mean f =
    Array.map
      (fun v -> v /. float_of_int n)
      (List.fold_left
         (Array.map2 (fun a g -> a +. f g))
         (Array.map (fun _ -> 0.) (List.hd examples))
         examples)
  in
  (mean Fun.id, mean Float.abs)

(* The largest [|log_sum_exp z| + max |z|] over the batch's logits [z]: the
   magnitudes an example's loss [log_sum_exp z - z_y] combines. *)
let loss_scale p x =
  let z = Nx.to_array (Model.apply p x) in
  let rows = Array.length z / classes in
  let scale = ref 0. in
  for r = 0 to rows - 1 do
    let row = Array.sub z (r * classes) classes in
    let m = Array.fold_left Float.max Float.neg_infinity row in
    let lse =
      m
      +. Float.log (Array.fold_left (fun a v -> a +. Float.exp (v -. m)) 0. row)
    in
    let big = Array.fold_left (fun a v -> Float.max a (Float.abs v)) 0. row in
    scale := Float.max !scale (Float.abs lse +. big)
  done;
  !scale

(* AdamW's bias-corrected direction at step [t] for a gradient element [g] and
   the moments [m] and [v] before the step (vega.mli). *)
let direction ~t m v g =
  let t = float_of_int t in
  let m = (b1 *. m) +. ((1. -. b1) *. g)
  and v = (b2 *. v) +. ((1. -. b2) *. g *. g) in
  m /. (1. -. (b1 ** t)) /. (Float.sqrt (v /. (1. -. (b2 ** t))) +. eps)

(* The step at [t] from the same state [(params, ostate)] on one device, giving
   loss [l1] and parameters [p1], and on two, giving [l2] and [p2].

   Premise, from rune.mli's Numerics and nx's bounds: each side computes each
   example's loss and gradient, through two contractions of at most 8 terms
   (within 8u each of the magnitudes they add), a tanh and a log-softmax over 3
   classes (within 3 ulps each), within 64u of the magnitudes that example
   combines, provided no intermediate cancels below them; and the batch's sum of
   examples, ordered by its program, adds 8u of their magnitudes ([u = 2^-53]).
   No bit equality between the two programs is assumed.

   So each side's loss is within [64u Z + 8u L] of the exact one, [Z] the
   largest magnitude an example's loss combines, [L] the loss. Each gradient
   element is within [E = 72u S], [S] the mean of its examples' magnitudes, and
   so is the eager gradient [g] computed here: both sides lie in [g - 2E, g +
   2E]. Over so small an interval the direction [d] is monotone up to second
   order, so the sides' directions differ by at most [|d (g + 2E) - d (g -
   2E)|], plus each side's own roundings of [d] and of the decay, [16u (|d| + wd
   |p|)], times [lr]; the update and its rounding add [4u |p'|]. *)
let follows_one_device ~t model (params, (ostate : (_, _) Vega.adam_state))
    batch (l1, p1) (l2, p2) =
  let g, s = gradient_scale model params batch in
  let z = loss_scale params (fst batch) in
  equal
    ~msg:(Printf.sprintf "loss of step %d" t)
    (float (2. *. ((64. *. u *. z) +. (8. *. u *. l1))))
    l1 l2;
  let theta = flat model params in
  let m = flat model ostate.Vega.mu and v = flat model ostate.Vega.nu in
  Array.iteri
    (fun i p ->
      let e = 72. *. u *. s.(i) and d = direction ~t m.(i) v.(i) in
      let tol =
        lr
        *. (Float.abs (d (g.(i) +. (2. *. e)) -. d (g.(i) -. (2. *. e)))
           +. 16. *. u
              *. (Float.abs (d g.(i)) +. (weight_decay *. Float.abs theta.(i)))
           )
        +. (4. *. u *. Float.abs p)
      in
      equal
        ~msg:(Printf.sprintf "parameter %d after step %d" i t)
        (float (Float.max tol (Float.ldexp 1. (-1074))))
        p p2.(i))
    p1

(* The two-device step runs over [Data] minibatches from its own state, given up
   to each call; before each call a host copy of its state also takes the
   one-device step, so each comparison is of one step from one state. *)
let split_batch_follows_one_device () =
  Compiled.here ();
  let module Two = (val two ()) in
  let host_model = model () and two_model = model () in
  let host_state = Nx.Ptree.pair host_model (Vega.adam_ptree host_model) in
  let two_state = Nx.Ptree.pair two_model (Vega.adam_ptree two_model) in
  let one = compiled host_state host_model in
  let split = compiled two_state two_model in
  let like = (initial (), Vega.adamw_init host_model (initial ())) in
  let params = Nx.Ptree.place (module Model) Two.on (initial ()) in
  let s = ref (params, Vega.adamw_init two_model params) in
  Seq.iteri
    (fun k (x, y) ->
      let pre = to_host two_state host_state ~like !s in
      let cut = Two.split ~axis:0 in
      let before = Rune.evaluations () in
      let next, l2 =
        split (Nx.Ptree.donate two_state !s) (Nx.place cut x, Nx.place cut y)
      in
      equal ~msg:"calls evaluated" int 0 (Rune.evaluations () - before);
      s := next;
      let (p1, _), l1 = one pre (x, y) in
      follows_one_device ~t:(k + 1) host_model pre (x, y)
        (Nx.item [] l1, flat host_model p1)
        (Nx.item [] l2, flat two_model (fst next)))
    (Data.batches2 ~batch_size:8 (dataset ()))

let () =
  exit
    (run "kaun"
       [
         group "training step"
           [
             test
               "compiled with its state given up, over a batch split on two \
                devices, follows one device"
               split_batch_follows_one_device;
           ];
       ])
