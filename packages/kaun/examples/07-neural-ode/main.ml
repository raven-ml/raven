(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A neural ODE: a kaun MLP is the field of an ODE that jera solves, and the
   loss on the solution's samples is differentiated through the adaptive solve.
   The data is the spiral y' = A y³ of Chen et al. (2018). *)

open Kaun
open Jera

module Mlp = struct
  type 'a t = { l1 : 'a Linear.t; l2 : 'a Linear.t }

  let walk c { l1; l2 } =
    let open Nx.Ptree.Walk in
    let l1 = field c "l1" Linear.walk l1 in
    let l2 = field c "l2" Linear.walk l2 in
    { l1; l2 }

  let apply p y =
    Linear.apply p.l2 (Nx.tanh (Linear.apply p.l1 (Nx.pow_s y 3.)))
end

let mlp = Nx.Ptree.instantiate (module Mlp)
let f32 = Nx.float32
let point = Nx.Ptree.tensor
let tol = Tol.v ~rel:1e-5 ~abs:1e-6

(* The states at [at] of the solution from [y0] under the field [f]. The steps
   land on every time of [at], so the budget exceeds their count. *)
let trajectory f ~at y0 =
  Solution.get
    (Ode.sample point Ode.tsit5 ~tol ~budget:4000 (fun _ y -> f y) ~at y0)

let () =
  Nx.Rng.with_key (Nx.Rng.key 0) @@ fun () ->
  (* The true dynamics, sampled at 1000 times on [0, 25]. *)
  let a = Nx.create f32 [| 2; 2 |] [| -0.1; 2.; -2.; -0.1 |] in
  let spiral y = Nx.matmul (Nx.pow_s y 3.) (Nx.transpose a) in
  let y0 = Nx.create f32 [| 2 |] [| 2.; 0. |] in
  let samples = 1000 in
  let at = Nx.linspace f32 0. 25. samples in
  let data = trajectory spiral ~at y0 in

  (* Training fits [batch] stretches of [window] samples, each starting at a
     random sample, solved together as one state of shape [[batch; 2]]. The
     starts are drawn on the host and passed to the compiled step, so each call
     fits other stretches. The loss's gradient flows through the solve's
     accepted steps, taken again with the tracked field. *)
  let batch = 20 and window = 10 in
  let relative = Nx.slice [ Nx.R (0, window) ] at in
  let loss starts p =
    let rows =
      Nx.add
        (Nx.reshape [| 1; window |] (Nx.arange Nx.int64 0 window 1))
        (Nx.reshape [| batch; 1 |] starts)
    in
    (* [[window; batch; 2]], as the solve stacks them. *)
    let target =
      Nx.transpose ~axes:[ 1; 0; 2 ]
        (Nx.reshape [| batch; window; 2 |]
           (Nx.take ~axis:0 ~indices:(Nx.reshape [| -1 |] rows) data))
    in
    let s =
      Ode.sample point Ode.tsit5 ~tol ~budget:500
        (fun _ y -> Mlp.apply p y)
        ~at:relative
        (Nx.take ~axis:0 ~indices:starts data)
    in
    (Nx.mean (Nx.abs (Nx.sub (Solution.get s) target)), Solution.evaluations s)
  in
  let steps = 4000 in
  let lr = Vega.Schedule.cosine_decay ~init_value:3e-3 ~decay_steps:steps () in
  let optimizer = Vega.adam_ptree mlp in
  let step =
    Rune.jit
      Nx.Ptree.(
        pair (pair mlp optimizer) tensor
        @-> returns (pair (pair mlp optimizer) (pair tensor tensor)))
      (fun ((params, ostate), starts) ->
        let l, grads, evaluations =
          Rune.value_and_grad_aux mlp Nx.Ptree.tensor (loss starts) params
        in
        let params, ostate =
          Vega.adam_step mlp ~lr:(lr ostate.Vega.step) ostate ~params ~grads
        in
        ((params, ostate), (l, evaluations)))
  in
  let random = Random.State.make [| 0 |] in
  let starts () =
    Nx.create Nx.int64 [| batch |]
      (Array.init batch (fun _ ->
           Int64.of_int (Random.State.int random (samples - window + 1))))
  in

  let params =
    {
      Mlp.l1 = Linear.init ~inputs:2 ~outputs:50;
      l2 = Linear.init ~inputs:50 ~outputs:2;
    }
  in
  let state = ref (params, Vega.adam_init mlp params) in
  for i = 1 to steps do
    let s, (l, evaluations) = step (!state, starts ()) in
    state := s;
    if i mod 500 = 0 then
      Printf.printf "step %4d  loss %.4f  field evaluations %ld\n%!" i
        (Nx.item [] l) (Nx.item [] evaluations)
  done;

  (* The learned field's trajectory from the start against the truth. *)
  let later = Nx.linspace f32 0. 25. 6 in
  let learned = trajectory (Mlp.apply (fst !state)) ~at:later y0 in
  let truth = trajectory spiral ~at:later y0 in
  Printf.printf "\n   t    true              learned\n";
  for i = 0 to 5 do
    Printf.printf "%5.1f  (% .3f, % .3f)  (% .3f, % .3f)\n"
      (Nx.item [ i ] later)
      (Nx.item [ i; 0 ] truth)
      (Nx.item [ i; 1 ] truth)
      (Nx.item [ i; 0 ] learned)
      (Nx.item [ i; 1 ] learned)
  done
