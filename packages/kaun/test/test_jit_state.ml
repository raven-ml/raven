(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The whole training step — forward, backward, Vega optimizer update — as one
   compiled program, with the optimizer state threaded through it.

   This is the regression suite for Vega's jit ergonomics: the optimizer state
   is a structure over the model's ([Vega.adam_ptree model]) sitting as one
   field of the step's state, a record walked with [Nx.Ptree.Walk.structure],
   and the learning rate derives inside the step from the schedule applied to
   the state's own counter. The counter is a tensor leaf advanced inside the
   compiled program, so the jitted trajectory matches the eager one and the
   counter reads [n] after [n] compiled calls — a host-int counter would burn
   the step into the trace and replay it stale. A run over a batch split across
   two devices, the state a copy on each, matches the single-device one.

   Deterministic init (no RNG) so every run sees identical weights and data. *)

open Windtrap
open Kaun

let rows = Nx.Placement.sharded ~axis:0 [ Nx.Device.cpu 1; Nx.Device.cpu 2 ]
let batch = 8
let inputs = 8
let hidden = 16
let outputs = 3
let steps = 8

module Mlp = struct
  type 'a t = { l1 : 'a Linear.t; l2 : 'a Linear.t }

  let walk c { l1; l2 } =
    let open Nx.Ptree.Walk in
    let l1 = field c "l1" Linear.walk l1 in
    let l2 = field c "l2" Linear.walk l2 in
    { l1; l2 }

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end

type model = Nx.float32_t Mlp.t

let model : model Nx.Ptree.t = Nx.Ptree.instantiate (module Mlp)

(* The step's state: the parameters and the optimizer state, at paths
   [params.l1.w] and [opt.mu.l1.w]. *)

module State = struct
  type state = { params : model; opt : model Vega.adam_state }
  type _ t = state

  let walk c { params; opt } =
    let open Nx.Ptree.Walk in
    let params = field c "params" (structure model) params in
    let opt = field c "opt" (structure (Vega.adam_ptree model)) opt in
    { params; opt }
end

let state = Nx.Ptree.instantiate (module State)

(* A step reads the state and the batch, and returns the next state and the
   loss. *)
let step_signature =
  Nx.Ptree.(state @-> tensor @-> tensor @-> returns (pair state tensor))

let fill i n =
  Array.init n (fun j -> sin (float_of_int ((i * 7919) + j)) *. 0.3)

let mat i r c = Nx.create Nx.float32 [| r; c |] (fill i (r * c))
let vec i n = Nx.create Nx.float32 [| n |] (fill i n)

let model_init () =
  {
    Mlp.l1 = { Linear.w = mat 3 inputs hidden; b = Some (vec 4 hidden) };
    l2 = { Linear.w = mat 5 hidden outputs; b = Some (vec 6 outputs) };
  }

let data_init () =
  ( Nx.create Nx.float32 [| batch; inputs |] (fill 7 (batch * inputs)),
    Nx.create Nx.float32 [| batch; outputs |] (fill 8 (batch * outputs)) )

let loss_fn x y p = Loss.mse (Mlp.apply p x) y
let sched = Vega.Schedule.cosine_decay ~init_value:0.05 ~decay_steps:64 ()

(* One training step: value_and_grad, gradient clipping, a scheduled learning
   rate derived from the state's counter, one Adam update. Run eagerly and,
   through [Rune.jit], compiled. *)
let train_step { State.params; opt } x y =
  let loss, grads = Rune.value_and_grad model (loss_fn x y) params in
  let grads = Vega.clip_by_global_norm model ~max_norm:2.0 grads in
  let params, opt =
    Vega.adam_step model ~lr:(sched opt.step) opt ~params ~grads
  in
  ({ State.params; opt }, loss)

let init () =
  let params = model_init () in
  { State.params; opt = Vega.adam_init model params }

let advance step0 s =
  let x, y = data_init () in
  let next, loss = step0 !s x y in
  s := next;
  (Nx.item [] loss, next.State.params)

let run_traj ~step0 n s0 =
  let s = ref s0 in
  Array.init n (fun _ -> advance step0 s)

let check_trajectory ~msg eps (a : (float * model) array)
    (b : (float * model) array) =
  Array.iteri
    (fun i (la, _) ->
      equal
        ~msg:(Printf.sprintf "%s: loss at step %d" msg (i + 1))
        (float eps) la
        (fst b.(i)))
    a;
  let _, last_a = a.(Array.length a - 1) in
  let _, last_b = b.(Array.length b - 1) in
  let leaf = ref 0 in
  ignore
    (Nx.Ptree.map2 model
       (fun (type a b) _ (x : (a, b) Nx.t) (y : (a, b) Nx.t) : (a, b) Nx.t ->
         incr leaf;
         let d =
           Nx.item [] (Nx.cast Nx.float64 (Nx.max (Nx.abs (Nx.sub x y))))
         in
         is_true
           ~msg:
             (Printf.sprintf "%s: leaf %d, max |eager - compiled| = %g" msg
                !leaf d)
           (d <= eps);
         x)
       last_a last_b)

let test_jit_matches_eager () =
  let eager = run_traj ~step0:train_step steps (init ()) in
  let compiled =
    run_traj ~step0:(Rune.jit step_signature train_step) steps (init ())
  in
  check_trajectory ~msg:"jit adam" 1e-6 eager compiled

let test_state_advances_across_compiled_calls () =
  let jitted = Rune.jit step_signature train_step in
  let s = ref (init ()) in
  for _ = 1 to steps do
    ignore (advance jitted s)
  done;
  let opt = !s.State.opt in
  equal ~msg:"counter reads n after n calls" int steps
    (Int32.to_int (Nx.item [] opt.step));
  (* The moments are not zero: the state genuinely updates. *)
  let abs_sum (type a b) (t : (a, b) Nx.t) : float =
    Nx.item [] (Nx.sum (Nx.abs (Nx.cast Nx.float64 t)))
  in
  let moved =
    Nx.Ptree.fold model (fun _ t moved -> moved || abs_sum t > 0.0) opt.mu false
  in
  is_true ~msg:"moments moved" moved;
  (* The schedule tracks the counter: the compiled program's rate at the next
     step matches the schedule read on the host. *)
  equal ~msg:"schedule follows the counter" (float 1e-6)
    (Vega.Schedule.eval sched steps)
    (Nx.item [] (sched opt.step))

let test_split_batch_matches_jit () =
  let jit =
    run_traj ~step0:(Rune.jit step_signature train_step) steps (init ())
  in
  (* The state enters from the host as a copy on each device, the batch split on
     axis 0. *)
  let split =
    let step = Rune.jit step_signature train_step in
    run_traj
      ~step0:(fun s x y -> step s (Nx.place rows x) (Nx.place rows y))
      steps (init ())
  in
  check_trajectory ~msg:"adam over a split batch" 1e-5 jit split

(* L-BFGS at a fixed rate. Its state carries the point, so the step reads the
   state and the batch, takes the gradient there, and returns the loss with the
   state, as [train_step] does. *)

let lopt : model Vega.lbfgs_state Nx.Ptree.t = Vega.lbfgs_ptree model

let lbfgs_signature =
  Nx.Ptree.(lopt @-> tensor @-> tensor @-> returns (pair tensor lopt))

let lbfgs_step (st : model Vega.lbfgs_state) x y =
  let loss, grads = Rune.value_and_grad model (loss_fn x y) st.params in
  (loss, Vega.lbfgs_step model ~lr:(Vega.lr 0.1) grads st)

let lbfgs_init () = Vega.lbfgs_init model ~history:4 (model_init ())

let run_lbfgs ~step0 n s0 =
  let x, y = data_init () in
  let s = ref s0 in
  let traj =
    Array.init n (fun _ ->
        let loss, (st : model Vega.lbfgs_state) = step0 !s x y in
        s := st;
        (Nx.item [] loss, st.params))
  in
  (traj, !s)

let test_lbfgs_jit_matches_eager () =
  let eager, _ = run_lbfgs ~step0:lbfgs_step steps (lbfgs_init ()) in
  let compiled, st =
    run_lbfgs ~step0:(Rune.jit lbfgs_signature lbfgs_step) steps (lbfgs_init ())
  in
  check_trajectory ~msg:"jit lbfgs" 1e-5 eager compiled;
  is_true ~msg:"the loss decreases" (fst eager.(steps - 1) < fst eager.(0));
  (* The memory fills through the compiled program: after [steps] calls the
     counter reads n and every completed slot, all but the newest, holds a pair
     of positive curvature. *)
  equal ~msg:"counter reads n after n calls" int steps
    (Int32.to_int (Nx.item [] st.step));
  let slot i m = Nx.Ptree.map model (fun _ t -> Nx.get [ i ] t) m in
  List.iter
    (fun i ->
      is_true ~msg:"a completed slot holds a curvature pair"
        (Nx.item [] (Nx.Ptree.dot model Nx.float32 (slot i st.y) (slot i st.s))
        > 0.0))
    [ 1; 2; 3 ]

let tests =
  [
    group "jitted optimizer state"
      [
        test "jit step matches the eager trajectory" test_jit_matches_eager;
        test "counter and schedule advance across compiled calls"
          test_state_advances_across_compiled_calls;
        test "a batch split over two devices matches one"
          test_split_batch_matches_jit;
        slow "jit lbfgs at a fixed rate matches the eager trajectory"
          test_lbfgs_jit_matches_eager;
      ];
  ]

let () = exit (run "kaun jit state" tests)
