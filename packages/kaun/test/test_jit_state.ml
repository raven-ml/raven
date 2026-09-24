(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The whole training step — forward, backward, Vega optimizer update — as one
   compiled program, with the optimizer state threaded through it.

   This is the regression suite for Vega's jit ergonomics: the optimizer state
   is a structure over the model's ([Vega.adam_ptree model]) sitting as one
   field of the step's input and output records, and the learning rate derives
   inside the step from the schedule applied to the state's own counter. The
   step records are adapted from pairs of the fields' structures with
   [Nx.Ptree.iso]. The counter is a tensor leaf advanced inside the compiled
   program, so the jitted trajectory matches the eager one and the counter reads
   [n] after [n] compiled calls — a host-int counter would burn the step into
   the trace and replay it stale. A [pmap2] run with the state replicated
   matches the single-device one.

   Deterministic init (no RNG) so every run sees identical weights and data. *)

open Windtrap
open Kaun

let dev = "CPU"
let devs2 = [ "CPU:1"; "CPU:2" ]
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

(* The step records: parameters, the optimizer state, the batch. *)

module Step_in = struct
  type t = {
    params : model;
    opt : model Vega.adam_state;
    x : Nx.float32_t;
    y : Nx.float32_t;
  }

  let ptree =
    Nx.Ptree.(
      iso
        (fun ((params, opt), (x, y)) -> { params; opt; x; y })
        (fun { params; opt; x; y } -> ((params, opt), (x, y)))
        (pair (pair model (Vega.adam_ptree model)) (pair tensor tensor)))
end

module Step_out = struct
  type t = { params : model; opt : model Vega.adam_state; loss : Nx.float32_t }

  let ptree =
    Nx.Ptree.(
      iso
        (fun ((params, opt), loss) -> { params; opt; loss })
        (fun { params; opt; loss } -> ((params, opt), loss))
        (pair (pair model (Vega.adam_ptree model)) tensor))
end

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
   through [Rune.jit2], compiled. *)
let train_step { Step_in.params; opt; x; y } =
  let loss, grads = Rune.value_and_grad model (loss_fn x y) params in
  let grads = Vega.clip_by_global_norm model ~max_norm:2.0 grads in
  let params, opt =
    Vega.adam_step model ~lr:(sched opt.step) opt ~params ~grads
  in
  { Step_out.params; opt; loss }

let init () =
  let params = model_init () in
  let opt = Vega.adam_init model params in
  let x, y = data_init () in
  { Step_in.params; opt; x; y }

let advance step0 s =
  let out = step0 !s in
  s := { !s with Step_in.params = out.Step_out.params; opt = out.Step_out.opt };
  (Nx.item [] out.Step_out.loss, out.Step_out.params)

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
    run_traj
      ~step0:(Rune.jit2 ~device:dev Step_in.ptree Step_out.ptree train_step)
      steps (init ())
  in
  check_trajectory ~msg:"jit adam" 1e-6 eager compiled

let test_state_advances_across_compiled_calls () =
  let jitted = Rune.jit2 ~device:dev Step_in.ptree Step_out.ptree train_step in
  let s = ref (init ()) in
  for _ = 1 to steps do
    ignore (advance jitted s)
  done;
  let opt = !s.Step_in.opt in
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

let test_pmap_matches_jit () =
  let jit =
    run_traj
      ~step0:(Rune.jit2 ~device:dev Step_in.ptree Step_out.ptree train_step)
      steps (init ())
  in
  (* Everything replicated except the batch, sharded on axis 0. *)
  let in_axes s =
    let n = Nx.Ptree.fold Step_in.ptree (fun _ _ n -> n + 1) s 0 in
    List.init (n - 2) (fun _ -> None) @ [ Some 0; Some 0 ]
  in
  let pmapped =
    run_traj
      ~step0:
        (Rune.pmap2 ~devices:devs2
           ~in_axes:(in_axes (init ()))
           Step_in.ptree Step_out.ptree train_step)
      steps (init ())
  in
  check_trajectory ~msg:"pmap adam" 1e-5 jit pmapped

(* L-BFGS at a fixed rate. Its state carries the point, so the step's input is
   the state plus the batch and its output the state itself; the objective
   evaluates inside the step, as [train_step]'s does. *)

let lopt : (model, Nx.float32_elt) Vega.lbfgs_state Nx.Ptree.t =
  Vega.lbfgs_ptree model

module Lbfgs_in = struct
  type t = {
    st : (model, Nx.float32_elt) Vega.lbfgs_state;
    x : Nx.float32_t;
    y : Nx.float32_t;
  }

  let ptree =
    Nx.Ptree.(
      iso
        (fun (st, (x, y)) -> { st; x; y })
        (fun { st; x; y } -> (st, (x, y)))
        (pair lopt (pair tensor tensor)))
end

let lbfgs_step { Lbfgs_in.st; x; y } =
  Vega.lbfgs_step model ~lr:(Vega.lr 0.1)
    (Rune.value_and_grad model (loss_fn x y))
    st

let lbfgs_init () =
  let x, y = data_init () in
  let st =
    Vega.lbfgs_init model ~history:4
      (Rune.value_and_grad model (loss_fn x y))
      (model_init ())
  in
  { Lbfgs_in.st; x; y }

let run_lbfgs ~step0 n s0 =
  let s = ref s0 in
  let traj =
    Array.init n (fun _ ->
        let st = step0 !s in
        s := { !s with Lbfgs_in.st };
        (Nx.item [] st.value, st.params))
  in
  (traj, !s.Lbfgs_in.st)

let test_lbfgs_jit_matches_eager () =
  let eager, _ = run_lbfgs ~step0:lbfgs_step steps (lbfgs_init ()) in
  let compiled, st =
    run_lbfgs
      ~step0:(Rune.jit2 ~device:dev Lbfgs_in.ptree lopt lbfgs_step)
      steps (lbfgs_init ())
  in
  check_trajectory ~msg:"jit lbfgs" 1e-5 eager compiled;
  is_true ~msg:"the loss decreases" (fst eager.(steps - 1) < fst eager.(0));
  (* The memory fills through the compiled program: after [steps] calls the
     counter reads n and every slot holds a pair of positive curvature. *)
  equal ~msg:"counter reads n after n calls" int steps
    (Int32.to_int (Nx.item [] st.step));
  is_true ~msg:"every slot holds a curvature pair"
    (Array.for_all (fun r -> r > 0.0) (Nx.to_array st.rho))

(* Without a rate the step line-searches, reading objective values on the host
   to pick its trials: jit must refuse it loudly at trace time rather than
   compile a trace that replays the first search's decisions. *)
let test_lbfgs_line_search_does_not_trace () =
  let searching { Lbfgs_in.st; x; y } =
    Vega.lbfgs_step model (Rune.value_and_grad model (loss_fn x y)) st
  in
  let jitted = Rune.jit2 ~device:dev Lbfgs_in.ptree lopt searching in
  raises_match
    (function Rune.Jit_error _ -> true | _ -> false)
    (fun () -> jitted (lbfgs_init ()))

let tests =
  [
    group "jitted optimizer state"
      [
        test "jit step matches the eager trajectory" test_jit_matches_eager;
        test "counter and schedule advance across compiled calls"
          test_state_advances_across_compiled_calls;
        test "pmap with replicated state matches jit" test_pmap_matches_jit;
        slow "jit lbfgs at a fixed rate matches the eager trajectory"
          test_lbfgs_jit_matches_eager;
        test "jit refuses a line-searching lbfgs step"
          test_lbfgs_line_search_does_not_trace;
      ];
  ]

let () = run "kaun jit state" tests
