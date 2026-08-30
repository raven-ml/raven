(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The whole training step — forward, backward, Vega optimizer update — as one
   compiled program, with the optimizer state threaded through it.

   This is the regression suite for Vega's jit ergonomics: the optimizer state
   is a parameter tree ([Vega.Adam_state (Model)]) sitting as one field of the
   step's input and output records, and the learning rate derives inside the
   step from the schedule applied to the state's own counter. The step records'
   traversals are one-line delegations to the field modules (ppx_ptree derives
   the same shape for users who prefer). The counter is a tensor leaf advanced
   inside the compiled program, so the jitted trajectory matches the eager one
   and the counter reads [n] after [n] compiled calls — a host-int counter
   would burn the step into the trace and replay it stale. A [pmap2] run with
   the state replicated matches the single-device one.

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

  let map f { l1; l2 } = { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 f p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter f { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end

module Model =
  (val Kaun.ptree (module Mlp) : Nx.Ptree.S with type t = Nx.float32_t Mlp.t)

module Opt = Vega.Adam_state (Model)

(* The step records: parameters, the optimizer state, the batch. Each field
   is walked by its own module's traversals. *)

module Step_in = struct
  type t = {
    params : Model.t;
    opt : Opt.t;
    x : Nx.float32_t;
    y : Nx.float32_t;
  }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s =
    {
      params = Model.map f s.params;
      opt = Opt.map f s.opt;
      x = f s.x;
      y = f s.y;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      params = Model.map2 f a.params b.params;
      opt = Opt.map2 f a.opt b.opt;
      x = f a.x b.x;
      y = f a.y b.y;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) s =
    Model.iter f s.params;
    Opt.iter f s.opt;
    f s.x;
    f s.y
end

module Step_out = struct
  type t = { params : Model.t; opt : Opt.t; loss : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s =
    { params = Model.map f s.params; opt = Opt.map f s.opt; loss = f s.loss }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      params = Model.map2 f a.params b.params;
      opt = Opt.map2 f a.opt b.opt;
      loss = f a.loss b.loss;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) s =
    Model.iter f s.params;
    Opt.iter f s.opt;
    f s.loss
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
  let loss, grads = Rune.value_and_grad (module Model) (loss_fn x y) params in
  let grads = Vega.clip_by_global_norm (module Model) ~max_norm:2.0 grads in
  let params, opt =
    Vega.adam_step (module Model) ~lr:(sched opt.step) opt ~params ~grads
  in
  { Step_out.params; opt; loss }

let init () =
  let params = model_init () in
  let opt = Vega.adam_init (module Model) params in
  let x, y = data_init () in
  { Step_in.params; opt; x; y }

let advance step0 s =
  let out = step0 !s in
  s := { !s with Step_in.params = out.Step_out.params; opt = out.Step_out.opt };
  (Nx.item [] out.Step_out.loss, out.Step_out.params)

let run_traj ~step0 n s0 =
  let s = ref s0 in
  Array.init n (fun _ -> advance step0 s)

let check_trajectory ~msg eps (a : (float * Model.t) array)
    (b : (float * Model.t) array) =
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
    (Model.map2
       (fun (type a b) (x : (a, b) Nx.t) (y : (a, b) Nx.t) : (a, b) Nx.t ->
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
      ~step0:
        (Rune.jit2 ~device:dev (module Step_in) (module Step_out) train_step)
      steps (init ())
  in
  check_trajectory ~msg:"jit adam" 1e-6 eager compiled

let test_state_advances_across_compiled_calls () =
  let jitted =
    Rune.jit2 ~device:dev (module Step_in) (module Step_out) train_step
  in
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
  let moved = ref false in
  Model.iter (fun t -> if abs_sum t > 0.0 then moved := true) opt.mu;
  is_true ~msg:"moments moved" !moved;
  (* The schedule tracks the counter: the compiled program's rate at the next
     step matches the schedule read on the host. *)
  equal ~msg:"schedule follows the counter" (float 1e-6)
    (Vega.Schedule.eval sched steps)
    (Nx.item [] (sched opt.step))

let test_pmap_matches_jit () =
  let jit =
    run_traj
      ~step0:
        (Rune.jit2 ~device:dev (module Step_in) (module Step_out) train_step)
      steps (init ())
  in
  (* Everything replicated except the batch, sharded on axis 0. *)
  let in_axes s =
    let n = ref 0 in
    Step_in.iter (fun _ -> incr n) s;
    List.init (!n - 2) (fun _ -> None) @ [ Some 0; Some 0 ]
  in
  let pmapped =
    run_traj
      ~step0:
        (Rune.pmap2 ~devices:devs2
           ~in_axes:(in_axes (init ()))
           (module Step_in)
           (module Step_out)
           train_step)
      steps (init ())
  in
  check_trajectory ~msg:"pmap adam" 1e-5 jit pmapped

let tests =
  [
    group "jitted optimizer state"
      [
        test "jit step matches the eager trajectory" test_jit_matches_eager;
        test "counter and schedule advance across compiled calls"
          test_state_advances_across_compiled_calls;
        test "pmap with replicated state matches jit" test_pmap_matches_jit;
      ];
  ]

let () = run "kaun jit state" tests
