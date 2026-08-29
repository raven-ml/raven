(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The whole training step — forward, backward, Vega optimizer update — as one
   compiled program, with the optimizer state threaded through it.

   This is the regression suite for Vega's jit ergonomics. The optimizer state
   is a parameter tree ([Vega.Adam_state.Make (Model)]), so it is one field of
   the step's input and output records, walked by its own traversals — no
   duplicated model structure, no option plumbing. The bias corrections and the
   step counter are scalar tensor leaves advanced inside the compiled program,
   so the jitted trajectory matches the eager one and the counter reads [n]
   after [n] compiled calls — a host-int counter would burn [t] into the trace
   and replay it stale. The learning rate derives inside the step from a tensor
   schedule evaluated at the state's own step leaf, and a [pmap2] run with the
   state replicated matches the single-device one.

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

type model = { l1 : Nx.float32_t Linear.t; l2 : Nx.float32_t Linear.t }

module Model = struct
  type t = model

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) m =
    { l1 = Linear.map f m.l1; l2 = Linear.map f m.l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { l1 = Linear.map2 f a.l1 b.l1; l2 = Linear.map2 f a.l2 b.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) m =
    Linear.iter f m.l1;
    Linear.iter f m.l2
end

module Opt = Vega.Adam_state.Make (Model)

let fill i n =
  Array.init n (fun j -> sin (float_of_int ((i * 7919) + j)) *. 0.3)

let mat i r c = Nx.create Nx.float32 [| r; c |] (fill i (r * c))
let vec i n = Nx.create Nx.float32 [| n |] (fill i n)

let model_init () =
  {
    l1 = { Linear.w = mat 3 inputs hidden; b = Some (vec 4 hidden) };
    l2 = { Linear.w = mat 5 hidden outputs; b = Some (vec 6 outputs) };
  }

let data_init () =
  ( Nx.create Nx.float32 [| batch; inputs |] (fill 7 (batch * inputs)),
    Nx.create Nx.float32 [| batch; outputs |] (fill 8 (batch * outputs)) )

let loss_fn x y p =
  Loss.mse (Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))) y

(* The step record: parameters, the optimizer state, the batch. The state field
   is walked by [Opt]'s own parameter tree. *)

type step_in = {
  params : model;
  opt : Opt.t;
  x : Nx.float32_t;
  y : Nx.float32_t;
}

module Step_in = struct
  type t = step_in

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

type step_out = { params' : model; opt' : Opt.t; loss : Nx.float32_t }

module Step_out = struct
  type t = step_out

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s =
    {
      params' = Model.map f s.params';
      opt' = Opt.map f s.opt';
      loss = f s.loss;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      params' = Model.map2 f a.params' b.params';
      opt' = Opt.map2 f a.opt' b.opt';
      loss = f a.loss b.loss;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) s =
    Model.iter f s.params';
    Opt.iter f s.opt';
    f s.loss
end

(* One training step: value_and_grad, gradient clipping, a scheduled learning
   rate derived from the state's counter, one Adam update. Run eagerly and,
   through [Rune.jit2], compiled. *)
let train_step { params; opt; x; y } =
  let loss, grads = Rune.value_and_grad (module Model) (loss_fn x y) params in
  let grads = Vega.clip_by_global_norm (module Model) ~max_norm:2.0 grads in
  let lr =
    Vega.Schedule.cosine_decay_t ~init_value:0.05 ~decay_steps:64 opt.step
  in
  let params', opt' = Vega.adam_step (module Model) ~lr opt ~params ~grads in
  { params'; opt'; loss }

let init () =
  let params = model_init () in
  let opt = Vega.adam_init (module Model) params in
  let x, y = data_init () in
  { params; opt; x; y }

let advance step0 s =
  let out = step0 !s in
  s := { !s with params = out.params'; opt = out.opt' };
  (Nx.item [] out.loss, out.params')

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
  let opt = !s.opt in
  equal ~msg:"counter reads n after n calls" int steps
    (Int32.to_int (Nx.item [] opt.step));
  (* The corrections must match the closed form at t = n, computed inside the
     compiled program: c1 = 1 - 0.9^n, c2 = 1 - 0.999^n. *)
  let c1 = Nx.item [] opt.c1 and c2 = Nx.item [] opt.c2 in
  is_true ~msg:"c1 matches the closed form"
    (Float.abs (c1 -. (1.0 -. (0.9 ** float_of_int steps))) < 1e-6);
  is_true ~msg:"c2 matches the closed form"
    (Float.abs (c2 -. (1.0 -. (0.999 ** float_of_int steps))) < 1e-6);
  (* The moments are not zero: the state genuinely updates. *)
  let abs_sum (type a b) (t : (a, b) Nx.t) : float =
    Nx.item [] (Nx.sum (Nx.abs (Nx.cast Nx.float64 t)))
  in
  let moved = ref false in
  ignore (Model.iter (fun t -> if abs_sum t > 0.0 then moved := true) opt.mu);
  is_true ~msg:"moments moved" !moved

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
        test "corrections and counter advance across compiled calls"
          test_state_advances_across_compiled_calls;
        test "pmap with replicated state matches jit" test_pmap_matches_jit;
      ];
  ]

let () = run "kaun jit state" tests
