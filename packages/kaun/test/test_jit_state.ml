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

   Deterministic init (no RNG) so every run sees identical weights and data.
   Every description is made where it is used, at the set its values lie on. *)

open Windtrap
open Kaun

(* Two devices of one set that nx.cpu computes, the batch split over them. *)
module Two =
  (val Nx.devices
         (List.map
            (fun n -> Result.get_ok (Rig.memory_device n))
            [ "kaun-jit-state-0"; "kaun-jit-state-1" ]))

let batch = 8
let inputs = 8
let hidden = 16
let outputs = 3
let steps = 8

module Mlp = struct
  type ('a, 'd) t = { l1 : ('a, 'd) Linear.t; l2 : ('a, 'd) Linear.t }

  let walk c { l1; l2 } =
    let open Nx.Ptree.Walk in
    let l1 = field c "l1" Linear.walk l1 in
    let l2 = field c "l2" Linear.walk l2 in
    { l1; l2 }

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end

type 'd model = ('d Nx.float32_t, 'd) Mlp.t

let model () = Nx.Ptree.instantiate (module Mlp)

(* The step's state: the parameters and the optimizer state, at paths
   [params.l1.w] and [opt.mu.l1.w]. *)

type 'd state = { params : 'd model; opt : ('d model, 'd) Vega.adam_state }

let state model =
  Nx.Ptree.v (fun c { params; opt } ->
      let open Nx.Ptree.Walk in
      let params = field c "params" (structure model) params in
      let opt = field c "opt" (structure (Vega.adam_ptree model)) opt in
      { params; opt })

(* A step reads the state and the batch, and returns the next state and the
   loss. *)
let step_signature model =
  let state = state model in
  Nx.Ptree.(state @-> tensor @-> tensor @-> returns (pair state tensor))

let fill i n =
  Array.init n (fun j -> sin (float_of_int ((i * 7919) + j)) *. 0.3)

(* The parameters at [at], whole on each of its devices. *)
let model_init at =
  let mat i r c =
    Nx.place at (Nx.create Nx.float32 [| r; c |] (fill i (r * c)))
  in
  let vec i n = Nx.place at (Nx.create Nx.float32 [| n |] (fill i n)) in
  {
    Mlp.l1 = { Linear.w = mat 3 inputs hidden; b = Some (vec 4 hidden) };
    l2 = { Linear.w = mat 5 hidden outputs; b = Some (vec 6 outputs) };
  }

let data_init at =
  ( Nx.place at
      (Nx.create Nx.float32 [| batch; inputs |] (fill 7 (batch * inputs))),
    Nx.place at
      (Nx.create Nx.float32 [| batch; outputs |] (fill 8 (batch * outputs))) )

let loss_fn x y p = Loss.mse (Mlp.apply p x) y

(* Eta-expanded, so the schedule serves the counter of every set. *)
let sched step =
  Vega.Schedule.cosine_decay ~init_value:0.05 ~decay_steps:64 () step

(* One training step: value_and_grad, gradient clipping, a scheduled learning
   rate derived from the state's counter, one Adam update. Run eagerly and,
   through [Rune.jit], compiled. *)
let train_step model { params; opt } x y =
  let loss, grads = Rune.value_and_grad model (loss_fn x y) params in
  let grads = Vega.clip_by_global_norm model ~max_norm:2.0 grads in
  let params, opt = Vega.adam_step model ~lr:sched opt ~params ~grads in
  ({ params; opt }, loss)

let init model at =
  let params = model_init at in
  { params; opt = Vega.adam_init model params }

let advance step0 ~data s =
  let x, y = data in
  let next, loss = step0 !s x y in
  s := next;
  (Nx.item [] loss, next.params)

let run_traj ~data ~step0 n s0 =
  let s = ref s0 in
  Array.init n (fun _ -> advance step0 ~data s)

(* On the host, or compiled on the host. *)
let host_traj ~jit =
  let model = model () in
  let step = train_step model in
  let step0 = if jit then Rune.jit (step_signature model) step else step in
  run_traj ~data:(data_init Nx.Host.on) ~step0 steps (init model Nx.Host.on)

let check_trajectory ~msg eps (a : (float * Nx.host model) array)
    (b : (float * 'd model) array) =
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
  (* The two may lie on two sets: compare what their reads give. *)
  let floats s x =
    List.map
      (fun (Nx.P t) -> Nx.to_array (Nx.cast Nx.float64 t))
      (fst (Nx.Ptree.flatten s x))
  in
  List.iter2
    (fun x y ->
      incr leaf;
      equal
        ~msg:(Printf.sprintf "%s: leaf %d" msg !leaf)
        (array (float eps))
        x y)
    (floats (model ()) last_a)
    (floats (model ()) last_b)

let test_jit_matches_eager () =
  check_trajectory ~msg:"jit adam" 1e-6 (host_traj ~jit:false)
    (host_traj ~jit:true)

let test_state_advances_across_compiled_calls () =
  let model = model () in
  let jitted = Rune.jit (step_signature model) (train_step model) in
  let s = ref (init model Nx.Host.on) in
  for _ = 1 to steps do
    ignore (advance jitted ~data:(data_init Nx.Host.on) s)
  done;
  let opt = !s.opt in
  equal ~msg:"counter reads n after n calls" int steps
    (Int32.to_int (Nx.item [] opt.step));
  (* The moments are not zero: the state genuinely updates. *)
  let abs_sum (type a b) (t : (a, b, _) Nx.t) : float =
    Nx.item [] (Nx.sum (Nx.abs (Nx.cast Nx.float64 t)))
  in
  let moved =
    Nx.Ptree.fold model (fun _ t acc -> acc +. abs_sum t) opt.mu 0.0
  in
  greater ~msg:"moments moved" float_exact ~than:0.0 moved;
  (* The schedule tracks the counter: the compiled program's rate at the next
     step matches the schedule read on the host. *)
  equal ~msg:"schedule follows the counter" (float 1e-6)
    (Vega.Schedule.eval sched steps)
    (Nx.item [] (sched opt.step))

let test_split_batch_matches_jit () =
  (* The state lies whole on each device, the batch split on axis 0. *)
  let split =
    let model = model () in
    let step = Rune.jit (step_signature model) (train_step model) in
    run_traj
      ~data:(data_init (Two.split ~axis:0))
      ~step0:step steps (init model Two.on)
  in
  check_trajectory ~msg:"adam over a split batch" 1e-5 (host_traj ~jit:true)
    split

(* L-BFGS at a fixed rate. Its state carries the point, so the step reads the
   state and the batch, takes the gradient there, and returns the loss with the
   state, as [train_step] does. *)

let model = model ()
let lopt = Vega.lbfgs_ptree model

let lbfgs_signature =
  Nx.Ptree.(lopt @-> tensor @-> tensor @-> returns (pair tensor lopt))

let lbfgs_step (st : (Nx.host model, Nx.host) Vega.lbfgs_state) x y =
  let loss, grads = Rune.value_and_grad model (loss_fn x y) st.params in
  (loss, Vega.lbfgs_step model ~lr:(Vega.lr 0.1) grads st)

let lbfgs_init () = Vega.lbfgs_init model ~history:4 (model_init Nx.Host.on)

let run_lbfgs ~step0 n s0 =
  let x, y = data_init Nx.Host.on in
  let s = ref s0 in
  let traj =
    Array.init n (fun _ ->
        let loss, (st : (Nx.host model, Nx.host) Vega.lbfgs_state) =
          step0 !s x y
        in
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
  less ~msg:"the loss decreases" float_exact ~than:(fst eager.(0))
    (fst eager.(steps - 1));
  (* The memory fills through the compiled program: after [steps] calls the
     counter reads n and every completed slot, all but the newest, holds a pair
     of positive curvature. *)
  equal ~msg:"counter reads n after n calls" int steps
    (Int32.to_int (Nx.item [] st.step));
  let slot i m = Nx.Ptree.map model (fun _ t -> Nx.get [ i ] t) m in
  List.iter
    (fun i ->
      let curvature =
        Nx.Ptree.dot model Nx.float32 (slot i st.y) (slot i st.s)
      in
      greater ~msg:"a completed slot holds a curvature pair" float_exact
        ~than:0.0 (Nx.item [] curvature))
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
        test "jit lbfgs at a fixed rate matches the eager trajectory"
          test_lbfgs_jit_matches_eager;
      ];
  ]

let () = exit (run "kaun jit state" tests)
