(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fit a linear model y = x . w + b with a training step over its parameters.

   The parameters are a record of two tensors, and its structure is what the
   optimizers walk: every step takes it first. A training step is function
   application: compute the gradients, transform them, step. This example shows
   how to:

   1. Describe the parameters' structure with a [walk] 2. Clip the gradients
   before any optimizer's step 3. Read the optimizer state as a structure of
   named leaves

   The gradients of the mean squared error are written by hand here, so the
   example needs no autodiff; with rune, [Rune.value_and_grad model loss]
   computes them from the loss alone. *)

module Linear = struct
  type 'a t = { w : 'a; b : 'a }

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" leaf b in
    { w; b }
end

type params = Nx.float32_t Linear.t

let model : params Nx.Ptree.t = Nx.Ptree.instantiate (module Linear)

let x =
  Nx.create Nx.float32 [| 16; 3 |]
    (Array.init 48 (fun i -> sin (float (i * i))))

let w_true = Nx.create Nx.float32 [| 3 |] [| 1.5; -2.0; 0.5 |]
let y = Nx.add_s (Nx.sum ~axes:[ 1 ] (Nx.mul x w_true)) 0.3

let start () =
  { Linear.w = Nx.zeros Nx.float32 [| 3 |]; b = Nx.scalar Nx.float32 0.0 }

(* The mean squared error and its gradients. *)
let loss_and_grads (m : params) =
  let r = Nx.sub (Nx.add (Nx.sum ~axes:[ 1 ] (Nx.mul x m.w)) m.b) y in
  let k = 2.0 /. float (Nx.shape x).(0) in
  let w =
    Nx.mul_s (Nx.sum ~axes:[ 0 ] (Nx.mul x (Nx.unsqueeze ~axes:[ 1 ] r))) k
  in
  (Nx.item [] (Nx.mean (Nx.square r)), { Linear.w; b = Nx.mul_s (Nx.sum r) k })

(* The training loop, for any optimizer: the gradients are clipped to a global
   norm of 1, then the optimizer steps. [init] and [step] are the
   optimizer's. *)
let train name init step =
  let params = ref (start ()) in
  let st = ref (init model !params) in
  for _ = 1 to 300 do
    let _, grads = loss_and_grads !params in
    let grads = Vega.clip_by_global_norm model ~max_norm:1.0 grads in
    let params', st' = step !st ~params:!params ~grads in
    params := params';
    st := st'
  done;
  Printf.printf "  %-6s loss = %.6f  w = %s  b = %.4f\n" name
    (fst (loss_and_grads !params))
    (Nx.to_string !params.w) (Nx.item [] !params.b);
  !st

let () =
  let lr = Vega.lr 0.05 in

  (* 1-2. The same loop drives every optimizer. AdamW, RAdam and LAMB share
     Adam's state; Lion moves every element by exactly its rate. *)
  Printf.printf "--- 300 clipped steps ---\n";
  let st =
    train "adamw" Vega.adamw_init (fun st ->
        Vega.adamw_step model ~lr ~weight_decay:1e-4 st)
  in
  ignore
    (train "radam" Vega.radam_init (fun st -> Vega.radam_step model ~lr st));
  ignore (train "lamb" Vega.lamb_init (fun st -> Vega.lamb_step model ~lr st));
  ignore
    (train "lion" Vega.lion_init (fun st ->
         Vega.lion_step model ~lr:(Vega.lr 0.02) st));
  ignore
    (train "sgd" Vega.sgd_init (fun st ->
         Vega.sgd_step model ~lr ~momentum:0.9 st));
  Printf.printf "\n";

  (* 3. The state is a structure over the parameters: every leaf has a path, the
     names a compiled step's signature and a checkpoint use. *)
  Printf.printf "--- AdamW's state ---\n";
  List.iter
    (fun v -> Format.printf "  %a@." Nx.Ptree.pp_visit v)
    (Nx.Ptree.visits (Vega.adam_ptree model) st);
  Printf.printf "  step = %ld\n" (Nx.item [] st.step)
