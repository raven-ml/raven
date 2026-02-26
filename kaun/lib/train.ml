(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('i, 'o) t = { model : ('i, 'o) Layer.t; optimizer : Optim.algorithm }
type 'l state = { vars : 'l Layer.vars; opt_state : Optim.state }

let make ~model ~optimizer = { model; optimizer }

let init t ~dtype =
  let vars = Layer.init t.model ~dtype in
  let opt_state = Optim.init t.optimizer (Layer.params vars) in
  { vars; opt_state }

let vars st = st.vars

let make_state t vars =
  let opt_state = Optim.init t.optimizer (Layer.params vars) in
  { vars; opt_state }

let step (type i o l in_elt) (t : (i, o) t) (st : l state) ~training ?ctx
    ~(loss : (o, l) Rune.t -> (float, l) Rune.t) (x : (i, in_elt) Rune.t) =
  let loss_val, grads, new_layer_state =
    Grad.value_and_grad_aux
      (fun params ->
        let vars' = Layer.with_params st.vars params in
        let pred, vars'' = Layer.apply t.model vars' ~training ?ctx x in
        (loss pred, Layer.state vars''))
      (Layer.params st.vars)
  in
  let new_params, opt_state =
    Optim.update t.optimizer st.opt_state (Layer.params st.vars) grads
  in
  let vars =
    Layer.with_params st.vars new_params |> fun v ->
    Layer.with_state v new_layer_state
  in
  (loss_val, { vars; opt_state })

exception Early_stop

let fit (type i o l in_elt) (t : (i, o) t) (st : l state) ?ctx ?report
    (data : ((i, in_elt) Rune.t * ((o, l) Rune.t -> (float, l) Rune.t)) Data.t)
    =
  let st = ref st in
  let i = ref 0 in
  (try
     Data.iter
       (fun (x, loss) ->
         incr i;
         let loss_val, st' = step t !st ~training:true ?ctx ~loss x in
         st := st';
         match report with
         | Some f -> f ~step:!i ~loss:(Rune.item [] loss_val) !st
         | None -> ())
       data
   with Early_stop -> ());
  !st

let predict (type i o l in_elt) (t : (i, o) t) (st : l state) ?ctx
    (x : (i, in_elt) Rune.t) =
  let y, _ = Layer.apply t.model st.vars ~training:false ?ctx x in
  y
