(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every step compiles with its state as a consumed argument: the parameters and
   the optimizer state go into [Rune.jit] and come back out, and the compiled
   trajectory matches the eager one leaf for leaf, the counter included. *)

open Windtrap

let dev = Rune.device "CPU"
let steps = 6

(* A float32 matrix and vector: Adafactor factors the one and not the other. *)
module Wb = struct
  type t = { w : Nx.float32_t; b : Nx.float32_t }

  module Walked = struct
    type nonrec _ t = t

    let walk c { w; b } =
      let open Nx.Ptree.Walk in
      let w = field c "w" tensor w in
      let b = field c "b" tensor b in
      { w; b }
  end

  let ptree : t Nx.Ptree.t = Nx.Ptree.instantiate (module Walked)
end

let fill i n = Array.init n (fun j -> sin (float_of_int ((i * 7919) + j)))

let wb i =
  {
    Wb.w = Nx.create Nx.float32 [| 4; 3 |] (fill i 12);
    b = Nx.create Nx.float32 [| 3 |] (fill (i + 1) 3);
  }

let target = lazy (wb 3)

let grads (x : Wb.t) =
  let t = Lazy.force target in
  { Wb.w = Nx.mul_s (Nx.sub x.w t.w) 2.0; b = Nx.mul_s (Nx.sub x.b t.b) 2.0 }

let max_diff (type a b) (x : (a, b) Nx.t) (y : (a, b) Nx.t) =
  Nx.item []
    (Nx.max (Nx.abs (Nx.sub (Nx.cast Nx.float64 x) (Nx.cast Nx.float64 y))))

let compiles state ~init ~step () =
  let both = Nx.Ptree.pair Wb.ptree (state Wb.ptree) in
  let body (params, st) = step st ~params ~grads:(grads params) in
  let run f =
    let rec go k x = if k = 0 then x else go (k - 1) (f x) in
    let params = wb 1 in
    go steps (params, init Wb.ptree params)
  in
  let eager = run body in
  let compiled =
    run
      (Rune.jit ~devices:[ dev ] Nx.Ptree.(consumes both @@ returns both) body)
  in
  ignore
    (Nx.Ptree.map2 both
       (fun path x y ->
         let d = max_diff x y in
         is_true
           ~msg:
             (Format.asprintf "%a: |eager - compiled| = %g" Nx.Ptree.Path.pp
                path d)
           (d <= 1e-5);
         x)
       eager compiled)

let lr = Vega.lr 0.01

let tests =
  [
    test "sgd"
      (compiles Vega.sgd_ptree ~init:Vega.sgd_init ~step:(fun st ->
           Vega.sgd_step Wb.ptree ~lr ~momentum:0.9 st));
    test "lars"
      (compiles Vega.sgd_ptree ~init:Vega.lars_init ~step:(fun st ->
           Vega.lars_step Wb.ptree ~lr ~nesterov:true st));
    test "adam"
      (compiles Vega.adam_ptree ~init:Vega.adam_init ~step:(fun st ->
           Vega.adam_step Wb.ptree ~lr st));
    (* [b2 = 0.99] crosses to rectified steps at the sixth. *)
    test "radam"
      (compiles Vega.adam_ptree ~init:Vega.radam_init ~step:(fun st ->
           Vega.radam_step Wb.ptree ~lr ~b2:0.99 st));
    test "lamb"
      (compiles Vega.adam_ptree ~init:Vega.lamb_init ~step:(fun st ->
           Vega.lamb_step Wb.ptree ~lr st));
    test "rmsprop"
      (compiles Vega.rmsprop_ptree ~init:Vega.rmsprop_init ~step:(fun st ->
           Vega.rmsprop_step Wb.ptree ~lr ~momentum:0.9 st));
    test "adagrad"
      (compiles Vega.adagrad_ptree ~init:Vega.adagrad_init ~step:(fun st ->
           Vega.adagrad_step Wb.ptree ~lr st));
    test "adan"
      (compiles Vega.adan_ptree ~init:Vega.adan_init ~step:(fun st ->
           Vega.adan_step Wb.ptree ~lr st));
    test "lion"
      (compiles Vega.lion_ptree ~init:Vega.lion_init ~step:(fun st ->
           Vega.lion_step Wb.ptree ~lr st));
    test "adafactor"
      (compiles Vega.adafactor_ptree
         ~init:(fun p x -> Vega.adafactor_init p x)
         ~step:(fun (st : Wb.t Vega.adafactor_state) ->
           let t = Nx.cast Nx.float32 (Nx.add_s st.step 1l) in
           Vega.adafactor_step Wb.ptree ~lr:(Nx.mul_s (Nx.rsqrt t) 1e-2) st));
  ]

let () =
  run "vega jit" [ group "a step compiles with its state consumed" tests ]
