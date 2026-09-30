(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rune_next
open Tolk_next

(* [u] with each parameter held as a buffer of its slot, which the harness binds
   as it binds a capture. *)
let held u =
  let buffer p =
    match Ops.arg p with
    | Param { slot; size = Some n; dtype; device = Some d; _ } ->
        (p, Ops.new_buffer ~slot d n dtype)
    | _ -> invalid_arg "a parameter that is not storage"
  in
  Ops.substitute u
    (List.filter_map
       (fun p -> if Op.equal (Ops.op p) Op.Param then Some (buffer p) else None)
       (Ops.toposort u))

(* The schedule that stores [y] into a new buffer on the host, and that
   buffer. *)
let schedule y =
  let tdt = Option.get (Lower.dtype (Nx.dtype y)) in
  let n = Array.fold_left ( * ) 1 (Nx.shape y) in
  let out = Ops.new_buffer (Single "CPU") n tdt in
  let shape = List.map (fun d -> Ops.Int d) (Array.to_list (Nx.shape y)) in
  let view = Ops.reshape out shape in
  let sink =
    Ops.sink [ Ops.after view [ Ops.store view (held (Lower.uop y)) ] ]
  in
  (fst (Schedule.create_linear_with_vars sink), out)

let kernels y =
  Ops.sink (List.map (fun call -> Ops.body call) (Ops.src (fst (schedule y))))

let compiled s y =
  let buffers = Traces.contents s in
  let dt = Nx.dtype y in
  let linear, out = schedule y in
  let slot u =
    match Ops.arg u with
    | Param { slot; _ } -> slot
    | _ -> invalid_arg "an argument that is not storage"
  in
  match Ops.src linear with
  | [ call ] ->
      let ren = Traces.host Nx.Device.host in
      let program = Codegen.to_program (Ops.body call) ren in
      let args = Ops.src_without_body call in
      let bound =
        List.filter_map
          (fun (k, a) ->
            Option.map (fun v -> (k, v)) (List.assoc_opt (slot a) buffers))
          (List.mapi (fun k a -> (k, a)) args)
      in
      let results = Run.on_host program bound in
      let k = Option.get (List.find_index (fun a -> slot a = slot out) args) in
      Nx.create dt (Nx.shape y)
        (Array.map
           (fun v -> Traces.of_const dt (v :> Dtype.const))
           (List.assoc k results))
  | calls ->
      Format.kasprintf invalid_arg "%d kernels: a host program runs one"
        (List.length calls)
