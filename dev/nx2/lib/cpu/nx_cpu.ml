(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module V = Nx_kernel.Spec.Contract_view

let name = "nx.cpu"
let computes_on = Rig.shares_host_memory

external copy :
  dst:('v, 's) Nx_array.t -> ('a, 'b) Nx_array.t -> Nx_array.answer
  = "nx_cpu_copy"

external cast :
  dst:('v, 's) Nx_array.t -> ('a, 'b) Nx_array.t -> Nx_array.answer
  = "nx_cpu_cast"

let apply1 (k : Nx_kernel.Prog.op1) ~dst x =
  match k with
  | Copy -> copy ~dst x
  | Cast -> cast ~dst x
  | Unary _ | Bitcast -> Declined

(* Contractions *)

(* [contract_c s v ~dst a b i] contracts [a] and [b], with the init [i] if
   [s] has one, laid out as [v]: the view's extents, offsets, then each
   operand's strides, 24 ints. *)
external contract_c :
  Nx_kernel.Spec.contract Nx_kernel.Spec.t ->
  int array ->
  dst:('v, 's) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('c, 'd) Nx_array.t ->
  ('e, 'f) Nx_array.t ->
  Nx_array.answer = "nx_cpu_contract_byte" "nx_cpu_contract"

let axes = V.[| Batch; Row; Column; Contracted |]
let operands = V.[| A; B; Init; Dst |]

(* The axes each operand has. *)
let has o x =
  match (o, x) with
  | V.A, V.Column | V.B, V.Row | (V.Init | V.Dst), V.Contracted -> false
  | _ -> true

(* Each domain's view and its numbers. *)
let view = Domain.DLS.new_key (fun () -> (V.make (), Array.make 24 0))

let contract s ~dst ops =
  let v, w = Domain.DLS.get view in
  if not (V.fill v s ~dst ops) then Nx_array.Declined
  else begin
    let init = Nx_kernel.Spec.init s in
    Array.iteri (fun i x -> w.(i) <- V.extent v x) axes;
    Array.iteri
      (fun i o ->
        let present = init || o <> V.Init in
        w.(4 + i) <- (if present then V.offset v o else 0);
        Array.iteri
          (fun j x ->
            w.(8 + (4 * i) + j) <-
              (if present && has o x then V.stride v o x else 0))
          axes)
      operands;
    let (Nx_array.Any d) = dst in
    let (Nx_array.Any a) = ops.(0) in
    let (Nx_array.Any b) = ops.(1) in
    let (Nx_array.Any i) = if init then ops.(2) else ops.(0) in
    contract_c s w ~dst:d a b i
  end
