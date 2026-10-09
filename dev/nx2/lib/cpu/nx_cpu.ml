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
   [s] has one, laid out as the view [v]. *)
external contract_c :
  Nx_kernel.Spec.contract Nx_kernel.Spec.t ->
  V.t ->
  dst:('v, 's) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('c, 'd) Nx_array.t ->
  ('e, 'f) Nx_array.t ->
  Nx_array.answer = "nx_cpu_contract_byte" "nx_cpu_contract"

(* Each domain's view. *)
let view = Domain.DLS.new_key V.make

let contract s ~dst ops =
  let v = Domain.DLS.get view in
  if not (V.fill v s ~dst ops) then Nx_array.Declined
  else
    let (Nx_array.Any d) = dst in
    let (Nx_array.Any a) = ops.(0) in
    let (Nx_array.Any b) = ops.(1) in
    let (Nx_array.Any i) = if Nx_kernel.Spec.init s then ops.(2) else ops.(0) in
    contract_c s v ~dst:d a b i
