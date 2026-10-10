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

external bitcast :
  dst:('v, 's) Nx_array.t -> ('a, 'b) Nx_array.t -> Nx_array.answer
  = "nx_cpu_bitcast"

external apply1_c :
  Nx_kernel.Prog.unary ->
  ('v, 's) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  Nx_array.answer = "nx_cpu_apply1"

let apply1 (k : Nx_kernel.Prog.op1) ~dst x =
  match k with
  | Copy -> copy ~dst x
  | Cast -> cast ~dst x
  | (Unary _ | Bitcast)
    when not
           (Nx_kernel.Prog.accepts1 k (Nx_array.dtype x) (Nx_array.dtype dst))
    ->
      Nx_array.Wrong_dtype
  | Bitcast -> bitcast ~dst x
  | Unary u -> apply1_c u dst x

(* Kinds of no, two and three operands (apply.c, iota.c): the kind passes
   as its value. *)

external fill : string -> ('v, 's) Nx_array.t -> Nx_array.answer
  = "nx_cpu_fill"

external iota : int -> ('v, 's) Nx_array.t -> Nx_array.answer = "nx_cpu_iota"

external apply2_c :
  Nx_kernel.Prog.op2 ->
  ('v, 's) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  Nx_array.answer = "nx_cpu_apply2"

external apply3_c :
  Nx_kernel.Prog.op3 ->
  ('v, 's) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  Nx_array.answer = "nx_cpu_apply3"

let apply0 (k : Nx_kernel.Prog.op0) ~dst =
  if not (Nx_kernel.Prog.accepts0 k (Nx_array.dtype dst)) then
    Nx_array.Wrong_dtype
  else match k with Fill b -> fill b dst | Iota i -> iota i dst

let apply2 k ~dst x y =
  if not (Nx_kernel.Prog.accepts2 k (Nx_array.dtype x)) then
    Nx_array.Wrong_dtype
  else apply2_c k dst x y

let apply3 k ~dst c x y =
  if not (Nx_kernel.Prog.accepts3 k (Nx_array.dtype c) (Nx_array.dtype x))
  then Nx_array.Wrong_dtype
  else apply3_c k dst c x y

let map _ ~dsts:_ _ = Nx_array.Declined
(* Reductions and scans (fold.c) *)

external reduce :
  Nx_kernel.Spec.reduce Nx_kernel.Spec.t ->
  dsts:Nx_array.any array ->
  Nx_array.any array ->
  Nx_array.answer = "nx_cpu_reduce"

external scan :
  Nx_kernel.Spec.scan Nx_kernel.Spec.t ->
  dsts:Nx_array.any array ->
  Nx_array.any array ->
  Nx_array.answer = "nx_cpu_scan"

(* Gathers and scatters (index.c) *)

external gather :
  Nx_kernel.Spec.gather Nx_kernel.Spec.t ->
  dst:('v, 's) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_gather"

external scatter :
  Nx_kernel.Spec.scatter Nx_kernel.Spec.t ->
  dst:('v, 's) Nx_array.t ->
  into:('v, 's) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_scatter"

(* Sorts (sort.c) *)

external sort :
  Nx_kernel.Spec.sort Nx_kernel.Spec.t ->
  values:('v, 's) Nx_array.t ->
  positions:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_sort"

(* Assemblies and folds (assemble.c) *)

external assemble :
  Nx_kernel.Spec.assemble Nx_kernel.Spec.t ->
  dst:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t array ->
  Nx_array.answer = "nx_cpu_assemble"

external fold :
  Nx_kernel.Spec.fold Nx_kernel.Spec.t ->
  dst:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_fold_pad"

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

let fft _ ~dst:_ _ = Nx_array.Declined
let linalg _ ~dsts:_ _ = Nx_array.Declined
