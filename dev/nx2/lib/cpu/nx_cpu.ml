(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module V = Nx_kernel.Contract_view

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

(* Maps (map.c) *)

external map :
  Nx_kernel.Prog.t ->
  dsts:Nx_array.any array ->
  Nx_kernel.load array ->
  Nx_array.answer = "nx_cpu_map"

(* Reductions and scans (fold.c). [fill_bits loads] is the bits of the
   first load's fill where it is padded: C computes one load. *)

external reduce_c :
  Nx_kernel.Prog.t ->
  int array ->
  (Nx_kernel.reduction * int) array ->
  Nx_array.any array ->
  Nx_kernel.load array ->
  string ->
  Nx_array.answer = "nx_cpu_reduce_byte" "nx_cpu_reduce"

external scan_c :
  Nx_kernel.Prog.t ->
  int ->
  Nx_kernel.reduction * int ->
  Nx_array.any array ->
  Nx_kernel.load array ->
  string ->
  Nx_array.answer = "nx_cpu_scan_byte" "nx_cpu_scan"

let fill_bits (loads : Nx_kernel.load array) =
  if Array.length loads = 0 then ""
  else
    match loads.(0) with
    | Plain _ -> ""
    | Padded { x; fill; _ } -> Nx_kernel.Prog.bits (Nx_array.dtype x) fill

(* Whether reduction [r] accepts the dtype of [p]'s output [k]. *)
let accepted p (r, k) =
  let (Nx_array.Dtype.Any dt) =
    Nx_kernel.Prog.dtype p (Nx_kernel.Prog.outs p).(k)
  in
  Nx_kernel.accepts r dt

let reduce p ~axes rs ~dsts loads =
  if not (Array.for_all (accepted p) rs) then Nx_array.Wrong_dtype
  else reduce_c p axes rs dsts loads (fill_bits loads)

let scan p ~axis rk ~dsts loads =
  if not (accepted p rk) then Nx_array.Wrong_dtype
  else scan_c p axis rk dsts loads (fill_bits loads)

(* Gathers and scatters (index.c) *)

external gather :
  axis:int ->
  dst:('v, 's) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_gather"

external scatter_c :
  Nx_kernel.combine ->
  int ->
  ('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_scatter_byte" "nx_cpu_scatter"

(* [unique] promises what the kernel does not need: it combines a target's
   updates in order either way. *)
let scatter c ~unique:_ ~axis ~dst ~into idx updates =
  let adds = c = Nx_kernel.Add in
  if adds && not (Nx_kernel.Prog.accepts2 (Binary Add) (Nx_array.dtype into))
  then Nx_array.Wrong_dtype
  else scatter_c c axis dst into idx updates

(* Sorts (sort.c) *)

external sort :
  axis:int ->
  descending:bool ->
  values:('v, 's) Nx_array.t ->
  positions:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_sort"

(* Assemblies and folds (assemble.c) *)

external assemble_c :
  string ->
  ('v, 's) Nx_array.t ->
  (Nx_array.Move.range array * ('v, 's) Nx_array.t) array ->
  Nx_array.answer = "nx_cpu_assemble"

let assemble ~fill ~dst pieces =
  assemble_c (Nx_kernel.Prog.bits (Nx_array.dtype dst) fill) dst pieces

external fold :
  Nx_kernel.pad ->
  dst:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_fold_pad"

(* Contractions (gemm.c) *)

(* [contract_c acc v ~dst a b init] contracts [a] and [b], with [init] if
   given, laid out as the view [v], in the dtype of code [acc]. *)
external contract_c :
  int ->
  V.t ->
  dst:('v, 's) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('c, 'd) Nx_array.t ->
  ('v, 's) Nx_array.t option ->
  Nx_array.answer = "nx_cpu_contract_byte" "nx_cpu_contract"

(* Each domain's view. *)
let view = Domain.DLS.new_key V.make

let contract ~batch ~contracting ~acc ~dst ~init a b =
  let v = Domain.DLS.get view in
  if not (V.fill v ~batch ~contracting ~dst ~init a b) then Nx_array.Declined
  else
    let (Nx_array.Dtype.Any acc) = acc in
    contract_c (Nx_array.Dtype.code acc) v ~dst a b init

(* Fourier transforms: declined at the dtypes they take. *)

let declined takes = if takes then Nx_array.Declined else Nx_array.Wrong_dtype
let c2c _ ~axes:_ ~dst:_ _ = Nx_array.Declined

let r2c ~axes:_ ~dst x =
  declined
    (Nx_kernel.same_precision (Nx_array.dtype x) (Nx_array.dtype dst))

let c2r ~axes:_ ~dst x =
  declined
    (Nx_kernel.same_precision (Nx_array.dtype dst) (Nx_array.dtype x))

(* Matrices (linalg.c) *)

external cholesky :
  Nx_kernel.triangle ->
  dst:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_cholesky"

external lu :
  lu:('v, 's) Nx_array.t ->
  pivots:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  perm:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_lu"

external qr :
  Nx_kernel.factors ->
  q:('v, 's) Nx_array.t ->
  r:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_qr"

external solve_triangular :
  Nx_kernel.triangle ->
  adjoint:bool ->
  unit_diagonal:bool ->
  dst:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer
  = "nx_cpu_solve_triangular_byte" "nx_cpu_solve_triangular"

(* nx.cpu computes no decomposition by iteration. *)
let takes a = Nx_kernel.full_precision (Nx_array.dtype a)
let svd _ ~u:_ ~s:_ ~vh:_ a = declined (takes a)
let svd_values ~s:_ a = declined (takes a)
let eigh ~w:_ ~v:_ a = declined (takes a)
let eigh_values ~w:_ a = declined (takes a)
let eig ~w:_ ~v:_ a = declined (takes a)
let eig_values ~w:_ a = declined (takes a)
