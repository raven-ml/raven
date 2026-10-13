(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype

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

(* Reductions and scans (reduce.c). nx.cpu computes a program's one operand,
   unchanged, into its own dtype: one [Sum], [Prod], [Max] or [Min] at a
   dtype the kernels reduce, or a [Moments] scan at float32 or float64. The
   externals take the monoid, read as an int, a padded load's pad and its
   fill's bits, and [threads]: [1] for one thread, [0] for as many as the
   job gives. *)

external reduce_c :
  Nx_kernel.monoid ->
  int array ->
  Nx_array.any ->
  ('v, 's) Nx_array.t ->
  (Nx_kernel.pad * string) option ->
  int ->
  Nx_array.answer = "nx_cpu_reduce_byte" "nx_cpu_reduce"

external scan_c :
  Nx_kernel.monoid ->
  int ->
  Nx_array.any ->
  ('v, 's) Nx_array.t ->
  (Nx_kernel.pad * string) option ->
  int ->
  Nx_array.answer = "nx_cpu_scan_byte" "nx_cpu_scan"

external moments_c :
  int ->
  Nx_array.any ->
  Nx_array.any ->
  ('v, 's) Nx_array.t ->
  (Nx_kernel.pad * string) option ->
  int ->
  Nx_array.answer = "nx_cpu_moments_scan_byte" "nx_cpu_moments_scan"

(* Whether the kernels reduce terms of [dt] by [m]: the four monoids at
   float32, float64 and the 8- to 64-bit integers, extremes at bool. *)
let reduces (type v s) (m : Nx_kernel.monoid) (dt : (v, s) D.t) =
  match (m, dt) with
  | Logsumexp, _ -> false
  | ( (Sum | Prod | Max | Min),
      ( Float32 | Float64 | Int8 | Uint8 | Int16 | Uint16 | Int32 | Uint32
      | Int64 | Uint64 ) ) ->
      true
  | (Max | Min), Bool -> true
  | (Sum | Prod), Bool -> false
  | ( (Sum | Prod | Max | Min),
      ( Float16 | Bfloat16 | Float8_e4m3fn | Float8_e5m2 | Float4_e2m1fn
      | Int4 | Uint4 | Complex128 | Complex64 | Bit ) ) ->
      false

(* Whether the kernels scan terms of [dt] by Moments. *)
let moments_dtype (type v s) (dt : (v, s) D.t) =
  match dt with Float32 | Float64 -> true | _ -> false

let same (Nx_array.Any d) x = D.equal (Nx_array.dtype d) (Nx_array.dtype x)

(* A load's operand, and a padded one's pad and its fill's bits. *)
let operand : Nx_kernel.load -> Nx_array.any = function
  | Plain x -> Any x
  | Padded { x; _ } -> Any x

let padding : Nx_kernel.load -> _ = function
  | Plain _ -> None
  | Padded { x; fill; pad } ->
      Some (pad, Nx_kernel.Prog.bits (Nx_array.dtype x) fill)

(* Whether reduction [r] accepts the dtype of [p]'s output [k]. *)
let accepted (p : Nx_kernel.Prog.t) (r, k) =
  let (D.Any dt) = Nx_kernel.Prog.dtype p (Iarray.get p.outs k) in
  Nx_kernel.accepts r dt

let rec all_accepted p rs i =
  i = Array.length rs || (accepted p rs.(i) && all_accepted p rs (i + 1))

(* nx.cpu computes a program that is its one operand's identity. A padded
   load passes its pad and its fill's bits. *)
let reduce_on ~threads p ~axes rs ~dsts (loads : Nx_kernel.load array) =
  if not (all_accepted p rs 0) then Nx_array.Wrong_dtype
  else if not (Nx_kernel.Prog.is_identity p) then Nx_array.Declined
  else
    match (rs, dsts, loads) with
    | [| (Monoid m, 0) |], [| dst |], [| l |] ->
        let (Any x) = operand l in
        if not (reduces m (Nx_array.dtype x) && same dst x) then
          Nx_array.Declined
        else reduce_c m axes dst x (padding l) threads
    | _ -> Nx_array.Declined

let scan_on ~threads p ~axis rk ~dsts (loads : Nx_kernel.load array) =
  if not (accepted p rk) then Nx_array.Wrong_dtype
  else if not (Nx_kernel.Prog.is_identity p) then Nx_array.Declined
  else
    match (rk, dsts, loads) with
    | (Monoid m, 0), [| dst |], [| l |] ->
        let (Any x) = operand l in
        if not (reduces m (Nx_array.dtype x) && same dst x) then
          Nx_array.Declined
        else scan_c m axis dst x (padding l) threads
    | (Moments, 0), [| mean; var |], [| l |] ->
        let (Any x) = operand l in
        let dt = Nx_array.dtype x in
        if not (moments_dtype dt && same mean x && same var x) then
          Nx_array.Declined
        else moments_c axis mean var x (padding l) threads
    | _ -> Nx_array.Declined

let reduce p ~axes rs ~dsts loads = reduce_on ~threads:0 p ~axes rs ~dsts loads
let scan p ~axis rk ~dsts loads = scan_on ~threads:0 p ~axis rk ~dsts loads

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

(* [contract_c ~batch ~contracting acc ~dst ~init a b] contracts [a] and
   [b], with [init] if given, in the dtype of code [acc], over the view C
   groups from the pairs on its stack ([nx_contract_view_fill]). *)
external contract_c :
  batch:(int * int) array ->
  contracting:(int * int) array ->
  int ->
  dst:('v, 's) Nx_array.t ->
  init:('v, 's) Nx_array.t option ->
  ('a, 'b) Nx_array.t ->
  ('c, 'd) Nx_array.t ->
  Nx_array.answer = "nx_cpu_contract_byte" "nx_cpu_contract"

let contract ~batch ~contracting ~acc ~dst ~init a b =
  let (Nx_array.Dtype.Any acc) = acc in
  contract_c ~batch ~contracting (Nx_array.Dtype.code acc) ~dst ~init a b

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
let svd ~s:_ ?vectors:_ a = declined (takes a)
let eigh ~w:_ ?v:_ a = declined (takes a)
let eig ~w ?v:_ a =
  declined
    (takes a
    && Nx_kernel.same_precision (Nx_array.dtype a) (Nx_array.dtype w))
