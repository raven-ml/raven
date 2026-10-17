(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype

let name = "nx.cpu"
let computes_on = Rig.shares_host_memory

(* Declines

   nx.cpu computes every case its contract offers it, so a kernel never
   answers [Declined]: one that would, a bug in nx.cpu, raises [Failure] at
   the call instead, naming the kernel and its operands. Each entry that
   could decline matches its answer itself, so that a call that computes
   builds no description. *)

let operand_name (Nx_array.Any a) =
  let s = Nx_array.Layout.shape (Nx_array.layout a) in
  Printf.sprintf "%s [%s]"
    (Nx_array.Dtype.name (Nx_array.dtype a))
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let declined kernel operands =
  failwith
    (Printf.sprintf
       "nx.cpu: %s declined a case of its contract, over %s: a bug in nx.cpu"
       kernel
       (String.concat ", " (List.map operand_name operands)))

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

let apply1_answer (k : Nx_kernel.Prog.op1) ~dst x =
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

let apply1 k ~dst x =
  match apply1_answer k ~dst x with
  | Nx_array.Declined -> declined "apply1" [ Any dst; Any x ]
  | answer -> answer

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
  else
    match match k with Fill b -> fill b dst | Iota i -> iota i dst with
    | Nx_array.Declined -> declined "apply0" [ Any dst ]
    | answer -> answer

let apply2 k ~dst x y =
  if not (Nx_kernel.Prog.accepts2 k (Nx_array.dtype x)) then
    Nx_array.Wrong_dtype
  else
    match apply2_c k dst x y with
    | Nx_array.Declined -> declined "apply2" [ Any dst; Any x; Any y ]
    | answer -> answer

let apply3 k ~dst c x y =
  if not (Nx_kernel.Prog.accepts3 k (Nx_array.dtype c) (Nx_array.dtype x))
  then Nx_array.Wrong_dtype
  else
    match apply3_c k dst c x y with
    | Nx_array.Declined -> declined "apply3" [ Any dst; Any c; Any x; Any y ]
    | answer -> answer

(* Loads: a load's operand, and a padded one's pad and its fill's bits. *)

let operand : Nx_kernel.load -> Nx_array.any = function
  | Plain x -> Any x
  | Padded { x; _ } -> Any x

let padding : Nx_kernel.load -> _ = function
  | Plain _ -> None
  | Padded { x; fill; pad } ->
      Some (pad, Nx_kernel.Prog.bits (Nx_array.dtype x) fill)

(* Maps (map.c): each load passes its padding, or none passes any where
   no load is padded, so that a plain map allocates nothing. *)

external map_c :
  Nx_kernel.Prog.t ->
  Nx_array.any array ->
  Nx_kernel.load array ->
  (Nx_kernel.pad * string) option array ->
  Nx_array.answer = "nx_cpu_map"

let padded : Nx_kernel.load -> bool = function
  | Plain _ -> false
  | Padded _ -> true

let map p ~dsts loads =
  let answer =
    if Array.exists padded loads then
      map_c p dsts loads (Array.map padding loads)
    else map_c p dsts loads [||]
  in
  match answer with
  | Nx_array.Declined ->
      declined "map"
        (Array.to_list dsts @ Array.to_list (Array.map operand loads))
  | answer -> answer

(* Reductions and scans (reduce.c). nx.cpu computes a program's one
   operand, unchanged: one reduction, reduced or scanned, as [reduction]
   takes it, and a [Moments] scan by its own external. The externals
   take the reduction, read as an int, a padded load's pad
   and its fill's bits, and [threads]: [1] for one thread, [0] for as many
   as the job gives. *)

(* The reductions reduce.c computes: the monoids in [Nx_kernel.monoid]'s
   order, then those it computes by a state of each block of terms. *)
type reduction =
  | Sum
  | Prod
  | Max
  | Min
  | Logsumexp
  | Arg_max
  | Arg_min
  | Moments

external reduce_c :
  reduction ->
  int array ->
  Nx_array.any array ->
  ('v, 's) Nx_array.t ->
  (Nx_kernel.pad * string) option ->
  int ->
  Nx_array.answer = "nx_cpu_reduce_byte" "nx_cpu_reduce"

(* A program's reductions, scan or Moments scan, [0], [1] or [2], by pairs
   of a reduction, read as an int, and an output. *)
external reduce_prog_c :
  int ->
  Nx_kernel.Prog.t ->
  int array ->
  int array ->
  int array ->
  Nx_array.any array ->
  Nx_kernel.load array ->
  (Nx_kernel.pad * string) option array ->
  int ->
  Nx_array.answer = "nx_cpu_reduce_prog_byte" "nx_cpu_reduce_prog"

external scan_c :
  reduction ->
  int ->
  Nx_array.any array ->
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

(* Whether the kernels compute [Logsumexp] and [Moments] of terms of [dt]:
   the floats, the narrower ones in float32. *)
let floats (type v s) (dt : (v, s) D.t) =
  match dt with
  | Float32 | Float64 | Float16 | Bfloat16 | Float8_e4m3fn | Float8_e5m2
  | Float4_e2m1fn ->
      true
  | Int8 | Uint8 | Int16 | Uint16 | Int32 | Uint32 | Int64 | Uint64 | Int4
  | Uint4 | Complex64 | Complex128 | Bool | Bit ->
      false

let of_monoid : Nx_kernel.monoid -> reduction = function
  | Sum -> Sum
  | Prod -> Prod
  | Max -> Max
  | Min -> Min
  | Logsumexp -> Logsumexp

(* The destinations of [r]: one, or a [Moments]' mean and variance, an
   [Arg]'s extremes and positions. *)
let width : Nx_kernel.reduction -> int = function
  | Monoid _ -> 1
  | Arg _ | Moments -> 2

(* Whether the kernels compute [r] of terms of [dt] into the destinations
   of [dsts] from [d], [r]'s, which the caller checked are there: of any
   dtype but an [Arg]'s positions, of int64. A predicate apart from the
   reduction it names, reading [dsts] in place, so that the reductions
   allocate nothing for it. *)
let computes (type v s) (r : Nx_kernel.reduction) (dt : (v, s) D.t) dsts d =
  match r with
  | Monoid Logsumexp | Moments -> floats dt
  | Monoid (Sum | Prod | Max | Min) -> true
  | Arg _ ->
      let (Nx_array.Any i) = dsts.(d + 1) in
      D.equal (Nx_array.dtype i) D.Int64

(* The reduction reduce.c computes for [r]. *)
let reduction : Nx_kernel.reduction -> reduction = function
  | Monoid m -> of_monoid m
  | Arg Max -> Arg_max
  | Arg Min -> Arg_min
  | Moments -> Moments

(* Whether reduction [r] accepts the dtype of [p]'s output [k]. *)
let accepted (p : Nx_kernel.Prog.t) (r, k) =
  let (D.Any dt) = Nx_kernel.Prog.dtype p (Iarray.get p.outs k) in
  Nx_kernel.accepts r dt

let rec all_accepted p rs i =
  i = Array.length rs || (accepted p rs.(i) && all_accepted p rs (i + 1))

(* Whether the layout [l] has extents [shape] from axis [i] on, read without
   allocating. *)
let rec extents l shape i =
  i = Array.length shape
  || (Nx_array.Layout.dim l i = shape.(i) && extents l shape (i + 1))

(* Whether load [i] on of [loads] have the loop's [shape] once loaded: a
   plain load its array's, read without allocating on the reductions' hot
   path, a padded one {!Nx_kernel.Shape.padded}'s. No load has none to
   differ. *)
let rec fits shape (loads : Nx_kernel.load array) i =
  i = Array.length loads
  || (match loads.(i) with
     | Plain x ->
         let l = Nx_array.layout x in
         Nx_array.Layout.rank l = Array.length shape && extents l shape 0
     | Padded { x; pad; _ } ->
         let s = Nx_array.Layout.shape (Nx_array.layout x) in
         Nx_kernel.Shape.padded pad s = Ok shape)
     && fits shape loads (i + 1)

(* The loads' paddings, or none where no load is padded. *)
let pads loads =
  if Array.exists padded loads then Array.map padding loads else [||]

(* [r]'s code in reduce.c. *)
let code : reduction -> int = function
  | Sum -> 0
  | Prod -> 1
  | Max -> 2
  | Min -> 3
  | Logsumexp -> 4
  | Arg_max -> 5
  | Arg_min -> 6
  | Moments -> 7

(* What the kernels answer a reduction [r] into [dsts] they do not compute,
   its terms' dtype accepted: destinations other than [r]'s, or an [Arg]'s
   positions of another dtype than int64. *)
let refusal r dsts =
  if Array.length dsts <> width r then Nx_array.Bad_arity
  else Nx_array.Wrong_dtype

(* Fills [codes] with each reduction of [rs] from [i] on, its code in
   reduce.c and its output, its destinations from [d] on: [Done], or the
   refusal of the first the kernels do not compute, or of destinations
   other than the reductions'. *)
let rec fill_codes p rs dsts codes i d =
  if i = Array.length rs then
    if d = Array.length dsts then Nx_array.Done else Nx_array.Bad_arity
  else
    let r, k = rs.(i) in
    let (D.Any dt) = Nx_kernel.Prog.dtype p (Iarray.get p.outs k) in
    if d + width r > Array.length dsts then Nx_array.Bad_arity
    else if not (computes r dt dsts d) then Nx_array.Wrong_dtype
    else begin
      codes.(2 * i) <- code (reduction r);
      codes.((2 * i) + 1) <- k;
      fill_codes p rs dsts codes (i + 1) (d + width r)
    end

(* Reductions of a program's outputs through it: each (reduction, output)
   pair of [rs] into its one or two destinations of [dsts] in turn, in one
   call, which checks every destination before it writes any. *)
let fused_reduce ~threads p ~shape ~axes rs ~dsts loads =
  let codes = Array.make (2 * Array.length rs) 0 in
  match fill_codes p rs dsts codes 0 0 with
  | Nx_array.Done ->
      reduce_prog_c 0 p codes shape axes dsts loads (pads loads) threads
  | refused -> refused

(* nx.cpu reads the terms of a program that is its one operand's identity
   in place, and those of another, or of several reductions, through the
   program, block by block (reduce.c's Programs), over the loop's [shape].
   A padded load passes its pad and its fill's bits. *)
let reduce_answer ~threads p ~shape ~axes rs ~dsts
    (loads : Nx_kernel.load array) =
  if not (all_accepted p rs 0) then Nx_array.Wrong_dtype
  else if not (fits shape loads 0) then Nx_array.Shape_mismatch
  else
    match (rs, loads) with
    | [| (r, 0) |], [| l |] when Nx_kernel.Prog.is_identity p ->
        let (Any x) = operand l in
        if Array.length dsts = width r && computes r (Nx_array.dtype x) dsts 0
        then reduce_c (reduction r) axes dsts x (padding l) threads
        else refusal r dsts
    | _ -> fused_reduce ~threads p ~shape ~axes rs ~dsts loads

(* A scan of a program other than its one operand's identity, through it,
   over its destinations' shape. *)
let fused_scan ~threads p ~axis (r, k) ~dsts loads =
  let (D.Any dt) = Nx_kernel.Prog.dtype p (Iarray.get p.outs k) in
  let shape (Nx_array.Any d) = Nx_array.Layout.shape (Nx_array.layout d) in
  match (r, dsts) with
  | Nx_kernel.Moments, [| mean; _ |] when floats dt ->
      reduce_prog_c 2 p [| code Moments; k |] (shape mean) [| axis |] dsts
        loads (pads loads) threads
  | _ when Array.length dsts = width r && computes r dt dsts 0 ->
      reduce_prog_c 1 p
        [| code (reduction r); k |]
        (shape dsts.(0)) [| axis |] dsts loads (pads loads) threads
  | _ -> refusal r dsts

let scan_answer ~threads p ~axis rk ~dsts (loads : Nx_kernel.load array) =
  if not (accepted p rk) then Nx_array.Wrong_dtype
  else if not (Nx_kernel.Prog.is_identity p) then
    fused_scan ~threads p ~axis rk ~dsts loads
  else
    match (rk, dsts, loads) with
    | (Moments, 0), [| mean; var |], [| l |] ->
        let (Any x) = operand l in
        moments_c axis mean var x (padding l) threads
    | (r, 0), _, [| l |] ->
        let (Any x) = operand l in
        if Array.length dsts = width r && computes r (Nx_array.dtype x) dsts 0
        then scan_c (reduction r) axis dsts x (padding l) threads
        else refusal r dsts
    | _ -> Nx_array.Bad_arity

(* The operands of a reduction or scan, to name a decline. *)
let loop_operands dsts loads =
  Array.to_list dsts @ Array.to_list (Array.map operand loads)

let reduce_on ~threads p ~shape ~axes rs ~dsts loads =
  match reduce_answer ~threads p ~shape ~axes rs ~dsts loads with
  | Nx_array.Declined -> declined "reduce" (loop_operands dsts loads)
  | answer -> answer

let scan_on ~threads p ~axis rk ~dsts loads =
  match scan_answer ~threads p ~axis rk ~dsts loads with
  | Nx_array.Declined -> declined "scan" (loop_operands dsts loads)
  | answer -> answer

let reduce p ~shape ~axes rs ~dsts loads =
  reduce_on ~threads:0 p ~shape ~axes rs ~dsts loads

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

external fold_c :
  Nx_kernel.pad ->
  dst:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_fold_pad"

let fold p ~dst x =
  match fold_c p ~dst x with
  | Nx_array.Declined -> declined "fold" [ Any dst; Any x ]
  | answer -> answer

(* Contractions (gemm.c) *)

(* Chain order where a contraction has the outputs to keep every target's
   chains in flight, 64 (NEON keeps 16 vectors of 4 chains, 4 pipes of 4
   cycles; AVX2 8 of 8); lane order where its rows or columns are few, as
   decoding a token's are, so that b streams once in the order it lies. *)
let chain_outputs = 64
let lane_most = 4

let contract_order ~rows ~columns : Nx_kernel.order =
  (* [rows · columns >= chain_outputs], without overflow: a factor of
     chain_outputs or more makes it, and below that the product fits. *)
  let outputs =
    rows >= chain_outputs || columns >= chain_outputs
    || rows * columns >= chain_outputs
  in
  if rows > lane_most && columns > lane_most && outputs then Chain else Lane

(* [contract_c order ~batch ~contracting acc ~dst ~init a b] contracts [a]
   and [b], with [init] if given, in [order] and the dtype of code [acc],
   over the view C groups from the pairs on its stack
   ([nx_contract_view_fill]). *)
external contract_c :
  Nx_kernel.order ->
  batch:(int * int) array ->
  contracting:(int * int) array ->
  int ->
  dst:('v, 's) Nx_array.t ->
  init:('v, 's) Nx_array.t option ->
  ('a, 'b) Nx_array.t ->
  ('c, 'd) Nx_array.t ->
  Nx_array.answer = "nx_cpu_contract_byte" "nx_cpu_contract"

(* An [acc] {!Nx_kernel.accumulates} refuses is the caller's error, refused
   before the kernel is asked. *)
let contract ~order ~batch ~contracting ~acc ~dst ~init a b =
  let (Nx_array.Dtype.Any dt) = acc in
  if not (Nx_kernel.accumulates dt) then Nx_array.Wrong_dtype
  else
    match
      contract_c order ~batch ~contracting (Nx_array.Dtype.code dt) ~dst
        ~init a b
    with
    | Nx_array.Declined ->
        let init =
          Option.fold ~none:[] ~some:(fun i -> [ Nx_array.Any i ]) init
        in
        declined "contract" ([ Nx_array.Any dst; Any a; Any b ] @ init)
    | answer -> answer

(* Fourier transforms (fft.c). The shapes are checked here; the externals take
   [threads] as reduce's do, and the direction as its int. *)

external c2c_c :
  int ->
  Nx_kernel.direction ->
  int array ->
  (Complex.t, 's) Nx_array.t ->
  (Complex.t, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_c2c"

external r2c_c :
  int ->
  int array ->
  (Complex.t, 'c) Nx_array.t ->
  (float, 'r) Nx_array.t ->
  Nx_array.answer = "nx_cpu_r2c"

external c2r_c :
  int ->
  int array ->
  (float, 'r) Nx_array.t ->
  (Complex.t, 'c) Nx_array.t ->
  Nx_array.answer = "nx_cpu_c2r"

let shape a = Nx_array.Layout.shape (Nx_array.layout a)

(* Whether [dst] has the shape [Nx_kernel.Shape] gives, if it gives one. *)
let fits want dst =
  match want with
  | Error _ -> false
  | Ok s ->
      let l = Nx_array.layout dst in
      let dim = Nx_array.Layout.dim l in
      let rec same i = i = Array.length s || (s.(i) = dim i && same (i + 1)) in
      Array.length s = Nx_array.Layout.rank l && same 0

let c2c_on ~threads d ~axes ~dst x =
  if not (fits (Nx_kernel.Shape.c2c ~axes (shape x)) dst) then
    Nx_array.Shape_mismatch
  else c2c_c threads d axes dst x

let r2c_on ~threads ~axes ~dst x =
  if not (Nx_kernel.same_precision (Nx_array.dtype x) (Nx_array.dtype dst)) then
    Nx_array.Wrong_dtype
  else if not (fits (Nx_kernel.Shape.r2c ~axes (shape x)) dst) then
    Nx_array.Shape_mismatch
  else r2c_c threads axes dst x

let c2r_on ~threads ~axes ~dst x =
  if not (Nx_kernel.same_precision (Nx_array.dtype dst) (Nx_array.dtype x)) then
    Nx_array.Wrong_dtype
  else
    let s = shape dst and k = Array.length axes - 1 in
    let n =
      if k >= 0 && axes.(k) >= 0 && axes.(k) < Array.length s then s.(axes.(k))
      else -1
    in
    if n < 0 || not (fits (Nx_kernel.Shape.c2r ~axes ~n (shape x)) dst) then
      Nx_array.Shape_mismatch
    else c2r_c threads axes dst x

let c2c d ~axes ~dst x = c2c_on ~threads:0 d ~axes ~dst x
let r2c ~axes ~dst x = r2c_on ~threads:0 ~axes ~dst x
let c2r ~axes ~dst x = c2r_on ~threads:0 ~axes ~dst x

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

external eigh_c :
  w:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t option ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_eigh"

let eigh ~w ?v a = eigh_c ~w v a

external svd_c :
  s:('v, 's) Nx_array.t ->
  (u:('v, 's) Nx_array.t * vh:('v, 's) Nx_array.t) option ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_svd"

let svd ~s ?vectors a = svd_c ~s vectors a

external eig_c :
  w:(Complex.t, 'c) Nx_array.t ->
  (Complex.t, 'c) Nx_array.t option ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_eig"

let eig ~w ?v a =
  let dt = Nx_array.dtype a in
  let fits (x : (Complex.t, _) Nx_array.t) =
    Nx_kernel.same_precision dt (Nx_array.dtype x)
  in
  let fit = fits w && Option.fold ~none:true ~some:fits v in
  if not (Nx_kernel.full_precision dt && fit) then Nx_array.Wrong_dtype
  else eig_c ~w v a
