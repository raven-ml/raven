(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype

let name = "nx.cpu"
let computes_on d = Rig.reaches Rig.host d

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

(* Elementwise kernels (copy.c, convert.c, apply.c, iota.c), which [map]
   runs for a program of one node: the kind passes as its value. Each reads
   its operands through the door at the dtypes its kind takes, so their
   types here are any. *)

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

external fill : string -> ('v, 's) Nx_array.t -> Nx_array.answer
  = "nx_cpu_fill"

external iota : int -> ('v, 's) Nx_array.t -> Nx_array.answer = "nx_cpu_iota"

external apply2_c :
  Nx_kernel.Prog.op2 ->
  ('v, 's) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  Nx_array.answer = "nx_cpu_apply2"

external apply3_c :
  Nx_kernel.Prog.op3 ->
  ('v, 's) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('w, 'r) Nx_array.t ->
  Nx_array.answer = "nx_cpu_apply3"

(* Loads: a load's operand, and a padded one's pad and its fill's bits. *)

let operand : Nx_kernel.load -> Nx_array.any = function
  | Plain x -> Any x
  | Padded { x; _ } -> Any x

let padding : Nx_kernel.load -> _ = function
  | Plain _ -> None
  | Padded { x; fill; pad } ->
      Some (pad, Nx_kernel.Prog.bits (Nx_array.dtype x) fill)

(* Loads' shapes, checked by Nx_kernel.Shape's rule before an external
   copies a padded load. *)

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

(* The shape of the destination [d], whose shape a map's or a scan's loop
   has. *)
let loop_of (Nx_array.Any d) = Nx_array.Layout.shape (Nx_array.layout d)

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

(* Whether loads [i] on of [loads] are plain, one per operand from [i] on,
   each of the dtype [ins] names. *)
let rec plain_of ins (loads : Nx_kernel.load array) i =
  if i = Array.length loads then i = Iarray.length ins
  else
    i < Iarray.length ins
    && (match loads.(i) with
       | Plain x ->
           let (D.Any dt) = Iarray.get ins i in
           D.equal (Nx_array.dtype x) dt
       | Padded _ -> false)
    && plain_of ins loads (i + 1)

(* [n], the one node of [p], into [dst] over plain loads of [p]'s operands'
   dtypes, by its kind's kernel, allocating nothing. The program has
   checked the kind against its operands' dtypes. *)
let single (p : Nx_kernel.Prog.t) n (Nx_array.Any dst) loads =
  let (D.Any out) = Nx_kernel.Prog.dtype p (Iarray.length p.nodes - 1) in
  if not (D.equal (Nx_array.dtype dst) out) then Nx_array.Wrong_dtype
  else
    match ((n : Nx_kernel.Prog.node), loads) with
    | Const (_, b), _ -> fill b dst
    | Coord axis, _ -> iota axis dst
    | Op1 (Copy, _, _), [| Nx_kernel.Plain x |] -> copy ~dst x
    | Op1 (Cast, _, _), [| Plain x |] -> cast ~dst x
    | Op1 (Bitcast, _, _), [| Plain x |] -> bitcast ~dst x
    | Op1 (Unary u, _, _), [| Plain x |] -> apply1_c u dst x
    | Op2 (k, _, _), [| Plain x; Plain y |] -> apply2_c k dst x y
    | Op3 (k, _, _, _), [| Plain c; Plain x; Plain y |] ->
        apply3_c k dst c x y
    | (In _ | Op1 _ | Op2 _ | Op3 _), _ -> Nx_array.Declined

(* A program of one node over plain loads, into one destination, runs by
   its kind's kernel, which reads its operands' layouts directly; any
   other through map.c. *)
let map p ~dsts loads =
  let answer =
    match Nx_kernel.Prog.single p with
    | Some n when Array.length dsts = 1 && plain_of p.ins loads 0 ->
        single p n dsts.(0) loads
    | Some _ | None ->
        if not (Array.exists padded loads) then map_c p dsts loads [||]
        else if Array.length dsts > 0 && not (fits (loop_of dsts.(0)) loads 0)
        then Nx_array.Shape_mismatch
        else map_c p dsts loads (Array.map padding loads)
  in
  match answer with
  | Nx_array.Declined ->
      declined "map"
        (Array.to_list dsts @ Array.to_list (Array.map operand loads))
  | answer -> answer

(* Reductions and scans (reduce.c), of a program's outputs through it,
   each load passing its padding: an external each, which reads the terms
   of a program that is its one load's identity in place and those of
   another block by block (reduce.c's Programs), and checks each
   reduction's destinations and a scan's carry. Neither allocates for a
   program of plain loads. *)

external reduce_c :
  Nx_kernel.Prog.t ->
  (Nx_kernel.reduction * int) array ->
  int array ->
  int iarray ->
  Nx_array.any array ->
  Nx_kernel.load array ->
  (Nx_kernel.pad * string) option array ->
  Nx_array.answer = "nx_cpu_reduce_byte" "nx_cpu_reduce"

(* A scan, over its first destination's shape. *)
external scan_c :
  Nx_kernel.Prog.t ->
  Nx_kernel.reduction * int ->
  int ->
  Nx_array.any array ->
  Nx_kernel.load array ->
  (Nx_kernel.pad * string) option array ->
  Nx_array.any option ->
  Nx_array.answer = "nx_cpu_scan_byte" "nx_cpu_scan"

(* Whether reduction [r] accepts the dtype of [p]'s output [k]. *)
let accepted (p : Nx_kernel.Prog.t) (r, k) =
  let (D.Any dt) = Nx_kernel.Prog.dtype p (Iarray.get p.outs k) in
  Nx_kernel.accepts r dt

let rec all_accepted p rs i =
  i = Array.length rs || (accepted p rs.(i) && all_accepted p rs (i + 1))

(* Whether a load of [loads] from [i] on is padded: a loop of its own, as
   [Array.exists]' closure would allocate on every reduction. *)
let rec any_padded (loads : Nx_kernel.load array) i =
  i < Array.length loads && (padded loads.(i) || any_padded loads (i + 1))

(* The loads' paddings, or none where no load is padded. *)
let pads loads = if any_padded loads 0 then Array.map padding loads else [||]

(* Whether the layouts [l] and [l'] have one extent from axis [i] on, of
   one rank. *)
let rec same_extents l l' i =
  i = Nx_array.Layout.rank l
  || Nx_array.Layout.dim l i = Nx_array.Layout.dim l' i
     && same_extents l l' (i + 1)

(* Whether load [i] on of [loads] have the shape of the destination [d]
   once loaded, as {!fits} says, read without allocating for a plain
   load. *)
let rec fits_dst (Nx_array.Any d as any) (loads : Nx_kernel.load array) i =
  let l = Nx_array.layout d in
  i = Array.length loads
  || (match loads.(i) with
     | Plain x ->
         let lx = Nx_array.layout x in
         Nx_array.Layout.rank lx = Nx_array.Layout.rank l && same_extents lx l 0
     | Padded _ -> fits (loop_of any) loads i)
     && fits_dst any loads (i + 1)

(* The operands of a reduction or scan, to name a decline. *)
let loop_operands dsts loads =
  Array.to_list dsts @ Array.to_list (Array.map operand loads)

let reduce_answer p ~shape ~axes rs ~dsts loads =
  if not (all_accepted p rs 0) then Nx_array.Wrong_dtype
  else if not (fits shape loads 0) then Nx_array.Shape_mismatch
  else reduce_c p rs shape axes dsts loads (pads loads)

let scan_answer ?carry p ~axis rk ~dsts loads =
  if not (accepted p rk) then Nx_array.Wrong_dtype
  else if Array.length dsts = 0 then Nx_array.Bad_arity
  else if not (fits_dst dsts.(0) loads 0) then Nx_array.Shape_mismatch
  else scan_c p rk axis dsts loads (pads loads) carry

let reduce p ~shape ~axes rs ~dsts loads =
  match reduce_answer p ~shape ~axes rs ~dsts loads with
  | Nx_array.Declined -> declined "reduce" (loop_operands dsts loads)
  | answer -> answer

let scan ?carry p ~axis rk ~dsts loads =
  match scan_answer ?carry p ~axis rk ~dsts loads with
  | Nx_array.Declined -> declined "scan" (loop_operands dsts loads)
  | answer -> answer

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
  (Nx_array.Move.range iarray * ('v, 's) Nx_array.t) array ->
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

(* Lane order also where the depth is at least [lane_depth] products per
   output, as a weight gradient's few outputs over many tokens are: chain
   order's threads take tiles of outputs, too few to share the depth, where
   lane order's take ranges of its blocks. On the Core Ultra 5 235, 8 × 8
   outputs over 16384 products took 84-125 µs in chain order against 42-57
   in lane, 16 × 16 over 65536 663-873 against 272-293; 64 × 64 over 262144
   14.1 ms against 15.0-16.5, and 8 × 8 over 4096 21-25 µs against 44-46. *)
let lane_depth = 128

let contract_order ~rows ~columns ~depth : Nx_kernel.order =
  (* [rows · columns >= chain_outputs], without overflow: a factor of
     chain_outputs or more makes it, and below that the product fits. *)
  let outputs =
    rows >= chain_outputs || columns >= chain_outputs
    || rows * columns >= chain_outputs
  in
  (* [rows · columns <= depth / lane_depth], without overflow. *)
  let per = depth / lane_depth in
  let long = rows <= per && columns <= per / Int.max 1 rows in
  if rows > lane_most && columns > lane_most && outputs && not long then Chain
  else Lane

(* [contract_c order ~batch ~contracting acc ~dst ~init a b sa sb segs]
   contracts [a] and [b], scaled by [sa] and [sb], with [init] if given and
   cut by [segs], in [order] and the dtype of code [acc], over the view C
   groups from the pairs on its stack ([nx_contract_view_fill]). *)
external contract_c :
  Nx_kernel.order ->
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  int ->
  dst:('v, 's) Nx_array.t ->
  init:('v, 's) Nx_array.t option ->
  ('a, 'b) Nx_array.t ->
  ('c, 'd) Nx_array.t ->
  Nx_kernel.scale option ->
  Nx_kernel.scale option ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t Nx_kernel.segments option ->
  Nx_array.answer = "nx_cpu_contract_byte" "nx_cpu_contract"

(* An [acc] {!Nx_kernel.accumulates} refuses is the caller's error, refused
   before the kernel is asked. *)
let contract ~order ~batch ~contracting ~segments ~acc ~dst ~init
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let (Nx_array.Dtype.Any dt) = acc in
  let (Nx_array.Any a) = oa.array in
  let (Nx_array.Any b) = ob.array in
  let scaled = Option.is_some oa.scale || Option.is_some ob.scale in
  let float = match acc with Any (Float32 | Float64) -> true | _ -> false in
  if not (Nx_kernel.accumulates dt) then Nx_array.Wrong_dtype
  else if scaled && not float then Nx_array.Declined
  else
    match
      contract_c order ~batch ~contracting (Nx_array.Dtype.code dt) ~dst
        ~init a b oa.scale ob.scale segments
    with
    | Nx_array.Declined ->
        let init =
          Option.fold ~none:[] ~some:(fun i -> [ Nx_array.Any i ]) init
        in
        declined "contract" ([ Nx_array.Any dst; Any a; Any b ] @ init)
    | answer -> answer

(* Fourier transforms (fft.c). The shapes are checked here; the externals take
   the direction as its int. *)

external c2c_c :
  Nx_kernel.direction ->
  int iarray ->
  (Complex.t, 's) Nx_array.t ->
  (Complex.t, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_c2c"

external r2c_c :
  int iarray ->
  (Complex.t, 'c) Nx_array.t ->
  (float, 'r) Nx_array.t ->
  Nx_array.answer = "nx_cpu_r2c"

external c2r_c :
  int iarray ->
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

let c2c d ~axes ~dst x =
  if not (fits (Nx_kernel.Shape.c2c ~axes (shape x)) dst) then
    Nx_array.Shape_mismatch
  else c2c_c d axes dst x

let r2c ~axes ~dst x =
  if not (Nx_kernel.same_precision (Nx_array.dtype x) (Nx_array.dtype dst)) then
    Nx_array.Wrong_dtype
  else if not (fits (Nx_kernel.Shape.r2c ~axes (shape x)) dst) then
    Nx_array.Shape_mismatch
  else r2c_c axes dst x

let c2r ~axes ~dst x =
  if not (Nx_kernel.same_precision (Nx_array.dtype dst) (Nx_array.dtype x)) then
    Nx_array.Wrong_dtype
  else
    let s = shape dst and k = Iarray.length axes - 1 in
    let n =
      let a = if k >= 0 then Iarray.get axes k else -1 in
      if a >= 0 && a < Array.length s then s.(a) else -1
    in
    if n < 0 || not (fits (Nx_kernel.Shape.c2r ~axes ~n (shape x)) dst) then
      Nx_array.Shape_mismatch
    else c2r_c axes dst x

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

(* One thread *)

external swap_one : bool -> bool = "nx_cpu_one_thread" [@@noalloc]

let one_thread f =
  let was = swap_one true in
  Fun.protect ~finally:(fun () -> ignore (swap_one was : bool)) f
