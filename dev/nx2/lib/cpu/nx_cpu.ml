(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Raises where the host lacks the instructions nx.cpu is compiled for
   (target.c): the first code of nx.cpu to run. *)
external require : unit -> unit = "nx_cpu_require"

let () = require ()

module D = Nx_array.Dtype

let name = "nx.cpu"
let computes_on d = Rig.reaches Rig.host d

(* Elementwise kernels (copy.c, convert.c, apply.c), which [map] runs for
   a program of one node: the kind passes as its value. Each reads its
   operands through the door at the dtypes its kind takes, so their types
   here are any. *)

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

(* Assemblies and folds (assemble.c) *)

external assemble_c :
  string ->
  ('v, 's) Nx_array.t ->
  (Nx_array.Move.range iarray * ('v, 's) Nx_array.t) array ->
  Nx_array.answer = "nx_cpu_assemble"

let assemble ~fill ~dst pieces =
  assemble_c (Nx_kernel.Prog.bits (Nx_array.dtype dst) fill) dst pieces

external fold :
  Nx_kernel.pad ->
  dst:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_fold_pad"

(* Loads' shapes, checked by Nx_kernel.Shape's rule before a padded load
   is realised: C sees the realised copy alone, and a misfit is refused
   before the copy is allocated. *)

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
         let s = Iarray.to_array (Nx_array.Layout.shape (Nx_array.layout x)) in
         Nx_kernel.Shape.padded pad s = Ok shape)
     && fits shape loads (i + 1)

(* The shape of the destination [d], whose shape a map's or a scan's loop
   has. *)
let loop_of (Nx_array.Any d) =
  Iarray.to_array (Nx_array.Layout.shape (Nx_array.layout d))

(* Padded loads

   A padded load is realised before a kernel reads it: a host array of the
   operand padded, before its windows, assembled from the operand's
   elements where they land and the pad's value elsewhere, then broadcast
   and windowed as a view of that array. An axis the operand broadcasts
   and the pad leaves alone is held at extent 1, so a convolution's copy
   holds its input once, never once per output channel. The C kernels
   then read plain loads alone. *)

(* The extent of an axis of extent [d] padded by [pad] along axis [i],
   before the windows. *)
let padded_extent (pad : Nx_kernel.pad) d i =
  Iarray.get pad.lo i + Iarray.get pad.hi i + d
  + if d > 0 then Iarray.get pad.interior i * (d - 1) else 0

(* The ranges of an axis of [held] elements of the copy, and of [x]'s [d]
   along it, that [x]'s elements land on: [x]'s [t] at [lo + t s], [s] the
   interior padding plus one, for each [t] that lands in [0, held). *)
let landing (pad : Nx_kernel.pad) ~held d i =
  let lo = Iarray.get pad.lo i and s = Iarray.get pad.interior i + 1 in
  let first = if lo >= 0 then 0 else (s - 1 - lo) / s in
  let last =
    if held - 1 - lo < 0 then -1 else min (d - 1) ((held - 1 - lo) / s)
  in
  let count = max 0 (last - first + 1) in
  let into = { Nx_array.Move.start = lo + (first * s); count; step = s } in
  (into, { Nx_array.Move.start = first; count; step = 1 })

(* [x] padded with [value] as [pad] says, over a fresh host array whose
   buffer is pushed onto [buffers] for the caller to free, or the refusal
   its assembly answers. [x]'s padded shape is one
   {!Nx_kernel.Shape.padded} admits. *)
let realise (type v s) ~buffers (x : (v, s) Nx_array.t) (value : v)
    (pad : Nx_kernel.pad) =
  let l = Nx_array.layout x and dt = Nx_array.dtype x in
  let d = Nx_array.Layout.dim l and r = Nx_array.Layout.rank l in
  let shape = Iarray.init r (fun i -> padded_extent pad (d i) i) in
  let alone i =
    Iarray.get pad.lo i = 0 && Iarray.get pad.hi i = 0
    && Iarray.get pad.interior i = 0
  in
  let held =
    Iarray.init r (fun i ->
        if d i > 1 && Nx_array.Layout.stride l i = 0 && alone i then 1
        else Iarray.get shape i)
  in
  let a = Nx_array.create Rig.host dt held in
  buffers := Nx_array.buffer a :: !buffers;
  let ranges =
    Array.init r (fun i -> landing pad ~held:(Iarray.get held i) (d i) i)
  in
  let region = Iarray.init r (fun i -> fst ranges.(i)) in
  let slice = Iarray.init r (fun i -> snd ranges.(i)) in
  let piece = Option.get (Nx_array.move (Slice slice) x) in
  match assemble ~fill:value ~dst:a [| (region, piece) |] with
  | Done ->
      let y = Option.get (Nx_array.move (Broadcast shape) a) in
      if Iarray.length pad.windows = 0 then Ok y
      else Ok (Option.get (Nx_array.move (Window pad.windows) y))
  | refusal -> Error refusal

(* Whether a load of [loads] from [i] on is padded: a loop of its own, as
   [Array.exists]' closure would allocate on every reduction. *)
let rec any_padded (loads : Nx_kernel.load array) i =
  i < Array.length loads
  && ((match loads.(i) with Plain _ -> false | Padded _ -> true)
     || any_padded loads (i + 1))

(* [f] of [loads], each padded one realised: the copies are freed once [f]
   returns or raises, or a copy does. The first refusal a copy answers is
   the answer. *)
let realised (loads : Nx_kernel.load array) f =
  let buffers = ref [] in
  let free () =
    List.iter (fun b -> ignore (Rig.Buffer.free b : bool)) !buffers
  in
  Fun.protect ~finally:free @@ fun () ->
  let rec go i (plain : Nx_kernel.load array) =
    if i = Array.length loads then f plain
    else
      match loads.(i) with
      | Plain _ -> go (i + 1) plain
      | Padded { x; fill; pad } -> (
          match realise ~buffers x fill pad with
          | Error refusal -> refusal
          | Ok y ->
              plain.(i) <- Plain y;
              go (i + 1) plain)
  in
  go 0 (Array.copy loads)

(* Maps (map.c), over plain loads. *)

external map_c :
  Nx_kernel.Prog.t ->
  Nx_array.any array ->
  Nx_kernel.load array ->
  Nx_array.answer = "nx_cpu_map"

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
   checked the kind against its operands' dtypes, so each kind's table
   has its row. A node no kernel matches runs through map.c. *)
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
    | (In _ | Op1 _ | Op2 _ | Op3 _), _ -> map_c p [| Any dst |] loads

(* A program of one node over plain loads, into one destination, runs by
   its kind's kernel, which reads its operands' layouts directly; any
   other through map.c. *)
let map_plain p ~dsts loads =
  if
    Array.length dsts = 1
    && Nx_kernel.Prog.is_single p
    && plain_of p.ins loads 0
  then
    let last = Iarray.get p.nodes (Iarray.length p.nodes - 1) in
    single p last dsts.(0) loads
  else map_c p dsts loads

let map p ~dsts loads =
  if not (any_padded loads 0) then map_plain p ~dsts loads
  else if Array.length dsts = 0 || not (fits (loop_of dsts.(0)) loads 0) then
    Nx_array.Shape_mismatch
  else realised loads (map_plain p ~dsts)

(* Reductions and scans (reduce.c), of a program's outputs through it, over
   plain loads: an external each, which reads the terms of a program that
   is its one load's identity in place and those of another block by block
   (reduce.c's Programs), and checks each reduction's destinations and a
   scan's carry. Neither allocates for a program of plain loads. *)

external reduce_c :
  Nx_kernel.Prog.t ->
  (Nx_kernel.reduction * int) array ->
  int array ->
  int iarray ->
  Nx_array.any array ->
  Nx_kernel.load array ->
  Nx_array.answer = "nx_cpu_reduce_byte" "nx_cpu_reduce"

(* A scan, over its first destination's shape. *)
external scan_c :
  Nx_kernel.Prog.t ->
  Nx_kernel.reduction * int ->
  int ->
  Nx_array.any array ->
  Nx_kernel.load array ->
  Nx_array.any option ->
  Nx_array.answer = "nx_cpu_scan_byte" "nx_cpu_scan"

(* Whether reduction [r] accepts the dtype of [p]'s output [k]. *)
let accepted (p : Nx_kernel.Prog.t) (r, k) =
  let (D.Any dt) = Nx_kernel.Prog.dtype p (Iarray.get p.outs k) in
  Nx_kernel.accepts r dt

let rec all_accepted p rs i =
  i = Array.length rs || (accepted p rs.(i) && all_accepted p rs (i + 1))

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

(* Whether [dsts] are the destinations of the reductions [rs] along [axes]
   of a loop of [shape]: one each, two for an Arg or a Moments, of the
   shape {!Nx_kernel.Shape.reduce} gives. A reduction's loop is given as a
   shape alone, which bounds no padded copy: C checks its destinations
   after the copy, so a padded reduction checks them first. *)
let reduces_into ~shape ~axes rs dsts =
  let count n ((r : Nx_kernel.reduction), _) =
    n + match r with Arg _ | Moments -> 2 | Monoid _ | Logsumexp -> 1
  in
  match Nx_kernel.Shape.reduce ~axes (Array.map fst rs) shape with
  | Error _ -> false
  | Ok want ->
      Array.length dsts = Array.fold_left count 0 rs
      && Array.for_all (fun d -> loop_of d = want) dsts

let reduce p ~shape ~axes rs ~dsts loads =
  if not (all_accepted p rs 0) then Nx_array.Wrong_dtype
  else if not (fits shape loads 0) then Nx_array.Shape_mismatch
  else if not (any_padded loads 0) then reduce_c p rs shape axes dsts loads
  else if not (reduces_into ~shape ~axes rs dsts) then Nx_array.Shape_mismatch
  else realised loads (reduce_c p rs shape axes dsts)

let scan ?carry p ~axis rk ~dsts loads =
  if not (accepted p rk) then Nx_array.Wrong_dtype
  else if Array.length dsts = 0 || not (fits_dst dsts.(0) loads 0) then
    Nx_array.Shape_mismatch
  else if not (any_padded loads 0) then scan_c p rk axis dsts loads carry
  else realised loads (fun loads -> scan_c p rk axis dsts loads carry)

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

(* Contractions (gemm.c) *)

(* [contract_c ~batch ~contracting acc ~dst ~init a b sa sb segs]
   contracts [a] and [b], scaled by [sa] and [sb], with [init] if given and
   cut by [segs], in the dtype of code [acc], over the view C groups from
   the pairs on its stack ([nx_contract_fill], gemm_view.c). *)
external contract_c :
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
let contract ~batch ~contracting ~segments ~acc ~dst ~init
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let (Nx_array.Dtype.Any dt) = acc in
  let (Nx_array.Any a) = oa.array in
  let (Nx_array.Any b) = ob.array in
  let scaled = Option.is_some oa.scale || Option.is_some ob.scale in
  let float = match acc with Any (Float32 | Float64) -> true | _ -> false in
  if not (Nx_kernel.accumulates dt) then Nx_array.Wrong_dtype
  else if scaled && not float then Nx_array.Declined
  else
    contract_c ~batch ~contracting (Nx_array.Dtype.code dt) ~dst ~init
      a b oa.scale ob.scale segments

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

let shape a = Iarray.to_array (Nx_array.Layout.shape (Nx_array.layout a))

(* Whether [dst] has the shape [Nx_kernel.Shape] gives, if it gives one. *)
let has_shape want dst =
  match want with
  | Error _ -> false
  | Ok s ->
      let l = Nx_array.layout dst in
      let dim = Nx_array.Layout.dim l in
      let rec same i = i = Array.length s || (s.(i) = dim i && same (i + 1)) in
      Array.length s = Nx_array.Layout.rank l && same 0

let c2c d ~axes ~dst x =
  if not (has_shape (Nx_kernel.Shape.c2c ~axes (shape x)) dst) then
    Nx_array.Shape_mismatch
  else c2c_c d axes dst x

let r2c ~axes ~dst x =
  if not (Nx_kernel.same_precision (Nx_array.dtype x) (Nx_array.dtype dst)) then
    Nx_array.Wrong_dtype
  else if not (has_shape (Nx_kernel.Shape.r2c ~axes (shape x)) dst) then
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
    if n < 0 || not (has_shape (Nx_kernel.Shape.c2r ~axes ~n (shape x)) dst)
    then Nx_array.Shape_mismatch
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

(* Whether the real result [x] is of [a]'s precision, as eigh and svd store
   it. *)
let real_of a x =
  Nx_kernel.same_precision (Nx_array.dtype a) (Nx_array.dtype x)

external eigh_c :
  w:(float, 'r) Nx_array.t ->
  ('v, 's) Nx_array.t option ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_eigh"

let eigh ~w ?v a =
  if not (real_of a w) then Nx_array.Wrong_dtype else eigh_c ~w v a

external svd_c :
  s:(float, 'r) Nx_array.t ->
  (u:('v, 's) Nx_array.t * vh:('v, 's) Nx_array.t) option ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_svd"

let svd ~s ?vectors a =
  if not (real_of a s) then Nx_array.Wrong_dtype else svd_c ~s vectors a

external eig_c :
  w:(Complex.t, 'c) Nx_array.t ->
  (Complex.t, 'c) Nx_array.t option ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_eig"

let eig ~w ?v a =
  let dt = Nx_array.dtype a in
  let paired (x : (Complex.t, _) Nx_array.t) =
    Nx_kernel.same_precision dt (Nx_array.dtype x)
  in
  let fit = paired w && Option.fold ~none:true ~some:paired v in
  if not (Nx_kernel.full_precision dt && fit) then Nx_array.Wrong_dtype
  else eig_c ~w v a

(* One thread *)

external swap_one : bool -> bool = "nx_cpu_one_thread" [@@noalloc]

let one_thread f =
  let was = swap_one true in
  Fun.protect ~finally:(fun () -> ignore (swap_one was : bool)) f
