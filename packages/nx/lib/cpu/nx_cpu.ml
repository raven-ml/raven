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
  (Nx_array.Move.range iarray * ('v, 's) Nx_array.t) iarray ->
  Nx_array.answer = "nx_cpu_assemble"

let assemble ~fill ~dst pieces =
  assemble_c (Nx_kernel.Prog.bits (Nx_array.dtype dst) fill) dst pieces

external fold :
  Nx_kernel.pad ->
  dst:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_fold_pad"

(* Padded loads' shapes. C checks that each load has the loop's shape,
   but sees a padded load only once realised: these checks, by
   Nx_kernel.Shape's rule, refuse a misfit before its copy is allocated. *)

(* Whether the shape [s] is [t] from axis [i] on: a loop of its own, as a
   local one's closure would allocate. *)
let rec same_from (s : int array) (t : int iarray) i =
  i = Array.length s || (s.(i) = Iarray.get t i && same_from s t (i + 1))

let same_shape s t = Array.length s = Iarray.length t && same_from s t 0

(* Whether each padded load of [loads] from [i] on has the loop's [shape]
   once loaded, {!Nx_kernel.Shape.padded}'s. *)
let rec padded_fit shape (loads : Nx_kernel.load iarray) i =
  i = Iarray.length loads
  || (match Iarray.get loads i with
     | Plain _ -> true
     | Padded { x; pad; _ } -> (
         let s = Iarray.to_array (Nx_array.Layout.shape (Nx_array.layout x)) in
         match Nx_kernel.Shape.padded pad s with
         | Ok s -> same_shape s shape
         | Error _ -> false))
     && padded_fit shape loads (i + 1)

(* The shape of the destination [d], whose shape a map's or a scan's loop
   has. *)
let loop_of (Nx_array.Any d) = Nx_array.Layout.shape (Nx_array.layout d)

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
let rec any_padded (loads : Nx_kernel.load iarray) i =
  i < Iarray.length loads
  && ((match Iarray.get loads i with Plain _ -> false | Padded _ -> true)
     || any_padded loads (i + 1))

(* Each padded load of [loads] from [i] on realised into [plain], which
   holds [loads]: [Done], or the first refusal a copy answers, after which
   no copy is made. A loop of its own, as a local one's closure would
   allocate. *)
let rec realise_from ~buffers (loads : Nx_kernel.load iarray)
    (plain : Nx_kernel.load array) i : Nx_array.answer =
  if i = Iarray.length loads then Done
  else
    match Iarray.get loads i with
    | Plain _ -> realise_from ~buffers loads plain (i + 1)
    | Padded { x; fill; pad } -> (
        match realise ~buffers x fill pad with
        | Error refusal -> refusal
        | Ok y ->
            plain.(i) <- Plain y;
            realise_from ~buffers loads plain (i + 1))

(* [f] of [loads], each padded one realised: the copies are freed once [f]
   returns or raises, or a copy does. The first refusal a copy answers is
   the answer. *)
let realised (loads : Nx_kernel.load iarray) f =
  let buffers = ref [] in
  let free () =
    List.iter (fun b -> ignore (Rig.Buffer.free b : bool)) !buffers
  in
  Fun.protect ~finally:free @@ fun () ->
  let plain = Iarray.to_array loads in
  match realise_from ~buffers loads plain 0 with
  | Done -> f (Iarray.of_array plain)
  | refusal -> refusal

(* Maps (map.c), over plain loads. *)

external map_c :
  Nx_kernel.Prog.t ->
  Nx_array.any iarray ->
  Nx_kernel.load iarray ->
  Nx_array.answer = "nx_cpu_map"

(* [n], the one node of [p], into the one of [dsts] over plain loads of
   [p]'s operands' dtypes, by its kind's kernel, allocating nothing. The
   program has checked the kind against its operands' dtypes, so each
   kind's table has its row. A node no kernel matches runs through
   map.c. *)
let single p n dsts (loads : Nx_kernel.load iarray) =
  let (Nx_array.Any dst) = Iarray.get dsts 0 in
  match ((n : Nx_kernel.Prog.node), loads) with
  | Const (_, b), _ -> fill b dst
  | Coord axis, _ -> iota axis dst
  | Op1 (Copy, _, _), [| Nx_kernel.Plain x |] -> copy ~dst x
  | Op1 (Cast, _, _), [| Plain x |] -> cast ~dst x
  | Op1 (Bitcast, _, _), [| Plain x |] -> bitcast ~dst x
  | Op1 (Unary u, _, _), [| Plain x |] -> apply1_c u dst x
  | Op2 (k, _, _), [| Plain x; Plain y |] -> apply2_c k dst x y
  | Op3 (k, _, _, _), [| Plain c; Plain x; Plain y |] -> apply3_c k dst c x y
  | (In _ | Op1 _ | Op2 _ | Op3 _), _ -> map_c p dsts loads

(* A program of one node over plain loads, into one destination, runs by
   its kind's kernel, which reads its operands' layouts directly; any
   other through map.c. *)
let map_plain p ~dsts loads =
  if Iarray.length dsts = 1 && Nx_kernel.Prog.is_single p then
    single p (Iarray.get p.nodes (Iarray.length p.nodes - 1)) dsts loads
  else map_c p dsts loads

(* A loop's destinations and loads as Nx_kernel's rules read them, which
   refuse first, so that map.c and reduce.c meet only calls that fit. *)
let any_dtype (Nx_array.Any a) = D.Any (Nx_array.dtype a)

let arrays =
  {
    Nx_kernel.arrays = Iarray.length;
    dtype = (fun a i -> any_dtype (Iarray.get a i));
  }

let reductions =
  {
    Nx_kernel.reductions = Iarray.length;
    kind = (fun rs i -> fst (Iarray.get rs i));
    output = (fun rs i -> snd (Iarray.get rs i));
  }

let loads_of =
  {
    Nx_kernel.arrays = Iarray.length;
    dtype =
      (fun (a : Nx_kernel.load iarray) i ->
        match Iarray.get a i with
        | Plain x -> D.Any (Nx_array.dtype x)
        | Padded { x; _ } -> D.Any (Nx_array.dtype x));
  }

(* The rank of a loop over the shape of [dsts]' first, [0] for none. *)
let rank_of dsts =
  if Iarray.length dsts = 0 then 0
  else
    let (Nx_array.Any d) = Iarray.get dsts 0 in
    Nx_array.Layout.rank (Nx_array.layout d)

let map p ~dsts loads =
  match
    Nx_kernel.map_refusal p ~rank:(rank_of dsts) ~dsts:arrays dsts
      ~loads:loads_of loads
  with
  | Some (r, _) -> r
  | None ->
      if not (any_padded loads 0) then map_plain p ~dsts loads
      else if not (padded_fit (loop_of (Iarray.get dsts 0)) loads 0) then
        Nx_array.Shape_mismatch
      else realised loads (map_plain p ~dsts)

(* Reductions and scans (reduce.c), of a program's outputs through it, over
   plain loads: an external each, which reads the terms of a program that
   is its one load's identity in place and those of another block by block
   (reduce.c's Programs), and checks each reduction's destinations and a
   scan's carry. Neither allocates for a program of plain loads. *)

external reduce_c :
  Nx_kernel.Prog.t ->
  (Nx_kernel.reduction * int) iarray ->
  int iarray ->
  int iarray ->
  Nx_array.any iarray ->
  Nx_kernel.load iarray ->
  Nx_array.answer = "nx_cpu_reduce_byte" "nx_cpu_reduce"

(* A scan, over its first destination's shape. *)
external scan_c :
  Nx_kernel.Prog.t ->
  Nx_kernel.reduction * int ->
  int ->
  Nx_array.any iarray ->
  Nx_kernel.load iarray ->
  Nx_array.any option ->
  Nx_array.answer = "nx_cpu_scan_byte" "nx_cpu_scan"

(* Whether the layout [l] from its axis [k] on is the loop's [shape] from
   axis [i] on without [axes] from [j] on. *)
let rec kept_from l (shape : int iarray) (axes : int iarray) i j k =
  if i = Iarray.length shape then k = Nx_array.Layout.rank l
  else if j < Iarray.length axes && Iarray.get axes j = i then
    kept_from l shape axes (i + 1) (j + 1) k
  else
    k < Nx_array.Layout.rank l
    && Nx_array.Layout.dim l k = Iarray.get shape i
    && kept_from l shape axes (i + 1) j (k + 1)

(* Whether each destination of [dsts] from [i] on has the loop's [shape]
   without [axes], the shape {!Nx_kernel.Shape.reduce} gives where it
   gives one, compared in place. A reduction's loop is given as a shape
   alone, which bounds no padded copy: C checks its destinations, their
   number and the rest of Shape.reduce's rule after the copy, so a padded
   reduction checks their shapes first. *)
let rec reduces_into ~shape ~axes (dsts : Nx_array.any iarray) i =
  i = Iarray.length dsts
  ||
  let (Nx_array.Any d) = Iarray.get dsts i in
  kept_from (Nx_array.layout d) shape axes 0 0 0
  && reduces_into ~shape ~axes dsts (i + 1)

let reduce p ~shape ~axes rs ~dsts loads =
  match
    Nx_kernel.reduce_refusal p ~rank:(Iarray.length shape) ~reductions rs
      ~dsts:arrays dsts ~loads:loads_of loads
  with
  | Some (r, _) -> r
  | None ->
      if not (any_padded loads 0) then reduce_c p rs shape axes dsts loads
      else if
        not (padded_fit shape loads 0 && reduces_into ~shape ~axes dsts 0)
      then Nx_array.Shape_mismatch
      else realised loads (reduce_c p rs shape axes dsts)

let scan ?carry p ~axis ((r, k) as rk) ~dsts loads =
  match
    Nx_kernel.scan_refusal p ~rank:(rank_of dsts) r ~output:k ~dsts:arrays
      dsts ?carry:(Option.map any_dtype carry) ~loads:loads_of loads
  with
  | Some (r, _) -> r
  | None ->
      if not (any_padded loads 0) then scan_c p rk axis dsts loads carry
      else if not (padded_fit (loop_of (Iarray.get dsts 0)) loads 0) then
        Nx_array.Shape_mismatch
      else realised loads (fun loads -> scan_c p rk axis dsts loads carry)

(* Gathers and scatters (index.c) *)

external gather :
  axis:int ->
  dst:('v, 's) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_gather"

external gather_ranges :
  dst:('v, 's) Nx_array.t ->
  starts:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ends:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_gather_ranges"

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

(* Numbering (unique.c) *)

external unique :
  ids:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  first:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  counts:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  count:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  Nx_array.answer = "nx_cpu_unique"

external unique_seed : int64 -> unit = "nx_cpu_unique_seed" [@@noalloc]
external unique_state : int64 array -> int64 = "nx_cpu_unique_state"

(* The hash's seed, from the system's random source through the runtime's
   self-initialisation (/dev/urandom on Unix, BCryptGenRandom on Windows). *)
let () = unique_seed (Random.State.bits64 (Random.State.make_self_init ()))

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

(* Fourier transforms (fft.c). fft.c trusts its operands' shapes: they are
   checked here, by Nx_kernel.Shape's rule, which states each transform's
   shapes once. fft.c passes the door the dtypes it reads and writes. The
   externals take the direction as its int. *)

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
  if not (has_shape (Nx_kernel.Shape.r2c ~axes (shape x)) dst) then
    Nx_array.Shape_mismatch
  else r2c_c axes dst x

let c2r ~axes ~dst x =
  let s = shape dst and k = Iarray.length axes - 1 in
  let n =
    let a = if k >= 0 then Iarray.get axes k else -1 in
    if a >= 0 && a < Array.length s then s.(a) else -1
  in
  if n < 0 || not (has_shape (Nx_kernel.Shape.c2r ~axes ~n (shape x)) dst)
  then Nx_array.Shape_mismatch
  else c2r_c axes dst x

(* Matrices (linalg.c), which checks its operands' dtypes and shapes and
   passes the door the dtypes it reads and writes. *)

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

external eigh :
  w:(float, 'r) Nx_array.t ->
  ?v:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_eigh"

external svd :
  s:(float, 'r) Nx_array.t ->
  ?vectors:(u:('v, 's) Nx_array.t * vh:('v, 's) Nx_array.t) ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_svd"

external eig :
  w:(Complex.t, 'c) Nx_array.t ->
  ?v:(Complex.t, 'c) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Nx_array.answer = "nx_cpu_eig"

(* One thread *)

external swap_one : bool -> bool = "nx_cpu_one_thread" [@@noalloc]

let one_thread f =
  let was = swap_one true in
  Fun.protect ~finally:(fun () -> ignore (swap_one was : bool)) f

(* Target tables (target.c) *)

external targets_c : unit -> string list = "nx_cpu_targets"
external swap_target : string -> string = "nx_cpu_swap_target"

let targets = targets_c ()

external floor_holds : int -> int -> int -> bool = "nx_cpu_floor_holds"

let floor_holds ~ecx1 ~ebx7 ~xcr0 = floor_holds ecx1 ebx7 xcr0

let with_target t f =
  let was = swap_target t in
  Fun.protect ~finally:(fun () -> ignore (swap_target was : string)) f

(* The largest cache (target.c) *)

external swap_llc : int -> int = "nx_cpu_swap_llc" [@@noalloc]

let with_cache n f =
  let was = swap_llc n in
  Fun.protect ~finally:(fun () -> ignore (swap_llc was : int)) f

(* Attention *)

external attention_c :
  Nx_kernel.Prog.t ->
  Nx_kernel.Prog.t option ->
  float ->
  ('w, 'r) Nx_array.t ->
  (float, 'c) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t option ->
  ('v, 's) Nx_array.t ->
  (('v, 's) Nx_array.t, (int64, Nx_array.Dtype.int64_elt) Nx_array.t)
  Nx_kernel.kv ->
  Nx_array.any iarray ->
  Nx_array.answer = "nx_cpu_attention_byte" "nx_cpu_attention"

(* The ranges of the mask, output 0, over a box of pairs of a loop of
   [rank] axes, by which the kernel skips the blocks it hides (attention.c,
   Classes): none for a constant mask. *)
let mask_ranges (p : Nx_kernel.Prog.t) ~rank =
  match Iarray.get p.nodes (Iarray.get p.outs 0) with
  | Const _ -> None
  | _ -> Some (Nx_kernel.Prog.ranges ~axes:rank ~outs:[| 0 |] p)

(* attention.c computes what Nx_kernel.attention_refusal lets through. *)
let attention p ~scale ~o ~lse ?lengths q kv loads =
  let rank = Nx_array.Layout.rank (Nx_array.layout q) in
  match
    Nx_kernel.attention_refusal p ~q:(Nx_array.dtype q) ~rank
      ~o:(Nx_array.dtype o) ~acc:(Nx_array.dtype lse) ~loads:arrays loads
  with
  | Some (r, _) -> r
  | None -> attention_c p (mask_ranges p ~rank) scale o lse lengths q kv loads
