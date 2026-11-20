(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ops

(* Two facts of every target staged so far. Shared memory serves [line] bytes at
   once, 32 banks of 4 bytes (CUDA C++ Programming Guide §2.3.4.2; one of
   RDNA4's two sets of 32 banks, ISA §12.1), and the widest copy, PTX's
   [cp.async], moves [copy] bytes. Each becomes a renderer's field when a
   target measures another value. An asynchronous copy, CUDA's alone, reads
   global memory in sectors of [sector] bytes: "the data access unit is
   32-byte regardless of whether global loads are cached in L1 or not" (CUDA
   C++ Best Practices Guide §10.2.1). *)
let line = 128
let copy = 16
let sector = 32

let extent u = Dtype.Value.to_int (vmax u) + 1

(* A staged product in registers takes several steps of its loop a trip
   ({!deepen}): a loop of those steps, inside the trips, is part of the trip's
   tile as the lanes are. It has its loop's axis type, so only its identity
   can mark it, as {!Codegen} marks the blocks of running sums it splits a
   loop into: those take the loop's identity followed by [Chains.chain], the
   steps take it followed by [step], one more, and so nest inside both the
   trips and the blocks. *)
let step = Chains.chain + 1

let is_step r =
  match List.rev (axis_id r) with s :: _ :: _ -> s = step | _ -> false

let is_inner r =
  List.mem (axis_type r) Axis_type.[ Warp; Local; Upcast ] || is_step r

let is_thread r = List.mem (axis_type r) Axis_type.[ Warp; Local ]

(* The elements of K a staged trip takes at most: 32. *)
let chunk = 32

let constant u =
  if Op.equal (op u) Op.Const then
    match value u with `Int z -> Some (Bigint.to_int z) | _ -> None
  else None

(* Whether [u] is a constant, or a cast of one, whose value [p] holds of. *)
let is_value p u =
  let c = if Op.equal (op u) Op.Cast then nth u 0 else u in
  Op.equal (op c) Op.Const && p (value u)

let is_zero =
  is_value (function
    | `Int z -> Bigint.(equal z zero)
    | `Float f -> f = 0.
    | _ -> false)

let is_true = is_value (function `Bool b -> b | _ -> false)
let reads_inner u = List.exists is_inner (Nodes.to_list (ranges u))

(* Depth *)

let depth ?(used = 0) (ren : Renderer.t) tiles =
  let stage = List.fold_left ( + ) 0 tiles in
  if
    (not (Setting.value Setting.stage))
    || (not (ren.has_local && ren.has_shared))
    || tiles = []
    || List.exists (fun b -> b < copy) tiles
  then 0
  else if ren.async_copy && used + (2 * stage) <= ren.shared_max then 2
  else if used + stage <= ren.shared_max then 1
  else 0

(* Layout *)

let rec pow2_factor n =
  if n > 0 && n mod 2 = 0 then 2 * pow2_factor (n / 2) else 1

(* A row of [cols] elements holds [chunks = p 2^j] copies, [p] odd. The 8 rows
   read at one chunk, and 8 consecutive chunks a copy writes, fall
   on 8 different 16-byte slots of a line when the low [g = min 2^j (line /
   copy)] bits of the chunk are permuted by the row: by [(row / per_phase) mod
   g], [per_phase = line / copy / g] rows sharing a permutation. With [p = 1]
   this is CuTe's [Swizzle<B, M, S>] and Triton's [perPhase] and [maxPhase];
   an odd count needs none, as its rows already cycle through the slots. *)
let layout ~itemsize ~cols col row =
  let flat = O.((row * int cols) + col) in
  let bytes = cols * itemsize in
  if itemsize > copy || bytes mod copy <> 0 then flat
  else
    let v = copy / itemsize in
    let g = min (pow2_factor (bytes / copy)) (line / copy) in
    if g = 1 then flat
    else
      let per_phase = line / copy / g in
      let phase = O.(row // int per_phase % int g) in
      let chunk = O.(col // int v lxor phase) in
      O.((row * int cols) + (chunk * int v) + (col % int v))

(* Tiles *)

(* [u] as a sum of terms [(c, t)], [c] times the atom [t], [None] standing for
   [1]. *)
let rec terms u =
  let scaled c a = List.map (fun (k, t) -> (k * c, t)) (terms a) in
  match (op u, src u) with
  | Op.Add, [ a; b ] -> terms a @ terms b
  | Op.Mul, [ a; c ] when constant c <> None ->
      scaled (Option.get (constant c)) a
  | Op.Mul, [ c; a ] when constant c <> None ->
      scaled (Option.get (constant c)) a
  | Op.Const, _ when constant u <> None -> [ (Option.get (constant u), None) ]
  | _ -> [ (1, Some u) ]

(* A dimension of a tile: its stride in memory, its extent, and the atoms
   whose sum, each times its coefficient, is the coordinate along it. A
   dimension of gathered rows has [rows = Some (c, t)]: its coordinate is a
   position in a row map, and [c t], [t] the row the map holds there, is
   that row's offset in memory; its stride is the size of the dimensions
   inside it. *)
type dim = {
  stride : int;
  extent : int;
  parts : (int * Ops.t) list;
  rows : (int * Ops.t) option;
}

(* The load of global memory through which alone [t] reads the threads and
   lanes, as a row map's: [t] is then the row at that load's coordinate. *)
let row_load t =
  let direct =
    toposort ~calls:Skip ~gate:(fun x -> not (Op.equal (op x) Op.Index)) t
  in
  (* The walk stops at loads, which it leaves out: they are the sources of
     the nodes it reaches. *)
  let loads =
    List.sort_uniq Ops.compare
      (List.filter
         (fun x -> Op.equal (op x) Op.Index && reads_inner x)
         (t :: List.concat_map src direct))
  in
  let reads_inner_directly =
    List.exists (fun x -> Op.equal (op x) Op.Range && is_inner x) direct
  in
  let global l = addrspace (nth l 0) = Some Dtype.Global in
  match loads with
  | [ l ] when global l && not reads_inner_directly -> Some l
  | _ -> None

(* The tile an index [idx] reads over the threads and lanes of a workgroup:
   the part of [idx] that the threads and lanes do not move (the global and
   loop ranges, a gather), and the dimensions, innermost first. A term [c t]
   whose [t] reads the threads and lanes only through one load, a row of a
   row map, is a dimension of rows, outermost, its coordinate the load's.
   [None] where a term mixes both otherwise, or the inner terms overlap. A
   range [inner] holds of counts with the threads and lanes. *)
let rec split ?(inner = is_inner) idx =
  let classify (c, a) =
    match a with
    | None -> `Outer (c, a)
    | Some t -> (
        let rs = Nodes.to_list (ranges t) in
        if not (List.exists inner rs) then `Outer (c, a)
        else if
          List.for_all inner rs && c > 0
          && Dtype.Value.to_int (vmin t) = 0
        then if extent t > 1 then `Inner (c, t) else `Unit
        else
          let map l = split (get_idx (nth l 1)) in
          match Option.map map (row_load t) with
          | Some (Some (_, [ d ])) when d.rows = None ->
              `Rows { d with rows = Some (c, t) }
          | _ -> `Mixed)
  in
  let classes = List.map classify (terms idx) in
  let rows = List.filter_map (function `Rows d -> Some d | _ -> None) classes in
  if List.mem `Mixed classes || List.length rows > 1 then None
  else
    let outer =
      List.fold_left
        (fun acc -> function
          | `Outer (c, None) -> O.(acc + int c)
          | `Outer (c, Some t) -> O.(acc + (t * int c))
          | _ -> acc)
        (int 0) classes
    in
    let inner =
      List.filter_map (function `Inner x -> Some x | _ -> None) classes
      |> List.stable_sort (fun (a, _) (b, _) -> Int.compare a b)
    in
    let rec group dims = function
      | [] -> Some (List.rev dims)
      | (c, t) :: rest -> (
          let e = extent t in
          match dims with
          | d :: ds when c = d.stride * d.extent ->
              let parts = (d.extent, t) :: d.parts in
              let d = { d with extent = d.extent * e; parts } in
              group (d :: ds) rest
          | d :: _ when c < d.stride * d.extent -> None
          | _ ->
              let d =
                { stride = c; extent = e; parts = [ (1, t) ]; rows = None }
              in
              group (d :: dims) rest)
    in
    (* A dimension of rows counts past the others in {!dense}: its stride
       there is their size. *)
    let past dims d =
      { d with stride = List.fold_left (fun n d -> n * d.extent) 1 dims }
    in
    match group [] inner with
    | None | Some [] -> None
    | Some dims -> Some (outer, dims @ List.map (past dims) rows)

(* The value of the integer expression [u] with each range bound by [env]. *)
let rec eval env u =
  match op u with
  | Op.Const -> value u
  | Op.Range -> `Int (Bigint.of_int (Tbl.find env u))
  | Op.Cast -> eval env (nth u 0)
  | o when Op.Set.mem o Op.Set.alu ->
      exec_alu o (dtype u) (List.map (eval env) (src u))
  | _ -> raise Exit

(* Whether the inner terms of the tile [dims] read each of its elements: over
   the values of the ranges they read, their sum takes as many values as the
   tile holds. A copy of the tile then reads only what the index reads. *)
let dense dims =
  let size = List.fold_left (fun n d -> n * d.extent) 1 dims in
  let rs =
    List.fold_left
      (fun acc d ->
        List.fold_left
          (fun acc (_, t) ->
            List.fold_left
              (fun acc r -> if List.memq r acc then acc else r :: acc)
              acc
              (Nodes.to_list (ranges t)))
          acc d.parts)
      [] dims
  in
  let points = List.fold_left (fun n r -> n * extent r) 1 rs in
  (* A workgroup's threads and a thread's lanes are at most a few thousand. *)
  points <= 1 lsl 16
  &&
  let env = Tbl.create 8 and seen = Hashtbl.create size in
  let value () =
    List.fold_left
      (fun acc d ->
        List.fold_left
          (fun acc (k, t) ->
            acc + (d.stride * k * Bigint.to_int (integer (eval env t))))
          acc d.parts)
      0 dims
  in
  let rec walk = function
    | [] -> Hashtbl.replace seen (value ()) ()
    | r :: rest ->
        for i = 0 to extent r - 1 do
          Tbl.replace env r i;
          walk rest
        done
  in
  match walk rs with
  | () -> Hashtbl.length seen = size
  | exception Exit -> false

(* Operands *)

(* An operand of a core read from global memory, as a tile. *)
(* An operand of a core read from global memory, as a tile: the buffer it
   reads, its elements' type, the part of its index the tile does not move,
   its dimensions, its gate if any ([0] read where it fails), each inner range
   the gate reads as the fields of the tile's coordinates that hold its bits
   ([(d, k, place, atom)]: the atom at [k] in dimension [d], holding the
   range's bits from [place]), the loops it steps through, outermost first,
   whether its outer part loads (a gather), and its size in elements. *)
type tile = {
  buf : Ops.t;
  dt : Dtype.t;
  outer : Ops.t;
  dims : dim list;
  valid : Ops.t option;
  fields : (Ops.t * (int * int * int * Ops.t) list) list;
  loops : Ops.t list;
  gathers : bool;
  size : int;
}

let bytes t = t.size * Dtype.itemsize t.dt

let is_serial r = List.mem (axis_type r) Axis_type.[ Weak; Loop ]

(* The loops of [outer]: its weak and loop ranges of more than one trip,
   outermost first, the order in which a node closing several nests them
   ({!Ops.compare_range_arg}). A range of one trip, as a core's [K] unrolled
   whole leaves, is no loop. *)
let loops_of outer =
  List.filter
    (fun r -> is_serial r && extent r > 1)
    (Nodes.to_list (ranges outer))
  |> List.sort compare_range_arg

let same_loops a b = List.equal ( == ) a.loops b.loops

(* Whether the loops [ls] of the core [w], outermost first, are a nest that
   two stages step through as one sequence of trips: none or one loop, or
   loops of constant extents that one node closes, closing no other loop of
   more than one trip inside the outermost but the steps of a trip
   ({!deepen}), which the tiles take whole. A loop closed elsewhere cannot
   run between them, since they end together. The trip after the last of an
   inner loop's is then the first of the next outer trip's. *)
let nest all w ls =
  match ls with
  | [] | [ _ ] -> true
  | l :: _ -> (
      let closes u =
        List.memq l (ended_ranges u)
        && Nodes.mem w (backward_slice_with_self ~calls:Skip u)
      in
      match List.find_opt closes all with
      | None -> false
      | Some y ->
          let inside =
            List.filter
              (fun r ->
                op r = Op.Range && is_serial r && extent r > 1
                && (not (is_step r))
                && compare_range_arg r l >= 0)
              (ended_ranges y)
            |> List.sort compare_range_arg
          in
          List.equal ( == ) inside ls
          && List.for_all (fun r -> constant (nth r 0) <> None) ls)

(* The trip of the nest [ls] that runs, counted outermost first, and the
   nest's count of trips: a loop's own range and bound where it is one. *)
let trip ls =
  match ls with
  | [ l ] -> (l, nth l 0)
  | _ ->
      let t, n =
        List.fold_left
          (fun (t, n) l -> (O.((t * int (extent l)) + l), n * extent l))
          (int 0, 1) ls
      in
      (t, int n)

(* Each loop of the nest [ls] at its trip [t], counted outermost first: [t]'s
   digits in the loops' extents. The outermost takes no remainder, so that the
   trip past the last one reads past the outermost loop's bound, where the
   copy's gate fails, rather than wrapping. *)
let digits ls t =
  match ls with
  | [ l ] -> [ (l, t) ]
  | _ ->
      let strides =
        fst
          (List.fold_right
             (fun l (acc, s) -> (s :: acc, s * extent l))
             ls ([], 1))
      in
      List.mapi
        (fun i (l, s) ->
          let q = O.(t // int s) in
          (l, if i = 0 then q else O.(q % int (extent l))))
        (List.combine ls strides)

(* The place of the lowest bit of the range [r] that the atom [t] holds: [d]
   where [t] is [r], [r / d], [r mod b] or [(r / d) mod b]. *)
let place r t =
  let div u =
    if u == r then Some 1
    else
      match (op u, src u) with
      | Op.Div, [ x; c ] when x == r -> constant c
      | _ -> None
  in
  match (op t, src t) with
  | Op.Mod, [ x; c ] when constant c <> None -> div x
  | _ -> div t

(* Each inner range [g] reads, as the fields of the tile [dims] that hold its
   bits, if the fields hold every bit of it once: the range is then the sum of
   its fields' values, each times its place. *)
let fields dims g =
  let of_range r =
    let found =
      List.concat
        (List.mapi
           (fun d dim ->
             List.filter_map
               (fun (k, a) -> Option.map (fun p -> (d, k, p, a)) (place r a))
               dim.parts)
           dims)
    in
    let env = Tbl.create 1 in
    let sum v =
      Tbl.replace env r v;
      List.fold_left
        (fun acc (_, _, p, a) ->
          acc + (p * Bigint.to_int (integer (eval env a))))
        0 found
    in
    let whole =
      List.for_all (fun v -> sum v = v) (List.init (extent r) Fun.id)
    in
    if found <> [] && whole then Some (r, found) else None
  in
  let rs = List.filter is_inner (Nodes.to_list (ranges g)) in
  let fs = List.filter_map of_range rs in
  if List.length fs = List.length rs then Some fs else None

(* The tile of a core's operand [x]: an index of global memory, gated by its
   own validity or by a [where] the core reads [0] outside, whose tile is
   dense and whose gate's threads and lanes the tile's coordinates hold. *)
let tile_of x =
  let x, gate =
    match (op x, src x) with
    | Op.Where, [ g; y; z ] when is_zero z -> (y, Some g)
    | _ -> (x, None)
  in
  match (op x, src x) with
  | Op.Index, [ buf; idx ] when addrspace buf = Some Dtype.Global -> (
      let valid =
        match (gate, get_valid idx) with
        | g, v when is_true v -> g
        | None, v -> Some v
        | Some g, v -> Some O.(g land v)
      in
      let held dims =
        List.for_all
          (fun d ->
            match d.rows with
            | Some (_, t) -> Option.is_some (fields dims t)
            | None -> true)
          dims
      in
      match split (get_idx idx) with
      | Some (outer, dims) when dense dims && held dims -> (
          let fs =
            match valid with Some g -> fields dims g | None -> Some []
          in
          match fs with
          | Some fields ->
              Some
                {
                  buf;
                  dt = dtype x;
                  outer;
                  dims;
                  valid;
                  fields;
                  loops = loops_of outer;
                  gathers =
                    List.exists (fun d -> Option.is_some d.rows) dims
                    || List.exists
                         (fun u -> Op.equal (op u) Op.Index)
                         (toposort ~calls:Skip outer);
                  size = List.fold_left (fun n d -> n * d.extent) 1 dims;
                }
          | None -> None)
      | _ -> None)
  | _ -> None

(* The tile of [x] that staging copies, in a kernel of the nodes [all]. A
   tile of gathered rows copies in one stage, each trip waiting for its
   copies, and is copied only where several warps read it: where a warp's
   index, a local range, does not move [x]'s index. On an RTX 5000 Ada,
   bfloat16 routed rows each of two warps read alone, blocks of 8 over 64
   tokens, ran 4.8% slower staged; at 512 tokens, staged and read by 4
   warps, MXFP4 routed rows ran 27% faster. Other tiles copied in one stage
   keep staging unmeasured either way. The heuristic asks before it splits
   any warp, so a core reading gathered rows keeps the unstaged splits. The
   staged tile's registers, 222 a thread against 125, halve the warps an SM
   holds, which pays where the grid fills its last wave and costs where it
   does not: bfloat16 experts' gate and up projection, in blocks of 64 rows,
   ran 2.5% faster staged over 48 blocks and 3.1-3.5% slower over 63. The
   last wave follows the card's count of SMs, which a target does not fix,
   so the splits that lose less, the unstaged, stay. The sign was measured
   on blocks of 64 rows; over blocks of 48 the two ran within 3%. A range
   [whole], a loop the tile planned takes whole, is none of its loops. *)
let staged_tile ?whole all x =
  let warps =
    List.filter
      (fun u -> Op.equal (op u) Op.Range && axis_type u = Axis_type.Local)
      all
  in
  let shared = List.exists (fun r -> not (Nodes.mem r (ranges x))) warps in
  let rows t = List.exists (fun d -> Option.is_some d.rows) t.dims in
  let unlooped t =
    match whole with
    | None -> t
    | Some k ->
        { t with loops = List.filter (fun l -> axis_id l <> axis_id k) t.loops }
  in
  match tile_of x with
  | Some t when rows t && not shared -> None
  | t -> Option.map unlooped t

(* Codes are integers that a load of integer storage gives, through integer
   arithmetic alone. A float's bits read as an integer, as a selection by sign
   computes ReLU of a bfloat16, are no codes. *)
let decoded o =
  let coded = Tbl.create 16 in
  let rec codes u =
    match Tbl.find_opt coded u with
    | Some b -> b
    | None ->
        let b =
          Dtype.is_int (dtype u)
          && (op u = Op.Index || List.exists codes (src u))
        in
        Tbl.add coded u b;
        b
  in
  let converts u =
    (op u = Op.Cast || op u = Op.Bitcast)
    && Dtype.is_float (dtype u)
    && codes (nth u 0)
  in
  List.exists converts
    (toposort ~calls:Skip ~gate:(fun u -> op u <> Op.Index) o)

let elementwise u =
  Op.Set.mem (op u) Op.Set.alu
  || List.mem (op u) Op.[ Cast; Bitcast; Const; Where ]

(* Decoded operands

   A decoded operand's loads are copied where each copy is one asynchronous
   copy of [copy] bytes, in the stage of the core's other operand: an MXFP4
   operand's codes, a sector of each row a trip ({!steps}). Other copies go
   through
   registers. gpt-oss-20b's Q8_0 projections, whose blocks of 34 bytes admit
   no asynchronous copy, staged so, copied by one warp and waited within the
   trip that copies, on the staged tile's smaller grid: on an RTX 5000 Ada
   their query, key and value ran 15-18% faster at 64 tokens but 1.09-1.19
   times slower at 512, where ptxas spilled at 255 registers, and their
   output ran 1.34-1.43 times slower at 64. *)

(* Whether every copy of [t] is one asynchronous copy of [copy] bytes: its
   innermost dimension contiguous in whole copies, every other term of its
   address a multiple of a copy, in storage that starts on one. *)
let copyable t =
  let n = Dtype.itemsize t.dt and d0 = List.hd t.dims in
  let whole c = c * n mod copy = 0 in
  let starts =
    match arg (buf_uop t.buf) with
    | Param p -> p.align >= copy && p.phase = 0
    | _ -> false
  in
  starts && n <= copy && d0.stride = 1 && whole d0.extent
  && List.for_all (fun (c, _) -> whole c) (terms t.outer)
  && List.for_all
       (fun d ->
         match d.rows with Some (c, _) -> whole c | None -> whole d.stride)
       (List.tl t.dims)

let simplified u =
  graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
    (After_sources Symbolic.sym)

(* The loads of global memory through which the operand [x] reads the
   threads and lanes, if [x] computes elementwise on them. The walk stops at
   loads and at ranges, whose bounds the operand does not compute on, and
   leaves them out: they are the sources of the nodes it reaches. The loads
   keep the walk's order, which the graph's structure fixes. *)
let global_loads x =
  let stops u = List.mem (op u) Op.[ Index; Range; Special ] in
  let walked = toposort ~calls:Skip ~gate:(fun u -> not (stops u)) x in
  if not (List.for_all elementwise walked) then None
  else
    Some
      (List.fold_left
         (fun acc u ->
           if
             Op.equal (op u) Op.Index
             && addrspace (nth u 0) = Some Dtype.Global
             && reads_inner u && not (List.memq u acc)
           then u :: acc
           else acc)
         [] (List.concat_map src walked)
      |> List.rev)

(* Held tiles

   A load whose one loop moves it by at most a line over all its trips reads
   the same lines at every trip, which the L1 keeps for the whole loop. An
   MXFP4 operand's scales are such a load: a byte a row a trip, 90 trips
   over a row of 2880, a warp's 8 rows in 8 lines. Held in shared memory,
   copied once before the loop, the same bytes are read each trip without a
   request to the L1, a warp's 8 rows at a pitch of 90 bytes on 8 different
   banks, as Marlin reads its scales from shared memory each step (Frantar
   et al., "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on
   Large Language Models", arXiv:2408.11743, §3.4). The copy reads what the
   loop reads: it is gated by the loop's bound and is one run of whole
   copies.

   Shared memory costs more than its bytes. An SM's L1 and shared memory
   are one store of 128 KB, shared memory carved from it in steps of 0, 8,
   16, 32, 64 or 100 KB (NVIDIA Ada Tuning Guide §1.4.2.2), so each staged
   byte can take L1 from the loads left there: on an RTX 5000 Ada,
   gpt-oss-20b's MXFP4 expert kernels given 14400 bytes of shared memory
   they do not read, their code otherwise the same, ran 8.3% to 28.6%
   slower. A tile is therefore held only where the operand's other loads are
   copied, and [cp.async.cg] caches those copies in the L2 alone: the
   operand then leaves nothing on the L1. Marlin copies every operand into
   shared memory so, "asynchronously via Ampere's cp.async instruction from
   global (or L2) to shared memory" (§3.4), and vLLM's [cp_async4], which
   copies its weights, is [cp.async.cg]
   ([csrc/quantization/marlin/marlin.cuh]). On gpt-oss-20b's real weights
   and routing at 32 values a trip, against neither copied: the scales held
   alone ran the down projection at 64 tokens 20% slower; the codes copied
   and the scales left as loads ran gate and up and down at 512 tokens
   0.0-0.4% faster; both, 4.8-5.6% faster. *)

(* The tile of [l] over the threads, lanes and the loop [k], if [k] moves
   [l] by at most a line over all its trips, the tile is one run of
   [copy]-byte copies at an address that keeps their alignment, and [l] has
   no gate of its own. *)
let held_tile k l =
  match (op l, src l) with
  | Op.Index, [ buf; idx ]
    when addrspace buf = Some Dtype.Global && is_true (get_valid idx) -> (
      let size = Dtype.itemsize (dtype l) in
      let idx = get_idx (simplified idx) in
      let inner r = is_inner r || r == k in
      let step d =
        List.find_map (fun (c, t) -> if t == k then Some c else None) d.parts
      in
      match split ~inner idx with
      | Some (outer, [ d ]) when d.stride = 1 && d.rows = None && dense [ d ]
        -> (
          match step d with
          | Some c when c * size * extent k <= line ->
              let t =
                {
                  buf;
                  dt = dtype l;
                  outer;
                  dims = [ d ];
                  valid = Some O.(k < nth k 0);
                  fields = [ (k, [ (0, c, 1, k) ]) ];
                  loops = [];
                  gathers = false;
                  size = d.extent;
                }
              in
              if copyable t then Some t else None
          | _ -> None)
      | _ -> None)
  | _ -> None

(* The loads of global memory of a decoded operand [x] through which it reads
   the threads and lanes, each [`Copied] with its tile where every copy of
   its tile a trip is one asynchronous copy ({!copyable}), its index
   simplified, as [(32 L + i) / 32] over lanes [i < 32] reads the loop [L]
   alone; else [`Held] with its held tile where it steps through one loop
   and has one; [None] where a load is neither. A load whose trip's tile
   copies whole is copied each trip, held or not: MXFP4's codes over a row
   of 256 values move 128 bytes over the loop, and held whole they would
   leave the operand no copied load. *)
let decoded_loads ?whole all x =
  let classify l =
    match (op l, src l) with
    | Op.Index, [ buf; idx ] -> (
        let simple = replace l ~src:[ buf; simplified idx ] in
        match staged_tile ?whole all simple with
        | Some t when copyable t -> Some (`Copied (l, t))
        | _ -> (
            match loops_of (get_idx (simplified idx)) with
            | [ k ] -> Option.map (fun t -> `Held (l, t)) (held_tile k l)
            | _ -> None))
    | _ -> None
  in
  match global_loads x with
  | None -> None
  | Some loads ->
      let classes = List.map classify loads in
      if List.exists Option.is_none classes then None
      else Some (List.filter_map Fun.id classes)

(* The elements of the threads and lanes [l] reads: the product of the
   extents of the inner ranges it reads. *)
let inner_elements l =
  List.fold_left
    (fun n r -> if is_inner r then n * extent r else n)
    1
    (Nodes.to_list (ranges l))

(* The loads staging copies for a core's operand [x], each with its tile: [x]
   itself where it is a load with a tile; where [x] computes elementwise on
   loads and is not decoded, the loads of global memory through which it reads
   the threads and lanes, where some have a tile and each that has none reads
   fewer of the threads' and lanes' elements than the tile of most elements;
   [None] otherwise. Of a computed operand's loads, those whose tiles step
   through the loops of its tile of most elements are copied, and the others
   stay loads: an attention output's per-head scale, while the core's loop
   runs within a head, is read once a trip of its own loop, and a norm's
   scale and shift beside its normalised rows, whose gate over a padded axis
   no tile of theirs holds, at each trip. The copies move the loads' values,
   which the operand then computes on as it did, so the operand's values do
   not change. A load with no tile that reads as many elements as the
   largest tile, as rows gathered through a copied map of row indices are,
   keeps the operand unstaged. A decoded operand's loads are copied where
   [async] holds, each load copied or held ({!decoded_loads}), and some
   copied. *)
let loads_of ~async ?whole all x =
  match staged_tile ?whole all x with
  | Some t -> Some [ (x, t) ]
  | None when decoded x && async -> (
      let copied =
        List.filter_map
          (function `Copied c -> Some c | `Held _ -> None)
          (Option.value (decoded_loads ?whole all x) ~default:[])
      in
      match copied with [] -> None | _ -> Some copied)
  | None when decoded x -> None
  | None -> (
      match global_loads x with
      | None | Some [] -> None
      | Some loads -> (
          let tiled, untiled =
            List.partition_map
              (fun l ->
                match staged_tile ?whole all l with
                | Some t -> Either.Left (l, t)
                | None -> Either.Right l)
              loads
          in
          match tiled with
          | [] -> None
          | (_, t) :: _ ->
              let largest =
                List.fold_left
                  (fun b (_, t) -> if t.size > b.size then t else b)
                  t tiled
              in
              if
                List.for_all (fun l -> inner_elements l < largest.size) untiled
              then Some (List.filter (fun (_, t) -> same_loops t largest) tiled)
              else None))

(* The loads of a core's [operands] that staging copies, each with its tile,
   or an operand with [None] where it stays unstaged. A computed operand's
   loads are copied only where their loops are those of each operand that is
   a load: a copy runs where the loops of its tile do. *)
let operand_tiles ~async ?whole all operands =
  let plain = List.filter_map (staged_tile ?whole all) operands in
  let on_plain (_, t) = List.for_all (fun p -> same_loops p t) plain in
  List.concat_map
    (fun x ->
      match loads_of ~async ?whole all x with
      | Some [ (l, t) ] when l == x -> [ (x, Some t) ]
      | Some ls when List.for_all on_plain ls ->
          List.map (fun (l, t) -> (l, Some t)) ls
      | _ -> [ (x, None) ])
    operands

(* Copies *)

(* A kernel's workgroup: the number of its threads and the thread's position
   in it, a warp's lanes first. *)
type workgroup = { threads : int; tid : Ops.t }

let workgroup all =
  let rs =
    List.filter (fun u -> Op.equal (op u) Op.Range && is_thread u) all
    |> List.sort (fun a b ->
           Stdlib.compare
             (axis_type a <> Axis_type.Warp, axis_id a)
             (axis_type b <> Axis_type.Warp, axis_id b))
  in
  let tid, threads =
    List.fold_left
      (fun (acc, n) r -> (O.(acc + (r * int n)), n * extent r))
      (int 0, 1) rs
  in
  if rs = [] then None else Some { threads; tid }

(* The elements of [t] a copy moves at once: [copy] bytes, or fewer where the
   innermost dimension is not contiguous or not a multiple. *)
let lanes t =
  let d0 = List.hd t.dims in
  let rec fit v =
    if v > 1 && (d0.stride <> 1 || d0.extent mod v <> 0) then fit (v / 2)
    else v
  in
  fit (max 1 (copy / Dtype.itemsize t.dt))

(* The address in the staged buffer of the element at coordinates [cs],
   innermost first. *)
let address t cs =
  let row =
    fst
      (List.fold_left2
         (fun (acc, m) d c -> (O.(acc + (c * int m)), m * d.extent))
         (int 0, 1) (List.tl t.dims) (List.tl cs))
  in
  layout ~itemsize:(Dtype.itemsize t.dt) ~cols:(List.hd t.dims).extent
    (List.hd cs) row

type stager = {
  copy_to :
    ?ahead:Ops.t ->
    ?within:Ops.t list ->
    ?at:Ops.t ->
    slot:Ops.t ->
    unit ->
    Ops.t;
  read : deps:Ops.t list -> slot:Ops.t -> Ops.t;
}

(* The staging of [t] by the workgroup [wg] into [buf], in slots of
   [t.size] elements. [copy_to ~slot ~at] copies the tile of the trip [at] of
   its loops, counted outermost first ({!digits}), or of the trip that runs
   by default, into slot [slot], under the gate [ahead] if given, at each
   trip of the ranges [within] (default none) besides those it reads;
   [read ~deps ~slot] reads it back, ordered after the copies [deps]. *)
let stager wg ~fresh t buf =
  let v = lanes t in
  let per = wg.threads * v in
  let copies = (t.size + per - 1) / per in
  let rc = if copies > 1 then Some (fresh copies) else None in
  let rv = if v > 1 then Some (fresh v) else None in
  let ranges = List.filter_map Fun.id [ rc; rv ] in
  (* Element [f] of the tile, row-major with its innermost dimension last,
     and its coordinates, innermost first. *)
  let f =
    let i = match rc with Some r -> O.(r * int wg.threads) | None -> int 0 in
    let lane = match rv with Some r -> r | None -> int 0 in
    O.(((i + wg.tid) * int v) + lane)
  in
  let coords =
    List.rev
      (fst
         (List.fold_left
            (fun (cs, below) d ->
              (O.(f // int below % int d.extent) :: cs, below * d.extent))
            ([], 1) t.dims))
  in
  (* [u] at the copy's element: each inner range it reads, the sum of its
     fields' values there. *)
  let at fields u =
    let coord = Array.of_list coords in
    let field (d, k, p, a) =
      O.(coord.(d) // int k % int (extent a) * int p)
    in
    let value fs =
      List.fold_left (fun acc f -> O.(acc + field f)) (int 0) fs
    in
    substitute ~calls:Skip ~pass:Once u
      (List.map (fun (r, fs) -> (r, value fs)) fields)
  in
  (* A dimension of rows moves by its row's offset, the row loaded at the
     copy's element. *)
  let gaddr =
    List.fold_left2
      (fun acc d c ->
        match d.rows with
        | None -> O.(acc + (c * int d.stride))
        | Some (k, r) ->
            (* {!tile_of} holds the fields of each range [r] reads. *)
            let fs = Option.get (fields t.dims r) in
            O.(acc + (at fs r * int k)))
      t.outer t.dims coords
  in
  let valid = Option.map (at t.fields) t.valid in
  let partial = if t.size mod per = 0 then None else Some O.(f < int t.size) in
  let laddr = address t coords in
  let copy_to ?ahead ?(within = []) ?at ~slot () =
    let moved u =
      match at with
      | Some a when t.loops <> [] ->
          substitute ~calls:Skip ~pass:Once u (digits t.loops a)
      | _ -> u
    in
    let conds =
      List.filter_map Fun.id [ Option.map moved valid; ahead; partial ]
    in
    let gidx =
      match conds with
      | [] -> moved gaddr
      | c :: cs -> Shape.valid (moved gaddr) (List.fold_left O.( land ) c cs)
    in
    let lidx = O.(laddr + (slot * int t.size)) in
    let lidx =
      match partial with Some p -> Shape.valid lidx p | None -> lidx
    in
    let dst = index (after buf within) [ lidx ] in
    end_ (store dst (index t.buf [ gidx ])) ranges
  in
  let read ~deps ~slot =
    let rcs =
      List.map
        (fun d ->
          List.fold_left
            (fun acc (k, a) -> O.(acc + (a * int k)))
            (int 0) d.parts)
        t.dims
    in
    let base = address t rcs in
    index (after buf deps) [ O.(base + (slot * int t.size)) ]
  in
  { copy_to; read }

(* The serial ranges the loop [l] of the core [w] runs inside: those the node
   closing [l] runs inside, and those it closes that sort before [l], as a node
   closing several ranges nests them in {!Ops.compare_range_arg}'s order, the
   first outermost. The first chunk's copy runs before [l]'s loop at each of
   their trips: copied once, outside them, slot [0] would hold a tile of [l]'s
   last trips at their second trip. *)
let around all w l =
  let closes u =
    List.memq l (ended_ranges u)
    && Nodes.mem w (backward_slice_with_self ~calls:Skip u)
  in
  match List.find_opt closes all with
  | None -> []
  | Some y ->
      let outer =
        List.filter
          (fun r -> op r = Op.Range && compare_range_arg r l < 0)
          (ended_ranges y)
      in
      List.filter is_serial (Nodes.to_list (ranges y) @ outer)

(* The kernel's LOCAL memory other than staging's: a grouped reduction's
   buffer, or a stage into LOCAL memory. Staging leaves such kernels as they
   are. *)
let has_locals all =
  List.exists
    (fun u ->
      match (op u, arg u) with
      | Op.Stage, Bufferize o -> o.addrspace = Dtype.Local
      | (Op.Alloc | Op.Buffer), _ -> addrspace u = Some Dtype.Local
      | Op.Reduce, _ -> List.exists is_thread (List.tl (src u))
      | _ -> false)
    all

(* The stages of the core [w] of [all] whose staged tiles [staged] step
   through [loops] and copy [sizes] bytes a trip, beside [used] bytes of
   other cores' stages ({!depth}): one where the tiles step through no loop
   or through loops that are no {!nest}, or where a tile's address gathers,
   since a gather is read at the next trip only inside the loop's bound. *)
let stages ~used ren all w staged ~loops sizes =
  match depth ~used ren sizes with
  | 0 -> 0
  | d ->
      let gathered = List.exists (fun t -> t.gathers) staged in
      if loops = [] || gathered || not (nest all w loops) then 1 else d

(* The staging of the core [w] of [all] beside [used] bytes of other cores'
   stages: the tiles of its operands ({!operand_tiles}), its staged tiles,
   the loops they step through, the same for each, and its stages; [None]
   where it stages nothing. *)
let staging (ren : Renderer.t) all ~used w =
  let operands = List.filteri (fun i _ -> i < 2) (src w) in
  let tiles = operand_tiles ~async:ren.async_copy all operands in
  let staged = List.filter_map snd tiles in
  match List.map (fun t -> t.loops) staged with
  | loops :: rest when List.for_all (List.equal ( == ) loops) rest -> (
      match stages ~used ren all w staged ~loops (List.map bytes staged) with
      | 0 -> None
      | d -> Some (tiles, staged, loops, d))
  | _ -> None

(* The held tiles of a core's [operands], whose loads staging copies as
   [tiles] gives them ({!operand_tiles}): each decoded operand's held loads
   ({!decoded_loads}), where it has copied loads and staging copies each. An
   operand whose loads are all held has no stage to carry it. Its loads are
   copied only asynchronously ({!loads_of}), so no target without
   [async_copy] holds a tile. *)
let held_tiles all tiles operands =
  let copied l = List.exists (fun (y, t) -> y == l && Option.is_some t) tiles in
  List.concat_map
    (fun x ->
      let loads =
        if decoded x then Option.value (decoded_loads all x) ~default:[]
        else []
      in
      let held =
        List.filter_map (function `Held h -> Some h | `Copied _ -> None) loads
      and others =
        List.filter_map
          (function `Copied (l, _) -> Some l | `Held _ -> None)
          loads
      in
      if others <> [] && List.for_all copied others then held else [])
    operands

(* Products in registers

   A sum of products a thread computes in its registers, a [Mul] that a
   [Reduce] of [Add] sums, reads its two operands as a core does: a tile over
   the workgroup's threads and the thread's lanes at each trip of the sum's
   loop. It is staged where every read of shared memory is a run along the
   sum, which the copies fill [copy] bytes at a time, and the 8 threads of a
   quarter of a SIMD group, which shared memory serves in one pass of 128
   bytes, read different slots of a line or the same address: each staged
   tile's innermost dimension is the sum's lanes, and each of its other
   dimensions moves first with a thread, so adjacent threads read adjacent
   rows, which {!layout} places on different slots. A tile whose rows a
   thread's lanes take in runs reads rows 8 apart across those threads, all
   on one slot. *)

let products all =
  let strip u = if Op.equal (op u) Op.Cast then nth u 0 else u in
  List.filter_map
    (fun r ->
      match arg r with
      | Reduce { op = Op.Add; _ } when Op.equal (op (strip (nth r 0))) Op.Mul
        ->
          Some (strip (nth r 0), List.tl (src r))
      | _ -> None)
    (List.filter (fun u -> Op.equal (op u) Op.Reduce) all)

(* Whether the tile [t] is read along the sum of the ranges [summed], its
   rows moving first with the threads. *)
let interleaved summed t =
  let lane (_, a) =
    Op.equal (op a) Op.Range
    && (axis_type a = Axis_type.Upcast || is_step a)
    && List.memq a summed
  in
  let threaded d =
    d.rows = None
    &&
    match List.assoc_opt 1 d.parts with
    | Some a -> Op.equal (op a) Op.Range && is_thread a
    | None -> false
  in
  match t.dims with
  | d0 :: rest ->
      d0.stride = 1 && d0.rows = None && List.for_all lane d0.parts
      && List.for_all threaded rest
  | [] -> false

(* The products in registers of [all] whose operands staging copies: none
   but in a workgroup that holds no other shared memory and reduces across
   no threads, on a renderer that stages a tile in two, which only an
   asynchronous copy does ({!depth}). *)
let staged_products (ren : Renderer.t) all =
  let staged (m, summed) =
    let tiles = operand_tiles ~async:true all (src m) in
    (not (List.exists decoded (src m)))
    && tiles <> []
    && List.for_all
         (function _, Some t -> interleaved summed t | _, None -> false)
         tiles
  in
  if
    Option.is_some (workgroup all)
    && (not (has_locals all))
    && depth ren [ copy ] = 2
  then List.map fst (List.filter staged (products all))
  else []

(* The cores of [all] whose operands staging copies: its tensor cores and its
   staged products in registers. *)
let cores ren all =
  List.filter (fun u -> Op.equal (op u) Op.Wmma) all @ staged_products ren all

let staged_bytes (_, staged, _, d) =
  d * List.fold_left (fun n t -> n + bytes t) 0 staged

(* The bytes of shared memory the stages of [all]'s cores before [w] take,
   as {!operands} stages them, in order. *)
let used_before ren all w =
  let rec go used = function
    | [] -> used
    | c :: _ when c == w -> used
    | c :: rest -> (
        match staging ren all ~used c with
        | Some st -> go (used + staged_bytes st) rest
        | None -> go used rest)
  in
  go 0 (cores ren all)

(* Trips of a row's sector

   A copy of [copy] bytes from the L2 reads a sector of [sector], so a row
   whose tile a trip is [copy] bytes, as MXFP4's codes are at 32 values a
   trip, reads each sector at two trips, twice; unstaged, one line of the
   L1 serves 8 trips. A trip of [u] steps of the loop [k] reads [u] times
   the row of a tile whose rows abut from one step to the next: [k] moves
   its address by its innermost dimension's extent. On an RTX 5000 Ada a
   dense MXFP4 product of 5760 by 2880 at 64 values a trip ran 16% faster
   at 64 tokens than unstaged with its weights in the L2, where 32 values
   a trip ran 3% slower, and 22% faster flushed. The trip costs registers:
   gpt-oss-20b's expert cores at 64 tokens take 146 against 128, 3
   workgroups an SM against 4, and on a real routing their down
   projection ran 14% slower than at 32 values a trip, their gate and up
   2% faster; at 512 tokens both ran 5-6% faster. *)

(* The bytes of each row of [t] that a step of [k] reads, where its rows
   abut from one step of [k] to the next. *)
let row_run k t =
  let d0 = List.hd t.dims in
  let by_k (c, a) =
    match a with
    | Some a when Op.equal (op a) Op.Range && axis_id a = axis_id k -> Some c
    | _ -> None
  in
  match List.find_map by_k (terms t.outer) with
  | Some c when d0.stride = 1 && c = d0.extent ->
      Some (d0.extent * Dtype.itemsize t.dt)
  | _ -> None

let steps (ren : Renderer.t) sink ~k =
  let all = toposort ~calls:Skip sink in
  let n = extent k in
  let cores = List.filter (fun u -> Op.equal (op u) Op.Wmma) all in
  let reads t =
    List.exists
      (fun r -> axis_id r = axis_id k)
      (Nodes.to_list (ranges t.outer))
  in
  (* The least steps at which each row a staged core's tiles read along [k]
     covers a sector, if [k] keeps more than one trip to pipeline and the
     stages keep their depth. *)
  let least w =
    match staging ren all ~used:(used_before ren all w) w with
    | None -> 1
    | Some (_, staged, _, d) -> (
        let runs = List.filter_map (row_run k) staged in
        let need =
          List.fold_left (fun u b -> max u ((sector + b - 1) / b)) 1 runs
        in
        let sizes u =
          List.map (fun t -> bytes t * if reads t then u else 1) staged
        in
        let used = used_before ren all w in
        let keeps u =
          u >= need && u < n && n mod u = 0
          && depth ~used ren (sizes u) >= d
        in
        match List.find_opt keeps (List.init n (fun i -> i + 1)) with
        | Some u -> u
        | None -> 1)
  in
  if
    (not ren.async_copy) || n <= 1
    || Option.is_none (workgroup all)
    || has_locals all
  then 1
  else List.fold_left (fun u w -> max u (least w)) 1 cores

(* Steps of a staged product's trip

   A staged product's sum is unrolled by a copy's run, 4 values of float32,
   and each of its loop's trips copies and waits at a barrier. Its trip takes
   [s] steps of that loop, the deepest that keeps a run of at most 32 values,
   leaves two trips, keeps the blocks of the sum's running sums whole
   ({!Chains.block}) and whose two stages fit shared memory: the loop of [n]
   trips becomes [n / s] trips of an inner loop of [s] steps, both read in
   the order of memory, so each output adds the same terms in the same order.
   Unrolled whole instead, a trip's reads of shared memory, 80 runs of 16
   bytes a thread at 20 values, all issued before its sums, spilled 0.9 to
   1.7 KB of registers a thread on sm_89. On an RTX 5000 Ada, hand-written
   kernels of the staged tile ran a product of 16384 rows over 440 values by
   1792 in 1.45 ms at 4 values a trip and 0.96 ms at 20 (design/smem-gemm). *)

let trip_steps (ren : Renderer.t) ~bytes ~lanes n =
  let keeps s =
    n mod s = 0
    && n / s >= 2
    && 2 * s * bytes <= ren.shared_max
    && n / s mod Chains.block ~lanes:(s * lanes) (n / s) = 0
  in
  let most = chunk / lanes in
  List.find_opt keeps (List.init most (fun i -> most - i))

(* [sink] with the loop of each staged product in registers
   ({!staged_products}) read in trips of several steps ({!trip_steps}), the
   steps a loop inside the trips that the tiles take whole. *)
let deepen (ren : Renderer.t) sink =
  let all = toposort ~calls:Skip sink in
  let plan m =
    let tiles = List.filter_map snd (operand_tiles ~async:true all (src m)) in
    match List.map (fun t -> t.loops) tiles with
    | (_ :: _ as ls) :: rest when List.for_all (List.equal ( == ) ls) rest -> (
        let x = List.nth ls (List.length ls - 1) in
        let lanes = List.map (fun t -> (List.hd t.dims).extent) tiles in
        let bytes = List.fold_left (fun n t -> n + bytes t) 0 tiles in
        match (constant (nth x 0), lanes) with
        | Some n, l :: ls when List.for_all (( = ) l) ls -> (
            match trip_steps ren ~bytes ~lanes:l n with
            | Some s when s > 1 -> Some (x, n, s)
            | _ -> None)
        | _ -> None)
    | _ -> None
  in
  let splits =
    List.sort_uniq
      (fun (a, _, _) (b, _, _) -> Ops.compare a b)
      (List.filter_map plan (staged_products ren all))
  in
  List.fold_left
    (fun sink (x, n, s) ->
      let o =
        replace x
          ~src:[ Shape.const_like (nth x 0) (`Int (Bigint.of_int (n / s))) ]
      in
      let b =
        range ~axis_type:(axis_type x) ~dtype:(dtype x) (Int s)
          (axis_id x @ [ step ])
      in
      substitute
        ~extra_pm:(Pattern_matcher.with_ctx Simplify.pm_flatten_range)
        ~calls:Skip ~pass:Fixed_point sink
        [ (x, O.((o * int s) + b)) ])
    sink splits

let operands (ren : Renderer.t) sink =
  let sink = deepen ren sink in
  let all = toposort ~calls:Skip sink in
  match workgroup all with
  | None -> sink
  | Some _ when has_locals all || depth ren [ copy ] = 0 -> sink
  | Some wg ->
      let next_id =
        ref
          (List.fold_left
             (fun n u ->
               if Op.equal (op u) Op.Range then
                 List.fold_left max n (axis_id u)
               else n)
             0 all)
      in
      let fresh n =
        incr next_id;
        range ~axis_type:Axis_type.Upcast (Int n) [ !next_id ]
      in
      let next_slot =
        ref
          (List.fold_left
             (fun n u ->
               match (op u, arg u) with
               | (Op.Buffer | Op.Alloc), Param p -> max n (p.slot + 1)
               | _ -> n)
             0 all)
      in
      (* Every core's stages first, then the held tiles in the room they
         leave, so that a held tile never takes a stage's room. *)
      let cores = cores ren all in
      let used = ref 0 in
      let stagings =
        List.filter_map
          (fun w ->
            match staging ren all ~used:!used w with
            | None -> None
            | Some st ->
                used := !used + staged_bytes st;
                Some (w, st))
          cores
      in
      let stage (w, (tiles, _, loops, d)) =
        let operands = List.filteri (fun i _ -> i < 2) (src w) in
        let plans =
          List.map
            (fun (x, t) ->
              match t with
              | None -> (x, None)
              | Some t ->
                  let buf =
                    Shape.alloc ~slot:!next_slot ~addrspace:Dtype.Local
                      [ Int (t.size * d) ] t.dt
                  in
                  incr next_slot;
                  (x, Some (stager wg ~fresh t buf)))
            tiles
        in
        (* Each held tile that fits, copied once, before every loop. *)
        let held =
          List.filter_map
            (fun (l, t) ->
              if !used + bytes t > ren.shared_max then None
              else begin
                used := !used + bytes t;
                let buf =
                  Shape.alloc ~slot:!next_slot ~addrspace:Dtype.Local
                    [ Int t.size ] t.dt
                in
                incr next_slot;
                let st = stager wg ~fresh t buf in
                Some (l, st, st.copy_to ~slot:(int 0) ())
              end)
            (held_tiles all tiles operands)
        in
        (* The first chunk's copies, before the reads; with two stages, the
           next chunk's, each trip, into the slot the next trip reads: of a
           nest, the next trip in its order. *)
        let first, ahead, rslot =
          match loops with
          | outermost :: _ when d = 2 ->
              let t, n = trip loops in
              let next = O.(t + int 1) in
              let within = around all w outermost in
              let firsts =
                List.filter_map
                  (fun (_, p) ->
                    Option.map
                      (fun st ->
                        st.copy_to ~within ~slot:(int 0) ~at:(int 0) ())
                      p)
                  plans
              and aheads =
                List.filter_map
                  (fun (_, p) ->
                    Option.map
                      (fun st ->
                        st.copy_to ~ahead:O.(next < n) ~slot:O.(next % int 2)
                          ~at:next ())
                      p)
                  plans
              in
              (firsts, aheads, O.(t % int 2))
          | _ ->
              ( List.filter_map
                  (fun (_, p) ->
                    Option.map (fun st -> st.copy_to ~slot:(int 0) ()) p)
                  plans,
                [],
                int 0 )
        in
        (* The reads wait for the first copies and the held ones together,
           at one barrier: a held copy issued before the loop completes
           with the first chunk's. The next chunk's copies run before
           them, which no read waits for: the slot's offset is ordered
           after them. *)
        let waits = first @ List.map (fun (_, _, c) -> c) held in
        let slot = if List.is_empty ahead then rslot else after rslot ahead in
        (* An operand reads its copies back: itself where it is a load,
           or the loads it computes on. *)
        let reads =
          List.filter_map
            (fun (x, p) ->
              Option.map (fun st -> (x, st.read ~deps:waits ~slot)) p)
            plans
          @ List.map
              (fun (l, st, _) -> (l, st.read ~deps:waits ~slot:(int 0)))
              held
        in
        let srcs =
          List.map
            (fun x ->
              match List.assq_opt x reads with
              | Some read -> read
              | None -> substitute ~calls:Skip ~pass:Once x reads)
            operands
        in
        let acc = List.filteri (fun i _ -> i >= 2) (src w) in
        (w, replace w ~src:(srcs @ acc))
      in
      match List.map stage stagings with
      | [] -> sink
      | subs -> substitute ~calls:Skip ~pass:Once sink subs

let admits (ren : Renderer.t) sink =
  let all = toposort ~calls:Skip sink in
  let core w =
    let tiles =
      List.map snd
        (operand_tiles ~async:ren.async_copy all
           (List.filteri (fun i _ -> i < 2) (src w)))
    in
    match tiles with
    | Some t :: rest ->
        List.for_all
          (function Some t' -> same_loops t t' | None -> false)
          rest
    | _ -> false
  in
  Option.is_some (workgroup all)
  && (not (has_locals all))
  && depth ren [ copy ] > 0
  && List.for_all (fun u -> (not (Op.equal (op u) Op.Wmma)) || core u) all


(* A staged core's tile. On an RTX 5000 Ada, bfloat16 products of 512 by 512 by
   2880 and of 64 rows by 5760 by 2880 ran 1.6 to 1.9 times slower without the
   halving of [tile]; at two stages the knee lies between 5760 and 11520
   threads. *)
let warp_tile = 64

(* A staged trip lasts about one copy's latency while its busiest compute
   unit copies few bytes, and those bytes over the unit's rate of fill when
   it copies many: [max (L, B / F)] for [B] bytes a trip (Volkov,
   "Understanding Latency Hiding on GPUs", UCB/EECS-2016-143, Sec. 3.5, the
   latency-bound and throughput-bound modes). A kernel then takes at least
   its busiest unit's bytes over the whole sum over [F], so of the plans
   whose sums keep their order the one whose busiest unit copies least a
   value of [K] fills the units best; and its trip is deepened until [B]
   reaches [in_flight], [L F] with a margin, past which a deeper trip only
   takes shared memory. On an RTX 5000 Ada a key projection of 64 tokens by
   512 over 2880, its operands read from memory, ran 8 workgroups of 64 by 64
   on 8 of its 100 SMs in 39.7 us, and 15.7 spread and deepened. *)
let tile (ren : Renderer.t) ast (w : wmma) ~core_threads ~workgroups
    ~sizes:(sn0, sm, sk) ~axes:(rn, rm, rk) ~in_run =
  let n, m, kd = w.dims in
  (* N's extent is the product's, which a model's widths fix (440 columns are
     55 tiles of 8): where no plan's upcast divides it and the core pads
     ({!Setting.tc_opt} [2]), N pads to a multiple of the whole warp tile's,
     its padded columns stored nowhere, and every plan below divides it. On
     an RTX 5000 Ada bfloat16 products of 65536 rows by 512 by 440 and by
     136, whose core divides N but whose tiles no plan divides, took 0.35
     and 0.20 ms staged on 448 and 192 columns against 0.67 and 0.71 on the
     unstaged core. *)
  let sn =
    let divides hn = sn0 mod max 1 (warp_tile / n / hn) = 0 in
    let un = warp_tile / n in
    if sn0 > 0 && un > 1 && Setting.value Setting.tc_opt >= 2
       && not (List.exists divides [ 1; 2 ])
    then (sn0 + un - 1) / un * un
    else sn0
  in
  let workgroups = Option.map (fun g -> g * sn / max 1 sn0) workgroups in
  let unroll =
    if chunk mod kd = 0 && sk mod (chunk / kd) = 0 then chunk / kd else 1
  in
  (* The most tiles of rows up to [u] that a run holds, one tile less at a
     time: a run of three tiles, a routed block of 48 rows, fits one warp,
     which loads the run's value of the other operand once. Halving would
     give it three warps of one tile, each loading it. *)
  let rec within u = if u <= 1 || in_run u then u else within (u - 1) in
  (* The upcasts of M and N, and their local splits, of a warp tile halved
     [hm] and [hn] times. *)
  let plan (hm, hn) =
    let um = within (warp_tile / m / hm) and un = warp_tile / n / hn in
    if um < 1 || un < 1 || sm = 0 || sn = 0 || sm mod um <> 0 || sn mod un <> 0
    then None
    else
      let lm = if sm / um mod 2 = 0 && in_run (um * 2) then 2 else 1
      and ln = if sn / un mod 2 = 0 then 2 else 1 in
      Some (um, un, lm, ln)
  in
  let planned h = Option.map (fun p -> (h, p)) (plan h) in
  let plans = List.filter_map planned [ (1, 1); (2, 1); (2, 2) ] in
  (* The halvings past [h], [M] then [N] in turn, down to the core's own
     tile. *)
  let rec halved (hm, hn) =
    let can_m = warp_tile / m / hm >= 2 and can_n = warp_tile / n / hn >= 2 in
    let next =
      if can_m && (hm <= hn || not can_n) then Some (hm * 2, hn)
      else if can_n then Some (hm, hn * 2)
      else None
    in
    match next with
    | None -> []
    | Some h -> Option.to_list (planned h) @ halved h
  in
  let all = toposort ~calls:Skip ast in
  let reads r t =
    List.exists
      (fun x -> axis_id x = axis_id r)
      (Nodes.to_list (ranges t.outer))
  in
  let threads (um, un, lm, ln) =
    match workgroups with
    | Some g -> g / (um * un * lm * ln) * w.threads * lm * ln
    | None -> max_int
  in
  (* The units' share of the busiest, in workgroups of the plan [p]. *)
  let busiest (um, un, lm, ln) =
    let g = max 1 (Option.value workgroups ~default:1 / (um * un * lm * ln)) in
    (g + ren.units - 1) / ren.units
  in
  (* The plan and the depth of the core [c], the largest plan [top]. *)
  let choose c top =
    let operands = List.filteri (fun i _ -> i < 2) (src c) in
    let used = used_before ren all c in
    (* The tiles staging copies at [u] of the core's steps a trip, the core's
       [K] none of their loops where a trip takes it whole, as a per-head
       scale is copied once a trip takes a head whole. *)
    let tiles whole =
      List.filter_map snd
        (operand_tiles ~async:ren.async_copy ?whole all operands)
    in
    let within = lazy (tiles None) and whole = lazy (tiles (Some rk)) in
    let staged u =
      Lazy.force (if sk = 0 || sk / u > 1 then within else whole)
    in
    (* The bytes a workgroup of [p] copies of each tile a trip of [u] steps:
       the core leaves a tile of its rows for one step, which [p]'s splits
       widen where it reads [M] or [N], and [u] where it reads [K]. *)
    let copied (um, un, lm, ln) u =
      List.map
        (fun t ->
          bytes t
          * (if reads rm t then um * lm else 1)
          * (if reads rn t then un * ln else 1)
          * if reads rk t then u else 1)
        (staged u)
    in
    let stages_at p u =
      let ts = staged u in
      match List.map (fun t -> t.loops) ts with
      | loops :: rest when List.for_all (List.equal ( == ) loops) rest ->
          stages ~used ren all c ts ~loops (copied p u)
      | _ -> 0
    in
    match stages_at top unroll with
    | 0 -> None
    | s0 ->
        let fits (_, p) = threads p >= core_threads / s0 in
        (* A core of two stages comes before a stream of several passes
           whatever its threads, which no unit count decides: the choice
           moves a sum's order. *)
        let two_stages = s0 = 2 && Option.is_some workgroups in
        let h0, p0 =
          match List.find_opt fits plans with
          | Some p -> p
          | None -> List.hd (List.rev plans)
        in
        (* The plan and the trip follow the units and their bytes in flight
           where the target states them, with copies overlapping cores, as
           they were measured, and known workgroups. *)
        let measured =
          ren.in_flight > 0 && ren.units > 0 && s0 = 2
          && Option.is_some workgroups
        in
        let carried p u = busiest p * List.fold_left ( + ) 0 (copied p u) in
        (* Smaller tiles than [p0]'s block their sums as [p0]'s do where its
           accumulators are blocked, so that no sum's order moves. *)
        let blocked (um, un, _, _) =
          um * m * un * n / w.threads * Dtype.itemsize (dtype c) <= Chains.held
        in
        let least b (_, p) =
          if carried p unroll < carried b unroll then p else b
        in
        let best =
          if measured && blocked p0 then List.fold_left least p0 (halved h0)
          else p0
        in
        let s = stages_at best unroll in
        (* The depths that keep the stages and the blocks of the sum's
           running sums, so that its order does not move; the least whose
           trip carries [in_flight], else the deepest. The blocks are
           {!Chains.block}'s where the core's [K] is its sum's innermost
           range: then the trips it takes whole, or the block it is split
           in, are the same terms at every depth, and the ranges outside see
           the same terms a trip of [K]. Every sum over [K] must have it
           innermost: one beside the core then reads fewer lanes a trip, so
           it leaves [K] more room and the core's blocks govern. *)
        let innermost =
          let over r =
            Op.equal (op r) Op.Reduce
            && List.exists (fun x -> axis_id x = axis_id rk) (List.tl (src r))
          in
          let first r =
            let rs = Chains.innermost (List.tl (src r)) in
            match List.find_opt (fun x -> extent x > 1) rs with
            | Some x -> axis_id x = axis_id rk
            | None -> false
          in
          match List.filter over all with
          | [] -> false
          | sums -> List.for_all first sums
        in
        let block u = Chains.block ~lanes:(kd * u) (sk / u) * u in
        let keeps u =
          u mod unroll = 0 && sk mod u = 0
          && kd * u <= Chains.chain
          && block u = block unroll
          && stages_at best u >= s
        in
        let depths = List.filter keeps (List.init sk (fun i -> i + 1)) in
        let flies u = carried best u >= ren.in_flight in
        let u =
          if not (measured && innermost) then unroll
          else
            match List.find_opt flies depths with
            | Some u -> u
            | None -> List.fold_left max unroll depths
        in
        let um, un, lm, ln = best in
        let when_ c opt = if c then [ opt ] else [] in
        let split axis amount target =
          Opt.Split { axis; amount; target; top = false }
        in
        Some
          ( when_ (sn > sn0) (Opt.Padto { axis = 0; amount = warp_tile / n })
            @ when_ (um > 1) (split 1 um Opt.Upcast)
            @ (if un > 4 then
                 [ split 0 4 Opt.Upcast; split 0 (un / 4) Opt.Upcast ]
               else when_ (un > 1) (split 0 un Opt.Upcast))
            @ when_ (lm = 2) (split 1 2 Opt.Local)
            @ when_ (ln = 2) (split 0 2 Opt.Local)
            @ when_ (u > 1) (split 2 u Opt.Upcast),
            two_stages )
  in
  let core =
    List.find_opt (fun u -> match arg u with Wmma w' -> w' = w | _ -> false) all
  in
  (* The staged tile is taken only where staging stages the core's operands,
     which it decides of the kernel as the core leaves it. *)
  match (core, plans) with
  | Some c, (_, top) :: _ when admits ren ast -> choose c top
  | _ -> None

let stages_product (ren : Renderer.t) sink =
  let all = toposort ~calls:Skip sink in
  let staged p =
    Option.is_some (staging ren all ~used:(used_before ren all p) p)
  in
  match staged_products ren all with
  | [] -> false
  | ps -> List.for_all staged ps

(* A staged product's tile. A workgroup of 128 threads, 16 along M by 8 along
   N, each holding 8 by 8 outputs, is the blocked contraction's in registers,
   two workgroups an SM. Its threads take the rows first, so that the reads
   of shared memory meet no conflict ({!products}), where CUTLASS's SIMT
   products of operands laid along K store their tiles transposed, from
   loads of one element. On an RTX 5000 Ada TESSERA's float32 products of
   16384 rows by weights stored [outputs; inputs] ran so at 23.9-30.1 TFLOP/s,
   against 15.4-18.5 split across threads ({!Heuristic}), and cuBLAS's SIMT
   kernels' 23-31.

   The sum is unrolled by a copy's run: 16 bytes, or where they do not
   divide it, 8, whose copies move half as much, taken only for a sum of at
   least [narrow_copies] times 16 bytes. A run of 4 bytes, one float32
   value, holds no lanes for the tile's innermost dimension
   ({!interleaved}). An attention's
   output projection, 4 heads of 110 float32 values, whose rows lie 440
   bytes apart, runs at 8 bytes; on an RTX 5000 Ada a float32 product of
   16384 rows over 110 by 440 took 94 us so against 147 split across
   threads, but a bfloat16 one over 36, 72 bytes, 18.4 us against 14.3-15.2,
   and one of 163840 rows over 10 float32 values 578-581 us against 584. *)
type product_tile = { threads : int * int; outputs : int; run : int }

let narrow_copies = 16

let product_tile (ren : Renderer.t) ~itemsize ~k ~short =
  let runs =
    List.filter_map
      (fun b -> if b mod itemsize = 0 then Some (b / itemsize) else None)
      [ copy; copy / 2 ]
  in
  let takes run =
    k mod run = 0
    &&
    if run * itemsize = copy then k / run >= 2
    else k * itemsize >= narrow_copies * copy
  in
  match List.find_opt takes runs with
  | Some run when depth ren [ copy ] = 2 ->
      Some { threads = (16, 8); outputs = (if short then 4 else 8); run }
  | _ -> None

(* Decoded panels on the host *)

(* The panel lies on the stack of the thread that runs the kernel: a rig pool's
   worker has 8 MiB, the calling thread its own, which a thread a program starts
   takes at the C library's default, 512 KiB on macOS. An eighth of that leaves
   the rest to the caller. *)
let panel_bytes = 64 * 1024

let by_axis rs =
  List.sort (fun a b -> List.compare Int.compare (axis_id a) (axis_id b)) rs

(* The decoded operands of [all]'s sums of products, each with the ranges its
   sum closes. *)
let decoded_operands all =
  let strip u = if Op.equal (op u) Op.Cast then nth u 0 else u in
  List.concat_map
    (fun r ->
      match arg r with
      | Reduce { op = Op.Add; _ } when Op.equal (op (strip (nth r 0))) Op.Mul ->
          List.filter_map
            (fun o -> if decoded o then Some (o, List.tl (src r)) else None)
            (src (strip (nth r 0)))
      | _ -> [])
    (List.filter (fun u -> Op.equal (op u) Op.Reduce) all)

(* [sink] with [o]'s values within one trip of the output [loops] it reads
   ([inner], its sum's ranges and its lanes) stored into a register buffer
   before those trips' other loops, and read back from it; the loops [o] reads
   renumbered to run outside the others. *)
let panel all sink ~loops ~lanes o inner =
  (* The panel's loops, then its lanes, each a stride of the extents after it: a
     trip's lanes lie together. *)
  let inner =
    List.stable_sort
      (fun a b -> Bool.compare (lanes a) (lanes b))
      (by_axis inner)
  in
  let address rs =
    List.fold_left (fun acc r -> O.((acc * int (extent r)) + r)) (int 0) rs
  in
  let next_id =
    ref
      (List.fold_left
         (fun n u ->
           if Op.equal (op u) Op.Range then List.fold_left max n (axis_id u)
           else n)
         0 all)
  in
  let fresh r =
    incr next_id;
    let axis_type = if lanes r then Axis_type.Upcast else Axis_type.Loop in
    range ~axis_type (Int (extent r)) [ !next_id ]
  in
  let copies = List.map fresh inner in
  let slot =
    List.fold_left
      (fun n u ->
        match (op u, arg u) with
        | (Op.Buffer | Op.Alloc), Param p -> max n (p.slot + 1)
        | _ -> n)
      0 all
  in
  let size = List.fold_left (fun n r -> n * extent r) 1 inner in
  let buf = Shape.alloc ~slot ~addrspace:Dtype.Reg [ Int size ] (dtype o) in
  let value = substitute ~calls:Skip ~pass:Once o (List.combine inner copies) in
  let stored = end_ (store (index buf [ address copies ]) value) copies in
  let read = index (after buf [ stored ]) [ address inner ] in
  (* Loops nest in the order of their axes: renumbered, the loops [o] reads take
     the first of the output loops' axes. *)
  let reads r = Nodes.mem r (ranges o) in
  let order =
    List.filter reads loops @ List.filter (fun r -> not (reads r)) loops
  in
  let renamed =
    List.map2
      (fun r at ->
        ( r,
          replace r
            ~arg:(Range { axis_type = axis_type r; axis_id = axis_id at }) ))
      order loops
  in
  let sink = substitute ~calls:Skip ~pass:Once sink [ (o, read) ] in
  substitute ~calls:Skip ~pass:Once sink renamed

let decodes (ren : Renderer.t) sink =
  let all = toposort ~calls:Skip sink in
  let summed =
    List.concat_map
      (fun u -> if Op.equal (op u) Op.Reduce then List.tl (src u) else [])
      all
  in
  (* The output loops: the weak ranges no sum closes. *)
  let loops =
    by_axis
      (List.filter
         (fun u ->
           Op.equal (op u) Op.Range
           && axis_type u = Axis_type.Weak
           && not (List.memq u summed))
         all)
  in
  let known r = Option.is_some (constant (nth r 0)) in
  let lanes r = axis_type r = Axis_type.Upcast in
  let inner o =
    List.filter (fun r -> not (List.memq r loops)) (Nodes.to_list (ranges o))
  in
  let stages (o, closed) =
    let inner = inner o in
    let size = List.fold_left (fun n r -> n * extent r) 1 inner in
    List.exists (fun r -> (not (Nodes.mem r (ranges o))) && extent r > 1) loops
    && inner <> []
    && List.for_all (fun r -> known r && (lanes r || List.memq r closed)) inner
    && size * Dtype.itemsize (dtype o) <= panel_bytes
  in
  match decoded_operands all with
  | [ ((o, _) as d) ]
    when Setting.value Setting.stage
         && ren.target.device = "CPU" && List.for_all known loops && stages d ->
      panel all sink ~loops ~lanes o (inner o)
  | _ -> sink
