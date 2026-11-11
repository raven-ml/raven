(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ops

(* Two facts of every target staged so far. Shared memory serves [line] bytes at
   once, 32 banks of 4 bytes (CUDA C++ Programming Guide §2.3.4.2; one of
   RDNA4's two sets of 32 banks, ISA §12.1), and the widest copy, PTX's
   [cp.async], moves [copy] bytes. Each becomes a renderer's field when a
   target measures another value. *)
let line = 128
let copy = 16

let extent u = Dtype.Value.to_int (vmax u) + 1
let is_inner r = List.mem (axis_type r) Axis_type.[ Warp; Local; Upcast ]
let is_thread r = List.mem (axis_type r) Axis_type.[ Warp; Local ]

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
   [None] where a term mixes both otherwise, or the inner terms overlap. *)
let rec split idx =
  let classify (c, a) =
    match a with
    | None -> `Outer (c, a)
    | Some t -> (
        let rs = Nodes.to_list (ranges t) in
        if not (reads_inner t) then `Outer (c, a)
        else if
          List.for_all is_inner rs && c > 0
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
   range's bits from [place]), the loop it steps through, whether its outer
   part loads (a gather), and its size in elements. *)
type tile = {
  buf : Ops.t;
  dt : Dtype.t;
  outer : Ops.t;
  dims : dim list;
  valid : Ops.t option;
  fields : (Ops.t * (int * int * int * Ops.t) list) list;
  loop : Ops.t option;
  gathers : bool;
  size : int;
}

let bytes t = t.size * Dtype.itemsize t.dt

let is_serial r = List.mem (axis_type r) Axis_type.[ Weak; Loop ]

(* The loop of [outer]: its one weak or loop range, if it has one. *)
let loop_of outer =
  match List.filter is_serial (Nodes.to_list (ranges outer)) with
  | [ l ] -> Some l
  | _ -> None

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
                  loop = loop_of outer;
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
   any warp, so a core reading gathered rows keeps the unstaged splits: a
   routed product of bfloat16 weights ran 3.6% slower on the staged tile. *)
let staged_tile all x =
  let warps =
    List.filter
      (fun u -> Op.equal (op u) Op.Range && axis_type u = Axis_type.Local)
      all
  in
  let shared = List.exists (fun r -> not (Nodes.mem r (ranges x))) warps in
  let rows t = List.exists (fun d -> Option.is_some d.rows) t.dims in
  match tile_of x with Some t when rows t && not shared -> None | t -> t

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
    slot:Ops.t ->
    at:Ops.t ->
    unit ->
    Ops.t;
  read : deps:Ops.t list -> slot:Ops.t -> Ops.t;
}

(* The staging of [t] by the workgroup [wg] into [buf], in slots of
   [t.size] elements. [copy_to ~slot ~at] copies the tile of the loop's trip
   [at] into slot [slot], under the gate [ahead] if given, at each trip of the
   ranges [within] (default none) besides those it reads; [read ~deps ~slot]
   reads it back, ordered after the copies [deps]. *)
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
  let copy_to ?ahead ?(within = []) ~slot ~at () =
    let moved u =
      match t.loop with
      | Some l when at != l -> substitute ~calls:Skip ~pass:Once u [ (l, at) ]
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

let operands (ren : Renderer.t) sink =
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
      let used = ref 0 in
      let stage w =
        let operands = List.filteri (fun i _ -> i < 2) (src w) in
        let tiles = List.map (fun x -> (x, staged_tile all x)) operands in
        let staged = List.filter_map snd tiles in
        let d = depth ~used:!used ren (List.map bytes staged) in
        (* The tiles step through one loop, or none: a copy runs where its
           tile's loop does. *)
        let loop =
          match List.map (fun t -> t.loop) staged with
          | l :: rest when List.for_all (Option.equal ( == ) l) rest -> Ok l
          | _ -> Error ()
        in
        match loop with
        | Error () -> None
        | Ok _ when List.is_empty staged || d = 0 -> None
        | Ok loop -> begin
          (* A gather in a tile's address is read at the next trip only
             inside the loop's bound: such a tile copies in one stage. *)
          let gathered = List.exists (fun t -> t.gathers) staged in
          let d = if Option.is_none loop || gathered then 1 else d in
          used :=
            !used + (d * List.fold_left (fun n t -> n + bytes t) 0 staged);
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
          (* The first chunk's copies, before the reads; with two stages, the
             next chunk's, each trip, into the slot the next trip reads. *)
          let first, ahead, rslot =
            match loop with
            | Some l when d = 2 ->
                let n = nth l 0 and next = O.(l + int 1) in
                let within = around all w l in
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
                (firsts, aheads, O.(l % int 2))
            | _ ->
                let at = match loop with Some l -> l | None -> int 0 in
                ( List.filter_map
                    (fun (_, p) ->
                      Option.map (fun st -> st.copy_to ~slot:(int 0) ~at ()) p)
                    plans,
                  [],
                  int 0 )
          in
          (* The reads wait for the first copies; the next chunk's copies run
             before them, which no read waits for: the slot's offset is
             ordered after them. *)
          let slot = if List.is_empty ahead then rslot else after rslot ahead in
          let srcs =
            List.map
              (fun (x, p) ->
                match p with
                | Some st -> st.read ~deps:first ~slot
                | None -> x)
              plans
          in
          let acc = List.filteri (fun i _ -> i >= 2) (src w) in
          Some (w, replace w ~src:(srcs @ acc))
        end
      in
      let cores = List.filter (fun u -> Op.equal (op u) Op.Wmma) all in
      match List.filter_map stage cores with
      | [] -> sink
      | subs -> substitute ~calls:Skip ~pass:Once sink subs

let admits (ren : Renderer.t) sink =
  let all = toposort ~calls:Skip sink in
  let core w =
    let tiles =
      List.filteri (fun i _ -> i < 2) (src w) |> List.map (staged_tile all)
    in
    match tiles with
    | Some t :: rest ->
        List.for_all
          (function
            | Some t' -> Option.equal ( == ) t.loop t'.loop | None -> false)
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
let chunk = 32

let tile (ren : Renderer.t) ast (w : wmma) ~core_threads ~workgroups
    ~sizes:(sn, sm, sk) ~in_run =
  let n, m, kd = w.dims and itemsize = Dtype.itemsize w.dtype_in in
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
  let plans = List.filter_map plan [ (1, 1); (2, 1); (2, 2) ] in
  let bytes (um, un, lm, ln) =
    let depth = kd * unroll * itemsize in
    [ lm * um * m * depth; ln * un * n * depth ]
  in
  (* The staged tile is taken only where staging stages the core's operands,
     which it decides of the kernel as the core leaves it. *)
  let stages =
    match plans with
    | p :: _ when admits ren ast -> depth ren (bytes p)
    | _ -> 0
  in
  if stages = 0 then None
  else
    let threads (um, un, lm, ln) =
      match workgroups with
      | Some g -> g / (um * un * lm * ln) * w.threads * lm * ln
      | None -> max_int
    in
    let fits p = threads p >= core_threads / stages in
    (* Whether the core beats a stream of several passes was measured only
       with copies overlapping its cores, on two stages, and known counts. *)
    let filled =
      stages = 2 && Option.is_some workgroups && List.exists fits plans
    in
    let um, un, lm, ln =
      match List.find_opt fits plans with
      | Some p -> p
      | None -> List.hd (List.rev plans)
    in
    let when_ c split = if c then [ split ] else [] in
    Some
      ( when_ (um > 1) (1, um, Opt.Upcast)
        @ (if un > 4 then [ (0, 4, Opt.Upcast); (0, un / 4, Opt.Upcast) ]
           else when_ (un > 1) (0, un, Opt.Upcast))
        @ when_ (lm = 2) (1, 2, Opt.Local)
        @ when_ (ln = 2) (0, 2, Opt.Local)
        @ when_ (unroll > 1) (2, unroll, Opt.Upcast),
        filled )

(* Decoded panels on the host *)

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
