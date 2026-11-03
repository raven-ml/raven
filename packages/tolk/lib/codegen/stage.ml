(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ops

(* Two facts of every target staged so far. Shared memory serves [line] bytes at
   once, 32 banks of 4 bytes (CUDA C++ Programming Guide §2.3.4.2; one of
   RDNA4's two sets of 32 banks, ISA §12.1), and the widest copy, PTX's
   [cp.async] and an [ldmatrix] row, moves [copy] bytes. Each becomes a
   renderer's field when a target measures another value. *)
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
   an ldmatrix reads at one chunk, and 8 consecutive chunks a copy writes, fall
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
   whose sum, each times its coefficient, is the coordinate along it. *)
type dim = { stride : int; extent : int; parts : (int * Ops.t) list }

(* The tile an index [idx] reads over the threads and lanes of a workgroup:
   the part of [idx] that the threads and lanes do not move (the global and
   loop ranges, a gather) and the dimensions, innermost first. [None] where a
   term mixes both, or the inner terms overlap. *)
let split idx =
  let classify (c, a) =
    match a with
    | None -> `Outer (c, a)
    | Some t ->
        let rs = Nodes.to_list (ranges t) in
        if not (reads_inner t) then `Outer (c, a)
        else if
          List.for_all is_inner rs && c > 0
          && Dtype.Value.to_int (vmin t) = 0
        then if extent t > 1 then `Inner (c, t) else `Unit
        else `Mixed
  in
  let classes = List.map classify (terms idx) in
  if List.mem `Mixed classes then None
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
              let d = { stride = c; extent = e; parts = [ (1, t) ] } in
              group (d :: dims) rest)
    in
    match group [] inner with
    | None | Some [] -> None
    | Some dims -> Some (outer, dims)

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

(* ldmatrix *)

let ldmatrix_rows ~trans ~nbits t =
  let lane = if trans then O.(t // int 2 % int 4) else O.(t % int 8 * int 4) in
  let h = if trans then O.(t % int 2) else int 0 in
  let bit k =
    let place = 8 lsl k in
    O.(t // int place % int 2)
  in
  (lane, h :: List.init (nbits - 1) bit)

(* The ldmatrix form, [Some trans], that delivers each lane of the warp [w]
   the elements its read [base] reads, of element bits [es], least significant
   first, at every value of [base]'s other ranges, each row it supplies on a
   16-byte boundary; [None] if neither form does. PTX ISA §9.7.16.5.15: lanes
   [8j] to [8j + 7] supply the rows of matrix [j]; without [.trans] lane [t]
   receives elements [2 (t mod 4)] and the next of row [t / 4] of each matrix,
   with [.trans] those of column [t / 4] in rows [2 (t mod 4)] and the next. *)
let ldmatrix base w es =
  let nbits = List.length es and n = 1 lsl List.length es in
  let rs = Nodes.to_list (ranges base) in
  let others = List.filter (fun r -> r != w && not (List.memq r es)) rs in
  let env = Tbl.create 8 in
  let value u = Bigint.to_int (integer (eval env u)) in
  let rec each = function
    | [] -> fun f -> f ()
    | r :: rest ->
        fun f ->
          List.for_all
            (fun i ->
              Tbl.replace env r i;
              each rest f)
            (List.init (extent r) Fun.id)
  in
  let fits trans =
    let lane, bits = ldmatrix_rows ~trans ~nbits w in
    let row_at =
      substitute ~calls:Skip ~pass:Once base
        ((w, lane) :: List.combine es bits)
    in
    let row s = Tbl.replace env w s; value row_at in
    let read t e =
      Tbl.replace env w t;
      List.iteri (fun k r -> Tbl.replace env r ((e lsr k) land 1)) es;
      value base
    in
    let at t e =
      let j = e / 2 and h = e mod 2 in
      if trans then row ((8 * j) + (2 * (t mod 4)) + h) + (t / 4)
      else row ((8 * j) + (t / 4)) + (2 * (t mod 4)) + h
    in
    each others (fun () ->
        List.for_all
          (fun t ->
            row t mod 8 = 0
            && List.for_all (fun e -> read t e = at t e) (List.init n Fun.id))
          (List.init 32 Fun.id))
  in
  if extent w <> 32 || not (List.mem (n / 2) [ 1; 2; 4 ]) then None
  else
    List.find_opt
      (fun trans -> try fits trans with Exit -> false)
      [ false; true ]

let is_ldmatrix x =
  Op.equal (op x) Op.Load
  &&
  match arg x with
  | String s -> String.starts_with ~prefix:"ldmatrix" s
  | _ -> false

(* The address lane [w] supplies to the load [x] ldmatrix delivers, its index
   the [Stack] of the lane's elements: each element's index with the lane
   made the lane [ldmatrix_rows] names, selected by the element bits
   [ldmatrix_rows] names, least significant first. An [After] keeps its
   ordering sources: the slot's offset is ordered after the next stage's
   copies, which keep their own lanes. The terms every element shares, as the
   slot's offset, are added outside the selection. *)
let lower_ldmatrix x =
  match src x with
  | [ ix ] when is_ldmatrix x && Op.equal (op (nth ix 1)) Op.Stack ->
      let lanes = src (nth ix 1) in
      let trans = arg x = String "ldmatrix.trans" in
      let is_lane r =
        Op.equal (op r) Op.Range
        && axis_type r = Axis_type.Local
        && List.hd (List.rev (axis_id r)) = 0
      in
      let w = List.find is_lane (toposort ~calls:Skip (nth ix 1)) in
      let rec log2 n = if n <= 1 then 0 else 1 + log2 (n / 2) in
      let nbits = log2 (List.length lanes) in
      let lane, bits = ldmatrix_rows ~trans ~nbits w in
      let holes =
        List.mapi
          (fun i a ->
            let name = Printf.sprintf "ldm%d" i in
            (a, variable ~dtype:(dtype a) name (vmin a) (vmax a)))
          (List.filter
             (fun u -> Op.equal (op u) Op.After)
             (toposort ~calls:Skip (nth ix 1)))
      in
      let sub l subs = substitute ~calls:Skip ~pass:Once l subs in
      let at_lane l =
        let back = List.map (fun (a, h) -> (h, a)) holes in
        sub (sub (sub l holes) [ (w, lane) ]) back
      in
      let lanes = List.map at_lane lanes in
      let rec pick ls = function
        | [] -> List.hd ls
        | b :: bs ->
            let half p = List.filteri (fun i _ -> i mod 2 = p) ls in
            if constant b = Some 0 then pick (half 0) bs
            else
              let c = O.(b <> int 0) in
              where c (pick (half 1) bs) (pick (half 0) bs)
      in
      let same (c, a) (c', a') = c = c' && Option.equal ( == ) a a' in
      let shared t =
        List.for_all (fun l -> List.exists (same t) (terms l)) lanes
      in
      let sum ts =
        List.fold_left
          (fun acc (c, a) ->
            O.(acc + (Option.value a ~default:(int 1) * int c)))
          (int 0) ts
      in
      let common = List.filter shared (terms (List.hd lanes)) in
      let rest l =
        sum (List.filter (fun t -> not (List.exists (same t) common)) (terms l))
      in
      let addr = O.(sum common + pick (List.map rest lanes) bits) in
      let n = int (List.length lanes) in
      let window = Ops.v Op.Shrink ~src:[ nth ix 0; addr; n ] in
      Some (replace x ~src:[ window ])
  | _ -> None

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

(* The loop of [outer]: its one weak or loop range, if it has one. *)
let loop_of outer =
  match
    List.filter
      (fun r -> List.mem (axis_type r) Axis_type.[ Weak; Loop ])
      (Nodes.to_list (ranges outer))
  with
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
      match split (get_idx idx) with
      | Some (outer, dims) when dense dims -> (
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
                    List.exists
                      (fun u -> Op.equal (op u) Op.Index)
                      (toposort ~calls:Skip outer);
                  size = List.fold_left (fun n d -> n * d.extent) 1 dims;
                }
          | None -> None)
      | _ -> None)
  | _ -> None

(* Copies *)

(* A kernel's workgroup: its thread ranges, a warp's lanes first, the number
   of its threads, and the thread's position in it. *)
type workgroup = { threads : int; tid : Ops.t; warp : Ops.t option }

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
  let warp = List.find_opt (fun r -> axis_type r = Axis_type.Warp) rs in
  if rs = [] then None else Some { threads; tid; warp }

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
  copy_to : ?ahead:Ops.t -> slot:Ops.t -> at:Ops.t -> unit -> Ops.t;
  read :
    lanes:(Ops.t * Ops.t list) option -> deps:Ops.t list -> slot:Ops.t -> Ops.t;
}

(* The staging of [t] by the workgroup [wg] into [buf], in slots of
   [t.size] elements. [copy_to ~slot ~at] copies the tile of the loop's trip
   [at] into slot [slot], under the gate [ahead] if given; [read ~deps
   ~slot] reads it back, ordered after the copies [deps]. *)
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
  let gaddr =
    List.fold_left2
      (fun acc d c -> O.(acc + (c * int d.stride)))
      t.outer t.dims coords
  in
  (* The gate at the copy's element: each inner range it reads, the sum of
     its fields' values there. *)
  let valid =
    let coord = Array.of_list coords in
    let field (d, k, p, a) =
      O.(coord.(d) // int k % int (extent a) * int p)
    in
    let value fs =
      List.fold_left (fun acc f -> O.(acc + field f)) (int 0) fs
    in
    Option.map
      (fun g ->
        substitute ~calls:Skip ~pass:Once g
          (List.map (fun (r, fs) -> (r, value fs)) t.fields))
      t.valid
  in
  let partial = if t.size mod per = 0 then None else Some O.(f < int t.size) in
  let laddr = address t coords in
  let copy_to ?ahead ~slot ~at () =
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
    end_ (store (index buf [ lidx ]) (index t.buf [ gidx ])) ranges
  in
  (* With [lanes], the warp's lane and the fragment's element ranges, a read
     ldmatrix delivers is marked as one. *)
  let read ~lanes ~deps ~slot =
    let rcs =
      List.map
        (fun d ->
          List.fold_left
            (fun acc (k, a) -> O.(acc + (a * int k)))
            (int 0) d.parts)
        t.dims
    in
    let base = address t rcs in
    let x = index (after buf deps) [ O.(base + (slot * int t.size)) ] in
    let form =
      match lanes with
      | Some (w, es) when Dtype.itemsize t.dt = 2 && es <> [] ->
          ldmatrix base w es
      | _ -> None
    in
    match form with
    | None -> x
    | Some trans ->
        let mark = if trans then "ldmatrix.trans" else "ldmatrix" in
        Ops.v Op.Load ~src:[ x ] ~arg:(String mark)
  in
  { copy_to; read }

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
        let tiles = List.map (fun x -> (x, tile_of x)) operands in
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
                let firsts =
                  List.filter_map
                    (fun (_, p) ->
                      Option.map
                        (fun st -> st.copy_to ~slot:(int 0) ~at:(int 0) ())
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
          (* Each operand's element ranges: the core's upcast axes of two
             that it folds into its fragment. *)
          let elements i =
            match arg w with
            | Wmma { upcast_axes = Some (a0, a1, _); _ } ->
                let axes = if i = 0 then a0 else a1 in
                let range_of ((t, id), size) =
                  if size <> 2 then None
                  else
                    List.find_opt
                      (fun r ->
                        Op.equal (op r) Op.Range
                        && Axis_type.equal (axis_type r) t
                        && axis_id r = id)
                      all
                in
                List.rev (List.filter_map range_of axes)
            | _ -> []
          in
          (* ldmatrix reads only an operand staged in two stages. In one
             stage the next chunk's loads wait on the barrier after this
             chunk's cores, and ptxas overlaps them only by moving cores past
             that barrier, which it does for narrow reads and not for
             ldmatrix: on an RTX 5000 Ada, gpt-oss's gathered expert
             products ran 1.05 to 1.12 times slower with it. *)
          let lanes i =
            match (ren.load_matrix, wg.warp) with
            | true, Some w when d = 2 -> Some (w, elements i)
            | _ -> None
          in
          let srcs =
            List.mapi
              (fun i (x, p) ->
                match p with
                | Some st -> st.read ~lanes:(lanes i) ~deps:first ~slot
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
    let tiles = List.filteri (fun i _ -> i < 2) (src w) |> List.map tile_of in
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

