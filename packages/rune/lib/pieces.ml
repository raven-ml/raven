(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program
module A = Nx_array
module L = Nx_array.Layout
module M = Nx_array.Move
module D = Nx.Dtype

exception Declined of string

exception
  Part of { check : int; position : int; failure : unit -> exn }

let decline fmt = Printf.ksprintf (fun s -> raise (Declined s)) fmt
let shape_of (f : _ Nx.Prim.form) = Iarray.to_array (L.shape f.layout)
let extents w = Array.map (fun (r : M.range) -> r.count) w

let device_of p =
  let s = Nx.Placement.devices p in
  List.find_opt
    (fun r -> Nx.Placement.equal p (Nx.Placement.device s r))
    (Nx.Devices.to_list s)

(* Whether a program needs splitting *)

(* A placement and the device it holds a value on, if it is one. *)
type seen = Seen : 'd Nx.Placement.t * Rig.t option -> seen

let needed (p : Program.t) forms =
  let several = ref false and devices = ref [] in
  (* Most values of a program share a placement: the last one's device is
     kept rather than looked up again. *)
  let last = ref None in
  let device_of (type d) (q : d Nx.Placement.t) =
    match !last with
    | Some (Seen (q', r)) when Nx.Placement.equal q q' -> r
    | Some _ | None ->
        let r = device_of q in
        last := Some (Seen (q, r));
        r
  in
  let note placement =
    match device_of placement with
    | None -> several := true
    | Some r ->
        if not (List.exists (Rig.equal r) !devices) then
          devices := r :: !devices
  in
  let is_var x =
    match Nx.Prim.payload p.owner x with Some (Var _) -> true | _ -> false
  in
  (* A value over several devices needs splitting. Of the values on one device,
     the arguments, the results it computes and the placements say where the
     program computes: a creation elsewhere computes there too, and a constant
     elsewhere is placed, or copied, there. *)
  let operand (type v s d) (x : (v, s, d) Nx.t) =
    if Option.is_none (device_of (Nx.placement x)) then several := true
  in
  let any (Nx.Prim.Any x) = operand x in
  Array.iter (fun (F f) -> note f.placement) forms;
  Array.iter
    (fun (V y) -> if is_var y then note (Nx.placement y) else operand y)
    p.outputs;
  (* A loop's step computes where its operations and placements say. *)
  let rec eqns (p : Program.t) =
    Array.iter
      (function
        | Op { op; _ } -> (
            let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
            List.iter any xs;
            match op with
            | Place (q, _) -> note q
            | Map { at = Some q; _ } ->
                if Option.is_none (device_of q) then several := true
            | _ -> ())
        | Do { c; _ } ->
            List.iter (fun (V x) -> operand x) (Program.operands c);
            List.iter eqns (Program.carried c))
      p.eqns
  in
  eqns p;
  !several || List.length !devices > 1

(* The engine

   A value over several devices is a stand-in of the splitting's own
   interpretation, holding its pieces; a value on one device is the tracer's
   variable, or a constant. Each step of a partition is an operation the tracer
   records, on one device, or a placement from one device onto another, a
   copy. *)

type ('v, 's, 'd) Nx.Prim.payload +=
  | Cut : ('v, 's, 'd) Nx.t iarray -> ('v, 's, 'd) Nx.Prim.payload
        (** The pieces, one per device of the placement, in order. *)

let rig_of p = Option.get (device_of p)

(* An empty value of [dt] and [shape] on the one device of [at]. *)
let empty (type v s e) (at : e Nx.Placement.t) (dt : (v, s) D.t) shape :
    (v, s, e) Nx.t =
  Nx.Repr.unsafe_of_array (Nx.Placement.devices at)
    (A.create (rig_of at) dt (Iarray.of_array shape))

let full x (w : M.range array) =
  let s = Nx.shape x in
  Array.length w = Array.length s
  && Array.for_all2
       (fun (r : M.range) n -> r.start = 0 && r.count = n && r.step = 1)
       w s

let bytes x = D.bytes (Nx.dtype x) (Array.fold_left ( * ) 1 (Nx.shape x))

(* The pieces of [x], a value of the splitting [ri] or a constant. *)
let pieces_of (type v s d) ri (x : (v, s, d) Nx.t) : (v, s, d) Nx.t iarray =
  match Nx.Prim.payload ri x with
  | Some (Cut ps) -> ps
  | _ -> (
      match device_of (Nx.placement x) with
      | Some _ -> Iarray.of_list [ x ]
      | None ->
          let s = Nx.Placement.devices (Nx.placement x) in
          Iarray.map (Nx.Repr.of_array s) (Nx.Repr.shards x))

(* Loops over windows

   A device's loop over a window of a loop's shape reads its coordinates from
   the window's start ({!Nx.Prim.offset}). A start baked into its program would
   make each device's program its own: along an axis where windows start
   apart, the loop reads its coordinates there as an operand instead, the
   window of the whole loop's coordinates, which a compiled program reads as a
   start each device's call binds ({!Compile}), so that every device's loop is
   one program. nx's partials pass a tile's start so too, as a loop's
   coordinate. *)

module P = Nx_kernel.Prog

let loop_shape (type r) (op : r Nx.Prim.t) =
  match op with
  | Map { layout; _ } | Reduce { layout; _ } | Scan { layout; _ } ->
      Iarray.to_array (L.shape layout)
  | _ -> [||]

(* [prog] with each [Coord c] for which [at.(c)] is [Some k] read as its
   operand [k], an [int64] after [prog]'s own. *)
let from_operands (prog : P.t) (at : int option array) =
  let more = Array.to_list (Array.map (fun _ -> D.Any D.Int64) at) in
  let more = List.filteri (fun a _ -> Option.is_some at.(a)) more in
  let ins = Iarray.append prog.ins (Iarray.of_list more) in
  let coord : P.node -> P.E.any option = function
    | Coord c when c < Array.length at && Option.is_some at.(c) ->
        Some (P.E.any (P.E.input (Option.get at.(c)) D.Int64))
    | _ -> None
  in
  P.subst ~ins coord prog

(* The set of a loop's operands, or of the placement of a creation. *)
let set_of (type d) (loads : d Nx.Prim.load iarray)
    (at : d Nx.Placement.t option) : d Nx.Devices.t option =
  match at with
  | Some q -> Some (Nx.Placement.devices q)
  | None when Iarray.length loads = 0 -> None
  | None -> (
      match Iarray.get loads 0 with
      | Plain x -> Some (Nx.Placement.devices (Nx.placement x))
      | Padded { x; _ } -> Some (Nx.Placement.devices (Nx.placement x)))

(* [op], a loop over the window of its loop's shape at [origin] on [here],
   reading its coordinates along each axis where windows start apart, where it
   reads them, as the window there of [arange] of the whole axis on [here],
   broadcast to its shape. Every other window starts at [0]. Attention, which
   is lowered whole, reads each coordinate offset by its window's start. *)
let windowed (type r) here (op : r Nx.Prim.t) (origin : Nx.Prim.origin) :
    r Nx.Prim.t =
  let { Nx.Prim.start; extents } = origin and shape = loop_shape op in
  let n = Array.length start in
  let apart a = start.(a) > 0 || shape.(a) < extents.(a) in
  let read (type d) (prog : P.t) (loads : d Nx.Prim.load iarray) set =
    let coords a =
      Iarray.exists (function P.Coord c -> c = a | _ -> false) prog.nodes
    in
    let next = ref (Iarray.length prog.ins) in
    let at =
      Array.init n (fun a ->
          if apart a && coords a then begin
            incr next;
            Some (!next - 1)
          end
          else None)
    in
    let operand a =
      let on = Nx.Placement.device set (rig_of here) in
      let all = Nx.arange_at on Nx.int64 0 extents.(a) 1 in
      let w = Nx.slice [ R (start.(a), start.(a) + shape.(a)) ] all in
      let dims = Array.mapi (fun i k -> if i = a then k else 1) shape in
      Nx.Prim.Plain (Nx.broadcast_to shape (Nx.reshape dims w))
    in
    if Array.for_all Option.is_none at then (prog, loads)
    else
      let more =
        List.filter_map
          (fun a -> Option.map (fun _ -> operand a) at.(a))
          (List.init n Fun.id)
      in
      (from_operands prog at, Iarray.append loads (Iarray.of_list more))
  in
  (* A loop has loads or names its placement: one with neither would lose its
     offset. *)
  let no_set () = invalid_arg "Rune.jit: a loop over a window lies nowhere" in
  match op with
  | _ when n = 0 -> op
  (* Attention's pair program reads its coordinates in the whole: each
     offset by its window's start. *)
  | Attention _ -> Nx.Prim.offset op origin
  | Map m -> (
      match set_of m.loads m.at with
      | None -> no_set ()
      | Some set ->
          let prog, loads = read m.prog m.loads set in
          let at = if Iarray.length loads = 0 then m.at else None in
          Map { m with prog; loads; at })
  | Reduce m -> (
      match set_of m.loads None with
      | None -> no_set ()
      | Some set ->
          let prog, loads = read m.prog m.loads set in
          Reduce { m with prog; loads })
  | Scan m -> (
      match set_of m.loads None with
      | None -> no_set ()
      | Some set ->
          let prog, loads = read m.prog m.loads set in
          Scan { m with prog; loads })
  | _ -> op

(* Gathered values

   A value an engine gathers ({!engine}) is its boxes, each a piece placed on
   the value's device that holds one window of it. A window of the value is
   gathered again from the boxes within it, which lie on that device: a device
   whose step reads its own window of a value gathered whole on it, as a
   closing's cotangent placed back at a cut weight, reads its own pieces, and
   the others' copies are never made. A split program's gathers are kept by
   their variable. *)

type gathered = (int, value * (M.range array * value) list) Hashtbl.t

let var_of x =
  match Nx.Prim.owner x with
  | None -> None
  | Some o -> (
      match Nx.Prim.payload o x with Some (Var n) -> Some n | _ -> None)

let boxes_of (gathered : gathered) x =
  match var_of x with
  | None -> None
  | Some n -> (
      match Hashtbl.find_opt gathered n with
      | Some (y, boxes) when Program.same (V x) y -> Some boxes
      | Some _ | None -> None)

(* The common window of [a] and [b], if it holds an element. *)
let meet (a : M.range array) (b : M.range array) =
  let m =
    Array.map2
      (fun (a : M.range) (b : M.range) ->
        let start = Int.max a.start b.start in
        let stop = Int.min (a.start + a.count) (b.start + b.count) in
        { M.start; count = stop - start; step = 1 })
      a b
  in
  if Array.exists (fun (r : M.range) -> r.count <= 0) m then None else Some m

(* [inner], a window of the whole, as a window of [outer]'s. *)
let relative (outer : M.range array) (inner : M.range array) =
  Array.map2
    (fun (o : M.range) (i : M.range) -> { i with start = i.start - o.start })
    outer inner

let engine ri ~(gathered : gathered) ~by moves : Nx.Prim.engine =
  let run ~by ?into:_ ~origin here _ op =
    Nx.Prim.eval ~by (windowed here op origin)
  in
  (* The value of [dt] and shape [s] that holds, at each window of [boxes],
     the box's piece, at the pieces' placement, kept as gathered. *)
  let assembled (type v s e) (dt : (v, s) D.t) s
      (boxes : (M.range array * (v, s, e) Nx.t) list) : (v, s, e) Nx.t =
    let pieces = List.map (fun (w, p) -> (Iarray.of_array w, p)) boxes in
    let y =
      Nx.Prim.eval ~by
        (Assemble { dtype = dt; shape = s; fill = D.zero dt; pieces })
    in
    Option.iter
      (fun n ->
        Hashtbl.replace gathered n
          (V y, List.map (fun (w, p) -> (w, V p)) boxes))
      (var_of y);
    y
  in
  let rec view : type v s d. M.t -> (v, s, d) Nx.t -> (v, s, d) Nx.t =
   fun mv x ->
    let regathered w =
      Iarray.for_all (fun (r : M.range) -> r.step = 1 && r.count > 0) w
      && not (full x (Iarray.to_array w))
    in
    match (mv, boxes_of gathered x) with
    | M.Slice w, Some boxes when regathered w -> (
        let w = Iarray.to_array w in
        (* Each box within [w]: its window of [w] and its piece's view. *)
        let within (dw, p) =
          Option.map
            (fun m ->
              let p = like x p and b = relative dw m in
              let p =
                if full p b then p else view (Slice (Iarray.of_array b)) p
              in
              (relative w m, p))
            (meet dw w)
        in
        match List.filter_map within boxes with
        | [ (_, p) ] when Nx.shape p = extents w -> p
        | inside -> assembled (Nx.dtype x) (Iarray.of_array (extents w)) inside)
    | _ -> Nx.Prim.eval ~by (Move (mv, x))
  in
  let window x w =
    if full x w then x else view (M.Slice (Iarray.of_array w)) x
  in
  (* [x] placed at [at], a copy where it lies on another device, noted with
     the value it makes. *)
  let put at x =
    let src = rig_of (Nx.placement x) and dst = rig_of at in
    let y = Nx.Prim.eval ~by (Place (at, x)) in
    if not (Rig.equal src dst) then
      moves :=
        ( Some (V y),
          Printf.sprintf "%s moves %d bytes from %s to %s" by (bytes x)
            (Rig.name src) (Rig.name dst) )
        :: !moves;
    y
  in
  let bring at ps =
    let d = rig_of at in
    let here (x, _) = Rig.equal (rig_of (Nx.placement x)) d in
    let x, w =
      match List.find_opt here ps with Some p -> p | None -> List.hd ps
    in
    put at (window x w)
  in
  let gather (type v s d e) (at : e Nx.Placement.t) (dt : (v, s) D.t) s
      (boxes : (M.range array * (v, s, d) Nx.t * M.range array) list) :
      (v, s, e) Nx.t =
    assembled dt s (List.map (fun (w, x, b) -> (w, put at (window x b))) boxes)
  in
  let assemble (type v s d e) (p : e Nx.Placement.t) s (dt : (v, s) D.t)
      (ps : (v, s, d) Nx.t option array) : (v, s, e) Nx.t =
    let set = Nx.Placement.devices p in
    let windows = Nx.Prim.pieces p s in
    let piece j =
      match ps.(j) with
      | Some v -> Option.get (Nx.Prim.coerce dt set v)
      | None ->
          let here, w = windows.(j) in
          empty here dt (extents w)
    in
    if Array.length windows = 1 then piece 0
    else
      let form =
        {
          Nx.Prim.dtype = dt;
          layout = L.contiguous (Iarray.of_array s);
          placement = p;
        }
      in
      Nx.Prim.traced ri form (Cut (Iarray.init (Array.length windows) piece))
  in
  {
    run;
    piece = (fun x j _ -> Iarray.get (pieces_of ri x) j);
    view;
    bring;
    gather;
    assemble;
  }

(* Splitting *)

type output = One of int | Over of form * int array

type t = {
  program : Program.t;
  forms : form array;
  args : int array;
  outputs : output array;
  moves : string list;
}

(* The forms of the pieces of a value of [f]: [f] itself on one device. *)
let piece_forms (F f) =
  match device_of f.placement with
  | Some _ -> [| F f |]
  | None ->
      Array.map
        (fun (here, w) ->
          F
            {
              Nx.Prim.dtype = f.dtype;
              layout = L.contiguous (Iarray.of_array (extents w));
              placement = here;
            })
        (Nx.Prim.pieces f.placement (shape_of f))

(* The placement of a value at [p] whose devices hold whole rows of their
   windows of the rest: each device's piece of a row is a row of its piece. *)
let by_rows p =
  Nx.Placement.with_leading_axis (Nx.Placement.without_leading_axis p)

let same_form (F f) (F g) =
  D.equal f.dtype g.dtype
  && shape_of f = shape_of g
  && Nx.Placement.equal f.placement g.placement

let pieces ri (V x) =
  Array.map (fun v -> V v) (Iarray.to_array (pieces_of ri x))

let all_pieces ri vs = Array.concat (Array.to_list (Array.map (pieces ri) vs))

(* The value of form [f] whose pieces are [ps]. *)
let joined ri (F f) ps =
  match device_of f.placement with
  | Some _ -> ps.(0)
  | None ->
      let set = Nx.Placement.devices f.placement in
      let ps = Iarray.map (Program.cast f.dtype set) (Iarray.of_array ps) in
      V (Nx.Prim.traced ri f (Cut ps))

(* Checks

   A check over several devices is computed where its flags lie: each device's
   window of the flags is a check of its own, whose position counts within
   that window, and each device's copy of a window whole on several is checked
   once, on [home] where it holds one. The data, moved to the flags' placement
   where they lie elsewhere, are read at the same position of each device's
   window: this is the only data a check moves. Only each device's outcome,
   one flag, reaches [home], where the call's check reads it ({!Compile}).
   A failure is a {!Part}: {!Compile} reads the position of each device's
   failing part and raises the first, in C order of the whole value. *)

let checked_parts = Atomic.make 0

(* The data of a check at device [j]'s window. *)
let rec data_pieces : type d e.
    _ -> (d, e) Nx.Prim.data -> int -> (d, e) Nx.Prim.data =
 fun ri data j ->
  match data with
  | [] -> []
  | x :: rest -> Iarray.get (pieces_of ri x) j :: data_pieces ri rest j

let rec placed_at : type d e.
    Nx.Prim.engine -> by:string -> d Nx.Placement.t ->
    (d, e) Nx.Prim.data -> (d, e) Nx.Prim.data =
 fun e ~by at data ->
  match data with
  | [] -> []
  | x :: rest ->
      let x =
        if Nx.Placement.equal (Nx.placement x) at then x
        else Nx.Prim.partition e ~by (Place (at, x))
      in
      x :: placed_at e ~by at rest

let checked (type d e) ri ~home e ~by (ok : (bool, _, d) Nx.t)
    (data : (d, e) Nx.Prim.data) (fail : int array -> e -> exn) =
  let shape = Nx.shape ok and at = Nx.placement ok in
  let check = Atomic.fetch_and_add checked_parts 1 in
  let data = placed_at e ~by at data in
  let windows = Nx.Prim.pieces at shape in
  let devices = Array.map (fun (here, _) -> rig_of here) windows in
  (* Each window's device that checks it: [home] where it holds that window,
     else the first. *)
  let chosen = Hashtbl.create 4 in
  Array.iteri
    (fun j (_, w) ->
      let key = Array.to_list w in
      match Hashtbl.find_opt chosen key with
      | Some k when Rig.equal devices.(k) home -> ()
      | Some _ when not (Rig.equal devices.(j) home) -> ()
      | Some _ | None -> Hashtbl.replace chosen key j)
    windows;
  (* A window's flags at [local], in the whole value's indices and position
     in C order. *)
  let at_position (w : M.range array) local =
    let index = Array.map2 (fun (r : M.range) c -> r.start + c) w local in
    let position = ref 0 in
    Array.iteri (fun a n -> position := (!position * n) + index.(a)) shape;
    (index, !position)
  in
  let part j =
    let _, w = windows.(j) in
    if Array.for_all (fun (r : M.range) -> r.count > 0) w then
      Nx.Prim.eval ~by
        (Check
           {
             ok = Iarray.get (pieces_of ri ok) j;
             data = data_pieces ri data j;
             fail =
               (fun local datum ->
                 let index, position = at_position w local in
                 Part
                   { check; position; failure = (fun () -> fail index datum) });
           })
  in
  List.iter part
    (List.sort compare (Hashtbl.fold (fun _ j l -> j :: l) chosen []))

(* The moves of [noted], each with the value it makes, that [p] keeps: those
   whose value an equation of [p] that reads to its outputs makes. *)
let kept (p : Program.t) noted =
  let live = Array.make p.slots false in
  let all = Array.init (Array.length p.outputs) Fun.id in
  let q = Program.prune p ~outputs:all in
  (* An equation with no results, such as a check, may number past the slots. *)
  let mark base = if base < p.slots then live.(base) <- true in
  Array.iter (function Op { base; _ } | Do { base; _ } -> mark base) q.eqns;
  List.filter_map
    (fun (v, m) ->
      match v with
      | Some (V y) -> (
          match Nx.Prim.payload p.owner y with
          | Some (Var n) when not live.(n) -> None
          | _ -> Some m)
      | None -> Some m)
    noted

(* [cut ~kind ~home ~name p forms] is [p] split, as {!split}, by a tracer of
   [kind], its captures, and its moves, each with the value it makes, in
   order. [home] is the device the split program computes on, where its
   checks and an iterate's stop lie. *)
let rec cut ~kind ~in_loop ~(home : Rig.t) ~name (p : Program.t) forms =
  if kind <> Trace.Jit && Array.length p.adds > 0 then
    decline "a total in a loop over several devices";
  if p.emits then decline "an emission over several devices";
  let leaves = Array.map piece_forms forms in
  let moves = ref [] and outputs = ref [||] in
  let gathered = Hashtbl.create 16 in
  let refuse _ ~by _ =
    invalid_arg (by ^ ": a value over several devices outside its partition")
  in
  Nx.Prim.interpret ~name Values refuse @@ fun ri ->
  let body vs =
    let env = Array.make p.slots None and at = ref 0 in
    Array.iteri
      (fun k (F f) ->
        let n = Array.length leaves.(k) in
        let ps = Array.sub vs !at n in
        at := !at + n;
        env.(p.params.(k)) <- Some (joined ri (F f) ps))
      forms;
    (* [p]'s equations over [env], its slots' values. *)
    let rec run (p : Program.t) env =
      let lookup (type v s d) (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
        match Nx.Prim.payload p.owner x with
        | Some (Var n) -> like x (Option.get env.(n))
        | _ -> x
      in
      let bind = ref 0 in
      let store (type v s d) (y : (v, s, d) Nx.t) =
        env.(!bind) <- Some (V y);
        incr bind
      in
      let operands vs = Array.map (fun (V x) -> V (lookup x)) vs in
      let bound base vs =
        Array.iteri (fun k v -> env.(base + k) <- Some v) vs
      in
      Array.iter
        (function
          | Op { by; op = traced; base } -> (
              let op = Nx.Prim.map lookup traced in
              let e = engine ri ~gathered ~by moves in
              match op with
              | Check { ok; data; fail } ->
                  if in_loop then
                    decline "a check in a loop's step over several devices";
                  checked ri ~home e ~by ok data fail
              | _ ->
                  let r = Nx.Prim.partition e ~by op in
                  bind := base;
                  Nx.Prim.iter_results store op r)
          | Do { c = Detach x; base } -> env.(base) <- Some (V (lookup x))
          | Do { c = (Scan _ | Iterate _ | Remat _ | Switch _) as c; base } ->
              let c = Program.map_operands { f = lookup } c in
              Array.iteri
                (fun k v -> env.(base + k) <- Some v)
                (looped ~in_loop ~home ~name ri moves c)
          (* Attention's derivative is its default's operations, each
             partitioned. *)
          | Do { c = Attention_linear a as c; base } ->
              let operands =
                Array.of_list
                  (Program.operands (Program.map_operands { f = lookup } c))
              in
              let nd = Array.length a.data in
              let q, captures =
                Trace.trace Staged ~name:"Nx.attention"
                  (Array.map Program.form operands) (fun vs ->
                    Program.attention_linear a.spec ~transposed:a.transposed
                      (Array.sub vs 0 nd)
                      (Array.sub vs nd (Array.length vs - nd)))
              in
              if Array.length captures > 0 then
                decline "Nx.attention's derivative over several devices";
              bound base (inlined q operands)
          (* A custom rule, a linear function's application, a call and an
             addition are the operations of their programs, inlined, each
             partitioned, as {!Compile} runs them on one device. *)
          | Do { c = Call { body; args; _ }; base } ->
              bound base (inlined body (operands args))
          | Do { c = Linear { apply = Some a; args; captures; _ }; base } ->
              bound base
                (inlined a (operands (Array.append args captures)))
          | Do { c = Custom c; base } ->
              let primal =
                inlined c.primal (operands (Array.append c.args c.captures))
              in
              bound base (Array.append primal (tangent_sums c.derivative))
          | Do { c = Accumulate { lanes; running; x }; base } ->
              let running = lookup running and x = lookup x in
              (* The program's running value out lies where its running value
                 in does, which an addition at another placement moves. *)
              let here = Nx.placement running in
              if not (Nx.Placement.equal here (Nx.placement x)) then
                decline "an addition at another placement than its total's";
              let q, captures =
                Trace.trace Staged ~name:"Rune.Total.add"
                  [| form (V running); form (V x) |]
                  (fun vs ->
                    let r = like running vs.(0) and x = like x vs.(1) in
                    [| V (added ~lanes r x) |])
              in
              if Array.length captures > 0 then
                decline "a total over several devices";
              bound base (inlined q [| V running; V x |])
          | Do { c; _ } -> decline "%s over several devices" (construct_name c))
        p.eqns
    (* [q]'s equations, each partitioned, over [operands], its parameters'
       values: its outputs. Its totals and emissions stay the enclosing
       program's to thread, so a program that holds one declines. *)
    and inlined (q : Program.t) operands =
      if Array.length q.adds > 0 || q.emits then
        decline "a total or an emission in a construct over several devices";
      let sub = Array.make q.slots None in
      Array.iteri (fun k v -> sub.(q.params.(k)) <- Some v) operands;
      run q sub;
      Array.map
        (fun (V y) ->
          match Nx.Prim.payload q.owner y with
          | Some (Var n) -> Option.get sub.(n)
          | _ -> V y)
        q.outputs
    (* A tangent map's additions' running values out, which no differentiation
       makes ({!Compile}'s own): the identity, at their forms. *)
    and tangent_sums = function
      | Tangent t ->
          let n = Array.length t.outputs and na = Array.length t.adds in
          let sum k =
            let filled _ =
              [| Loops.filled ~zero:false (form t.outputs.(n - na + k)) |]
            in
            let q, _ = Trace.trace Staged ~name:"Rune.custom_jvp" [||] filled in
            (inlined q [||]).(0)
          in
          Array.init na sum
      | Pullback _ -> [||]
    in
    run p env;
    let lookup (type v s d) (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
      match Nx.Prim.payload p.owner x with
      | Some (Var n) -> like x (Option.get env.(n))
      | _ -> x
    in
    let pieces =
      Array.map
        (fun (V y) ->
          let y = lookup y in
          Array.map (fun v -> V v) (Iarray.to_array (pieces_of ri y)))
        p.outputs
    in
    let next = ref 0 in
    outputs :=
      Array.mapi
        (fun j ps ->
          let k = !next in
          next := k + Array.length ps;
          let (F f) = form p.outputs.(j) in
          match device_of f.placement with
          | Some _ -> One k
          | None -> Over (F f, Array.init (Array.length ps) (( + ) k)))
        pieces;
    Array.concat (Array.to_list pieces)
  in
  let forms = Array.concat (Array.to_list leaves) in
  let program, captures = Trace.trace kind ~name forms body in
  let noted = List.rev !moves in
  ( {
      program;
      forms;
      args = Array.map Array.length leaves;
      outputs = !outputs;
      moves = kept program noted;
    },
    captures,
    noted )

(* A loop over values of several devices, [c] over the splitting [ri]'s values:
   the loop of its programs split, over its operands' pieces. A scan reads each
   device's piece of a row from its piece of the rows. It declines rows cut
   along axis 0, which eager computation does not read where they lie, and
   rows cut along an axis of one element, whose rows [Nx.get], which reads
   them eagerly, may route elsewhere than the step's rows lie. *)
and looped ~in_loop ~home ~name ri moves c =
  (* The moves a loop's programs keep, each trip's, noted with the loop's first
     result once it is made. *)
  let inner = ref [] in
  let split ?(in_loop = true) q forms =
    let s, captures, _ = cut ~kind:Staged ~in_loop ~home ~name q forms in
    inner := !inner @ s.moves;
    (s, captures)
  in
  let note ?(each = " each trip") r =
    let tag = if Array.length r > 0 then Some r.(0) else None in
    List.iter (fun m -> moves := (tag, m ^ each) :: !moves) !inner;
    r
  in
  (* The indices of the outputs of [s] that are output [j]'s pieces. *)
  let outs (s : t) j =
    match s.outputs.(j) with One k -> [| k |] | Over (_, ks) -> ks
  in
  (* Each carry's next value is split as the carry is. *)
  let carries (s : t) nc =
    let first = ref 0 in
    for j = 0 to nc - 1 do
      let ks = outs s j in
      let same i k =
        same_form s.forms.(!first + i) (form s.program.outputs.(k))
      in
      if
        Array.length ks <> s.args.(j)
        || not (Array.for_all Fun.id (Array.mapi same ks))
      then
        decline "%s whose carry's next value lies elsewhere"
          (construct_name c);
      first := !first + s.args.(j)
    done
  in
  let results (s : t) r =
    Array.mapi
      (fun j f -> joined ri f (Array.map (fun k -> r.(k)) (outs s j)))
      (Program.forms c)
  in
  let forms vs = Array.map form vs in
  match c with
  | Scan { step; length; trips; reverse; init; xs; captures; seeds = [||] } ->
      (* Rows with no element lie anywhere: each device then reads empty
         pieces of whole rows. *)
      let rows (V x) =
        let at = Nx.placement x and shape = Nx.shape x in
        let q = by_rows at in
        if Nx.Placement.equal q at then V x
        else if Array.fold_left ( * ) 1 shape > 0 then
          decline "a scan over rows cut across devices along them"
        else
          let f =
            {
              Nx.Prim.dtype = Nx.dtype x;
              layout = L.contiguous (Iarray.of_array shape);
              placement = q;
            }
          in
          let piece (here, w) = V (empty here f.dtype (extents w)) in
          joined ri (F f) (Array.map piece (Nx.Prim.pieces q shape))
      in
      let xs = Array.map rows xs in
      let s, more =
        split step
          (Array.concat [ forms init; Array.map Loops.row xs; forms captures ])
      in
      carries s (Array.length init);
      let first v = (pieces ri v).(0) in
      results s
        (note @@ Program.perform
           (Scan
              {
                step = s.program;
                length;
                trips = Option.map first trips;
                reverse;
                init = all_pieces ri init;
                xs = all_pieces ri xs;
                captures = Array.append (all_pieces ri captures) more;
                seeds = [||];
              }))
  | Iterate
      {
        name = entry;
        step;
        until;
        max;
        rows = None;
        stopped;
        init;
        step_captures;
        until_captures;
        seeds = [||];
      } ->
      let carry = forms init in
      let s, step_more =
        split step (Array.append carry (forms step_captures))
      in
      let u, until_more, noted =
        cut ~kind:Staged ~in_loop:true ~home ~name until
          (Array.append carry (forms until_captures))
      in
      carries s (Array.length init);
      (* The stop is one boolean, read on [home] before each trip: where
         several devices hold it, replicas being equal, [home]'s, or else the
         first device's. *)
      let device k =
        let (F f) = form u.program.outputs.(k) in
        if L.numel f.layout = 0 then None else Some (rig_of f.placement)
      in
      let at_home k =
        match device k with Some d -> Rig.equal d home | None -> false
      in
      let until, k =
        match u.outputs.(0) with
        | One k -> (u.program, k)
        | Over (_, ks) ->
            let ks =
              List.filter (fun k -> device k <> None) (Array.to_list ks)
            in
            let k =
              match List.find_opt at_home ks with
              | Some k -> k
              | None -> List.hd ks
            in
            (Program.prune u.program ~outputs:[| k |], k)
      in
      inner := !inner @ kept until noted;
      (match device k with
      | Some d when not (Rig.equal d home) ->
          inner :=
            !inner
            @ [
                Printf.sprintf "%s moves its stop, 1 byte, from %s to %s" entry
                  (Rig.name d) (Rig.name home);
              ]
      | _ -> ());
      let stopped =
        Option.map
          (fun i ->
            if s.args.(i) <> 1 then
              decline "an iterate whose lanes stop apart over several devices";
            Array.fold_left ( + ) 0 (Array.sub s.args 0 i))
          stopped
      in
      results s
        (note @@ Program.perform
           (Iterate
              {
                name = entry;
                step = s.program;
                until;
                max;
                rows = None;
                stopped;
                init = all_pieces ri init;
                step_captures =
                  Array.append (all_pieces ri step_captures) step_more;
                until_captures =
                  Array.append (all_pieces ri until_captures) until_more;
                seeds = [||];
              }))
  | Remat { body; args; seeds = [||]; differentiated } ->
      (* The barrier stays the compiler's: each piece of an operand is stored
         apart there, and read where it lies. *)
      let s, more = split ~in_loop body (forms args) in
      results s
        (note ~each:"" @@ Program.perform
           (Remat
              {
                body = s.program;
                args = Array.append (all_pieces ri args) more;
                seeds = [||];
                differentiated;
              }))
  | Switch { name = entry; index; branches; captures; seeds = [||] } ->
      (* Each branch split over the captures' pieces, their captures as one.
         The index, a 0-d value, is read on [home]: its copy there, or else
         the first device's, which the compiler then moves. *)
      let cuts = Array.map (fun b -> split b (forms captures)) branches in
      let s = fst cuts.(0) in
      (* A switch's results are one storage per piece: a branch whose results
         lie elsewhere than the first's, as a map's untaken lanes do, has none
         to share. *)
      let alike ((b : t), _) =
        let ys = b.program.outputs and zs = s.program.outputs in
        Array.length ys = Array.length zs
        && Array.for_all2 (fun y z -> same_form (form y) (form z)) ys zs
      in
      if not (Array.for_all alike cuts) then
        decline "%s whose branches' results lie apart" (construct_name c);
      let params = Array.fold_left ( + ) 0 s.args in
      let programs, more =
        Program.unite
          (Array.map (fun ((s : t), m) -> (s.program, m)) cuts)
          ~params
      in
      let at_home (V x) = Rig.equal (rig_of (Nx.placement x)) home in
      let copies = pieces ri index in
      let index =
        match List.find_opt at_home (Array.to_list copies) with
        | Some v -> v
        | None ->
            let (V x) = copies.(0) in
            inner :=
              !inner
              @ [
                  Printf.sprintf "%s moves its index, 4 bytes, from %s to %s"
                    entry
                    (Rig.name (rig_of (Nx.placement x)))
                    (Rig.name home);
                ];
            copies.(0)
      in
      results s
        (note ~each:"" @@ Program.perform
           (Switch
              {
                name = entry;
                index;
                branches = programs;
                captures = Array.append (all_pieces ri captures) more;
                seeds = [||];
              }))
  | Iterate { rows = Some _; _ } ->
      decline "an iterate that stacks its steps over several devices"
  | Scan _ | Iterate _ -> decline "a total over several devices"
  | c -> decline "%s over several devices" (construct_name c)

let split ~name ~home p forms =
  let s, _, _ = cut ~kind:Jit ~in_loop:false ~home ~name p forms in
  s
