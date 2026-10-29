(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx_kernel.Prog
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module Prim = Nx.Prim

(* Re-raised with the backtrace of its raise, which a plain [raise] would
   restart at the handler. *)
let reraise e = Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

type ('v, 's, 'd) Prim.payload +=
  | Lane : ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Prim.payload
  | Shared : ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Prim.payload

(* Lane values *)

let batched i x =
  match Prim.payload i x with Some (Lane b) -> Some b | _ -> None

let owned i x = Option.is_some (batched i x)

(* Shared running values

   A total's running value held around the map, as the map answers
   {!Program.Running}: one value every lane adds to, its rows in turn, which no
   operation reads. A running value held inside the map is a lane value or a
   constant, each lane's own. *)

let shared i x =
  match Prim.payload i x with Some (Shared r) -> Some r | _ -> None

let share i r = Prim.traced i (Prim.form r) (Shared r)
let tail a = Iarray.sub a ~pos:1 ~len:(Iarray.length a - 1)

(* One lane of [b]: its layout without axis 0, its placement without its leading
   axis. *)
let lane_form (type v s d) (b : (v, s, d) Nx.t) : (v, s, d) Prim.form =
  let f = Prim.form b in
  let l = f.layout in
  {
    dtype = f.dtype;
    layout =
      L.v ~offset:(L.offset l) ~strides:(tail (L.strides l)) (tail (L.shape l));
    placement = Nx.Placement.without_leading_axis f.placement;
  }

let lane i b = Prim.traced i (lane_form b) (Lane b)
let lead n a = Iarray.append (Iarray.of_list [ n ]) a
let broadcast n x = Nx.broadcast_to (Array.append [| n |] (Nx.shape x)) x

let batch i n x =
  match Prim.payload i x with
  | Some (Lane b) -> b
  | Some (Shared _) ->
      invalid_arg "Rune: a total's running value is read inside a map"
  | _ -> broadcast n x

(* A construct's sums under the map: each shared where its seed is a running
   value around the map, and otherwise every lane's, a lane value or the
   identity. *)
let sharing i count seeds =
  if seeds = [||] then Array.make count false
  else Array.map (fun (Program.V x) -> Option.is_some (shared i x)) seeds

let share_v i (Program.V x) = Program.V (share i x)

let unshare_v i (Program.V x) =
  match shared i x with
  | Some r -> Program.V r
  | None -> invalid_arg "Rune: a map's running value out is no longer shared"

(* A construct's seeds performed outward, and its sums' ends back inside. *)
let seeds_out i n shared seeds =
  Array.mapi
    (fun k (Program.V x as v) ->
      if shared.(k) then unshare_v i v else Program.V (batch i n x))
    seeds

let ends_in i shared ends =
  Array.mapi
    (fun k (Program.V x as v) ->
      if shared.(k) then share_v i v else Program.V (lane i x))
    ends

(* A program's running values in, inside the map [w], and out of it. *)
let totals_in w shared running =
  Array.mapi
    (fun k (Program.V x as v) ->
      if shared.(k) then share_v w v else Program.V (lane w x))
    running

let totals_out w n shared outs =
  Array.mapi
    (fun k (Program.V x as v) ->
      if shared.(k) then unshare_v w v else Program.V (batch w n x))
    outs

(* [Program.threaded]'s lanes: every total but the shared ones has lanes. *)
let laned n shared = (n, Array.map not shared)

(* A program's flags from those of the sums [feeds] says its totals feed. *)
let fed shared feeds = Array.map (fun j -> shared.(j)) feeds

(* A result's value: a batch, or a value every lane shares. *)
type out = Batch of Program.value | Shared of Program.value

(* [op]'s results, of the forms eager execution gives one lane: lane values over
   the batches [compute ()] gives in position order, and the shared values
   themselves. [op] is checked against its rule first, so that a lane's misuse
   raises as it would for one lane, whatever the number of lanes. *)
let results (type r) i ~by (op : r Prim.t) compute : r =
  let outs = ref None in
  Prim.results ~by
    (fun k f ->
      let os =
        match !outs with
        | Some os -> os
        | None ->
            let os = compute () in
            outs := Some os;
            os
      in
      let coerce (Program.V x) =
        match Prim.coerce f.dtype (Nx.Placement.devices f.placement) x with
        | Some x -> x
        | None ->
            invalid_arg "Rune: a result of another dtype or set than its lane's"
      in
      match os.(k) with
      | Batch b -> Prim.traced i f (Lane (coerce b))
      | Shared x -> coerce x)
    op

(* The values of [op] issued outward, in position order. *)
let values (type r) ~by (op : r Prim.t) =
  let r = Prim.eval ~by op and vs = ref [] in
  Prim.iter_results (fun y -> vs := Program.V y :: !vs) op r;
  Array.of_list (List.rev !vs)

let issued ~by op () = Array.map (fun v -> Batch v) (values ~by op)

(* Operations' attributes past the lane axis *)

(* [p] reading the coordinates one axis further: its loop gains axis 0. *)
let shift_coords (p : P.t) =
  if not (Iarray.exists (function P.Coord _ -> true | _ -> false) p.nodes)
  then p
  else
    P.v ~ins:p.ins
      (Iarray.map (function P.Coord a -> P.Coord (a + 1) | n -> n) p.nodes)
      ~outs:p.outs

let shift_window (w : M.window) = { w with axis = w.axis + 1 }

(* A pad that leaves the lane axis as it is. *)
let lead_pad (pad : Nx_kernel.pad) : Nx_kernel.pad =
  {
    lo = lead 0 pad.lo;
    hi = lead 0 pad.hi;
    interior = lead 0 pad.interior;
    windows = Iarray.map shift_window pad.windows;
  }

let load i n : 'd Prim.load -> 'd Prim.load = function
  | Plain x -> Plain (batch i n x)
  | Padded { x; fill; pad } ->
      Padded { x = batch i n x; fill; pad = lead_pad pad }

let loop_layout n layout = L.contiguous (lead n (L.shape layout))
let every n = { M.start = 0; count = n; step = 1 }

(* Loops

   A loop's output that reads no load with lanes, and for a scan no carry with
   lanes, is the same in every lane, as JAX's batching leaves an output of
   unbatched operands unbatched: forward mode under a map of tangents keeps its
   primal shared, so it computes once and a read of it reads. A loop is cut into
   a part over the lanes and a part every lane shares, each its program's
   outputs over the loads they read. Where the shared part reads no load it
   would be a creation with no operand to lie beside, so the loop stays whole
   over the lanes. *)

(* Whether each of [loads] holds lanes. *)
let laned_loads i loads =
  Array.init (Iarray.length loads) (fun j ->
      match Iarray.get loads j with
      | Prim.Plain x -> owned i x
      | Padded { x; _ } -> owned i x)

(* Some of a program's outputs over the loads they read: [loads] the original
   positions of its loads, [index] each original output's position, or [-1]. *)
type part = { prog : P.t; loads : int array; index : int array }

(* A program's outputs that read lanes, and its parts. *)
type plan = { laned : bool array; lanes : part option; shared : part option }

let positions flags =
  Array.of_list
    (List.filter (fun j -> flags.(j)) (List.init (Array.length flags) Fun.id))

let operands = function
  | P.In _ | Coord _ | Const _ -> []
  | Op1 (_, _, a) -> [ a ]
  | Op2 (_, a, b) -> [ a; b ]
  | Op3 (_, a, b, c) -> [ a; b; c ]

(* [p]'s outputs [outs], in order, over the loads they read. *)
let restrict (p : P.t) outs =
  let nodes = p.nodes and nout = Iarray.length p.outs in
  let n = Iarray.length nodes in
  let reach = Array.make n false in
  List.iter (fun k -> reach.(Iarray.get p.outs k) <- true) outs;
  for v = n - 1 downto 0 do
    if reach.(v) then
      List.iter (fun a -> reach.(a) <- true) (operands (Iarray.get nodes v))
  done;
  let read = Array.make (Iarray.length p.ins) false in
  Iarray.iteri
    (fun v node ->
      match node with P.In j when reach.(v) -> read.(j) <- true | _ -> ())
    nodes;
  let loads = positions read in
  let at_load = Array.make (Iarray.length p.ins) (-1) in
  Array.iteri (fun k j -> at_load.(j) <- k) loads;
  let at = Array.make n (-1) and kept = ref [] and next = ref 0 in
  Iarray.iteri
    (fun v node ->
      if reach.(v) then begin
        let node : P.node =
          match node with
          | P.In j -> In at_load.(j)
          | (Coord _ | Const _) as c -> c
          | Op1 (k, dt, a) -> Op1 (k, dt, at.(a))
          | Op2 (k, a, b) -> Op2 (k, at.(a), at.(b))
          | Op3 (k, a, b, c) -> Op3 (k, at.(a), at.(b), at.(c))
        in
        kept := node :: !kept;
        at.(v) <- !next;
        incr next
      end)
    nodes;
  let index = Array.make nout (-1) in
  List.iteri (fun m k -> index.(k) <- m) outs;
  let prog =
    P.v
      ~ins:(Iarray.of_array (Array.map (Iarray.get p.ins) loads))
      (Iarray.of_list (List.rev !kept))
      ~outs:(Iarray.of_list (List.map (fun k -> at.(Iarray.get p.outs k)) outs))
  in
  { prog; loads; index }

let plan (p : P.t) laned_load =
  let nodes = p.nodes in
  let reads = Array.make (Iarray.length nodes) false in
  Iarray.iteri
    (fun v node ->
      reads.(v) <-
        (match node with
        | P.In j -> laned_load.(j)
        | node -> List.exists (fun a -> reads.(a)) (operands node)))
    nodes;
  let laned = Array.map (fun o -> reads.(o)) (Iarray.to_array p.outs) in
  let nout = Array.length laned in
  let part flag =
    match List.filter (fun k -> laned.(k) = flag) (List.init nout Fun.id) with
    | [] -> None
    | outs -> Some (restrict p outs)
  in
  match part false with
  | None | Some { loads = [||]; _ } ->
      (* Outputs that read no load would lie on no operand's devices: they
         compute with the lanes, every lane's alike. *)
      let whole =
        {
          prog = p;
          loads = Array.init (Array.length laned_load) Fun.id;
          index = Array.init nout Fun.id;
        }
      in
      { laned = Array.make nout true; lanes = Some whole; shared = None }
  | Some _ as shared -> { laned; lanes = part true; shared }

type 'd outs = Outs : ('d, 'r) Prim.outs -> 'd outs

let rec outs_of : type d. D.any list -> d outs = function
  | [] -> Outs []
  | D.Any dt :: rest ->
      let (Outs o) = outs_of rest in
      Outs (dt :: o)

let out_dtypes (p : P.t) = List.map (P.dtype p) (Iarray.to_list p.outs)

type 'd red = Red : ('d, 'a) Prim.reduction -> 'd red
type 'd reds = Reds : ('d, 'r) Prim.reductions -> 'd reds

let rec reds_list : type d r. (d, r) Prim.reductions -> d red list = function
  | [] -> []
  | r :: rs -> Red r :: reds_list rs

let rec of_list : type d. d red list -> d reds = function
  | [] -> Reds []
  | Red r :: l ->
      let (Reds rs) = of_list l in
      Reds (r :: rs)

let output : type d a. (d, a) Prim.reduction -> int = function
  | Monoid (_, k, _) | Logsumexp (k, _) | Moments (k, _) | Arg (_, k, _) -> k

let arity : type d a. (d, a) Prim.reduction -> int = function
  | Monoid _ | Logsumexp _ -> 1
  | Moments _ | Arg _ -> 2

let reading : type d a. int -> (d, a) Prim.reduction -> (d, a) Prim.reduction =
 fun k -> function
  | Monoid (m, _, dt) -> Monoid (m, k, dt)
  | Logsumexp (_, dt) -> Logsumexp (k, dt)
  | Moments (_, dt) -> Moments (k, dt)
  | Arg (e, _, dt) -> Arg (e, k, dt)

(* The loads of [part] of [loads], batched for the part over the lanes. *)
let part_loads i n ~laned part loads =
  let l = Iarray.of_array (Array.map (Iarray.get loads) part.loads) in
  if laned then Iarray.map (load i n) l else l

let map_rule (type r) i n ~by (op : r Prim.t) : r =
  match op with
  | Map m ->
      let pl = plan m.prog (laned_loads i m.loads) in
      let issue ~laned part =
        let (Outs o) = outs_of (out_dtypes part.prog) in
        let loads = part_loads i n ~laned part m.loads in
        if laned then
          values ~by
            (Map
               {
                 layout = loop_layout n m.layout;
                 prog = shift_coords part.prog;
                 outs = o;
                 loads;
                 at = None;
               })
        else
          values ~by
            (Map
               {
                 layout = m.layout;
                 prog = part.prog;
                 outs = o;
                 loads;
                 at = None;
               })
      in
      results i ~by op (fun () ->
          let lv = Option.map (issue ~laned:true) pl.lanes in
          let sv = Option.map (issue ~laned:false) pl.shared in
          Array.mapi
            (fun k laned ->
              if laned then
                Batch (Option.get lv).((Option.get pl.lanes).index.(k))
              else Shared (Option.get sv).((Option.get pl.shared).index.(k)))
            pl.laned)
  | _ -> invalid_arg "Rune: a map expected"

let reduce_rule (type r) i n ~by (op : r Prim.t) : r =
  match op with
  | Reduce r ->
      let pl = plan r.prog (laned_loads i r.loads) in
      let reds = reds_list r.reductions in
      let issue ~laned part =
        let mine =
          List.filter (fun (Red x) -> pl.laned.(output x) = laned) reds
        in
        if mine = [] then [||]
        else
          let (Reds rs) =
            of_list
              (List.map
                 (fun (Red x) -> Red (reading part.index.(output x) x))
                 mine)
          in
          let loads = part_loads i n ~laned part r.loads in
          if laned then
            values ~by
              (Reduce
                 {
                   layout = loop_layout n r.layout;
                   axes = Iarray.map succ r.axes;
                   prog = shift_coords part.prog;
                   reductions = rs;
                   loads;
                 })
          else
            values ~by
              (Reduce
                 {
                   layout = r.layout;
                   axes = r.axes;
                   prog = part.prog;
                   reductions = rs;
                   loads;
                 })
      in
      results i ~by op (fun () ->
          let lv = Option.fold ~none:[||] ~some:(issue ~laned:true) pl.lanes in
          let sv =
            Option.fold ~none:[||] ~some:(issue ~laned:false) pl.shared
          in
          let at_l = ref 0 and at_s = ref 0 in
          Array.concat
            (List.map
               (fun (Red x) ->
                 let laned = pl.laned.(output x) in
                 let at = if laned then at_l else at_s in
                 let k = !at in
                 at := k + arity x;
                 Array.init (arity x) (fun j ->
                     if laned then Batch lv.(k + j) else Shared sv.(k + j)))
               reds))
  | _ -> invalid_arg "Rune: a reduction expected"

let rec data : type d e.
    Prim.interpretation -> int -> (d, e) Prim.data -> (d, e) Prim.data =
 fun i n d -> match d with [] -> [] | x :: d -> batch i n x :: data i n d

(* Operations over batches *)

let linalg (type d r) i n (l : (d, r) Prim.linalg) : (d, r) Prim.linalg =
  let b x = batch i n x in
  match l with
  | Cholesky c -> Cholesky { c with a = b c.a }
  | Lu a -> Lu (b a)
  | Qr q -> Qr { q with a = b q.a }
  | Svd_values s -> Svd_values { s with a = b s.a }
  | Svd s -> Svd { s with a = b s.a }
  | Eigh_values e -> Eigh_values { e with a = b e.a }
  | Eigh e -> Eigh { e with a = b e.a }
  | Eig_values e -> Eig_values { e with a = b e.a }
  | Eig e -> Eig { e with a = b e.a }
  | Solve_triangular s -> Solve_triangular { s with a = b s.a; b = b s.b }

(* A contraction maps the lane axis as JAX's dot_general does: a batch pair in
   front where both operands have lanes, and otherwise a free axis of the one
   that has them, moved to the front of the result. A scale has its operand's
   lanes; segments' offsets with lanes make the lane axis a batch pair, as their
   extents lead with the batch's. A contraction whose lanes are its [init]'s
   alone maps [a]'s. *)
let contract (type v s d) i n ~by (op : (v, s, d) Nx.t Prim.t) () =
  match op with
  | Contract c ->
      let b x = batch i n x in
      let scaled = function
        | Some (Prim.Exponent x) -> owned i x
        | Some (Value x) -> owned i x
        | None -> false
      in
      let scale on = function
        | Some (Prim.Exponent x) when on -> Some (Prim.Exponent (b x))
        | Some (Value x) when on -> Some (Prim.Value (b x))
        | s -> s
      in
      let offsets =
        match c.segments with
        | Some (Rows { offsets; _ } | Contracting { offsets; _ }) ->
            owned i offsets
        | None -> false
      in
      let on_a = owned i c.a || scaled c.a_scale in
      let on_b = owned i c.b || scaled c.b_scale in
      let both = (on_a && on_b) || offsets in
      let on_a = both || on_a || not on_b and on_b = both || on_b in
      let da = Bool.to_int on_a and db = Bool.to_int on_b in
      let pairs = Iarray.map (fun (x, y) -> (x + da, y + db)) in
      let nb = Iarray.length c.batch and nc = Iarray.length c.contracting in
      let groups = match c.segments with Some (Contracting _) -> 1 | _ -> 0 in
      (* The lane axis in the result: the groups, the batch pairs, [a]'s free
         axes, then [b]'s. *)
      let p =
        if both then groups
        else if on_a then groups + nb
        else groups + nb + (Nx.ndim c.a - nb - nc)
      in
      let offsets o = if both then b o else o in
      let segments =
        Option.map
          (function
            | Nx_kernel.Rows r ->
                Nx_kernel.Rows
                  {
                    offsets = offsets r.offsets;
                    a_axis = r.a_axis + da;
                    b_axis = r.b_axis + db;
                  }
            | Contracting r ->
                Contracting { r with offsets = offsets r.offsets })
          c.segments
      in
      let to_lane x = if p = 0 then x else Nx.moveaxis 0 p x in
      let y =
        Prim.eval ~by
          (Contract
             {
               c with
               batch =
                 (if both then lead (0, 0) (pairs c.batch) else pairs c.batch);
               contracting = pairs c.contracting;
               a = (if on_a then b c.a else c.a);
               b = (if on_b then b c.b else c.b);
               init = Option.map (fun x -> to_lane (b x)) c.init;
               a_scale = scale on_a c.a_scale;
               b_scale = scale on_b c.b_scale;
               segments;
             })
      in
      [| Batch (Program.V (if p = 0 then y else Nx.moveaxis p 0 y)) |]
  | _ -> invalid_arg "Rune: a contraction expected"

let move (type v s d) i n ~by mv (x : (v, s, d) Nx.t) () =
  let b = batch i n x in
  let eval mv b = Prim.eval ~by (Move (mv, b)) in
  let y =
    match (mv : M.t) with
    | Reshape s -> eval (Reshape (lead n s)) b
    | Broadcast s ->
        (* Ones between the lane axis and [x]'s, so that aligning from the right
           keeps the lane axis in front. *)
        let k = Iarray.length s - Nx.ndim x in
        let ones = Iarray.init k (fun _ -> 1) in
        let s' = Iarray.append ones (Iarray.of_array (Nx.shape x)) in
        let b = if k = 0 then b else eval (Reshape (lead n s')) b in
        eval (Broadcast (lead n s)) b
    | Permute p -> eval (Permute (lead 0 (Iarray.map succ p))) b
    | Slice rs -> eval (Slice (lead (every n) rs)) b
    | Window ws -> eval (Window (Iarray.map shift_window ws)) b
  in
  [| Batch (Program.V y) |]

(* A lane's [Place] onto the placement it has passes its batch through: placing
   the batch at that placement with a leading axis would gather a batch cut
   along it onto every device. *)
let place (type v s d e) i n ~by (q : e Nx.Placement.t) (x : (v, s, d) Nx.t) ()
    =
  let b = batch i n x in
  let through =
    Nx.Placement.(equal (without_leading_axis (Nx.placement b)) q)
  in
  if through then [| Batch (Program.V b) |]
  else
    let q' = Nx.Placement.with_leading_axis q in
    [| Batch (Program.V (Prim.eval ~by (Place (q', b)))) |]

(* Lanes that stop apart

   A loop whose lanes stop apart runs over every lane until the last one stops,
   and a stopped lane is held: it keeps its values, and computes the step at the
   first lane still running, its donor, on the donor's values detached, so that
   no lane computes at a point no lane reached and a held lane's derivative
   reaches no other lane. Inside the step a held lane's check passes, its
   additions are dropped, its lane index is its donor's, and the map's lanes
   raise, the lanes standing at different steps. *)

(* The lanes that run, one boolean per lane, and each lane's donor, an [int32]
   per lane: itself where it runs. *)
type held = { running : Program.value; donors : Program.value }

let as_bool (type v s d) (x : (v, s, d) Nx.t) : (bool, Nx.bool_elt, d) Nx.t =
  match D.equal_witness (Nx.dtype x) Nx.bool with
  | Some Type.Equal -> x
  | None -> invalid_arg "Rune: lane flags that are not booleans"

let as_int32 (type v s d) (x : (v, s, d) Nx.t) : (int32, Nx.int32_elt, d) Nx.t =
  match D.equal_witness (Nx.dtype x) Nx.int32 with
  | Some Type.Equal -> x
  | None -> invalid_arg "Rune: lane indices that are not int32"

(* [vs] at one brand, the first one's, for [b.k]. *)
type at_one = { k : 'd. 'd Prim.any array -> Program.value }

let at_one vs b =
  let (Program.V r) = vs.(0) in
  let at (Program.V x) =
    match
      Prim.coerce (Nx.dtype x) (Nx.Placement.devices (Nx.placement r)) x
    with
    | Some x -> Prim.Any x
    | None -> invalid_arg "Rune: lane values on two sets"
  in
  b.k (Array.map at vs)

(* [b.k c x] with [c], a value per lane, and [x], a batch, at one brand. *)
type beside = {
  k : 'w 'q 'v 's 'd. ('w, 'q, 'd) Nx.t -> ('v, 's, 'd) Nx.t -> Program.value;
}

let beside c x b =
  at_one [| c; x |]
    {
      k =
        (fun vs ->
          match vs with
          | [| Prim.Any c; Prim.Any x |] -> b.k c x
          | _ -> invalid_arg "Rune: lane values");
    }

(* [x] at [y]'s dtype, which it has. *)
let same (type v s w r d) (y : (v, s, d) Nx.t) (x : (w, r, d) Nx.t) :
    (v, s, d) Nx.t =
  match D.equal_witness (Nx.dtype x) (Nx.dtype y) with
  | Some Type.Equal -> x
  | None -> invalid_arg "Rune: lane values of two dtypes"

(* [c], a value per lane, along [x]'s lanes. *)
let along c x =
  let r = Nx.ndim x - 1 in
  Nx.broadcast_to (Nx.shape x)
    (Nx.reshape (Array.append [| Nx.numel c |] (Array.make r 1)) c)

(* [Nx.where c y x], whose transpose gives the operand it does not select [-0.],
   the additive identity that keeps every sign, where [Nx.where]'s gives [+0.].
   A held carry receives a lane's cotangent through the step and through the
   hold, one of them that zero, which would turn a [-0.] into [+0.]. *)
let keep (type v s d) c (y : (v, s, d) Nx.t) (x : (v, s, d) Nx.t) =
  match D.kind (Nx.dtype y) with
  | Float | Complex -> Nx.neg (Nx.where c (Nx.neg y) (Nx.neg x))
  | Signed | Unsigned | Boolean -> Nx.where c y x

(* The batch [x'] where [running] holds, and [x] elsewhere. *)
let hold running x' x =
  at_one [| running; x'; x |]
    {
      k =
        (function
        | [| Prim.Any c; Prim.Any y; Prim.Any x |] ->
            Program.V (keep (along (as_bool c) y) y (same y x))
        | _ -> invalid_arg "Rune: lane values");
    }

(* The batch [x] where [running] holds, and elsewhere the row of its donor,
   detached. With [~kept], the default, a held lane's cotangent is the [-0.] of
   {!keep}, for a carry that receives a held lane's cotangent through its hold
   too; without, it is [+0.], for a capture whose lane receives nothing else. *)
let lend ?(kept = true) { running; donors } v =
  at_one [| running; donors; v |]
    {
      k =
        (function
        | [| Prim.Any c; Prim.Any d; Prim.Any x |] ->
            let rows = Nx.take ~axis:0 (Nx.cast Nx.int64 (as_int32 d)) x in
            let rows = Program.like rows (Program.perform (Detach rows)).(0) in
            let c = along (as_bool c) x in
            Program.V (if kept then keep c x rows else Nx.where c x rows)
        | _ -> invalid_arg "Rune: lane values");
    }

let not_flags (Program.V f) = Program.V (Nx.logical_not (as_bool f))

let or_flags a b =
  beside a b
    { k = (fun a b -> Program.V (Nx.logical_or (as_bool a) (as_bool b))) }

let and_flags a b =
  beside a b
    { k = (fun a b -> Program.V (Nx.logical_and (as_bool a) (as_bool b))) }

(* Each lane's count [t], one more where [running] holds. *)
let count_up running t =
  beside running t
    {
      k =
        (fun r t ->
          Program.V (Nx.add (as_int32 t) (Nx.cast Nx.int32 (as_bool r))));
    }

(* Each lane's donor: itself where it runs, else the first lane that runs. *)
let donors n (Program.V running) =
  let r = as_bool running in
  let own = Nx.arange_at (Nx.placement r) Nx.int32 0 n 1 in
  if n = 0 then Program.V own
  else
    let first = Nx.cast Nx.int32 (Nx.argmax ~axis:0 (Nx.cast Nx.int32 r)) in
    Program.V (Nx.where r own (Nx.broadcast_to [| n |] first))

(* The lane flags [running] at [x]'s brand. *)
let flags_at (type v s d) ~what (Program.V running) (x : (v, s, d) Nx.t) :
    (bool, Nx.bool_elt, d) Nx.t =
  let r = as_bool running in
  match Prim.coerce (Nx.dtype r) (Nx.Placement.devices (Nx.placement x)) r with
  | Some r -> r
  | None ->
      invalid_arg
        (what ^ ": a value on another set than the lanes held apart beside it")

(* A batch of additions whose held lanes' values are the identity. *)
let dropped running b =
  let r = flags_at ~what:"Rune.Total.add" running b in
  Nx.where (along r b) b
    (Program.like b (Loops.filled ~zero:false (Program.form (Program.V b))))

(* The rows of an emission's batch [x] that ran, over its lanes and its first
   [lead] axes: those of the lanes [running] holds, where [mask] holds. *)
let held_rows h (Program.V x) ~lead mask =
  let r = flags_at ~what:"Rune.Probe.emit" h.running x in
  let shape = Array.sub (Nx.shape x) 0 (lead + 1) in
  let r =
    Nx.broadcast_to shape
      (Nx.reshape (Array.append [| Nx.numel r |] (Array.make lead 1)) r)
  in
  match mask with
  | None -> Program.V r
  | Some m -> Program.V (Nx.logical_and r (Program.like r m))

let rule (type r) n held i ~by (op : r Prim.t) : r =
  let b x = batch i n x in
  let loads ls = Iarray.map (load i n) ls in
  let batched (bop : r Prim.t) = results i ~by op (issued ~by bop) in
  match op with
  | Map _ -> map_rule i n ~by op
  | Reduce _ -> reduce_rule i n ~by op
  | Scan s
    when not
           (Option.fold ~none:false
              ~some:(fun (Prim.Any c) -> owned i c)
              s.carry
           || (plan s.prog (laned_loads i s.loads)).laned.(output s.reduction))
    ->
      (* An output and carry every lane shares. *)
      let k = output s.reduction in
      let part = restrict s.prog [ k ] in
      let loads = part_loads i n ~laned:false part s.loads in
      let scan =
        Prim.Scan
          {
            s with
            prog = part.prog;
            reduction = reading part.index.(k) s.reduction;
            loads;
          }
      in
      results i ~by op (fun () ->
          Array.map (fun v -> Shared v) (values ~by scan))
  | Scan s ->
      batched
        (Scan
           {
             s with
             layout = loop_layout n s.layout;
             axis = s.axis + 1;
             prog = shift_coords s.prog;
             loads = loads s.loads;
             carry = Option.map (fun (Prim.Any c) -> Prim.Any (b c)) s.carry;
           })
  | Gather g -> batched (Gather { axis = g.axis + 1; idx = b g.idx; x = b g.x })
  | Scatter s ->
      batched
        (Scatter
           {
             s with
             axis = s.axis + 1;
             idx = b s.idx;
             updates = b s.updates;
             into = b s.into;
           })
  | Sort s -> batched (Sort { s with axis = s.axis + 1; x = b s.x })
  | Fold f ->
      batched (Fold { shape = lead n f.shape; pad = lead_pad f.pad; x = b f.x })
  | Assemble a ->
      let piece (r, x) = (lead (every n) r, b x) in
      batched
        (Assemble
           { a with shape = lead n a.shape; pieces = List.map piece a.pieces })
  | Contract _ -> results i ~by op (contract i n ~by op)
  | Fft (C2c f) ->
      batched (Fft (C2c { f with axes = Iarray.map succ f.axes; x = b f.x }))
  | Fft (R2c f) ->
      batched (Fft (R2c { f with axes = Iarray.map succ f.axes; x = b f.x }))
  | Fft (C2r f) ->
      batched (Fft (C2r { f with axes = Iarray.map succ f.axes; x = b f.x }))
  | Linalg l -> batched (Linalg (linalg i n l))
  | Copy x -> batched (Copy (b x))
  | Move (mv, x) -> results i ~by op (move i n ~by mv x)
  | Bitcast (dt, x) -> batched (Bitcast (dt, b x))
  | Place (q, x) -> results i ~by op (place i n ~by q x)
  | Check { ok; data = d; fail } ->
      (* Checked as one lane's, then a failure receives the index within the
         failing lane. *)
      Prim.results ~by (fun _ _ -> invalid_arg "Rune: a check's result") op;
      let fail idx e = fail (Array.sub idx 1 (Array.length idx - 1)) e in
      let ok = b ok in
      (* A held lane's check passes. *)
      let ok =
        match held with
        | None -> ok
        | Some h ->
            let r = flags_at ~what:"Nx.check" h.running ok in
            Nx.logical_or ok (Nx.logical_not (along r ok))
      in
      Prim.eval ~by (Check { ok; data = data i n d; fail })

(* Constructs *)

let operand i (Program.V x) = Option.is_some (Prim.payload i x)

(* A call answers whose program names the map, a lane value read or none. *)
let claims i axis : Program.construct -> bool = function
  (* A callback runs once per lane, whatever it reads, so a construct holding
     one is mapped. *)
  | c when Program.calls c -> true
  | Lanes (a, x) -> Program.same_axis a axis || owned i x
  (* An iterate whose programs read the map's lanes may stop them apart. *)
  | Iterate { step; until; _ }
    when Program.mentions axis step || Program.mentions axis until ->
      true
  | Switch { branches; _ } when Array.exists (Program.mentions axis) branches ->
      true
  | Lane_index (a, _) -> Program.same_axis a axis
  | Add _ | Emit _ -> true
  | Call { args; site; seeds; _ } ->
      Array.exists (operand i) args
      || Array.exists (operand i) seeds
      || List.exists (Program.same_axis axis) (Program.axes site)
  | c -> List.exists (operand i) (Program.operands c)

(* Custom rules

   A custom rule ({!Program.Custom}) or a linear call ({!Program.Linear}) over
   lane values, as {!Custom} calls trace them. It is performed outward with each
   of its programs traced again at its signature over the batches and evaluated
   over their lanes. A pullback or a transpose runs in a backward pass outside
   the map, so its additions reach a running value every lane shares. *)

module Rules = struct
  open Program

  let sub = Array.sub
  let residual_count = Custom.residual_count

  (* The map, its number of lanes, and [map ~shared f bs], the batches of [f]'s
     results over the lanes of the batches [bs] inside a map of the same name
     and size, the last of [bs] and of the results running values, shared where
     [shared] says. *)
  type ctx = {
    i : Prim.interpretation;
    n : int;
    map :
      shared:bool array ->
      (value array -> value array) ->
      value array ->
      value array;
  }

  (* [p] over the lanes of [vs], its results' batches, its totals threaded:
     shared where [shared] says, and otherwise every lane's. *)
  let over m ~shared ~results (p : Program.t) vs =
    let nv = Array.length vs in
    let out =
      Program.threaded ~lanes:(laned m.n shared) p (fun running ->
          m.map ~shared
            (fun lanes ->
              eval
                ~totals:(sub lanes nv (Array.length lanes - nv))
                p (sub lanes 0 nv))
            (Array.append vs running))
    in
    sub out 0 results

  let every (p : Program.t) b = Array.make (Array.length p.adds) b

  (* A custom rule over a map's values: the rule traced again over its operands'
     batches, every program evaluated over their lanes, so that the
     differentiations around the map apply it; its results are the lanes of the
     batched call's. A pullback's and a transpose's additions, which a backward
     pass makes outside the map, add each lane's rows in turn to the running
     value around that pass. *)
  let rule m c =
    let batched = Array.map (fun (V x) -> V (batch m.i m.n x)) in
    let lanes ~keep r =
      Array.mapi (fun k (V b as v) -> if keep k then v else V (lane m.i b)) r
    in
    (* [primal] over the operands' batches, its results and residuals batched in
       order, then the values the batched derivative [d] captures, which are its
       further residuals. *)
    let primal_over ~name ~shared ops (p : Program.t) ~ny more =
      Trace.trace Body ~name (Array.map form ops) (fun vs ->
          let nr = residual_count p ~ny in
          Array.append (over m ~shared ~results:(ny + nr) p vs) more)
    in
    (* Each derivative program is traced at its own signature, its parameters
       batched in their order. *)
    let batched_derivative ~name ~shared ~results (d : Program.t) forms
        (p : Program.t) ~ny =
      let res = sub p.outputs ny (residual_count p ~ny) in
      let forms =
        Array.append forms (Array.map (fun v -> stacked m.n (form v)) res)
      in
      Trace.trace Body ~name forms (over m ~shared:(every d shared) ~results d)
    in
    match c with
    | Custom
        {
          name;
          primal;
          derivative;
          ny;
          args;
          captures;
          residuals;
          applied;
          seeds;
        } ->
        let na = Array.length args in
        let ops = Array.append (batched args) (batched captures) in
        let derivative, more =
          match derivative with
          | Tangent t ->
              let forms = Array.map form (sub ops 0 na) in
              (* A tangent map's sums start from the identity, every lane's. *)
              let t, more =
                batched_derivative ~name ~shared:false ~results:ny t forms
                  primal ~ny
              in
              (Tangent t, more)
          | Pullback b ->
              let ys = sub primal.outputs 0 ny in
              let forms = Array.map (fun v -> stacked m.n (form v)) ys in
              let b, more =
                batched_derivative ~name ~shared:true ~results:na b forms primal
                  ~ny
              in
              (Pullback b, more)
        in
        let nr = residual_count primal ~ny in
        let shared = sharing m.i (Array.length primal.adds) seeds in
        let primal, captured = primal_over ~name ~shared ops primal ~ny more in
        let captures =
          Array.append (sub ops na (Array.length ops - na)) captured
        in
        let args = sub ops 0 na in
        let c =
          Custom
            {
              name;
              primal;
              derivative;
              ny;
              args;
              captures;
              residuals;
              applied;
              seeds = seeds_out m.i m.n shared seeds;
            }
        in
        (* Unused residuals hold nothing to take lanes of; the primal's sums
           follow the residuals. *)
        let base = ny + nr + Array.length more in
        let unused k = (not residuals) && k >= ny && k < base in
        let ns = Array.length shared in
        let sum k = k >= base && k < base + ns && shared.(k - base) in
        Some
          (Array.mapi
             (fun k (V b as v) ->
               if unused k then v
               else if sum k then share_v m.i v
               else V (lane m.i b))
             (perform c))
    | Linear { name; apply; transpose; args; captures; residuals; forms } ->
        let na = Array.length args and ny = Array.length forms in
        let args = batched args and captures = batched captures in
        let residuals = batched residuals in
        let forms = Array.map (stacked m.n) forms in
        let apply, captures =
          match apply with
          | None -> (None, captures)
          | Some a ->
              let a, more =
                Trace.trace Body ~name
                  (Array.map form (Array.append args captures))
                  (over m ~shared:(every a false) ~results:ny a)
              in
              (Some a, Array.append captures more)
        in
        let transpose =
          match transpose with
          | Traced t -> t
          | Host _ ->
              (* Only a tape's forward application, which has none, maps it. *)
              invalid_arg
                (name
               ^ ": a custom_vjp rule has no forward derivative; give the \
                  function a custom_jvp rule")
        in
        let transpose, more =
          Trace.trace Body ~name
            (Array.append forms (Array.map form residuals))
            (over m ~shared:(every transpose true) ~results:na transpose)
        in
        let residuals = Array.append residuals more in
        Some
          (lanes
             ~keep:(fun _ -> false)
             (perform
                (Linear
                   {
                     name;
                     apply;
                     transpose = Traced transpose;
                     args;
                     captures;
                     residuals;
                     forms;
                   })))
    | _ -> None
end

(* Programs' maps

   A loop or a remat over lane values is performed outward with its program
   mapped: [eval p] traced under a map of its own, of the same name and lanes,
   over the batches and the shared values, as JAX batches a jaxpr. *)

(* A form's batch over [n] lanes. *)
let stacked_form (type v s d) n (f : (v, s, d) Nx.Prim.form) :
    (v, s, d) Nx.Prim.form =
  let shape = Iarray.to_array (Nx_array.Layout.shape f.layout) in
  {
    f with
    layout =
      Nx_array.Layout.contiguous (Iarray.of_array (Array.append [| n |] shape));
    placement = Nx.Placement.with_leading_axis f.placement;
  }

let lane_v i (Program.V x) = Program.V (lane i x)
let batch_v i n (Program.V x) = Program.V (batch i n x)
let owned_v i (Program.V x) = owned i x

(* A batch [n; length; ...] read row by row: [length; n; ...]. *)
let rows_of i (Program.V x as v) =
  match batched i x with Some b -> Program.V (Nx.moveaxis 0 1 b) | None -> v

(* [vs], lane values of [w] where [mask] holds. *)
let lanes_of w mask vs =
  Array.map2 (fun m v -> if m then lane_v w v else v) mask vs

(* A callback over this map's lanes, performed outward as one callback over
   their batches. With [~lanes], its function runs once on them, a held lane
   given its donor's rows. Otherwise it runs once per lane that runs, in lane
   order, on the lane's rows; a held lane's results are zeros. The flags of the
   lanes that run are an argument, read on the host. *)
let callback ?held i n (cb : Program.callback) : Program.callback =
  let forms = Array.map (Program.stacked n) cb.forms in
  let args = Array.map (batch_v i n) cb.args in
  match (cb.lanes, held) with
  | Some _, None -> { cb with maps = n :: cb.maps; args; forms }
  | Some _, Some h ->
      let args = Array.map (lend ~kept:false h) args in
      { cb with maps = n :: cb.maps; args; forms }
  | None, _ ->
      let flags = match held with Some h -> [| h.running |] | None -> [||] in
      let nf = Array.length flags in
      let run vs =
        let runs =
          if nf = 0 then fun _ -> true
          else
            let (Program.V r) = vs.(0) in
            let r = Nx.to_array (Nx.copy (as_bool r)) in
            fun j -> r.(j)
        in
        let vs = Array.sub vs nf (Array.length vs - nf) in
        let row j =
          if not (runs j) then Array.map (Loops.filled ~zero:true) cb.forms
          else
            let ys =
              cb.run (Array.map (fun (Program.V x) -> Program.V (Nx.get [ j ] x)) vs)
            in
            Program.conform cb ys;
            ys
        in
        let rows = List.init n row in
        Array.mapi
          (fun k f ->
            match rows with
            | [] -> Loops.filled ~zero:true f
            | first :: _ ->
                let (Program.V y) = first.(k) in
                Program.V
                  (Nx.stack ~axis:0
                     (List.map (fun ys -> Program.like y ys.(k)) rows)))
          forms
      in
      {
        run;
        lanes = None;
        maps = [];
        args = Array.append flags args;
        forms;
        paths = cb.paths;
      }

(* [Lanes] of this map is every lane's value, the batch, a constant; of another
   map, every lane's batch, a lane value. An addition to a running value around
   the map adds each lane's rows in turn, lane 0 first, a held lane's row the
   identity; one to a lane's own running value is the lanes' additions. Where
   lanes are held, the map's lanes raise. *)
let rec answer ~name ?held i axis n (c : Program.construct) k =
  let one x = Effect.Deep.continue k [| Program.V x |] in
  match c with
  | Lanes (a, _) when Program.same_axis a axis && Option.is_some held ->
      Effect.Deep.discontinue k
        (Invalid_argument
           "Rune.lanes: the lanes of the map are held apart, at an iterate's \
            different steps or a switch's different branches")
  | Lanes (a, x) when Program.same_axis a axis -> one (batch i n x)
  | Lanes (a, x) ->
      let b = batch i n x in
      let r = Program.like b (Program.perform (Lanes (a, b))).(0) in
      one (lane i (Nx.swapaxes 0 1 r))
  | Lane_index (_, p) -> (
      (* [p] cuts no axis, so the indices' batch at [p] gives lanes at [p]; a
         held lane's is its donor's. *)
      match held with
      | Some h ->
          let (Program.V d) = h.donors in
          one (lane i (Nx.place p (as_int32 d)))
      | None -> one (lane i (Nx.arange_at p Nx.int32 0 n 1)))
  | Add (t, x) ->
      (* No collect or tracer holds the total: an addition of no consequence,
         offered outward for the interpretations that refuse it. *)
      ignore (Program.perform (Add (t, batch i n x)));
      Effect.Deep.continue k [||]
  | Accumulate { lanes; running; x } -> (
      match shared i running with
      | None -> Effect.Deep.discontinue k Program.Evaluate
      | Some r ->
          let b = batch i n x in
          let b = match held with Some h -> dropped h.running b | None -> b in
          let lanes = lanes + 1 in
          let y = Program.perform (Accumulate { lanes; running = r; x = b }) in
          one (share i (Program.like r y.(0))))
  | Emit (Emission e) ->
      (* The map's lanes lead the lanes the emission has, and a held lane's rows
         did not run. *)
      let values = Array.map (batch_v i n) e.values in
      let mask = Option.map (batch_v i n) e.mask in
      let mask =
        match (held, values) with
        | None, _ | _, [||] -> mask
        | Some h, _ -> Some (held_rows h values.(0) ~lead:(e.lead + 1) mask)
      in
      let e = Program.Emission { e with lead = e.lead + 1; values; mask } in
      ignore (Program.perform (Emit e));
      Effect.Deep.continue k [||]
  | Callback cb ->
      let r = Program.perform (Callback (callback ?held i n cb)) in
      Effect.Deep.continue k (Array.map (lane_v i) r)
  | Detach x ->
      let b = batch i n x in
      one (lane i (Program.like b (Program.perform (Detach b)).(0)))
  | Custom _ | Linear _ ->
      let r = Rules.rule { i; n; map = map ~name ?held axis n } c in
      Effect.Deep.continue k (Option.get r)
  | Scan { trips = Some (V t); _ } when owned i t ->
      Effect.Deep.continue k (masked_scan ~name ?held i axis n c)
  | Scan { step; length; trips; reverse; init; xs; captures; seeds } ->
      Effect.Deep.continue k
        (scan ~name ?held i axis n step ~length ~trips ~reverse ~init ~xs
           ~captures ~seeds)
  | Remat { body; args; seeds; differentiated } ->
      Effect.Deep.continue k
        (remat ~name ?held i axis n body args seeds ~differentiated)
  | Call { body; args; site; seeds } ->
      Effect.Deep.continue k (call ~name ?held i axis n body args site seeds)
  | Iterate _ -> Effect.Deep.continue k (iterate ~name ?held i axis n c)
  | Switch _ -> Effect.Deep.continue k (switch ~name ?held i axis n c)
  (* An operator over lane values applies over the lanes: its arguments' and
     those of its data that hold them. *)
  | Operator o ->
      let mask = Array.map (owned_v i) o.data in
      let r =
        Program.perform
          (Operator
             {
               o with
               lanes = (n, mask) :: o.lanes;
               args = Array.map (batch_v i n) o.args;
               data =
                 Array.map2
                   (fun l v -> if l then batch_v i n v else v)
                   mask o.data;
               forms = Array.map (Program.stacked n) o.forms;
             })
      in
      Effect.Deep.continue k (Array.map (lane_v i) r)
  | Root _ | Solve _ -> Effect.Deep.continue k (root ~name i axis n c)

(* A scan's map is the scan of its step's map, whose carry has lanes for each
   carry that has them at the call or gains them from the step: a fixpoint, as
   JAX's scan batches its carry. A carry that gains lanes starts from its init
   broadcast; a row with lanes is read along the batch's axis 1. A sum whose
   seed is a running value around the map stays shared, so that each addition
   adds its lanes in turn; the others are every lane's. *)
and scan ~name ?held i axis n (step : Program.t) ~length ~trips ~reverse ~init
    ~xs ~captures ~seeds =
  let nc = Array.length init and nx = Array.length xs in
  let ncap = Array.length captures in
  let ny = Array.length step.outputs - nc - Array.length step.adds in
  let bxs = Array.map (rows_of i) xs and lx = Array.map (owned_v i) xs in
  let bcaps =
    Array.map (fun v -> if owned_v i v then batch_v i n v else v) captures
  in
  let lcap = Array.map (owned_v i) captures in
  let shared = sharing i (Array.length step.adds) seeds in
  let carry lc k = if lc.(k) then batch_v i n init.(k) else init.(k) in
  let rec fix lc (cf : Program.form array) =
    let gained = Array.make nc false in
    let ly = ref [||] in
    let forms =
      Array.concat [ cf; Array.map Loops.row bxs; Array.map Program.form bcaps ]
    in
    let mapped vs =
      match Loops.split [ nc; nx; ncap ] vs with
      | [ c; r; cap ] ->
          let ys = ref [||] in
          let (_ : Program.value array) =
            Program.threaded ~lanes:(laned n shared) step (fun running ->
                interpret ~name ~axis ?held n (fun w ->
                    let args =
                      Array.concat
                        [
                          lanes_of w lc c; lanes_of w lx r; lanes_of w lcap cap;
                        ]
                    in
                    let totals = totals_in w shared running in
                    let out = Program.eval ~totals step args in
                    let c' =
                      Array.init nc (fun k ->
                          let laned = owned_v w out.(k) in
                          gained.(k) <- laned && not lc.(k);
                          if lc.(k) then batch_v w n out.(k) else out.(k))
                    in
                    let y = Array.sub out nc ny in
                    ly := Array.map (owned_v w) y;
                    let y =
                      Array.map
                        (fun v -> if owned_v w v then batch_v w n v else v)
                        y
                    in
                    let adds =
                      Array.sub out (nc + ny) (Array.length out - nc - ny)
                    in
                    ys := Array.append c' y;
                    totals_out w n shared adds))
          in
          !ys
      | _ -> invalid_arg "Rune: a scan's map"
    in
    let prog, own = Trace.trace Body ~name forms mapped in
    if Array.exists Fun.id gained then
      let lc = Array.map2 ( || ) lc gained in
      fix lc
        (Array.mapi
           (fun k f -> if gained.(k) then Program.form (carry lc k) else f)
           cf)
    else (prog, own, lc, !ly)
  in
  let lc = Array.map (owned_v i) init in
  let prog, own, lc, ly =
    fix lc (Array.init nc (fun k -> Program.form (carry lc k)))
  in
  let r =
    Program.perform
      (Scan
         {
           step = prog;
           length;
           trips;
           reverse;
           init = Array.init nc (carry lc);
           xs = bxs;
           captures = Array.append bcaps own;
           seeds = seeds_out i n shared seeds;
         })
  in
  let base = nc + ny in
  let ends = ends_in i shared (Array.sub r base (Array.length r - base)) in
  Array.append
    (Array.mapi
       (fun k v ->
         if k < nc then if lc.(k) then lane_v i v else v
         else if ly.(k - nc) then
           let (Program.V y) = v in
           Program.V (lane i (Nx.moveaxis 1 0 y))
         else v)
       (Array.sub r 0 base))
    ends

(* An iterate's map. Its carry has lanes where its init has them or a step gives
   them, a fixpoint. Where its stop is every lane's, the lanes stop together:
   the iterate of its programs' maps. Otherwise, and inside the step of an
   iterate whose lanes stop apart, they stop apart: every carry holds each
   lane's, beside one more, each lane's stop, which each lane tests at its init
   and after each of its steps; the loop runs until every lane has stopped, and
   a lane that has stopped is held. *)
and iterate ~name ?held i axis n (c : Program.construct) =
  match c with
  | Iterate it ->
      let nc = Array.length it.init in
      let ny = Array.length it.step.outputs - nc - Array.length it.step.adds in
      if it.rows = Some Taken then
        invalid_arg (it.name ^ ": a map of an iterate stacking its steps taken");
      let laned = Array.map (owned_v i) in
      let lsc = laned it.step_captures and luc = laned it.until_captures in
      let batched ls vs =
        Array.map2 (fun l v -> if l then batch_v i n v else v) ls vs
      in
      let bsc = batched lsc it.step_captures
      and buc = batched luc it.until_captures in
      let carry lc k =
        if lc.(k) then batch_v i n it.init.(k) else it.init.(k)
      in
      let forms = Array.map Program.form in
      (* The carries with lanes, and their forms. *)
      let rec fix lc cf =
        let gained = Array.make nc false in
        let (_ : Program.t * Program.value array) =
          Trace.trace Body ~name
            (Array.append cf (forms bsc))
            (fun vs ->
              interpret ~name ~axis ?held n (fun w ->
                  let args =
                    Array.append
                      (lanes_of w lc (Array.sub vs 0 nc))
                      (lanes_of w lsc (Array.sub vs nc (Array.length bsc)))
                  in
                  let out = Program.eval it.step args in
                  Array.iteri
                    (fun k v -> gained.(k) <- owned_v w v && not lc.(k))
                    (Array.sub out 0 nc);
                  [||]))
        in
        if Array.exists Fun.id gained then
          let lc = Array.map2 ( || ) lc gained in
          fix lc
            (Array.mapi
               (fun k f -> if gained.(k) then Program.form (carry lc k) else f)
               cf)
        else (lc, cf)
      in
      let lc = laned it.init in
      let lc, cf =
        fix lc (Array.init nc (fun k -> Program.form (carry lc k)))
      in
      let stop_laned =
        let laned = ref false in
        let (_ : Program.t * Program.value array) =
          Trace.trace Body ~name
            (Array.append cf (forms buc))
            (fun vs ->
              interpret ~name ~axis ?held n (fun w ->
                  let args =
                    Array.append
                      (lanes_of w lc (Array.sub vs 0 nc))
                      (lanes_of w luc (Array.sub vs nc (Array.length buc)))
                  in
                  laned := owned_v w (Program.eval it.until args).(0);
                  [||]))
        in
        !laned
      in
      if (not stop_laned) && Option.is_none held then
        together ~name i axis n c lc cf ~bsc ~buc ~lsc ~luc ~ny
      else apart ~name ?held i axis n c ~bsc ~buc ~lsc ~luc ~ny
  | _ -> invalid_arg "Rune: an iterate's map"

(* Lanes that stop together: the iterate of its step's and stop's maps, its sums
   shared or every lane's as a scan's. *)
and together ~name i axis n c lc cf ~bsc ~buc ~lsc ~luc ~ny =
  match c with
  | Iterate it ->
      let nc = Array.length lc in
      let ly = ref [||] in
      let sum_adds, _, feeds = Program.sums [ it.until; it.step ] in
      let shared = sharing i (Array.length sum_adds) it.seeds in
      let mapped (p : Program.t) feeds caps lcaps keep =
        let shared = fed shared feeds in
        Trace.trace Body ~name
          (Array.append cf (Array.map Program.form caps))
          (fun vs ->
            Program.threaded ~lanes:(laned n shared) p (fun running ->
                interpret ~name ~axis n (fun w ->
                    let totals = totals_in w shared running in
                    let args =
                      Array.append
                        (lanes_of w lc (Array.sub vs 0 nc))
                        (lanes_of w lcaps (Array.sub vs nc (Array.length caps)))
                    in
                    let out = Program.eval ~totals p args in
                    let nr = Array.length out - Array.length p.adds in
                    Array.append
                      (keep w (Array.sub out 0 nr))
                      (totals_out w n shared
                         (Array.sub out nr (Array.length p.adds))))))
      in
      let step, sown =
        mapped it.step (List.nth feeds 1) bsc lsc (fun w out ->
            let c =
              Array.init nc (fun k ->
                  if lc.(k) then batch_v w n out.(k) else out.(k))
            in
            let y = Array.sub out nc ny in
            ly := Array.map (owned_v w) y;
            Array.append c
              (Array.map (fun v -> if owned_v w v then batch_v w n v else v) y))
      in
      let until, uown =
        mapped it.until (List.nth feeds 0) buc luc (fun _ out -> [| out.(0) |])
      in
      let r =
        Program.perform
          (Iterate
             {
               it with
               step;
               until;
               init =
                 Array.init nc (fun k ->
                     if lc.(k) then batch_v i n it.init.(k) else it.init.(k));
               step_captures = Array.append bsc sown;
               until_captures = Array.append buc uown;
               seeds = seeds_out i n shared it.seeds;
             })
      in
      let stacked = if Option.is_some it.rows then 1 + ny else 0 in
      let base = nc + stacked in
      Array.append
        (Array.mapi
           (fun k v ->
             if k < nc then if lc.(k) then lane_v i v else v
             else if k = nc then v
             else if !ly.(k - nc - 1) then
               let (Program.V y) = v in
               Program.V (lane i (Nx.moveaxis 1 0 y))
             else v)
           (Array.sub r 0 base))
        (ends_in i shared (Array.sub r base (Array.length r - base)))
  | _ -> invalid_arg "Rune: an iterate's map"

(* Lanes that stop apart. Each lane tests its stop at its init, outside the
   loop, and after each of its steps, inside it. *)
and apart ~name ?held i axis n c ~bsc ~buc ~lsc ~luc ~ny =
  match c with
  | Iterate it ->
      let nc = Array.length it.init in
      let rows = Option.is_some it.rows in
      let all = Array.make nc true in
      let sum_adds, sum_forms, feeds = Program.sums [ it.until; it.step ] in
      let until_feeds = List.nth feeds 0 and step_feeds = List.nth feeds 1 in
      let shared = sharing i (Array.length sum_adds) it.seeds in
      (* [p] over the lanes of [vs] where [laned], its totals from the sums
         [acc]: its results' batches, and the sums' ends, a held lane's kept. *)
      let over ?h p feeds laned vs acc =
        evaluated ~name ~axis ?held:h n p feeds ~shared laned vs acc
      in
      let b0 = Array.map (batch_v i n) it.init in
      let acc0 = starts i n shared sum_forms it.seeds in
      let flags, acc1 =
        over ?h:held it.until until_feeds (Array.append all luc)
          (Array.append b0 buc) acc0
      in
      (* A lane an outer loop holds starts stopped. *)
      let stopped0 =
        match held with
        | None -> flags.(0)
        | Some h -> or_flags flags.(0) (not_flags h.running)
      in
      let trips0 =
        let (Program.V f) = stopped0 in
        Program.V (Nx.zeros_at (Nx.placement f) Nx.int32 [| n |])
      in
      let ys = Array.sub it.step.outputs nc ny in
      let thread f = Program.threading ~lanes:(laned n shared) sum_adds f in
      let totals = lanes_placed n shared sum_adds in
      let step_of cf =
        let outs = ref [||] in
        let p, own =
          Trace.trace ~totals Body ~name
            (Array.append cf (Array.map Program.form bsc))
            (fun vs ->
              let b = Array.sub vs 0 nc and stopped = vs.(nc) in
              let base = if rows then nc + 2 else nc + 1 in
              let caps = Array.sub vs base (Array.length vs - base) in
              thread (fun acc ->
                  let running = not_flags stopped in
                  let h = { running; donors = donors n running } in
                  let bd = Array.map (lend h) b in
                  let capsd =
                    Array.map2 (fun l v -> if l then lend h v else v) lsc caps
                  in
                  let out, acc =
                    over ~h it.step step_feeds (Array.append all lsc)
                      (Array.append bd capsd) acc
                  in
                  let b' =
                    Array.init nc (fun k -> hold running out.(k) b.(k))
                  in
                  let ys' =
                    Array.init ny (fun j ->
                        hold running
                          out.(nc + j)
                          (Loops.filled ~zero:true
                             (Program.stacked n (Program.form ys.(j)))))
                  in
                  let f, acc =
                    over ~h it.until until_feeds (Array.append all luc)
                      (Array.append b' buc) acc
                  in
                  let stopped' = or_flags stopped f.(0) in
                  let trips =
                    if rows then [| count_up running vs.(nc + 1) |] else [||]
                  in
                  let carry = Array.concat [ b'; [| stopped' |]; trips ] in
                  outs := Array.map Program.form carry;
                  (acc, Array.append carry ys')))
        in
        (p, own, !outs)
      in
      let init =
        Array.concat
          [ b0; [| stopped0 |]; (if rows then [| trips0 |] else [||]) ]
      in
      let cf = Array.map Program.form init in
      let step, own, _ = step_of cf in
      let until, uown =
        Trace.trace Body ~name cf (fun vs ->
            let (Program.V s) = vs.(nc) in
            [| Program.V (Nx.all (as_bool s)) |])
      in
      let r =
        Program.perform
          (Iterate
             {
               it with
               step;
               until;
               stopped = Some nc;
               init;
               step_captures = Array.append bsc own;
               until_captures = uown;
               seeds = acc1;
             })
      in
      let carried = Array.length init in
      let lanes v = lane_v i v in
      let rows_lanes (Program.V y) = Program.V (lane i (Nx.moveaxis 1 0 y)) in
      Array.concat
        [
          Array.map lanes (Array.sub r 0 nc);
          (if rows then
             Array.append
               [| lanes r.(nc + 1) |]
               (Array.map rows_lanes (Array.sub r (carried + 1) ny))
           else [||]);
          ends_in i shared
            (Array.sub r
               (carried + if rows then 1 + ny else 0)
               (Array.length sum_adds));
        ]
  | _ -> invalid_arg "Rune: an iterate's map"

(* Where a loop's sums start: every lane's identity where no seed is given, and
   otherwise each seed, a shared one as the running value around the map. *)
and starts i n shared forms seeds =
  if seeds = [||] then
    Array.map (fun f -> Loops.filled ~zero:false (Program.stacked n f)) forms
  else seeds_out i n shared seeds

(* Each of [adds]'s totals' running value where it lies: every lane's stacked, a
   shared one as it is. *)
and lanes_placed n shared adds =
  Array.mapi
    (fun k (Program.Addition (t, into, out)) ->
      let into =
        match Program.where t into.placement with
        | At (Some p) -> { into with placement = p }
        | At None | Nowhere -> into
      in
      let into = if shared.(k) then into else stacked_form n into in
      Program.Addition (t, into, out))
    adds

(* [p] over the lanes of [vs] where [laned], its totals from the sums [acc]: its
   results' batches, and the sums' ends. A shared sum adds a held lane's rows as
   the identity; every lane's keeps a held lane's. A sum [p] cannot take at the
   form it traced its running value in at, as one a held lane's computed on a
   set, is added to after a run from the identity. *)
and evaluated ~name ~axis ?held n (p : Program.t) feeds ~shared laned vs acc =
  let shared = fed shared feeds in
  let fits =
    Array.mapi
      (fun k (Program.Addition (_, into, _)) ->
        let (Program.V a) = acc.(feeds.(k)) in
        shared.(k) || Option.is_some (Program.coerce_at into a))
      p.adds
  in
  let na = Array.length feeds in
  let out =
    interpret ~name ~axis ?held n (fun w ->
        let totals =
          Array.mapi
            (fun k (Program.Addition (_, into, _)) ->
              if shared.(k) then share_v w acc.(feeds.(k))
              else if fits.(k) then lane_v w acc.(feeds.(k))
              else Loops.filled ~zero:false (Program.F into))
            p.adds
        in
        let out = Program.eval ~totals p (lanes_of w laned vs) in
        let nr = Array.length out - na in
        Array.append
          (Array.map (batch_v w n) (Array.sub out 0 nr))
          (totals_out w n shared (Array.sub out nr na)))
  in
  let nr = Array.length out - na in
  let acc = Array.copy acc in
  Array.iteri
    (fun k j ->
      let o = out.(nr + k) in
      acc.(j) <-
        (if shared.(k) then o
         else
           let o = if fits.(k) then o else Loops.add acc.(j) o in
           match held with Some h -> hold h.running o acc.(j) | None -> o))
    feeds;
  (Array.sub out 0 nr, acc)

(* A scan whose rows that run differ between lanes: it runs the most rows any
   lane runs, a lane holding its carry from its last row on, and computing the
   step at its donor's values meanwhile. *)
and masked_scan ~name ?held i axis n (c : Program.construct) =
  match c with
  | Scan ({ trips = Some trips; _ } as sc) ->
      let nc = Array.length sc.init and nx = Array.length sc.xs in
      let ny = Array.length sc.step.outputs - nc - Array.length sc.step.adds in
      let lcap = Array.map (owned_v i) sc.captures in
      let bcaps =
        Array.map2 (fun l v -> if l then batch_v i n v else v) lcap sc.captures
      in
      let b0 = Array.map (batch_v i n) sc.init in
      let bxs =
        Array.map
          (fun v ->
            let (Program.V b) = batch_v i n v in
            Program.V (Nx.moveaxis 0 1 b))
          sc.xs
      in
      let (Program.V tb) = batch_v i n trips in
      let tb = as_int32 tb in
      let at = Nx.beside tb in
      let most =
        if n = 0 then Program.V (Nx.scalar_at at Nx.int32 0l)
        else Program.V (Nx.max tb)
      in
      let row = Program.V (Nx.arange_at at Nx.int32 0 sc.length 1) in
      let sum_adds, sum_forms, feeds = Program.sums [ sc.step ] in
      let shared = sharing i (Array.length sum_adds) sc.seeds in
      let feeds = List.hd feeds in
      let all = Array.make (nc + nx) true in
      let ys = Array.sub sc.step.outputs nc ny in
      let totals = lanes_placed n shared sum_adds in
      let step_of cf =
        let outs = ref [||] in
        let p, own =
          Trace.trace ~totals Body ~name
            (Array.concat
               [
                 cf;
                 Array.map Loops.row bxs;
                 [| Loops.row row |];
                 Array.map Program.form bcaps;
               ])
            (fun vs ->
              let b = Array.sub vs 0 nc and xr = Array.sub vs nc nx in
              let k = vs.(nc + nx) in
              let caps = Array.sub vs (nc + nx + 1) (Array.length bcaps) in
              Program.threading ~lanes:(laned n shared) sum_adds (fun acc ->
                  let runs =
                    beside (Program.V tb) k
                      {
                        k =
                          (fun t k ->
                            Program.V
                              (Nx.less
                                 (Nx.broadcast_to [| n |] (as_int32 k))
                                 (as_int32 t)));
                      }
                  in
                  let running =
                    match held with
                    | None -> runs
                    | Some h -> and_flags runs h.running
                  in
                  let h = { running; donors = donors n running } in
                  let args =
                    Array.concat
                      [
                        Array.map (lend h) b;
                        Array.map (lend h) xr;
                        Array.map2
                          (fun l v -> if l then lend h v else v)
                          lcap caps;
                      ]
                  in
                  let out, acc' =
                    evaluated ~name ~axis ~held:h n sc.step feeds ~shared
                      (Array.append all lcap) args acc
                  in
                  let b' =
                    Array.init nc (fun j -> hold running out.(j) b.(j))
                  in
                  let ys' =
                    Array.init ny (fun j ->
                        hold running
                          out.(nc + j)
                          (Loops.filled ~zero:true
                             (Program.stacked n (Program.form ys.(j)))))
                  in
                  outs := Array.map Program.form b';
                  (acc', Array.append b' ys')))
        in
        (p, own, !outs)
      in
      let step, own, _ = step_of (Array.map Program.form b0) in
      let r =
        Program.perform
          (Scan
             {
               sc with
               step;
               trips = Some most;
               init = b0;
               xs = Array.append bxs [| row |];
               captures = Array.append bcaps own;
               seeds = starts i n shared sum_forms sc.seeds;
             })
      in
      let ns = Array.length sum_adds in
      let base = Array.length r - ns in
      Array.append
        (Array.mapi
           (fun k v ->
             if k >= nc && k < nc + ny then
               let (Program.V y) = v in
               Program.V (lane i (Nx.moveaxis 1 0 y))
             else lane_v i v)
           (Array.sub r 0 base))
        (ends_in i shared (Array.sub r base ns))
  | _ -> invalid_arg "Rune: a scan's map"

(* A root or a linear solve over lane values is performed outward over every
   lane's values, its system's programs mapped: a lane's operator stays, which
   the linear solve's map applies over the lanes. *)
and root ~name i axis n (c : Program.construct) =
  let batched = Array.map (batch_v i n) in
  let forms = Array.map Program.form in
  (* Each program of the system is a function of its operands alone, traced
     staged: a body would run what reads none of them at the call, where under
     a jit it is a capture the system does not carry. *)
  let mapped (p : Program.t) vs =
    let q, own =
      Trace.trace Staged ~name (forms vs) (fun bs ->
          interpret ~name ~axis n (fun w ->
              let out = Program.eval p (Array.map (lane_v w) bs) in
              Array.map (batch_v w n) out))
    in
    if own <> [||] then invalid_arg (name ^ ": a root's program captures");
    q
  in
  let system (s : Program.system) x th caps =
    {
      s with
      residual = mapped s.residual (Array.append x th);
      jacobian = mapped s.jacobian (Array.concat [ x; x; th ]);
      transpose =
        Program.once (fun ~name ->
            mapped (s.transpose ~name) (Array.concat [ x; x; th ]));
      solve = mapped s.solve (Array.concat [ x; x; th; caps ]);
    }
  in
  let r =
    match c with
    | Root rt ->
        let x = batched rt.solution and th = batched rt.theta in
        let caps = batched rt.captures in
        Program.perform
          (Root
             {
               rt with
               system = system rt.system x th caps;
               solution = x;
               theta = th;
               captures = caps;
             })
    | Solve sv ->
        let b = batched sv.b and data = batched sv.data in
        let nx = Array.length b in
        let x = Array.sub data 0 nx
        and th = Array.sub data nx (Array.length data - nx) in
        let caps = batched sv.captures in
        Program.perform
          (Solve
             {
               sv with
               system = system sv.system x th caps;
               b;
               data;
               captures = caps;
             })
    | _ -> invalid_arg "Rune: a root's map"
  in
  Array.map (lane_v i) r

(* A switch whose index every lane shares is the switch of its branches' maps;
   one whose index is per lane runs its branches apart. *)
and switch ~name ?held i axis n (c : Program.construct) =
  match c with
  | Switch sw ->
      let (Program.V idx) = sw.index in
      let lcap = Array.map (owned_v i) sw.captures in
      let bcaps =
        Array.map2 (fun l v -> if l then batch_v i n v else v) lcap sw.captures
      in
      if owned i idx then switch_apart ~name ?held i axis n sw ~lcap ~bcaps
      else switch_shared ~name ?held i axis n sw ~lcap ~bcaps
  | _ -> invalid_arg "Rune: a switch's map"

and switch_shared ~name ?held i axis n (sw : Program.switch) ~lcap ~bcaps =
  let sum_adds, _, feeds = Program.sums (Array.to_list sw.branches) in
  let shared = sharing i (Array.length sum_adds) sw.seeds in
  let mapped (b : Program.t) feeds vs =
    let shared = fed shared feeds in
    Program.threaded ~lanes:(laned n shared) b (fun running ->
        interpret ~name ~axis ?held n (fun w ->
            let totals = totals_in w shared running in
            let out = Program.eval ~totals b (lanes_of w lcap vs) in
            let nr = Array.length out - Array.length totals in
            Array.append
              (Array.map (batch_v w n) (Array.sub out 0 nr))
              (totals_out w n shared (Array.sub out nr (Array.length totals)))))
  in
  let traced =
    Array.mapi
      (fun k b ->
        Trace.trace Staged ~name
          (Array.map Program.form bcaps)
          (mapped b (List.nth feeds k)))
      sw.branches
  in
  let progs, own = Program.unite traced ~params:(Array.length bcaps) in
  let r =
    Program.perform
      (Switch
         {
           sw with
           branches = progs;
           captures = Array.append bcaps own;
           seeds = seeds_out i n shared sw.seeds;
         })
  in
  let base = Array.length r - Array.length sum_adds in
  Array.append
    (Array.map (lane_v i) (Array.sub r 0 base))
    (ends_in i shared (Array.sub r base (Array.length sum_adds)))

(* Each branch some lane takes runs over every lane, a lane that did not take it
   held at its donor, as an iterate's stopped lanes are, and each lane takes its
   own branch's results, a lane an outer mask holds its donor's. The lanes'
   indices are read once where they are values, and otherwise each branch runs
   under a switch on whether a lane takes it, a flag the device can gate on. *)
and switch_apart ~name ?held i axis n (sw : Program.switch) ~lcap ~bcaps =
  let (Program.V idx) = sw.index in
  let branches = sw.branches in
  let nb = Array.length branches in
  let results = Program.switched branches in
  let no = Array.length results in
  let sum_adds, sum_forms, feeds = Program.sums (Array.to_list branches) in
  let ns = Array.length sum_forms in
  let b = as_int32 (batch i n idx) in
  let shared = sharing i ns sw.seeds in
  let acc0 = starts i n shared sum_forms sw.seeds in
  let zeros () =
    Array.map (fun f -> Loops.filled ~zero:true (Program.stacked n f)) results
  in
  (* Each running lane's index in range, a check, naming the first lane
     outside. *)
  let inside =
    Nx.logical_and (Nx.greater_equal_s b 0l) (Nx.less_s b (Int32.of_int nb))
  in
  let ok =
    match held with
    | None -> inside
    | Some h ->
        Nx.logical_or inside
          (Nx.logical_not (flags_at ~what:sw.name h.running b))
  in
  Nx.check Nx.Ptree.tensor ok b (fun at k ->
      Invalid_argument
        (Printf.sprintf "%s: index %ld outside 0 .. %d, in lane %d" sw.name
           (Nx.item [] k) (nb - 1) at.(0)));
  let chose k = Program.V (Nx.equal_s b (Int32.of_int k)) in
  (* The lanes that run branch [k]: those that chose it, of those running. *)
  let takes k =
    match held with None -> chose k | Some h -> and_flags h.running (chose k)
  in
  let run k running vs acc =
    let h = { running; donors = donors n running } in
    let caps =
      Array.map2 (fun l v -> if l then lend ~kept:false h v else v) lcap vs
    in
    evaluated ~name ~axis ~held:h n branches.(k) (List.nth feeds k) ~shared lcap
      caps acc
  in
  (* Each lane's results, its own branch's: a lane an outer mask holds chose its
     donor's. *)
  let merge runs =
    let res = zeros () in
    List.iter
      (fun (k, out) ->
        Array.iteri (fun j o -> res.(j) <- hold (chose k) o res.(j)) out)
      runs;
    res
  in
  let res, acc =
    if Option.is_none (Prim.owner b) then begin
      (* Every lane's index, read once. *)
      let ks = if n = 0 then [||] else Nx.to_array b in
      let taken =
        List.filter
          (fun k -> Array.exists (fun x -> Int32.to_int x = k) ks)
          (List.init nb Fun.id)
      in
      let acc = ref acc0 and runs = ref [] in
      List.iter
        (fun k ->
          let out, a = run k (takes k) bcaps !acc in
          acc := a;
          runs := (k, out) :: !runs)
        taken;
      (merge (List.rev !runs), !acc)
    end
    else begin
      let acc = ref acc0 and runs = ref [] in
      for k = 0 to nb - 1 do
        let running = takes k in
        let (Program.V r) = running in
        let any = Nx.cast Nx.int32 (Nx.any (as_bool r)) in
        let forms = Array.map Program.form (Array.append [| running |] bcaps) in
        let nc = Array.length bcaps in
        (* The sums thread the branches, so that a shared one stays a running
           value. *)
        let thread f = Program.threading ~lanes:(laned n shared) sum_adds f in
        let skip _ = thread (fun acc -> (acc, zeros ())) in
        let go vs =
          thread (fun acc ->
              let out, a = run k vs.(0) (Array.sub vs 1 nc) acc in
              (a, out))
        in
        let totals = lanes_placed n shared sum_adds in
        let traced =
          [|
            Trace.trace ~totals Staged ~name forms skip;
            Trace.trace ~totals Staged ~name forms go;
          |]
        in
        let progs, own = Program.unite traced ~params:(Array.length forms) in
        let r =
          Program.perform
            (Switch
               {
                 name = sw.name;
                 index = Program.V any;
                 branches = progs;
                 captures = Array.concat [ [| running |]; bcaps; own ];
                 seeds = !acc;
               })
        in
        acc := Array.sub r no ns;
        runs := (k, Array.sub r 0 no) :: !runs
      done;
      (merge (List.rev !runs), !acc)
    end
  in
  Array.append (Array.map (lane_v i) res) (ends_in i shared acc)

(* A remat's map is the remat of its function's map, as JAX's checkpoint
   batches, so that a reverse mode around the map sees the remat. *)
and remat ~name ?held i axis n (body : Program.t) args seeds ~differentiated =
  let la = Array.map (owned_v i) args in
  let bargs =
    Array.map (fun v -> if owned_v i v then batch_v i n v else v) args
  in
  let laned_out = ref [||] in
  let no = Array.length body.outputs - Array.length body.adds in
  let shared = sharing i (Array.length body.adds) seeds in
  let mapped vs =
    Program.threaded ~lanes:(laned n shared) body (fun running ->
        interpret ~name ~axis ?held n (fun w ->
            let totals = totals_in w shared running in
            let out = Program.eval ~totals body (lanes_of w la vs) in
            laned_out := Array.sub (Array.map (owned_v w) out) 0 no;
            Array.append
              (Array.map
                 (fun v -> if owned_v w v then batch_v w n v else v)
                 (Array.sub out 0 no))
              (totals_out w n shared (Array.sub out no (Array.length shared)))))
  in
  let prog, own =
    (* Traced as the remat's function was: every operation staged. *)
    Trace.trace Staged ~name (Array.map Program.form bargs) mapped
  in
  let r =
    Program.perform
      (Remat
         {
           body = prog;
           args = Array.append bargs own;
           seeds = seeds_out i n shared seeds;
           differentiated;
         })
  in
  Array.append
    (Array.mapi
       (fun k v -> if !laned_out.(k) then lane_v i v else v)
       (Array.sub r 0 no))
    (ends_in i shared (Array.sub r no (Array.length shared)))

(* A call's map is the call of its program's map, as JAX batches a compiled
   call's jaxpr: derived once per derivation at the call's site, traced as a
   compiled function is, and compiled. The map's name enters the derivation only
   where the program names it, so maps of other names share the program
   otherwise. *)
and call ~name ?held i axis n (body : Program.t) args site seeds =
  let la = Array.map (owned_v i) args in
  let bargs =
    Array.map (fun v -> if owned_v i v then batch_v i n v else v) args
  in
  (* Where lanes are held, two more arguments, every lane's: which lanes run and
     each lane's donor, so that the derived program holds them apart as the map
     around it does. *)
  let m = Array.length bargs in
  let extra =
    match held with Some h -> [| h.running; h.donors |] | None -> [||]
  in
  let bargs = Array.append bargs extra in
  let named = List.exists (Program.same_axis axis) (Program.axes site) in
  let forms = Array.map Program.form bargs in
  let no = Array.length body.outputs - Array.length body.adds in
  let shared = sharing i (Array.length body.adds) seeds in
  let made () =
    let laned_out = ref [||] in
    let mapped vs =
      let held =
        Option.map (fun _ -> { running = vs.(m); donors = vs.(m + 1) }) held
      in
      let vs = Array.sub vs 0 m in
      Program.threaded ~lanes:(laned n shared) body (fun running ->
          interpret ~name ~axis ?held n (fun w ->
              let totals = totals_in w shared running in
              let out = Program.eval ~totals body (lanes_of w la vs) in
              laned_out := Array.sub (Array.map (owned_v w) out) 0 no;
              Array.append
                (Array.map
                   (fun v -> if owned_v w v then batch_v w n v else v)
                   (Array.sub out 0 no))
                (totals_out w n shared (Array.sub out no (Array.length shared)))))
    in
    let prog, _ = Trace.trace Jit ~name forms mapped in
    let totals =
      Array.map (fun (Program.Addition (_, f, _)) -> Program.F f) prog.adds
    in
    ((prog, Array.append forms totals), !laned_out, None)
  in
  let lanes = Array.append la (Array.map (fun _ -> true) extra) in
  let derivation = Program.Vmap { lanes; shared; n; named } in
  let d = Program.derive ~map:(axis, n) site derivation forms made in
  let r =
    Program.perform
      (Call
         {
           body = d.program;
           args = bargs;
           site = d.at;
           seeds = seeds_out i n shared seeds;
         })
  in
  Array.append
    (Array.mapi
       (fun k v -> if d.flags.(k) then lane_v i v else v)
       (Array.sub r 0 no))
    (ends_in i shared (Array.sub r no (Array.length shared)))

(* [f]'s results' batches over the lanes of the batches [bs], in a map of this
   one's name, size and held lanes. *)
and map ~name ?held axis n ~shared f bs =
  let ns = Array.length shared in
  interpret ~name ~axis ?held n (fun j ->
      let tail vs = Array.sub vs (Array.length vs - ns) ns in
      let head vs = Array.sub vs 0 (Array.length vs - ns) in
      let lanes =
        Array.append
          (Array.map (fun (Program.V b) -> Program.V (lane j b)) (head bs))
          (totals_in j shared (tail bs))
      in
      let out = f lanes in
      Array.append
        (Array.map (fun (Program.V y) -> Program.V (batch j n y)) (head out))
        (totals_out j n shared (tail out)))

and interpret :
    'a.
    name:string ->
    axis:Program.axis ->
    ?held:held ->
    int ->
    (Prim.interpretation -> 'a) ->
    'a =
 fun ~name ~axis ?held n f ->
  let owner = ref None in
  let effc (type a) (e : a Effect.t) :
      ((a, _) Effect.Deep.continuation -> _) option =
    match (e, !owner) with
    | Program.Construct c, Some i when claims i axis c ->
        Some (answer ~name ?held i axis n c)
    | Program.Within, _ -> Some (fun k -> Effect.Deep.continue k true)
    | Program.Bound a, _ when Program.same_axis a axis ->
        Some (fun k -> Effect.Deep.continue k (Some n))
    (* No differentiation around the map runs a custom rule called inside it:
       the rule may capture lanes, which only its traced call batches. *)
    | Program.Rule _, Some _ -> Some (fun k -> Effect.Deep.continue k None)
    (* A running value around the map is one every lane shares. *)
    | Program.Running (t, f), Some i ->
        Some
          (fun k ->
            Effect.Deep.continue k (Option.map (share i) (Program.running t f)))
    | Program.Replace (t, x), Some i when Option.is_some (Prim.payload i x) ->
        Some
          (fun k ->
            match shared i x with
            | Some r ->
                Effect.perform (Program.Replace (t, r));
                Effect.Deep.continue k ()
            | None ->
                Effect.Deep.discontinue k
                  (Invalid_argument
                     "Rune: a lane's running value replaces one around its map"))
    | _ -> None
  in
  Effect.Deep.match_with
    (fun () ->
      Prim.interpret ~name Values
        (fun i ~by op -> rule n held i ~by op)
        (fun i ->
          owner := Some i;
          f i))
    ()
    { retc = Fun.id; exnc = reraise; effc }

(* A map's sites offer once each, every lane's rows merged. *)
let run ~name ~axis n f =
  Program.gathering ~lanes:true (fun () -> interpret ~name ~axis n f)
