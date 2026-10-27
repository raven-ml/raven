(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx_kernel.Prog
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Prim = Nx.Prim

type ('v, 's, 'd) Prim.payload +=
  | Dual : {
      primal : ('v, 's, 'd) Nx.t;
      tangent : ('v, 's, 'd) Nx.t;
      fresh : bool;
    }
      -> ('v, 's, 'd) Prim.payload

let primal i x =
  match Prim.payload i x with Some (Dual d) -> d.primal | _ -> x

let tangent i x =
  match Prim.payload i x with Some (Dual d) -> Some d.tangent | _ -> None

(* A value of [i] whose primal [y] an [Nx.item] reads, as an eager derivative
   branches on it. [fresh] where [t] is its own memory, in C order. *)
let dual i ~fresh y t =
  Prim.twinned i y (Dual { primal = y; tangent = t; fresh })

let seed i x t = dual i ~fresh:false x t

(* A value of the dtype [dt], checked. *)
let expect (type v s d) (dt : (v, s) D.t) (Prim.Any x : d Prim.any) :
    (v, s, d) Nx.t =
  match D.equal_witness (Nx.dtype x) dt with
  | Some Type.Equal -> x
  | None -> invalid_arg "Rune: a tangent of another dtype"

(* Loads *)

let primal_load i : 'd Prim.load -> 'd Prim.load = function
  | Plain x -> Plain (primal i x)
  | Padded { x; fill; pad } -> Padded { x = primal i x; fill; pad }

let tangent_load i : 'd Prim.load -> 'd Prim.load option = function
  | Plain x -> Option.map (fun t -> Prim.Plain t) (tangent i x)
  | Padded { x; pad; _ } ->
      Option.map
        (fun t -> Prim.Padded { x = t; fill = D.zero (Nx.dtype t); pad })
        (tangent i x)

let load_has i : 'd Prim.load -> bool = function
  | Plain x -> Option.is_some (tangent i x)
  | Padded { x; _ } -> Option.is_some (tangent i x)

let load_dtype : 'd Prim.load -> D.any = function
  | Plain x -> D.Any (Nx.dtype x)
  | Padded { x; _ } -> D.Any (Nx.dtype x)

(* The loads of a loop over primals, then tangents. *)
let jvp_loads i loads =
  let ps = Iarray.to_list (Iarray.map (primal_load i) loads) in
  let ts = List.filter_map (tangent_load i) (Iarray.to_list loads) in
  Iarray.of_list (ps @ ts)

(* Which loads have a tangent. *)
let has i loads =
  Array.init (Iarray.length loads) (fun j -> load_has i (Iarray.get loads j))

(* The same, bit [j] for load [j], of fewer loads than an [int] has bits. *)
let mask i loads =
  let m = ref 0 in
  for j = Iarray.length loads - 1 downto 0 do
    m := (!m lsl 1) lor Bool.to_int (load_has i (Iarray.get loads j))
  done;
  !m

let flags m n = Array.init n (fun j -> (m lsr j) land 1 = 1)
let masked loads = Iarray.length loads < Sys.int_size - 1

(* Outputs of dtypes known at run time *)

type 'd outs = Outs : ('d, 'r) Prim.outs * ('r -> 'd Prim.any list) -> 'd outs

let rec outs_of : type d. D.any list -> d outs = function
  | [] -> Outs ([], fun () -> [])
  | D.Any dt :: rest ->
      let (Outs (o, read)) = outs_of rest in
      Outs (dt :: o, fun (y, r) -> Prim.Any y :: read r)

(* [p]'s outputs at every index of [layout] over [loads], computed by maps of
   one node each where one map of [p] would take more operands than every
   library takes. *)
let run_map (type d) ~by layout (p : P.t) (loads : d Prim.load iarray) :
    d Prim.any array =
  let dts k = P.dtype p (Iarray.get p.outs k) in
  let fused =
    Iarray.length loads + Iarray.length p.outs <= Nx_kernel.loop_width
  in
  if fused then
    let (Outs (outs, read)) = outs_of (List.init (Iarray.length p.outs) dts) in
    Array.of_list (read (Prim.eval ~by (Map { layout; prog = p; outs; loads })))
  else
    let one (type v s) (dt : (v, s) D.t) node (loads : d Prim.load list) :
        d Prim.any =
      let ins = Iarray.of_list (List.map load_dtype loads) in
      let prog = P.of_node ~ins node in
      let loads = Iarray.of_list loads in
      let y, () = Prim.eval ~by (Map { layout; prog; outs = [ dt ]; loads }) in
      Any y
    in
    let vals = Array.make (Iarray.length p.nodes) (None : d Prim.any option) in
    let get k = Option.get vals.(k) in
    let load k = match get k with Prim.Any x -> Prim.Plain x in
    Iarray.iteri
      (fun k n ->
        let (D.Any dt) = P.dtype p k in
        let v : d Prim.any =
          match (n : P.node) with
          | In j ->
              (* A copy, so that an output is fresh memory. *)
              one dt (Op1 (Copy, D.Any dt, 0)) [ Iarray.get loads j ]
          | Coord _ | Const _ -> one dt n []
          | Op1 (k, d, a) -> one dt (Op1 (k, d, 0)) [ load a ]
          | Op2 (k, a, c) -> one dt (Op2 (k, 0, 1)) [ load a; load c ]
          | Op3 (k, a, c, e) ->
              one dt (Op3 (k, 0, 1, 2)) [ load a; load c; load e ]
        in
        vals.(k) <- Some v)
      p.nodes;
    Array.map get (Iarray.to_array p.outs)

(* Maps *)

(* A map's derivative: its program's primal and tangent outputs, each output
   that has a tangent followed by it, and which outputs have one. *)
type plan = { prog : P.t; flags : bool array; any : bool }

let plan prog tangents =
  let b = Scalar.start prog ~tangents ~extra:[||] in
  let m = Scalar.outputs b in
  let flags = Array.init m (fun k -> Scalar.tangent b k <> None) in
  let outs =
    List.concat
      (List.init m (fun k ->
           match Scalar.tangent b k with
           | Some t -> [ Scalar.value b k; t ]
           | None -> [ Scalar.value b k ]))
  in
  {
    prog = Scalar.finish b ~outs:(Array.of_list outs);
    flags;
    any = Array.exists Fun.id flags;
  }

(* Plans per domain, found by their program and its loads' tangents: the last
   one found, which a chain of one kind finds again, then a table, emptied when
   it holds [entries]. nx's functions build each kind's one-node program
   once. *)
type entry = { p : P.t; m : int; plan : plan }

type cache = {
  mutable last : entry option;
  table : (P.t * int, plan) Hashtbl.t;
}

let entries = 4096

let cache =
  Domain.DLS.new_key (fun () -> { last = None; table = Hashtbl.create 64 })

let plan_of (prog : P.t) m =
  let c = Domain.DLS.get cache in
  match c.last with
  | Some e when e.p == prog && e.m = m -> e.plan
  | _ ->
      let plan =
        match Hashtbl.find_opt c.table (prog, m) with
        | Some plan -> plan
        | None ->
            if Hashtbl.length c.table >= entries then Hashtbl.reset c.table;
            let plan = plan prog (flags m (Iarray.length prog.ins)) in
            Hashtbl.add c.table (prog, m) plan;
            plan
      in
      c.last <- Some { p = prog; m; plan };
      plan

(* The outputs of a map's derivative and how its result gives the map's, each
   output with a tangent a dual. *)
type ('d, 'r) jouts = J : ('d, 'q) Prim.outs * ('q -> 'r) -> ('d, 'r) jouts

let rec jouts : type d r.
    Prim.interpretation -> (d, r) Prim.outs -> bool array -> int -> (d, r) jouts
    =
 fun i outs flags k ->
  match outs with
  | [] -> J ([], fun () -> ())
  | dt :: rest ->
      let (J (o, f)) = jouts i rest flags (k + 1) in
      if flags.(k) then
        J (dt :: dt :: o, fun (y, (t, q)) -> (dual i ~fresh:true y t, f q))
      else J (dt :: o, fun (y, q) -> (y, f q))

(* The map's result from [vals], its derivative's outputs in order. *)
let rec of_values : type d r.
    Prim.interpretation ->
    (d, r) Prim.outs ->
    bool array ->
    d Prim.any array ->
    int ->
    int ->
    r =
 fun i outs flags vals k at ->
  match outs with
  | [] -> ()
  | dt :: rest ->
      let y = expect dt vals.(at) in
      if flags.(k) then
        let t = expect dt vals.(at + 1) in
        (dual i ~fresh:true y t, of_values i rest flags vals (k + 1) (at + 2))
      else (y, of_values i rest flags vals (k + 1) (at + 1))

let map_rule (type d r) i ~by layout prog (outs : (d, r) Prim.outs)
    (loads : d Prim.load iarray) : r =
  let p =
    if masked loads then plan_of prog (mask i loads)
    else plan prog (has i loads)
  in
  if not p.any then
    Prim.eval ~by
      (Map { layout; prog; outs; loads = Iarray.map (primal_load i) loads })
  else
    let jl = jvp_loads i loads in
    if Iarray.length jl + Iarray.length p.prog.outs > Nx_kernel.loop_width
    then of_values i outs p.flags (run_map ~by layout p.prog jl) 0 0
    else
      match outs with
      | [ dt ] ->
          let y, (t, ()) =
            Prim.eval ~by
              (Map { layout; prog = p.prog; outs = [ dt; dt ]; loads = jl })
          in
          (dual i ~fresh:true y t, ())
      | _ ->
          let (J (o, f)) = jouts i outs p.flags 0 in
          f
            (Prim.eval ~by
               (Map { layout; prog = p.prog; outs = o; loads = jl }))

(* Complex values and their parts' pairs: bitcasts between them are linear. *)
let same_coordinates (D.Any a) (D.Any b) =
  match (a, b) with
  | D.Complex64, D.Float32 | D.Float32, D.Complex64 -> true
  | D.Complex128, D.Float64 | D.Float64, D.Complex128 -> true
  | _ -> false

let rule (type r) name i ~by (op : r Prim.t) : r =
  (* [x]'s dual from [f] of its primal and of its tangent. *)
  let both ~fresh f x =
    let y = f (primal i x) in
    match tangent i x with Some t -> dual i ~fresh y (f t) | None -> y
  in
  match op with
  | Map { layout; prog; outs; loads } -> map_rule i ~by layout prog outs loads
  | Assemble a ->
      let assemble fill pieces =
        Prim.eval ~by (Assemble { a with fill; pieces })
      in
      let y =
        assemble a.fill (List.map (fun (r, x) -> (r, primal i x)) a.pieces)
      in
      let piece (r, x) =
        match tangent i x with
        | Some t -> (r, t)
        | None -> (r, Nx.zeros_like (primal i x))
      in
      dual i ~fresh:true y (assemble (D.zero a.dtype) (List.map piece a.pieces))
  | Copy x -> both ~fresh:true (fun x -> Prim.eval ~by (Copy x)) x
  | Move (mv, x) -> both ~fresh:false (fun x -> Prim.eval ~by (Move (mv, x))) x
  | Bitcast (dt, x) ->
      if same_coordinates (D.Any dt) (D.Any (Nx.dtype x)) then
        both ~fresh:false (fun x -> Prim.eval ~by (Bitcast (dt, x))) x
      else Prim.eval ~by (Bitcast (dt, primal i x))
  | Place (q, x) -> (
      let y = Prim.eval ~by (Place (q, primal i x)) in
      match tangent i x with
      | Some t -> dual i ~fresh:false y (Prim.eval ~by (Place (q, t)))
      | None -> y)
  | Check _ -> Prim.eval ~by (Prim.map (fun x -> primal i x) op)
  | Reduce _ | Scan _ | Gather _ | Scatter _ | Sort _ | Fold _ | Contract _
  | Fft _ | Linalg _ ->
      invalid_arg
        (Printf.sprintf "%s: %s has no derivative yet" name (Prim.name op))

(* Constructs *)

let owned i (Program.V x) =
  match Prim.owner x with Some j -> j == i | None -> false

(* A construct over duals: [Detach] and an addition are performed outward on the
   primal, so that every differentiation around detaches the value and a total
   receives the primal. Loops, calls and maps have no rule of their own yet:
   they run where they were performed, inside the interpretation, which receives
   their programs' operations. *)
let answer i (c : Program.construct) k =
  match c with
  | Detach x -> Effect.Deep.continue k (Program.perform (Detach (primal i x)))
  | Add (t, x) ->
      ignore (Program.perform (Add (t, primal i x)));
      Effect.Deep.continue k [||]
  | Lanes _ | Lane_index _ | Scan _ | Iterate _ | Call _ ->
      Effect.Deep.discontinue k Program.Evaluate

let run ~name f =
  let owner = ref None in
  let effc (type a) (e : a Effect.t) :
      ((a, _) Effect.Deep.continuation -> _) option =
    match (e, !owner) with
    | Program.Construct c, Some i
      when List.exists (owned i) (Program.operands c) ->
        Some (answer i c)
    | _ -> None
  in
  Effect.Deep.match_with
    (fun () ->
      Prim.interpret ~name Values
        (fun i ~by op -> rule name i ~by op)
        (fun i ->
          owner := Some i;
          f i))
    ()
    { retc = Fun.id; exnc = raise; effc }

let tangent i x =
  match Prim.payload i x with
  | Some (Dual d) -> Some (if d.fresh then d.tangent else Nx.copy d.tangent)
  | _ -> None
