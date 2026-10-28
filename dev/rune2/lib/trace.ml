(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program
module P = Nx_kernel.Prog
module Prim = Nx.Prim

type kind = Body | Jit | Tape

type state = {
  kind : kind;
  name : string;
  mutable next : int;  (** The next variable's number. *)
  mutable eqns : eqn list;  (** Newest first. *)
  mutable captures : (value * int) list;
      (** Each capture and its variable's number, newest first. *)
  mutable adds : (addition * value) list;  (** Newest first. *)
}

let fresh st i form =
  let n = st.next in
  st.next <- n + 1;
  Nx.Prim.traced i form (Var n)

(* The number of [x]'s variable, if [x] is a capture already. *)
let captured st x =
  List.find_map (fun (y, n) -> if same y x then Some n else None) st.captures

(* [x] as an equation of [i]'s reads it: [i]'s variable, a constant, or the
   variable of a capture. *)
let operand (type v s d) st i (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
  match Nx.Prim.owner x with
  | None -> x
  | Some j when j == i -> x
  | Some _ -> (
      match st.kind with
      | Jit ->
          invalid_arg
            (st.name
           ^ ": the function captures a value another transformation traces; \
              pass it as an argument")
      | Body | Tape -> (
          match captured st (V x) with
          | Some n -> Nx.Prim.traced i (Nx.Prim.form x) (Var n)
          | None ->
              st.captures <- (V x, st.next) :: st.captures;
              fresh st i (Nx.Prim.form x)))

let record_op (type r) st i ~by (op : r Nx.Prim.t) : r =
  let op = Nx.Prim.map (fun x -> operand st i x) op in
  let base = st.next and n = ref 0 in
  let r =
    Nx.Prim.results ~by
      (fun k form ->
        n := Int.max !n (k + 1);
        Nx.Prim.traced i form (Var (base + k)))
      op
  in
  st.next <- base + !n;
  st.eqns <- Op { by; op; base } :: st.eqns;
  r

(* Partial evaluation

   The tape tracer records only what reads its variables. A loop's outputs that
   read none of them compute at the call, outward, over the loads they read; the
   others are recorded over theirs, the values they recompute. *)

let owns i x = match Nx.Prim.owner x with Some j -> j == i | None -> false

let load_owned i : 'd Prim.load -> bool = function
  | Plain x -> owns i x
  | Padded { x; _ } -> owns i x

(* Outputs [outs] of a program over the loads they read, the loads [kept]. *)
type part = { prog : P.t; outs : int array; kept : int array }

let part (p : P.t) outs =
  let keep = Scalar.reads p ~outs in
  let kept =
    Array.of_list
      (List.filter (fun j -> keep.(j)) (List.init (Array.length keep) Fun.id))
  in
  { prog = Scalar.prune p ~keep ~outs; outs; kept }

(* A program's outputs split by whether they read a load of [mask]:
   [unknown.(k)] for output [k], and the parts of each side, where it has
   outputs. *)
type plan = {
  unknown : bool array;
  known : part option;
  recorded : part option;
}

let plan (p : P.t) owned =
  let deps = Scalar.depends p owned in
  let unknown = Array.map (fun o -> deps.(o)) (Iarray.to_array p.outs) in
  let side flag =
    let outs =
      List.filter
        (fun k -> unknown.(k) = flag)
        (List.init (Array.length unknown) Fun.id)
    in
    match outs with [] -> None | outs -> Some (part p (Array.of_list outs))
  in
  { unknown; known = side false; recorded = side true }

(* Plans per domain, found by their program and its owned loads, as jvp's are:
   the last one found, then a table emptied when it holds [entries]. *)
type entry = { p : P.t; m : int; plan : plan }

type cache = {
  mutable last : entry option;
  table : (P.t * int, plan) Hashtbl.t;
}

let entries = 4096

let cache =
  Domain.DLS.new_key (fun () -> { last = None; table = Hashtbl.create 64 })

let plan_of i (p : P.t) (loads : 'd Prim.load iarray) =
  let n = Iarray.length loads in
  let owned () = Array.init n (fun j -> load_owned i (Iarray.get loads j)) in
  if n >= Sys.int_size - 1 then plan p (owned ())
  else
    let m = ref 0 in
    for j = n - 1 downto 0 do
      m := (!m lsl 1) lor Bool.to_int (load_owned i (Iarray.get loads j))
    done;
    let m = !m in
    let c = Domain.DLS.get cache in
    match c.last with
    | Some e when e.p == p && e.m = m -> e.plan
    | _ ->
        let plan =
          match Hashtbl.find_opt c.table (p, m) with
          | Some plan -> plan
          | None ->
              if Hashtbl.length c.table >= entries then Hashtbl.reset c.table;
              let plan = plan p (owned ()) in
              Hashtbl.add c.table (p, m) plan;
              plan
        in
        c.last <- Some { p; m; plan };
        plan

let kept part loads =
  Iarray.init (Array.length part.kept) (fun j -> Iarray.get loads part.kept.(j))

(* A map's outputs split in two maps' and how their results give its own. *)
type ('d, 'r) outs_split =
  | O :
      ('d, 'k) Prim.outs * ('d, 'u) Prim.outs * ('k -> 'u -> 'r)
      -> ('d, 'r) outs_split

let rec split_outs : type d r.
    (d, r) Prim.outs -> bool array -> int -> (d, r) outs_split =
 fun outs unknown k ->
  match outs with
  | [] -> O ([], [], fun () () -> ())
  | dt :: rest ->
      let (O (ko, uo, f)) = split_outs rest unknown (k + 1) in
      if unknown.(k) then O (ko, dt :: uo, fun kr (y, ur) -> (y, f kr ur))
      else O (dt :: ko, uo, fun (y, kr) ur -> (y, f kr ur))

(* A reduction's reductions split likewise, each renumbered to its side's
   program by [index]. *)
type ('d, 'r) reds_split =
  | R :
      ('d, 'k) Prim.reductions * ('d, 'u) Prim.reductions * ('k -> 'u -> 'r)
      -> ('d, 'r) reds_split

let output : type d a. (d, a) Prim.reduction -> int = function
  | Monoid (_, k, _) | Logsumexp (k, _) | Moments (k, _) | Arg (_, k, _) -> k

let renumber : type d a. int -> (d, a) Prim.reduction -> (d, a) Prim.reduction =
 fun k -> function
  | Monoid (m, _, dt) -> Monoid (m, k, dt)
  | Logsumexp (_, dt) -> Logsumexp (k, dt)
  | Moments (_, dt) -> Moments (k, dt)
  | Arg (e, _, dt) -> Arg (e, k, dt)

let rec split_reds : type d r.
    (d, r) Prim.reductions -> bool array -> (int -> int) -> (d, r) reds_split =
 fun rs unknown index ->
  match rs with
  | [] -> R ([], [], fun () () -> ())
  | r :: rest ->
      let (R (kr, ur, f)) = split_reds rest unknown index in
      let k = output r in
      let r = renumber (index k) r in
      if unknown.(k) then R (kr, r :: ur, fun kv (y, uv) -> (y, f kv uv))
      else R (r :: kr, ur, fun (y, kv) uv -> (y, f kv uv))

(* The position of each output of [part] among its outputs. *)
let positions n = function
  | None -> Array.make n (-1)
  | Some part ->
      let a = Array.make n (-1) in
      Array.iteri (fun i o -> a.(o) <- i) part.outs;
      a

let map_parts (type d r) st i ~by layout prog (outs : (d, r) Prim.outs)
    (loads : d Prim.load iarray) : r =
  let plan = plan_of i prog loads in
  match (outs, plan.known, plan.recorded) with
  | [ a; b ], Some k, Some u when not plan.unknown.(0) ->
      (* A primal and its tangent, as forward mode makes them: without the
         general split. *)
      let y, () =
        Prim.eval ~by
          (Map { layout; prog = k.prog; outs = [ a ]; loads = kept k loads })
      in
      let t, () =
        record_op st i ~by
          (Map { layout; prog = u.prog; outs = [ b ]; loads = kept u loads })
      in
      (y, (t, ()))
  | _ ->
      let (O (ko, uo, f)) = split_outs outs plan.unknown 0 in
      let known : type k. (d, k) Prim.outs -> k = function
        | [] -> ()
        | _ :: _ as ko ->
            let part = Option.get plan.known in
            Prim.eval ~by
              (Map
                 {
                   layout;
                   prog = part.prog;
                   outs = ko;
                   loads = kept part loads;
                 })
      in
      let recorded : type u. (d, u) Prim.outs -> u = function
        | [] -> ()
        | _ :: _ as uo ->
            let part = Option.get plan.recorded in
            record_op st i ~by
              (Map
                 {
                   layout;
                   prog = part.prog;
                   outs = uo;
                   loads = kept part loads;
                 })
      in
      let kr = known ko in
      f kr (recorded uo)

let reduce_parts (type d r) st i ~by layout axes prog
    (reductions : (d, r) Prim.reductions) (loads : d Prim.load iarray) : r =
  let plan = plan_of i prog loads in
  let n = Array.length plan.unknown in
  let k_at = positions n plan.known and u_at = positions n plan.recorded in
  let index k = if plan.unknown.(k) then u_at.(k) else k_at.(k) in
  let (R (kr, ur, f)) = split_reds reductions plan.unknown index in
  let known : type k. (d, k) Prim.reductions -> k = function
    | [] -> ()
    | _ :: _ as kr ->
        let part = Option.get plan.known in
        Prim.eval ~by
          (Reduce
             {
               layout;
               axes;
               prog = part.prog;
               reductions = kr;
               loads = kept part loads;
             })
  in
  let recorded : type u. (d, u) Prim.reductions -> u = function
    | [] -> ()
    | _ :: _ as ur ->
        let part = Option.get plan.recorded in
        record_op st i ~by
          (Reduce
             {
               layout;
               axes;
               prog = part.prog;
               reductions = ur;
               loads = kept part loads;
             })
  in
  let kv = known kr in
  f kv (recorded ur)

let scan_part (type d r) st i ~by layout axis prog
    (reduction : (d, r) Prim.reduction) (loads : d Prim.load iarray)
    (carry : d Prim.any option) : r =
  let plan = plan_of i prog loads in
  let carried = match carry with Some (Any c) -> owns i c | None -> false in
  if carried || plan.unknown.(output reduction) then
    record_op st i ~by (Scan { layout; axis; prog; reduction; loads; carry })
  else
    let part = Option.get plan.known in
    let at = positions (Array.length plan.unknown) plan.known in
    let reduction = renumber at.(output reduction) reduction in
    Prim.eval ~by
      (Scan
         {
           layout;
           axis;
           prog = part.prog;
           reduction;
           loads = kept part loads;
           carry;
         })

let rule (type r) st i ~by (op : r Nx.Prim.t) : r =
  match (st.kind, op) with
  | Tape, Map { layout; prog; outs; loads } ->
      map_parts st i ~by layout prog outs loads
  | Tape, Reduce { layout; axes; prog; reductions; loads } ->
      reduce_parts st i ~by layout axes prog reductions loads
  | Tape, Scan { layout; axis; prog; reduction; loads; carry } ->
      scan_part st i ~by layout axis prog reduction loads carry
  | _ -> record_op st i ~by op

let owned i (V x) = owns i x

(* Whether [i]'s handler answers [c]. A call is left to its default, which
   issues its equations here; a map's operation with no such map around is left
   to its default, which raises. *)
let claims st i = function
  | Call _ -> false
  | Add (_, x) -> st.kind <> Tape || owns i x
  | (Lanes (a, _) | Lane_index a) when bound a = None -> false
  | c -> st.kind = Jit || List.exists (owned i) (operands c)

let record st i = function
  | Add _ when st.kind = Tape ->
      invalid_arg (st.name ^ ": a tangent is added to a total")
  | Add (t, x) ->
      let x = operand st i x in
      st.adds <- (Addition (t, Nx.Prim.form x), V x) :: st.adds;
      [||]
  | c ->
      let c = map_operands { f = (fun x -> operand st i x) } c in
      let forms = forms c in
      let base = st.next in
      let r = Array.map (fun (F f) -> V (fresh st i f)) forms in
      st.eqns <- Do { c; base } :: st.eqns;
      r

let trace kind ~name inputs f =
  let st =
    {
      kind;
      name;
      next = Array.length inputs;
      eqns = [];
      captures = [];
      adds = [];
    }
  in
  let owner = ref None in
  let body i =
    owner := Some i;
    let params =
      Array.mapi (fun k (F form) -> V (Nx.Prim.traced i form (Var k))) inputs
    in
    let outs = f params in
    Array.map (fun (V y) -> V (operand st i y)) outs
  in
  (* The handler lies outside the interpretation, as its rule does: what it
     computes goes to the interpretations around the tracer. *)
  let effc (type a) (e : a Effect.t) :
      ((a, _) Effect.Deep.continuation -> _) option =
    match (e, !owner) with
    | Construct (Custom _ as c), Some i
      when st.kind = Tape && List.exists (owned i) (operands c) ->
        (* A custom rule over a tape's variables, as a tangent map that applies
           one to its tangents: its primal runs where it was performed, and its
           operations are recorded. *)
        Some (fun k -> Effect.Deep.discontinue k Evaluate)
    | Construct c, Some i when claims st i c ->
        Some (fun k -> Effect.Deep.continue k (record st i c))
    | Staging, Some _ when st.kind = Jit ->
        Some (fun k -> Effect.Deep.continue k true)
    | _ -> None
  in
  let reach = match kind with Body | Tape -> Nx.Prim.Values | Jit -> Extent in
  let traced () =
    Effect.Deep.match_with
      (fun () ->
        Nx.Prim.interpret ~name reach (fun i ~by op -> rule st i ~by op) body)
      ()
      { retc = Fun.id; exnc = raise; effc }
  in
  (* A tape's primal computes at the call, so its draws are the scope's. *)
  let outs =
    match kind with
    | Tape -> traced ()
    | Body | Jit -> Nx.Rng.refusing name traced
  in
  let owner = Option.get !owner in
  let captures = List.rev st.captures and adds = List.rev st.adds in
  let params =
    Array.append
      (Array.init (Array.length inputs) Fun.id)
      (Array.of_list (List.map snd captures))
  in
  let program =
    Program.make ~owner ~params (List.rev st.eqns)
      ~outputs:(Array.append outs (Array.of_list (List.map snd adds)))
      ~adds:(Array.of_list (List.map fst adds))
      ~slots:st.next
  in
  (program, Array.of_list (List.map fst captures))
