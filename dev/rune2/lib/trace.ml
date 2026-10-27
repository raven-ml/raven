(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program

type kind = Body | Jit

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
      | Body -> (
          match captured st (V x) with
          | Some n -> Nx.Prim.traced i (Nx.Prim.form x) (Var n)
          | None ->
              st.captures <- (V x, st.next) :: st.captures;
              fresh st i (Nx.Prim.form x)))

let rule (type r) st i ~by (op : r Nx.Prim.t) : r =
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

let owned i (V x) =
  match Nx.Prim.owner x with Some j -> j == i | None -> false

(* Whether [i]'s handler answers [c]. A call is left to its default, which
   issues its equations here; a map's operation with no such map around is left
   to its default, which raises. *)
let claims st i = function
  | Call _ -> false
  | Add _ -> true
  | (Lanes (a, _) | Lane_index a) when bound a = None -> false
  | c -> st.kind = Jit || List.exists (owned i) (operands c)

let record st i = function
  | Add (t, x) ->
      let x = operand st i x in
      st.adds <- (Addition (t, Nx.dtype x), V x) :: st.adds;
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
    | Construct c, Some i when claims st i c ->
        Some (fun k -> Effect.Deep.continue k (record st i c))
    | _ -> None
  in
  let reach = match kind with Body -> Nx.Prim.Values | Jit -> Extent in
  let outs =
    Nx.Rng.refusing name (fun () ->
        Effect.Deep.match_with
          (fun () ->
            Nx.Prim.interpret ~name reach
              (fun i ~by op -> rule st i ~by op)
              body)
          ()
          { retc = Fun.id; exnc = raise; effc })
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
