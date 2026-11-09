(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Prog = Nx_kernel.Prog

type t = {
  schema : Schema.t;
  held : (string * Expr.packed) list;
  merge : (string * Expr.packed) list;
  finish : (string * Expr.packed) list;
  folds : bool;
}

let row_field = "\000row"
let int64 = Type.int64
let float64 = Type.float64
let float32 = Type.float32
let fixed ty = Expr.Column ty
let lit n = Expr.typed (fixed int64) (Lit (Kind.int, n))
let reduce ty r a = Expr.typed (fixed ty) (Reduce (r, a))
let branch ty c a b = Expr.typed (fixed ty) (If (c, a, b))
let is_null a = Expr.typed (fixed Type.bool) (Is_null a)
let compare_ op a b = Expr.typed (fixed Type.bool) (Compare (op, a, b))
let add a b = Expr.typed (fixed int64) (Int (Add, a, b))

(* [nx_cast into from a] is [a] converted as [Nx.cast] converts it, which is how
   {!Reduce} reads a column in another dtype: it wraps and rounds where a
   checked {!Expr.cast} fails. *)
let nx_cast into from a =
  let (Eval.Dtype src) = Option.get (Eval.dtype from) in
  let (Eval.Dtype dst) = Option.get (Eval.dtype into) in
  let prog =
    Prog.of_exprs
      ~ins:(Iarray.of_list [ Nx.Dtype.Any src ])
      (Iarray.of_list [ Prog.E.(any (cast dst (input 0 src))) ])
  in
  Expr.typed (fixed into) (Nx_map (prog, [ Expr.Packed a ]))

exception Blocks

(* A reduction lowers to fields: a held value on each row and a merge of the
   field over state and held rows, so that a row is a state of one row. [rank]
   is the reduction's place in evaluation order. *)
type field = {
  name : string;
  ty : Type.any;
  held : Expr.packed;
  merge : Expr.packed;
  rank : int;
}

type st = {
  order : Expr.packed list;
  mutable fields : field list;
  mutable moments : (Expr.packed * (Record.t, Expr.row) Expr.t) list;
      (* The state of [var] and [std], one per operand. *)
}

let field st rank ty held merge =
  let name = "\000" ^ string_of_int (List.length st.fields) in
  let merge = Expr.Packed (merge (Expr.read ty name)) in
  let f = { name; ty = Type.Any ty; held = Expr.Packed held; merge; rank } in
  st.fields <- f :: st.fields;
  Expr.read ty name

(* Integers of at most 32 bits sum exactly in [int64] below this many values,
   as {!Reduce.mean} says. *)
let exact_values = 1 lsl 31

let small_int : type a. a Type.t -> bool = function
  | Int8 | Int16 | Int32 | Uint8 | Uint16 | Uint32 -> true
  | _ -> false

(* An [arg_min] or [arg_max]: a state row wins at its own place, and a later
   row at the rows before it plus its place. A group of no value is [at]
   itself, which is null over the blocking aggregate's slot, where the place's
   arithmetic would leave the state's row count under the null. *)
let arg st j ty a r v =
  let extreme = field st j ty a (reduce ty v) in
  let rows = field st j int64 (lit 1) (reduce int64 Sum) in
  let null = Expr.typed (fixed int64) Null in
  let place = branch int64 (is_null a) null (lit 0) in
  field st j int64 place (fun p ->
      let at = reduce int64 r extreme in
      branch int64 (is_null at) at
        (branch int64
           (compare_ `Eq at (lit 0))
           (reduce int64 First p)
           (add (reduce int64 First rows) (add at (lit (-1))))))

let mean st j ty a =
  let sum ty x = field st j ty x (reduce ty Sum) in
  let total = sum float64 (nx_cast float64 ty a) in
  let c = sum int64 (branch int64 (is_null a) (lit 0) (lit 1)) in
  let total =
    if not (small_int ty) then total
    else
      branch float64
        (compare_ `Ge c (lit exact_values))
        total
        (nx_cast float64 int64 (sum int64 (nx_cast int64 ty a)))
  in
  (* A null count leaves the quotient's slot, [total / 0], as a blocking mean
     does; a null branch would put [0.] there. *)
  let n =
    branch float64
      (compare_ `Eq c (lit 0))
      (Expr.typed (fixed float64) Null)
      (nx_cast float64 int64 c)
  in
  Expr.typed (fixed float64) (Float (Div, total, n))

(* [var] and [std] keep one state per operand, a record of a fold's fields
   ({!Reduce.frame}), so that their finish reads its results and the fold has
   no second definition here. A held row is a state of one value: [n] is [0]
   or [1], [c] the value, [down] [1] and the rest [0]. *)
let moments_type =
  Type.record
    [
      ("c", Type.Any float64);
      ("n", Type.Any int64);
      ("m", Type.Any float64);
      ("s", Type.Any float64);
      ("down", Type.Any float64);
      ("var", Type.Any float64);
      ("std", Type.Any float64);
    ]

let flit x = Expr.typed (fixed float64) (Lit (Kind.float, x))

let moments st j name ty a =
  let state =
    let same (Expr.Packed a', _) = Expr.same a a' in
    match List.find_opt same st.moments with
    | Some (_, r) -> r
    | None ->
        let x = nx_cast float64 ty a in
        let n = branch int64 (is_null a) (lit 0) (lit 1) in
        let held =
          Expr.typed (fixed moments_type)
            (Record_of
               Expr.
                 [
                   "c" := x;
                   "n" := n;
                   "m" := flit 0.;
                   "s" := flit 0.;
                   "down" := flit 1.;
                   "var" := flit 0.;
                   "std" := flit 0.;
                 ])
        in
        let r =
          field st j moments_type held (fun r ->
              Expr.typed (fixed moments_type) (Reduce (Moments, r)))
        in
        st.moments <- (Expr.Packed a, r) :: st.moments;
        r
  in
  Expr.typed (fixed float64) (Field (Kind.float, name, state))

let reduction : type a b s r.
    st -> int -> (a, b) Expr.reduction -> (a, s) Expr.t -> (b, r) Expr.t =
 fun st j r a ->
  if not (Expr.row_local a) then raise Blocks;
  let ty = match Expr.typing a with Column ty -> ty | _ -> raise Blocks in
  let plain r = field st j ty a (reduce ty r) in
  let sum ty x = field st j ty x (reduce ty Sum) in
  match r with
  | Count -> sum int64 (branch int64 (is_null a) (lit 0) (lit 1))
  | Sum -> (
      match (ty, Type.kind ty) with
      | Float16, _ -> nx_cast ty float32 (sum float32 (nx_cast float32 ty a))
      | _, Float -> plain Sum
      | _, Int -> sum int64 (nx_cast int64 ty a)
      | _ -> raise Blocks)
  | Mean -> mean st j ty a
  | Min -> plain Min
  | Max -> plain Max
  | First -> plain First
  | Last -> plain Last
  | Only -> plain Only
  | Arg_min -> arg st j ty a Arg_min Min
  | Arg_max -> arg st j ty a Arg_max Max
  | Var -> moments st j "var" ty a
  | Std -> moments st j "std" ty a
  | Median | Quantile _ | N_unique | Collect | Moments | Pivot -> raise Blocks

(* The evaluation order of [st.order] ranks the reductions: [map_node] rebuilds
   operands in no stated order, so {!v} sorts the fields by it. *)
let rec lower : type a s. st -> (a, s) Expr.t -> (a, s) Expr.t =
 fun st e ->
  let rank e =
    List.find_index (fun (Expr.Packed e') -> Expr.same e e') st.order
  in
  (match Expr.typing e with Extension _ -> raise Blocks | _ -> ());
  match Expr.node e with
  | Rows -> field st 0 int64 (lit 1) (reduce int64 Sum)
  | Reduce (r, a) ->
      (* Only to check it: the operand's own nodes are held, as they are. *)
      ignore (lower st a);
      reduction st (Option.get (rank e)) r a
  | Over _ | Window _ | Shift _ | Rank _ | Per_list _ | App _ | Of_option _
  | Const _ | Spread _ ->
      raise Blocks
  | n ->
      Expr.typed (Expr.typing e)
        (Expr.map_node { map = (fun e -> lower st e) } n)

let v ~by outputs schema =
  let order =
    List.concat_map (fun (_, Expr.Packed e) -> Expr.reductions e) outputs
  in
  let st = { order; fields = []; moments = [] } in
  let lowered (n, Expr.Packed e) = (n, Expr.Packed (lower st e)) in
  match List.map lowered outputs with
  | exception Blocks -> None
  | finish
    when List.exists (fun (_, Expr.Packed e) -> Expr.can_fail e) finish ->
      None
  | finish ->
      let fields =
        List.stable_sort
          (fun a b -> Int.compare a.rank b.rank)
          (List.rev st.fields)
      in
      let key k = (k, Option.get (Schema.find schema k)) in
      let keys = List.map key by in
      let read (k, Type.Any ty) = (k, Expr.Packed (Expr.read ty k)) in
      let row = Expr.Packed (reduce int64 First (Expr.read int64 row_field)) in
      Some
        {
          schema =
            Schema.v
              (keys
              @ List.map (fun f -> (f.name, f.ty)) fields
              @ [ (row_field, Type.Any int64) ]);
          held =
            List.map read keys @ List.map (fun f -> (f.name, f.held)) fields;
          merge =
            List.map (fun f -> (f.name, f.merge)) fields @ [ (row_field, row) ];
          finish = List.map read keys @ finish;
          folds = st.moments <> [];
        }
