(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next

type iop = Add | Sub | Mul | Div | Mod
type fop = Fadd | Fsub | Fmul | Fdiv
type cmp = [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ]

type 'a expr =
  | Col : 'a Type.t * string -> 'a expr
  | Lit : 'a Type.t * 'a -> 'a expr
  | Null : 'a Type.t -> 'a expr
  | Int : iop * int expr * int expr -> int expr
  | Float : fop * float expr * float expr -> float expr
  | Cmp : cmp * 'a expr * 'a expr -> bool expr
  | And : bool expr * bool expr -> bool expr
  | Or : bool expr * bool expr -> bool expr
  | Not : bool expr -> bool expr
  | If : bool expr * 'a expr * 'a expr -> 'a expr
  | Is_null : 'a expr -> bool expr
  | Coalesce : 'a expr list -> 'a expr
  | Is_in : 'a list * 'a expr -> bool expr
  | Store : 'a Type.t * 'a expr -> 'a expr
  | Map : int Type.t * ('a -> int) * 'a expr -> int expr
  | Bind : int Type.t * ('a option -> int option) * 'a expr -> int expr

type out = Out : string * 'a expr -> out | Keep of string list

type plan =
  | Table of Talon_next.t
  | Select of out list * plan
  | Derive of out list * plan
  | Filter of bool expr * plan
  | Slice of { offset : int; length : int; plan : plan }
  | Append of plan * plan

let literal : type a s. a Type.t -> (a -> (a, s) Expr.t) option =
 fun ty ->
  let k = Type.kind ty in
  let is k' = Kind.provably_equal k k' in
  match (is Kind.int, is Kind.float, is Kind.bool, is Kind.string) with
  | Some Equal, _, _, _ -> Some Expr.int
  | _, Some Equal, _, _ -> Some Expr.float
  | _, _, Some Equal, _ -> Some Expr.bool
  | _, _, _, Some Equal -> Some Expr.string
  | None, None, None, None -> (
      match (is Kind.date, is Kind.instant, is Kind.span) with
      | Some Equal, _, _ -> Some Expr.date
      | _, Some Equal, _ -> Some Expr.instant
      | _, _, Some Equal -> Some Expr.span
      | None, None, None -> None)

let rec type_of : type a. a expr -> a Type.t = function
  | Col (ty, _) | Lit (ty, _) | Null ty | Store (ty, _) -> ty
  | Map (ty, _, _) -> ty
  | Bind (ty, _, _) -> ty
  | Int (_, a, b) -> common a b
  | Float (_, a, b) -> common a b
  | If (_, a, b) -> common a b
  | Coalesce es -> Option.get (Type.common (List.map type_of es))
  | Cmp _ -> Type.bool
  | And _ -> Type.bool
  | Or _ -> Type.bool
  | Not _ -> Type.bool
  | Is_null _ -> Type.bool
  | Is_in _ -> Type.bool

(* [common a b] is the type at which [a] and [b] meet. *)
and common : type a. a expr -> a expr -> a Type.t =
 fun a b -> Option.get (Type.common [ type_of a; type_of b ])

(* Translation *)

let rec expr : type a. a expr -> (a, Expr.row) Expr.t = function
  | Col (ty, n) -> Col.v (Type.kind ty) n
  | Lit (ty, v) -> (Option.get (literal ty)) v
  | Null _ -> Expr.null
  | Int (op, a, b) ->
      let op =
        match op with
        | Add -> Expr.( + )
        | Sub -> Expr.( - )
        | Mul -> Expr.( * )
        | Div -> Expr.( / )
        | Mod -> Expr.( mod )
      in
      op (expr a) (expr b)
  | Float (op, a, b) ->
      let op =
        match op with
        | Fadd -> Expr.( +. )
        | Fsub -> Expr.( -. )
        | Fmul -> Expr.( *. )
        | Fdiv -> Expr.( /. )
      in
      op (expr a) (expr b)
  | Cmp (op, a, b) ->
      let op =
        match op with
        | `Eq -> Expr.( = )
        | `Ne -> Expr.( <> )
        | `Lt -> Expr.( < )
        | `Le -> Expr.( <= )
        | `Gt -> Expr.( > )
        | `Ge -> Expr.( >= )
      in
      op (expr a) (expr b)
  | And (a, b) -> Expr.(expr a && expr b)
  | Or (a, b) -> Expr.(expr a || expr b)
  | Not a -> Expr.not (expr a)
  | If (c, a, b) -> Expr.if_ (expr c) (expr a) (expr b)
  | Is_null a -> Expr.is_null (expr a)
  | Coalesce es -> Expr.coalesce (List.map expr es)
  | Is_in (vs, a) -> Expr.is_in vs (expr a)
  | Store (ty, a) -> Expr.store ty (expr a)
  | Map (ty, f, a) -> Expr.(store ty (const f $ expr a))
  | Bind (ty, f, a) -> Expr.(store ty (of_option (const f $ option (expr a))))

let out = function
  | Out (n, e) -> Expr.(n := expr e)
  | Keep ns -> Expr.keep (Sel.names ns)

let rec query = function
  | Table t -> Query.of_table t
  | Select (os, p) -> Query.select (List.map out os) (query p)
  | Derive (os, p) -> Query.derive (List.map out os) (query p)
  | Filter (e, p) -> Query.filter (expr e) (query p)
  | Slice { offset; length; plan } -> Query.slice ~offset ~length (query plan)
  | Append (p, rest) -> Query.append (query rest) (query p)

(* [derived cs outs] is the columns [cs] with [outs] in place of those of their
   names, then the others. *)
let derived cs outs =
  List.map
    (fun (n, c) -> (n, Option.value ~default:c (List.assoc_opt n outs)))
    cs
  @ List.filter (fun (n, _) -> not (List.mem_assoc n cs)) outs

let rec schema = function
  | Table t -> Schema.columns (Talon_next.schema t)
  | Select (os, p) -> List.concat_map (out_schema (schema p)) os
  | Derive (os, p) ->
      let s = schema p in
      derived s (List.concat_map (out_schema s) os)
  | Filter (_, p) | Slice { plan = p; _ } | Append (p, _) -> schema p

and out_schema s = function
  | Out (n, e) -> [ (n, Type.Any (type_of e)) ]
  | Keep ns -> List.map (fun n -> (n, List.assoc n s)) ns

(* Evaluation *)

type cell = Cell : 'a Type.t * 'a option -> cell
type row = (string * cell) list

let read : type a. a Type.t -> cell -> a option =
 fun ty (Cell (ty', v)) ->
  match Kind.provably_equal (Type.kind ty') (Type.kind ty) with
  | Some Equal -> v
  | None -> invalid_arg "Reference: a cell of another kind"

(* [stored ty v] is the value [v] has once stored as [ty]. *)
let stored : type a. a Type.t -> a -> a =
 fun ty v ->
  match ty with
  | Float32 -> Int32.float_of_bits (Int32.bits_of_float v)
  | _ -> v

let signed bits v =
  let m = 1 lsl bits in
  let w = v land (m - 1) in
  if w >= m / 2 then w - m else w

(* OCaml's integers wrap modulo 2{^ 63}, which keeps the low 32 bits exact. *)
let wrap : int Type.t -> int -> int =
 fun ty v ->
  match ty with
  | Int8 -> signed 8 v
  | Int16 -> signed 16 v
  | Int32 -> signed 32 v
  | Uint8 -> v land 0xff
  | Uint16 -> v land 0xffff
  | Uint32 -> v land 0xffff_ffff
  | _ -> invalid_arg "Reference: arithmetic wider than 32 bits"

let int ty op x y =
  match op with
  | Add -> Some (wrap ty (x + y))
  | Sub -> Some (wrap ty (x - y))
  | Mul -> Some (wrap ty (x * y))
  | (Div | Mod) when y = 0 -> None
  | Div -> Some (wrap ty (x / y))
  | Mod -> Some (wrap ty (x mod y))

let float ty op x y =
  let r =
    match op with
    | Fadd -> x +. y
    | Fsub -> x -. y
    | Fmul -> x *. y
    | Fdiv -> x /. y
  in
  stored ty r

let compare (op : cmp) c =
  match op with
  | `Eq -> c = 0
  | `Ne -> c <> 0
  | `Lt -> c < 0
  | `Le -> c <= 0
  | `Gt -> c > 0
  | `Ge -> c >= 0

(* [Data reason] is a failure of an expression, at the row being evaluated. *)
exception Data of string

let held ty v =
  if Type.holds ty v then v
  else raise (Data (Format.asprintf "%a does not hold %d" Type.pp ty v))

let rec eval : type a. row -> a expr -> a option =
 fun row e ->
  let both f a b =
    let x = eval row a in
    match (x, eval row b) with Some x, Some y -> f x y | _ -> None
  in
  match e with
  | Col (ty, n) -> read ty (List.assoc n row)
  | Lit (ty, v) -> Some (stored ty v)
  | Null _ -> None
  | Int (op, a, b) -> both (int (type_of e) op) a b
  | Float (op, a, b) -> both (fun x y -> Some (float (type_of e) op x y)) a b
  | Cmp (op, a, b) ->
      both
        (fun x y -> Some (compare op (Type.compare_value (common a b) x y)))
        a b
  | And (a, b) -> (
      let x = eval row a in
      match (x, eval row b) with
      | Some false, _ | _, Some false -> Some false
      | Some true, Some true -> Some true
      | _ -> None)
  | Or (a, b) -> (
      let x = eval row a in
      match (x, eval row b) with
      | Some true, _ | _, Some true -> Some true
      | Some false, Some false -> Some false
      | _ -> None)
  | Not a -> Option.map not (eval row a)
  | If (c, a, b) ->
      let c = eval row c in
      let a = eval row a in
      let b = eval row b in
      if c = Some true then a else b
  | Is_null a -> Some (Option.is_none (eval row a))
  | Coalesce es -> List.find_map Fun.id (List.map (eval row) es)
  | Store (_, a) -> eval row a
  | Is_in (vs, a) ->
      let ty = type_of a in
      let same x v = Type.compare_value ty x (stored ty v) = 0 in
      Some
        (Option.fold ~none:false
           ~some:(fun x -> List.exists (same x) vs)
           (eval row a))
  | Map (ty, f, a) -> Option.map (fun x -> held ty (f x)) (eval row a)
  | Bind (ty, f, a) -> Option.map (held ty) (f (eval row a))

let out_cells row = function
  | Out (n, e) -> [ (n, Cell (type_of e, eval row e)) ]
  | Keep ns -> List.map (fun n -> (n, List.assoc n row)) ns

type column = Column : 'a Type.t * 'a option array -> column

(* An extension's values are read as its storage's. *)
let storage : type a. a Type.t -> Type.any = function
  | Ext { storage; _ } -> Any storage
  | ty -> Any ty

let decode c =
  let (Type.Any st) = match Column.type_ c with Any ty -> storage ty in
  let c = Result.get_ok (Column.of_layout (Any st) (Column.layout c)) in
  Column (st, Column.options (Type.kind st) c)

let table_rows t =
  let column n =
    let (Column (ty, vs)) = decode (column t n) in
    Array.map (fun v -> (n, Cell (ty, v))) vs
  in
  let columns =
    List.map column (List.map fst (Schema.columns (Talon_next.schema t)))
  in
  List.init (rows t) (fun i -> List.map (fun c -> c.(i)) columns)

(* [Failed (row, reason)] is a step's failure at its input's row [row]. *)
exception Failed of int * string

(* [step f rs] is [f] on each row of [rs], as one step. *)
let step f rs =
  Seq.mapi
    (fun i r ->
      match f r with v -> v | exception Data why -> raise (Failed (i, why)))
    rs

let rec rows : plan -> row Seq.t = function
  | Table t -> List.to_seq (table_rows t)
  | Select (os, p) -> step (fun r -> List.concat_map (out_cells r) os) (rows p)
  | Derive (os, p) ->
      step (fun r -> derived r (List.concat_map (out_cells r) os)) (rows p)
  | Filter (e, p) ->
      Seq.filter_map Fun.id
        (step (fun r -> if eval r e = Some true then Some r else None) (rows p))
  | Slice { offset; length; plan } when offset >= 0 ->
      let stop =
        if length > max_int - offset then max_int else offset + length
      in
      Seq.drop offset (Seq.take stop (rows plan))
  | Slice { offset; length; plan } ->
      let rs = List.of_seq (rows plan) in
      let start = List.length rs + offset in
      List.to_seq
        (List.filteri (fun i _ -> i >= start && i - start < length) rs)
  | Append (p, rest) -> Seq.append (rows p) (rows rest)

let run p =
  match List.of_seq (rows p) with
  | exception Failed (row, why) -> Error (row, why)
  | rs ->
      let column (n, Type.Any ty) =
        let (Type.Any ty) = storage ty in
        let vs = List.map (fun r -> read ty (List.assoc n r)) rs in
        (n, Column (ty, Array.of_list vs))
      in
      Ok (List.map column (schema p))

let values e p =
  let null = "the value is null; read it through Expr.option" in
  let value r = match eval r e with Some v -> v | None -> raise (Data null) in
  match Array.of_seq (step value (rows p)) with
  | vs -> Ok vs
  | exception Failed (row, why) -> Error (row, why)
