(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next

type iop = Add | Sub | Mul | Div | Mod
type fop = Fadd | Fsub | Fmul | Fdiv
type cmp = [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ]

type (_, _) reduction =
  | Count : ('a, int) reduction
  | Sum : (int, int) reduction
  | Mean : (int, float) reduction
  | Min : ('a, 'a) reduction
  | Max : ('a, 'a) reduction
  | First : ('a, 'a) reduction
  | Last : ('a, 'a) reduction
  | Only : ('a, 'a) reduction
  | Median : ('a, float) reduction
  | Quantile : float -> ('a, float) reduction
  | N_unique : ('a, int) reduction
  | Arg_min : ('a, int) reduction
  | Arg_max : ('a, int) reduction

type key = { name : string; desc : bool; nulls_first : bool }

type ('a, 's) term =
  | Col : 'a Type.t * string -> ('a, Expr.row) term
  | Lit : 'a Type.t * 'a -> ('a, 's) term
  | Null : 'a Type.t -> ('a, 's) term
  | Int : iop * (int, 's) term * (int, 's) term -> (int, 's) term
  | Float : fop * (float, 's) term * (float, 's) term -> (float, 's) term
  | Cmp : cmp * ('a, 's) term * ('a, 's) term -> (bool, 's) term
  | And : (bool, 's) term * (bool, 's) term -> (bool, 's) term
  | Or : (bool, 's) term * (bool, 's) term -> (bool, 's) term
  | Not : (bool, 's) term -> (bool, 's) term
  | If : (bool, 's) term * ('a, 's) term * ('a, 's) term -> ('a, 's) term
  | Is_null : ('a, 's) term -> (bool, 's) term
  | Coalesce : ('a, 's) term list -> ('a, 's) term
  | Is_in : 'a list * ('a, 's) term -> (bool, 's) term
  | Store : 'a Type.t * ('a, 's) term -> ('a, 's) term
  | Map : int Type.t * ('a -> int) * ('a, 's) term -> (int, 's) term
  | Bind :
      int Type.t * ('a option -> int option) * ('a, 's) term
      -> (int, 's) term
  | Rows : (int, Expr.agg) term
  | Reduce : ('a, 'b) reduction * ('a, Expr.row) term -> ('b, Expr.agg) term
  | Over : string list * key list * ('a, 's) term -> ('a, Expr.row) term
  | Shift : int * ('a, Expr.row) term -> ('a, Expr.row) term
  | Rank : ('a, Expr.row) term -> (int, Expr.row) term

type 'a expr = ('a, Expr.row) term

type 's out =
  | Out : string * ('a, 's) term -> 's out
  | Keep : string list -> Expr.row out

type on = Keys of (string * string) list | Position | All

type plan =
  | Table of Talon_next.t
  | Select of Expr.row out list * plan
  | Derive of Expr.row out list * plan
  | Filter of bool expr * plan
  | Slice of { offset : int; length : int; plan : plan }
  | Append of plan * plan
  | Aggregate of string list * Expr.agg out list * plan
  | Join of {
      kind : Join.kind;
      each_left : Join.count;
      each_right : Join.count;
      on : on;
      left : plan;
      right : plan;
    }

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

let rec type_of : type a s. (a, s) term -> a Type.t = function
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
  | Rows -> Type.int64
  | Rank _ -> Type.int64
  | Over (_, _, a) -> type_of a
  | Shift (_, a) -> type_of a
  | Reduce (r, a) -> reduced r (type_of a)

and reduced : type a b. (a, b) reduction -> a Type.t -> b Type.t =
 fun r ty ->
  match r with
  | Count -> Type.int64
  | Sum -> Type.int64
  | N_unique -> Type.int64
  | Arg_min -> Type.int64
  | Arg_max -> Type.int64
  | Mean -> Type.float64
  | Median -> Type.float64
  | Quantile _ -> Type.float64
  | Min -> ty
  | Max -> ty
  | First -> ty
  | Last -> ty
  | Only -> ty

(* [common a b] is the type at which [a] and [b] meet. *)
and common : type a s. (a, s) term -> (a, s) term -> a Type.t =
 fun a b -> Option.get (Type.common [ type_of a; type_of b ])

(* Translation *)

let order { name; desc; nulls_first } =
  let k = if desc then Order.desc name else Order.asc name in
  if nulls_first then Order.nulls_first k else k

let rec expr : type a s. (a, s) term -> (a, s) Expr.t = function
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
  | Rows -> Expr.rows
  | Over (by, keys, a) -> Expr.over ~by ~order:(List.map order keys) (expr a)
  | Shift (n, a) -> Expr.shift n (expr a)
  | Rank a -> Expr.rank (expr a)
  | Reduce (r, a) -> reduction r (expr a)

and reduction : type a b.
    (a, b) reduction -> (a, Expr.row) Expr.t -> (b, Expr.agg) Expr.t =
 fun r a ->
  match r with
  | Count -> Expr.count a
  | Sum -> Expr.sum a
  | Mean -> Expr.mean a
  | Min -> Expr.min a
  | Max -> Expr.max a
  | First -> Expr.first a
  | Last -> Expr.last a
  | Only -> Expr.only a
  | Median -> Expr.median a
  | Quantile p -> Expr.quantile p a
  | N_unique -> Expr.n_unique a
  | Arg_min -> Expr.arg_min a
  | Arg_max -> Expr.arg_max a

let out : type s. s out -> s Expr.out = function
  | Out (n, e) -> Expr.(n := expr e)
  | Keep ns -> Expr.keep (Sel.names ns)

let cond = function
  | Keys ks -> List.fold_left (fun c (l, r) -> Join.(c && eq l r)) Join.all ks
  | Position -> Join.position
  | All -> Join.all

let keys = function Keys ks -> ks | Position | All -> []

let rec query = function
  | Table t -> Query.of_table t
  | Select (os, p) -> Query.select (List.map out os) (query p)
  | Derive (os, p) -> Query.derive (List.map out os) (query p)
  | Filter (e, p) -> Query.filter (expr e) (query p)
  | Slice { offset; length; plan } -> Query.slice ~offset ~length (query plan)
  | Append (p, rest) -> Query.append (query rest) (query p)
  | Aggregate (by, os, p) -> Query.aggregate ~by (List.map out os) (query p)
  | Join { kind; each_left; each_right; on; left; right } ->
      Query.join ~kind ~each_left ~each_right ~on:(cond on) (query right)
        (query left)

(* [derived cs outs] is the columns [cs] with [outs] in place of those of their
   names, then the others. *)
let derived cs outs =
  List.map
    (fun (n, c) -> (n, Option.value ~default:c (List.assoc_opt n outs)))
    cs
  @ List.filter (fun (n, _) -> not (List.mem_assoc n cs)) outs

(* [meet a b] is the type at which a left key of type [a] meets a right key of
   type [b]. *)
let meet (Type.Any a as t) (Type.Any b) =
  if Type.equal a b then t
  else
    match Kind.provably_equal (Type.kind b) (Type.kind a) with
    | Some Equal -> Type.Any (Option.get (Type.common [ a; b ]))
    | None -> invalid_arg "Reference: keys that do not meet"

(* [unkeyed on rs] is the right columns [rs] but the keys of [on]. *)
let unkeyed on rs =
  List.filter
    (fun (n, _) -> not (List.exists (fun (_, r) -> String.equal r n) (keys on)))
    rs

let rec schema = function
  | Table t -> Schema.columns (Talon_next.schema t)
  | Select (os, p) -> List.concat_map (out_schema (schema p)) os
  | Derive (os, p) ->
      let s = schema p in
      derived s (List.concat_map (out_schema s) os)
  | Filter (_, p) | Slice { plan = p; _ } | Append (p, _) -> schema p
  | Aggregate (by, os, p) ->
      let s = schema p in
      List.map (fun n -> (n, List.assoc n s)) by
      @ List.concat_map (out_schema s) os
  | Join { kind; on; left; right; _ } -> (
      let ls = schema left and rs = schema right in
      let key (n, t) =
        match List.assoc_opt n (keys on) with
        | Some r -> (n, meet t (List.assoc r rs))
        | None -> (n, t)
      in
      match kind with
      | Semi | Anti -> ls
      | Inner | Left -> ls @ unkeyed on rs
      | Full -> List.map key ls @ unkeyed on rs)

and out_schema : type s. _ -> s out -> _ =
 fun s -> function
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

(* Frames *)

type failure = Data of string | Raised of exn

(* The rows of a frame, in its order, and their rows [at] in the step's input. A
   frame that is [one] has one value, a reduction's. [failed] is the earliest
   failure, and at a row the first. *)
type frame = {
  rows : row array;
  at : int array;
  one : bool;
  failed : (int * failure) option ref;
}

let frame rows at = { rows; at; one = false; failed = ref None }
let size fr = if fr.one then 1 else Array.length fr.rows

let start fr =
  Array.fold_left Int.min (if fr.at = [||] then 0 else max_int) fr.at

let record fr row why =
  match !(fr.failed) with
  | Some (r, _) when r <= row -> ()
  | _ -> fr.failed := Some (row, why)

let fail fr i why = record fr (if fr.one then start fr else fr.at.(i)) why

(* [Failed (row, reason)] is a step's failure at its input's row [row]. *)
exception Failed of int * string

let check failed =
  match !failed with
  | None -> ()
  | Some (row, Data why) -> raise (Failed (row, why))
  | Some (_, Raised exn) -> raise exn

(* [ocaml fr ty f vs] is [f] of each value of [vs] that [ty] holds. *)
let ocaml fr ty f vs =
  Array.mapi
    (fun i v ->
      match f v with
      | None -> None
      | Some w when Type.holds ty w -> Some w
      | Some w ->
          fail fr i (Data (Format.asprintf "%a does not hold %d" Type.pp ty w));
          None
      | exception exn ->
          fail fr i (Raised exn);
          None)
    vs

let compare_cells (Cell (ty, x)) c =
  match (x, read ty c) with
  | None, None -> 0
  | None, Some _ -> 1
  | Some _, None -> -1
  | Some x, Some y -> Type.compare_value ty x y

let compare_keys keys r r' =
  List.fold_left
    (fun c { name; desc; nulls_first } ->
      if c <> 0 then c
      else
        let (Cell (ty, x) as a) = List.assoc name r in
        let y = read ty (List.assoc name r') in
        match (x, y) with
        | None, None -> 0
        | None, Some _ -> if nulls_first then -1 else 1
        | Some _, None -> if nulls_first then 1 else -1
        | Some _, Some _ ->
            let c = compare_cells a (List.assoc name r') in
            if desc then -c else c)
    0 keys

(* [partition same is] is [is] in groups of [same] elements, in order of first
   appearance. *)
let rec partition same = function
  | [] -> []
  | i :: is ->
      let g, rest = List.partition (same i) is in
      (i :: g) :: partition same rest

(* Reductions *)

(* nx's sort order of floats: [-0.] before [0.], NaN last. *)
let float_order x y =
  match (Float.is_nan x, Float.is_nan y) with
  | true, true -> 0
  | true, false -> 1
  | false, true -> -1
  | false, false -> (
      match Float.compare x y with
      | 0 -> Bool.compare (Float.sign_bit y) (Float.sign_bit x)
      | c -> c)

let quantile p xs =
  let s = List.sort float_order xs |> Array.of_list in
  let n = Array.length s in
  let h = p *. float_of_int (n - 1) in
  let lo = int_of_float h in
  let a = s.(lo) and b = s.(Int.min (lo + 1) (n - 1)) in
  let f = h -. Float.of_int lo in
  if f = 0. || a = b then a else a +. (f *. (b -. a))

let to_float : type a. a Type.t -> a -> float =
 fun ty v ->
  match
    ( Kind.provably_equal (Type.kind ty) Kind.int,
      Kind.provably_equal (Type.kind ty) Kind.float )
  with
  | Some Equal, _ -> float_of_int v
  | _, Some Equal -> v
  | None, None -> invalid_arg "Reference: a quantile of no number"

let reduce : type a b.
    frame -> (a, b) reduction -> a Type.t -> a option array -> b option =
 fun fr r ty vs ->
  let values = List.filter_map Fun.id (Array.to_list vs) in
  let n = List.length values in
  let cmp = Type.compare_value ty in
  (* The position of the first value that no other one is [better] than. *)
  let extreme better =
    let best = ref None in
    Array.iteri
      (fun i v ->
        match (v, !best) with
        | Some x, Some (_, y) when better (cmp x y) -> best := Some (i, x)
        | Some x, None -> best := Some (i, x)
        | _ -> ())
      vs;
    !best
  in
  let least = extreme (fun c -> c < 0)
  and greatest = extreme (fun c -> c > 0) in
  match r with
  | Count -> Some n
  | Sum -> Some (List.fold_left ( + ) 0 values)
  | Mean when n = 0 -> None
  | Mean -> Some (float_of_int (List.fold_left ( + ) 0 values) /. float_of_int n)
  | Min -> Option.map snd least
  | Max -> Option.map snd greatest
  | Arg_min -> Option.map fst least
  | Arg_max -> Option.map fst greatest
  | First -> List.nth_opt values 0
  | Last -> List.nth_opt (List.rev values) 0
  | Only ->
      if List.exists (fun y -> cmp (List.hd values) y <> 0) values then
        record fr (start fr) (Data "only finds several values");
      List.nth_opt values 0
  | Median when n = 0 -> None
  | Median -> Some (quantile 0.5 (List.map (to_float ty) values))
  | Quantile _ when n = 0 -> None
  | Quantile p -> Some (quantile p (List.map (to_float ty) values))
  | N_unique ->
      let distinct =
        partition
          (fun x y -> Option.equal (fun x y -> cmp x y = 0) x y)
          (Array.to_list vs)
      in
      Some (List.length distinct)

(* Expressions, node by node over a frame *)

let rec eval : type a s. frame -> (a, s) term -> a option array =
 fun fr e ->
  let both f a b =
    let x = eval fr a in
    Array.map2
      (fun x y -> match (x, y) with Some x, Some y -> f x y | _ -> None)
      x (eval fr b)
  in
  match e with
  | Col (ty, n) -> Array.map (fun r -> read ty (List.assoc n r)) fr.rows
  | Lit (ty, v) -> Array.make (size fr) (Some (stored ty v))
  | Null _ -> Array.make (size fr) None
  | Int (op, a, b) -> both (int (type_of e) op) a b
  | Float (op, a, b) -> both (fun x y -> Some (float (type_of e) op x y)) a b
  | Cmp (op, a, b) ->
      both
        (fun x y -> Some (compare op (Type.compare_value (common a b) x y)))
        a b
  | And (a, b) ->
      let x = eval fr a in
      Array.map2
        (fun x y ->
          match (x, y) with
          | Some false, _ | _, Some false -> Some false
          | Some true, Some true -> Some true
          | _ -> None)
        x (eval fr b)
  | Or (a, b) ->
      let x = eval fr a in
      Array.map2
        (fun x y ->
          match (x, y) with
          | Some true, _ | _, Some true -> Some true
          | Some false, Some false -> Some false
          | _ -> None)
        x (eval fr b)
  | Not a -> Array.map (Option.map not) (eval fr a)
  | If (c, a, b) ->
      let c = eval fr c in
      let a = eval fr a in
      let b = eval fr b in
      Array.mapi (fun i c -> if c = Some true then a.(i) else b.(i)) c
  | Is_null a -> Array.map (fun v -> Some (Option.is_none v)) (eval fr a)
  | Coalesce es ->
      let vs = List.map (eval fr) es in
      Array.init (size fr) (fun i -> List.find_map (fun v -> v.(i)) vs)
  | Store (_, a) -> eval fr a
  | Is_in (vs, a) ->
      let ty = type_of a in
      let same x v = Type.compare_value ty x (stored ty v) = 0 in
      Array.map
        (fun x ->
          Some
            (Option.fold ~none:false ~some:(fun x -> List.exists (same x) vs) x))
        (eval fr a)
  | Map (ty, f, a) -> ocaml fr ty (Option.map f) (eval fr a)
  | Bind (ty, f, a) -> ocaml fr ty f (eval fr a)
  | Rows -> Array.make (size fr) (Some (Array.length fr.rows))
  | Reduce (r, a) ->
      let v = reduce fr r (type_of a) (eval { fr with one = false } a) in
      Array.make (size fr) v
  | Shift (n, a) ->
      let vs = eval fr a in
      Array.init (size fr) (fun i ->
          if i - n >= 0 && i - n < Array.length vs then vs.(i - n) else None)
  | Rank a ->
      let ty = type_of a in
      let vs = eval fr a in
      let below x = function
        | Some y -> Type.compare_value ty y x < 0
        | None -> false
      in
      Array.map
        (Option.map (fun x ->
             1
             + Array.fold_left (fun k y -> if below x y then k + 1 else k) 0 vs))
        vs
  | Over (by, keys, a) ->
      let same i j =
        List.for_all
          (fun n ->
            compare_cells (List.assoc n fr.rows.(i)) (List.assoc n fr.rows.(j))
            = 0)
          by
      in
      let out = Array.make (size fr) None in
      let over g =
        let g =
          List.stable_sort
            (fun i j -> compare_keys keys fr.rows.(i) fr.rows.(j))
            g
        in
        let g = Array.of_list g in
        let sub =
          {
            fr with
            rows = Array.map (Array.get fr.rows) g;
            at = Array.map (Array.get fr.at) g;
          }
        in
        Array.iteri (fun j v -> out.(g.(j)) <- v) (eval sub a)
      in
      List.iter over (partition same (List.init (size fr) Fun.id));
      out

(* [outs fr os] is the cells of the outputs [os] on each value of [fr]. *)
let outs : type s. frame -> s out list -> row array =
 fun fr os ->
  let cells : s out -> (string * cell array) list = function
    | Out (n, e) ->
        let ty = type_of e in
        [ (n, Array.map (fun v -> Cell (ty, v)) (eval fr e)) ]
    | Keep ns -> List.map (fun n -> (n, Array.map (List.assoc n) fr.rows)) ns
  in
  let cs = List.concat_map cells os in
  Array.init (size fr) (fun i -> List.map (fun (n, c) -> (n, c.(i))) cs)

let rec local : type a s. (a, s) term -> bool = function
  | Over _ | Shift _ | Rank _ -> false
  | Col _ | Lit _ | Null _ | Rows -> true
  | Int (_, a, b) -> local a && local b
  | Float (_, a, b) -> local a && local b
  | Cmp (_, a, b) -> local a && local b
  | And (a, b) | Or (a, b) -> local a && local b
  | If (c, a, b) -> local c && local a && local b
  | Coalesce es -> List.for_all local es
  | Not a -> local a
  | Is_null a -> local a
  | Is_in (_, a) -> local a
  | Store (_, a) -> local a
  | Map (_, _, a) -> local a
  | Bind (_, _, a) -> local a
  | Reduce (_, a) -> local a

let local_outs os =
  List.for_all (function Out (_, e) -> local e | Keep _ -> true) os

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

(* [step local f rs] is [f] over the rows [rs] of a step's input, one row at a
   time if [local], else all at once. *)
let step local f rs =
  let over fr =
    let out = f fr in
    check fr.failed;
    Array.to_seq out
  in
  if local then
    Seq.concat (Seq.mapi (fun i r -> over (frame [| r |] [| i |])) rs)
  else fun () ->
    let rs = Array.of_seq rs in
    over (frame rs (Array.init (Array.length rs) Fun.id)) ()

let aggregate by os rs () =
  let rs = Array.of_seq rs in
  let same i j =
    List.for_all
      (fun n -> compare_cells (List.assoc n rs.(i)) (List.assoc n rs.(j)) = 0)
      by
  in
  let groups =
    match partition same (List.init (Array.length rs) Fun.id) with
    | [] when by = [] -> [ [] ]
    | gs -> gs
  in
  let failed = ref None in
  let group g =
    let at = Array.of_list g in
    let fr = { rows = Array.map (Array.get rs) at; at; one = true; failed } in
    let keys = List.map (fun n -> (n, List.assoc n rs.(List.hd g))) by in
    keys @ (outs fr os).(0)
  in
  let out = List.map group groups in
  check failed;
  List.to_seq out ()

(* Joins *)

let key_text : type a. a Type.t -> (a option -> string) option =
 fun ty ->
  let text lit = function
    | None -> "∅"
    | Some v -> Format.asprintf "%a" Expr.pp (lit v)
  in
  match ty with Datetime _ -> None | _ -> Option.map text (literal ty)

(* [same_key a b] is [true] iff the cells [a] and [b] are one key. *)
let same_key (Cell (ty, _) as a) (Cell (ty', _) as b) =
  let (Type.Any t) = meet (Any ty) (Any ty') in
  match (read t a, read t b) with
  | None, None -> true
  | Some x, Some y -> Type.compare_value t x y = 0
  | _ -> false

let allows (count : Join.count) k =
  match count with
  | Any -> true
  | At_most_one -> k <= 1
  | One -> k = 1
  | At_least_one -> k >= 1

let phrase : Join.count -> string = function
  | Any -> "any"
  | At_most_one -> "at most one"
  | One -> "one"
  | At_least_one -> "at least one"

(* [assertion side count names i r k] fails at row [i] of [side], [r], if
   [count] does not allow its [k] matches, naming its keys [names]. *)
let assertion side count names i r k =
  if not (allows count k) then begin
    let key n =
      let (Cell (ty, v)) = List.assoc n r in
      match key_text ty with
      | Some text -> Format.asprintf "%a is %s" Type.pp_quoted n (text v)
      | None ->
          Format.kasprintf invalid_arg "Reference: no text for %a" Type.pp ty
    in
    let whose =
      match names with
      | [] -> ""
      | ns -> " whose " ^ String.concat " and " (List.map key ns)
    in
    let why =
      Printf.sprintf "the %s row%s matches %d rows, not %s" side whose k
        (phrase count)
    in
    raise (Failed (i, why))
  end

let join kind (each_left, each_right) on (lschema, rschema) ls rs () =
  let ks = keys on in
  let null (n, Type.Any ty) =
    let (Type.Any st) = storage ty in
    (n, Cell (st, None))
  in
  let pair l = function
    | Some r ->
        l @ List.map (fun (n, _) -> (n, List.assoc n r)) (unkeyed on rschema)
    | None -> l @ List.map null (unkeyed on rschema)
  in
  (* A right row without a match, its keys in the left's. *)
  let unmatched r =
    pair
      (List.map
         (fun ((n, _) as c) ->
           match List.assoc_opt n ks with
           | Some rn -> (n, List.assoc rn r)
           | None -> null c)
         lschema)
      (Some r)
  in
  let out l js =
    match ((kind : Join.kind), js) with
    | (Inner | Left | Full), _ :: _ -> List.map (fun r -> pair l (Some r)) js
    | (Left | Full), [] -> [ pair l None ]
    | Inner, [] -> []
    | Semi, js -> if js = [] then [] else [ l ]
    | Anti, js -> if js = [] then [ l ] else []
  in
  match on with
  | All ->
      let rs = Array.of_seq rs in
      let n = ref 0 in
      let each l =
        assertion "left" each_left [] !n l (Array.length rs);
        incr n;
        List.to_seq (out l (Array.to_list rs))
      in
      let last () =
        Array.iteri (fun j r -> assertion "right" each_right [] j r !n) rs;
        if kind = Full && !n = 0 then Array.to_seq (Array.map unmatched rs) ()
        else Seq.Nil
      in
      Seq.append (Seq.concat_map each ls) last ()
  | Keys _ | Position ->
      let ls = Array.of_seq ls in
      let rs = Array.of_seq rs in
      let matches i l j r =
        match on with
        | Position -> i = j
        | _ ->
            List.for_all
              (fun (ln, rn) -> same_key (List.assoc ln l) (List.assoc rn r))
              ks
      in
      let hits =
        Array.mapi
          (fun i l ->
            List.filter
              (fun j -> matches i l j rs.(j))
              (List.init (Array.length rs) Fun.id))
          ls
      in
      Array.iteri
        (fun i l ->
          assertion "left" each_left (List.map fst ks) i l
            (List.length hits.(i)))
        ls;
      let counts =
        Array.mapi
          (fun j _ ->
            Array.fold_left
              (fun k js -> if List.mem j js then k + 1 else k)
              0 hits)
          rs
      in
      Array.iteri
        (fun j r ->
          assertion "right" each_right (List.map snd ks) j r counts.(j))
        rs;
      let pairs =
        List.concat
          (Array.to_list
             (Array.mapi
                (fun i l -> out l (List.map (Array.get rs) hits.(i)))
                ls))
      in
      let rest =
        if kind <> Full then []
        else
          List.filteri (fun j _ -> counts.(j) = 0) (Array.to_list rs)
          |> List.map unmatched
      in
      List.to_seq (pairs @ rest) ()

let rec rows : plan -> row Seq.t = function
  | Table t -> List.to_seq (table_rows t)
  | Select (os, p) -> step (local_outs os) (fun fr -> outs fr os) (rows p)
  | Derive (os, p) ->
      step (local_outs os)
        (fun fr -> Array.map2 derived fr.rows (outs fr os))
        (rows p)
  | Filter (e, p) ->
      let keep fr =
        let k = eval fr e in
        Array.of_list
          (List.filteri (fun i _ -> k.(i) = Some true) (Array.to_list fr.rows))
      in
      step (local e) keep (rows p)
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
  | Aggregate (by, os, p) -> aggregate by os (rows p)
  | Join { kind; each_left; each_right; on; left; right } ->
      join kind (each_left, each_right) on
        (schema left, schema right)
        (rows left) (rows right)

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
  let value fr =
    let v = (eval fr e).(0) in
    check fr.failed;
    match v with Some v -> [| v |] | None -> raise (Failed (fr.at.(0), null))
  in
  match Array.of_seq (step true value (rows p)) with
  | vs -> Ok vs
  | exception Failed (row, why) -> Error (row, why)
