(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type frame = { columns : Column.t array; rows : int }

let frame b = { columns = Table.columns b; rows = Table.rows b }

type cause = Data of string | Raised of exn * Printexc.raw_backtrace
type failure = { row : int; cause : cause }

let not_lowered what = invalid_arg (what ^ " is not implemented yet")

(* Columns

   A value is a column of the frame's rows, or of one row where it computes from
   literals alone, which nx broadcasts. *)

type dtype = Dtype : ('a, 'b) Nx.dtype -> dtype

let dtype : type a. a Type.t -> dtype option = function
  | Bool -> Some (Dtype Nx.bool)
  | Int8 -> Some (Dtype Nx.int8)
  | Int16 -> Some (Dtype Nx.int16)
  | Int32 -> Some (Dtype Nx.int32)
  | Int64 -> Some (Dtype Nx.int64)
  | Uint8 -> Some (Dtype Nx.uint8)
  | Uint16 -> Some (Dtype Nx.uint16)
  | Uint32 -> Some (Dtype Nx.uint32)
  | Uint64 -> Some (Dtype Nx.uint64)
  | Float16 -> Some (Dtype Nx.float16)
  | Float32 -> Some (Dtype Nx.float32)
  | Float64 -> Some (Dtype Nx.float64)
  | _ -> None

(* [tensor dt c] is the values of [c], a column of a type stored one element per
   row, cast to [dt]. *)
let tensor dt c =
  match Column.data c with Fixed (P x) -> Nx.cast dt x | _ -> assert false

(* [valid cs] is where every column of [cs] holds a value, [None] where none has
   a null. *)
let valid cs =
  let both v c =
    match (v, Column.valid c) with
    | v, None | None, v -> v
    | Some a, Some b -> Some (Nx.logical_and a b)
  in
  List.fold_left both None cs

(* [fixed ty ?valid x] is the column of [ty] stored as [x], null where [valid]
   is [false], with zeros under its nulls. *)
let fixed ty ?valid x =
  let length = Nx.dim 0 x in
  let valid = Option.map (Nx.broadcast_to [| length |]) valid in
  let x =
    match valid with Some v -> Nx.where v x (Nx.zeros_like x) | None -> x
  in
  Column.make ty ?valid ~length (Fixed (P x))

let boolean ?valid x = fixed (Type.Any Type.bool) ?valid x

(* [column t vs] is the column of the values [vs] at the typing [t]. *)
let column : type a. a Expr.typing -> a option array -> Column.t =
 fun t vs ->
  let encoded ty vs =
    match Column.encode ty (Array.length vs) (Array.get vs) with
    | Ok c -> c
    | Error _ -> assert false (* Binding checked that [ty] holds [vs]. *)
  in
  match t with
  | Column ty -> encoded ty vs
  | Extension d ->
      let c = encoded d.storage (Array.map (Option.map d.enc) vs) in
      Column.with_data (Any d.type_) (Column.data c) c
  | Value -> not_lowered "an OCaml value"

let type_of : type a. a Expr.typing -> Type.any = function
  | Column ty -> Any ty
  | Extension d -> Any d.type_
  | Value -> not_lowered "an OCaml value"

(* [meet ta tb] is the type at which operands of the typings [ta] and [tb]
   meet. *)
let meet : type a. a Expr.typing -> a Expr.typing -> Type.any =
 fun ta tb ->
  match (ta, tb) with
  | Column ta, Column tb -> Any (Option.get (Type.common [ ta; tb ]))
  | ta, _ -> type_of ta

(* [widen from t] converts a column of type [from] to the type [t] that contains
   it: binding leaves each operand at its own type, and its operation casts it
   to the type they meet at. The conversion is chosen when the expression
   compiles, so one that no unit lowers yet is refused before any data is
   read. *)
let widen (Type.Any from) (Type.Any ty as t) =
  match (from, ty, dtype from, dtype ty) with
  | _ when Type.equal from ty -> Fun.id
  | Categorical _, Categorical _, _, _ ->
      fun c -> Column.with_data t (Column.data c) c
  | Categorical dict, String, _, _ ->
      let text = Column.v Type.string (Iarray.to_array dict) in
      fun c ->
        let codes = tensor Nx.int64 c in
        let null = Nx.full Nx.int64 [| Column.length c |] (-1L) in
        let codes =
          Option.fold ~none:codes
            ~some:(fun v -> Nx.where v codes null)
            (Column.valid c)
        in
        Column.take codes text
  | _, _, Some _, Some (Dtype dt) ->
      fun c -> Column.with_data t (Fixed (P (tensor dt c))) c
  | _ ->
      not_lowered (Format.asprintf "widening %a to %a" Type.pp from Type.pp ty)

(* [words a b] is the order words ({!Key.value}) of the rows of the compound
   columns [a] and [b], of one type, which compare as their values do. A
   compound value's word is a code relative to the rows it is computed over, so
   those of both columns are computed at once. *)
let words a b =
  let w = Key.value Order (Column.concat [ a; b ]) in
  let n = Column.length a in
  (Nx.shrink [| (0, n) |] w, Nx.shrink [| (n, Nx.dim 0 w) |] w)

(* Operations *)

let arithmetic : type a.
    a Type.t -> Expr.arith -> Column.t -> Column.t -> Column.t =
 fun ty op a b ->
  let integer, signed =
    match ty with
    | Int8 | Int16 | Int32 | Int64 -> (true, true)
    | Uint8 | Uint16 | Uint32 | Uint64 -> (true, false)
    | _ -> (false, false)
  in
  let (Dtype dt) = Option.get (dtype ty) in
  let x = tensor dt a and y = tensor dt b in
  (* nx leaves a signed type's least value divided by [-1] unspecified, and
     talon wraps it: [x / -1] is [neg x], and [x mod -1] is [x mod 1], 0. Both
     divide by [y] with [-1] replaced by [1]. *)
  let quotient f =
    if not signed then f x y
    else
      let one = Nx.ones_like y in
      let minus_one = Nx.equal y (Nx.neg one) in
      let q = f x (Nx.where minus_one one y) in
      match op with Div -> Nx.where minus_one (Nx.neg x) q | _ -> q
  in
  let r =
    match op with
    | Add -> Nx.add x y
    | Sub -> Nx.sub x y
    | Mul -> Nx.mul x y
    | Pow -> Nx.pow x y
    | Div -> quotient Nx.div
    | Mod -> quotient Nx.mod_
  in
  let valid = valid [ a; b ] in
  let valid =
    match op with
    | (Div | Mod) when integer ->
        let nonzero = Nx.not_equal y (Nx.zeros_like y) in
        Some (Option.fold ~none:nonzero ~some:(Nx.logical_and nonzero) valid)
    | _ -> valid
  in
  fixed (Any ty) ?valid r

(* [ordered op x y] compares [x] and [y] by talon's total order: floats with
   every NaN equal, and after every other value. *)
let ordered (type a b) (op : Expr.compare) (x : (a, b) Nx.t) (y : (a, b) Nx.t) =
  let r =
    match op with
    | `Eq -> Nx.equal x y
    | `Ne -> Nx.not_equal x y
    | `Lt -> Nx.less x y
    | `Le -> Nx.less_equal x y
    | `Gt -> Nx.greater x y
    | `Ge -> Nx.greater_equal x y
  in
  if not (Nx_dtype.is_float (Nx.dtype x)) then r
  else
    let nan_x = Nx.isnan x and nan_y = Nx.isnan y in
    let only a b = Nx.logical_and a (Nx.logical_not b) in
    match op with
    | `Eq -> Nx.logical_or r (Nx.logical_and nan_x nan_y)
    | `Ne -> only r (Nx.logical_and nan_x nan_y)
    | `Lt -> Nx.logical_or r (only nan_y nan_x)
    | `Le -> Nx.logical_or r nan_y
    | `Gt -> Nx.logical_or r (only nan_x nan_y)
    | `Ge -> Nx.logical_or r nan_x

let compare op a b =
  let r =
    match (Column.data a, Column.data b) with
    | Fixed (P x), Fixed q when Nx.ndim x = 1 ->
        ordered op x (Nx.unpack (Nx.dtype x) q)
    | _ ->
        let x, y = words a b in
        ordered op x y
  in
  boolean ?valid:(valid [ a; b ]) r

(* [truth c] is where the boolean column [c] is [true], and [falsity c] where it
   is [false]: neither holds under a null. *)
let truth c =
  let x = tensor Nx.bool c in
  Option.fold ~none:x ~some:(Nx.logical_and x) (Column.valid c)

let falsity c =
  let x = Nx.logical_not (tensor Nx.bool c) in
  Option.fold ~none:x ~some:(Nx.logical_and x) (Column.valid c)

(* Kleene: a conjunction is valid where it is [true] or an operand is [false], a
   disjunction where it is [true] or both operands are [false]. *)
let logic (op : Expr.logic) a b =
  let combine = match op with And -> Nx.logical_and | Or -> Nx.logical_or in
  let r = combine (truth a) (truth b) in
  let valid =
    match (Column.valid a, Column.valid b) with
    | None, None -> None
    | _ ->
        let decided =
          match op with
          | And -> Nx.logical_or (falsity a) (falsity b)
          | Or -> Nx.logical_and (falsity a) (falsity b)
        in
        Some (Nx.logical_or r decided)
  in
  boolean ?valid r

let is_null c =
  match Column.valid c with
  | Some v -> boolean (Nx.logical_not v)
  | None -> boolean (Nx.zeros Nx.bool [| Column.length c |])

(* [choose m a b] is [a] where [m] holds and [b] elsewhere, [a] and [b] of one
   type. Text, lists, records and tensors are taken from both. *)
let choose m a b =
  match (Column.data a, Column.data b) with
  | Fixed (P x), Fixed p when Nx.ndim x = 1 ->
      let y = Nx.unpack (Nx.dtype x) p in
      let valid =
        match (Column.valid a, Column.valid b) with
        | None, None -> None
        | va, vb ->
            let all c = Nx.ones Nx.bool [| Column.length c |] in
            let va = Option.value va ~default:(all a) in
            Some (Nx.where m va (Option.value vb ~default:(all b)))
      in
      fixed (Column.type_ a) ?valid (Nx.where m x y)
  | _ ->
      let length =
        Int.max (Nx.dim 0 m) (Int.max (Column.length a) (Column.length b))
      in
      let rows c =
        if Column.length c = length then Nx.arange Nx.int64 0 length 1
        else Nx.zeros Nx.int64 [| length |]
      in
      let from_b = Nx.add_s (rows b) (Int64.of_int (Column.length a)) in
      Column.take (Nx.where m (rows a) from_b) (Column.concat [ a; b ])

let rec coalesce = function
  | [] -> assert false (* Binding gives a coalesce an operand. *)
  | [ c ] -> c
  | c :: cs -> (
      match Column.valid c with None -> c | Some v -> choose v c (coalesce cs))

(* [is_in vs] finds the words of a column's rows among the sorted words of the
   values [vs]: words of fixed-width values sorted once, and codes of compound
   values computed with the column's rows. *)
let is_in vs =
  let m = Column.length vs in
  let found s x a =
    let i = Nx.searchsorted ~side:`Left s x in
    let hit =
      Nx.equal (Nx.take ~indices:(Nx.minimum_s i (Int64.of_int (m - 1))) s) x
    in
    boolean (Option.fold ~none:hit ~some:(Nx.logical_and hit) (Column.valid a))
  in
  match Column.data vs with
  | _ when m = 0 -> fun a -> boolean (Nx.zeros Nx.bool [| Column.length a |])
  | Fixed (P x) when Nx.ndim x = 1 ->
      let s, _ = Nx.sort (Key.value Order vs) in
      fun a -> found s (Key.value Order a) a
  | _ ->
      fun a ->
        let x, w = words a vs in
        found (fst (Nx.sort w)) x a

(* Compiling

   Each node compiles once per call of [outputs], [predicate] or [values]. A
   node that the expressions reach twice gets a slot, which holds its value
   during the evaluation of a frame. *)

type env = {
  frame : frame;
  slots : Column.t option array;
  mutable failure : failure option;
}

type node = { eval : env -> Column.t; mutable slot : int }

type state = {
  names : string list;
  mutable nodes : (Expr.packed * node) list;
  mutable slots : int;
}

let state s = { names = Schema.names s; nodes = []; slots = 0 }

let get env n =
  if n.slot < 0 then n.eval env
  else
    match env.slots.(n.slot) with
    | Some c -> c
    | None ->
        let c = n.eval env in
        env.slots.(n.slot) <- Some c;
        c

let rec compile : type a s. state -> (a, s) Expr.t -> node =
 fun st e ->
  match List.find_opt (fun (Expr.Packed e', _) -> Expr.same e e') st.nodes with
  | Some (_, n) ->
      if n.slot < 0 then begin
        n.slot <- st.slots;
        st.slots <- st.slots + 1
      end;
      n
  | None ->
      let n = { eval = lower st e; slot = -1 } in
      st.nodes <- (Expr.Packed e, n) :: st.nodes;
      n

and lower : type a s. state -> (a, s) Expr.t -> env -> Column.t =
 fun st e ->
  let typing = Expr.typing e in
  let unary a f =
    let a = compile st a in
    fun env -> f (get env a)
  in
  let binary a b f =
    let a = compile st a and b = compile st b in
    fun env -> f (get env a) (get env b)
  in
  match (Expr.node e, typing) with
  | (Handle (_, n) | Read (_, n) | Ext_handle (_, n)), _ ->
      let i = Option.get (List.find_index (String.equal n) st.names) in
      fun env -> env.frame.columns.(i)
  | Lit (_, v), _ ->
      let c = column typing [| Some v |] in
      Fun.const c
  | Null, _ ->
      let c = column typing [| None |] in
      Fun.const c
  | Int (op, a, b), Column ty -> binary a b (arithmetic ty op)
  | Float (op, a, b), Column ty -> binary a b (arithmetic ty op)
  | Compare (op, a, b), _ ->
      let t = meet (Expr.typing a) (Expr.typing b) in
      let wa = widen (type_of (Expr.typing a)) t
      and wb = widen (type_of (Expr.typing b)) t in
      binary a b (fun a b -> compare op (wa a) (wb b))
  | Logic (op, a, b), _ -> binary a b (logic op)
  | Not a, _ -> unary a (fun a -> boolean ?valid:(Column.valid a) (falsity a))
  | If (c, a, b), _ ->
      let t = type_of typing in
      let wa = widen (type_of (Expr.typing a)) t
      and wb = widen (type_of (Expr.typing b)) t in
      let c = compile st c and a = compile st a and b = compile st b in
      fun env -> choose (truth (get env c)) (wa (get env a)) (wb (get env b))
  | Is_null a, _ -> unary a is_null
  | Coalesce es, _ ->
      let t = type_of typing in
      let operand e = (compile st e, widen (type_of (Expr.typing e)) t) in
      let es = List.map operand es in
      fun env -> coalesce (List.map (fun (e, w) -> w (get env e)) es)
  | Is_in (vs, a), _ ->
      let vs =
        column (Expr.typing a) (Array.of_list (List.map Option.some vs))
      in
      unary a (is_in vs)
  | Store (ty, a), _ -> (
      match Expr.typing a with
      | Value -> not_lowered (Format.asprintf "%a" Expr.pp e)
      | ta -> unary a (widen (type_of ta) (Any ty)))
  | _ -> not_lowered (Format.asprintf "%a" Expr.pp e)

(* Calls *)

let env st frame = { frame; slots = Array.make st.slots None; failure = None }

let fail env row cause =
  match env.failure with
  | Some f when f.row <= row -> ()
  | _ -> env.failure <- Some { row; cause }

(* [fill env c] is [c], or the one row of literals [c] repeated on each of the
   frame's rows. *)
let fill env c =
  let n = env.frame.rows in
  if Column.length c = n then c else Column.take (Nx.zeros Nx.int64 [| n |]) c

let outputs s os =
  let st = state s in
  let nodes = List.map (fun (_, Expr.Packed e) -> compile st e) os in
  fun frame ->
    let env = env st frame in
    let cs = List.map (fun n -> fill env (get env n)) nodes in
    (cs, env.failure)

let predicate s p =
  let st = state s in
  let n = compile st p in
  fun frame ->
    let env = env st frame in
    let m = truth (fill env (get env n)) in
    (m, env.failure)

let values : type a.
    Schema.t -> (a, Expr.row) Expr.t -> frame -> a option array * failure option
    =
 fun s e ->
  let st = state s in
  let n = compile st e in
  let decoder : Column.t -> (int -> a option, int * string) result =
    match Expr.typing e with
    | Column ty -> Column.decoder ty
    | Extension d ->
        fun c ->
          let c = Column.with_data (Any d.storage) (Column.data c) c in
          Result.map
            (fun get i -> Option.map d.dec (get i))
            (Column.decoder d.storage c)
    | Value -> not_lowered (Format.asprintf "%a" Expr.pp e)
  in
  fun frame ->
    let env = env st frame in
    let c = fill env (get env n) in
    let vs = Array.make frame.rows None in
    let rec read get i stop =
      if i < stop then
        match get i with
        | v ->
            vs.(i) <- v;
            read get (i + 1) stop
        | exception exn ->
            fail env i (Raised (exn, Printexc.get_raw_backtrace ()))
    in
    let rec decode stop =
      match decoder (Column.sub c ~offset:0 ~length:stop) with
      | Ok get -> read get 0 stop
      | Error (row, why) ->
          fail env row (Data why);
          decode row
    in
    decode (match env.failure with Some f -> f.row | None -> frame.rows);
    (vs, env.failure)
