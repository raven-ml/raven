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
      Column.make (Any d.type_) ?valid:(Column.valid c)
        ~length:(Column.length c) (Column.data c)
  | Value -> not_lowered "an OCaml value"

let type_of : type a. a Expr.typing -> Type.any = function
  | Column ty -> Any ty
  | Extension d -> Any d.type_
  | Value -> not_lowered "an OCaml value"

(* [widen t c] is [c] at the type [t] that contains [c]'s: binding leaves each
   operand at its own type, and its operation casts it to the type they meet
   at. *)
let widen (Type.Any ty as t) c =
  let (Any from) = Column.type_ c in
  let length = Column.length c and valid = Column.valid c in
  match (from, ty, dtype ty, Column.data c) with
  | _ when Type.equal from ty -> c
  | Categorical _, Categorical _, _, data -> Column.make t ?valid ~length data
  | Categorical dict, String, _, Fixed (P codes) ->
      let codes = Nx.cast Nx.int64 codes in
      let codes =
        match valid with
        | Some v -> Nx.where v codes (Nx.full Nx.int64 [| length |] (-1L))
        | None -> codes
      in
      Column.take codes (Column.v Type.string (Iarray.to_array dict))
  | _, _, Some (Dtype dt), Fixed (P x) ->
      Column.make t ?valid ~length (Fixed (P (Nx.cast dt x)))
  | _ ->
      not_lowered (Format.asprintf "widening %a to %a" Type.pp from Type.pp ty)

(* [words a b] is the order words ({!Key.value}) of [a]'s and [b]'s rows, one
   type, which compare as their values do. A compound value's word is a code
   relative to the rows it is computed over, so those of both columns are
   computed at once. *)
let words a b =
  match Column.data a with
  | Fixed (P x) when Nx.ndim x = 1 -> (Key.value Order a, Key.value Order b)
  | _ ->
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
     talon wraps it: [x / -1] is [neg x], and [x mod -1] is [x mod 1], 0. *)
  let quotient f =
    if not signed then f x y
    else
      let one = Nx.ones_like y in
      let minus_one = Nx.equal y (Nx.neg one) in
      Nx.where minus_one (f (Nx.neg x) one) (f x (Nx.where minus_one one y))
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

let compare (op : Expr.compare) a b =
  let x, y = words a b in
  let r =
    match op with
    | `Eq -> Nx.equal x y
    | `Ne -> Nx.not_equal x y
    | `Lt -> Nx.less x y
    | `Le -> Nx.less_equal x y
    | `Gt -> Nx.greater x y
    | `Ge -> Nx.greater_equal x y
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

(* [is_in vs a] finds the words of [a]'s rows among the sorted words of the
   values [vs]. *)
let is_in vs a =
  let m = Column.length vs in
  if m = 0 then boolean (Nx.zeros Nx.bool [| Column.length a |])
  else
    let x, w = words a vs in
    let s, _ = Nx.sort w in
    let i = Nx.searchsorted ~side:`Left s x in
    let hit =
      Nx.equal (Nx.take ~indices:(Nx.minimum_s i (Int64.of_int (m - 1))) s) x
    in
    boolean (Option.fold ~none:hit ~some:(Nx.logical_and hit) (Column.valid a))

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
      let t =
        match (Expr.typing a, Expr.typing b) with
        | Column ta, Column tb -> Type.Any (Option.get (Type.common [ ta; tb ]))
        | ta, _ -> type_of ta
      in
      binary a b (fun a b -> compare op (widen t a) (widen t b))
  | Logic (op, a, b), _ -> binary a b (logic op)
  | Not a, _ -> unary a (fun a -> boolean ?valid:(Column.valid a) (falsity a))
  | If (c, a, b), _ ->
      let t = type_of typing in
      let c = compile st c and a = compile st a and b = compile st b in
      fun env ->
        choose (truth (get env c)) (widen t (get env a)) (widen t (get env b))
  | Is_null a, _ -> unary a is_null
  | Coalesce es, _ ->
      let t = type_of typing and es = List.map (compile st) es in
      fun env -> coalesce (List.map (fun e -> widen t (get env e)) es)
  | Is_in (vs, a), _ ->
      let vs =
        column (Expr.typing a) (Array.of_list (List.map Option.some vs))
      in
      unary a (is_in vs)
  | Store (ty, a), _ -> (
      match Expr.typing a with
      | Value -> not_lowered (Format.asprintf "%a" Expr.pp e)
      | _ -> unary a (widen (Any ty)))
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
          let c =
            Column.make (Any d.storage) ?valid:(Column.valid c)
              ~length:(Column.length c) (Column.data c)
          in
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
