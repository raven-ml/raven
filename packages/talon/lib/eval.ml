(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [reduced] holds where the outputs reduce over the segments. *)
type frame = {
  columns : Column.t array;
  rows : int;
  segments : Reduce.segments;
  reduced : bool;
}

let frame b =
  let rows = Table.rows b in
  {
    columns = Table.columns b;
    rows;
    segments = Reduce.one rows;
    reduced = false;
  }

let groups b segments = { (frame b) with segments; reduced = true }

type cause = Data of Error.t | Raised of exn * Printexc.raw_backtrace
type failure = { row : int; cause : cause }

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
   a validity: the operands' bits, and-ed word by word. *)
let valid cs =
  let both v c =
    match (v, Column.validity c) with
    | v, None | None, v -> v
    | Some a, Some b -> Some (Nx.logical_and a b)
  in
  List.fold_left both None cs

(* [fixed ty ?valid x] is the column of [ty] stored as [x], null where [valid]
   is clear. The values under its nulls are [x]'s. *)
let fixed ty ?valid x =
  let length = Nx.dim 0 x in
  let validity = Option.map (Nx.broadcast_to [| length |]) valid in
  Column.make ty ?validity ~length (Fixed (P x))

let boolean ?valid x = fixed (Type.Any Type.bool) ?valid x

(* [column t vs] is the column of the literals [vs] at the typing [t]. *)
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
  | Value -> assert false (* An OCaml value is no column. *)

let type_of : type a. a Expr.typing -> Type.any = function
  | Column ty -> Any ty
  | Extension d -> Any d.type_
  | Value -> assert false (* An OCaml value is no column. *)

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
   compiles. *)
let widen (Type.Any from as f) (Type.Any ty as t) =
  match (dtype from, dtype ty) with
  | _ when Type.equal from ty -> Fun.id
  | Some _, Some (Dtype dt) ->
      fun c -> Column.with_data t (Fixed (P (tensor dt c))) c
  | _ -> (
      let cast = Kernels.cast f t in
      fun c ->
        match cast c with
        | c, None -> c
        | _, Some _ -> assert false (* A widening keeps every value. *))

(* [words use a b] is the words ({!Key.value}) of the rows of the compound
   columns [a] and [b], of one type, which compare as their values do for [use].
   A compound value's word is a code relative to the rows it is computed over,
   so those of both columns are computed at once. Equality needs only
   [Identity], which hashes the rows where [Order] sorts them. *)
let words use a b =
  let w = Key.value use (Column.concat [ a; b ]) in
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
        let nonzero = Nx.cast Nx.bit (Nx.not_equal y (Nx.zeros_like y)) in
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

(* [text_name op] names the function whose text comparison reads bytes. *)
let text_name : Expr.compare -> string = function
  | `Eq -> "Expr.( = )"
  | `Ne -> "Expr.( <> )"
  | `Lt -> "Expr.( < )"
  | `Le -> "Expr.( <= )"
  | `Gt -> "Expr.( > )"
  | `Ge -> "Expr.( >= )"

let compare op a b =
  let r =
    match (Column.data a, Column.data b) with
    | Fixed (P x), Fixed q when Nx.ndim x = 1 ->
        ordered op x (Nx.unpack (Nx.dtype x) q)
    | Bytes r, Bytes one when Column.length b = 1 ->
        let s = Strings.compare ~by:(text_name op) r one in
        ordered op s (Nx.zeros_like s)
    | Bytes one, Bytes r when Column.length a = 1 ->
        let s = Strings.compare ~by:(text_name op) r one in
        ordered op (Nx.zeros_like s) s
    | _ ->
        let use : Key.use =
          match op with `Eq | `Ne -> Identity | `Lt | `Le | `Gt | `Ge -> Order
        in
        let x, y = words use a b in
        ordered op x y
  in
  boolean ?valid:(valid [ a; b ]) r

(* [held c] is where [c] holds a value, as a condition. *)
let held c = Option.map (Nx.cast Nx.bool) (Column.validity c)

(* [truth c] is where the boolean column [c] is [true], and [falsity c] where it
   is [false]: neither holds under a null. *)
let truth c =
  let x = tensor Nx.bool c in
  Option.fold ~none:x ~some:(Nx.logical_and x) (held c)

let falsity c =
  let x = Nx.logical_not (tensor Nx.bool c) in
  Option.fold ~none:x ~some:(Nx.logical_and x) (held c)

(* Kleene: a conjunction is valid where both operands are [true] or either is
   [false], and a disjunction where either is [true] or both are [false]. A
   valid row has both operands valid or one that decides it, so the operands'
   values combined as they are, those under nulls included, are its Kleene
   value: [a && true] and [a || false] are [a], under its nulls too. *)
let logic (op : Expr.logic) a b =
  let x = tensor Nx.bool a and y = tensor Nx.bool b in
  let r = match op with And -> Nx.logical_and x y | Or -> Nx.logical_or x y in
  let valid =
    match (Column.validity a, Column.validity b) with
    | None, None -> None
    | _ ->
        let decided =
          match op with
          | And ->
              Nx.logical_or
                (Nx.logical_and (truth a) (truth b))
                (Nx.logical_or (falsity a) (falsity b))
          | Or ->
              Nx.logical_or
                (Nx.logical_or (truth a) (truth b))
                (Nx.logical_and (falsity a) (falsity b))
        in
        Some (Nx.cast Nx.bit decided)
  in
  boolean ?valid r

let is_null c =
  match held c with
  | Some v -> boolean (Nx.logical_not v)
  | None -> boolean (Nx.zeros Nx.bool [| Column.length c |])

(* [choose m a b] is [a] where [m] holds and [b] elsewhere, [a] and [b] of one
   type. Text, lists, records and tensors are taken from both. *)
let choose m a b =
  match (Column.data a, Column.data b) with
  | Fixed (P x), Fixed p when Nx.ndim x = 1 ->
      let y = Nx.unpack (Nx.dtype x) p in
      let valid =
        match (Column.validity a, Column.validity b) with
        | None, None -> None
        | va, vb ->
            let all c = Nx.ones Nx.bit [| Column.length c |] in
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
      Column.gather (Nx.where m (rows a) from_b) (Column.concat [ a; b ])

let rec coalesce = function
  | [] -> assert false (* Binding gives a coalesce an operand. *)
  | [ c ] -> c
  | c :: cs -> (
      if Column.known_zero c then c
      else match held c with Some v -> choose v c (coalesce cs) | None -> c)

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
    boolean (Option.fold ~none:hit ~some:(Nx.logical_and hit) (held a))
  in
  match Column.data vs with
  | _ when m = 0 -> fun a -> boolean (Nx.zeros Nx.bool [| Column.length a |])
  | Fixed (P x) when Nx.ndim x = 1 ->
      let s, _ = Nx.sort (Key.value Order vs) in
      fun a -> found s (Key.value Order a) a
  | _ ->
      fun a ->
        let x, w = words Identity a vs in
        found (fst (Nx.sort w)) x a

(* [lifted ty cs x] is the column of [ty] stored as [x], which an nx operation
   computes over the columns [cs]: null where one of them is. *)
let lifted ty cs x = fixed (Any ty) ?valid:(valid cs) x

(* [width cs] is the rows that the columns [cs] broadcast to, and [wide cs dt c]
   is [c]'s values cast to [dt] and broadcast to them. *)
let width cs =
  List.fold_left
    (fun n c -> if Column.length c = 1 then n else Column.length c)
    1 cs

let wide cs dt c = Nx.broadcast_to [| width cs |] (tensor dt c)

(* Compiling

   Each node compiles once per call of [outputs], [predicate] or [values]. A
   node of a column typing that the expressions reach twice gets a slot, which
   holds its value during the evaluation of a frame. A node of OCaml values
   computes at each place it is reached: its function is pure, so it computes
   the same values there. Operands evaluate from left to right, as the order of
   failures at a row demands, so [binary] and [ternary] name each result.

   A node evaluates to one value per row of the frame or, under a reduction, one
   per segment: [extent] values, the [i]th of which fails at the frame's row
   [origin i], a segment's first row. *)

type env = {
  frame : frame;
  extent : int;
  origin : int -> int;
  slots : Column.t option array;
  failure : failure option ref;
}

let per_row env = { env with extent = env.frame.rows; origin = Fun.id }

let per_segment env segments =
  let first = lazy (Nx.to_array (Reduce.first segments)) in
  {
    env with
    frame = { env.frame with segments };
    extent = Reduce.count segments;
    origin = (fun i -> Int64.to_int (Lazy.force first).(i));
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

let fail env i cause =
  let row = env.origin i in
  match !(env.failure) with
  | Some f when f.row <= row -> ()
  | _ -> env.failure := Some { row; cause }

let data why = Data (Error.v (why ^ "."))

(* [checked env (c, f)] is [c], recording its failure [f]. A column of literals
   fails at its first value, if [env] has values. *)
let checked env (c, f) =
  let fails (i, e) = if i < env.extent then fail env i (Data e) in
  Option.iter fails f;
  c

(* [fill env c] is [c], or the one row of literals [c] repeated [env.extent]
   times. *)
let fill env c =
  let n = env.extent in
  if Column.length c = n then c else Column.gather (Nx.zeros Nx.int64 [| n |]) c

(* [each env f] is [f i] on each value [i] before the earliest failure, and
   [None] from it on. An exception that [f] raises fails at its row, which ends
   the calls. *)
let each env f =
  let vs = Array.make env.extent None in
  let rec go i =
    let stop = match !(env.failure) with Some f -> f.row | None -> max_int in
    if i < env.extent && env.origin i < stop then
      match f i with
      | v ->
          vs.(i) <- v;
          go (i + 1)
      | exception exn ->
          fail env i (Raised (exn, Printexc.get_raw_backtrace ()))
  in
  go 0;
  vs

let decoder : type a.
    a Expr.typing -> Column.t -> (int -> a option, int * string) result =
  function
  | Column ty -> Column.decoder ty
  | Extension d ->
      fun c ->
        let c = Column.with_data (Any d.storage) (Column.data c) c in
        Result.map
          (fun get i -> Option.map d.dec (get i))
          (Column.decoder d.storage c)
  | Value -> assert false (* An OCaml value is no column. *)

(* [decode env dec c] is [c]'s rows decoded to OCaml by [dec] ({!decoder}). A
   row that [dec] refuses fails. *)
let decode env dec c =
  match dec c with
  | Ok get -> each env get
  | Error (row, why) ->
      fail env row (data why);
      each env (Result.get_ok (dec (Column.sub c ~offset:0 ~length:row)))

(* [store env t vs] is the column of the OCaml values [vs] at the column typing
   [t]. A value that [t] does not hold fails at its row, and is null from it
   on. *)
let store : type a. env -> a Expr.typing -> a option array -> Column.t =
 fun env t vs ->
  let encode ty vs =
    let n = Array.length vs in
    match Column.encode ty n (Array.get vs) with
    | Ok c -> c
    | Error (row, why) ->
        fail env row (data why);
        Result.get_ok
          (Column.encode ty n (fun i -> if i < row then vs.(i) else None))
  in
  match t with
  | Column ty -> encode ty vs
  | Extension d ->
      let c = encode d.storage (each env (fun i -> Option.map d.enc vs.(i))) in
      Column.with_data (Any d.type_) (Column.data c) c
  | Value -> assert false (* An OCaml value is no column. *)

let operand_dtype e =
  match Expr.typing e with
  | Column ty -> Option.get (dtype ty)
  | _ -> assert false (* A lift's operand has a column type. *)

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
    fun env ->
      let a = get env a in
      f a (get env b)
  in
  let unary_checked a f =
    let a = compile st a in
    fun env -> checked env (f (get env a))
  in
  let binary_checked a b f =
    let a = compile st a and b = compile st b in
    fun env ->
      let a = get env a in
      checked env (f a (get env b))
  in
  (* Text operations read a categorical as its text. *)
  let textual a f =
    let w = widen (type_of (Expr.typing a)) (Any Type.string) in
    unary_checked a (fun c -> f (w c))
  in
  let ternary a b c f =
    let a = compile st a and b = compile st b and c = compile st c in
    fun env ->
      let a = get env a in
      let b = get env b in
      f a b (get env c)
  in
  match (Expr.node e, typing) with
  | (Handle (_, n) | Read (_, n) | Ext_handle (_, n)), _ ->
      let i = Option.get (List.find_index (String.equal n) st.names) in
      fun env -> env.frame.columns.(i)
  | (Lit (_, v) | Const v), _ ->
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
  | Not a, _ ->
      unary a (fun a ->
          let x = Nx.logical_not (tensor Nx.bool a) in
          Column.with_data (Any Type.bool) (Fixed (P x)) a)
  | If (c, a, b), _ ->
      let t = type_of typing in
      let wa = widen (type_of (Expr.typing a)) t
      and wb = widen (type_of (Expr.typing b)) t in
      ternary c a b (fun c a b -> choose (truth c) (wa a) (wb b))
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
  | Store (ty, a), _ -> unary a (widen (type_of (Expr.typing a)) (Any ty))
  | (App _ | Of_option _), _ ->
      let vs = ocaml st e in
      fun env -> store env typing (vs env)
  | Nx_unary (op, a), Column ty ->
      let (Dtype dt) = Option.get (dtype ty) in
      unary a (fun a -> lifted ty [ a ] (Nx.Op.eval (Unary (op, tensor dt a))))
  | Nx_binary (op, a, b), Column ty ->
      let (Dtype dt) = Option.get (dtype ty) in
      binary a b (fun a b ->
          let cs = [ a; b ] in
          lifted ty cs (Nx.Op.eval (Binary (op, wide cs dt a, wide cs dt b))))
  | Nx_compare (op, a, b), _ ->
      let (Dtype dt) = operand_dtype a in
      binary a b (fun a b ->
          let cs = [ a; b ] in
          lifted Type.bool cs
            (Nx.Op.eval (Compare (op, wide cs dt a, wide cs dt b))))
  | Nx_where (c, a, b), Column ty ->
      let (Dtype dt) = Option.get (dtype ty) in
      ternary c a b (fun c a b ->
          let cs = [ c; a; b ] in
          lifted ty cs
            (Nx.Op.eval (Where (wide cs Nx.bool c, wide cs dt a, wide cs dt b))))
  | Nx_cast (ty, a), _ ->
      let (Dtype dt) = Option.get (dtype ty) in
      let (Dtype from) = operand_dtype a in
      unary a (fun a ->
          lifted ty [ a ] (Nx.Op.eval (Convert (Cast, dt, tensor from a))))
  | Rows, _ -> fun env -> Reduce.rows env.frame.segments
  | Reduce (r, a), _ ->
      let a = compile st a and ty = type_of typing in
      fun env ->
        let rows = per_row env in
        let c, f =
          Reduce.reduce r ty env.frame.segments (fill rows (get rows a))
        in
        Option.iter (fun (i, why) -> fail env i (data why)) f;
        c
  | Over { by; order; e = x }, _ ->
      (* [x]'s nodes evaluate in the refined frame, so they share no slot with
         the nodes outside. *)
      let reduces = Expr.reduces x and scope = st.nodes in
      st.nodes <- [];
      let body = compile st x in
      st.nodes <- scope;
      let column n = Option.get (List.find_index (String.equal n) st.names) in
      let by = List.map column by in
      let order = List.map (fun (k : Order.t) -> (column k.name, k)) order in
      fun env ->
        let cs = env.frame.columns in
        let s =
          Reduce.refine env.frame.segments
            ~by:(List.map (Array.get cs) by)
            ~order:(List.map (fun (i, k) -> (cs.(i), k)) order)
        in
        if reduces then Reduce.broadcast s (get (per_segment env s) body)
        else get { env with frame = { env.frame with segments = s } } body
  | Shift (n, a), _ ->
      let a = compile st a in
      fun env -> Reduce.shift env.frame.segments n (fill env (get env a))
  | Rank a, _ ->
      let a = compile st a in
      fun env -> Reduce.rank env.frame.segments (fill env (get env a))
  | Cast (ty, a), _ ->
      unary_checked a (Kernels.cast (type_of (Expr.typing a)) (Any ty))
  | Text (op, a), _ -> textual a (Kernels.text op)
  | Calendar op, _ -> (
      let t e = type_of (Expr.typing e) in
      match op with
      | Add_span (a, d) -> binary_checked a d (Kernels.add (t a) (t d))
      | Diff (a, b) ->
          let m = meet (Expr.typing a) (Expr.typing b) in
          let wa = widen (t a) m and wb = widen (t b) m in
          let diff = Kernels.diff m in
          binary_checked a b (fun a b -> diff (wa a) (wb b))
      | Part (f, a) -> unary a (Kernels.field f (t a))
      | Floor (step, a) -> unary_checked a (Kernels.floor step (t a))
      | Offset (step, a) -> unary_checked a (Kernels.offset step (t a))
      | Parse_with (fmt, ty, a) -> textual a (Kernels.parse_with fmt (Any ty))
      | Format_with (fmt, a) -> unary a (Form.format_with fmt))
  (* Binding traces lifts away, an option stands only where [ocaml] reads it,
     and arithmetic and nx operations have a column type. *)
  | (Lift _ | Lift2 _ | Option _), _
  | ( (Int _ | Float _ | Nx_unary _ | Nx_binary _ | Nx_where _),
      (Extension _ | Value) ) ->
      invalid_arg (Format.asprintf "Eval: %a is not bound" Expr.pp e)

(* [ocaml st e] computes [e]'s values as OCaml values, [None] where [e] is
   null. *)
and ocaml : type a s. state -> (a, s) Expr.t -> env -> a option array =
 fun st e ->
  match Expr.node e with
  | Const v -> fun env -> Array.make env.extent (Some v)
  | App (f, a) ->
      let f = ocaml st f and a = ocaml st a in
      fun env ->
        let f = f env in
        let a = a env in
        each env (fun i ->
            match (f.(i), a.(i)) with Some f, Some a -> Some (f a) | _ -> None)
  | Option a ->
      let a = ocaml st a in
      fun env -> Array.map Option.some (a env)
  | Of_option a ->
      let a = ocaml st a in
      fun env -> Array.map Option.join (a env)
  | _ ->
      let n = compile st e and dec = decoder (Expr.typing e) in
      fun env -> decode env dec (fill env (get env n))

(* Calls *)

let env st frame =
  let env =
    {
      frame;
      extent = frame.rows;
      origin = Fun.id;
      slots = Array.make st.slots None;
      failure = ref None;
    }
  in
  if frame.reduced then per_segment env frame.segments else env

let outputs s os =
  let st = state s in
  let nodes = List.map (fun (_, Expr.Packed e) -> compile st e) os in
  fun frame ->
    let env = env st frame in
    let cs = List.map (fun n -> fill env (get env n)) nodes in
    (cs, !(env.failure))

let predicate s p =
  let st = state s in
  let n = compile st p in
  fun frame ->
    let env = env st frame in
    let m = truth (fill env (get env n)) in
    (m, !(env.failure))

let values s e =
  let st = state s in
  let vs = ocaml st e in
  fun frame ->
    let env = env st frame in
    let vs = vs env in
    (vs, !(env.failure))
