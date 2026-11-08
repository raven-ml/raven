(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [reduced] holds where the outputs reduce over the segments. Reductions run
   over [frames]: the segments, made when a reduction first reads them, or
   inside a window's reduction each row's window, where [frames] is not the
   frame of [segments]. A frame whose segments change takes [with_segments].

   [origin j] is the row of the enclosing run that the frame's row [j] came
   from, the row itself outside a list frame. A list frame, whose segments can
   be empty, knows each segment's row, [list_row]; any other frame's segment
   starts at its first row. *)
type frame = {
  columns : Column.t array;
  rows : int;
  segments : Reduce.segments;
  mutable frames : Reduce.frame option;
  reduced : bool;
  origin : int -> int;
  list_row : (int -> int) option;
}

let with_segments f segments =
  { f with segments; frames = None; list_row = None }

let frames f =
  match f.frames with
  | Some r -> r
  | None ->
      let r = Reduce.frame f.segments in
      f.frames <- Some r;
      r

let frame b =
  let rows = Table.rows b in
  {
    columns = Table.columns b;
    rows;
    segments = Reduce.one rows;
    frames = None;
    reduced = false;
    origin = Fun.id;
    list_row = None;
  }

let groups b segments = { (frame b) with segments; reduced = true }

type cause = Data of Error.t | Raised of exn * Printexc.raw_backtrace
type failure = { row : int; cause : cause }

(* Columns

   A value is a column of the frame's rows, or of one row where it computes from
   literals alone, which nx broadcasts. *)

type dtype = Dtype : ('a, 'b) Nx.dtype -> dtype

(* [dtype ty] is the dtype in which nx operations compute [ty]'s values: bytes
   for [bool], whose columns keep bits. *)
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

let boolean ?valid (x : Nx.host Nx.bit_t) = fixed (Type.Any Type.bool) ?valid x

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
  (Nx.slice [ Nx.R (0, n) ] w, Nx.slice [ Nx.R (n, Nx.dim 0 w) ] w)

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
let ordered (type a b) (op : Expr.compare) (x : (a, b, Nx.host) Nx.t)
    (y : (a, b, Nx.host) Nx.t) =
  let r =
    match op with
    | `Eq -> Nx.equal x y
    | `Ne -> Nx.not_equal x y
    | `Lt -> Nx.less x y
    | `Le -> Nx.less_equal x y
    | `Gt -> Nx.greater x y
    | `Ge -> Nx.greater_equal x y
  in
  if not (Nx.Dtype.is Float (Nx.dtype x)) then r
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
  let valid = valid [ a; b ] in
  let masked r = Option.fold ~none:r ~some:(Nx.logical_and r) valid in
  let r =
    match (Column.data a, Column.data b) with
    | Fixed (P x), Fixed q when Nx.ndim x = 1 ->
        Nx.cast Nx.bit (ordered op x (Nx.unpack (Nx.dtype x) q))
    (* Text is compared with one row's bytes, or by words, and a batch of one
       row takes either way: both are masked under the nulls, so that the bits
       there are the same whatever the batches. *)
    | Bytes r, Bytes one when Column.length b = 1 ->
        masked (Strings.compare ~by:(text_name op) op r one)
    | Bytes one, Bytes r when Column.length a = 1 ->
        masked (Strings.compare ~by:(text_name op) (Expr.mirror op) r one)
    | _ ->
        let use : Key.use =
          match op with `Eq | `Ne -> Identity | `Lt | `Le | `Gt | `Ge -> Order
        in
        (* A compound value's word is relative to the rows computed with it, so
           the comparison is masked under the nulls: there it would depend on
           the batch. *)
        let x, y = words use a b in
        masked (Nx.cast Nx.bit (ordered op x y))
  in
  boolean ?valid r

(* [held c] is where [c] holds a value, as a condition. *)
let held c = Option.map (Nx.cast Nx.bool) (Column.validity c)

(* [truth c] is where the boolean column [c] is [true] and not null, as bits. *)
let truth c =
  let x = tensor Nx.bit c in
  Option.fold ~none:x ~some:(Nx.logical_and x) (Column.validity c)

(* Kleene logic, on the bits of the values [x] and [y] and of the validities
   [va] and [vb]. A conjunction is valid where both operands are, or where one
   is a valid [false]; a disjunction, where both are, or where one is a valid
   [true]. With [decides] the test for that deciding value, this is [va] and
   ([vb] or [decides x]), or [vb] and [decides y]. A missing validity is all
   [true], which leaves [va] or [decides y] where [b] has none. A valid row has
   both operands valid or one that decides it, so the operands' values combined
   as they are, those under nulls included, are its Kleene value: [a && true]
   and [a || false] are [a], under its nulls too. *)
let logic (op : Expr.logic) a b =
  let x = tensor Nx.bit a and y = tensor Nx.bit b in
  let r, decides =
    match op with
    | And -> (Nx.logical_and x y, Nx.logical_not)
    | Or -> (Nx.logical_or x y, Fun.id)
  in
  let valid =
    match (Column.validity a, Column.validity b) with
    | None, None -> None
    | Some va, None -> Some (Nx.logical_or va (decides y))
    | None, Some vb -> Some (Nx.logical_or vb (decides x))
    | Some va, Some vb ->
        let by_a = Nx.logical_and va (Nx.logical_or vb (decides x)) in
        Some (Nx.logical_or by_a (Nx.logical_and vb (decides y)))
  in
  boolean ?valid r

let is_null c =
  match Column.validity c with
  | Some v -> boolean (Nx.logical_not v)
  | None -> boolean (Nx.zeros Nx.bit [| Column.length c |])

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

(* Up to this many texts, [is_in] compares a column's rows with them in one
   read of its bytes, rather than coding the rows with them, which sorts the
   rows and passes over them tens of times. *)
let compared_texts = 8

(* [is_in vs] finds the words of a column's rows among the sorted words of the
   values [vs]: words of fixed-width values sorted once, and codes of compound
   values computed with the column's rows. A few texts are compared with the
   rows' bytes instead. *)
let is_in vs =
  let m = Column.length vs in
  let held hit a =
    boolean
      (Option.fold ~none:hit ~some:(Nx.logical_and hit) (Column.validity a))
  in
  let found s x a =
    let i = Nx.searchsorted ~side:`Left s x in
    let hit = Nx.equal (Nx.take (Nx.minimum_s i (Int64.of_int (m - 1))) s) x in
    held (Nx.cast Nx.bit hit) a
  in
  let coded a =
    let x, w = words Identity a vs in
    found (fst (Nx.sort w)) x a
  in
  match Column.data vs with
  | _ when m = 0 -> fun a -> boolean (Nx.zeros Nx.bit [| Column.length a |])
  | Fixed (P x) when Nx.ndim x = 1 ->
      let s, _ = Nx.sort (Key.value Order vs) in
      fun a -> found s (Key.value Order a) a
  | Bytes texts when m <= compared_texts && Column.known_zero vs -> (
      fun a ->
        match Column.data a with
        | Bytes r -> held (Strings.member ~by:"Expr.is_in" r texts) a
        | _ -> coded a)
  | _ -> coded

(* [cut edges] is the bin of each row of a column among [edges], which binding
   checked ascend strictly: from the number [r] of edges at or below the row's
   word, the bin is [r - 1], and the row is null where [r] is [0] or every edge.
   NaN's word is above every other, so a NaN row is null too. Words of
   fixed-width edges are in order already, and codes of compound edges are
   computed with the column's rows. *)
let cut edges =
  let m = Int64.of_int (Column.length edges) in
  let bin s x a =
    let r = Nx.searchsorted ~side:`Right s x in
    let inside = Nx.logical_and (Nx.greater_equal_s r 1L) (Nx.less_s r m) in
    let inside = Nx.cast Nx.bit inside in
    let valid =
      Option.fold ~none:inside ~some:(Nx.logical_and inside) (Column.validity a)
    in
    fixed (Any Type.int64) ~valid (Nx.sub_s r 1L)
  in
  match Column.data edges with
  | Fixed (P x) when Nx.ndim x = 1 ->
      let s = Key.value Order edges in
      fun a -> bin s (Key.value Order a) a
  | _ ->
      fun a ->
        let x, s = words Order a edges in
        bin s x a

(* [lifted ty cs x] is the column of [ty] stored as [x], which an nx operation
   computes over the columns [cs]: null where one of them is. A [bool] [x] is
   packed into bits. *)
let lifted : type a b c.
    a Type.t -> Column.t list -> (b, c, Nx.host) Nx.t -> Column.t =
 fun ty cs x ->
  match ty with
  | Bool -> boolean ?valid:(valid cs) (Nx.cast Nx.bit x)
  | _ -> fixed (Any ty) ?valid:(valid cs) x

(* [width cs] is the rows that the columns [cs] broadcast to. *)
let width cs =
  List.fold_left
    (fun n c -> if Column.length c = 1 then n else Column.length c)
    1 cs

(* [map ty p cs] is the column of [ty] that the program [p] of one output
   computes in one pass over the columns [cs], each read in its operand's dtype
   and broadcast to the rows: the map a lift recorded, with nx's semantics. *)
let map ty (p : Nx_kernel.Prog.t) cs =
  let n = width cs in
  let load j c =
    let (Nx.Dtype.Any dt) = Iarray.get p.ins j in
    Nx.Prim.Plain (Nx.broadcast_to [| n |] (tensor dt c))
  in
  let (Any out) = Nx_kernel.Prog.dtype p (Iarray.get p.outs 0) in
  let y, () =
    Nx.Prim.eval ~by:"Talon.Expr.nx"
      (Map
         {
           layout = Nx_array.Layout.contiguous [| n |];
           prog = p;
           outs = [ out ];
           loads = Iarray.of_list (List.mapi load cs);
           at = None;
         })
  in
  lifted ty cs y

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
  parts : Reduce.parts option array;
  failure : failure option ref;
}

let per_row env =
  { env with extent = env.frame.rows; origin = env.frame.origin }

(* [over_segments env frame] evaluates one value per segment of [frame]. *)
let over_segments env frame =
  let first = lazy (Nx.to_array (Reduce.first frame.segments)) in
  let origin =
    match frame.list_row with
    | Some row -> row
    | None -> fun i -> frame.origin (Int64.to_int (Lazy.force first).(i))
  in
  { env with frame; extent = Reduce.count frame.segments; origin }

(* [per_segment env ~same segments] is [env] over [segments], which cut the
   frame as its own do if [same]. *)
let per_segment env ~same segments =
  let frame =
    if same then { env.frame with segments; frames = None }
    else with_segments env.frame segments
  in
  over_segments env frame

(* A node that a reduction reads gets a [part], which holds, for a frame, the
   parts its reductions share. *)
type node = { eval : env -> Column.t; mutable slot : int; mutable part : int }

type state = {
  mutable names : string list;
  mutable nodes : (Expr.packed * node) list;
  mutable slots : int;
  mutable parts : int;
}

let state s = { names = Schema.names s; nodes = []; slots = 0; parts = 0 }

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

(* A list frame

   [list_frame] lays out the frame of [per_list]'s lists: one row per position
   of the lists of each row that has them all, valid and of one length, the
   segment of a row being its list. [e] elements cost [e] words of positions
   per list that its child does not hold whole and per outer column read. *)

let list_data c =
  match Column.data c with
  | List { offsets; child } -> (offsets, child)
  | _ -> invalid_arg "Eval: per_list reads a column that is not a list"

let lengths n offsets =
  Nx.sub
    (Nx.slice [ Nx.R (1, n + 1) ] offsets)
    (Nx.slice [ Nx.R (0, n) ] offsets)

(* [lockstep env names ls valid] is [valid] without the rows whose lists [ls]
   differ in length, compared where all are valid. The first such row fails,
   and the others are past it. *)
let lockstep env names ls valid =
  match ls with
  | [] -> valid
  | l0 :: rest ->
      let differ =
        List.fold_left
          (fun acc l -> Nx.logical_or acc (Nx.not_equal l l0))
          (Nx.zeros_like valid) rest
      in
      let bad = Nx.positions (Nx.logical_and valid differ) in
      if Nx.dim 0 bad = 0 then valid
      else begin
        let i = Int64.to_int (Nx.item [ 0 ] bad) in
        let pp ppf (n, l) =
          Format.fprintf ppf "%a has %Ld" Type.pp_quoted n (Nx.item [ i ] l)
        in
        let why =
          Format.asprintf
            "per_list reads its lists in lockstep, and their lengths differ: %a"
            (Format.pp_print_list
               ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
               pp)
            (List.combine names ls)
        in
        fail env i (data why);
        Nx.logical_and valid (Nx.logical_not differ)
      end

(* [list_frame env names cs reads] is the frame of the lists [cs] at the rows
   of [env] that have a value, and [slot], the place of each row's frame among
   them or [-1]: [None] if every row has one. [reads] says where each column of
   the frame comes from. *)
let list_frame env names cs reads =
  let n = env.extent in
  let data = List.map list_data cs in
  let ls = List.map (fun (o, _) -> lengths n o) data in
  let valid =
    List.fold_left
      (fun acc c ->
        match Column.validity c with
        | None -> acc
        | Some v -> Nx.logical_and acc (Nx.cast Nx.bool v))
      (Nx.ones Nx.bool [| n |])
      cs
  in
  let live = lockstep env names ls valid in
  let rows = Nx.positions live in
  let m = Nx.dim 0 rows in
  let sizes = Nx.take rows (List.hd ls) in
  let ids = Nx.positions sizes in
  let e = Nx.dim 0 ids in
  let outer = Nx.take ids rows in
  let start = Nx.sub (Nx.cumsum sizes) sizes in
  let place = lazy (Nx.sub (Nx.arange Nx.int64 0 e 1) (Nx.take ids start)) in
  (* A child read whole needs no gather: every row live, offsets from 0 and
     the child no longer than the lists. *)
  let element (offsets, child) =
    if
      m = n
      && Int64.equal (Nx.item [ 0 ] offsets) 0L
      && Int64.equal (Nx.item [ n ] offsets)
           (Int64.of_int (Column.length child))
    then child
    else
      let first = Nx.take outer offsets in
      Column.gather (Nx.add first (Lazy.force place)) child
  in
  let elements = lazy (Array.of_list (List.map element data)) in
  let column = function
    | `Element k -> (Lazy.force elements).(k)
    | `Outer i -> Column.gather outer env.frame.columns.(i)
  in
  let row_of a = lazy (Nx.to_array a) in
  let outer_rows = row_of outer and live_rows = row_of rows in
  let frame =
    {
      columns = Array.of_list (List.map column reads);
      rows = e;
      segments = Reduce.of_ids ~segments:m ~sizes ids;
      frames = None;
      reduced = true;
      origin = (fun j -> env.origin (Int64.to_int (Lazy.force outer_rows).(j)));
      list_row =
        Some (fun i -> env.origin (Int64.to_int (Lazy.force live_rows).(i)));
    }
  in
  let slot =
    if m = n then None
    else
      let count = Nx.sub_s (Nx.cumsum (Nx.cast Nx.int64 live)) 1L in
      Some (Nx.where live count (Nx.full Nx.int64 [| n |] (-1L)))
  in
  (frame, slot)

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
      let n = { eval = lower st e; slot = -1; part = -1 } in
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
          let x = Nx.logical_not (tensor Nx.bit a) in
          Column.with_data (Any Type.bool) (Fixed (P x)) a)
  | If (c, a, b), _ ->
      let t = type_of typing in
      let wa = widen (type_of (Expr.typing a)) t
      and wb = widen (type_of (Expr.typing b)) t in
      ternary c a b (fun c a b ->
          choose (Nx.cast Nx.bool (truth c)) (wa a) (wb b))
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
  | Cut (edges, a), _ ->
      let edges = column (Expr.typing a) (Array.map Option.some edges) in
      unary a (cut edges)
  | Store (ty, a), _ -> unary a (widen (type_of (Expr.typing a)) (Any ty))
  | (App _ | Of_option _), _ ->
      let vs = ocaml st e in
      fun env -> store env typing (vs env)
  | Nx_map (p, es), Column ty ->
      let es = List.map (fun (Expr.Packed e) -> compile st e) es in
      fun env ->
        let cs = List.map (get env) es in
        map ty p cs
  | Rows, _ ->
      fun env ->
        Reduce.fixed (Any Type.int64) ((frames env.frame).sizes ())
  | Reduce (r, a), _ ->
      let a = compile st a and ty = type_of typing in
      if a.part < 0 then begin
        a.part <- st.parts;
        st.parts <- st.parts + 1
      end;
      fun env ->
        let p =
          match env.parts.(a.part) with
          | Some p -> p
          | None ->
              let rows = per_row env in
              let c = fill rows (get rows a) in
              let p = Reduce.parts (frames env.frame) c in
              env.parts.(a.part) <- Some p;
              p
        in
        let c, f = Reduce.reduce r ty p in
        Option.iter (fun (i, why) -> fail env i (data why)) f;
        c
  | Over { by; order; e = x }, _ ->
      (* [x]'s nodes evaluate in the refined frame, so they share no slot with
         the nodes outside. *)
      let same = List.is_empty by in
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
        let refined = per_segment env ~same s in
        if reduces then Reduce.broadcast s (get refined body)
        else get { env with frame = refined.frame } body
  | Window { on; before; after; exclude; r }, _ ->
      (* [r]'s nodes evaluate over the windows, so they share no slot with the
         nodes outside. Only rows with a window have one, and a window fails
         at its row. *)
      let operand : type b s. (b, s) Expr.t option -> node option =
       fun e -> Option.map (compile st) e
      in
      let on = operand on and before = operand before in
      let after = operand after in
      let scope = st.nodes in
      st.nodes <- [];
      let body = compile st r in
      st.nodes <- scope;
      fun env ->
        let column = Option.map (fun n -> fill env (get env n)) in
        let w =
          Window.v env.frame.segments ~on:(column on) ~before:(column before)
            ~after:(column after) ~exclude
        in
        let frame = { env.frame with frames = Some (Window.frame w) } in
        let extent = Window.windows w
        and origin i = frame.origin (Window.row w i) in
        let env = { env with frame; extent; origin } in
        Window.to_rows w (fill env (get env body))
  | Per_list (os, r), _ ->
      (* [r]'s nodes evaluate over the elements, so they share no slot with
         the nodes outside, and read the columns [reads] by their names there.
         A name reads its output first: a conjunct moved below a select can
         sit over an input with a column of the output's name. *)
      let outs = Expr.list_outs os in
      let names = List.map fst outs in
      let lists = List.map (fun (_, Expr.Packed e) -> compile st e) outs in
      let reads = Expr.reads r in
      let source n =
        match List.find_index (String.equal n) names with
        | Some k -> `Element k
        | None ->
            `Outer (Option.get (List.find_index (String.equal n) st.names))
      in
      let sources = List.map source reads in
      let scope = st.nodes and outer_names = st.names in
      st.nodes <- [];
      st.names <- reads;
      let body = compile st r in
      st.nodes <- scope;
      st.names <- outer_names;
      fun env ->
        let cs = List.map (fun n -> fill env (get env n)) lists in
        let frame, slot = list_frame env names cs sources in
        let m = Reduce.count frame.segments in
        let origin = Option.get frame.list_row in
        let env = { env with frame; extent = m; origin } in
        let c = fill env (get env body) in
        Option.fold ~none:c ~some:(fun slot -> Reduce.rows_at slot c) slot
  | Shift (n, a), _ ->
      let a = compile st a in
      fun env -> Reduce.shift env.frame.segments n (fill env (get env a))
  | Rank a, _ ->
      let a = compile st a in
      fun env -> Reduce.rank env.frame.segments (fill env (get env a))
  | Cast (ty, a), _ ->
      unary_checked a (Kernels.cast (type_of (Expr.typing a)) (Any ty))
  | Text (op, a), _ -> textual a (Kernels.text op)
  | Concat (sep, es), _ ->
      let operand e =
        (compile st e, widen (type_of (Expr.typing e)) (Any Type.string))
      in
      let es = List.map operand es in
      fun env -> Kernels.concat sep (List.map (fun (e, w) -> w (get env e)) es)
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
  | (Int _ | Float _ | Nx_map _), (Extension _ | Value) ->
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
      parts = Array.make st.parts None;
      failure = ref None;
    }
  in
  if frame.reduced then over_segments env frame else env

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
