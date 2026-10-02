(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Format.asprintf
let err fmt = Format.kasprintf invalid_arg fmt

(* Types *)

type row
type agg
type fn = { f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }
type fn2 = { f2 : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }
type arith = Add | Sub | Mul | Div | Mod | Pow
type compare = [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ]
type logic = And | Or

type (_, _) reduction =
  | Count : ('a, int) reduction
  | Sum : ('a, 'a) reduction
  | Min : ('a, 'a) reduction
  | Max : ('a, 'a) reduction
  | First : ('a, 'a) reduction
  | Last : ('a, 'a) reduction
  | Only : ('a, 'a) reduction
  | Mean : ('a, float) reduction
  | Std : ('a, float) reduction
  | Var : ('a, float) reduction
  | Median : ('a, float) reduction
  | Quantile : float -> ('a, float) reduction
  | Ewm : float -> ('a, float) reduction
  | N_unique : ('a, int) reduction
  | Arg_min : ('a, int) reduction
  | Arg_max : ('a, int) reduction
  | Collect : ('a, 'a array) reduction

type ('e, 's) ext = {
  type_ : Type.ext Type.t;
  storage : 's Type.t;
  ordered : bool;
  dec : 's -> 'e;
  enc : 'e -> 's;
}

type pattern = Strings.pattern =
  | Literal of string
  | Prefix of string
  | Suffix of string
  | Pieces of string list

type field =
  [ `Year
  | `Month
  | `Day
  | `Hour
  | `Minute
  | `Second
  | `Nanosecond
  | `Weekday
  | `Yearday ]

type policy = [ `Earlier | `Later | `Null | `Fail ]

type ('a, 's) t = { id : int; node : 'a node; typing : 'a typing option }
and 's out = out_repr

and out_repr =
  | Named of string * packed
  | Keep of Sel.t
  | Across : 'a Kind.t * Sel.t * (string -> ('a, row) t -> out_repr) -> out_repr
  | Each : Sel.t * 's column -> out_repr
  | Unpack of (Record.t, row) t

and 's column = { column : 'a. string -> ('a, row) t -> 's out }
and packed = Packed : ('a, 's) t -> packed

and 'a typing =
  | Column : 'a Type.t -> 'a typing
  | Extension : ('a, 's) ext -> 'a typing
  | Value : 'a typing

and 'a node =
  | Handle : 'a Kind.t * string -> 'a node
  | Ext_handle : ('a, 's) ext * string -> 'a node
  | Read : 'a Type.t * string -> 'a node
  | Lit : 'a Kind.t * 'a -> 'a node
  | Null : 'a node
  | Rows : int node
  | Int : arith * (int, 's0) t * (int, 's1) t -> int node
  | Float : arith * (float, 's0) t * (float, 's1) t -> float node
  | Compare : compare * ('a, 's0) t * ('a, 's1) t -> bool node
  | Logic : logic * (bool, 's0) t * (bool, 's1) t -> bool node
  | Not : (bool, 's) t -> bool node
  | If : (bool, 's0) t * ('a, 's1) t * ('a, 's2) t -> 'a node
  | Is_null : ('a, 's) t -> bool node
  | Coalesce : ('a, 's) t list -> 'a node
  | Is_in : 'a list * ('a, 's) t -> bool node
  | Cut : 'a array * ('a, 's) t -> int node
  | Cast : 'a Type.t * ('b, 's) t -> 'a node
  | Lift : fn * ('a, 's) t -> 'a node
  | Lift2 : fn2 * ('a, 's0) t * ('a, 's1) t -> 'a node
  | Nx_unary : Nx_backend.unary * ('a, 's) t -> 'a node
  | Nx_binary : Nx_backend.binary * ('a, 's0) t * ('a, 's1) t -> 'a node
  | Nx_compare : Nx_backend.compare * ('a, 's0) t * ('a, 's1) t -> bool node
  | Nx_where : (bool, 's0) t * ('a, 's1) t * ('a, 's2) t -> 'a node
  | Nx_cast : 'a Type.t * ('b, 's) t -> 'a node
  | Reduce : ('a, 'b) reduction * ('a, 's) t -> 'b node
  | Over : { by : string list; order : Order.t list; e : ('a, 's) t } -> 'a node
  | Rolling : Window.t * ('a, 's) t -> 'a node
  | Shift : int * ('a, 's) t -> 'a node
  | Rank : ('a, 's) t -> int node
  | Const : 'a -> 'a node
  | App : ('a -> 'b, 's0) t * ('a, 's1) t -> 'b node
  | Option : ('a, 's) t -> 'a option node
  | Of_option : ('a option, 's) t -> 'a node
  | Store : 'a Type.t * ('a, 's) t -> 'a node
  | Batch :
      (('a, 'b) Nx.t -> ('c, 'd) Nx.t) * (('a, 'b) Nx.t, 's) t
      -> ('c, 'd) Nx.t node
  | Record_outs : 's out list -> Record.t node
  | Fields : (string * packed) list -> Record.t node
  | Field : 'a Kind.t * string * (Record.t, 's) t -> 'a node
  | Storage : ('e, 'a) ext * ('e, 's) t -> 'a node
  | Wrap : ('a, 'st) ext * ('st, 's) t -> 'a node
  | Text : 'a text_op * (string, 's) t -> 'a node
  | Calendar : 'a calendar_op -> 'a node

and 'a text_op =
  | Length : int text_op
  | Slice : { offset : int; length : int } -> string text_op
  | Lower : string text_op
  | Upper : string text_op
  | Matches : pattern -> bool text_op
  | Parse : 'a Type.t -> 'a text_op

and 'a calendar_op =
  | Add_span : ('a, 's0) t * (Time.span, 's1) t -> 'a calendar_op
  | Diff : ('a, 's0) t * ('a, 's1) t -> Time.span calendar_op
  | Part : field * Tz.zone option * ('a, 's) t -> int calendar_op
  | Floor : Tz.zone option * Time.step * ('a, 's) t -> 'a calendar_op
  | Offset : Tz.zone option * Time.step * ('a, 's) t -> 'a calendar_op
  | Localize : {
      zone : Tz.zone;
      ambiguous : policy;
      gap : policy;
      a : (Time.instant, 's) t;
    }
      -> Time.instant calendar_op
  | Windows : {
      zone : Tz.zone option;
      every : Time.step;
      period : Time.step;
      a : ('a, 's) t;
    }
      -> 'a array calendar_op
  | Parse_with : string * 'a Type.t * (string, 's) t -> 'a calendar_op
  | Format_with : string * ('a, 's) t -> string calendar_op

(* Operands *)

(* [operands n] is [n]'s operands, in order. *)
let operands : type a. a node -> packed list = function
  | Handle _ | Ext_handle _ | Read _ | Lit _ | Null | Rows | Const _ -> []
  | Int (_, a, b) -> [ Packed a; Packed b ]
  | Float (_, a, b) -> [ Packed a; Packed b ]
  | Compare (_, a, b) -> [ Packed a; Packed b ]
  | Logic (_, a, b) -> [ Packed a; Packed b ]
  | Nx_binary (_, a, b) -> [ Packed a; Packed b ]
  | Nx_compare (_, a, b) -> [ Packed a; Packed b ]
  | Lift2 (_, a, b) -> [ Packed a; Packed b ]
  | App (f, a) -> [ Packed f; Packed a ]
  | If (c, a, b) -> [ Packed c; Packed a; Packed b ]
  | Nx_where (c, a, b) -> [ Packed c; Packed a; Packed b ]
  | Coalesce es -> List.map (fun e -> Packed e) es
  | Not a -> [ Packed a ]
  | Is_null a -> [ Packed a ]
  | Rank a -> [ Packed a ]
  | Option a -> [ Packed a ]
  | Of_option a -> [ Packed a ]
  | Is_in (_, a) -> [ Packed a ]
  | Cut (_, a) -> [ Packed a ]
  | Lift (_, a) -> [ Packed a ]
  | Nx_unary (_, a) -> [ Packed a ]
  | Shift (_, a) -> [ Packed a ]
  | Cast (_, a) -> [ Packed a ]
  | Nx_cast (_, a) -> [ Packed a ]
  | Reduce (_, a) -> [ Packed a ]
  | Rolling (_, a) -> [ Packed a ]
  | Store (_, a) -> [ Packed a ]
  | Batch (_, a) -> [ Packed a ]
  | Field (_, _, r) -> [ Packed r ]
  | Storage (_, a) -> [ Packed a ]
  | Wrap (_, a) -> [ Packed a ]
  | Text (_, a) -> [ Packed a ]
  | Over { e; _ } -> [ Packed e ]
  (* An unbound node, which no bound expression holds. *)
  | Record_outs _ -> []
  | Fields fs -> List.map snd fs
  | Calendar op -> (
      match op with
      | Add_span (a, d) -> [ Packed a; Packed d ]
      | Diff (a, b) -> [ Packed a; Packed b ]
      | Part (_, _, a) -> [ Packed a ]
      | Floor (_, _, a) -> [ Packed a ]
      | Offset (_, _, a) -> [ Packed a ]
      | Localize { a; _ } -> [ Packed a ]
      | Windows { a; _ } -> [ Packed a ]
      | Parse_with (_, _, a) -> [ Packed a ]
      | Format_with (_, a) -> [ Packed a ])

(* Names *)

let reduction_name : type a b. (a, b) reduction -> string = function
  | Count -> "count"
  | Sum -> "sum"
  | Min -> "min"
  | Max -> "max"
  | First -> "first"
  | Last -> "last"
  | Only -> "only"
  | Mean -> "mean"
  | Std -> "std"
  | Var -> "var"
  | Median -> "median"
  | Quantile _ -> "quantile"
  | Ewm _ -> "ewm"
  | N_unique -> "n_unique"
  | Arg_min -> "arg_min"
  | Arg_max -> "arg_max"
  | Collect -> "collect"

let text_op_name : type a. a text_op -> string = function
  | Length -> "Str.length"
  | Slice _ -> "Str.slice"
  | Lower -> "Str.lower"
  | Upper -> "Str.upper"
  | Matches _ -> "Str.matches"
  | Parse _ -> "Str.parse"

let calendar_op_name : type a. a calendar_op -> string = function
  | Add_span _ -> "Temporal.add"
  | Diff _ -> "Temporal.diff"
  | Part _ -> "Temporal.field"
  | Floor _ -> "Temporal.floor"
  | Offset _ -> "Temporal.offset"
  | Localize _ -> "Temporal.localize"
  | Windows _ -> "Temporal.windows"
  | Parse_with _ -> "Temporal.parse"
  | Format_with _ -> "Temporal.format"

(* Hash-consing *)

(* Functions, [const] values, declarations and zones compare physically, at
   whatever types they are held. *)
let same_value x y = Obj.repr x == Obj.repr y
let same a b = Int.equal a.id b.id

let rec lit_equal : type a. a Kind.t -> a -> a -> bool =
 fun k v0 v1 ->
  match k with
  | Float -> Int64.equal (Int64.bits_of_float v0) (Int64.bits_of_float v1)
  | Int -> Int.equal v0 v1
  | Bool -> Bool.equal v0 v1
  | String -> String.equal v0 v1
  | Binary -> String.equal (v0 :> string) (v1 :> string)
  | Date -> Time.Date.equal v0 v1
  | Instant -> Time.equal v0 v1
  | Span -> Time.Span.equal v0 v1
  | List k ->
      Array.length v0 = Array.length v1 && Array.for_all2 (lit_equal k) v0 v1
  | Record | Tensor _ -> v0 == v1
  | Ext -> ( match v0 with _ -> .)

let typing_equal : type a b. a typing -> b typing -> bool =
 fun t0 t1 ->
  match (t0, t1) with
  | Column ty0, Column ty1 -> Type.equal ty0 ty1
  | Extension d0, Extension d1 -> same_value d0 d1
  | Value, Value -> true
  | (Column _ | Extension _ | Value), _ -> false

(* [values_equal a0 vs0 a1 vs1] is [true] iff the operands [a0] and [a1], one
   expression, have a column type at which the values [vs0] and [vs1] are
   equal. *)
let values_equal : type a b s0 s1.
    (a, s0) t -> a list -> (b, s1) t -> b list -> bool =
 fun a0 vs0 a1 vs1 ->
  match (a0.typing, a1.typing) with
  | Some (Column ty0), Some (Column ty1) -> (
      match Kind.equal_witness (Type.kind ty0) (Type.kind ty1) with
      | Some Equal ->
          let cmp = Type.compare_value ty0 in
          List.equal (fun v0 v1 -> Int.equal (cmp v0 v1) 0) vs0 vs1
      | None -> false)
  | _ -> false

let step_equal s0 s1 =
  match (s0, s1) with
  | Time.Months n0, Time.Months n1
  | Time.Weeks n0, Time.Weeks n1
  | Time.Days n0, Time.Days n1 ->
      Int.equal n0 n1
  | Time.Exact d0, Time.Exact d1 -> Time.Span.equal d0 d1
  | (Time.Months _ | Time.Weeks _ | Time.Days _ | Time.Exact _), _ -> false

let pattern_equal p0 p1 =
  match (p0, p1) with
  | Literal s0, Literal s1 | Prefix s0, Prefix s1 | Suffix s0, Suffix s1 ->
      String.equal s0 s1
  | Pieces ss0, Pieces ss1 -> List.equal String.equal ss0 ss1
  | (Literal _ | Prefix _ | Suffix _ | Pieces _), _ -> false

let zone_equal z0 z1 =
  match (z0, z1) with
  | None, None -> true
  | Some z0, Some z1 -> z0 == z1
  | (None | Some _), _ -> false

(* Reductions of one name differ only by a quantile's probability or an ewm's
   alpha. *)
let reduction_equal : type a b c d. (a, b) reduction -> (c, d) reduction -> bool
    =
 fun r0 r1 ->
  match (r0, r1) with
  | Quantile p0, Quantile p1 -> Float.equal p0 p1
  | Ewm a0, Ewm a1 -> Float.equal a0 a1
  | _ -> String.equal (reduction_name r0) (reduction_name r1)

let text_op_equal : type a b. a text_op -> b text_op -> bool =
 fun op0 op1 ->
  match (op0, op1) with
  | Length, Length | Lower, Lower | Upper, Upper -> true
  | Slice s0, Slice s1 ->
      Int.equal s0.offset s1.offset && Int.equal s0.length s1.length
  | Matches p0, Matches p1 -> pattern_equal p0 p1
  | Parse ty0, Parse ty1 -> Type.equal ty0 ty1
  | (Length | Slice _ | Lower | Upper | Matches _ | Parse _), _ -> false

(* [calendar_equal op0 op1] is [true] iff [op0] and [op1] are one operation with
   equal attributes, whatever their operands. *)
let calendar_equal : type a b. a calendar_op -> b calendar_op -> bool =
 fun op0 op1 ->
  match (op0, op1) with
  | Add_span _, Add_span _ | Diff _, Diff _ -> true
  | Part (f0, z0, _), Part (f1, z1, _) -> f0 = f1 && zone_equal z0 z1
  | Floor (z0, s0, _), Floor (z1, s1, _) | Offset (z0, s0, _), Offset (z1, s1, _)
    ->
      zone_equal z0 z1 && step_equal s0 s1
  | Localize l0, Localize l1 ->
      l0.zone == l1.zone && l0.ambiguous = l1.ambiguous && l0.gap = l1.gap
  | Windows w0, Windows w1 ->
      zone_equal w0.zone w1.zone
      && step_equal w0.every w1.every
      && step_equal w0.period w1.period
  | Parse_with (f0, ty0, _), Parse_with (f1, ty1, _) ->
      String.equal f0 f1 && Type.equal ty0 ty1
  | Format_with (f0, _), Format_with (f1, _) -> String.equal f0 f1
  | ( ( Add_span _ | Diff _ | Part _ | Floor _ | Offset _ | Localize _
      | Windows _ | Parse_with _ | Format_with _ ),
      _ ) ->
      false

(* [node_equal n0 n1] is [true] iff [n0] and [n1] are one operation with equal
   attributes on the same operands. The operands are compared first: they type
   the values of [is_in] and [cut]. Unbound nodes, which binding rewrites, are
   never hash-consed. *)
let node_equal : type a b. a node -> b node -> bool =
 fun n0 n1 ->
  List.equal (fun (Packed a) (Packed b) -> same a b) (operands n0) (operands n1)
  &&
  match (n0, n1) with
  | Handle (k0, c0), Handle (k1, c1) ->
      String.equal c0 c1 && Option.is_some (Kind.equal_witness k0 k1)
  | Ext_handle (d0, c0), Ext_handle (d1, c1) ->
      String.equal c0 c1 && same_value d0 d1
  | Read (ty0, c0), Read (ty1, c1) -> String.equal c0 c1 && Type.equal ty0 ty1
  | Lit (k0, v0), Lit (k1, v1) -> (
      match Kind.equal_witness k0 k1 with
      | Some Equal -> lit_equal k0 v0 v1
      | None -> false)
  | Shift (i0, _), Shift (i1, _) -> Int.equal i0 i1
  | Int (o0, _, _), Int (o1, _, _) | Float (o0, _, _), Float (o1, _, _) ->
      o0 = o1
  | Compare (o0, _, _), Compare (o1, _, _) -> o0 = o1
  | Logic (o0, _, _), Logic (o1, _, _) -> o0 = o1
  | Nx_unary (o0, _), Nx_unary (o1, _) -> o0 = o1
  | Nx_binary (o0, _, _), Nx_binary (o1, _, _) -> o0 = o1
  | Nx_compare (o0, _, _), Nx_compare (o1, _, _) -> o0 = o1
  | Is_in (vs0, a0), Is_in (vs1, a1) ->
      same_value vs0 vs1 || values_equal a0 vs0 a1 vs1
  | Cut (es0, a0), Cut (es1, a1) ->
      same_value es0 es1
      || values_equal a0 (Array.to_list es0) a1 (Array.to_list es1)
  | Cast (ty0, _), Cast (ty1, _)
  | Nx_cast (ty0, _), Nx_cast (ty1, _)
  | Store (ty0, _), Store (ty1, _) ->
      Type.equal ty0 ty1
  | Lift (f0, _), Lift (f1, _) -> same_value f0.f f1.f
  | Lift2 (f0, _, _), Lift2 (f1, _, _) -> same_value f0.f2 f1.f2
  | Reduce (r0, _), Reduce (r1, _) -> reduction_equal r0 r1
  | Over o0, Over o1 ->
      List.equal String.equal o0.by o1.by
      && List.equal Order.equal o0.order o1.order
  | Rolling (w0, _), Rolling (w1, _) -> Window.equal w0 w1
  | Const v0, Const v1 -> same_value v0 v1
  | Batch (f0, _), Batch (f1, _) -> same_value f0 f1
  | Fields fs0, Fields fs1 ->
      List.equal (fun (n0, _) (n1, _) -> String.equal n0 n1) fs0 fs1
  | Field (k0, n0, _), Field (k1, n1, _) ->
      String.equal n0 n1 && Option.is_some (Kind.equal_witness k0 k1)
  | Storage (d0, _), Storage (d1, _) -> same_value d0 d1
  | Wrap (d0, _), Wrap (d1, _) -> same_value d0 d1
  | Text (op0, _), Text (op1, _) -> text_op_equal op0 op1
  | Calendar op0, Calendar op1 -> calendar_equal op0 op1
  | Null, Null
  | Rows, Rows
  | Not _, Not _
  | If _, If _
  | Is_null _, Is_null _
  | Coalesce _, Coalesce _
  | Nx_where _, Nx_where _
  | Rank _, Rank _
  | App _, App _
  | Option _, Option _
  | Of_option _, Of_option _ ->
      true
  | ( ( Handle _ | Ext_handle _ | Read _ | Lit _ | Null | Rows | Int _ | Float _
      | Compare _ | Logic _ | Not _ | If _ | Is_null _ | Coalesce _ | Is_in _
      | Cut _ | Cast _ | Lift _ | Lift2 _ | Nx_unary _ | Nx_binary _
      | Nx_compare _ | Nx_where _ | Nx_cast _ | Reduce _ | Over _ | Rolling _
      | Shift _ | Rank _ | Const _ | App _ | Option _ | Of_option _ | Store _
      | Batch _ | Record_outs _ | Fields _ | Field _ | Storage _ | Wrap _
      | Text _ | Calendar _ ),
      _ ) ->
      false

let lit_hash : type a. a Kind.t -> a -> int =
 fun k v ->
  match k with
  | Float -> Hashtbl.hash (Int64.bits_of_float v)
  | Int -> Hashtbl.hash v
  | Bool -> Hashtbl.hash v
  | String -> Hashtbl.hash v
  | Date -> Hashtbl.hash (Time.Date.to_days v)
  | Instant -> Hashtbl.hash (Time.to_ns v)
  | Span -> Hashtbl.hash (Time.Span.to_ns v)
  | Binary | List _ | Record | Tensor _ | Ext -> 0

(* A hash agrees with [node_equal]: it reads the identities of operands and only
   leaves that [node_equal] compares by value, never the values it compares with
   [Type.compare_value]. *)
let node_hash : type a. a node -> int =
 fun n ->
  let leaf =
    match n with
    | Handle (_, c) | Ext_handle (_, c) | Read (_, c) -> Hashtbl.hash c
    | Lit (k, v) -> lit_hash k v
    | _ -> 0
  in
  Hashtbl.hash (leaf, List.map (fun (Packed e) -> e.id) (operands n))

(* [term] is unboxed, so a term in the table is the expression itself and the
   table holds it weakly. *)
type term = T : ('a, 's) t -> term [@@unboxed]

module Table = Weak.Make (struct
  type t = term

  let equal (T e0) (T e1) =
    match (e0.typing, e1.typing) with
    | Some t0, Some t1 -> typing_equal t0 t1 && node_equal e0.node e1.node
    | _ -> false

  let hash (T e) = node_hash e.node
end)

let table = Table.create 4096
let lock = Mutex.create ()
let null = { id = 0; node = Null; typing = None }
let rows = { id = 1; node = Rows; typing = None }
let next_id = Atomic.make 2

(* Unbound expressions are trees: binding reads them and nothing compares them,
   so each has an identity of its own. *)
let make node = { id = Atomic.fetch_and_add next_id 1; node; typing = None }

(* A structurally equal expression may be live at another type, so a new
   expression takes its identity rather than being it. The new expression joins
   the table too, so that the identity lives as long as one of them does. *)
let typed typing node =
  Mutex.protect lock (fun () ->
      let e = { id = -1; node; typing = Some typing } in
      let id =
        match Table.find_opt table (T e) with
        | Some (T e) -> e.id
        | None -> Atomic.fetch_and_add next_id 1
      in
      let e = { e with id } in
      Table.add table (T e);
      e)

let node e = e.node

(* Formatting *)

let keywords =
  String.split_on_char ' '
    "and as assert asr begin class constraint do done downto effect else end \
     exception external false for fun function functor if in include inherit \
     initializer land lazy let lor lsl lsr lxor match method mod module \
     mutable new nonrec object of open or private rec sig struct then to true \
     try type val virtual when while with _"

(* The values of this module, which a bare column name would shadow. *)
let values =
  String.split_on_char ' '
    "across agg arg_max arg_min batch bool cast coalesce collect const count \
     cut date each ewm field first float if_ instant int is_in is_null keep \
     last max mean median min n_unique not null nx nx2 of_option only option \
     over pp pp_out quantile rank record rolling rows shift span std store \
     string sum unpack var"

let is_lowercase_ident n =
  String.length n > 0
  && (match n.[0] with 'a' .. 'z' | '_' -> true | _ -> false)
  && String.for_all
       (function
         | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' | '\'' -> true
         | _ -> false)
       n

let pp_name ppf n =
  if is_lowercase_ident n && not (List.mem n keywords || List.mem n values) then
    Format.pp_print_string ppf n
  else Type.pp_quoted ppf n

let pp_pattern ppf = function
  | Literal s -> Format.fprintf ppf "literal %a" Type.pp_quoted s
  | Prefix s -> Format.fprintf ppf "prefix %a" Type.pp_quoted s
  | Suffix s -> Format.fprintf ppf "suffix %a" Type.pp_quoted s
  | Pieces ss -> Format.fprintf ppf "pieces %a" (Type.pp_list Type.pp_quoted) ss

let pp_zone ppf z = Type.pp_quoted ppf (Tz.name z)

let pp_policy ppf = function
  | `Earlier -> Format.pp_print_string ppf "`Earlier"
  | `Later -> Format.pp_print_string ppf "`Later"
  | `Null -> Format.pp_print_string ppf "`Null"
  | `Fail -> Format.pp_print_string ppf "`Fail"

let field_name : field -> string = function
  | `Year -> "`Year"
  | `Month -> "`Month"
  | `Day -> "`Day"
  | `Hour -> "`Hour"
  | `Minute -> "`Minute"
  | `Second -> "`Second"
  | `Nanosecond -> "`Nanosecond"
  | `Weekday -> "`Weekday"
  | `Yearday -> "`Yearday"

let unary_name : Nx_backend.unary -> string = function
  | Neg -> "neg"
  | Recip -> "recip"
  | Abs -> "abs"
  | Sqrt -> "sqrt"
  | Sign -> "sign"
  | Exp -> "exp"
  | Log -> "log"
  | Log1p -> "log1p"
  | Expm1 -> "expm1"
  | Sin -> "sin"
  | Cos -> "cos"
  | Tan -> "tan"
  | Asin -> "asin"
  | Acos -> "acos"
  | Atan -> "atan"
  | Sinh -> "sinh"
  | Cosh -> "cosh"
  | Tanh -> "tanh"
  | Trunc -> "trunc"
  | Ceil -> "ceil"
  | Floor -> "floor"
  | Round -> "round"
  | Erf -> "erf"

let binary_name : Nx_backend.binary -> string = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Fdiv -> "div"
  | Idiv -> "div"
  | Mod -> "mod"
  | Pow -> "pow"
  | Atan2 -> "atan2"
  | Maximum -> "maximum"
  | Minimum -> "minimum"
  | And -> "logical_and"
  | Or -> "logical_or"
  | Xor -> "logical_xor"

let compare_name : Nx_backend.compare -> string = function
  | Equal -> "equal"
  | Not_equal -> "not_equal"
  | Less -> "less"
  | Less_equal -> "less_equal"

let ext_name d =
  match d.type_ with Type.Ext { name; _ } -> name | _ -> assert false

(* Precedences, loosest first: [||] 1, [&&] 2, comparisons and [$] 3, additive
   operators 4, multiplicative ones 5, [**] 6, application 7, atoms 8. *)

let arith_op op ~float =
  match op with
  | Add -> ((if float then "+." else "+"), 4)
  | Sub -> ((if float then "-." else "-"), 4)
  | Mul -> ((if float then "*." else "*"), 5)
  | Div -> ((if float then "/." else "/"), 5)
  | Mod -> ("mod", 5)
  | Pow -> ("**", 6)

let compare_op : compare -> string = function
  | `Eq -> "="
  | `Ne -> "<>"
  | `Lt -> "<"
  | `Le -> "<="
  | `Gt -> ">"
  | `Ge -> ">="

let wrap ctx level ppf pp =
  if ctx > level then Format.fprintf ppf "@[<hov 1>(%t)@]" pp else pp ppf

(* [is_negative k v] is [true] iff the literal [v] formats with a sign. *)
let is_negative : type a. a Kind.t -> a -> bool =
 fun k v ->
  match k with
  | Int -> v < 0
  | Float -> Float.sign_bit v && not (Float.is_nan v)
  | Span -> Int64.compare (Time.Span.to_ns v) 0L < 0
  | _ -> false

let signed n = if n < 0 then Printf.sprintf "(%d)" n else string_of_int n
let pattern p ppf = Format.fprintf ppf "(%a)" pp_pattern p

let rec pp_at : type a s. int -> Format.formatter -> (a, s) t -> unit =
 fun ctx ppf e ->
  let infix level ~right sym a b =
    let l, r = if right then (level + 1, level) else (level, level + 1) in
    wrap ctx level ppf (fun ppf ->
        Format.fprintf ppf "@[<hov 2>%a %s@ %a@]" (pp_at l) a sym (pp_at r) b)
  in
  let app name args =
    wrap ctx 7 ppf (fun ppf ->
        Format.fprintf ppf "@[<hov 2>%s" name;
        List.iter (fun arg -> Format.fprintf ppf "@ %t" arg) args;
        Format.fprintf ppf "@]")
  in
  let arg e ppf = pp_at 8 ppf e in
  let str s ppf = Format.pp_print_string ppf s in
  let pp_values (type v w) (a : (v, w) t) (vs : v list) ppf =
    match a.typing with
    | Some (Column ty) -> Type.pp_list (Type.pp_value ty) ppf vs
    | _ -> Format.pp_print_string ppf "[…]"
  in
  match e.node with
  | Handle (_, n) | Ext_handle (_, n) | Read (_, n) -> pp_name ppf n
  | Lit (k, v) ->
      if is_negative k v then wrap ctx 7 ppf (fun ppf -> Type.pp_lit k ppf v)
      else Type.pp_lit k ppf v
  | Null -> Format.pp_print_string ppf "null"
  | Rows -> Format.pp_print_string ppf "rows"
  | Int (op, a, b) ->
      let sym, level = arith_op op ~float:false in
      infix level ~right:(level = 6) sym a b
  | Float (op, a, b) ->
      let sym, level = arith_op op ~float:true in
      infix level ~right:(level = 6) sym a b
  | Compare (op, a, b) -> infix 3 ~right:false (compare_op op) a b
  | Logic (And, a, b) -> infix 2 ~right:true "&&" a b
  | Logic (Or, a, b) -> infix 1 ~right:true "||" a b
  | App (f, a) -> infix 3 ~right:false "$" f a
  | Not a -> app "not" [ arg a ]
  | If (c, a, b) -> app "if_" [ arg c; arg a; arg b ]
  | Is_null a -> app "is_null" [ arg a ]
  | Coalesce es -> app "coalesce" [ (fun ppf -> Type.pp_list (pp_at 0) ppf es) ]
  | Is_in (vs, a) -> app "is_in" [ pp_values a vs; arg a ]
  | Cut (edges, a) -> app "cut" [ pp_values a (Array.to_list edges); arg a ]
  | Cast (ty, a) -> app "cast" [ (fun ppf -> Type.pp ppf ty); arg a ]
  | Lift (_, a) -> app "nx" [ str "<fn>"; arg a ]
  | Lift2 (_, a, b) -> app "nx2" [ str "<fn>"; arg a; arg b ]
  | Nx_unary (op, a) -> app (unary_name op) [ arg a ]
  | Nx_binary (op, a, b) -> app (binary_name op) [ arg a; arg b ]
  | Nx_compare (op, a, b) -> app (compare_name op) [ arg a; arg b ]
  | Nx_where (c, a, b) -> app "where" [ arg c; arg a; arg b ]
  | Nx_cast (ty, a) -> app "convert" [ (fun ppf -> Type.pp ppf ty); arg a ]
  | Reduce (Quantile p, a) ->
      app "quantile" [ (fun ppf -> Type.pp_lit Float ppf p); arg a ]
  | Reduce (Ewm alpha, a) ->
      app "ewm"
        [
          (fun ppf -> Format.fprintf ppf "~alpha:%a" (Type.pp_lit Float) alpha);
          arg a;
        ]
  | Reduce (r, a) -> app (reduction_name r) [ arg a ]
  | Over { by; order; e } ->
      let labelled label pp = function
        | [] -> []
        | vs ->
            [
              (fun ppf ->
                Format.fprintf ppf "~%s:%a" label (Type.pp_list pp) vs);
            ]
      in
      app "over"
        (labelled "by" Type.pp_quoted by
        @ labelled "order" Order.pp order
        @ [ arg e ])
  | Rolling (w, e) ->
      app "rolling"
        [ (fun ppf -> Format.fprintf ppf "(%a)" Window.pp w); arg e ]
  | Shift (n, a) -> app "shift" [ str (signed n); arg a ]
  | Rank a -> app "rank" [ arg a ]
  | Const _ -> Format.pp_print_string ppf "<const>"
  | Option a -> app "option" [ arg a ]
  | Of_option a -> app "of_option" [ arg a ]
  | Store (ty, a) -> app "store" [ (fun ppf -> Type.pp ppf ty); arg a ]
  | Batch (_, a) -> app "batch" [ str "<fn>"; arg a ]
  | Record_outs os -> app "record" [ (fun ppf -> Type.pp_list pp_out ppf os) ]
  | Fields fs ->
      let pp_field ppf (n, Packed e) =
        Format.fprintf ppf "@[<hov 2>%a :=@ %a@]" Type.pp_quoted n (pp_at 0) e
      in
      app "record" [ (fun ppf -> Type.pp_list pp_field ppf fs) ]
  | Field (k, name, r) ->
      app "field"
        [
          (fun ppf -> Kind.pp ppf k);
          (fun ppf -> Type.pp_quoted ppf name);
          arg r;
        ]
  | Storage (d, a) ->
      app "Ext.storage"
        [ (fun ppf -> Format.fprintf ppf "<%s>" (ext_name d)); arg a ]
  | Wrap (d, a) ->
      app "Ext.wrap"
        [ (fun ppf -> Format.fprintf ppf "<%s>" (ext_name d)); arg a ]
  | Text (op, a) ->
      let attributes =
        match op with
        | Slice { offset; length } ->
            [
              (fun ppf -> Format.fprintf ppf "~offset:%s" (signed offset));
              (fun ppf -> Format.fprintf ppf "~length:%d" length);
            ]
        | Matches p -> [ pattern p ]
        | Parse ty -> [ (fun ppf -> Type.pp ppf ty) ]
        | Length | Lower | Upper -> []
      in
      app (text_op_name op) (attributes @ [ arg a ])
  | Calendar op ->
      let zone = function
        | None -> []
        | Some z -> [ (fun ppf -> Format.fprintf ppf "~zone:%a" pp_zone z) ]
      in
      let step s ppf =
        let printed = strf "%a" Time.pp_step s in
        if String.starts_with ~prefix:"-" printed then
          Format.fprintf ppf "(%s)" printed
        else Format.pp_print_string ppf printed
      in
      app (calendar_op_name op)
        (match op with
        | Add_span (a, d) -> [ arg a; arg d ]
        | Diff (a, b) -> [ arg a; arg b ]
        | Part (f, z, a) -> (str (field_name f) :: zone z) @ [ arg a ]
        | Floor (z, s, a) -> zone z @ [ step s; arg a ]
        | Offset (z, s, a) -> zone z @ [ step s; arg a ]
        | Localize { zone = z; ambiguous; gap; a } ->
            [
              (fun ppf -> pp_zone ppf z);
              (fun ppf ->
                Format.fprintf ppf "~ambiguous:%a" pp_policy ambiguous);
              (fun ppf -> Format.fprintf ppf "~gap:%a" pp_policy gap);
              arg a;
            ]
        | Windows { zone = z; every; period; a } ->
            zone z
            @ [
                (fun ppf -> Format.fprintf ppf "~every:%a" Time.pp_step every);
                (fun ppf -> Format.fprintf ppf "~period:%a" Time.pp_step period);
                arg a;
              ]
        | Parse_with (fmt, ty, a) ->
            [
              (fun ppf -> Type.pp_quoted ppf fmt);
              (fun ppf -> Type.pp ppf ty);
              arg a;
            ]
        | Format_with (fmt, a) -> [ (fun ppf -> Type.pp_quoted ppf fmt); arg a ])

and pp_out ppf = function
  | Named (n, Packed e) ->
      Format.fprintf ppf "@[<hov 2>%a :=@ %a@]" Type.pp_quoted n (pp_at 0) e
  | Keep sel -> Format.fprintf ppf "@[<hov 2>keep@ %a@]" Sel.pp_arg sel
  | Across (k, sel, _) ->
      Format.fprintf ppf "@[<hov 2>across@ %a@ %a@ <fn>@]" Kind.pp k Sel.pp_arg
        sel
  | Each (sel, _) ->
      Format.fprintf ppf "@[<hov 2>each@ %a@ <fn>@]" Sel.pp_arg sel
  | Unpack r -> Format.fprintf ppf "@[<hov 2>unpack@ %a@]" (pp_at 8) r

let pp ppf e = pp_at 0 ppf e
let pp_arg ppf e = pp_at 8 ppf e

(* Binding *)

(* The shape of the subexpressions binding builds: shapes are erased in nodes,
   and the root's is restored by coercion. *)
type erased

type 'a elab =
  | Known of 'a typing * ('a, erased) t
      (** A bound expression of this typing. *)
  | Flexible of { default : 'a typing option; at : 'a typing -> ('a, erased) t }
      (** An expression whose typing its context gives: [at t] binds it at [t],
          reporting what [t] does not hold, and [default] is its typing where no
          context gives one. *)
  | Broken  (** An expression with a problem, already reported. *)

type env = {
  schema : Schema.t;
  problems : Problem.t list ref;  (** In reverse order. *)
  seen : (string, unit) Hashtbl.t;  (** The messages of [problems]. *)
}

let problem env p =
  let key = strf "%a" Problem.pp p in
  if not (Hashtbl.mem env.seen key) then begin
    Hashtbl.add env.seen key ();
    env.problems := p :: !(env.problems)
  end

let report env fmt =
  Format.kasprintf (fun s -> problem env (Problem.v "%s" s)) fmt

let int64 = Column Type.int64
let bool_typing = Column Type.bool
let known t n = Known (t, typed t n)
let read ty n = typed (Column ty) (Read (ty, n))
let conj a b = typed bool_typing (Logic (And, a, b))

let pp_typing : type a. Format.formatter -> a typing -> unit =
 fun ppf -> function
  | Column ty -> Type.pp ppf ty
  | Extension d -> Type.pp ppf d.type_
  | Value -> Format.pp_print_string ppf "an OCaml value"

let missing env name = problem env (Problem.missing name env.schema)

let untyped env e =
  report env "%a has no type: give it one with store." pp e;
  None

let default_type : type a. a Kind.t -> a Type.t option = function
  | Int -> Some Type.int64
  | Float -> Some Type.float64
  | Bool -> Some Type.bool
  | String -> Some Type.string
  | Binary -> Some Type.binary
  | Date -> Some Type.date
  | Instant -> Some (Type.datetime ~zone:"UTC" Type.Ns)
  | Span -> Some (Type.duration Type.Ns)
  | List _ | Record | Tensor _ | Ext -> None

(* [kind_types k] says which types [k] binds. *)
let kind_types : type a. a Kind.t -> string = function
  | Int -> "int8 to int64 and uint8 to uint64"
  | Float -> "float16, float32 or float64"
  | String -> "string or categorical"
  | Instant -> "datetime"
  | Span -> "duration or clock"
  | k -> strf "%a" Kind.pp k

let handle_name : type a. a Kind.t -> string =
 fun k ->
  match k with
  | Bool | Int | Float | String | Binary | Date | Instant | Span ->
      strf "Col.%a" Kind.pp k
  | List _ | Record | Tensor _ | Ext -> strf "Col.v %a" Kind.pp k

let first_default els =
  List.find_map
    (function
      | Flexible { default = Some (Column _ as t); _ } -> Some t | _ -> None)
    els

(* [at t el] is [el] bound at the typing [t] its operation met at. *)
let at : type a. a typing -> a elab -> (a, erased) t =
 fun t -> function
  | Known (_, b) -> b
  | Flexible { at; _ } -> at t
  | Broken -> assert false

(* [resolve env e el] is [el] bound where an operand needs a column type. *)
let resolve : type a s.
    env -> (a, s) t -> a elab -> (a typing * (a, erased) t) option =
 fun env e -> function
  | Known (Value, _) ->
      report env
        "%a is an OCaml value, not a column: give it a type with store." pp e;
      None
  | Known (t, b) -> Some (t, b)
  | Flexible { default = Some (Column _ as t); at } -> Some (t, at t)
  | Flexible _ -> untyped env e
  | Broken -> None

(* [resolve_value env e el] is like [resolve], where an OCaml value may
   stand. *)
let resolve_value env e = function
  | Known (t, b) -> Some (t, b)
  | Flexible { default = Some t; at } -> Some (t, at t)
  | Flexible _ -> untyped env e
  | Broken -> None

type 'a met = Met of 'a typing | Unknown | Failed

let pp_and pp ppf vs =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " and ")
    pp ppf vs

(* [meet env els] is the typing at which the operands [els] meet. *)
let meet : type a. env -> a elab list -> a met =
 fun env els ->
  let known =
    List.filter_map (function Known (t, _) -> Some t | _ -> None) els
  in
  if List.exists (function Broken -> true | _ -> false) els then Failed
  else
    match known with
    | [] -> Unknown
    | _ when List.exists (function Value -> true | _ -> false) known ->
        report env
          "an OCaml value is not an operand: give it a type with store.";
        Failed
    | Column _ :: _
      when List.for_all (function Column _ -> true | _ -> false) known -> (
        let tys =
          List.filter_map (function Column ty -> Some ty | _ -> None) known
        in
        match Type.common tys with
        | Some ty -> Met (Column ty)
        | None ->
            report env "%a do not meet: cast first." (pp_and Type.pp) tys;
            Failed)
    | Extension d :: rest
      when List.for_all
             (function Extension d' -> same_value d d' | _ -> false)
             rest ->
        Met (Extension d)
    | _ ->
        report env "%a do not meet." (pp_and pp_typing) known;
        Failed

(* [operation_at els met build] binds an operation whose result has the typing
   [met] at which its operands [els] meet. *)
let operation_at els met build =
  match met with
  | Failed -> Broken
  | Met t -> known t (build (at t))
  | Unknown ->
      Flexible
        { default = first_default els; at = (fun t -> typed t (build (at t))) }

let operation env els build = operation_at els (meet env els) build

(* [meet_or_default env e els] is the typing at which the operands [els] of [e]
   meet, their own default where all are flexible. *)
let meet_or_default env e els =
  match meet env els with
  | Failed -> None
  | Met t -> Some t
  | Unknown -> (
      match first_default els with Some t -> Some t | None -> untyped env e)

(* [expect env e t el] is [el] bound at the typing [t], which must contain its
   own. *)
let expect : type a s.
    env -> (a, s) t -> a typing -> a elab -> (a, erased) t option =
 fun env e t el ->
  match (el, t) with
  | Flexible { at; _ }, _ -> Some (at t)
  | Broken, _ -> None
  | Known (Column ty, b), Column want -> (
      match Type.common [ ty; want ] with
      | Some c when Type.equal c want -> Some b
      | _ ->
          report env "%a is %a, which %a does not contain." pp e Type.pp ty
            Type.pp want;
          None)
  | Known (Extension d, b), Extension d' when same_value d d' -> Some b
  | Known (t', _), _ ->
      let remedy =
        match (t', t) with
        | Extension _, Column _ -> ": use Ext.storage"
        | _ -> ""
      in
      report env "%a is %a, where %a is expected%s." pp e pp_typing t' pp_typing
        t remedy;
      None

let ordered : type a. env -> string -> a typing -> unit =
 fun env what -> function
  | Column (Ext _ as ty) ->
      report env
        "%s orders values, and %a is an extension read without its \
         declaration: read it with an Ext.t declared ~ordered:true."
        what Type.pp ty
  | Column ty when Type.has_ext ty ->
      report env
        "%s orders values, and %a holds an extension type and has no order."
        what Type.pp ty
  | Extension d when not d.ordered ->
      report env
        "%s orders values, and the declaration of %a is not ~ordered:true." what
        Type.pp d.type_
  | Column _ | Extension _ | Value -> ()

let holds : type a. env -> string -> a typing -> a -> bool =
 fun env what t v ->
  match t with
  | Column ty ->
      let held = Type.holds ty v in
      if not held then
        report env "%a does not hold the %s %a." Type.pp ty what
          (Type.pp_value ty) v;
      held
  | Extension d ->
      let s = d.enc v in
      let held = Type.holds d.storage s in
      if not held then
        report env "%a does not hold %a, the storage of the %s." Type.pp
          d.storage (Type.pp_value d.storage) s what;
      held
  | Value -> true

(* Every value is checked, so that each is reported. A value outside a
   categorical dictionary has no order: a node holding one must not reach
   sorting or hash-consing, which compare values. *)
let all_hold env what t vs =
  List.fold_left (fun ok v -> holds env what t v && ok) true vs

(* [constant e] is the value of [e] when it is integer arithmetic of literals
   that is not null. *)
let rec constant : type s. (int, s) t -> int option =
 fun e ->
  match e.node with
  | Lit (_, v) -> Some v
  | Int (op, a, b) -> (
      match (constant a, constant b) with
      | Some x, Some y -> (
          match op with
          | Add -> Some (x + y)
          | Sub -> Some (x - y)
          | Mul -> Some (x * y)
          | Div when y <> 0 -> Some (x / y)
          | Mod when y <> 0 -> Some (x mod y)
          | Div | Mod | Pow -> None)
      | _ -> None)
  | _ -> None

let is_int_or_float : type a. a Type.t -> bool = function
  | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64 -> true
  | Float16 | Float32 | Float64 -> true
  | _ -> false

let sum_type : type a. a Type.t -> a Type.t option =
 fun ty ->
  match (ty, Type.kind ty) with
  | _, Int -> Some Type.int64
  | _, Float -> Some ty
  | Duration _, _ -> Some ty
  | _ -> None

let reduction_typing : type a b.
    env -> (a, b) reduction -> a typing -> b typing option =
 fun env r t ->
  let name = reduction_name r in
  let float_of () =
    match t with
    | Column ty when is_int_or_float ty -> Some (Column Type.float64)
    | _ ->
        let remedy =
          match t with Extension _ -> ": use Ext.storage" | _ -> ""
        in
        report env "%s takes integers or floats, not %a%s." name pp_typing t
          remedy;
        None
  in
  let no_type () =
    report env
      "%s has no type over %a: apply it to Ext.storage, or read the column \
       with each."
      name pp_typing t;
    None
  in
  (match r with Min | Max | Arg_min | Arg_max -> ordered env name t | _ -> ());
  match r with
  | Count -> Some int64
  | N_unique -> Some int64
  | First -> Some t
  | Last -> Some t
  | Only -> Some t
  | Min -> Some t
  | Max -> Some t
  | Arg_min -> Some int64
  | Arg_max -> Some int64
  | Sum -> (
      match t with
      | Column ty -> (
          match sum_type ty with
          | Some ty -> Some (Column ty)
          | None ->
              report env "sum takes integers, floats or durations, not %a."
                Type.pp ty;
              None)
      | Extension _ | Value -> no_type ())
  | Mean -> float_of ()
  | Std -> float_of ()
  | Var -> float_of ()
  | Median -> float_of ()
  | Quantile _ -> float_of ()
  | Ewm _ -> float_of ()
  | Collect -> (
      match t with
      | Column ty -> Some (Column (Type.list ty))
      | Extension _ | Value -> no_type ())

(* [castable ty0 ty1] is [Ok ()] iff [cast] converts [ty0] to [ty1], and
   otherwise names the function that does, if one does. *)
let rec castable : type a b.
    a Type.t -> b Type.t -> (unit, string option) result =
 fun ty0 ty1 ->
  let numeric : type c. c Type.t -> bool = function
    | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
    | Float16 | Float32 | Float64 ->
        true
    | _ -> false
  in
  let text : type c. c Type.t -> bool = function
    | String | Categorical _ -> true
    | _ -> false
  in
  match (ty0, ty1) with
  | _ when numeric ty0 && numeric ty1 -> Ok ()
  | _ when text ty0 && text ty1 -> Ok ()
  | Datetime d0, Datetime d1 ->
      if Bool.equal (Option.is_some d0.zone) (Option.is_some d1.zone) then Ok ()
      else Error (Some "Temporal.localize")
  | Duration _, Duration _ | Clock _, Clock _ | Date, Date | Binary, Binary ->
      Ok ()
  | List t0, List t1 -> castable t0 t1
  | Record fs0, Record fs1
    when List.equal (fun (n0, _) (n1, _) -> String.equal n0 n1) fs0 fs1 ->
      List.fold_left2
        (fun acc (_, Type.Any t0) (_, Type.Any t1) ->
          Result.bind acc (fun () -> castable t0 t1))
        (Ok ()) fs0 fs1
  | Tensor (_, s0), Tensor (_, s1) when Iarray.equal Int.equal s0 s1 -> Ok ()
  | Ext _, _ | _, Ext _ -> Error (Some "Ext.storage and Ext.wrap")
  | _, Clock _ when text ty0 -> Error (Some "Temporal.parse")
  | ( _,
      ( Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
      | Float16 | Float32 | Float64 | Date | Datetime _ ) )
    when text ty0 ->
      Error (Some "Str.parse")
  | (Date | Clock _ | Datetime _), _ when text ty1 ->
      Error (Some "Temporal.format")
  | _ -> Error None

let is_temporal_key : type a. a Type.t -> bool = function
  | Datetime _ | Date | Clock _ | Duration _ -> true
  | _ -> false

let unit_rank : Type.unit_ -> int = function
  | S -> 0
  | Ms -> 1
  | Us -> 2
  | Ns -> 3

let has_zone_directive fmt =
  let n = String.length fmt in
  let rec loop i =
    i + 1 < n
    &&
    if Char.equal fmt.[i] '%' then Char.equal fmt.[i + 1] 'z' || loop (i + 2)
    else loop (i + 1)
  in
  loop 0

(* [zone_rule env what zone ty_zone] checks that [zone] is given exactly when
   the datetime type's zone [ty_zone] is. *)
let zone_rule env what zone ty_zone =
  match (zone, ty_zone) with
  | None, Some z ->
      report env
        "%s reads a datetime with the zone %a on a wall clock, which needs \
         ~zone."
        what Type.pp_quoted z
  | Some _, None ->
      report env
        "%s reads a wall-clock value, which takes no ~zone: only a datetime \
         with a zone does."
        what
  | None, None | Some _, Some _ -> ()

let not_temporal : type a b. env -> string -> a Type.t -> b elab =
 fun env what ty ->
  report env "%s does not take %a." what Type.pp ty;
  Broken

let whole_days span =
  Int64.equal
    (Int64.rem (Time.Span.to_ns span) (Time.Span.to_ns (Time.Span.days 1)))
    0L

(* An exact step on a datetime is whole ticks, since the result is of the
   column's type. *)
let calendar_step : type a. env -> string -> a Type.t -> Time.step -> unit =
 fun env what ty step ->
  match (ty, step) with
  | Date, Time.Exact _ ->
      report env "%s moves a date by calendar steps only." what
  | Datetime { unit_; _ }, Time.Exact s ->
      let tick = Type.ns_per_unit unit_ in
      if not (Int64.equal (Int64.rem (Time.Span.to_ns s) tick) 0L) then
        report env "%s moves %a by multiples of %a, not %a." what Type.pp ty
          Time.Span.pp (Time.Span.of_ns tick) Time.Span.pp s
  | _ -> ()

(* Lifts *)

type (_, _) Nx.Repr.node +=
  | Traced : 'c Type.t * ('c, erased) t -> ('a, 'b) Nx.Repr.node

exception Not_elementwise of string

type dtype = Dtype : ('a, 'b) Nx.dtype -> dtype

let dtype_of : type a. a Type.t -> dtype option = function
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

(* The type and the literal of an element of a dtype. *)
type 'a scalar =
  | Scalar : 'c Type.t * 'c Kind.t * ('a -> 'c option) -> 'a scalar

let int64_value v =
  let i = Int64.to_int v in
  if Int64.equal (Int64.of_int i) v then Some i else None

let scalar_of : type a b. (a, b) Nx.dtype -> a scalar option = function
  | Nx.Bool -> Some (Scalar (Type.bool, Kind.bool, Option.some))
  | Nx.Int8 -> Some (Scalar (Type.int8, Kind.int, Option.some))
  | Nx.Int16 -> Some (Scalar (Type.int16, Kind.int, Option.some))
  | Nx.Int32 ->
      Some (Scalar (Type.int32, Kind.int, fun v -> Some (Int32.to_int v)))
  | Nx.Int64 -> Some (Scalar (Type.int64, Kind.int, int64_value))
  | Nx.UInt8 -> Some (Scalar (Type.uint8, Kind.int, Option.some))
  | Nx.UInt16 -> Some (Scalar (Type.uint16, Kind.int, Option.some))
  | Nx.UInt32 ->
      Some
        (Scalar
           ( Type.uint32,
             Kind.int,
             fun v -> Some (Int32.to_int v land 0xFFFF_FFFF) ))
  | Nx.UInt64 ->
      Some
        (Scalar
           ( Type.uint64,
             Kind.int,
             fun v -> if Int64.compare v 0L < 0 then None else int64_value v ))
  | Nx.Float16 -> Some (Scalar (Type.float16, Kind.float, Option.some))
  | Nx.Float32 -> Some (Scalar (Type.float32, Kind.float, Option.some))
  | Nx.Float64 -> Some (Scalar (Type.float64, Kind.float, Option.some))
  | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 | Nx.Int4 | Nx.UInt4
  | Nx.Complex64 | Nx.Complex128 ->
      None

type traced = Expr_of : 'c Type.t * ('c, erased) t -> traced

let no_type dt =
  Not_elementwise
    (strf "computes in %a, which talon has no type for" Nx.pp_dtype dt)

(* [traced x] is the expression that the value [x] traces, a constant becoming a
   literal. *)
let traced : type a b. (a, b) Nx.t -> traced =
 fun x ->
  match Nx.Repr.v x with
  | Traced tr -> (
      match Nx.Repr.Traced.node tr with
      | Traced (ty, e) -> Expr_of (ty, e)
      | _ ->
          raise
            (Not_elementwise "reads a value traced by another transformation"))
  | Host _ | Placed _ -> (
      let dt = Nx.dtype x in
      match scalar_of dt with
      | None -> raise (no_type dt)
      | Some (Scalar (ty, k, value)) -> (
          let v =
            Nx.item (List.map (fun _ -> 0) (Array.to_list (Nx.shape x))) x
          in
          match value v with
          | Some c -> Expr_of (ty, typed (Column ty) (Lit (k, c)))
          | None ->
              raise (Not_elementwise "creates a constant outside OCaml's int")))

let result dt ty n =
  Nx.Repr.Traced.v ~context:Nx.Placement.host Nx.Placement.host dt [||]
    (Traced (ty, typed (Column ty) n))

(* [same_type tx ty] is a witness that the traced types [tx] and [ty], of one
   dtype, are equal. *)
let same_type : type a b. a Type.t -> b Type.t -> (a, b) Stdlib.Type.eq =
 fun tx ty ->
  match Kind.equal_witness (Type.kind tx) (Type.kind ty) with
  | Some Equal when Type.equal tx ty -> Equal
  | _ -> raise (Not_elementwise "mixes dtypes")

let moves_nothing : Nx.Op.move -> bool = function
  | Reshape s | Expand s -> Array.length s = 0
  | Permute p -> Array.length p = 0
  | Shrink s -> Array.length s = 0
  | Flip f -> Array.length f = 0
  | Window _ -> false

let run : type r. r Nx.Op.t -> r =
 fun op ->
  let is_traced (Nx.P x) =
    match Nx.Repr.v x with Traced _ -> true | _ -> false
  in
  if not (List.exists is_traced (Nx.Op.operands op)) then Nx.Op.eval op
  else
    match op with
    | Unary (u, x) ->
        let (Expr_of (tx, ex)) = traced x in
        result (Nx.dtype x) tx (Nx_unary (u, ex))
    | Binary (b, x, y) ->
        let (Expr_of (tx, ex)) = traced x in
        let (Expr_of (ty, ey)) = traced y in
        let Equal = same_type tx ty in
        result (Nx.dtype x) tx (Nx_binary (b, ex, ey))
    | Compare (c, x, y) ->
        let (Expr_of (tx, ex)) = traced x in
        let (Expr_of (ty, ey)) = traced y in
        let Equal = same_type tx ty in
        result Nx.bool Type.bool (Nx_compare (c, ex, ey))
    | Where (c, x, y) ->
        let (Expr_of (tc, ec)) = traced c in
        let (Expr_of (tx, ex)) = traced x in
        let (Expr_of (ty, ey)) = traced y in
        let Equal = same_type tc Type.bool in
        let Equal = same_type tx ty in
        result (Nx.dtype x) tx (Nx_where (ec, ex, ey))
    | Convert (Cast, dt, x) -> (
        let (Expr_of (_, ex)) = traced x in
        match scalar_of dt with
        | Some (Scalar (ty, _, _)) -> result dt ty (Nx_cast (ty, ex))
        | None -> raise (no_type dt))
    | Move (x, move) when moves_nothing move -> x
    | Contiguous x -> x
    | op ->
        raise
          (Not_elementwise
             (strf "performs %s, which is not elementwise" (Nx.Op.name op)))

let input dt ty e =
  Nx.Repr.Traced.v ~context:Nx.Placement.host Nx.Placement.host dt [||]
    (Traced (ty, e))

(* [uses ids e] is [true] iff the traced expression [e] uses each expression
   whose identity is in [ids]. *)
let uses ids e =
  let seen = Hashtbl.create 16 in
  let rec walk : type a s. (a, s) t -> unit =
   fun e ->
    if not (Hashtbl.mem seen e.id) then begin
      Hashtbl.add seen e.id ();
      match e.node with
      | Nx_unary _ | Nx_binary _ | Nx_compare _ | Nx_where _ | Nx_cast _ ->
          List.iter (fun (Packed e) -> walk e) (operands e.node)
      | _ -> ()
    end
  in
  walk e;
  List.for_all (Hashtbl.mem seen) ids

(* [lift env name ty ids f] is the expression that [f], the function of the lift
   [name], computes under the tracer on the expressions of identities [ids],
   traced at the type [ty], which its result must keep. A result that does not
   read one of them would drop its nulls. *)
let lift : type a c d.
    env ->
    string ->
    a Type.t ->
    int list ->
    (unit -> (c, d) Nx.t) ->
    (a, erased) t option =
 fun env name ty ids f ->
  (* The tracer takes every operation [f] performs, so that a value [f] builds
     from no operand is a literal of the expression. *)
  let claims _ = true in
  match traced (Nx.Op.intercept { run; claims } f) with
  | exception Not_elementwise what ->
      report env "the function of %s %s." name what;
      None
  | Expr_of (ty', e) -> (
      match Kind.equal_witness (Type.kind ty') (Type.kind ty) with
      | Some Equal when Type.equal ty' ty ->
          if uses ids e then Some e
          else begin
            report env "the function of %s ignores %s." name
              (match ids with [ _ ] -> "its argument" | _ -> "an argument");
            None
          end
      | _ ->
          report env "the function of %s turns %a into %a, which it must keep."
            name Type.pp ty Type.pp ty';
          None)

let no_dtype env name ty =
  report env "%s takes integer, float and boolean operands, not %a." name
    Type.pp ty;
  None

let trace1 env (fn : fn) ty a =
  match dtype_of ty with
  | None -> no_dtype env "nx" ty
  | Some (Dtype dt) ->
      let x = input dt ty a in
      lift env "nx" ty [ a.id ] (fun () -> fn.f x)

let trace2 env (fn : fn2) ty a b =
  match dtype_of ty with
  | None -> no_dtype env "nx2" ty
  | Some (Dtype dt) ->
      let x = input dt ty a and y = input dt ty b in
      lift env "nx2" ty [ a.id; b.id ] (fun () -> fn.f2 x y)

(* Expressions *)

let rec elab : type a s. env -> (a, s) t -> a elab =
 fun env e ->
  match e.node with
  | _ when Option.is_some e.typing -> err "Expr: %a is bound" pp e
  | Fields _ | Nx_unary _ | Nx_binary _ | Nx_compare _ | Nx_where _ | Nx_cast _
    ->
      err "Expr: %a is bound" pp e
  | Handle (k, n) -> (
      match Schema.find env.schema n with
      | None ->
          missing env n;
          Broken
      | Some (Any ty) -> (
          match Kind.provably_equal k (Type.kind ty) with
          | Some Equal -> known (Column ty) e.node
          | None ->
              report env "%s reads %s, but %a is %a." (handle_name k)
                (kind_types k) Type.pp_quoted n Type.pp ty;
              Broken))
  | Ext_handle (d, n) -> (
      match Schema.find env.schema n with
      | None ->
          missing env n;
          Broken
      | Some (Any ty) when Type.equal ty d.type_ -> known (Extension d) e.node
      | Some (Any ty) ->
          report env "Ext.col binds %a, but %a is %a." Type.pp d.type_
            Type.pp_quoted n Type.pp ty;
          Broken)
  | Read (ty, n) -> (
      match Schema.find env.schema n with
      | None ->
          missing env n;
          Broken
      | Some (Any ty') when Type.equal ty ty' -> known (Column ty) e.node
      | Some (Any ty') ->
          report env "%a was read as %a, but it is %a here." Type.pp_quoted n
            Type.pp ty Type.pp ty';
          Broken)
  | Lit (k, v) ->
      let default = Option.map (fun ty -> Column ty) (default_type k) in
      Flexible
        {
          default;
          at =
            (fun t ->
              ignore (holds env "literal" t v);
              typed t e.node);
        }
  | Null -> Flexible { default = None; at = (fun t -> typed t Null) }
  | Const v ->
      Flexible
        {
          default = Some Value;
          at =
            (fun t ->
              ignore (holds env "value" t v);
              typed t e.node);
        }
  | Rows -> known int64 Rows
  | Int (op, a, b) -> (
      let ea = elab env a in
      let eb = elab env b in
      match operation env [ ea; eb ] (fun at -> Int (op, at ea, at eb)) with
      | Flexible { default; at } as el -> (
          match constant e with
          | Some v ->
              let at t =
                let b = at t in
                ignore (holds env "constant" t v);
                b
              in
              Flexible { default; at }
          | None -> el)
      | el -> el)
  | Float (op, a, b) ->
      let ea = elab env a in
      let eb = elab env b in
      operation env [ ea; eb ] (fun at -> Float (op, at ea, at eb))
  | Compare (op, a, b) -> (
      let ea = elab env a in
      let eb = elab env b in
      match meet_or_default env e [ ea; eb ] with
      | None -> Broken
      | Some t ->
          (match op with
          | `Lt | `Le | `Gt | `Ge -> ordered env (compare_op op) t
          | `Eq | `Ne -> ());
          known bool_typing (Compare (op, at t ea, at t eb)))
  | Logic (op, a, b) -> (
      let a = expect env a bool_typing (elab env a) in
      let b = expect env b bool_typing (elab env b) in
      match (a, b) with
      | Some a, Some b -> known bool_typing (Logic (op, a, b))
      | _ -> Broken)
  | Not a -> (
      match expect env a bool_typing (elab env a) with
      | Some a -> known bool_typing (Not a)
      | None -> Broken)
  | If (c, a, b) -> (
      let c = expect env c bool_typing (elab env c) in
      let ea = elab env a in
      let eb = elab env b in
      let met = meet env [ ea; eb ] in
      match c with
      | Some c -> operation_at [ ea; eb ] met (fun at -> If (c, at ea, at eb))
      | None -> Broken)
  | Is_null a -> on_operand env a (fun _ a -> known bool_typing (Is_null a))
  | Coalesce es ->
      let els = List.map (elab env) es in
      operation env els (fun at -> Coalesce (List.map at els))
  | Is_in (vs, a) ->
      on_operand env a (fun t a ->
          if all_hold env "value" t vs then known bool_typing (Is_in (vs, a))
          else Broken)
  | Cut (edges, a) ->
      on_operand env a (fun t a ->
          ordered env "cut" t;
          if all_hold env "edge" t (Array.to_list edges) then
            known int64 (Cut (sorted_edges t edges, a))
          else Broken)
  | Cast (ty, a) ->
      on_operand env a (fun t a ->
          match t with
          | Column ta -> (
              match castable ta ty with
              | Ok () -> known (Column ty) (Cast (ty, a))
              | Error converter ->
                  let pp_use ppf =
                    Option.iter (Format.fprintf ppf ": use %s")
                  in
                  report env "cast does not convert %a to %a%a." Type.pp ta
                    Type.pp ty pp_use converter;
                  Broken)
          | t ->
              report env "cast does not convert %a to %a: use Ext.storage."
                pp_typing t Type.pp ty;
              Broken)
  | Lift (fn, a) -> (
      let lift_at t a =
        match t with
        | Column ty -> trace1 env fn ty a
        | Extension _ | Value ->
            report env "nx takes integer, float and boolean operands, not %a."
              pp_typing t;
            None
      in
      match elab env a with
      | Known (t, a) -> (
          match lift_at t a with Some b -> Known (t, b) | None -> Broken)
      | Flexible { default; at } ->
          let at t =
            let a = at t in
            match lift_at t a with
            | Some b -> b
            | None -> typed t (Lift (fn, a))
          in
          Flexible { default; at }
      | Broken -> Broken)
  | Lift2 (fn, a, b) -> (
      let ea = elab env a in
      let eb = elab env b in
      let lift_at t =
        let a = at t ea and b = at t eb in
        match t with
        | Column ty -> (
            match trace2 env fn ty a b with
            | Some b -> b
            | None -> typed t (Lift2 (fn, a, b)))
        | Extension _ | Value ->
            report env "nx2 takes integer, float and boolean operands, not %a."
              pp_typing t;
            typed t (Lift2 (fn, a, b))
      in
      match meet env [ ea; eb ] with
      | Failed -> Broken
      | Met t -> Known (t, lift_at t)
      | Unknown -> Flexible { default = first_default [ ea; eb ]; at = lift_at }
      )
  | Reduce (r, a) ->
      on_operand env a (fun t a ->
          match reduction_typing env r t with
          | Some rt -> known rt (Reduce (r, a))
          | None -> Broken)
  | Over { by; order; e = x } ->
      let rec keys seen = function
        | [] -> ()
        | n :: ns ->
            if not (List.mem n seen) then begin
              if Option.is_none (Schema.find env.schema n) then missing env n;
              if List.mem n ns then
                report env "over ~by: %a is named twice." Type.pp_quoted n
            end;
            keys (n :: seen) ns
      in
      keys [] by;
      List.iter (fun (_, p) -> problem env p) (Order.check order env.schema);
      on_operand env x (fun t x -> known t (Over { by; order; e = x }))
  | Rolling (w, x) ->
      (match w with
      | Window.Rows _ -> ()
      | Window.Times { on; _ } -> (
          match Schema.find env.schema on with
          | None -> missing env on
          | Some (Any ty) when is_temporal_key ty -> ()
          | Some (Any ty) ->
              report env
                "a time window reads datetime, date, clock or duration keys, \
                 but %a is %a."
                Type.pp_quoted on Type.pp ty));
      on_operand env x (fun t x -> known t (Rolling (w, x)))
  | Shift (n, x) -> on_operand env x (fun t x -> known t (Shift (n, x)))
  | Rank x ->
      on_operand env x (fun t x ->
          ordered env "rank" t;
          known int64 (Rank x))
  | App (f, a) -> (
      let f =
        match elab env f with
        | Known (_, f) -> Some f
        | Flexible { at; _ } -> Some (at Value)
        | Broken -> None
      in
      let a = resolve_value env a (elab env a) in
      match (f, a) with
      | Some f, Some (_, a) ->
          Flexible
            { default = Some Value; at = (fun t -> typed t (App (f, a))) }
      | _ -> Broken)
  | Option a -> on_operand env a (fun _ a -> known Value (Option a))
  | Of_option a -> (
      match resolve_value env a (elab env a) with
      | Some (_, a) ->
          Flexible
            { default = Some Value; at = (fun t -> typed t (Of_option a)) }
      | None -> Broken)
  | Store (ty, a) -> (
      match expect env a (Column ty) (elab env a) with
      | Some a -> known (Column ty) (Store (ty, a))
      | None -> Broken)
  | Batch (f, x) ->
      on_operand env x (fun t x ->
          match t with
          | Column ty -> (
              match batch_type env f ty with
              | Some rt -> known (Column rt) (Batch (f, x))
              | None -> Broken)
          | t ->
              report env "batch takes a tensor column, not %a." pp_typing t;
              Broken)
  | Record_outs os -> record env (List.concat_map (outs env) os)
  | Field (k, name, r) ->
      on_operand env r (fun t r : a elab ->
          match t with
          | Column (Record fields) -> (
              match List.assoc_opt name fields with
              | None ->
                  report env "the record has no field %a: its fields are %a."
                    Type.pp_quoted name (pp_and Type.pp_quoted)
                    (List.map fst fields);
                  Broken
              | Some (Any fty) -> (
                  match Kind.provably_equal k (Type.kind fty) with
                  | Some Equal -> known (Column fty) (Field (k, name, r))
                  | None ->
                      report env "field %a reads %s, but the field %a is %a."
                        Kind.pp k (kind_types k) Type.pp_quoted name Type.pp fty;
                      Broken))
          | t ->
              report env "field reads a record, not %a." pp_typing t;
              Broken)
  | Storage (d, x) -> (
      match expect env x (Extension d) (elab env x) with
      | Some x -> known (Column d.storage) (Storage (d, x))
      | None -> Broken)
  | Wrap (d, x) -> (
      match expect env x (Column d.storage) (elab env x) with
      | Some x -> known (Extension d) (Wrap (d, x))
      | None -> Broken)
  | Text (op, a) -> text env op a
  | Calendar op -> calendar env op

(* [on_operand env a k] is [k t b], [b] the operand [a] bound where it needs a
   column type and [t] its typing, or [Broken]. *)
and on_operand : type a s r.
    env -> (a, s) t -> (a typing -> (a, erased) t -> r elab) -> r elab =
 fun env a k ->
  match resolve env a (elab env a) with Some (t, b) -> k t b | None -> Broken

(* An extension's edges stay as given, so that binding twice gives one
   expression: the evaluator sorts them by their encodings. *)
and sorted_edges : type a. a typing -> a array -> a array =
 fun t edges ->
  match t with
  | Column ty ->
      Array.of_list
        (List.sort_uniq (Type.compare_value ty) (Array.to_list edges))
  | Extension _ | Value -> edges

and batch_type : type a b c d.
    env ->
    ((a, b) Nx.t -> (c, d) Nx.t) ->
    (a, b) Nx.t Type.t ->
    (c, d) Nx.t Type.t option =
 fun env f ty ->
  match ty with
  | Tensor (dt, shape) ->
      let empty = Nx.zeros dt (Array.append [| 0 |] (Iarray.to_array shape)) in
      let r = f empty in
      let rs = Nx.shape r in
      if Array.length rs < 2 || rs.(0) <> 0 then begin
        report env
          "batch's function returns shape %a on an empty batch, which is not \
           an empty batch of cells."
          Nx.pp_shape rs;
        None
      end
      else
        Some (Type.tensor (Nx.dtype r) (Array.sub rs 1 (Array.length rs - 1)))
  | _ ->
      report env "batch takes a tensor column, not %a." Type.pp ty;
      None

and record : env -> (string * packed) list -> Record.t elab =
 fun env fields ->
  match Problem.repeated (List.map fst fields) with
  | _ :: _ as twice ->
      List.iter
        (report env "record: the output %a appears twice." Type.pp_quoted)
        twice;
      Broken
  | [] ->
      let field_type (n, Packed b) =
        match b.typing with
        | Some (Column ty) -> (n, Type.Any ty)
        | Some (Extension d) -> (n, Type.Any d.type_)
        | Some Value | None -> assert false
      in
      known (Column (Type.record (List.map field_type fields))) (Fields fields)

(* [outs env o] is the named, bound outputs of [o]. *)
and outs : env -> out_repr -> (string * packed) list =
 fun env o ->
  let select sel =
    let names, ps = Sel.check sel env.schema in
    List.iter (problem env) ps;
    List.filter_map
      (fun n -> Option.map (fun ty -> (n, ty)) (Schema.find env.schema n))
      names
  in
  match o with
  | Named (n, Packed e) -> (
      match resolve env e (elab env e) with
      | Some (_, b) -> [ (n, Packed b) ]
      | None -> [])
  | Keep sel ->
      List.map (fun (n, Type.Any ty) -> (n, Packed (read ty n))) (select sel)
  | Across (k, sel, f) ->
      List.concat_map
        (fun (n, Type.Any ty) ->
          match Kind.provably_equal k (Type.kind ty) with
          | Some Equal -> outs env (f n (make (Handle (k, n))))
          | None ->
              report env
                "across %a: %a is %a, which %a does not read: narrow the \
                 selector with Sel.of_kind."
                Kind.pp k Type.pp_quoted n Type.pp ty Kind.pp k;
              [])
        (select sel)
  | Each (sel, { column }) ->
      List.concat_map
        (fun (n, Type.Any ty) -> outs env (column n (make (Read (ty, n)))))
        (select sel)
  | Unpack r -> (
      match resolve env r (elab env r) with
      | Some (Column (Record fields), r) ->
          List.map
            (fun (n, Type.Any fty) ->
              (n, Packed (typed (Column fty) (Field (Type.kind fty, n, r)))))
            fields
      | Some (t, _) ->
          report env "unpack reads a record, not %a." pp_typing t;
          []
      | None -> [])

and text : type a s. env -> a text_op -> (string, s) t -> a elab =
 fun env op a ->
  on_operand env a (fun t a : a elab ->
      match t with
      | Column (String | Categorical _) -> (
          let known t = known t (Text (op, a)) in
          match op with
          | Length -> known int64
          | Slice _ -> known (Column Type.string)
          | Lower -> known (Column Type.string)
          | Upper -> known (Column Type.string)
          | Matches _ -> known bool_typing
          | Parse ty -> (
              match ty with
              | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32
              | Uint64 | Float16 | Float32 | Float64 | Date | Datetime _
              | Categorical _ ->
                  known (Column ty)
              | _ ->
                  report env
                    "Str.parse reads bool, integer, float, date, datetime and \
                     categorical types, not %a."
                    Type.pp ty;
                  Broken))
      | t ->
          report env "%s takes text, not %a." (text_op_name op) pp_typing t;
          Broken)

and calendar : type a. env -> a calendar_op -> a elab =
 fun env op ->
  let what = calendar_op_name op in
  let column a =
    match resolve env a (elab env a) with
    | Some (Column ty, b) -> Some (ty, b)
    | Some (t, _) ->
        report env "%s takes a temporal column, not %a." what pp_typing t;
        None
    | None -> None
  in
  let known t n = known t (Calendar n) in
  match op with
  | Add_span (a, d) -> (
      match column a with
      | None -> Broken
      | Some (ta, a') -> (
          let unit_of : type c. c Type.t -> Type.unit_ option = function
            | Datetime { unit_; _ } -> Some unit_
            | Duration u | Clock u -> Some u
            | Date -> Some S
            | _ -> None
          in
          match unit_of ta with
          | None -> not_temporal env what ta
          | Some u -> (
              (match (ta, d.node) with
              | Date, Lit (_, span) when not (whole_days span) ->
                  report env "%s advances a date by whole days, not %a." what
                    Time.Span.pp span
              | _ -> ());
              match elab env d with
              | Flexible { at; _ } ->
                  known (Column ta)
                    (Add_span (a', at (Column (Type.duration u))))
              | Known (Column (Duration du), d')
                when unit_rank du <= unit_rank u ->
                  known (Column ta) (Add_span (a', d'))
              | Known (t, _) ->
                  report env
                    "%s advances %a by durations of its unit or a coarser one, \
                     not %a."
                    what Type.pp ta pp_typing t;
                  Broken
              | Broken -> Broken)))
  | Diff (a, b) -> (
      let ea = elab env a in
      let eb = elab env b in
      match meet_or_default env a [ ea; eb ] with
      | Some (Column ty as t) -> (
          let d u =
            known (Column (Type.duration u)) (Diff (at t ea, at t eb))
          in
          match ty with
          | Datetime { unit_; _ } -> d unit_
          | Duration u | Clock u -> d u
          | Date -> d S
          | _ -> not_temporal env what ty)
      | Some t ->
          report env "%s does not take %a." what pp_typing t;
          Broken
      | None -> Broken)
  | Part (f, zone, a) -> (
      match column a with
      | None -> Broken
      | Some (ta, a') -> (
          let time_field =
            match f with
            | `Hour | `Minute | `Second | `Nanosecond -> true
            | _ -> false
          in
          let ok () = known int64 (Part (f, zone, a')) in
          match ta with
          | Date when time_field ->
              report env "a date has no %s." (field_name f);
              Broken
          | Clock _ when not time_field ->
              report env "a clock has no %s." (field_name f);
              Broken
          | Date | Clock _ ->
              zone_rule env what zone None;
              ok ()
          | Datetime { zone = z; _ } ->
              zone_rule env what zone z;
              ok ()
          | _ -> not_temporal env what ta))
  | Floor (zone, step, a) -> (
      match dated env what zone a with
      | Some (ta, a') ->
          calendar_step env what ta step;
          known (Column ta) (Floor (zone, step, a'))
      | None -> Broken)
  | Offset (zone, step, a) -> (
      match dated env what zone a with
      | Some (ta, a') ->
          calendar_step env what ta step;
          known (Column ta) (Offset (zone, step, a'))
      | None -> Broken)
  | Localize { zone; ambiguous; gap; a } -> (
      match column a with
      | Some (Datetime { zone = None; unit_ }, a') ->
          let t = Column (Type.datetime ~zone:(Tz.name zone) unit_) in
          known t (Localize { zone; ambiguous; gap; a = a' })
      | Some (ta, _) ->
          report env "%s takes a datetime without a zone, not %a." what Type.pp
            ta;
          Broken
      | None -> Broken)
  | Windows { zone; every; period; a } -> (
      match dated env what zone a with
      | Some (ta, a') ->
          calendar_step env what ta every;
          calendar_step env what ta period;
          known
            (Column (Type.list ta))
            (Windows { zone; every; period; a = a' })
      | None -> Broken)
  | Parse_with (fmt, ty, a) ->
      on_operand env a (fun t a' ->
          match t with
          | Column (String | Categorical _) -> (
              let z = has_zone_directive fmt in
              match ty with
              | Datetime { zone = Some _; _ } when not z ->
                  report env
                    "%s reads %a, which has a zone, so its format needs %%z."
                    what Type.pp ty;
                  Broken
              | (Date | Clock _ | Datetime { zone = None; _ }) when z ->
                  report env
                    "%s reads %a, which has no zone, so its format takes no \
                     %%z."
                    what Type.pp ty;
                  Broken
              | Date | Clock _ | Datetime _ ->
                  known (Column ty) (Parse_with (fmt, ty, a'))
              | _ -> not_temporal env what ty)
          | t ->
              report env "%s takes text, not %a." what pp_typing t;
              Broken)
  | Format_with (fmt, a) -> (
      match column a with
      | None -> Broken
      | Some (ta, a') -> (
          let ok () = known (Column Type.string) (Format_with (fmt, a')) in
          match ta with
          | Datetime { zone = Some _; _ } -> ok ()
          | (Date | Clock _ | Datetime _) when has_zone_directive fmt ->
              report env "%s writes %%z of a datetime with a zone, not of %a."
                what Type.pp ta;
              Broken
          | Date | Clock _ | Datetime _ -> ok ()
          | _ -> not_temporal env what ta))

(* [dated env what zone a] binds [a], a date or a datetime whose calendar [zone]
   reads. *)
and dated : type a s.
    env ->
    string ->
    Tz.zone option ->
    (a, s) t ->
    (a Type.t * (a, erased) t) option =
 fun env what zone a ->
  match resolve env a (elab env a) with
  | Some (Column (Date as ta), a') ->
      zone_rule env what zone None;
      Some (ta, a')
  | Some (Column (Datetime { zone = z; _ } as ta), a') ->
      zone_rule env what zone z;
      Some (ta, a')
  | Some (t, _) ->
      report env "%s takes a date or a datetime, not %a." what pp_typing t;
      None
  | None -> None

let env schema = { schema; problems = ref []; seen = Hashtbl.create 8 }

let finish env v =
  match !(env.problems) with [] -> Ok v | ps -> Error (List.rev ps)

let bind_predicate schema p =
  let env = env schema in
  match expect env p bool_typing (elab env p) with
  | Some b -> finish env (b :> (_, _) t)
  | None -> Error (List.rev !(env.problems))

let typing e =
  match e.typing with
  | Some t -> t
  | None -> invalid_arg "Expr.typing: the expression is not bound"

let bind_value schema e =
  let env = env schema in
  match resolve_value env e (elab env e) with
  | Some (_, b) -> finish env (b :> (_, _) t)
  | None -> Error (List.rev !(env.problems))

let bind_out schema o =
  let env = env schema in
  finish env (outs env o)

(* Analyses *)

let check_bound fn e =
  if Option.is_none e.typing then err "Expr.%s: the expression is not bound" fn

let reads e =
  check_bound "reads" e;
  let add acc n = if List.mem n acc then acc else n :: acc in
  let rec walk : type a s. string list -> (a, s) t -> string list =
   fun acc e ->
    match e.node with
    | Handle (_, n) | Ext_handle (_, n) | Read (_, n) -> add acc n
    | Over { by; order; e } ->
        let acc = List.fold_left add acc by in
        walk
          (List.fold_left (fun acc (k : Order.t) -> add acc k.name) acc order)
          e
    | Rolling (Window.Times { on; _ }, a) -> walk (add acc on) a
    | n -> List.fold_left (fun acc (Packed e) -> walk acc e) acc (operands n)
  in
  List.rev (walk [] e)

let row_local e =
  check_bound "row_local" e;
  let rec local : type a s. (a, s) t -> bool =
   fun e ->
    match e.node with
    | Over _ | Rolling _ | Shift _ | Rank _ -> false
    | n -> List.for_all (fun (Packed e) -> local e) (operands n)
  in
  local e

let reduces e =
  check_bound "reduces" e;
  let rec per_frame : type a s. (a, s) t -> bool =
   fun e ->
    match e.node with
    | Rows | Reduce _ -> true
    | Over _ | Rolling _ -> false
    | n -> List.exists (fun (Packed e) -> per_frame e) (operands n)
  in
  per_frame e

(* [widens a ty] is [true] iff [ty] contains the column type of [a], so that a
   cast of [a] to [ty] keeps every value. *)
let widens : type a b s. (a, s) t -> b Type.t -> bool =
 fun a ty ->
  match a.typing with
  | Some (Column ta) -> (
      match Kind.equal_witness (Type.kind ta) (Type.kind ty) with
      | Some Equal -> (
          match Type.common [ ta; ty ] with
          | Some c -> Type.equal c ty
          | None -> false)
      | None -> false)
  | _ -> false

let can_fail e =
  check_bound "can_fail" e;
  let rec fails : type a s. (a, s) t -> bool =
   fun e ->
    match e.node with
    | Cast (ty, a) when not (widens a ty) -> true
    | Text (Parse _, _)
    | Calendar
        ( Add_span _ | Diff _ | Floor _ | Offset _ | Localize _ | Windows _
        | Parse_with _ )
    | Of_option _ | App _ | Batch _ ->
        true
    | n -> List.exists (fun (Packed e) -> fails e) (operands n)
  in
  fails e

type mapper = { map : 'a 's. ('a, 's) t -> ('a, 's) t }

(* [map_node m n] is the bound node [n] with [m] applied to each of its
   operands. *)
let map_node : type a. mapper -> a node -> a node =
 fun { map } n ->
  match n with
  | Handle _ | Ext_handle _ | Read _ | Lit _ | Null | Rows | Const _ -> n
  (* An unbound node, which no bound expression holds. *)
  | Record_outs _ -> n
  | Int (op, a, b) -> Int (op, map a, map b)
  | Float (op, a, b) -> Float (op, map a, map b)
  | Compare (op, a, b) -> Compare (op, map a, map b)
  | Logic (op, a, b) -> Logic (op, map a, map b)
  | Not a -> Not (map a)
  | If (c, a, b) -> If (map c, map a, map b)
  | Is_null a -> Is_null (map a)
  | Coalesce es -> Coalesce (List.map map es)
  | Is_in (vs, a) -> Is_in (vs, map a)
  | Cut (edges, a) -> Cut (edges, map a)
  | Cast (ty, a) -> Cast (ty, map a)
  | Lift (fn, a) -> Lift (fn, map a)
  | Lift2 (fn, a, b) -> Lift2 (fn, map a, map b)
  | Nx_unary (op, a) -> Nx_unary (op, map a)
  | Nx_binary (op, a, b) -> Nx_binary (op, map a, map b)
  | Nx_compare (op, a, b) -> Nx_compare (op, map a, map b)
  | Nx_where (c, a, b) -> Nx_where (map c, map a, map b)
  | Nx_cast (ty, a) -> Nx_cast (ty, map a)
  | Reduce (r, a) -> Reduce (r, map a)
  | Over { by; order; e } -> Over { by; order; e = map e }
  | Rolling (w, a) -> Rolling (w, map a)
  | Shift (k, a) -> Shift (k, map a)
  | Rank a -> Rank (map a)
  | App (f, a) -> App (map f, map a)
  | Option a -> Option (map a)
  | Of_option a -> Of_option (map a)
  | Store (ty, a) -> Store (ty, map a)
  | Batch (f, a) -> Batch (f, map a)
  | Fields fs -> Fields (List.map (fun (n, Packed e) -> (n, Packed (map e))) fs)
  | Field (k, name, r) -> Field (k, name, map r)
  | Storage (d, a) -> Storage (d, map a)
  | Wrap (d, a) -> Wrap (d, map a)
  | Text (op, a) -> Text (op, map a)
  | Calendar op ->
      Calendar
        (match op with
        | Add_span (a, d) -> Add_span (map a, map d)
        | Diff (a, b) -> Diff (map a, map b)
        | Part (f, z, a) -> Part (f, z, map a)
        | Floor (z, s, a) -> Floor (z, s, map a)
        | Offset (z, s, a) -> Offset (z, s, map a)
        | Localize { zone; ambiguous; gap; a } ->
            Localize { zone; ambiguous; gap; a = map a }
        | Windows { zone; every; period; a } ->
            Windows { zone; every; period; a = map a }
        | Parse_with (fmt, ty, a) -> Parse_with (fmt, ty, map a)
        | Format_with (fmt, a) -> Format_with (fmt, map a))

let rename f e =
  check_bound "rename" e;
  let key (k : Order.t) =
    let k' = if k.desc then Order.desc (f k.name) else Order.asc (f k.name) in
    if k.nulls_first then Order.nulls_first k' else k'
  in
  let rec go : type a s. (a, s) t -> (a, s) t =
   fun e ->
    typed (typing e)
      (match e.node with
      | Handle (k, n) -> Handle (k, f n)
      | Ext_handle (d, n) -> Ext_handle (d, f n)
      | Read (ty, n) -> Read (ty, f n)
      | Over { by; order; e } ->
          Over { by = List.map f by; order = List.map key order; e = go e }
      | Rolling (Window.Times { on; before; after }, a) ->
          Rolling (Window.time ~after ~before (f on), go a)
      | n -> map_node { map = go } n)
  in
  go e

(* Constants *)

(* [literal e] is [Some (ty, Some v)] if [e] is a literal of value [v] at its
   column type [ty], and [Some (ty, None)] if it is null. *)
let literal : type a s. (a, s) t -> (a Type.t * a option) option =
 fun e ->
  match (e.node, e.typing) with
  | Null, Some (Column ty) -> Some (ty, None)
  | Lit (_, v), Some (Column ty) ->
      Option.map (fun v -> (ty, Some v)) (Type.value ty v)
  | _ -> None

let is_lit e = match e.node with Lit _ -> true | _ -> false
let is_null_lit e = match e.node with Null -> true | _ -> false

(* [at_typing t v] is the literal [v] typed [t], null if [v] is [None]. *)
let at_typing : type a s. a typing -> a option -> (a, s) t =
 fun t v ->
  match (t, v) with
  | Column ty, Some v -> typed t (Lit (Type.kind ty, v))
  | _ -> typed t Null

(* [with_typing t e] is [e] if its typing is [t]. *)
let with_typing : type a s0 s1. a typing -> (a, s0) t -> (a, s1) t option =
 fun t e ->
  match e.typing with
  | Some t' when typing_equal t t' -> Some (typed t' e.node)
  | _ -> None

type evaluator = { eval : 'a. ('a, row) t -> 'a option option }

(* [is_constant e] is [true] iff [e] is a literal or null of a column type. *)
let is_constant e =
  match (e.node, e.typing) with
  | (Lit _ | Null), Some (Column _) -> true
  | _ -> false

(* [folded ev e] is the literal or the operand that the bound operation [e] is,
   if it is one, with [e]'s typing. *)
let folded : type a s0 s. evaluator -> (a, s0) t -> (a, s) t option =
 fun ev e ->
  let t = typing e in
  let null () = Some (at_typing t None) in
  let constants () =
    List.for_all (fun (Packed e) -> is_constant e) (operands e.node)
  in
  match e.node with
  | (Int _ | Float _ | Compare _ | Not _) when constants () ->
      Option.map (at_typing t) (ev.eval (typed t e.node))
  | Logic (o, a, b) -> (
      let truth e = Option.map snd (literal e) in
      match (o, truth a, truth b) with
      | And, Some (Some false), _ | And, _, Some (Some false) ->
          Some (at_typing t (Some false))
      | Or, Some (Some true), _ | Or, _, Some (Some true) ->
          Some (at_typing t (Some true))
      | And, Some (Some true), _ | Or, Some (Some false), _ -> with_typing t b
      | And, _, Some (Some true) | Or, _, Some (Some false) -> with_typing t a
      | _, Some None, Some None -> null ()
      | _ -> None)
  | Is_null a ->
      if is_lit a then Some (at_typing t (Some false))
      else if is_null_lit a then Some (at_typing t (Some true))
      else None
  | If (c, a, b) -> (
      match literal c with
      | Some (_, Some true) -> with_typing t a
      | Some (_, (Some false | None)) -> with_typing t b
      | None -> None)
  | Store (_, a) when is_lit a || is_null_lit a -> with_typing t a
  | Coalesce es -> (
      let rec live = function
        | [] -> []
        | e :: es ->
            if is_null_lit e then live es
            else if is_lit e then [ e ]
            else e :: live es
      in
      match live es with
      | [] -> null ()
      | [ e ] when Option.is_some (with_typing t e) -> with_typing t e
      | es' when List.compare_lengths es' es < 0 ->
          Some (typed t (Coalesce es'))
      | _ -> None)
  | _ -> None

let fold_constants ev e =
  check_bound "fold_constants" e;
  let rec fold : type a s. (a, s) t -> (a, s) t =
   fun e ->
    let e = typed (typing e) (map_node { map = fold } e.node) in
    Option.value ~default:e (folded ev e)
  in
  fold e

(* Outputs *)

let check_utf_8 fn s =
  if not (String.is_valid_utf_8 s) then err "%s: %S is not valid UTF-8" fn s

let ( := ) name e =
  check_utf_8 "Expr.( := )" name;
  Named (name, Packed e)

let out_name = function Named (n, _) -> Some n | _ -> None
let keep sel = Keep sel
let across k sel f = Across (k, sel, f)
let each sel column = Each (sel, column)

(* Literals *)

let int n = make (Lit (Kind.int, n))
let float x = make (Lit (Kind.float, x))
let bool b = make (Lit (Kind.bool, b))

let string s =
  check_utf_8 "Expr.string" s;
  make (Lit (Kind.string, s))

let instant t = make (Lit (Kind.instant, t))
let span d = make (Lit (Kind.span, d))
let date d = make (Lit (Kind.date, d))

(* Nested values *)

let record os = make (Record_outs os)
let field k name r = make (Field (k, name, r))
let unpack r = Unpack r

(* Text *)

module Str = struct
  type nonrec pattern = pattern

  let check_piece fn s =
    if String.equal s "" then err "Expr.Str.%s: empty pattern" fn;
    check_utf_8 ("Expr.Str." ^ fn) s

  let literal s =
    check_piece "literal" s;
    Literal s

  let prefix s =
    check_piece "prefix" s;
    Prefix s

  let suffix s =
    check_piece "suffix" s;
    Suffix s

  let pieces ss =
    if List.is_empty ss then err "Expr.Str.pieces: no pieces";
    List.iter (check_piece "pieces") ss;
    Pieces ss

  let text op a = make (Text (op, a))
  let length a = text Length a

  let slice ~offset ~length a =
    if length < 0 then err "Expr.Str.slice: ~length:%d is negative" length;
    text (Slice { offset; length }) a

  let lower a = text Lower a
  let upper a = text Upper a
  let matches p a = text (Matches p) a
  let parse ty a = text (Parse ty) a
end

(* Time *)

module Temporal = struct
  type nonrec field = field
  type nonrec policy = policy

  let calendar op = make (Calendar op)

  let check_step fn = function
    | (Time.Months n | Time.Weeks n | Time.Days n) when n > 0 -> ()
    | Time.Exact d when Time.Span.to_ns d > 0L -> ()
    | step -> err "Expr.Temporal.%s: %a is not positive" fn Time.pp_step step

  (* [check_format fmt] raises unless [fmt] holds only the directives of
     {!parse}. *)
  let check_format fn fmt =
    let n = String.length fmt in
    let rec loop i =
      if i < n then
        if Char.equal fmt.[i] '%' then
          if i + 1 = n then err "Expr.Temporal.%s: %S ends with %%" fn fmt
          else
            match fmt.[i + 1] with
            | 'Y' | 'm' | 'd' | 'H' | 'M' | 'S' | 'f' | 'z' | '%' -> loop (i + 2)
            | c ->
                err "Expr.Temporal.%s: %S holds the unknown directive %%%c" fn
                  fmt c
        else loop (i + 1)
    in
    check_utf_8 ("Expr.Temporal." ^ fn) fmt;
    loop 0

  let add a d = calendar (Add_span (a, d))
  let diff a b = calendar (Diff (a, b))
  let field f ?zone a = calendar (Part (f, zone, a))

  let floor ?zone step a =
    check_step "floor" step;
    calendar (Floor (zone, step, a))

  let offset ?zone step a = calendar (Offset (zone, step, a))

  let localize zone ~ambiguous ~gap a =
    calendar (Localize { zone; ambiguous; gap; a })

  let windows ?zone ~every ~period a =
    check_step "windows" every;
    check_step "windows" period;
    calendar (Windows { zone; every; period; a })

  let parse fmt ty a =
    check_format "parse" fmt;
    calendar (Parse_with (fmt, ty, a))

  let format fmt a =
    check_format "format" fmt;
    calendar (Format_with (fmt, a))
end

(* Reductions *)

let reduce r a = make (Reduce (r, a))
let count a = reduce Count a
let sum a = reduce Sum a
let min a = reduce Min a
let max a = reduce Max a
let first a = reduce First a
let last a = reduce Last a
let only a = reduce Only a
let mean a = reduce Mean a
let std a = reduce Std a
let var a = reduce Var a
let median a = reduce Median a

let quantile p a =
  if not (0. <= p && p <= 1.) then err "Expr.quantile: %g is not in [0;1]" p;
  reduce (Quantile p) a

let ewm ~alpha a =
  if not (0. < alpha && alpha <= 1.) then
    err "Expr.ewm: ~alpha:%g is not in (0;1]" alpha;
  reduce (Ewm alpha) a

let n_unique a = reduce N_unique a
let arg_min a = reduce Arg_min a
let arg_max a = reduce Arg_max a
let collect a = reduce Collect a

(* Frames *)

let over ?(by = []) ?(order = []) e = make (Over { by; order; e })
let rolling w e = make (Rolling (w, e))
let shift n a = make (Shift (n, a))
let rank a = make (Rank a)

(* OCaml values *)

let const v = make (Const v)
let ( $ ) f a = make (App (f, a))
let option a = make (Option a)
let of_option a = make (Of_option a)
let store ty a = make (Store (ty, a))
let batch f x = make (Batch (f, x))

(* Elementwise operations *)

let int_op op a b = make (Int (op, a, b))
let float_op op a b = make (Float (op, a, b))
let cmp op a b = make (Compare (op, a, b))
let ( + ) a b = int_op Add a b
let ( - ) a b = int_op Sub a b
let ( * ) a b = int_op Mul a b
let ( / ) a b = int_op Div a b
let ( mod ) a b = int_op Mod a b
let ( +. ) a b = float_op Add a b
let ( -. ) a b = float_op Sub a b
let ( *. ) a b = float_op Mul a b
let ( /. ) a b = float_op Div a b
let ( ** ) a b = float_op Pow a b
let ( = ) a b = cmp `Eq a b
let ( <> ) a b = cmp `Ne a b
let ( < ) a b = cmp `Lt a b
let ( > ) a b = cmp `Gt a b
let ( <= ) a b = cmp `Le a b
let ( >= ) a b = cmp `Ge a b
let ( && ) a b = make (Logic (And, a, b))
let ( || ) a b = make (Logic (Or, a, b))
let not a = make (Not a)
let if_ c a b = make (If (c, a, b))
let is_null a = make (Is_null a)
let coalesce es = make (Coalesce es)
let is_in vs a = make (Is_in (vs, a))
let cut edges a = make (Cut (Array.copy edges, a))
let cast ty a = make (Cast (ty, a))
let nx fn a = make (Lift (fn, a))
let nx2 fn a b = make (Lift2 (fn, a, b))
