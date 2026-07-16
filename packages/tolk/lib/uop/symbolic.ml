(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Three-layer symbolic folder. Rules are expressed against Upat
   patterns; each rule receives the capture bindings and returns the
   rewritten node. *)

(* Uop / Const helpers *)

module U = Uop

let dtype_v u = Dtype.val_of (Uop.dtype u)

let const_int_v u =
  match Uop.op u, Uop.arg u with
  | Ops.Const, Uop.Arg.Value c ->
      (match Const.view c with
       | Const.Int n -> Some (Int64.to_int n)
       | _ -> None)
  | _ -> None

let const_bool_v u =
  match Uop.op u, Uop.arg u with
  | Ops.Const, Uop.Arg.Value c ->
      (match Const.view c with
       | Const.Bool b -> Some b
       | _ -> None)
  | _ -> None

let is_invalid_const u =
  match Uop.op u, Uop.arg u with
  | Ops.Const, Uop.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

let pm_remove_invalid : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    op ~name:"i" Ops.Const
    => fun bs ->
         let i = bs $ "i" in
         match U.dtype i, U.arg i with
         | Dtype.Val dtype, U.Arg.Value c when Const.view c = Const.Invalid ->
             Some (U.const (Const.zero dtype))
         | _ -> None;
  ]

let is_index_dtype dtype = Dtype.scalar dtype = Dtype.Index

let cast_to_index node =
  match U.op node, U.src node with
  | Ops.Cast, [| src |] when is_index_dtype (U.dtype node) -> Some src
  | _ -> None

let cast_from_int node =
  match U.op node, U.src node with
  | Ops.Cast, [| src |] when Dtype.is_int (U.dtype src) -> Some src
  | _ -> None

let select_index_dtype node =
  let scalar =
    if U.vmin node < -0x8000_0000 || U.vmax node > 0x7fff_ffff
    then Dtype.Val.int64
    else Dtype.Val.int32
  in
  let count = Dtype.count (U.dtype node) in
  Dtype.Val (if count <= 1 then scalar else Dtype.Val.vec count scalar)

let retype_const node dtype =
  match U.op node, U.arg node, dtype with
  | Ops.Const, U.Arg.Value c, Dtype.Val dtype ->
      Some (U.const (Const.of_view (Dtype.Val.scalarize dtype) (Const.view c)))
  | _ -> None

let index_range_value node (r : U.range_view) =
  let size =
    match cast_to_index r.size with
    | Some size -> Some size
    | None when Dtype.is_int (U.dtype r.size) -> Some r.size
    | None -> None
  in
  match size with
  | None -> None
  | Some size ->
      Some
        (U.replace node
           ~src:(Array.of_list (size :: r.parents))
           ~dtype:(U.dtype size) ())

(* Concrete integer value of a index binary operand: the source of a
   [cast_to_index], or the operand itself when it is already a concrete int.
   Accepting a bare int lets the index cast bubble past an already-lowered
   operand (e.g. an integer stride) instead of stalling below it. *)
let index_binary_operand node =
  match cast_to_index node with
  | Some src -> Some src
  | None -> if Dtype.is_int (U.dtype node) then Some node else None

let lower_index_binary node =
  let src = U.src node in
  (* Comparisons of index operands lower alongside index-valued
     binaries: the reference matches any binary whose operands are
     index casts and casts the result back to the node dtype (a no-op
     for the bool result of a comparison). *)
  let index_operands =
    Array.length src = 2
    && is_index_dtype (U.dtype src.(0))
    && is_index_dtype (U.dtype src.(1))
  in
  if
    not (Ops.Group.is_binary (U.op node))
    || Array.length src <> 2
    || not (is_index_dtype (U.dtype node) || index_operands)
  then None
  else
    match index_binary_operand src.(0), index_binary_operand src.(1) with
    | Some lhs, Some rhs
      when cast_to_index src.(0) <> None || cast_to_index src.(1) <> None ->
        let dtype =
          Dtype.least_upper_dtype
            [ select_index_dtype node; U.dtype lhs; U.dtype rhs ]
        in
        let value =
          U.alu_binary ~op:(U.op node)
            ~lhs:(U.cast ~src:lhs ~dtype)
            ~rhs:(U.cast ~src:rhs ~dtype)
        in
        if Dtype.equal (U.dtype value) (U.dtype node) then Some value
        else Some (U.cast ~src:value ~dtype:(U.dtype node))
    | _ -> None

let lower_index_const node =
  match U.op node, U.arg node with
  | Ops.Const, U.Arg.Value c
    when is_index_dtype (U.dtype node) && Const.view c <> Const.Invalid ->
      Option.map
        (fun src -> U.cast ~src ~dtype:(U.dtype node))
        (retype_const node (select_index_dtype node))
  | _ -> None

let lower_index_where node =
  match U.op node, U.src node with
  | Ops.Where, [| cond; t; f |] when is_index_dtype (U.dtype node) -> (
      match cast_to_index t, cast_to_index f with
      | Some t, Some f ->
          let dtype = Dtype.least_upper_dtype [ U.dtype t; U.dtype f ] in
          let value =
            U.O.where cond (U.cast ~src:t ~dtype) (U.cast ~src:f ~dtype)
          in
          Some (U.cast ~src:value ~dtype:(U.dtype node))
      | _ -> None)
  | _ -> None

let lower_index_range node =
  match U.as_range node with
  | Some r when is_index_dtype (U.dtype node) ->
      Option.map
        (fun range -> U.cast ~src:range ~dtype:(U.dtype node))
        (index_range_value node r)
  | _ -> None

let lower_index_stack node =
  match U.op node, U.src node with
  | Ops.Stack, src when is_index_dtype (U.dtype node) ->
      let inners = Array.map cast_to_index src in
      if Array.exists Option.is_none inners then None
      else
        let dtype = select_index_dtype node in
        let scalar_dtype = Dtype.scalarize dtype in
        let srcs =
          Array.to_list inners
          |> List.map (function
               | Some src -> U.cast ~src ~dtype:scalar_dtype
               | None -> assert false)
        in
        Some
          (U.cast
             ~src:(U.stack ~dtype:(Dtype.Val.scalarize (Dtype.val_of dtype)) srcs)
             ~dtype:(U.dtype node))
  | _ -> None

let lower_index_special node =
  match U.op node, U.src node with
  | Ops.Special, [| size |] when is_index_dtype (U.dtype node) -> (
      let concrete_size =
        match cast_to_index size with
        | Some size -> Some size
        | None when is_index_dtype (U.dtype size) ->
            retype_const size (select_index_dtype size)
        | None when Dtype.is_int (U.dtype size) -> Some size
        | None -> None
      in
      match concrete_size with
      | None -> None
      | Some size ->
          Some
            (U.cast
               ~src:(U.replace node ~src:[| size |] ~dtype:Dtype.int32 ())
               ~dtype:(U.dtype node)))
  | _ -> None

let lower_index_param node =
  match U.op node, U.addrspace node with
  | Ops.Param, Some Dtype.Alu when is_index_dtype (U.dtype node) ->
      Some
        (U.cast
           ~src:(U.replace node ~dtype:Dtype.int32 ())
           ~dtype:(U.dtype node))
  | _ -> None

let lower_index_bind node =
  match U.as_bind node with
  | Some { var; value } when is_index_dtype (U.dtype node) -> (
      match cast_to_index var, cast_to_index value with
      | Some var, Some value ->
          Some (U.cast ~src:(U.bind ~var ~value) ~dtype:(U.dtype node))
      | _ -> None)
  | _ -> None

let lower_index_casts node =
  match U.as_index node with
  | Some { ptr; idxs } ->
      let changed = ref false in
      let idxs =
        List.map
          (fun idx ->
            match cast_from_int idx with
            | None -> idx
            | Some idx ->
                changed := true;
                idx)
          idxs
      in
      if !changed then Some (U.replace node ~src:(Array.of_list (ptr :: idxs)) ())
      else None
  | None -> (
      match U.op node, U.src node with
      | Ops.Shrink, [| buf; idx; len |] -> (
          match cast_from_int idx, cast_from_int len with
          | None, None -> None
          | idx', len' ->
              Some
                (U.replace node
                   ~src:
                     [| buf;
                        Option.value idx' ~default:idx;
                        Option.value len' ~default:len |]
                   ()))
      | _ -> None)

let strip_sink_like_index_casts node =
  match U.op node with
  | Ops.Sink | Ops.Noop | Ops.End ->
      let changed = ref false in
      let src =
        Array.map
          (fun s ->
            match cast_to_index s with
            | None -> s
            | Some s ->
                changed := true;
                s)
          (U.src node)
      in
      if !changed then Some (U.replace node ~src ()) else None
  | _ -> None

let lower_index_dtype_rule node =
  match lower_index_binary node with
  | Some _ as r -> r
  | None ->
      List.find_map (fun f -> f node)
        [
          lower_index_const;
          lower_index_where;
          lower_index_range;
          lower_index_stack;
          lower_index_special;
          lower_index_param;
          lower_index_bind;
          lower_index_casts;
          strip_sink_like_index_casts;
        ]

let pm_lower_index_dtype : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    ops ~name:"node" Ops.Group.all
    => fun bs -> lower_index_dtype_rule (bs $ "node");
  ]

let is_zero_const node =
  match U.op node, U.arg node with
  | Ops.Const, U.Arg.Value c -> (
      match Const.view c with
      | Const.Int 0L | Const.Float 0.0 | Const.Bool false -> true
      | Const.Int _ | Const.Float _ | Const.Bool _ | Const.Invalid -> false)
  | _ -> false

let invalid_where node =
  match U.op node, U.src node with
  | Ops.Where, [| gate; idx; invalid |] when is_invalid_const invalid ->
      Some (gate, idx)
  | _ -> None

let indexed_or_casted node =
  match U.as_index node with
  | Some index -> Some (node, index, None)
  | None -> (
      match U.op node, U.src node with
      | Ops.Cast, [| idx |] -> (
          match U.as_index idx with
          | Some index -> Some (idx, index, Some (U.dtype node))
          | None -> None)
      | _ -> None)

let index_valid idx =
  match invalid_where idx with
  | Some (valid, idx) -> valid, idx
  | None -> U.const_bool true, idx

let const_true_uprod = function
  | [] -> U.const_bool true
  | xs -> U.uprod xs

let ranges_subset a b =
  List.for_all (fun r -> List.exists (U.equal r) b) a

let index_nodes u =
  U.toposort u |> List.filter (fun n -> U.op n = Ops.Index)

let where_on_load cond value =
  match indexed_or_casted value with
  | None -> None
  | Some (index_node, { ptr; idxs }, cast_dtype) -> (
      match idxs with
      | [ idx ] ->
          let load_valid, idx = index_valid idx in
          let in_load = U.split_uop load_valid Ops.And in
          let idx_ranges = U.ranges idx in
          let idx_indexes = index_nodes idx in
          let can_move clause =
            ranges_subset (U.ranges clause) idx_ranges
            && List.for_all
                 (fun node ->
                   U.op node <> Ops.Index
                   || List.exists (U.equal node) idx_indexes)
                 (U.toposort clause)
          in
          let where_clauses = U.split_uop cond Ops.And in
          let moved, keep =
            where_clauses
            |> List.filter (fun clause ->
                 not (List.exists (U.equal clause) in_load))
            |> List.partition can_move
          in
          if List.length keep = List.length where_clauses then None
          else
            let valid = U.uprod (load_valid :: moved) in
            let idx =
              if U.equal valid (U.const_bool true) then idx
              else U.O.where valid idx (U.invalid ())
            in
            let next = U.replace index_node ~src:[| ptr; idx |] () in
            let next =
              match cast_dtype with
              | None -> next
              | Some dtype -> U.cast ~src:next ~dtype
            in
            Some (U.O.where (const_true_uprod keep) next (U.zero_like next))
      | _ -> None)

let move_where_on_load_rule node =
  match U.op node, U.src node with
  | Ops.Where, [| cond; lhs; rhs |] when is_zero_const rhs ->
      where_on_load cond lhs
  | Ops.Where, [| cond; lhs; rhs |] when is_zero_const lhs ->
      where_on_load (U.O.not_ cond) rhs
  | _ -> None

let pm_move_where_on_load : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    op ~name:"where" Ops.Where
    => fun bs -> move_where_on_load_rule (bs $ "where");
  ]

let const_float_v u =
  match Uop.op u, Uop.arg u with
  | Ops.Const, Uop.Arg.Value c ->
      (match Const.view c with
       | Const.Float f -> Some f
       | _ -> None)
  | _ -> None

let const_nan_like u =
  match Uop.dtype u with
  | Dtype.Val v when Dtype.Val.is_float v ->
      Some (Uop.const (Const.of_scalar v (`Float Float.nan)))
  | _ -> None

let is_int_uop u =
  match Uop.dtype u with
  | Dtype.Val v -> Dtype.Val.is_int v
  | Dtype.Ptr _ -> false

let range_kind_is_reduce r =
  match Uop.as_range r with
  | Some { kind; _ } ->
      Axis_type.equal kind Axis_type.Reduce
      || Axis_type.equal kind Axis_type.Group_reduce
      || Axis_type.equal kind Axis_type.Unroll
  | None -> false

let depends_on_reduce_range u =
  List.exists range_kind_is_reduce (Uop.ranges u)

let depends_on_nonreduce_range u =
  List.exists
    (fun r -> not (range_kind_is_reduce r))
    (Uop.ranges u)

let same_shape a b =
  try
    let a = Uop.shape a and b = Uop.shape b in
    List.length a = List.length b && List.for_all2 Uop.equal a b
  with Invalid_argument _ -> false

let rec gcd_int a b =
  if b = 0 then abs a else gcd_int b (a mod b)

let floor_div x y =
  if y = 0 then 0
  else
    let q = x / y and r = x mod y in
    if r <> 0 && ((r < 0) <> (y < 0)) then q - 1 else q

let floor_mod x y = x - (floor_div x y * y)

let int_bounds (v : Dtype.Val.t) =
  match Dtype.min (Dtype.Val v), Dtype.max (Dtype.Val v) with
  | `SInt lo, `SInt hi -> Some (Int64.to_int lo, Int64.to_int hi)
  | `UInt lo, `UInt hi -> Some (Int64.to_int lo, Int64.to_int hi)
  | _ -> None

let overflows u (v : Dtype.Val.t) =
  match int_bounds v with
  | Some (lo, hi) -> Uop.vmin u < lo || Uop.vmax u > hi
  | None -> true

let rec contains_param_slot slot u =
  (match Uop.as_param u with
   | Some { param; _ } -> param.slot = slot
   | None -> false)
  || Array.exists (contains_param_slot slot) (Uop.src u)

(* {1 exec_alu}

   Constant folding for scalar ALU ops. Promotes bool operands to int
   where the op is arithmetic and dispatches on int-vs-float. The result
   is NOT truncated to the target dtype: like the reference's
   [exec_alu(..., truncate_output=False)] at its symbolic fold site, the
   folded constant keeps full host precision and only the renderer or
   runtime narrows it. Any binary op with an [Invalid] operand folds to
   [Invalid], regardless of dtype. *)

let const_as_float c =
  match Const.view c with
  | Const.Float f -> Some f
  | Const.Int n -> Some (Int64.to_float n)
  | Const.Bool b -> Some (if b then 1.0 else 0.0)
  | Const.Invalid -> None

let const_as_int c =
  match Const.view c with
  | Const.Int n -> Some (Int64.to_int n)
  | Const.Bool b -> Some (if b then 1 else 0)
  | Const.Float _ | Const.Invalid -> None

let const_of_target ~(target : Dtype.Val.t) v =
  let open Const in
  if Dtype.Val.is_bool target then
    (match v with
     | `Bool b -> Some (of_scalar target (`Bool b))
     | `Int n -> Some (of_scalar target (`Int (Int64.of_int n)))
     | `Float f -> Some (of_scalar target (`Float f)))
  else if Dtype.Val.is_int target then
    (match v with
     | `Bool b -> Some (of_scalar target (`Bool b))
     | `Int n -> Some (of_scalar target (`Int (Int64.of_int n)))
     | `Float f -> Some (of_scalar target (`Float f)))
  else if Dtype.Val.is_float target then
    (match v with
     | `Bool b -> Some (of_scalar target (`Bool b))
     | `Int n -> Some (of_scalar target (`Float (float_of_int n)))
     | `Float f -> Some (of_scalar target (`Float f)))
  else None

let any_invalid args = List.exists (fun c -> Const.view c = Const.Invalid) args

let exec_unary op (target : Dtype.Val.t) c =
  let is_float_target = Dtype.Val.is_float target in
  if is_float_target then
    match const_as_float c with
    | None -> None
    | Some x ->
        let r = match op with
          | Ops.Neg -> Some (-. x)
          | Ops.Exp2 ->
              Some (try 2.0 ** x with _ -> Float.infinity)
          | Ops.Log2 ->
              Some
                (if x > 0.0 then log x /. log 2.0
                 else if x = 0.0 then Float.neg_infinity
                 else Float.nan)
          | Ops.Sqrt ->
              Some (if x >= 0.0 then sqrt x else Float.nan)
          | Ops.Reciprocal ->
              Some
                (if x <> 0.0 then 1.0 /. x
                 else Float.copy_sign Float.infinity x)
          | Ops.Sin ->
              Some
                (if not (Float.is_finite x) then Float.nan else sin x)
          | Ops.Trunc -> Some (Float.trunc x)
          | _ -> None
        in
        (match r with
         | None -> None
         | Some f -> const_of_target ~target (`Float f))
  else
    match const_as_int c with
    | None -> None
    | Some x ->
        let r = match op with
          | Ops.Neg -> Some (-x)
          | Ops.Trunc -> Some x
          | _ -> None
        in
        (match r with
         | None -> None
         | Some n -> const_of_target ~target (`Int n))

let exec_binary op (target : Dtype.Val.t) a b =
  let result_is_bool = Dtype.Val.is_bool target in
  if result_is_bool then
    match op with
    | Ops.Cmplt | Ops.Cmpne | Ops.Cmpeq ->
        (match op with
         (* CMPEQ/CMPNE follow IEEE for floats (nan <> nan, 0.0 = -0.0);
            for ints/bool the structural [Const.equal] is value equality. *)
         | Ops.Cmpeq ->
             (match Const.view a, Const.view b with
              | Const.Float x, Const.Float y -> Some (Const.bool (x = y))
              | _ -> Some (Const.bool (Const.equal a b)))
         | Ops.Cmpne ->
             (match Const.view a, Const.view b with
              | Const.Float x, Const.Float y -> Some (Const.bool (x <> y))
              | _ -> Some (Const.bool (not (Const.equal a b))))
         (* CMPLT compares ints exactly; floats compare under IEEE. *)
         | Ops.Cmplt ->
             (match const_as_int a, const_as_int b with
              | Some x, Some y -> Some (Const.bool (x < y))
              | _ ->
                  (match const_as_float a, const_as_float b with
                   | Some x, Some y -> Some (Const.bool (x < y))
                   | _ -> None))
         | _ -> None)
    | Ops.And | Ops.Or | Ops.Xor ->
        (match const_as_int a, const_as_int b with
         | Some x, Some y ->
             let r = match op with
               | Ops.And -> x land y
               | Ops.Or -> x lor y
               | Ops.Xor -> x lxor y
               | _ -> 0
             in
             Some (Const.bool (r <> 0))
         | _ -> None)
    | _ -> None
  else if Dtype.Val.is_float target then
    match const_as_float a, const_as_float b with
    | Some x, Some y ->
        let r = match op with
          | Ops.Add -> Some (x +. y)
          | Ops.Sub -> Some (x -. y)
          | Ops.Mul -> Some (x *. y)
          | Ops.Fdiv ->
              Some
                (if y = 0.0 then Float.copy_sign Float.infinity x
                 else x /. y)
          | Ops.Max -> Some (max x y)
          | Ops.Pow ->
              (try
                 let p = x ** y in
                 if Float.is_nan p || not (Float.is_finite p) then
                   if x > 0.0 && not (Float.is_finite y) then
                     if abs_float x > 1.0 = (y > 0.0)
                     then Some Float.infinity
                     else Some 0.0
                   else Some Float.nan
                 else Some p
               with _ -> Some Float.nan)
          | _ -> None
        in
        (match r with
         | None -> None
         | Some f -> const_of_target ~target (`Float f))
    | _ -> None
  else
    match const_as_int a, const_as_int b with
    | Some x, Some y ->
        let r = match op with
          | Ops.Add -> Some (x + y)
          | Ops.Sub -> Some (x - y)
          | Ops.Mul -> Some (x * y)
          | Ops.Cdiv ->
              Some (if y = 0 then 0 else x / y)
          | Ops.Cmod ->
              Some (if y = 0 then x else x - (x / y) * y)
          | Ops.Floordiv ->
              Some (floor_div x y)
          | Ops.Floormod ->
              Some (floor_mod x y)
          | Ops.Max -> Some (max x y)
          | Ops.Xor -> Some (x lxor y)
          | Ops.Or -> Some (x lor y)
          | Ops.And -> Some (x land y)
          | Ops.Shl -> Some (x lsl y)
          | Ops.Shr -> Some (x asr y)
          | _ -> None
        in
        (match r with
         | None -> None
         | Some n -> const_of_target ~target (`Int n))
    | _ -> None

let exec_ternary op (target : Dtype.Val.t) a b c =
  match op with
  | Ops.Where ->
      (match Const.view a with
       | Const.Bool true -> Some b
       | Const.Bool false -> Some c
       | Const.Int n -> Some (if n <> 0L then b else c)
       | _ -> None)
  | Ops.Mulacc ->
      if Dtype.Val.is_float target then
        (match const_as_float a, const_as_float b, const_as_float c with
         | Some x, Some y, Some z ->
             const_of_target ~target (`Float ((x *. y) +. z))
         | _ -> None)
      else
        (match const_as_int a, const_as_int b, const_as_int c with
         | Some x, Some y, Some z ->
             const_of_target ~target (`Int ((x * y) + z))
         | _ -> None)
  | _ -> None

let exec_alu op (target : Dtype.Val.t) args =
  let is_binary = Ops.Group.is_binary op in
  if is_binary && any_invalid args
  then Some (Const.invalid ~dtype:target ())
  else
    match args with
    | [ a ] when Ops.Group.is_unary op -> exec_unary op target a
    | [ a; b ] when is_binary -> exec_binary op target a b
    | [ a; b; c ] when Ops.Group.is_ternary op ->
        exec_ternary op target a b c
    | _ -> None

(* phase 1: the most generic folding rules *)

(* [invalid_pat] narrows further via a callback guard because Upat has
   no Const-value pattern for Invalid. *)
let invalid_pat = Upat.op ~name:"i" Ops.Const

let const_of_uop u =
  match Uop.op u, Uop.arg u with
  | Ops.Const, Uop.Arg.Value c -> Some c
  | _ -> None

let is_max_identity u =
  match Uop.dtype u, const_of_uop u with
  | Dtype.Val dtype, Some c ->
      Const.equal c (Const.min_value (Dtype.Val.scalarize dtype))
  | _ -> false

let scalar_const_as_int u =
  match const_of_uop u with
  | Some c -> const_as_int c
  | None -> None

let const_value_of_const c =
  match Const.view c with
  | Const.Bool b -> Uop.Const_scalar (`Bool b)
  | Const.Int n -> Uop.Const_scalar (`Int n)
  | Const.Float f -> Uop.Const_scalar (`Float f)
  | Const.Invalid -> Uop.Const_invalid

let cast_const target c =
  match Const.view c with
  | Const.Bool b -> Some (Const.of_scalar target (`Bool b))
  | Const.Int n -> Some (Const.of_scalar target (`Int n))
  | Const.Float f -> Some (Const.of_scalar target (`Float f))
  | Const.Invalid -> None

let const_node_from_lanes dtype lanes =
  match lanes with
  | [ c ] when Dtype.Val.count dtype = 1 -> Uop.const c
  | _ ->
      Uop.const_of_dtype dtype
        (Uop.Const_tuple (List.map const_value_of_const lanes))

let is_signed_int_scalar = function
  | Dtype.Int8 | Dtype.Int16 | Dtype.Int32 | Dtype.Int64 -> true
  | _ -> false

let raw_mask bytes =
  if bytes >= 8 then -1L
  else Int64.sub (Int64.shift_left 1L (bytes * 8)) 1L

let low_bits bytes n = Int64.logand n (raw_mask bytes)

let sign_extend bytes raw =
  if bytes >= 8 then raw
  else
    let bits = bytes * 8 in
    let sign = Int64.shift_left 1L (bits - 1) in
    if Int64.logand raw sign = 0L then raw
    else Int64.logor raw (Int64.lognot (raw_mask bytes))

let raw_bits_of_const src c =
  let bytes = Dtype.Val.itemsize src in
  match Const.view c with
  | Const.Bool b -> Some (if b then 1L else 0L)
  | Const.Int n when Dtype.Val.is_int src || Dtype.Val.is_bool src ->
      Some (low_bits bytes n)
  | Const.Float f ->
      (match Dtype.Val.scalar src with
       | Dtype.Float32 ->
           Some (low_bits bytes (Int64.of_int32 (Int32.bits_of_float f)))
       | Dtype.Float64 -> Some (Int64.bits_of_float f)
       | _ -> None)
  | Const.Int _ | Const.Invalid -> None

let storage_of_raw_bits dst raw =
  let bytes = Dtype.Val.itemsize dst in
  let raw = low_bits bytes raw in
  if Dtype.Val.is_bool dst then Some (`Bool (raw <> 0L))
  else if Dtype.Val.is_int dst then
    let n =
      if is_signed_int_scalar (Dtype.Val.scalar dst)
      then sign_extend bytes raw
      else raw
    in
    Some (`Int n)
  else
    match Dtype.Val.scalar dst with
    | Dtype.Float32 -> Some (`Float (Int32.float_of_bits (Int64.to_int32 raw)))
    | Dtype.Float64 -> Some (`Float (Int64.float_of_bits raw))
    | _ -> None

let bitcast_const_storage ~src ~dst c =
  let src = Dtype.Val.scalarize src and dst = Dtype.Val.scalarize dst in
  match Dtype.storage_fmt_for_dtype src, Dtype.storage_fmt_for_dtype dst with
  | Some _, Some _ when Dtype.Val.itemsize src = Dtype.Val.itemsize dst ->
      Option.bind (raw_bits_of_const src c) (storage_of_raw_bits dst)
  | _ -> None

let const_lanes count u =
  match Uop.op u, Uop.arg u with
  | Ops.Const, Uop.Arg.Value c ->
      let n = Dtype.Val.count (Const.dtype c) in
      if n = 1 || n = count then Some (List.init count (fun _ -> c))
      else None
  | Ops.Stack, _ ->
      let srcs = Uop.src u in
      if Array.length srcs <> count then None
      else
        let rec loop i acc =
          if i < 0 then Some acc
          else
            match const_of_uop srcs.(i) with
            | Some c -> loop (i - 1) (c :: acc)
            | None -> None
        in
        loop (Array.length srcs - 1) []
  | _ -> None

let fold_const_alu root =
  match Uop.dtype root with
  | Dtype.Ptr _ -> None
  | Dtype.Val dtype ->
      let count = Dtype.Val.count dtype in
      let scalar_dtype = Dtype.Val.scalarize dtype in
      let srcs = Array.to_list (Uop.src root) in
      let lanes = List.map (const_lanes count) srcs in
      if List.exists Option.is_none lanes then None
      else
        let lanes = List.map Option.get lanes in
        let lane i = List.map (fun lane_consts -> List.nth lane_consts i) lanes in
        let rec fold i acc =
          if i = count then Some (List.rev acc)
          else
            match exec_alu (Uop.op root) scalar_dtype (lane i) with
            | Some c -> fold (i + 1) (c :: acc)
            | None -> None
        in
        match fold 0 [] with
        | None -> None
        | Some [ c ] -> Some (Uop.const c)
        | Some cs ->
            Some
              (Uop.const_of_dtype dtype
                 (Uop.Const_tuple (List.map const_value_of_const cs)))

(* Build a numeric const matching [c]'s dtype with value [v]. *)
let const_numeric_like c v =
  match Uop.dtype c with
  | Dtype.Val dtv when Dtype.Val.is_float dtv ->
      Uop.const (Const.of_scalar dtv (`Float v))
  | Dtype.Val dtv when Dtype.Val.is_int dtv ->
      Uop.const (Const.of_scalar dtv (`Int (Int64.of_int (int_of_float v))))
  | _ -> Uop.const_like c (int_of_float v)

(* Read [c]'s numeric value as a float. Returns [None] for non-numeric
   (e.g. Invalid). *)
let const_numeric_v c =
  match const_float_v c with
  | Some f -> Some f
  | None ->
      (match const_int_v c with
       | Some n -> Some (float_of_int n)
       | None -> None)

let const_bound_like u n =
  match Uop.dtype u with
  | Dtype.Val v when Dtype.Val.is_bool v ->
      Uop.broadcast (Uop.const_bool (n <> 0)) (Dtype.Val.count v)
  | _ -> Uop.const_like u n

let simplify_pow x c =
  match const_numeric_v c with
  | None -> None
  | Some e ->
      if e < 0.0 then
        let neg_c = const_numeric_like c (-. e) in
        let r = Uop.alu_unary ~op:Ops.Reciprocal ~src:x in
        Some (Uop.alu_binary ~op:Ops.Pow ~lhs:r ~rhs:neg_c)
      else if e = 0.0 then Some (const_numeric_like x 1.0)
      else if Float.of_int (Float.to_int (e -. 0.5)) +. 0.5 = e then
        (* half-integer: x^e = x^(e-0.5) * sqrt(x) *)
        let c' = const_numeric_like c (e -. 0.5) in
        let half = Uop.alu_binary ~op:Ops.Pow ~lhs:x ~rhs:c' in
        let s = Uop.alu_unary ~op:Ops.Sqrt ~src:x in
        Some (Uop.alu_binary ~op:Ops.Mul ~lhs:half ~rhs:s)
      else if Float.of_int (Float.to_int e) = e then
        (* integer >= 0: repeated squaring *)
        let n = Float.to_int e in
        let c' = const_numeric_like c (Float.of_int (n / 2)) in
        let y = Uop.alu_binary ~op:Ops.Pow ~lhs:x ~rhs:c' in
        let y2 = Uop.alu_binary ~op:Ops.Mul ~lhs:y ~rhs:y in
        if n mod 2 = 1
        then Some (Uop.alu_binary ~op:Ops.Mul ~lhs:y2 ~rhs:x)
        else Some y2
      else None

let div_op_of_mod_op = function
  | Ops.Cmod -> Some Ops.Cdiv
  | Ops.Floormod -> Some Ops.Floordiv
  | _ -> None

(* Decompose an ADD term into (mod_op, div_op, base, div, mul):
   - [base % div]          -> (base, div, 1)
   - [(base % div) * mul]  -> (base, div, mul) *)
let decompose_mod_mul u =
  match Uop.op u, Uop.src u with
  | (Ops.Cmod | Ops.Floormod as mod_op), [| base; d |] ->
      (match const_int_v d with
       | Some dv ->
           Option.map
             (fun div_op -> (mod_op, div_op, base, dv, 1))
             (div_op_of_mod_op mod_op)
       | None -> None)
  | Ops.Mul, [| inner; c |] ->
      (match const_int_v c, Uop.op inner, Uop.src inner with
       | Some mv, (Ops.Cmod | Ops.Floormod as mod_op), [| base; d |] ->
           (match const_int_v d with
            | Some dv ->
                Option.map
                  (fun div_op -> (mod_op, div_op, base, dv, mv))
                  (div_op_of_mod_op mod_op)
            | None -> None)
       | _ -> None)
  | _ -> None

(* [(base % div) * mul + q * (div * mul)] collapses to [base * mul] when
   [q = base // div] (or the equivalent via nested CDIV). A third variant
   collapses [((base // div) % d) * div + base % div] to [base % (div*d)]. *)
let fold_add_divmod_recombine root =
  let terms = Uop.split_uop root Ops.Add in
  let terms_arr = Array.of_list terms in
  let n = Array.length terms_arr in
  let result = ref None in
  let i = ref 0 in
  while Option.is_none !result && !i < n do
    (match decompose_mod_mul terms_arr.(!i) with
     | None -> ()
     | Some (mod_op, div_op, base, div, mul) ->
         let target = div * mul in
         let j = ref 0 in
         while Option.is_none !result && !j < n do
           if !j <> !i then begin
             let v = terms_arr.(!j) in
             (match Uop.op v, Uop.src v with
              | Ops.Mul, [| q; c |] when
                  (match const_int_v c with
                   | Some cv -> cv = target
                   | None -> false) ->
                  let exact =
                    match Uop.op q, Uop.src q with
                    | op, [| qbase; qd |] when Ops.equal op div_op ->
                        (match const_int_v qd with
                         | Some qdv when qdv = div ->
                             Uop.equal qbase base
                         | _ -> false)
                    | _ -> false
                  in
                  let exact_nested =
                    if exact then false
                    else match Uop.op base, Uop.src base with
                    | op, [| bb; bd |] when Ops.equal op div_op ->
                        (match const_int_v bd, Uop.op q, Uop.src q with
                         | Some bdv, qop, [| qbase; qd |]
                           when Ops.equal qop div_op ->
                             (match const_int_v qd with
                              | Some qdv when qdv = bdv * div ->
                                  Uop.equal qbase bb
                              | _ -> false)
                         | _ -> false)
                    | _ -> false
                  in
                  if exact || exact_nested then begin
                    let head =
                      Uop.O.(base * Uop.const_like base mul)
                    in
                    let rest =
                      Array.to_list terms_arr
                      |> List.mapi (fun k t -> k, t)
                      |> List.filter_map (fun (k, t) ->
                           if k = !i || k = !j then None
                           else Some t)
                    in
                    result := Some (Uop.usum (head :: rest))
                  end
                  else if div > 0 then begin
                    match Uop.op q, Uop.src q with
                    | qop, [| qinner; qd |] when Ops.equal qop mod_op ->
                        (match const_int_v qd with
                         | Some d_inner when d_inner > 0 ->
                             (match Uop.op qinner, Uop.src qinner with
                              | qinner_op, [| qb; qb_d |]
                                when Ops.equal qinner_op div_op ->
                                  let ok =
                                    Uop.equal qb base
                                    && (match const_int_v qb_d with
                                        | Some qdv -> qdv = div
                                        | _ -> false)
                                  in
                                  if ok then begin
                                    let new_mod_rhs =
                                      Uop.const_like base (div * d_inner)
                                    in
                                    let new_mod =
                                      Uop.alu_binary ~op:mod_op
                                        ~lhs:base ~rhs:new_mod_rhs
                                    in
                                    let head =
                                      Uop.O.(new_mod * Uop.const_like base mul)
                                    in
                                    let rest =
                                      Array.to_list terms_arr
                                      |> List.mapi (fun k t -> k, t)
                                      |> List.filter_map (fun (k, t) ->
                                           if k = !i || k = !j
                                           then None
                                           else Some t)
                                    in
                                    result := Some
                                      (Uop.usum (head :: rest))
                                  end
                              | _ -> ())
                         | _ -> ())
                    | _ -> ()
                  end
              | _ -> ())
           end;
           incr j
         done);
    incr i
  done;
  !result

let is_const_int_eq u v =
  match const_int_v u with Some n -> n = v | None -> false

let index_stack_const u stk c =
  match const_int_v c with
  | None -> None
  | Some i ->
      let srcs = Uop.src stk in
      if i < 0 || i >= Array.length srcs then None
      else
        let selected = srcs.(i) in
        if Dtype.equal (Uop.dtype u) (Uop.dtype selected)
        then Some selected
        else None

let non_cmp_binary =
  List.filter (fun o -> not (Ops.Group.is_comparison o)) Ops.Group.binary

let invalid_is_index i =
  match Uop.dtype i with
  | Dtype.Val v -> Dtype.Val.scalar v = Dtype.Index
  | Dtype.Ptr _ -> false

let invalid_for_result u =
  Uop.invalid ~dtype:(Dtype.val_of (Uop.dtype u)) ()

let propagate_invalid_comparison ~op ~cond ~valid_lhs ~valid_rhs ~invalid =
  let cmp = Uop.alu_binary ~op ~lhs:valid_lhs ~rhs:valid_rhs in
  if invalid_is_index invalid then Some cmp
  else
    let invalid_bool = Uop.cast ~src:invalid ~dtype:(Uop.dtype cmp) in
    Some (Uop.O.where cond cmp invalid_bool)

(* A LOAD through an INDEX whose scalar offset is [Invalid] folds to [0];
   a STORE through it becomes a NOOP. Two pattern variants cover the
   bare INDEX and [CAST(INDEX)] (when the address is widened). *)

let invalid_index =
  let open Upat in
  index ~name:"idx" invalid_pat any

let invalid_index_or_casted =
  let open Upat in
  [ invalid_index; cast invalid_index ]

let noop_void () = Uop.noop ~dtype:(Dtype.Val Dtype.Val.void) ()

let zero_of_dtype dt =
  match dt with
  | Dtype.Val v when Dtype.Val.is_bool v -> Uop.const (Const.bool false)
  | Dtype.Val v when Dtype.Val.is_int v -> Uop.const (Const.int v 0)
  | Dtype.Val v when Dtype.Val.is_float v -> Uop.const (Const.float v 0.0)
  | _ -> Uop.const (Const.int Dtype.Val.weakint 0)

let make_rule_invalid_load inner =
  let open Upat in
  op ~src:[ inner ] ~name:"ld" ~allow_any_len:true Ops.Load
  => fun bs ->
       if is_invalid_const (bs $ "i")
       then
         let ld = bs $ "ld" in
         let s = Uop.src ld in
         if Array.length s > 1 then Some s.(1)
         else Some (zero_of_dtype (Uop.dtype ld))
       else None

let make_rule_invalid_store_idx inner =
  let open Upat in
  op ~src:[ inner; any ] ~allow_any_len:true Ops.Store
  => fun bs ->
       if is_invalid_const (bs $ "i") then Some (noop_void ()) else None

let pm_invalid_load_store : Upat.Pattern_matcher.t =
  Upat.Pattern_matcher.make
    (List.concat_map
       (fun inner ->
         [ make_rule_invalid_load inner; make_rule_invalid_store_idx inner ])
       invalid_index_or_casted)

(* Index-typed invalid gates are never read past a cast or comparison, so the
   gate drops and the valid value is used directly. Runs before pm_data_invalid
   so the drop wins over the general gate-lifting rules. *)
let pm_index_invalid : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    (let cond = var "cond" and x = var "x" in
     cast ~name:"cast" (where cond x invalid_pat) => fun bs ->
       let i = bs $ "i" in
       if is_invalid_const i && invalid_is_index i
       then Some (Uop.cast ~src:(bs $ "x") ~dtype:(Uop.dtype (bs $ "cast")))
       else None);

    (let cond = var "cond" and x = var "x" and y = var "y" in
     ops ~src:[ where cond x invalid_pat; y ] ~name:"alu" Ops.Group.comparison
     => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         propagate_invalid_comparison ~op:(Uop.op (bs $ "alu"))
           ~cond:(bs $ "cond") ~valid_lhs:(bs $ "x") ~valid_rhs:(bs $ "y")
           ~invalid:i);

    (let cond = var "cond" and x = var "x" and y = var "y" in
     ops ~src:[ y; where cond x invalid_pat ] ~name:"alu" Ops.Group.comparison
     => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         propagate_invalid_comparison ~op:(Uop.op (bs $ "alu"))
           ~cond:(bs $ "cond") ~valid_lhs:(bs $ "y") ~valid_rhs:(bs $ "x")
           ~invalid:i);
  ]

(* Everywhere else Invalid poisons the value: ops move inside the gate so the
   Invalid reaches the LOAD/STORE and folds there. Prepended to symbolic_simple
   so that [0 * something_that_might_be_invalid] does not become [0]. *)
let pm_data_invalid : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    (* Bare Invalid poisons a unary or bitcast result. *)
    (ops ~src:[ invalid_pat ] Ops.Group.unary => fun bs ->
       let i = bs $ "i" in if is_invalid_const i then Some i else None);

    (bitcast ~name:"bc" (op ~name:"i" Ops.Const) => fun bs ->
       let i = bs $ "i" in
       if is_invalid_const i
       then Some (Uop.cast ~src:i ~dtype:(Uop.dtype (bs $ "bc")))
       else None);

    (* Unary(invalid_gate) -> cond.where(op(x), invalid). *)
    (let cond = var "cond" and x = var "x" in
     ops ~src:[ where cond x invalid_pat ] ~name:"alu" Ops.Group.unary
     => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         let alu = bs $ "alu" and cond = bs $ "cond" and x = bs $ "x" in
         Some (Uop.O.where cond (Uop.alu_unary ~op:(Uop.op alu) ~src:x)
                 (invalid_for_result alu)));

    (* BITCAST(invalid_gate) -> cond.where(x.bitcast(d), i.bitcast(d)). *)
    (let cond = var "cond" and x = var "x" in
     bitcast ~name:"bc" (where cond x invalid_pat) => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         let bc = bs $ "bc"
         and cond = bs $ "cond" and x = bs $ "x" in
         let dt = Uop.dtype bc in
         Some (Uop.O.where cond
                 (Uop.bitcast ~src:x ~dtype:dt)
                 (Uop.bitcast ~src:i ~dtype:dt)));

    (* Binary(invalid_gate, y) -> cond.where(op(x, y), invalid) for non-cmps
       (comparisons are handled in pm_index_invalid). *)
    (let cond = var "cond" and x = var "x" and y = var "y" in
     ops ~src:[ where cond x invalid_pat; y ] ~name:"alu" non_cmp_binary
     => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         let alu = bs $ "alu" and cond = bs $ "cond"
         and x = bs $ "x" and y = bs $ "y" in
         Some (Uop.O.where cond
                 (Uop.alu_binary ~op:(Uop.op alu) ~lhs:x ~rhs:y)
                 (invalid_for_result alu)));

    (* Binary(y, invalid_gate) -> cond.where(op(y, x), invalid) for non-cmps. *)
    (let cond = var "cond" and x = var "x" and y = var "y" in
     ops ~src:[ y; where cond x invalid_pat ] ~name:"alu" non_cmp_binary
     => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         let alu = bs $ "alu" and cond = bs $ "cond"
         and x = bs $ "x" and y = bs $ "y" in
         Some (Uop.O.where cond
                 (Uop.alu_binary ~op:(Uop.op alu) ~lhs:y ~rhs:x)
                 (invalid_for_result alu)));

    (* Bare Invalid poisons a non-comparison binary. *)
    (ops ~src:[ invalid_pat; any ] non_cmp_binary => fun bs ->
       let i = bs $ "i" in if is_invalid_const i then Some i else None);

    (ops ~src:[ any; invalid_pat ] non_cmp_binary => fun bs ->
       let i = bs $ "i" in if is_invalid_const i then Some i else None);

    (* An Invalid condition poisons the whole where. *)
    (let a = var "a" in
     where invalid_pat a any => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else Some (Uop.cast ~src:i ~dtype:(Uop.dtype (bs $ "a"))));

    (* A gated-Invalid condition lifts its gate out of the where. *)
    (let cond = var "cond" and x = var "x" and a = var "a" and b = var "b" in
     where (where cond x invalid_pat) a b => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         let cond = bs $ "cond" and x = bs $ "x"
         and a = bs $ "a" and b = bs $ "b" in
         Some
           (Uop.O.where cond (Uop.O.where x a b)
              (Uop.cast ~src:i ~dtype:(Uop.dtype a))));

    (* Normalize where(cond, Invalid, val) -> !cond.where(val, Invalid).
       If val is also Invalid, fold to Invalid. *)
    (where (var "cond") invalid_pat (var "val") => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         let cond = bs $ "cond" and v = bs $ "val" in
         if is_invalid_const v then Some i
         else Some (Uop.O.where (Uop.O.not_ cond) v i));

    (* where(a, where(cond, x, Invalid), c) ->
       lifted.where(a.where(x, c), Invalid) with
       [lifted = cond] if [a = cond], else [!a | cond]. *)
    (let a = var "a" and c = var "c" in
     let cond = var "cond" and x = var "x" in
     where a (where cond x invalid_pat) c => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         let a = bs $ "a" and c = bs $ "c" in
         if is_invalid_const c then None
         else
           let cond = bs $ "cond" and x = bs $ "x" in
           let lifted =
             if Uop.equal a cond then cond
             else Uop.alu_binary ~op:Ops.Or ~lhs:(Uop.O.not_ a) ~rhs:cond
           in
           Some (Uop.O.where lifted (Uop.O.where a x c) i));

    (* where(a, b, where(cond, x, Invalid))
       -> (a | cond).where(a.where(b, x), Invalid). *)
    (let a = var "a" and b = var "b" in
     let cond = var "cond" and x = var "x" in
     where a b (where cond x invalid_pat) => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         let a = bs $ "a" and b = bs $ "b" in
         if is_invalid_const b then None
         else
           let cond = bs $ "cond" and x = bs $ "x" in
           let lifted = Uop.alu_binary ~op:Ops.Or ~lhs:a ~rhs:cond in
           Some (Uop.O.where lifted (Uop.O.where a b x) i));
  ]

let propagate_invalid : Upat.Pattern_matcher.t =
  Upat.Pattern_matcher.(pm_index_invalid ++ pm_data_invalid ++ pm_invalid_load_store)

let fold_mul_zero x =
  match const_float_v x with
  | Some f when not (Float.is_finite f) -> const_nan_like x
  | _ -> Some (Uop.const_like x 0)

let symbolic_simple : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.(propagate_invalid ++ make [
    (* x + 0 -> x *)
    rewrite1 (fun x -> O.(x + zero)) (fun x -> Some x);
    (let x = var "x" and c = cvar ~name:"c" () in
     alu [ x; c ] Ops.Add => fun bs ->
       match const_int_v (bs $ "c"), const_float_v (bs $ "c") with
       | Some 0, _ -> Some (bs $ "x")
       | _, Some f when f = 0.0 -> Some (bs $ "x")
       | _ -> None);
    (let x = var "x"
     and c0 = cvar ~name:"c0" () and c1 = cvar ~name:"c1" () in
     O.((x + c0) + c1) => fun bs ->
       let x = bs $ "x" in
       match const_int_v (bs $ "c0"), const_int_v (bs $ "c1") with
       | Some a, Some b
         when (match Uop.dtype x with
               | Dtype.Val dtype -> not (Dtype.Val.is_unsigned dtype)
               | Dtype.Ptr _ -> false) ->
           let c = Uop.const_like x (a + b) in
           Some Uop.O.(x + c)
       | _ -> None);

    (* x - 0 -> x *)
    rewrite1 (fun x -> O.(x - zero)) (fun x -> Some x);
    (let x = var "x" and c = cvar ~name:"c" () in
     alu [ x; c ] Ops.Sub => fun bs ->
       let x = bs $ "x" and c = bs $ "c" in
       match const_int_v c, const_float_v c with
       | Some 0, _ -> Some (bs $ "x")
       | _, Some f when f = 0.0 -> Some (bs $ "x")
       | Some n, _ -> (
           match Uop.dtype x with
           | Dtype.Val target when not (Dtype.Val.is_unsigned target) ->
               Some Uop.O.(x + Uop.const_like x (-n))
           | _ -> None)
       | _, Some f -> (
           match Uop.dtype x with
           | Dtype.Val target ->
               Option.map
                 (fun c -> Uop.alu_binary ~op:Ops.Add ~lhs:x ~rhs:(Uop.const c))
                 (const_of_target ~target (`Float (-. f)))
           | Dtype.Ptr _ -> None)
       | _ -> None);

    (* x * 1 -> x *)
    rewrite1 (fun x -> O.(x * one)) (fun x -> Some x);
    (let x = var "x" and c = cvar ~name:"c" () in
     alu [ x; c ] Ops.Mul => fun bs ->
       match const_int_v (bs $ "c"), const_float_v (bs $ "c") with
       | Some 1, _ -> Some (bs $ "x")
       | _, Some f when f = 1.0 -> Some (bs $ "x")
       | _ -> None);

    (* x << 0 -> x; x >> 0 -> x *)
    (let x = var "x" and c = cvar ~name:"c" () in
     ops ~src:[ x; c ] [ Ops.Shl; Ops.Shr ] => fun bs ->
       match const_int_v (bs $ "c") with
       | Some 0 -> Some (bs $ "x")
       | _ -> None);

    (* cdiv(x, x) -> 1; x // x -> 1. *)
    rewrite1 (fun x -> O.(cdiv x x)) (fun x -> Some (Uop.const_like x 1));
    (rewrite1 (fun x -> alu [ x; x ] Ops.Floordiv)
       (fun x -> Some (Uop.const_like x 1)));

    (* cdiv(x, 1) -> x; x // 1 -> x. *)
    rewrite1 (fun x -> O.(cdiv x one)) (fun x -> Some x);
    (rewrite1 (fun x -> alu [ x; one ] Ops.Floordiv) (fun x -> Some x));

    (* cdiv(x, -1) -> -x; x // -1 -> -x. *)
    rewrite1 (fun x -> O.(cdiv x neg_one)) (fun x ->
       Some (Uop.O.neg x));
    (rewrite1 (fun x -> alu [ x; neg_one ] Ops.Floordiv)
       (fun x -> Some (Uop.O.neg x)));

    (* cmod(x, x) -> 0; x mod x -> 0. *)
    rewrite1 (fun x -> O.(cmod x x)) (fun x -> Some (Uop.const_like x 0));
    (rewrite1 (fun x -> alu [ x; x ] Ops.Floormod)
       (fun x -> Some (Uop.const_like x 0)));

    (* x < x -> false (or a vector of falses matching x's lane count). *)
    (rewrite1 (fun x -> O.(x < x)) (fun x ->
       let n = Dtype.count (Uop.dtype x) in
       Some (Uop.broadcast (Uop.const_bool false) n)));

    (* x ^ x -> 0 (on ints/bool) *)
    (rewrite1 (fun x -> alu [ x; x ] Ops.Xor) (fun x ->
       if Dtype.is_int (Uop.dtype x) || Dtype.is_bool (Uop.dtype x)
       then Some (Uop.const_like x 0)
       else None));

    (* x & 0 -> 0 *)
    (rewrite1 (fun x -> alu [ x; zero ] Ops.And)
       (fun x -> Some (Uop.const_like x 0)));

    (* x ^ 0 -> x (ints/bool only) *)
    (rewrite1 (fun x -> alu [ x; zero ] Ops.Xor) (fun x ->
       if Dtype.is_int (Uop.dtype x) || Dtype.is_bool (Uop.dtype x)
       then Some x
       else None));

    (* (x ^ y) ^ y -> x *)
    (rewrite2
       (fun x y -> alu [ alu [ x; y ] Ops.Xor; y ] Ops.Xor)
       (fun x _ -> Some x));

    (* (x & mask) >> k  ->  x >> k  when mask only clears bits below k. *)
    (let x = var "x"
     and mask = cvar ~name:"mask" ()
     and k = cvar ~name:"k" () in
     alu [ alu [ x; mask ] Ops.And; k ] Ops.Shr => fun bs ->
       let x = bs $ "x" and mask = bs $ "mask" and k = bs $ "k" in
       match const_int_v mask, const_int_v k with
       | Some mv, Some kv when mv lor ((1 lsl kv) - 1) = -1 ->
           Some (Uop.alu_binary ~op:Ops.Shr ~lhs:x ~rhs:k)
       | _ -> None);

    (* (x & mask) // c  ->  x // c  when c is a power of two and mask clears
       exactly the low bits the division discards. *)
    (let x = var "x" and mask = cvar ~name:"mask" () and c = cvar ~name:"c" () in
     alu [ alu [ x; mask ] Ops.And; c ] Ops.Floordiv => fun bs ->
       let x = bs $ "x" and mask = bs $ "mask" and c = bs $ "c" in
       match const_int_v mask, const_int_v c with
       | Some mv, Some cv
         when cv > 0 && cv land (cv - 1) = 0 && mv lor (cv - 1) = -1 ->
           Some (Uop.alu_binary ~op:Ops.Floordiv ~lhs:x ~rhs:c)
       | _ -> None);

    (* x != x -> False (ints/bool only, vectorised to match lanes). *)
    (rewrite1 (fun x -> alu [ x; x ] Ops.Cmpne) (fun x ->
       if Dtype.is_int (Uop.dtype x) || Dtype.is_bool (Uop.dtype x)
       then
         let n = Dtype.count (Uop.dtype x) in
         Some (Uop.broadcast (Uop.const_bool false) n)
       else None));

    (* Evaluate unary ALU on Consts or STACKs of Consts. *)
    (ops ~name:"root" Ops.Group.unary => fun bs ->
       fold_const_alu (bs $ "root"));

    (* Evaluate binary ALU on Consts or STACKs of Consts (Threefry handled
       separately). *)
    (let binary_non_threefry =
       List.filter (fun o -> o <> Ops.Threefry) Ops.Group.binary
     in
     ops ~name:"root" binary_non_threefry => fun bs ->
       fold_const_alu (bs $ "root"));

    (* Evaluate ternary ALU on Consts or STACKs of Consts. *)
    (ops ~name:"root" [ Ops.Where; Ops.Mulacc ] => fun bs ->
       fold_const_alu (bs $ "root"));

    (* variations of div/mod recombination, for both truncating and floor ops. *)
    (op ~dtype:Dtype.index ~name:"x" Ops.Add
     => fun bs -> fold_add_divmod_recombine (bs $ "x"));

    (* (x:u64 & 0xFFFFFFFF).cast(u32) -> x.cast(u32). *)
    (let x = var_scalar "x" Dtype.Uint64 in
     let mask = cvar ~name:"m" ~dtype:Dtype.uint64 () in
     cast ~dtype:Dtype.uint32 (alu [ x; mask ] Ops.And) => fun bs ->
       match const_int_v (bs $ "m") with
       | Some 0xFFFFFFFF ->
           Some (Uop.cast ~src:(bs $ "x") ~dtype:Dtype.uint32)
       | _ -> None);

    (* ((a:u64 * 2^32) | y:u32.cast(u64)).cast(u32) -> y *)
    (let a = var_scalar "a" Dtype.Uint64
     and y = var_scalar "y" Dtype.Uint32
     and k = cvar ~name:"k" ~dtype:Dtype.uint64 () in
     let spliced =
       alu [ alu [ a; k ] Ops.Mul; cast ~dtype:Dtype.uint64 y ] Ops.Or
     in
     cast ~dtype:Dtype.uint32 spliced => fun bs ->
       if is_const_int_eq (bs $ "k") (1 lsl 32) then Some (bs $ "y") else None);

    (* ((a:u64 << 32) | y:u32.cast(u64)).cast(u32) -> y *)
    (let a = var_scalar "a" Dtype.Uint64
     and y = var_scalar "y" Dtype.Uint32
     and k = cvar ~name:"k" () in
     let spliced =
       alu [ alu [ a; k ] Ops.Shl; cast ~dtype:Dtype.uint64 y ] Ops.Or
     in
     cast ~dtype:Dtype.uint32 spliced => fun bs ->
       if is_const_int_eq (bs $ "k") 32 then Some (bs $ "y") else None);

    (* cdiv(((a:u64 * 2^32) | _:u32.cast(u64)), 2^32) -> a *)
    (let a = var_scalar "a" Dtype.Uint64
     and y = var_scalar "y" Dtype.Uint32
     and k1 = cvar ~name:"k1" ~dtype:Dtype.uint64 ()
     and k2 = cvar ~name:"k2" ~dtype:Dtype.uint64 () in
     let spliced =
       alu [ alu [ a; k1 ] Ops.Mul; cast ~dtype:Dtype.uint64 y ] Ops.Or
     in
     alu [ spliced; k2 ] Ops.Cdiv => fun bs ->
       if is_const_int_eq (bs $ "k1") (1 lsl 32)
          && is_const_int_eq (bs $ "k2") (1 lsl 32)
       then Some (bs $ "a")
       else None);

    (* ((a:u64 << 32) | _:u32.cast(u64)) >> 32 -> a *)
    (let a = var_scalar "a" Dtype.Uint64
     and y = var_scalar "y" Dtype.Uint32
     and k1 = cvar ~name:"k1" ()
     and k2 = cvar ~name:"k2" () in
     let spliced =
       alu [ alu [ a; k1 ] Ops.Shl; cast ~dtype:Dtype.uint64 y ] Ops.Or
     in
     alu [ spliced; k2 ] Ops.Shr => fun bs ->
       if is_const_int_eq (bs $ "k1") 32 && is_const_int_eq (bs $ "k2") 32
       then Some (bs $ "a")
       else None);

    (* (base % y) % y -> base % y *)
    (let y = var "y" in
     let base = op ~src:[ any; y ] ~name:"base" Ops.Cmod in
     O.(cmod base y) => fun bs -> Some (bs $ "base"));
    (let y = var "y" in
     let base = op ~src:[ any; y ] ~name:"base" Ops.Floormod in
     alu [ base; y ] Ops.Floormod => fun bs -> Some (bs $ "base"));

    (* x (bool) and c -> x if c else c *)
    (let x = var_scalar "x" Dtype.Bool in
     let c = cvar ~name:"c" () in
     alu [ x; c ] Ops.And => fun bs ->
       let x = bs $ "x" and c = bs $ "c" in
       match const_bool_v c with
       | Some true -> Some x
       | Some false -> Some c
       | None -> None);

    (* x (bool) or c -> c if c else x *)
    (let x = var_scalar "x" Dtype.Bool in
     let c = cvar ~name:"c" () in
     alu [ x; c ] Ops.Or => fun bs ->
       let x = bs $ "x" and c = bs $ "c" in
       match const_bool_v c with
       | Some true -> Some c
       | Some false -> Some x
       | None -> None);

    (* Idempotent ALUs with two equal operands. *)
    (rewrite1 (fun x -> ops ~src:[ x; x ] Ops.Group.idempotent) (fun x -> Some x));

    (* !!x -> x *)
    (let x = var_scalar "x" Dtype.Bool in
     let inner = op ~src:[ x; false_ ] Ops.Cmpeq in
     op ~src:[ inner; false_ ] Ops.Cmpeq
     => fun bs -> Some (bs $ "x"));

    (* where(cond, true, false) -> cond *)
    (let cond = var_scalar "cond" Dtype.Bool in
     where cond true_ false_ => fun bs -> Some (bs $ "cond"));

    (* where(cond, false, true) -> !cond *)
    (let cond = var_scalar "cond" Dtype.Bool in
     where cond false_ true_ => fun bs -> Some (Uop.O.not_ (bs $ "cond")));

    (* where(x == y, 1, 0) -> where(x != y, 0, 1) *)
    (let x = var "x"
     and y = var "y"
     and one = cvar ~name:"one" ()
     and zero = cvar ~name:"zero" () in
     let cond = alu [ x; y ] Ops.Cmpeq in
     where cond one zero => fun bs ->
       match scalar_const_as_int (bs $ "one"), scalar_const_as_int (bs $ "zero") with
       | Some 1, Some 0 ->
           let x = bs $ "x" and y = bs $ "y" in
           Some
             (Uop.O.where
                (Uop.alu_binary ~op:Ops.Cmpne ~lhs:x ~rhs:y)
                (bs $ "zero") (bs $ "one"))
       | _ -> None);

    (* CAST(bool -> int) != const. Since cast(False)=0 and cast(True)=1,
       compare the uncast bool directly for [0], its negation for [1],
       and [true] for any other scalar integer constant. *)
    (let x = var_scalar "x" Dtype.Bool and c = cvar ~name:"c" () in
     let cast_x = cast ~name:"cast" x in
     alu [ cast_x; c ] Ops.Cmpne => fun bs ->
       let cast_x = bs $ "cast" in
       if not (Dtype.is_int (Uop.dtype cast_x)) then None
       else
         match scalar_const_as_int (bs $ "c") with
         | Some 0 -> Some (bs $ "x")
         | Some 1 -> Some (Uop.O.not_ (bs $ "x"))
         | Some _ -> Some (Uop.const_bool true)
         | None -> None);

    (* where(a, b, b) -> b (noop conditional) *)
    (rewrite1 (fun v -> where any v v) (fun v -> Some v));

    (* where(const gate, c0, c1) -> c0 or c1 based on gate *)
    (let gate = cvar ~name:"gate" () in
     where gate (var "c0") (var "c1") => fun bs ->
       match const_bool_v (bs $ "gate") with
       | Some true -> Some (bs $ "c0")
       | Some false -> Some (bs $ "c1")
       | None -> None);

    (* INDEX(STACK(...), const) selects the indexed stack source when the
       current typed representation already agrees on the result dtype. *)
    (op ~src:[ op ~name:"stk" Ops.Stack; cvar ~name:"c" () ]
       ~name:"idx" Ops.Index
     => fun bs -> index_stack_const (bs $ "idx") (bs $ "stk") (bs $ "c"));

    (* RESHAPE to the same shape is a no-op. *)
    (op ~name:"root" Ops.Reshape => fun bs ->
       let root = bs $ "root" in
       match Uop.src root with
       | [| src; _ |] when same_shape root src -> Some src
       | _ -> None);

    (* trunc on int-typed input -> input *)
    (rewrite1 (fun x -> op ~src:[ x ] Ops.Trunc) (fun x ->
       if Dtype.is_int (Uop.dtype x) || Dtype.is_bool (Uop.dtype x)
       then Some x
       else None));

    (* -(-x) -> x. *)
    (rewrite1
       (fun x -> alu [ alu [ x ] Ops.Neg ] Ops.Neg)
       (fun x -> Some x));

    (* Cast of a constant -> same const with root dtype. *)
    (cast ~name:"root" (cvar ~name:"c" ()) => fun bs ->
       let root = bs $ "root" in
       match Uop.dtype root, const_of_uop (bs $ "c") with
       | Dtype.Val target, Some c ->
           Option.map Uop.const (cast_const target c)
       | _ -> None);

    (* Cast of STACK constants -> lane-wise cast. Tolk represents tinygrad tuple
       vector constants structurally as STACK. *)
    (cast ~name:"root" (op ~name:"stk" Ops.Stack) => fun bs ->
       let root = bs $ "root" and stk = bs $ "stk" in
       match Uop.dtype root with
       | Dtype.Ptr _ -> None
       | Dtype.Val target ->
           let scalar_target = Dtype.Val.scalarize target in
           let srcs = Array.to_list (Uop.src stk) in
           if List.length srcs <> Dtype.Val.count target then None
           else
             let rec loop acc = function
             | [] -> Some (const_node_from_lanes target (List.rev acc))
             | u :: us ->
                 (match const_of_uop u with
                  | Some c ->
                      (match cast_const scalar_target c with
                       | Some c -> loop (c :: acc) us
                       | None -> None)
                  | None -> None)
             in
             loop [] srcs);

    (* Same-dtype cast / bitcast -> input. *)
    (ops ~name:"root" [ Ops.Cast; Ops.Bitcast ] => fun bs ->
       let root = bs $ "root" in
       let s = Uop.src root in
       if Array.length s <> 1 then None
       else if Dtype.equal (Uop.dtype root) (Uop.dtype s.(0))
       then Some s.(0)
       else None);

    (* b.cast(a).cast(b) -> b if a preserves all values in b. *)
    (let x = var "x" in
     cast ~name:"b" (cast ~name:"a" x) => fun bs ->
       let x = bs $ "x" and a = bs $ "a" and b = bs $ "b" in
       match Uop.dtype x, Uop.dtype a, Uop.dtype b with
       | Dtype.Val vx, Dtype.Val va, Dtype.Val vb
         when Dtype.Val.equal vx vb && Dtype.Val.can_lossless_cast vb va ->
           Some x
       | _ -> None);

    (* Bitcast of scalar-broadcast CONST -> reinterpret each lane. *)
    (bitcast ~name:"root" (cvar ~name:"c" ()) => fun bs ->
       let root = bs $ "root" and c = bs $ "c" in
       match Uop.dtype root, Uop.dtype c, const_of_uop c with
       | Dtype.Val target, Dtype.Val source, Some value
         when Dtype.Val.count target = Dtype.Val.count source ->
           Option.map
             (fun storage -> Uop.const (Const.of_scalar target storage))
             (bitcast_const_storage ~src:source ~dst:target value)
       | _ -> None);

    (* Bitcast of STACK constants -> lane-wise bitcast. This covers Tolk's
       structural representation of tinygrad tuple vector constants. *)
    (bitcast ~name:"root" (op ~name:"stk" Ops.Stack) => fun bs ->
       let root = bs $ "root" and stk = bs $ "stk" in
       match Uop.dtype root with
       | Dtype.Val target ->
           let srcs = Array.to_list (Uop.src stk) in
           if List.length srcs <> Dtype.Val.count target then None
           else
             let scalar_target = Dtype.Val.scalarize target in
             let rec loop source acc = function
             | [] -> Some (const_node_from_lanes target (List.rev acc))
             | u :: us -> (
                 match Uop.dtype u, const_of_uop u with
                 | Dtype.Val dtype, Some c ->
                     let scalar_source = Dtype.Val.scalarize dtype in
                     if not (Dtype.Val.equal scalar_source source) then None
                     else
                       (match
                          bitcast_const_storage ~src:source ~dst:scalar_target c
                        with
                        | Some storage ->
                            loop source
                              (Const.of_scalar scalar_target storage :: acc)
                              us
                        | None -> None)
                 | _ -> None)
             in
             (match srcs with
              | [] -> None
              | u :: _ -> (
                  match Uop.dtype u with
                  | Dtype.Val source ->
                      loop (Dtype.Val.scalarize source) [] srcs
                  | _ -> None))
       | _ -> None);

    (* x.cast(bool) -> x != 0 *)
    (rewrite1 (fun x -> cast ~dtype:Dtype.bool x) (fun x ->
       Some (Uop.O.ne x (Uop.const_like x 0))));

    (* ** pow ** *)
    (let x = var "x" and c = cvar ~name:"c" () in
     alu [ x; c ] Ops.Pow => fun bs ->
       simplify_pow (bs $ "x") (bs $ "c"));

    (* Positive constant base: c^x -> c if c = 1, else (x*log2(c)).exp2(). *)
    (let c = cvar ~name:"c" () and x = var "x" in
     alu [ c; x ] Ops.Pow => fun bs ->
       let c = bs $ "c" and x = bs $ "x" in
       match const_numeric_v c with
       | Some f when f = 1.0 -> Some c
       | Some f when f > 0.0 ->
           let log2_c = const_numeric_like x (log f /. log 2.0) in
           let prod = Uop.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:log2_c in
           Some (Uop.alu_unary ~op:Ops.Exp2 ~src:prod)
       | _ -> None);

    (* bool MUL -> AND (so downstream rules don't miscompute bool arithmetic). *)
    (let x = var_scalar "x" Dtype.Bool and y = var_scalar "y" Dtype.Bool in
     O.(x * y) => fun bs ->
       Some (Uop.alu_binary ~op:Ops.And ~lhs:(bs $ "x") ~rhs:(bs $ "y")));

    (* bool ADD -> OR *)
    (let x = var_scalar "x" Dtype.Bool and y = var_scalar "y" Dtype.Bool in
     O.(x + y) => fun bs ->
       Some (Uop.alu_binary ~op:Ops.Or ~lhs:(bs $ "x") ~rhs:(bs $ "y")));

    (* x * 0 -> 0 (or NaN if x is a Const NaN/Inf float). *)
    (rewrite1 (fun x -> O.(x * zero)) fold_mul_zero);
    (rewrite1 (fun x -> O.(zero * x)) fold_mul_zero);

    (* x / x -> 1 (float) *)
    (rewrite1 (fun x -> alu [ x; x ] Ops.Fdiv)
       (fun x -> Some (Uop.const_like x 1)));

    (* (x * x2) / x2 -> x (fdiv, can be wrong if x2 is 0). *)
    (rewrite2
       (fun x x2 -> alu [ alu [ x; x2 ] Ops.Mul; x2 ] Ops.Fdiv)
       (fun x _ -> Some x));

    (* 0 / 0 -> nan (fdiv). *)
    (let z = cvar ~name:"z" () in
     alu [ z; zero ] Ops.Fdiv => fun bs ->
       let z = bs $ "z" in
       match const_int_v z with
       | Some 0 -> const_nan_like z
       | _ -> None);

    (* (x * 0) / 0 -> nan (fdiv). *)
    (rewrite1
       (fun x -> alu [ alu [ x; zero ] Ops.Mul; zero ] Ops.Fdiv)
       (fun x -> const_nan_like x));

    (* a.where(b.where(c, d), d) -> (a & b).where(c, d). *)
    (rewrite4
       (fun a b c d -> where a (where b c d) d)
       (fun a b c d ->
         Some (Uop.O.where (Uop.alu_binary ~op:Ops.And ~lhs:a ~rhs:b) c d)));

    (* bool max(x, y) -> x | y. *)
    (let x = var_scalar "x" Dtype.Bool and y = var_scalar "y" Dtype.Bool in
     alu [ x; y ] Ops.Max => fun bs ->
       Some (Uop.alu_binary ~op:Ops.Or ~lhs:(bs $ "x") ~rhs:(bs $ "y")));
  ])

(* phase 2 *)

(* Two-stage ALU folding on associative ops: x.op(c1).op(c2) -> x.op(c1.op(c2)). *)
let rule_two_stage_associative_for assoc_op =
  let open Upat in
  let x = var "x"
  and c1 = cvar ~name:"c1" () and c2 = cvar ~name:"c2" () in
  alu [ alu [ x; c1 ] assoc_op; c2 ] assoc_op => fun bs ->
    let x = bs $ "x" and c1 = bs $ "c1" and c2 = bs $ "c2" in
    if
      assoc_op = Ops.Add
      &&
      match Uop.dtype x with
      | Dtype.Val dtype -> Dtype.Val.is_unsigned dtype
      | Dtype.Ptr _ -> false
    then None
    else
      let combined = Uop.alu_binary ~op:assoc_op ~lhs:c1 ~rhs:c2 in
      Some (Uop.alu_binary ~op:assoc_op ~lhs:x ~rhs:combined)

let two_stage_associative_rules =
  List.map rule_two_stage_associative_for Ops.Group.associative

(* [x < c]: if [x = sum(np) + sum(p)] where each term in [np] has common
   integer factor [d > 1] dividing [c] and the [p] "residual" sum stays
   within [\[0; d)], then [sum(np)/d < c/d] is equivalent. *)
let lt_folding x c =
  let terms = Uop.split_uop x Ops.Add in
  let p, np = List.partition (fun u -> Uop.const_factor u = 1) terms in
  if np = [] then None
  else
    let factors = List.map Uop.const_factor np in
    let d = List.fold_left gcd_int c factors in
    if d <= 1 then None
    else
      let p_vmin = List.fold_left (fun acc u -> acc + Uop.vmin u) 0 p in
      let p_vmax = List.fold_left (fun acc u -> acc + Uop.vmax u) 0 p in
      if p_vmin < 0 || p_vmax >= d then None
      else
        let np_sum = Uop.usum np in
        match Uop.divides np_sum d with
        | None -> None
        | Some q ->
	        let rhs = Uop.const_like q (c / d) in
	        Some Uop.O.(q < rhs)

(* A simplex [a0*x0 + a1*x1 + ...] with all [ai > 0] and [xi >= 0] can be
   canonicalised to [x0 + x1 + ...] when testing [> 0]. *)
let canonicalize_simplex x =
  let terms = Uop.split_uop x Ops.Add in
  let changed = ref false in
  let exception Reject in
  try
    let ret = List.map (fun u ->
      let u' =
        match Uop.op u, Uop.src u with
        | Ops.Mul, [| inner; c |] ->
            (match const_int_v c with
             | Some cv when cv > 0 ->
                 changed := true; inner
             | _ -> u)
        | _ -> u
      in
      let is_irreducible = List.mem (Uop.op u') Ops.Group.irreducible in
      if not (is_irreducible && Uop.vmin u' >= 0) then raise Reject;
      u') terms in
    if !changed then Some (Uop.usum ret) else None
  with Reject -> None

(* Memoized: an unmemoized walk revisits shared subgraphs and goes
   exponential on wide unrolled ALU chains. *)
let structural_lane_count_cache : int Uop.Ref_tbl.t = Uop.Ref_tbl.create 256

let rec structural_lane_count u =
  match Uop.Ref_tbl.find_opt structural_lane_count_cache u with
  | Some n -> n
  | None ->
      let n = compute_structural_lane_count u in
      Uop.Ref_tbl.add structural_lane_count_cache u n;
      n

and compute_structural_lane_count u =
  let count = Dtype.count (Uop.dtype u) in
  if count > 1 then count
  else
    match Uop.op u with
    | Ops.Stack -> Array.length (Uop.src u)
    | Ops.Index -> 1
    | op when Ops.Group.is_alu op || op = Ops.Cast || op = Ops.Bitcast ->
        Uop.src u
        |> Array.fold_left
             (fun acc child -> max acc (structural_lane_count child))
             count
    | _ -> count

let index_pushing : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    (op ~src:[ op ~name:"src" Ops.Stack; cvar ~name:"idx" () ] Ops.Index
     => fun bs ->
       match Uop.const_int_value (bs $ "idx") with
       | Some i ->
           let src = Uop.src (bs $ "src") in
           if i >= 0 && i < Array.length src then Some src.(i) else None
       | None -> None);
    (op ~name:"stk" Ops.Stack => fun bs ->
       let stk = bs $ "stk" in
       let srcs = Uop.src stk in
       if Array.length srcs = 0 then None
       else
         let first_src =
           match Uop.op srcs.(0), Uop.src srcs.(0) with
           | Ops.Index, idx_src when Array.length idx_src = 2 -> Some idx_src.(0)
           | _ -> None
         in
         match first_src with
         | None -> None
         | Some first_src ->
             try
               for i = 0 to Array.length srcs - 1 do
                 match Uop.op srcs.(i), Uop.src srcs.(i) with
                 | Ops.Index, idx_src when Array.length idx_src = 2 ->
                     if not (Uop.equal idx_src.(0) first_src) then raise Exit;
                     if Uop.const_int_value idx_src.(1) <> Some i then raise Exit
                 | _ -> raise Exit
               done;
               if Uop.shape stk = Uop.shape first_src then Some first_src else None
             with Exit -> None);
  ]

(* Fold a STACK whose sources are INDEX(src, 0), INDEX(src, 1), ... over the
   same [src] back into [src], when the shapes agree. Concrete dims compare
   by value: they are dtype-erased ints in the reference. Runs after memory
   coalescing, where lane re-stacks of freshly vectorized loads appear. *)
let pm_fold_lane_stack : Upat.Pattern_matcher.t =
  let open Upat in
  let dim_equal a b =
    Uop.equal a b
    || (match Uop.const_int_value a, Uop.const_int_value b with
       | Some x, Some y -> x = y
       | _ -> false)
  in
  Pattern_matcher.make [
    (op ~name:"stk" Ops.Stack => fun bs ->
       let stk = bs $ "stk" in
       let srcs = Uop.src stk in
       if Array.length srcs = 0 then None
       else
         let first_src =
           match Uop.op srcs.(0), Uop.src srcs.(0) with
           | Ops.Index, idx_src when Array.length idx_src = 2 ->
               Some idx_src.(0)
           | _ -> None
         in
         match first_src with
         | None -> None
         | Some first_src -> (
             try
               Array.iteri
                 (fun i x ->
                   match Uop.op x, Uop.src x with
                   | Ops.Index, idx_src when Array.length idx_src = 2 ->
                       if not (Uop.equal idx_src.(0) first_src) then
                         raise Exit;
                       if Uop.const_int_value idx_src.(1) <> Some i then
                         raise Exit
                   | _ -> raise Exit)
                 srcs;
               let sa = Uop.shape stk and sb = Uop.shape first_src in
               if List.length sa = List.length sb
                  && List.for_all2 dim_equal sa sb
               then Some first_src
               else None
             with Exit -> None));
  ]

let symbolic : Upat.Pattern_matcher.t =
  let open Upat in
  let rec compare_tuplize a b =
    if Uop.equal a b then 0
    else
      let c = Ops.compare (Uop.op a) (Uop.op b) in
      if c <> 0 then c
      else
        let c = Uop.Arg.compare (Uop.arg a) (Uop.arg b) in
        if c <> 0 then c
        else
          let c = Dtype.compare (Uop.dtype a) (Uop.dtype b) in
          if c <> 0 then c
          else
            let sa = Uop.src a and sb = Uop.src b in
            let rec cmp i =
              if i = Array.length sa || i = Array.length sb then
                Int.compare (Array.length sa) (Array.length sb)
              else
                let c = compare_tuplize sa.(i) sb.(i) in
                if c <> 0 then c else cmp (i + 1)
            in
            cmp 0
  in
  let phase_2_rules = [
    (* x | !x -> True *)
    (let x = var_scalar "x" Dtype.Bool in
     let neg_x = op ~src:[ x; false_ ] Ops.Cmpeq in
     alu [ x; neg_x ] Ops.Or => fun _ -> Some (Uop.const_bool true));

    (* Canonical operand order for index-mode commutative ops. *)
    (ops ~dtype:Dtype.index ~name:"x" Ops.Group.commutative => fun bs ->
       let x = bs $ "x" in
       let s = Uop.src x in
       if Array.length s <> 2 then None
       else if compare_tuplize s.(1) s.(0) < 0
       then Some (Uop.replace x ~src:[| s.(1); s.(0) |] ())
       else None);

    (* (x * c0) + (x * c1) -> x * (c0 + c1). *)
    (let x = var "x"
     and c0 = cvar ~name:"c0" () and c1 = cvar ~name:"c1" () in
     O.((x * c0) + (x * c1)) => fun bs ->
       let x = bs $ "x" and c0 = bs $ "c0" and c1 = bs $ "c1" in
       Some Uop.O.(x * (c0 + c1)));

    (* y + (x * c0) + (x * c1) -> y + x*(c0+c1). *)
    (let x = var "x" and y = var "y"
     and c0 = cvar ~name:"c0" () and c1 = cvar ~name:"c1" () in
     O.((y + x * c0) + (x * c1)) => fun bs ->
       let x = bs $ "x" and y = bs $ "y"
       and c0 = bs $ "c0" and c1 = bs $ "c1" in
       Some Uop.O.(y + (x * (c0 + c1))));

    (* (x + x) -> x * 2. *)
    (rewrite1 (fun x -> O.(x + x))
       (fun x -> Some Uop.O.(x * Uop.const_like x 2)));

    (* y + x + x -> y + x*2 (associative variant). *)
    (rewrite2 (fun x y -> O.((y + x) + x))
       (fun x y -> Some Uop.O.(y + (x * Uop.const_like x 2))));

    (* (x + x * c) -> x * (c + 1). *)
    (let x = var "x" and c = cvar ~name:"c" () in
     O.(x + x * c) => fun bs ->
       let x = bs $ "x" and c = bs $ "c" in
       Some Uop.O.(x * (c + Uop.const_like c 1)));

    (* y + x + x*c -> y + x*(c+1). *)
    (let x = var "x" and y = var "y" and c = cvar ~name:"c" () in
     O.((y + x) + (x * c)) => fun bs ->
       let x = bs $ "x" and y = bs $ "y" and c = bs $ "c" in
       Some Uop.O.(y + (x * (c + Uop.const_like c 1))));

    (* y + x*c + x -> y + x*(c+1). *)
    (let x = var "x" and y = var "y" and c = cvar ~name:"c" () in
     O.((y + (x * c)) + x) => fun bs ->
       let x = bs $ "x" and y = bs $ "y" and c = bs $ "c" in
       Some Uop.O.(y + (x * (c + Uop.const_like c 1))));

    (* y * (x + c) -> (y*x) + (y*c)  (distribution, int only). *)
    (let x = var_scalar "x" Dtype.Index
     and y = cvar ~name:"y" () and c = cvar ~name:"c" () in
     O.(y * (x + c)) => fun bs ->
       let x = bs $ "x" and y = bs $ "y" and c = bs $ "c" in
       Some Uop.O.((y * x) + (y * c)));

    (* (x / x2) / x3 -> x / (x2 * x3)  when x2 and x3 differ. *)
    (let x = var "x" and x2 = var "x2" and x3 = var "x3" in
     O.((x / x2) / x3) => fun bs ->
       let x2 = bs $ "x2" and x3 = bs $ "x3" in
       if Uop.equal x2 x3 then None
       else Some Uop.O.((bs $ "x") / (x2 * x3)));
    (let x = var "x"
     and c1 = cvar ~name:"c1" () and c2 = cvar ~name:"c2" () in
     alu [ alu [ x; c1 ] Ops.Floordiv; c2 ] Ops.Floordiv => fun bs ->
       let x = bs $ "x" and c1 = bs $ "c1" and c2 = bs $ "c2" in
       if Uop.vmin c2 > 0 then
         Some
           (Uop.alu_binary ~op:Ops.Floordiv ~lhs:x
              ~rhs:Uop.O.(c1 * c2))
       else None);

    (* ALU/variable with min==max -> const. *)
    (ops ~name:"x"
       [ Ops.Cmplt; Ops.Cmpne; Ops.Cdiv; Ops.Cmod;
         Ops.Floordiv; Ops.Floormod;
         Ops.Param; Ops.Bind; Ops.Special ]
     => fun bs ->
       let x = bs $ "x" in
       let lo = Uop.vmin x and hi = Uop.vmax x in
       if lo = hi && lo <> min_int && lo <> max_int
       then Some (const_bound_like x lo)
       else None);

    (* RANGE with const vmin == vmax -> const. *)
    (op ~name:"x" Ops.Range => fun bs ->
       let x = bs $ "x" in
       let lo = Uop.vmin x and hi = Uop.vmax x in
       if lo = hi && lo <> min_int && lo <> max_int
       then Some (Uop.const_like x lo)
       else None);

    (* max(x, y) -> x if x.vmin >= y.vmax; -> y if x.vmax <= y.vmin. *)
    (rewrite2 (fun x y -> alu [ x; y ] Ops.Max) (fun x y ->
       if is_max_identity x then Some y
       else if is_max_identity y then Some x
       else if Uop.vmin x >= Uop.vmax y then Some x
       else if Uop.vmax x <= Uop.vmin y then Some y
       else None));
  ]
  (* two-stage associative folding sits between max folding and the lt rules,
     matching tinygrad's rule order. *)
  @ two_stage_associative_rules
  @ [
    (* c0*x < c1 for positive int c0, c1: rewrites to x < ceil(c1/c0). *)
    (let x = var_scalar "x" Dtype.Index
     and c0 = cvar ~name:"c0" ()
     and c1 = cvar ~name:"c1" () in
     O.((c0 * x) < c1) => fun bs ->
       let x = bs $ "x" and c0 = bs $ "c0" and c1 = bs $ "c1" in
       match const_int_v c0, const_int_v c1 with
       | Some c0v, Some c1v when c0v > 0 && c1v > 0 ->
           let bound = (c1v + c0v - 1) / c0v in
           Some Uop.O.(x < Uop.const_like x bound)
       | _ -> None);

    (* c0*x < c1 for negative c0 (not -1), c1 <= 0: rewrites to
       -x < -floor(-c1/-c0). *)
    (let x = var_scalar "x" Dtype.Index
     and c0 = cvar ~name:"c0" ()
     and c1 = cvar ~name:"c1" () in
     O.((c0 * x) < c1) => fun bs ->
       let x = bs $ "x" and c0 = bs $ "c0" and c1 = bs $ "c1" in
       match const_int_v c0, const_int_v c1 with
       | Some c0v, Some c1v when c0v < 0 && c0v <> -1 && c1v <= 0 ->
           (* floor((-c1)/(-c0)) with both positive. *)
           let bound = (-c1v) / (-c0v) in
           Some Uop.O.(Uop.O.neg x < Uop.const_like x (-bound))
       | _ -> None);

    (* (x//d) < c  for d > 0:
       - if c > 0: x < c*d
       - else:     x < c*d - (d-1) *)
    (let x = var_scalar "x" Dtype.Index
     and d = cvar ~name:"d" ()
     and c = cvar ~name:"c" () in
     O.((x // d) < c) => fun bs ->
       let x = bs $ "x" and d = bs $ "d" and c = bs $ "c" in
       match const_int_v d, const_int_v c with
       | Some dv, Some cv when dv > 0 ->
           let bound = if cv > 0 then cv * dv else cv * dv - (dv - 1) in
           Some Uop.O.(x < Uop.const_like x bound)
       | _ -> None);
    (let x = var_scalar "x" Dtype.Index
     and d = cvar ~name:"d" ()
     and c = cvar ~name:"c" () in
     O.(cdiv x d < c) => fun bs ->
       let x = bs $ "x" and d = bs $ "d" and c = bs $ "c" in
       match const_int_v d, const_int_v c with
       | Some dv, Some cv when dv > 0 ->
           let bound = if cv > 0 then cv * dv else cv * dv - (dv - 1) in
           Some Uop.O.(x < Uop.const_like x bound)
       | _ -> None);

    (* Move add/mul consts to the tail: (x + c1) + y -> (x + y) + c1.
       Guard: only fire when [y] is not itself a Const, otherwise this
       ping-pongs with the commutative canonicalisation when both outer
       operands are Consts. *)
    (let x = var "x" and y = var "y" and c1 = cvar ~name:"c1" () in
     O.((x + c1) + y) => fun bs ->
       let y = bs $ "y" in
       if Uop.op y = Ops.Const then None
       else
         let x = bs $ "x" and c1 = bs $ "c1" in
         Some Uop.O.((x + y) + c1));
    (let x = var "x" and y = var "y" and c1 = cvar ~name:"c1" () in
     O.((x * c1) * y) => fun bs ->
       let y = bs $ "y" in
       if Uop.op y = Ops.Const then None
       else
         let x = bs $ "x" and c1 = bs $ "c1" in
         Some Uop.O.((x * y) * c1));

    (* x*(-1) < y*(-1)  ->  y < x. *)
    (let x = var_scalar "x" Dtype.Index and y = var "y" in
     O.(alu [ x; neg_one ] Ops.Mul < alu [ y; neg_one ] Ops.Mul) => fun bs ->
       Some Uop.O.((bs $ "y") < (bs $ "x")));

    (* Generic lt folding: lifts a common factor out of an ADD-split LHS. *)
    (let x = var_scalar "x" Dtype.Index
     and c = cvar ~name:"c" () in
     O.(x < c) => fun bs ->
       match const_int_v (bs $ "c") with
       | Some cv when cv > 0 -> lt_folding (bs $ "x") cv
       | _ -> None);

    (* Canonicalise a simplex with positive coefficients > 0. *)
    (let x = var_scalar "x" Dtype.Index in
     op ~src:[ O.(x < one); true_ ] Ops.Cmpne => fun bs ->
       match canonicalize_simplex (bs $ "x") with
       | None -> None
       | Some newx ->
           Some (Uop.O.ne
                   Uop.O.(newx < Uop.const_like newx 1)
                   (Uop.const_bool true)));

    (* Binary(where(c, t, f), where(c, tt, ff)) -> where(c, op(t,tt), op(f,ff))
       when at least one branch is const on both sides. *)
    (let c = var "c" in
     let t = var "t" and f = var "f" in
     let tt = var "tt" and ff = var "ff" in
     ops ~src:[ where c t f; where c tt ff ] ~name:"alu" Ops.Group.binary
     => fun bs ->
       let alu = bs $ "alu"
       and c = bs $ "c" and t = bs $ "t" and tt = bs $ "tt"
       and f = bs $ "f" and ff = bs $ "ff" in
       let t_const = Uop.op t = Ops.Const && Uop.op tt = Ops.Const in
       let f_const = Uop.op f = Ops.Const && Uop.op ff = Ops.Const in
       if not (t_const || f_const) then None
       else
         let lhs = Uop.alu_binary ~op:(Uop.op alu) ~lhs:t ~rhs:tt in
         let rhs = Uop.alu_binary ~op:(Uop.op alu) ~lhs:f ~rhs:ff in
         Some (Uop.O.where c lhs rhs));

    (* (y + where(c, t, f)) + where(c, tt, ff) collapses when t&tt or
       f&ff are consts: -> y + where(c, t+tt, f+ff). *)
    (let c = var "c" and y = var "y" in
     let t = var "t" and f = var "f" in
     let tt = var "tt" and ff = var "ff" in
     O.((y + where c t f) + where c tt ff) => fun bs ->
       let c = bs $ "c" and y = bs $ "y"
       and t = bs $ "t" and tt = bs $ "tt"
       and f = bs $ "f" and ff = bs $ "ff" in
       let t_const = Uop.op t = Ops.Const && Uop.op tt = Ops.Const in
       let f_const = Uop.op f = Ops.Const && Uop.op ff = Ops.Const in
       if not (t_const || f_const) then None
       else
         let merged = Uop.O.where c Uop.O.(t + tt) Uop.O.(f + ff) in
         Some Uop.O.(y + merged));

    (* Binary op on two int64 sources narrows to int32 math when no operand
       or result overflows int32, then casts the result back to int64. *)
    (let x = var_scalar "x" Dtype.Int64 and y = var_scalar "y" Dtype.Int64 in
     ops ~src:[ x; y ] ~name:"u" Ops.Group.binary => fun bs ->
       let u = bs $ "u" and x = bs $ "x" and y = bs $ "y" in
       let i32 = Dtype.Val.int32 in
       if overflows u i32 || overflows x i32 || overflows y i32 then None
       else
         let xc = Uop.cast ~src:x ~dtype:Dtype.int32 in
         let yc = Uop.cast ~src:y ~dtype:Dtype.int32 in
         let narrowed = Uop.alu_binary ~op:(Uop.op u) ~lhs:xc ~rhs:yc in
         Some (Uop.cast ~src:narrowed ~dtype:(Uop.dtype u)));

    (* Narrowing cast chain: [x.cast(a).cast(b)] where [x], [a], [b] are
       ints, [a]'s range covers [x.vmin..x.vmax]. Collapse to
       [x.cast(b)]. *)
    (let x = var_scalar "x" Dtype.Index in
     cast ~name:"b" (cast ~name:"a" x) => fun bs ->
       let x = bs $ "x" and a = bs $ "a" and b = bs $ "b" in
       if not (Dtype.is_int (Uop.dtype x) && Dtype.is_int (Uop.dtype a))
       then None
       else
         match Uop.dtype a with
         | Dtype.Val av ->
             (match int_bounds av with
              | Some (lo, hi) when lo <= Uop.vmin x && Uop.vmax x <= hi ->
                  Some (Uop.cast ~src:x ~dtype:(Uop.dtype b))
              | _ -> None)
         | _ -> None);

    (* -1 * (x + c) -> -x + -c. *)
    (let x = var "x" and c = cvar ~name:"c" () in
     O.(neg_one * (x + c)) => fun bs ->
       let x = bs $ "x" and c = bs $ "c" in
       Some Uop.O.(neg x + neg c));

    (* cond.not.where(t, f) -> cond.where(f, t) when f is not Invalid. *)
    (let cond = var_scalar "cond" Dtype.Bool in
     let inner_not = op ~src:[ cond; false_ ] Ops.Cmpeq in
     where inner_not (var "t") (var "f") => fun bs ->
       let c = bs $ "cond" and t = bs $ "t" and f = bs $ "f" in
       if is_invalid_const f then None
       else Some (Uop.O.where c f t));
    (let cond = var_scalar "cond" Dtype.Bool in
     let inner_not = op ~src:[ cond; true_ ] Ops.Cmpne in
     where inner_not (var "t") (var "f") => fun bs ->
       let c = bs $ "cond" and t = bs $ "t" and f = bs $ "f" in
       if is_invalid_const f then None
       else Some (Uop.O.where c f t));

    (* (c0 + x) < c1 -> x < (c1 - c0). *)
    (let x = var "x"
     and c0 = cvar ~name:"c0" () and c1 = cvar ~name:"c1" () in
     O.((c0 + x) < c1) => fun bs ->
       let x = bs $ "x" and c0 = bs $ "c0" and c1 = bs $ "c1" in
       Some Uop.O.(x < (c1 - c0)));

    (* A range mod its own upper bound is just the range. *)
    (let end_p = var "end" in
     let r = op ~name:"r" ~src:[ end_p ] Ops.Range in
     O.(r mod end_p) => fun bs -> Some (bs $ "r"));
    (let end_p = var "end" in
     let r = op ~name:"r" ~src:[ end_p ] Ops.Range in
     O.(cmod r end_p) => fun bs -> Some (bs $ "r"));
	    (let end_p = var "end" in
	     let r = op ~name:"r" ~src:[ end_p ] Ops.Range in
	     alu [ r; end_p ] Ops.Floormod => fun bs -> Some (bs $ "r"));

	    (* A range divided by its own upper bound is 0. *)
	    (let end_p = var "end" in
	     let r = op ~name:"r" ~src:[ end_p ] Ops.Range in
	     O.(r // end_p) => fun bs -> Some (Uop.const_like (bs $ "r") 0));
    (let end_p = var "end" in
     let r = op ~name:"r" ~src:[ end_p ] Ops.Range in
     O.(cdiv r end_p) => fun bs -> Some (Uop.const_like (bs $ "r") 0));
	    (let end_p = var "end" in
	     let r = op ~name:"r" ~src:[ end_p ] Ops.Range in
	     alu [ r; end_p ] Ops.Floordiv => fun bs ->
	       Some (Uop.const_like (bs $ "r") 0));

	    (* AFTER: replace non-side-effecting deps with their transitive deps. *)
    (op ~name:"after" Ops.After => fun bs ->
       let after = bs $ "after" in
       let s = Uop.src after in
       if Array.length s < 2 then None
       else
         let side_effectful y = match Uop.op y with
           | Ops.Range | Ops.Store | Ops.Call | Ops.Function
           | Ops.Barrier | Ops.End | Ops.Linear
           | Ops.Stage -> true
           | _ -> false
         in
         let changed = ref false in
         let seen = Uop.Ref_tbl.create (Array.length s) in
         let add_dedup dst y =
           if Uop.Ref_tbl.mem seen y then ()
           else (Uop.Ref_tbl.add seen y (); dst := y :: !dst)
         in
         let new_deps = ref [] in
         for i = 1 to Array.length s - 1 do
           let d = s.(i) in
           if side_effectful d then add_dedup new_deps d
           else begin
             changed := true;
             Array.iter (add_dedup new_deps) (Uop.src d)
           end
         done;
         if not !changed then None
         else
           let new_src = Array.of_list (s.(0) :: List.rev !new_deps) in
           Some (Uop.replace after ~src:new_src ()));

    (* AFTER with a single src is just the src. *)
    (op ~name:"after" Ops.After => fun bs ->
       let s = Uop.src (bs $ "after") in
       if Array.length s = 1 then Some s.(0) else None);

    (* STACK(const, const, ...) is already the current vector constant form. *)
    (op ~name:"vec" Ops.Stack => fun bs ->
       let vec = bs $ "vec" in
       let s = Uop.src vec in
       if Array.length s = 0 then None
       else
         let exception Not_all_const in
         try
           Array.iter (fun u ->
             match Uop.op u, Uop.arg u with
             | Ops.Const, Uop.Arg.Value _ -> ()
             | _ -> raise Not_all_const) s;
           None
         with Not_all_const -> None);

    (* (x + c).cast(int) -> x.cast + c.cast. *)
    (let x = var_scalar "x" Dtype.Index and c = cvar ~name:"c" () in
     cast ~name:"cast" (alu [ x; c ] Ops.Add) => fun bs ->
       let cast = bs $ "cast"
       and x = bs $ "x" and c = bs $ "c" in
       if not (Dtype.is_int (Uop.dtype cast)) then None
       else
         let dt = Uop.dtype cast in
         Some (Uop.alu_binary ~op:Ops.Add
                 ~lhs:(Uop.cast ~src:x ~dtype:dt)
                 ~rhs:(Uop.cast ~src:c ~dtype:dt)));

    (* cast/long folding: intermediate cast that doesn't narrow can be
       dropped. *)
    (let x = var "x" in
     cast ~name:"b" (cast ~name:"a" x) => fun bs ->
       let x = bs $ "x" and a = bs $ "a" and b = bs $ "b" in
       match Uop.dtype x, Uop.dtype a with
       | Dtype.Val vx, Dtype.Val va when Dtype.Val.can_lossless_cast vx va ->
           Some (Uop.cast ~src:x ~dtype:(Uop.dtype b))
       | _ -> None);
  ] in
  let base =
    Pattern_matcher.(symbolic_simple ++ make phase_2_rules)
  in
  Pattern_matcher.(base ++ Divandmod.div_and_mod_symbolic)

(* phase 3 (symbolic 2.0) *)

(* A gated LOAD whose gate folded to a constant collapses to its taken branch. *)
let rule_const_gated_load =
  let open Upat in
  op ~name:"ld" Ops.Load
  => fun bs ->
       let ld = bs $ "ld" in
       match Uop.as_load ld with
       | Some { src; alt = Some alt; gate = Some gate } -> (
           match const_bool_v gate with
           | Some true -> Some (Uop.load ~src ())
           | Some false -> Some alt
           | None -> None)
       | _ -> None

let rules_invalid_load_store = [ rule_const_gated_load ]

(* {1 simplify_valid}

   Validity predicates are conjunctions of comparisons like [X < c] or
   [!(X < c)] (i.e. [X >= c]). [parse_valid] recognises one such clause
   and returns the bounded subject. [simplify_valid] dedups AND-split
   clauses and substitutes each bound expression with a tighter fresh
   variable, simplifies, then substitutes back: if the substitution
   round-trips to the same canonical form under each candidate, the
   simplification holds. *)

let parse_valid v =
  match Uop.op v, Uop.src v with
  | Ops.Cmpne, [| inner; rhs |]
    when (match const_bool_v rhs with Some true -> true | _ -> false) ->
      (match Uop.op inner, Uop.src inner with
       | Ops.Cmplt, [| lhs; rhs2 |] when Dtype.is_int (Uop.dtype lhs) ->
           Some (lhs, false, Uop.vmin rhs2)
       | _ -> None)
  | Ops.Cmplt, [| lhs; rhs |] when Dtype.is_int (Uop.dtype lhs) ->
      Some (lhs, true, Uop.vmax rhs - 1)
  | _ -> None

let fake_var ~index ~lo ~hi ~(dtype : Dtype.Val.t) () =
  let name = Printf.sprintf "fake%d" index in
  Uop.variable ~name ~min_val:lo ~max_val:hi ~dtype ()

(* [uop_given_valid ~try_simplex valid u] rewrites [u] under the
   assumption that every AND-clause of [valid] holds. For each bounded
   expression [X] in the valid, try two candidates:

   1. Substitute [X] with a fresh variable of its bounded range.
   2. If [X = X0 + X1 + ...] of irreducibles with [lo = 1], substitute
      each [Xi] with a fresh variable of range [\[1; Xi.vmax\]] and accept
      only if every branch simplifies to the same term.

   For [Ops.Stack] with two sources, independently accept a branch
   that collapses one lane. Finally, substitute every [X] in the bounds
   map with its whole-clause fake and simplify. *)
let uop_given_valid ?(try_simplex = true) valid u =
  let clauses = Uop.split_uop valid Ops.And in
  let bounds = Uop.Tbl.create 8 in
  let order = ref [] in
  List.iter (fun c ->
    match parse_valid c with
    | None -> ()
    | Some (expr, is_upper, bound) ->
        let cur =
          match Uop.Tbl.find_opt bounds expr with
          | Some pair -> pair
          | None ->
              let pair = (ref None, ref None) in
              order := expr :: !order;
              Uop.Tbl.add bounds expr pair;
              pair
        in
        let lo_r, hi_r = cur in
        if is_upper then hi_r := Some bound
        else lo_r := Some bound)
    clauses;
  let order = List.rev !order in
  let all_candidates = ref [] in
  let uop_ref = ref u in
  let all_same_uop xs =
    match xs with
    | [] -> false
    | first :: rest -> List.for_all (Uop.equal first) rest
  in
  let try_candidate candidate =
    let u_cur = !uop_ref in
    let subs_list = List.map (fun (x, newx) -> (x, newx)) candidate in
    let newuops =
      List.map (fun (x, newx) ->
        Uop.substitute [ (x, newx) ] u_cur
        |> fun r -> (x, newx, r)) subs_list
    in
    let any_unchanged =
      List.exists (fun (_, _, r) -> Uop.equal r u_cur) newuops
    in
    if any_unchanged then ()
    else
      let finals = List.map (fun (x, newx, r) ->
        let simp = Uop.simplify r in
        let unsubbed = Uop.substitute [ (newx, x) ] simp in
        Uop.simplify unsubbed) newuops
      in
      if all_same_uop finals then uop_ref := List.hd finals
      else
        match Uop.op u_cur, Uop.src u_cur with
        | Ops.Stack, [| s0; s1 |] ->
            let fst_srcs = List.map (fun f ->
              match Uop.src f with
              | [| a; _ |] -> Some a
              | _ -> None) finals
            in
            let snd_srcs = List.map (fun f ->
              match Uop.src f with
              | [| _; b |] -> Some b
              | _ -> None) finals
            in
            let unwrap = List.filter_map (fun x -> x) in
            let fst_srcs = unwrap fst_srcs in
            let snd_srcs = unwrap snd_srcs in
            (if List.length fst_srcs = List.length finals
                && all_same_uop fst_srcs then
              uop_ref :=
                Uop.replace u_cur
                  ~src:[| List.hd fst_srcs; s1 |] ());
            (if List.length snd_srcs = List.length finals
                && all_same_uop snd_srcs then
              uop_ref :=
                Uop.replace u_cur
                  ~src:[| s0; List.hd snd_srcs |] ())
        | _ -> ()
  in
  List.iteri (fun i expr ->
    let lo_r, hi_r = Uop.Tbl.find bounds expr in
    let default_lo = Uop.vmin expr in
    let default_hi = Uop.vmax expr in
    let lo = Option.value !lo_r ~default:default_lo in
    let hi = Option.value !hi_r ~default:default_hi in
    if lo = min_int || hi = max_int then ()
    else
      let dt = match Uop.dtype expr with
        | Dtype.Val v -> v
        | _ -> Dtype.Val.weakint
      in
      let fake = fake_var ~index:i ~lo ~hi ~dtype:dt () in
      all_candidates := (expr, fake) :: !all_candidates;
      if try_simplex then begin
        try_candidate [ (expr, fake) ];
        let is_simplex =
          Uop.op expr = Ops.Add && lo = 1
          && List.for_all
               (fun u -> List.mem (Uop.op u) Ops.Group.irreducible)
               (Uop.split_uop expr Ops.Add)
        in
        if is_simplex then
          let simplex_cands = List.map (fun xi ->
            let xi_dt = match Uop.dtype xi with
              | Dtype.Val v -> v
              | _ -> Dtype.Val.weakint
            in
            let hi_xi = Uop.vmax xi in
            (xi, fake_var ~index:i ~lo:1 ~hi:hi_xi ~dtype:xi_dt ())
          ) (Uop.split_uop expr Ops.Add) in
          try_candidate simplex_cands
      end)
    order;
  let final_subs = List.rev !all_candidates in
  if final_subs = [] then !uop_ref
  else
    let s_uop = Uop.substitute final_subs !uop_ref in
    if Uop.equal s_uop !uop_ref then !uop_ref
    else
      let reverse = List.map (fun (a, b) -> (b, a)) final_subs in
      Uop.simplify (Uop.substitute reverse (Uop.simplify s_uop))

(* [_valid_priority v valids] is the sort key used by {!simplify_valid}.
   A clause is ordered earlier when its subject appears in the backward
   slice of other clauses, so that simplifying later clauses can use the
   tighter bound established by earlier ones. *)
let _valid_priority v valids =
  match parse_valid v with
  | None -> 0
  | Some (subject, _, _) ->
      List.fold_left (fun acc other ->
        if List.memq subject (Uop.toposort other) then acc - 1
        else acc) 0 valids

(* [simplify_valid] deduplicates AND clauses in [valid], orders them by
   {!_valid_priority}, then applies constraint propagation pairwise:
   each clause is simplified under the conjunction of those already
   accepted. Only runs for non-indexing valids. *)
let simplify_valid valid =
  (* Guard: this simplification is for pure validity predicates. Skip
     when the valid's backward slice contains [Ops.Index] — those are
     indexing expressions, and [uop_given_valid] is not sound there. *)
  let contains_index =
    List.exists (fun u -> Uop.op u = Ops.Index) (Uop.toposort valid)
  in
  if contains_index then None
  else
    let valids = Uop.split_uop valid Ops.And in
    let sorted =
      List.stable_sort
        (fun a b -> compare (_valid_priority a valids) (_valid_priority b valids))
        valids
    in
    let seen = Uop.Tbl.create 8 in
    let deduped = List.filter (fun c ->
      if Uop.Tbl.mem seen c then false
      else (Uop.Tbl.add seen c (); true)) sorted in
    let rec loop acc = function
      | [] -> List.rev acc
      | c :: rest ->
          let c' =
            match acc with
            | [] -> c
            | _ -> uop_given_valid (Uop.uprod (List.rev acc)) c
          in
          loop (c' :: acc) rest
    in
    let result_terms = loop [] deduped in
    (* [None] iff processing was a no-op: no dedup and no simplification. *)
    let same_as_sorted =
      List.length result_terms = List.length sorted
      && List.for_all2 Uop.equal result_terms sorted
    in
    if same_as_sorted then None
    else Some (Uop.uprod result_terms)

let pm_simplify_valid =
  let open Upat in
  Pattern_matcher.make [
    (op ~name:"valid" Ops.And => fun bs -> simplify_valid (bs $ "valid"));

    (let cond = var "cond" and x = var "x" in
     where cond x invalid_pat => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         let cond = bs $ "cond" and x = bs $ "x" in
         match Uop.dtype x with
         | Dtype.Val v when Dtype.Val.scalar v = Dtype.Index ->
             let x' = uop_given_valid cond x in
             if Uop.equal x x' then None
             else Some (Uop.O.where cond x' i)
         | _ -> None);

  ]

let pm_drop_and_clauses =
  let open Upat in
  Pattern_matcher.make [
    (let cond = var "cond" and x = var "x" in
     where cond x invalid_pat => fun bs ->
       let i = bs $ "i" in
       if not (is_invalid_const i) then None
       else
         let cond = bs $ "cond" and x = bs $ "x" in
         let x_ranges = Uop.ranges x in
         let clauses = Uop.split_uop cond Ops.And in
         let keep, drop =
           List.partition
             (fun c ->
               List.exists (fun r -> List.memq r x_ranges) (Uop.ranges c))
             clauses
         in
         if drop = [] then None
         else
           let new_cond = match keep with
             | [] -> Uop.const_bool true
             | xs -> Uop.uprod xs
           in
           Some (Uop.O.where new_cond x i));
  ]

let sym : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.(symbolic ++ pm_simplify_valid
                  ++ make rules_invalid_load_store
                  ++ make [
    (* ALU(STACK(x), STACK(y)) -> STACK(ALU(x,y), ..., ALU(x,y)). *)
    (let x = var "x" and y = var "y" in
     ops ~name:"alu"
       ~src:[ op ~src:[ x ] Ops.Stack; op ~src:[ y ] Ops.Stack ]
       Ops.Group.alu
     => fun bs ->
       let alu = bs $ "alu" and x = bs $ "x" and y = bs $ "y" in
       let alu_dt = Uop.dtype alu in
       match alu_dt with
       | Dtype.Val dtv ->
           let scalar_dt = Dtype.Val (Dtype.Val.scalarize dtv) in
           let count = Dtype.count alu_dt in
           let inner = Uop.alu_binary ~op:(Uop.op alu) ~lhs:x ~rhs:y in
           let inner = Uop.cast ~src:inner ~dtype:scalar_dt in
           Some
             (Uop.stack ~dtype:(Dtype.Val.scalarize dtv)
                (List.init count (fun _ -> inner)))
       | _ -> None);

    (* where(s, a, b).cast(d) -> where(s, a.cast(d), b.cast(d)). *)
    (let s = var "s" and a = var "a" and b = var "b" in
     cast ~name:"cast" (where s a b) => fun bs ->
       let cast = bs $ "cast"
       and s = bs $ "s" and a = bs $ "a" and b = bs $ "b" in
       let dt = Uop.dtype cast in
       Some (Uop.O.where s
               (Uop.cast ~src:a ~dtype:dt) (Uop.cast ~src:b ~dtype:dt)));

    (* store(index, load(index)) -> Noop  (self-store elimination). *)
    (let i = op ~name:"index" Ops.Index in
     store i (load i) => fun _ ->
       Some (Uop.noop ~dtype:(Dtype.Val Dtype.Val.void) ()));

    (* store(index, gate.where(alt, load(index))) -> gated store of alt. *)
    (let index = op ~name:"index" Ops.Index in
     let gate = var "gate" and alt = var "alt" in
     store index (where gate alt (load index)) => fun bs ->
       let index = bs $ "index" and gate = bs $ "gate" and alt = bs $ "alt" in
       let index_src = Uop.src index in
       if Array.length index_src < 2 then None
       else
        let buf = index_src.(0) in
        let idxs = Array.to_list index_src |> List.tl in
        let idxs = List.map (fun idx -> Uop.valid ~src:idx ~cond:gate) idxs in
        let dst =
          Uop.index ~ptr:buf ~idxs ~as_ptr:(Dtype.is_ptr (Uop.dtype index)) ()
        in
        Some (Uop.store ~dst ~value:alt ()));

    (* Store of Invalid -> Noop. *)
    (store ~name:"st" any invalid_pat => fun bs ->
       if is_invalid_const (bs $ "i")
       then Some (Uop.noop ~dtype:(Dtype.Val Dtype.Val.void) ())
       else None);

    (* store(buf.index(idx), cond.where(val, Invalid), ...ranges)
       -> store(buf.index(cond.where(idx, Invalid)), val, ...ranges). *)
    (let idx_node = op ~name:"index" Ops.Index in
     let cond = var "cond" and value = var "val" in
     let wh = where cond value invalid_pat in
     op ~src:[ idx_node; wh ] ~name:"store" ~allow_any_len:true Ops.Store
     => fun bs ->
          let i = bs $ "i" in
          if not (is_invalid_const i) then None
          else
            let index = bs $ "index" in
            let cond = bs $ "cond" and value = bs $ "val" in
            let store = bs $ "store" in
            let index_src = Uop.src index in
            if Array.length index_src < 2 then None
            else
              let buf = index_src.(0) in
              let idxs = Array.to_list index_src |> List.tl in
              let idxs =
                List.map (fun idx -> Uop.valid ~src:idx ~cond) idxs
              in
              let new_index =
                Uop.index ~ptr:buf ~idxs
                  ~as_ptr:(Dtype.is_ptr (Uop.dtype index)) ()
              in
              let gate =
                let src = Uop.src store in
                if Array.length src = 3 then Some src.(2) else None
              in
              Some (Uop.store ~dst:new_index ~value ?gate ()));

    (* A constant multiplier on the reduced expression can float past an
       [Ops.Add] reduce: [reduce(x * c, ranges) = reduce(x, ranges) * c]. *)
    (let x = var "x" and c = cvar ~name:"c" () in
     let mul = alu [ x; c ] Ops.Mul in
     op ~src:[ mul ] ~name:"r" ~allow_any_len:true Ops.Reduce
     => fun bs ->
          let r = bs $ "r" and x = bs $ "x" and c = bs $ "c" in
          match Uop.Arg.as_reduce_arg (Uop.arg r) with
          | Some { op = Ops.Add; _ } ->
              let rsrc = Uop.src r in
              let new_src = Array.copy rsrc in
              new_src.(0) <- x;
              let new_r = Uop.replace r ~src:new_src () in
              Some (Uop.alu_binary ~op:Ops.Mul ~lhs:new_r ~rhs:c)
          | _ -> None);

    (* [reduce(x0 * x1 * ... , ranges)] with [arg] in [{Add, Max}] moves
       every MUL-term that neither uses a reduced range nor is itself a
       reduced range outside the reduce. Non-negative terms are required
       for MAX. *)
    (let mul_body = op Ops.Mul in
     op ~src:[ mul_body ] ~name:"r" ~allow_any_len:true Ops.Reduce
     => fun bs ->
          let r = bs $ "r" in
          match Uop.Arg.as_reduce_arg (Uop.arg r) with
          | Some { op; axes = [] } when op = Ops.Add || op = Ops.Max ->
              if not (Dtype.equal (Uop.dtype r) (Uop.dtype (Uop.src r).(0)))
              then None
              else
                let rsrc = Uop.src r in
                let body = rsrc.(0) in
                let ranges =
                  Array.sub rsrc 1 (Array.length rsrc - 1) |> Array.to_list
                in
                let terms = Uop.split_uop body Ops.Mul in
                let inside = ref [] and outside = ref [] in
                List.iter (fun m ->
                  let m_refs_range =
                    List.exists (fun rn -> Uop.in_backward_slice rn m) ranges
                  in
                  let m_is_range = List.memq m ranges in
                  let vmin_ok = op <> Ops.Max || Uop.vmin m >= 0 in
                  if (not m_refs_range) && (not m_is_range) && vmin_ok
                  then outside := m :: !outside
                  else inside := m :: !inside)
                  terms;
                let outside = List.rev !outside in
                let inside = List.rev !inside in
                if outside = [] then None
                else
                  let new_body = match inside with
                    | [] -> Uop.const_like body 1
                    | xs -> Uop.uprod xs
                  in
                  let new_src = Array.copy rsrc in
                  new_src.(0) <- new_body;
                  let new_r = Uop.replace r ~src:new_src () in
                  let out_prod = Uop.uprod outside in
                  Some (Uop.alu_binary ~op:Ops.Mul ~lhs:new_r ~rhs:out_prod)
          | _ -> None);

    (* (x * x).reciprocal -> x.reciprocal * x.reciprocal. *)
    (rewrite1
       (fun x -> op ~src:[ alu [ x; x ] Ops.Mul ] Ops.Reciprocal)
       (fun x ->
         let r = Uop.alu_unary ~op:Ops.Reciprocal ~src:x in
         Some (Uop.alu_binary ~op:Ops.Mul ~lhs:r ~rhs:r)));

    (* (x * x * x).reciprocal -> (1/x)*(1/x)*(1/x). *)
    (rewrite1
       (fun x ->
         op ~src:[ alu [ alu [ x; x ] Ops.Mul; x ] Ops.Mul ] Ops.Reciprocal)
       (fun x ->
         let r = Uop.alu_unary ~op:Ops.Reciprocal ~src:x in
         Some (Uop.alu_binary ~op:Ops.Mul
                 ~lhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:r ~rhs:r) ~rhs:r)));

    (* (x * c).reciprocal -> (1/x) * (1/c). *)
    (let x = var "x" and c = cvar ~name:"c" () in
     op ~src:[ alu [ x; c ] Ops.Mul ] Ops.Reciprocal => fun bs ->
       let x = bs $ "x" and c = bs $ "c" in
       let rx = Uop.alu_unary ~op:Ops.Reciprocal ~src:x in
       let rc = Uop.alu_unary ~op:Ops.Reciprocal ~src:c in
       Some (Uop.alu_binary ~op:Ops.Mul ~lhs:rx ~rhs:rc));

    (* x * (1/(1+x)) -> 1 - 1/(1+x). *)
    (let x = var "x" in
     let d = op ~name:"d" ~src:[ O.(x + one) ] Ops.Reciprocal in
     O.(x * d) => fun bs ->
       let d = bs $ "d" in
       Some Uop.O.(Uop.const_like d 1 - d));

    (* x * (1/(1+x) * y) -> y * (1 - 1/(1+x)). *)
    (let x = var "x" and y = var "y" in
     let d = op ~name:"d" ~src:[ O.(x + one) ] Ops.Reciprocal in
     O.(x * (d * y)) => fun bs ->
       let y = bs $ "y" and d = bs $ "d" in
       Some Uop.O.(y * (Uop.const_like d 1 - d)));

    (* x * (1/(1+x) + y) -> (1 - 1/(1+x)) + x*y. *)
    (let x = var "x" and y = var "y" in
     let d = op ~name:"d" ~src:[ O.(x + one) ] Ops.Reciprocal in
     O.(x * (d + y)) => fun bs ->
       let x = bs $ "x" and y = bs $ "y" and d = bs $ "d" in
       Some Uop.O.((Uop.const_like d 1 - d) + (x * y)));

    (* GROUP with a single source -> the source (peephole cleanup). *)
    (rewrite1 (fun x -> op ~src:[ x ] Ops.Group) (fun x -> Some x));

    (* SINK/GROUP flattening: when a child is SINK/GROUP/NOOP/STACK/UNROLL,
       expand its srcs inline. *)
    (ops ~name:"root" [ Ops.Sink; Ops.Group ] => fun bs ->
       let root = bs $ "root" in
       let remove_like = function
         | Ops.Noop | Ops.Stack | Ops.Sink | Ops.Group -> true
         | _ -> false
       in
       let s = Uop.src root in
       if not (Array.exists (fun u -> remove_like (Uop.op u)) s) then None
       else
         let flat =
           Array.fold_right
             (fun u acc ->
               if remove_like (Uop.op u)
               then Array.to_list (Uop.src u) @ acc
               else u :: acc)
             s []
         in
         Some (Uop.replace root ~src:(Array.of_list flat) ()));

    (* -1 * (x + y) -> -x + -y (general, on all numeric types). *)
    (rewrite2 (fun x y -> O.(neg_one * (x + y)))
       (fun x y -> Some Uop.O.(neg x + neg y)));

    (* (x + y) * c  ->  x*c + y*c  (int only; floats hit NaN issues). *)
    (let x = var_scalar "x" Dtype.Index
     and y = var "y" and c = cvar ~name:"c" () in
     O.((x + y) * c) => fun bs ->
       let x = bs $ "x" and y = bs $ "y" and c = bs $ "c" in
       Some Uop.O.((x * c) + (y * c)));
  ])

(* top-level simplifier *)

(* Run [symbolic] to fixed point, then install as [Uop.simplify_ref].
   This mirrors tinygrad, where [UOp.simplify] runs the phase-2 [symbolic]
   matcher (which itself carries [div_and_mod_symbolic]), not the heavier
   phase-3 [sym]: [sym]'s [pm_simplify_valid] re-enters [Uop.simplify], so
   using it here would make simplification mutually recursive. *)
let simplify u =
  let rec loop u =
    let u' = Uop.graph_rewrite (fun n ->
      Upat.Pattern_matcher.rewrite symbolic n) u in
    if Uop.equal u u' then u else loop u'
  in
  loop u

let () = Uop.simplify_ref := simplify
