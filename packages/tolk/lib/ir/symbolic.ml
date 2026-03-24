(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Symbolic simplification rules for Kernel IR.

   Three phases:
   - symbolic_simple (phase 1): generic folding
   - symbolic (phase 2): deeper algebraic rules + divandmod
   - sym (phase 3): full symbolic + GEP/vectorize + decompositions *)

module K = Kernel

(* Helpers *)

let is_const_int n node =
  match K.view node with
  | Const { value; _ } ->
      (match Const.view value with Int v -> v = Int64.of_int n | _ -> false)
  | _ -> false

let is_const_float f node =
  match K.view node with
  | Const { value; _ } ->
      (match Const.view value with Float v -> v = f | _ -> false)
  | _ -> false

let const_int_val node =
  match K.view node with
  | Const { value; _ } ->
      (match Const.view value with Int v -> Some v | _ -> None)
  | _ -> None

let is_const_bool b node =
  match K.view node with
  | Const { value; _ } ->
      (match Const.view value with Bool v -> v = b | _ -> false)
  | _ -> false

let is_index_dtype dt =
  Dtype.is_int dt && Dtype.equal (Dtype.scalar_of dt) Dtype.index

(* GEP pushing *)

let gep_vectorize node =
  match K.view node with
  | Gep { src; idxs; _ } ->
      (match K.view src with
      | Vectorize { srcs; _ } ->
          let n = List.length srcs in
          if List.for_all (fun i -> i >= 0 && i < n) idxs then
            match idxs with
            | [idx] -> Some (List.nth srcs idx)
            | _ ->
                let extracted = List.map (fun i -> List.nth srcs i) idxs in
                Some (K.vectorize ~srcs:extracted)
          else None
      | _ -> None)
  | _ -> None

let gep_const node =
  match K.view node with
  | Gep { src; idxs; _ } ->
      (match K.view src with
      | Const { value; _ } ->
          (match idxs with
          | [_] -> Some (K.const value)
          | _ ->
              let c = K.const value in
              Some (K.vectorize ~srcs:(List.init (List.length idxs) (fun _ -> c))))
      | _ -> None)
  | _ -> None

let gep_vconst node =
  match K.view node with
  | Gep { src; idxs; _ } ->
      (match K.view src with
      | Vconst { values; dtype } ->
          let n = List.length values in
          if List.for_all (fun i -> i >= 0 && i < n) idxs then
            match idxs with
            | [idx] -> Some (K.const (List.nth values idx))
            | _ ->
                Some
                  (K.vconst
                     ~values:(List.map (List.nth values) idxs)
                     ~dtype:(Dtype.vec (Dtype.scalar_of dtype) (List.length idxs)))
          else None
      | _ -> None)
  | _ -> None

let gep_void node =
  match K.view node with
  | Gep { src; dtype; _ } when Dtype.equal dtype Dtype.void -> Some src
  | _ -> None

let cat_to_vectorize node =
  match K.view node with
  | Cat { srcs; _ } ->
      let expanded =
        List.concat_map
          (fun s ->
            let count = Dtype.count (K.dtype_or Dtype.void s) in
            List.init count (fun i -> K.gep ~src:s ~idx:i))
          srcs
      in
      Some (K.vectorize ~srcs:expanded)
  | _ -> None

let vectorize_in_order_gep node =
  match K.view node with
  | Vectorize { srcs = (first :: _ as srcs); _ } ->
      (match K.view first with
      | Gep { src = base; idxs = [0]; _ } ->
          let count = List.length srcs in
          if count = Dtype.count (K.dtype_or Dtype.void base) then
            let rec check i = function
              | [] -> true
              | s :: rest ->
                  (match K.view s with
                  | Gep { src; idxs = [idx]; _ } ->
                      src == base && idx = i && check (i + 1) rest
                  | _ -> false)
            in
            if check 0 srcs then Some base else None
          else None
      | _ -> None)
  | _ -> None

let gep_through_alu node =
  match K.view node with
  | Gep { src; idxs = [idx]; dtype } when Dtype.equal (Dtype.scalar_of dtype) Dtype.index
    ->
      (match K.view src with
      | Binary { op; lhs; rhs; dtype = alu_dt } when Dtype.is_int alu_dt ->
          Some
            (K.binary ~op
               ~lhs:(K.gep ~src:lhs ~idx)
               ~rhs:(K.gep ~src:rhs ~idx))
      | Unary { op; src = inner; dtype = alu_dt } when Dtype.is_int alu_dt ->
          Some (K.unary ~op ~src:(K.gep ~src:inner ~idx))
      | Cast { src = inner; _ } ->
          Some (K.cast ~src:(K.gep ~src:inner ~idx) ~dtype:(Dtype.to_any dtype))
      | _ -> None)
  | _ -> None

(* GEP with identity permutation on the full vector → remove the GEP.
   Skips ptr-typed sources (Ptrcat, Cast with ptr dtype) — those need
   to stay so gep_after_load can update the Load dtype. *)
let gep_identity node =
  match K.view node with
  | Gep { src; idxs; _ } when not (K.is_ptr src) ->
      let src_count = match K.dtype src with
        | Some dt -> Dtype.count dt | None -> -1
      in
      if src_count > 0 && idxs = List.init src_count Fun.id then Some src
      else None
  | _ -> None

let gep_pushing =
  K.first_match
    [ gep_vectorize; gep_const; gep_vconst; gep_void; gep_identity;
      cat_to_vectorize; vectorize_in_order_gep; gep_through_alu ]

(* ALU/Vectorize reordering *)

let alu_through_broadcast_vectorize node =
  let is_broadcast srcs =
    match srcs with
    | x :: _ -> List.for_all (fun s -> s == x) srcs
    | [] -> false
  in
  match K.view node with
  | Binary { op; lhs; rhs; _ } ->
      (match K.view lhs, K.view rhs with
      | Vectorize { srcs = lsrcs; dtype = ldt },
        Vectorize { srcs = rsrcs; dtype = rdt }
        when Dtype.count (Dtype.any_to_val ldt) = Dtype.count (Dtype.any_to_val rdt)
             && Dtype.count (Dtype.any_to_val ldt) > 0
             && is_broadcast lsrcs && is_broadcast rsrcs ->
          let scalar = K.binary ~op ~lhs:(List.hd lsrcs) ~rhs:(List.hd rsrcs) in
          Some (K.vectorize ~srcs:(List.init (Dtype.count (Dtype.any_to_val ldt)) (fun _ -> scalar)))
      | _ -> None)
  | Unary { op; src; _ } ->
      (match K.view src with
      | Vectorize { srcs; dtype } when Dtype.count (Dtype.any_to_val dtype) > 0 && is_broadcast srcs ->
          let scalar = K.unary ~op ~src:(List.hd srcs) in
          Some (K.vectorize ~srcs:(List.init (Dtype.count (Dtype.any_to_val dtype)) (fun _ -> scalar)))
      | _ -> None)
  | _ -> None

(* Constant folding *)

let exec_unary op v =
  match op with
  | `Neg -> -.v
  | `Exp2 -> Float.pow 2.0 v
  | `Log2 -> Float.log v /. Float.log 2.0
  | `Sin -> Float.sin v
  | `Sqrt -> Float.sqrt v
  | `Recip -> 1.0 /. v
  | `Trunc -> if v >= 0.0 then Float.floor v else Float.ceil v

let exec_binary_float op l r =
  match op with
  | `Add -> Some (l +. r)
  | `Sub -> Some (l -. r)
  | `Mul -> Some (l *. r)
  | `Fdiv -> Some (l /. r)
  | `Max -> Some (Float.max l r)
  | `Pow -> Some (Float.pow l r)
  | _ -> None

let exec_binary_int op (l : int64) (r : int64) =
  match op with
  | `Add -> Some (Int64.add l r)
  | `Sub -> Some (Int64.sub l r)
  | `Mul -> Some (Int64.mul l r)
  | `Idiv -> if r = 0L then None else Some (Int64.div l r)
  | `Mod -> if r = 0L then None else Some (Int64.rem l r)
  | `Max -> Some (if Int64.compare l r >= 0 then l else r)
  | `Shl -> Some (Int64.shift_left l (Int64.to_int r))
  | `Shr -> Some (Int64.shift_right l (Int64.to_int r))
  | `And -> Some (Int64.logand l r)
  | `Or -> Some (Int64.logor l r)
  | `Xor -> Some (Int64.logxor l r)
  | `Cmplt -> Some (if Int64.compare l r < 0 then 1L else 0L)
  | `Cmpeq -> Some (if l = r then 1L else 0L)
  | `Cmpne -> Some (if l <> r then 1L else 0L)
  | _ -> None

let const_fold node =
  match K.view node with
  | Unary { op; src; _ } ->
      (match K.view src with
      | Const { value; dtype } ->
          (match Const.view value with
          | Float f -> Some (K.const (Const.float dtype (exec_unary op f)))
          | _ -> None)
      | _ -> None)
  | Binary { op; lhs; rhs; _ } ->
      (match K.view lhs, K.view rhs with
      | Const { value = lv; dtype = ld }, Const { value = rv; _ } ->
          (match Const.view lv, Const.view rv with
          | Float lf, Float rf ->
              (match op with
              | `Cmplt -> Some (K.const (Const.bool (lf < rf)))
              | `Cmpeq -> Some (K.const (Const.bool (lf = rf)))
              | `Cmpne -> Some (K.const (Const.bool (lf <> rf)))
              | _ ->
                  Option.map (fun r -> K.const (Const.float ld r))
                    (exec_binary_float op lf rf))
          | Int li, Int ri ->
              (match op with
              | `Cmplt -> Some (K.const (Const.bool (Int64.compare li ri < 0)))
              | `Cmpeq -> Some (K.const (Const.bool (li = ri)))
              | `Cmpne -> Some (K.const (Const.bool (li <> ri)))
              | _ ->
                  Option.map (fun r -> K.const (Const.int64 ld r))
                    (exec_binary_int op li ri))
          | Bool lb, Bool rb ->
              (match op with
              | `And -> Some (K.const (Const.bool (lb && rb)))
              | `Or -> Some (K.const (Const.bool (lb || rb)))
              | `Xor | `Cmpne -> Some (K.const (Const.bool (lb <> rb)))
              | `Cmpeq -> Some (K.const (Const.bool (lb = rb)))
              | _ -> None)
          | _ -> None)
      | _ -> None)
  | _ -> None

(* Phase 1: self-folding and identity rules (symbolic_simple) *)

let self_fold node =
  match K.view node with
  | Binary { op = `Idiv; lhs; rhs; dtype } when lhs == rhs ->
      Some (K.const (Const.int64 dtype 1L))
  | Binary { op = `Idiv; lhs; rhs; _ } when is_const_int (-1) rhs ->
      Some (K.unary ~op:`Neg ~src:lhs)
  | Binary { op = `Mod; lhs; rhs; dtype } when lhs == rhs ->
      Some (K.const (Const.int64 dtype 0L))
  | Binary { op = `Mod; lhs; rhs; _ } ->
      (match K.view lhs with
      | Binary { op = `Mod; rhs = inner_rhs; _ } when inner_rhs == rhs ->
          Some lhs
      | _ -> None)
  | Binary { op = `Cmplt; lhs; rhs; _ } when lhs == rhs ->
      Some (K.const_bool false)
  | Binary { op = `Xor; lhs; rhs; dtype } when lhs == rhs ->
      Some (K.const (Const.int64 dtype 0L))
  | Binary { op = `Cmpne; lhs; rhs; dtype }
    when lhs == rhs && Dtype.is_int dtype ->
      Some (K.const_bool false)
  | _ -> None

let identity_fold node =
  match K.view node with
  | Binary { op = `Add; lhs; rhs; _ } ->
      if is_const_int 0 rhs || is_const_float 0.0 rhs then Some lhs
      else if is_const_int 0 lhs || is_const_float 0.0 lhs then Some rhs
      else None
  | Binary { op = `Sub; lhs; rhs; _ } ->
      if is_const_int 0 rhs || is_const_float 0.0 rhs then Some lhs else None
  | Binary { op = `Mul; lhs; rhs; dtype } ->
      if is_const_int 1 rhs || is_const_float 1.0 rhs then Some lhs
      else if is_const_int 1 lhs || is_const_float 1.0 lhs then Some rhs
      else if is_const_int 0 rhs || is_const_int 0 lhs then
        Some (K.const (Const.int64 dtype 0L))
      else None
  | Binary { op = (`Idiv | `Fdiv); lhs; rhs; _ } ->
      if is_const_int 1 rhs || is_const_float 1.0 rhs then Some lhs else None
  | Binary { op = `Or; lhs; rhs; _ } ->
      if is_const_int 0 rhs then Some lhs
      else if is_const_int 0 lhs then Some rhs
      else None
  | Binary { op = `And; lhs; rhs; _ } ->
      if is_const_int 0 rhs then Some rhs
      else if is_const_int 0 lhs then Some lhs
      else None
  | Binary { op = `Xor; lhs; rhs; _ } ->
      if is_const_int 0 rhs then Some lhs
      else if is_const_int 0 lhs then Some rhs
      else None
  | Cast { src; dtype } ->
      if Dtype.to_any (K.dtype_or Dtype.void src) = dtype then Some src else None
  | _ -> None

let float_div_fold node =
  match K.view node with
  | Binary { op = `Fdiv; lhs; rhs; _ } when lhs == rhs ->
      Some (K.const (Const.float (K.dtype_or Dtype.float32 node) 1.0))
  | Binary { op = `Fdiv; lhs; rhs; _ } ->
      (match K.view lhs with
      | Binary { op = `Mul; lhs = x; rhs = y; _ } when y == rhs -> Some x
      | Binary { op = `Mul; lhs = y; rhs = x; _ } when y == rhs -> Some x
      | _ -> None)
  | _ -> None

let where_fold node =
  match K.view node with
  | Ternary { op = `Where; b; c; _ } when b == c -> Some b
  | Ternary { op = `Where; a; b; c; _ } ->
      (match K.view a with
      | Const { value; _ } ->
          (match Const.view value with
          | Bool true -> Some b
          | Bool false -> Some c
          | Int v -> if v <> 0L then Some b else Some c
          | _ -> None)
      | _ -> None)
  | _ -> None

let idempotent_fold node =
  match K.view node with
  | Binary { op = (`And | `Or | `Max); lhs; rhs; _ } when lhs == rhs ->
      Some lhs
  | _ -> None

let trunc_int node =
  match K.view node with
  | Unary { op = `Trunc; src; _ } when Dtype.is_int (K.dtype_or Dtype.void src)
    -> Some src
  | _ -> None

let bool_arith_fold node =
  let dt = K.dtype_or Dtype.void node in
  if not (Dtype.is_bool dt) then None
  else
    match K.view node with
    | Binary { op = `Mul; lhs; rhs; _ } ->
        Some (K.binary ~op:`And ~lhs ~rhs)
    | Binary { op = (`Add | `Max); lhs; rhs; _ } ->
        Some (K.binary ~op:`Or ~lhs ~rhs)
    | Binary { op = `And; lhs = x; rhs; _ } ->
        if is_const_bool true rhs then Some x
        else if is_const_bool false rhs then Some rhs
        else None
    | Binary { op = `Or; lhs = x; rhs; _ } ->
        if is_const_bool true rhs then Some rhs
        else if is_const_bool false rhs then Some x
        else None
    | _ -> None

let double_not_fold node =
  match K.view node with
  | Binary { op = `Cmpne; lhs; rhs; _ } when is_const_bool true rhs ->
      (match K.view lhs with
      | Binary { op = `Cmpne; lhs = z; rhs = inner_rhs; _ }
        when is_const_bool true inner_rhs -> Some z
      | _ -> None)
  | _ -> None

let bool_where_identity node =
  match K.view node with
  | Ternary { op = `Where; a; b; c; _ }
    when Dtype.is_bool (K.dtype_or Dtype.void a) ->
      (match K.view b, K.view c with
      | Const { value = bv; _ }, Const { value = cv; _ } ->
          (match Const.view bv, Const.view cv with
          | Bool true, Bool false -> Some a
          | Bool false, Bool true ->
              Some (K.binary ~op:`Cmpne ~lhs:a ~rhs:(K.const_bool true))
          | _ -> None)
      | _ -> None)
  | _ -> None

let const_cast_fold node =
  match K.view node with
  | Cast { src; dtype = Dtype.T dtype } ->
      (match K.view src with
      | Const { value; _ } ->
          (match Const.view value with
          | Int v ->
              if Dtype.is_int dtype then Some (K.const (Const.int64 dtype v))
              else if Dtype.is_float dtype then
                Some (K.const (Const.float dtype (Int64.to_float v)))
              else if Dtype.is_bool dtype then
                Some (K.const (Const.bool (v <> 0L)))
              else None
          | Float f ->
              if Dtype.is_float dtype then Some (K.const (Const.float dtype f))
              else if Dtype.is_int dtype then
                Some (K.const (Const.int64 dtype (Int64.of_float f)))
              else if Dtype.is_bool dtype then
                Some (K.const (Const.bool (f <> 0.0)))
              else None
          | Bool b ->
              if Dtype.is_int dtype then
                Some (K.const (Const.int64 dtype (if b then 1L else 0L)))
              else if Dtype.is_float dtype then
                Some (K.const (Const.float dtype (if b then 1.0 else 0.0)))
              else None)
      | _ -> None)
  | _ -> None

let double_cast_fold node =
  match K.view node with
  | Cast { src; dtype = b_dt } ->
      (match K.view src with
      | Cast { src = x; dtype = a_dt } ->
          let x_dt = K.dtype_or Dtype.void x in
          let b = Dtype.any_to_val b_dt and a = Dtype.any_to_val a_dt in
          if Dtype.equal x_dt b && Dtype.can_lossless_cast b a then
            Some x
          else None
      | _ -> None)
  | _ -> None

let cast_to_bool node =
  match K.view node with
  | Cast { src; dtype = Dtype.T dt } when Dtype.is_bool dt ->
      Some (K.binary ~op:`Cmpne ~lhs:src ~rhs:(K.zero_like src))
  | _ -> None

let nested_where_fold node =
  match K.view node with
  | Ternary { op = `Where; a; b = inner; c = d_outer; _ } ->
      (match K.view inner with
      | Ternary { op = `Where; a = b_cond; b = c_val; c = d_inner; _ }
        when d_inner == d_outer ->
          Some
            (K.ternary ~op:`Where
               ~a:(K.binary ~op:`And ~lhs:a ~rhs:b_cond)
               ~b:c_val ~c:d_outer)
      | _ -> None)
  | _ -> None

let divmod_reconstitute node =
  match K.view node with
  | Binary { op = `Add; dtype; _ }
    when is_index_dtype dtype ->
      let terms = Divandmod.split_add node in
      List.find_mapi
        (fun i u ->
          let base_div_mul =
            match K.view u with
            | Binary { op = `Mod; lhs = base; rhs = d_node; _ } ->
                Option.map (fun d -> (base, d, 1L)) (const_int_val d_node)
            | Binary { op = `Mul; lhs = mod_node; rhs = m_node; _ } ->
                (match K.view mod_node with
                | Binary { op = `Mod; lhs = base; rhs = d_node; _ } ->
                    (match const_int_val d_node, const_int_val m_node with
                    | Some d, Some m -> Some (base, d, m)
                    | _ -> None)
                | _ -> None)
            | _ -> None
          in
          match base_div_mul with
          | None -> None
          | Some (base, div, mul) ->
              List.find_mapi
                (fun j v ->
                  if i = j then None
                  else
                    match K.view v with
                    | Binary { op = `Mul; lhs = q; rhs = dm_node; _ } ->
                        (match const_int_val dm_node with
                        | Some dm when dm = Int64.mul div mul ->
                            (match K.view q with
                            | Binary { op = `Idiv; lhs = q_base; rhs = q_div; _ }
                              ->
                                (match const_int_val q_div with
                                | Some qd when qd = div && q_base == base ->
                                    let remaining =
                                      List.filteri
                                        (fun k _ -> k <> i && k <> j)
                                        terms
                                    in
                                    let base_mul =
                                      if mul = 1L then base
                                      else
                                        K.binary ~op:`Mul ~lhs:base
                                          ~rhs:(K.const (Const.int64 Dtype.index mul))
                                    in
                                    Some
                                      (List.fold_left
                                        (fun acc t -> K.binary ~op:`Add ~lhs:acc ~rhs:t)
                                        base_mul remaining)
                                | _ -> None)
                            | _ -> None)
                        | _ -> None)
                    | _ -> None)
                terms)
        terms
  | _ -> None

(* Phase 2: deeper algebraic rules *)

let is_commutative = function
  | `Add | `Mul | `And | `Or | `Xor -> true
  | _ -> false

let commutative_flip node =
  match K.view node with
  | Binary { op; lhs; rhs; dtype }
    when is_index_dtype dtype
         && is_commutative op
         && K.compare_structure rhs lhs < 0 ->
      Some (K.binary ~op ~lhs:rhs ~rhs:lhs)
  | _ -> None

let combine_terms node =
  match K.view node with
  | Binary { op = `Add; lhs; rhs; dtype } when lhs == rhs && Dtype.is_int dtype ->
      Some (K.binary ~op:`Mul ~lhs ~rhs:(K.const (Const.int64 dtype 2L)))
  | Binary { op = `Add; lhs; rhs; _ } ->
      let extract_mul_const n =
        match K.view n with
        | Binary { op = `Mul; lhs = x; rhs = c; _ } ->
            Option.map (fun cv -> (x, cv)) (const_int_val c)
        | _ -> None
      in
      let dt = K.dtype_or Dtype.index node in
      (match extract_mul_const lhs, extract_mul_const rhs with
      | Some (x1, c1), Some (x2, c2) when x1 == x2 ->
          Some (K.binary ~op:`Mul ~lhs:x1
                  ~rhs:(K.const (Const.int64 dt (Int64.add c1 c2))))
      | None, Some (x2, c2) when lhs == x2 ->
          Some (K.binary ~op:`Mul ~lhs
                  ~rhs:(K.const (Const.int64 dt (Int64.add c2 1L))))
      | Some (x1, c1), None when x1 == rhs ->
          Some (K.binary ~op:`Mul ~lhs:x1
                  ~rhs:(K.const (Const.int64 dt (Int64.add c1 1L))))
      | _ -> None)
  | _ -> None

let associative_fold node =
  match K.view node with
  | Binary { op; lhs; rhs = c2; _ } when is_commutative op ->
      (match K.view lhs, const_int_val c2 with
      | Binary { op = inner_op; lhs = x; rhs = c1; _ }, Some _
        when inner_op = op ->
          (match const_int_val c1 with
          | Some _ ->
              Some (K.binary ~op ~lhs:x ~rhs:(K.binary ~op ~lhs:c1 ~rhs:c2))
          | None -> None)
      | _ -> None)
  | _ -> None

let const_to_end node =
  let try_op op lhs rhs =
    if not (K.is_const rhs) then
      match K.view lhs with
      | Binary { op = inner_op; lhs = x; rhs = c1; _ }
        when inner_op = op && K.is_const c1 ->
          Some
            (K.binary ~op
               ~lhs:(K.binary ~op ~lhs:x ~rhs)
               ~rhs:c1)
      | _ -> None
    else None
  in
  match K.view node with
  | Binary { op = (`Add as op); lhs; rhs; _ } -> try_op op lhs rhs
  | Binary { op = (`Mul as op); lhs; rhs; _ } -> try_op op lhs rhs
  | _ -> None

let nested_div_fold node =
  match K.view node with
  | Binary { op = `Idiv; lhs; rhs = c2; _ } ->
      (match K.view lhs with
      | Binary { op = `Idiv; lhs = x; rhs = c1; _ } ->
          Some (K.binary ~op:`Idiv ~lhs:x
                  ~rhs:(K.binary ~op:`Mul ~lhs:c1 ~rhs:c2))
      | _ -> None)
  | _ -> None

let range_self_divmod node =
  match K.view node with
  | Binary { op = `Mod; lhs; rhs; _ } ->
      (match K.view lhs with
      | Range { size; _ } when size == rhs -> Some lhs
      | _ -> None)
  | Binary { op = `Idiv; lhs; rhs; dtype } ->
      (match K.view lhs with
      | Range { size; _ } when size == rhs ->
          Some (K.const (Const.int64 dtype 0L))
      | _ -> None)
  | _ -> None

let max_fold node =
  match K.view node with
  | Binary { op = `Max; lhs; rhs; _ } ->
      if Divandmod.vmin lhs >= Divandmod.vmax rhs then Some lhs
      else if Divandmod.vmax lhs <= Divandmod.vmin rhs then Some rhs
      else None
  | _ -> None

let range_collapse node =
  let dtype = K.dtype_or Dtype.void node in
  if not (is_index_dtype dtype) then None
  else
    match K.view node with
    | Binary _ | Unary _ | Ternary _ | Define_var _ | Range _ ->
        let lo = Divandmod.vmin node and hi = Divandmod.vmax node in
        if lo = hi && lo <> Int64.min_int && hi <> Int64.max_int then
          Some (K.const (Const.int64 dtype lo))
        else None
    | _ -> None

let lt_const_fold node =
  match K.view node with
  | Binary { op = `Cmplt; lhs; rhs; _ } ->
      (* c0 + x < c1 or x + c0 < c1 -> x < c1 - c0 *)
      (match K.view lhs with
      | Binary { op = `Add; lhs = a; rhs = b; _ } ->
          let c0, x =
            if K.is_const a then Some a, b
            else if K.is_const b then Some b, a
            else None, a
          in
          (match c0 with
          | Some c0 ->
              Some (K.binary ~op:`Cmplt ~lhs:x
                      ~rhs:(K.binary ~op:`Sub ~lhs:rhs ~rhs:c0))
          | None -> None)
      | _ -> None)
  | _ -> None

let try_both_orderings lhs rhs f =
  match f lhs rhs with
  | Some _ as r -> r
  | None -> f rhs lhs

let distribute_neg node =
  match K.view node with
  | Binary { op = `Mul; lhs; rhs; _ } ->
      try_both_orderings lhs rhs (fun neg_one sum ->
        if is_const_int (-1) neg_one then
          match K.view sum with
          | Binary { op = `Add; lhs = x; rhs = c; _ } when K.is_const c ->
              Some
                (K.binary ~op:`Add
                   ~lhs:(K.unary ~op:`Neg ~src:x)
                   ~rhs:(K.unary ~op:`Neg ~src:c))
          | _ -> None
        else None)
  | _ -> None

let float_div_chain node =
  match K.view node with
  | Binary { op = `Fdiv; lhs; rhs = x3; _ } ->
      (match K.view lhs with
      | Binary { op = `Fdiv; lhs = x; rhs = x2; _ } when not (x2 == x3) ->
          Some (K.binary ~op:`Fdiv ~lhs:x
                  ~rhs:(K.binary ~op:`Mul ~lhs:x2 ~rhs:x3))
      | _ -> None)
  | _ -> None

let distribute_const_mul node =
  match K.view node with
  | Binary { op = `Mul; lhs; rhs; dtype }
    when is_index_dtype dtype && K.is_const lhs ->
      (match K.view rhs with
      | Binary { op = `Add; lhs = x; rhs = c; _ } ->
          Some
            (K.binary ~op:`Add
               ~lhs:(K.binary ~op:`Mul ~lhs ~rhs:x)
               ~rhs:(K.binary ~op:`Mul ~lhs ~rhs:c))
      | _ -> None)
  | _ -> None

let where_not_inversion node =
  match K.view node with
  | Ternary { op = `Where; a; b = t; c = f; _ } ->
      (match K.view a with
      | Binary { op = `Cmpne; lhs = cond; rhs; _ }
        when is_const_bool true rhs ->
          Some (K.ternary ~op:`Where ~a:cond ~b:f ~c:t)
      | _ -> None)
  | _ -> None

let lt_mul_fold node =
  match K.view node with
  | Binary { op = `Cmplt; lhs; rhs; _ } ->
      let extract_cmul n =
        match K.view n with
        | Binary { op = `Mul; lhs = a; rhs = b; dtype } when is_index_dtype dtype ->
            (match const_int_val a with
            | Some cv -> Some (cv, b, dtype)
            | None -> Option.map (fun cv -> (cv, a, dtype)) (const_int_val b))
        | _ -> None
      in
      (match extract_cmul lhs with
      | Some (c0, x, dtype) ->
          (match const_int_val rhs with
          | Some c1 when c0 > 0L && c1 > 0L ->
              let ceil_div = Int64.div (Int64.add c1 (Int64.sub c0 1L)) c0 in
              Some (K.binary ~op:`Cmplt ~lhs:x
                      ~rhs:(K.const (Const.int64 dtype ceil_div)))
          | Some c1 when c0 < 0L && c0 <> -1L && c1 <= 0L ->
              let div_val =
                Int64.neg (Int64.div (Int64.neg c1) (Int64.neg c0))
              in
              Some (K.binary ~op:`Cmplt
                      ~lhs:(K.unary ~op:`Neg ~src:x)
                      ~rhs:(K.const (Const.int64 dtype (Int64.neg div_val))))
          | _ -> None)
      | None -> None)
  | _ -> None

let lt_div_fold node =
  match K.view node with
  | Binary { op = `Cmplt; lhs; rhs; _ } ->
      (match K.view lhs with
      | Binary { op = `Idiv; lhs = x; rhs = d; dtype }
        when is_index_dtype dtype ->
          (match const_int_val d, const_int_val rhs with
          | Some dv, Some cv when dv > 0L ->
              let bound =
                if cv > 0L then Int64.mul cv dv
                else Int64.sub (Int64.mul cv dv) (Int64.sub dv 1L)
              in
              Some (K.binary ~op:`Cmplt ~lhs:x
                      ~rhs:(K.const (Const.int64 dtype bound)))
          | _ -> None)
      | _ -> None)
  | _ -> None

let lt_sign_flip node =
  match K.view node with
  | Binary { op = `Cmplt; lhs; rhs; _ } ->
      (match K.view lhs, K.view rhs with
      | Binary { op = `Mul; lhs = x; rhs = lc; _ },
        Binary { op = `Mul; lhs = y; rhs = rc; _ }
        when is_const_int (-1) lc && is_const_int (-1) rc ->
          Some (K.binary ~op:`Cmplt ~lhs:y ~rhs:x)
      | _ -> None)
  | _ -> None

let cast_chain_fold node =
  match K.view node with
  | Cast { src; dtype = b_dt } ->
      (match K.view src with
      | Cast { src = x; dtype = a_dt } ->
          if Dtype.can_lossless_cast (K.dtype_or Dtype.void x) (Dtype.any_to_val a_dt) then
            Some (K.cast ~src:x ~dtype:b_dt)
          else None
      | _ -> None)
  | _ -> None

let is_side_effecting node =
  match K.view node with
  | Range _ | Store _ | End _ | Unroll _ | Barrier -> true
  | _ -> false

let after_cleanup node =
  match K.view node with
  | After { src; deps = [] } -> Some src
  | _ -> None

let bool_or_not node =
  match K.view node with
  | Binary { op = `Or; lhs = x; rhs; _ }
    when Dtype.is_bool (K.dtype_or Dtype.void node) ->
      (* x | !x -> true, or !x | x -> true *)
      let is_not_of target n =
        match K.view n with
        | Binary { op = `Cmpne; lhs = nx; rhs = t; _ }
          when nx == target && is_const_bool true t -> true
        | _ -> false
      in
      if is_not_of x rhs || is_not_of rhs x then Some (K.const_bool true)
      else None
  | _ -> None

(* SINK/GROUP cleanup *)

let is_removable_from_sink node =
  match K.view node with
  | Unroll _ | Vectorize _ -> true
  | _ -> false

let flatten_removable srcs =
  List.concat_map
    (fun s -> if is_removable_from_sink s then K.children s else [ s ])
    srcs

let sink_cleanup node =
  match K.view node with
  | Sink { srcs; kernel_info } when List.exists is_removable_from_sink srcs ->
      Some (K.sink ~kernel_info:(Option.get kernel_info) (flatten_removable srcs))
  | Group { srcs } when List.exists is_removable_from_sink srcs ->
      Some (K.group (flatten_removable srcs))
  | _ -> None

let group_singleton node =
  match K.view node with
  | Group { srcs = [ x ] } -> Some x
  | _ -> None

let empty_unroll node =
  match K.view node with
  | Unroll { src; axes = []; _ } -> Some src
  | _ -> None

(* Phase 3: POW decomposition, reciprocal algebra, etc. *)

let pow_fold node =
  match K.view node with
  | Binary { op = `Pow; lhs = base; rhs = exp; dtype }
    when Dtype.is_float dtype ->
      Some (Decomposition.xpow ~base ~exponent:exp)
  | _ -> None

let where_cast_push node =
  match K.view node with
  | Cast { src; dtype } ->
      (match K.view src with
      | Ternary { op = `Where; a = s; b = a; c = b; _ } ->
          Some (K.ternary ~op:`Where ~a:s
                  ~b:(K.cast ~src:a ~dtype)
                  ~c:(K.cast ~src:b ~dtype))
      | _ -> None)
  | _ -> None

let reciprocal_algebra node =
  match K.view node with
  | Unary { op = `Recip; src; _ } ->
      (match K.view src with
      | Binary { op = `Mul; lhs = x1; rhs = x2; _ } when x1 == x2 ->
          let rx = K.unary ~op:`Recip ~src:x1 in
          Some (K.binary ~op:`Mul ~lhs:rx ~rhs:rx)
      | Binary { op = `Mul; lhs = x; rhs = c; _ } when K.is_const c ->
          Some (K.binary ~op:`Mul
                  ~lhs:(K.unary ~op:`Recip ~src:x)
                  ~rhs:(K.unary ~op:`Recip ~src:c))
      | _ -> None)
  (* x * 1/(1+x) -> 1 - 1/(1+x) *)
  | Binary { op = `Mul; lhs = x; rhs; _ } ->
      (match K.view rhs with
      | Unary { op = `Recip; src = sum; _ } ->
          (match K.view sum with
          | Binary { op = `Add; lhs = a; rhs = b; _ }
            when (a == x && is_const_int 1 b) ||
                 (b == x && is_const_int 1 a) ->
              let one = K.const (Const.float (K.dtype_or Dtype.float32 node) 1.0) in
              Some (K.binary ~op:`Sub ~lhs:one ~rhs)
          | _ -> None)
      | _ -> None)
  | _ -> None

let distribute_neg_full node =
  match K.view node with
  | Binary { op = `Mul; lhs; rhs; _ } ->
      try_both_orderings lhs rhs (fun neg_one sum ->
        if is_const_int (-1) neg_one then
          match K.view sum with
          | Binary { op = `Add; lhs = x; rhs = y; _ } ->
              Some (K.binary ~op:`Add
                      ~lhs:(K.unary ~op:`Neg ~src:x)
                      ~rhs:(K.unary ~op:`Neg ~src:y))
          | _ -> None
        else None)
  | _ -> None

let distribute_mul_index node =
  match K.view node with
  | Binary { op = `Mul; lhs; rhs; dtype }
    when is_index_dtype dtype && K.is_const rhs ->
      (match K.view lhs with
      | Binary { op = `Add; lhs = x; rhs = y; _ } ->
          Some (K.binary ~op:`Add
                  ~lhs:(K.binary ~op:`Mul ~lhs:x ~rhs)
                  ~rhs:(K.binary ~op:`Mul ~lhs:y ~rhs))
      | _ -> None)
  | _ -> None

(* Propagate Invalid_index *)

let is_invalid node =
  match K.view node with Invalid_index _ -> true | _ -> false

let is_comparison = function
  | `Cmplt | `Cmpeq | `Cmpne -> true
  | _ -> false

let decompose_invalid_gate node =
  match K.view node with
  | Ternary { op = `Where; a = cond; b = x; c = inv; _ }
    when is_invalid inv -> Some (cond, x, inv)
  | _ -> None

let invalid_is_index inv =
  match K.view inv with
  | Invalid_index { dtype; _ } -> Dtype.equal (Dtype.scalar_of dtype) Dtype.index
  | _ -> false

let propagate_invalid node =
  match K.view node with
  | Cast { src; dtype } ->
      (match decompose_invalid_gate src with
      | Some (cond, x, inv) ->
          if invalid_is_index inv then Some (K.cast ~src:x ~dtype)
          else
            Some (K.ternary ~op:`Where ~a:cond
                    ~b:(K.cast ~src:x ~dtype)
                    ~c:(K.cast ~src:inv ~dtype))
      | None -> None)
  | Unary { op; src; _ } ->
      (match decompose_invalid_gate src with
      | Some (cond, x, inv) ->
          Some (K.ternary ~op:`Where ~a:cond ~b:(K.unary ~op ~src:x) ~c:inv)
      | None ->
          (match K.view src with Invalid_index _ -> Some src | _ -> None))
  | Binary { op; lhs; rhs; _ } when not (is_comparison op) ->
      (match decompose_invalid_gate lhs, decompose_invalid_gate rhs with
      | Some (cond, x, inv), None ->
          Some (K.ternary ~op:`Where ~a:cond
                  ~b:(K.binary ~op ~lhs:x ~rhs) ~c:inv)
      | None, Some (cond, x, inv) ->
          Some (K.ternary ~op:`Where ~a:cond
                  ~b:(K.binary ~op ~lhs ~rhs:x) ~c:inv)
      | _ ->
          if is_invalid lhs then Some lhs
          else if is_invalid rhs then Some rhs
          else None)
  | Binary { op; lhs; rhs; _ } when is_comparison op ->
      let handle_side gate ~build =
        match decompose_invalid_gate gate with
        | Some (cond, x, inv) ->
            if invalid_is_index inv then Some (build x)
            else
              Some (K.ternary ~op:`Where ~a:cond
                      ~b:(build x)
                      ~c:(K.cast ~src:inv ~dtype:(Dtype.to_any Dtype.bool)))
        | None -> None
      in
      (match handle_side lhs
               ~build:(fun x -> K.binary ~op ~lhs:x ~rhs) with
      | Some _ as r -> r
      | None ->
          handle_side rhs
            ~build:(fun x -> K.binary ~op ~lhs ~rhs:x))
  | Ternary { op = `Where; a = cond; b = inv; c = val_; _ }
    when is_invalid inv ->
      if is_invalid val_ then Some inv
      else
        let not_cond =
          K.binary ~op:`Cmpeq ~lhs:cond ~rhs:(K.const (Const.bool false))
        in
        Some (K.ternary ~op:`Where ~a:not_cond ~b:val_ ~c:inv)
  | Ternary { op = `Where; a; b = gate; c; _ }
    when not (is_invalid c) ->
      (match decompose_invalid_gate gate with
      | Some (cond, x, inv) ->
          Some (K.ternary ~op:`Where ~a:cond
                  ~b:(K.ternary ~op:`Where ~a ~b:x ~c)
                  ~c:inv)
      | None -> None)
  | Ternary { op = `Where; a; b; c = gate; _ }
    when not (is_invalid b) ->
      (match decompose_invalid_gate gate with
      | Some (cond, x, inv) ->
          Some (K.ternary ~op:`Where ~a:cond
                  ~b:(K.ternary ~op:`Where ~a ~b ~c:x)
                  ~c:inv)
      | None -> None)
  | Bitcast { src; dtype } when is_invalid src ->
      Some (K.cast ~src ~dtype:(Dtype.to_any dtype))
  | Bitcast { src; dtype } ->
      (match decompose_invalid_gate src with
      | Some (cond, x, inv) ->
          Some (K.ternary ~op:`Where ~a:cond
                  ~b:(K.bitcast ~src:x ~dtype)
                  ~c:(K.bitcast ~src:inv ~dtype))
      | None -> None)
  | _ -> None

let fold_gated_load_store node =
  match K.view node with
  | Load { src; alt; dtype } ->
      (match K.view src with
      | Index { idxs; _ } when List.exists is_invalid idxs ->
          let zero =
            match alt with
            | Some a -> a
            | None ->
                if Dtype.is_float dtype then K.const (Const.float dtype 0.0)
                else K.const (Const.int64 dtype 0L)
          in
          Some zero
      | _ -> None)
  | Store { dst; value; ranges } ->
      (match decompose_invalid_gate value with
      | Some (cond, val_, _inv) ->
          (match K.view dst with
          | Index { ptr; idxs; gate; _ } ->
              let gated_idxs =
                List.map
                  (fun idx ->
                    K.ternary ~op:`Where ~a:cond ~b:idx
                      ~c:(K.invalid_index ()))
                  idxs
              in
              let combined_gate = match gate with
                | Some g -> K.binary ~op:`And ~lhs:cond ~rhs:g
                | None -> cond
              in
              Some (K.store
                      ~dst:(K.index ~ptr ~idxs:gated_idxs ~gate:combined_gate ())
                      ~value:val_ ~ranges)
          | _ -> None)
      | None -> None)
  | _ -> None

(* Composed passes *)

let phase1_rules =
  [ propagate_invalid;
    const_fold;
    self_fold;
    identity_fold;
    float_div_fold;
    where_fold;
    idempotent_fold;
    trunc_int;
    bool_arith_fold;
    double_not_fold;
    bool_where_identity;
    const_cast_fold;
    double_cast_fold;
    cast_to_bool;
    nested_where_fold;
    divmod_reconstitute ]

let phase2_rules =
  [ commutative_flip;
    combine_terms;
    associative_fold;
    const_to_end;
    nested_div_fold;
    range_self_divmod;
    max_fold;
    range_collapse;
    lt_const_fold;
    distribute_neg;
    float_div_chain;
    distribute_const_mul;
    where_not_inversion;
    lt_mul_fold;
    lt_div_fold;
    lt_sign_flip;
    cast_chain_fold;
    after_cleanup;
    bool_or_not;
    Divandmod.div_and_mod_symbolic;
    sink_cleanup;
    group_singleton;
    empty_unroll ]

let symbolic_simple = K.first_match phase1_rules

let symbolic = K.first_match (phase1_rules @ phase2_rules)

let sym =
  K.first_match
    ([ gep_pushing; alu_through_broadcast_vectorize ]
     @ phase1_rules @ phase2_rules
     @ [ pow_fold; where_cast_push; reciprocal_algebra;
         distribute_neg_full; distribute_mul_index;
         fold_gated_load_store ])
