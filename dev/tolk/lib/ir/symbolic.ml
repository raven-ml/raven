(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module K = Kernel

(* GEP pushing *)

(* GEP(Vectorize(...), i) → lane i *)
let gep_vectorize node =
  match K.view node with
  | Gep { src; idx; _ } -> (
      match K.view src with
      | Vectorize { srcs; _ } when idx >= 0 && idx < List.length srcs ->
          Some (List.nth srcs idx)
      | _ -> None)
  | _ -> None

(* GEP(Const(c), _) → Const(c) *)
let gep_const node =
  match K.view node with
  | Gep { src; _ } -> (
      match K.view src with
      | Const { value; _ } -> Some (K.const value)
      | _ -> None)
  | _ -> None

(* GEP(void_src, _) → void_src. Matches void and effect nodes (Store, End, etc.)
   whose dtype is either void or absent. *)
let gep_void node =
  match K.view node with
  | Gep { src; dtype; _ } when dtype = Dtype.void -> Some src
  | _ -> None

(* CAT(a, b, ...) → Vectorize(Gep(a,0), ..., Gep(b,0), ...)
   CAT can't be rendered; expand to a single Vectorize with GEPs. *)
let cat_to_vectorize node =
  match K.view node with
  | Cat { srcs; _ } ->
      let expanded =
        List.concat_map
          (fun s ->
            let count = (K.dtype_or Dtype.void s).count in
            List.init count (fun i -> K.gep ~src:s ~idx:i))
          srcs
      in
      Some (K.vectorize ~srcs:expanded)
  | _ -> None

(* Vectorize(Gep(x,0), Gep(x,1), ..., Gep(x,n-1)) → x
   when indices are in-order and cover the full source.
   Inverse of Vectorize(Gep) splitting. *)
let vectorize_in_order_gep node =
  match K.view node with
  | Vectorize { srcs = (first :: _ as srcs); _ } -> (
      match K.view first with
      | Gep { src = base; idx = 0; _ } ->
          let count = List.length srcs in
          let base_count = (K.dtype_or Dtype.void base).count in
          if count = base_count then
            let rec check i = function
              | [] -> true
              | s :: rest -> (
                  match K.view s with
                  | Gep { src; idx; _ } -> src == base && idx = i && check (i + 1) rest
                  | _ -> false)
            in
            if check 0 srcs then Some base else None
          else None
      | _ -> None)
  | _ -> None

let gep_pushing =
  K.first_match
    [ gep_vectorize; gep_const; gep_void; cat_to_vectorize;
      vectorize_in_order_gep ]

(* ALU/Vectorize reordering *)

(* ALU(Vectorize(x, x, ...), Vectorize(y, y, ...)) → Vectorize(ALU(x, y), ...)
   when the Vectorize is a broadcast (all lanes identical). *)
let alu_through_broadcast_vectorize node =
  match K.view node with
  | Binary { op; lhs; rhs; _ } -> (
      match (K.view lhs, K.view rhs) with
      | ( Vectorize { srcs = lsrcs; dtype = ldt },
          Vectorize { srcs = rsrcs; dtype = rdt } )
        when ldt.count = rdt.count && ldt.count > 0 ->
          let lx = List.hd lsrcs in
          let rx = List.hd rsrcs in
          let l_broadcast = List.for_all (fun s -> s == lx) lsrcs in
          let r_broadcast = List.for_all (fun s -> s == rx) rsrcs in
          if l_broadcast && r_broadcast then
            let scalar = K.binary ~op ~lhs:lx ~rhs:rx in
            Some (K.vectorize ~srcs:(List.init ldt.count (fun _ -> scalar)))
          else None
      | _ -> None)
  | Unary { op; src; _ } -> (
      match K.view src with
      | Vectorize { srcs; dtype } when dtype.count > 0 ->
          let x = List.hd srcs in
          if List.for_all (fun s -> s == x) srcs then
            let scalar = K.unary ~op ~src:x in
            Some (K.vectorize ~srcs:(List.init dtype.count (fun _ -> scalar)))
          else None
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
  | `Trunc -> Float.round v

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
  | Unary { op; src; _ } -> (
      match K.view src with
      | Const { value; dtype } -> (
          match Const.view value with
          | Float f ->
              let r = exec_unary op f in
              Some (K.const (Const.float dtype r))
          | _ -> None)
      | _ -> None)
  | Binary { op; lhs; rhs; _ } -> (
      match (K.view lhs, K.view rhs) with
      | Const { value = lv; dtype = ld }, Const { value = rv; _ } -> (
          match (Const.view lv, Const.view rv) with
          | Float lf, Float rf -> (
              match exec_binary_float op lf rf with
              | Some r -> Some (K.const (Const.float ld r))
              | None -> None)
          | Int li, Int ri -> (
              match exec_binary_int op li ri with
              | Some r -> Some (K.const (Const.int64 ld r))
              | None -> None)
          | Bool lb, Bool rb -> (
              match op with
              | `And -> Some (K.const (Const.bool (lb && rb)))
              | `Or -> Some (K.const (Const.bool (lb || rb)))
              | `Xor -> Some (K.const (Const.bool (lb <> rb)))
              | `Cmpeq -> Some (K.const (Const.bool (lb = rb)))
              | `Cmpne -> Some (K.const (Const.bool (lb <> rb)))
              | _ -> None)
          | _ -> None)
      | _ -> None)
  | _ -> None

(* Identity folding *)

let is_const_int n node =
  match K.view node with
  | Const { value; _ } -> (
      match Const.view value with Int v -> v = Int64.of_int n | _ -> false)
  | _ -> false

let is_const_float f node =
  match K.view node with
  | Const { value; _ } -> (
      match Const.view value with Float v -> v = f | _ -> false)
  | _ -> false

let identity_fold node =
  match K.view node with
  | Binary { op = `Add; lhs; rhs; _ } ->
      if is_const_int 0 rhs then Some lhs
      else if is_const_int 0 lhs then Some rhs
      else if is_const_float 0.0 rhs then Some lhs
      else if is_const_float 0.0 lhs then Some rhs
      else None
  | Binary { op = `Sub; lhs; rhs; _ } ->
      if is_const_int 0 rhs then Some lhs
      else if is_const_float 0.0 rhs then Some lhs
      else None
  | Binary { op = `Mul; lhs; rhs; _ } ->
      if is_const_int 1 rhs then Some lhs
      else if is_const_int 1 lhs then Some rhs
      else if is_const_float 1.0 rhs then Some lhs
      else if is_const_float 1.0 lhs then Some rhs
      else None
  | Binary { op = (`Idiv | `Fdiv); lhs; rhs; _ } ->
      if is_const_int 1 rhs then Some lhs
      else if is_const_float 1.0 rhs then Some lhs
      else None
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
      if K.dtype_or Dtype.void src = dtype then Some src else None
  | _ -> None

(* SINK/GROUP cleanup *)

(* Strip UNROLL and VECTORIZE wrappers from SINK and GROUP children.
   These are expansion bookkeeping transparent at the sink level. *)
let is_removable_from_sink node =
  match K.view node with
  | Unroll _ | Vectorize _ -> true
  | _ -> false

let sink_cleanup node =
  match K.view node with
  | Sink { srcs; kernel_info } ->
      if List.exists is_removable_from_sink srcs then
        let flat =
          List.concat_map
            (fun s ->
              if is_removable_from_sink s then K.children s
              else [ s ])
            srcs
        in
        Some (K.sink ~kernel_info:(Option.get kernel_info) flat)
      else None
  | Group { srcs } ->
      if List.exists is_removable_from_sink srcs then
        let flat =
          List.concat_map
            (fun s ->
              if is_removable_from_sink s then K.children s
              else [ s ])
            srcs
        in
        Some (K.group flat)
      else None
  | _ -> None

(* Empty UNROLL is NOOP: UNROLL(x, []) → x *)
let empty_unroll node =
  match K.view node with
  | Unroll { src; axes = []; _ } -> Some src
  | _ -> None

(* sym *)

let sym =
  K.first_match
    [
      gep_pushing;
      alu_through_broadcast_vectorize;
      const_fold;
      identity_fold;
      sink_cleanup;
      empty_unroll;
    ]
