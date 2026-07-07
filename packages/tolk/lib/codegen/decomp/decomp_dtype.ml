(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop

let int64_to_int_checked n =
  if Int64.compare n (Int64.of_int min_int) < 0
     || Int64.compare n (Int64.of_int max_int) > 0
  then None
  else Some (Int64.to_int n)

let const_float_dt dt x = Uop.const (Const.float (Dtype.val_of dt) x)

let fconst_like node x =
  let v = Dtype.val_of (Uop.dtype node) in
  Uop.const (Const.float v x)

let float_div lhs rhs =
  Uop.alu_binary ~op:Ops.Mul ~lhs
    ~rhs:(Uop.alu_unary ~op:Ops.Reciprocal ~src:rhs)

(* Long decomposition: int64 -> int32 pairs.

   Emulates int64/uint64 as a pair of 32-bit halves tagged "0" (low) and
   "1" (high). Rewrites arithmetic, comparisons, bit operations, shifts,
   casts, loads/stores, and constants over the pair representation. *)

let is_long_dtype (dt : Dtype.Val.t) =
  let s = Dtype.Val.scalar dt in
  s = Dtype.Int64 || s = Dtype.Uint64

let long_to_int_dtype (dt : Dtype.Val.t) =
  match Dtype.Val.scalar dt with
  | Dtype.Int64 -> Dtype.Val.int32
  | Dtype.Uint64 -> Dtype.Val.uint32
  | _ -> dt

(* Shift by a constant amount expressed as mul/div by a power of two on
   [x]'s own integer dtype. *)
let shr_i x n =
  let v = Dtype.val_of (Uop.dtype x) in
  let op =
    if Dtype.Val.is_int v && not (Dtype.Val.is_unsigned v) then Ops.Floordiv
    else Ops.Cdiv
  in
  Uop.alu_binary ~op ~lhs:x
    ~rhs:(Uop.const (Const.int64 v (Int64.shift_left 1L n)))

let shl_i x n =
  let v = Dtype.val_of (Uop.dtype x) in
  Uop.alu_binary ~op:Ops.Mul ~lhs:x
    ~rhs:(Uop.const (Const.int64 v (Int64.shift_left 1L n)))

let const_of_node_int node =
  match Uop.op node, Uop.arg node with
  | Ops.Const, Uop.Arg.Value c ->
      (match Const.view c with
       | Const.Int n -> int64_to_int_checked n
       | _ -> None)
  | _ -> None

let expr_shr x y =
  match const_of_node_int y with
  | Some n -> shr_i x n
  | None -> Uop.alu_binary ~op:Ops.Shr ~lhs:x ~rhs:y

let expr_shl x y =
  match const_of_node_int y with
  | Some n -> shl_i x n
  | None -> Uop.alu_binary ~op:Ops.Shl ~lhs:x ~rhs:y

(* [reindex idx off mul]: rebuild an INDEX node so that the scalar offset
   is scaled by [mul] and shifted by [off]. Used to stride through the
   [lo; hi; lo; hi; ...] layout of the decomposed buffer. *)
let reindex (idx : Uop.t) off mul =
  match Uop.op idx, Uop.src idx with
  | Ops.Shrink, [| ptr; offset; _size |] ->
      if mul <> 1 then invalid_arg "Decomp_dtype.reindex: SHRINK with mul <> 1";
      let open Uop.O in
      let as_ptr = match Uop.dtype idx with Dtype.Ptr _ -> true | _ -> false in
      Uop.index ~ptr ~idxs:[(offset + int_ off)] ~as_ptr ()
  | _ ->
      match Uop.as_index idx with
      | Some { ptr; idxs = i :: idxs } ->
          let open Uop.O in
          let as_ptr =
            match Uop.dtype idx with Dtype.Ptr _ -> true | _ -> false
          in
          Uop.index ~ptr ~idxs:((i * int_ mul + int_ off) :: idxs) ~as_ptr ()
      | Some { idxs = []; _ } -> idx
      | None -> idx

(* [unpack32 v]: split a 32-bit value into its low and high 16-bit halves
   as uint32 values. Used for 32x32 partial-product expansion in l2i MUL. *)
let unpack32 (v : Uop.t) : Uop.t * Uop.t =
  let u = Uop.bitcast ~src:v ~dtype:Dtype.uint32 in
  let lo = Uop.alu_binary ~op:Ops.And ~lhs:u
    ~rhs:(Uop.const (Const.int Dtype.Val.uint32 0xFFFF)) in
  let hi = shr_i u 16 in
  (lo, hi)

type l2i_op =
  | L2i_neg | L2i_shl | L2i_shr | L2i_add | L2i_sub | L2i_mul
  | L2i_idiv | L2i_mod
  | L2i_cmplt | L2i_cmpeq | L2i_cmpne
  | L2i_xor | L2i_or | L2i_and
  | L2i_where | L2i_max | L2i_bitcast

let rec l2i (op : l2i_op) (dt : Dtype.Val.t) (uops : Uop.t list) : Uop.t * Uop.t =
  let zero = Uop.const (Const.int dt 0) in
  let a0, a1 = match uops with
    | [a0; a1] -> (a0, a1)
    | [a0; a1; _] when op = L2i_shl || op = L2i_shr -> (a0, a1)
    | [a0; a1; _; _] -> (a0, a1)
    | [_; t_lo; t_hi; _; _] when op = L2i_where -> (t_lo, t_hi)
    | _ -> failwith "l2i: unexpected operand count"
  in
  let b0, b1 = match uops with
    | [_; _; b0] when op = L2i_shl || op = L2i_shr ->
        (b0, zero)
    | [_; _; b0; b1] -> (b0, b1)
    | _ -> (zero, zero)
  in
  match op with
  | L2i_neg -> l2i L2i_sub dt [zero; zero; a0; a1]
  | L2i_shl ->
      let mask31 = Uop.const (Const.int dt 31) in
      let b0_mod = Uop.alu_binary ~op:Ops.And ~lhs:b0 ~rhs:mask31 in
      let lo = expr_shl a0 b0_mod in
      let one_c = Uop.const (Const.int dt 1) in
      let comp = Uop.alu_binary ~op:Ops.Sub ~lhs:mask31 ~rhs:b0_mod in
      let hi = Uop.alu_binary ~op:Ops.Or
        ~lhs:(expr_shl a1 b0_mod)
        ~rhs:(expr_shr (expr_shr a0 one_c) comp)
      in
      let ge32 = Uop.alu_binary ~op:Ops.Cmplt ~lhs:mask31 ~rhs:b0 in
      (Uop.alu_ternary ~op:Ops.Where ~a:ge32 ~b:zero ~c:lo,
       Uop.alu_ternary ~op:Ops.Where ~a:ge32 ~b:lo ~c:hi)
  | L2i_shr ->
      let mask31 = Uop.const (Const.int dt 31) in
      let b0_mod = Uop.alu_binary ~op:Ops.And ~lhs:b0 ~rhs:mask31 in
      let one_c = Uop.const (Const.int dt 1) in
      let comp = Uop.alu_binary ~op:Ops.Sub ~lhs:mask31 ~rhs:b0_mod in
      let lo = Uop.alu_binary ~op:Ops.Or
        ~lhs:(expr_shr a0 b0_mod)
        ~rhs:(expr_shl (expr_shl a1 one_c) comp)
      in
      let hi = expr_shr a1 b0_mod in
      let ge32 = Uop.alu_binary ~op:Ops.Cmplt ~lhs:mask31 ~rhs:b0 in
      (Uop.alu_ternary ~op:Ops.Where ~a:ge32 ~b:hi ~c:lo,
       Uop.alu_ternary ~op:Ops.Where ~a:ge32 ~b:zero ~c:hi)
  | L2i_add ->
      let low = Uop.alu_binary ~op:Ops.Add ~lhs:a0 ~rhs:b0 in
      let carry =
        Uop.cast ~dtype:(Dtype.Val dt)
          ~src:(Uop.alu_binary ~op:Ops.Cmplt
                  ~lhs:(Uop.bitcast ~src:low ~dtype:Dtype.uint32)
                  ~rhs:(Uop.bitcast ~src:a0 ~dtype:Dtype.uint32))
      in
      let sum_hi = Uop.alu_binary ~op:Ops.Add ~lhs:a1 ~rhs:b1 in
      (low, Uop.alu_binary ~op:Ops.Add ~lhs:sum_hi ~rhs:carry)
  | L2i_sub ->
      let borrow =
        Uop.cast ~dtype:(Dtype.Val dt)
          ~src:(Uop.alu_binary ~op:Ops.Cmplt
                  ~lhs:(Uop.bitcast ~src:a0 ~dtype:Dtype.uint32)
                  ~rhs:(Uop.bitcast ~src:b0 ~dtype:Dtype.uint32))
      in
      let diff_hi = Uop.alu_binary ~op:Ops.Sub ~lhs:a1 ~rhs:b1 in
      (Uop.alu_binary ~op:Ops.Sub ~lhs:a0 ~rhs:b0,
       Uop.alu_binary ~op:Ops.Sub ~lhs:diff_hi ~rhs:borrow)
  | L2i_cmplt ->
      let hi_lt = Uop.alu_binary ~op:Ops.Cmplt ~lhs:a1 ~rhs:b1 in
      let hi_eq = Uop.alu_binary ~op:Ops.Cmpeq ~lhs:a1 ~rhs:b1 in
      let lo_lt = Uop.alu_binary ~op:Ops.Cmplt
        ~lhs:(Uop.bitcast ~src:a0 ~dtype:Dtype.uint32)
        ~rhs:(Uop.bitcast ~src:b0 ~dtype:Dtype.uint32)
      in
      let both = Uop.alu_binary ~op:Ops.And ~lhs:hi_eq ~rhs:lo_lt in
      (Uop.alu_binary ~op:Ops.Or ~lhs:hi_lt ~rhs:both, zero)
  | L2i_cmpeq ->
      (Uop.alu_binary ~op:Ops.And
         ~lhs:(Uop.alu_binary ~op:Ops.Cmpeq ~lhs:a0 ~rhs:b0)
         ~rhs:(Uop.alu_binary ~op:Ops.Cmpeq ~lhs:a1 ~rhs:b1),
       zero)
  | L2i_cmpne ->
      (Uop.alu_binary ~op:Ops.Or
         ~lhs:(Uop.alu_binary ~op:Ops.Cmpne ~lhs:a0 ~rhs:b0)
         ~rhs:(Uop.alu_binary ~op:Ops.Cmpne ~lhs:a1 ~rhs:b1),
       zero)
  | L2i_xor ->
      (Uop.alu_binary ~op:Ops.Xor ~lhs:a0 ~rhs:b0,
       Uop.alu_binary ~op:Ops.Xor ~lhs:a1 ~rhs:b1)
  | L2i_or ->
      (Uop.alu_binary ~op:Ops.Or ~lhs:a0 ~rhs:b0,
       Uop.alu_binary ~op:Ops.Or ~lhs:a1 ~rhs:b1)
  | L2i_and ->
      (Uop.alu_binary ~op:Ops.And ~lhs:a0 ~rhs:b0,
       Uop.alu_binary ~op:Ops.And ~lhs:a1 ~rhs:b1)
  | L2i_where ->
      (match uops with
       | [cond; t_lo; t_hi; f_lo; f_hi] ->
           (Uop.alu_ternary ~op:Ops.Where ~a:cond ~b:t_lo ~c:f_lo,
            Uop.alu_ternary ~op:Ops.Where ~a:cond ~b:t_hi ~c:f_hi)
       | _ -> failwith "l2i Where: need 5 operands")
  | L2i_max ->
      let cmp, _ = l2i L2i_cmplt dt uops in
      l2i L2i_where dt (cmp :: b0 :: b1 :: a0 :: [a1])
  | L2i_mul ->
      (* 32x32 partial product expansion: split a0,b0 into 16-bit halves
         and sum via recursive ADD carries. *)
      let dt_val = Dtype.Val dt in
      let (a00, a01) = unpack32 a0 in
      let (b00, b01) = unpack32 b0 in
      let p_a00_b01 = Uop.alu_binary ~op:Ops.Mul ~lhs:a00 ~rhs:b01 in
      let p_a01_b00 = Uop.alu_binary ~op:Ops.Mul ~lhs:a01 ~rhs:b00 in
      let p_a00_b00 = Uop.alu_binary ~op:Ops.Mul ~lhs:a00 ~rhs:b00 in
      let p_a01_b01 = Uop.alu_binary ~op:Ops.Mul ~lhs:a01 ~rhs:b01 in
      let mid_lo_a = Uop.bitcast ~src:(shl_i p_a00_b01 16) ~dtype:dt_val in
      let mid_hi_a = Uop.bitcast ~src:(shr_i p_a00_b01 16) ~dtype:dt_val in
      let mid_lo_b = Uop.bitcast ~src:(shl_i p_a01_b00 16) ~dtype:dt_val in
      let mid_hi_b = Uop.bitcast ~src:(shr_i p_a01_b00 16) ~dtype:dt_val in
      let (mid_lo, mid_hi) =
        l2i L2i_add dt [mid_lo_a; mid_hi_a; mid_lo_b; mid_hi_b] in
      let lo_base = Uop.bitcast ~src:p_a00_b00 ~dtype:dt_val in
      let hi_base =
        let hi32 = Uop.bitcast ~src:p_a01_b01 ~dtype:dt_val in
        let cross_ab = Uop.alu_binary ~op:Ops.Mul ~lhs:a0 ~rhs:b1 in
        let cross_ba = Uop.alu_binary ~op:Ops.Mul ~lhs:a1 ~rhs:b0 in
        Uop.alu_binary ~op:Ops.Add
          ~lhs:(Uop.alu_binary ~op:Ops.Add ~lhs:hi32 ~rhs:cross_ab)
          ~rhs:cross_ba
      in
      l2i L2i_add dt [mid_lo; mid_hi; lo_base; hi_base]
  | L2i_bitcast ->
      (* Pure reinterpretation: each half is bitcast to the narrow dtype.
         For long->double, recombination happens in the caller (via the
         bitcast rule). *)
      let dt_val = Dtype.Val dt in
      (Uop.bitcast ~src:a0 ~dtype:dt_val, Uop.bitcast ~src:a1 ~dtype:dt_val)
  | L2i_idiv | L2i_mod ->
      (* TAOCP 4.3.1 shift-subtract long division over 64-bit operands.
         For signed [dt], takes absolute values first, then applies
         C-style sign adjustment afterwards. *)
      let uint = Dtype.Val.uint32 in
      let zero_u = Uop.const (Const.int uint 0) in
      let one_u = Uop.const (Const.int uint 1) in
      let dt_val = Dtype.Val dt in
      let uint_val = Dtype.Val uint in
      let signed = (Dtype.Val.scalar dt = Dtype.Int32) in
      let zero_sign = Uop.const (Const.int dt 0) in
      let (a0u, a1u, b0u, b1u, a_neg_opt, b_neg_opt) =
        if signed then
          let ua0 = Uop.bitcast ~src:a0 ~dtype:uint_val in
          let ua1 = Uop.bitcast ~src:a1 ~dtype:uint_val in
          let ub0 = Uop.bitcast ~src:b0 ~dtype:uint_val in
          let ub1 = Uop.bitcast ~src:b1 ~dtype:uint_val in
          let a_neg = Uop.alu_binary ~op:Ops.Cmplt ~lhs:a1 ~rhs:zero_sign in
          let b_neg = Uop.alu_binary ~op:Ops.Cmplt ~lhs:b1 ~rhs:zero_sign in
          let (na0, na1) = l2i L2i_neg uint [ua0; ua1] in
          let (nb0, nb1) = l2i L2i_neg uint [ub0; ub1] in
          let a0' = Uop.alu_ternary ~op:Ops.Where ~a:a_neg ~b:na0 ~c:ua0 in
          let a1' = Uop.alu_ternary ~op:Ops.Where ~a:a_neg ~b:na1 ~c:ua1 in
          let b0' = Uop.alu_ternary ~op:Ops.Where ~a:b_neg ~b:nb0 ~c:ub0 in
          let b1' = Uop.alu_ternary ~op:Ops.Where ~a:b_neg ~b:nb1 ~c:ub1 in
          (a0', a1', b0', b1', Some a_neg, Some b_neg)
        else
          (a0, a1, b0, b1, None, None)
      in
      let q = ref (zero_u, zero_u) in
      let r = ref (zero_u, zero_u) in
      for i = 63 downto 0 do
        let (r0, r1) = !r in
        let (sr0, sr1) = l2i L2i_shl uint [r0; r1; one_u; zero_u] in
        let shift_const = Uop.const (Const.int uint i) in
        let (bit_lo, _) =
          l2i L2i_shr uint [a0u; a1u; shift_const; zero_u]
        in
        let bit = Uop.alu_binary ~op:Ops.And ~lhs:bit_lo ~rhs:one_u in
        let new_r0 = Uop.alu_binary ~op:Ops.Or ~lhs:sr0 ~rhs:bit in
        let new_r1 = sr1 in
        let (cmp_lo, _) =
          l2i L2i_cmplt uint [new_r0; new_r1; b0u; b1u]
        in
        let cond = Uop.O.not_ cmp_lo in
        let (diff0, diff1) =
          l2i L2i_sub uint [new_r0; new_r1; b0u; b1u]
        in
        let cond_u = Uop.cast ~src:cond ~dtype:uint_val in
        let (q0, q1) = !q in
        let q' =
          if i < 32 then
            (Uop.alu_binary ~op:Ops.Or ~lhs:q0
               ~rhs:(shl_i cond_u (i mod 32)), q1)
          else
            (q0, Uop.alu_binary ~op:Ops.Or ~lhs:q1
                   ~rhs:(shl_i cond_u (i mod 32)))
        in
        q := q';
        let (wr0, wr1) =
          l2i L2i_where uint [cond; diff0; diff1; new_r0; new_r1]
        in
        r := (wr0, wr1)
      done;
      let (q0, q1) = !q in
      let (r0, r1) = !r in
      if signed then
        let a_neg = match a_neg_opt with Some v -> v | None -> assert false in
        (match op with
         | L2i_mod ->
             let (nr0, nr1) = l2i L2i_neg uint [r0; r1] in
             let br0 = Uop.bitcast ~src:r0 ~dtype:dt_val in
             let br1 = Uop.bitcast ~src:r1 ~dtype:dt_val in
             let bnr0 = Uop.bitcast ~src:nr0 ~dtype:dt_val in
             let bnr1 = Uop.bitcast ~src:nr1 ~dtype:dt_val in
             (Uop.alu_ternary ~op:Ops.Where ~a:a_neg ~b:bnr0 ~c:br0,
              Uop.alu_ternary ~op:Ops.Where ~a:a_neg ~b:bnr1 ~c:br1)
         | _ ->
             let b_neg =
               match b_neg_opt with Some v -> v | None -> assert false in
             let (nq0, nq1) = l2i L2i_neg uint [q0; q1] in
             let bq0 = Uop.bitcast ~src:q0 ~dtype:dt_val in
             let bq1 = Uop.bitcast ~src:q1 ~dtype:dt_val in
             let bnq0 = Uop.bitcast ~src:nq0 ~dtype:dt_val in
             let bnq1 = Uop.bitcast ~src:nq1 ~dtype:dt_val in
             let qsign = Uop.alu_binary ~op:Ops.Xor ~lhs:a_neg ~rhs:b_neg in
             (Uop.alu_ternary ~op:Ops.Where ~a:qsign ~b:bnq0 ~c:bq0,
              Uop.alu_ternary ~op:Ops.Where ~a:qsign ~b:bnq1 ~c:bq1))
      else
        (match op with
         | L2i_mod ->
             (Uop.bitcast ~src:r0 ~dtype:dt_val,
              Uop.bitcast ~src:r1 ~dtype:dt_val)
         | _ ->
             (Uop.bitcast ~src:q0 ~dtype:dt_val,
              Uop.bitcast ~src:q1 ~dtype:dt_val))

(* Pointer widen: base dtype becomes its long->int counterpart, element
   count doubles since each long occupies two int32 slots. *)
let widen_long_ptr (dtype : Dtype.Ptr.t) size =
  let new_base = long_to_int_dtype (Dtype.Ptr.base dtype) in
  Dtype.Ptr.create new_base
    ~addrspace:(Dtype.Ptr.addrspace dtype)
    ~size:(if size < 0 then size else size * 2)

(* Classify an op into an l2i dispatch tag for the generic ALU fanout. *)
let classify_alu_op op =
  match op with
  | Ops.Add -> Some L2i_add | Ops.Sub -> Some L2i_sub
  | Ops.Mul -> Some L2i_mul | Ops.Shl -> Some L2i_shl
  | Ops.Shr -> Some L2i_shr | Ops.And -> Some L2i_and
  | Ops.Or -> Some L2i_or | Ops.Xor -> Some L2i_xor
  | Ops.Cmplt -> Some L2i_cmplt | Ops.Cmpeq -> Some L2i_cmpeq
  | Ops.Cmpne -> Some L2i_cmpne | Ops.Max -> Some L2i_max
  | Ops.Neg -> Some L2i_neg | Ops.Where -> Some L2i_where
  | Ops.Cdiv -> Some L2i_idiv | Ops.Cmod -> Some L2i_mod
  | _ -> None

(* [rule_long_defines] narrows definitions with long storage and doubles
   explicit storage size, matching tinygrad's GroupOp.Defines rule. *)
let rule_long_defines =
  let open Upat in
  ops ~name:"n" Ops.Group.defines => fun bs ->
    let n = bs $ "n" in
    match Uop.dtype n with
    | Dtype.Ptr pty when is_long_dtype (Dtype.Ptr.base pty) ->
        let size = Dtype.Ptr.size pty in
        let new_pty = widen_long_ptr pty size in
        Some (Uop.replace n ~dtype:(Dtype.Ptr new_pty) ())
    | Dtype.Val dv when is_long_dtype dv ->
        Some (Uop.replace n ~dtype:(Dtype.Val (long_to_int_dtype dv)) ())
    | _ -> None

(* Tagged INDEX that produces a long value (i.e. the dtype-narrowed
   [Uop.index ~as_ptr:false ...]) -> stride by two and narrow to int32. *)
let rule_long_index_tagged =
  let open Upat in
  op ~name:"ix" Ops.Index => fun bs ->
    let n = bs $ "ix" in
    match Uop.dtype n, Uop.node_tag n with
    | Dtype.Val dv, Some tag when is_long_dtype dv ->
        let off = if String.equal tag "1" then 1 else 0 in
        let narrow = long_to_int_dtype dv in
        Some (Uop.replace (reindex n off 2) ~dtype:(Dtype.Val narrow) ())
    | _ -> None

(* Untagged STORE of a long value -> two tagged stores (low, high). *)
let rule_long_store =
  let open Upat in
  op ~name:"st" Ops.Store => fun bs ->
    let n = bs $ "st" in
    if Uop.node_tag n <> None then None
    else
      match Uop.as_store n with
      | None -> None
      | Some { dst; value; gate } ->
        (match Uop.dtype value with
         | Dtype.Val dv when is_long_dtype dv ->
             let store_lo =
               Uop.with_tag "0"
                 (Uop.store ~dst:(reindex dst 0 2)
                    ~value:(Uop.with_tag "0" value) ?gate ())
             in
             let store_hi =
               Uop.with_tag "1"
                 (Uop.store ~dst:(reindex dst 1 2)
                    ~value:(Uop.with_tag "1" value) ?gate ())
             in
             Some (Uop.group [ store_lo; store_hi ])
         | _ -> None)

(* Tagged LOAD of a long value -> load the matching half of the widened
   buffer. *)
let rule_long_load =
  let open Upat in
  op ~name:"ld" Ops.Load => fun bs ->
    let n = bs $ "ld" in
    match Uop.dtype n with
    | Dtype.Val dv when is_long_dtype dv ->
        (match Uop.node_tag n with
         | Some tag ->
             (match Uop.as_load n with
              | None -> None
              | Some { src; alt; gate } ->
               let off = if tag = "1" then 1 else 0 in
               let alt = Option.map (Uop.with_tag tag) alt in
               Some (Uop.load ~src:(reindex src off 2) ?alt ?gate ()))
         | None -> None)
    | _ -> None

(* Tagged long CONST -> 32-bit constant of the matching half. *)
let rule_long_const =
  let open Upat in
  op ~name:"c" Ops.Const => fun bs ->
    let n = bs $ "c" in
    match Uop.dtype n, Uop.arg n with
    | Dtype.Val dv, Uop.Arg.Value v when is_long_dtype dv ->
        let narrow = long_to_int_dtype dv in
        (match Uop.node_tag n, Const.view v with
         | Some "1", Const.Int bits ->
             let hi = Int64.shift_right_logical bits 32 in
             let hi =
               match Dtype.truncate narrow (`Int hi) with
               | `Int n -> n
               | _ -> assert false
             in
             Some (Uop.const (Const.int64 narrow hi))
         | (Some _ | None), Const.Int bits ->
             let lo = Int64.logand bits 0xFFFFFFFFL in
             let lo =
               match Dtype.truncate narrow (`Int lo) with
               | `Int n -> n
               | _ -> assert false
             in
             Some (Uop.const (Const.int64 narrow lo))
         | _ -> None)
    | _ -> None

(* CAST between two long dtypes (int64 <-> uint64): equivalent to a
   bitcast of each narrow half. Selected by the CAST node's tag. *)
let rule_long_cast_long_to_long =
  let open Upat in
  op ~name:"c" Ops.Cast => fun bs ->
    let n = bs $ "c" in
    let tag = Uop.node_tag n in
    if tag = None then None
    else
      match Uop.dtype n with
      | Dtype.Val dv when is_long_dtype dv ->
          let srcs = Uop.src n in
          if Array.length srcs <> 1 then None
          else
            let a = srcs.(0) in
            (match Uop.dtype a with
             | Dtype.Val adv when is_long_dtype adv ->
                 let src_narrow = long_to_int_dtype adv in
                 let dst_narrow = long_to_int_dtype dv in
                 let half t =
                   let h = Uop.cast ~src:(Uop.with_tag t a)
                     ~dtype:(Dtype.Val src_narrow) in
                   Uop.bitcast ~src:h ~dtype:(Dtype.Val dst_narrow)
                 in
                 (match tag with
                  | Some "0" -> Some (half "0")
                  | Some "1" -> Some (half "1")
                  | _ -> None)
             | _ -> None)
      | _ -> None

(* CAST whose result is long: the operand is non-long (int or float).
   Inlines the direction-specific logic; picks the half by the CAST
   node's tag. *)
let rule_long_cast_to_long =
  let open Upat in
  op ~name:"c" Ops.Cast => fun bs ->
    let n = bs $ "c" in
    let tag = Uop.node_tag n in
    if tag = None then None
    else
      match Uop.dtype n with
      | Dtype.Val dv when is_long_dtype dv ->
          let srcs = Uop.src n in
          if Array.length srcs <> 1 then None
          else
            let a = srcs.(0) in
            let adv_opt =
              match Uop.dtype a with Dtype.Val x -> Some x | _ -> None
            in
            (match adv_opt with
             | None -> None
             | Some adv when is_long_dtype adv -> let _ = adv in None
             | Some adv ->
                 let narrow = long_to_int_dtype dv in
                 let narrow_val = Dtype.Val narrow in
                 if Dtype.Val.is_float adv then begin
                   (* float -> long (truncate toward zero).
                      lo = cast(src, narrow);
                      hi = cast(src / 2^32, narrow)
                           - ((src < 0) & (lo != 0)) *)
                   let lo = Uop.cast ~src:a ~dtype:narrow_val in
                   let two_pow_32 =
                     Uop.const (Const.float adv 4294967296.0) in
                   let hi_float = float_div a two_pow_32 in
                   let hi_int = Uop.cast ~src:hi_float ~dtype:narrow_val in
                   let is_neg =
                     Uop.alu_binary ~op:Ops.Cmplt ~lhs:a
                       ~rhs:(fconst_like a 0.0)
                   in
                   let lo_ne_zero =
                     Uop.alu_binary ~op:Ops.Cmpne ~lhs:lo
                       ~rhs:(Uop.const (Const.int narrow 0)) in
                   let adj =
                     Uop.cast
                       ~src:(Uop.alu_binary ~op:Ops.And
                               ~lhs:is_neg ~rhs:lo_ne_zero)
                       ~dtype:narrow_val
                   in
                   let hi =
                     Uop.alu_binary ~op:Ops.Sub ~lhs:hi_int ~rhs:adj in
                   (match tag with
                    | Some "0" -> Some lo
                    | Some "1" -> Some hi
                    | _ -> None)
                 end
                 else begin
                   (* int -> long (sign-extend).
                      lo = cast(src, narrow);
                      hi = (src < 0) ? -1 : 0 *)
                   let lo = Uop.cast ~src:a ~dtype:narrow_val in
                   let hi =
                     Uop.alu_ternary ~op:Ops.Where
                       ~a:(Uop.alu_binary ~op:Ops.Cmplt ~lhs:a
                             ~rhs:(Uop.const_like a 0))
                       ~b:(Uop.const (Const.int narrow (-1)))
                       ~c:(Uop.const (Const.int narrow 0))
                   in
                   (match tag with
                    | Some "0" -> Some lo
                    | Some "1" -> Some hi
                    | _ -> None)
                 end)
      | _ -> None

(* CAST whose operand is long and result is non-long (int or float):
   expand the long operand into its (lo, hi) halves and combine
   inline. *)
let rule_long_cast_from_long =
  let open Upat in
  op ~name:"c" Ops.Cast => fun bs ->
    let n = bs $ "c" in
    match Uop.dtype n with
    | Dtype.Val dv when is_long_dtype dv -> let _ = dv in None
    | _ ->
        let srcs = Uop.src n in
        if Array.length srcs <> 1 then None
        else
          let a = srcs.(0) in
          (match Uop.dtype a, Uop.dtype n with
           | Dtype.Val adv, Dtype.Val tdv when is_long_dtype adv ->
               let narrow = long_to_int_dtype adv in
               let narrow_val = Dtype.Val narrow in
               let a0 =
                 Uop.cast ~src:(Uop.with_tag "0" a) ~dtype:narrow_val in
               let a1 =
                 Uop.cast ~src:(Uop.with_tag "1" a) ~dtype:narrow_val in
               if Dtype.Val.is_float tdv then begin
                 (* long -> float: small-value fast path + two-half
                    reconstruction in float32. *)
                 let tdv_val = Dtype.Val tdv in
                 let zero_a1 = Uop.const_like a1 0 in
                 let minus_one_a1 = Uop.const (Const.int narrow (-1)) in
                 let zero_a0 = Uop.const_like a0 0 in
                 let hi_zero = Uop.alu_binary ~op:Ops.Cmpeq ~lhs:a1
                                 ~rhs:zero_a1 in
                 let hi_m1 = Uop.alu_binary ~op:Ops.Cmpeq ~lhs:a1
                               ~rhs:minus_one_a1 in
                 let lo_neg = Uop.alu_binary ~op:Ops.Cmplt ~lhs:a0
                                ~rhs:zero_a0 in
                 let lo_ge0 = Uop.O.not_ lo_neg in
                 let small =
                   Uop.alu_binary ~op:Ops.Or
                     ~lhs:(Uop.alu_binary ~op:Ops.And
                             ~lhs:hi_zero ~rhs:lo_ge0)
                     ~rhs:(Uop.alu_binary ~op:Ops.And
                             ~lhs:hi_m1 ~rhs:lo_neg)
                 in
                 let f32 = Dtype.Val Dtype.Val.float32 in
                 let small_branch = Uop.cast ~src:a0 ~dtype:tdv_val in
                 let hi_f32 = Uop.cast ~src:a1 ~dtype:f32 in
                 let two_pow_32 =
                   Uop.const (Const.float Dtype.Val.float32 4294967296.0) in
                 let hi_scaled = Uop.alu_binary ~op:Ops.Mul
                   ~lhs:hi_f32 ~rhs:two_pow_32 in
                 let lo_u = Uop.bitcast ~src:a0 ~dtype:Dtype.uint32 in
                 let lo_f32 = Uop.cast ~src:lo_u ~dtype:f32 in
                 let sum_f32 = Uop.alu_binary ~op:Ops.Add
                   ~lhs:hi_scaled ~rhs:lo_f32 in
                 let big_branch = Uop.cast ~src:sum_f32 ~dtype:tdv_val in
                 Some (Uop.alu_ternary ~op:Ops.Where ~a:small
                         ~b:small_branch ~c:big_branch)
               end
               else begin
                 (* long -> int (narrow the low half). *)
                 let lo_u = Uop.bitcast ~src:a0 ~dtype:Dtype.uint32 in
                 Some (Uop.cast ~src:lo_u ~dtype:(Dtype.Val tdv))
               end
           | _ -> None)

(* BITCAST whose result is long and source is a long (i.e. int64<->uint64):
   expand operand into narrow pair and dispatch through [l2i L2i_bitcast].
   Selected by the BITCAST node's tag. *)
let rule_long_bitcast =
  let open Upat in
  op ~name:"b" Ops.Bitcast => fun bs ->
    let n = bs $ "b" in
    let tag = Uop.node_tag n in
    if tag = None then None
    else
      match Uop.dtype n with
      | Dtype.Val dv when is_long_dtype dv ->
          let srcs = Uop.src n in
          if Array.length srcs <> 1 then None
          else
            let a = srcs.(0) in
            (match Uop.dtype a with
             | Dtype.Val adv when is_long_dtype adv ->
                 let src_narrow = long_to_int_dtype adv in
                 let dst_narrow = long_to_int_dtype dv in
                 let a0 = Uop.cast
                     ~src:(Uop.with_tag "0" a) ~dtype:(Dtype.Val src_narrow) in
                 let a1 = Uop.cast
                     ~src:(Uop.with_tag "1" a) ~dtype:(Dtype.Val src_narrow) in
                 let lo, hi = l2i L2i_bitcast dst_narrow [a0; a1] in
                 (match tag with
                  | Some "0" -> Some lo
                  | Some "1" -> Some hi
                  | _ -> None)
             | _ -> None)
      | _ -> None

(* Comparisons whose operands are long-valued reduce to the (lo, _)
   component of [l2i] on the four tagged halves of the operands. *)
let rule_long_cmp =
  let open Upat in
  ops ~name:"c" [ Ops.Cmplt; Ops.Cmpeq; Ops.Cmpne ] => fun bs ->
    let n = bs $ "c" in
    let srcs = Uop.src n in
    if Array.length srcs <> 2 then None
    else
      let lhs = srcs.(0) and rhs = srcs.(1) in
      match Uop.dtype lhs with
      | Dtype.Val dv when is_long_dtype dv ->
          let dt = long_to_int_dtype dv in
          let l2i_op =
            match Uop.op n with
            | Ops.Cmplt -> L2i_cmplt | Ops.Cmpeq -> L2i_cmpeq
            | Ops.Cmpne -> L2i_cmpne | _ -> assert false
          in
          let args = [
            Uop.with_tag "0" lhs; Uop.with_tag "1" lhs;
            Uop.with_tag "0" rhs; Uop.with_tag "1" rhs;
          ] in
          Some (fst (l2i l2i_op dt args))
      | _ -> None

(* Generic ALU (unary/binary/ternary) whose result dtype is long and
   which has been tagged "0" or "1" by a downstream reader. Expands each
   long operand into a pair of 32-bit sources tagged "0" and "1",
   narrows each via CAST, runs [l2i], and returns the requested half. *)
let rule_long_alu =
  let open Upat in
  ops ~name:"__root__" Ops.Group.alu => fun bs ->
    let n = bs $ "__root__" in
    let tag = Uop.node_tag n in
    if tag = None then None
    else
      let op = Uop.op n in
      if not (Ops.Group.is_alu op) then None
      else
        match Uop.dtype n with
        | Dtype.Val dv when is_long_dtype dv ->
            (match classify_alu_op op with
             | None -> None
             | Some l2i_op ->
                 let dt = long_to_int_dtype dv in
                 let narrow = Dtype.Val dt in
                 let expanded =
                   Array.fold_right (fun c acc ->
                     match Uop.dtype c with
                     | Dtype.Val cdv when is_long_dtype cdv ->
                         Uop.cast ~src:(Uop.with_tag "0" c) ~dtype:narrow ::
                         Uop.cast ~src:(Uop.with_tag "1" c) ~dtype:narrow ::
                         acc
                     | _ -> c :: acc)
                     (Uop.src n) []
                 in
                 let lo, hi = l2i l2i_op dt expanded in
                 (match tag with
                  | Some "0" -> Some lo
                  | Some "1" -> Some hi
                  | _ -> None))
        | _ -> None

let pm_long_decomp : Upat.Pattern_matcher.t =
  Upat.Pattern_matcher.make [
    rule_long_index_tagged;
    rule_long_defines;
    rule_long_store;
    rule_long_load;
    rule_long_const;
    rule_long_cast_long_to_long;
    rule_long_cast_to_long;
    rule_long_cast_from_long;
    rule_long_bitcast;
    rule_long_cmp;
    rule_long_alu;
  ]

type float_decomp_ctx = {
  from_dtype : Dtype.scalar;
  to_dtype : Dtype.scalar;
}

(* Float decomposition: emulated float storage <-> promoted float arithmetic. *)

let float_tag s = Dtype.scalar_to_string s

let scalar_bits s = Dtype.Val.bitsize (Dtype.Val.of_scalar s)

let scalar_val ?(count = 1) s =
  Dtype.Val.vec count (Dtype.Val.of_scalar s)

let float_value_dtype ?(count = 1) s = Dtype.Val (scalar_val ~count s)

let f2f_dt_scalar = function
  | Dtype.Float64 -> Dtype.Uint64
  | Dtype.Float32 -> Dtype.Uint32
  | Dtype.Float16 | Dtype.Bfloat16 -> Dtype.Uint16
  | Dtype.Fp8e4m3 | Dtype.Fp8e5m2
  | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz -> Dtype.Uint8
  | _ -> invalid_arg "Dtype.f2f_dt: not a float dtype"

let f2f_dt ?(count = 1) s = scalar_val ~count (f2f_dt_scalar s)

let is_fp8_scalar = function
  | Dtype.Fp8e4m3 | Dtype.Fp8e5m2
  | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz -> true
  | _ -> false

let is_fp8_fnuz_scalar = function
  | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz -> true
  | _ -> false

let pow2_bits n = Int64.shift_left 1L n

let mask_bits n =
  if n >= 64 then -1L else Int64.sub (pow2_bits n) 1L

let int_const_like_uop u n =
  Uop.const (Const.int64 (Dtype.val_of (Uop.dtype u)) n)

let int_const_val dt n = Uop.const (Const.int64 dt n)

let iand a b = Uop.alu_binary ~op:Ops.And ~lhs:a ~rhs:b
let ior a b = Uop.alu_binary ~op:Ops.Or ~lhs:a ~rhs:b
let iadd a b = Uop.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b
let isub a b = Uop.alu_binary ~op:Ops.Sub ~lhs:a ~rhs:b
let imul a b = Uop.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b
let icmplt a b = Uop.alu_binary ~op:Ops.Cmplt ~lhs:a ~rhs:b
let icmpne a b = Uop.alu_binary ~op:Ops.Cmpne ~lhs:a ~rhs:b
let icmpeq a b = Uop.alu_binary ~op:Ops.Cmpeq ~lhs:a ~rhs:b
let iwhere c t f = Uop.alu_ternary ~op:Ops.Where ~a:c ~b:t ~c:f

let shl_const x n =
  if n = 0 then x else imul x (int_const_like_uop x (pow2_bits n))

let shr_const x n =
  if n = 0 then x else
    Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:(int_const_like_uop x (pow2_bits n))

let cast_to_val dt x = Uop.cast ~src:x ~dtype:(Dtype.Val dt)
let bitcast_to_val dt x = Uop.bitcast ~src:x ~dtype:(Dtype.Val dt)

let rne v s =
  let one = int_const_like_uop v 1L in
  let rounded = shr_const v s in
  let guard = iand (shr_const v (s - 1)) one in
  let sticky =
    icmpne (iand v (int_const_like_uop v (mask_bits (s - 1))))
      (int_const_like_uop v 0L)
  in
  let lsb = iand rounded one in
  iadd rounded
    (iand guard
       (ior (Uop.cast ~src:sticky ~dtype:(Uop.dtype v)) lsb))

let rec f2f v fr to_ =
  let fs = scalar_bits fr in
  let fb = Decomp_transcendental.exponent_bias (Dtype.Val (Dtype.Val.of_scalar fr)) in
  let fe, fm = Dtype.finfo (Dtype.Val (Dtype.Val.of_scalar fr)) in
  let ts = scalar_bits to_ in
  let tb = Decomp_transcendental.exponent_bias (Dtype.Val (Dtype.Val.of_scalar to_)) in
  let te, tm = Dtype.finfo (Dtype.Val (Dtype.Val.of_scalar to_)) in
  let count = Dtype.count (Uop.dtype v) in
  let fr_uint = f2f_dt ~count fr in
  let to_uint = f2f_dt ~count to_ in
  let to_float = float_value_dtype ~count to_ in
  if fe <= te && fm < tm then begin
    let sign =
      shl_const
        (cast_to_val to_uint
           (iand v (int_const_like_uop v (pow2_bits (fs - 1)))))
        (ts - fs)
    in
    let nosign =
      cast_to_val to_uint
        (iand v (int_const_like_uop v (Int64.sub (pow2_bits (fs - 1)) 1L)))
    in
    let exp = shr_const nosign fm in
    let norm =
      iadd (shl_const nosign (tm - fm))
        (int_const_val to_uint (Int64.shift_left (Int64.of_int (tb - fb)) tm))
    in
    let nan =
      ior (shl_const nosign (tm - fm))
        (int_const_val to_uint
           (Int64.shift_left (Int64.of_int ((1 lsl te) - 1)) tm))
    in
    let body =
      if is_fp8_fnuz_scalar fr then
        let fnuz_nan =
          Uop.alu_binary ~op:Ops.And
            ~lhs:(icmpne sign (int_const_val to_uint 0L))
            ~rhs:(icmpeq nosign (int_const_val to_uint 0L))
        in
        let qnan =
          int_const_val to_uint
            (Int64.logor
               (Int64.shift_left (Int64.of_int ((1 lsl te) - 1)) tm)
               (Int64.shift_left 1L (tm - 1)))
        in
        iwhere fnuz_nan qnan
          (ior sign (iwhere (icmpeq exp (int_const_val to_uint 0L))
                       (int_const_val to_uint 0L) norm))
      else
        let is_nan =
          if fr = Dtype.Fp8e4m3 then
            icmpeq nosign (int_const_val to_uint (Int64.of_int ((1 lsl (fm + fe)) - 1)))
          else
            icmpeq exp (int_const_val to_uint (Int64.of_int ((1 lsl fe) - 1)))
        in
        ior sign
          (iwhere (icmpeq exp (int_const_val to_uint 0L))
             (int_const_val to_uint 0L) (iwhere is_nan nan norm))
    in
    Uop.bitcast ~src:body ~dtype:to_float
  end else if fe >= te && fm > tm then begin
    let v =
      bitcast_to_val fr_uint
        (f2f_clamp (Uop.bitcast ~src:v ~dtype:(float_value_dtype ~count fr)) to_)
    in
    let sign =
      iand (shr_const v (fs - ts))
        (int_const_like_uop v (pow2_bits (ts - 1)))
    in
    let nosign =
      iand v (int_const_like_uop v (Int64.sub (pow2_bits (fs - 1)) 1L))
    in
    let norm =
      cast_to_val to_uint
        (isub (rne nosign (fm - tm))
           (int_const_like_uop nosign
              (Int64.shift_left (Int64.of_int (fb - tb)) tm)))
    in
    let underflow =
      icmplt
        (iand (shr_const v fm)
           (int_const_like_uop v (Int64.of_int ((1 lsl fe) - 1))))
        (int_const_like_uop v (Int64.of_int (1 + fb - tb)))
    in
    let nan_mantissa =
      if to_ = Dtype.Fp8e4m3 then
        int_const_like_uop sign (Int64.of_int ((1 lsl tm) - 1))
      else
        iand (shr_const nosign (fm - tm))
          (int_const_like_uop nosign (Int64.of_int ((1 lsl tm) - 1)))
    in
    let nan =
      cast_to_val to_uint
        (ior (ior sign nan_mantissa)
           (int_const_like_uop sign
              (Int64.shift_left (Int64.of_int ((1 lsl te) - 1)) tm)))
    in
    let is_nan =
      icmpeq
        (iand (shr_const v fm)
           (int_const_like_uop v (Int64.of_int ((1 lsl fe) - 1))))
        (int_const_like_uop v (Int64.of_int ((1 lsl fe) - 1)))
    in
    if is_fp8_fnuz_scalar to_ then
      iwhere is_nan
        (int_const_val to_uint (pow2_bits (ts - 1)))
        (iwhere underflow (int_const_val to_uint 0L)
           (ior (cast_to_val to_uint sign) norm))
    else
      iwhere is_nan nan
        (ior (cast_to_val to_uint sign)
           (iwhere underflow (int_const_val to_uint 0L) norm))
  end else
    invalid_arg "Dtype.f2f: unsupported float decomposition"

and f2f_clamp ?(sat = true) val_ dt =
  let e, m = Dtype.finfo (Dtype.Val (Dtype.Val.of_scalar dt)) in
  let max_exp, max_man =
    if is_fp8_fnuz_scalar dt then ((1 lsl e) - 1, (1 lsl m) - 1)
    else if dt = Dtype.Fp8e4m3 then ((1 lsl e) - 1, (1 lsl m) - 2)
    else ((1 lsl e) - 2, (1 lsl m) - 1)
  in
  let mx_value =
    (2.0 ** Float.of_int (max_exp - Decomp_transcendental.exponent_bias (Dtype.Val (Dtype.Val.of_scalar dt))))
    *. (1.0 +. (Float.of_int max_man /. Float.of_int (1 lsl m)))
  in
  let mx = const_float_dt (Uop.dtype val_) mx_value in
  let sat_value =
    if is_fp8_scalar dt && sat then mx
    else const_float_dt (Uop.dtype val_) Float.infinity
  in
  let neg_mx = Uop.alu_unary ~op:Ops.Neg ~src:mx in
  let neg_sat = Uop.alu_unary ~op:Ops.Neg ~src:sat_value in
  iwhere (icmpne val_ val_) val_
    (iwhere (icmplt val_ neg_mx) neg_sat
       (iwhere (icmplt mx val_) sat_value val_))

let f2f_load x fr to_ =
  let count = Dtype.count (Uop.dtype x) in
  let uint_fr = f2f_dt ~count fr in
  if count = 1 then f2f (Uop.replace x ~dtype:(Dtype.Val uint_fr) ()) fr to_
  else
    match Uop.as_load x with
    | None -> invalid_arg "Dtype.f2f_load: expected load"
    | Some { src; _ } ->
        let scalar_uint_fr = f2f_dt fr in
        Uop.stack
          (List.init count (fun i ->
             let ld =
               Uop.replace x
                 ~src:[| reindex src i 1 |]
                 ~dtype:(Dtype.Val scalar_uint_fr) ()
             in
             f2f ld fr to_))

let f2f_store st idx val_ fr to_ =
  let count = Dtype.count (Uop.dtype val_) in
  if count = 1 then
    Uop.replace st
      ~src:[| idx; f2f (Uop.bitcast ~src:val_ ~dtype:(Dtype.Val (f2f_dt to_))) to_ fr |]
      ()
  else
    Uop.group
      (List.init count (fun i ->
         let value =
           f2f
             (Uop.bitcast
                ~src:(Uop.index ~ptr:val_ ~idxs:[ Uop.const_int i ] ())
                ~dtype:(Dtype.Val (f2f_dt to_)))
             to_ fr
         in
         Uop.replace st ~src:[| reindex idx i 1; value |] ()))

let dtype_scalar_count = function
  | Dtype.Val dt -> Some (Dtype.Val.scalar dt, Dtype.Val.count dt)
  | _ -> None

let same_scalar s = function
  | Dtype.Val dt -> Dtype.Val.scalar dt = s
  | _ -> false

let rule_float_defines_index_shrink ctx =
  let open Upat in
  ops ~name:"x" (Ops.Group.defines @ [ Ops.Index; Ops.Shrink ]) => fun bs ->
    let x = bs $ "x" in
    let tag = Some (float_tag ctx.from_dtype) in
    match Uop.dtype x with
    | Dtype.Ptr p when Dtype.Val.scalar (Dtype.Ptr.base p) = ctx.from_dtype ->
        let count = Dtype.Val.count (Dtype.Ptr.base p) in
        let base = f2f_dt ~count ctx.from_dtype in
        Some (Uop.replace x ~dtype:(Dtype.Ptr (Dtype.Ptr.with_base base p))
                ~node_tag:tag ())
    | Dtype.Val dt when Dtype.Val.scalar dt = ctx.from_dtype ->
        let src = Uop.src x in
        if Uop.op x = Ops.Index && Array.length src > 0
           && (Uop.op src.(0) = Ops.Load || Uop.op src.(0) = Ops.Stack)
        then None
        else
          let count = Dtype.Val.count dt in
          Some
            (Uop.replace x
               ~dtype:(Dtype.Val (f2f_dt ~count ctx.from_dtype))
               ~node_tag:tag ())
    | _ -> None

let rule_float_load ctx =
  let open Upat in
  op ~name:"x" Ops.Load => fun bs ->
    let x = bs $ "x" in
    if same_scalar ctx.from_dtype (Uop.dtype x) then
      Some (f2f_load x ctx.from_dtype ctx.to_dtype)
    else None

let rule_float_bitcast_load ctx =
  let open Upat in
  op ~name:"bc" Ops.Bitcast => fun bs ->
    let bc = bs $ "bc" in
    match Uop.src bc with
    | [| ld |] when Uop.op ld = Ops.Load
                  && same_scalar ctx.from_dtype (Uop.dtype ld) ->
        let count = Dtype.count (Uop.dtype ld) in
        Some
          (Uop.bitcast
             ~src:(Uop.replace ld
                     ~dtype:(Dtype.Val (f2f_dt ~count ctx.from_dtype)) ())
             ~dtype:(Uop.dtype bc))
    | _ -> None

let rule_float_bitcast_from ctx =
  let open Upat in
  op ~name:"bc" Ops.Bitcast => fun bs ->
    let bc = bs $ "bc" in
    match Uop.src bc, Uop.dtype bc with
    | [| x |], Dtype.Val bdt
      when same_scalar ctx.to_dtype (Uop.dtype x)
           && Dtype.Val.bitsize bdt = scalar_bits ctx.from_dtype ->
        Some
          (Uop.replace bc
             ~src:[| f2f
                       (Uop.bitcast ~src:x
                          ~dtype:(Dtype.Val
                                    (f2f_dt ~count:(Dtype.Val.count bdt)
                                       ctx.to_dtype)))
                       ctx.to_dtype ctx.from_dtype |]
             ())
    | _ -> None

let rule_float_bitcast_to ctx =
  let open Upat in
  op ~name:"bc" Ops.Bitcast => fun bs ->
    let bc = bs $ "bc" in
    match Uop.src bc with
    | [| x |] when same_scalar ctx.from_dtype (Uop.dtype bc) ->
        let count = Dtype.count (Uop.dtype bc) in
        Some
          (f2f
             (Uop.bitcast ~src:x
                ~dtype:(Dtype.Val (f2f_dt ~count ctx.from_dtype)))
             ctx.from_dtype ctx.to_dtype)
    | _ -> None

let rule_float_cast ctx =
  let open Upat in
  op ~name:"x" Ops.Cast => fun bs ->
    let x = bs $ "x" in
    match Uop.src x with
    | [| val_ |] when same_scalar ctx.from_dtype (Uop.dtype x) ->
        let count = Dtype.count (Uop.dtype x) in
        Some
          (f2f_clamp
             (Uop.cast ~src:val_
                ~dtype:(float_value_dtype ~count ctx.to_dtype))
             ctx.from_dtype)
    | _ -> None

let rule_float_all ctx =
  let open Upat in
  ops ~name:"x" (List.filter (fun op -> op <> Ops.Bitcast) Ops.Group.all) => fun bs ->
    let x = bs $ "x" in
    match dtype_scalar_count (Uop.dtype x) with
    | Some (s, count) when s = ctx.from_dtype ->
        let to_dt = float_value_dtype ~count ctx.to_dtype in
        let src =
          Array.map
            (fun child ->
               if same_scalar ctx.from_dtype (Uop.dtype child) then
                 Uop.cast ~src:child ~dtype:to_dt
               else child)
            (Uop.src x)
        in
        Some (Uop.replace x ~dtype:to_dt ~src ())
    | _ -> None

let rule_float_store_bitcast ctx =
  let open Upat in
  op ~name:"st" Ops.Store => fun bs ->
    let st = bs $ "st" in
    match Uop.as_store st with
    | Some { dst; value; gate = None }
      when Uop.op value = Ops.Bitcast
           && same_scalar ctx.from_dtype (Uop.dtype value)
           && Uop.node_tag dst = Some (float_tag ctx.from_dtype) ->
        let count = Dtype.count (Uop.dtype value) in
        Some
          (Uop.replace st
             ~src:[| dst;
                     Uop.replace value
                       ~dtype:(Dtype.Val (f2f_dt ~count ctx.from_dtype)) () |]
             ())
    | Some _ | None -> None

let rule_float_store ctx =
  let open Upat in
  op ~name:"st" Ops.Store => fun bs ->
    let st = bs $ "st" in
    match Uop.as_store st with
    | Some { dst; value; gate = None }
      when same_scalar ctx.to_dtype (Uop.dtype value) ->
        let idx =
          match Uop.op dst, Uop.src dst with
          | Ops.Cast, [| raw |] -> raw
          | _ -> dst
        in
        if Uop.node_tag idx = Some (float_tag ctx.from_dtype) then
          Some (f2f_store st idx value ctx.from_dtype ctx.to_dtype)
        else None
    | Some _ | None -> None

let pm_float_decomp (ctx : float_decomp_ctx) : Upat.Pattern_matcher.t =
  Upat.Pattern_matcher.make [
    rule_float_defines_index_shrink ctx;
    rule_float_load ctx;
    rule_float_bitcast_load ctx;
    rule_float_bitcast_from ctx;
    rule_float_bitcast_to ctx;
    rule_float_cast ctx;
    rule_float_all ctx;
    rule_float_store_bitcast ctx;
    rule_float_store ctx;
  ]

type dtype_decomp_ctx = {
  detected : (Dtype.scalar, unit) Hashtbl.t;
}

let decomposable_scalar = function
  | Dtype.Float16 | Dtype.Bfloat16
  | Dtype.Fp8e4m3 | Dtype.Fp8e5m2
  | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz
  | Dtype.Int64 | Dtype.Uint64 -> true
  | _ -> false

let canonical_decomp_scalar = function
  | Dtype.Uint64 -> Dtype.Int64
  | scalar -> scalar

let detect_decomp_dtype ctx node =
  let add scalar =
    if decomposable_scalar scalar then
      Hashtbl.replace ctx.detected (canonical_decomp_scalar scalar) ()
  in
  (match Uop.dtype node with
   | Dtype.Val dt -> add (Dtype.Val.scalar dt)
   | Dtype.Ptr ptr -> add (Dtype.Val.scalar (Dtype.Ptr.base ptr)));
  None

let pm_dtype_decomps = detect_decomp_dtype

let dtype_of_scalar scalar =
  Dtype.Val (Dtype.Val.of_scalar scalar)

let should_emulate renderer scalar =
  (not (Renderer.supports_dtype renderer (dtype_of_scalar scalar)))
  ||
  List.exists
    (fun (from_dtype, _) -> from_dtype = scalar)
    (Renderer.emulated_float_dtypes renderer)

let float_decomp_target renderer scalar =
  if is_fp8_scalar scalar
     && not (should_emulate renderer Dtype.Float16)
  then Dtype.Float16
  else Dtype.Float32

let safe_rewrite pm node =
  try Upat.Pattern_matcher.rewrite pm node with
  | Invalid_argument msg
    when String.equal msg "Uop.index: expected pointer ptr" -> None

let do_dtype_decomps (renderer : Renderer.t) (sink : Uop.t) : Uop.t =
  let ctx = { detected = Hashtbl.create 8 } in
  ignore (Uop.graph_rewrite ~name:"detect dtypes" (pm_dtype_decomps ctx) sink);
  let dtypes =
    Hashtbl.fold (fun dtype () acc -> dtype :: acc) ctx.detected []
    |> List.sort compare
    |> List.filter (should_emulate renderer)
  in
  let rewrite pm name sink =
    Uop.graph_rewrite ~name ~bottom_up:true (safe_rewrite pm) sink
  in
  List.fold_left
    (fun sink dtype ->
       match dtype with
       | Dtype.Int64 ->
           rewrite pm_long_decomp "decomp long -> int" sink
       | Dtype.Float16 | Dtype.Bfloat16
       | Dtype.Fp8e4m3 | Dtype.Fp8e5m2
       | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz ->
           let ctx =
             { from_dtype = dtype; to_dtype = float_decomp_target renderer dtype }
           in
           rewrite (pm_float_decomp ctx)
             (Printf.sprintf "decomp %s -> %s"
                (Dtype.scalar_to_string dtype)
                (Dtype.scalar_to_string ctx.to_dtype))
             sink
       | _ -> sink)
    sink dtypes
