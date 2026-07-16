(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop


let const_float_v v x = Uop.const (Const.float v x)

(* Threefry2x32 counter-based PRNG. Splits a 64-bit counter and a 64-bit
   key into 32-bit halves, runs 5 rounds, repacks to 64 bits. *)

let threefry_rotations = [|
  [| 13; 15; 26; 6 |];
  [| 17; 29; 16; 24 |];
|]

let threefry_key_magic = 0x1BD11BDA

let threefry2x32 x key =
  let u32 = Dtype.uint32 in
  let u64 = Dtype.uint64 in
  let two32 = Int64.shift_left 1L 32 in
  let mask32 = Uop.const (Const.int64 Dtype.uint64 0xFFFFFFFFL) in
  let c_two32 = Uop.const (Const.int64 Dtype.uint64 two32) in
  let u32c n = Uop.const (Const.int Dtype.uint32 n) in
  let low32 u =
    Uop.cast ~dtype:u32
      ~src:(Uop.alu_binary ~op:Ops.And ~lhs:u ~rhs:mask32)
  in
  let high32 u =
    Uop.cast ~dtype:u32
      ~src:(Uop.alu_binary ~op:Ops.And
              ~lhs:(Uop.alu_binary ~op:Ops.Cdiv ~lhs:u ~rhs:c_two32)
              ~rhs:mask32)
  in
  (* (v << r) + (v >> (32 - r)) via MUL/CDIV; no logical shift op. *)
  let rot32 v r =
    let hi = Uop.alu_binary ~op:Ops.Mul ~lhs:v ~rhs:(u32c (1 lsl r)) in
    let lo = Uop.alu_binary ~op:Ops.Cdiv ~lhs:v
               ~rhs:(u32c (1 lsl (32 - r)))
    in
    Uop.alu_binary ~op:Ops.Add ~lhs:hi ~rhs:lo
  in
  let key0 = low32 key and key1 = high32 key in
  let ks = [|
    key1;
    Uop.alu_binary ~op:Ops.Xor
      ~lhs:(Uop.alu_binary ~op:Ops.Xor ~lhs:key0 ~rhs:key1)
      ~rhs:(u32c threefry_key_magic);
    key0;
  |] in
  let round xr0 xr1 i =
    let rots = threefry_rotations.(i mod 2) in
    let a, b = Array.fold_left (fun (a, b) r ->
      let sum = Uop.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
      let mixed = Uop.alu_binary ~op:Ops.Xor ~lhs:sum ~rhs:(rot32 b r) in
      sum, mixed) (xr0, xr1) rots
    in
    let k0 = ks.(i mod 3) and k1 = ks.((i + 1) mod 3) in
    Uop.alu_binary ~op:Ops.Add ~lhs:a ~rhs:k0,
    Uop.alu_binary ~op:Ops.Add
      ~lhs:(Uop.alu_binary ~op:Ops.Add ~lhs:b ~rhs:k1)
      ~rhs:(u32c (i + 1))
  in
  let init0 = Uop.alu_binary ~op:Ops.Add ~lhs:(low32 x) ~rhs:ks.(2) in
  let init1 = Uop.alu_binary ~op:Ops.Add ~lhs:(high32 x) ~rhs:ks.(0) in
  let rec loop i (a, b) =
    if i >= 5 then a, b else loop (i + 1) (round a b i)
  in
  let xr0, xr1 = loop 0 (init0, init1) in
  (* Combine as uint64: (xr1 * 2^32) | xr0 *)
  let to_u64 v = Uop.cast ~dtype:u64 ~src:v in
  let hi = Uop.alu_binary ~op:Ops.Mul ~lhs:(to_u64 xr1) ~rhs:c_two32 in
  Uop.alu_binary ~op:Ops.Or ~lhs:hi ~rhs:(to_u64 xr0)

(* Integer division magic *)

(* Hacker's Delight 10-1: find [(m, s)] with [x // d = (x * m) >> s]
   for [0 <= x <= vmax] and [d > 0]. *)
let magicgu vmax d =
  if d <= 0 then invalid_arg "Decomp_op.magicgu: d must be positive";
  let nc = ((vmax + 1) / d) * d - 1 in
  let rec bits_of n acc = if n = 0 then acc else bits_of (n asr 1) (acc + 1) in
  let nbits = bits_of vmax 0 in
  let rec find s =
    if s > 2 * nbits then
      invalid_arg "Decomp_op.magicgu: no solution"
    else
      let pow2s = 1 lsl s in
      if pow2s > nc * (d - 1 - (pow2s - 1) mod d) then
        (pow2s + d - 1 - (pow2s - 1) mod d) / d, s
      else find (s + 1)
  in
  find 0

let int64_to_int_checked n =
  if Int64.compare n (Int64.of_int min_int) < 0
     || Int64.compare n (Int64.of_int max_int) > 0
  then None
  else Some (Int64.to_int n)

let dtype_int_bounds (dt : Dtype.t) =
  match Dtype.min dt, Dtype.max dt with
  | `SInt lo, `SInt hi -> Some (lo, hi)
  | `UInt lo, `UInt hi when Int64.compare hi 0L >= 0 -> Some (lo, hi)
  | _ -> None

let safe_mul_int64 a b =
  let p = Int64.mul a b in
  if Int64.equal a 0L || Int64.equal (Int64.div p a) b then Some p else None

let abs_int64_checked n =
  if Int64.equal n Int64.min_int then None else Some (Int64.abs n)

let next_integer_dtype (dt : Dtype.t) =
  let next_scalar =
    match dt with
    | Dtype.Weakint -> Some Dtype.Uint8
    | Dtype.Int8 -> Some Dtype.Int16
    | Dtype.Int16 -> Some Dtype.Int32
    | Dtype.Int32 -> Some Dtype.Int64
    | Dtype.Int64 -> Some Dtype.Uint64
    | Dtype.Uint8 -> Some Dtype.Uint16
    | Dtype.Uint16 -> Some Dtype.Uint32
    | Dtype.Uint32 -> Some Dtype.Uint64
    | Dtype.Uint64 | Dtype.Uint128 | Dtype.Uint256
    | Dtype.Fp8e4m3 | Dtype.Fp8e5m2
    | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz | Dtype.Float16
    | Dtype.Bfloat16 | Dtype.Float32 | Dtype.Float64 | Dtype.Bool | Dtype.Void
    | Dtype.Index | Dtype.Weakfloat ->
        None
  in
  match next_scalar with
  | Some next -> if Dtype.is_int next then Some next else None
  | None -> None

let shifted_div ~is_unsigned x x_for_mul m s =
  let m_c = Uop.const_like x_for_mul m in
  let s_c = Uop.const_like x_for_mul s in
  let xm = Uop.alu_binary ~op:Ops.Mul ~lhs:x_for_mul ~rhs:m_c in
  let shr_op = Uop.alu_binary ~op:Ops.Shr ~lhs:xm ~rhs:s_c in
  let q =
    if is_unsigned then shr_op
    else
      let one_c = Uop.const_like shr_op 1 in
      let zero_c = Uop.const_like shr_op 0 in
      let cond = Uop.O.(x < Uop.const_like x 0) in
      let adj = Uop.O.where cond one_c zero_c in
      Uop.alu_binary ~op:Ops.Add ~lhs:shr_op ~rhs:adj
  in
  if Dtype.equal (Uop.dtype q) (Uop.dtype x) then q
  else Uop.cast ~src:q ~dtype:(Uop.dtype x)

(* Magic-multiplication division by a positive integer constant. *)
let rec fast_idiv ?(dont_cast = false) ~is_metal ~supports_dtype x d =
  if d <= 0 || is_metal then None
  else
    let dt = Some (Uop.dtype x) in
    match dt with
    | None -> None
    | Some v ->
        let is_int = Dtype.is_int v in
        if not is_int then None
        else
          let is_unsigned =
            Uop.vmin x >= 0 || Dtype.is_unsigned v
          in
          let vmin = Uop.vmin x and vmax = Uop.vmax x in
          if vmin = min_int || vmax = max_int then None
          else if vmin > -d && vmax < d then Some (Uop.const_like x 0)
          else
            match dtype_int_bounds v with
            | None -> None
            | Some (dtype_lo, dtype_hi) ->
                let vmin64 = max (Int64.of_int vmin) dtype_lo in
                let vmax64 = min (Int64.of_int vmax) dtype_hi in
                let abs_vmin = abs_int64_checked vmin64 in
                let vmax_for_magic =
                  match abs_vmin with
                  | None -> None
                  | Some a -> int64_to_int_checked (max vmax64 a)
                in
                match vmax_for_magic with
                | None -> None
                | Some vmax_for_magic ->
                    let m, s = magicgu vmax_for_magic d in
                    let m64 = Int64.of_int m in
                    (match
                       ( safe_mul_int64 m64 vmin64,
                         safe_mul_int64 m64 vmax64 )
                     with
                     | Some lo, Some hi
                       when Int64.compare lo dtype_lo >= 0
                            && Int64.compare hi dtype_hi <= 0 ->
                         Some (shifted_div ~is_unsigned x x m s)
                     | _ ->
                         let pow2_factor = d land (-d) in
                         let try_cast () =
                           if dont_cast then None
                           else
                             match next_integer_dtype v with
                             | Some next_dt
                               when supports_dtype next_dt ->
                                 let next_lo, next_hi =
                                   match dtype_int_bounds next_dt with
                                   | Some bounds -> bounds
                                   | None -> dtype_lo, dtype_hi
                                 in
                                 (match
                                    ( safe_mul_int64 m64 vmin64,
                                      safe_mul_int64 m64 vmax64 )
                                  with
                                  | Some lo, Some hi
                                    when Int64.compare lo next_lo >= 0
                                         && Int64.compare hi next_hi <= 0 ->
                                      let x' =
                                        Uop.cast ~src:x
                                          ~dtype:next_dt
                                      in
                                      Some (shifted_div ~is_unsigned x x' m s)
                                  | _ -> None)
                             | _ -> None
                         in
                         if pow2_factor > 1 then
                           let x' =
                             Uop.alu_binary ~op:Ops.Cdiv ~lhs:x
                               ~rhs:(Uop.const_like x pow2_factor)
                           in
                           match
                             fast_idiv ~dont_cast:true ~is_metal
                               ~supports_dtype x' (d / pow2_factor)
                           with
                           | Some _ as ret -> ret
                           | None -> try_cast ()
                         else try_cast ())

(* Backend capability flags threaded through late-rewrite pattern
   construction. Each [has_*] is [true] iff the backend natively supports
   the corresponding op. *)
type supported_ops = {
  has_exp2 : bool;
  has_log2 : bool;
  has_sin : bool;
  has_sqrt : bool;
  has_neg : bool;
  has_sub : bool;
  has_max : bool;
  has_shl : bool;
  has_shr : bool;
  has_and : bool;
  has_or : bool;
  has_cmplt : bool;
  has_cmpeq : bool;
  has_fdiv : bool;
  has_threefry : bool;
  has_mulacc : bool;
  is_metal : bool;
  supports_dtype : Dtype.t -> bool;
  disable_fast_idiv : bool;
  force_transcendental : bool;
}
(* Reads an integer constant out of a [Uop.t] that is a scalar [Const]. *)
let const_int64_value node =
  match Uop.op node, Uop.arg node with
  | Ops.Const, Uop.Arg.Value v ->
      (match Const.view v with Const.Int n -> Some n | _ -> None)
  | _ -> None

let const_bool_value node =
  match Uop.op node, Uop.arg node with
  | Ops.Const, Uop.Arg.Value v ->
      (match Const.view v with Const.Bool b -> Some b | _ -> None)
  | _ -> None

let is_power_of_two n =
  Int64.compare n 0L > 0 && Int64.equal (Int64.logand n (Int64.sub n 1L)) 0L

let log2_of_power n =
  if not (is_power_of_two n) then None
  else
    let rec loop k n =
      if Int64.equal n 1L then k
      else loop (k + 1) (Int64.shift_right_logical n 1)
    in
    Some (loop 0 n)

(* Bool-negation predicate: [n] is structurally [x != true]. *)
let as_logical_not n =
  match Uop.op n with
  | Ops.Cmpne ->
      let s = Uop.src n in
      if Array.length s = 2 then
        match const_int64_value s.(1), const_bool_value s.(1) with
        | Some 1L, _ | _, Some true -> Some s.(0)
        | _ -> None
      else None
  | _ -> None

let is_signed_int_node n =
  let dt = Uop.dtype n in
  Dtype.is_int dt && not (Dtype.is_unsigned dt)

let signed_int_dtype n =
  let dt = Uop.dtype n in
  if Dtype.is_int dt && not (Dtype.is_unsigned dt) then Some dt else None

let const_int64_for dt n = Uop.const (Const.int64 dt n)

let const_int64_value_signed n =
  let dt = Uop.dtype n in
  if Dtype.is_int dt && not (Dtype.is_unsigned dt) then
    (match const_int64_value n with
     | Some v -> Some (dt, v)
     | None -> None)
  else None

let is_neg_one node =
  match Uop.op node, Uop.arg node with
  | Ops.Const, Uop.Arg.Value v -> (
      match Const.view v with
      | Const.Int n -> Int64.equal n (-1L)
      | Const.Float f -> Float.equal f (-1.0)
      | _ -> false)
  | _ -> false

let as_mul_neg_one n =
  match Uop.op n, Uop.src n with
  | Ops.Mul, [| a; b |] ->
      if is_neg_one b then Some a else if is_neg_one a then Some b else None
  | _ -> None

let as_mul_const_signed n =
  match Uop.op n, Uop.src n with
  | Ops.Mul, [| a; b |] ->
      (match const_int64_value_signed b with
       | Some (dt, v) -> Some (a, dt, v)
       | None ->
           (match const_int64_value_signed a with
            | Some (dt, v) -> Some (b, dt, v)
            | None -> None))
  | _ -> None

(* THREEFRY x key -> software implementation. *)
let rule_threefry (ops : supported_ops) node =
  if ops.has_threefry then None
  else match Uop.op node with
    | Ops.Threefry ->
        let s = Uop.src node in
        if Array.length s = 2 && Dtype.equal (Uop.dtype node) Dtype.uint64 then
          Some (threefry2x32 s.(0) s.(1))
        else None
    | _ -> None

let floor_same_as_trunc a b =
  (Uop.vmin a >= 0 && Uop.vmin b > 0)
  || (Uop.vmax a <= 0 && Uop.vmax b < 0)

let floor_fixup_condition a b r =
  let zero_a = Uop.const_like a 0 in
  let zero_b = Uop.const_like b 0 in
  let zero_r = Uop.const_like r 0 in
  let has_remainder = Uop.O.(ne r zero_r) in
  let sign_diff = Uop.O.(ne (a < zero_a) (b < zero_b)) in
  Uop.alu_binary ~op:Ops.And ~lhs:has_remainder ~rhs:sign_diff

(* FLOORDIV a b -> CDIV a b, with a one-step correction when floor and
   truncating division differ. *)
let rule_floordiv_to_idiv _ops node =
  match Uop.op node, Uop.dtype node, Uop.src node with
  | Ops.Floordiv, dt, [| a; b |] when Dtype.is_int dt ->
      let q = Uop.alu_binary ~op:Ops.Cdiv ~lhs:a ~rhs:b in
      if floor_same_as_trunc a b then Some q
      else
        let fixup =
          Uop.cast ~src:(floor_fixup_condition a b (Uop.alu_binary ~op:Ops.Cmod ~lhs:a ~rhs:b))
            ~dtype:(Uop.dtype q)
        in
        Some (Uop.alu_binary ~op:Ops.Sub ~lhs:q ~rhs:fixup)
  | _ -> None

(* FLOORMOD by 2^k -> x & (2^k - 1). This is correct for any signed
   input under two's-complement floor-mod semantics. *)
let rule_floormod_and (ops : supported_ops) node =
  if not ops.has_and then None
  else match Uop.op node, Uop.dtype node, Uop.src node with
    | Ops.Floormod, dt, [| x; c |] when Dtype.is_int dt ->
        (match const_int64_value c with
         | Some cv when is_power_of_two cv ->
             Some
               (Uop.alu_binary ~op:Ops.And ~lhs:x
                  ~rhs:(Uop.const (Const.int64 dt (Int64.sub cv 1L))))
         | _ -> None)
    | _ -> None

(* FLOORMOD a b -> CMOD a b, with a correction toward [b] when the
   truncating remainder has the wrong sign. *)
let rule_floormod_to_mod _ops node =
  match Uop.op node, Uop.dtype node, Uop.src node with
  | Ops.Floormod, dt, [| a; b |] when Dtype.is_int dt ->
      let r = Uop.alu_binary ~op:Ops.Cmod ~lhs:a ~rhs:b in
      if floor_same_as_trunc a b then Some r
      else
        let fixup =
          Uop.alu_ternary ~op:Ops.Where
            ~a:(floor_fixup_condition a b r) ~b
            ~c:(Uop.const_like b 0)
        in
        Some (Uop.alu_binary ~op:Ops.Add ~lhs:r ~rhs:fixup)
  | _ -> None

(* MAX x y -> where(x < y, y, x). *)
let rule_max (ops : supported_ops) node =
  if ops.has_max || not ops.has_cmplt then None
  else match Uop.op node with
    | Ops.Max ->
        let s = Uop.src node in
        if Array.length s = 2 then
          let x = s.(0) and y = s.(1) in
          Some (Uop.O.where (Uop.O.(x < y)) y x)
        else None
    | _ -> None

(* Early simplifying rewrites that lower floor div/mod before dtype and
   late codegen decompositions. *)
let get_simplifying_rewrite_patterns (ops : supported_ops) (node : Uop.t) :
    Uop.t option =
  let rules =
    [ rule_floordiv_to_idiv; rule_floormod_and; rule_floormod_to_mod;
      rule_threefry; rule_max ]
  in
  let rec try_rules = function
    | [] -> None
    | r :: rest ->
        (match r ops node with
         | Some _ as v -> v
         | None -> try_rules rest)
  in
  try_rules rules

(* De Morgan: ¬x ∧ ¬y -> ¬(x ∨ y) on booleans. *)
let rule_de_morgan (ops : supported_ops) node =
  if not ops.has_or then None
  else match Uop.op node with
    | Ops.And ->
        let s = Uop.src node in
        (match Uop.dtype node, s with
         | dt, [| a; b |] when Dtype.is_bool dt ->
             (match as_logical_not a, as_logical_not b with
              | Some x, Some y ->
                  let or_ = Uop.alu_binary ~op:Ops.Or ~lhs:x ~rhs:y in
                  Some (Uop.O.not_ or_)
              | _ -> None)
         | _ -> None)
    | _ -> None

(* x * 2^k -> x << k on integer x. *)
let rule_mul_to_shl (ops : supported_ops) node =
  if not ops.has_shl then None
  else match Uop.op node with
    | Ops.Mul ->
        let s = Uop.src node in
        let try_shift base c_node =
          match const_int64_value c_node with
          | Some cv ->
              (match log2_of_power cv with
               | Some n when n > 0 ->
                   (match Uop.dtype node with
                    | dt when Dtype.is_int dt ->
                        Some (Uop.alu_binary ~op:Ops.Shl ~lhs:base
                                ~rhs:(Uop.const (Const.int dt n)))
                    | _ -> None)
               | Some _ | None -> None)
          | None -> None
        in
        (match s with
         | [| x; c |] ->
             (match try_shift x c with
              | Some _ as r -> r
              | None -> try_shift c x)
         | _ -> None)
    | _ -> None

(* Unsigned x / 2^k -> x >> k. *)
let rule_udiv_to_shr (ops : supported_ops) node =
  if not ops.has_shr then None
  else match Uop.op node with
    | Ops.Cdiv ->
        let s = Uop.src node in
        (match Uop.dtype node, s with
         | dt, [| x; c |]
           when Dtype.is_int dt && Dtype.is_unsigned dt ->
             (match const_int64_value c with
              | Some cv ->
                  (match log2_of_power cv with
                   | Some n when n > 0 ->
                       Some (Uop.alu_binary ~op:Ops.Shr ~lhs:x
                               ~rhs:(Uop.const (Const.int dt n)))
                   | Some _ | None -> None)
              | None -> None)
         | _ -> None)
    | _ -> None

(* Signed x / 2^k -> (x + fixup) >> k, where fixup = (x<0 ? 2^k-1 : 0). *)
let rule_sdiv_to_shr (ops : supported_ops) node =
  if not ops.has_shr then None
  else match Uop.op node with
    | Ops.Cdiv ->
        let s = Uop.src node in
        (match Uop.dtype node, s with
         | dt, [| x; c |]
           when Dtype.is_int dt && not (Dtype.is_unsigned dt) ->
             (match const_int64_value c with
              | Some cv ->
                  (match log2_of_power cv with
                   | Some n when n > 0 ->
                       let lt_zero =
                         Uop.alu_binary ~op:Ops.Cmplt ~lhs:x
                           ~rhs:(Uop.const (Const.int64 dt 0L))
                       in
                       let cond =
                         if Uop.vmin lt_zero = Uop.vmax lt_zero then
                           Uop.const_like lt_zero (Uop.vmin lt_zero)
                         else lt_zero
                       in
                       let correction =
                         Uop.alu_ternary ~op:Ops.Where ~a:cond
                           ~b:(Uop.const (Const.int64 dt (Int64.sub cv 1L)))
                           ~c:(Uop.const (Const.int64 dt 0L))
                       in
                       Some (Uop.alu_binary ~op:Ops.Shr
                               ~lhs:(Uop.alu_binary ~op:Ops.Add ~lhs:x ~rhs:correction)
                               ~rhs:(Uop.const (Const.int dt n)))
                   | Some _ | None -> None)
              | None -> None)
         | _ -> None)
    | _ -> None

(* x / d (constant d > 0, non-power-of-two) -> magic multiply-shift. *)
let rule_fast_idiv_late (ops : supported_ops) node =
  if not ops.has_shr || ops.disable_fast_idiv then None
  else match Uop.op node with
    | Ops.Cdiv ->
        let s = Uop.src node in
        (match Uop.dtype node, s with
         | dt, [| x; d |]
           when Dtype.is_int dt
                && (Uop.vmin x >= 0 || Dtype.is_unsigned dt) ->
             (match const_int64_value d with
              | Some dv when Int64.compare dv 0L > 0 ->
                  (match log2_of_power dv with
                   | Some _ -> None
                   | None ->
                       (match int64_to_int_checked dv with
                        | Some d ->
                            fast_idiv ~is_metal:ops.is_metal
                              ~supports_dtype:ops.supports_dtype x d
                        | None -> None))
              | _ -> None)
         | _ -> None)
    | _ -> None

(* x % d -> x - d * (x // d). *)
let rule_mod_from_idiv (ops : supported_ops) node =
  if not ops.has_shr || ops.disable_fast_idiv then None
  else match Uop.op node with
    | Ops.Cmod ->
        let s = Uop.src node in
        (match Uop.dtype node, s with
         | dt, [| x; d |]
           when Dtype.is_int dt
                && (Uop.vmin x >= 0 || Dtype.is_unsigned dt) ->
             (match const_int64_value d with
              | Some dv when ops.has_and && is_power_of_two dv -> None
              | _ ->
                  let q = Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:d in
                  let dq = Uop.alu_binary ~op:Ops.Mul ~lhs:d ~rhs:q in
                  Some (Uop.alu_binary ~op:Ops.Sub ~lhs:x ~rhs:dq))
         | _ -> None)
    | _ -> None

(* x * -1 -> neg x. *)
let rule_mul_neg_one (ops : supported_ops) node =
  if not ops.has_neg then None
  else
    match as_mul_neg_one node with
    | Some x -> Some (Uop.alu_unary ~op:Ops.Neg ~src:x)
    | None -> None

(* x + neg y -> x - y. *)
let rule_add_neg_to_sub (ops : supported_ops) node =
  if not ops.has_neg || not ops.has_sub then None
  else match Uop.op node with
    | Ops.Add ->
        let s = Uop.src node in
        (match s with
         | [| a; b |] ->
             let try_neg x n =
               if Uop.op n = Ops.Neg then
                 let sn = Uop.src n in
                 if Array.length sn = 1 then
                   Some (Uop.alu_binary ~op:Ops.Sub ~lhs:x ~rhs:sn.(0))
                 else None
               else None
             in
             (match try_neg a b with Some _ as r -> r | None -> try_neg b a)
         | _ -> None)
    | _ -> None

(* Late signed-CMPLT canonicalizations. Simplex expects equalities in a
   particular shape, so these mirror tinygrad's late-only comparison rules. *)
let rule_not_cmplt_const (ops : supported_ops) node =
  if not ops.has_cmplt then None
  else
    match as_logical_not node with
    | Some cmp when Uop.op cmp = Ops.Cmplt ->
        (match Uop.src cmp with
         | [| x; c |]
           when is_signed_int_node x
                && Option.is_some (const_int64_value_signed c) ->
             (match const_int64_value_signed c with
              | Some (dt, cv) ->
                  Some
                    (Uop.alu_binary ~op:Ops.Cmplt
                       ~lhs:(const_int64_for dt (Int64.sub cv 1L))
                       ~rhs:x)
              | None -> None)
         | [| c; x |]
           when is_signed_int_node x
                && Option.is_some (const_int64_value_signed c) ->
             (match const_int64_value_signed c with
              | Some (dt, cv) ->
                  Some
                    (Uop.alu_binary ~op:Ops.Cmplt ~lhs:x
                       ~rhs:(const_int64_for dt (Int64.add cv 1L)))
              | None -> None)
         | _ -> None)
    | _ -> None

let rule_negated_signed_cmplt (ops : supported_ops) node =
  if not ops.has_cmplt then None
  else match Uop.op node, Uop.src node with
    | Ops.Cmplt, [| lhs; rhs |] ->
        (match as_mul_neg_one lhs with
         | Some x when is_signed_int_node x ->
             (match as_mul_const_signed rhs with
              | Some (y, dt, cv) when is_signed_int_node y ->
                  Some
                    (Uop.alu_binary ~op:Ops.Cmplt
                       ~lhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:y
                               ~rhs:(const_int64_for dt (Int64.neg cv)))
                       ~rhs:x)
              | _ ->
                  (match const_int64_value_signed rhs with
                   | Some (dt, cv) ->
                       Some
                         (Uop.alu_binary ~op:Ops.Cmplt
                            ~lhs:(const_int64_for dt (Int64.neg cv))
                            ~rhs:x)
                   | None -> None))
         | _ -> None)
    | _ -> None

let rule_bounded_cmplt_to_eq (ops : supported_ops) node =
  if not ops.has_cmplt then None
  else match Uop.op node, Uop.src node with
    | Ops.And, [| a; b |] ->
        let try_pair left right =
          match Uop.op left, Uop.src left, Uop.op right, Uop.src right with
          | Ops.Cmplt, [| c1; x1 |], Ops.Cmplt, [| x2; c2 |]
            when Uop.equal x1 x2 && is_signed_int_node x1 ->
              (match const_int64_value c1, const_int64_value c2,
                     signed_int_dtype x1 with
               | Some lo, Some hi, Some dt
                 when Int64.equal (Int64.add lo 1L) (Int64.sub hi 1L) ->
                   Some
                     (Uop.alu_binary ~op:Ops.Cmpeq ~lhs:x1
                        ~rhs:(const_int64_for dt (Int64.add lo 1L)))
               | _ -> None)
          | _ -> None
        in
        (match try_pair a b with Some _ as r -> r | None -> try_pair b a)
    | _ -> None

(* ¬(x ≠ y) -> x = y. Simplex expects the inequality form, so CMPEQ is
   rebuilt only at the late rewrite step. *)
let rule_not_ne_to_eq (ops : supported_ops) node =
  if not ops.has_cmpeq then None
  else match Uop.op node with
    | Ops.Cmpne ->
        let s = Uop.src node in
        (match s with
         | [| a; b |] when Uop.op a = Ops.Cmpne ->
             (match const_int64_value b, const_bool_value b with
              | Some n, _ when Int64.equal n 1L ->
                  let sa = Uop.src a in
                  if Array.length sa = 2 then
                    Some (Uop.alu_binary ~op:Ops.Cmpeq ~lhs:sa.(0) ~rhs:sa.(1))
                  else None
              | _, Some true ->
                  let sa = Uop.src a in
                  if Array.length sa = 2 then
                    Some (Uop.alu_binary ~op:Ops.Cmpeq ~lhs:sa.(0) ~rhs:sa.(1))
                  else None
              | _ -> None)
         | _ -> None)
    | _ -> None

(* a * b + c -> mulacc a b c; also (x << n) + c -> mulacc x (2^n) c. *)
let rule_mulacc_fuse (ops : supported_ops) node =
  if not ops.has_mulacc then None
  else match Uop.op node with
    | Ops.Add ->
        let s = Uop.src node in
        let try_fuse l r =
          match Uop.op l with
          | Ops.Mul ->
              let sl = Uop.src l in
              if Array.length sl = 2 then
                Some (Uop.alu_ternary ~op:Ops.Mulacc ~a:sl.(0) ~b:sl.(1) ~c:r)
              else None
          | Ops.Shl ->
              let sl = Uop.src l in
              if Array.length sl = 2 && ops.has_shl then
                match const_int64_value sl.(1) with
                | Some n ->
                    (match int64_to_int_checked n with
                     | Some shift ->
                         let dt = Uop.dtype l in
                         let two_n = Int64.shift_left 1L shift in
                         let two_c = Uop.const (Const.int64 dt two_n) in
                         Some (Uop.alu_ternary ~op:Ops.Mulacc
                                 ~a:sl.(0) ~b:two_c ~c:r)
                     | None -> None)
                | None -> None
              else None
          | _ -> None
        in
        (match s with
         | [| l; r |] ->
             (match try_fuse l r with
              | Some _ as res -> res
              | None -> try_fuse r l)
         | _ -> None)
    | _ -> None

(* 1 / x via FDIV. *)
let rule_recip_to_fdiv (ops : supported_ops) node =
  if not ops.has_fdiv then None
  else match Uop.op node with
    | Ops.Reciprocal ->
        let s = Uop.src node in
        (match s with
         | [| x |] ->
             let v = Uop.dtype x in
             Some (Uop.alu_binary ~op:Ops.Fdiv ~lhs:(const_float_v v 1.0) ~rhs:x)
         | _ -> None)
    | _ -> None

(* a * (1 / b) -> a / b. *)
let rule_mul_recip_to_fdiv (ops : supported_ops) node =
  if not ops.has_fdiv then None
  else match Uop.op node with
    | Ops.Mul ->
        let s = Uop.src node in
        let try_fdiv a d =
          match Uop.dtype node with
          | dt when Dtype.is_float dt && Uop.op d = Ops.Fdiv ->
             let sd = Uop.src d in
             if Array.length sd = 2 then
               (match Uop.op sd.(0), Uop.arg sd.(0) with
                | Ops.Const, Uop.Arg.Value v ->
                    (match Const.view v with
                     | Const.Float 1.0 ->
                         Some (Uop.alu_binary ~op:Ops.Fdiv ~lhs:a ~rhs:sd.(1))
                     | _ -> None)
                | _ -> None)
             else None
          | _ -> None
        in
        (match s with
         | [| a; d |] ->
             (match try_fdiv a d with Some _ as r -> r | None -> try_fdiv d a)
         | _ -> None)
    | _ -> None

(* Late-rewrite driver: threads the backend capability flags through an
   ordered list of point rewrites. Matches the
   [get_late_rewrite_patterns] block. *)
let get_late_rewrite_patterns (ops : supported_ops) (node : Uop.t) : Uop.t option =
  let rules =
    [ rule_de_morgan;
      rule_mul_to_shl; rule_udiv_to_shr; rule_sdiv_to_shr;
      rule_fast_idiv_late; rule_mod_from_idiv;
      rule_mul_neg_one; rule_add_neg_to_sub;
      rule_not_cmplt_const; rule_negated_signed_cmplt;
      rule_bounded_cmplt_to_eq;
      rule_not_ne_to_eq; rule_mulacc_fuse;
      rule_recip_to_fdiv; rule_mul_recip_to_fdiv; ]
  in
  let rec try_rules = function
    | [] -> None
    | r :: rest ->
        (match r ops node with
         | Some _ as v -> v
         | None -> try_rules rest)
  in
  try_rules rules
