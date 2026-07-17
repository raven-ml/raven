(* Smoke tests for codegen/decomp modules. *)

open Windtrap
open Tolk
open Tolk_uop

(* magicgu round-trip: [x // d] = [(x * m) >> s]. *)
let magicgu_correct () =
  let m, s = Decomp_op.magicgu 1000 7 in
  let ok =
    List.for_all (fun x ->
      x / 7 = (x * m) asr s)
      [ 0; 1; 2; 7; 8; 14; 100; 999; 1000 ]
  in
  is_true ~msg:"magicgu divides" ok

(* threefry2x32: at least terminates and produces a uint64 uop. *)
let threefry_produces_uint64 () =
  let x = Uop.const (Const.int64 Dtype.uint64 42L) in
  let key = Uop.const (Const.int64 Dtype.uint64 99L) in
  let r = Decomp_op.threefry2x32 x key in
  is_true ~msg:"dtype is uint64"
    (Dtype.equal (Uop.dtype r) Dtype.uint64)

(* Long-to-int decomposition: construct an op whose result is int64, tag
   the root as half "0" or "1", apply [pm_long_decomp], and check that
   the result is a narrow (int32) uop free of int64 sub-nodes. *)
let contains_long (u : Uop.t) =
  let nodes = Uop.toposort u in
  List.exists (fun n ->
    let d = Uop.dtype n in
    Dtype.equal d Dtype.int64 || Dtype.equal d Dtype.uint64)
    nodes

let contains_op op (u : Uop.t) =
  List.exists (fun n -> Uop.op n = op) (Uop.toposort u)

let const_int64_value node =
  match Uop.op node, Uop.arg node with
  | Ops.Const, Uop.Arg.Value v ->
      (match Const.view v with Const.Int n -> Some n | _ -> None)
  | _ -> None

let const_float_value node =
  match Uop.op node, Uop.arg node with
  | Ops.Const, Uop.Arg.Value v ->
      (match Const.view v with Const.Float x -> Some x | _ -> None)
  | _ -> None

let supported_ops ?(has_and = true) ?(has_max = true) ?(has_cmplt = true)
    ?(has_threefry = true) ?(disable_fast_idiv = true)
    ?(has_shr = true) ?(has_sqrt = true) ?(is_metal = false)
    ?(supports_dtype = fun _ -> true) () :
    Decomp_op.supported_ops =
  { has_shl = true; has_shr; has_and; has_or = true;
    has_max; has_cmplt; has_cmpeq = true;
    has_neg = true; has_sub = true; has_mulacc = false;
    has_fdiv = false; has_threefry; disable_fast_idiv;
    is_metal; supports_dtype;
    has_exp2 = true; has_log2 = true; has_sin = true; has_sqrt;
    force_transcendental = false }

(* Sqrt lowers through xpow, which produces a Where at the root for the
   zero-zero fixup. *)
let sqrt_decomposition_builds_where () =
  let base = Uop.const_float 2.0 in
  let sqrt = Uop.alu_unary ~op:Ops.Sqrt ~src:base in
  match
    Decomp_transcendental.get_transcendental_patterns
      (supported_ops ~has_sqrt:false ()) sqrt
  with
  | Some r -> is_true ~msg:"outer op is Where" (Uop.op r = Ops.Where)
  | None -> is_true ~msg:"sqrt decomposition fired" false

let log2_denormal_scale_uses_float_power () =
  let x = Uop.const_float 1.0 in
  let log2 = Uop.alu_unary ~op:Ops.Log2 ~src:x in
  match
    Decomp_transcendental.get_transcendental_patterns
      { (supported_ops ()) with has_log2 = false } log2
  with
  | Some r ->
      let scale = 2.0 ** 64.0 in
      let has_scale =
        List.exists
          (fun n ->
            match const_float_value n with
            | Some f -> Float.equal f scale
            | None -> false)
          (Uop.toposort r)
      in
      is_true ~msg:"f32 log2 denormal scale is 2.0 ** 64.0" has_scale
  | None -> is_true ~msg:"log2 decomposition fired" false

let sin_f16_cody_waite_casts_quadrant_to_f32 () =
  let f16 = Dtype.float16 in
  let f32_dt = Dtype.float32 in
  let i16_dt = Dtype.int16 in
  let x = Uop.const (Const.float f16 1.0) in
  let sin = Uop.alu_unary ~op:Ops.Sin ~src:x in
  match
    Decomp_transcendental.get_transcendental_patterns
      { (supported_ops ()) with has_sin = false } sin
  with
  | Some r ->
      let has_quadrant_cast =
        List.exists
          (fun n ->
            Uop.op n = Ops.Cast
            && Dtype.equal (Uop.dtype n) f32_dt
            &&
            let src = Uop.src n in
            Array.length src = 1 && Dtype.equal (Uop.dtype src.(0)) i16_dt)
          (Uop.toposort r)
      in
      is_true ~msg:"f16 Cody-Waite casts int quadrant directly to f32"
        has_quadrant_cast
  | None -> is_true ~msg:"sin decomposition fired" false

let decomposes_free_of_long u =
  let tagged = Uop.with_tag "0" u in
  let rewritten =
    Uop.graph_rewrite
      (Upat.Pattern_matcher.rewrite Decomp_dtype.pm_long_decomp) tagged
  in
  (* After decomposition we should have a narrow result and no long nodes
     remaining in the reachable graph. *)
  let narrow_result =
    let d = Uop.dtype rewritten in
    Dtype.equal d Dtype.int32 || Dtype.equal d Dtype.uint32
  in
  narrow_result && not (contains_long rewritten)

let rewrite_long_half tag u =
  Uop.graph_rewrite
    (Upat.Pattern_matcher.rewrite Decomp_dtype.pm_long_decomp)
    (Uop.with_tag tag u)

let mul_long_decomposes () =
  let a = Uop.const (Const.int64 Dtype.int64 0x1234567890abcdefL) in
  let b = Uop.const (Const.int64 Dtype.int64 0xfedcba9876543210L) in
  let prod = Uop.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b in
  is_true ~msg:"MUL of int64 lowers to int32 without int64 residue"
    (decomposes_free_of_long prod)

let idiv_long_decomposes () =
  let a = Uop.const (Const.int64 Dtype.int64 1_000_000_000_000L) in
  let b = Uop.const (Const.int64 Dtype.int64 7L) in
  let q = Uop.alu_binary ~op:Ops.Cdiv ~lhs:a ~rhs:b in
  is_true ~msg:"IDIV of int64 lowers to int32 without int64 residue"
    (decomposes_free_of_long q)

let mod_long_decomposes () =
  let a = Uop.const (Const.int64 Dtype.int64 1_000_000_000_000L) in
  let b = Uop.const (Const.int64 Dtype.int64 7L) in
  let r = Uop.alu_binary ~op:Ops.Cmod ~lhs:a ~rhs:b in
  is_true ~msg:"MOD of int64 lowers to int32 without int64 residue"
    (decomposes_free_of_long r)

let cast_float_to_long_decomposes () =
  let f = Uop.const (Const.float Dtype.float64 1.5e10) in
  let casted = Uop.cast ~src:f ~dtype:Dtype.int64 in
  is_true ~msg:"CAST float64->int64 lowers to int32 without int64 residue"
    (decomposes_free_of_long casted)

let cast_long_to_int_decomposes () =
  let a = Uop.const (Const.int64 Dtype.int64 42L) in
  let casted = Uop.cast ~src:a ~dtype:Dtype.int32 in
  (* Result is already int32; just check no int64 residue. *)
  let rewritten =
    Uop.graph_rewrite
      (Upat.Pattern_matcher.rewrite Decomp_dtype.pm_long_decomp) casted
  in
  is_true ~msg:"CAST int64->int32 has no int64 residue"
    (not (contains_long rewritten))

let bitcast_long_to_long_decomposes () =
  let a =
    Uop.variable ~name:"a" ~min_val:0 ~max_val:100 ~dtype:Dtype.int64 ()
  in
  let b = Uop.bitcast ~src:a ~dtype:Dtype.uint64 in
  let half tag =
    Upat.Pattern_matcher.rewrite Decomp_dtype.pm_long_decomp
      (Uop.with_tag tag b)
  in
  match half "0", half "1" with
  | Some lo, Some hi ->
      is_true ~msg:"BITCAST int64->uint64 rewrites both narrow halves"
        (Dtype.equal (Uop.dtype lo) Dtype.uint32
         && Dtype.equal (Uop.dtype hi) Dtype.uint32)
  | _ -> is_true ~msg:"BITCAST int64->uint64 rule fired" false

let long_const_halves_are_truncated_to_int32 () =
  let c = Uop.const (Const.int64 Dtype.int64 (-2147483648L)) in
  let lo = rewrite_long_half "0" c in
  let hi = rewrite_long_half "1" c in
  let const_int u =
    match Uop.op u, Uop.arg u with
    | Ops.Const, Uop.Arg.Value v ->
        (match Const.view v with Const.Int n -> Some n | _ -> None)
    | _ -> None
  in
  is_true ~msg:"low and high halves are signed int32 truncated"
    (const_int lo = Some (-2147483648L)
     && const_int hi = Some (-1L)
     && Dtype.equal (Uop.dtype lo) Dtype.int32
     && Dtype.equal (Uop.dtype hi) Dtype.int32)

let untagged_long_const_is_low_half () =
  let c = Uop.const (Const.int64 Dtype.int64 0x0000000100000002L) in
  let rewritten =
    Uop.graph_rewrite
      (Upat.Pattern_matcher.rewrite Decomp_dtype.pm_long_decomp) c
  in
  is_true ~msg:"untagged long CONST lowers to low half"
    (Dtype.equal (Uop.dtype rewritten) Dtype.int32
     && const_int64_value rewritten = Some 2L)

let unbounded_long_param_keeps_unbounded_size () =
  (* A param with no shape stands in for an unbounded buffer: the dtype
     narrows to int32 in both the node and its argument, and the missing
     size is left untouched. *)
  let buf = Uop.param ~slot:0 ~dtype:Dtype.int64 ~addrspace:Dtype.Global () in
  match Upat.Pattern_matcher.rewrite Decomp_dtype.pm_long_decomp buf with
  | Some rewritten ->
      let arg_narrowed =
        match Uop.as_param rewritten with
        | Some { param; _ } -> Dtype.equal param.dtype Dtype.int32
        | None -> false
      in
      is_true ~msg:"unbounded long param narrows to int32 in node and arg"
        (Dtype.equal (Uop.dtype rewritten) Dtype.int32 && arg_narrowed)
  | None -> is_true ~msg:"long param define rewrites" false

let untagged_long_index_is_not_rewritten () =
  let buf =
    Uop.param ~slot:0 ~dtype:Dtype.int64 ~shape:(Uop.const_int 8)
      ~addrspace:Dtype.Global ()
  in
  let idx = Uop.index ~ptr:buf ~idxs:[ Uop.const_int 3 ] () in
  match Upat.Pattern_matcher.rewrite Decomp_dtype.pm_long_decomp idx with
  | None -> ()
  | Some _ -> is_true ~msg:"untagged INDEX should not rewrite" false

let tagged_long_index_reindexes_before_define () =
  let buf =
    Uop.param ~slot:0 ~dtype:Dtype.int64 ~shape:(Uop.const_int 8)
      ~addrspace:Dtype.Global ()
  in
  let idx = Uop.index ~ptr:buf ~idxs:[ Uop.const_int 3 ] () in
  match
    Upat.Pattern_matcher.rewrite Decomp_dtype.pm_long_decomp
      (Uop.with_tag "1" idx)
  with
  | Some rewritten ->
      (match Uop.as_index rewritten with
       | Some { idxs = [ i ]; _ } ->
           is_true ~msg:"tagged INDEX is narrowed and offset to high half"
             (Dtype.equal (Uop.dtype rewritten) Dtype.int32
              && Uop.vmin i = 7 && Uop.vmax i = 7)
       | _ -> is_true ~msg:"rewritten node remains INDEX" false)
  | None -> is_true ~msg:"tagged INDEX rule fired" false

let tagged_long_index_preserves_multi_index_tail () =
  let buf =
    Uop.param ~slot:0 ~dtype:Dtype.int64 ~shape:(Uop.const_int 8)
      ~addrspace:Dtype.Global ()
  in
  let idx =
    Uop.index ~ptr:buf ~idxs:[ Uop.const_int 3; Uop.const_int 5 ] ()
  in
  match
    Upat.Pattern_matcher.rewrite Decomp_dtype.pm_long_decomp
      (Uop.with_tag "1" idx)
  with
  | Some rewritten ->
      (match Uop.as_index rewritten with
       | Some { idxs = [ i; tail ]; _ } ->
           is_true ~msg:"tagged INDEX rewrites first index and keeps tail"
             (Uop.vmin i = 7 && Uop.vmax i = 7
              && Uop.vmin tail = 5 && Uop.vmax tail = 5)
       | _ -> is_true ~msg:"rewritten node keeps two indexes" false)
  | None -> is_true ~msg:"tagged INDEX rule fired" false

let float_to_long_high_half_uses_reciprocal () =
  let f = Uop.const (Const.float Dtype.float64 1.5e10) in
  let casted = Uop.cast ~src:f ~dtype:Dtype.int64 in
  let hi = rewrite_long_half "1" casted in
  let has_pow32_divisor =
    List.exists
      (fun n ->
         Uop.op n = Ops.Mul
         &&
         match Uop.src n with
         | [| _; rhs |] when Uop.op rhs = Ops.Reciprocal -> (
             match Uop.src rhs with
             | [| divisor |] -> const_float_value divisor = Some 4294967296.0
             | _ -> false)
         | _ -> false)
      (Uop.toposort hi)
  in
  is_true ~msg:"float->long high half divides by 2^32"
    (has_pow32_divisor && not (contains_long hi))

let long_variable_shl_uses_native_shift () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:100 ~dtype:Dtype.int64 ()
  in
  let s =
    Uop.variable ~name:"s" ~min_val:0 ~max_val:31 ~dtype:Dtype.int32 ()
  in
  let shifted = Uop.alu_binary ~op:Ops.Shl ~lhs:x ~rhs:s in
  let lo = rewrite_long_half "0" shifted in
  is_true ~msg:"variable long SHL lowers to narrow SHL"
    (contains_op Ops.Shl lo && not (contains_long lo))

let long_variable_shr_uses_native_shift () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:100 ~dtype:Dtype.int64 ()
  in
  let s =
    Uop.variable ~name:"s" ~min_val:0 ~max_val:31 ~dtype:Dtype.int32 ()
  in
  let shifted = Uop.alu_binary ~op:Ops.Shr ~lhs:x ~rhs:s in
  let lo = rewrite_long_half "0" shifted in
  is_true ~msg:"variable long SHR lowers to narrow SHR"
    (contains_op Ops.Shr lo && not (contains_long lo))

let floordiv_lowering_uses_cdiv_for_same_sign () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:100 ~dtype:Dtype.int32 ()
  in
  let d = Uop.const (Const.int Dtype.int32 3) in
  let q = Uop.alu_binary ~op:Ops.Floordiv ~lhs:x ~rhs:d in
  match Decomp_op.get_simplifying_rewrite_patterns (supported_ops ()) q with
  | Some r -> is_true ~msg:"rewrites to Cdiv" (Uop.op r = Ops.Cdiv)
  | None -> is_true ~msg:"rule fired" false

let floordiv_lowering_corrects_mixed_sign () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let d = Uop.const (Const.int Dtype.int32 3) in
  let q = Uop.alu_binary ~op:Ops.Floordiv ~lhs:x ~rhs:d in
  match Decomp_op.get_simplifying_rewrite_patterns (supported_ops ()) q with
  | Some r -> is_true ~msg:"rewrites to corrected Sub" (Uop.op r = Ops.Sub)
  | None -> is_true ~msg:"rule fired" false

let floormod_power_of_two_uses_and_for_negative_input () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let d = Uop.const (Const.int Dtype.int32 4) in
  let r = Uop.alu_binary ~op:Ops.Floormod ~lhs:x ~rhs:d in
  match Decomp_op.get_simplifying_rewrite_patterns (supported_ops ()) r with
  | Some r -> is_true ~msg:"rewrites to And" (Uop.op r = Ops.And)
  | None -> is_true ~msg:"rule fired" false

let late_cmod_power_of_two_rejects_negative_signed_input () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let d = Uop.const (Const.int Dtype.int32 4) in
  let r = Uop.alu_binary ~op:Ops.Cmod ~lhs:x ~rhs:d in
  let got = Decomp_op.get_late_rewrite_patterns (supported_ops ()) r in
  is_true ~msg:"negative signed Cmod is not rewritten to And" (got = None)

let fast_idiv_small_range_folds_to_zero () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:6 ~dtype:Dtype.int32 ()
  in
  let d = Uop.const (Const.int Dtype.int32 7) in
  let q = Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:d in
  match
    Decomp_op.get_late_rewrite_patterns
      (supported_ops ~disable_fast_idiv:false ()) q
  with
  | Some r ->
      is_true ~msg:"small range division rewrites to zero"
        (Uop.const_int_value r = Some 0)
  | None -> is_true ~msg:"small range fast idiv rule fired" false

let fast_idiv_rejects_non_native_divisor () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:100 ~dtype:Dtype.int64 ()
  in
  let d = Uop.const (Const.int64 Dtype.int64 Int64.max_int) in
  let q = Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:d in
  let got =
    Decomp_op.get_late_rewrite_patterns
      (supported_ops ~disable_fast_idiv:false ()) q
  in
  is_true ~msg:"fast idiv rejects divisors outside native int" (got = None)

let fast_idiv_recursion_uses_cdiv () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:90_000 ~dtype:Dtype.int32
      ()
  in
  let d = Uop.const (Const.int Dtype.int32 6) in
  let q = Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:d in
  match
    Decomp_op.get_late_rewrite_patterns
      (supported_ops ~disable_fast_idiv:false
         ~supports_dtype:(fun dt -> not (Dtype.equal dt Dtype.int64))
         ())
      q
  with
  | Some r ->
      is_true ~msg:"fast idiv recursion keeps truncating CDIV"
        (contains_op Ops.Cdiv r && not (contains_op Ops.Fdiv r))
  | None -> is_true ~msg:"recursive fast idiv rule fired" false

let fast_idiv_disabled_for_metal () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:2_000_000_000
      ~dtype:Dtype.int32 ()
  in
  let d = Uop.const (Const.int Dtype.int32 7) in
  let q = Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:d in
  let got =
    Decomp_op.get_late_rewrite_patterns
      (supported_ops ~disable_fast_idiv:false ~is_metal:true ()) q
  in
  is_true ~msg:"fast idiv is disabled for Metal" (got = None)

let fast_idiv_promotion_requires_supported_dtype () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:2_000_000_000
      ~dtype:Dtype.int32 ()
  in
  let d = Uop.const (Const.int Dtype.int32 7) in
  let q = Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:d in
  let reject =
    Decomp_op.get_late_rewrite_patterns
      (supported_ops ~disable_fast_idiv:false
         ~supports_dtype:(fun dt -> not (Dtype.equal dt Dtype.int64))
         ())
      q
  in
  let accept =
    Decomp_op.get_late_rewrite_patterns
      (supported_ops ~disable_fast_idiv:false ()) q
  in
  is_true ~msg:"fast idiv casts only to supported promotion dtype"
    (reject = None
     &&
     match accept with
     | Some r -> contains_op Ops.Cast r
     | None -> false)

let late_cmod_power_of_two_without_and_uses_generic_rule () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:100 ~dtype:Dtype.int32 ()
  in
  let d = Uop.const (Const.int Dtype.int32 4) in
  let r = Uop.alu_binary ~op:Ops.Cmod ~lhs:x ~rhs:d in
  match
    Decomp_op.get_late_rewrite_patterns
      (supported_ops ~has_and:false ~disable_fast_idiv:false ()) r
  with
  | Some r -> is_true ~msg:"Cmod uses x - d*Cdiv when And is unavailable" (Uop.op r = Ops.Sub)
  | None -> is_true ~msg:"generic Cmod rule fired" false

let signed_cdiv_pow2_nonnegative_uses_constant_condition () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:100 ~dtype:Dtype.int32 ()
  in
  let d = Uop.const (Const.int Dtype.int32 4) in
  let q = Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:d in
  match Decomp_op.get_late_rewrite_patterns (supported_ops ()) q with
  | Some r ->
      is_true ~msg:"signed pow2 CDIV skips dynamic x<0 condition"
        (Uop.op r = Ops.Shr && not (contains_op Ops.Cmplt r))
  | None -> is_true ~msg:"signed pow2 CDIV rule fired" false

let late_mul_by_one_is_not_shifted () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:100 ~dtype:Dtype.int32 ()
  in
  let one = Uop.const (Const.int Dtype.int32 1) in
  let mul = Uop.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:one in
  let got = Decomp_op.get_late_rewrite_patterns (supported_ops ()) mul in
  is_true ~msg:"x * 1 is not rewritten to x << 0" (got = None)

let late_div_by_one_is_not_shifted () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:100 ~dtype:Dtype.uint32 ()
  in
  let one = Uop.const (Const.int Dtype.uint32 1) in
  let div = Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:one in
  let got = Decomp_op.get_late_rewrite_patterns (supported_ops ()) div in
  is_true ~msg:"x / 1 is not rewritten to x >> 0" (got = None)

let late_rewrite_does_not_lower_max () =
  let x = Uop.const_float 1.0 in
  let y = Uop.const_float 2.0 in
  let max_ = Uop.alu_binary ~op:Ops.Max ~lhs:x ~rhs:y in
  let got =
    Decomp_op.get_late_rewrite_patterns
      (supported_ops ~has_max:false ~has_cmplt:true ()) max_
  in
  is_true ~msg:"Max lowering belongs to simplifying rewrites" (got = None)

let threefry_rewrite_requires_uint64 () =
  let x = Uop.const (Const.int Dtype.uint32 42) in
  let key = Uop.const (Const.int Dtype.uint32 99) in
  let fry = Uop.alu_binary ~op:Ops.Threefry ~lhs:x ~rhs:key in
  let got =
    Decomp_op.get_simplifying_rewrite_patterns
      (supported_ops ~has_threefry:false ()) fry
  in
  is_true ~msg:"Threefry software rewrite is only for uint64" (got = None)

let early_decomp u =
  let ops = supported_ops () in
  Uop.graph_rewrite
    (Uop.first_match
       [
         Upat.Pattern_matcher.rewrite Symbolic.symbolic_simple;
         Upat.Pattern_matcher.rewrite Divandmod.div_and_mod_symbolic;
         Decomp_op.get_simplifying_rewrite_patterns ops;
       ])
    u

let early_floordiv_by_zero_raises () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.index ()
  in
  let zero = Uop.const_int 0 in
  let q = Uop.alu_binary ~op:Ops.Floordiv ~lhs:x ~rhs:zero in
  raises ~msg:"floor div by zero is caught before trunc lowering"
    Division_by_zero (fun () -> ignore (early_decomp q))

let early_floormod_by_zero_raises () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.index ()
  in
  let zero = Uop.const_int 0 in
  let r = Uop.alu_binary ~op:Ops.Floormod ~lhs:x ~rhs:zero in
  raises ~msg:"floor mod by zero is caught before trunc lowering"
    Division_by_zero (fun () -> ignore (early_decomp r))

let late_not_x_lt_const_canonicalizes () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let c = Uop.const (Const.int Dtype.int32 5) in
  let cmp = Uop.O.(not_ (x < c)) in
  match Decomp_op.get_late_rewrite_patterns (supported_ops ()) cmp with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"not (x < c) rewrites to c-1 < x"
        (Uop.op r = Ops.Cmplt
         && Array.length src = 2
         && const_int64_value src.(0) = Some 4L
         && Uop.equal src.(1) x)
  | None -> is_true ~msg:"late CMPLT rule fired" false

let late_not_const_lt_x_canonicalizes () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let c = Uop.const (Const.int Dtype.int32 5) in
  let cmp = Uop.O.(not_ (c < x)) in
  match Decomp_op.get_late_rewrite_patterns (supported_ops ()) cmp with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"not (c < x) rewrites to x < c+1"
        (Uop.op r = Ops.Cmplt
         && Array.length src = 2
         && Uop.equal src.(0) x
         && const_int64_value src.(1) = Some 6L)
  | None -> is_true ~msg:"late CMPLT rule fired" false

let late_not_ne_uses_cmpne_true_shape () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let y =
    Uop.variable ~name:"y" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let ne = Uop.O.ne x y in
  let not_ne = Uop.O.not_ ne in
  match Decomp_op.get_late_rewrite_patterns (supported_ops ()) not_ne with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"not (x != y) rewrites from CMPNE(_, true) to CMPEQ"
        (Uop.op r = Ops.Cmpeq
         && Array.length src = 2
         && Uop.equal src.(0) x && Uop.equal src.(1) y)
  | None -> is_true ~msg:"late not-ne rule fired" false

let late_add_neg_is_commutative () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let y =
    Uop.variable ~name:"y" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let sum =
    Uop.alu_binary ~op:Ops.Add ~lhs:(Uop.alu_unary ~op:Ops.Neg ~src:y) ~rhs:x
  in
  match Decomp_op.get_late_rewrite_patterns (supported_ops ()) sum with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"neg y + x rewrites to x - y"
        (Uop.op r = Ops.Sub
         && Array.length src = 2
         && Uop.equal src.(0) x && Uop.equal src.(1) y)
  | None -> is_true ~msg:"late add-neg rule fired" false

let late_mul_recip_is_commutative () =
  let a =
    Uop.variable ~name:"a" ~min_val:(-10) ~max_val:10
      ~dtype:Dtype.float32 ()
  in
  let b =
    Uop.variable ~name:"b" ~min_val:1 ~max_val:10 ~dtype:Dtype.float32 ()
  in
  let one = Uop.const (Const.float Dtype.float32 1.0) in
  let recip = Uop.alu_binary ~op:Ops.Fdiv ~lhs:one ~rhs:b in
  let mul = Uop.alu_binary ~op:Ops.Mul ~lhs:recip ~rhs:a in
  match
    Decomp_op.get_late_rewrite_patterns
      { (supported_ops ()) with has_fdiv = true } mul
  with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"(1 / b) * a rewrites to a / b"
        (Uop.op r = Ops.Fdiv
         && Array.length src = 2
         && Uop.equal src.(0) a && Uop.equal src.(1) b)
  | None -> is_true ~msg:"late mul-recip rule fired" false

let late_negated_mul_cmplt_canonicalizes () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let y =
    Uop.variable ~name:"y" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let neg_one = Uop.const (Const.int Dtype.int32 (-1)) in
  let three = Uop.const (Const.int Dtype.int32 3) in
  let lhs = Uop.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:neg_one in
  let rhs = Uop.alu_binary ~op:Ops.Mul ~lhs:y ~rhs:three in
  let cmp = Uop.alu_binary ~op:Ops.Cmplt ~lhs ~rhs in
  match Decomp_op.get_late_rewrite_patterns (supported_ops ()) cmp with
  | Some r ->
      let src = Uop.src r in
      let mul = if Array.length src = 2 then src.(0) else r in
      let msrc = Uop.src mul in
      is_true ~msg:"-x < y*c rewrites to y*(-c) < x"
        (Uop.op r = Ops.Cmplt
         && Uop.op mul = Ops.Mul
         && Array.length msrc = 2
         && Uop.equal msrc.(0) y
         && const_int64_value msrc.(1) = Some (-3L)
         && Uop.equal src.(1) x)
  | None -> is_true ~msg:"late CMPLT rule fired" false

let late_negated_const_cmplt_canonicalizes () =
  let x =
    Uop.variable ~name:"x" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let neg_one = Uop.const (Const.int Dtype.int32 (-1)) in
  let five = Uop.const (Const.int Dtype.int32 5) in
  let lhs = Uop.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:neg_one in
  let cmp = Uop.alu_binary ~op:Ops.Cmplt ~lhs ~rhs:five in
  match Decomp_op.get_late_rewrite_patterns (supported_ops ()) cmp with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"-x < c rewrites to -c < x"
        (Uop.op r = Ops.Cmplt
         && Array.length src = 2
         && const_int64_value src.(0) = Some (-5L)
         && Uop.equal src.(1) x)
  | None -> is_true ~msg:"late CMPLT rule fired" false

let late_bounded_cmplt_collapses_to_eq () =
  let x =
    Uop.variable ~name:"x" ~min_val:0 ~max_val:10 ~dtype:Dtype.int32 ()
  in
  let c3 = Uop.const (Const.int Dtype.int32 3) in
  let c5 = Uop.const (Const.int Dtype.int32 5) in
  let lo = Uop.alu_binary ~op:Ops.Cmplt ~lhs:c3 ~rhs:x in
  let hi = Uop.alu_binary ~op:Ops.Cmplt ~lhs:x ~rhs:c5 in
  let bounded = Uop.alu_binary ~op:Ops.And ~lhs:lo ~rhs:hi in
  match Decomp_op.get_late_rewrite_patterns (supported_ops ()) bounded with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"(c-1)<x & x<(c+1) rewrites to x==c"
        (Uop.op r = Ops.Cmpeq
         && Array.length src = 2
         && Uop.equal src.(0) x
         && const_int64_value src.(1) = Some 4L)
  | None -> is_true ~msg:"late CMPLT rule fired" false

let bf16_load_promotes_to_f32 () =
  let buf =
    Uop.param ~slot:0 ~dtype:Dtype.bfloat16 ~shape:(Uop.const_int 1)
      ~addrspace:Dtype.Global ()
  in
  let idx = Uop.index ~ptr:buf ~idxs:[ Uop.const_int 0 ] () in
  let load = Uop.load ~src:idx () in
  let ctx : Decomp_dtype.float_decomp_ctx =
    { from_dtype = Dtype.Bfloat16; to_dtype = Dtype.Float32 }
  in
  let rewritten =
    Uop.graph_rewrite
      (Upat.Pattern_matcher.rewrite (Decomp_dtype.pm_float_decomp ctx))
      load
  in
  let has_uint16_load =
    List.exists
      (fun n -> Uop.op n = Ops.Load && Dtype.equal (Uop.dtype n) Dtype.uint16)
      (Uop.toposort rewritten)
  in
  is_true ~msg:"bf16 load is promoted through uint16 storage to f32"
    (Dtype.equal (Uop.dtype rewritten) Dtype.float32 && has_uint16_load)

let bf16_vector_load_reindexes_shrink () =
  (* A two-lane view is now expressed by the SHRINK's shape rather than a
     vector element dtype: the buffer holds scalar bf16 and the SHRINK
     selects a width-2 window. *)
  let buf =
    Uop.param ~slot:0 ~dtype:Dtype.bfloat16 ~shape:(Uop.const_int 5)
      ~addrspace:Dtype.Global ()
  in
  let shrink =
    Uop.shrink ~src:buf ~offset:(Uop.const_int 3) ~size:(Uop.const_int 2)
  in
  let load = Uop.load ~src:shrink () in
  let ctx : Decomp_dtype.float_decomp_ctx =
    { from_dtype = Dtype.Bfloat16; to_dtype = Dtype.Float32 }
  in
  let rewritten =
    Uop.graph_rewrite
      (Upat.Pattern_matcher.rewrite (Decomp_dtype.pm_float_decomp ctx))
      load
  in
  let nodes = Uop.toposort rewritten in
  let has_offset_index =
    List.exists
      (fun n ->
         match Uop.as_index n with
         | Some { idxs = [ i ]; _ } -> Uop.vmin i = 3 || Uop.vmin i = 4
         | _ -> false)
      nodes
  in
  is_true ~msg:"bf16 vector load converts SHRINK lanes to INDEX offsets"
    (Uop.op rewritten = Ops.Stack && has_offset_index
     && not (contains_op Ops.Shrink rewritten))

let f32_store_demotes_to_bf16_bits () =
  let buf =
    Uop.param ~slot:0 ~dtype:Dtype.bfloat16 ~shape:(Uop.const_int 1)
      ~addrspace:Dtype.Global ()
  in
  let raw_idx = Uop.index ~ptr:buf ~idxs:[ Uop.const_int 0 ] () in
  let idx = Uop.with_tag (Dtype.to_string Dtype.bfloat16) raw_idx in
  let value =
    Uop.variable ~name:"v" ~min_val:(-10) ~max_val:10
      ~dtype:Dtype.float32 ()
  in
  let store = Uop.store ~dst:idx ~value () in
  let ctx : Decomp_dtype.float_decomp_ctx =
    { from_dtype = Dtype.Bfloat16; to_dtype = Dtype.Float32 }
  in
  match
    Upat.Pattern_matcher.rewrite (Decomp_dtype.pm_float_decomp ctx) store
  with
  | Some rewritten ->
      (match Uop.as_store rewritten with
       | Some { value; _ } ->
           is_true ~msg:"f32 store is converted to bf16 uint16 bits"
             (Dtype.equal (Uop.dtype value) Dtype.uint16)
       | None -> is_true ~msg:"store remains a store" false)
  | None -> is_true ~msg:"float store rule fired" false

let gated_f32_store_is_not_float_decomposed () =
  let buf =
    Uop.param ~slot:0 ~dtype:Dtype.bfloat16 ~shape:(Uop.const_int 1)
      ~addrspace:Dtype.Global ()
  in
  let raw_idx = Uop.index ~ptr:buf ~idxs:[ Uop.const_int 0 ] () in
  let idx = Uop.with_tag (Dtype.to_string Dtype.bfloat16) raw_idx in
  let value =
    Uop.variable ~name:"v" ~min_val:(-10) ~max_val:10
      ~dtype:Dtype.float32 ()
  in
  let store = Uop.store ~dst:idx ~value ~gate:(Uop.const_bool true) () in
  let ctx : Decomp_dtype.float_decomp_ctx =
    { from_dtype = Dtype.Bfloat16; to_dtype = Dtype.Float32 }
  in
  match Upat.Pattern_matcher.rewrite (Decomp_dtype.pm_float_decomp ctx) store with
  | None -> ()
  | Some _ -> is_true ~msg:"gated store should not match f2f store rule" false

let () =
  run "tolk.uop.decomp"
    [
      group "transcendentals"
        [ test "sqrt decomposition builds Where"
            sqrt_decomposition_builds_where;
          test "log2 denormal scale uses float power"
            log2_denormal_scale_uses_float_power;
          test "sin f16 Cody-Waite casts quadrant to f32"
            sin_f16_cody_waite_casts_quadrant_to_f32 ];
      group "integer division"
        [ test "magicgu is correct" magicgu_correct ];
      group "prng"
        [ test "threefry2x32 is uint64" threefry_produces_uint64 ];
      group "long decomposition"
        [ test "MUL lowers" mul_long_decomposes;
          test "IDIV lowers" idiv_long_decomposes;
          test "MOD lowers" mod_long_decomposes;
          test "CAST float->long lowers" cast_float_to_long_decomposes;
          test "CAST long->int lowers" cast_long_to_int_decomposes;
          test "BITCAST long->long lowers" bitcast_long_to_long_decomposes;
          test "CONST halves truncate to int32"
            long_const_halves_are_truncated_to_int32;
          test "untagged CONST is low half"
            untagged_long_const_is_low_half;
          test "unbounded param keeps unbounded size"
            unbounded_long_param_keeps_unbounded_size;
          test "untagged INDEX is not rewritten"
            untagged_long_index_is_not_rewritten;
          test "tagged INDEX reindexes before define"
            tagged_long_index_reindexes_before_define;
          test "tagged INDEX preserves multi-index tail"
            tagged_long_index_preserves_multi_index_tail;
          test "CAST float->long high half uses reciprocal"
            float_to_long_high_half_uses_reciprocal;
          test "variable SHL uses native narrow shift"
            long_variable_shl_uses_native_shift;
          test "variable SHR uses native narrow shift"
            long_variable_shr_uses_native_shift;
        ];
      group "simplifying rewrites"
        [ test "Floordiv same sign lowers to Cdiv"
            floordiv_lowering_uses_cdiv_for_same_sign;
          test "Floordiv mixed sign lowers with correction"
            floordiv_lowering_corrects_mixed_sign;
          test "Floormod power of two lowers to And for negative input"
            floormod_power_of_two_uses_and_for_negative_input;
          test "late Cmod power of two rejects negative signed input"
            late_cmod_power_of_two_rejects_negative_signed_input;
          test "fast idiv small range folds to zero"
            fast_idiv_small_range_folds_to_zero;
          test "fast idiv rejects non-native divisor"
            fast_idiv_rejects_non_native_divisor;
          test "fast idiv recursion uses Cdiv"
            fast_idiv_recursion_uses_cdiv;
          test "fast idiv is disabled for Metal"
            fast_idiv_disabled_for_metal;
          test "fast idiv promotion checks dtype support"
            fast_idiv_promotion_requires_supported_dtype;
          test "late Cmod power of two uses generic rule without And"
            late_cmod_power_of_two_without_and_uses_generic_rule;
          test "signed Cdiv pow2 uses constant condition"
            signed_cdiv_pow2_nonnegative_uses_constant_condition;
          test "late Mul by one is not shifted"
            late_mul_by_one_is_not_shifted;
          test "late Cdiv by one is not shifted"
            late_div_by_one_is_not_shifted;
          test "late rewrite does not lower Max"
            late_rewrite_does_not_lower_max;
          test "Threefry rewrite requires uint64"
            threefry_rewrite_requires_uint64;
          test "early Floordiv by zero raises before trunc lowering"
            early_floordiv_by_zero_raises;
          test "early Floormod by zero raises before trunc lowering"
            early_floormod_by_zero_raises;
        ];
      group "late comparisons"
        [ test "not (x < c) canonicalizes"
            late_not_x_lt_const_canonicalizes;
          test "not (c < x) canonicalizes"
            late_not_const_lt_x_canonicalizes;
          test "not-ne uses CMPNE true shape"
            late_not_ne_uses_cmpne_true_shape;
          test "add-neg rewrite is commutative"
            late_add_neg_is_commutative;
          test "mul-recip rewrite is commutative"
            late_mul_recip_is_commutative;
          test "-x < y*c canonicalizes"
            late_negated_mul_cmplt_canonicalizes;
          test "-x < c canonicalizes"
            late_negated_const_cmplt_canonicalizes;
          test "bounded CMPLT collapses to equality"
            late_bounded_cmplt_collapses_to_eq;
        ];
      group "float decomposition"
        [ test "bf16 load promotes to f32" bf16_load_promotes_to_f32;
          test "bf16 vector load reindexes SHRINK"
            bf16_vector_load_reindexes_shrink;
          test "f32 store demotes to bf16 bits" f32_store_demotes_to_bf16_bits;
          test "gated f32 store is not decomposed"
            gated_f32_store_is_not_float_decomposed;
        ];
    ]
