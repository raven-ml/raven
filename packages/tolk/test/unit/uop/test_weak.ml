(* Per-rule tests for Weak: committing weak dtypes to concrete widths. *)

open Windtrap
open Tolk_uop
module U = Uop
module D = Dtype
module C = Const

let dtype = Testable.make ~pp:D.pp ~equal:D.equal
let src node i = (U.src node).(i)

let rewrite pm u = U.graph_rewrite (Upat.Pattern_matcher.rewrite pm) u

(* The pass is driven from a sink: a weak-rooted expression is only lowered
   once a node that is not itself weak demands a width from it. *)
let lower u = src (rewrite (Weak.pm_lower_index_dtype ()) (U.sink [ u ])) 0

let const_int_of node =
  match U.as_const node with
  | Some c -> (
      match C.view c with C.Int n -> Some (Z.to_int n) | _ -> None)
  | _ -> None

let i32 n = U.const (C.int D.int32 n)
let i8 n = U.const (C.int D.int8 n)

(* A value nobody constrains takes its default width. *)

let unconstrained_int_const_commits_at_int32 () =
  let r = lower U.O.(U.const_int 1 + U.const_int 2) in
  equal dtype ~msg:"the sum takes int32" D.int32 (U.dtype r);
  equal (option int) ~msg:"the left operand keeps its value" (Some 1)
    (const_int_of (src r 0));
  equal dtype ~msg:"the left operand takes int32" D.int32
    (U.dtype (src r 0))

let unconstrained_overflowing_const_commits_at_int64 () =
  let big = U.const (C.int D.weakint 0x1_0000_0000) in
  let r = lower U.O.(big + U.const_int 1) in
  equal dtype ~msg:"an int32-overflowing value takes int64" D.int64 (U.dtype r)

let unconstrained_float_const_commits_at_default_float () =
  let r = lower U.O.(U.const_float 1.5 * U.const_float 2.0) in
  equal dtype ~msg:"weakfloat takes default_float" D.default_float (U.dtype r)

(* Demand from a peer. *)

let peer_commits_weak_const_at_its_own_width () =
  let r = rewrite Weak.pm_commit_weak U.O.(i8 3 + U.const_int 1) in
  equal dtype ~msg:"the peer's width wins" D.int8 (U.dtype r);
  equal dtype ~msg:"a derivable const retains its weak payload" D.weakint
    (U.dtype (src r 1));
  is_true ~msg:"no cast is introduced" (Ops.equal (U.op (src r 1)) Ops.Const)

let peer_commits_weak_alu_by_cast () =
  let x = U.variable ~name:"x" ~min_val:0 ~max_val:4 ~dtype:D.weakint () in
  let r = rewrite Weak.pm_commit_weak U.O.(i32 3 + (x + U.const_int 1)) in
  equal dtype ~msg:"the peer's width wins" D.int32 (U.dtype r);
  equal dtype ~msg:"the demanded weak expression commits at the peer width"
    D.int32 (U.dtype (src r 1))

let all_weak_sources_stay_weak () =
  let e = U.O.(U.const_int 1 + U.const_int 2) in
  is_true ~msg:"nothing to commit against"
    (Option.is_none (Upat.Pattern_matcher.rewrite Weak.pm_commit_weak e))

let store_commits_value_at_destination_dtype () =
  let buf =
    U.param ~slot:0 ~dtype:D.float32 ~shape:(U.stack [ U.const_int 8 ])
      ~addrspace:D.Global ()
  in
  let idx = U.index ~ptr:buf ~idxs:[ i32 0 ] () in
  let store = U.store ~dst:idx ~value:(U.const_float 1.0) () in
  let r = rewrite Weak.pm_commit_weak store in
  equal dtype ~msg:"the destination's dtype wins" D.float32
    (U.dtype (src r 1))

(* Demand from a consumer's cast: a floor, never a narrowing. *)

let consumer_cast_widens () =
  let x = U.variable ~name:"x" ~min_val:0 ~max_val:4 ~dtype:D.weakint () in
  let e = U.cast ~src:U.O.(x + U.const_int 1) ~dtype:D.int64 in
  let r = rewrite Weak.pm_cast_weak e in
  equal dtype ~msg:"the cast is preserved" D.int64 (U.dtype r);
  equal dtype ~msg:"the computation happens at the demanded width" D.int64
    (U.dtype (src r 0))

let consumer_cast_never_narrows () =
  let big = U.const (C.int D.weakint 0x1_0000_0000) in
  let e = U.cast ~src:U.O.(big + U.const_int 1) ~dtype:D.int8 in
  let r = rewrite Weak.pm_cast_weak e in
  equal dtype ~msg:"the cast is preserved" D.int8 (U.dtype r);
  equal dtype ~msg:"the value keeps the width its range needs" D.int64
    (U.dtype (src r 0))

let cast_preserves_operand_widths () =
  let large = 1 lsl 40 in
  let x = U.variable ~name:"dividend" ~min_val:0 ~max_val:large
      ~dtype:D.weakint () in
  let quotient = U.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:(U.const_int large) in
  let r = rewrite Weak.pm_cast_weak (U.cast ~src:quotient ~dtype:D.int32) in
  let division = List.find (fun u -> U.op u = Ops.Cdiv) (U.toposort r) in
  equal dtype ~msg:"division operands need int64 even though its result fits int32"
    D.int64 (U.dtype (src division 0))

let consecutive_weak_casts_preserve_integer_conversion () =
  let x = U.const (C.float D.float32 1.5) in
  let e = U.cast ~src:(U.cast ~src:x ~dtype:D.weakint) ~dtype:D.weakfloat in
  let r = lower e |> Symbolic.simplify in
  match U.as_const r with
  | Some c ->
      (match C.view c with
       | C.Float value -> equal float_exact 1.0 value
       | _ -> fail "expected a floating constant")
  | _ -> fail "expected a constant"

let weak_integer_cast_is_a_value_conversion () =
  let x = U.const (C.float D.float32 1.5) in
  let integer = U.cast ~src:x ~dtype:D.weakint in
  let r = lower U.O.(integer + U.const_int 1) |> Symbolic.simplify in
  equal dtype D.int32 (U.dtype r);
  equal (option int) (Some 2) (const_int_of r)

(* Whole-pass lowering. *)

let range_arithmetic_lowers_to_concrete_int () =
  let r =
    U.range ~size:(U.const_int 16) ~axis:0 ~kind:Axis_type.Weak ()
  in
  let e = U.O.(r * U.const_int 4) in
  let lowered = lower e in
  is_true ~msg:"no weak dtype survives"
    (List.for_all
       (fun n -> U.op n = Ops.Const || not (D.is_weak (U.dtype n)))
       (U.toposort lowered));
  equal dtype ~msg:"index math lands at int32" D.int32 (U.dtype lowered)

let comparison_unifies_operand_widths () =
  let r =
    U.range ~size:(U.const_int 16) ~axis:0 ~kind:Axis_type.Weak ()
  in
  let lowered = lower U.O.(r < U.const_int 8) in
  equal dtype ~msg:"a comparison is bool" D.bool (U.dtype lowered);
  is_true ~msg:"no weak dtype survives"
    (List.for_all
       (fun n -> U.op n = Ops.Const || not (D.is_weak (U.dtype n)))
       (U.toposort lowered))

let gated_long_index_narrows_for_small_buffers () =
  let buf =
    U.param ~slot:0 ~dtype:D.float32 ~shape:(U.stack [ U.const_int 8 ])
      ~addrspace:D.Global ()
  in
  let long_idx = U.cast ~src:(i32 3) ~dtype:D.int64 in
  let gate = U.variable ~name:"g" ~min_val:0 ~max_val:1 ~dtype:D.bool () in
  let e = U.index ~ptr:buf ~idxs:[ U.valid ~src:long_idx ~cond:gate ] () in
  let r = lower e in
  equal dtype ~msg:"an index into an 8-element buffer fits int32" D.int32
    (U.dtype (src (src r 1) 1))

let gated_long_index_keeps_wide_storage () =
  let buf = U.param ~slot:0 ~dtype:D.float32 ~shape:(U.const_int (1 lsl 33)) () in
  let index = U.variable ~name:"index" ~min_val:0 ~max_val:(1 lsl 32)
      ~dtype:D.int64 () in
  let gate = U.variable ~name:"g" ~min_val:0 ~max_val:1 ~dtype:D.bool () in
  let r = lower (U.index ~ptr:buf ~idxs:[ U.valid ~src:index ~cond:gate ] ()) in
  equal dtype D.int64 (U.dtype (src (src r 1) 1))

let uint64_width_and_unrepresentable_const () =
  let maximum = Z.pred (Z.shift_left Z.one 64) in
  let value n = U.const (C.integer D.weakint n) in
  let r = lower (value maximum) in
  equal dtype D.uint64 (U.dtype r);
  (match U.as_const r with
   | Some c ->
       (match C.view c with
        | C.Int n -> is_true (Z.equal maximum n)
        | _ -> fail "expected integer constant")
   | _ -> fail "expected constant");
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> lower (value (Z.succ maximum)))

let uncast_preserves_operand_and_result_types () =
  let x = U.variable ~name:"concrete" ~min_val:0 ~max_val:10 ~dtype:D.int32 () in
  let sum = U.O.(x + i32 1) in
  let bare = rewrite Weak.pm_uncast_const sum in
  equal dtype D.int32 (U.dtype bare);
  is_true (U.op (src bare 1) = Ops.Const);
  let constants = U.O.(i32 1 + i32 2) in
  is_true ~msg:"both constants keep the width of the expression"
    (U.equal constants (rewrite Weak.pm_uncast_const constants));
  let count = U.variable ~name:"shift" ~min_val:0 ~max_val:31 ~dtype:D.uint32 () in
  let shift = U.alu_binary ~op:Ops.Shl ~lhs:(i32 1) ~rhs:count in
  is_true ~msg:"a concrete operand meet cannot hide a weak shifted value"
    (U.equal shift (rewrite Weak.pm_uncast_const shift))

(* A committed cast goes only where the literal keeps its value through it:
   uint8 300 is 44, so the cast of 300 stays and the cast of 44 goes. *)
let uncast_keeps_a_wrapping_cast () =
  let x = U.param ~slot:0 ~dtype:D.uint8 () in
  let bound n = U.cconst (C.int D.weakint n) D.uint8 in
  let kept = U.O.(x < bound 300) in
  is_true ~msg:"x < uint8 300 keeps its cast"
    (U.equal kept (rewrite Weak.pm_uncast_const kept));
  let stripped = rewrite Weak.pm_uncast_const U.O.(x < bound 44) in
  is_true ~msg:"x < uint8 44 loses its cast" (U.op (src stripped 1) = Ops.Const)

let final_constants_state_width_on_each_edge () =
  let literal = U.const_int 1 in
  let i = U.variable ~name:"integer" ~min_val:0 ~max_val:10 ~dtype:D.int32 () in
  let f = U.variable ~name:"float" ~min_val:0 ~max_val:10 ~dtype:D.float32 () in
  let root = U.sink [ U.O.(i + literal); U.O.(f + literal); U.const_bool true ] in
  let result = rewrite Weak.pm_cast_const root in
  equal dtype D.int32 (U.dtype (src (src result 0) 1));
  equal dtype D.float32 (U.dtype (src (src result 1) 1));
  let boolean = src result 2 in
  is_true (U.op boolean = Ops.Cast && U.op (src boolean 0) = Ops.Const);
  equal dtype D.bool (U.dtype boolean);
  is_true ~msg:"final commitment is stable"
    (U.equal result (rewrite Weak.pm_cast_const result))

let late_simplification_preserves_committed_literals () =
  let x = U.variable ~name:"bf16" ~min_val:0 ~max_val:10 ~dtype:D.bfloat16 () in
  let literal = U.const (C.float D.bfloat16 1.5) in
  let sum = U.O.(x + literal) in
  is_true ~msg:"late decomposition keeps the emulated operand width"
    (U.equal sum (rewrite Symbolic.symbolic_simple sum));
  let early = rewrite Symbolic.symbolic sum in
  is_true ~msg:"full symbolic still exposes literals for folding"
    (Array.exists (fun s -> U.op s = Ops.Const) (U.src early))

let () =
  exit (run "tolk.uop.weak"
    [
      group "default width"
        [
          test "unconstrained int const"
            unconstrained_int_const_commits_at_int32;
          test "overflowing int const"
            unconstrained_overflowing_const_commits_at_int64;
          test "float const" unconstrained_float_const_commits_at_default_float;
          test "uint64 width and overflow" uint64_width_and_unrepresentable_const;
        ];
      group "peer demand"
        [
          test "weak const rebuilt at the peer's width"
            peer_commits_weak_const_at_its_own_width;
          test "weak alu cast to the peer's width"
            peer_commits_weak_alu_by_cast;
          test "all-weak sources stay weak" all_weak_sources_stay_weak;
          test "store commits at the destination dtype"
            store_commits_value_at_destination_dtype;
        ];
      group "cast demand"
        [
          test "a cast preserves operand widths" cast_preserves_operand_widths;
          test "consecutive weak casts preserve integer conversion"
            consecutive_weak_casts_preserve_integer_conversion;
          test "a weak integer cast converts its value"
            weak_integer_cast_is_a_value_conversion;
          test "a wider cast widens" consumer_cast_widens;
          test "a narrower cast does not narrow" consumer_cast_never_narrows;
        ];
      group "literal edges"
        [ test "uncasting preserves both derived types" uncast_preserves_operand_and_result_types;
          test "uncasting keeps a wrapping cast" uncast_keeps_a_wrapping_cast;
          test "late simplification preserves committed literals" late_simplification_preserves_committed_literals;
          test "final constants state edge widths" final_constants_state_width_on_each_edge ];
      group "whole pass"
        [
          test "range arithmetic" range_arithmetic_lowers_to_concrete_int;
          test "comparison" comparison_unifies_operand_widths;
          test "gated long index narrows"
            gated_long_index_narrows_for_small_buffers;
          test "gated long index keeps its width for huge buffers"
            gated_long_index_keeps_wide_storage;
        ];
    ])
