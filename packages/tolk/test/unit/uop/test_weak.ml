(* Per-rule tests for Weak: committing weak dtypes to concrete widths. *)

open Windtrap
open Tolk_uop
module U = Uop
module D = Dtype
module C = Const

let dtype = testable ~pp:D.pp ~equal:D.equal ()
let src node i = (U.src node).(i)

let rewrite pm u = U.graph_rewrite (Upat.Pattern_matcher.rewrite pm) u

(* The pass is driven from a sink: a weak-rooted expression is only lowered
   once a node that is not itself weak demands a width from it. *)
let lower u = src (rewrite (Weak.pm_lower_index_dtype ()) (U.sink [ u ])) 0

let const_int_of node =
  match U.op node, U.arg node with
  | Ops.Const, U.Arg.Value c -> (
      match C.view c with C.Int n -> Some (Int64.to_int n) | _ -> None)
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
  equal dtype ~msg:"the weak const is rebuilt, not cast" D.int8
    (U.dtype (src r 1));
  is_true ~msg:"no cast is introduced" (Ops.equal (U.op (src r 1)) Ops.Const)

let peer_commits_weak_alu_by_cast () =
  let x = U.variable ~name:"x" ~min_val:0 ~max_val:4 ~dtype:D.weakint () in
  let r = rewrite Weak.pm_commit_weak U.O.(i32 3 + (x + U.const_int 1)) in
  equal dtype ~msg:"the peer's width wins" D.int32 (U.dtype r);
  is_true ~msg:"a weak non-const takes a cast"
    (Ops.equal (U.op (src r 1)) Ops.Cast)

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

(* Whole-pass lowering. *)

let range_arithmetic_lowers_to_concrete_int () =
  let r =
    U.range ~size:(U.const_int 16) ~axis:0 ~kind:Axis_type.Weak ()
  in
  let e = U.O.(r * U.const_int 4) in
  let lowered = lower e in
  is_true ~msg:"no weak dtype survives"
    (List.for_all
       (fun n -> not (D.is_weak (U.dtype n)))
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
       (fun n -> not (D.is_weak (U.dtype n)))
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

let () =
  run "tolk.uop.weak"
    [
      group "default width"
        [
          test "unconstrained int const"
            unconstrained_int_const_commits_at_int32;
          test "overflowing int const"
            unconstrained_overflowing_const_commits_at_int64;
          test "float const" unconstrained_float_const_commits_at_default_float;
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
          test "a wider cast widens" consumer_cast_widens;
          test "a narrower cast does not narrow" consumer_cast_never_narrows;
        ];
      group "whole pass"
        [
          test "range arithmetic" range_arithmetic_lowers_to_concrete_int;
          test "comparison" comparison_unifies_operand_widths;
          test "gated long index narrows"
            gated_long_index_narrows_for_small_buffers;
        ];
    ]
