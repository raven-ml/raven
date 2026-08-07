(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Pins that a bufferize of a range-independent value does not fan its
   consumer out over the bufferize's ranges.

   The scheduler can emit a LOCAL STAGE whose source ignores the ranges it is
   staged over, read back by an INDEX that supplies no index at all:

     RECIPROCAL  float  [scale]              scalar, uses no range
     RANGE       index  ['n']                LOOP, used by nothing else
     STAGE       float  [reciprocal, range]  addrspace=LOCAL, removable=true
     INDEX       float  [stage]              zero index sources

   [Uop.shape] of an INDEX is its index shapes followed by the pointer axes it
   does not consume, so that INDEX carries the whole ['n'] axis with no range
   behind it. Every consumer inherits the axis: the store's value broadcasts to
   [n] lanes, devectorization emits one store per lane, and the n stores all
   land on the same scalar address.

   Both kernels below compute the same scalar. The one with the staged
   reciprocal must lower to the same number of stores as the one that uses the
   reciprocal directly.

   CURRENTLY RED, on every renderer and at every width, including the width 64
   used here. Lowering the staged kernel raises
   [Failure "Coalese: multiple stores to the same offset"]. It is the same
   construct that made the float16 GPT-2 training step uncompilable: there the
   staged value is [1 / loss_scale], shared between the scalar loss unscale and
   the vocabulary-shaped gradient unscale, and the range is the 50257-wide
   vocabulary axis.

   The owner is the scheduler, not codegen. [Indexing.wrap_realized_src] indexes
   a realized source with the parent's ranges selected by the source's realized
   axes, which yields too few indices when the parent has lower rank;
   [Rangeify.stage_index_sources] then declines to remove the stage rather than
   rejecting the mismatch. Fixing it means finding why a rank-0 node is assigned
   rank-2 output ranges in the first place, so this test stays out of
   [@runtest] until then. *)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let f32 = Dtype.float32

let kernel_info name =
  {
    U.name;
    axis_types = [];
    dont_use_locals = false;
    applied_opts = [];
    opts_to_apply = Some [];
    estimates = None;
    beam = 0;
  }

let param ~slot size =
  U.param ~slot ~dtype:f32 ~shape:(U.const_int size) ()

(* [sum(x) * scale] scaled back down by [1 / scale], stored to [out.(0)].
   [staged] routes the reciprocal through a LOCAL bufferize over a [width]-wide
   loop range that its value does not depend on. *)
let unscale_kernel ~staged ~width =
  let out = param ~slot:0 1 in
  let x = param ~slot:1 16 in
  let scale = param ~slot:2 1 in
  let r = U.range ~size:(U.const_int 16) ~axis:0 ~kind:Axis_type.Reduce () in
  let acc =
    U.reduce
      ~src:(U.index ~ptr:x ~idxs:[ r ] ())
      ~ranges:[ r ] ~op:Ops.Add ~dtype:f32
  in
  let s = U.index ~ptr:scale ~idxs:[ U.const_int 0 ] () in
  let scaled = U.alu_binary ~op:Ops.Mul ~lhs:acc ~rhs:s in
  let recip = U.alu_unary ~op:Ops.Reciprocal ~src:s in
  let recip =
    if not staged then recip
    else
      let loop =
        U.range ~size:(U.const_int width) ~axis:1 ~kind:Axis_type.Loop ()
      in
      let stage =
        U.stage ~src:recip ~ranges:[ loop ]
          ~opts:
            { device = None; addrspace = Dtype.Local; removable = true }
      in
      U.index ~ptr:stage ~idxs:[] ()
  in
  let value = U.alu_binary ~op:Ops.Mul ~lhs:scaled ~rhs:recip in
  let dst = U.index ~ptr:out ~idxs:[ U.const_int 0 ] () in
  U.sink
    ~kernel_info:(kernel_info (if staged then "staged" else "plain"))
    [ U.store ~dst ~value () ]

let store_count sink =
  List.length
    (List.filter (fun u -> U.op u = Ops.Store) (U.toposort sink))

let backends =
  [
    ("clang", Cstyle.clang_no_abi Gpu_target.X86_64);
    ("cuda", Cstyle.cuda Gpu_target.SM80);
  ]

let () =
  run "Stage broadcast"
    [
      group "range-independent bufferize"
        (List.map
           (fun (backend, ren) ->
             test
               (Printf.sprintf "%s: staged reciprocal does not fan out the store" backend)
               (fun () ->
                 let lower staged =
                   Codegen.full_rewrite_to_sink ren
                     (unscale_kernel ~staged ~width:64)
                 in
                 (* Sequenced so that a failure names the staged kernel: the
                    plain one is expected to lower on every renderer. *)
                 let plain = store_count (lower false) in
                 let staged = store_count (lower true) in
                 equal int
                   ~msg:"staged kernel stores as many times as the plain one"
                   plain staged))
           backends);
    ]
