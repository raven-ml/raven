(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates elementwise_add.actual for the debug golden test. Dune runs
   this with DEBUG=6 so graph_rewrite emits print_uops after each named
   stage. Stderr is redirected to the output file. *)

open Tolk
open Tolk_ir
module K = Kernel

let global_fptr = Dtype.ptr_of Dtype.float32 ~addrspace:Global ~size:(-1)
let idx n = K.const (Const.int Dtype.index n)

let make_kernel ~name ~opts_to_apply =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let p2 = K.param ~idx:2 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 256) ~axis:0 ~kind:Axis_kind.Global () in
  let ld_a = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let ld_b = K.load ~src:(K.index ~ptr:p1 ~idxs:[ r0 ] ()) () in
  let add = K.binary ~op:`Add ~lhs:ld_a ~rhs:ld_b in
  let st = K.store ~dst:(K.index ~ptr:p2 ~idxs:[ r0 ] ()) ~value:add ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  K.sink
    ~kernel_info:{ K.name = name;
      axis_kinds = [ Axis_kind.Global ]; dont_use_locals = false;
      applied_opts = []; opts_to_apply; estimates = None }
    [ e ]

let ren = Cstyle.clang_no_abi

let saved_stderr = Unix.dup Unix.stderr

let run_test ~name ~sink =
  let path = Filename.concat Sys.argv.(1) (name ^ ".actual") in
  let fd = Unix.openfile path [ O_WRONLY; O_CREAT; O_TRUNC ] 0o644 in
  Unix.dup2 fd Unix.stderr;
  Unix.close fd;
  ignore (Pipeline.full_rewrite_to_sink ~optimize:true ren sink);
  flush stderr;
  Unix.dup2 saved_stderr Unix.stderr

let () =
  (* Test 1: no optimization (scalar) *)
  run_test ~name:"elementwise_add"
    ~sink:(make_kernel ~name:"elementwise_add" ~opts_to_apply:(Some []));
  (* Test 2: auto-optimized (float4 upcast) *)
  run_test ~name:"elementwise_add_opt"
    ~sink:(make_kernel ~name:"elementwise_add_opt" ~opts_to_apply:None)
