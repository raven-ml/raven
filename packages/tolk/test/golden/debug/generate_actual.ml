(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates elementwise_add.actual for the debug golden test. Dune runs
   this with DEBUG=6 so graph_rewrite emits print_uops after each named
   stage. Stderr is redirected to the output file. *)

open Tolk
open Tolk_uop
module U = Uop

let global_fptr size =
  Dtype.Ptr.create (Dtype.val_of Dtype.float32) ~addrspace:Global ~size
let idx n = U.const (Const.int Dtype.Val.weakint n)

let make_kernel ~name ~opts_to_apply ~ptr_size =
  let ptr = Dtype.Ptr (global_fptr ptr_size) in
  let p0 = U.param ~slot:0 ~dtype:ptr () in
  let p1 = U.param ~slot:1 ~dtype:ptr () in
  let p2 = U.param ~slot:2 ~dtype:ptr () in
  let r0 = U.range ~size:(idx 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ~as_ptr:true ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ~as_ptr:true ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:ld_b in
  let st = U.store ~dst:(U.index ~ptr:p2 ~idxs:[r0] ~as_ptr:true ()) ~value:add () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink
    ~kernel_info:{ U.name = name;
      axis_types = [ Axis_type.Global ]; dont_use_locals = false;
      applied_opts = []; opts_to_apply; estimates = None; beam = 0 }
    [ e ]

let ren = Cstyle.clang_no_abi Gpu_target.X86_64

let saved_stdout = Unix.dup Unix.stdout

let run_test ~name ~sink =
  let path = Filename.concat Sys.argv.(1) (name ^ ".actual") in
  let fd = Unix.openfile path [ O_WRONLY; O_CREAT; O_TRUNC ] 0o644 in
  Unix.dup2 fd Unix.stdout;
  Unix.close fd;
  ignore (Codegen.full_rewrite_to_sink ~optimize:true ren sink);
  flush stdout;
  Unix.dup2 saved_stdout Unix.stdout

let () =
  run_test ~name:"elementwise_add"
    ~sink:(make_kernel ~name:"elementwise_add" ~opts_to_apply:(Some [])
             ~ptr_size:(-1));
  run_test ~name:"elementwise_add_opt"
    ~sink:(make_kernel ~name:"elementwise_add_opt" ~opts_to_apply:None
             ~ptr_size:256)
