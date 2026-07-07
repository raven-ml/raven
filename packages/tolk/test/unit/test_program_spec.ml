(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop

module U = Uop
module E = Program_spec.Estimates

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)
let shape dims =
  let dims = List.map (fun n -> U.const (Const.int Dtype.Val.weakint n)) dims in
  U.stack ~dtype:Dtype.Val.weakint dims

let param slot dt =
  U.param ~slot ~dtype:(Dtype.Ptr (global_ptr dt)) ~shape:(shape [ 1 ])
    ~addrspace:Global ()

let buffer slot dt =
  U.replace (param slot dt) ~op:Ops.Buffer ()

let i32 n = U.const (Const.int Dtype.Val.int32 n)
let f32 x = U.const (Const.float Dtype.Val.float32 x)
let define_var name lo hi =
  U.replace
    (U.variable ~name ~min_val:lo ~max_val:hi ~dtype:Dtype.Val.int32 ())
    ~src:[| shape [] |] ()
let index ptr idx = U.index ~ptr ~idxs:[idx] ~as_ptr:true ()
let load src = U.load ~src ()
let store dst value = U.store ~dst ~value ()
let add lhs rhs = U.alu_binary ~op:Ops.Add ~lhs ~rhs
let mul lhs rhs = U.alu_binary ~op:Ops.Mul ~lhs ~rhs
let floordiv lhs rhs = U.alu_binary ~op:Ops.Floordiv ~lhs ~rhs
let floormod lhs rhs = U.alu_binary ~op:Ops.Floormod ~lhs ~rhs
let neg src = U.alu_unary ~op:Ops.Neg ~src
let mulacc a b c = U.alu_ternary ~op:Ops.Mulacc ~a ~b ~c
let range size = U.range ~size ~axis:0 ~kind:Axis_type.Loop ()
let special dim size =
  U.special ~name:(Gpu_dim.to_special_name dim) ~size
    ~dtype:(Dtype.val_of (U.dtype size))
    ()

let spec_of ?estimates ?aux program =
  Program_spec.of_program ~name:"kern" ~src:"" ~device:"CPU" ?estimates ?aux
    program

let empty_spec ?estimates () = spec_of ?estimates []

let expect_int_estimate label expected = function
  | E.Int n -> equal int expected n ~msg:label
  | E.Symbolic _ -> failwith (label ^ ": expected exact int estimate")

let () =
  run "Program_spec"
    [
      group "Extraction"
        [
          test "reads and writes are deduplicated" (fun () ->
            let p0 = param 0 Dtype.Val.float32 in
            let p1 = param 1 Dtype.Val.float32 in
            let c0 = i32 0 in
            let idx1 = index p0 c0 in
            let idx2 = index p1 c0 in
            let ld1 = load idx2 in
            let ld2 = load idx2 in
            let st1 = store idx1 ld1 in
            let st2 = store idx1 ld2 in
            let spec = spec_of [ p0; p1; c0; idx1; idx2; ld1; ld2; st1; st2 ] in
            equal (list int) [ 0; 1 ] (Program_spec.globals spec);
            equal (list int) [ 0 ] (Program_spec.outs spec);
            equal (list int) [ 1 ] (Program_spec.ins spec));
          test "buffer tracing passes through cast and after" (fun () ->
            let p0 = param 0 Dtype.Val.float32 in
            let c0 = i32 0 in
            let idx = index p0 c0 in
            let dep = U.barrier () in
            let sequenced = U.after ~src:idx ~deps:[ dep ] in
            let casted = U.cast ~src:sequenced ~dtype:(U.dtype sequenced) in
            let ld = load casted in
            let spec = spec_of [ p0; c0; idx; dep; sequenced; casted; ld ] in
            equal (list int) [ 0 ] (Program_spec.ins spec));
          test "buffer args are treated as globals" (fun () ->
            let b0 = buffer 0 Dtype.Val.float32 in
            let b1 = buffer 1 Dtype.Val.float32 in
            let c0 = i32 0 in
            let out_idx = index b0 c0 in
            let in_idx = index b1 c0 in
            let ld = load in_idx in
            let st = store out_idx ld in
            let spec = spec_of [ b0; b1; c0; out_idx; in_idx; ld; st ] in
            equal (list int) [ 0; 1 ] (Program_spec.globals spec);
            equal (list int) [ 0 ] (Program_spec.outs spec);
            equal (list int) [ 1 ] (Program_spec.ins spec));
          test "thread-group launch expressions are preserved" (fun () ->
            let m = define_var "m" 1 32 in
            let c4 = i32 4 in
            let groups = mul m c4 in
            let gid = special (Gpu_dim.Group_id 0) groups in
            let lid = special (Gpu_dim.Local_id 1) m in
            let spec = spec_of [ m; c4; groups; gid; lid ] in
            match Program_spec.launch_kind spec with
            | Program_spec.Thread_groups ->
                let global, local = Program_spec.launch_dims spec [ "m", 3 ] in
                equal (array int) [| 12; 1; 1 |] global;
                begin match local with
                | None -> failwith "expected local dims"
                | Some local -> equal (array int) [| 1; 3; 1 |] local
                end
            | _ -> failwith "expected thread-group launch metadata");
          test "launch variables are resolved by name" (fun () ->
            let m = define_var "m" 0 7 in
            let n = define_var "n" 0 15 in
            let gid = special (Gpu_dim.Group_id 0) n in
            let global, _local =
              Program_spec.launch_dims (spec_of [ m; n; gid ])
                [ "m", 3; "n", 9 ]
            in
            equal (array int) [| 9; 1; 1 |] global);
          test "launch floor div and mod use Python semantics" (fun () ->
            let n = define_var "n" (-10) 10 in
            let three = i32 3 in
            let groups = floordiv n three in
            let locals = floormod n three in
            let gid = special (Gpu_dim.Group_id 0) groups in
            let lid = special (Gpu_dim.Local_id 1) locals in
            let global, local =
              Program_spec.launch_dims
                (spec_of [ n; three; groups; locals; gid; lid ])
                [ "n", -7 ]
            in
            equal (array int) [| -3; 1; 1 |] global;
            begin match local with
            | None -> failwith "expected local dims"
            | Some local -> equal (array int) [| 1; 2; 1 |] local
            end);
          test "global idx uses flat thread launch" (fun () ->
            let threads = define_var "threads" 1 64 in
            let gid = special (Gpu_dim.Global_idx 2) threads in
            let spec = spec_of [ threads; gid ] in
            match Program_spec.launch_kind spec with
            | Program_spec.Threads ->
                let global, local =
                  Program_spec.launch_dims spec [ "threads", 11 ]
                in
                equal (array int) [| 1; 1; 11 |] global;
                equal (option pass) None local
            | _ -> failwith "expected flat thread launch metadata");
          test "core_id is explicit runtime metadata" (fun () ->
            let arg = define_var "arg" 0 9 in
            let cid = define_var "core_id" 0 7 in
            let spec = spec_of [ arg; cid ] in
            match Program_spec.core_id spec with
            | None -> failwith "expected core_id metadata"
            | Some core_id ->
                equal int 1 core_id.var_index;
                equal int 8 (Program_spec.thread_count core_id);
                begin match Program_spec.launch_kind spec with
                | Program_spec.Serial -> ()
                | _ -> failwith "core_id should not synthesize GPU launch metadata"
                end;
                let global, local = Program_spec.launch_dims spec [] in
                equal (array int) [| 8; 1; 1 |] global;
                begin match local with
                | None -> failwith "expected serial local dims"
                | Some local -> equal (array int) [| 1; 1; 1 |] local
                end);
          test "program_info mirrors extracted metadata" (fun () ->
            let m = define_var "m" 1 32 in
            let p0 = param 0 Dtype.Val.float32 in
            let p1 = param 1 Dtype.Val.float32 in
            let c0 = i32 0 in
            let groups = mul m (i32 4) in
            let gid = special (Gpu_dim.Group_id 0) groups in
            let out_idx = index p0 c0 in
            let in_idx = index p1 c0 in
            let ld = load in_idx in
            let st = store out_idx ld in
            let spec =
              spec_of [ m; p0; p1; c0; groups; gid; out_idx; in_idx; ld; st ]
            in
            let info = Program_spec.program_info spec in
            equal string "kern" info.name;
            equal (list string) [] info.aux;
            equal (list int) [ 0; 1 ] info.globals;
            equal (list int) [ 0 ] info.outs;
            equal (list int) [ 1 ] info.ins;
            equal int 1 (List.length info.vars);
            begin match info.global_size with
            | [ U.Launch_sym u; U.Launch_int 1; U.Launch_int 1 ] ->
                is_true (U.equal groups u)
            | _ -> failwith "expected symbolic launch metadata"
            end;
            equal (option (list int)) (Some [ 1; 1; 1 ]) info.local_size);
          test "program_info preserves renderer aux metadata" (fun () ->
            let spec = spec_of ~aux:[ "((0,dtypes.float.ptr(-1)))" ] [] in
            let info = Program_spec.program_info spec in
            equal (list string) [ "((0,dtypes.float.ptr(-1)))" ] info.aux);
          test "duplicate launch axis is rejected" (fun () ->
            let c4 = i32 4 in
            let gid0 = special (Gpu_dim.Group_id 0) c4 in
            let gid1 = special (Gpu_dim.Group_id 0) c4 in
            raises_invalid_arg "group_id axis 0 appears more than once"
              (fun () -> ignore (spec_of [ c4; gid0; gid1 ])));
          test "mixed launch models are rejected" (fun () ->
            let c4 = i32 4 in
            let group = special (Gpu_dim.Group_id 0) c4 in
            let flat = special (Gpu_dim.Global_idx 1) c4 in
            raises_invalid_arg
              "launch metadata cannot mix flat-thread and thread-group specials"
              (fun () -> ignore (spec_of [ c4; group; flat ])));
          test "core_id lower bound must be zero" (fun () ->
            let cid = define_var "core_id" 2 7 in
            raises_invalid_arg "core_id must have lower bound 0"
              (fun () -> ignore (spec_of [ cid ])));
          test "exact estimates can be forwarded" (fun () ->
            let estimates =
              E.of_uop U.{ ops = Int 7; lds = Int 11; mem = Int 13 }
            in
            let est = Program_spec.estimates (empty_spec ~estimates ()) in
            expect_int_estimate "ops" 7 est.ops;
            expect_int_estimate "lds" 11 est.lds;
            expect_int_estimate "mem" 13 est.mem);
          test "symbolic estimates require caller handling" (fun () ->
            let sym_node = U.variable ~name:"n" ~min_val:1 ~max_val:100 () in
            let estimates =
              E.of_uop U.{ ops = Sym sym_node; lds = Int 1; mem = Int 2 }
            in
            match estimates.ops with
            | E.Symbolic _ -> ()
            | _ -> failwith "expected symbolic ops estimate");
        ];
      group "Estimates.of_program"
        [
          test "counts basic ALU ops" (fun () ->
            let a = f32 1.0 in
            let b = f32 2.0 in
            let c = add a b in
            let d = neg a in
            let est = E.of_program [ a; b; c; d ] in
            expect_int_estimate "ops" 2 est.ops);
          test "mulacc counts as 2 FLOPs" (fun () ->
            let a = f32 1.0 in
            let b = f32 2.0 in
            let c = f32 3.0 in
            let d = mulacc a b c in
            let est = E.of_program [ a; b; c; d ] in
            expect_int_estimate "ops" 2 est.ops);
          test "loop multiplier stacks" (fun () ->
            let c10 = i32 10 in
            let r = range c10 in
            let a = f32 1.0 in
            let body = add a a in
            let end_ = U.end_ ~value:body ~ranges:[ r ] in
            let est = E.of_program [ c10; r; a; body; end_ ] in
            expect_int_estimate "ops" 10 est.ops);
          test "special multiplier stacks" (fun () ->
            let c8 = i32 8 in
            let idx = special (Gpu_dim.Global_idx 0) c8 in
            let a = f32 1.0 in
            let body = add a a in
            let est = E.of_program [ c8; idx; a; body ] in
            expect_int_estimate "ops" 8 est.ops);
          test "load/store tracks lds and memory bytes" (fun () ->
            let p0 = param 0 Dtype.Val.float32 in
            let c0 = i32 0 in
            let idx = index p0 c0 in
            let ld = load idx in
            let st = store idx ld in
            let est = E.of_program [ p0; c0; idx; ld; st ] in
            expect_int_estimate "lds" 8 est.lds;
            expect_int_estimate "mem" 8 est.mem);
          test "index arithmetic excluded from FLOPs" (fun () ->
            let p0 = param 0 Dtype.Val.float32 in
            let c0 = i32 0 in
            let c1 = i32 1 in
            let idx_expr = add c0 c1 in
            let idx = index p0 idx_expr in
            let ld = load idx in
            let est = E.of_program [ p0; c0; c1; idx_expr; idx; ld ] in
            expect_int_estimate "ops" 0 est.ops);
        ];
    ]
