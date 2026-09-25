(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop

module U = Uop
module E = Program_spec.Estimates

let param slot dt =
  U.param ~slot ~dtype:dt ~shape:(U.const_int 1) ~addrspace:Dtype.Global ()

let buffer slot dt =
  U.buffer ~slot ~dtype:dt ~shape:(U.const_int 1) ~addrspace:Dtype.Global ()

let i32 n = U.const (Const.int Dtype.int32 n)
let f32 x = U.const (Const.float Dtype.float32 x)
let define_var name lo hi =
  U.variable ~name ~min_val:lo ~max_val:hi ~dtype:Dtype.int32 ()
let index ptr idx = U.index ~ptr ~idxs:[ idx ] ()
let load src = U.load ~src ()
let store dst value = U.store ~dst ~value ()
let add lhs rhs = U.alu_binary ~op:Ops.Add ~lhs ~rhs
let mul lhs rhs = U.alu_binary ~op:Ops.Mul ~lhs ~rhs
let floordiv lhs rhs = U.alu_binary ~op:Ops.Floordiv ~lhs ~rhs
let floormod lhs rhs = U.alu_binary ~op:Ops.Floormod ~lhs ~rhs
let neg src = U.alu_unary ~op:Ops.Neg ~src
let mulacc a b c = U.alu_ternary ~op:Ops.Mulacc ~a ~b ~c
let range size = U.range ~size ~axis:0 ~kind:Axis_type.Weak ()

(* A tensor-core node with scalar operands: [Estimates] reads the FLOP count
   off [dims] and [threads], never off the operand widths. *)
let wmma ~dims ~threads =
  let info : U.wmma_info =
    {
      dims;
      dtype_in = Dtype.float16;
      threads;
      tc_upcast_axes = None;
    }
  in
  let a = U.const (Const.float Dtype.float16 1.0) in
  let c = f32 0.0 in
  U.wmma ~a ~b:a ~c ~info
let special dim size =
  U.special ~name:(Gpu_dim.to_special_name dim) ~size ~dtype:(U.dtype size) ()

let spec_of ?estimates program =
  Program_spec.of_program ~name:"kern" ~src:"" ~device:"CPU" ?estimates program

let empty_spec ?estimates () = spec_of ?estimates []

let expect_int_estimate label expected = function
  | E.Int n -> equal int expected n ~msg:label
  | E.Symbolic _ -> failwith (label ^ ": expected exact int estimate")

let check_launch_dimension dimension bindings expected =
  let gid = special (Gpu_dim.Group_id 0) dimension in
  let spec = spec_of (U.toposort gid) in
  let global = fst (Program_spec.launch_dims spec bindings) in
  equal (array int) [|expected; 1; 1|] global;
  let global = fst (U.program_launch_dims (Program_spec.program_info spec)
      ~var_vals:bindings) in
  is_true (global = [U.Launch_value_int expected; U.Launch_value_int 1; U.Launch_value_int 1])

let () =
  run "Program_spec"
    [
      group "Extraction"
        [
          test "incomplete scalar metadata is rejected before ABI construction" (fun () ->
            let output = param 0 Dtype.int64 in
            let at = index output (U.const_int 0) in
            let cases = [
              U.param ~slot:1 ~dtype:Dtype.int64 ~addrspace:Dtype.Alu ~name:"value" ();
              U.param ~slot:1 ~dtype:Dtype.int64 ~addrspace:Dtype.Alu
                ~vmin_vmax:(Bound.zero, Bound.int 7) ();
              U.param ~slot:1 ~dtype:Dtype.int64 ~addrspace:Dtype.Alu ()] in
            List.iter (fun value ->
                raises (Invalid_argument
                    "Program_spec: scalar parameter slot 1 requires a name and bounds")
                  (fun () -> spec_of [output; value; at; store at value])) cases);
          test "bounded scalar metadata retains the complete ABI" (fun () ->
            let output = param 0 Dtype.int64 in
            let value = U.param ~slot:1 ~dtype:Dtype.int64 ~addrspace:Dtype.Alu
                ~name:"value" ~vmin_vmax:(Bound.zero, Bound.int 7) () in
            let at = index output (U.const_int 0) in
            let spec = spec_of [output; value; at; store at value]
                |> Program_spec.with_lib Bytes.empty in
            let obj = Program_spec.to_elf spec in
            equal (list int) [0; 1]
              (List.map (fun (arg : Tiny_elf.argument) -> arg.slot) obj.signature);
            equal (list string) ["value"]
              (List.map (fun (v : Program_spec.var) -> v.name) (Program_spec.vars spec)));
          test "reads and writes are deduplicated" (fun () ->
            let p0 = param 0 Dtype.float32 in
            let p1 = param 1 Dtype.float32 in
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
            let p0 = param 0 Dtype.float32 in
            let c0 = i32 0 in
            let idx = index p0 c0 in
            let dep = U.barrier () in
            let sequenced = U.after ~src:idx ~deps:[ dep ] in
            let casted = U.cast ~src:sequenced ~dtype:(U.dtype sequenced) in
            let ld = load casted in
            let spec = spec_of [ p0; c0; idx; dep; sequenced; casted; ld ] in
            equal (list int) [ 0 ] (Program_spec.ins spec));
          test "buffer args are treated as globals" (fun () ->
            let b0 = buffer 0 Dtype.float32 in
            let b1 = buffer 1 Dtype.float32 in
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
          test "missing launch variables identify the program" (fun () ->
            let n = define_var "n" 1 32 in
            let gid = special (Gpu_dim.Group_id 0) n in
            raises_match
              (function
                | Invalid_argument msg ->
                    msg = "program \"kern\": sym_infer: missing variable \"n\""
                | _ -> false)
              (fun () -> Program_spec.launch_dims (spec_of [ n; gid ]) []));
          test "launch dimensions retain exact intermediate products" (fun () ->
            let variable name = U.variable ~name ~min_val:1 ~max_val:max_int () in
            let n = variable "launch_n" and m = variable "launch_m" in
            let dimension = U.O.((n * m) // U.const_int max_int) in
            let value = 1 lsl 32 in
            check_launch_dimension dimension ["launch_n", value; "launch_m", value] 4);
          test "launch dimensions preserve Python signed shifts" (fun () ->
            let n = U.variable ~name:"shift_n" ~min_val:(-1000) ~max_val:1000 () in
            let shifted = U.alu_binary ~op:Ops.Shr ~lhs:n ~rhs:(U.const_int 1) in
            check_launch_dimension U.O.(shifted + U.const_int 10) ["shift_n", -7] 6);
          test "launch casts convert values without storage narrowing" (fun () ->
            let n = U.variable ~name:"cast_n" ~min_val:(-1000) ~max_val:(1 lsl 30) () in
            check_launch_dimension (U.cast ~src:n ~dtype:Dtype.int8) ["cast_n", 300] 300;
            let floating = U.cast ~src:n ~dtype:Dtype.float32 in
            check_launch_dimension (U.cast ~src:floating ~dtype:Dtype.weakint)
              ["cast_n", 16_777_217] 16_777_217;
            let boolean = U.cast ~src:n ~dtype:Dtype.bool in
            let integer = U.cast ~src:boolean ~dtype:Dtype.weakint in
            check_launch_dimension U.O.(integer + U.const_int 1) ["cast_n", -7] 2);
          test "launch bitcasts retain the source representation" (fun () ->
            let bits = U.variable ~name:"bits" ~min_val:0 ~max_val:0x7fff_ffff
                ~dtype:Dtype.int32 () in
            let floating = U.bitcast ~src:bits ~dtype:Dtype.float32 in
            check_launch_dimension (U.cast ~src:floating ~dtype:Dtype.weakint)
              ["bits", 0x3f80_0000] 1);
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
                is_none local
            | _ -> failwith "expected flat thread launch metadata");
          test "core_id is an ordinary scalar" (fun () ->
            let arg = define_var "arg" 0 9 in
            let cid = define_var "core_id" 2 7 in
            let spec = spec_of [ arg; cid ] in
            equal (list string) [ "arg"; "core_id" ]
              (List.map (fun (v : Program_spec.var) -> v.name) (Program_spec.vars spec));
            is_true (Program_spec.launch_kind spec = Program_spec.Serial);
            let global, local = Program_spec.launch_dims spec [] in
            equal (array int) [| 1; 1; 1 |] global;
            equal (option (array int)) (Some [| 1; 1; 1 |]) local);
          test "program_info mirrors extracted metadata" (fun () ->
            let m = define_var "m" 1 32 in
            let p0 = param 0 Dtype.float32 in
            let p1 = param 1 Dtype.float32 in
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
            equal (list int) [ 0; 1 ] info.globals;
            equal (list int) [ 0 ] info.outs;
            equal (list int) [ 1 ] info.ins;
            equal int 1 (List.length info.vars);
            begin match info.global_size with
            | [ U.Launch_sym u; U.Launch_int 1; U.Launch_int 1 ] ->
                is_true (U.equal groups u)
            | _ -> failwith "expected symbolic launch metadata"
            end;
            is_true (info.local_size = [ U.Launch_int 1; U.Launch_int 1; U.Launch_int 1 ]));
          test "program_info preserves symbolic local dimensions" (fun () ->
            let n = define_var "n" 1 32 in
            let lid = special (Gpu_dim.Local_id 0) n in
            let info = Program_spec.program_info (spec_of [ n; lid ]) in
            let _, local = U.program_launch_dims info ~var_vals:[ "n", 8 ] in
            is_true (local = [ U.Launch_value_int 8; U.Launch_value_int 1; U.Launch_value_int 1 ]));
          test "duplicate launch axis is rejected" (fun () ->
            let c4 = i32 4 in
            let gid0 = special (Gpu_dim.Group_id 0) c4 in
            let gid1 = special (Gpu_dim.Group_id 0) c4 in
            raises (Invalid_argument "group_id axis 0 appears more than once")
              (fun () -> ignore (spec_of [ c4; gid0; gid1 ])));
          test "mixed launch models are rejected" (fun () ->
            let c4 = i32 4 in
            let group = special (Gpu_dim.Group_id 0) c4 in
            let flat = special (Gpu_dim.Global_idx 1) c4 in
            raises
              (Invalid_argument
                 "launch metadata cannot mix flat-thread and thread-group \
                  specials")
              (fun () -> ignore (spec_of [ c4; group; flat ])));
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
          test "wmma counts 2*M*N*K per warp, divided across threads"
            (fun () ->
              (* One tensor-core call does 2*M*N*K FLOPs for the whole warp,
                 and the estimate is per thread: 2*8*16*16/32 = 128. *)
              let w = wmma ~dims:(8, 16, 16) ~threads:32 in
              let est = E.of_program (U.toposort w) in
              expect_int_estimate "ops" 128 est.ops);
          test "wmma thread count divides the FLOP factor" (fun () ->
            (* Same shape on a 64-wide warp halves the per-thread count,
               so the divisor is read from the node rather than assumed. *)
            let w = wmma ~dims:(8, 16, 16) ~threads:64 in
            let est = E.of_program (U.toposort w) in
            expect_int_estimate "ops" 64 est.ops);
          test "wmma FLOPs scale with the loop multiplier" (fun () ->
            let c10 = i32 10 in
            let r = range c10 in
            let w = wmma ~dims:(8, 16, 16) ~threads:32 in
            let end_ = U.end_ ~value:w ~ranges:[ r ] in
            let est =
              E.of_program ((c10 :: r :: U.toposort w) @ [ end_ ])
            in
            expect_int_estimate "ops" 1280 est.ops);
          test "loop multiplier stacks" (fun () ->
            let c10 = i32 10 in
            let r = range c10 in
            let a = f32 1.0 in
            let body = add a a in
            let end_ = U.end_ ~value:body ~ranges:[ r ] in
            let est = E.of_program [ c10; r; a; body; end_ ] in
            expect_int_estimate "ops" 10 est.ops);
          test "an unbounded loop contributes no multiplier" (fun () ->
            (* A void range has no trip count, so its body is counted once
               rather than at the size sitting in its source. *)
            let c10 = i32 10 in
            let r = U.loop ~axis:0 in
            let a = f32 1.0 in
            let body = add a a in
            let end_ = U.backedge ~body ~loop:r ~cond:(U.const_bool false) in
            let est = E.of_program [ c10; r; a; body; end_ ] in
            expect_int_estimate "ops" 1 est.ops);
          test "a loop bounded by a loaded value counts at its bound"
            (fun () ->
              (* The id-bounded loop of a gated kernel: its trip count is
                 known only when the kernel runs. *)
              let p0 = param 0 Dtype.int32 in
              let c0 = i32 0 in
              let id = load (index p0 c0) in
              let selects = U.alu_binary ~op:Ops.Cmplt ~lhs:c0 ~rhs:id in
              let size = U.alu_ternary ~op:Ops.Where ~a:selects ~b:(i32 6) ~c:c0 in
              let r = range size in
              let a = f32 1.0 in
              let body = add a a in
              let end_ = U.end_ ~value:body ~ranges:[ r ] in
              let est = E.of_program [ r; a; body; end_ ] in
              expect_int_estimate "ops" 6 est.ops);
          test "special multiplier stacks" (fun () ->
            let c8 = i32 8 in
            let idx = special (Gpu_dim.Global_idx 0) c8 in
            let a = f32 1.0 in
            let body = add a a in
            let est = E.of_program [ c8; idx; a; body ] in
            expect_int_estimate "ops" 8 est.ops);
          test "load/store tracks lds and memory bytes" (fun () ->
            let p0 = param 0 Dtype.float32 in
            let c0 = i32 0 in
            let idx = index p0 c0 in
            let ld = load idx in
            let st = store idx ld in
            let est = E.of_program [ p0; c0; idx; ld; st ] in
            expect_int_estimate "lds" 8 est.lds;
            expect_int_estimate "mem" 8 est.mem);
          test "index arithmetic excluded from FLOPs" (fun () ->
            let p0 = param 0 Dtype.float32 in
            let c0 = i32 0 in
            let c1 = i32 1 in
            let idx_expr = add c0 c1 in
            let idx = index p0 idx_expr in
            let ld = load idx in
            let est = E.of_program [ p0; c0; c1; idx_expr; idx; ld ] in
            expect_int_estimate "ops" 0 est.ops);
          test "repeated reads cap memory at buffer size" (fun () ->
            let p0 = param 0 Dtype.float32 in
            let c0 = i32 0 in
            let c10 = i32 10 in
            let r = range c10 in
            let idx = index p0 c0 in
            let ld = load idx in
            let end_ = U.end_ ~value:ld ~ranges:[ r ] in
            let est = E.of_program [ p0; c0; c10; r; idx; ld; end_ ] in
            (* The loop reads the 4-byte buffer ten times: lds counts every
               access, but mem counts the buffer footprint only once. *)
            expect_int_estimate "lds" 40 est.lds;
            expect_int_estimate "mem" 4 est.mem);
        ];
    ]
