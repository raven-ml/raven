(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_ir
module P = Program

let global_ptr dt = Dtype.ptr_of dt ~addrspace:Global ~size:(-1)

let spec_of ?(estimates : Program_spec.Estimates.t option) b =
  Program_spec.of_program ~name:"kern" ?estimates (P.finish b)

let empty_spec ?estimates () = spec_of ?estimates (P.create ())

let () =
  run "Program_spec"
    [
      group "Extraction"
        [
          test "reads and writes are deduplicated" (fun () ->
            let ptr = global_ptr Dtype.float32 in
            let b = P.create () in
            let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
            let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
            let c0 =
              P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 })
            in
            let idx1 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
            let idx2 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
            let ld1 = P.emit b (Load { src = idx2; alt = None; dtype = Dtype.float32 }) in
            let ld2 = P.emit b (Load { src = idx2; alt = None; dtype = Dtype.float32 }) in
            let _ = P.emit b (Store { dst = idx1; value = ld1 }) in
            let _ = P.emit b (Store { dst = idx1; value = ld2 }) in
            let spec = spec_of b in
            equal (list int) [ 0 ] (Program_spec.outs spec);
            equal (list int) [ 1 ] (Program_spec.ins spec));
          test "thread-group launch expressions are preserved" (fun () ->
            let b = P.create () in
            let dv = P.emit b (Define_var { name = "m"; lo = 1; hi = 32; dtype = Dtype.int32 }) in
            let c4 =
              P.emit b (Const { value = Const.int Dtype.int32 4; dtype = Dtype.int32 })
            in
            let mul = P.emit b (Binary { op = `Mul; lhs = dv; rhs = c4; dtype = Dtype.int32 }) in
            let _ = P.emit b (Special { dim = Special_dim.Group_id 0; size = mul; dtype = Dtype.index }) in
            let _ = P.emit b (Special { dim = Special_dim.Local_id 1; size = dv; dtype = Dtype.index }) in
            let spec = spec_of b in
            match Program_spec.launch_kind spec with
            | Program_spec.Thread_groups ->
                let global, local = Program_spec.launch_dims spec [ 3 ] in
                equal (array int) [| 12; 1; 1 |] global;
                begin match local with
                | None -> failwith "expected local dims"
                | Some local -> equal (array int) [| 1; 3; 1 |] local
                end
            | _ -> failwith "expected thread-group launch metadata");
          test "launch variables are keyed by instruction identity" (fun () ->
            let b = P.create () in
            let _ = P.emit b (Define_var { name = "n"; lo = 0; hi = 7; dtype = Dtype.int32 }) in
            let dv1 = P.emit b (Define_var { name = "n"; lo = 0; hi = 15; dtype = Dtype.int32 }) in
            let _ = P.emit b (Special { dim = Special_dim.Group_id 0; size = dv1; dtype = Dtype.index }) in
            let global, _local = Program_spec.launch_dims (spec_of b) [ 3; 9 ] in
            equal (array int) [| 9; 1; 1 |] global);
          test "global idx uses thread launch" (fun () ->
            let b = P.create () in
            let dv = P.emit b (Define_var { name = "threads"; lo = 1; hi = 64; dtype = Dtype.int32 }) in
            let _ = P.emit b (Special { dim = Special_dim.Global_idx 2; size = dv; dtype = Dtype.index }) in
            let spec = spec_of b in
            match Program_spec.launch_kind spec with
            | Program_spec.Threads ->
                let global, local = Program_spec.launch_dims spec [ 11 ] in
                equal (array int) [| 1; 1; 11 |] global;
                equal (option pass) None local
            | _ -> failwith "expected flat thread launch metadata");
          test "core_id is explicit runtime metadata" (fun () ->
            let b = P.create () in
            let _ = P.emit b (Define_var { name = "arg"; lo = 0; hi = 9; dtype = Dtype.int32 }) in
            let _ = P.emit b (Define_var { name = "core_id"; lo = 0; hi = 7; dtype = Dtype.int32 }) in
            let spec = spec_of b in
            match Program_spec.core_id spec with
            | None -> failwith "expected core_id metadata"
            | Some core_id ->
                equal int 1 core_id.var_index;
                equal int 8 (Program_spec.thread_count core_id);
                begin match Program_spec.launch_kind spec with
                | Program_spec.Serial -> ()
                | _ -> failwith "core_id should not synthesize GPU launch metadata"
                end);
          test "duplicate launch axis is rejected" (fun () ->
            let b = P.create () in
            let c4 =
              P.emit b (Const { value = Const.int Dtype.int32 4; dtype = Dtype.int32 })
            in
            let _ = P.emit b (Special { dim = Special_dim.Group_id 0; size = c4; dtype = Dtype.index }) in
            let _ = P.emit b (Special { dim = Special_dim.Group_id 0; size = c4; dtype = Dtype.index }) in
            raises_invalid_arg "group_id axis 0 appears more than once" (fun () ->
                ignore (spec_of b)));
          test "mixed launch models are rejected" (fun () ->
            let b = P.create () in
            let c4 =
              P.emit b (Const { value = Const.int Dtype.int32 4; dtype = Dtype.int32 })
            in
            let _ = P.emit b (Special { dim = Special_dim.Group_id 0; size = c4; dtype = Dtype.index }) in
            let _ = P.emit b (Special { dim = Special_dim.Global_idx 1; size = c4; dtype = Dtype.index }) in
            raises_invalid_arg
              "launch metadata cannot mix flat-thread and thread-group specials"
              (fun () -> ignore (spec_of b)));
          test "core_id lower bound must be zero" (fun () ->
            let b = P.create () in
            let _ = P.emit b (Define_var { name = "core_id"; lo = 2; hi = 7; dtype = Dtype.int32 }) in
            raises_invalid_arg "core_id must have lower bound 0" (fun () ->
                ignore (spec_of b)));
          test "exact estimates are forwarded" (fun () ->
            let estimates =
              Program_spec.Estimates.of_kernel
                Kernel.{ ops = Int 7; lds = Int 11; mem = Int 13 }
            in
            let est = Program_spec.estimates (empty_spec ~estimates ()) in
            begin match est.ops with
            | Program_spec.Estimates.Int 7 -> ()
            | _ -> failwith "expected exact ops estimate"
            end;
            begin match est.lds with
            | Program_spec.Estimates.Int 11 -> ()
            | _ -> failwith "expected exact lds estimate"
            end;
            begin match est.mem with
            | Program_spec.Estimates.Int 13 -> ()
            | _ -> failwith "expected exact mem estimate"
            end);
          test "symbolic estimates require caller handling" (fun () ->
            let estimates =
              Program_spec.Estimates.of_kernel
                Kernel.{ ops = Symbolic "n"; lds = Int 1; mem = Int 2 }
            in
            match estimates.ops with
            | Program_spec.Estimates.Symbolic "n" -> ()
            | _ -> failwith "expected symbolic ops estimate");
        ];
      group "Estimates.of_program"
        [
          test "counts basic ALU ops" (fun () ->
            let b = P.create () in
            let a =
              P.emit b (Const { value = Const.float Dtype.float32 1.0; dtype = Dtype.float32 })
            in
            let c =
              P.emit b (Const { value = Const.float Dtype.float32 2.0; dtype = Dtype.float32 })
            in
            let _ = P.emit b (Binary { op = `Add; lhs = a; rhs = c; dtype = Dtype.float32 }) in
            let _ = P.emit b (Unary { op = `Neg; src = a; dtype = Dtype.float32 }) in
            let est = Program_spec.Estimates.of_program (P.finish b) in
            begin match est.ops with
            | Program_spec.Estimates.Int 2 -> ()
            | Program_spec.Estimates.Int n ->
                failwith (Printf.sprintf "expected 2 FLOPs, got %d" n)
            | _ -> failwith "expected exact int ops estimate"
            end);
          test "mulacc counts as 2 FLOPs" (fun () ->
            let b = P.create () in
            let a =
              P.emit b (Const { value = Const.float Dtype.float32 1.0; dtype = Dtype.float32 })
            in
            let c =
              P.emit b (Const { value = Const.float Dtype.float32 2.0; dtype = Dtype.float32 })
            in
            let d =
              P.emit b (Const { value = Const.float Dtype.float32 3.0; dtype = Dtype.float32 })
            in
            let _ =
              P.emit b (Ternary { op = `Mulacc; a; b = c; c = d; dtype = Dtype.float32 })
            in
            let est = Program_spec.Estimates.of_program (P.finish b) in
            begin match est.ops with
            | Program_spec.Estimates.Int 2 -> ()
            | Program_spec.Estimates.Int n ->
                failwith (Printf.sprintf "expected 2 FLOPs, got %d" n)
            | _ -> failwith "expected exact int ops estimate"
            end);
          test "loop multiplier stacks" (fun () ->
            let b = P.create () in
            let c10 =
              P.emit b (Const { value = Const.int Dtype.int32 10; dtype = Dtype.int32 })
            in
            let range =
              P.emit b
                (Range
                   { size = c10; dtype = Dtype.int32; axis = 0; sub = [];
                     kind = Axis_kind.Loop })
            in
            let a =
              P.emit b (Const { value = Const.float Dtype.float32 1.0; dtype = Dtype.float32 })
            in
            let add =
              P.emit b (Binary { op = `Add; lhs = a; rhs = a; dtype = Dtype.float32 })
            in
            let _ = P.emit b (End_range { dep = add; range }) in
            let est = Program_spec.Estimates.of_program (P.finish b) in
            begin match est.ops with
            | Program_spec.Estimates.Int 10 -> ()
            | Program_spec.Estimates.Int n ->
                failwith (Printf.sprintf "expected 10 FLOPs (1 op * 10 iters), got %d" n)
            | _ -> failwith "expected exact int ops estimate"
            end);
          test "load/store tracks lds bytes" (fun () ->
            let ptr = global_ptr Dtype.float32 in
            let b = P.create () in
            let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
            let c0 =
              P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 })
            in
            let idx = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
            let ld = P.emit b (Load { src = idx; alt = None; dtype = Dtype.float32 }) in
            let _ = P.emit b (Store { dst = idx; value = ld }) in
            let est = Program_spec.Estimates.of_program (P.finish b) in
            begin match est.lds with
            | Program_spec.Estimates.Int n when n = 4 + 4 -> ()
            | Program_spec.Estimates.Int n ->
                failwith (Printf.sprintf "expected 8 lds bytes (4 load + 4 store), got %d" n)
            | _ -> failwith "expected exact int lds estimate"
            end);
          test "index arithmetic excluded from FLOPs" (fun () ->
            let ptr = global_ptr Dtype.float32 in
            let b = P.create () in
            let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
            let c0 =
              P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 })
            in
            let c1 =
              P.emit b (Const { value = Const.int Dtype.int32 1; dtype = Dtype.int32 })
            in
            (* This add is used as an index operand — should be excluded. *)
            let idx_expr =
              P.emit b (Binary { op = `Add; lhs = c0; rhs = c1; dtype = Dtype.int32 })
            in
            let idx =
              P.emit b
                (Index { ptr = p0; idxs = [ idx_expr ]; gate = None; dtype = ptr })
            in
            let _ = P.emit b (Load { src = idx; alt = None; dtype = Dtype.float32 }) in
            let est = Program_spec.Estimates.of_program (P.finish b) in
            begin match est.ops with
            | Program_spec.Estimates.Int 0 -> ()
            | Program_spec.Estimates.Int n ->
                failwith (Printf.sprintf "expected 0 FLOPs (index add excluded), got %d" n)
            | _ -> failwith "expected exact int ops estimate"
            end);
        ];
    ]
