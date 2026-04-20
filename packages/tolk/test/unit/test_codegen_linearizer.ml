(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_ir
module K = Kernel
module P = Program

(* Helpers *)

let dt = Dtype.Val.float32
let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)
let ptr = global_ptr dt

let i32 n = K.const (Const.int Dtype.Val.int32 n)
let f32 x = K.const (Const.float Dtype.Val.float32 x)

let loop_range ~axis size =
  K.range ~size ~axis ~kind:Axis_kind.Loop ~dtype:Dtype.Val.int32 ()

let reduce_range ~axis size =
  K.range ~size ~axis ~kind:Axis_kind.Reduce ~dtype:Dtype.Val.int32 ()

let global_range ~axis size =
  K.range ~size ~axis ~kind:Axis_kind.Global ~dtype:Dtype.Val.int32 ()

let load_one_elem () =
  let p0 = K.param ~idx:0 ~dtype:ptr in
  K.load ~src:(K.index ~ptr:p0 ~idxs:[ i32 0 ] ()) ()

let contains haystack needle =
  let hl = String.length haystack and nl = String.length needle in
  if nl = 0 then true
  else if nl > hl then false
  else
    let rec loop i =
      if i > hl - nl then false
      else if String.sub haystack i nl = needle then true
      else loop (i + 1)
    in
    loop 0

let pp_view view = Format.asprintf "%a" P.pp_view view
let pp_program program = Format.asprintf "%a" P.pp program

let fail_view msg view =
  failwith (Printf.sprintf "%s: %s" msg (pp_view view))

let find_positions (program : P.t) pred =
  let acc = ref [] in
  P.iteri (fun i view -> if pred view then acc := i :: !acc) program;
  List.rev !acc

let find_unique_position label program pred =
  match find_positions program pred with
  | [ i ] -> i
  | xs ->
      failwith
        (Printf.sprintf "%s: expected one match, got %d\n%s" label
           (List.length xs) (pp_program program))

let count program pred =
  let n = ref 0 in
  P.iteri (fun _ view -> if pred view then incr n) program;
  !n

let linearize sink =
  let sink = Linearizer.pm_split_ends sink in
  let sink = Linearizer.pm_add_control_flow sink in
  Linearizer.linearize sink

let count_ranges prog = count prog (function P.Range _ -> true | _ -> false)

let count_end_ranges prog =
  count prog (function P.End_range _ -> true | _ -> false)

let find_ranges prog =
  find_positions prog (function P.Range _ -> true | _ -> false)

let find_end_ranges prog =
  find_positions prog (function P.End_range _ -> true | _ -> false)

let find_range ~axis prog =
  find_unique_position "range" prog (function
    | P.Range { axis = a; _ } -> a = axis
    | _ -> false)

let find_range_by_kind ~kind prog =
  find_unique_position "range" prog (function
    | P.Range { kind = k; _ } -> k = kind
    | _ -> false)

let find_load prog =
  find_unique_position "load" prog (function P.Load _ -> true | _ -> false)

let find_store prog =
  find_unique_position "store" prog (function P.Store _ -> true | _ -> false)

let raises_linearize substring fn =
  raises_match (function Failure msg -> contains msg substring | _ -> false) fn

let test_unlowered_rejected name build_node =
  raises_linearize (name ^ " must be lowered before linearize") (fun () ->
      ignore (linearize (K.sink [ build_node () ])))

let () =
  run "Linearizer"
    [
      group "Late kernel to program"
        [
          test "multi-range End lowers to nested End_range pairs" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let r0 = loop_range ~axis:0 (i32 2) in
            let r1 = loop_range ~axis:1 (i32 3) in
            let sum = K.binary ~op:`Add ~lhs:r0 ~rhs:r1 in
            let idx = K.index ~ptr:p0 ~idxs:[ sum ] () in
            let st = K.store ~dst:idx ~value:(f32 1.0) ~ranges:[] in
            let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
            let program = linearize (K.sink [ e ]) in
            P.validate program;
            equal int 2 (count_ranges program);
            equal int 2 (count_end_ranges program);
            let outer = find_range ~axis:0 program in
            let inner = find_range ~axis:1 program in
            let inner_end =
              find_unique_position "inner end" program (function
                | P.End_range { range } ->
                    (match P.view program range with
                     | P.Range { axis = 1; _ } -> true
                     | _ -> false)
                | _ -> false)
            in
            let outer_end =
              find_unique_position "outer end" program (function
                | P.End_range { range } ->
                    (match P.view program range with
                     | P.Range { axis = 0; _ } -> true
                     | _ -> false)
                | _ -> false)
            in
            is_true (outer < inner);
            is_true (inner < inner_end);
            is_true (inner_end < outer_end));
          test "outer-range loads are scheduled before entering inner ranges"
            (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let p1 = K.param ~idx:1 ~dtype:ptr in
            let r0 = loop_range ~axis:0 (i32 2) in
            let idx_in = K.index ~ptr:p0 ~idxs:[ r0 ] () in
            let ld = K.load ~src:idx_in () in
            let r1 = loop_range ~axis:1 (i32 3) in
            let sum = K.binary ~op:`Add ~lhs:r0 ~rhs:r1 in
            let idx_out = K.index ~ptr:p1 ~idxs:[ sum ] () in
            let st = K.store ~dst:idx_out ~value:ld ~ranges:[] in
            let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
            let program = linearize (K.sink [ e ]) in
            P.validate program;
            let load_pos = find_load program in
            is_true (find_range ~axis:0 program < load_pos);
            is_true (load_pos < find_range ~axis:1 program));
          test "After nodes stay in Program ownership after linearize"
            (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let idx = K.index ~ptr:p0 ~idxs:[ i32 0 ] () in
            let ld = K.load ~src:idx () in
            let af = K.after ~src:ld ~deps:[ f32 1.0 ] in
            let program = linearize (K.sink [ af ]) in
            P.validate program;
            let after_pos =
              find_unique_position "after" program (function
                | P.After _ -> true
                | _ -> false)
            in
            (match P.view program after_pos with
             | P.After { src; deps = [ dep ]; dtype } ->
                 is_true (Dtype.Val.equal dtype dt);
                 (match (P.view program src, P.view program dep) with
                  | P.Load _, P.Const { value; _ } ->
                      (match Const.view value with
                       | Float f -> is_true (f = 1.0)
                       | _ -> failwith "expected Float const")
                  | src_view, dep_view ->
                      failwith
                        (Printf.sprintf "unexpected After operands:\n%s\n%s"
                           (pp_view src_view) (pp_view dep_view)))
             | view -> fail_view "expected After" view));
          test "effect-only After nodes preserve store ordering" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let idx0 = K.index ~ptr:p0 ~idxs:[ i32 0 ] () in
            let st0 = K.store ~dst:idx0 ~value:(f32 1.0) ~ranges:[] in
            let idx1 = K.index ~ptr:p0 ~idxs:[ i32 1 ] () in
            let st1 = K.store ~dst:idx1 ~value:(f32 2.0) ~ranges:[] in
            let af = K.after ~src:st0 ~deps:[ st1 ] in
            let program = linearize (K.sink [ af ]) in
            let after_pos =
              find_unique_position "effect after" program (function
                | P.After _ -> true
                | _ -> false)
            in
            (match P.view program after_pos with
             | P.After { src; deps = [ dep ]; dtype } ->
                 is_true (Dtype.Val.equal dtype Dtype.Val.void);
                 (match (P.view program src, P.view program dep) with
                  | P.Store _, P.Store _ -> ()
                  | src_view, dep_view ->
                      failwith
                        (Printf.sprintf
                           "unexpected void After operands:\n%s\n%s"
                           (pp_view src_view) (pp_view dep_view)))
             | view -> fail_view "expected effect-only After" view));
          test "nested alt-index loads stay between the two ranges" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let p1 = K.param ~idx:1 ~dtype:ptr in
            let r0 = loop_range ~axis:0 (i32 2) in
            let gate = K.binary ~op:`Cmplt ~lhs:r0 ~rhs:(i32 2) in
            let idx_gated = K.index ~ptr:p0 ~idxs:[ r0 ] ~gate () in
            let ld = K.load ~src:idx_gated ~alt:(f32 2.0) () in
            let r1 = loop_range ~axis:1 (i32 3) in
            let add = K.binary ~op:`Add ~lhs:ld ~rhs:(f32 1.0) in
            let flat_idx =
              K.binary ~op:`Add
                ~lhs:(K.binary ~op:`Mul ~lhs:r0 ~rhs:(i32 3))
                ~rhs:r1
            in
            let idx_out = K.index ~ptr:p1 ~idxs:[ flat_idx ] () in
            let st = K.store ~dst:idx_out ~value:add ~ranges:[] in
            let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
            let program = linearize (K.sink [ e ]) in
            P.validate program;
            let outer = find_range ~axis:0 program in
            let inner = find_range ~axis:1 program in
            let load_pos = find_load program in
            is_true (outer < load_pos);
            is_true (load_pos < inner);
            is_true
              (List.exists
                 (fun pos ->
                   match P.view program pos with
                   | P.Binary { op = `Cmplt; _ } | P.Index _ -> true
                   | _ -> false)
                 (List.init (inner - outer - 1) (fun i -> outer + i + 1))));
          test "gated stores become IF/STORE/ENDIF"
            (fun () ->
            (* Gated stores are converted to IF/STORE/ENDIF in the linearizer. *)
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let gate = K.const (Const.bool true) in
            let idx = K.index ~ptr:p0 ~idxs:[ i32 0 ] ~gate () in
            let st = K.store ~dst:idx ~value:(f32 1.0) ~ranges:[] in
            let program = linearize (K.sink [ st ]) in
            P.validate program;
            equal int 1
              (count program (function P.If _ -> true | _ -> false));
            equal int 1
              (count program (function P.Endif _ -> true | _ -> false)));
          test "equal-priority nodes use structural tie-breaks" (fun () ->
            let sub = K.binary ~op:`Sub ~lhs:(i32 2) ~rhs:(i32 1) in
            let add = K.binary ~op:`Add ~lhs:(i32 1) ~rhs:(i32 2) in
            let program = linearize (K.sink [ add; sub ]) in
            P.validate program;
            let add_pos =
              find_unique_position "add" program (function
                | P.Binary { op = `Add; _ } -> true
                | _ -> false)
            in
            let sub_pos =
              find_unique_position "sub" program (function
                | P.Binary { op = `Sub; _ } -> true
                | _ -> false)
            in
            is_true (add_pos < sub_pos));
          test "late bias loads are scheduled after reduce end" (fun () ->
            let out_ptr = global_ptr dt in
            let in_ptr = global_ptr dt in
            let bias_ptr = global_ptr dt in
            let reg_ptr = Dtype.Ptr.create dt ~addrspace:Reg ~size:1 in
            let p0 = K.param ~idx:0 ~dtype:out_ptr in
            let p1 = K.param ~idx:1 ~dtype:in_ptr in
            let p2 = K.param ~idx:2 ~dtype:bias_ptr in
            let dreg = K.define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let reg_idx = K.index ~ptr:dreg ~idxs:[ i32 0 ] () in
            let st_init = K.store ~dst:reg_idx ~value:(f32 0.0) ~ranges:[] in
            let r0 = reduce_range ~axis:0 (i32 4) in
            let idx_in = K.index ~ptr:p1 ~idxs:[ r0 ] () in
            let st_acc =
              K.store ~dst:reg_idx ~value:(K.load ~src:idx_in ()) ~ranges:[]
            in
            let e = K.end_ ~value:st_acc ~ranges:[ r0 ] () in
            let acc_after = K.after ~src:dreg ~deps:[ e ] in
            let acc_val =
              K.load ~src:(K.index ~ptr:acc_after ~idxs:[ i32 0 ] ()) ()
            in
            let idx_bias = K.index ~ptr:p2 ~idxs:[ i32 0 ] () in
            let ld_bias = K.load ~src:idx_bias () in
            let add = K.binary ~op:`Add ~lhs:acc_val ~rhs:ld_bias in
            let idx_out = K.index ~ptr:p0 ~idxs:[ i32 0 ] () in
            let st_out = K.store ~dst:idx_out ~value:add ~ranges:[] in
            let program = linearize (K.sink [ st_init; st_out ]) in
            P.validate program;
            let last_end =
              find_end_ranges program |> List.rev |> List.hd
            in
            let bias_load =
              find_positions program (function
                | P.Load { src; _ } ->
                    (match P.view program src with
                     | P.Index { ptr; _ } ->
                         (match P.view program ptr with
                          | P.Param { idx = 2; _ } -> true
                          | _ -> false)
                     | _ -> false)
                | _ -> false)
              |> List.hd
            in
            is_true (last_end < bias_load));
          test "outer ops are placed before loop phis" (fun () ->
            let out_ptr = global_ptr dt in
            let in_ptr = global_ptr dt in
            let bias_ptr = global_ptr dt in
            let reg_ptr = Dtype.Ptr.create dt ~addrspace:Reg ~size:1 in
            let p0 = K.param ~idx:0 ~dtype:out_ptr in
            let p1 = K.param ~idx:1 ~dtype:in_ptr in
            let p2 = K.param ~idx:2 ~dtype:bias_ptr in
            let dreg = K.define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let reg_idx = K.index ~ptr:dreg ~idxs:[ i32 0 ] () in
            let st_init = K.store ~dst:reg_idx ~value:(f32 0.0) ~ranges:[] in
            let idx_bias = K.index ~ptr:p2 ~idxs:[ i32 0 ] () in
            let ld_bias = K.load ~src:idx_bias () in
            let r0 = reduce_range ~axis:0 (i32 4) in
            let idx_in = K.index ~ptr:p1 ~idxs:[ r0 ] () in
            let ld_in = K.load ~src:idx_in () in
            let add_in = K.binary ~op:`Add ~lhs:ld_in ~rhs:ld_bias in
            let st_reg = K.store ~dst:reg_idx ~value:add_in ~ranges:[] in
            let e = K.end_ ~value:st_reg ~ranges:[ r0 ] () in
            let acc_after = K.after ~src:dreg ~deps:[ e ] in
            let acc_val =
              K.load ~src:(K.index ~ptr:acc_after ~idxs:[ i32 0 ] ()) ()
            in
            let add_out = K.binary ~op:`Add ~lhs:acc_val ~rhs:ld_bias in
            let idx_out = K.index ~ptr:p0 ~idxs:[ i32 0 ] () in
            let st_out = K.store ~dst:idx_out ~value:add_out ~ranges:[] in
            let program = linearize (K.sink [ st_init; st_out ]) in
            P.validate program;
            let range_pos =
              find_unique_position "range" program (function
                | P.Range _ -> true
                | _ -> false)
            in
            let pre_range_loads =
              List.filter
                (fun pos ->
                  match P.view program pos with
                  | P.Load { src; _ } ->
                      (match P.view program src with
                       | P.Index { ptr; _ } ->
                           (match P.view program ptr with
                            | P.Param { idx = 2; _ } -> true
                            | _ -> false)
                       | _ -> false)
                  | _ -> false)
                (find_positions program (function
                  | P.Load _ -> true
                  | _ -> false))
            in
            equal int 1 (List.length pre_range_loads);
            is_true (List.hd pre_range_loads < range_pos));
          test "loop-carried reg stores stay inside the range" (fun () ->
            let input_ptr = global_ptr dt in
            let reg_ptr = Dtype.Ptr.create dt ~addrspace:Reg ~size:4 in
            let p0 = K.param ~idx:0 ~dtype:input_ptr in
            let dreg = K.define_reg ~size:4 ~dtype:reg_ptr ~slot:0 in
            let ri n = K.index ~ptr:dreg ~idxs:[ i32 n ] () in
            let st_init n = K.store ~dst:(ri n) ~value:(f32 0.0) ~ranges:[] in
            let r0 = loop_range ~axis:0 (i32 4) in
            let idx_in = K.index ~ptr:p0 ~idxs:[ r0 ] () in
            let ld = K.load ~src:idx_in () in
            let st_loop n =
              let add = K.binary ~op:`Add ~lhs:ld ~rhs:(f32 0.0) in
              K.store ~dst:(ri n) ~value:add ~ranges:[]
            in
            let e = K.end_ ~value:ld ~ranges:[ r0 ] () in
            let program =
              linearize
                (K.sink
                   [
                     st_init 0; st_init 1; st_init 2; st_init 3;
                     st_loop 0; st_loop 1; st_loop 2; st_loop 3;
                     e;
                   ])
            in
            P.validate program;
            let range_pos =
              find_unique_position "range" program (function
                | P.Range _ -> true
                | _ -> false)
            in
            let end_pos =
              find_unique_position "end" program (function
                | P.End_range _ -> true
                | _ -> false)
            in
            P.iteri
              (fun i view ->
                match view with
                | P.Store { dst; value } ->
                    (match P.view program dst with
                     | P.Index { ptr; _ } ->
                         (match P.view program ptr with
                          | P.Define_reg _ when i < range_pos ->
                              (match P.view program value with
                               | P.Const _ -> ()
                               | other ->
                                   fail_view "expected reg init before range"
                                     other)
                          | P.Define_reg _
                            when i > range_pos && i < end_pos ->
                              (match P.view program value with
                               | P.Binary { op = `Add; _ } -> ()
                               | other ->
                                   fail_view
                                     "expected ALU-fed reg store inside range"
                                     other)
                          | _ -> ())
                     | _ -> ())
                | _ -> ())
              program);
          test "gated loads without alts are rejected" (fun () ->
            raises_linearize
              "gated loads require an alt value before linearize"
              (fun () ->
                let p0 = K.param ~idx:0 ~dtype:ptr in
                let gate = K.const (Const.bool true) in
                let idx = K.index ~ptr:p0 ~idxs:[ i32 0 ] ~gate () in
                let ld = K.load ~src:idx () in
                ignore (linearize (K.sink [ ld ]))));
          test "load alts require gated indices" (fun () ->
            raises_linearize "Load alt requires gated Index" (fun () ->
                let p0 = K.param ~idx:0 ~dtype:ptr in
                let idx = K.index ~ptr:p0 ~idxs:[ i32 0 ] () in
                let ld = K.load ~src:idx ~alt:(f32 0.0) () in
                ignore (linearize (K.sink [ ld ]))));
          test "unlowered Reduce nodes are rejected" (fun () ->
            raises_linearize "Reduce must be lowered before linearize"
              (fun () ->
                let p0 = K.param ~idx:0 ~dtype:ptr in
                let r0 = reduce_range ~axis:0 (i32 4) in
                let idx = K.index ~ptr:p0 ~idxs:[ r0 ] () in
                let ld = K.load ~src:idx () in
                let red =
                  K.reduce ~op:`Add ~src:ld ~ranges:[ r0 ] ~dtype:dt
                in
                ignore (linearize (K.sink [ red ]))));
        ];
      group "CFG context"
        [
          test "sibling ends under sink are ordered" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let p1 = K.param ~idx:1 ~dtype:ptr in
            let r0 = loop_range ~axis:0 (i32 4) in
            let st0 =
              K.store
                ~dst:(K.index ~ptr:p0 ~idxs:[ r0 ] ())
                ~value:(f32 1.0) ~ranges:[]
            in
            let e0 = K.end_ ~value:st0 ~ranges:[ r0 ] () in
            let r1 = loop_range ~axis:1 (i32 4) in
            let st1 =
              K.store
                ~dst:(K.index ~ptr:p1 ~idxs:[ r1 ] ())
                ~value:(f32 1.0) ~ranges:[]
            in
            let e1 = K.end_ ~value:st1 ~ranges:[ r1 ] () in
            let program = linearize (K.sink [ e0; e1 ]) in
            P.validate program;
            equal int 2 (count_ranges program);
            equal int 2 (count_end_ranges program);
            let ranges = find_ranges program in
            let ends = find_end_ranges program in
            is_true (List.hd ends < List.nth ranges 1));
          test "three-range end exercises cfg nesting" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let r0 = loop_range ~axis:0 (i32 4) in
            let r1 = loop_range ~axis:1 (i32 4) in
            let r2 = loop_range ~axis:2 (i32 4) in
            let sum = K.binary ~op:`Add ~lhs:r0 ~rhs:r1 in
            let sum2 = K.binary ~op:`Add ~lhs:sum ~rhs:r2 in
            let idx = K.index ~ptr:p0 ~idxs:[ sum2 ] () in
            let st = K.store ~dst:idx ~value:(f32 1.0) ~ranges:[] in
            let e = K.end_ ~value:st ~ranges:[ r0; r1; r2 ] () in
            let program = linearize (K.sink [ e ]) in
            P.validate program;
            equal int 3 (count_ranges program);
            equal int 3 (count_end_ranges program);
            let ranges = find_ranges program in
            let ends = find_end_ranges program in
            is_true (List.nth ranges 0 < List.nth ranges 1);
            is_true (List.nth ranges 1 < List.nth ranges 2);
            is_true (List.nth ends 0 < List.nth ends 1);
            is_true (List.nth ends 1 < List.nth ends 2);
            is_true (List.nth ranges 2 < List.nth ends 0);
            is_true (List.nth ranges 1 < List.nth ranges 2);
            is_true (List.nth ends 0 < List.nth ends 1));
          test "two independent reduces are sequenced" (fun () ->
            let out_ptr = global_ptr dt in
            let in_ptr_a = global_ptr dt in
            let in_ptr_b = global_ptr dt in
            let reg_ptr = Dtype.Ptr.create dt ~addrspace:Reg ~size:1 in
            let p0 = K.param ~idx:0 ~dtype:out_ptr in
            let p1 = K.param ~idx:1 ~dtype:in_ptr_a in
            let p2 = K.param ~idx:2 ~dtype:in_ptr_b in
            let make_reduce dreg param axis =
              let ri = K.index ~ptr:dreg ~idxs:[ i32 0 ] () in
              let st_init = K.store ~dst:ri ~value:(f32 0.0) ~ranges:[] in
              let r = reduce_range ~axis (i32 4) in
              let idx = K.index ~ptr:param ~idxs:[ r ] () in
              let st_acc =
                K.store ~dst:ri ~value:(K.load ~src:idx ()) ~ranges:[]
              in
              let e = K.end_ ~value:st_acc ~ranges:[ r ] () in
              (st_init, e)
            in
            let dreg_a = K.define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let dreg_b = K.define_reg ~size:1 ~dtype:reg_ptr ~slot:1 in
            let st_init_a, e0 = make_reduce dreg_a p1 0 in
            let st_init_b, e1 = make_reduce dreg_b p2 1 in
            let af_a = K.after ~src:dreg_a ~deps:[ e0 ] in
            let af_b = K.after ~src:dreg_b ~deps:[ e1 ] in
            let ld_res_a =
              K.load ~src:(K.index ~ptr:af_a ~idxs:[ i32 0 ] ()) ()
            in
            let ld_res_b =
              K.load ~src:(K.index ~ptr:af_b ~idxs:[ i32 0 ] ()) ()
            in
            let sum = K.binary ~op:`Add ~lhs:ld_res_a ~rhs:ld_res_b in
            let idx_out = K.index ~ptr:p0 ~idxs:[ i32 0 ] () in
            let st_out = K.store ~dst:idx_out ~value:sum ~ranges:[] in
            let program =
              linearize (K.sink [ st_init_a; st_init_b; st_out ])
            in
            P.validate program;
            equal int 2 (count_ranges program);
            equal int 2 (count_end_ranges program);
            let ranges = find_ranges program in
            let ends = find_end_ranges program in
            is_true (List.nth ends 0 < List.nth ranges 1));
          test "three sibling ends are chain-ordered" (fun () ->
            let make_branch idx axis =
              let p = K.param ~idx ~dtype:(global_ptr dt) in
              let r = loop_range ~axis (i32 4) in
              let st =
                K.store
                  ~dst:(K.index ~ptr:p ~idxs:[ r ] ())
                  ~value:(f32 1.0) ~ranges:[]
              in
              K.end_ ~value:st ~ranges:[ r ] ()
            in
            let e0 = make_branch 0 0 in
            let e1 = make_branch 1 1 in
            let e2 = make_branch 2 2 in
            let program = linearize (K.sink [ e0; e1; e2 ]) in
            P.validate program;
            equal int 3 (count_ranges program);
            equal int 3 (count_end_ranges program);
            let ranges = find_ranges program in
            let ends = find_end_ranges program in
            is_true (List.nth ends 0 < List.nth ranges 1);
            is_true (List.nth ends 1 < List.nth ranges 2));
        ];
      group "Error paths"
        [
          test "unlowered Unroll is rejected" (fun () ->
            test_unlowered_rejected "Unroll" (fun () ->
                K.unroll ~src:(load_one_elem ()) ~axes:[ (0, 4) ] ~dtype:dt));
          test "unlowered Contract is rejected" (fun () ->
            test_unlowered_rejected "Contract" (fun () ->
                K.contract ~src:(load_one_elem ()) ~axes:[ (0, 4) ]
                  ~dtype:dt));
          test "unlowered Bufferize is rejected" (fun () ->
            test_unlowered_rejected "Bufferize" (fun () ->
                let buf_ptr = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1) in
                let opts : Kernel.bufferize_opts =
                  { device = None; addrspace = Global; removable = false }
                in
                K.bufferize ~src:(load_one_elem ()) ~ranges:[] ~dtype:buf_ptr
                  ~opts));
          test "unlowered Vcat is rejected" (fun () ->
            test_unlowered_rejected "Vcat" (fun () ->
                let v = K.vectorize ~srcs:[ f32 1.0; f32 2.0 ] in
                K.vcat ~srcs:[ v; v ]));
          test "unlowered Ptrcat is rejected" (fun () ->
            test_unlowered_rejected "Ptrcat" (fun () ->
                let p0 = K.param ~idx:0 ~dtype:ptr in
                let p1 = K.param ~idx:1 ~dtype:ptr in
                K.ptrcat ~srcs:[ p0; p1 ] ~dtype:ptr));
          test "unlowered Invalid_index is rejected" (fun () ->
            test_unlowered_rejected "Invalid_index" (fun () ->
                K.invalid_index ()));
          test "empty Group is rejected" (fun () ->
            raises_linearize "empty Group" (fun () ->
                ignore (linearize (K.sink [ K.group [] ]))));
        ];
      group "Priority ordering"
        [
          test "params ordered by index" (fun () ->
            let p2 = K.param ~idx:2 ~dtype:ptr in
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let p1 = K.param ~idx:1 ~dtype:ptr in
            let ld n p = K.load ~src:(K.index ~ptr:p ~idxs:[ i32 0 ] ()) () in
            let sum =
              K.binary ~op:`Add ~lhs:(ld 0 p0)
                ~rhs:
                  (K.binary ~op:`Add ~lhs:(ld 1 p1)
                     ~rhs:(ld 2 p2))
            in
            let program = linearize (K.sink [ sum ]) in
            P.validate program;
            let find_param idx =
              find_unique_position "param" program (function
                | P.Param { idx = i; _ } -> i = idx
                | _ -> false)
            in
            is_true (find_param 0 < find_param 1);
            is_true (find_param 1 < find_param 2));
          test "define_var ordered by name" (fun () ->
            let vb =
              K.define_var ~name:"b" ~lo:0 ~hi:10 ~dtype:Dtype.Val.int32 ()
            in
            let va =
              K.define_var ~name:"a" ~lo:0 ~hi:10 ~dtype:Dtype.Val.int32 ()
            in
            let sum = K.binary ~op:`Add ~lhs:va ~rhs:vb in
            let program = linearize (K.sink [ sum ]) in
            P.validate program;
            let find_var name =
              find_unique_position "var" program (function
                | P.Define_var { name = n; _ } -> n = name
                | _ -> false)
            in
            is_true (find_var "a" < find_var "b"));
          test "define_local before define_reg" (fun () ->
            let local_ptr = Dtype.Ptr.create dt ~addrspace:Local ~size:256 in
            let reg_ptr = Dtype.Ptr.create dt ~addrspace:Reg ~size:1 in
            let dl = K.define_local ~size:256 ~dtype:local_ptr in
            let dr = K.define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let st ptr_node =
              K.store
                ~dst:(K.index ~ptr:ptr_node ~idxs:[ i32 0 ] ())
                ~value:(f32 0.0) ~ranges:[]
            in
            let program = linearize (K.sink [ st dl; st dr ]) in
            P.validate program;
            let pos_local =
              find_unique_position "define_local" program (function
                | P.Define_local _ -> true
                | _ -> false)
            in
            let pos_reg =
              find_unique_position "define_reg" program (function
                | P.Define_reg _ -> true
                | _ -> false)
            in
            is_true (pos_local < pos_reg));
          test "nested range increases run_count" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let r_outer = loop_range ~axis:0 (i32 4) in
            let r_inner = loop_range ~axis:1 (i32 8) in
            let sum = K.binary ~op:`Add ~lhs:r_outer ~rhs:r_inner in
            let idx = K.index ~ptr:p0 ~idxs:[ sum ] () in
            let st = K.store ~dst:idx ~value:(f32 1.0) ~ranges:[] in
            let e = K.end_ ~value:st ~ranges:[ r_outer; r_inner ] () in
            let program = linearize (K.sink [ e ]) in
            P.validate program;
            let outer = find_range ~axis:0 program in
            let inner = find_range ~axis:1 program in
            let store_pos = find_store program in
            is_true (outer < inner);
            is_true (inner < store_pos));
        ];
      group "Split ends"
        [
          test "three ranges with mixed kinds are sorted" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let r_global = global_range ~axis:0 (i32 4) in
            let r_loop = loop_range ~axis:1 (i32 4) in
            let r_reduce = reduce_range ~axis:2 (i32 4) in
            let sum = K.binary ~op:`Add ~lhs:r_global ~rhs:r_loop in
            let sum2 = K.binary ~op:`Add ~lhs:sum ~rhs:r_reduce in
            let idx = K.index ~ptr:p0 ~idxs:[ sum2 ] () in
            let st = K.store ~dst:idx ~value:(f32 1.0) ~ranges:[] in
            let e =
              K.end_ ~value:st ~ranges:[ r_global; r_loop; r_reduce ] ()
            in
            let program = linearize (K.sink [ e ]) in
            P.validate program;
            equal int 3 (count_ranges program);
            equal int 3 (count_end_ranges program);
            let pos_reduce = find_range_by_kind ~kind:Axis_kind.Reduce program in
            let pos_loop = find_range_by_kind ~kind:Axis_kind.Loop program in
            let pos_global = find_range_by_kind ~kind:Axis_kind.Global program in
            is_true (pos_global < pos_loop);
            is_true (pos_loop < pos_reduce));
          test "end with zero ranges passes through" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let idx = K.index ~ptr:p0 ~idxs:[ i32 0 ] () in
            let st = K.store ~dst:idx ~value:(f32 1.0) ~ranges:[] in
            let e = K.end_ ~value:st ~ranges:[] () in
            let program = linearize (K.sink [ e ]) in
            P.validate program;
            equal int 0 (count_ranges program);
            equal int 0 (count_end_ranges program);
            equal int 1
              (count program (function P.Store _ -> true | _ -> false)));
        ];
      group "Emission"
        [
          test "barrier emission" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let idx = K.index ~ptr:p0 ~idxs:[ i32 0 ] () in
            let st = K.store ~dst:idx ~value:(f32 1.0) ~ranges:[] in
            let program = linearize (K.sink [ st; K.barrier ]) in
            P.validate program;
            equal int 1
              (count program (function P.Barrier -> true | _ -> false)));
          test "special emission" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let sp =
              K.special ~dim:(Special_dim.Global_idx 0) ~size:(i32 32) ()
            in
            let idx = K.index ~ptr:p0 ~idxs:[ sp ] () in
            let st = K.store ~dst:idx ~value:(f32 1.0) ~ranges:[] in
            let program = linearize (K.sink [ st ]) in
            P.validate program;
            ignore
              (find_unique_position "special" program (function
                | P.Special { dim = Special_dim.Global_idx 0; _ } -> true
                | _ -> false)));
          test "cast and bitcast emission" (fun () ->
            let c1f = f32 1.0 in
            let casted = K.cast ~src:c1f ~dtype:(Dtype.int32) in
            let bitcoded = K.bitcast ~src:c1f ~dtype:Dtype.Val.int32 in
            let sum = K.binary ~op:`Add ~lhs:casted ~rhs:bitcoded in
            let program = linearize (K.sink [ sum ]) in
            P.validate program;
            equal int 1
              (count program (function P.Cast _ -> true | _ -> false));
            equal int 1
              (count program (function P.Bitcast _ -> true | _ -> false)));
          test "vectorize emission" (fun () ->
            let v = K.vectorize ~srcs:[ f32 1.0; f32 2.0; f32 3.0; f32 4.0 ] in
            let program = linearize (K.sink [ v ]) in
            P.validate program;
            ignore
              (find_unique_position "vectorize" program (function
                | P.Vectorize { srcs; _ } -> List.length srcs = 4
                | _ -> false)));
          test "gep emission" (fun () ->
            let v = K.vectorize ~srcs:[ f32 1.0; f32 2.0 ] in
            let add = K.binary ~op:`Add ~lhs:v ~rhs:v in
            let program = linearize (K.sink [ K.gep ~src:add ~idx:1 ]) in
            P.validate program;
            ignore
              (find_unique_position "gep" program (function
                | P.Gep { idxs = [1]; _ } -> true
                | _ -> false)));
          test "custom and custom_inline emission" (fun () ->
            let ci =
              K.custom_inline ~fmt:"get_val(%d)" ~args:[ i32 0 ]
                ~dtype:Dtype.Val.int32
            in
            let ce = K.custom ~fmt:"barrier()" ~args:[] in
            let af = K.after ~src:ci ~deps:[ ce ] in
            let program = linearize (K.sink [ af ]) in
            P.validate program;
            equal int 1
              (count program (function P.Custom _ -> true | _ -> false));
            equal int 1
              (count program (function
                | P.Custom_inline _ -> true
                | _ -> false)));
          test "after on ptr maps directly" (fun () ->
            let reg_ptr = Dtype.Ptr.create dt ~addrspace:Reg ~size:1 in
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let dreg = K.define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let reg_idx = K.index ~ptr:dreg ~idxs:[ i32 0 ] () in
            let st_init = K.store ~dst:reg_idx ~value:(f32 0.0) ~ranges:[] in
            let af = K.after ~src:dreg ~deps:[ st_init ] in
            let ld =
              K.load ~src:(K.index ~ptr:af ~idxs:[ i32 0 ] ()) ()
            in
            let st_out =
              K.store
                ~dst:(K.index ~ptr:p0 ~idxs:[ i32 0 ] ())
                ~value:ld ~ranges:[]
            in
            let program = linearize (K.sink [ st_out ]) in
            P.validate program;
            equal int 0
              (count program (function P.After _ -> true | _ -> false)));
          test "group forwards first source" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let st n =
              K.store
                ~dst:(K.index ~ptr:p0 ~idxs:[ i32 n ] ())
                ~value:(f32 1.0) ~ranges:[]
            in
            let g = K.group [ st 0; st 1 ] in
            let program = linearize (K.sink [ g ]) in
            P.validate program;
            equal int 2
              (count program (function P.Store _ -> true | _ -> false)));
        ];
    ]
