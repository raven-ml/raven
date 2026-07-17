(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

module P = struct
  type t = U.t list
  type binary_op = [ `Add | `Sub | `Cmplt | `Other of Ops.t ]

  type view =
    | Range of { axis : int; kind : Axis_type.t }
    | End_range of { range : int }
    | Load of { src : int }
    | Store of { dst : int; value : int }
    | After of { src : int; deps : int list; dtype : Dtype.t }
    | Binary of { op : binary_op }
    | Const of { value : Const.t; dtype : Dtype.t }
    | Index of { ptr : int }
    | Param of { idx : int; dtype : Dtype.t }
    | If of { cond : int }
    | Endif of { if_ : int }
    | Define_reg of { size : int; dtype : Dtype.t }
    | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
    | Define_local of { size : int; dtype : Dtype.t }
    | Special of { name : string; size : int }
    | Cast of { src : int }
    | Bitcast of { src : int }
    | Vectorize of { srcs : int list }
    | Custom of { fmt : string }
    | Custom_inline of { fmt : string }
    | Barrier
    | Other of U.t

  let index_map program =
    let tbl = U.Ref_tbl.create (List.length program) in
    List.iteri (fun i u -> U.Ref_tbl.replace tbl u i) program;
    tbl

  let index tbl u =
    match U.Ref_tbl.find_opt tbl u with
    | Some i -> i
    | None ->
        failwith
          (Printf.sprintf "linearized child is missing from program: %s"
             (Format.asprintf "%a" U.pp u))

  let value_dtype u =
    match U.addrspace u with Some _ -> Dtype.void | None -> U.dtype u

  let int_const u =
    match U.op u, U.Arg.as_value (U.arg u) with
    | Ops.Const, Some c -> (
        match Const.view c with
        | Int n -> Some (Int64.to_int n)
        | _ -> None)
    | _ -> None

  let view_with tbl u =
    match U.op u with
    | Ops.Range -> (
        match U.as_range u with
        | Some { axis; kind; _ } -> Range { axis; kind }
        | None -> Other u)
    | Ops.End -> (
        match U.as_end u with
        | Some { ranges = [ range ]; _ } -> End_range { range = index tbl range }
        | _ -> Other u)
    | Ops.Load -> (
        match U.as_load u with
        | Some { src; _ } -> Load { src = index tbl src }
        | None -> Other u)
    | Ops.Store -> (
        match U.as_store u with
        | Some { dst; value; _ } ->
            Store { dst = index tbl dst; value = index tbl value }
        | None -> Other u)
    | Ops.After ->
        let src = U.src u in
        if Array.length src = 0 then Other u
        else
          After
            {
              src = index tbl src.(0);
              deps =
                Array.to_list src |> List.tl |> List.map (fun dep -> index tbl dep);
              dtype = value_dtype u;
            }
    | Ops.Add -> Binary { op = `Add }
    | Ops.Sub -> Binary { op = `Sub }
    | Ops.Cmplt -> Binary { op = `Cmplt }
    | op when Ops.Group.is_binary op -> Binary { op = `Other op }
    | Ops.Const -> (
        match U.Arg.as_value (U.arg u) with
        | Some value -> Const { value; dtype = U.dtype u }
        | None -> Other u)
    | Ops.Index -> (
        match U.as_index u with
        | Some { ptr; _ } -> Index { ptr = index tbl ptr }
        | None -> Other u)
    | Ops.Param -> (
        match U.as_param u with
        | Some { param = { slot; name; vmin_vmax; addrspace; _ }; _ }
          when addrspace = Dtype.Alu ->
            (match name, vmin_vmax with
             | Some name, Some (lo, hi) ->
                 Define_var { name; lo; hi; dtype = U.dtype u }
             | _ -> Param { idx = slot; dtype = U.dtype u })
        | Some { param = { slot; _ }; _ } -> Param { idx = slot; dtype = U.dtype u }
        | None -> Other u)
    | Ops.If -> (
        match U.as_if u with
        | Some { cond; _ } -> If { cond = index tbl cond }
        | None -> Other u)
    | Ops.Endif ->
        let src = U.src u in
        if Array.length src = 1 then Endif { if_ = index tbl src.(0) } else Other u
    | Ops.Buffer -> (
        match U.addrspace u with
        | Some Dtype.Local ->
            Define_local
              { size = Option.value (int_const (U.src u).(0)) ~default:(-1);
                dtype = U.dtype u }
        | Some Dtype.Reg ->
            Define_reg
              { size = Option.value (int_const (U.src u).(0)) ~default:(-1);
                dtype = U.dtype u }
        | _ -> Other u)
    | Ops.Special -> (
        match U.as_special u with
        | Some { name; size } ->
            Special { name; size = Option.value (int_const size) ~default:(-1) }
        | None -> Other u)
    | Ops.Cast -> Cast { src = index tbl (U.src u).(0) }
    | Ops.Bitcast -> Bitcast { src = index tbl (U.src u).(0) }
    | Ops.Stack -> Vectorize { srcs = Array.to_list (U.src u) |> List.map (index tbl) }
    | Ops.Custom -> (
        match U.Arg.as_string (U.arg u) with
        | Some fmt -> Custom { fmt }
        | None -> Other u)
    | Ops.Customi -> (
        match U.Arg.as_string (U.arg u) with
        | Some fmt -> Custom_inline { fmt }
        | None -> Other u)
    | Ops.Barrier -> Barrier
    | _ -> Other u

  let view program i =
    let tbl = index_map program in
    view_with tbl (List.nth program i)

  let iteri f program =
    let tbl = index_map program in
    List.iteri (fun i u -> f i (view_with tbl u)) program

  let length = List.length
  let validate = Spec.verify_list Spec.program_spec

  let pp_view fmt = function
    | Range _ -> Format.pp_print_string fmt "Range"
    | End_range _ -> Format.pp_print_string fmt "End_range"
    | Load _ -> Format.pp_print_string fmt "Load"
    | Store _ -> Format.pp_print_string fmt "Store"
    | After _ -> Format.pp_print_string fmt "After"
    | Binary _ -> Format.pp_print_string fmt "Binary"
    | Const _ -> Format.pp_print_string fmt "Const"
    | Index _ -> Format.pp_print_string fmt "Index"
    | Param _ -> Format.pp_print_string fmt "Param"
    | If _ -> Format.pp_print_string fmt "If"
    | Endif _ -> Format.pp_print_string fmt "Endif"
    | Define_reg _ -> Format.pp_print_string fmt "Define_reg"
    | Define_var _ -> Format.pp_print_string fmt "Define_var"
    | Define_local _ -> Format.pp_print_string fmt "Define_local"
    | Special _ -> Format.pp_print_string fmt "Special"
    | Cast _ -> Format.pp_print_string fmt "Cast"
    | Bitcast _ -> Format.pp_print_string fmt "Bitcast"
    | Vectorize _ -> Format.pp_print_string fmt "Vectorize"
    | Custom _ -> Format.pp_print_string fmt "Custom"
    | Custom_inline _ -> Format.pp_print_string fmt "Custom_inline"
    | Barrier -> Format.pp_print_string fmt "Barrier"
    | Other u -> U.pp fmt u

  let pp fmt program = Render.pp_uops fmt program
end

(* Helpers *)

let dt = Dtype.float32
let ptr = dt

let i32 n = U.const (Const.int Dtype.int32 n)
let f32 x = U.const (Const.float Dtype.float32 x)

let define_local ~size ~dtype =
  U.buffer ~slot:0 ~dtype ~shape:(i32 size) ~addrspace:Dtype.Local ()

let define_reg ~size ~dtype ~slot =
  U.buffer ~slot ~dtype ~shape:(i32 size) ~addrspace:Dtype.Reg ()

let define_var ~name ~lo ~hi ~dtype () =
  U.variable ~name ~min_val:lo ~max_val:hi ~dtype ()

let loop_range ~axis size =
  U.range ~size ~axis ~kind:Axis_type.Loop ~dtype:Dtype.int32 ()

let reduce_range ~axis size =
  U.range ~size ~axis ~kind:Axis_type.Reduce ~dtype:Dtype.int32 ()

let global_range ~axis size =
  U.range ~size ~axis ~kind:Axis_type.Global ~dtype:Dtype.int32 ()

let load_one_elem () =
  let p0 = U.param ~slot:0 ~dtype:ptr () in
  U.load ~src:(U.index ~ptr:p0 ~idxs:[(i32 0)] ()) ()

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

let range_sub r =
  match U.as_range r with
  | Some { sub; _ } -> sub
  | None -> failwith "expected Range"

let find_load prog =
  find_unique_position "load" prog (function P.Load _ -> true | _ -> false)

let find_store prog =
  find_unique_position "store" prog (function P.Store _ -> true | _ -> false)

let raises_linearize substring fn =
  raises_match (function Failure msg -> contains msg substring | _ -> false) fn

let test_unlowered_rejected name build_node =
  raises_linearize (name ^ " must be lowered before linearize") (fun () ->
      ignore (linearize (U.sink [ build_node () ])))

let () =
  run "Linearizer"
    [
      group "Late kernel to program"
        [
          test "multi-range End lowers to nested End_range pairs" (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let r0 = loop_range ~axis:0 (i32 2) in
            let r1 = loop_range ~axis:1 (i32 3) in
            let sum = U.alu_binary ~op:Ops.Add ~lhs:r0 ~rhs:r1 in
            let idx = U.index ~ptr:p0 ~idxs:[sum] () in
            let st = U.store ~dst:idx ~value:(f32 1.0) () in
            let e = U.end_ ~value:st ~ranges:[ r0; r1 ] in
            let program = linearize (U.sink [ e ]) in
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
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let p1 = U.param ~slot:1 ~dtype:ptr () in
            let r0 = loop_range ~axis:0 (i32 2) in
            let idx_in = U.index ~ptr:p0 ~idxs:[r0] () in
            let ld = U.load ~src:idx_in () in
            let r1 = loop_range ~axis:1 (i32 3) in
            let sum = U.alu_binary ~op:Ops.Add ~lhs:r0 ~rhs:r1 in
            let idx_out = U.index ~ptr:p1 ~idxs:[sum] () in
            let st = U.store ~dst:idx_out ~value:ld () in
            let e = U.end_ ~value:st ~ranges:[ r0; r1 ] in
            let program = linearize (U.sink [ e ]) in
            P.validate program;
            let load_pos = find_load program in
            is_true (find_range ~axis:0 program < load_pos);
            is_true (load_pos < find_range ~axis:1 program));
          test "After nodes stay in Program ownership after linearize"
            (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let idx = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let ld = U.load ~src:idx () in
            let af = U.after ~src:ld ~deps:[ f32 1.0 ] in
            let program = linearize (U.sink [ af ]) in
            let after_pos =
              find_unique_position "after" program (function
                | P.After _ -> true
                | _ -> false)
            in
            (match P.view program after_pos with
             | P.After { src; deps = [ dep ]; dtype } ->
                 is_true (Dtype.equal dtype dt);
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
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let idx0 = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let st0 = U.store ~dst:idx0 ~value:(f32 1.0) () in
            let idx1 = U.index ~ptr:p0 ~idxs:[(i32 1)] () in
            let st1 = U.store ~dst:idx1 ~value:(f32 2.0) () in
            let af = U.after ~src:st0 ~deps:[ st1 ] in
            let program = linearize (U.sink [ af ]) in
            let after_pos =
              find_unique_position "effect after" program (function
                | P.After _ -> true
                | _ -> false)
            in
            (match P.view program after_pos with
             | P.After { src; deps = [ dep ]; dtype } ->
                 is_true (Dtype.equal dtype Dtype.void);
                 (match (P.view program src, P.view program dep) with
                  | P.Store _, P.Store _ -> ()
                  | src_view, dep_view ->
                      failwith
                        (Printf.sprintf
                           "unexpected void After operands:\n%s\n%s"
                           (pp_view src_view) (pp_view dep_view)))
             | view -> fail_view "expected effect-only After" view));
          test "nested alt-index loads stay between the two ranges" (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let p1 = U.param ~slot:1 ~dtype:ptr () in
            let r0 = loop_range ~axis:0 (i32 2) in
            let gate = U.alu_binary ~op:Ops.Cmplt ~lhs:r0 ~rhs:(i32 2) in
            let idx_gated = U.index ~ptr:p0 ~idxs:[r0] () in
            let ld = U.load ~src:idx_gated ~alt:(f32 2.0) ~gate () in
            let r1 = loop_range ~axis:1 (i32 3) in
            let add = U.alu_binary ~op:Ops.Add ~lhs:ld ~rhs:(f32 1.0) in
            let flat_idx =
              U.alu_binary ~op:Ops.Add
                ~lhs:(U.alu_binary ~op:Ops.Mul ~lhs:r0 ~rhs:(i32 3))
                ~rhs:r1
            in
            let idx_out = U.index ~ptr:p1 ~idxs:[flat_idx] () in
            let st = U.store ~dst:idx_out ~value:add () in
            let e = U.end_ ~value:st ~ranges:[ r0; r1 ] in
            let program = linearize (U.sink [ e ]) in
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
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let gate = U.const (Const.bool true) in
            let idx = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let st = U.store ~dst:idx ~value:(f32 1.0) ~gate () in
            let sink = U.sink [ st ] in
            let sink = Linearizer.pm_split_ends sink in
            let sink = Linearizer.pm_add_control_flow sink in
            let program = Linearizer.linearize sink in
            P.validate program;
            let if_pos =
              find_unique_position "if" program (function
                | P.If _ -> true
                | _ -> false)
            in
            let store_pos = find_store program in
            let endif_pos =
              find_unique_position "endif" program (function
                | P.Endif _ -> true
                | _ -> false)
            in
            is_true (if_pos < store_pos);
            is_true (store_pos < endif_pos);
            (match P.view program store_pos with
             | P.Store _ -> ()
             | view -> fail_view "expected ungated Store" view));
          test "single casted gated stores become IF/STORE/ENDIF"
            (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let gate = U.const (Const.bool true) in
            let idx = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let dst = U.cast ~src:idx ~dtype:(U.dtype idx) in
            let st = U.store ~dst ~value:(f32 1.0) ~gate () in
            let program = U.sink [ st ] |> linearize in
            P.validate program;
            ignore
              (find_unique_position "if" program (function
                | P.If _ -> true
                | _ -> false));
            ignore
              (find_unique_position "endif" program (function
                | P.Endif _ -> true
                | _ -> false)));
          test "bitcasted gated stores are not linearize-cleanup matches"
            (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let gate = U.const (Const.bool true) in
            let idx = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let i32_ptr = Dtype.int32 in
            let dst = U.bitcast ~src:idx ~dtype:i32_ptr in
            let st = U.store ~dst ~value:(i32 1) ~gate () in
            let program = U.sink [ st ] |> linearize in
            equal int 0
              (count program (function P.If _ | P.Endif _ -> true | _ -> false));
            equal int 1
              (List.length
                 (List.filter
                    (fun u ->
                      match U.as_store u with
                      | Some { gate = Some gate'; _ } -> U.equal gate gate'
                      | _ -> false)
                    program)));
          test "nested-cast gated stores are not linearize-cleanup matches"
            (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let gate = U.const (Const.bool true) in
            let idx = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let i32_ptr = Dtype.int32 in
            let dst =
              U.cast ~src:(U.cast ~src:idx ~dtype:i32_ptr)
                ~dtype:(U.dtype idx)
            in
            let st = U.store ~dst ~value:(f32 1.0) ~gate () in
            let program = U.sink [ st ] |> linearize in
            equal int 0
              (count program (function P.If _ | P.Endif _ -> true | _ -> false));
            equal int 1
              (List.length
                 (List.filter
                    (fun u ->
                      match U.as_store u with
                      | Some { gate = Some gate'; _ } -> U.equal gate gate'
                      | _ -> false)
                    program)));
          test "equal-priority nodes use structural tie-breaks" (fun () ->
            let sub = U.alu_binary ~op:Ops.Sub ~lhs:(i32 2) ~rhs:(i32 1) in
            let add = U.alu_binary ~op:Ops.Add ~lhs:(i32 1) ~rhs:(i32 2) in
            let program = linearize (U.sink [ add; sub ]) in
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
            let out_ptr = dt in
            let in_ptr = dt in
            let bias_ptr = dt in
            let reg_ptr = dt in
            let p0 = U.param ~slot:0 ~dtype:out_ptr () in
            let p1 = U.param ~slot:1 ~dtype:in_ptr () in
            let p2 = U.param ~slot:2 ~dtype:bias_ptr () in
            let dreg = define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let reg_idx = U.index ~ptr:dreg ~idxs:[(i32 0)] () in
            let st_init = U.store ~dst:reg_idx ~value:(f32 0.0) () in
            let r0 = reduce_range ~axis:0 (i32 4) in
            let idx_in = U.index ~ptr:p1 ~idxs:[r0] () in
            let st_acc =
              U.store ~dst:reg_idx ~value:(U.load ~src:idx_in ()) ()
            in
            let e = U.end_ ~value:st_acc ~ranges:[ r0 ] in
            let acc_after = U.after ~src:dreg ~deps:[ e ] in
            let acc_val =
              U.load ~src:(U.index ~ptr:acc_after ~idxs:[(i32 0)] ()) ()
            in
            let idx_bias = U.index ~ptr:p2 ~idxs:[(i32 0)] () in
            let ld_bias = U.load ~src:idx_bias () in
            let add = U.alu_binary ~op:Ops.Add ~lhs:acc_val ~rhs:ld_bias in
            let idx_out = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let st_out = U.store ~dst:idx_out ~value:add () in
            let program = linearize (U.sink [ st_init; st_out ]) in
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
            let out_ptr = dt in
            let in_ptr = dt in
            let bias_ptr = dt in
            let reg_ptr = dt in
            let p0 = U.param ~slot:0 ~dtype:out_ptr () in
            let p1 = U.param ~slot:1 ~dtype:in_ptr () in
            let p2 = U.param ~slot:2 ~dtype:bias_ptr () in
            let dreg = define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let reg_idx = U.index ~ptr:dreg ~idxs:[(i32 0)] () in
            let st_init = U.store ~dst:reg_idx ~value:(f32 0.0) () in
            let idx_bias = U.index ~ptr:p2 ~idxs:[(i32 0)] () in
            let ld_bias = U.load ~src:idx_bias () in
            let r0 = reduce_range ~axis:0 (i32 4) in
            let idx_in = U.index ~ptr:p1 ~idxs:[r0] () in
            let ld_in = U.load ~src:idx_in () in
            let add_in = U.alu_binary ~op:Ops.Add ~lhs:ld_in ~rhs:ld_bias in
            let st_reg = U.store ~dst:reg_idx ~value:add_in () in
            let e = U.end_ ~value:st_reg ~ranges:[ r0 ] in
            let acc_after = U.after ~src:dreg ~deps:[ e ] in
            let acc_val =
              U.load ~src:(U.index ~ptr:acc_after ~idxs:[(i32 0)] ()) ()
            in
            let add_out = U.alu_binary ~op:Ops.Add ~lhs:acc_val ~rhs:ld_bias in
            let idx_out = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let st_out = U.store ~dst:idx_out ~value:add_out () in
            let program = linearize (U.sink [ st_init; st_out ]) in
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
            let input_ptr = dt in
            let reg_ptr = dt in
            let p0 = U.param ~slot:0 ~dtype:input_ptr () in
            let dreg = define_reg ~size:4 ~dtype:reg_ptr ~slot:0 in
            let ri n = U.index ~ptr:dreg ~idxs:[(i32 n)] () in
            let st_init n = U.store ~dst:(ri n) ~value:(f32 0.0) () in
            let r0 = loop_range ~axis:0 (i32 4) in
            let idx_in = U.index ~ptr:p0 ~idxs:[r0] () in
            let ld = U.load ~src:idx_in () in
            let st_loop n =
              let add = U.alu_binary ~op:Ops.Add ~lhs:ld ~rhs:(f32 0.0) in
              U.store ~dst:(ri n) ~value:add ()
            in
            let e = U.end_ ~value:ld ~ranges:[ r0 ] in
            let program =
              linearize
                (U.sink
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
            let sink_pos = List.length program - 1 in
            P.iteri
              (fun i view ->
                match view with
                | P.Store { dst; value } ->
                    (match P.view program dst with
                     | P.Index { ptr; _ } ->
                         (match P.view program ptr with
                          | P.Define_reg _ ->
                              (match P.view program value with
                               | P.Const _ -> ()
                               | P.Binary { op = `Add; _ } ->
                                   is_true
                                     ~msg:
                                       "expected ALU-fed reg store inside \
                                        linearized range"
                                     (i > range_pos && i < sink_pos)
                               | other ->
                                   fail_view
                                     "expected reg init or ALU-fed reg store"
                                     other)
                          | _ -> ())
                     | _ -> ())
                | _ -> ())
              program);
          test "gated loads without alts are rejected" (fun () ->
            raises_linearize
              "gated loads require an alt value before linearize"
              (fun () ->
                let p0 = U.param ~slot:0 ~dtype:ptr () in
                let gate = U.const (Const.bool true) in
                let idx = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
                let ld = U.load ~src:idx () in
                let ld = U.replace ld ~src:[| idx; gate |] () in
                ignore (linearize (U.sink [ ld ]))));
          test "unlowered Reduce nodes are rejected" (fun () ->
            raises_linearize "Reduce must be lowered before linearize"
              (fun () ->
                let p0 = U.param ~slot:0 ~dtype:ptr () in
                let r0 = reduce_range ~axis:0 (i32 4) in
                let idx = U.index ~ptr:p0 ~idxs:[r0] () in
                let ld = U.load ~src:idx () in
                let red =
                  U.reduce ~op:Ops.Add ~src:ld ~ranges:[ r0 ] ~dtype:dt
                in
                ignore (linearize (U.sink [ red ]))));
          test "graph IF nodes are rejected" (fun () ->
            raises_linearize "IF/ENDIF must be inserted by linearize cleanups"
              (fun () ->
                let if_ =
                  U.if_ ~cond:(U.const (Const.bool true))
                    ~idx_for_dedup:(i32 0)
                in
                ignore (linearize (U.sink [ if_ ]))));
          test "graph ENDIF nodes are rejected" (fun () ->
            raises_linearize "IF/ENDIF must be inserted by linearize cleanups"
              (fun () ->
                let if_ =
                  U.if_ ~cond:(U.const (Const.bool true))
                    ~idx_for_dedup:(i32 0)
                in
                ignore (linearize (U.sink [ U.endif ~if_ ]))));
        ];
      group "CFG context"
        [
          test "sibling ends under sink are ordered" (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let p1 = U.param ~slot:1 ~dtype:ptr () in
            let r0 = loop_range ~axis:0 (i32 4) in
            let st0 =
              U.store
                ~dst:(U.index ~ptr:p0 ~idxs:[r0] ())
                ~value:(f32 1.0) ()
            in
            let e0 = U.end_ ~value:st0 ~ranges:[ r0 ] in
            let r1 = loop_range ~axis:1 (i32 4) in
            let st1 =
              U.store
                ~dst:(U.index ~ptr:p1 ~idxs:[r1] ())
                ~value:(f32 1.0) ()
            in
            let e1 = U.end_ ~value:st1 ~ranges:[ r1 ] in
            let program = linearize (U.sink [ e0; e1 ]) in
            P.validate program;
            equal int 2 (count_ranges program);
            equal int 2 (count_end_ranges program);
            let ranges = find_ranges program in
            let ends = find_end_ranges program in
            is_true (List.hd ends < List.nth ranges 1));
          test "three-range end exercises cfg nesting" (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let r0 = loop_range ~axis:0 (i32 4) in
            let r1 = loop_range ~axis:1 (i32 4) in
            let r2 = loop_range ~axis:2 (i32 4) in
            let sum = U.alu_binary ~op:Ops.Add ~lhs:r0 ~rhs:r1 in
            let sum2 = U.alu_binary ~op:Ops.Add ~lhs:sum ~rhs:r2 in
            let idx = U.index ~ptr:p0 ~idxs:[sum2] () in
            let st = U.store ~dst:idx ~value:(f32 1.0) () in
            let e = U.end_ ~value:st ~ranges:[ r0; r1; r2 ] in
            let program = linearize (U.sink [ e ]) in
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
            let out_ptr = dt in
            let in_ptr_a = dt in
            let in_ptr_b = dt in
            let reg_ptr = dt in
            let p0 = U.param ~slot:0 ~dtype:out_ptr () in
            let p1 = U.param ~slot:1 ~dtype:in_ptr_a () in
            let p2 = U.param ~slot:2 ~dtype:in_ptr_b () in
            let make_reduce dreg param axis =
              let ri = U.index ~ptr:dreg ~idxs:[(i32 0)] () in
              let st_init = U.store ~dst:ri ~value:(f32 0.0) () in
              let r = reduce_range ~axis (i32 4) in
              let idx = U.index ~ptr:param ~idxs:[r] () in
              let st_acc =
                U.store ~dst:ri ~value:(U.load ~src:idx ()) ()
              in
              let e = U.end_ ~value:st_acc ~ranges:[ r ] in
              (st_init, e)
            in
            let dreg_a = define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let dreg_b = define_reg ~size:1 ~dtype:reg_ptr ~slot:1 in
            let st_init_a, e0 = make_reduce dreg_a p1 0 in
            let st_init_b, e1 = make_reduce dreg_b p2 1 in
            let af_a = U.after ~src:dreg_a ~deps:[ e0 ] in
            let af_b = U.after ~src:dreg_b ~deps:[ e1 ] in
            let ld_res_a =
              U.load ~src:(U.index ~ptr:af_a ~idxs:[(i32 0)] ()) ()
            in
            let ld_res_b =
              U.load ~src:(U.index ~ptr:af_b ~idxs:[(i32 0)] ()) ()
            in
            let sum = U.alu_binary ~op:Ops.Add ~lhs:ld_res_a ~rhs:ld_res_b in
            let idx_out = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let st_out = U.store ~dst:idx_out ~value:sum () in
            let program =
              linearize (U.sink [ st_init_a; st_init_b; st_out ])
            in
            P.validate program;
            equal int 2 (count_ranges program);
            equal int 2 (count_end_ranges program);
            let ranges = find_ranges program in
            let ends = find_end_ranges program in
            is_true (List.nth ends 0 < List.nth ranges 1));
          test "three sibling ends are chain-ordered" (fun () ->
            let make_branch idx axis =
              let p = U.param ~slot:idx ~dtype:dt () in
              let r = loop_range ~axis (i32 4) in
              let st =
                U.store
                  ~dst:(U.index ~ptr:p ~idxs:[r] ())
                  ~value:(f32 1.0) ()
              in
              U.end_ ~value:st ~ranges:[ r ]
            in
            let e0 = make_branch 0 0 in
            let e1 = make_branch 1 1 in
            let e2 = make_branch 2 2 in
            let program = linearize (U.sink [ e0; e1; e2 ]) in
            P.validate program;
            equal int 3 (count_ranges program);
            equal int 3 (count_end_ranges program);
            let ranges = find_ranges program in
            let ends = find_end_ranges program in
            is_true (List.nth ends 0 < List.nth ranges 1);
            is_true (List.nth ends 1 < List.nth ranges 2));
          test "cyclic control-flow edge is rejected" (fun () ->
            let p = U.param ~slot:0 ~dtype:dt () in
            let child_r = loop_range ~axis:1 (i32 4) in
            let parent_r =
              loop_range ~axis:0 (i32 4)
              |> fun r ->
              U.replace r
                ~src:(Array.append (U.src r) [| child_r |])
                ()
            in
            let child_idx =
              U.index ~ptr:p
                ~idxs:[ U.alu_binary ~op:Ops.Add ~lhs:parent_r ~rhs:child_r ]
                ()
            in
            let child_store = U.store ~dst:child_idx ~value:(f32 1.0) () in
            let child_end = U.end_ ~value:child_store ~ranges:[ child_r ] in
            let parent_end = U.end_ ~value:child_end ~ranges:[ parent_r ] in
            raises_match
              (function
                | Failure msg ->
                    String.equal msg "linearizer control-flow cycle"
                | _ -> false)
              (fun () ->
                ignore (Linearizer.pm_add_control_flow (U.sink [ parent_end ]))));
        ];
      group "Error paths"
        [
          test "empty Group is rejected" (fun () ->
            raises_linearize "empty Group" (fun () ->
                ignore (linearize (U.sink [ U.group [] ]))));
        ];
      group "Priority ordering"
        [
          test "params ordered by index" (fun () ->
            let p2 = U.param ~slot:2 ~dtype:ptr () in
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let p1 = U.param ~slot:1 ~dtype:ptr () in
            let ld n p =
              U.load ~src:(U.index ~ptr:p ~idxs:[(i32 0)] ()) ()
            in
            let sum =
              U.alu_binary ~op:Ops.Add ~lhs:(ld 0 p0)
                ~rhs:
                  (U.alu_binary ~op:Ops.Add ~lhs:(ld 1 p1)
                     ~rhs:(ld 2 p2))
            in
            let program = linearize (U.sink [ sum ]) in
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
              define_var ~name:"b" ~lo:0 ~hi:10 ~dtype:Dtype.int32 ()
            in
            let va =
              define_var ~name:"a" ~lo:0 ~hi:10 ~dtype:Dtype.int32 ()
            in
            let sum = U.alu_binary ~op:Ops.Add ~lhs:va ~rhs:vb in
            let program = linearize (U.sink [ sum ]) in
            P.validate program;
            let find_var name =
              find_unique_position "var" program (function
                | P.Define_var { name = n; _ } -> n = name
                | _ -> false)
            in
            is_true (find_var "a" < find_var "b"));
          test "define_reg before define_local" (fun () ->
            let local_ptr = dt in
            let reg_ptr = dt in
            let dl = define_local ~size:256 ~dtype:local_ptr in
            let dr = define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let st ptr_node =
              U.store
                ~dst:(U.index ~ptr:ptr_node ~idxs:[(i32 0)] ())
                ~value:(f32 0.0) ()
            in
            let program = linearize (U.sink [ st dl; st dr ]) in
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
            is_true (pos_reg < pos_local));
          test "nested range increases run_count" (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let r_outer = loop_range ~axis:0 (i32 4) in
            let r_inner = loop_range ~axis:1 (i32 8) in
            let sum = U.alu_binary ~op:Ops.Add ~lhs:r_outer ~rhs:r_inner in
            let idx = U.index ~ptr:p0 ~idxs:[sum] () in
            let st = U.store ~dst:idx ~value:(f32 1.0) () in
            let e = U.end_ ~value:st ~ranges:[ r_outer; r_inner ] in
            let program = linearize (U.sink [ e ]) in
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
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let r_global = global_range ~axis:0 (i32 4) in
            let r_loop = loop_range ~axis:1 (i32 4) in
            let r_reduce = reduce_range ~axis:2 (i32 4) in
            let sum = U.alu_binary ~op:Ops.Add ~lhs:r_global ~rhs:r_loop in
            let sum2 = U.alu_binary ~op:Ops.Add ~lhs:sum ~rhs:r_reduce in
            let idx = U.index ~ptr:p0 ~idxs:[sum2] () in
            let st = U.store ~dst:idx ~value:(f32 1.0) () in
            let e =
              U.end_ ~value:st ~ranges:[ r_global; r_loop; r_reduce ]
            in
            let program = linearize (U.sink [ e ]) in
            P.validate program;
            equal int 3 (count_ranges program);
            equal int 3 (count_end_ranges program);
            let pos_reduce =
              find_range_by_kind ~kind:Axis_type.Reduce program
            in
            let pos_loop = find_range_by_kind ~kind:Axis_type.Loop program in
            let pos_global =
              find_range_by_kind ~kind:Axis_type.Global program
            in
            is_true (pos_global < pos_loop);
            is_true (pos_loop < pos_reduce));
          test "same-axis ranges are split by full range argument" (fun () ->
            let r0 =
              U.range ~size:(i32 4) ~axis:0 ~sub:[ 0 ]
                ~kind:Axis_type.Loop ~dtype:Dtype.int32 ()
            in
            let r1 =
              U.range ~size:(i32 4) ~axis:0 ~sub:[ 1 ]
                ~kind:Axis_type.Loop ~dtype:Dtype.int32 ()
            in
            let e = U.end_ ~value:(i32 1) ~ranges:[ r0; r1 ] in
            match Linearizer.do_split_ends e with
            | None -> failwith "expected split End"
            | Some outer ->
                (match U.as_end outer with
                 | Some { value = inner; ranges = [ outer_range ] } ->
                     equal (list int) [ 0 ] (range_sub outer_range);
                     (match U.as_end inner with
                      | Some { ranges = [ inner_range ]; _ } ->
                          equal (list int) [ 1 ] (range_sub inner_range)
                      | _ -> failwith "expected inner End")
                 | _ -> failwith "expected outer End"));
          test "end with zero ranges passes through" (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let idx = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let st = U.store ~dst:idx ~value:(f32 1.0) () in
            let e = U.end_ ~value:st ~ranges:[] in
            let program = linearize (U.sink [ e ]) in
            P.validate program;
            equal int 0 (count_ranges program);
            equal int 0 (count_end_ranges program);
            equal int 1
              (count program (function P.Store _ -> true | _ -> false)));
        ];
      group "Emission"
        [
          test "barrier emission" (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let idx = U.index ~ptr:p0 ~idxs:[(i32 0)] () in
            let st = U.store ~dst:idx ~value:(f32 1.0) () in
            let program = linearize (U.sink [ st; U.barrier () ]) in
            P.validate program;
            equal int 1
              (count program (function P.Barrier -> true | _ -> false)));
          test "special emission" (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let sp =
              U.special ~name:"idx0" ~size:(i32 32) ~dtype:Dtype.int32 ()
            in
            let idx = U.index ~ptr:p0 ~idxs:[sp] () in
            let st = U.store ~dst:idx ~value:(f32 1.0) () in
            let program = linearize (U.sink [ st ]) in
            P.validate program;
            ignore
              (find_unique_position "special" program (function
                | P.Special { name = "idx0"; _ } -> true
                | _ -> false)));
          test "cast and bitcast emission" (fun () ->
            let c1f = f32 1.0 in
            let casted = U.cast ~src:c1f ~dtype:(Dtype.int32) in
            let bitcoded = U.bitcast ~src:c1f ~dtype:Dtype.int32 in
            let sum = U.alu_binary ~op:Ops.Add ~lhs:casted ~rhs:bitcoded in
            let program = linearize (U.sink [ sum ]) in
            P.validate program;
            equal int 1
              (count program (function P.Cast _ -> true | _ -> false));
            equal int 1
              (count program (function P.Bitcast _ -> true | _ -> false)));
          test "vectorize emission" (fun () ->
            let v = U.stack [ f32 1.0; f32 2.0; f32 3.0; f32 4.0 ] in
            let program = linearize (U.sink [ v ]) in
            P.validate program;
            ignore
              (find_unique_position "vectorize" program (function
                | P.Vectorize { srcs; _ } -> List.length srcs = 4
                | _ -> false)));
          test "value index emission" (fun () ->
            let v = U.stack [ f32 1.0; f32 2.0 ] in
            let add = U.alu_binary ~op:Ops.Add ~lhs:v ~rhs:v in
            let program =
              linearize
                (U.sink [ U.index ~ptr:add ~idxs:[ U.const_int 1 ] () ])
            in
            ignore
              (find_unique_position "value index" program (function
                | P.Index _ -> true
                | _ -> false)));
          test "custom and custom_inline emission" (fun () ->
            let ci =
              U.custom_inline ~fmt:"get_val(%d)" ~args:[ i32 0 ]
                ~dtype:Dtype.int32
            in
            let ce = U.custom ~fmt:"barrier()" ~args:[] in
            let af = U.after ~src:ci ~deps:[ ce ] in
            let program = linearize (U.sink [ af ]) in
            equal int 1
              (count program (function P.Custom _ -> true | _ -> false));
            equal int 1
              (count program (function
                | P.Custom_inline _ -> true
                | _ -> false)));
          test "after on ptr stays in program" (fun () ->
            let reg_ptr = dt in
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let dreg = define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let reg_idx = U.index ~ptr:dreg ~idxs:[(i32 0)] () in
            let st_init = U.store ~dst:reg_idx ~value:(f32 0.0) () in
            let af = U.after ~src:dreg ~deps:[ st_init ] in
            let ld =
              U.load ~src:(U.index ~ptr:af ~idxs:[(i32 0)] ()) ()
            in
            let st_out =
              U.store
                ~dst:(U.index ~ptr:p0 ~idxs:[(i32 0)] ())
                ~value:ld ()
            in
            let program = linearize (U.sink [ st_out ]) in
            P.validate program;
            let after_pos =
              find_unique_position "ptr after" program (function
                | P.After _ -> true
                | _ -> false)
            in
            (match P.view program after_pos with
             | P.After { src; deps = [ dep ]; dtype } ->
                 is_true (Dtype.equal dtype Dtype.void);
                 (match (P.view program src, P.view program dep) with
                  | P.Define_reg _, P.Store _ -> ()
                  | src_view, dep_view ->
                      failwith
                        (Printf.sprintf
                           "unexpected pointer After operands:\n%s\n%s"
                           (pp_view src_view) (pp_view dep_view)))
             | view -> fail_view "expected pointer After" view));
          test "group forwards first source" (fun () ->
            let p0 = U.param ~slot:0 ~dtype:ptr () in
            let st n =
              U.store
                ~dst:(U.index ~ptr:p0 ~idxs:[(i32 n)] ())
                ~value:(f32 1.0) ()
            in
            let g = U.group [ st 0; st 1 ] in
            let program = linearize (U.sink [ g ]) in
            P.validate program;
            equal int 2
              (count program (function P.Store _ -> true | _ -> false)));
        ];
    ]
