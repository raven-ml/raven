(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop

module U = Uop

let test_renderer ?extra_matcher () =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:false
    ~has_shared:false ~shared_max:0 ?extra_matcher
    ~render:(fun ?name:_ _ -> "") ()

let with_spec value fn =
  let old = Sys.getenv_opt "SPEC" in
  Unix.putenv "SPEC" value;
  Fun.protect fn ~finally:(fun () ->
      match old with
      | Some v -> Unix.putenv "SPEC" v
      | None -> Unix.putenv "SPEC" "")

let custom_fmt node =
  match U.op node, U.Arg.as_string (U.arg node) with
  | (Ops.Custom | Ops.Customi), Some fmt -> Some fmt
  | _ -> None

let param_slots root =
  root
  |> U.toposort
  |> List.filter_map (fun node ->
       match U.as_param node with
       | Some { param; _ } -> Some param.slot
       | None -> None)

let named_param_slots root =
  root
  |> U.toposort
  |> List.filter_map (fun node ->
       match U.as_param node with
       | Some { param = { name = Some name; slot; _ }; _ } ->
           Some (name, slot)
       | Some _ | None -> None)

let is_invalid_const node =
  match U.op node, U.arg node with
  | Ops.Const, U.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

let has_invalid_const root =
  List.exists is_invalid_const (U.toposort root)

let global_ptr ?(slot = 0) () =
  U.param ~slot ~dtype:Dtype.float32 ~addrspace:Dtype.Global ()

let () =
  run "Codegen_lower"
    [
      group "final cleanup"
        [
          test "final rewrite concretizes leftover index dtypes" (fun () ->
            let p = global_ptr () in
            let r =
              U.range ~size:(U.const_int 8) ~axis:0 ~kind:Axis_type.Loop ()
            in
            let dst = U.index ~ptr:p ~idxs:[ r ] () in
            let st =
              U.store ~dst ~value:(U.const (Const.float Dtype.float32 1.0)) ()
            in
            let root = U.sink [ U.end_ ~value:st ~ranges:[ r ] ] in
            let lowered =
              with_spec "1" (fun () ->
                  Codegen_lower.lower (test_renderer ()) root)
            in
            is_true ~msg:"no reachable index dtype remains"
              (not
                 (List.exists
                    (fun node -> Dtype.equal (U.dtype node) Dtype.index)
                    (U.toposort lowered)));
            Spec.type_verify Spec.program_spec lowered);
          test "memory operands feeding ALU become explicit loads" (fun () ->
            let p0 = global_ptr ~slot:0 () in
            let p1 = global_ptr ~slot:1 () in
            let idx = U.const_int 0 in
            let value =
              U.alu_unary ~op:Ops.Neg
                ~src:(U.index ~ptr:p0 ~idxs:[ idx ] ())
            in
            let dst = U.index ~ptr:p1 ~idxs:[ idx ] () in
            let lowered =
              Codegen_lower.lower (test_renderer ())
                (U.sink [ U.store ~dst ~value () ])
            in
            let topo = U.toposort lowered in
            is_true ~msg:"ALU consumes an explicit load"
              (List.exists
                 (fun node ->
                   U.op node = Ops.Neg
                   &&
                   match U.src node with
                   | [| src |] -> U.op src = Ops.Load
                   | _ -> false)
                 topo);
            is_true ~msg:"store destination is not loaded"
              (List.for_all
                 (fun node ->
                   match U.as_store node with
                   | Some { dst; _ } -> U.op dst <> Ops.Load
                   | None -> true)
                 topo));
          test "gater leaves already-gated invalid-index load unchanged"
            (fun () ->
            let p = global_ptr () in
            let idx =
              U.variable ~name:"i" ~min_val:0 ~max_val:100 ~dtype:Dtype.int32 ()
            in
            let gate =
              U.alu_binary ~op:Ops.Cmplt ~lhs:idx
                ~rhs:(U.const (Const.int Dtype.int32 3))
            in
            let invalid_idx =
              U.O.where gate idx (U.invalid ~dtype:Dtype.int32 ())
            in
            let mop = U.index ~ptr:p ~idxs:[ invalid_idx ] () in
            let load =
              U.load ~src:mop
                ~alt:(U.const (Const.float Dtype.float32 0.0))
                ~gate ()
            in
            let lowered = Gater.pm_move_gates_from_index load in
            (match U.as_load lowered with
            | Some { src; gate = Some load_gate; _ } ->
                is_true ~msg:"gate preserved" (U.equal gate load_gate);
                is_true ~msg:"invalid remains in index"
                  (has_invalid_const src)
            | _ -> failwith "expected gated load");
            is_true ~msg:"load unchanged" (U.equal load lowered));
          test "gater strips both image indexes with same invalid gate"
            (fun () ->
            let p = global_ptr () in
            let gate =
              U.param ~slot:(-1) ~dtype:Dtype.bool ~name:"gate"
                ~addrspace:Dtype.Alu ()
            in
            let y = U.const_int 3 in
            let x = U.const_int 5 in
            let yi = U.O.where gate y (U.invalid ()) in
            let xi = U.O.where gate x (U.invalid ()) in
            let src = U.index ~ptr:p ~idxs:[ yi; xi ] () in
            let load = U.load ~src () in
            let lowered = Gater.pm_move_gates_from_index load in
            match U.as_load lowered with
            | Some { src; gate = Some load_gate; alt = Some _ } ->
                is_true ~msg:"gate moved to load" (U.equal gate load_gate);
                is_true ~msg:"invalid removed from image index"
                  (not (has_invalid_const src));
                (match U.as_index src with
                | Some { idxs = [ y'; x' ]; _ } ->
                    is_true ~msg:"y index stripped" (U.equal y y');
                    is_true ~msg:"x index stripped" (U.equal x x')
                | Some _ -> failwith "expected two image indexes"
                | None -> failwith "expected index")
            | _ -> failwith "expected gated load");
          test "gater strips stacked image load coordinates with same invalid gate"
            (fun () ->
            let p =
              U.param ~slot:0 ~dtype:Dtype.float32
                ~shape:
                  (U.stack [ U.const_int 4; U.const_int 4; U.const_int 4 ])
                ~addrspace:Dtype.Global ()
            in
            let gate =
              U.param ~slot:(-1) ~dtype:Dtype.bool ~name:"gate"
                ~addrspace:Dtype.Alu ()
            in
            let y = U.const_int 3 in
            let x = U.const_int 5 in
            let coord =
              U.stack ~dtype:Dtype.index
                [ U.O.where gate y (U.invalid ());
                  U.O.where gate x (U.invalid ()) ]
            in
            let src = U.index ~ptr:p ~idxs:[ coord ] () in
            let lowered = Gater.pm_move_gates_from_index (U.load ~src ()) in
            match U.as_load lowered with
            | Some { src; gate = Some load_gate; alt = Some _ } ->
                is_true ~msg:"gate moved to load" (U.equal gate load_gate);
                is_true ~msg:"invalid removed from image index"
                  (not (has_invalid_const src));
                (match U.as_index src with
                | Some { idxs = [ coord' ]; _ } ->
                    (match U.op coord', U.src coord' with
                    | Ops.Stack, [| y'; x' |] ->
                        is_true ~msg:"y coordinate stripped" (U.equal y y');
                        is_true ~msg:"x coordinate stripped" (U.equal x x')
                    | _ -> failwith "expected stacked image coordinate")
                | Some _ -> failwith "expected one stacked image index"
                | None -> failwith "expected index")
            | _ -> failwith "expected gated load");
          test "gater strips stacked image store coordinates with same invalid gate"
            (fun () ->
            let p =
              U.param ~slot:0 ~dtype:Dtype.float32
                ~shape:
                  (U.stack [ U.const_int 4; U.const_int 4; U.const_int 4 ])
                ~addrspace:Dtype.Global ()
            in
            let gate =
              U.param ~slot:(-1) ~dtype:Dtype.bool ~name:"gate"
                ~addrspace:Dtype.Alu ()
            in
            let y = U.const_int 3 in
            let x = U.const_int 5 in
            let coord =
              U.stack ~dtype:Dtype.index
                [ U.O.where gate y (U.invalid ());
                  U.O.where gate x (U.invalid ()) ]
            in
            let dst = U.index ~ptr:p ~idxs:[ coord ] () in
            let store = U.store ~dst ~value:(U.const_float 1.0) () in
            let lowered = Gater.pm_move_gates_from_index store in
            match U.as_store lowered with
            | Some { dst; gate = Some store_gate; value } ->
                is_true ~msg:"gate moved to store" (U.equal gate store_gate);
                is_true ~msg:"value preserved" (U.equal value (U.const_float 1.0));
                is_true ~msg:"invalid removed from image index"
                  (not (has_invalid_const dst));
                (match U.as_index dst with
                | Some { idxs = [ coord' ]; _ } ->
                    (match U.op coord', U.src coord' with
                    | Ops.Stack, [| y'; x' |] ->
                        is_true ~msg:"y coordinate stripped" (U.equal y y');
                        is_true ~msg:"x coordinate stripped" (U.equal x x')
                    | _ -> failwith "expected stacked image coordinate")
                | Some _ -> failwith "expected one stacked image index"
                | None -> failwith "expected index")
            | _ -> failwith "expected gated store");
          test "gater strips only the first variadic invalid index" (fun () ->
            let p = global_ptr () in
            let gate =
              U.param ~slot:(-1) ~dtype:Dtype.bool ~name:"gate"
                ~addrspace:Dtype.Alu ()
            in
            let i0 = U.const_int 3 in
            let i1 = U.const_int 5 in
            let g0 = U.O.where gate i0 (U.invalid ()) in
            let g1 = U.O.where gate i1 (U.invalid ()) in
            let tail = U.const_int 7 in
            let src = U.index ~ptr:p ~idxs:[ g0; g1; tail ] () in
            let load = U.load ~src () in
            let lowered = Gater.pm_move_gates_from_index load in
            match U.as_load lowered with
            | Some { src; gate = Some load_gate; alt = Some _ } ->
                is_true ~msg:"gate moved to load" (U.equal gate load_gate);
                (match U.as_index src with
                | Some { idxs = [ i0'; g1'; tail' ]; _ } ->
                    is_true ~msg:"first index stripped" (U.equal i0 i0');
                    is_true ~msg:"second invalid index preserved"
                      (has_invalid_const g1');
                    is_true ~msg:"tail preserved" (U.equal tail tail')
                | Some _ -> failwith "expected three indexes"
                | None -> failwith "expected index")
            | _ -> failwith "expected gated load");
          test "PARAM slot -1 is numbered from existing param count" (fun () ->
            let p0 = U.param ~slot:0 ~dtype:Dtype.int32 () in
            let p2 = U.param ~slot:2 ~dtype:Dtype.int32 () in
            let n =
              U.param ~slot:(-1) ~dtype:Dtype.weakint ~name:"n"
                ~vmin_vmax:(0, 8) ~addrspace:Dtype.Alu ()
            in
            let m =
              U.param ~slot:(-1) ~dtype:Dtype.weakint ~name:"m"
                ~vmin_vmax:(0, 8) ~addrspace:Dtype.Alu ()
            in
            let root = U.sink [ p0; p2; n; m ] in
            let lowered = Codegen_lower.lower (test_renderer ()) root in
            let slots = param_slots lowered in
            is_true ~msg:"all params are numbered"
              (List.for_all (fun slot -> slot >= 0) slots);
            let named_slots = named_param_slots lowered in
            equal (option int) ~msg:"n follows numbered param count"
              (Some 2) (List.assoc_opt "n" named_slots);
            equal (option int) ~msg:"m follows n"
              (Some 3) (List.assoc_opt "m" named_slots));
          test "lowering flattens sink-like children" (fun () ->
            let a = U.const (Const.float Dtype.float32 1.0) in
            let b = U.const (Const.float Dtype.float32 2.0) in
            let c = U.const (Const.float Dtype.float32 3.0) in
            let stack = U.stack ~dtype:Dtype.float32 [ b; c ] in
            let noop = U.noop ~dtype:Dtype.void () in
            let root = U.sink [ U.sink [ a ]; stack; noop ] in
            let lowered = Codegen_lower.lower (test_renderer ()) root in
            match U.op lowered, Array.to_list (U.src lowered) with
            | Ops.Sink, [ a'; b'; c' ] ->
                is_true ~msg:"nested sink is flattened" (U.equal a a');
                is_true ~msg:"stack is flattened" (U.equal b b');
                is_true ~msg:"noop is flattened away" (U.equal c c')
            | _ -> failwith "expected cleaned sink children");
          test "invalid index gate moves onto store" (fun () ->
            let p = global_ptr () in
            let gate =
              U.param ~slot:(-1) ~dtype:Dtype.bool ~name:"gate"
                ~addrspace:Dtype.Alu ()
            in
            let idx =
              U.O.where gate (U.const_int 0) (U.invalid ())
            in
            let dst = U.index ~ptr:p ~idxs:[idx] () in
            let root =
              U.sink
                [ U.store ~dst ~value:(U.const_float 1.0) () ]
            in
            let lowered = Codegen_lower.lower (test_renderer ()) root in
            let stores =
              U.toposort lowered |> List.filter_map U.as_store
            in
            match stores with
            | [ { dst; gate = Some store_gate; _ } ] ->
                (match U.as_param store_gate with
                 | Some { param; _ } ->
                     equal (option string) (Some "gate") param.name;
                     equal bool true (Dtype.equal (U.dtype store_gate) Dtype.bool)
                 | None -> failwith "expected store gate param");
                (match U.as_index dst with
                 | Some { idxs = [ idx ]; _ } ->
                     equal bool false
                       (List.exists is_invalid_const (U.toposort idx))
                 | Some _ -> failwith "expected scalar store index"
                 | None -> failwith "expected store index")
            | _ -> failwith "expected one gated store");
          test "range comparison invalid value becomes gated store" (fun () ->
            let p = global_ptr () in
            let r =
              U.range ~size:(U.const_int 256) ~axis:0
                ~kind:Axis_type.Global ()
            in
            let gate =
              U.alu_binary ~op:Ops.Cmplt ~lhs:r ~rhs:(U.const_int 200)
            in
            let dst = U.index ~ptr:p ~idxs:[r] () in
            let value =
              U.O.where gate (U.const_float 1.0)
                (U.invalid ~dtype:Dtype.float32 ())
            in
            let st = U.store ~dst ~value () in
            let root =
              U.sink
                ~kernel_info:
                  {
                    U.name = "gated_store";
                    axis_types = [ Axis_type.Global ];
                    dont_use_locals = false;
                    applied_opts = [];
                    opts_to_apply = Some [];
                    estimates = None;
                    beam = 0;
                  }
                [ U.end_ ~value:st ~ranges:[ r ] ]
            in
            let lowered = Codegen_lower.lower (test_renderer ()) root in
            let stores =
              U.toposort lowered |> List.filter_map U.as_store
            in
            match stores with
            | [ { gate = Some gate; _ } ] ->
                is_true ~msg:"store gate remains a comparison"
                  (U.op gate = Ops.Cmplt)
            | _ -> failwith "expected one gated store with comparison gate");
          test "SPEC verifies final program spec" (fun () ->
            let extra_matcher node =
              match custom_fmt node with
              | Some "make_movement()" ->
                  let one = U.const (Const.int Dtype.int32 1) in
                  Some (U.reshape ~src:one ~shape:one)
              | _ -> None
            in
            let marker =
              U.custom_inline ~fmt:"make_movement()" ~args:[]
                ~dtype:Dtype.int32
            in
            raises_match
              (function Spec.Verification_failed _ -> true | _ -> false)
              (fun () ->
                 with_spec "1" (fun () ->
                     ignore
                       (Codegen_lower.lower
                          (test_renderer ~extra_matcher ())
                          (U.sink [ marker ])))));
        ];
    ]
