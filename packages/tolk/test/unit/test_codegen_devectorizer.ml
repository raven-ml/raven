(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_ir

module K = Kernel

let dt = Dtype.Val.float32
let ptr = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)
let pp_kernel kernel = Format.asprintf "%a" K.pp kernel

let stub_renderer ?(supports_float4 = false) () =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~supports_float4 ~render:(fun ?name:_ _ -> "") ()

let expect_dtype msg expected actual =
  if not (Dtype.Val.equal expected actual) then
    failwith
      (Printf.sprintf "%s: expected %s, got %s" msg
         (Format.asprintf "%a" Dtype.Val.pp expected)
         (Format.asprintf "%a" Dtype.Val.pp actual))

let expect_ptr_dtype msg expected actual =
  if not (Dtype.Ptr.equal expected actual) then
    failwith
      (Printf.sprintf "%s: expected %s, got %s" msg
         (Format.asprintf "%a" Dtype.Ptr.pp expected)
         (Format.asprintf "%a" Dtype.Ptr.pp actual))

let topo_array root =
  let arr = Array.of_list (K.toposort root) in
  let tbl = K.Ref_tbl.create (Array.length arr) in
  Array.iteri (fun i node -> K.Ref_tbl.replace tbl node i) arr;
  (arr, tbl)

let node_at (arr, _) i = arr.(i)
let id_of (_, tbl) node = K.Ref_tbl.find tbl node
let topo_length (arr, _) = Array.length arr

let find_sink root =
  let arr, _ = topo_array root in
  let found = ref None in
  Array.iteri
    (fun i n ->
      match K.view n with
      | K.Sink _ -> (
          match !found with
          | None -> found := Some i
          | Some _ -> failwith "expected a single Sink")
      | _ -> ())
    arr;
  match !found with Some i -> i | None -> failwith "expected a Sink"

let reachable_indices root (idx : int) =
  let topo = topo_array root in
  let len = topo_length topo in
  let seen = Array.make len false in
  let rec visit r =
    if r >= 0 && r < len && not seen.(r) then begin
      seen.(r) <- true;
      List.iter
        (fun dep -> visit (id_of topo dep))
        (K.children (node_at topo r))
    end
  in
  visit idx;
  seen

let count_reachable root ~root_idx pred =
  let topo = topo_array root in
  let seen = reachable_indices root root_idx in
  let count = ref 0 in
  Array.iteri
    (fun i n -> if seen.(i) && pred (K.view n) then incr count)
    (fst topo);
  !count

let find_reachable root ~root_idx pred =
  let topo = topo_array root in
  let seen = reachable_indices root root_idx in
  let result = ref None in
  Array.iteri
    (fun i n ->
      if !result = None && seen.(i) && pred (K.view n) then
        result := Some (i, K.view n))
    (fst topo);
  !result

let find_all_reachable root ~root_idx pred =
  let topo = topo_array root in
  let seen = reachable_indices root root_idx in
  let results = ref [] in
  Array.iteri
    (fun i n ->
      if seen.(i) && pred (K.view n) then
        results := (i, K.view n) :: !results)
    (fst topo);
  List.rev !results

(* Helpers for common node construction *)

let f32 f = K.const (Const.float Dtype.Val.float32 f)
let i32 n = K.const (Const.int Dtype.Val.int32 n)
let idx n = K.const (Const.int Dtype.Val.index n)
let idx0 = idx 0

let rec unwrap_const n =
  match K.view n with
  | K.Const { value; _ } -> Some value
  | K.Cast { src; _ } -> unwrap_const src
  | _ -> None

let no_geps lowered sink =
  equal int 0
    (count_reachable lowered ~root_idx:sink (function
      | K.Gep _ -> true
      | _ -> false))

let no_reduces lowered sink =
  equal int 0
    (count_reachable lowered ~root_idx:sink (function
      | K.Reduce _ -> true
      | _ -> false))

let has_reachable lowered sink pred =
  is_true
    (count_reachable lowered ~root_idx:sink pred >= 1)

(* Expect to find a reachable node matching pred; fail with msg if absent *)
let expect_reachable lowered sink msg pred =
  match find_reachable lowered ~root_idx:sink pred with
  | Some v -> v
  | None -> failwith (msg ^ ":\n" ^ pp_kernel lowered)

(* Check that devectorization produces a Vectorize of per-lane ops.
   Used by binary, unary, cast, and bitcast scalarization tests. *)
let check_scalarized_vectorize lowered sink ~vec_dt ~lane_count ~lane_pred
    ~desc =
  let _, view =
    expect_reachable lowered sink ("expected vectorized scalar " ^ desc)
      (function
      | K.Vectorize { dtype; srcs } when Dtype.Val.equal (Dtype.val_of dtype) vec_dt ->
          List.for_all (fun r -> lane_pred (K.view r)) srcs
      | _ -> false)
  in
  match view with
  | K.Vectorize { srcs; dtype } ->
      expect_dtype (desc ^ " dtype") vec_dt (Dtype.val_of dtype);
      equal int lane_count (List.length srcs);
      srcs
  | _ -> failwith ("expected Vectorize: " ^ pp_kernel lowered)

(* Test runner *)

let () =
  run "Devectorizer"
    [
      group "pm_reduce"
        [
          test "reduce_to_acc creates accumulator loop" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let p1 = K.param ~idx:1 ~dtype:ptr in
            let r =
              K.range ~size:(idx 8) ~axis:0 ~kind:Axis_kind.Reduce ()
            in
            let ld = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r ] ()) () in
            let red = K.reduce ~op:`Add ~src:ld ~ranges:[ r ] ~dtype:dt in
            let st =
              K.store ~dst:(K.index ~ptr:p1 ~idxs:[ idx0 ] ())
                ~value:red ~ranges:[]
            in
            let lowered = K.sink [ st ] |> Devectorizer.pm_reduce in
            let sink = find_sink lowered in
            no_reduces lowered sink;
            has_reachable lowered sink (function
              | K.Define_reg _ -> true
              | _ -> false);
            has_reachable lowered sink (function
              | K.End _ -> true
              | _ -> false);
            has_reachable lowered sink (function
              | K.Const { value; dtype } when Dtype.Val.equal dtype dt ->
                  (match Const.view value with
                   | Const.Float 0.0 -> true
                   | _ -> false)
              | _ -> false);
            K.validate lowered);
          test "reduce identity elements match op" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let make_reduce op axis =
              let r =
                K.range ~size:(idx 4) ~axis ~kind:Axis_kind.Reduce ()
              in
              let ld =
                K.load ~src:(K.index ~ptr:p0 ~idxs:[ r ] ()) ()
              in
              K.reduce ~op ~src:ld ~ranges:[ r ] ~dtype:dt
            in
            let red_add = make_reduce `Add 0 in
            let red_mul = make_reduce `Mul 1 in
            let red_max = make_reduce `Max 2 in
            let mk_store pidx value =
              let p = K.param ~idx:pidx ~dtype:ptr in
              K.store ~dst:(K.index ~ptr:p ~idxs:[ idx0 ] ())
                ~value ~ranges:[]
            in
            let kernel =
              K.sink
                [ mk_store 1 red_add; mk_store 2 red_mul;
                  mk_store 3 red_max ]
            in
            let lowered = Devectorizer.pm_reduce kernel in
            let sink = find_sink lowered in
            let stores =
              find_all_reachable lowered ~root_idx:sink (function
                | K.Store { value; _ } ->
                    (match K.view value with K.Const _ -> true | _ -> false)
                | _ -> false)
            in
            let store_vals =
              List.filter_map
                (fun (_, v) ->
                  match v with
                  | K.Store { value; _ } -> (
                      match K.view value with
                      | K.Const { value = cv; _ } -> (
                          match Const.view cv with
                          | Const.Float f -> Some f
                          | _ -> None)
                      | _ -> None)
                  | _ -> None)
                stores
            in
            is_true (List.mem 0.0 store_vals);
            is_true (List.mem 1.0 store_vals);
            is_true (List.mem neg_infinity store_vals);
            K.validate lowered);
          test "reduce lowers parallel reduces" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let p1 = K.param ~idx:1 ~dtype:ptr in
            let p2 = K.param ~idx:2 ~dtype:ptr in
            let r =
              K.range ~size:(idx 4) ~axis:0 ~kind:Axis_kind.Reduce ()
            in
            let ld = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r ] ()) () in
            let red_add = K.reduce ~op:`Add ~src:ld ~ranges:[ r ] ~dtype:dt in
            let red_max = K.reduce ~op:`Max ~src:ld ~ranges:[ r ] ~dtype:dt in
            let mk_store p value =
              K.store ~dst:(K.index ~ptr:p ~idxs:[ idx0 ] ())
                ~value ~ranges:[]
            in
            let kernel =
              K.sink [ mk_store p1 red_add; mk_store p2 red_max ]
            in
            let lowered = Devectorizer.pm_reduce kernel in
            let sink = find_sink lowered in
            no_reduces lowered sink;
            has_reachable lowered sink (function
              | K.End { ranges = [ _ ]; _ } -> true
              | _ -> false);
            K.validate lowered);
          test "reduce folds WMMA accumulate" (fun () ->
            let dt2 = Dtype.Val.vec 2 Dtype.Val.float32 in
            let va = K.vectorize ~srcs:[ f32 1.0; f32 2.0 ] in
            let vb = K.vectorize ~srcs:[ f32 3.0; f32 4.0 ] in
            let vc = K.vectorize ~srcs:[ f32 5.0; f32 6.0 ] in
            let w =
              K.wmma ~name:"WMMA_test" ~a:va ~b:vb ~c:vc ~dtype:dt2
                ~dims:(1, 1, 1) ~dtype_in:Dtype.Float32 ~dtype_out:Dtype.Float32
                ~device:"TEST" ~threads:1
                ~upcast_axes:([ (0, 1) ], [ (0, 1) ], [ (0, 2) ])
                ~reduce_axes:[]
            in
            let sum =
              K.binary ~op:`Add ~lhs:w
                ~rhs:(K.vectorize ~srcs:[ f32 10.0; f32 20.0 ])
            in
            let lowered = K.sink [ sum ] |> Devectorizer.pm_reduce in
            let sink = find_sink lowered in
            ignore
              (expect_reachable lowered sink
                 "expected WMMA with folded accumulate in c operand"
                 (function
                 | K.Wmma { c; _ } ->
                     (match K.view c with
                      | K.Binary { op = `Add; _ } -> true
                      | _ -> false)
                 | _ -> false));
            K.validate lowered);
        ];
      group "pm_add_loads"
        [
          test "add_loads inserts loads only for value uses of Index" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let p1 = K.param ~idx:1 ~dtype:ptr in
            let idx0_node = K.index ~ptr:p0 ~idxs:[ idx0 ] ~as_ptr:false () in
            let neg = K.unary ~op:`Neg ~src:idx0_node in
            let st =
              K.store ~dst:(K.index ~ptr:p1 ~idxs:[ idx0 ] ~as_ptr:false ())
                ~value:neg ~ranges:[]
            in
            let lowered = K.sink [ st ] |> Devectorizer.pm_add_loads in
            let topo = topo_array lowered in
            equal int 9 (topo_length topo);
            let sink = find_sink lowered in
            let _, load_view =
              expect_reachable lowered sink "expected inserted Load"
                (function
                | K.Load { src; alt = None; _ } -> (
                    match K.view src with
                    | K.Index { ptr = ptr_ref; idxs = [ _ ]; gate = None; _ }
                      ->
                        (match K.view ptr_ref with
                         | K.Param { idx = 0; _ } -> true
                         | _ -> false)
                    | _ -> false)
                | _ -> false)
            in
            (match load_view with
            | K.Load { dtype; _ } ->
                expect_dtype "inserted load dtype" dt dtype
            | _ -> failwith "unreachable");
            let _, neg_view =
              expect_reachable lowered sink
                "expected Neg to consume inserted Load"
                (function
                | K.Unary { op = `Neg; src; _ } ->
                    (match K.view src with K.Load _ -> true | _ -> false)
                | _ -> false)
            in
            (match neg_view with
            | K.Unary { op = `Neg; dtype; _ } ->
                expect_dtype "neg dtype" dt dtype
            | _ -> failwith "unreachable");
            let _, idx_view =
              expect_reachable lowered sink
                "expected store destination Index to stay untouched"
                (function
                | K.Index { ptr = ptr_ref; idxs = [ _ ]; gate = None; _ } ->
                    (match K.view ptr_ref with
                     | K.Param { idx = 1; _ } -> true
                     | _ -> false)
                | _ -> false)
            in
            (match idx_view with
            | K.Index { dtype = Dtype.Ptr pty; _ } ->
                expect_ptr_dtype "store destination pointer dtype" ptr pty
            | K.Index { dtype = Dtype.Val _; _ } ->
                failwith "store destination Index should be ptr-typed after pm_add_loads"
            | _ -> failwith "unreachable");
            K.validate lowered);
        ];
      group "pm_devectorize"
        [
          test "splits vector ALU into scalar ops" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:(Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)) in
            let mk_load i =
              K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx i ] ()) ()
            in
            let v1 =
              K.vectorize
                ~srcs:[ mk_load 0; mk_load 1; mk_load 2; mk_load 3; mk_load 4 ]
            in
            let v2 =
              K.vectorize
                ~srcs:[ mk_load 5; mk_load 6; mk_load 7; mk_load 8; mk_load 9 ]
            in
            let add = K.binary ~op:`Add ~lhs:v1 ~rhs:v2 in
            let lowered =
              K.sink [ add ] |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            (* sym's sink_cleanup flattens the Vectorize, so we check for
               5 scalar Adds directly reachable from the sink. *)
            let scalar_adds =
              find_all_reachable lowered ~root_idx:sink (function
                | K.Binary { op = `Add; dtype = lane_dt; _ } ->
                    Dtype.Val.equal lane_dt dt
                | _ -> false)
            in
            equal int 5 (List.length scalar_adds);
            K.validate lowered);
          test "splits small vector comparisons" (fun () ->
            let i32_ptr = Dtype.Ptr.create Dtype.Val.int32 ~addrspace:Global ~size:(-1) in
            let p0 = K.param ~idx:0 ~dtype:i32_ptr in
            let p1 = K.param ~idx:1 ~dtype:i32_ptr in
            let ld0 = K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx0 ] ()) () in
            let ld1 = K.load ~src:(K.index ~ptr:p1 ~idxs:[ idx0 ] ()) () in
            let v1 = K.vectorize ~srcs:[ ld0; ld1 ] in
            let v2 = K.vectorize ~srcs:[ ld1; ld0 ] in
            let lowered =
              K.sink [ K.binary ~op:`Cmpeq ~lhs:v1 ~rhs:v2 ]
              |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            let scalar_cmps =
              find_all_reachable lowered ~root_idx:sink (function
                | K.Binary { op = `Cmpeq; dtype; _ } ->
                    Dtype.Val.equal dtype Dtype.Val.bool
                | _ -> false)
            in
            equal int 2 (List.length scalar_cmps);
            K.validate lowered);
          test "scalarizes unary ops on vectors" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:(Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)) in
            let mk_load i =
              K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx i ] ()) ()
            in
            let vec = K.vectorize ~srcs:[ mk_load 0; mk_load 1; mk_load 2 ] in
            let lowered =
              K.sink [ K.unary ~op:`Neg ~src:vec ]
              |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            let scalar_negs =
              find_all_reachable lowered ~root_idx:sink (function
                | K.Unary { op = `Neg; dtype; _ } ->
                    Dtype.Val.equal dtype dt
                | _ -> false)
            in
            equal int 3 (List.length scalar_negs);
            K.validate lowered);
          test "scalarizes Cast on vectors" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:(Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)) in
            let ld0 = K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx 0 ] ()) () in
            let ld1 = K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx 1 ] ()) () in
            let vec = K.vectorize ~srcs:[ ld0; ld1 ] in
            let cst = K.cast ~src:vec ~dtype:(Dtype.Val (Dtype.Val.vec 2 Dtype.Val.int32)) in
            let lowered =
              K.sink [ cst ] |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            let scalar_casts =
              find_all_reachable lowered ~root_idx:sink (function
                | K.Cast { dtype; _ } ->
                    Dtype.equal dtype Dtype.int32
                | _ -> false)
            in
            equal int 2 (List.length scalar_casts);
            K.validate lowered);
          test "scalarizes Bitcast on vectors" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:(Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)) in
            let ld0 = K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx 0 ] ()) () in
            let ld1 = K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx 1 ] ()) () in
            let vec = K.vectorize ~srcs:[ ld0; ld1 ] in
            let bc = K.bitcast ~src:vec ~dtype:(Dtype.Val.vec 2 Dtype.Val.int32) in
            let lowered =
              K.sink [ bc ] |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            let scalar_bitcasts =
              find_all_reachable lowered ~root_idx:sink (function
                | K.Bitcast { dtype; _ } ->
                    Dtype.Val.equal dtype Dtype.Val.int32
                | _ -> false)
            in
            equal int 2 (List.length scalar_bitcasts);
            K.validate lowered);
          test "reorders Cast after After" (fun () ->
            (* cast_after_after: After(Cast(x, dt), deps) -> Cast(After(x, deps), dt) *)
            let i32_ptr = Dtype.Ptr.create Dtype.Val.int32 ~addrspace:Global ~size:(-1) in
            let p0 = K.param ~idx:0 ~dtype:i32_ptr in
            let ld = K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx0 ] ()) () in
            let cst = K.cast ~src:ld ~dtype:(Dtype.float32) in
            let aft = K.after ~src:cst ~deps:[ idx0 ] in
            let lowered =
              K.sink [ aft ] |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            let _, view =
              expect_reachable lowered sink
                "expected After(Cast) to become Cast(After)"
                (function K.Cast _ -> true | _ -> false)
            in
            (match view with
            | K.Cast { src = after_ref; dtype } ->
                expect_dtype "reordered cast dtype" Dtype.Val.float32 (Dtype.val_of dtype);
                (match K.view after_ref with
                | K.After { src = load_ref; deps = [ _ ] } ->
                    (match K.view load_ref with
                    | K.Load _ -> ()
                    | _ -> failwith "expected Load under After")
                | _ -> failwith "expected After under Cast")
            | _ -> failwith "expected Cast wrapping After");
            K.validate lowered);
          test "splits oversized WMMA" (fun () ->
            let dt2 = Dtype.Val.vec 2 Dtype.Val.float32 in
            let dt4 = Dtype.Val.vec 4 Dtype.Val.float32 in
            let p0 = K.param ~idx:0 ~dtype:(Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)) in
            let mk_load i =
              K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx i ] ()) ()
            in
            let va = K.vectorize ~srcs:[ mk_load 0; mk_load 1 ] in
            let vb = K.vectorize ~srcs:[ mk_load 2; mk_load 3 ] in
            let vc =
              K.vectorize ~srcs:[ mk_load 4; mk_load 5; mk_load 6; mk_load 7 ]
            in
            let w =
              K.wmma ~name:"WMMA_test" ~a:va ~b:vb ~c:vc ~dtype:dt4
                ~dims:(1, 1, 1) ~dtype_in:Dtype.Float32
                ~dtype_out:Dtype.Float32 ~device:"TEST" ~threads:1
                ~upcast_axes:([ (0, 1) ], [ (0, 1) ], [ (0, 2) ])
                ~reduce_axes:[]
            in
            let lowered =
              K.sink [ w ] |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            (* sym's sink_cleanup flattens the Vectorize, so we check that
               the oversized WMMA was split into 2 smaller dt2 WMMAs. *)
            equal int 2
              (count_reachable lowered ~root_idx:sink (function
                | K.Wmma { dtype; _ } when Dtype.Val.equal dtype dt2 -> true
                | _ -> false));
            K.validate lowered);
          test "scalarizes vector register buffers" (fun () ->
            let vec_dt = Dtype.Val.vec 2 Dtype.Val.float32 in
            let reg_ptr =
              Dtype.Ptr.create vec_dt ~addrspace:Reg ~size:1
            in
            let def = K.define_reg ~size:1 ~dtype:reg_ptr ~slot:0 in
            let idx_ld = K.index ~ptr:def ~idxs:[ idx0 ] () in
            let ld = K.load ~src:idx_ld () in
            let idx_st = K.index ~ptr:def ~idxs:[ idx0 ] () in
            let st = K.store ~dst:idx_st ~value:ld ~ranges:[] in
            let lowered =
              K.sink [ st ] |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            let _, dreg_view =
              expect_reachable lowered sink
                "expected Define_reg to scalarize"
                (function K.Define_reg _ -> true | _ -> false)
            in
            (match dreg_view with
            | K.Define_reg { size = 2; dtype } ->
                expect_ptr_dtype "scalarized register dtype"
                  (Dtype.Ptr.create dt ~addrspace:Reg ~size:2)
                  dtype
            | _ -> failwith ("expected Define_reg to scalarize: "
                             ^ pp_kernel lowered));
            (* pm_devectorize scalarizes the buffer and vectorizes the
               index, but the Load stays vector-typed. Register buffers
               are skipped by correct_load_store. *)
            let _, ld_view =
              expect_reachable lowered sink
                "expected vector Load to remain after register devectorize"
                (function
                | K.Load { dtype; _ } when Dtype.Val.equal dtype vec_dt -> true
                | _ -> false)
            in
            (match ld_view with
            | K.Load { dtype; _ } ->
                expect_dtype "register load stays vector" vec_dt dtype
            | _ -> failwith "unreachable");
            let _, st_view =
              expect_reachable lowered sink
                "expected store in scalarized register kernel"
                (function K.Store _ -> true | _ -> false)
            in
            (match st_view with K.Store _ -> () | _ -> failwith "unreachable"));
          test "scalarizes vector local buffers" (fun () ->
            let vec_dt = Dtype.Val.vec 2 Dtype.Val.float32 in
            let local_ptr =
              Dtype.Ptr.create vec_dt ~addrspace:Local ~size:1
            in
            let def = K.define_local ~size:1 ~dtype:local_ptr in
            let idx_ld = K.index ~ptr:def ~idxs:[ idx0 ] () in
            let ld = K.load ~src:idx_ld () in
            let idx_st = K.index ~ptr:def ~idxs:[ idx0 ] () in
            let st = K.store ~dst:idx_st ~value:ld ~ranges:[] in
            let lowered =
              K.sink [ st ] |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            let _, dloc_view =
              expect_reachable lowered sink
                "expected Define_local to scalarize"
                (function K.Define_local _ -> true | _ -> false)
            in
            (match dloc_view with
            | K.Define_local { size = 2; dtype } ->
                expect_ptr_dtype "scalarized local dtype"
                  (Dtype.Ptr.create dt ~addrspace:Local ~size:2)
                  dtype
            | _ -> failwith ("expected Define_local to scalarize: "
                             ^ pp_kernel lowered)));
          test "rewrites vector index on local/reg" (fun () ->
            let vec_dt = Dtype.Val.vec 2 Dtype.Val.float32 in
            let local_ptr =
              Dtype.Ptr.create vec_dt ~addrspace:Local ~size:4
            in
            let def = K.define_local ~size:4 ~dtype:local_ptr in
            let var = K.define_var ~name:"i" ~lo:0 ~hi:3 () in
            let ld =
              K.load ~src:(K.index ~ptr:def ~idxs:[ var ] ()) ()
            in
            let lowered =
              K.sink [ ld ] |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            (* The vector index is rewritten to scalar indices with stride
               multiplication.  Check we get 2 scalar loads. *)
            let scalar_loads =
              find_all_reachable lowered ~root_idx:sink (function
                | K.Load { dtype; _ } -> Dtype.Val.equal dtype dt
                | _ -> false)
            in
            equal int 2 (List.length scalar_loads);
            K.validate lowered);
          test "preserves WHERE with Invalid_index" (fun () ->
            let vec_dt = Dtype.Val.vec 2 Dtype.Val.index in
            let p0 = K.param ~idx:0 ~dtype:(Dtype.Ptr.create Dtype.Val.bool ~addrspace:Global ~size:(-1)) in
            let cond = K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx0 ] ()) () in
            let val_vec = K.vectorize ~srcs:[ idx 0; idx 1 ] in
            let inv = K.invalid_index ~lanes:2 () in
            let wh =
              K.ternary ~op:`Where
                ~a:cond
                ~b:val_vec ~c:inv
            in
            let lowered =
              K.sink [ wh ] |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            let _, view =
              expect_reachable lowered sink
                "expected WHERE with Invalid_index to be preserved"
                (function
                | K.Ternary { op = `Where; _ } -> true
                | _ -> false)
            in
            (match view with
            | K.Ternary { dtype; _ } ->
                expect_dtype "preserved Where dtype" vec_dt dtype
            | _ -> failwith "unreachable");
            K.validate lowered);
          test "drops true gate from Index" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let gate = K.const (Const.bool true) in
            let gated_idx = K.index ~ptr:p0 ~idxs:[ idx0 ] ~gate () in
            let ld = K.load ~src:gated_idx () in
            let lowered =
              K.sink [ ld ] |> Devectorizer.pm_devectorize (stub_renderer ())
            in
            let sink = find_sink lowered in
            equal int 0
              (count_reachable lowered ~root_idx:sink (function
                | K.Index { gate = Some _; _ } -> true
                | _ -> false));
            has_reachable lowered sink (function
              | K.Index { gate = None; _ } -> true
              | _ -> false);
            K.validate lowered);
        ];
      group "pm_correct_load_store"
        [
          (* Tinygrad's correct_load_store matches Load(Cast(Index)) /
             Store(Cast(Index)). Input must wrap Index in Cast. Output
             is Vcat of scalar Loads / Group of scalar Stores,
             each with a new Index src (not Gep). *)
          test "splits vector load to scalar" (fun () ->
            let ren = stub_renderer () in
            let vec_ptr =
              Dtype.Ptr.create (Dtype.Val.vec 4 dt) ~addrspace:Global ~size:(-1)
            in
            let p0 = K.param ~idx:0 ~dtype:vec_ptr in
            let index = K.index ~ptr:p0 ~idxs:[ idx0 ] () in
            let cast_idx =
              K.cast ~src:index ~dtype:(Dtype.Ptr vec_ptr)
            in
            let ld = K.load ~src:cast_idx () in
            let lowered =
              K.sink [ ld ] |> Devectorizer.pm_devectorize ren
            in
            let sink = find_sink lowered in
            (* pm_devectorize splits the vector load and sym simplifies
               the Vcat away; verify 4 scalar loads with Index sources. *)
            let scalar_loads =
              find_all_reachable lowered ~root_idx:sink (function
                | K.Load { dtype; _ } -> Dtype.Val.equal dtype dt
                | _ -> false)
            in
            equal int 4 (List.length scalar_loads);
            List.iter
              (fun (_, view) ->
                match view with
                | K.Load { src; dtype; _ } ->
                    expect_dtype "scalar load dtype" dt dtype;
                    (match K.view src with
                    | K.Index _ -> ()
                    | _ ->
                        failwith ("expected Index source for split load: "
                                  ^ pp_kernel lowered))
                | _ -> failwith "unreachable")
              scalar_loads;
            K.validate lowered);
          test "splits vector store to scalar" (fun () ->
            let ren = stub_renderer () in
            let vec_ptr =
              Dtype.Ptr.create (Dtype.Val.vec 4 dt) ~addrspace:Global ~size:(-1)
            in
            let p0 = K.param ~idx:0 ~dtype:vec_ptr in
            let index = K.index ~ptr:p0 ~idxs:[ idx0 ] () in
            let cast_idx =
              K.cast ~src:index ~dtype:(Dtype.Ptr vec_ptr)
            in
            let vec_val =
              K.vectorize ~srcs:[ f32 1.0; f32 2.0; f32 3.0; f32 4.0 ]
            in
            let st = K.store ~dst:cast_idx ~value:vec_val ~ranges:[] in
            let lowered =
              K.sink [ st ] |> Devectorizer.pm_devectorize ren
            in
            let sink = find_sink lowered in
            let _, view =
              expect_reachable lowered sink
                "expected vector Store split into Group of scalar Stores"
                (function
                | K.Group { srcs } ->
                    List.for_all
                      (fun r ->
                        match K.view r with
                        | K.Store _ -> true
                        | _ -> false)
                      srcs
                | _ -> false)
            in
            (match view with
            | K.Group { srcs } ->
                equal int 4 (List.length srcs);
                List.iteri
                  (fun lane_idx st_node ->
                    match K.view st_node with
                    | K.Store { dst; value; _ } ->
                        (* dst is an Index (not Gep) *)
                        (match K.view dst with
                        | K.Index _ -> ()
                        | _ ->
                            failwith ("expected Index dst for split store: "
                                      ^ pp_kernel lowered));
                        (* value is scalar (Gep may simplify away) *)
                        (match K.dtype_opt value with
                        | Some vdt ->
                            expect_dtype "scalar store value" dt (Dtype.val_of vdt)
                        | None ->
                            failwith ("expected scalar value dtype: "
                                      ^ pp_kernel lowered))
                    | _ ->
                        failwith ("expected Store in Group: "
                                  ^ pp_kernel lowered))
                  srcs
            | _ -> failwith "unreachable"));
          test "preserves alt per lane" (fun () ->
            let ren = stub_renderer () in
            let vec_ptr =
              Dtype.Ptr.create (Dtype.Val.vec 2 dt) ~addrspace:Global ~size:(-1)
            in
            let p0 = K.param ~idx:0 ~dtype:vec_ptr in
            let gate = K.const (Const.bool true) in
            let index =
              K.index ~ptr:p0 ~idxs:[ idx0 ] ~gate ()
            in
            let cast_idx =
              K.cast ~src:index ~dtype:(Dtype.Ptr vec_ptr)
            in
            let vec_alt = K.vectorize ~srcs:[ f32 42.0; f32 99.0 ] in
            let ld = K.load ~src:cast_idx ~alt:vec_alt () in
            let lowered =
              K.sink [ ld ] |> Devectorizer.pm_devectorize ren
            in
            let sink = find_sink lowered in
            (* pm_devectorize splits the vector load and sym simplifies
               the Vcat away; verify 2 scalar loads with alt preserved. *)
            let scalar_loads =
              find_all_reachable lowered ~root_idx:sink (function
                | K.Load { alt = Some _; dtype; _ } ->
                    Dtype.Val.equal dtype dt
                | _ -> false)
            in
            equal int 2 (List.length scalar_loads);
            List.iter
              (fun (_, view) ->
                match view with
                | K.Load { alt = Some _; dtype; _ } ->
                    expect_dtype "scalar split load dtype" dt dtype
                | _ -> failwith "unreachable")
              scalar_loads);
          test "skips Reg addrspace" (fun () ->
            let ren = stub_renderer () in
            let reg_ptr =
              Dtype.Ptr.create (Dtype.Val.vec 2 dt) ~addrspace:Reg ~size:(-1)
            in
            let p0 = K.param ~idx:0 ~dtype:reg_ptr in
            let ld =
              K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx0 ] ()) ()
            in
            let lowered =
              K.sink [ ld ] |> Devectorizer.pm_devectorize ren
            in
            let sink = find_sink lowered in
            equal int 0
              (count_reachable lowered ~root_idx:sink (function
                | K.Vectorize _ -> true
                | _ -> false));
            has_reachable lowered sink (function
              | K.Load { dtype; _ } -> Dtype.Val.equal dtype (Dtype.Val.vec 2 dt)
              | _ -> false));
          test "skips when renderer supports width" (fun () ->
            let ren =
              stub_renderer ~supports_float4:true ()
            in
            let vec_ptr =
              Dtype.Ptr.create (Dtype.Val.vec 4 dt) ~addrspace:Global ~size:(-1)
            in
            let p0 = K.param ~idx:0 ~dtype:vec_ptr in
            let ld =
              K.load ~src:(K.index ~ptr:p0 ~idxs:[ idx0 ] ()) ()
            in
            let lowered =
              K.sink [ ld ] |> Devectorizer.pm_devectorize ren
            in
            let sink = find_sink lowered in
            equal int 0
              (count_reachable lowered ~root_idx:sink (function
                | K.Vectorize _ -> true
                | _ -> false));
            has_reachable lowered sink (function
              | K.Load { dtype; _ } -> Dtype.Val.equal dtype (Dtype.Val.vec 4 dt)
              | _ -> false));
        ];
      group "pm_render"
        [
          test "adds a zero alt to gated loads" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let gate = K.const (Const.bool true) in
            let gated_idx = K.index ~ptr:p0 ~idxs:[ idx0 ] ~gate () in
            let ld = K.load ~src:gated_idx () in
            let lowered =
              K.sink [ ld ] |> Devectorizer.pm_render
            in
            let sink = find_sink lowered in
            let _, view =
              expect_reachable lowered sink
                "expected masked load alt insertion"
                (function
                | K.Load { alt = Some alt; _ } ->
                    (match K.view alt with
                    | K.Const { value; _ } ->
                        (match Const.view value with
                        | Const.Float 0.0 -> true
                        | _ -> false)
                    | _ -> false)
                | _ -> false)
            in
            (match view with
            | K.Load { alt = Some alt; dtype; _ } ->
                expect_dtype "masked load dtype" dt dtype;
                (match K.view alt with
                | K.Const { value; dtype } ->
                    (match Const.view value with
                    | Const.Float 0.0 ->
                        expect_dtype "masked load alt dtype" dt dtype
                    | _ -> failwith "expected zero alt constant")
                | _ -> failwith "expected zero alt constant")
            | _ -> failwith "unreachable");
            K.validate lowered);
          test "folds Where after gated load into alt" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let gate = K.const (Const.bool true) in
            let alt_val = K.const (Const.float dt 9.0) in
            let gated_idx = K.index ~ptr:p0 ~idxs:[ idx0 ] ~gate () in
            let ld = K.load ~src:gated_idx () in
            let wh = K.ternary ~op:`Where ~a:gate ~b:ld ~c:alt_val in
            let lowered =
              K.sink [ wh ] |> Devectorizer.pm_render
            in
            let sink = find_sink lowered in
            let _, view =
              expect_reachable lowered sink
                "expected Where(gated Load, alt) to fold into Load alt"
                (function
                | K.Load { alt = Some alt; _ } ->
                    (match unwrap_const alt with
                    | Some v ->
                        (match Const.view v with
                        | Const.Float 9.0 -> true
                        | _ -> false)
                    | None -> false)
                | _ -> false)
            in
            (match view with
            | K.Load { alt = Some alt; dtype; _ } ->
                expect_dtype "folded where load dtype" dt dtype;
                (match unwrap_const alt with
                | Some value ->
                    (match Const.view value with
                    | Const.Float 9.0 -> ()
                    | _ -> failwith
                             "expected Where(gated Load, alt) to fold")
                | None -> failwith
                         "expected Where(gated Load, alt) to fold")
            | _ -> failwith "unreachable");
            K.validate lowered);
          test "folds Where with negated gate into alt" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let gate = K.const (Const.bool true) in
            let negated_gate =
              K.binary ~op:`Xor ~lhs:gate
                ~rhs:(K.const (Const.bool true))
            in
            let alt_val = K.const (Const.float dt 5.0) in
            let gated_idx =
              K.index ~ptr:p0 ~idxs:[ idx0 ] ~gate:negated_gate ()
            in
            let ld = K.load ~src:gated_idx () in
            let wh = K.ternary ~op:`Where ~a:gate ~b:alt_val ~c:ld in
            let lowered =
              K.sink [ wh ] |> Devectorizer.pm_render
            in
            let sink = find_sink lowered in
            equal int 0
              (count_reachable lowered ~root_idx:sink (function
                | K.Ternary { op = `Where; _ } -> true
                | _ -> false));
            let _, view =
              expect_reachable lowered sink
                "expected negated gate Where to fold into Load alt"
                (function
                | K.Load { alt = Some alt; _ } ->
                    (match unwrap_const alt with
                    | Some v ->
                        (match Const.view v with
                        | Const.Float 5.0 -> true
                        | _ -> false)
                    | None -> false)
                | _ -> false)
            in
            (match view with
            | K.Load { alt = Some _; dtype; _ } ->
                expect_dtype "negated gate folded load dtype" dt dtype
            | _ -> failwith "unreachable");
            K.validate lowered);
          test "folds Where with Cast-wrapped gated load" (fun () ->
            let load_dt = Dtype.Val.int32 in
            let cast_dt = Dtype.Val.float32 in
            let load_ptr =
              Dtype.Ptr.create load_dt ~addrspace:Global ~size:(-1)
            in
            let p0 = K.param ~idx:0 ~dtype:load_ptr in
            let gate = K.const (Const.bool true) in
            let gated_idx = K.index ~ptr:p0 ~idxs:[ idx0 ] ~gate () in
            let ld = K.load ~src:gated_idx () in
            let casted_load = K.cast ~src:ld ~dtype:(Dtype.Val cast_dt) in
            let alt_val = K.const (Const.float cast_dt 5.0) in
            let wh =
              K.ternary ~op:`Where ~a:gate ~b:casted_load ~c:alt_val
            in
            let lowered =
              K.sink [ wh ] |> Devectorizer.pm_render
            in
            let sink = find_sink lowered in
            equal int 0
              (count_reachable lowered ~root_idx:sink (function
                | K.Ternary { op = `Where; _ } -> true
                | _ -> false));
            has_reachable lowered sink (function
              | K.Cast { dtype; _ } when Dtype.equal dtype (Dtype.Val cast_dt) -> true
              | _ -> false);
            has_reachable lowered sink (function
              | K.Load { alt = Some _; _ } -> true
              | _ -> false);
            K.validate lowered);
          test "Where with different gate does not fold" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let gate1 = K.const (Const.bool true) in
            let r =
              K.range
                ~size:(idx 10) ~axis:0 ~kind:Axis_kind.Loop ()
            in
            let gate2 =
              K.binary ~op:`Cmplt ~lhs:r ~rhs:(idx 5)
            in
            let alt_val = K.const (Const.float dt 5.0) in
            let gated_idx =
              K.index ~ptr:p0 ~idxs:[ idx0 ] ~gate:gate1 ()
            in
            let ld = K.load ~src:gated_idx () in
            let wh = K.ternary ~op:`Where ~a:gate2 ~b:ld ~c:alt_val in
            let lowered =
              K.sink [ wh ] |> Devectorizer.pm_render
            in
            let sink = find_sink lowered in
            has_reachable lowered sink (function
              | K.Ternary { op = `Where; _ } -> true
              | _ -> false);
            K.validate lowered);
        ];
      group "integration"
        [
          test "full pipeline: reduce + gated load" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let p1 = K.param ~idx:1 ~dtype:ptr in
            let r =
              K.range ~size:(idx 4) ~axis:0 ~kind:Axis_kind.Reduce ()
            in
            let gate =
              K.binary ~op:`Cmplt ~lhs:r ~rhs:(idx 3)
            in
            let ld =
              K.load ~src:(K.index ~ptr:p0 ~idxs:[ r ] ~gate ()) ()
            in
            let red = K.reduce ~op:`Add ~src:ld ~ranges:[ r ] ~dtype:dt in
            let st =
              K.store ~dst:(K.index ~ptr:p1 ~idxs:[ idx0 ] ())
                ~value:red ~ranges:[]
            in
            let result =
              K.sink [ st ]
              |> Devectorizer.pm_reduce
              |> Devectorizer.pm_add_loads
              |> Devectorizer.pm_devectorize (stub_renderer ())
              |> Devectorizer.pm_render
            in
            let sink = find_sink result in
            no_reduces result sink;
            equal int 0
              (count_reachable result ~root_idx:sink (function
                | K.Load { alt = None; src; _ } ->
                    let rec has_gate n =
                      match K.view n with
                      | K.Index { gate = Some _; _ } -> true
                      | K.Cast { src; _ } | K.Bitcast { src; _ } ->
                          has_gate src
                      | _ -> false
                    in
                    has_gate src
                | _ -> false));
            K.validate result);
        ];
    ]
