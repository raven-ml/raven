(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_ir

module K = Kernel

let dt = Dtype.float32
let ptr = Dtype.Ptr.create dt ~addrspace:Global ()
let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ()
let pp_kernel root = Format.asprintf "%a" K.pp root
let i32 n = K.const (Const.int Dtype.int32 n)
let idx_const n = K.const (Const.int Dtype.index n)

let kernel_info ?opts_to_apply () =
  {
    K.name = "";
    axis_kinds = [];
    dont_use_locals = false;
    applied_opts = [];
    opts_to_apply;
    estimates = None;
    metadata_tags = [];
  }

let metal_like_tc =
  {
    Renderer.dims = (8, 8, 8);
    threads = 32;
    elements_per_thread = (2, 2, 2);
    dtype_in = Dtype.Float32;
    dtype_out = Dtype.Float32;
    opts = [ "u0"; "l0"; "l1"; "l1"; "l0"; "l1" ];
    swizzle =
      ( ([ "r1"; "l1"; "l2"; "r2"; "l4" ], [ "r0" ], [ "u0"; "l0"; "l3" ]),
        ([ "l0"; "r0"; "r1"; "l3"; "r2" ], [ "u0" ], [ "l1"; "l2"; "l4" ]) );
  }

let noop_renderer ?(tensor_cores = []) () =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~tensor_cores
    ~render:(fun ?name:_ _ -> "")
    ()

let topo_array root =
  let arr = Array.of_list (K.toposort root) in
  let tbl = K.Ref_tbl.create (Array.length arr) in
  Array.iteri (fun i node -> K.Ref_tbl.replace tbl node i) arr;
  (arr, tbl)

let id_of (_, tbl) node = K.Ref_tbl.find tbl node
let topo_length (arr, _) = Array.length arr

let reachable_set root =
  let topo = topo_array root in
  let len = topo_length topo in
  let arr, _ = topo in
  let sink_idx =
    let found = ref None in
    Array.iteri
      (fun i n ->
        match K.view n with K.Sink _ -> found := Some i | _ -> ())
      arr;
    Option.get !found
  in
  let seen = Array.make len false in
  let rec visit idx =
    if idx >= 0 && idx < len && not seen.(idx) then begin
      seen.(idx) <- true;
      List.iter
        (fun dep -> visit (id_of topo dep))
        (K.children arr.(idx))
    end
  in
  visit sink_idx;
  (topo, seen, sink_idx)

let find_sink root =
  let arr, _ = topo_array root in
  let found = ref None in
  Array.iteri
    (fun _ n ->
      match K.view n with K.Sink _ -> found := Some n | _ -> ())
    arr;
  Option.get !found

let count_reachable root pred =
  let (topo, seen, _) = reachable_set root in
  let arr, _ = topo in
  let count = ref 0 in
  Array.iteri
    (fun i n -> if seen.(i) && pred (K.view n) then incr count)
    arr;
  !count

let sink_children root =
  match K.view (find_sink root) with
  | K.Sink { srcs; _ } -> srcs
  | _ -> failwith "expected Sink"

let const_int_value node =
  match K.view node with
  | K.Const { value; _ } -> (
      match Const.view value with
      | Int n -> Int64.to_int n
      | _ -> failwith "expected int const")
  | _ ->
      failwith
        (Printf.sprintf "expected Const, got %s"
           (Format.asprintf "%a" K.pp_view node))

let sink_int_values root =
  List.map const_int_value (sink_children root)

let rec take n xs =
  if n <= 0 then []
  else match xs with [] -> [] | x :: xs -> x :: take (n - 1) xs

(* Shared assertion: no Contract or Unroll markers remain after expansion *)
let assert_no_contract_unroll expanded =
  let _ = find_sink expanded in
  equal int 0
    (count_reachable expanded (function
      | K.Contract _ | K.Unroll _ -> true
      | _ -> false))

(* Build vectorize -> unroll -> contract -> sink, expand, assert clean *)
let expand_vec_contract ~consts ~unroll_axes ~contract_axes ~vec_width =
  let vec = K.vectorize ~srcs:consts in
  let unroll =
    K.unroll ~src:vec ~axes:unroll_axes ~dtype:Dtype.int32
  in
  let contract =
    K.contract ~src:unroll ~axes:contract_axes
      ~dtype:(Dtype.vec Dtype.int32 vec_width)
  in
  let root = K.sink ~kernel_info:(kernel_info ()) [ contract ] in
  let expanded = Expander.expand root in
  assert_no_contract_unroll expanded;
  expanded

let grouped_reduce_kernel () =
  let p1 = K.param ~idx:1 ~dtype:ptr in
  let r0 = K.range ~size:(idx_const 2) ~axis:0 ~kind:Axis_kind.Local () in
  let r1 =
    K.range ~size:(idx_const 4) ~axis:1 ~kind:Axis_kind.Group_reduce ()
  in
  let idx = K.index ~ptr:p1 ~idxs:[ r0; r1 ] () in
  let ld = K.load ~src:idx () in
  let red = K.reduce ~op:`Add ~src:ld ~ranges:[ r1 ] ~dtype:dt in
  K.sink ~kernel_info:(kernel_info ()) [ red ]

let () =
  run "Expander"
    [
      group "expander core"
        [
          test "expand lowers reachable contract markers" (fun () ->
            let _ = K.range ~size:(idx_const 4) ~axis:0
              ~kind:Axis_kind.Global () in
            let cf = K.const (Const.float Dtype.float32 3.0) in
            let contract =
              K.contract ~src:cf ~axes:[ (0, 2) ]
                ~dtype:(Dtype.vec Dtype.float32 2)
            in
            let root = K.sink ~kernel_info:(kernel_info ()) [ contract ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            let children = sink_children expanded in
            equal int 2 (List.length children);
            is_true (List.hd children == List.nth children 1);
            K.validate expanded);

          test "expand lowers tensor-core contract and unroll markers" (fun () ->
            let _renderer =
              noop_renderer ~tensor_cores:[ metal_like_tc ] ()
            in
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let p1 = K.param ~idx:1 ~dtype:ptr in
            let c8 = idx_const 8 in
            let r0 = K.range ~size:c8 ~axis:0 ~kind:Axis_kind.Global () in
            let r1 = K.range ~size:c8 ~axis:1 ~kind:Axis_kind.Global () in
            let r2 = K.range ~size:c8 ~axis:2 ~kind:Axis_kind.Reduce () in
            let idx0 = K.index ~ptr:p0 ~idxs:[ r0; r2 ] () in
            let idx1 = K.index ~ptr:p1 ~idxs:[ r2; r1 ] () in
            let ld0 = K.load ~src:idx0 () in
            let ld1 = K.load ~src:idx1 () in
            let zero = K.const (Const.float Dtype.float32 0.0) in
            let wmma =
              K.wmma ~name:"__metal_simdgroup_matrix_fma" ~a:ld0 ~b:ld1
                ~c:zero ~dtype:dt ~dims:(8, 8, 8)
                ~dtype_in:Dtype.Float32 ~dtype_out:Dtype.Float32
                ~device:"TEST" ~threads:32
                ~upcast_axes:([ (0, 2) ], [ (0, 2) ], [ (0, 2); (1, 2) ])
                ~reduce_axes:[ 2 ]
            in
            let unroll =
              K.unroll ~src:wmma ~axes:[ (0, 2); (1, 2); (2, 2) ] ~dtype:dt
            in
            let contract =
              K.contract ~src:unroll ~axes:[ (0, 2); (1, 2) ]
                ~dtype:(Dtype.vec dt 4)
            in
            let root =
              K.sink
                ~kernel_info:
                  (kernel_info
                     ~opts_to_apply:
                       [
                         K.Opt.Tc
                           { axis = 0; tc_select = -1; tc_opt = 0;
                             use_tc = 1 };
                       ]
                     ())
                [ contract ]
            in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            equal int 1
              (count_reachable expanded (function
                | K.Wmma _ -> true
                | _ -> false));
            is_true
              (count_reachable expanded (function
                | K.Gep _ -> true
                | _ -> false) > 0);
            K.validate expanded);

          test "expand fully contracts consumed unroll markers" (fun () ->
            let consts = List.init 4 (fun i -> i32 i) in
            let expanded =
              expand_vec_contract ~consts ~unroll_axes:[ (1, 4) ]
                ~contract_axes:[ (1, 4) ] ~vec_width:4
            in
            equal (list int) [ 0; 1; 2; 3 ] (sink_int_values expanded);
            K.validate expanded);

          test "expand flattens nested unroll markers" (fun () ->
            let consts = List.init 8 (fun i -> i32 i) in
            let vec = K.vectorize ~srcs:consts in
            let unroll1 =
              K.unroll ~src:vec ~axes:[ (1, 4) ]
                ~dtype:(Dtype.vec Dtype.int32 2)
            in
            let unroll2 =
              K.unroll ~src:unroll1 ~axes:[ (2, 2) ] ~dtype:Dtype.int32
            in
            let root = K.sink ~kernel_info:(kernel_info ()) [ unroll2 ] in
            let expanded = Expander.expand root in
            let _ = find_sink expanded in
            equal int 0
              (count_reachable expanded (function
                | K.Unroll _ -> true
                | _ -> false));
            equal (list int) [ 0; 1; 2; 3; 4; 5; 6; 7 ]
              (sink_int_values expanded);
            K.validate expanded);

          test "expand preserves remaining unroll axes after contract" (fun () ->
            let consts = List.init 8 (fun i -> i32 i) in
            let vec = K.vectorize ~srcs:consts in
            let unroll =
              K.unroll ~src:vec ~axes:[ (1, 4); (2, 2) ] ~dtype:Dtype.int32
            in
            let contract =
              K.contract ~src:unroll ~axes:[ (1, 4) ]
                ~dtype:(Dtype.vec Dtype.int32 4)
            in
            let root = K.sink ~kernel_info:(kernel_info ()) [ contract ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            equal (list int) [ 0; 2; 4; 6; 1; 3; 5; 7 ]
              (sink_int_values expanded);
            K.validate expanded);

          test "expand contract without unroll repeats scalar" (fun () ->
            let contract =
              K.contract ~src:(i32 7) ~axes:[ (2, 2) ]
                ~dtype:(Dtype.vec Dtype.int32 2)
            in
            let root = K.sink ~kernel_info:(kernel_info ()) [ contract ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            let children = sink_children expanded in
            equal int 2 (List.length children);
            is_true (List.hd children == List.nth children 1);
            equal (list int) [ 7; 7 ] (sink_int_values expanded);
            K.validate expanded);

          test "expand contract axis order" (fun () ->
            let consts = List.init 16 (fun i -> i32 i) in
            let expanded =
              expand_vec_contract ~consts
                ~unroll_axes:[ (1, 4); (2, 4) ]
                ~contract_axes:[ (1, 4) ] ~vec_width:4
            in
            equal (list int)
              [ 0; 4; 8; 12; 1; 5; 9; 13; 2; 6; 10; 14; 3; 7; 11; 15 ]
              (sink_int_values expanded);
            K.validate expanded);

          test "expand contract half-expand duplicates lanes" (fun () ->
            let consts = List.init 4 (fun i -> i32 i) in
            let vec = K.vectorize ~srcs:consts in
            let unroll =
              K.unroll ~src:vec ~axes:[ (1, 4) ] ~dtype:Dtype.int32
            in
            let contract =
              K.contract ~src:unroll ~axes:[ (1, 4); (2, 2) ]
                ~dtype:(Dtype.vec Dtype.int32 8)
            in
            let root = K.sink ~kernel_info:(kernel_info ()) [ contract ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            equal (list int) [ 0; 0; 1; 1; 2; 2; 3; 3 ]
              (sink_int_values expanded);
            K.validate expanded);

          test "expand add broadcast" (fun () ->
            let consts = List.init 4 (fun i -> i32 i) in
            let vec = K.vectorize ~srcs:consts in
            let unroll =
              K.unroll ~src:vec ~axes:[ (1, 4) ] ~dtype:Dtype.int32
            in
            let add = K.binary ~op:`Add ~lhs:unroll ~rhs:(i32 3) in
            let root = K.sink ~kernel_info:(kernel_info ()) [ add ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            equal int 1
              (count_reachable expanded (function
                | K.Binary _ -> true
                | _ -> false));
            is_true
              (count_reachable expanded (function
                 | K.Vectorize { srcs = x :: xs; _ } ->
                     List.for_all (fun s -> s == x) xs
                 | _ -> false)
              > 0);
            K.validate expanded);

          test "expand same-axis add" (fun () ->
            let consts_a = List.init 4 (fun i -> i32 i) in
            let consts_b = List.init 4 (fun i -> i32 (i * 4)) in
            let unroll_a =
              K.unroll ~src:(K.vectorize ~srcs:consts_a)
                ~axes:[ (1, 4) ] ~dtype:Dtype.int32
            in
            let unroll_b =
              K.unroll ~src:(K.vectorize ~srcs:consts_b)
                ~axes:[ (1, 4) ] ~dtype:Dtype.int32
            in
            let add = K.binary ~op:`Add ~lhs:unroll_a ~rhs:unroll_b in
            let root = K.sink ~kernel_info:(kernel_info ()) [ add ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            equal int 1
              (count_reachable expanded (function
                | K.Binary _ -> true
                | _ -> false));
            equal int 2
              (count_reachable expanded (function
                | K.Vectorize _ -> true
                | _ -> false));
            K.validate expanded);

          test "expand different-axis add" (fun () ->
            let consts_a = List.init 4 (fun i -> i32 (i * 4)) in
            let consts_b = List.init 4 (fun i -> i32 i) in
            let unroll_a =
              K.unroll ~src:(K.vectorize ~srcs:consts_a)
                ~axes:[ (1, 4) ] ~dtype:Dtype.int32
            in
            let unroll_b =
              K.unroll ~src:(K.vectorize ~srcs:consts_b)
                ~axes:[ (2, 4) ] ~dtype:Dtype.int32
            in
            let add = K.binary ~op:`Add ~lhs:unroll_a ~rhs:unroll_b in
            let root = K.sink ~kernel_info:(kernel_info ()) [ add ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            equal int 1
              (count_reachable expanded (function
                | K.Binary _ -> true
                | _ -> false));
            equal int 2
              (count_reachable expanded (function
                | K.Vectorize _ -> true
                | _ -> false));
            K.validate expanded);

          test "expand contract multi-axis order" (fun () ->
            let build axes =
              let consts = List.init 16 (fun i -> i32 i) in
              let vec = K.vectorize ~srcs:consts in
              let unroll =
                K.unroll ~src:vec
                  ~axes:[ (1, 2); (2, 2); (3, 2); (4, 2) ]
                  ~dtype:Dtype.int32
              in
              let contract =
                K.contract ~src:unroll ~axes
                  ~dtype:(Dtype.vec Dtype.int32 4)
              in
              K.sink ~kernel_info:(kernel_info ()) [ contract ]
            in
            let assert_prefix axes expected =
              let expanded = Expander.expand (build axes) in
              assert_no_contract_unroll expanded;
              equal (list int) expected
                (sink_int_values expanded |> take 4);
              K.validate expanded
            in
            assert_prefix [ (3, 2); (2, 2) ] [ 0; 4; 2; 6 ];
            assert_prefix [ (2, 2); (3, 2) ] [ 0; 2; 4; 6 ]);
        ];

      group "contract and expand edge cases"
        [
          test "contract axis 2 from 2-axis UNROLL" (fun () ->
            let consts = List.init 16 (fun i -> i32 i) in
            let expanded =
              expand_vec_contract ~consts
                ~unroll_axes:[ (1, 4); (2, 4) ]
                ~contract_axes:[ (2, 4) ] ~vec_width:4
            in
            let vals = sink_int_values expanded in
            equal (list int) [ 0; 1; 2; 3 ] (take 4 vals);
            equal (list int) [ 12; 13; 14; 15 ]
              (take 4 (List.filteri (fun i _ -> i >= 12) vals));
            K.validate expanded);

          test "contract axis 2 from 4-axis UNROLL" (fun () ->
            let consts = List.init 16 (fun i -> i32 i) in
            let expanded =
              expand_vec_contract ~consts
                ~unroll_axes:[ (1, 2); (2, 2); (3, 2); (4, 2) ]
                ~contract_axes:[ (2, 2) ] ~vec_width:2
            in
            let vals = sink_int_values expanded in
            equal (list int) [ 0; 4 ] (take 2 vals);
            equal (list int) [ 10; 14 ]
              (take 2 (List.filteri (fun i _ -> i >= 12) vals));
            K.validate expanded);

          test "contract middle axis of 3-axis UNROLL" (fun () ->
            let consts = List.init 8 (fun i -> i32 i) in
            let expanded =
              expand_vec_contract ~consts
                ~unroll_axes:[ (1, 2); (2, 2); (3, 2) ]
                ~contract_axes:[ (2, 2) ] ~vec_width:2
            in
            equal (list int) [ 0; 2; 1; 3; 4; 6; 5; 7 ]
              (sink_int_values expanded);
            K.validate expanded);

          test "different-axis add with flipped operands" (fun () ->
            let consts_a = List.init 4 (fun i -> i32 (i * 4)) in
            let consts_b = List.init 4 (fun i -> i32 i) in
            let unroll_a =
              K.unroll ~src:(K.vectorize ~srcs:consts_a)
                ~axes:[ (1, 4) ] ~dtype:Dtype.int32
            in
            let unroll_b =
              K.unroll ~src:(K.vectorize ~srcs:consts_b)
                ~axes:[ (2, 4) ] ~dtype:Dtype.int32
            in
            let add = K.binary ~op:`Add ~lhs:unroll_b ~rhs:unroll_a in
            let root = K.sink ~kernel_info:(kernel_info ()) [ add ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            equal int 1
              (count_reachable expanded (function
                | K.Binary _ -> true
                | _ -> false));
            equal int 2
              (count_reachable expanded (function
                | K.Vectorize _ -> true
                | _ -> false));
            K.validate expanded);

          test "contract simple exact GEP indices" (fun () ->
            let consts = List.init 4 (fun i -> i32 i) in
            let expanded =
              expand_vec_contract ~consts ~unroll_axes:[ (1, 4) ]
                ~contract_axes:[ (1, 4) ] ~vec_width:4
            in
            equal (list int) [ 0; 1; 2; 3 ] (sink_int_values expanded);
            K.validate expanded);
        ];

      group "edge cases"
        [
          test "empty UNROLL is a no-op" (fun () ->
            let unroll =
              K.unroll ~src:(i32 42) ~axes:[] ~dtype:Dtype.int32
            in
            let root = K.sink ~kernel_info:(kernel_info ()) [ unroll ] in
            let expanded = Expander.expand root in
            let _ = find_sink expanded in
            equal int 0
              (count_reachable expanded (function
                | K.Unroll _ -> true
                | _ -> false));
            K.validate expanded);

          test "push broadcast through AFTER" (fun () ->
            let c5 = i32 5 in
            let r0 =
              K.range ~size:(idx_const 4) ~axis:0 ~kind:Axis_kind.Global ()
            in
            let end_node = K.end_ ~value:c5 ~ranges:[ r0 ] in
            let bcast = K.vectorize ~srcs:[ c5; c5; c5; c5 ] in
            let after = K.after ~src:bcast ~deps:[ end_node ] in
            let root = K.sink ~kernel_info:(kernel_info ()) [ after ] in
            let expanded = Expander.expand root in
            let _ = find_sink expanded in
            is_true
              (count_reachable expanded (function
                 | K.After { src; _ } ->
                     (K.dtype_or Dtype.void src).count = 1
                 | _ -> false)
              > 0);
            K.validate expanded);

          test "push broadcast through END" (fun () ->
            let c5 = i32 5 in
            let r0 =
              K.range ~size:(idx_const 4) ~axis:0 ~kind:Axis_kind.Global ()
            in
            let bcast = K.vectorize ~srcs:[ c5; c5; c5; c5 ] in
            let end_node = K.end_ ~value:bcast ~ranges:[ r0 ] in
            let root = K.sink ~kernel_info:(kernel_info ()) [ end_node ] in
            let expanded = Expander.expand root in
            let _ = find_sink expanded in
            is_true
              (count_reachable expanded (function
                 | K.End { value; _ } ->
                     (K.dtype_or Dtype.void value).count = 1
                 | _ -> false)
              > 0);
            K.validate expanded);

          test "double UNROLL axis order" (fun () ->
            let consts = List.init 8 (fun i -> i32 i) in
            let vec = K.vectorize ~srcs:consts in
            let inner =
              K.unroll ~src:vec ~axes:[ (1, 4) ]
                ~dtype:(Dtype.vec Dtype.int32 2)
            in
            let outer =
              K.unroll ~src:inner ~axes:[ (2, 2) ] ~dtype:Dtype.int32
            in
            let contract =
              K.contract ~src:outer ~axes:[ (1, 4); (2, 2) ]
                ~dtype:(Dtype.vec Dtype.int32 8)
            in
            let root = K.sink ~kernel_info:(kernel_info ()) [ contract ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            K.validate expanded);
        ];

      group "pre-expand"
        [
          test "converts Upcast range to Unroll" (fun () ->
            let r0 =
              K.range ~size:(idx_const 4) ~axis:0 ~kind:Axis_kind.Upcast ()
            in
            let add = K.binary ~op:`Add ~lhs:r0 ~rhs:r0 in
            let root = K.sink ~kernel_info:(kernel_info ()) [ add ] in
            let expanded = Expander.expand root in
            let _ = find_sink expanded in
            equal int 0
              (count_reachable expanded (function
                 | K.Range { kind = Axis_kind.Upcast; _ } -> true
                 | _ -> false));
            equal int 0
              (count_reachable expanded (function
                | K.Unroll _ -> true
                | _ -> false));
            K.validate expanded);

          test "converts Unroll range to Unroll marker" (fun () ->
            let r0 =
              K.range ~size:(idx_const 3) ~axis:0 ~kind:Axis_kind.Unroll ()
            in
            let add = K.binary ~op:`Add ~lhs:r0 ~rhs:r0 in
            let root = K.sink ~kernel_info:(kernel_info ()) [ add ] in
            let expanded = Expander.expand root in
            let _ = find_sink expanded in
            equal int 0
              (count_reachable expanded (function
                 | K.Range { kind = Axis_kind.Unroll; _ } -> true
                 | _ -> false));
            equal int 0
              (count_reachable expanded (function
                | K.Unroll _ -> true
                | _ -> false));
            K.validate expanded);

          test "ignores Reduce range" (fun () ->
            let c8 = idx_const 8 in
            let r0 = K.range ~size:c8 ~axis:0 ~kind:Axis_kind.Reduce () in
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let idx = K.index ~ptr:p0 ~idxs:[ r0 ] () in
            let ld = K.load ~src:idx () in
            let red = K.reduce ~op:`Add ~src:ld ~ranges:[ r0 ] ~dtype:dt in
            let root = K.sink ~kernel_info:(kernel_info ()) [ red ] in
            let expanded = Expander.expand root in
            let _ = find_sink expanded in
            equal int 1
              (count_reachable expanded (function
                 | K.Range { kind = Axis_kind.Reduce; _ } -> true
                 | _ -> false));
            equal int 0
              (count_reachable expanded (function
                | K.Unroll _ -> true
                | _ -> false));
            K.validate expanded);

          test "fix_reduce_unroll wraps source in CONTRACT" (fun () ->
            let consts = List.init 4 (fun i -> i32 i) in
            let vec = K.vectorize ~srcs:consts in
            let unroll =
              K.unroll ~src:vec ~axes:[ (0, 4) ] ~dtype:Dtype.int32
            in
            let cf = K.const (Const.float Dtype.float32 1.0) in
            let r1 =
              K.range ~size:(idx_const 8) ~axis:1 ~kind:Axis_kind.Reduce ()
            in
            let red =
              K.reduce ~op:`Add ~src:cf ~ranges:[ unroll; r1 ] ~dtype:dt
            in
            let root = K.sink ~kernel_info:(kernel_info ()) [ red ] in
            let expanded = Expander.expand root in
            let _ = find_sink expanded in
            is_true
              (count_reachable expanded (function
                 | K.Reduce { ranges; _ } ->
                     List.for_all
                       (fun r ->
                         match K.view r with K.Range _ -> true | _ -> false)
                       ranges
                 | _ -> false)
              > 0);
            K.validate expanded);

          test "fix_store_unroll wraps store in CONTRACT" (fun () ->
            let consts = List.init 4 (fun i -> i32 i) in
            let vec = K.vectorize ~srcs:consts in
            let unroll =
              K.unroll ~src:vec ~axes:[ (0, 4) ] ~dtype:Dtype.int32
            in
            let i32_ptr = global_ptr Dtype.int32 in
            let p0 = K.param ~idx:0 ~dtype:i32_ptr in
            let idx = K.index ~ptr:p0 ~idxs:[ idx_const 0 ] () in
            let store = K.store ~dst:idx ~value:(i32 7) ~ranges:[ unroll ] in
            let root = K.sink ~kernel_info:(kernel_info ()) [ store ] in
            let expanded = Expander.expand root in
            let _ = find_sink expanded in
            equal int 0
              (count_reachable expanded (function
                | K.Unroll _ -> true
                | _ -> false));
            K.validate expanded);
        ];

      group "group for reduce"
        [
          test "basic group-reduce transform" (fun () ->
            let expanded = Expander.expand (grouped_reduce_kernel ()) in
            let _ = find_sink expanded in
            equal int 1
              (count_reachable expanded (function
                | K.Bufferize _ -> true
                | _ -> false));
            equal int 2
              (count_reachable expanded (function
                | K.Reduce _ -> true
                | _ -> false));
            K.validate expanded);

          test "no-op without Group_reduce ranges" (fun () ->
            let r0 =
              K.range ~size:(idx_const 8) ~axis:0 ~kind:Axis_kind.Reduce ()
            in
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let idx = K.index ~ptr:p0 ~idxs:[ r0 ] () in
            let ld = K.load ~src:idx () in
            let red = K.reduce ~op:`Add ~src:ld ~ranges:[ r0 ] ~dtype:dt in
            let root = K.sink ~kernel_info:(kernel_info ()) [ red ] in
            let expanded = Expander.expand root in
            let _ = find_sink expanded in
            equal int 0
              (count_reachable expanded (function
                | K.Bufferize _ -> true
                | _ -> false));
            equal int 1
              (count_reachable expanded (function
                | K.Reduce _ -> true
                | _ -> false));
            K.validate expanded);

          test "new reduce loop axis is original + 100" (fun () ->
            let expanded = Expander.expand (grouped_reduce_kernel ()) in
            let _ = find_sink expanded in
            equal int 1
              (count_reachable expanded (function
                 | K.Range { axis = 101; kind = Axis_kind.Reduce; _ } -> true
                 | _ -> false));
            K.validate expanded);

          test "upstream locals in buffer ranges" (fun () ->
            let expanded = Expander.expand (grouped_reduce_kernel ()) in
            let _ = find_sink expanded in
            let buf_node =
              List.find
                (fun n ->
                  match K.view n with K.Bufferize _ -> true | _ -> false)
                (K.toposort expanded)
            in
            (match K.view buf_node with
             | K.Bufferize { ranges; _ } ->
                 is_true
                   (List.exists
                      (fun r -> match K.view r with
                         | K.Range { kind = Axis_kind.Local; _ } -> true
                         | _ -> false)
                      ranges);
                 is_true
                   (List.exists
                      (fun r -> match K.view r with
                         | K.Range { kind = Axis_kind.Group_reduce; _ } -> true
                         | _ -> false)
                      ranges)
             | _ -> failwith (pp_kernel expanded));
            K.validate expanded);
        ];

      group "full pipeline"
        [
          test "expand rewrites grouped reduce through bufferize plus index"
            (fun () ->
            let expanded = Expander.expand (grouped_reduce_kernel ()) in
            let _ = find_sink expanded in
            equal int 1
              (count_reachable expanded (function
                | K.Bufferize _ -> true
                | _ -> false));
            equal int 2
              (count_reachable expanded (function
                | K.Reduce _ -> true
                | _ -> false));
            let buf_node =
              List.find
                (fun n ->
                  match K.view n with K.Bufferize _ -> true | _ -> false)
                (K.toposort expanded)
            in
            begin
              match K.view buf_node with
              | K.Bufferize { ranges; _ } ->
                  equal int 2 (List.length ranges);
                  is_true
                    (List.exists
                       (fun r ->
                         match K.view r with
                         | K.Range { kind = Axis_kind.Local; _ } -> true
                         | _ -> false)
                       ranges);
                  is_true
                    (List.exists
                       (fun r ->
                         match K.view r with
                         | K.Range { kind = Axis_kind.Group_reduce; _ } ->
                             true
                         | _ -> false)
                       ranges)
              | _ -> failwith (pp_kernel expanded)
            end;
            equal int 1
              (count_reachable expanded (function
                | K.Index { ptr; _ } -> (
                    match K.view ptr with
                    | K.Bufferize _ -> true
                    | _ -> false)
                | _ -> false));
            equal int 1
              (count_reachable expanded (function
                | K.Reduce { ranges = [ r ]; _ } -> (
                    match K.view r with
                    | K.Range { kind = Axis_kind.Reduce; _ } -> true
                    | _ -> false)
                | _ -> false));
            K.validate expanded);

          test "expand consumes upcast ranges in reduce" (fun () ->
            let p0 = K.param ~idx:0 ~dtype:ptr in
            let c4 = idx_const 4 in
            let c8 = idx_const 8 in
            let r0 = K.range ~size:c4 ~axis:0 ~kind:Axis_kind.Upcast () in
            let r1 = K.range ~size:c8 ~axis:1 ~kind:Axis_kind.Reduce () in
            let idx = K.index ~ptr:p0 ~idxs:[ r0; r1 ] () in
            let ld = K.load ~src:idx () in
            let red =
              K.reduce ~op:`Add ~src:ld ~ranges:[ r0; r1 ] ~dtype:dt
            in
            let root = K.sink ~kernel_info:(kernel_info ()) [ red ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            is_true
              (count_reachable expanded (function
                 | K.Reduce { ranges = [ _ ]; _ } -> true
                 | _ -> false)
              = 1);
            try K.validate expanded
            with exn ->
              failwith
                (Printexc.to_string exn ^ "\n" ^ pp_kernel expanded));

          test "expand consumes upcast ranges in store" (fun () ->
            let i32_ptr = global_ptr Dtype.int32 in
            let p0 = K.param ~idx:0 ~dtype:i32_ptr in
            let c4 = idx_const 4 in
            let r0 = K.range ~size:c4 ~axis:0 ~kind:Axis_kind.Upcast () in
            let idx = K.index ~ptr:p0 ~idxs:[ r0 ] () in
            let store = K.store ~dst:idx ~value:(i32 7) ~ranges:[ r0 ] in
            let root = K.sink ~kernel_info:(kernel_info ()) [ store ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            equal int 1
              (count_reachable expanded (function
                | K.Store _ -> true
                | _ -> false));
            K.validate expanded);

          test "expand consumes upcast ranges in end" (fun () ->
            let i32_ptr = global_ptr Dtype.int32 in
            let p0 = K.param ~idx:0 ~dtype:i32_ptr in
            let c4 = idx_const 4 in
            let r0 = K.range ~size:c4 ~axis:0 ~kind:Axis_kind.Upcast () in
            let idx = K.index ~ptr:p0 ~idxs:[ r0 ] () in
            let store = K.store ~dst:idx ~value:(i32 7) ~ranges:[] in
            let end_node = K.end_ ~value:store ~ranges:[ r0 ] in
            let root = K.sink ~kernel_info:(kernel_info ()) [ end_node ] in
            let expanded = Expander.expand root in
            assert_no_contract_unroll expanded;
            equal int 1
              (count_reachable expanded (function
                | K.Store _ -> true
                | _ -> false));
            equal int 1
              (count_reachable expanded (function
                | K.End _ -> true
                | _ -> false));
            K.validate expanded);
        ];
    ]
