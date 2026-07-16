(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop

module U = Uop

let count pred root = List.length (U.find_nodes pred root)

let custom_fmt n =
  match U.op n, U.Arg.as_string (U.arg n) with
  | (Ops.Custom | Ops.Customi), Some fmt -> Some fmt
  | _ -> None

let coord ?(cast = false) x y =
  let maybe_cast u = if cast then U.cast ~src:u ~dtype:Dtype.weakint else u in
  let x = maybe_cast x and y = maybe_cast y in
  U.stack ~dtype:(U.dtype x) [ x; y ]

(* An image is a float parameter whose shape is [(height, width, 4)]. *)
let image_param ?(slot = 0) shape =
  U.param ~slot ~dtype:Dtype.float32
    ~shape:(U.stack (List.map U.const_int shape)) ()

let buffer_param ?(slot = 1) dtype =
  U.param ~slot ~dtype ~shape:(U.const_int 1) ()

let scalar_zero () = U.const (Const.float Dtype.float32 0.0)

let lowered_float_buffer ?(slot = 0) size =
  U.param ~slot ~dtype:Dtype.float32 ~shape:(U.const_int size) ()

let lowered_load buf offset =
  U.load
    ~src:(U.index ~ptr:buf ~idxs:[ U.const_int offset ] ())
    ~dtype:Dtype.float32 ()

let lowered_store buf offset value =
  U.store
    ~dst:(U.index ~ptr:buf ~idxs:[ U.const_int offset ] ())
    ~value:(U.const (Const.float Dtype.float32 value)) ()

let run_matcher pm root =
  U.graph_rewrite ~name:"test"
    (fun n -> Upat.Pattern_matcher.rewrite pm n)
    root

let with_env name value f =
  let old = try Some (Sys.getenv name) with Not_found -> None in
  Unix.putenv name value;
  Fun.protect f ~finally:(fun () ->
      match old with
      | Some old -> Unix.putenv name old
      | None -> Unix.putenv name "")

let test_renderer ?pre_matcher ?extra_matcher () =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:false
    ~has_shared:false ~shared_max:0 ?pre_matcher ?extra_matcher
    ~render:(fun ?name:_ _ -> "") ()

let qcom_renderer () =
  Renderer.make ~name:"qcom" ~device:"QCOM" ~has_local:false
    ~has_shared:false ~shared_max:0 ~image_pitch_alignment:64
    ~render:(fun ?name:_ _ -> "") ()

let () =
  run "Coalese image selection"
    [
      group "image valid dimensions"
        [
          test "missing pitch alignment yields no candidates" (fun () ->
            equal (list (pair int int)) []
              (Coalese.image_valid_dims ~image_pitch_alignment:None
                 ~base:Dtype.float32 ~size:1024 ()));
          test "rejects non-float image bases" (fun () ->
            equal (list (pair int int)) []
              (Coalese.image_valid_dims ~image_pitch_alignment:(Some 64)
                 ~base:Dtype.int32 ~size:1024 ()));
          test "aligned sizes enumerate height-width candidates" (fun () ->
            equal (list (pair int int)) [ (4, 64); (2, 128); (1, 256) ]
              (Coalese.image_valid_dims ~image_pitch_alignment:(Some 64)
                 ~base:Dtype.float32 ~size:1024 ()));
          test "one-row fallback uses byte alignment" (fun () ->
            equal (list (pair int int)) [ (1, 4) ]
              (Coalese.image_valid_dims ~image_pitch_alignment:(Some 64)
                 ~base:Dtype.float32 ~size:16 ());
            equal (list (pair int int)) [ (1, 10) ]
              (Coalese.image_valid_dims ~image_pitch_alignment:(Some 20)
                 ~base:Dtype.float32 ~size:40 ());
            equal (list (pair int int)) []
              (Coalese.image_valid_dims ~image_pitch_alignment:(Some 64)
                 ~base:Dtype.float32 ~size:40 ());
            equal (list (pair int int)) []
              (Coalese.image_valid_dims ~image_pitch_alignment:(Some 64)
                 ~base:Dtype.float16 ~size:40 ());
            equal (list (pair int int)) [ (1, 8) ]
              (Coalese.image_valid_dims ~osx:true
                 ~image_pitch_alignment:(Some 256)
                 ~base:Dtype.float16 ~size:32 ()));
        ];
      group "coalese image selection"
        [
          test "shrink of float param becomes image index" (fun () ->
            let param =
              U.param ~slot:0 ~dtype:Dtype.float32
                ~shape:(U.const_int 16) ()
            in
            let shrink =
              U.shrink ~src:param ~offset:(U.const_int 0)
                ~size:(U.const_int 4)
            in
            let root =
              with_env "IMAGE" "1" (fun () ->
                  U.graph_rewrite ~name:"add images" ~bottom_up:true
                    (fun n ->
                      Upat.Pattern_matcher.rewrite
                        (Coalese.pm_simplify_add_image (qcom_renderer ()))
                        n)
                    shrink)
            in
            match U.as_index root with
            | Some { ptr; _ } ->
                equal (list int) [ 1; 4; 4 ] (U.max_shape ptr)
            | None -> failwith "expected image index");
          test "opencl target pitch enables image index" (fun () ->
            let param =
              U.param ~slot:0 ~dtype:Dtype.float32
                ~shape:(U.const_int 16) ()
            in
            let shrink =
              U.shrink ~src:param ~offset:(U.const_int 0)
                ~size:(U.const_int 4)
            in
            let root =
              with_env "IMAGE" "1" (fun () ->
                  U.graph_rewrite ~name:"add images" ~bottom_up:true
                    (fun n ->
                      Upat.Pattern_matcher.rewrite
                        (Coalese.pm_simplify_add_image
                           (Cstyle.opencl "IMAGE_PITCH_ALIGNMENT=64"))
                        n)
                    shrink)
            in
            match U.as_index root with
            | Some { ptr; _ } ->
                equal (list int) [ 1; 4; 4 ] (U.max_shape ptr)
            | None -> failwith "expected image index");
        ];
      group "memory coalesing"
        [
          test "adjacent float loads become coalesced load with lane indexes"
            (fun () ->
              let buf = lowered_float_buffer 16 in
              let loads = List.init 4 (lowered_load buf) in
              let root = Coalese.memory_coalesing (test_renderer ()) (U.sink loads) in
              let nodes = U.toposort root in
              let coalesced_loads =
                List.filter
                  (fun n ->
                    match U.as_load n with
                    | Some { src; _ } ->
                        U.op src = Ops.Shrink
                        && Dtype.equal (U.dtype n) Dtype.float32
                    | None -> false)
                  nodes
              in
              let lane_indexes =
                List.filter
                  (fun n ->
                    match U.as_index n with
                    | Some { ptr; _ } -> U.op ptr = Ops.Load
                    | None -> false)
                  nodes
              in
              equal int 1 (List.length coalesced_loads);
              equal int 4 (List.length lane_indexes));
          test "adjacent float stores become vector store with vector value"
            (fun () ->
              let buf = lowered_float_buffer 16 in
              let stores =
                List.init 4 (fun i -> lowered_store buf i (Float.of_int i))
              in
              let root =
                Coalese.memory_coalesing (test_renderer ()) (U.sink stores)
              in
              let nodes = U.toposort root in
              let vector_stores =
                List.filter_map
                  (fun n ->
                    match U.as_store n with
                    | Some { value; _ }
                      when U.op value = Ops.Stack
                           && Array.length (U.src value) = 4 ->
                        Some n
                    | _ -> None)
                  nodes
              in
              equal int 1 (List.length vector_stores));
          test "duplicate stores to one offset are rejected" (fun () ->
            let buf = lowered_float_buffer 16 in
            let stores =
              [
                lowered_store buf 0 1.0;
                lowered_store buf 0 2.0;
              ]
            in
            raises_match
              (function
                | Failure msg ->
                    String.equal msg
                      "Coalese: multiple stores to the same offset"
                | _ -> false)
              (fun () ->
                ignore
                  (Coalese.memory_coalesing (test_renderer ())
                     (U.sink stores))));
          test "gated memory ops are rejected" (fun () ->
            let buf = lowered_float_buffer 16 in
            let idx =
              U.index ~ptr:buf ~idxs:[ U.const_int 0 ] ()
            in
            let store =
              U.store ~dst:idx
                ~value:(U.const (Const.float Dtype.float32 1.0))
                ~gate:(U.const_bool true) ()
            in
            raises_match
              (function
                | Failure msg ->
                    String.equal msg
                      "memory coalesing does not support gated loads/stores"
                | _ -> false)
              (fun () ->
                ignore
                  (Coalese.memory_coalesing (test_renderer ())
                     (U.sink [ store ]))));
          test "non-index memory ops are rejected" (fun () ->
            let buf = lowered_float_buffer 16 in
            let load = U.load ~src:buf ~dtype:Dtype.float32 () in
            raises_match
              (function
                | Failure msg ->
                    String.equal msg
                      "memory coalesing should be on INDEX, not PARAM"
                | _ -> false)
              (fun () ->
                ignore
                  (Coalese.memory_coalesing (test_renderer ())
                     (U.sink [ load ]))));
        ];
      group "late cleanup"
        [
          test "remove invalid keeps index sentinels" (fun () ->
            let root =
              U.sink [ U.invalid (); U.invalid ~dtype:Dtype.float32 () ]
            in
            let root =
              U.graph_rewrite ~name:"remove invalids"
                (fun n ->
                  Upat.Pattern_matcher.rewrite Symbolic.pm_remove_invalid n)
                root
            in
            equal int 1
              (count
                 (fun n ->
                   match U.op n, U.arg n with
                   | Ops.Const, U.Arg.Value c ->
                       Const.view c = Const.Invalid
                       && Dtype.equal (U.dtype n) Dtype.index
                   | _ -> false)
                 root);
            equal int 1
              (count
                 (fun n ->
                   match U.op n, U.arg n with
                   | Ops.Const, U.Arg.Value c ->
                       Const.view c = Const.Float 0.0
                       && Dtype.equal (U.dtype n) Dtype.float32
                   | _ -> false)
                 root));
          test "load-store indexing strips coordinate casts" (fun () ->
            let img = image_param [ 4; 4; 4 ] in
            let x = U.const (Const.int Dtype.int32 0) in
            let y = U.const (Const.int Dtype.int32 1) in
            let idx = coord ~cast:true x y in
            let node = U.index ~ptr:img ~idxs:[ idx ] () in
            let root =
              run_matcher
                Upat.Pattern_matcher.(
                  Symbolic.pm_lower_index_dtype ++ Coalese.indexing_simplify)
                node
            in
            let idx =
              match U.as_index root with
              | Some { idxs = [ idx ]; _ } -> idx
              | Some _ -> failwith "expected scalar image coordinate"
              | None -> failwith "expected index"
            in
            equal int 0 (count (fun n -> U.op n = Ops.Cast) idx));
          test "lower index dtype concretizes weak binary math" (fun () ->
            let x =
              U.cast
                ~src:(U.const (Const.int Dtype.int32 1))
                ~dtype:Dtype.weakint
            in
            let y =
              U.cast
                ~src:(U.const (Const.int Dtype.int32 2))
                ~dtype:Dtype.weakint
            in
            let root = run_matcher Symbolic.pm_lower_index_dtype U.O.(x + y) in
            match U.op root, U.src root with
            | Ops.Cast, [| add |] ->
                equal bool true (Dtype.equal (U.dtype root) Dtype.weakint);
                equal bool true (Dtype.equal (U.dtype add) Dtype.int32)
            | _ -> failwith "expected weakint cast around concrete add");
          test "move where on value index keeps loads late" (fun () ->
            let buf = buffer_param Dtype.float32 in
            let axis =
              U.range ~axis:0 ~kind:Axis_type.Loop ~size:(U.const_int 2) ()
            in
            let gate = U.O.(axis < U.const_int 1) in
            let zero = scalar_zero () in
            let value =
              U.index ~ptr:buf ~idxs:[(U.const_int 0)] ()
            in
            let root =
              run_matcher Symbolic.pm_move_where_on_load
                (U.O.where gate value zero)
            in
            match U.op root, U.src root with
            | Ops.Where, [| keep; index; alt |] -> (
                if not (U.equal keep gate) then failwith "expected outer gate";
                if
                  not
                    (match U.op alt, U.arg alt with
                  | Ops.Const, U.Arg.Value c -> (
                      match Const.view c with
                      | Const.Int 0L | Const.Float 0.0 | Const.Bool false ->
                          true
                      | Const.Int _ | Const.Float _ | Const.Bool _
                      | Const.Invalid ->
                          false)
                  | _ -> false)
                then failwith "expected zero outer alt";
                match U.as_index index with
                | Some { idxs = [ idx ]; _ } -> (
                    match U.const_int_value idx with
                    | Some 0 -> ()
                    | Some _ | None -> failwith "expected unchanged index")
                | _ -> failwith "expected index")
            | _ -> failwith "expected where around valid index");
          test "lower pipeline applies renderer pre and extra matchers" (fun () ->
            let pre_matcher node =
              match custom_fmt node with
              | Some "pre({0})" ->
                  Some
                    (U.custom_inline ~fmt:"pre_done({0})"
                       ~args:(Array.to_list (U.src node))
                       ~dtype:(U.dtype node))
              | _ -> None
            in
            let extra_matcher node =
              match custom_fmt node with
              | Some "extra({0})" ->
                  Some
                    (U.custom_inline ~fmt:"extra_done({0})"
                       ~args:(Array.to_list (U.src node))
                       ~dtype:(U.dtype node))
              | _ -> None
            in
            let arg = U.const_float 1.0 in
            let root =
              U.sink
                [ U.custom_inline ~fmt:"pre({0})" ~args:[ arg ]
                    ~dtype:Dtype.float32;
                  U.custom_inline ~fmt:"extra({0})" ~args:[ arg ]
                    ~dtype:Dtype.float32 ]
            in
            let lowered =
              Codegen_lower.lower
                (test_renderer ~pre_matcher ~extra_matcher ())
                root
            in
            equal int 1
              (count
                 (fun n ->
                   match custom_fmt n with
                   | Some "pre_done({0})" -> true
                   | _ -> false)
                 lowered);
            equal int 1
              (count
                 (fun n ->
                   match custom_fmt n with
                   | Some "extra_done({0})" -> true
                   | _ -> false)
                 lowered));
        ];
    ]
