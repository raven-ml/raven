(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_ir
module K = Kernel

(* Constants *)

let float4 = Dtype.Val.vec 4 Dtype.Val.float32
let int2 = Dtype.Val.vec 2 Dtype.Val.int32
let cl = Cstyle.opencl

(* Helpers *)

let global dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)
let img_ptr = global float4
let buf_ptr = global float4

let contains s sub =
  let sl = String.length s and subl = String.length sub in
  subl <= sl &&
  let rec loop i =
    i <= sl - subl && (String.sub s i subl = sub || loop (i + 1))
  in
  loop 0

let rewrite k = Images.rewrite cl k

let render r k =
  Renderer.render r (Linearizer.linearize (Images.rewrite r k))

let find_unique msg pred root =
  match K.find_nodes pred root with
  | [n] -> n
  | ns ->
      failwith
        (Printf.sprintf "%s: expected 1, got %d" msg (List.length ns))

let count pred root = List.length (K.find_nodes pred root)

let failure_contains needle fn =
  raises_match
    (function Failure msg -> contains msg needle | _ -> false)
    fn

let assert_rendered msg s sub =
  if not (contains s sub) then
    failwith (Printf.sprintf "%s: expected %S in output" msg sub)

let assert_dtype msg expected actual =
  if not (Dtype.Val.equal expected actual) then
    failwith
      (Printf.sprintf "%s: expected %s, got %s" msg
         (Format.asprintf "%a" Dtype.Val.pp expected)
         (Format.asprintf "%a" Dtype.Val.pp actual))

(* Kernel builders *)

let float4_zero () =
  let z = K.const_float 0.0 in
  K.vectorize ~srcs:[ z; z; z; z ]

let mk_ungated_load () =
  let img = K.param_image ~idx:0 ~dtype:img_ptr ~width:4 ~height:4 in
  let buf = K.param ~idx:1 ~dtype:buf_ptr in
  let c0 = K.const_int 0 and c1 = K.const_int 1 in
  let src = K.index ~ptr:img ~idxs:[ c0; c1 ] () in
  let dst = K.index ~ptr:buf ~idxs:[ c0 ] () in
  K.sink [ K.store ~dst ~value:(K.load ~src ()) ~ranges:[] ]

let mk_gated_load () =
  let img = K.param_image ~idx:0 ~dtype:img_ptr ~width:4 ~height:4 in
  let buf = K.param ~idx:1 ~dtype:buf_ptr in
  let c0 = K.const_int 0 and c1 = K.const_int 1 in
  let gate = K.const_bool true in
  let src = K.index ~ptr:img ~idxs:[ c0; c1 ] ~gate () in
  let dst = K.index ~ptr:buf ~idxs:[ c0 ] () in
  K.sink [ K.store ~dst ~value:(K.load ~src ~alt:(float4_zero ()) ()) ~ranges:[] ]

let mk_ungated_store () =
  let img = K.param_image ~idx:0 ~dtype:img_ptr ~width:4 ~height:4 in
  let buf = K.param ~idx:1 ~dtype:buf_ptr in
  let c0 = K.const_int 0 and c1 = K.const_int 1 in
  let src = K.index ~ptr:buf ~idxs:[ c0 ] () in
  let dst = K.index ~ptr:img ~idxs:[ c0; c1 ] () in
  K.sink [ K.store ~dst ~value:(K.load ~src ()) ~ranges:[] ]

let mk_gated_store () =
  let img = K.param_image ~idx:0 ~dtype:img_ptr ~width:4 ~height:4 in
  let buf = K.param ~idx:1 ~dtype:buf_ptr in
  let c0 = K.const_int 0 and c1 = K.const_int 1 in
  let gate = K.const_bool true in
  let src = K.index ~ptr:buf ~idxs:[ c0 ] () in
  let dst = K.index ~ptr:img ~idxs:[ c0; c1 ] ~gate () in
  K.sink [ K.store ~dst ~value:(K.load ~src ()) ~ranges:[] ]

let mk_mixed () =
  let f32_ptr = global Dtype.Val.float32 in
  let img = K.param_image ~idx:0 ~dtype:img_ptr ~width:4 ~height:4 in
  let p1 = K.param ~idx:1 ~dtype:f32_ptr in
  let p2 = K.param ~idx:2 ~dtype:buf_ptr in
  let c0 = K.const_int 0 and c1 = K.const_int 1 in
  (* buffer load/store *)
  let idx_buf = K.index ~ptr:p1 ~idxs:[ c0 ] () in
  let st_buf =
    K.store ~dst:idx_buf ~value:(K.load ~src:idx_buf ()) ~ranges:[]
  in
  (* image load → buffer store *)
  let idx_img = K.index ~ptr:img ~idxs:[ c0; c1 ] () in
  let idx_out = K.index ~ptr:p2 ~idxs:[ c0 ] () in
  let st_out =
    K.store ~dst:idx_out ~value:(K.load ~src:idx_img ()) ~ranges:[]
  in
  (* buffer load → image store *)
  let idx_img2 = K.index ~ptr:img ~idxs:[ c1; c0 ] () in
  let idx_in = K.index ~ptr:p2 ~idxs:[ c1 ] () in
  let st_img =
    K.store ~dst:idx_img2 ~value:(K.load ~src:idx_in ()) ~ranges:[]
  in
  K.sink [ st_buf; st_out; st_img ]

let mk_no_images () =
  let f32_ptr = global Dtype.Val.float32 in
  let p0 = K.param ~idx:0 ~dtype:f32_ptr in
  let p1 = K.param ~idx:1 ~dtype:f32_ptr in
  let c0 = K.const_int 0 in
  let src = K.index ~ptr:p0 ~idxs:[ c0 ] () in
  let dst = K.index ~ptr:p1 ~idxs:[ c0 ] () in
  K.sink [ K.store ~dst ~value:(K.load ~src ()) ~ranges:[] ]

(* Error-case builders *)

let mk_bad_dtype () =
  let bad = global Dtype.Val.int32 in
  let img = K.param_image ~idx:0 ~dtype:bad ~width:4 ~height:4 in
  let buf = K.param ~idx:1 ~dtype:buf_ptr in
  let c0 = K.const_int 0 and c1 = K.const_int 1 in
  let src = K.index ~ptr:img ~idxs:[ c0; c1 ] () in
  let dst = K.index ~ptr:buf ~idxs:[ c0 ] () in
  K.sink [ K.store ~dst ~value:(K.load ~src ()) ~ranges:[] ]

let mk_wrong_idx_count n =
  let img = K.param_image ~idx:0 ~dtype:img_ptr ~width:4 ~height:4 in
  let buf = K.param ~idx:1 ~dtype:buf_ptr in
  let idxs = List.init n (fun i -> K.const_int i) in
  let src = K.index ~ptr:img ~idxs () in
  let c0 = K.const_int 0 in
  let dst = K.index ~ptr:buf ~idxs:[ c0 ] () in
  K.sink [ K.store ~dst ~value:(K.load ~src ()) ~ranges:[] ]

let mk_wrong_load_dtype () =
  let f32_ptr = global Dtype.Val.float32 in
  let img = K.param_image ~idx:0 ~dtype:f32_ptr ~width:4 ~height:4 in
  let buf = K.param ~idx:1 ~dtype:f32_ptr in
  let c0 = K.const_int 0 and c1 = K.const_int 1 in
  let src = K.index ~ptr:img ~idxs:[ c0; c1 ] () in
  let dst = K.index ~ptr:buf ~idxs:[ c0 ] () in
  K.sink [ K.store ~dst ~value:(K.load ~src ()) ~ranges:[] ]

let mk_wrong_store_dtype () =
  let f32_ptr = global Dtype.Val.float32 in
  let img = K.param_image ~idx:0 ~dtype:f32_ptr ~width:4 ~height:4 in
  let buf = K.param ~idx:1 ~dtype:f32_ptr in
  let c0 = K.const_int 0 and c1 = K.const_int 1 in
  let src = K.index ~ptr:buf ~idxs:[ c0 ] () in
  let dst = K.index ~ptr:img ~idxs:[ c0; c1 ] () in
  K.sink [ K.store ~dst ~value:(K.load ~src ()) ~ranges:[] ]

let mk_gated_no_alt () =
  let img = K.param_image ~idx:0 ~dtype:img_ptr ~width:4 ~height:4 in
  let buf = K.param ~idx:1 ~dtype:buf_ptr in
  let c0 = K.const_int 0 and c1 = K.const_int 1 in
  let gate = K.const_bool true in
  let src = K.index ~ptr:img ~idxs:[ c0; c1 ] ~gate () in
  let dst = K.index ~ptr:buf ~idxs:[ c0 ] () in
  K.sink [ K.store ~dst ~value:(K.load ~src ()) ~ranges:[] ]

let mk_alt_no_gate () =
  let img = K.param_image ~idx:0 ~dtype:img_ptr ~width:4 ~height:4 in
  let buf = K.param ~idx:1 ~dtype:buf_ptr in
  let c0 = K.const_int 0 and c1 = K.const_int 1 in
  let src = K.index ~ptr:img ~idxs:[ c0; c1 ] () in
  let dst = K.index ~ptr:buf ~idxs:[ c0 ] () in
  K.sink
    [ K.store ~dst ~value:(K.load ~src ~alt:(float4_zero ()) ()) ~ranges:[] ]

(* Runner *)

let () =
  run "Images"
    [
      group "Index rewriting"
        [
          test "ungated image index becomes int2" (fun () ->
            let root = rewrite (mk_ungated_load ()) in
            let n =
              find_unique "int2"
                (fun n ->
                  match K.view n with
                  | Custom_inline { fmt; _ } -> contains fmt "(int2)"
                  | _ -> false)
                root
            in
            match K.view n with
            | Custom_inline { fmt; args; dtype } ->
                equal string "(int2)({0}, {1})" fmt;
                equal int 2 (List.length args);
                assert_dtype "int2 index" int2 dtype
            | _ -> assert false);
          test "non-image index unchanged" (fun () ->
            equal int 3
              (count
                 (fun n ->
                   match K.view n with Index _ -> true | _ -> false)
                 (rewrite (mk_mixed ()))));
        ];
      group "Load rewriting"
        [
          test "ungated image load becomes read_imagef" (fun () ->
            let root = rewrite (mk_ungated_load ()) in
            let n =
              find_unique "read_imagef"
                (fun n ->
                  match K.view n with
                  | Custom_inline { fmt; _ } -> contains fmt "read_imagef"
                  | _ -> false)
                root
            in
            match K.view n with
            | Custom_inline { fmt; args; dtype } ->
                equal string "read_imagef({0}, smp, {1})" fmt;
                equal int 2 (List.length args);
                assert_dtype "read_imagef" float4 dtype
            | _ -> assert false);
          test "gated image load becomes conditional read_imagef" (fun () ->
            let root = rewrite (mk_gated_load ()) in
            let n =
              find_unique "gated read_imagef"
                (fun n ->
                  match K.view n with
                  | Custom_inline { fmt; _ } -> contains fmt "read_imagef"
                  | _ -> false)
                root
            in
            match K.view n with
            | Custom_inline { fmt; args; dtype } ->
                equal string "({2}?read_imagef({0}, smp, {1}):{3})" fmt;
                equal int 4 (List.length args);
                assert_dtype "gated read_imagef" float4 dtype
            | _ -> assert false);
          test "non-image load unchanged" (fun () ->
            equal int 2
              (count
                 (fun n ->
                   match K.view n with Load _ -> true | _ -> false)
                 (rewrite (mk_mixed ()))));
        ];
      group "Store rewriting"
        [
          test "ungated image store becomes write_imagef" (fun () ->
            let root = rewrite (mk_ungated_store ()) in
            let n =
              find_unique "write_imagef"
                (fun n ->
                  match K.view n with
                  | Custom { fmt; _ } -> contains fmt "write_imagef"
                  | _ -> false)
                root
            in
            match K.view n with
            | Custom { fmt; args } ->
                equal string "write_imagef({0}, {1}, {2});" fmt;
                equal int 3 (List.length args)
            | _ -> assert false);
          test "gated image store becomes conditional write_imagef" (fun () ->
            let root = rewrite (mk_gated_store ()) in
            let n =
              find_unique "gated write_imagef"
                (fun n ->
                  match K.view n with
                  | Custom { fmt; _ } -> contains fmt "write_imagef"
                  | _ -> false)
                root
            in
            match K.view n with
            | Custom { fmt; args } ->
                equal string "if ({3}) write_imagef({0}, {1}, {2});" fmt;
                equal int 4 (List.length args)
            | _ -> assert false);
          test "non-image store unchanged" (fun () ->
            equal int 2
              (count
                 (fun n ->
                   match K.view n with Store _ -> true | _ -> false)
                 (rewrite (mk_mixed ()))));
        ];
      group "Passthrough"
        [
          test "Param_image preserved" (fun () ->
            let root = rewrite (mk_ungated_load ()) in
            let n =
              find_unique "Param_image"
                (fun n ->
                  match K.view n with
                  | Param_image _ -> true
                  | _ -> false)
                root
            in
            match K.view n with
            | Param_image { idx; width; height; _ } ->
                equal int 0 idx;
                equal int 4 width;
                equal int 4 height
            | _ -> assert false);
          test "no-image kernel is identity" (fun () ->
            let k = mk_no_images () in
            equal int
              (List.length (K.toposort k))
              (List.length (K.toposort (rewrite k))));
        ];
      group "Device support"
        [
          test "CL accepts images" (fun () ->
            ignore (Images.rewrite Cstyle.opencl (mk_ungated_load ())));
          test "QCOM accepts images" (fun () ->
            ignore (Images.rewrite Cstyle.qcom (mk_ungated_load ())));
          test "Metal rejects images" (fun () ->
            failure_contains "does not support" (fun () ->
              ignore
                (Images.rewrite Cstyle.metal (mk_ungated_load ()))));
          test "CUDA rejects images" (fun () ->
            failure_contains "does not support" (fun () ->
              ignore
                (Images.rewrite
                   (Cstyle.cuda Gpu_target.SM80)
                   (mk_ungated_load ()))));
          test "no images on unsupported renderer passes" (fun () ->
            ignore (Images.rewrite Cstyle.metal (mk_no_images ())));
        ];
      group "Validation"
        [
          test "rejects unsupported base dtype" (fun () ->
            failure_contains "unsupported base dtype" (fun () ->
              ignore (rewrite (mk_bad_dtype ()))));
          test "rejects 1D image access" (fun () ->
            failure_contains "exactly two coordinates" (fun () ->
              ignore (rewrite (mk_wrong_idx_count 1))));
          test "rejects 3D image access" (fun () ->
            failure_contains "exactly two coordinates" (fun () ->
              ignore (rewrite (mk_wrong_idx_count 3))));
          test "rejects non-float4 load" (fun () ->
            failure_contains "must produce float4" (fun () ->
              ignore (rewrite (mk_wrong_load_dtype ()))));
          test "rejects non-float4 store" (fun () ->
            failure_contains "must write float4" (fun () ->
              ignore (rewrite (mk_wrong_store_dtype ()))));
          test "rejects gated load without alt" (fun () ->
            failure_contains "requires alt value" (fun () ->
              ignore (rewrite (mk_gated_no_alt ()))));
          test "rejects alt without gate" (fun () ->
            failure_contains "requires gated index" (fun () ->
              ignore (rewrite (mk_alt_no_gate ()))));
        ];
      group "Rendered output"
        [
          test "gated load renders correctly" (fun () ->
            let out = render cl (mk_gated_load ()) in
            assert_rendered "gated read_imagef" out "?read_imagef(";
            assert_rendered "gated alt colon" out ":");
          test "gated store renders correctly" (fun () ->
            let out = render cl (mk_gated_store ()) in
            assert_rendered "if-guarded write" out "if (";
            assert_rendered "write_imagef" out "write_imagef(");
        ];
    ]
