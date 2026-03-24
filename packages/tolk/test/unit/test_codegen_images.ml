(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_ir
module P = Program

(* Helpers *)

let float4_dt = Dtype.vec Dtype.float32 4
let int2_dt = Dtype.vec Dtype.int32 2
let cl = Cstyle.opencl

let rewrite prog = Images.rewrite cl prog
let render_with_images r prog = Renderer.render r (Images.rewrite r prog)

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

let assert_contains msg haystack needle =
  if not (contains haystack needle) then
    failwith
      (Printf.sprintf "%s: expected output to contain %S, got:\n%s" msg needle
         haystack)

let pp_view view = Format.asprintf "%a" P.pp_view view
let fail_view msg view = failwith (Printf.sprintf "%s: %s" msg (pp_view view))
let pp_program program = Format.asprintf "%a" P.pp program

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

let raises_failure needle fn =
  raises_match
    (function Failure msg -> contains msg needle | _ -> false)
    fn

let assert_dtype msg expected actual =
  if not (Dtype.equal expected actual) then
    failwith
      (Printf.sprintf "%s: expected %s, got %s" msg
         (Format.asprintf "%a" Dtype.pp expected)
         (Format.asprintf "%a" Dtype.pp actual))

(* Program builder helpers *)

let global_ptr dt = Dtype.ptr_of dt ~addrspace:Global ~size:(-1)
let f32_image_ptr = global_ptr Dtype.float32
let vec_ptr = global_ptr float4_dt

let emit_i32 b n =
  P.emit b (Const { value = Const.int Dtype.int32 n; dtype = Dtype.int32 })

(* Common preamble: image param at idx 0, buffer param at idx 1, constants 0
   and 1. Returns (builder, image_param, buf_param, c0, c1, image_ptr_dt). *)
let emit_image_and_buf ?(image_dt = Dtype.float32) ?(buf_dt = vec_ptr)
    ?(width = 4) ?(height = 4) () =
  let image_ptr = global_ptr image_dt in
  let b = P.create () in
  let p0 =
    P.emit b (Param_image { idx = 0; dtype = image_ptr; width; height })
  in
  let p1 = P.emit b (Param { idx = 1; dtype = buf_dt }) in
  let c0 = emit_i32 b 0 in
  let c1 = emit_i32 b 1 in
  (b, p0, p1, c0, c1, image_ptr)

let emit_gate b = P.emit b (Const { value = Const.bool true; dtype = Dtype.bool })

let emit_float4_alt b =
  let z0 =
    P.emit b
      (Const { value = Const.float Dtype.float32 0.0; dtype = Dtype.float32 })
  in
  P.emit b (Vectorize { srcs = [ z0; z0; z0; z0 ]; dtype = float4_dt })

let finish_store b dst value = ignore (P.emit b (Store { dst; value })); P.finish b

(* Program builders *)

let make_ungated_load ?(image_dt = Dtype.float32) ?(width = 4) ?(height = 4)
    () =
  let b, p0, p1, c0, c1, iptr =
    emit_image_and_buf ~image_dt ~width ~height ()
  in
  let idx0 =
    P.emit b (Index { ptr = p0; idxs = [ c0; c1 ]; gate = None; dtype = iptr })
  in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = float4_dt }) in
  let idx1 =
    P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = vec_ptr })
  in
  finish_store b idx1 ld

let make_gated_load () =
  let b, p0, p1, c0, c1, iptr = emit_image_and_buf () in
  let gate = emit_gate b in
  let alt = emit_float4_alt b in
  let idx0 =
    P.emit b
      (Index { ptr = p0; idxs = [ c0; c1 ]; gate = Some gate; dtype = iptr })
  in
  let ld =
    P.emit b (Load { src = idx0; alt = Some alt; dtype = float4_dt })
  in
  let idx1 =
    P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = vec_ptr })
  in
  finish_store b idx1 ld

let make_ungated_store () =
  let b, p0, p1, c0, c1, iptr = emit_image_and_buf () in
  let idx1 =
    P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = vec_ptr })
  in
  let ld = P.emit b (Load { src = idx1; alt = None; dtype = float4_dt }) in
  let idx0 =
    P.emit b (Index { ptr = p0; idxs = [ c0; c1 ]; gate = None; dtype = iptr })
  in
  finish_store b idx0 ld

let make_gated_store () =
  let b, p0, p1, c0, c1, iptr = emit_image_and_buf () in
  let gate = emit_gate b in
  let idx1 =
    P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = vec_ptr })
  in
  let ld = P.emit b (Load { src = idx1; alt = None; dtype = float4_dt }) in
  let idx0 =
    P.emit b
      (Index { ptr = p0; idxs = [ c0; c1 ]; gate = Some gate; dtype = iptr })
  in
  finish_store b idx0 ld

let make_mixed () =
  let f32_ptr = global_ptr Dtype.float32 in
  let b = P.create () in
  let p0 =
    P.emit b
      (Param_image { idx = 0; dtype = f32_image_ptr; width = 4; height = 4 })
  in
  let p1 = P.emit b (Param { idx = 1; dtype = f32_ptr }) in
  let p2 = P.emit b (Param { idx = 2; dtype = vec_ptr }) in
  let c0 = emit_i32 b 0 in
  let c1 = emit_i32 b 1 in
  let idx_buf =
    P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = f32_ptr })
  in
  let ld_buf =
    P.emit b (Load { src = idx_buf; alt = None; dtype = Dtype.float32 })
  in
  ignore (P.emit b (Store { dst = idx_buf; value = ld_buf }));
  let idx_img =
    P.emit b
      (Index
         { ptr = p0; idxs = [ c0; c1 ]; gate = None; dtype = f32_image_ptr })
  in
  let ld_img =
    P.emit b (Load { src = idx_img; alt = None; dtype = float4_dt })
  in
  let idx_out =
    P.emit b (Index { ptr = p2; idxs = [ c0 ]; gate = None; dtype = vec_ptr })
  in
  ignore (P.emit b (Store { dst = idx_out; value = ld_img }));
  let idx_img2 =
    P.emit b
      (Index
         { ptr = p0; idxs = [ c1; c0 ]; gate = None; dtype = f32_image_ptr })
  in
  let idx_in =
    P.emit b (Index { ptr = p2; idxs = [ c1 ]; gate = None; dtype = vec_ptr })
  in
  let ld_vec =
    P.emit b (Load { src = idx_in; alt = None; dtype = float4_dt })
  in
  finish_store b idx_img2 ld_vec

let make_no_images () =
  let f32_ptr = global_ptr Dtype.float32 in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = f32_ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = f32_ptr }) in
  let c0 = emit_i32 b 0 in
  let idx0 =
    P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = f32_ptr })
  in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = Dtype.float32 }) in
  let idx1 =
    P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = f32_ptr })
  in
  finish_store b idx1 ld

(* Error-case builders *)

let make_bad_dtype () =
  let image_ptr = global_ptr Dtype.int32 in
  let b = P.create () in
  let p0 =
    P.emit b (Param_image { idx = 0; dtype = image_ptr; width = 4; height = 4 })
  in
  let p1 = P.emit b (Param { idx = 1; dtype = vec_ptr }) in
  let c0 = emit_i32 b 0 in
  let c1 = emit_i32 b 1 in
  let idx0 =
    P.emit b
      (Index { ptr = p0; idxs = [ c0; c1 ]; gate = None; dtype = image_ptr })
  in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = float4_dt }) in
  let idx1 =
    P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = vec_ptr })
  in
  finish_store b idx1 ld

let make_wrong_idx_count n =
  let b, p0, p1, _, _, iptr = emit_image_and_buf () in
  let idxs = List.init n (fun i -> emit_i32 b i) in
  let idx0 =
    P.emit b (Index { ptr = p0; idxs; gate = None; dtype = iptr })
  in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = float4_dt }) in
  let c0 = emit_i32 b 0 in
  let idx1 =
    P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = vec_ptr })
  in
  finish_store b idx1 ld

let make_wrong_load_or_store_dtype ~store () =
  let f32_ptr = global_ptr Dtype.float32 in
  let b, p0, p1, c0, c1, iptr =
    emit_image_and_buf ~buf_dt:f32_ptr ()
  in
  if store then begin
    let idx1 =
      P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = f32_ptr })
    in
    let ld =
      P.emit b (Load { src = idx1; alt = None; dtype = Dtype.float32 })
    in
    let idx0 =
      P.emit b
        (Index { ptr = p0; idxs = [ c0; c1 ]; gate = None; dtype = iptr })
    in
    finish_store b idx0 ld
  end
  else begin
    let idx0 =
      P.emit b
        (Index { ptr = p0; idxs = [ c0; c1 ]; gate = None; dtype = iptr })
    in
    let ld =
      P.emit b (Load { src = idx0; alt = None; dtype = Dtype.float32 })
    in
    let idx1 =
      P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = f32_ptr })
    in
    finish_store b idx1 ld
  end

let make_gated_no_alt () =
  let b, p0, p1, c0, c1, iptr = emit_image_and_buf () in
  let gate = emit_gate b in
  let idx0 =
    P.emit b
      (Index { ptr = p0; idxs = [ c0; c1 ]; gate = Some gate; dtype = iptr })
  in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = float4_dt }) in
  let idx1 =
    P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = vec_ptr })
  in
  finish_store b idx1 ld

let make_alt_no_gate () =
  let b, p0, p1, c0, c1, iptr = emit_image_and_buf () in
  let alt = emit_float4_alt b in
  let idx0 =
    P.emit b
      (Index { ptr = p0; idxs = [ c0; c1 ]; gate = None; dtype = iptr })
  in
  let ld =
    P.emit b (Load { src = idx0; alt = Some alt; dtype = float4_dt })
  in
  let idx1 =
    P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = vec_ptr })
  in
  finish_store b idx1 ld

(* Runner *)

let () =
  run "Images"
    [
      group "Index rewriting"
        [
          test "ungated image index becomes int2" (fun () ->
            let program = rewrite (make_ungated_load ()) in
            let pos =
              find_unique_position "int2 custom_inline" program (function
                | P.Custom_inline { fmt; _ } -> contains fmt "(int2)"
                | _ -> false)
            in
            match P.view program pos with
            | Custom_inline { fmt; args; dtype } ->
                equal string "(int2)({0}, {1})" fmt;
                equal int 2 (List.length args);
                assert_dtype "int2 index" int2_dt dtype
            | v -> fail_view "expected Custom_inline" v);
          test "non-image index unchanged" (fun () ->
            let program = rewrite (make_mixed ()) in
            equal int 3
              (count program (function P.Index _ -> true | _ -> false)));
        ];
      group "Load rewriting"
        [
          test "ungated image load becomes read_imagef" (fun () ->
            let program = rewrite (make_ungated_load ()) in
            let pos =
              find_unique_position "read_imagef" program (function
                | P.Custom_inline { fmt; _ } -> contains fmt "read_imagef"
                | _ -> false)
            in
            match P.view program pos with
            | Custom_inline { fmt; args; dtype } ->
                equal string "read_imagef({0}, smp, {1})" fmt;
                equal int 2 (List.length args);
                assert_dtype "read_imagef" float4_dt dtype
            | v -> fail_view "expected Custom_inline" v);
          test "gated image load becomes conditional read_imagef" (fun () ->
            let program = rewrite (make_gated_load ()) in
            let pos =
              find_unique_position "gated read_imagef" program (function
                | P.Custom_inline { fmt; _ } -> contains fmt "read_imagef"
                | _ -> false)
            in
            match P.view program pos with
            | Custom_inline { fmt; args; dtype } ->
                equal string "({2}?read_imagef({0}, smp, {1}):{3})" fmt;
                equal int 4 (List.length args);
                assert_dtype "gated read_imagef" float4_dt dtype
            | v -> fail_view "expected Custom_inline" v);
          test "non-image load unchanged" (fun () ->
            let program = rewrite (make_mixed ()) in
            equal int 2
              (count program (function P.Load _ -> true | _ -> false)));
        ];
      group "Store rewriting"
        [
          test "ungated image store becomes write_imagef" (fun () ->
            let program = rewrite (make_ungated_store ()) in
            let pos =
              find_unique_position "write_imagef" program (function
                | P.Custom { fmt; _ } -> contains fmt "write_imagef"
                | _ -> false)
            in
            match P.view program pos with
            | Custom { fmt; args } ->
                equal string "write_imagef({0}, {1}, {2});" fmt;
                equal int 3 (List.length args)
            | v -> fail_view "expected Custom" v);
          test "gated image store becomes conditional write_imagef" (fun () ->
            let program = rewrite (make_gated_store ()) in
            let pos =
              find_unique_position "gated write_imagef" program (function
                | P.Custom { fmt; _ } -> contains fmt "write_imagef"
                | _ -> false)
            in
            match P.view program pos with
            | Custom { fmt; args } ->
                equal string "if ({3}) write_imagef({0}, {1}, {2});" fmt;
                equal int 4 (List.length args)
            | v -> fail_view "expected Custom" v);
          test "non-image store unchanged" (fun () ->
            let program = rewrite (make_mixed ()) in
            equal int 2
              (count program (function P.Store _ -> true | _ -> false)));
        ];
      group "Passthrough"
        [
          test "Param_image preserved" (fun () ->
            let program = rewrite (make_ungated_load ()) in
            let pos =
              find_unique_position "Param_image" program (function
                | P.Param_image _ -> true | _ -> false)
            in
            match P.view program pos with
            | Param_image { idx; width; height; _ } ->
                equal int 0 idx;
                equal int 4 width;
                equal int 4 height
            | v -> fail_view "expected Param_image" v);
          test "no-image program is identity" (fun () ->
            let original = make_no_images () in
            equal int (P.length original) (P.length (rewrite original)));
        ];
      group "Device support"
        [
          test "CL accepts images" (fun () ->
            ignore (Images.rewrite Cstyle.opencl (make_ungated_load ())));
          test "QCOM accepts images" (fun () ->
            ignore (Images.rewrite Cstyle.qcom (make_ungated_load ())));
          test "Metal rejects images" (fun () ->
            raises_failure "does not support" (fun () ->
              ignore (Images.rewrite Cstyle.metal (make_ungated_load ()))));
          test "CUDA rejects images" (fun () ->
            raises_failure "does not support" (fun () ->
              ignore
                (Images.rewrite
                   (Cstyle.cuda Gpu_target.SM80)
                   (make_ungated_load ()))));
          test "no images on unsupported renderer passes" (fun () ->
            ignore (Images.rewrite Cstyle.metal (make_no_images ())));
        ];
      group "Dtype acceptance"
        [
          test "Float16 image accepted" (fun () ->
            let program =
              rewrite
                (make_ungated_load ~image_dt:Dtype.float16 ~width:8 ~height:8
                   ())
            in
            ignore
              (find_unique_position "read_imagef from f16" program (function
                | P.Custom_inline { fmt; _ } -> contains fmt "read_imagef"
                | _ -> false)));
        ];
      group "Validation"
        [
          test "rejects unsupported base dtype" (fun () ->
            raises_failure "unsupported image base dtype" (fun () ->
              ignore (rewrite (make_bad_dtype ()))));
          test "rejects 1D image access" (fun () ->
            raises_failure "exactly two coordinates" (fun () ->
              ignore (rewrite (make_wrong_idx_count 1))));
          test "rejects 3D image access" (fun () ->
            raises_failure "exactly two coordinates" (fun () ->
              ignore (rewrite (make_wrong_idx_count 3))));
          test "rejects non-float4 load" (fun () ->
            raises_failure "must produce float4" (fun () ->
              ignore (rewrite (make_wrong_load_or_store_dtype ~store:false ()))));
          test "rejects non-float4 store" (fun () ->
            raises_failure "must write float4" (fun () ->
              ignore (rewrite (make_wrong_load_or_store_dtype ~store:true ()))));
          test "rejects gated load without alt" (fun () ->
            raises_failure "require an alt value" (fun () ->
              ignore (rewrite (make_gated_no_alt ()))));
          test "rejects alt without gate" (fun () ->
            raises_failure "requires a gated index" (fun () ->
              ignore (rewrite (make_alt_no_gate ()))));
        ];
      group "Rendered output"
        [
          test "gated load renders correctly" (fun () ->
            let out = render_with_images cl (make_gated_load ()) in
            assert_contains "gated read_imagef" out "?read_imagef(";
            assert_contains "gated alt colon" out ":");
          test "gated store renders correctly" (fun () ->
            let out = render_with_images cl (make_gated_store ()) in
            assert_contains "if-guarded write" out "if (";
            assert_contains "write_imagef" out "write_imagef(");
        ];
    ]
