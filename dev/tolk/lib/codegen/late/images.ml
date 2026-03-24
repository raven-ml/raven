(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir.Program
module Program = Tolk_ir.Program

let strf = Printf.sprintf

let err_unsupported ren =
  strf "Images.rewrite: renderer %s does not support OpenCL image parameters"
    (Renderer.name ren)

let err_bad_dtype s =
  strf "Images.rewrite: unsupported image base dtype %s"
    (Tolk_ir.Dtype.scalar_to_string s)

let err_need_2_coords = "Images.rewrite: image access requires exactly two coordinates"
let err_load_not_f4 = "Images.rewrite: image loads must produce float4 values"
let err_store_not_f4 = "Images.rewrite: image stores must write float4 values"
let err_gate_no_alt = "Images.rewrite: gated image loads require an alt value"
let err_alt_no_gate = "Images.rewrite: image load alt requires a gated index"

let int2_dtype = Tolk_ir.Dtype.vec Tolk_ir.Dtype.int32 2
let float4_dtype = Tolk_ir.Dtype.vec Tolk_ir.Dtype.float32 4

let supports_images (renderer : Renderer.t) =
  match Renderer.device renderer with "CL" | "QCOM" -> true | _ -> false

let is_image_param program ref_ =
  match Program.view program ref_ with Param_image _ -> true | _ -> false

let validate_image_access (dtype : Tolk_ir.Dtype.ptr) idxs =
  (match Tolk_ir.Dtype.scalar (Tolk_ir.Dtype.base dtype) with
  | Tolk_ir.Dtype.Float16 | Tolk_ir.Dtype.Float32 -> ()
  | s -> failwith (err_bad_dtype s));
  if List.length idxs <> 2 then failwith err_need_2_coords

let has_param_images (program : Tolk_ir.Program.t) =
  try
    Program.iteri
      (fun _id v -> match v with Param_image _ -> raise_notrace Exit | _ -> ())
      program;
    false
  with Exit -> true

let rewrite (renderer : Renderer.t) (program : Tolk_ir.Program.t) =
  if has_param_images program && not (supports_images renderer) then
    failwith (err_unsupported renderer);
  Program.rebuild
    (fun ~emit ~map_ref instr ->
      match instr with
      | Param_image _ -> None
      | Index { ptr; idxs; dtype; _ } when is_image_param program ptr ->
          validate_image_access dtype idxs;
          Some
            (emit
               (Custom_inline
                  { fmt = "(int2)({0}, {1})";
                    args = List.map map_ref idxs;
                    dtype = int2_dtype }))
      | Load { src; alt; dtype } -> (
          match Program.view program src with
          | Index { ptr; idxs; gate; dtype = idx_dtype }
            when is_image_param program ptr ->
              validate_image_access idx_dtype idxs;
              if not (Tolk_ir.Dtype.equal dtype float4_dtype) then
                failwith err_load_not_f4;
              let ptr = map_ref ptr and idx = map_ref src in
              Some
                (emit
                   (match gate, alt with
                   | None, None ->
                       Custom_inline
                         { fmt = "read_imagef({0}, smp, {1})";
                           args = [ ptr; idx ]; dtype }
                   | Some g, Some a ->
                       Custom_inline
                         { fmt = "({2}?read_imagef({0}, smp, {1}):{3})";
                           args = [ ptr; idx; map_ref g; map_ref a ]; dtype }
                   | Some _, None -> failwith err_gate_no_alt
                   | None, Some _ -> failwith err_alt_no_gate))
          | _ -> None)
      | Store { dst; value } -> (
          match Program.view program dst with
          | Index { ptr; idxs; gate; dtype } when is_image_param program ptr ->
              validate_image_access dtype idxs;
              let vdt =
                Option.value ~default:Tolk_ir.Dtype.void
                  (Program.dtype program value)
              in
              if not (Tolk_ir.Dtype.equal vdt float4_dtype) then
                failwith err_store_not_f4;
              let args = [ map_ref ptr; map_ref dst; map_ref value ] in
              Some
                (emit
                   (match gate with
                   | None ->
                       Custom { fmt = "write_imagef({0}, {1}, {2});"; args }
                   | Some g ->
                       Custom
                         { fmt = "if ({3}) write_imagef({0}, {1}, {2});";
                           args = args @ [ map_ref g ] }))
          | _ -> None)
      | _ -> None)
    program
