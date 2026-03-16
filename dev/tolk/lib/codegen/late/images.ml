(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir.Program
module Program = Tolk_ir.Program

type image_index = {
  ptr : int;
  idxs : int list;
  gate : int option;
  dtype : Tolk_ir.Dtype.ptr;
}

let int2_dtype = Tolk_ir.Dtype.vec Tolk_ir.Dtype.int32 2
let float4_dtype = Tolk_ir.Dtype.vec Tolk_ir.Dtype.float32 4

let supports_images (renderer : Renderer.t) =
  match Renderer.device renderer with "CL" | "QCOM" -> true | _ -> false

let fail_unsupported (renderer : Renderer.t) =
  failwith
    (Printf.sprintf
       "Images.rewrite: renderer %s does not support OpenCL image parameters"
       (Renderer.name renderer))

let image_param_dtype_of (program : Tolk_ir.Program.t) ref_ =
  match Program.view program ref_ with
  | Param_image { dtype; _ } -> Some dtype
  | _ -> None

let image_index_of (program : Tolk_ir.Program.t) ref_ =
  match Program.view program ref_ with
  | Index { ptr; idxs; gate; dtype } -> (
      match image_param_dtype_of program ptr with
      | Some _ -> Some { ptr; idxs; gate; dtype }
      | None -> None)
  | _ -> None

let validate_image_param_dtype (ptr_dtype : Tolk_ir.Dtype.ptr) =
  match ptr_dtype.base.scalar with
  | Tolk_ir.Dtype.Float16 | Tolk_ir.Dtype.Float32 -> ()
  | scalar ->
      failwith
        (Printf.sprintf "Images.rewrite: unsupported image base dtype %s"
           (Tolk_ir.Dtype.scalar_to_string scalar))

let validate_image_index info =
  validate_image_param_dtype info.dtype;
  if List.length info.idxs <> 2 then
    failwith "Images.rewrite: image access requires exactly two coordinates"

let has_param_images (program : Tolk_ir.Program.t) =
  let found = ref false in
  Program.iteri
    (fun _id v -> match v with Param_image _ -> found := true | _ -> ())
    program;
  !found

let rewrite (renderer : Renderer.t) (program : Tolk_ir.Program.t) :
    Tolk_ir.Program.t =
  if has_param_images program && not (supports_images renderer) then
    fail_unsupported renderer;
  Program.rebuild
    (fun ~emit ~map_ref instr ->
      match instr with
      | Param_image _ ->
          None (* keep as-is; renderer handles image typing *)
      | Index { ptr; idxs; dtype; _ } -> (
          match image_param_dtype_of program ptr with
          | None -> None
          | Some _ ->
              let info = { ptr; idxs; gate = None; dtype } in
              validate_image_index info;
              Some
                (emit
                   (Custom_inline
                      {
                        fmt = "(int2)({0}, {1})";
                        args = List.map map_ref idxs;
                        dtype = int2_dtype;
                      })))
      | Load { src; alt; dtype } -> (
          match image_index_of program src with
          | None -> None
          | Some info ->
              validate_image_index info;
              if not (Tolk_ir.Dtype.equal dtype float4_dtype) then
                failwith
                  "Images.rewrite: image loads must produce float4 values";
              let ptr = map_ref info.ptr in
              let idx = map_ref src in
              let rewritten =
                match (info.gate, alt) with
                | None, None ->
                    Custom_inline
                      {
                        fmt = "read_imagef({0}, smp, {1})";
                        args = [ ptr; idx ];
                        dtype;
                      }
                | Some gate, Some alt_ref ->
                    Custom_inline
                      {
                        fmt = "({2}?read_imagef({0}, smp, {1}):{3})";
                        args = [ ptr; idx; map_ref gate; map_ref alt_ref ];
                        dtype;
                      }
                | Some _, None ->
                    failwith
                      "Images.rewrite: gated image loads require an alt value"
                | None, Some _ ->
                    failwith
                      "Images.rewrite: image load alt requires a gated index"
              in
              Some (emit rewritten))
      | Store { dst; value } -> (
          match image_index_of program dst with
          | None -> None
          | Some info ->
              validate_image_index info;
              if
                not
                  (Tolk_ir.Dtype.equal
                     (Option.value ~default:Tolk_ir.Dtype.void
                        (Program.dtype program value))
                     float4_dtype)
              then
                failwith "Images.rewrite: image stores must write float4 values";
              let args = [ map_ref info.ptr; map_ref dst; map_ref value ] in
              Some
                (emit
                   (match info.gate with
                   | None ->
                       Custom { fmt = "write_imagef({0}, {1}, {2});"; args }
                   | Some gate ->
                       Custom
                         {
                           fmt = "if ({3}) write_imagef({0}, {1}, {2});";
                           args = args @ [ map_ref gate ];
                         })))
      | _ -> None)
    program
