(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Lower Param_image nodes into OpenCL image intrinsics (read_imagef /
   write_imagef).  Runs on Kernel IR before linearization.

   This replaces tinygrad's ImageDType handling which threads image types
   through the entire pipeline.  Tolk keeps images as Param_image nodes
   and lowers them here in a single late pass. *)

open Tolk_ir
module K = Kernel

let strf = Printf.sprintf

let float4 = Dtype.Val.vec 4 Dtype.Val.float32
let int2 = Dtype.Val.vec 2 Dtype.Val.int32

let is_image node = match K.view node with
  | Param_image _ -> true | _ -> false

(* Decompose an image Index into (image_param, coord_node, gate).
   The coord_node is a custom_inline "(int2)({0}, {1})" built from the
   two index operands.  Validates dtype and index count. *)
let decompose_image_index node = match K.view node with
  | Index { ptr; idxs; gate; _ } when is_image ptr ->
      (match K.dtype_opt ptr with
       | Some dt ->
           (match Dtype.scalar dt with
            | Float16 | Float32 -> ()
            | s -> failwith (strf
                "images: unsupported base dtype %s" (Dtype.scalar_to_string s)));
           if List.length idxs <> 2 then
             failwith "images: image access requires exactly two coordinates"
       | None -> ());
      let coords = K.custom_inline ~fmt:"(int2)({0}, {1})" ~args:idxs ~dtype:int2 in
      Some (ptr, coords, gate)
  | _ -> None

(* Load/Store on image params become read_imagef/write_imagef intrinsics.
   Index nodes that feed non-image consumers are left unchanged. *)
let rewrite_rule (node : K.t) : K.t option =
  match K.view node with
  | Load { src; alt; dtype } -> (
      match decompose_image_index src with
      | None -> None
      | Some (ptr, coords, gate) ->
          if not (Dtype.Val.equal dtype float4) then
            failwith "images: image loads must produce float4";
          Some (match gate, alt with
            | None, None ->
                K.custom_inline ~fmt:"read_imagef({0}, smp, {1})"
                  ~args:[ptr; coords] ~dtype
            | Some g, Some a ->
                K.custom_inline
                  ~fmt:"({2}?read_imagef({0}, smp, {1}):{3})"
                  ~args:[ptr; coords; g; a] ~dtype
            | Some _, None ->
                failwith "images: gated image load requires alt value"
            | None, Some _ ->
                failwith "images: image load alt requires gated index"))
  | Store { dst; value; _ } -> (
      match decompose_image_index dst with
      | None -> None
      | Some (ptr, coords, gate) ->
          if not (Dtype.equal (K.dtype value) (Dtype.Val float4)) then
            failwith "images: image stores must write float4";
          Some (match gate with
            | None ->
                K.custom ~fmt:"write_imagef({0}, {1}, {2});"
                  ~args:[ptr; coords; value]
            | Some g ->
                K.custom ~fmt:"if ({3}) write_imagef({0}, {1}, {2});"
                  ~args:[ptr; coords; value; g]))
  | _ -> None

let supports_images renderer =
  match Renderer.device renderer with "CL" | "QCOM" -> true | _ -> false

let rewrite renderer root =
  let has_images = List.exists (fun n -> match K.view n with
    | Param_image _ -> true | _ -> false) (K.toposort root) in
  if has_images && not (supports_images renderer) then
    failwith (strf "images: renderer %s does not support images"
      (Renderer.name renderer));
  K.graph_rewrite rewrite_rule root
