(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* TESSERA's 2B teacher, an 8.26 GB checkpoint whose entries past 4 GiB torch
   writes with ZIP64 data descriptors, loads whole: its state dict holds the
   2,064,266,242 fp32 parameters its model card states
   (https://huggingface.co/geotessera/TESSERA-V-2.0-2B-Teacher, Architecture),
   at the shapes of its modules, in 199 tensors, the keys its pickle names (a
   baseline). The dune file says how to run it. *)

open Windtrap
module Pt = Nx_io.Pytorch

let field k = function
  | Pt.Dict kv -> List.assoc (Pt.String k) kv
  | _ -> failwith "no dict"

(* Width 4096 and FFN 16384 in each backbone, 1024 per head, the fusion's two
   modalities and FFN 8192, and the reducer from 8192 to 16384 to 1024. *)
let shapes =
  let layer = "s2_backbone.transformer_encoder.layers.0." in
  [
    (layer ^ "q_proj.weight", [| 4096; 4096 |]);
    (layer ^ "q_norm.weight", [| 1024 |]);
    (layer ^ "linear1.weight", [| 16384; 4096 |]);
    ( "s1_backbone.transformer_encoder.layers.3.linear2.weight",
      [| 4096; 16384 |] );
    ("fusion_modality_embed", [| 2; 4096 |]);
    ("fusion_transformer.layers.1.linear1.weight", [| 8192; 4096 |]);
    ("dim_reducer.0.weight", [| 16384; 8192 |]);
    ("dim_reducer.4.weight", [| 1024; 16384 |]);
  ]

let () =
  let path =
    match Sys.getenv_opt "NX_IO_PT_TEACHER" with
    | Some p -> p
    | None -> failwith "NX_IO_PT_TEACHER is not set: the dune file says how"
  in
  let weights =
    Pt.tensors (field "encoder_state_dict" (Nx_io.load_pytorch path))
  in
  let all =
    List.map
      (fun n -> (n, Option.get (Nx_io.Archive.find n weights)))
      (Nx_io.Archive.names weights)
  in
  exit
    (run "nx.io PyTorch teacher"
       [
         test
           "its state dict holds 199 float32 tensors of 2,064,266,242 \
            parameters" (fun () ->
             equal ~msg:"tensors" int 199 (List.length all);
             List.iter
               (fun (n, Nx.P x) ->
                 equal ~msg:n string "float32" (Nx.Dtype.name (Nx.dtype x)))
               all;
             equal ~msg:"parameters" int 2_064_266_242
               (List.fold_left (fun s (_, Nx.P x) -> s + Nx.numel x) 0 all));
         cases "its tensors have their modules' shapes" ~id:fst shapes
           (fun (n, shape) ->
             let (Nx.P x) = List.assoc n all in
             equal (array int) shape (Nx.shape x));
       ])
