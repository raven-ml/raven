(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* coco_eval GT DT KIND OUT: evaluates the results file DT against the
   instances file GT, for KIND bbox or segm, and writes the precision and
   recall tables to the npz file OUT. ../gen/coco.py --check compares them
   with pycocotools' on the whole of val2017. *)

let read jsont file =
  match
    Jsont_bytesrw.decode_string jsont
      (In_channel.with_open_bin file In_channel.input_all)
  with
  | Ok v -> v
  | Error e -> failwith e

let () =
  match Sys.argv with
  | [| _; gt; dt; kind; out |] ->
      let d = read Mynd_coco.dataset_jsont gt in
      let e =
        match kind with
        | "bbox" ->
            Mynd_coco.evaluate_boxes d (read Mynd_coco.box_detections_jsont dt)
        | "segm" ->
            Mynd_coco.evaluate_masks d (read Mynd_coco.mask_detections_jsont dt)
        | k -> failwith ("unknown kind " ^ k)
      in
      Nx_io.save_npz out
        (Nx_io.Archive.of_list
           [ ("precision", Nx.P e.precision); ("recall", Nx.P e.recall) ])
  | _ ->
      prerr_endline "usage: coco_eval GT DT KIND OUT";
      exit 2
