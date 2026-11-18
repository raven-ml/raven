(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* masks.exe DIR OUT: raven's draws of the steps test_parity checks on the
   fixture DIR's tile, for tiny.py to feed TESSERA's code. OUT/distil holds the
   dropout masks of the distillation step (Tiny.step), NAME.npy for each, 1
   where a value is kept, masks.json their names in the order the forward draws
   them, and pixels.npy the tile's index of each place's pixel. OUT/barlow holds
   the pretraining group's (Tiny.views): its views' tokens, a1_s2.npy to
   a2_s1.npy, alpha.npy and perm.npy, and the masks of its three forwards,
   vV_NAME.npy, V 0 for the first views, 1 the second and 2 the mixup, with
   masks.json. *)

open Tessera

let save dir name x = Nx_io.save_npy (Filename.concat dir (name ^ ".npy")) x

let save_masks dir prefix masks =
  List.iter (fun (n, x) -> save dir (prefix ^ n) (Nx.cast Nx.uint8 x)) masks;
  Out_channel.with_open_bin (Filename.concat dir "masks.json") (fun oc ->
      Printf.fprintf oc "[%s]\n"
        (String.concat ", "
           (List.map (fun (n, _) -> Printf.sprintf "%S" n) masks)))

let () =
  let dir = Sys.argv.(1) and out = Sys.argv.(2) in
  let file f = Filename.concat dir f in
  List.iter
    (fun d -> if not (Sys.file_exists d) then Sys.mkdir d 0o755)
    [ out; Filename.concat out "distil"; Filename.concat out "barlow" ];
  let distil = Filename.concat out "distil" in
  let m = Model.load (file "student.pt") in
  let pixels, b, keys = Tiny.step dir in
  save_masks distil "" (Model.masks ~dropout:Model.v2_dropout ~keys m b);
  save distil "pixels" pixels;
  let barlow = Filename.concat out "barlow" in
  let m = Model.load (file "teacher.pt") in
  let v = Tiny.views dir in
  let keys i = Nx.Rng.of_tensor (Nx.get [ i ] (Nx.Rng.to_tensor v.keys)) in
  List.iteri
    (fun i b ->
      let masks = Model.masks ~dropout:Model.v2_dropout ~keys:(keys i) m b in
      save_masks barlow (Printf.sprintf "v%d_" i) masks)
    [ v.a1; v.a2; v.a1 ];
  List.iter
    (fun (name, (b : Nx.host Batch.t)) ->
      save barlow (name ^ "_s2") b.s2.tokens;
      save barlow (name ^ "_s1") b.s1.tokens)
    [ ("a1", v.a1); ("a2", v.a2) ];
  save barlow "alpha" v.alpha;
  save barlow "perm" v.perm
