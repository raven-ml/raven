(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Goldens and small datasets shared by mynd.coco's tests. Every golden is
   written by contrib/mynd/test/gen/coco.py. *)

open Windtrap
open Mynd_coco

(* Goldens *)

let archives = Hashtbl.create 4

let archive file =
  match Hashtbl.find_opt archives file with
  | Some a -> a
  | None ->
      let a = Nx_io.load_npz ("golden/" ^ file) in
      Hashtbl.add archives file a;
      a

let entry file dt key =
  match Nx_io.Archive.find key (archive file) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/%s has no %s" file key

let floats file key = Nx.to_array (entry file Nx.float64 key)

let ints file key =
  Array.map Int64.to_int (Nx.to_array (entry file Nx.int64 key))

let decode jsont s =
  match Jsont_bytesrw.decode_string jsont s with
  | Ok v -> v
  | Error e -> failf "decode: %s" e

let read jsont file =
  decode jsont
    (In_channel.with_open_bin ("golden/" ^ file) In_channel.input_all)

let encode jsont v =
  match Jsont_bytesrw.encode_string jsont v with
  | Ok s -> s
  | Error e -> failf "encode: %s" e

(* The first 24 val2017 images with a CC BY or CC BY-SA licence and their
   annotations, and detections drawn near them. *)
let fixture = lazy (read dataset_jsont "val24.json")
let box_dets = lazy (read box_detections_jsont "dets24.json")
let mask_dets = lazy (read mask_detections_jsont "dets24.json")

(* [split lengths flat] cuts [flat] into consecutive pieces of [lengths]. *)
let split lengths flat =
  let at = ref 0 in
  Array.map
    (fun n ->
      let a = Array.sub flat !at n in
      at := !at + n;
      a)
    lengths

(* Small datasets *)

let image ?(height = 100) ?(width = 100) id : image =
  { id; file_name = Printf.sprintf "%d.jpg" id; height; width }

let category id : category = { id; name = string_of_int id }
let box x y w h = { x; y; w; h }

let ann ?(crowd = false) ?area ?(segmentation = Polygons []) ?(id = 1) ~image
    ~category b =
  let area = Option.value area ~default:(b.w *. b.h) in
  { id; image; category; box = b; area; crowd; segmentation }

let det ~image ~category score region = { image; category; score; region }

let dataset ?(images = [ image 1 ]) ?(categories = [ category 1 ]) anns =
  {
    images = Array.of_list images;
    categories = Array.of_list categories;
    annotations = Array.of_list anns;
  }
