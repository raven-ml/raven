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

(* The tables' entries. [k] is a category's position by increasing id, [a] a
   range (all, small, medium, large) and [m] a limit (1, 10, 100). *)
let precision (e : evaluation) ~t ~r ~k ~a ~m =
  Nx.item [ t; r; k; a; m ] e.precision

let recall (e : evaluation) ~t ~k ~a ~m = Nx.item [ t; k; a; m ] e.recall

(* Bits of tables, NaN where pycocotools writes -1. *)
let of_coco a = Array.map (fun v -> if v = -1. then Float.nan else v) a
let bits (e : evaluation) = (Nx.to_array e.precision, Nx.to_array e.recall)

(* A drawn dataset: up to four images of 40 by 40 pixels and three
   categories, annotations of integer boxes, a tenth of them crowds, and
   detections near them with distinct scores, or scores on a grid of tenths
   when [ties]. *)
let gen_case ~ties =
  Gen.(
    let* ni, nk = pair (int_range 1 4) (int_range 1 3) in
    let gbox =
      let+ x = int_range 0 30
      and+ y = int_range 0 30
      and+ w = int_range 1 10
      and+ h = int_range 1 10 in
      box (Float.of_int x) (Float.of_int y) (Float.of_int w) (Float.of_int h)
    in
    let gann =
      let+ i = int_range 1 ni
      and+ k = int_range 1 nk
      and+ b = gbox
      and+ c = int_range 0 9 in
      ann ~crowd:(c = 0) ~image:i ~category:k b
    in
    let* anns = list ~size:(int_range 0 12) gann in
    let jitter (b : box) =
      let+ dx = int_range (-2) 2 and+ dy = int_range (-2) 2 in
      box (b.x +. Float.of_int dx) (b.y +. Float.of_int dy) b.w b.h
    in
    let free =
      let+ i = int_range 1 ni and+ k = int_range 1 nk and+ b = gbox in
      (i, k, b)
    in
    let gdet =
      match anns with
      | [] -> free
      | _ ->
          let near =
            let* (a : annotation) = of_list anns in
            let+ b = jitter a.box in
            (a.image, a.category, b)
          in
          frequency [ (3, near); (1, free) ]
    in
    let* dets = list ~size:(int_range 0 14) gdet in
    let n = List.length dets in
    let+ scores =
      if ties then
        list ~size:(constant n)
          (map (fun s -> Float.of_int s /. 10.) (int_range 1 9))
      else permutation (List.init n (fun s -> Float.of_int (s + 1) /. 64.))
    in
    let dets =
      List.map2 (fun (i, k, b) s -> det ~image:i ~category:k s b) dets scores
    in
    let anns = List.mapi (fun j a -> { a with id = j + 1 }) anns in
    ( dataset
        ~images:(List.init ni (fun i -> image ~height:40 ~width:40 (i + 1)))
        ~categories:(List.init nk (fun k -> category (k + 1)))
        anns,
      Array.of_list dets ))
