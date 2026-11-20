(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Datasets *)

type box = { x : float; y : float; w : float; h : float }
type segmentation = Polygons of float array list | Runs of Mynd.Rle.t
type image = { id : int; file_name : string; height : int; width : int }
type category = { id : int; name : string }

type annotation = {
  id : int;
  image : int;
  category : int;
  box : box;
  area : float;
  crowd : bool;
  segmentation : segmentation;
}

type dataset = {
  images : image array;
  categories : category array;
  annotations : annotation array;
}

(* Codecs *)

(* A decoder that raises [Invalid_argument] fails the decode with its
   message. *)
let guard f x =
  try f x with Invalid_argument why -> Jsont.Error.msg Jsont.Meta.none why

let box_jsont =
  let field b = function 0 -> b.x | 1 -> b.y | 2 -> b.w | _ -> b.h in
  Jsont.t4 ~kind:"bbox"
    ~dec:(fun x y w h -> { x; y; w; h })
    ~enc:field Jsont.number

let counts_jsont =
  let list =
    Jsont.map
      ~dec:(fun c -> `List c)
      ~enc:(function `List c -> c | `String _ -> invalid_arg "counts")
      (Jsont.array Jsont.int)
  in
  let string =
    Jsont.map
      ~dec:(fun s -> `String s)
      ~enc:(function `String s -> s | `List _ -> invalid_arg "counts")
      Jsont.string
  in
  Jsont.any ~kind:"counts" ~dec_array:list ~dec_string:string
    ~enc:(function `List _ -> list | `String _ -> string)
    ()

let runs_jsont =
  let dec size counts =
    let height = size.(0) and width = size.(1) in
    match counts with
    | `String s -> Mynd.Rle.of_string ~height ~width s
    | `List c -> Mynd.Rle.v ~height ~width c
  in
  Jsont.Object.map ~kind:"RLE" (fun size counts -> guard (dec size) counts)
  |> Jsont.Object.mem "size" (Jsont.tn ~n:2 Jsont.int) ~enc:(fun r ->
      [| r.Mynd.Rle.height; r.width |])
  |> Jsont.Object.mem "counts" counts_jsont ~enc:(fun r ->
      `String (Mynd.Rle.to_string r))
  |> Jsont.Object.finish

let segmentation_jsont =
  let polygons =
    Jsont.map
      ~dec:(fun p -> Polygons p)
      ~enc:(function Polygons p -> p | Runs _ -> invalid_arg "polygons")
      (Jsont.list (Jsont.array Jsont.number))
  in
  let runs =
    Jsont.map
      ~dec:(fun r -> Runs r)
      ~enc:(function Runs r -> r | Polygons _ -> invalid_arg "runs")
      runs_jsont
  in
  Jsont.any ~kind:"segmentation" ~dec_array:polygons ~dec_object:runs
    ~enc:(function Polygons _ -> polygons | Runs _ -> runs)
    ()

let crowd_jsont =
  Jsont.map ~kind:"iscrowd"
    ~dec:(function
      | 0 -> false
      | 1 -> true
      | n -> Jsont.Error.msgf Jsont.Meta.none "iscrowd is %d, not 0 or 1" n)
    ~enc:(fun c -> if c then 1 else 0)
    Jsont.int

let image_jsont =
  Jsont.Object.map ~kind:"image" (fun id file_name height width ->
      { id; file_name; height; width })
  |> Jsont.Object.mem "id" Jsont.int ~enc:(fun (i : image) -> i.id)
  |> Jsont.Object.mem "file_name" Jsont.string ~enc:(fun i -> i.file_name)
  |> Jsont.Object.mem "height" Jsont.int ~enc:(fun i -> i.height)
  |> Jsont.Object.mem "width" Jsont.int ~enc:(fun i -> i.width)
  |> Jsont.Object.finish

let category_jsont =
  Jsont.Object.map ~kind:"category" (fun id name : category -> { id; name })
  |> Jsont.Object.mem "id" Jsont.int ~enc:(fun (c : category) -> c.id)
  |> Jsont.Object.mem "name" Jsont.string ~enc:(fun (c : category) -> c.name)
  |> Jsont.Object.finish

let annotation_jsont =
  Jsont.Object.map ~kind:"annotation"
    (fun id image category box area crowd segmentation ->
      { id; image; category; box; area; crowd; segmentation })
  |> Jsont.Object.mem "id" Jsont.int ~enc:(fun (a : annotation) -> a.id)
  |> Jsont.Object.mem "image_id" Jsont.int ~enc:(fun (a : annotation) ->
      a.image)
  |> Jsont.Object.mem "category_id" Jsont.int ~enc:(fun (a : annotation) ->
      a.category)
  |> Jsont.Object.mem "bbox" box_jsont ~enc:(fun a -> a.box)
  |> Jsont.Object.mem "area" Jsont.number ~enc:(fun a -> a.area)
  |> Jsont.Object.mem "iscrowd" crowd_jsont ~enc:(fun a -> a.crowd)
  |> Jsont.Object.mem "segmentation" segmentation_jsont
       ~dec_absent:(Polygons []) ~enc:(fun a -> a.segmentation)
  |> Jsont.Object.finish

let dataset_jsont =
  Jsont.Object.map ~kind:"dataset" (fun images categories annotations ->
      { images; categories; annotations })
  |> Jsont.Object.mem "images" (Jsont.array image_jsont) ~enc:(fun d ->
      d.images)
  |> Jsont.Object.mem "categories" (Jsont.array category_jsont) ~enc:(fun d ->
      d.categories)
  |> Jsont.Object.mem "annotations" (Jsont.array annotation_jsont)
       ~dec_absent:[||] ~enc:(fun d -> d.annotations)
  |> Jsont.Object.finish

(* Masks *)

let runs ~height ~width s =
  let fn = "Mynd_coco.runs" in
  if height < 0 || width < 0 then
    invalid_argf "%s: an image of %d by %d pixels" fn height width;
  match s with
  | Runs r ->
      if r.height <> height || r.width <> width then
        invalid_argf "%s: runs of %d by %d pixels in an image of %d by %d" fn
          r.height r.width height width;
      r
  | Polygons ps ->
      let counts =
        List.fold_left
          (fun acc p ->
            let c = Raster.polygon fn ~height ~width p in
            match acc with None -> Some c | Some a -> Some (Raster.union a c))
          None ps
      in
      let counts = Option.value counts ~default:[| height * width |] in
      Mynd.Rle.v ~height ~width counts

(* Detections *)

type 'r detection = { image : int; category : int; score : float; region : 'r }

let detection_jsont ~kind key region =
  Jsont.Object.map ~kind (fun image category region score ->
      { image; category; score; region })
  |> Jsont.Object.mem "image_id" Jsont.int ~enc:(fun (d : _ detection) ->
      d.image)
  |> Jsont.Object.mem "category_id" Jsont.int ~enc:(fun (d : _ detection) ->
      d.category)
  |> Jsont.Object.mem key region ~enc:(fun d -> d.region)
  |> Jsont.Object.mem "score" Jsont.number ~enc:(fun d -> d.score)
  |> Jsont.Object.finish

let box_detections_jsont =
  Jsont.array (detection_jsont ~kind:"box detection" "bbox" box_jsont)

let mask_detections_jsont =
  Jsont.array (detection_jsont ~kind:"mask detection" "segmentation" runs_jsont)
