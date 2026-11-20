# COCO evaluation

This page evaluates a model's detections with `Mynd_coco`, the library
`mynd.coco`: reading COCO's files, turning a model's boxes and masks into
detections, the precision of each category, evaluating some categories,
masks for training, and drawing polygons that are not COCO's. The rules
each call follows are in [`mynd_coco.mli`](../lib/coco/mynd_coco.mli):
the protocol and IoU in its Evaluation section, the tables and their
thresholds in its Tables section, the polygon rule in its Masks section.

## Reading COCO's files

`Mynd_coco.dataset_jsont` maps an instances file and
`Mynd_coco.box_detections_jsont` a results file. With val2017's
annotations and a model's results on disk, the primary metric is:

<!-- $MDX skip -->
```ocaml
let read jsont file =
  In_channel.with_open_bin file In_channel.input_all
  |> Jsont_bytesrw.decode_string jsont
  |> Result.get_ok

let () =
  let d = read Mynd_coco.dataset_jsont "instances_val2017.json" in
  let r = read Mynd_coco.box_detections_jsont "results.json" in
  let s = Mynd_coco.(summary (evaluate_boxes d r)) in
  Printf.printf "AP %.3f\n" s.ap
```

The programs below read a dataset of two images and two categories from
a string. Each annotation's box is a corner, a width and a height. The
first annotation's segmentation is a polygon, and the others have none:

```ocaml
let decode jsont s =
  match Jsont_bytesrw.decode_string jsont s with
  | Ok v -> v
  | Error e -> failwith e

let d =
  decode Mynd_coco.dataset_jsont
    {|{"images": [{"id": 1, "file_name": "a.jpg", "height": 40, "width": 60},
                  {"id": 2, "file_name": "b.jpg", "height": 40, "width": 60}],
       "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
       "annotations": [
         {"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 6, 20, 12],
          "area": 240, "iscrowd": 0,
          "segmentation": [[4, 6, 24, 6, 24, 18, 4, 18]]},
         {"id": 2, "image_id": 1, "category_id": 2, "bbox": [30, 10, 16, 16],
          "area": 256, "iscrowd": 0},
         {"id": 3, "image_id": 2, "category_id": 1, "bbox": [10, 10, 30, 20],
          "area": 600, "iscrowd": 0}]}|}

let () = assert (Array.length d.annotations = 3)
```

## Boxes from a model

A detector gives each image's boxes as corners `x₀ y₀ x₁ y₁` in float32,
with scores and categories. `Mynd.Box.to_xywh` turns the corners into
COCO's boxes in float32, the conversion published pipelines make before
they write a results file, and each row becomes a detection of OCaml
floats:

```ocaml
let detections image boxes scores categories =
  let xywh = Nx.to_array (Nx.cast Nx.float64 (Mynd.Box.to_xywh boxes)) in
  Array.mapi
    (fun i score ->
      let region =
        Mynd_coco.
          {
            x = xywh.(4 * i);
            y = xywh.((4 * i) + 1);
            w = xywh.((4 * i) + 2);
            h = xywh.((4 * i) + 3);
          }
      in
      { Mynd_coco.image; category = categories.(i); score; region })
    scores

let dets =
  Array.append
    (detections 1
       (Nx.create Nx.float32 [| 3; 4 |]
          [| 4.; 6.; 24.; 18.; 30.; 10.; 42.; 26.; 0.; 0.; 10.; 10. |])
       [| 0.9; 0.8; 0.3 |] [| 1; 2; 1 |])
    (detections 2
       (Nx.create Nx.float32 [| 1; 4 |] [| 10.; 10.; 40.; 30. |])
       [| 0.7 |] [| 1 |])
```

The cat's two boxes are found above a false positive, and the dog's box
covers three quarters of its annotation, an IoU of 0.75 that reaches six
of the ten IoU thresholds. Its precision is that of one box found, and
`summary` averages the two categories:

```ocaml
let e = Mynd_coco.evaluate_boxes d dets
let s = Mynd_coco.summary e
let print v = Printf.sprintf "%.3f" v

let () =
  assert (print s.ap = "0.800");
  assert (print s.ap75 = "1.000");
  assert (print s.ar1 = "0.800")
```

The tables hold more than the twelve numbers. `e.precision` has axes
IoU threshold, recall threshold, category, area range and limit, with
NaN where no annotation of the category counts. Each category's AP is
the mean of its defined entries over the first two axes, at range all
(`0`) and limit 100 (`2`):

```ocaml
let per_category (e : Mynd_coco.evaluation) =
  let p = Nx.slice [ Nx.A; Nx.A; Nx.A; Nx.I 0; Nx.I 2 ] e.precision in
  let defined = Nx.logical_not (Nx.isnan p) in
  let sum = Nx.sum ~axes:[ 0; 1 ] (Nx.where defined p (Nx.zeros_like p)) in
  Nx.div sum (Nx.sum ~axes:[ 0; 1 ] (Nx.cast Nx.float64 defined))

let () =
  let ap = Nx.to_array (per_category e) in
  assert (Array.map print ap = [| "1.000"; "0.600" |])
```

The order of the categories is `e.categories`, by increasing id. This
mean adds in nx's order; `summary` adds exactly.

## Some categories

The evaluator raises on a detection whose category the dataset does not
list, so evaluating some categories filters both the dataset and the
detections:

```ocaml
let only keep (d : Mynd_coco.dataset) dets =
  let filter f a = Array.of_list (List.filter f (Array.to_list a)) in
  let kept_category (c : Mynd_coco.category) = keep c.id
  and kept_annotation (a : Mynd_coco.annotation) = keep a.category
  and kept_detection (det : _ Mynd_coco.detection) = keep det.category in
  ( {
      d with
      categories = filter kept_category d.categories;
      annotations = filter kept_annotation d.annotations;
    },
    filter kept_detection dets )

let () =
  let d', dets' = only (fun c -> c = 2) d dets in
  let s = Mynd_coco.(summary (evaluate_boxes d' dets')) in
  assert (print s.ap = "0.600")
```

## Masks

A model that predicts masks gives them as booleans `[n; h; w; 1]`.
`Mynd.Rle.encode` turns each into runs, and `Mynd_coco.evaluate_masks`
compares them with the masks of the annotations: polygons by
`Mynd_coco.runs`, and the dataset's runs as they are. An annotation
without a segmentation covers nothing, so this one is masked from its box:

```ocaml
let box_mask ~height ~width (b : Mynd_coco.box) =
  Nx.init Nx.bool [| height; width; 1 |] (fun i ->
      let x = Float.of_int i.(1) and y = Float.of_int i.(0) in
      x >= b.x && x < b.x +. b.w && y >= b.y && y < b.y +. b.h)

let runs_of b = Mynd.Rle.encode (box_mask ~height:40 ~width:60 b)

let masked =
  let mask (a : Mynd_coco.annotation) =
    { a with segmentation = Mynd_coco.Runs (runs_of a.box) }
  in
  { d with annotations = Array.map mask d.annotations }

let mask_dets =
  Array.map
    (fun (det : Mynd_coco.box Mynd_coco.detection) ->
      { det with region = runs_of det.region })
    dets

let () =
  let s = Mynd_coco.(summary (evaluate_masks masked mask_dets)) in
  assert (print s.ap = "0.800")
```

A mask detection's area, which picks its area range, is its pixel count.
A results file for masks often holds a `bbox` beside each mask;
`Mynd_coco.mask_detections_jsont` skips it, so the area stays the
mask's.

## Masks for training

A training program reads each image by its `file_name` and the masks of
its annotations by `Mynd_coco.runs`, the masks COCO's tools give and
the models trained on COCO saw. `Mynd.Rle.decode` turns them into masks
`[h; w; 1]`, and `Mynd.Box.of_xywh` turns the boxes into corners:

```ocaml
let targets (d : Mynd_coco.dataset) (im : Mynd_coco.image) =
  let anns =
    List.filter
      (fun (a : Mynd_coco.annotation) -> a.image = im.id && not a.crowd)
      (Array.to_list d.annotations)
  in
  let masks =
    List.map
      (fun (a : Mynd_coco.annotation) ->
        Mynd.Rle.decode
          (Mynd_coco.runs ~height:im.height ~width:im.width a.segmentation))
      anns
  in
  let xywh =
    List.concat_map
      (fun (a : Mynd_coco.annotation) -> [ a.box.x; a.box.y; a.box.w; a.box.h ])
      anns
  in
  let boxes =
    Mynd.Box.of_xywh
      (Nx.create Nx.float32 [| List.length anns; 4 |]
         (Array.of_list xywh))
  in
  (im.file_name, masks, boxes)

let () =
  let file, masks, boxes = targets d d.images.(0) in
  assert (file = "a.jpg");
  assert (Mynd.Rle.area (Mynd.Rle.encode (List.hd masks)) = 240);
  assert (Nx.to_array boxes = [| 4.; 6.; 24.; 18.; 30.; 10.; 46.; 26. |])
```

The first annotation's polygon covers its 20 by 12 cells. Its second has
no segmentation, so its mask is empty.

## Drawing other polygons

COCO's rule moves vertices onto a grid of fifths of a pixel, so it
belongs to COCO's annotations. To fill another polygon, such as a region
drawn over a result, take the pixels whose centres `(j + ½, i + ½)` lie
inside it. A centre is inside when a ray to its right crosses the
polygon's edges an odd number of times. An edge crosses the ray when one
end lies above the centre and the other does not, and the centre lies
left of the edge, which the sign of a cross product tells once it is
read along the edge's direction:

```ocaml
let fill ~height ~width (xy : float array) =
  let centres axis =
    Nx.init Nx.float64 [| height; width; 1 |] (fun i ->
        Float.of_int i.(axis) +. 0.5)
  in
  let cy = centres 0 and cx = centres 1 in
  let k = Array.length xy / 2 in
  let inside = ref (Nx.zeros Nx.bool [| height; width; 1 |]) in
  for e = 0 to k - 1 do
    let e' = (e + 1) mod k in
    let x0 = xy.(2 * e) and y0 = xy.((2 * e) + 1) in
    let x1 = xy.(2 * e') and y1 = xy.((2 * e') + 1) in
    let straddles = Nx.not_equal (Nx.greater_s cy y0) (Nx.greater_s cy y1) in
    (* (cx - x0) (y1 - y0) - (cy - y0) (x1 - x0), negative when the
       centre lies left of an edge that goes down. *)
    let cross =
      Nx.sub
        (Nx.mul_s (Nx.sub_s cx x0) (y1 -. y0))
        (Nx.mul_s (Nx.sub_s cy y0) (x1 -. x0))
    in
    let left = if y1 > y0 then Nx.less_s cross 0. else Nx.greater_s cross 0. in
    inside := Nx.logical_xor !inside (Nx.logical_and straddles left)
  done;
  !inside
```

On a triangle the two rules differ along the slanted edge:

```ocaml
let () =
  let t = [| 2.2; 3.1; 20.7; 5.5; 9.9; 19.4 |] in
  let centre = Mynd.Rle.encode (fill ~height:23 ~width:31 t) in
  let coco = Mynd_coco.runs ~height:23 ~width:31 (Mynd_coco.Polygons [ t ]) in
  assert (centre.counts <> coco.counts)
```
