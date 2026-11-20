(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** COCO's object annotations, the masks they describe, and the detection metric
    computed against them.

    A {e dataset} lists images, categories and annotations as COCO's instances
    files hold them. An {e annotation} marks one object of a category in an
    image with a box, an area and a segmentation. A {e crowd} annotation marks a
    group of objects too close to tell apart. It is never an object to find, and
    a detection inside it counts neither as right nor as wrong.

    COCO's boxes are a corner [(x, y)], a width [w] and a height [h], in the
    cell coordinates of {!Mynd}, so the box covers [\[x, x + w) × \[y, y + h)]
    and {!Mynd.Box.of_xywh} turns an array of them into mynd's boxes. A
    {e segmentation} is polygons, whose union is the object, or runs
    ({!Mynd.Rle.t}). {!runs} rasterises polygons by COCO's rule, so its masks
    are those COCO's tools give, the ones models trained on COCO saw.

    A {e detection} is a model's claim: an image, a category, a score and a
    region, a box or a mask. {!evaluate_boxes} and {!evaluate_masks} match
    detections to annotations by COCO's protocol and give tables of precision
    and recall; {!val-summary} gives the twelve numbers COCO reports. On a valid
    dataset whose annotation ids are positive and distinct, and on mask
    detections that pycocotools reads without a [bbox], every match and every
    entry of the tables equals pycocotools 2.0.11's [COCOeval] computed with
    each floating-point operation rounded on its own. Elsewhere pycocotools
    departs from its own protocol, and these functions keep to it. They tell
    annotations apart by position, where pycocotools takes a match with an
    annotation of id [0] for none and two annotations of one id for one. They
    give a mask detection the area range of its pixel count, where pycocotools
    takes the area of a results file's [bbox] when one is present. They raise on
    a detection of an unlisted category, which pycocotools drops, and on the
    other values their docs rule out, and they give NaN where pycocotools gives
    [-1].

    The life of an evaluation:
    + read the dataset ({!dataset_jsont}) or build it;
    + run the model and gather its detections in an array;
    + {!evaluate_boxes} or {!evaluate_masks};
    + {!val-summary}, or read the tables.

    Every value here is a host value. Unless its doc says otherwise, a function
    raises [Invalid_argument] at the call on a value its doc rules out.

    Reading a dataset and a model's results and printing the primary metric:
    {[
    let read jsont file =
      In_channel.with_open_bin file In_channel.input_all
      |> Jsont_bytesrw.decode_string jsont
      |> Result.get_ok

    let () =
      let d = read Mynd_coco.dataset_jsont "instances_val2017.json" in
      let r = read Mynd_coco.box_detections_jsont "results.json" in
      let s = Mynd_coco.(summary (evaluate_boxes d r)) in
      Printf.printf "AP %.3f\n" s.ap
    ]} *)

(** {1:datasets Datasets} *)

type box = { x : float; y : float; w : float; h : float }
(** The type for COCO's boxes. A box is {e valid} when its values are finite and
    [w] and [h] are non-negative. *)

(** The type for segmentations. COCO gives an object as polygons and a crowd as
    runs. *)
type segmentation =
  | Polygons of float array list
      (** Polygons [[|x₀; y₀; x₁; y₁; …|]], each closed from its last vertex to
          its first, whose union is the object. *)
  | Runs of Mynd.Rle.t

type image = { id : int; file_name : string; height : int; width : int }
(** The type for images. *)

type category = { id : int; name : string }
(** The type for categories. *)

type annotation = {
  id : int;  (** Its identifier in the file. Evaluation never reads it. *)
  image : int;  (** The [id] of its image. *)
  category : int;  (** The [id] of its category. *)
  box : box;
  area : float;
      (** The object's area as the file states it. For COCO's polygons it is the
          polygons' geometric area, which differs from the pixel count of their
          mask. *)
  crowd : bool;
  segmentation : segmentation;
}
(** The type for annotations. *)

type dataset = {
  images : image array;
  categories : category array;
  annotations : annotation array;
}
(** The type for datasets. A dataset is {e valid} when no two images and no two
    categories share an [id], and every annotation names a listed image and
    category, has a valid box and has an [area] that is finite and non-negative.
    Annotations are told apart by their position in [annotations]. *)

val box_jsont : box Jsont.t
(** [box_jsont] maps COCO's [bbox], [[x, y, w, h]]. *)

val segmentation_jsont : segmentation Jsont.t
(** [segmentation_jsont] maps COCO's [segmentation] member as {!dataset_jsont}
    reads and writes it. *)

val dataset_jsont : dataset Jsont.t
(** [dataset_jsont] maps COCO's instances files: the members [images],
    [categories] and [annotations], with the fields above under COCO's names
    ([image_id], [category_id], [bbox] as [[x, y, w, h]] and [iscrowd] as [0] or
    [1]), other members skipped. An absent [annotations] decodes as none, so
    COCO's image-information files decode. A segmentation decodes from a list of
    polygons, from runs with [counts] a list or a string ({!Mynd.Rle.of_string})
    and [size] [[height, width]], and from an absent member as [Polygons []]; it
    encodes as polygons or a string. Decoding fails on any other absent field.
*)

(** {1:masks Masks}

    {e COCO's rule} gives a polygon's mask in an image of [h] rows and [w]
    columns. It works on a grid of fifths of a pixel, in float64 with each
    operation rounded on its own. Each vertex [(x, y)] moves to
    [(trunc ((5x) + ½), trunc ((5y) + ½))], [trunc] rounding toward zero. Each
    edge becomes grid points one step apart along its longer axis, [x] on a tie,
    from its end of lesser coordinate on that axis; at step [t] the other
    coordinate is [trunc ((c + (s t)) + ½)], for [c] that end's coordinate and
    [s] the edge's slope. A {e crossing} of column [j] is a step of this closed
    path between grid columns [5j + 2] and [5j + 3]. It lies at row
    [⌈(v - 2)/5⌉], clamped to [[0, h]], for [v] the lesser grid row of the step.
    Pixel [(i, j)] is covered when an odd number of column [j]'s crossings lie
    at rows up to [i]. So a polygon that crosses itself covers what it encloses
    an odd number of times, a polygon of fewer than three vertices covers no
    pixel, and vertices may lie outside the image.

    Vertices move by up to a tenth of a pixel and a crossing takes the upper
    point of its step, so the rule differs from covering the pixels whose
    centres lie inside the polygon. *)

val runs : height:int -> width:int -> segmentation -> Mynd.Rle.t
(** [runs ~height ~width s] is the mask [s] covers in an image of [height] rows
    and [width] columns: [r] for [Runs r], and for polygons the union of each
    polygon's mask by COCO's rule.

    Raises [Invalid_argument] if [height] or [width] is negative, a polygon has
    an odd number of coordinates or a coordinate [x] that is not finite or has
    [|5x + ½| ≥ 2³¹], or [r]'s size is not [height] by [width]. *)

(** {1:detections Detections}

    Within an image and a category, detections of equal score rank in their
    array's order. *)

type 'r detection = { image : int; category : int; score : float; region : 'r }
(** The type for detections of regions ['r]: {!type-box} or {!Mynd.Rle.t}. *)

val box_detections_jsont : box detection array Jsont.t
(** [box_detections_jsont] maps COCO's results files for boxes: an array of
    objects with [image_id], [category_id], [bbox] and [score], in the file's
    order, other members skipped. *)

val mask_detections_jsont : Mynd.Rle.t detection array Jsont.t
(** [mask_detections_jsont] maps COCO's results files for masks: as
    {!box_detections_jsont} with [segmentation] in place of [bbox], runs as
    {!segmentation_jsont} reads them, written with [counts] a string. A [bbox]
    member is skipped. *)
