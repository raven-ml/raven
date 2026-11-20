(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* ocamlopt on arm64, power and riscv contracts a product feeding a sum
   into one fused multiply-add; pycocotools' bbIou rounds the two apart on
   every target. *)
let mul a b = Sys.opaque_identity (a *. b)

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

(* Evaluation *)

type evaluation = {
  categories : category array;
  precision : Nx.host Nx.float64_t;
  recall : Nx.host Nx.float64_t;
}

(* The thresholds pycocotools' Params.setDetParams computes with
   np.linspace(.5, .95, 10) and np.linspace(.0, 1., 101). Written as values,
   since the formula, k * step + start, fuses on arm64 and gives
   0.7999999999999999 for the seventh. *)
let iou_thresholds =
  [| 0.5; 0.55; 0.6; 0.65; 0.7; 0.75; 0.8; 0.85; Float.pred 0.9; 0.95 |]

let recall_thresholds =
  let above = [ 35; 41; 47; 57; 69; 70; 82; 83; 94; 95 ] in
  Array.init 101 (fun i ->
      let r = Float.of_int i /. 100. in
      if List.mem i above then Float.succ r else r)

(* Area ranges all, small, medium and large, bounds included, and the
   limits on detections an image. *)
let ranges =
  [|
    (0., 1e10); (0., 32. *. 32.); (32. *. 32., 96. *. 96.); (96. *. 96., 1e10);
  |]

let limits = [| 1; 10; 100 |]
let kept = 100
let largest_area = 1e10

(* pycocotools' precision eps, np.spacing(1). *)
let eps = epsilon_float
let nt = Array.length iou_thresholds
let nr = Array.length recall_thresholds
let na = Array.length ranges
let nm = Array.length limits

(* Ids to positions in the order of increasing id, raising on a shared
   id. *)
let index fn what ids =
  let sorted = Array.copy ids in
  Array.sort Int.compare sorted;
  let pos = Hashtbl.create (Array.length ids) in
  Array.iteri
    (fun i id ->
      if Hashtbl.mem pos id then invalid_argf "%s: two %s of id %d" fn what id;
      Hashtbl.add pos id i)
    sorted;
  (sorted, pos)

let find fn what pos id =
  match Hashtbl.find_opt pos id with
  | Some i -> i
  | None -> invalid_argf "%s: no %s of id %d" fn what id

let valid_box fn what b =
  let ok v = Float.is_finite v in
  if not (ok b.x && ok b.y && ok b.w && ok b.h && b.w >= 0. && b.h >= 0.) then
    invalid_argf "%s: %s box [%g, %g, %g, %g] is not valid" fn what b.x b.y b.w
      b.h

(* What evaluation reads of a dataset: images and categories by increasing
   id, and each annotation's image and category positions. *)
type checked = {
  images : image array;
  categories : category array;
  ann_image : int array;
  ann_category : int array;
  image_pos : (int, int) Hashtbl.t;
  category_pos : (int, int) Hashtbl.t;
}

let check fn (d : dataset) =
  let ids, image_pos =
    index fn "images" (Array.map (fun (i : image) -> i.id) d.images)
  in
  let images =
    let by_id = Hashtbl.create (Array.length d.images) in
    Array.iter (fun (i : image) -> Hashtbl.replace by_id i.id i) d.images;
    Array.map (Hashtbl.find by_id) ids
  in
  let cids, category_pos =
    index fn "categories" (Array.map (fun (c : category) -> c.id) d.categories)
  in
  let categories =
    let by_id = Hashtbl.create (Array.length d.categories) in
    Array.iter (fun (c : category) -> Hashtbl.replace by_id c.id c) d.categories;
    Array.map (Hashtbl.find by_id) cids
  in
  Array.iter
    (fun (a : annotation) ->
      valid_box fn "an annotation's" a.box;
      if not (Float.is_finite a.area && a.area >= 0.) then
        invalid_argf "%s: an annotation's area is %g" fn a.area)
    d.annotations;
  let ann_image =
    Array.map
      (fun (a : annotation) -> find fn "image" image_pos a.image)
      d.annotations
  in
  let ann_category =
    Array.map
      (fun (a : annotation) -> find fn "category" category_pos a.category)
      d.annotations
  in
  { images; categories; ann_image; ann_category; image_pos; category_pos }

let check_detection fn c (det : _ detection) area =
  let i = find fn "image" c.image_pos det.image in
  let k = find fn "category" c.category_pos det.category in
  if Float.is_nan det.score then
    invalid_argf "%s: a detection's score is NaN" fn;
  if area > largest_area then
    invalid_argf "%s: a detection's area %g lies past every range" fn area;
  (i, k)

(* One image and category under one range: the kept detections' scores by
   rank, and for each IoU threshold [t] and rank [r], at [t * nd + r],
   whether the detection took an annotation and whether it is ignored. *)
type img_eval = {
  scores : float array;
  taken : bool array;
  ignored : bool array;
  counted : int;
}

(* evaluateImg (pycocotools 2.0.11, cocoeval.py) for one range: annotations
   that count first, then each detection in rank order takes the best
   annotation it may. *)
let match_img ~iou ~crowd ~ann_area ~det_area ~scores ~dets ~anns (lo, hi) =
  let ng = Array.length anns and nd = Array.length dets in
  let ig =
    Array.map (fun g -> crowd g || ann_area g < lo || ann_area g > hi) anns
  in
  let order = Array.init ng Fun.id in
  Array.stable_sort (fun a b -> Bool.compare ig.(a) ig.(b)) order;
  let g_ig = Array.map (fun o -> ig.(o)) order in
  let g_crowd = Array.map (fun o -> crowd anns.(o)) order in
  let taken = Array.make (nt * nd) false in
  let ignored = Array.make (nt * nd) false in
  let g_taken = Array.make ng false in
  for t = 0 to nt - 1 do
    Array.fill g_taken 0 ng false;
    for r = 0 to nd - 1 do
      let best = ref iou_thresholds.(t) and m = ref (-1) in
      let g = ref 0 in
      while !g < ng do
        if g_taken.(!g) && not g_crowd.(!g) then incr g
        else if !m > -1 && (not g_ig.(!m)) && g_ig.(!g) then g := ng
        else begin
          let v = iou.((r * ng) + order.(!g)) in
          if v >= !best then begin
            best := v;
            m := !g
          end;
          incr g
        end
      done;
      if !m >= 0 then begin
        ignored.((t * nd) + r) <- g_ig.(!m);
        taken.((t * nd) + r) <- true;
        g_taken.(!m) <- true
      end
    done
  done;
  for r = 0 to nd - 1 do
    let a = det_area dets.(r) in
    if a < lo || a > hi then
      for t = 0 to nt - 1 do
        let j = (t * nd) + r in
        if not taken.(j) then ignored.(j) <- true
      done
  done;
  let counted = Array.fold_left (fun n b -> if b then n else n + 1) 0 ig in
  { scores; taken; ignored; counted }

(* Indices [0, n) stably sorted by descending [score]. *)
let by_score n score =
  let o = Array.init n Fun.id in
  Array.stable_sort (fun a b -> Float.compare (score b) (score a)) o;
  o

(* accumulate (cocoeval.py) for one category, range and limit, into the
   flat tables. *)
let accumulate ~precision ~recall ~k ~nk ~a ~mi (es : img_eval array) =
  let m = limits.(mi) in
  let counted = Array.fold_left (fun n e -> n + e.counted) 0 es in
  if counted > 0 then begin
    let take e = Int.min m (Array.length e.scores) in
    let nd = Array.fold_left (fun n e -> n + take e) 0 es in
    (* Each rank's image evaluation and its rank there. *)
    let src = Array.make nd 0 and at = Array.make nd 0 in
    let scores = Array.make nd 0. in
    let j = ref 0 in
    Array.iteri
      (fun ei e ->
        for r = 0 to take e - 1 do
          src.(!j) <- ei;
          at.(!j) <- r;
          scores.(!j) <- e.scores.(r);
          incr j
        done)
      es;
    let order = by_score nd (fun i -> scores.(i)) in
    let n = Float.of_int counted in
    let rc = Array.make nd 0. and pr = Array.make nd 0. in
    for t = 0 to nt - 1 do
      let tp = ref 0 and fp = ref 0 in
      Array.iteri
        (fun q i ->
          let e = es.(src.(i)) in
          let j = (t * Array.length e.scores) + at.(i) in
          if not e.ignored.(j) then if e.taken.(j) then incr tp else incr fp;
          let tp = Float.of_int !tp and fp = Float.of_int !fp in
          rc.(q) <- tp /. n;
          pr.(q) <- tp /. (fp +. tp +. eps))
        order;
      recall.((((((t * nk) + k) * na) + a) * nm) + mi) <-
        (if nd > 0 then rc.(nd - 1) else 0.);
      for q = nd - 1 downto 1 do
        if pr.(q) > pr.(q - 1) then pr.(q - 1) <- pr.(q)
      done;
      (* np.searchsorted (rc, thresholds, side='left'): the first rank whose
         recall reaches each threshold, and 0 from the first that none
         reaches. *)
      let q = ref 0 in
      Array.iteri
        (fun ri thr ->
          while !q < nd && rc.(!q) < thr do
            incr q
          done;
          let v = if !q < nd then pr.(!q) else 0. in
          precision.((((((((t * nr) + ri) * nk) + k) * na) + a) * nm) + mi) <- v)
        recall_thresholds
    done
  end

(* evaluate and accumulate (cocoeval.py). [place j] is detection [j]'s
   image and category positions, [det_area j] its area and [pair_iou j g]
   its IoU with annotation [g]. *)
let evaluate (d : dataset) c ~place ~det_area ~pair_iou
    (dets : _ detection array) =
  let nk = Array.length c.categories and ni = Array.length c.images in
  let place = Array.init (Array.length dets) place in
  (* Annotations and detections of each image and category, in array
     order, keyed [k ni + i] so that keys sort by category, then image. *)
  let groups = Hashtbl.create 1024 in
  let add key f =
    let g, dd = Option.value (Hashtbl.find_opt groups key) ~default:([], []) in
    Hashtbl.replace groups key (f (g, dd))
  in
  for j = Array.length d.annotations - 1 downto 0 do
    add
      ((c.ann_category.(j) * ni) + c.ann_image.(j))
      (fun (g, dd) -> (j :: g, dd))
  done;
  for j = Array.length dets - 1 downto 0 do
    let i, k = place.(j) in
    add ((k * ni) + i) (fun (g, dd) -> (g, j :: dd))
  done;
  let keys = Hashtbl.fold (fun key _ acc -> key :: acc) groups [] in
  let keys = List.sort Int.compare keys in
  let evals = Array.make_matrix nk na [] in
  let crowd g = d.annotations.(g).crowd in
  let ann_area g = d.annotations.(g).area in
  List.iter
    (fun key ->
      let g, dd = Hashtbl.find groups key in
      let anns = Array.of_list g and dd = Array.of_list dd in
      let o = by_score (Array.length dd) (fun r -> dets.(dd.(r)).score) in
      let nd = Int.min kept (Array.length o) and ng = Array.length anns in
      let ranked = Array.init nd (fun r -> dd.(o.(r))) in
      let iou = Array.make (nd * ng) 0. in
      for r = 0 to nd - 1 do
        for g = 0 to ng - 1 do
          iou.((r * ng) + g) <- pair_iou ranked.(r) anns.(g)
        done
      done;
      let scores = Array.map (fun j -> dets.(j).score) ranked in
      let k = key / ni in
      for a = 0 to na - 1 do
        let e =
          match_img ~iou ~crowd ~ann_area ~det_area ~scores ~dets:ranked ~anns
            ranges.(a)
        in
        evals.(k).(a) <- e :: evals.(k).(a)
      done)
    keys;
  let precision = Array.make (nt * nr * nk * na * nm) Float.nan in
  let recall = Array.make (nt * nk * na * nm) Float.nan in
  for k = 0 to nk - 1 do
    for a = 0 to na - 1 do
      let es = Array.of_list (List.rev evals.(k).(a)) in
      for mi = 0 to nm - 1 do
        accumulate ~precision ~recall ~k ~nk ~a ~mi es
      done
    done
  done;
  {
    categories = c.categories;
    precision = Nx.create Nx.float64 [| nt; nr; nk; na; nm |] precision;
    recall = Nx.create Nx.float64 [| nt; nk; na; nm |] recall;
  }

(* bbIou (maskApi.c) for one pair. *)
let box_iou (d : box) (g : box) crowd =
  let w = Float.min (d.w +. d.x) (g.w +. g.x) -. Float.max d.x g.x in
  if w <= 0. then 0.
  else
    let h = Float.min (d.h +. d.y) (g.h +. g.y) -. Float.max d.y g.y in
    if h <= 0. then 0.
    else
      let i = mul w h and da = mul d.w d.h in
      let u = if crowd then da else da +. mul g.w g.h -. i in
      i /. u

let evaluate_boxes d (dets : box detection array) =
  let fn = "Mynd_coco.evaluate_boxes" in
  let c = check fn d in
  let det_area j =
    let b = dets.(j).region in
    b.w *. b.h
  in
  let place j =
    valid_box fn "a detection's" dets.(j).region;
    check_detection fn c dets.(j) (det_area j)
  in
  let pair_iou j g =
    let a = d.annotations.(g) in
    box_iou dets.(j).region a.box a.crowd
  in
  evaluate d c ~place ~det_area ~pair_iou dets
