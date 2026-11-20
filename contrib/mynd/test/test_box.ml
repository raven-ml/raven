(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

let archive file = (file, lazy (Nx_io.load_npz file))

(* torchvision's box IoU and pycocotools' boxes of masks, from golden/annot.npz
   (contrib/mynd/test/gen/annot.py); torchvision's NMS, from golden/select.npz
   (contrib/mynd/test/gen/select.py) *)
let annot = archive "golden/annot.npz"
let select = archive "golden/select.npz"

let entry (file, npz) dt key =
  match Nx_io.Archive.find key (Lazy.force npz) with
  | Some p -> Nx.unpack dt p
  | None -> failf "%s has no %s" file key

let floats x = Nx.to_array (Nx.cast Nx.float64 x)

let close tol want got =
  equal (array int) (Nx.shape want) (Nx.shape got);
  equal
    (array (if tol = 0. then float_exact else float tol))
    (floats want) (floats got)

let boxes a = Nx.create Nx.float32 [| Array.length a / 4; 4 |] a

(* Boxes *)

let conversions =
  test "conversions" (fun () ->
      let b = boxes [| 1.; 2.; 4.; 8.; -3.; 0.5; -1.; 0.75 |] in
      equal (array float_exact)
        [| 1.; 2.; 3.; 6.; -3.; 0.5; 2.; 0.25 |]
        (floats (V.Box.to_xywh b));
      equal (array float_exact)
        [| 2.5; 5.; 3.; 6.; -2.; 0.625; 2.; 0.25 |]
        (floats (V.Box.to_cxcywh b));
      equal (array float_exact) (floats b)
        (floats (V.Box.of_xywh (V.Box.to_xywh b)));
      equal (array float_exact) (floats b)
        (floats (V.Box.of_cxcywh (V.Box.to_cxcywh b))))

let gen_box =
  Gen.(
    let+ x = float_range (-50.) 50.
    and+ y = float_range (-50.) 50.
    and+ w = float_range (-5.) 40.
    and+ h = float_range (-5.) 40. in
    [| x; y; x +. w; y +. h |])

let areas =
  prop "area is the product of the sides, 0 when empty" gen_box (fun b ->
      let w = b.(2) -. b.(0) and h = b.(3) -. b.(1) in
      cover "empty" (w <= 0. || h <= 0.);
      let want = if w <= 0. || h <= 0. then 0. else w *. h in
      equal float_exact want
        (Nx.item [ 0 ] (V.Box.area (Nx.create Nx.float64 [| 1; 4 |] b))))

let iou_golden =
  test "box IoU is torchvision's" (fun () ->
      let a = entry annot Nx.float32 "box.a"
      and b = entry annot Nx.float32 "box.b" in
      close 0. (entry annot Nx.float32 "box.iou") (V.Box.iou a b))

(* IoU against its definition in float64, over boxes on a grid of eighths, so
   that every area is exact. *)
let iou_law =
  prop "IoU is the intersection over the union"
    Gen.(pair gen_box gen_box)
    (fun (a, b) ->
      let q a = Array.map (fun v -> Float.round (v *. 8.) /. 8.) a in
      let a = q a and b = q b in
      let area a =
        Float.max 0. (a.(2) -. a.(0)) *. Float.max 0. (a.(3) -. a.(1))
      in
      let area a = if a.(2) <= a.(0) || a.(3) <= a.(1) then 0. else area a in
      let i =
        Float.max 0. (Float.min a.(2) b.(2) -. Float.max a.(0) b.(0))
        *. Float.max 0. (Float.min a.(3) b.(3) -. Float.max a.(1) b.(1))
      in
      let u = area a +. area b -. i in
      cover "disjoint" (i = 0.);
      cover "overlapping" (i > 0.);
      let want = if u = 0. then 0. else i /. u in
      let got =
        V.Box.iou
          (Nx.create Nx.float64 [| 1; 4 |] a)
          (Nx.create Nx.float64 [| 1; 4 |] b)
      in
      equal (array int) [| 1; 1 |] (Nx.shape got);
      equal float_exact want (Nx.item [ 0; 0 ] got))

let iou_batches =
  test "IoU pairs every box with every box, batch axes broadcast" (fun () ->
      let a =
        Nx.create Nx.float64 [| 2; 3; 4 |]
          (Array.init 24 (fun i ->
               Float.of_int (i mod 7) *. if i mod 4 >= 2 then 2. else 1.))
      in
      let b =
        Nx.create Nx.float64 [| 1; 2; 4 |] [| 0.; 0.; 4.; 4.; 1.; 1.; 6.; 9. |]
      in
      let r = V.Box.iou a b in
      equal (array int) [| 2; 3; 2 |] (Nx.shape r);
      equal float_exact
        (Nx.item [ 1; 2; 1 ] r)
        (Nx.item [ 0; 0 ]
           (V.Box.iou
              (Nx.slice [ Nx.I 1; Nx.R (2, 3) ] a)
              (Nx.slice [ Nx.I 0; Nx.R (1, 2) ] b))))

let empty_iou =
  test "empty boxes have IoU 0" (fun () ->
      let z = boxes [| 1.; 1.; 1.; 5. |] in
      equal (array float_exact) [| 0. |] (floats (V.Box.iou z z)))

let of_mask =
  test "a mask's box" (fun () ->
      let m = Nx.zeros Nx.bool [| 3; 5; 6; 1 |] in
      let m =
        Nx.set [ Nx.I 0; Nx.R (1, 3); Nx.R (2, 5) ] (Nx.scalar Nx.bool true) m
      in
      let m = Nx.set [ Nx.I 1; Nx.I 4; Nx.I 0 ] (Nx.scalar Nx.bool true) m in
      equal (array float_exact)
        [| 2.; 1.; 5.; 3.; 0.; 4.; 1.; 5.; 0.; 0.; 0.; 0. |]
        (floats (V.Box.of_mask Nx.float32 m)))

let of_mask_golden =
  test "a mask's box is pycocotools'" (fun () ->
      let m = Nx.cast Nx.bool (entry annot Nx.uint8 "m") in
      let m = Nx.unsqueeze ~axes:[ -1 ] m in
      let want = entry annot Nx.float64 "bbox" in
      (* pycocotools gives [0 0 0 0] for an empty mask too. *)
      close 0. want (V.Box.to_xywh (V.Box.of_mask Nx.float64 m)))

let of_mask_empty =
  cases "a mask with no row or no column has the box 0 0 0 0"
    ~id:(fun s -> String.concat "x" (List.map string_of_int (Array.to_list s)))
    [ [| 3; 0; 1 |]; [| 0; 4; 1 |]; [| 0; 0; 1 |]; [| 2; 0; 3; 1 |] ]
    (fun shape ->
      let b = V.Box.of_mask Nx.float32 (Nx.zeros Nx.bool shape) in
      let lead = Array.sub shape 0 (Array.length shape - 3) in
      equal (array int) (Array.append lead [| 4 |]) (Nx.shape b);
      equal (array float_exact)
        (Array.make (Array.length (floats b)) 0.)
        (floats b))

(* Moving boxes *)

(* Dyadic boxes moved by dyadic transforms, the exact rule rounded once into
   float64, from golden/geometry.npz (contrib/mynd/test/gen/geometry.py). *)
let geometry = archive "golden/geometry.npz"

let f64 shape a = Nx.create Nx.float64 shape a
let mat a = f64 [| 3; 3 |] a
let one a = f64 [| 1; 4 |] a
let fl = Float.of_int

(* Floats compared as numbers: [-0.] is [0.], and NaN equals NaN under
   [float_exact]. *)
let numbers x = Array.map (fun v -> v +. 0.) (floats x)
let identity = [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]

(* The flips of an image [w] columns wide or [h] rows high: [x ↦ w - x],
   [y ↦ h - y]. *)
let flip_x w = [| -1.; 0.; w; 0.; 1.; 0.; 0.; 0.; 1. |]
let flip_y h = [| 1.; 0.; 0.; 0.; -1.; h; 0.; 0.; 1. |]

(* The quarter turns: clockwise of an image [h] rows high, [(x, y) ↦ (h - y,
   x)]; anticlockwise of one [w] columns wide, [(x, y) ↦ (y, w - x)]. *)
let turn_cw h = [| 0.; -1.; h; 1.; 0.; 0.; 0.; 0.; 1. |]
let turn_ccw w = [| 0.; 1.; 0.; -1.; 0.; w; 0.; 0.; 1. |]
let shift x y = [| 1.; 0.; x; 0.; 1.; y; 0.; 0.; 1. |]
let scale sx sy = [| sx; 0.; 0.; 0.; sy; 0.; 0.; 0.; 1. |]

let turn a =
  let c = Float.cos a and s = Float.sin a in
  [| c; -.s; 0.; s; c; 0.; 0.; 0.; 1. |]

(* [d = 1 - x / 4]: [0] on the line [x = 4]. *)
let horizon = [| 1.; 0.; 0.; 0.; 1.; 0.; -0.25; 0.; 1. |]

let moved_cells =
  cases "array operations move a box as they move its cells" ~id:fst
    [
      ("identity", (identity, [| 1.; 2.; 4.; 5. |]));
      ("flip of 10 columns", (flip_x 10., [| 6.; 2.; 9.; 5. |]));
      ("clockwise quarter turn of 8 rows", (turn_cw 8., [| 3.; 1.; 6.; 4. |]));
      ("crop from row 1, column 2", (shift (-2.) (-1.), [| -1.; 1.; 2.; 4. |]));
    ]
    (fun (_, (m, want)) ->
      equal (array float_exact) want
        (numbers (V.Box.transform (mat m) (one [| 1.; 2.; 4.; 5. |]))))

type move =
  | Flip_x
  | Flip_y
  | Turn_cw
  | Turn_ccw
  | Crop of int * int * int * int
  | Scale of int * int

let gen_warp =
  Gen.(
    let* h = int_range 1 12 in
    let* w = int_range 1 12 in
    let* i0 = int_range 0 (h - 1) in
    let* i1 = int_range i0 (h - 1) in
    let* j0 = int_range 0 (w - 1) in
    let* j1 = int_range j0 (w - 1) in
    let* ci = int_range 0 (h - 1) in
    let* cj = int_range 0 (w - 1) in
    let* ch = int_range 1 (h - ci) in
    let* cw = int_range 1 (w - cj) in
    let* sx = int_range 1 3 in
    let* sy = int_range 1 3 in
    let+ move =
      of_list
        [
          Flip_x;
          Flip_y;
          Turn_cw;
          Turn_ccw;
          Crop (ci, cj, ch, cw);
          Scale (sx, sy);
        ]
    in
    ((h, w), (i0, i1, j0, j1), move))

(* The box of a block of pixels, moved and clipped to the new image, is the box
   of the block's mask warped by [nearest] at the warp's positions. A block
   moved off the image clips to an empty box, and its mask to no pixel. *)
let warp_law =
  prop "a box moves as a warp moves its pixels" gen_warp
    (fun ((h, w), (i0, i1, j0, j1), move) ->
      let m, (h', w') =
        match move with
        | Flip_x -> (flip_x (fl w), (h, w))
        | Flip_y -> (flip_y (fl h), (h, w))
        | Turn_cw -> (turn_cw (fl h), (w, h))
        | Turn_ccw -> (turn_ccw (fl w), (w, h))
        | Crop (ci, cj, ch, cw) -> (shift (fl (-cj)) (fl (-ci)), (ch, cw))
        | Scale (sx, sy) -> (scale (fl sx) (fl sy), (sy * h, sx * w))
      in
      cover "flip" (move = Flip_x || move = Flip_y);
      cover "quarter turn of a non-square image"
        ((move = Turn_cw || move = Turn_ccw) && h <> w);
      cover "crop" (match move with Crop _ -> true | _ -> false);
      cover "scaling" (match move with Scale _ -> true | _ -> false);
      cover "a box on the border"
        (i0 = 0 || j0 = 0 || i1 = h - 1 || j1 = w - 1);
      let mask =
        Nx.init Nx.bool [| h; w; 1 |] (fun ix ->
            ix.(0) >= i0 && ix.(0) <= i1 && ix.(1) >= j0 && ix.(1) <= j1)
      in
      let m = mat m in
      let at =
        V.Transform.apply (Nx.inv m) (V.Sample.centres Nx.float64 (h', w'))
      in
      let moved = V.Sample.nearest ~edge:`Zero mask at in
      let b = one [| fl j0; fl i0; fl (j1 + 1); fl (i1 + 1) |] in
      let size = f64 [| 4 |] [| fl w'; fl h'; fl w'; fl h' |] in
      let got = Nx.maximum_s (Nx.minimum (V.Box.transform m b) size) 0. in
      if Nx.item [] (Nx.any moved) then
        equal (array float_exact)
          (floats (V.Box.of_mask Nx.float64 moved))
          (numbers got)
      else (
        cover "a box cropped away" true;
        equal float_exact 0. (Nx.item [ 0 ] (V.Box.area got))))

let kinds =
  [
    ("affine", 0);
    ("projective, d > 0", 1);
    ("projective, d < 0", 2);
    ("d of both signs", 3);
    ("d = 0 at a corner", 4);
    ("empty", 5);
  ]

let exact_moves =
  cases "a box moves to the least box of its image, rounded once" ~id:fst kinds
    (fun (_, k) ->
      let kind = Nx.to_array (entry geometry Nx.int64 "transform.kind") in
      let rows =
        List.filter
          (fun i -> kind.(i) = Int64.of_int k)
          (List.init (Array.length kind) Fun.id)
      in
      let rows =
        Nx.create Nx.int64
          [| List.length rows |]
          (Array.of_list (List.map Int64.of_int rows))
      in
      let pick key = Nx.take ~axis:0 rows (entry geometry Nx.float64 key) in
      equal (array float_exact)
        (numbers (pick "transform.want"))
        (numbers (V.Box.transform (pick "transform.m") (pick "transform.b"))))

(* Transforms with entries in [-2, 2], translations in [-20, 20], and a last
   row [0 0 1] or [p q ±1] with [p] and [q] up to a tenth. *)
let gen_transform =
  Gen.(
    let e = float_range (-2.) 2. and t = float_range (-20.) 20. in
    let+ a = array ~size:(constant 4) e
    and+ tx = t
    and+ ty = t
    and+ pq =
      frequency
        [
          (1, constant (0., 0.));
          (2, pair (float_range (-0.1) 0.1) (float_range (-0.1) 0.1));
        ]
    and+ s = frequency [ (3, constant 1.); (1, constant (-1.)) ] in
    [| a.(0); a.(1); tx; a.(2); a.(3); ty; fst pq; snd pq; s |])

let neg_law =
  prop "-m moves a box as m does" (Gen.pair gen_transform gen_box)
    (fun (m, b) ->
      let r = numbers (V.Box.transform (mat m) (one b)) in
      cover "NaN" (Float.is_nan r.(0));
      cover "finite" (Float.is_finite r.(0));
      equal (array float_exact) r
        (numbers (V.Box.transform (Nx.neg (mat m)) (one b))))

(* A transform whose first row is [0 0 t] collapses every box to a segment of
   width [0]. *)
let gen_collapse =
  Gen.(
    let+ m = gen_transform in
    [| 0.; 0.; m.(2); m.(3); m.(4); m.(5); 0.; 0.; 1. |])

let compose_law =
  prop "moving a box twice gives a box holding the box moved once"
    Gen.(
      triple gen_transform
        (frequency [ (3, gen_transform); (1, gen_collapse) ])
        gen_box)
    (fun (a, c, b) ->
      let a = mat a and c = mat c and b = one b in
      let inner = floats (V.Box.transform c b) in
      let twice = floats (V.Box.transform a (V.Box.transform c b)) in
      let once = floats (V.Box.transform (Nx.matmul a c) b) in
      (* An empty inner box moves to [0 0 0 0], while the box moved once holds
         the segment the first move collapsed the box to. *)
      let collapsed = inner.(2) <= inner.(0) || inner.(3) <= inner.(1) in
      cover "the first move collapses the box" collapsed;
      cover "the box moved twice is NaN" (Float.is_nan twice.(0));
      if not (collapsed || Float.is_nan twice.(0)) then (
        cover "both are finite boxes" true;
        (* The first move rounds the box's corners; its error, relative to
           the coordinates, is far below 1e-9. *)
        let slack v = 1e-9 *. (1. +. Float.abs v) in
        at_most float_exact ~than:(once.(0) +. slack once.(0)) twice.(0);
        at_most float_exact ~than:(once.(1) +. slack once.(1)) twice.(1);
        at_most float_exact ~than:(twice.(2) +. slack twice.(2)) once.(2);
        at_most float_exact ~than:(twice.(3) +. slack twice.(3)) once.(3)))

(* Flips, quarter turns, dyadic scalings and integer translations, which move
   boxes on a grid of quarters exactly. *)
let gen_axis_map =
  Gen.(
    let* k = int_range 0 4 in
    let* t = int_range (-8) 8 in
    let* u = int_range (-8) 8 in
    let+ sx = of_list [ -2.; -0.5; 0.5; 1.; 2.; 4. ]
    and+ sy = of_list [ -2.; -0.5; 0.5; 1.; 2.; 4. ] in
    match k with
    | 0 -> flip_x (fl t)
    | 1 -> turn_cw (fl t)
    | 2 -> turn_ccw (fl t)
    | 3 -> shift (fl t) (fl u)
    | _ -> scale sx sy)

let gen_quarter_box =
  Gen.(
    let q lo hi = map (fun k -> fl k /. 4.) (int_range lo hi) in
    let+ x = q (-40) 40 and+ y = q (-40) 40 and+ w = q 1 40 and+ h = q 1 40 in
    [| x; y; x +. w; y +. h |])

let axis_law =
  prop "axis maps move boxes as their product does"
    Gen.(triple gen_axis_map gen_axis_map gen_quarter_box)
    (fun (a, c, b) ->
      let a = mat a and c = mat c and b = one b in
      equal (array float_exact)
        (numbers (V.Box.transform (Nx.matmul a c) b))
        (numbers (V.Box.transform a (V.Box.transform c b))))

let both_names ((n, _), (n', _)) = n ^ ", " ^ n'
let nan = Float.nan
let inf = Float.infinity

let moved_empties =
  let empties =
    [
      ("x1 = x0", [| 2.; 1.; 2.; 5. |]);
      ("x1 < x0", [| 3.; 1.; 2.; 5. |]);
      ("y1 = y0", [| 1.; 4.; 3.; 4. |]);
      ("y1 < y0", [| 1.; 5.; 3.; 4. |]);
      ("both", [| 3.; 5.; 2.; 4. |]);
      ("x0 = inf", [| inf; 0.; 5.; 1. |]);
      ("across d = 0", [| 5.; 1.; 3.; 5. |]);
    ]
  and moves =
    [
      ("identity", identity);
      ("30 degrees", turn (Float.pi /. 6.));
      ("d = 1 - x / 4", horizon);
      ("NaN in m", [| nan; 0.; 0.; 0.; 1.; 0.; nan; 0.; 1. |]);
    ]
  in
  cases "an empty box moves to 0 0 0 0" ~id:both_names
    (List.concat_map (fun b -> List.map (fun m -> (b, m)) moves) empties)
    (fun ((_, b), (_, m)) ->
      equal (array float_exact) [| 0.; 0.; 0.; 0. |]
        (floats (V.Box.transform (mat m) (one b))))

let moved_nans =
  let boxes =
    [
      ("x0 = NaN", [| nan; 0.; 5.; 1. |]);
      ("y1 = NaN", [| 0.; 0.; 5.; nan |]);
      ("x0 = -inf", [| -.inf; 0.; 5.; 1. |]);
      ("x1 = inf", [| 0.; 0.; inf; 1. |]);
      ("the plane", [| -.inf; -.inf; inf; inf |]);
    ]
  and moves =
    [
      ("identity", identity);
      ("30 degrees", turn (Float.pi /. 6.));
      ("d = 1 + x / 4", [| 1.; 0.; 0.; 0.; 1.; 0.; 0.25; 0.; 1. |]);
    ]
  in
  cases "a box with a non-finite coordinate or meeting d = 0 moves to NaN"
    ~id:both_names
    (List.concat_map (fun b -> List.map (fun m -> (b, m)) moves) boxes
    @ [
        (("across x = 4", [| 1.; 0.; 5.; 1. |]), ("d = 1 - x / 4", horizon));
        ( ("a corner on x = 4", [| 1.; 0.; 4.; 1. |]),
          ("d = 1 - x / 4", horizon) );
        ( ("a corner on x = 4", [| 4.; 0.; 5.; 1. |]),
          ("d = x / 4 - 1", Array.map Float.neg horizon) );
        ( ("finite", [| 1.; 0.; 3.; 1. |]),
          ("NaN in d", [| 1.; 0.; 0.; 0.; 1.; 0.; nan; 0.; 1. |]) );
      ])
    (fun ((_, b), (_, m)) ->
      equal (array float_exact) [| nan; nan; nan; nan |]
        (floats (V.Box.transform (mat m) (one b))))

let infinite_shift =
  test "a box moved by an infinite translation (baseline)" (fun () ->
      let r = V.Box.transform (mat (shift inf 0.)) (one [| 1.; 2.; 4.; 5. |]) in
      expect
        (String.concat " "
           (Array.to_list (Array.map (Printf.sprintf "%g") (floats r))))
      @@ __POS_OF__ {| inf 2 inf 5 |})

let transform_shapes =
  group "shapes of moved boxes"
    [
      test "no box" (fun () ->
          let m = Nx.stack [ mat identity; mat (turn 0.5) ] in
          equal (array int) [| 2; 0; 4 |]
            (Nx.shape (V.Box.transform m (Nx.zeros Nx.float64 [| 2; 0; 4 |]))));
      test "each image's boxes move by its transform" (fun () ->
          let ms = [ identity; turn 0.5; horizon ] in
          let b =
            f64 [| 3; 2; 4 |]
              (Array.init 24 (fun i -> fl ((i * 7) mod 11) +. fl (i mod 4)))
          in
          let got = V.Box.transform (Nx.stack (List.map mat ms)) b in
          List.iteri
            (fun i m ->
              equal (array float_exact)
                (numbers (V.Box.transform (mat m) (Nx.get [ i ] b)))
                (numbers (Nx.get [ i ] got)))
            ms);
      test "a transform of no batch axis moves every box" (fun () ->
          let b =
            f64 [| 2; 3; 4 |] (Array.init 24 (fun i -> fl ((i * 5) mod 13)))
          in
          let m = mat (turn 0.5) in
          let got = V.Box.transform m b in
          equal (array int) [| 2; 3; 4 |] (Nx.shape got);
          equal (array float_exact)
            (numbers (V.Box.transform m (Nx.get [ 1 ] b)))
            (numbers (Nx.get [ 1 ] got)));
      test "boxes of two axes an image" (fun () ->
          let b =
            f64 [| 2; 3; 5; 4 |] (Array.init 120 (fun i -> fl ((i * 5) mod 17)))
          in
          let m = Nx.stack [ mat identity; mat (turn 0.5) ] in
          let got = V.Box.transform m b in
          equal (array int) [| 2; 3; 5; 4 |] (Nx.shape got);
          equal (array float_exact)
            (numbers (V.Box.transform (mat (turn 0.5)) (Nx.get [ 1 ] b)))
            (numbers (Nx.get [ 1 ] got)));
      test "compiled is eager" (fun () ->
          let m = Nx.stack [ mat (turn 0.5); mat horizon ] in
          let b =
            f64 [| 2; 3; 4 |]
              [| 1.; 0.; 3.; 1.; 1.; 0.; 5.; 1.; 3.; 0.; 2.; 1.;
                 0.; 0.; 2.; 2.; 1.; 1.; 6.; 4.; nan; 0.; 1.; 1. |]
          in
          equal (array float_exact)
            (numbers (V.Box.transform m b))
            (numbers (Rune.jit' (V.Box.transform m) b)));
      test "boxes end in 4" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Box.transform") (fun () ->
              V.Box.transform (mat identity) (Nx.zeros Nx.float64 [| 2; 3 |])));
      test "fewer axes than the transforms' batch axes and the box's"
        (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Box.transform") (fun () ->
              V.Box.transform
                (Nx.stack (List.init 4 (fun _ -> mat identity)))
                (Nx.zeros Nx.float64 [| 4 |])));
      test "batch axes that do not broadcast" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Box.transform") (fun () ->
              V.Box.transform
                (Nx.stack [ mat identity; mat identity ])
                (Nx.zeros Nx.float64 [| 3; 1; 4 |])));
    ]

type flt = F : string * (float, _) Nx.dtype * float -> flt

(* Boxes of integers moved by axis maps stay exact in every float format. A
   turn by 30 degrees moves boxes within [[0, 8]] to corners below 16, each
   rounding two products and a sum, by at most half a unit in the last place
   of the values below 16, compared with float64 on the same rounded inputs. *)
let moved_dtypes =
  cases "boxes move in every float format"
    ~id:(fun (F (n, _, _)) -> n)
    [
      F ("float16", Nx.float16, 3. *. Float.ldexp 1. (-8));
      F ("bfloat16", Nx.bfloat16, 3. *. Float.ldexp 1. (-5));
      F ("float32", Nx.float32, 3. *. Float.ldexp 1. (-21));
    ]
    (fun (F (_, dt, tol)) ->
      let b =
        f64 [| 3; 4 |] [| 1.; 2.; 4.; 5.; 0.; 0.; 8.; 3.; 6.; 1.; 7.; 8. |]
      in
      List.iter
        (fun m ->
          let want = V.Box.transform (mat m) b in
          let got =
            V.Box.transform (Nx.cast dt (mat m)) (Nx.cast dt b)
            |> Nx.cast Nx.float64
          in
          equal (array float_exact) (numbers want) (numbers got))
        [ flip_x 10.; turn_cw 8.; turn_ccw 8.; shift (-2.) 3.; scale 2. 0.5 ];
      let m = Nx.cast dt (mat (turn (Float.pi /. 6.))) and b = Nx.cast dt b in
      let want =
        V.Box.transform (Nx.cast Nx.float64 m) (Nx.cast Nx.float64 b)
      in
      equal (array (float tol)) (floats want) (floats (V.Box.transform m b)))

(* Turns away from multiples of a quarter, with a small projective part, keep
   [d] near [1]. The law assumes each side of the result at one corner, at
   least half a pixel from every other corner's coordinate, which the steps
   of the differences never cross. *)
let gen_smooth =
  Gen.(
    let+ a = float_range 0.2 1.37
    and+ p = float_range (-0.01) 0.01
    and+ q = float_range (-0.01) 0.01
    and+ x = float_range 0. 10.
    and+ y = float_range 0. 10.
    and+ w = float_range 1. 10.
    and+ h = float_range 1. 10. in
    ( Nx.matmul (mat (turn a)) (mat [| 1.; 0.; 0.; 0.; 1.; 0.; p; q; 1. |]),
      [| x; y; x +. w; y +. h |] ))

let apart m b =
  let x0 = b.(0) and y0 = b.(1) and x1 = b.(2) and y1 = b.(3) in
  let c = f64 [| 4; 2 |] [| x0; y0; x1; y0; x1; y1; x0; y1 |] in
  let p = floats (V.Transform.apply m c) in
  let gap k =
    let v = List.sort compare (List.init 4 (fun i -> p.((2 * i) + k))) in
    Float.min (List.nth v 1 -. List.nth v 0) (List.nth v 3 -. List.nth v 2)
  in
  Float.min (gap 0) (gap 1) > 0.5

let weights = f64 [| 1; 4 |] [| 1.; -2.; 0.5; 3. |]
let weighed (m, b) = Nx.sum (Nx.mul (V.Box.transform m b) weights)

let transform_grads =
  prop "the derivative of a moved box is its corners'" gen_smooth
    (fun (m, b) ->
      assume (apart m b);
      let p = Nx.Ptree.(pair tensor tensor) in
      match Rune.check_grads p weighed (m, one b) with
      | Ok () -> ()
      | Error e -> fail e)

let finite_grads =
  cases "the derivative is finite at boxes near d = 0, NaN or empty" ~id:fst
    [
      ("one side of d = 0, near it", (horizon, [| 1.; 0.; 3.999; 1. |]));
      ("across d = 0", (horizon, [| 1.; 0.; 5.; 1. |]));
      ("a corner on d = 0", (horizon, [| 1.; 0.; 4.; 1. |]));
      ("empty", (turn 0.5, [| 3.; 0.; 2.; 1. |]));
      ("a NaN coordinate", (turn 0.5, [| nan; 0.; 2.; 1. |]));
      ("an infinite coordinate", (turn 0.5, [| 0.; 0.; inf; 1. |]));
    ]
    (fun (_, (m, b)) ->
      let gm, gb =
        Rune.grad Nx.Ptree.(pair tensor tensor) weighed (mat m, one b)
      in
      let finite g = Array.map Float.is_finite (floats g) in
      equal (array bool) (Array.make 9 true) (finite gm);
      equal (array bool) (Array.make 4 true) (finite gb))

(* At an axis map each side of the result is at two tied corners, whose
   tangents add to the side's derivative. *)
let tie_grads =
  cases "the derivative at tied corners is the side's" ~id:fst
    [
      ( "flip",
        ( flip_x 10.,
          [| 0.; 0.; -1.; 0.; 0.; 1.; 0.; 0.; -1.; 0.; 0.; 0.; 0.; 0.; 0.; 1. |]
        ) );
      ( "clockwise quarter turn",
        ( turn_cw 8.,
          [| 0.; 0.; 0.; -1.; 1.; 0.; 0.; 0.; 0.; -1.; 0.; 0.; 0.; 0.; 1.; 0. |]
        ) );
      ( "scaling",
        ( scale 2. 3.,
          [| 2.; 0.; 0.; 0.; 0.; 3.; 0.; 0.; 0.; 0.; 2.; 0.; 0.; 0.; 0.; 3. |]
        ) );
      ( "scaling by a negative",
        ( scale (-2.) 0.5,
          [|
            0.; 0.; -2.; 0.; 0.; 0.5; 0.; 0.; -2.; 0.; 0.; 0.; 0.; 0.; 0.; 0.5;
          |] ) );
    ]
    (fun (_, (m, jacobian)) ->
      let row k =
        Rune.grad'
          (fun b ->
            Nx.sum (Nx.slice [ Nx.A; Nx.I k ] (V.Box.transform (mat m) b)))
          (one [| 1.; 2.; 4.; 5. |])
      in
      equal (array float_exact) jacobian
        (Array.concat (List.init 4 (fun k -> numbers (row k)))))

(* Generalized intersection over union *)

let giou1 a b = Nx.item [ 0; 0 ] (V.Box.giou (one a) (one b))

let giou_cases =
  cases "GIoU of boxes whose value the definition gives" ~id:fst
    [
      ("equal boxes", ([| 0.; 0.; 2.; 3. |], [| 0.; 0.; 2.; 3. |], 1.));
      ("apart", ([| 0.; 0.; 1.; 1. |], [| 3.; 0.; 4.; 1. |], -0.5));
      ("nested", ([| 0.; 0.; 4.; 4. |], [| 1.; 1.; 2.; 2. |], 1. /. 16.));
      ("touching", ([| 0.; 0.; 1.; 1. |], [| 1.; 0.; 2.; 1. |], 0.));
      ("zero width", ([| 5.; 0.; 5.; 1. |], [| 0.; 0.; 1.; 1. |], -0.8));
      ("a point", ([| 5.; 5.; 5.; 5. |], [| 0.; 0.; 1.; 1. |], -0.96));
      ("one segment twice", ([| 0.; 0.; 0.; 1. |], [| 0.; 0.; 0.; 1. |], 0.));
      ("two segments apart", ([| 0.; 0.; 0.; 1. |], [| 3.; 0.; 3.; 1. |], -1.));
      ("inverted", ([| 5.; 0.; 3.; 1. |], [| 0.; 0.; 1.; 1. |], -2. /. 3.));
    ]
    (fun (_, (a, b, want)) ->
      (* Each value is two quotients of small integers and their difference,
         each rounded by at most half a unit in the last place of 1. *)
      equal (float (2. *. epsilon_float)) want (giou1 a b))

let giou_kinds =
  [
    "overlapping";
    "disjoint";
    "nested";
    "touching";
    "zero width";
    "zero height";
    "point";
  ]

(* Coordinates on quarters make [i], [u] and [c] exact, so [iou], [p / c] and
   their difference round once each, each by at most [2⁻⁵⁴] or [2⁻⁵³]: within
   [epsilon_float] of the exact value. *)
let giou_exact =
  cases "GIoU is the paper's Algorithm 2 where it needs no ordering"
    ~id:Fun.id giou_kinds (fun k ->
      let kind = Nx.to_array (entry geometry Nx.int64 "giou.kind") in
      let index = ref (-1) in
      List.iteri (fun i k' -> if k' = k then index := i) giou_kinds;
      let rows =
        List.filter
          (fun i -> kind.(i) = Int64.of_int !index)
          (List.init (Array.length kind) Fun.id)
      in
      let n = List.length rows in
      let rows =
        Nx.create Nx.int64 [| n |] (Array.of_list (List.map Int64.of_int rows))
      in
      let pick key = Nx.take ~axis:0 rows (entry geometry Nx.float64 key) in
      let p = Nx.reshape [| n; 1; 4 |] (pick "giou.p")
      and g = Nx.reshape [| n; 1; 4 |] (pick "giou.g") in
      equal (array (float epsilon_float))
        (floats (pick "giou.exact"))
        (floats (V.Box.giou p g)))

(* [F (name, dtype, tolerance)] is a float format and a tolerance there. *)
let giou_golden =
  cases "GIoU is torchvision's where its c - u is not below 0"
    ~id:(fun (F (n, _, _)) -> n)
    [ F ("float32", Nx.float32, 0.); F ("float64", Nx.float64, 0.) ]
    (fun (F (_, dt, _)) ->
      let bits = Nx.Dtype.bits dt in
      let get k = entry geometry dt (Printf.sprintf "giou.%s%d" k bits) in
      let clamped = Printf.sprintf "giou.clamped%d" bits in
      greater int64 ~than:0L (Nx.item [] (entry geometry Nx.int64 clamped));
      close 0. (get "want") (V.Box.giou (get "a") (get "b")))

(* Boxes of every kind: arbitrary, inverted ones among them; tiny boxes in
   large ones far from the origin; boxes of zero width; and strips, two boxes
   of one height overlapping across, whose union is their enclosing box, so
   that [c = u] in the reals and [c - u] rounds below [0] in float32 for
   about a fifth of them. *)
let gen_any_pair =
  Gen.(
    let tiny =
      let+ x = float_range 1000. 4000.
      and+ y = float_range 1000. 4000.
      and+ w = float_range 100. 400.
      and+ h = float_range 100. 400.
      and+ fx = float_range 0. 1.
      and+ fy = float_range 0. 1.
      and+ iw = float_range 1e-3 1.
      and+ ih = float_range 1e-3 1. in
      let ix = x +. (fx *. (w -. 1.)) and iy = y +. (fy *. (h -. 1.)) in
      ([| x; y; x +. w; y +. h |], [| ix; iy; ix +. iw; iy +. ih |])
    and segment =
      let+ a = gen_box and+ x = float_range (-50.) 50. in
      (a, [| x; a.(1); x; a.(3) |])
    and strip =
      let+ x = float_range (-50.) 50.
      and+ y = float_range (-50.) 50.
      and+ w = float_range 1. 40.
      and+ h = float_range 1. 40.
      and+ f = float_range 0. 1.
      and+ e = float_range 0. 40. in
      ([| x; y; x +. w; y +. h |], [| x +. (f *. w); y; x +. w +. e; y +. h |])
    in
    frequency
      [ (2, pair gen_box gen_box); (1, tiny); (1, segment); (2, strip) ])

let f32 a = Nx.create Nx.float32 [| 1; 4 |] a

(* [c - u] in float32, each operation rounded once: a float32 operation's
   exact result rounded to float64 and then to float32 is rounded once. *)
let enclosure_less_union a b =
  let r x = Int32.float_of_bits (Int32.bits_of_float x) in
  let area x0 y0 x1 y1 =
    let w = r (x1 -. x0) and h = r (y1 -. y0) in
    if w <= 0. || h <= 0. then 0. else r (w *. h)
  in
  let side lo lo' hi hi' =
    Float.max (r (Float.min hi hi' -. Float.max lo lo')) 0.
  in
  let i = r (side a.(0) b.(0) a.(2) b.(2) *. side a.(1) b.(1) a.(3) b.(3)) in
  let areas = area a.(0) a.(1) a.(2) a.(3) +. area b.(0) b.(1) b.(2) b.(3) in
  let u = r (r areas -. i) in
  let lo k = Float.min a.(k) b.(k) and hi k = Float.max a.(k) b.(k) in
  area (lo 0) (lo 1) (hi 2) (hi 3) -. u

let giou_range =
  prop "GIoU lies in [-1, iou a b]" ~count:300 gen_any_pair (fun (a, b) ->
      let a = f32 a and b = f32 b in
      cover "c - u rounds below 0"
        (enclosure_less_union (floats a) (floats b) < 0.);
      let iou = Nx.item [ 0; 0 ] (V.Box.iou a b)
      and g = Nx.item [ 0; 0 ] (V.Box.giou a b) in
      let empty x = x.(2) <= x.(0) || x.(3) <= x.(1) in
      cover "an empty box" (empty (floats a) || empty (floats b));
      cover "nested, a tiny box far from the origin"
        (Nx.item [ 0; 0 ] a > 999.);
      at_most float_exact ~than:iou g;
      at_most float_exact ~than:g (-1.))

let giou_symmetric =
  prop "GIoU of b and a is the transpose of a and b's" gen_any_pair
    (fun (a, b) ->
      let a = Nx.stack [ one a; one b ] |> Nx.reshape [| 2; 4 |]
      and b = Nx.stack [ one b; one a; one a ] |> Nx.reshape [| 3; 4 |] in
      equal (array float_exact)
        (floats (V.Box.giou a b))
        (floats (Nx.transpose (V.Box.giou b a))))

let giou_scaled =
  prop "GIoU is unchanged by scaling both boxes by a power of 2"
    Gen.(pair gen_any_pair (int_range (-8) 8))
    (fun ((a, b), k) ->
      (* Coordinates on a grid of [2⁻²⁰] keep every area clear of the
         subnormals, where scaling would round. *)
      let grid = Array.map (fun v -> Float.round (Float.ldexp v 20)) in
      let a = Array.map (fun v -> Float.ldexp v (-20)) (grid a)
      and b = Array.map (fun v -> Float.ldexp v (-20)) (grid b) in
      let s = Float.ldexp 1. k in
      let scaled x = Nx.mul_s (one x) s in
      equal (array float_exact)
        (floats (V.Box.giou (one a) (one b)))
        (floats (V.Box.giou (scaled a) (scaled b))))

(* The other box is at least 1 a side, so [c] and [u] are at least [1], and
   every box lies within [[-10, 20]]. A change of [2⁻²⁰] in a width then moves
   [i] and [c] by less than [30 · 2⁻²⁰], and GIoU by less than [1e-4]. *)
let giou_continuous =
  prop "GIoU is continuous as a box collapses or inverts"
    Gen.(
      let+ x = float_range (-10.) 10.
      and+ y = float_range (-10.) 10.
      and+ h = float_range 0. 10.
      and+ w = float_range (-.Float.ldexp 1. (-20)) (Float.ldexp 1. (-20))
      and+ ox = float_range (-10.) 10.
      and+ oy = float_range (-10.) 10.
      and+ ow = float_range 1. 10.
      and+ oh = float_range 1. 10. in
      ([| x; y; x; y +. h |], w, [| ox; oy; ox +. ow; oy +. oh |]))
    (fun (a, w, o) ->
      let wide = Array.copy a in
      wide.(2) <- a.(2) +. w;
      cover "inverted" (w < 0.);
      equal (float 1e-4) (giou1 a o) (giou1 wide o))

let giou_shapes =
  group "shapes of GIoU"
    [
      test "no box" (fun () ->
          equal (array int) [| 0; 3 |]
            (Nx.shape
               (V.Box.giou
                  (Nx.zeros Nx.float64 [| 0; 4 |])
                  (Nx.ones Nx.float64 [| 3; 4 |]))));
      test "matched pairs are the diagonal" (fun () ->
          let a =
            f64 [| 3; 4 |]
              [| 0.; 0.; 2.; 2.; 1.; 1.; 5.; 3.; 4.; 0.; 2.; 1. |]
          and b =
            f64 [| 3; 4 |]
              [| 1.; 0.; 3.; 3.; 0.; 0.; 1.; 1.; 6.; 2.; 7.; 2. |]
          in
          let matched =
            V.Box.giou (Nx.reshape [| 3; 1; 4 |] a) (Nx.reshape [| 3; 1; 4 |] b)
          in
          equal (array int) [| 3; 1; 1 |] (Nx.shape matched);
          let all = floats (V.Box.giou a b) in
          equal (array float_exact)
            [| all.(0); all.(4); all.(8) |]
            (floats matched));
      test "a NaN coordinate gives NaN in its row and column" (fun () ->
          let a = f64 [| 2; 4 |] [| nan; 0.; 1.; 1.; 0.; 0.; 2.; 2. |]
          and b = f64 [| 2; 4 |] [| 0.; 0.; 1.; 1.; 5.; 5.; 6.; nan |] in
          let g = floats (V.Box.giou a b) in
          equal (array bool)
            [| true; true; false; true |]
            (Array.map Float.is_nan g));
      test "boxes end in 4" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Box.giou") (fun () ->
              V.Box.giou
                (Nx.zeros Nx.float64 [| 2; 3 |])
                (Nx.zeros Nx.float64 [| 2; 4 |])));
    ]

let loss (a, b) = Nx.sum (Nx.rsub_s 1. (V.Box.giou a b))

let giou_grads =
  prop "the derivative of GIoU away from ties"
    Gen.(pair gen_quarter_box gen_quarter_box)
    (fun (a, b) ->
      (* Moving each coordinate off the grid of quarters keeps the sides of
         the intersection and the enclosing box from tying. *)
      let a = Array.mapi (fun i v -> v +. (0.1 *. fl (i + 1))) a in
      let p = Nx.Ptree.(pair tensor tensor) in
      match Rune.check_grads p loss (one a, one b) with
      | Ok () -> ()
      | Error e -> fail e)

let giou_finite =
  cases "the derivative of GIoU is finite where c, u or an area is 0" ~id:fst
    [
      ("c = 0", ([| 0.; 0.; 0.; 1. |], [| 0.; 0.; 0.; 1. |]));
      ("u = 0", ([| 0.; 0.; 0.; 1. |], [| 3.; 0.; 3.; 1. |]));
      ("an empty box", ([| 5.; 0.; 3.; 1. |], [| 0.; 0.; 1.; 1. |]));
      ("a point", ([| 5.; 5.; 5.; 5. |], [| 0.; 0.; 1.; 1. |]));
    ]
    (fun (_, (a, b)) ->
      let p = Nx.Ptree.(pair tensor tensor) in
      let ga, gb = Rune.grad p loss (one a, one b) in
      let finite g = Array.map Float.is_finite (floats g) in
      equal (array bool) (Array.make 4 true) (finite ga);
      equal (array bool) (Array.make 4 true) (finite gb))

let target = [| 0.; 0.; 1.; 1. |]
let descent a = floats (Rune.grad' (fun a -> loss (a, one target)) (one a))

let giou_widens =
  test "descent widens a box just wider than a segment" (fun () ->
      let g = descent [| 5.; 0.; 5. +. Float.ldexp 1. (-20); 1. |] in
      (* Descent moves [x₁ - x₀] by [-(g₂ - g₀)]. *)
      greater float_exact ~than:g.(2) g.(0))

(* Inverted boxes in x, y or both, inside the target's span or reaching out of
   it. *)
let gen_inverted =
  Gen.(
    let* inside = bool in
    let side = if inside then float_range 0.01 0.99 else float_range (-3.) 4. in
    let+ x0 = side and+ x1 = side and+ y0 = side and+ y1 = side
    and+ which = int_range 0 2 in
    let lo a b = Float.min a b and hi a b = Float.max a b in
    match which with
    | 0 -> [| hi x0 x1; lo y0 y1; lo x0 x1; hi y0 y1 |]
    | 1 -> [| lo x0 x1; hi y0 y1; hi x0 x1; lo y0 y1 |]
    | _ -> [| hi x0 x1; hi y0 y1; lo x0 x1; lo y0 y1 |])

let giou_shrinks =
  prop "descent never widens an inverted box" gen_inverted (fun a ->
      assume (a.(2) < a.(0) || a.(3) < a.(1));
      let g = descent a in
      let inside = Array.for_all (fun v -> v > 0. && v < 1.) a in
      cover "inside the target's span" inside;
      cover "reaching out of it" (not inside);
      at_most float_exact ~than:0. (-.(g.(2) -. g.(0)));
      at_most float_exact ~than:0. (-.(g.(3) -. g.(1)));
      if inside then
        equal (array float_exact) [| 0.; 0.; 0.; 0. |]
          (Array.map (fun v -> v +. 0.) g))

(* Coordinates of eighths within [[0, 1]] keep every difference, area and sum
   exact in float16 and bfloat16, so only [i / u], [p / c] and their
   difference round, each by at most half a unit in the last place of 1. *)
let giou_dtypes =
  cases "GIoU in every float format, boxes within [0, 1]"
    ~id:(fun (F (n, _, _)) -> n)
    [
      F ("float16", Nx.float16, 3. *. Float.ldexp 1. (-11));
      F ("bfloat16", Nx.bfloat16, 3. *. Float.ldexp 1. (-8));
      F ("float32", Nx.float32, 3. *. Float.ldexp 1. (-24));
    ]
    (fun (F (_, dt, tol)) ->
      let a =
        f64 [| 3; 4 |]
          [| 0.; 0.; 0.5; 0.5; 0.25; 0.125; 0.875; 0.75; 0.5; 0.5; 0.25; 1. |]
      and b =
        f64 [| 2; 4 |] [| 0.125; 0.25; 0.625; 1.; 0.75; 0.; 1.; 0.25 |]
      in
      let a = Nx.cast dt a and b = Nx.cast dt b in
      let want = V.Box.giou (Nx.cast Nx.float64 a) (Nx.cast Nx.float64 b) in
      equal (array (float tol)) (floats want) (floats (V.Box.giou a b)))

(* float16's largest float is 65504. A box of 300 by 300 has an infinite
   area, and so its IoU with itself is NaN. Boxes of 200 by 200 and 50 by 50
   have finite areas and IoU [0], and the box enclosing both, 300 by 300, has
   an infinite area. *)
let giou_overflow =
  cases "float16 boxes whose areas overflow give NaN GIoU" ~id:fst
    [
      ( "a box of 300 by 300",
        ([| 0.; 0.; 300.; 300. |], [| 0.; 0.; 300.; 300. |], nan) );
      ( "an enclosing box of 300 by 300",
        ([| 0.; 0.; 200.; 200. |], [| 250.; 250.; 300.; 300. |], 0.) );
    ]
    (fun (_, (a, b, iou)) ->
      let box x = Nx.create Nx.float16 [| 1; 4 |] x in
      equal float_exact iou (Nx.item [ 0; 0 ] (V.Box.iou (box a) (box b)));
      equal float_exact nan (Nx.item [ 0; 0 ] (V.Box.giou (box a) (box b))))

(* Non-maximum suppression *)

(* [greedy ?classes iou t s] is greedy NMS from the IoUs [iou] (row-major n ×
   n): the boxes by descending score, equal scores lowest index first and NaN
   first, each kept unless a kept one before it has IoU above [t]. *)
let greedy ?classes iou t s =
  let n = Array.length s in
  let order =
    List.init n Fun.id
    |> List.stable_sort (fun i j ->
        match (Float.is_nan s.(i), Float.is_nan s.(j)) with
        | true, true -> 0
        | true, false -> -1
        | false, true -> 1
        | false, false -> compare s.(j) s.(i))
  in
  let keep = Array.make n false in
  let kept = ref [] in
  List.iter
    (fun i ->
      let same j =
        match classes with None -> true | Some c -> c.(i) = c.(j)
      in
      if not (List.exists (fun j -> same j && iou.((i * n) + j) > t) !kept) then (
        keep.(i) <- true;
        kept := i :: !kept))
    order;
  keep

let golden_nms =
  cases "nms is torchvision's"
    ~id:(fun (c, t) -> Printf.sprintf "%s%g" (if c then "classes, " else "") t)
    (List.concat_map
       (fun t -> [ (false, t); (true, t) ])
       [ 0.3; 0.5; 0.7; 0.85 ])
    (fun (by_class, t) ->
      let b = entry select Nx.float32 "nms.boxes"
      and s = entry select Nx.float32 "nms.scores" in
      let classes =
        if by_class then Some (entry select Nx.int64 "nms.classes") else None
      in
      let key =
        Printf.sprintf "nms.%s%g" (if by_class then "classes." else "") t
      in
      equal (array bool)
        (Nx.to_array (entry select Nx.bool key))
        (Nx.to_array (V.Box.nms ?classes ~iou:t b s)))

let gen_boxes =
  Gen.(
    let* n = int_range 0 24 in
    let box =
      let+ x = float_range 0. 30.
      and+ y = float_range 0. 30.
      and+ w = float_range 1. 20.
      and+ h = float_range 1. 20. in
      [| x; y; x +. w; y +. h |]
    in
    let score =
      frequency
        [
          (8, map (fun k -> Float.of_int k /. 4.) (int_range 0 8));
          (1, constant Float.nan);
          (1, constant (-0.));
        ]
    in
    let+ boxes = array ~size:(constant n) box
    and+ scores = array ~size:(constant n) score
    and+ classes = array ~size:(constant n) (int_range 0 2)
    and+ t = float_range 0. 0.9 in
    (boxes, scores, classes, t))

let greedy_law =
  prop "nms is greedy suppression" ~count:300
    Gen.(pair bool gen_boxes)
    (fun (by_class, (bs, ss, cs, t)) ->
      let n = Array.length bs in
      let b =
        Nx.create Nx.float64 [| n; 4 |] (Array.concat (Array.to_list bs))
      in
      let s = Nx.create Nx.float64 [| n |] ss in
      let iou = floats (V.Box.iou b b) in
      let ties x =
        Array.fold_left (fun c y -> if y = x then c + 1 else c) 0 ss
      in
      cover "a tie" (Array.exists (fun x -> ties x > 1) ss);
      cover "a NaN score" (Array.exists Float.is_nan ss);
      cover "a -0 beside a 0"
        (Array.exists (fun x -> 1. /. x < 0.) ss
        && Array.exists (( = ) 0.) ss
        && Array.exists (fun x -> 1. /. x > 0. && x = 0.) ss);
      let classes, cl =
        if by_class then
          ( Some (Nx.create Nx.int64 [| n |] (Array.map Int64.of_int cs)),
            Some cs )
        else (None, None)
      in
      let keep, k = V.Box.nms_rounds ?classes ~iou:t b s in
      equal (array bool) (greedy ?classes:cl iou t ss) (Nx.to_array keep);
      less int ~than:(n + 2) (Int32.to_int (Nx.item [] k)))

(* The second box's IoU with the first is about [t] in float32, and for 0.85
   exactly [t] rounded up into float32: greater than [t] as a real number,
   though equal to [t] compared in float32. *)
(* [-0.] and [0.] are equal scores: the lower position comes first. *)
let signed_zeros =
  test "-0 and 0 tie" (fun () ->
      let b =
        Nx.create Nx.float32 [| 2; 4 |] [| 0.; 0.; 2.; 2.; 0.; 0.; 2.; 2. |]
      in
      let s = Nx.create Nx.float32 [| 2 |] [| -0.; 0. |] in
      equal (array bool) [| true; false |]
        (Nx.to_array (V.Box.nms ~iou:0.5 b s)))

let threshold =
  cases "the threshold is compared as a real number" ~id:string_of_float
    [ 0.85; 0.3; 0.7; 0.5 ] (fun t ->
      let t32 = Int32.float_of_bits (Int32.bits_of_float t) in
      let b =
        Nx.create Nx.float32 [| 2; 4 |] [| 0.; 0.; 1.; 1.; 0.; 0.; 1.; t32 |]
      in
      let s = Nx.create Nx.float32 [| 2 |] [| 1.; 0.5 |] in
      let iou = Nx.item [ 0; 1 ] (V.Box.iou b b) in
      if t = 0.85 then equal float_exact t32 iou;
      equal (array bool)
        [| true; not (iou > t) |]
        (Nx.to_array (V.Box.nms ~iou:t b s)))

(* [steps n iou t s] is the parallel greedy's steps, each keeping every
   undecided box no earlier undecided box conflicts with and dropping those it
   suppresses: the dependence length. *)
let steps n iou t s =
  let rank = Array.make n 0 in
  List.iteri
    (fun r i -> rank.(i) <- r)
    (List.init n Fun.id |> List.stable_sort (fun i j -> compare s.(j) s.(i)));
  let conflict i j = i <> j && iou.((i * n) + j) > t in
  let undecided = Array.make n true in
  let count = ref 0 in
  while Array.exists Fun.id undecided do
    incr count;
    let roots =
      List.filter
        (fun i ->
          undecided.(i)
          && not
               (List.exists
                  (fun j ->
                    undecided.(j) && rank.(j) < rank.(i) && conflict i j)
                  (List.init n Fun.id)))
        (List.init n Fun.id)
    in
    List.iter
      (fun i ->
        undecided.(i) <- false;
        for j = 0 to n - 1 do
          if conflict i j then undecided.(j) <- false
        done)
      roots
  done;
  !count

(* A chain: each box overlaps the next above the threshold, by descending score,
   so each decision waits for the one before. *)
let chain n =
  ( Nx.create Nx.float64 [| n; 4 |]
      (Array.concat
         (List.init n (fun i ->
              let x = Float.of_int i in
              [| x; 0.; x +. 4.; 1. |]))),
    Nx.create Nx.float64 [| n |] (Array.init n (fun i -> Float.of_int (n - i)))
  )

let rounds_nms =
  cases "rounds stay within twice the dependence length and one" ~id:fst
    [
      ( "detections 0.5",
        ( entry select Nx.float32 "nms.boxes" |> Nx.cast Nx.float64,
          entry select Nx.float32 "nms.scores" |> Nx.cast Nx.float64,
          0.5 ) );
      ( "detections 0.7",
        ( entry select Nx.float32 "nms.boxes" |> Nx.cast Nx.float64,
          entry select Nx.float32 "nms.scores" |> Nx.cast Nx.float64,
          0.7 ) );
      ("chain of 40", (fst (chain 40), snd (chain 40), 0.5));
    ]
    (fun (_, (b, s, t)) ->
      let n = Nx.dim 0 b in
      let l = steps n (floats (V.Box.iou b b)) t (floats s) in
      let _, k = V.Box.nms_rounds ~iou:t b s in
      less int ~than:((2 * l) + 2) (Int32.to_int (Nx.item [] k)))

let batches_nms =
  test "each set of leading axes selects alone" (fun () ->
      let b = entry select Nx.float32 "nms.boxes"
      and s = entry select Nx.float32 "nms.scores" in
      let b2 = Nx.stack [ b; Nx.flip ~axes:[ 0 ] b ]
      and s2 = Nx.stack [ s; Nx.flip ~axes:[ 0 ] s ] in
      let k = V.Box.nms ~iou:0.5 b2 s2 in
      equal (array bool)
        (Nx.to_array (V.Box.nms ~iou:0.5 b s))
        (Nx.to_array (Nx.slice [ Nx.I 0 ] k));
      equal (array bool)
        (Nx.to_array
           (V.Box.nms ~iou:0.5 (Nx.flip ~axes:[ 0 ] b) (Nx.flip ~axes:[ 0 ] s)))
        (Nx.to_array (Nx.slice [ Nx.I 1 ] k)))

let compiled_nms =
  test "compiled is eager" (fun () ->
      let b = entry select Nx.float32 "nms.boxes"
      and s = entry select Nx.float32 "nms.scores" in
      equal (array bool)
        (Nx.to_array (V.Box.nms ~iou:0.5 b s))
        (Nx.to_array (Rune.jit' (fun b -> V.Box.nms ~iou:0.5 b s) b)))

let refusals =
  group "misuse"
    [
      test "a mask without its channel axis" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Box.of_mask") (fun () ->
              V.Box.of_mask Nx.float32 (Nx.zeros Nx.bool [| 4; 4 |])));
      test "boxes end in 4" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Box.area") (fun () ->
              V.Box.area (Nx.zeros Nx.float32 [| 3 |])));
    ]

let nms_refusals =
  group "misuse of nms"
    [
      test "scores that are not one a box" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Box.nms") (fun () ->
              V.Box.nms ~iou:0.5
                (Nx.zeros Nx.float32 [| 3; 4 |])
                (Nx.zeros Nx.float32 [| 2 |])));
      test "boxes end in 4" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Box.nms") (fun () ->
              V.Box.nms ~iou:0.5
                (Nx.zeros Nx.float32 [| 3; 5 |])
                (Nx.zeros Nx.float32 [| 3 |])));
    ]

let () =
  exit
    (run "mynd.box"
       [
         conversions;
         areas;
         iou_golden;
         iou_law;
         iou_batches;
         empty_iou;
         of_mask;
         of_mask_golden;
         of_mask_empty;
         moved_cells;
         warp_law;
         exact_moves;
         neg_law;
         compose_law;
         axis_law;
         moved_empties;
         moved_nans;
         infinite_shift;
         transform_shapes;
         moved_dtypes;
         transform_grads;
         finite_grads;
         tie_grads;
         giou_cases;
         giou_exact;
         giou_golden;
         giou_range;
         giou_symmetric;
         giou_scaled;
         giou_continuous;
         giou_shapes;
         giou_grads;
         giou_finite;
         giou_widens;
         giou_shrinks;
         giou_dtypes;
         giou_overflow;
         golden_nms;
         greedy_law;
         signed_zeros;
         threshold;
         rounds_nms;
         batches_nms;
         compiled_nms;
         refusals;
         nms_refusals;
       ])
