(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* mynd's rows over synthetic frames.

   ToGrayscale, GaussianBlur, Sobel and Canny time the call that replaces
   sowilo's of the same row (CHANGES.md's table), on sowilo's frames, so their
   rows compare with sowilo's. The other rows time what sowilo lacked. *)

open Mynd

module Fixtures = struct
  (* [image ~width ~height ~seed palette] is the RGB frame bench_mynd.py draws,
     a palette of the radius and angle over [[-1, 1]²] plus Gaussian noise of
     deviation 0.05, clipped to [[0, 1]] and truncated to uint8. Its noise comes
     from Nx's generator seeded with [seed], numpy's in the script: the same
     distribution, other values. *)
  let image ~width ~height ~seed palette =
    let open Nx in
    let x = linspace float32 (-1.) 1. width
    and y = linspace float32 (-1.) 1. height in
    let xv, yv =
      match meshgrid ~indexing:`Xy [ x; y ] with
      | [ xv; yv ] -> (xv, yv)
      | _ -> assert false
    in
    let radial = sqrt (add (square xv) (square yv)) and angle = atan2 yv xv in
    let r, g, b =
      match palette with
      | `Sunset ->
          ( add_s (mul_s (cos angle) 0.4) 0.6,
            add_s (mul_s (sin (mul_s radial Float.pi)) 0.35) 0.4,
            add_s (mul_s (exp (mul_s radial (-1.5))) 0.6) 0.2 )
      | `Forest ->
          ( add_s (mul_s (exp (mul_s radial (-2.))) 0.3) 0.2,
            add_s (mul_s (cos (mul_s angle 2.)) 0.5) 0.4,
            add_s (mul_s (sin (mul_s radial Float.pi)) 0.4) 0.3 )
    in
    let noise =
      mul_s (Rng.normal ~key:(Rng.key seed) float32 [| height; width |]) 0.05
    in
    let channel c =
      cast uint8 (mul_s (clamp ~min:0. ~max:1. (add c noise)) 255.)
    in
    stack ~axis:(-1) [ channel r; channel g; channel b ]

  let img_1080 () =
    convert Nx.float32 (image ~width:1920 ~height:1080 ~seed:7 `Sunset)

  let img_720 () =
    convert Nx.float32 (image ~width:1280 ~height:720 ~seed:19 `Forest)

  let gray_1080 () = Colour.luma (img_1080 ())
  let gray_720 () = Colour.luma (img_720 ())

  (* The 720p frame's positions under a turn of 10° about its centre: each
     output centre's place in the input. *)
  let turned_720 () =
    let img = img_720 () in
    let h = Nx.dim 0 img and w = Nx.dim 1 img in
    let f32 = Nx.float32 in
    let about =
      Nx.create f32 [| 2 |] [| Float.of_int w /. 2.; Float.of_int h /. 2. |]
    in
    let turn = Transform.rotation (Nx.scalar f32 (10. *. Float.pi /. 180.)) in
    let m =
      Nx.matmul
        (Transform.translation about)
        (Nx.matmul turn (Transform.translation (Nx.neg about)))
    in
    (img, Transform.apply (Nx.inv m) (Sample.centres f32 (h, w)))

  (* [objects] objects of a 1080p frame, each detected by [per] boxes jittered
     by up to a tenth of its size with scores in [[0, 1)]: a detector's output
     before suppression, [[objects per; 4]] and [[objects per]]. *)
  let detections ~objects ~per () =
    let s = Random.State.make [| 3 |] in
    let n = objects * per in
    let boxes = Array.make (4 * n) 0. and scores = Array.make n 0. in
    for o = 0 to objects - 1 do
      let cx = Random.State.float s 1920. and cy = Random.State.float s 1080. in
      let bw = 40. +. Random.State.float s 200.
      and bh = 40. +. Random.State.float s 200. in
      for k = 0 to per - 1 do
        let i = (o * per) + k in
        let jitter d = Random.State.float s (0.2 *. d) -. (0.1 *. d) in
        let x = cx +. jitter bw and y = cy +. jitter bh in
        let w = bw +. jitter bw and h = bh +. jitter bh in
        boxes.(4 * i) <- x -. (w /. 2.);
        boxes.((4 * i) + 1) <- y -. (h /. 2.);
        boxes.((4 * i) + 2) <- x +. (w /. 2.);
        boxes.((4 * i) + 3) <- y +. (h /. 2.);
        scores.(i) <- Random.State.float s 1.
      done
    done;
    (Nx.create Nx.float32 [| n; 4 |] boxes, Nx.create Nx.float32 [| n |] scores)

  let rectangle k = Nx.ones Nx.bool [| k; k |]

  let disk r =
    let k = (2 * r) + 1 in
    Nx.init Nx.bool [| k; k |] (fun i ->
        let dy = i.(0) - r and dx = i.(1) - r in
        (dy * dy) + (dx * dx) <= r * r)
end

(* A case's work is done once the tensor's set has computed it. *)
let force tensor =
  Nx.sync tensor;
  tensor

let f32 = Nx.scalar Nx.float32

(* sowilo's rows *)

let grayscale img = force (Colour.luma img)

(* sowilo's [gaussian_blur ~sigma:1.2 ~ksize:5]. *)
let gaussian (sigma, img) =
  force (Filter.gaussian ~edge:`Zero ~radius:2 sigma img)

(* sowilo's [sobel]: its two gradients. *)
let sobel gray =
  let gx, gy = Filter.gradient ~edge:`Zero `Sobel gray in
  ignore (force gx);
  force gy

(* sowilo's [canny ~low:0.2 ~high:0.6] at its default deviation 1.4, whose blur
   took 2⌈3σ⌉ + 1 taps: the guide's composition (doc/edges.md) with `Sobel,
   whose strengths are sowilo's over 8. *)
let canny (sigma, gray) =
  let edge = `Symmetric_whole in
  let b = Filter.gaussian ~edge ~radius:5 sigma gray in
  let ((gx, gy) as g) = Filter.gradient ~edge `Sobel b in
  let m = Nx.hypot gx gy in
  let candidates =
    Nx.logical_and (Filter.gradient_maxima g) (Nx.greater_s m (0.2 /. 8.))
  in
  let strong = Nx.greater_s m (0.6 /. 8.) in
  force (Mask.reconstruct `Eight ~marker:strong candidates)

(* mynd's own rows *)

let resize kernel img =
  force (Sample.resize ~edge:`Replicate kernel (720, 1280) img)

let sample_at (img, p) = force (Sample.at ~edge:`Replicate `Triangle img p)
let components mask = force (Mask.components `Eight mask)
let nms (boxes, scores) = force (Box.nms ~iou:0.5 boxes scores)
let median k gray = force (Filter.median ~edge:`Zero (k, k) gray)
let erode (fp, gray) = force (Filter.erode fp gray)

(* Fixtures are built in each case's setup, inside thumper's forked worker,
   never in the parent: building a tensor spawns the backend's thread pool, and
   a pool created before the fork leaves the child dispatching onto threads that
   no longer exist. *)
let benches =
  let open Fixtures in
  let case = Thumper.bench_with_setup in
  [
    case ~setup:img_1080 "ToGrayscale/1080p" grayscale;
    case ~setup:(fun () -> (f32 1.2, img_1080 ())) "GaussianBlur/1080p" gaussian;
    case ~setup:gray_720 "Sobel/720p" sobel;
    case ~setup:(fun () -> (f32 1.4, gray_1080 ())) "Canny/1080p" canny;
    case ~setup:img_1080 "Resize/Box/1080p" (resize `Box);
    case ~setup:img_1080 "Resize/Triangle/1080p" (resize `Triangle);
    case ~setup:img_1080 "Resize/Cubic/1080p" (resize (`Cubic (-0.5)));
    case ~setup:img_1080 "Resize/Lanczos3/1080p" (resize (`Lanczos 3));
    case ~setup:turned_720 "SampleAt/Triangle/720p" sample_at;
    case
      ~setup:(fun () -> Nx.greater_s (gray_720 ()) 0.5)
      "Components/720p" components;
    case ~setup:(detections ~objects:96 ~per:32) "NMS/3072" nms;
    case ~setup:gray_720 "Median/3x3/720p" (median 3);
    case ~setup:gray_720 "Median/5x5/720p" (median 5);
    case ~setup:(fun () -> (rectangle 3, gray_720 ())) "Erode/3x3/720p" erode;
    case ~setup:(fun () -> (disk 2, gray_720 ())) "Erode/Disk2/720p" erode;
  ]

let () = exit (Thumper.run "mynd" [ Thumper.group "Mynd" benches ])
