(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sowilo image processing benchmarks over synthetic images. *)

module Fixtures = struct
  (* [image ~width ~height ~seed palette] is the RGB frame
     scripts/generate_fixtures.py draws, a palette of the radius and angle
     over [[-1, 1]²] plus Gaussian noise of deviation 0.05, clipped to [[0, 1]]
     and truncated to uint8. Its noise comes from Nx's generator seeded with
     [seed], numpy's in the script: the same distribution, other values. *)
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

  let img_1080 =
    lazy (Sowilo.to_float (image ~width:1920 ~height:1080 ~seed:7 `Sunset))

  let img_720 =
    lazy (Sowilo.to_float (image ~width:1280 ~height:720 ~seed:19 `Forest))

  let gray_1080 = lazy (Sowilo.to_grayscale (Lazy.force img_1080))
  let gray_720 = lazy (Sowilo.to_grayscale (Lazy.force img_720))
  let img_1080 () = Lazy.force img_1080
  let gray_1080 () = Lazy.force gray_1080
  let gray_720 () = Lazy.force gray_720
end

(* A case's work is done once the tensor's set has computed it. *)
let force_tensor tensor =
  Nx.sync tensor;
  tensor

let bench_grayscale img =
  let gray = Sowilo.to_grayscale img in
  force_tensor gray

let bench_gaussian img =
  let blurred = Sowilo.gaussian_blur ~sigma:1.2 ~ksize:5 img in
  force_tensor blurred

let bench_sobel img =
  let gx, _gy = Sowilo.sobel img in
  force_tensor gx

let bench_canny img =
  let edges = Sowilo.canny ~low:0.2 ~high:0.6 img in
  force_tensor edges

(* Fixtures are forced in each case's setup, inside thumper's forked worker —
   never in the parent: building a tensor spawns the backend's thread pool, and
   a pool created before the fork leaves the child dispatching onto threads that
   no longer exist. *)
let all_benchmarks =
  [
    Thumper.bench_with_setup ~setup:Fixtures.img_1080 "ToGrayscale/1080p"
      bench_grayscale;
    Thumper.bench_with_setup ~setup:Fixtures.img_1080 "GaussianBlur/1080p"
      bench_gaussian;
    Thumper.bench_with_setup ~setup:Fixtures.gray_720 "Sobel/720p" bench_sobel;
    Thumper.bench_with_setup ~setup:Fixtures.gray_1080 "Canny/1080p" bench_canny;
  ]
  |> fun benches -> [ Thumper.group "Sowilo" benches ]

let () = Thumper.run "sowilo" all_benchmarks |> exit
