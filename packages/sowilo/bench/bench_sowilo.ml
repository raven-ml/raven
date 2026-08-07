(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sowilo image processing benchmarks using synthetic PNG fixtures. *)

module Fixtures = struct
  (* Resolve fixtures next to the executable, not the working directory: the
     bench rule runs with the bench dir as cwd, dune exec with the project root
     — the exe path is the one stable anchor in both. *)
  let data_dir = Filename.concat (Filename.dirname Sys.executable_name) "data"

  let load_image name =
    let path = Filename.concat data_dir name in
    let nx_img = Nx_io.load_image path in
    Sowilo.to_float nx_img

  let img_1080 = lazy (load_image "img_1920x1080.png")
  let img_720 = lazy (load_image "img_1280x720.png")
  let gray_1080 = lazy (Sowilo.to_grayscale (Lazy.force img_1080))
  let gray_720 = lazy (Sowilo.to_grayscale (Lazy.force img_720))
  let img_1080 () = Lazy.force img_1080
  let gray_1080 () = Lazy.force gray_1080
  let gray_720 () = Lazy.force gray_720
end

let force_tensor tensor = Nx.to_buffer tensor

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

let () = Thumper.run "sowilo" all_benchmarks
