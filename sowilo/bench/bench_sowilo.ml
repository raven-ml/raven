(* Sowilo image processing benchmarks using synthetic PNG fixtures. *)

module Fixtures = struct
  let data_dir = Filename.concat (Sys.getcwd ()) "sowilo/bench/data"

  let load_image name =
    let path = Filename.concat data_dir name in
    let nx_img = Nx_io.load_image path in
    Rune.of_buffer (Nx.to_buffer nx_img)

  let img_1080 = lazy (load_image "img_1920x1080.png")
  let img_720 = lazy (load_image "img_1280x720.png")
  let gray_1080 = lazy (Sowilo.to_grayscale (Lazy.force img_1080))
  let gray_720 = lazy (Sowilo.to_grayscale (Lazy.force img_720))
  let img_1080 () = Lazy.force img_1080
  let gray_1080 () = Lazy.force gray_1080
  let gray_720 () = Lazy.force gray_720
end

let force_tensor tensor = ignore (Rune.to_buffer tensor)

let bench_grayscale img =
  let gray = Sowilo.to_grayscale img in
  force_tensor gray

let bench_gaussian img =
  let blurred = Sowilo.gaussian_blur ~ksize:(5, 5) ~sigmaX:1.2 img in
  force_tensor blurred

let bench_sobel img =
  let sobel_x = Sowilo.sobel ~dx:1 ~dy:0 img in
  force_tensor sobel_x

let bench_canny img =
  let edges = Sowilo.canny ~threshold1:55. ~threshold2:120. img in
  force_tensor edges

let all_benchmarks =
  let color_1080 = Fixtures.img_1080 () in
  let gray_1080 = Fixtures.gray_1080 () in
  let gray_720 = Fixtures.gray_720 () in
  [
    Ubench.bench "ToGrayscale/1080p" (fun () -> bench_grayscale color_1080);
    Ubench.bench "GaussianBlur/1080p" (fun () -> bench_gaussian color_1080);
    Ubench.bench "Sobel/720p" (fun () -> bench_sobel gray_720);
    Ubench.bench "Canny/1080p" (fun () -> bench_canny gray_1080);
  ]
  |> fun benches -> [ Ubench.group "Sowilo" benches ]

let () = Ubench.run_cli all_benchmarks
