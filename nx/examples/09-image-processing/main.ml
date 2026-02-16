(** Load, transform, and save images as arrays — convolutions, pooling, and
    pixel math.

    Create a synthetic grayscale gradient, blur it, detect edges with Sobel
    filters, and downsample with max pooling. Results are saved as PNG files. *)

open Nx

let () =
  let h = 64 and w = 64 in

  (* --- Create a gradient image with a bright rectangle --- *)
  let img =
    init UInt8 [| h; w |] (fun idx ->
        let y = idx.(0) and x = idx.(1) in
        (* Background: horizontal gradient. *)
        let base = x * 255 / (w - 1) in
        (* Bright rectangle in the center. *)
        if y >= 16 && y < 48 && x >= 16 && x < 48 then 220 else base)
  in
  Printf.printf "Created %dx%d grayscale image\n" h w;

  (* Save the original. *)
  Nx_io.save_image "gradient.png" (contiguous img);
  Printf.printf "Saved: gradient.png\n";

  (* --- Gaussian blur with a 3×3 kernel --- *)

  (* Convert to float for convolution: correlate2d expects [N;C;H;W]. *)
  let img_f = cast Float32 img |> contiguous |> reshape [| 1; 1; h; w |] in

  let blur_kernel =
    create Float32 [| 1; 1; 3; 3 |]
      [|
        1.0 /. 16.0;
        2.0 /. 16.0;
        1.0 /. 16.0;
        2.0 /. 16.0;
        4.0 /. 16.0;
        2.0 /. 16.0;
        1.0 /. 16.0;
        2.0 /. 16.0;
        1.0 /. 16.0;
      |]
  in
  let blurred = correlate2d ~padding_mode:`Same img_f blur_kernel in
  let blurred_img =
    reshape [| h; w |] blurred
    |> clamp ~min:0.0 ~max:255.0 |> cast UInt8 |> contiguous
  in
  Nx_io.save_image "blurred.png" blurred_img;
  Printf.printf "Saved: blurred.png\n";

  (* --- Sobel edge detection --- *)
  let sobel_x =
    create Float32 [| 1; 1; 3; 3 |]
      [| -1.0; 0.0; 1.0; -2.0; 0.0; 2.0; -1.0; 0.0; 1.0 |]
  in
  let sobel_y =
    create Float32 [| 1; 1; 3; 3 |]
      [| -1.0; -2.0; -1.0; 0.0; 0.0; 0.0; 1.0; 2.0; 1.0 |]
  in

  let gx = correlate2d ~padding_mode:`Same img_f sobel_x in
  let gy = correlate2d ~padding_mode:`Same img_f sobel_y in
  let edges = sqrt (add (mul gx gx) (mul gy gy)) in
  let edge_h = (shape edges).(2) and edge_w = (shape edges).(3) in
  let edges_img =
    reshape [| edge_h; edge_w |] edges
    |> clamp ~min:0.0 ~max:255.0 |> cast UInt8 |> contiguous
  in
  Nx_io.save_image "edges.png" edges_img;
  Printf.printf "Saved: edges.png\n";

  (* --- Max pooling: 2× downsample --- *)
  let pooled, _indices = max_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) img_f in
  let pool_h = (shape pooled).(2) and pool_w = (shape pooled).(3) in
  let pooled_img =
    reshape [| pool_h; pool_w |] pooled
    |> clamp ~min:0.0 ~max:255.0 |> cast UInt8 |> contiguous
  in
  Nx_io.save_image "pooled.png" pooled_img;
  Printf.printf "Saved: pooled.png (%dx%d → %dx%d)\n" h w pool_h pool_w;

  Printf.printf "\nAll images saved to the current directory.\n"
