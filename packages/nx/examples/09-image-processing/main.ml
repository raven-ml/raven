(** Load, transform, and save images as arrays: convolutions, pooling, and pixel
    math.

    Create a synthetic grayscale gradient, blur it, detect edges with Sobel
    filters, and downsample with max pooling. Results are saved as PNG files. *)

(* Save a float [h; w] array as a grey PNG. An image has the shape [h; w;
   channels], so the pixels gain a channel axis of extent 1. *)
let save path x =
  let pixels = Nx.clamp ~min:0.0 ~max:255.0 x |> Nx.cast Nx.uint8 in
  Nx_io.save_png path (Nx.unsqueeze ~axes:[ 2 ] pixels);
  Printf.printf "Saved: %s\n" path

let () =
  let h = 64 and w = 64 in

  (* --- Create a gradient image with a bright rectangle --- *)
  let img =
    Nx.init Nx.uint8 [| h; w |] (fun idx ->
        let y = idx.(0) and x = idx.(1) in
        (* Background: horizontal gradient. *)
        let base = x * 255 / (w - 1) in
        (* Bright rectangle in the center. *)
        if y >= 16 && y < 48 && x >= 16 && x < 48 then 220 else base)
  in
  Printf.printf "Created %dx%d grayscale image\n" h w;

  (* Convert to float for convolution. [correlate] works over the kernel's
     axes, the last two here, so the image stays [h; w]. *)
  let img_f = Nx.cast Nx.float32 img in
  save "gradient.png" img_f;

  (* --- Gaussian blur with a 3x3 kernel --- *)
  let blur_kernel =
    Nx.create Nx.float32 [| 3; 3 |]
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
  let blurred = Nx.correlate ~padding:`Same img_f blur_kernel in
  save "blurred.png" blurred;

  (* --- Sobel edge detection --- *)
  let sobel_x =
    Nx.create Nx.float32 [| 3; 3 |]
      [| -1.0; 0.0; 1.0; -2.0; 0.0; 2.0; -1.0; 0.0; 1.0 |]
  in
  let sobel_y =
    Nx.create Nx.float32 [| 3; 3 |]
      [| -1.0; -2.0; -1.0; 0.0; 0.0; 0.0; 1.0; 2.0; 1.0 |]
  in
  let gx = Nx.correlate ~padding:`Same img_f sobel_x in
  let gy = Nx.correlate ~padding:`Same img_f sobel_y in
  let edges = Nx.(Infix.(sqrt ((gx * gx) + (gy * gy)))) in
  save "edges.png" edges;

  (* --- Max pooling: 2x downsample ---

     [sliding_window] views each axis as windows of 2 elements, one every 2:
     [h; w] becomes [h/2; w; 2], then [h/2; w/2; 2; 2]. The maximum over the
     two window axes is each 2x2 block's. *)
  let pooled =
    img_f
    |> Nx.sliding_window ~axis:0 ~window:2 ~step:2
    |> Nx.sliding_window ~axis:1 ~window:2 ~step:2
    |> Nx.max ~axes:[ 2; 3 ]
  in
  save "pooled.png" pooled;
  Printf.printf "Pooled: %dx%d -> %dx%d\n" h w (Nx.dim 0 pooled)
    (Nx.dim 1 pooled);

  Printf.printf "\nAll images saved to the current directory.\n"
