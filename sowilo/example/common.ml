(* common.ml *)
open Rune

(* Alias plotting modules for brevity *)
module Plot = Hugin.Plotting
module Ax = Hugin.Axes
module Art = Hugin.Artist

(* Change this to your image path *)
let image_path = "sowilo/example/lena.png"

(* Helper to load image as Rune tensor *)
let load_image_rune path =
  let nx_img = Nx_io.load_image path in
  Rune.of_bigarray Rune.ocaml (Nx.to_bigarray nx_img)

(* Helper to convert Rune tensor to Nx tensor for plotting *)
let rune_to_nx (rune_tensor : ('a, 'b, 'dev) Rune.t) : ('a, 'b) Nx.t =
  let data = Rune.unsafe_to_bigarray rune_tensor in
  let nx_tensor = Nx.of_bigarray data in
  let shape = Rune.shape rune_tensor in
  Nx.reshape shape nx_tensor

(* Helper to visualize Sobel output (int16) as uint8 *)
let visualize_sobel (sobel_img : 'dev int16_t) : 'dev uint8_t =
  (* Calculate absolute values *)
  let abs_sobel = Rune.abs sobel_img in

  (* Use astype and Nx reductions for min/max *)
  let abs_sobel_f = Rune.astype Rune.float32 abs_sobel in
  let min_val_t = Rune.min ~keepdims:false abs_sobel_f in
  let max_val_t = Rune.max ~keepdims:false abs_sobel_f in

  (* Check if tensors are actually scalars before getting item *)
  let min_val =
    if Rune.ndim min_val_t = 0 then Rune.unsafe_get [] min_val_t else 0.0
    (* Default or error *)
  in
  let max_val =
    if Rune.ndim max_val_t = 0 then Rune.unsafe_get [] max_val_t else 255.0
    (* Default or error *)
  in

  let range = max_val -. min_val in
  if range <= 1e-6 then (* Avoid division by zero if image is flat *)
    Rune.zeros Rune.ocaml Rune.uint8 (Rune.shape sobel_img)
  else
    (* Scale to 0.0 - 1.0 *)
    let range_scalar = Rune.scalar Rune.ocaml Rune.float32 range in
    let min_scalar = Rune.scalar Rune.ocaml Rune.float32 min_val in
    let scaled_f = Rune.div (Rune.sub abs_sobel_f min_scalar) range_scalar in
    (* Convert to uint8 [0, 255] *)
    Sowilo.to_uint8 scaled_f

(* Simple timing function *)
let time_it name f =
  Printf.printf "Running %s...\n%!" name;
  let t1 = Unix.gettimeofday () in
  let result = f () in
  let t2 = Unix.gettimeofday () in
  Printf.printf "%s took: %.4f seconds\n%!" name (t2 -. t1);
  result

(* Plotting helper for single image *)
let plot_single title img ?(cmap = Art.Colormap.viridis) () =
  let fig = Hugin.figure () in
  let ax = Hugin.subplot ~nrows:1 ~ncols:1 ~index:1 fig in
  let nx_img = rune_to_nx img in
  ignore
    (ax
    |> Plot.imshow ~data:nx_img ~cmap
    |> Ax.set_title title |> Ax.set_xticks [] |> Ax.set_yticks []);
  Hugin.show fig;
  print_endline "Plot window closed."

(* Plotting helper for comparing two images *)
let plot_compare ?(cmap1 = Art.Colormap.viridis) ?(cmap2 = Art.Colormap.viridis)
    title1 img1 title2 img2 =
  let fig = Hugin.figure ~width:1000 ~height:500 () in
  let ax1 = Hugin.subplot ~nrows:1 ~ncols:2 ~index:1 fig in
  let nx_img1 = rune_to_nx img1 in
  let nx_img2 = rune_to_nx img2 in
  ignore
    (ax1
    |> Plot.imshow ~data:nx_img1 ~cmap:cmap1
    |> Ax.set_title title1 |> Ax.set_xticks [] |> Ax.set_yticks []);
  let ax2 = Hugin.subplot ~nrows:1 ~ncols:2 ~index:2 fig in
  ignore
    (ax2
    |> Plot.imshow ~data:nx_img2 ~cmap:cmap2
    |> Ax.set_title title2 |> Ax.set_xticks [] |> Ax.set_yticks []);
  Hugin.show fig;
  print_endline "Plot window closed."
