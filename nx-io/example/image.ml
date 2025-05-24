open Nx
open Nx_io

let output_dir = "test_data"

let () =
  if Array.length Sys.argv < 2 then (
    Printf.printf "Usage: %s <image_path>\n" Sys.executable_name;
    exit 1);

  let img_path = Sys.argv.(1) in

  (* Load color image (requires uint8_t) *)
  let () =
    let img_color = load_image img_path in
    Format.printf "Loaded color image '%s' (shape: %a)\n" img_path pp_shape
      (shape img_color)
  in

  (* Load grayscale image *)
  let img_gray = load_image ~grayscale:true img_path in
  Format.printf "Loaded grayscale image '%s' (shape: %a)\n" img_path pp_shape
    (shape img_gray);

  (* Save the grayscale image *)
  let gray_save_path = Filename.concat output_dir "lena_grayscale.png" in
  save_image img_gray gray_save_path;
  Printf.printf "Saved grayscale image to: %s\n" gray_save_path
