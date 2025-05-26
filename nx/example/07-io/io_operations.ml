open Nx
open Nx_io

let () =
  (* Create test directory *)
  let test_dir = "test_data" in
  (try Unix.mkdir test_dir 0o755
   with Unix.Unix_error (Unix.EEXIST, _, _) -> ());

  Printf.printf "=== Nx I/O Operations Example ===\n\n";

  (* 1. NPY files *)
  Printf.printf "1. NPY Format:\n";

  (* Save a float32 array *)
  let arr_f32 = arange Float32 0 12 1 |> reshape [| 3; 4 |] in
  let npy_path = Filename.concat test_dir "example.npy" in
  save_npy arr_f32 npy_path;
  Printf.printf "   Saved array to %s\n" npy_path;

  (* Load it back *)
  let loaded = load_npy npy_path in
  let arr_loaded = to_float32 loaded in
  Format.printf "   Loaded array (shape: %a):\n%a\n\n" pp_shape
    (shape arr_loaded) pp arr_loaded;

  (* 2. NPZ archives *)
  Printf.printf "2. NPZ Format (compressed archives):\n";

  (* Create multiple arrays *)
  let counts = arange Int32 0 5 1 in
  let matrix = full Float64 [| 2; 3 |] 3.14 in

  (* Save as NPZ *)
  let npz_path = Filename.concat test_dir "archive.npz" in
  save_npz [ ("counts", P counts); ("matrix", P matrix) ] npz_path;
  Printf.printf "   Saved archive to %s\n" npz_path;

  (* Load entire archive *)
  let archive = load_npz npz_path in
  Printf.printf "   Archive contains %d arrays:\n" (Hashtbl.length archive);
  Hashtbl.iter
    (fun name (P arr) ->
      Format.printf "     - %s: %a %a\n" name pp_dtype (dtype arr) pp_shape
        (shape arr))
    archive;

  (* Load single member *)
  let (P matrix_loaded) = load_npz_member ~path:npz_path ~name:"matrix" in
  let matrix_f64 = to_float64 (P matrix_loaded) in
  Format.printf "   Loaded 'matrix':\n%a\n\n" pp matrix_f64;

  (* 3. Image files *)
  Printf.printf "3. Image Format:\n";
  Printf.printf "   (Requires an image file to demonstrate)\n";

  (* Create a simple grayscale image *)
  let img_gray =
    create UInt8 [| 100; 100 |] (Array.init 10000 (fun i -> i mod 256))
  in
  let img_path = Filename.concat test_dir "gradient.png" in
  save_image img_gray img_path;
  Printf.printf "   Saved grayscale gradient to %s\n" img_path;

  (* Create a simple color image (RGB) *)
  let img_color = zeros UInt8 [| 100; 100; 3 |] in
  (* Make a red square in the center *)
  let red_value = scalar UInt8 255 in
  for i = 30 to 70 do
    for j = 30 to 70 do
      set [ i; j; 0 ] img_color red_value (* Red channel *)
    done
  done;
  let color_path = Filename.concat test_dir "red_square.png" in
  save_image img_color color_path;
  Printf.printf "   Saved color image to %s\n" color_path;

  Printf.printf "\nAll test files saved to '%s/'\n" test_dir
