open Hugin

let () =
  if Array.length Sys.argv < 2 then (
    Printf.printf "Usage: %s <image_path>\n" Sys.executable_name;
    exit 1);

  let image_path = Sys.argv.(1) in
  let img = Ndarray_io.load_image image_path in
  let fig = imshow ~title:"Image" img in
  show fig
