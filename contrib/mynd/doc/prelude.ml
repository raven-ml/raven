(* The pages read and write image files by name. These shadow nx.io's
   [load_image] and [save_png], and only those, so that their names resolve
   in a fresh temporary directory removed when the run ends, and the pages
   write nothing into the build tree. A page that calls another nx.io reader
   or writer, such as [save_jpeg], must be redirected here too. A change of
   directory would break the toplevel's relative paths to its .cmi files. *)
module Nx_io = struct
  include Nx_io

  let dir = Filename.temp_dir "mynd-doc" ""

  let () =
    at_exit (fun () ->
        Array.iter
          (fun f -> Sys.remove (Filename.concat dir f))
          (Sys.readdir dir);
        Sys.rmdir dir)

  let load_image path = load_image (Filename.concat dir path)

  let save_png ?dpi ?srgb path x =
    save_png ?dpi ?srgb (Filename.concat dir path) x
end
