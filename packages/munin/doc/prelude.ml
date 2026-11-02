(* Each guide runs in its own process, against a store in a fresh temporary
   directory that is removed at exit, and its examples' temporary files go to
   the same directory. *)

let rec rm_rf path =
  if Sys.is_directory path then begin
    Array.iter (fun e -> rm_rf (Filename.concat path e)) (Sys.readdir path);
    Sys.rmdir path
  end
  else Sys.remove path

let () =
  let dir = Filename.temp_dir "munin-doc" "" in
  at_exit (fun () -> rm_rf dir);
  let tmp = Filename.concat dir "tmp" in
  Sys.mkdir tmp 0o700;
  Filename.set_temp_dir_name tmp;
  Unix.putenv "RAVEN_TRACKING_DIR" (Filename.concat dir "store")
