(* The guides' examples log to a store in a fresh temporary directory, removed
   at exit, and read the files they log from the build directory, where this
   prelude writes them. Each guide runs in its own process, possibly at the
   same time as the others: files are written whole and renamed into place, so
   a reader never sees one half written. *)

let rec rm_rf path =
  if Sys.is_directory path then begin
    Array.iter (fun e -> rm_rf (Filename.concat path e)) (Sys.readdir path);
    Sys.rmdir path
  end
  else Sys.remove path

let write path text =
  let temp_dir = Filename.dirname path in
  let tmp = Filename.temp_file ~temp_dir "doc" ".tmp" in
  let oc = open_out_bin tmp in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc text);
  Sys.rename tmp path

let () =
  let store = Filename.temp_dir "munin-doc" "" in
  at_exit (fun () -> rm_rf store);
  Unix.putenv "RAVEN_TRACKING_DIR" store;
  (try Sys.mkdir "data" 0o755
   with Sys_error _ when Sys.is_directory "data" -> ());
  write "data/measurements.csv" "x,y\n1.0,2.0\n";
  write "model_epoch1.safetensors" "epoch 1 weights";
  write "model_epoch2.safetensors" "epoch 2 weights";
  write "config.yaml" "lr: 0.001\n";
  write "confusion_matrix.png" "png";
  write "features.csv" "a,b\n1,2\n"
