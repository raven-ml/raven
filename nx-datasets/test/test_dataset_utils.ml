open Alcotest
open Unix

let test_cache_dir_resolution () =
  let original_nx = Sys.getenv_opt "NX_DATASETS_CACHE" in
  let original_xdg = Sys.getenv_opt "XDG_CACHE_HOME" in

  Fun.protect
    (fun () ->
      (* Test NX_DATASETS_CACHE priority *)
      putenv "NX_DATASETS_CACHE" "/tmp/nx-cache";
      putenv "XDG_CACHE_HOME" "/tmp/xdg-cache";
      let path1 = Nx_datasets.get_cache_dir "iris" in
      check string "NX_DATASETS_CACHE takes priority"
        "/tmp/nx-cache/ocaml-nx/datasets/iris/" path1;

      (* Test XDG_CACHE_HOME fallback *)
      putenv "NX_DATASETS_CACHE" "";
      putenv "XDG_CACHE_HOME" "/tmp/xdg-cache";
      let path2 = Nx_datasets.get_cache_dir "mnist" in
      check string "XDG_CACHE_HOME used when NX_DATASETS_CACHE unset"
        "/tmp/xdg-cache/ocaml-nx/datasets/mnist/" path2;
      
      (* Test HOME fallback *)
      putenv "NX_DATASETS_CACHE" "";
      putenv "XDG_CACHE_HOME" "";
      let home = Sys.getenv "HOME" in
      let expected = Filename.concat home ".cache/ocaml-nx/datasets/cifar10/" in
      let path3 = Nx_datasets.get_cache_dir "cifar10" in
      check string "Falls back to HOME/.cache when no env vars set"
        expected path3)
    ~finally:(fun () ->
      (* Restore original environment variables *)
      (match original_nx with
       | Some v -> putenv "NX_DATASETS_CACHE" v
       | None -> putenv "NX_DATASETS_CACHE" "");
      (match original_xdg with
       | Some v -> putenv "XDG_CACHE_HOME" v
       | None -> putenv "XDG_CACHE_HOME" ""))

let () =
  run "Dataset Utils"
    [
      ( "Cache Directory Resolution",
        [
          test_case "Environment variable precedence" `Quick test_cache_dir_resolution;
        ] );
    ]
