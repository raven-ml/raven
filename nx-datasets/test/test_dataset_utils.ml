open Windtrap

let build_expected base dataset =
  let path = List.fold_left Filename.concat base [ "datasets"; dataset ] in
  let sep = Filename.dir_sep.[0] in
  if path <> "" && path.[String.length path - 1] = sep then path
  else path ^ Filename.dir_sep

let getenv_of_list env var = List.assoc_opt var env

let test_cache_dir_resolution () =
  let temp_dir = Filename.get_temp_dir_name () in
  let home_dir = Filename.concat temp_dir "home-dir" in
  let custom_cache_dir = Filename.concat temp_dir "nx-cache" in
  let xdg_cache_dir = Filename.concat temp_dir "xdg-cache" in
  let base_env = [ ("HOME", home_dir); ("USERPROFILE", home_dir) ] in

  (* RAVEN_CACHE_ROOT has highest priority *)
  let env_with_custom =
    ("RAVEN_CACHE_ROOT", custom_cache_dir)
    :: ("XDG_CACHE_HOME", xdg_cache_dir)
    :: base_env
  in
  let path1 =
    Nx_datasets.get_cache_dir ~getenv:(getenv_of_list env_with_custom) "iris"
  in
  let expected1 = build_expected custom_cache_dir "iris" in
  equal ~msg:"RAVEN_CACHE_ROOT takes priority" string expected1 path1;

  (* XDG_CACHE_HOME is used when RAVEN_CACHE_ROOT is unset or empty *)
  let env_with_xdg =
    ("RAVEN_CACHE_ROOT", "") :: ("XDG_CACHE_HOME", xdg_cache_dir) :: base_env
  in
  let path2 =
    Nx_datasets.get_cache_dir ~getenv:(getenv_of_list env_with_xdg) "mnist"
  in
  let expected2 =
    build_expected (Filename.concat xdg_cache_dir "raven") "mnist"
  in
  equal ~msg:"XDG_CACHE_HOME used when RAVEN_CACHE_ROOT unset" string expected2 path2;

  (* HOME fallback when neither cache env var is provided *)
  let env_with_home_only =
    ("RAVEN_CACHE_ROOT", "") :: ("XDG_CACHE_HOME", "") :: base_env
  in
  let path3 =
    Nx_datasets.get_cache_dir
      ~getenv:(getenv_of_list env_with_home_only)
      "cifar10"
  in
  let home_cache =
    Filename.concat (Filename.concat home_dir ".cache") "raven"
  in
  let expected3 = build_expected home_cache "cifar10" in
  equal ~msg:"Falls back to HOME/.cache when no env vars set" string expected3 path3

let () =
  run "Dataset Utils"
    [
      group "Cache Directory Resolution"
        [
          test "Environment variable precedence"
            test_cache_dir_resolution;
        ];
    ]
