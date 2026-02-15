open Windtrap
open Nx_datasets

let check_shape name expected_shape tensor =
  let actual_shape = Nx.shape tensor in
  equal ~msg:name (array int) expected_shape actual_shape

(* ───── Generators ───── *)

let test_make_blobs () =
  let x, y = make_blobs ~n_samples:100 ~n_features:2 ~centers:(`N 3) () in
  check_shape "X shape" [| 100; 2 |] x;
  check_shape "y shape" [| 100 |] y

let test_make_classification () =
  let x, y =
    make_classification ~n_samples:100 ~n_features:20 ~n_classes:3 ()
  in
  check_shape "X shape" [| 100; 20 |] x;
  check_shape "y shape" [| 100 |] y

let test_make_gaussian_quantiles () =
  let x, y = make_gaussian_quantiles ~n_samples:100 ~n_features:2 () in
  check_shape "X shape" [| 100; 2 |] x;
  check_shape "y shape" [| 100 |] y

let test_make_circles () =
  let x, y = make_circles ~n_samples:100 () in
  check_shape "X shape" [| 100; 2 |] x;
  check_shape "y shape" [| 100 |] y

let test_make_moons () =
  let x, y = make_moons ~n_samples:100 () in
  check_shape "X shape" [| 100; 2 |] x;
  check_shape "y shape" [| 100 |] y

let test_make_regression () =
  let x, y, coef =
    make_regression ~n_samples:100 ~n_features:10 ~coef:true ()
  in
  check_shape "X shape" [| 100; 10 |] x;
  check_shape "y shape" [| 100 |] y;
  match coef with
  | Some c -> check_shape "coef shape" [| 10; 1 |] c
  | None -> fail "Expected coefficients"

let test_make_friedman1 () =
  let x, y = make_friedman1 ~n_samples:100 () in
  check_shape "X shape" [| 100; 10 |] x;
  check_shape "y shape" [| 100 |] y

let test_make_s_curve () =
  let x, t = make_s_curve ~n_samples:100 () in
  check_shape "X shape" [| 100; 3 |] x;
  check_shape "t shape" [| 100 |] t

let test_make_swiss_roll () =
  let x, t = make_swiss_roll ~n_samples:100 () in
  check_shape "X shape" [| 100; 3 |] x;
  check_shape "t shape" [| 100 |] t

let test_make_low_rank_matrix () =
  let x = make_low_rank_matrix ~n_samples:50 ~n_features:100 () in
  check_shape "X shape" [| 50; 100 |] x

let test_make_spd_matrix () =
  let x = make_spd_matrix ~n_dim:30 () in
  check_shape "X shape" [| 30; 30 |] x

let test_make_biclusters () =
  let x, row_labels, col_labels = make_biclusters () in
  check_shape "X shape" [| 100; 100 |] x;
  check_shape "row_labels shape" [| 100 |] row_labels;
  check_shape "col_labels shape" [| 100 |] col_labels

(* ───── Dataset Utils ───── *)

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
  equal ~msg:"XDG_CACHE_HOME used when RAVEN_CACHE_ROOT unset" string expected2
    path2;

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
  equal ~msg:"Falls back to HOME/.cache when no env vars set" string expected3
    path3

(* ───── Entry Point ───── *)

let () =
  run "Nx_datasets"
    [
      group "Generators"
        [
          group "Classification"
            [
              test "make_blobs" test_make_blobs;
              test "make_classification" test_make_classification;
              test "make_gaussian_quantiles" test_make_gaussian_quantiles;
              test "make_circles" test_make_circles;
              test "make_moons" test_make_moons;
            ];
          group "Regression"
            [
              test "make_regression" test_make_regression;
              test "make_friedman1" test_make_friedman1;
            ];
          group "Manifold"
            [
              test "make_s_curve" test_make_s_curve;
              test "make_swiss_roll" test_make_swiss_roll;
            ];
          group "Matrix"
            [
              test "make_low_rank_matrix" test_make_low_rank_matrix;
              test "make_spd_matrix" test_make_spd_matrix;
              test "make_biclusters" test_make_biclusters;
            ];
        ];
      group "Dataset Utils"
        [
          group "Cache Directory Resolution"
            [ test "Environment variable precedence" test_cache_dir_resolution ];
        ];
    ]
