(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module HF = Kaun_huggingface
module Ptree = Kaun.Ptree

let dtype = Rune.float32

let rec rm_rf path =
  if Sys.file_exists path then
    if Sys.is_directory path then (
      Sys.readdir path
      |> Array.iter (fun entry -> rm_rf (Filename.concat path entry));
      Unix.rmdir path)
    else Sys.remove path

let rec mkdir_p path =
  let parent = Filename.dirname path in
  if path = parent || Sys.file_exists path then ()
  else (
    mkdir_p parent;
    Unix.mkdir path 0o755)

let with_temp_cache f =
  let cache_dir = Filename.temp_dir "kaun_hf_cache" "" in
  Fun.protect ~finally:(fun () -> rm_rf cache_dir) (fun () -> f cache_dir)

let revision_dir cache_dir model_id =
  let sanitized =
    String.map (fun c -> if Char.equal c '/' then '-' else c) model_id
  in
  Filename.concat cache_dir (Filename.concat sanitized "main")

let save_shard path tensors =
  Nx_io.save_safetensor path
    (List.map (fun (name, tensor) -> (name, Nx_io.P tensor)) tensors)

let write_index path mapping =
  let weight_map =
    `Assoc (List.map (fun (name, shard) -> (name, `String shard)) mapping)
  in
  Yojson.Safe.to_file path (`Assoc [ ("weight_map", weight_map) ])

let to_params mapping =
  let fields = Ptree.Dict.fields_exn ~ctx:"hf params" mapping in
  fun ~name -> Ptree.Dict.get_tensor_exn fields ~name dtype

let unwrap_params = function
  | HF.Cached params -> params
  | HF.Downloaded (params, _) -> params

let assert_tensor_equal name expected actual =
  let equal_val =
    Rune.all (Rune.equal expected actual) |> Rune.to_array |> fun arr ->
    Array.get arr 0
  in
  equal ~msg:name bool true equal_val

let make_config ~cache_dir =
  let open HF.Config in
  {
    default with
    cache_dir;
    offline_mode = true;
    force_download = false;
    show_progress = false;
  }

let test_load_sharded () =
  with_temp_cache (fun cache_dir ->
      let model_id = "author/sharded" in
      let rev_dir = revision_dir cache_dir model_id in
      mkdir_p rev_dir;
      let shard_1 =
        Filename.concat rev_dir "model-00001-of-00002.safetensors"
      in
      let shard_2 =
        Filename.concat rev_dir "model-00002-of-00002.safetensors"
      in
      let index_path = Filename.concat rev_dir "model.safetensors.index.json" in
      let weight = Nx.arange Nx.float32 0 6 1 |> Nx.reshape [| 2; 3 |] in
      let bias = Nx.arange Nx.float32 0 3 1 in
      let proj = Nx.arange Nx.float32 10 16 1 |> Nx.reshape [| 3; 2 |] in
      save_shard shard_1 [ ("layer.weight", weight); ("layer.bias", bias) ];
      save_shard shard_2 [ ("layer2.weight", proj) ];
      write_index index_path
        [
          ("layer.weight", "model-00001-of-00002.safetensors");
          ("layer.bias", "model-00001-of-00002.safetensors");
          ("layer2.weight", "model-00002-of-00002.safetensors");
        ];
      let config = make_config ~cache_dir in
      let params = HF.load_safetensors ~config ~model_id () |> unwrap_params in
      let get = to_params params in
      assert_tensor_equal "layer.weight" (Rune.of_nx weight)
        (get ~name:"layer.weight");
      assert_tensor_equal "layer.bias" (Rune.of_nx bias)
        (get ~name:"layer.bias");
      assert_tensor_equal "layer2.weight" (Rune.of_nx proj)
        (get ~name:"layer2.weight"))

let test_load_single_file_fallback () =
  with_temp_cache (fun cache_dir ->
      let model_id = "author/single" in
      let rev_dir = revision_dir cache_dir model_id in
      mkdir_p rev_dir;
      let path = Filename.concat rev_dir "model.safetensors" in
      let weight = Nx.arange Nx.float32 0 4 1 |> Nx.reshape [| 2; 2 |] in
      let bias = Nx.arange Nx.float32 4 6 1 in
      save_shard path [ ("layer.weight", weight); ("layer.bias", bias) ];
      let config = make_config ~cache_dir in
      let params = HF.load_safetensors ~config ~model_id () |> unwrap_params in
      let get = to_params params in
      assert_tensor_equal "layer.weight" (Rune.of_nx weight)
        (get ~name:"layer.weight");
      assert_tensor_equal "layer.bias" (Rune.of_nx bias)
        (get ~name:"layer.bias"))

let () =
  run "Kaun.Huggingface"
    [
      group "load_safetensors"
        [
          test "loads sharded safetensors" test_load_sharded;
          test "falls back to single file" test_load_single_file_fallback;
        ];
    ]
