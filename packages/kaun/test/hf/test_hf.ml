(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Everything here runs without network access: download tests exercise the
   cache and offline paths against a seeded temporary cache directory. *)

open Windtrap
module Checkpoint = Kaun.Checkpoint
module Hf = Kaun_hf

let f32 = Nx.float32
let vec xs = Nx.create f32 [| Array.length xs |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

let entry name ckpt =
  match Checkpoint.get name ckpt with Rune.Ptree.P x -> Nx.cast f32 x

let check_entry ~msg expected name ckpt =
  equal ~msg (array (float 0.0)) expected (to_arr (entry name ckpt))

(* Filesystem helpers *)

let rec mkdir_p path =
  if path = "" || path = "." || Sys.file_exists path then ()
  else begin
    mkdir_p (Filename.dirname path);
    Sys.mkdir path 0o755
  end

let rec rm_rf path =
  if Sys.is_directory path then begin
    Array.iter (fun e -> rm_rf (Filename.concat path e)) (Sys.readdir path);
    Sys.rmdir path
  end
  else Sys.remove path

(* Runs [f] with a fresh cache directory, removed afterwards even on failure. *)
let with_cache_dir f =
  let dir = Filename.temp_dir "kaun_hf" "" in
  Fun.protect
    ~finally:(fun () -> if Sys.file_exists dir then rm_rf dir)
    (fun () -> f dir)

(* Seeds [file] of [repo_id] into [cache_dir] by calling [write] on its cache
   path, as if a previous run had downloaded it. *)
let seed ~cache_dir ~repo_id ~file write =
  let path = Hf.cache_path ~cache_dir ~file repo_id in
  mkdir_p (Filename.dirname path);
  write path;
  path

let write_string path s =
  let oc = open_out_bin path in
  output_string oc s;
  close_out oc

(* Cache paths *)

let test_cache_path_layout () =
  equal string
    (Filename.concat "/c" "openai-community-gpt2/main/config.json"
    |> String.split_on_char '/'
    |> String.concat Filename.dir_sep)
    (Hf.cache_path ~cache_dir:"/c" ~file:"config.json" "openai-community/gpt2")

let test_cache_path_revision () =
  is_true
    (Hf.cache_path ~cache_dir:"/c" ~revision:"v1.2" ~file:"model.safetensors"
       "gpt2"
    = Filename.concat "/c"
        (Filename.concat "gpt2" (Filename.concat "v1.2" "model.safetensors")))

let test_cache_path_env () =
  let saved = Sys.getenv_opt "RAVEN_CACHE_ROOT" in
  Fun.protect
    ~finally:(fun () ->
      Unix.putenv "RAVEN_CACHE_ROOT" (Option.value saved ~default:""))
    (fun () ->
      Unix.putenv "RAVEN_CACHE_ROOT" "/raven-cache";
      equal ~msg:"file path" string
        (Filename.concat "/raven-cache"
           (Filename.concat "huggingface"
              (Filename.concat "gpt2" (Filename.concat "main" "vocab.json"))))
        (Hf.cache_path ~file:"vocab.json" "gpt2"))

(* Downloading (cache and offline behaviour only; no network) *)

let test_cached_file_served () =
  with_cache_dir @@ fun cache_dir ->
  let seeded =
    seed ~cache_dir ~repo_id:"acme/tiny" ~file:"vocab.json" (fun path ->
        write_string path "{}")
  in
  let got =
    Hf.download_file ~cache_dir ~offline:true ~file:"vocab.json" "acme/tiny"
  in
  equal ~msg:"path" string seeded got;
  (* A cached file short-circuits the network even when online. *)
  equal ~msg:"path (online)" string seeded
    (Hf.download_file ~cache_dir ~file:"vocab.json" "acme/tiny")

let test_offline_miss_raises () =
  with_cache_dir @@ fun cache_dir ->
  raises_failure "Not cached (offline): acme/tiny/vocab.json" (fun () ->
      Hf.download_file ~cache_dir ~offline:true ~file:"vocab.json" "acme/tiny")

let test_clear_cache () =
  with_cache_dir @@ fun cache_dir ->
  let a =
    seed ~cache_dir ~repo_id:"acme/a" ~file:"f" (fun p -> write_string p "a")
  in
  let b =
    seed ~cache_dir ~repo_id:"acme/b" ~file:"f" (fun p -> write_string p "b")
  in
  Hf.clear_cache ~cache_dir ~repo_id:"acme/a" ();
  is_true ~msg:"repo a removed" (not (Sys.file_exists a));
  is_true ~msg:"repo b kept" (Sys.file_exists b);
  Hf.clear_cache ~cache_dir ();
  is_true ~msg:"all removed" (not (Sys.file_exists b))

(* Loading (from a seeded cache, offline) *)

let test_load_config () =
  with_cache_dir @@ fun cache_dir ->
  let _ =
    seed ~cache_dir ~repo_id:"acme/tiny" ~file:"config.json" (fun path ->
        write_string path {|{"n_layer": 2}|})
  in
  match Hf.load_config ~cache_dir ~offline:true "acme/tiny" with
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem "n_layer" mems with
      | Some (_, Jsont.Number (n, _)) ->
          equal ~msg:"n_layer" int 2 (int_of_float n)
      | _ -> failf "n_layer missing from parsed config")
  | _ -> failf "unexpected config JSON shape"

let test_load_single_file () =
  with_cache_dir @@ fun cache_dir ->
  let _ =
    seed ~cache_dir ~repo_id:"acme/tiny" ~file:"model.safetensors" (fun path ->
        Checkpoint.save path
          (Checkpoint.concat
             [
               Checkpoint.of_tensor "w" (vec [| 1.0; 2.0 |]);
               Checkpoint.of_tensor "b" (vec [| 3.0 |]);
             ]))
  in
  let ckpt = Hf.load_checkpoint ~cache_dir ~offline:true "acme/tiny" in
  equal ~msg:"names" (list string) [ "b"; "w" ] (Checkpoint.names ckpt);
  check_entry ~msg:"w" [| 1.0; 2.0 |] "w" ckpt;
  check_entry ~msg:"b" [| 3.0 |] "b" ckpt

let test_load_sharded () =
  with_cache_dir @@ fun cache_dir ->
  let repo_id = "acme/sharded" in
  let _ =
    seed ~cache_dir ~repo_id ~file:"model.safetensors.index.json" (fun path ->
        write_string path
          {|{"metadata": {}, "weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors", "c": "model-00001.safetensors"}}|})
  in
  let _ =
    seed ~cache_dir ~repo_id ~file:"model-00001.safetensors" (fun path ->
        Checkpoint.save path
          (Checkpoint.concat
             [
               Checkpoint.of_tensor "a" (vec [| 1.0 |]);
               Checkpoint.of_tensor "c" (vec [| 3.0 |]);
             ]))
  in
  let _ =
    seed ~cache_dir ~repo_id ~file:"model-00002.safetensors" (fun path ->
        Checkpoint.save path (Checkpoint.of_tensor "b" (vec [| 2.0 |])))
  in
  let ckpt = Hf.load_checkpoint ~cache_dir ~offline:true repo_id in
  equal ~msg:"names" (list string) [ "a"; "b"; "c" ] (Checkpoint.names ckpt);
  check_entry ~msg:"a" [| 1.0 |] "a" ckpt;
  check_entry ~msg:"b" [| 2.0 |] "b" ckpt;
  check_entry ~msg:"c" [| 3.0 |] "c" ckpt

let test_load_missing_raises () =
  with_cache_dir @@ fun cache_dir ->
  raises_failure "No safetensors found for acme/empty" (fun () ->
      Hf.load_checkpoint ~cache_dir ~offline:true "acme/empty")

(* Adapting foreign checkpoints *)

let two_entries () =
  Checkpoint.concat
    [
      Checkpoint.of_tensor "a" (vec [| 1.0; 2.0 |]);
      Checkpoint.of_tensor "b" (vec [| 3.0 |]);
    ]

let test_rename () =
  let ckpt = Hf.rename (function "a" -> "x.w" | n -> n) (two_entries ()) in
  equal ~msg:"names" (list string) [ "b"; "x.w" ] (Checkpoint.names ckpt);
  check_entry ~msg:"renamed entry keeps its tensor" [| 1.0; 2.0 |] "x.w" ckpt

let test_rename_collision () =
  raises_invalid_arg "Kaun_hf.rename: duplicate name \"b\"" (fun () ->
      Hf.rename (fun _ -> "b") (two_entries ()))

let test_rename_empty () =
  raises_invalid_arg "Kaun_hf.rename: empty entry name" (fun () ->
      Hf.rename (fun _ -> "") (Checkpoint.of_tensor "a" (vec [| 1.0 |])))

let test_transpose () =
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let ckpt = Hf.transpose "w" (Checkpoint.of_tensor "w" x) in
  equal ~msg:"shape" (array int) [| 3; 2 |] (Nx.shape (entry "w" ckpt));
  check_entry ~msg:"values" [| 1.0; 4.0; 2.0; 5.0; 3.0; 6.0 |] "w" ckpt

let test_transpose_errors () =
  raises_invalid_arg "Kaun_hf.transpose: no entry named \"w\"" (fun () ->
      Hf.transpose "w" Checkpoint.empty);
  raises_invalid_arg
    "Kaun_hf.transpose: entry \"v\" has 1 axes, needs at least 2" (fun () ->
      Hf.transpose "v" (Checkpoint.of_tensor "v" (vec [| 1.0 |])))

let test_split () =
  let fused =
    Nx.create f32 [| 2; 6 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0 |]
  in
  let ckpt =
    Hf.split "qkv" ~into:[ "q"; "k"; "v" ] (Checkpoint.of_tensor "qkv" fused)
  in
  equal ~msg:"names" (list string) [ "k"; "q"; "v" ] (Checkpoint.names ckpt);
  equal ~msg:"shape" (array int) [| 2; 2 |] (Nx.shape (entry "q" ckpt));
  check_entry ~msg:"q" [| 1.0; 2.0; 7.0; 8.0 |] "q" ckpt;
  check_entry ~msg:"k" [| 3.0; 4.0; 9.0; 10.0 |] "k" ckpt;
  check_entry ~msg:"v" [| 5.0; 6.0; 11.0; 12.0 |] "v" ckpt

let test_split_axis () =
  let x = Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let ckpt =
    Hf.split ~axis:0 "x" ~into:[ "top"; "bottom" ] (Checkpoint.of_tensor "x" x)
  in
  check_entry ~msg:"top" [| 1.0; 2.0 |] "top" ckpt;
  check_entry ~msg:"bottom" [| 3.0; 4.0 |] "bottom" ckpt

let test_split_errors () =
  raises_invalid_arg "Kaun_hf.split: no entry named \"x\"" (fun () ->
      Hf.split "x" ~into:[ "a" ] Checkpoint.empty);
  let ckpt = Checkpoint.of_tensor "x" (vec [| 1.0; 2.0; 3.0 |]) in
  raises_invalid_arg "Kaun_hf.split: empty name list" (fun () ->
      Hf.split "x" ~into:[] ckpt);
  raises_invalid_arg
    "Kaun_hf.split: axis 0 of entry \"x\" has size 3, not a multiple of 2"
    (fun () -> Hf.split "x" ~into:[ "a"; "b" ] ckpt);
  raises_invalid_arg "Kaun_hf.split: axis out of bounds for entry \"x\""
    (fun () -> Hf.split ~axis:1 "x" ~into:[ "a" ] ckpt);
  let two = two_entries () in
  raises_invalid_arg "Kaun_hf.split: duplicate name \"b\"" (fun () ->
      Hf.split "a" ~into:[ "b"; "c" ] two)

(* End to end: a GPT-2-style fused attention block, remapped into a typed
   [Attention.params] record through [Checkpoint.to_params]. *)

let test_remap_into_attention () =
  let module Attention = Kaun.Attention in
  (* Conv1D layout: fused qkv weight is [d; 3d], no transpose needed. The out
     projection is stored [out; in] (nn.Linear layout) to exercise
     [transpose]. *)
  let fused_w =
    Nx.create f32 [| 2; 6 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0 |]
  in
  let fused_b = vec [| 0.5; 0.25; 0.125; 1.5; 2.5; 3.5 |] in
  let out_w = Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let out_b = vec [| 0.75; 1.25 |] in
  let hf =
    Checkpoint.concat
      [
        Checkpoint.of_tensor "attn.c_attn.weight" fused_w;
        Checkpoint.of_tensor "attn.c_attn.bias" fused_b;
        Checkpoint.of_tensor "attn.c_proj.weight" out_w;
        Checkpoint.of_tensor "attn.c_proj.bias" out_b;
      ]
  in
  let ours =
    hf
    |> Hf.split "attn.c_attn.weight" ~into:[ "q.w"; "k.w"; "v.w" ]
    |> Hf.split "attn.c_attn.bias" ~into:[ "q.b"; "k.b"; "v.b" ]
    |> Hf.transpose "attn.c_proj.weight"
    |> Hf.rename (function
      | "attn.c_proj.weight" -> "out.w"
      | "attn.c_proj.bias" -> "out.b"
      | n -> n)
  in
  let like = Nx.Rng.run ~seed:0 @@ fun () -> Attention.init ~embed_dim:2 in
  let p = Checkpoint.to_params (module Attention) ~like ours in
  equal ~msg:"q.w" (array (float 0.0)) [| 1.0; 2.0; 7.0; 8.0 |] (to_arr p.q.w);
  equal ~msg:"v.b"
    (array (float 0.0))
    [| 2.5; 3.5 |]
    (to_arr (Option.get p.v.b));
  equal ~msg:"out.w transposed"
    (array (float 0.0))
    [| 1.0; 3.0; 2.0; 4.0 |] (to_arr p.out.w)

let () =
  run "kaun hf"
    [
      group "cache paths"
        [
          test "cache_path lays out cache_dir/repo/revision/file"
            test_cache_path_layout;
          test "cache_path uses the revision" test_cache_path_revision;
          test "cache_path honours RAVEN_CACHE_ROOT" test_cache_path_env;
        ];
      group "downloading"
        [
          test "cached files are served without the network"
            test_cached_file_served;
          test "offline misses raise" test_offline_miss_raises;
          test "clear_cache removes one repository or all" test_clear_cache;
        ];
      group "loading"
        [
          test "load_config parses a cached config.json" test_load_config;
          test "single-file checkpoints load" test_load_single_file;
          test "sharded checkpoints merge their shards" test_load_sharded;
          test "repositories without safetensors raise" test_load_missing_raises;
        ];
      group "adapting"
        [
          test "rename rewrites entry names" test_rename;
          test "rename rejects colliding names" test_rename_collision;
          test "rename rejects empty names" test_rename_empty;
          test "transpose swaps the last two axes" test_transpose;
          test "transpose rejects missing and 1-D entries" test_transpose_errors;
          test "split cuts an entry along its last axis" test_split;
          test "split honours the axis argument" test_split_axis;
          test "split rejects bad axes, sizes and names" test_split_errors;
          test "a fused, transposed block loads into Attention.params"
            test_remap_into_attention;
        ];
    ]
