(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Everything here runs without network access: download tests exercise the
   cache and offline paths against a seeded temporary cache directory. *)

open Windtrap
module Archive = Nx_io.Archive
module Hf = Kaun_hf

let f32 = Nx.float32
let vec xs = Nx.create f32 [| Array.length xs |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] t)

let check_entry ~msg expected name a =
  let shape = [| Array.length expected |] in
  equal ~msg (array float_exact) expected
    (to_arr (Archive.tensor ~shape f32 name a))

let save path entries =
  Nx_io.save_safetensors path
    (Archive.of_list (List.map (fun (name, t) -> (name, Nx.P t)) entries))

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

(* Runs [f] with a fresh cache directory, removed afterwards even on failure. A
   loaded file stays mapped until its tensors are collected, and Windows may
   refuse to delete a mapped file. *)
let with_cache_dir f =
  let dir = Filename.temp_dir "kaun_hf" "" in
  Fun.protect
    ~finally:(fun () ->
      Gc.full_major ();
      try if Sys.file_exists dir then rm_rf dir
      with Sys_error _ when Sys.win32 -> ())
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
    (Filename.concat "/c"
       (Filename.concat "openai-community-gpt2"
          (Filename.concat "main" "config.json")))
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
  raises (Failure "Not cached (offline): acme/tiny/vocab.json") (fun () ->
      Hf.download_file ~cache_dir ~offline:true ~file:"vocab.json" "acme/tiny")

(* Puts a [curl] on PATH that creates [log] and writes to its standard output a
   partial file, then either completes it or, for a URL that contains "missing",
   fails as curl does on a 404. *)
let with_curl_stand_in ~log f =
  let bin = Filename.temp_dir "kaun_hf_bin" "" in
  let curl = Filename.concat bin "curl" in
  write_string curl
    (String.concat "\n"
       [
         "#!/bin/sh";
         Printf.sprintf ": > %s" (Filename.quote log);
         "for url; do :; done";
         "case \"$url\" in *missing*) printf 'part'; exit 22 ;; esac";
         "printf 'payload'";
         "";
       ]);
  Unix.chmod curl 0o755;
  let path = Sys.getenv "PATH" in
  Unix.putenv "PATH" (bin ^ ":" ^ path);
  Fun.protect
    ~finally:(fun () ->
      Unix.putenv "PATH" path;
      rm_rf bin)
    f

let read_string path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let test_download_is_atomic () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  with_cache_dir @@ fun cache_dir ->
  let log = Filename.concat cache_dir "curl-ran" in
  with_curl_stand_in ~log @@ fun () ->
  let local = Hf.download_file ~cache_dir ~file:"vocab.json" "acme/tiny" in
  let dir = Filename.dirname local in
  equal ~msg:"path" string
    (Hf.cache_path ~cache_dir ~file:"vocab.json" "acme/tiny")
    local;
  equal ~msg:"contents" string "payload" (read_string local);
  equal ~msg:"only the file remains" (array string) [| "vocab.json" |]
    (Sys.readdir dir);
  raises
    (Failure
       "Failed to download \
        https://huggingface.co/acme/tiny/resolve/main/missing.json") (fun () ->
      Hf.download_file ~cache_dir ~file:"missing.json" "acme/tiny");
  equal ~msg:"a failed download leaves nothing" (array string)
    [| "vocab.json" |] (Sys.readdir dir)

(* A umask without the owner's write bit makes a new file one only its creating
   descriptor can write: curl writes through that descriptor. *)
let test_download_under_umask () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  if Unix.geteuid () = 0 then skip ~reason:"root writes any file" ();
  with_cache_dir @@ fun cache_dir ->
  let log = Filename.concat cache_dir "curl-ran" in
  with_curl_stand_in ~log @@ fun () ->
  mkdir_p
    (Filename.dirname (Hf.cache_path ~cache_dir ~file:"vocab.json" "acme/tiny"));
  let umask = Unix.umask 0o277 in
  let local =
    Fun.protect
      ~finally:(fun () -> ignore (Unix.umask umask))
      (fun () -> Hf.download_file ~cache_dir ~file:"vocab.json" "acme/tiny")
  in
  equal ~msg:"contents" string "payload" (read_string local)

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

(* Downloading by digest *)

(* Puts a [curl] on PATH that writes the file a [file://] URL names, before
   any query, to its standard output, and creates [log]. *)
let with_file_curl ~log f =
  let bin = Filename.temp_dir "kaun_hf_bin" "" in
  let curl = Filename.concat bin "curl" in
  write_string curl
    (String.concat "\n"
       [
         "#!/bin/sh";
         Printf.sprintf ": > %s" (Filename.quote log);
         "for url; do :; done";
         "path=${url#file://}";
         "path=${path%%\\?*}";
         "exec cat \"$path\"";
         "";
       ]);
  Unix.chmod curl 0o755;
  let path = Sys.getenv "PATH" in
  Unix.putenv "PATH" (bin ^ ":" ^ path);
  Fun.protect
    ~finally:(fun () ->
      Unix.putenv "PATH" path;
      rm_rf bin)
    f

(* FIPS 180-2, appendix B, and the digest of no bytes. The 56-byte message
   pads into a second block. *)
let digests =
  [
    ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    ( "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" );
    ( String.make 1_000_000 'a',
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0" );
    ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  ]

let test_download_url_digests () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  with_cache_dir @@ fun dir ->
  let log = Filename.concat dir "curl-ran" in
  with_file_curl ~log @@ fun () ->
  let cache_dir = Filename.concat dir "cache" in
  List.iteri
    (fun i (bytes, sha256) ->
      let src = Filename.concat dir (Printf.sprintf "m%d.bin" i) in
      write_string src bytes;
      let url = "file://" ^ src ^ "?sig=x" in
      let local = Hf.download_url ~cache_dir ~sha256 url in
      let msg = Printf.sprintf "message %d" i in
      equal ~msg string
        (Filename.concat (Filename.concat cache_dir sha256)
           (Printf.sprintf "m%d.bin" i))
        local;
      equal ~msg:(msg ^ ", its bytes") string bytes (read_string local))
    digests

let test_download_url_refuses_another_digest () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  with_cache_dir @@ fun dir ->
  let log = Filename.concat dir "curl-ran" in
  with_file_curl ~log @@ fun () ->
  let cache_dir = Filename.concat dir "cache" in
  let src = Filename.concat dir "f.pt" in
  write_string src "abc";
  let url = "file://" ^ src in
  let want = String.make 64 '0' in
  raises
    (Failure
       (url
      ^ ": SHA-256 \
         ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad, \
         expected " ^ want))
    (fun () -> Hf.download_url ~cache_dir ~sha256:want url);
  equal ~msg:"nothing is left in the cache" (array string) [||]
    (Sys.readdir (Filename.concat cache_dir want))

let test_download_url_cache () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  with_cache_dir @@ fun dir ->
  let cache_dir = Filename.concat dir "cache" in
  let src = Filename.concat dir "f.pt" in
  let url = "file://" ^ src in
  let sha256 =
    "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
  in
  raises (Failure ("Not cached (offline): " ^ url)) (fun () ->
      Hf.download_url ~cache_dir ~offline:true ~sha256 url);
  write_string src "abc";
  let first =
    with_file_curl ~log:(Filename.concat dir "first") @@ fun () ->
    Hf.download_url ~cache_dir ~sha256 url
  in
  equal ~msg:"the digest names the file in lowercase" string
    (Filename.concat
       (Filename.concat cache_dir (String.lowercase_ascii sha256))
       "f.pt")
    first;
  let log = Filename.concat dir "second" in
  with_file_curl ~log @@ fun () ->
  equal ~msg:"served from the cache" string first
    (Hf.download_url ~cache_dir ~sha256 url);
  equal ~msg:"served offline" string first
    (Hf.download_url ~cache_dir ~offline:true ~sha256 url);
  is_true ~msg:"curl never ran" (not (Sys.file_exists log))

let test_download_url_rejects () =
  let sha256 = String.make 64 'a' in
  raises
    (Invalid_argument
       "Kaun_hf.download_url: \"abc\" is not 64 hexadecimal digits")
    (fun () -> Hf.download_url ~sha256:"abc" "https://example.com/f");
  let bad = String.make 63 'a' ^ "g" in
  raises
    (Invalid_argument
       (Printf.sprintf "Kaun_hf.download_url: %S is not 64 hexadecimal digits"
          bad))
    (fun () -> Hf.download_url ~sha256:bad "https://example.com/f");
  List.iter
    (fun url ->
      raises
        (Invalid_argument
           (Printf.sprintf "Kaun_hf.download_url: %s names no file" url))
        (fun () -> Hf.download_url ~sha256 url))
    [
      "https://example.com";
      "https://example.com/";
      "https://example.com/a/?q=1";
      "https://example.com/..";
      "https://example.com/a%5C..\\..\\x";
    ]

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
        save path [ ("w", vec [| 1.0; 2.0 |]); ("b", vec [| 3.0 |]) ])
  in
  let weights = Hf.load_safetensors ~cache_dir ~offline:true "acme/tiny" in
  equal ~msg:"names" (list string) [ "b"; "w" ] (Archive.names weights);
  check_entry ~msg:"w" [| 1.0; 2.0 |] "w" weights;
  check_entry ~msg:"b" [| 3.0 |] "b" weights

(* A repository cached as one file is loaded without asking the Hub whether it
   has a shard index. *)
let test_load_single_file_stays_local () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  with_cache_dir @@ fun cache_dir ->
  let _ =
    seed ~cache_dir ~repo_id:"acme/tiny" ~file:"model.safetensors" (fun path ->
        save path [ ("w", vec [| 1.0; 2.0 |]) ])
  in
  let log = Filename.concat cache_dir "curl-ran" in
  with_curl_stand_in ~log @@ fun () ->
  let weights = Hf.load_safetensors ~cache_dir "acme/tiny" in
  check_entry ~msg:"w" [| 1.0; 2.0 |] "w" weights;
  is_true ~msg:"curl never ran" (not (Sys.file_exists log))

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
        save path [ ("a", vec [| 1.0 |]); ("c", vec [| 3.0 |]) ])
  in
  let _ =
    seed ~cache_dir ~repo_id ~file:"model-00002.safetensors" (fun path ->
        save path [ ("b", vec [| 2.0 |]) ])
  in
  let weights = Hf.load_safetensors ~cache_dir ~offline:true repo_id in
  equal ~msg:"names" (list string) [ "a"; "b"; "c" ] (Archive.names weights);
  check_entry ~msg:"a" [| 1.0 |] "a" weights;
  check_entry ~msg:"b" [| 2.0 |] "b" weights;
  check_entry ~msg:"c" [| 3.0 |] "c" weights

let test_load_sharded_missing () =
  with_cache_dir @@ fun cache_dir ->
  let repo_id = "acme/sharded" in
  let _ =
    seed ~cache_dir ~repo_id ~file:"model.safetensors.index.json" (fun path ->
        write_string path
          {|{"metadata": {}, "weight_map": {"a": "model-00001.safetensors", "b": "model-00001.safetensors"}}|})
  in
  let _ =
    seed ~cache_dir ~repo_id ~file:"model-00001.safetensors" (fun path ->
        save path [ ("a", vec [| 1.0 |]) ])
  in
  raises
    (Failure
       "acme/sharded: tensor \"b\" missing in shard model-00001.safetensors")
    (fun () -> Hf.load_safetensors ~cache_dir ~offline:true repo_id)

let test_load_missing_raises () =
  with_cache_dir @@ fun cache_dir ->
  raises (Failure "No safetensors found for acme/empty") (fun () ->
      Hf.load_safetensors ~cache_dir ~offline:true "acme/empty")

(* Importing foreign weights *)

(* A GPT-2-style attention block built directly from the file's entries: the
   fused query, key and value projection is stored [d; 3d], [inputs;
   outputs], so each of its three pieces is transposed, and the output
   projection [outputs; inputs], [Linear]'s layout. *)
let test_import_attention () =
  let module Attention = Kaun.Attention in
  let d = 2 in
  let weights =
    Archive.of_list
      [
        ( "attn.c_attn.weight",
          Nx.P
            (Nx.create f32
               [| d; 3 * d |]
               [|
                 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0;
               |]) );
        ("attn.c_attn.bias", Nx.P (vec [| 0.5; 0.25; 0.125; 1.5; 2.5; 3.5 |]));
        ( "attn.c_proj.weight",
          Nx.P (Nx.create f32 [| d; d |] [| 1.0; 2.0; 3.0; 4.0 |]) );
        ("attn.c_proj.bias", Nx.P (vec [| 0.75; 1.25 |]));
      ]
  in
  let float ~shape name = Archive.float ~shape f32 name weights in
  let fused =
    List.combine
      (Nx.split ~axis:1 3 (float ~shape:[| d; 3 * d |] "attn.c_attn.weight"))
      (Nx.split ~axis:0 3 (float ~shape:[| 3 * d |] "attn.c_attn.bias"))
  in
  let linear (w, b) = { Kaun.Linear.w; b = Some b } in
  let p : (Nx.host Nx.float32_t, Nx.host) Attention.t =
    match
      List.map (fun (w, b) -> linear (Nx.matrix_transpose w, b)) fused
    with
    | [ q; k; v ] ->
        let out =
          linear
            ( float ~shape:[| d; d |] "attn.c_proj.weight",
              float ~shape:[| d |] "attn.c_proj.bias" )
        in
        { q; k; v; out }
    | _ -> assert false
  in
  equal ~msg:"q.w transposed" (array float_exact) [| 1.0; 7.0; 2.0; 8.0 |]
    (to_arr p.q.w);
  equal ~msg:"v.b" (array float_exact) [| 2.5; 3.5 |]
    (to_arr (Option.get p.v.b));
  equal ~msg:"out.w" (array float_exact) [| 1.0; 2.0; 3.0; 4.0 |]
    (to_arr p.out.w);
  raises
    (Failure
       "Nx_io.Archive.float: attn.c_proj.bias: shape [2] in the archive, [3] \
        asked for") (fun () -> float ~shape:[| 3 |] "attn.c_proj.bias")

let () =
  exit
    (run "kaun hf"
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
             test "a download is renamed into place" test_download_is_atomic;
             test "a download writes under a umask that denies the owner write"
               test_download_under_umask;
             test "clear_cache removes one repository or all" test_clear_cache;
             test "download_url keeps a file under its digest"
               test_download_url_digests;
             test "download_url refuses a file of another digest"
               test_download_url_refuses_another_digest;
             test "download_url serves its cache without the network"
               test_download_url_cache;
             test "download_url rejects a bad digest or a URL with no name"
               test_download_url_rejects;
           ];
         group "loading"
           [
             test "load_config parses a cached config.json" test_load_config;
             test "single-file weights load" test_load_single_file;
             test "a cached single file loads without the network"
               test_load_single_file_stays_local;
             test "sharded weights hold each indexed tensor from its shard"
               test_load_sharded;
             test "a tensor missing from its shard raises, naming it"
               test_load_sharded_missing;
             test "repositories without safetensors raise"
               test_load_missing_raises;
           ];
         group "importing"
           [
             test "a fused, transposed block is built from its entries"
               test_import_attention;
           ];
       ])
