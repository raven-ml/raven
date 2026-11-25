(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nothing here touches the network: a cache is seeded in a temporary directory,
   and a [curl] stand-in on [PATH] serves what a test downloads. *)

open Windtrap
module Archive = Nx_io.Archive
module Hf = Kaun_hf

let write path s =
  let oc = open_out_bin path in
  output_string oc s;
  close_out oc

let read path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let rec mkdir_p path =
  if not (Sys.file_exists path) then begin
    mkdir_p (Filename.dirname path);
    Sys.mkdir path 0o755
  end

let listing dir =
  if Sys.file_exists dir then
    List.sort compare (Array.to_list (Sys.readdir dir))
  else []

let path parts = List.fold_left Filename.concat (List.hd parts) (List.tl parts)

(* Writes [file] of [repo_id] at its cache path, as if an earlier run had
   downloaded it. *)
let seed ~cache_dir ?revision ~repo_id ~file write_to =
  let path = Hf.cache_path ~cache_dir ?revision ~file repo_id in
  mkdir_p (Filename.dirname path);
  write_to path;
  path

let vec xs = Nx.create Nx.float32 [| Array.length xs |] xs

let safetensors entries path =
  Nx_io.save_safetensors path
    (Archive.of_list
       (List.map (fun (name, xs) -> (name, Nx.P (vec xs))) entries))

let entry name weights =
  let shape =
    match Archive.find name weights with
    | Some (Nx.P t) -> Nx.shape t
    | None -> failf "no entry %S" name
  in
  Nx.to_array (Archive.tensor ~shape Nx.float32 name weights)

(* A [curl] on [PATH] that writes its arguments to the file it returns, one per
   line, then writes the file a [file://] URL names, before any query or
   fragment, fails as curl does on a 404 after a partial write for a URL holding
   "missing", and writes "payload" for any other URL. [HF_TOKEN] is unset, so no
   inherited token reaches the log. *)
let curl_stand_in () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  let bin = temp_dir ~prefix:"bin" () in
  let log = Filename.concat (temp_dir ~prefix:"curl" ()) "arguments" in
  let curl = Filename.concat bin "curl" in
  write curl
    (String.concat "\n"
       [
         "#!/bin/sh";
         Printf.sprintf "printf '%%s\\n' \"$@\" > %s" (Filename.quote log);
         "for url; do :; done";
         "case \"$url\" in";
         "  file://*) path=${url#file://}; path=${path%%\\?*}; exec cat \
          \"${path%%#*}\" ;;";
         "  *missing*) printf 'part'; exit 22 ;;";
         "esac";
         "printf 'payload'";
         "";
       ]);
  Unix.chmod curl 0o755;
  setenv "PATH" (Some (bin ^ ":" ^ Sys.getenv "PATH"));
  setenv "HF_TOKEN" None;
  log

(* The arguments curl ran with, if it ran. *)
let ran log =
  if Sys.file_exists log then
    Some (List.filter (( <> ) "") (String.split_on_char '\n' (read log)))
  else None

let curl_never_ran log =
  equal ~msg:"curl's arguments" (option (list string)) None (ran log)

let existing paths = List.filter Sys.file_exists paths

(* The cache *)

let cache_path_layout () =
  equal string
    (path [ "/c"; "openai-community-gpt2"; "v1.2"; "config.json" ])
    (Hf.cache_path ~cache_dir:"/c" ~revision:"v1.2" ~file:"config.json"
       "openai-community/gpt2");
  equal ~msg:"the default revision" string
    (path [ "/c"; "gpt2"; "main"; "config.json" ])
    (Hf.cache_path ~cache_dir:"/c" ~file:"config.json" "gpt2")

let touches_nothing () =
  let dir = temp_dir () in
  ignore
    (Hf.cache_path
       ~cache_dir:(Filename.concat dir "cache")
       ~file:"config.json" "gpt2");
  equal ~msg:"files made" (list string) [] (listing dir)

(* The variable raven's cache root comes from. *)
type root = Raven_cache_root | Xdg_cache_home | Home

let environments =
  [
    ("RAVEN_CACHE_ROOT set", Raven_cache_root);
    ("XDG_CACHE_HOME set and RAVEN_CACHE_ROOT not", Xdg_cache_home);
    ("HOME alone", Home);
  ]

(* Sets [RAVEN_CACHE_ROOT], [XDG_CACHE_HOME] and [HOME] to fresh directories,
   unsetting those before [root]'s in that order, and is the cache root they
   give. *)
let cache_root root =
  let raven = temp_dir () and xdg = temp_dir () and home = temp_dir () in
  setenv "RAVEN_CACHE_ROOT"
    (if root = Raven_cache_root then Some raven else None);
  setenv "XDG_CACHE_HOME" (if root = Home then None else Some xdg);
  setenv "HOME" (Some home);
  match root with
  | Raven_cache_root -> raven
  | Xdg_cache_home -> Filename.concat xdg "raven"
  | Home -> path [ home; ".cache"; "raven" ]

let cache_path_default (_, root) =
  let root = cache_root root in
  equal string
    (path [ root; "huggingface"; "gpt2"; "main"; "vocab.json" ])
    (Hf.cache_path ~file:"vocab.json" "gpt2")

let clears () =
  let cache_dir = temp_dir () in
  let seeded repo_id =
    seed ~cache_dir ~repo_id ~file:"f" (fun p -> write p "x")
  in
  let a = seeded "acme/a" and b = seeded "acme/b" in
  Hf.clear_cache ~cache_dir ~repo_id:"acme/a" ();
  equal ~msg:"files left after clearing acme/a" (list string) [ b ]
    (existing [ a; b ]);
  Hf.clear_cache ~cache_dir ();
  equal ~msg:"files left after clearing all" (list string) []
    (existing [ a; b ])

(* Fetching files *)

let serves_cached () =
  let log = curl_stand_in () in
  let cache_dir = temp_dir () in
  let seeded =
    seed ~cache_dir ~repo_id:"acme/tiny" ~file:"vocab.json" (fun p ->
        write p "{}")
  in
  equal ~msg:"offline" string seeded
    (Hf.download_file ~cache_dir ~offline:true ~file:"vocab.json" "acme/tiny");
  equal ~msg:"online" string seeded
    (Hf.download_file ~cache_dir ~file:"vocab.json" "acme/tiny");
  curl_never_ran log

(* The stand-in would serve the file: a [Failure] with curl never run is the
   offline refusal, never a failed download. *)
let offline_miss () =
  let log = curl_stand_in () in
  let cache_dir = temp_dir () in
  raises_match
    (fun e -> Exn.failure e)
    (fun () ->
      Hf.download_file ~cache_dir ~offline:true ~file:"vocab.json" "acme/tiny");
  curl_never_ran log

let no_curl () =
  setenv "PATH" (Some (temp_dir ()));
  let cache_dir = temp_dir () in
  raises_match
    (fun e -> Exn.failure e)
    (fun () -> Hf.download_file ~cache_dir ~file:"vocab.json" "acme/tiny")

let renamed_into_place () =
  ignore (curl_stand_in ());
  let cache_dir = temp_dir () in
  let local = Hf.download_file ~cache_dir ~file:"vocab.json" "acme/tiny" in
  equal ~msg:"path" string
    (Hf.cache_path ~cache_dir ~file:"vocab.json" "acme/tiny")
    local;
  equal ~msg:"contents" string "payload" (read local);
  equal ~msg:"files beside it" (list string) [ "vocab.json" ]
    (listing (Filename.dirname local))

let failure_leaves_nothing () =
  let log = curl_stand_in () in
  let cache_dir = temp_dir () in
  raises_match
    (fun e -> Exn.failure e)
    (fun () -> Hf.download_file ~cache_dir ~file:"missing.json" "acme/tiny");
  is_some ~msg:"curl ran" (ran log);
  equal (list string) []
    (listing
       (Filename.dirname
          (Hf.cache_path ~cache_dir ~file:"missing.json" "acme/tiny")))

(* A umask without the owner's write bit makes a new file one only its creating
   descriptor can write: the download goes through that descriptor. *)
let under_umask () =
  ignore (curl_stand_in ());
  if Unix.geteuid () = 0 then skip ~reason:"root writes any file" ();
  let cache_dir = temp_dir () in
  mkdir_p
    (Filename.dirname (Hf.cache_path ~cache_dir ~file:"vocab.json" "acme/tiny"));
  let umask = Unix.umask 0o277 in
  let local =
    Fun.protect
      ~finally:(fun () -> ignore (Unix.umask umask))
      (fun () -> Hf.download_file ~cache_dir ~file:"vocab.json" "acme/tiny")
  in
  equal string "payload" (read local)

let bearer t = "Authorization: Bearer " ^ t

(* The token passed, [HF_TOKEN], and the header curl must send. *)
let tokens =
  [
    ("~token", (Some "t0k", Some "env"), Some (bearer "t0k"));
    ("HF_TOKEN", (None, Some "env"), Some (bearer "env"));
    ("neither", (None, None), None);
  ]

let sends_token (_, (token, hf_token), header) =
  let log = curl_stand_in () in
  setenv "HF_TOKEN" hf_token;
  ignore
    (Hf.download_file ?token ~cache_dir:(temp_dir ()) ~file:"f.json" "acme/tiny");
  let args = Option.value (ran log) ~default:[] in
  match header with
  | Some h -> mem ~msg:"the header" string h args
  | None ->
      satisfies ~claim:"no Authorization header" (list string)
        (List.for_all (fun a ->
             not (String.starts_with ~prefix:"Authorization" a)))
        args

(* The Hub serves a file of a repository at a revision at
   https://huggingface.co/<repo_id>/resolve/<revision>/<file> (the Hub's
   documentation, "Download files from the Hub"). *)
let fetches_revision () =
  let log = curl_stand_in () in
  let cache_dir = temp_dir () in
  let local =
    Hf.download_file ~cache_dir ~revision:"v1" ~file:"f.json" "acme/tiny"
  in
  equal ~msg:"path" string
    (Hf.cache_path ~cache_dir ~revision:"v1" ~file:"f.json" "acme/tiny")
    local;
  mem ~msg:"the URL" string "https://huggingface.co/acme/tiny/resolve/v1/f.json"
    (Option.value (ran log) ~default:[])

(* FIPS 180-2, appendix B, and the digest of no bytes. The 56-byte message pads
   into a second block. *)
let digests =
  [
    ( "abc",
      lazy "abc",
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" );
    ( "two blocks",
      lazy "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" );
    ( "a million a",
      lazy (String.make 1_000_000 'a'),
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0" );
    ( "no bytes",
      lazy "",
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" );
  ]

(* A file to fetch by a [file://] URL ending in [suffix], which curl does not
   read. *)
let source ?(suffix = "?sig=x") bytes =
  let src = Filename.concat (temp_dir ()) "m.bin" in
  write src bytes;
  "file://" ^ src ^ suffix

let keeps_by_digest (_, bytes, sha256) =
  let bytes = Lazy.force bytes in
  ignore (curl_stand_in ());
  let cache_dir = temp_dir () in
  let local = Hf.download_url ~cache_dir ~sha256 (source bytes) in
  equal ~msg:"path" string (path [ cache_dir; sha256; "m.bin" ]) local;
  equal ~msg:"bytes" string bytes (read local)

let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

let refuses_another_digest () =
  ignore (curl_stand_in ());
  let cache_dir = temp_dir () in
  let want = String.make 64 '0' in
  raises_match
    (function
      | Failure m ->
          let has sub = Exn.failure ~sub (Failure m) in
          has abc && has want
      | _ -> false)
    (fun () -> Hf.download_url ~cache_dir ~sha256:want (source "abc"));
  equal ~msg:"the digest's directory" (list string) []
    (listing (Filename.concat cache_dir want))

let lowercase_name () =
  ignore (curl_stand_in ());
  let cache_dir = temp_dir () in
  equal string
    (path [ cache_dir; abc; "m.bin" ])
    (Hf.download_url ~cache_dir
       ~sha256:(String.uppercase_ascii abc)
       (source "abc"))

let names_by_last_segment suffix =
  ignore (curl_stand_in ());
  let cache_dir = temp_dir () in
  equal string
    (path [ cache_dir; abc; "m.bin" ])
    (Hf.download_url ~cache_dir ~sha256:abc (source ~suffix "abc"))

(* The cached file's bytes have another digest than its name: a call that read
   it again would refuse it. *)
let url_served_cached () =
  let log = curl_stand_in () in
  let cache_dir = temp_dir () in
  let local = path [ cache_dir; abc; "f.pt" ] in
  mkdir_p (Filename.dirname local);
  write local "not abc";
  let url = "https://example.com/weights/f.pt" in
  equal ~msg:"online" string local (Hf.download_url ~cache_dir ~sha256:abc url);
  equal ~msg:"offline" string local
    (Hf.download_url ~cache_dir ~offline:true ~sha256:abc url);
  curl_never_ran log

(* As for download_file: a [Failure] with curl never run is the offline
   refusal. *)
let url_offline_miss () =
  let log = curl_stand_in () in
  let url = "https://example.com/f.pt" in
  raises_match
    (fun e -> Exn.failure e)
    (fun () ->
      Hf.download_url ~cache_dir:(temp_dir ()) ~offline:true ~sha256:abc url);
  curl_never_ran log

let url_cache_default (_, root) =
  let log = curl_stand_in () in
  let local = path [ cache_root root; "sha256"; abc; "f.pt" ] in
  mkdir_p (Filename.dirname local);
  write local "abc";
  equal string local
    (Hf.download_url ~offline:true ~sha256:abc "https://example.com/f.pt");
  curl_never_ran log

let invalid_url_arguments =
  let sha256 = String.make 64 'a' in
  [
    ("a digest of 3 digits", "abc", "https://example.com/f");
    ("a digest with a g", String.make 63 'a' ^ "g", "https://example.com/f");
    ("no path", sha256, "https://example.com");
    ("an empty name", sha256, "https://example.com/");
    ("an empty name before a query", sha256, "https://example.com/a/?q=1");
    ("a backslash", sha256, "https://example.com/a%5C..\\..\\x");
  ]

(* Loading models *)

let parses_config () =
  let log = curl_stand_in () in
  let cache_dir = temp_dir () in
  ignore
    (seed ~cache_dir ~repo_id:"acme/tiny" ~file:"config.json" (fun p ->
         write p {|{"n_layer": 2}|}));
  let n_layer =
    match Hf.load_config ~cache_dir ~offline:true "acme/tiny" with
    | Jsont.Object (mems, _) -> (
        match Jsont.Json.find_mem "n_layer" mems with
        | Some (_, Jsont.Number (n, _)) -> Some n
        | _ -> None)
    | _ -> None
  in
  equal (option float_exact) (Some 2.) n_layer;
  curl_never_ran log

let config_parse_error () =
  let log = curl_stand_in () in
  let cache_dir = temp_dir () in
  ignore
    (seed ~cache_dir ~repo_id:"acme/tiny" ~file:"config.json" (fun p ->
         write p {|{"n_layer": |}));
  raises_match
    (fun e -> Exn.failure e)
    (fun () -> Hf.load_config ~cache_dir ~offline:true "acme/tiny");
  curl_never_ran log

let single_file () =
  let log = curl_stand_in () in
  let cache_dir = temp_dir () in
  ignore
    (seed ~cache_dir ~repo_id:"acme/tiny" ~file:"model.safetensors"
       (safetensors [ ("w", [| 1.; 2. |]); ("b", [| 3. |]) ]));
  let weights = Hf.load_safetensors ~cache_dir ~offline:true "acme/tiny" in
  equal ~msg:"names" (list string) [ "b"; "w" ] (Archive.names weights);
  equal ~msg:"w" (array float_exact) [| 1.; 2. |] (entry "w" weights);
  equal ~msg:"b" (array float_exact) [| 3. |] (entry "b" weights);
  curl_never_ran log

(* A repository cached as one file is loaded without asking the Hub whether it
   has a shard index. *)
let single_file_stays_local () =
  let log = curl_stand_in () in
  let cache_dir = temp_dir () in
  ignore
    (seed ~cache_dir ~repo_id:"acme/tiny" ~file:"model.safetensors"
       (safetensors [ ("w", [| 1.; 2. |]) ]));
  let weights = Hf.load_safetensors ~cache_dir "acme/tiny" in
  equal ~msg:"w" (array float_exact) [| 1.; 2. |] (entry "w" weights);
  curl_never_ran log

let index weight_map p =
  write p
    (Printf.sprintf {|{"metadata": {}, "weight_map": {%s}}|}
       (String.concat ", "
          (List.map
             (fun (name, shard) -> Printf.sprintf "%S: %S" name shard)
             weight_map)))

let sharded_repo ~weight_map shards =
  let cache_dir = temp_dir () in
  let repo_id = "acme/sharded" in
  ignore
    (seed ~cache_dir ~repo_id ~file:"model.safetensors.index.json"
       (index weight_map));
  List.iter
    (fun (file, entries) ->
      ignore (seed ~cache_dir ~repo_id ~file (safetensors entries)))
    shards;
  fun () -> Hf.load_safetensors ~cache_dir ~offline:true repo_id

let sharded () =
  let log = curl_stand_in () in
  let load =
    sharded_repo
      ~weight_map:
        [
          ("a", "model-00001.safetensors");
          ("b", "model-00002.safetensors");
          ("c", "model-00001.safetensors");
        ]
      [
        ( "model-00001.safetensors",
          [ ("a", [| 1. |]); ("c", [| 3. |]); ("unlisted", [| 9. |]) ] );
        ("model-00002.safetensors", [ ("b", [| 2. |]) ]);
      ]
  in
  let weights = load () in
  equal ~msg:"names" (list string) [ "a"; "b"; "c" ] (Archive.names weights);
  List.iter
    (fun (name, v) ->
      equal ~msg:name (array float_exact) [| v |] (entry name weights))
    [ ("a", 1.); ("b", 2.); ("c", 3.) ];
  curl_never_ran log

let load_refusals =
  [
    ( "a tensor its shard lacks",
      fun () ->
        sharded_repo
          ~weight_map:
            [
              ("a", "model-00001.safetensors"); ("b", "model-00001.safetensors");
            ]
          [ ("model-00001.safetensors", [ ("a", [| 1. |]) ]) ]
          () );
    ( "a tensor named twice",
      fun () ->
        sharded_repo
          ~weight_map:
            [
              ("a", "model-00001.safetensors"); ("a", "model-00002.safetensors");
            ]
          [
            ("model-00001.safetensors", [ ("a", [| 1. |]) ]);
            ("model-00002.safetensors", [ ("a", [| 2. |]) ]);
          ]
          () );
    ( "neither an index nor model.safetensors",
      fun () ->
        Hf.load_safetensors ~cache_dir:(temp_dir ()) ~offline:true "acme/empty"
    );
  ]

let () =
  exit
    (run "kaun hf"
       [
         group "download_file"
           [
             test "serves a cached file without the network" serves_cached;
             test "raises Failure offline on a file not cached" offline_miss;
             test "raises Failure without curl on the PATH" no_curl;
             test "renames a complete download into its cache path"
               renamed_into_place;
             test "leaves nothing of a failed download" failure_leaves_nothing;
             test "writes under a umask that denies the owner write" under_umask;
             cases "sends the token, else HF_TOKEN, as a bearer token"
               ~id:(fun (id, _, _) -> id)
               tokens sends_token;
             test "fetches the revision" fetches_revision;
           ];
         group "download_url"
           [
             cases "keeps a file under its FIPS 180-4 digest"
               ~id:(fun (id, _, _) -> id)
               digests keeps_by_digest;
             test "names the file by the digest in lowercase" lowercase_name;
             cases
               "names the file by the URL's last segment before a query or \
                fragment"
               ~id:Fun.id
               [ "?sig=x"; "#f"; "?q=1#f" ]
               names_by_last_segment;
             test
               "refuses a file of another digest, naming both, and keeps \
                nothing"
               refuses_another_digest;
             test "serves a cached file without reading it or the network"
               url_served_cached;
             test "raises Failure offline on a file not cached" url_offline_miss;
             cases
               "defaults its cache to the sha256 directory of raven's cache \
                root"
               ~id:fst environments url_cache_default;
             cases
               "raises on a digest of other than 64 hexadecimal digits or a \
                URL naming no file"
               ~id:(fun (id, _, _) -> id)
               invalid_url_arguments
               (fun (_, sha256, url) ->
                 let log = curl_stand_in () in
                 raises_match
                   (fun e -> Exn.invalid_arg e)
                   (fun () ->
                     Hf.download_url ~cache_dir:(temp_dir ()) ~sha256 url);
                 curl_never_ran log);
           ];
         group "load_config"
           [
             test "parses config.json" parses_config;
             test "raises Failure on a JSON parse error" config_parse_error;
           ];
         group "load_safetensors"
           [
             test "loads the single model.safetensors" single_file;
             test "loads a cached single file without the network"
               single_file_stays_local;
             test "holds each tensor the index names, from its shard" sharded;
             cases "raises Failure on a broken repository" ~id:fst load_refusals
               (fun (_, load) ->
                 let log = curl_stand_in () in
                 raises_match (fun e -> Exn.failure e) load;
                 curl_never_ran log);
           ];
         group "cache_path"
           [
             test "is cache_dir/repo/revision/file, repo's slashes as dashes"
               cache_path_layout;
             test "is computed and touches nothing" touches_nothing;
             cases
               "defaults cache_dir to the huggingface directory of raven's \
                cache root"
               ~id:fst environments cache_path_default;
           ];
         group "clear_cache"
           [
             test "removes repo_id's files, or every repository's without it"
               clears;
           ];
       ])
