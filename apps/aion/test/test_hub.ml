(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Aion

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    Sys.mkdir dir 0o755
  end

(* A cache holding [codecs/NAME/model.safetensors] where Kaun_hf caches the
   repository's file at Hub's revision. *)
let cache_with name archive =
  let cache_dir = Filename.temp_dir "aion" "hub" in
  let file = "codecs/" ^ name ^ "/model.safetensors" in
  let path =
    Kaun_hf.cache_path ~cache_dir ~revision:Hub.revision ~file Hub.repo
  in
  mkdir_p (Filename.dirname path);
  Nx_io.save_safetensors path archive;
  cache_dir

let edges = Nx.create Nx.float32 [| 3 |] [| -1.; 0.5; 2. |]

let cached =
  test "codec reads the codec's file at the revision" (fun () ->
      let entry = "_quantizer._index_to_val" in
      let cache_dir =
        cache_with "FLUX_G" (Nx_io.Archive.of_list [ (entry, Nx.P edges) ])
      in
      let a = Hub.codec ~cache_dir ~offline:true "FLUX_G" in
      equal (list string) [ entry ] (Nx_io.Archive.names a);
      let read = Nx_io.Archive.tensor ~shape:[| 3 |] Nx.float32 entry a in
      equal (array float_exact) (Nx.to_array edges) (Nx.to_array read))

let missing =
  test "codec raises offline without the file" (fun () ->
      let cache_dir = Filename.temp_dir "aion" "empty" in
      raises_match Exn.failure (fun () ->
          Hub.codec ~cache_dir ~offline:true "FLUX_G"))

(* Scalar.load's three layouts of codebooks, planted in the cache. Row [i] holds
   [i + j / 10], so a transposed or reordered channel shows. *)
let book i n =
  Nx.create Nx.float32 [| n |]
    (Array.init n (fun j -> float_of_int i +. (float_of_int j /. 10.)))

let entries l = Nx_io.Archive.of_list (List.map (fun (k, v) -> (k, Nx.P v)) l)

let load m l =
  let cache_dir = cache_with (Scalar.directory m) (entries l) in
  Scalar.load ~cache_dir ~offline:true m

let rows c n = Array.concat (List.init c (fun i -> Nx.to_array (book i n)))

let layouts =
  cases "Scalar.load reads each layout of codebooks"
    ~id:(fun (n, _, _, _) -> n)
    [
      ( "one codebook beside its reservoir",
        Modality.Flux_g,
        [
          ("_quantizer._index_to_val", book 0 4);
          ("_quantizer._reservoir", book 7 9);
        ],
        1 );
      ("the redshift grid", Modality.Z, [ ("_quantizer.buckets", book 0 4) ], 1);
      ( "one codebook per channel",
        Modality.Xp_bp,
        List.init 55 (fun i ->
            (Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i, book i 4)),
        55 );
    ]
    (fun (_, m, l, c) ->
      let t = load m l in
      equal (array int) [| c; 4 |] (Nx.shape (Scalar.edges t));
      equal (array float_exact) (rows c 4) (Nx.to_array (Scalar.edges t)))

let refused =
  cases "Scalar.load refuses a file without its codebooks"
    ~id:(fun (n, _, _) -> n)
    [
      ("no codebook", Modality.Flux_g, [ ("_quantizer._reservoir", book 0 4) ]);
      ( "54 of 55 channels",
        Modality.Xp_rp,
        List.init 54 (fun i ->
            (Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i, book i 4))
      );
      ( "56 of 55 channels",
        Modality.Xp_rp,
        List.init 56 (fun i ->
            (Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i, book i 4))
      );
      ( "channels of two lengths",
        Modality.Xp_rp,
        List.init 55 (fun i ->
            ( Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i,
              book i (if i = 9 then 3 else 4) )) );
    ]
    (fun (_, m, l) -> raises_match Exn.failure (fun () -> load m l))

(* Each model's weights at its repository's commit. *)
let models =
  cases "weights reads each model's file at its commit"
    ~id:(fun (n, _, _, _) -> n)
    [
      ("base", Hub.Base, Hub.repo, Hub.revision);
      ( "large",
        Hub.Large,
        "polymathic-ai/aion-large",
        "cfdb89e8a22ddbdbbc4fe65c30fa2eb6eb7f9bcd" );
    ]
    (fun (_, m, repo, revision) ->
      let cache_dir = Filename.temp_dir "aion" "weights" in
      let path =
        Kaun_hf.cache_path ~cache_dir ~revision ~file:"model.safetensors" repo
      in
      mkdir_p (Filename.dirname path);
      Nx_io.save_safetensors path
        (Nx_io.Archive.of_list [ ("w", Nx.P edges) ]);
      let a = Hub.weights ~cache_dir ~offline:true m in
      equal (list string) [ "w" ] (Nx_io.Archive.names a);
      let empty = Filename.temp_dir "aion" "empty" in
      raises_match (Exn.failure ~sub:repo) (fun () ->
          Hub.weights ~cache_dir:empty ~offline:true m))

let () =
  exit
    (run "aion.hub"
       [
         group "codec" [ cached; missing ];
         group "load" [ layouts; refused ];
         models;
       ])
