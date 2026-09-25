(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What each layer's and each index's walk visits: its leaf paths and its
   reports, and the checkpoint names they give. *)

open Windtrap
open Kaun

let f32 = Nx.float32

let visits s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

let names s x = Checkpoint.names (Checkpoint.of_value s x)

(* Each layer's checkpoint names are the ones its traversals gave before
   structures had a walk, sorted. *)
let check_layer ~msg s x ~visits:expected ~names:old =
  equal ~msg:(msg ^ " visits") (list string) expected (visits s x);
  equal ~msg:(msg ^ " names") (list string) (List.sort compare old) (names s x)

let leaves paths = List.map (fun p -> p ^ ": a leaf") paths

(* Layers *)

let test_linear () =
  let linear = Nx.Ptree.instantiate (module Linear) in
  check_layer ~msg:"with a bias" linear
    (Linear.make ~inputs:3 ~outputs:2 f32)
    ~visits:[ "w: a leaf"; "b: Some"; "b: a leaf" ]
    ~names:[ "w"; "b" ];
  check_layer ~msg:"without a bias" linear
    (Linear.make ~bias:false ~inputs:3 ~outputs:2 f32)
    ~visits:[ "w: a leaf"; "b: None" ] ~names:[ "w" ]

let test_conv () =
  let conv = Nx.Ptree.instantiate (module Conv) in
  check_layer ~msg:"conv" conv
    (Conv.make ~in_channels:1 ~out_channels:2 ~kernel_size:(3, 3) f32)
    ~visits:[ "w: a leaf"; "b: Some"; "b: a leaf" ]
    ~names:[ "w"; "b" ];
  check_layer ~msg:"conv without bias" conv
    (Conv.make ~bias:false ~in_channels:1 ~out_channels:2 ~kernel_size:(3, 3)
       f32)
    ~visits:[ "w: a leaf"; "b: None" ] ~names:[ "w" ]

let test_norms () =
  check_layer ~msg:"embedding"
    (Nx.Ptree.instantiate (module Embedding))
    (Embedding.make ~vocab:5 ~dim:2 f32)
    ~visits:(leaves [ "table" ]) ~names:[ "table" ];
  check_layer ~msg:"layer norm"
    (Nx.Ptree.instantiate (module Layer_norm))
    (Layer_norm.make ~dim:2 f32)
    ~visits:(leaves [ "gamma"; "beta" ])
    ~names:[ "gamma"; "beta" ];
  check_layer ~msg:"rms norm"
    (Nx.Ptree.instantiate (module Rms_norm))
    (Rms_norm.make ~dim:2 f32) ~visits:(leaves [ "gamma" ]) ~names:[ "gamma" ];
  let p, s = Batch_norm.init ~features:2 in
  check_layer ~msg:"batch norm"
    (Nx.Ptree.instantiate (module Batch_norm))
    p
    ~visits:(leaves [ "gamma"; "beta" ])
    ~names:[ "gamma"; "beta" ];
  check_layer ~msg:"batch norm statistics"
    (Nx.Ptree.instantiate (module Batch_norm.Stats))
    s
    ~visits:(leaves [ "mean"; "var" ])
    ~names:[ "mean"; "var" ]

let test_attention () =
  let paths = [ "q.w"; "q.b"; "k.w"; "k.b"; "v.w"; "v.b"; "out.w"; "out.b" ] in
  let with_bias proj =
    [ proj ^ ".w: a leaf"; proj ^ ".b: Some"; proj ^ ".b: a leaf" ]
  in
  check_layer ~msg:"attention"
    (Nx.Ptree.instantiate (module Attention))
    (Attention.make ~embed_dim:4 f32)
    ~visits:(List.concat_map with_bias [ "q"; "k"; "v"; "out" ])
    ~names:paths;
  let c = Attention.Cache.make ~slots:2 ~kv_heads:1 ~head_dim:2 f32 in
  let cache = Nx.Ptree.instantiate (module Attention.Cache) in
  check_layer ~msg:"cache" cache c
    ~visits:(leaves [ "keys"; "values" ])
    ~names:[ "keys"; "values" ];
  check_layer ~msg:"a decoder's caches" (Nx.Ptree.list cache) [ c; c ]
    ~visits:
      ("the root: length 2"
      :: leaves [ "0.keys"; "0.values"; "1.keys"; "1.values" ])
    ~names:[ "0.keys"; "0.values"; "1.keys"; "1.values" ]

let test_cast () =
  let p = Attention.make ~embed_dim:4 f32 in
  let half = Nx.Ptree.cast (module Attention) Nx.bfloat16 p in
  let dtypes =
    Nx.Ptree.fold
      (Nx.Ptree.instantiate (module Attention))
      (fun _ x acc -> Nx_dtype.to_string (Nx.dtype x) :: acc)
      half []
  in
  equal ~msg:"every projection is cast" (list string)
    (List.init 8 (fun _ -> "bfloat16"))
    dtypes

(* Cache indices *)

let int32s shape xs = Nx.create Nx.int32 shape xs

let test_index_whole () =
  equal ~msg:"whole" (list string)
    [
      "tokens: case \"whole\"";
      "tokens.pos: a leaf";
      "every: int 1";
      "window: None";
      "columns: None";
    ]
    (visits Cache_index.ptree (Cache_index.whole ~batch:2 ~seq:3 ()))

let test_index_tabled () =
  let index =
    Cache_index.window 2
      (Cache_index.every 4
         (Cache_index.rows ~every:[ 4; 2 ] ~context:8 [| 3; 2 |]))
  in
  equal ~msg:"rows, read in blocks of 4 under a window" (list string)
    [
      "tokens: case \"tabled\"";
      "tokens.row: None";
      "tokens.pos: a leaf";
      "tokens.table: a leaf";
      "tokens.blocks: length 2";
      "tokens.blocks.0: int 4";
      "tokens.blocks.0: a leaf";
      "tokens.blocks.1: int 2";
      "tokens.blocks.1: a leaf";
      "every: int 4";
      "window: Some";
      "window: int 2";
      "columns: None";
    ]
    (visits Cache_index.ptree index);
  let made =
    Cache_index.make ~row:(int32s [| 1 |] [| 0l |])
      ~pos:(int32s [| 1; 1 |] [| 0l |])
      ~table:(int32s [| 2; 2 |] [| 0l; 1l; 2l; 3l |])
      ()
  in
  let selected = Cache_index.select (int32s [| 1; 1; 1 |] [| 0l |]) made in
  equal ~msg:"a row and a selection" (list string)
    [
      "tokens: case \"tabled\"";
      "tokens.row: Some";
      "tokens.row: a leaf";
      "tokens.pos: a leaf";
      "tokens.table: a leaf";
      "tokens.blocks: length 0";
      "every: int 1";
      "window: None";
      "columns: Some";
      "columns: a leaf";
    ]
    (visits Cache_index.ptree selected)

let test_index_keys () =
  (* Everything a compiled program depends on changes what the index visits. *)
  let index = Cache_index.rows ~every:[ 4 ] ~context:8 [| 3; 2 |] in
  let differ ~msg a b =
    is_true ~msg (visits Cache_index.ptree a <> visits Cache_index.ptree b)
  in
  differ ~msg:"a window" index (Cache_index.window 2 index);
  differ ~msg:"another window"
    (Cache_index.window 2 index)
    (Cache_index.window 3 index);
  differ ~msg:"the blocks read" index (Cache_index.every 4 index);
  differ ~msg:"a block size"
    (Cache_index.rows ~every:[ 4 ] ~context:8 [| 3; 2 |])
    (Cache_index.rows ~every:[ 2 ] ~context:8 [| 3; 2 |]);
  differ ~msg:"whole or tabled" (Cache_index.whole ~batch:2 ~seq:3 ()) index

let test_index_round_trip () =
  let round_trip ~msg index =
    let leaves, _ = Nx.Ptree.flatten Cache_index.ptree index in
    let copy =
      Nx.Ptree.rebuild Cache_index.ptree ~like:index
        (List.map (fun (Nx.P x) -> Nx.P (Nx.copy x)) leaves)
    in
    equal
      ~msg:(msg ^ ": the copy visits what the index visits")
      (list string)
      (visits Cache_index.ptree index)
      (visits Cache_index.ptree copy);
    let values l =
      List.map
        (fun (Nx.P x) -> Nx.to_array (Nx.cast Nx.float64 x))
        (fst (Nx.Ptree.flatten Cache_index.ptree l))
    in
    equal
      ~msg:(msg ^ ": and holds the same tensors")
      (list (array float_exact))
      (values index) (values copy);
    equal
      ~msg:(msg ^ ": and masks alike")
      (array bool)
      (Nx.to_array (Cache_index.mask index))
      (Nx.to_array (Cache_index.mask copy))
  in
  round_trip ~msg:"blocks and a window"
    (Cache_index.window 2 (Cache_index.rows ~every:[ 4 ] ~context:8 [| 3; 2 |]));
  round_trip ~msg:"a row and a selection"
    (Cache_index.select
       (int32s [| 1; 1; 1 |] [| 1l |])
       (Cache_index.make ~row:(int32s [| 1 |] [| 1l |])
          ~pos:(int32s [| 1; 1 |] [| 1l |])
          ~table:(int32s [| 2; 2 |] [| 0l; 1l; 2l; 3l |])
          ()))

let tests =
  [
    group "layers"
      [
        test "linear" test_linear;
        test "conv" test_conv;
        test "embeddings and norms" test_norms;
        test "attention and caches" test_attention;
        test "a layer casts through its walk" test_cast;
      ];
    group "cache index"
      [
        test "a whole index" test_index_whole;
        test "a tabled index" test_index_tabled;
        test "what a program depends on is visited" test_index_keys;
        test "rebuilt from its tensors" test_index_round_trip;
      ];
  ]

let () = run "kaun structures" tests
