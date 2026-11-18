(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* gpt-oss against transformers recorded one block at a time, for checkpoints
   whose whole forward pass fits neither transformers nor an eager run here.
   [gen/reference_stream.py] recorded, for three prompts, the router's experts
   of every token in every block, the residual stream after each block's
   attention and after the block at a few positions (its mean, root mean
   square and largest magnitude, its first features and fixed projections on
   rows of +1 and -1), whole streams at the last position, the normalised last
   stream, and the largest logits and the argmax of every position.

   The checkpoint is imported at the dtype it stores, which copies nothing;
   each block runs compiled, at float32. A stream's error is in units of the
   reference stream's root mean square at that position, and must stay below
   2e-4: float32 against float32 differs by the kernels and the order of the
   sums, compounded over the blocks. Every expert choice and argmax is equal.

   The small MXFP4 checkpoint's recording runs in runtest. gpt-oss-20b's runs
   when its 13.8 GB are in raven's cache at the recorded revision; see the
   README. *)

open Windtrap
open Kaun
open Fixture

let tol = 2e-4

(* The projection rows of the recorder, from the same generator. *)
let signs ~rows ~dim =
  let state = ref 12345 in
  Array.init (rows * dim) (fun _ ->
      state := ((!state * 1103515245) + 12345) land 0x7FFFFFFF;
      if (!state lsr 16) land 1 = 1 then 1.0 else -1.0)

let worst_of a = Array.fold_left (fun m d -> if d <= m then m else d) 0.0 a

(* The error of row [t] of the stream [x], [tokens * dim] values, against a
   recorded summary. A value that is not finite gives an infinite error. *)
let summary_error ~signs ~dim summary x t =
  let v = Array.sub x (t * dim) dim in
  let n = float_of_int dim in
  let rms' = to_number (mem "rms" summary) in
  let mean = Array.fold_left ( +. ) 0.0 v /. n in
  let rms = sqrt (Array.fold_left (fun s e -> s +. (e *. e)) 0.0 v /. n) in
  let absmax = Array.fold_left (fun m e -> Float.max m (Float.abs e)) 0.0 v in
  let absmax' = to_number (mem "absmax" summary) in
  let projected j =
    let s = ref 0.0 in
    for i = 0 to dim - 1 do
      s := !s +. (signs.((j * dim) + i) *. v.(i))
    done;
    !s
  in
  let errors =
    Array.concat
      [
        Array.mapi
          (fun i e -> Float.abs (v.(i) -. e) /. rms')
          (floats (mem "first" summary));
        Array.mapi
          (fun j e -> Float.abs (projected j -. e) /. (rms' *. sqrt n))
          (floats (mem "projection" summary));
        [|
          Float.abs (mean -. to_number (mem "mean" summary)) /. rms';
          Float.abs (rms -. rms') /. rms';
          Float.abs (absmax -. absmax') /. absmax';
        |];
      ]
  in
  if Array.for_all Float.is_finite v then worst_of errors else Float.infinity

(* A whole row against a recorded one, in the same unit. *)
let row_error ~dim recorded x t =
  let v = Array.sub x (t * dim) dim in
  let rms' =
    sqrt
      (Array.fold_left (fun s e -> s +. (e *. e)) 0.0 recorded
      /. float_of_int dim)
  in
  if Array.for_all Float.is_finite v then
    worst_of (Array.mapi (fun i e -> Float.abs (v.(i) -. e) /. rms') recorded)
  else Float.infinity

let within msg e = less float_exact ~msg ~than:tol e

(* The router's experts as sets, token by token: [Nx.top_k] and torch.topk may
   order equal logits differently. *)
let routing msg ~k recorded experts =
  let set a t = List.sort compare (Array.to_list (Array.sub a (t * k) k)) in
  for t = 0 to (Array.length recorded / k) - 1 do
    equal
      ~msg:(Printf.sprintf "%s, token %d" msg t)
      (list int) (set recorded t) (set experts t)
  done

(* The configuration and weights [fx] was recorded from: a small checkpoint,
   or a repository read from raven's cache at its revision, never
   downloaded. *)
let source fx =
  match find "checkpoint" fx with
  | Some name -> checkpoint (to_string name)
  | None ->
      let repo = to_string (mem "repo" fx) in
      let revision = to_string (mem "revision" fx) in
      let config = Kaun_hf.load_config ~offline:true ~revision repo in
      ( Gpt_oss.config_of_json config,
        Kaun_hf.load_safetensors ~offline:true ~revision repo )

(* The prompt runner of [fx]'s checkpoint. Each block is cast to float32 for
   its call alone and runs compiled, one program for both layer kinds, as in
   {!Layer_loop}. *)
let stream fx =
  let cfg, weights = source fx in
  let (Gpt_oss.Dtype stored) = Gpt_oss.stored_dtype weights in
  let host _ ~axis:_ = Nx.Host.on in
  let p = Gpt_oss.of_hf ~placement:host cfg stored weights in
  let dim = cfg.dim and k = cfg.experts_per_token and f32 = Nx.float32 in
  let signs = signs ~rows:(to_int (mem "projections" fx)) ~dim in
  let compiled =
    Rune.jit
      Nx.Ptree.(
        instantiate (module Gpt_oss.Block)
        @-> v Cache_index.walk @-> tensor @-> returns tensor)
      (fun b index x ->
        let cache =
          Attention.Cache.make ~slots:0 ~kv_heads:cfg.n_kv_heads
            ~head_dim:cfg.head_dim f32 (Nx.placement x)
        in
        fst (Gpt_oss.block cfg b cache index x))
  in
  fun recorded () ->
    let ids = ints (mem "ids" recorded) in
    let tokens = Array.length ids in
    let positions = Array.to_list (ints (mem "positions" recorded)) in
    let last = tokens - 1 in
    let summaries recorded x =
      worst_of
        (Array.of_list
           (List.map2
              (fun s t -> summary_error ~signs ~dim s x t)
              (to_list recorded) positions))
    in
    let ids_t =
      Nx.create Nx.int64 [| 1; tokens |] (Array.map Int64.of_int ids)
    in
    let x =
      ref
        (Nx.cast f32
           (Nx.reshape [| tokens; dim |] (Embedding.apply p.tok ids_t)))
    in
    within "embedding" (summaries (mem "embedded" recorded) (flat !x));
    let whole = Cache_index.whole ~batch:1 ~seq:tokens () in
    let index = Gpt_oss.windows cfg whole in
    List.iteri
      (fun i (layer, (b, rb)) ->
        let block what = Printf.sprintf "block %d %s" i what in
        let b = Nx.Ptree.cast (module Gpt_oss.Block) f32 b in
        let x3 = Nx.reshape [| 1; tokens; dim |] !x in
        let cache =
          Attention.Cache.make ~slots:0 ~kv_heads:cfg.n_kv_heads
            ~head_dim:cfg.head_dim f32 Nx.Host.on
        in
        let attended, _ =
          Gpt_oss.attention cfg b cache (index layer)
            (Rms_norm.apply ~eps:cfg.norm_eps b.attn_norm x3)
        in
        let middle = Nx.add x3 attended in
        within (block "after attention")
          (summaries (mem "after_attention" rb) (flat middle));
        let experts, _ =
          Moe.route ~k
            (Linear.apply b.router
               (Nx.reshape [| tokens; dim |]
                  (Rms_norm.apply ~eps:cfg.norm_eps b.ffn_norm middle)))
        in
        routing (block "router") ~k
          (ints (mem "experts" rb))
          (Array.map Int64.to_int (Nx.to_array (Nx.reshape [| -1 |] experts)));
        let y = Nx.reshape [| tokens; dim |] (compiled b (index layer) x3) in
        let e = summaries (mem "after_block" rb) (flat y) in
        let e =
          match find "after_block_last" rb with
          | None -> e
          | Some row -> Float.max e (row_error ~dim (floats row) (flat y) last)
        in
        within (block "after experts") e;
        x := y;
        Gc.full_major ())
      (List.combine cfg.layers
         (List.combine p.blocks (to_list (mem "blocks" recorded))));
    let m =
      Nx.Ptree.cast (module Gpt_oss.Params) f32
        {
          Gpt_oss.tok =
            (if cfg.tied then p.tok
             else { Embedding.table = Nx.zeros stored [| 1; 1 |] });
          blocks = [];
          norm = p.norm;
          head = p.head;
        }
    in
    let normed = flat (Rms_norm.apply ~eps:cfg.norm_eps m.norm !x) in
    within "final norm"
      (Float.max (summaries (mem "normed" recorded) normed)
         (row_error ~dim (floats (mem "normed_last" recorded)) normed last));
    let logits = flat (Gpt_oss.logits cfg m !x) in
    let vocab = Array.length logits / tokens in
    List.iteri
      (fun t (ids, values) ->
        let scale = worst_of (Array.map Float.abs values) in
        Array.iteri
          (fun j id ->
            within
              (Printf.sprintf "logit %d of position %d" id t)
              (Float.abs (logits.((t * vocab) + id) -. values.(j)) /. scale))
          ids)
      (List.combine
         (List.map ints (to_list (mem "top_ids" recorded)))
         (List.map floats (to_list (mem "top_values" recorded))));
    let argmax t =
      let best = ref 0 in
      for v = 1 to vocab - 1 do
        if logits.((t * vocab) + v) > logits.((t * vocab) + !best) then
          best := v
      done;
      !best
    in
    equal ~msg:"argmax of every position" (array int)
      (ints (mem "argmax_per_position" recorded))
      (Array.init tokens argmax)

let prompts ?(skip = fun () -> ()) name =
  let fx = load name in
  let run = lazy (stream fx) in
  List.map
    (fun (prompt, recorded) ->
      test ~timeout:3600. prompt (fun () ->
          skip ();
          Lazy.force run recorded ()))
    (members (mem "prompts" fx))

(* gpt-oss-20b is read from the cache only, never downloaded here. *)
let in_cache name () =
  let fx = load name in
  let repo = to_string (mem "repo" fx) in
  let revision = to_string (mem "revision" fx) in
  let config = Kaun_hf.cache_path ~revision ~file:"config.json" repo in
  if not (Sys.file_exists config) then
    skip ~reason:(repo ^ " is not in the cache at " ^ revision) ()

let () =
  run "gpt-oss streamed against transformers"
    [
      group "small MXFP4" (prompts "gpt-oss-mxfp4-stream");
      group "gpt-oss-20b"
        (prompts ~skip:(in_cache "gpt-oss-20b-stream") "gpt-oss-20b-stream");
    ]
  |> exit
