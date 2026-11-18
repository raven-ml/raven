(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* gpt-oss against the transformers implementation, eagerly on the host, on
   two small checkpoints of random weights, one with float experts and one
   with MXFP4 experts, as [Gpt_oss.of_hf] loads them. [gen/reference.py]
   wrote the checkpoints and recorded the fixtures at float32 on the CPU.

   The MoE block of one layer: the router's logits, experts and weights, and
   the block's output on a batch, a ragged pair and an input scaled until the
   activation clamps. The whole model, whose sliding window of 4 binds on the
   12-token prompt: each layer's attention alone on its recorded input; the
   argmax of every position and the largest logits of the last; the residual
   stream after every block; a ragged batch of two prompts. At bfloat16 the
   model keeps about three digits of the float32 reference, the gap of its
   weights and activations. *)

open Windtrap
open Kaun
open Fixture

let ids_tensor rows =
  let batch = Array.length rows and seq = Array.length rows.(0) in
  Nx.create Nx.int64 [| batch; seq |]
    (Array.map Int64.of_int (Array.concat (Array.to_list rows)))

let host _ ~axis:_ = Nx.Host.on

(* The MoE block *)

let block ~tol cfg (b : _ Gpt_oss.block) case () =
  let shape = ints (mem "shape" case) in
  let hidden = Nx.create Nx.float32 shape (floats (mem "hidden" case)) in
  let x = Nx.reshape [| -1; shape.(Array.length shape - 1) |] hidden in
  let logits = Linear.apply b.router x in
  close ~tol "router logits" (floats (mem "router_logits" case)) (flat logits);
  let ((ids, weights) as routing) =
    Moe.route ~k:cfg.Gpt_oss.experts_per_token logits
  in
  equal ~msg:"selected experts" (array int)
    (ints (mem "experts" case))
    (Array.map Int64.to_int (Nx.to_array (Nx.reshape [| -1 |] ids)));
  close ~tol "expert weights"
    (floats (mem "expert_weights" case))
    (flat weights);
  close ~tol "output"
    (floats (mem "output" case))
    (flat (Moe.apply ~limit:cfg.swiglu_limit b.moe routing x))

(* The whole model *)

(* Each layer's attention alone on its recorded input, and, at float32, that
   the window hides something on this prompt. *)
let attention (type b) ~tol ~exact fx cfg p (dt : (float, b) Nx.dtype) () =
  let n = Array.length (ints (mem "ids" fx)) in
  List.iteri
    (fun i (layer, (b : _ Gpt_oss.block)) ->
      let recorded = mem (string_of_int i) (mem "attention" fx) in
      let attend layer x =
        let cache =
          Attention.Cache.make ~slots:0 ~kv_heads:cfg.Gpt_oss.n_kv_heads
            ~head_dim:cfg.head_dim dt Nx.Host.on
        in
        let index = Cache_index.whole ~batch:1 ~seq:n () in
        flat
          (fst
             (Gpt_oss.attention cfg b cache
                (Gpt_oss.windows cfg index layer)
                (Nx.cast dt x)))
      in
      let x =
        Nx.create Nx.float32 [| 1; n; cfg.dim |] (floats (mem "input" recorded))
      in
      let want = floats (mem "output" recorded) in
      close ~tol
        (Printf.sprintf "attention of block %d" i)
        want (attend layer x);
      if layer = Gpt_oss.Sliding && exact then
        let unbound = attend Gpt_oss.Full x in
        let gap = ref 0.0 in
        Array.iteri
          (fun j e -> gap := Float.max !gap (Float.abs (e -. unbound.(j))))
          want;
        greater float_exact ~msg:"the window binds on this prompt" ~than:1e-3
          !gap)
    (List.combine cfg.layers p.Gpt_oss.blocks)

let logits ~tol ~exact fx cfg p () =
  let tokens = ints (mem "ids" fx) in
  let n = Array.length tokens in
  let h = Gpt_oss.hidden cfg p (ids_tensor [| tokens |]) in
  let logits = Gpt_oss.logits cfg p h in
  let argmax =
    Array.map Int64.to_int (Nx.to_array (Nx.argmax ~axis:2 logits))
  in
  let top_ids = ints (mem "last_top_ids" fx) in
  let top_values = floats (mem "last_top_values" fx) in
  (if exact then
     equal ~msg:"argmax of every position" (array int)
       (ints (mem "argmax_per_position" fx))
       argmax
   else
     (* At half precision the last argmax may be any of the reference's
        nearly tied largest logits. *)
     let margin =
       match Array.find_index (fun id -> id = argmax.(n - 1)) top_ids with
       | None -> infinity
       | Some i ->
           (top_values.(0) -. top_values.(i)) /. Float.abs top_values.(0)
     in
     less float_exact ~msg:"argmax of the last position" ~than:tol margin);
  let last = flat (Nx.slice [ I 0; I (n - 1) ] logits) in
  close ~tol "the largest logits of the last position" top_values
    (Array.map (fun i -> last.(i)) top_ids);
  close ~tol "the first logits of the last position"
    (floats (mem "last_first_values" fx))
    (Array.sub last 0 8)

let streams ~tol fx cfg p () =
  let tokens = ints (mem "ids" fx) in
  let n = Array.length tokens in
  let layers = List.length cfg.Gpt_oss.layers in
  List.iter
    (fun k ->
      let first l = List.filteri (fun i _ -> i < k) l in
      let sub = { p with Gpt_oss.blocks = first p.Gpt_oss.blocks } in
      let cfg' = { cfg with layers = first cfg.layers } in
      let h = Gpt_oss.hidden cfg' sub (ids_tensor [| tokens |]) in
      let h =
        if k = layers then Rms_norm.apply ~eps:cfg.norm_eps p.norm h else h
      in
      close ~tol
        (Printf.sprintf "residual stream after block %d" k)
        (floats (mem (string_of_int k) (mem "hidden" fx)))
        (flat (Nx.slice [ I 0; I (n - 1); R (0, 8) ] h)))
    (List.init layers (fun i -> i + 1))

(* The prompt and a shorter one, left-padded, in one batch. *)
let ragged ~tol fx cfg p dt () =
  let tokens = ints (mem "ids" fx) and short = ints (mem "short_ids" fx) in
  let n = Array.length tokens and m = Array.length short in
  let padded = Array.append (Array.make (n - m) 0) short in
  let index = Cache_index.rows ~context:n [| n; m |] in
  let caches = Gpt_oss.cache ~placement:host cfg ~slots:(2 * n) dt in
  let ids = ids_tensor [| tokens; padded |] in
  let h, _ = Gpt_oss.cached cfg p caches index ids in
  let both = flat (Gpt_oss.logits cfg p (Nx.slice [ A; I (n - 1) ] h)) in
  let vocab = Array.length both / 2 in
  let row r key =
    Array.map (fun i -> both.((r * vocab) + i)) (ints (mem key fx))
  in
  close ~tol "the long row"
    (floats (mem "last_top_values" fx))
    (row 0 "last_top_ids");
  close ~tol "the short row"
    (floats (mem "short_top_values" fx))
    (row 1 "short_top_ids")

let load_model name dt =
  lazy
    (let cfg, weights = checkpoint name in
     (cfg, Gpt_oss.of_hf ~placement:host cfg dt weights))

let blocks name =
  let fx = load name in
  let loaded = load_model name Nx.float32 in
  List.map
    (fun (case, recorded) ->
      test case (fun () ->
          let cfg, p = Lazy.force loaded in
          let b = List.nth p.Gpt_oss.blocks (to_int (mem "layer" fx)) in
          block ~tol:1e-5 cfg b recorded ()))
    (members (mem "block" fx))

let model (type b) name (dt : (float, b) Nx.dtype) =
  let fx = load name in
  let exact = Nx.Dtype.equal dt Nx.float32 in
  let tol = if exact then 1e-5 else 5e-2 in
  let loaded = load_model name dt in
  let run f () =
    let cfg, p = Lazy.force loaded in
    f cfg p ()
  in
  [
    test "attention of each block alone"
      (run (fun cfg p -> attention ~tol ~exact fx cfg p dt));
    test "logits" (run (fun cfg p -> logits ~tol ~exact fx cfg p));
    test "residual streams" (run (fun cfg p -> streams ~tol fx cfg p));
    test "ragged batch" (run (fun cfg p -> ragged ~tol fx cfg p dt));
  ]

let () =
  run "gpt-oss against transformers"
    [
      group "float experts, MoE block" (blocks "gpt-oss-bf16");
      group "MXFP4 experts, MoE block" (blocks "gpt-oss-mxfp4");
      group "float experts" (model "gpt-oss-bf16" Nx.float32);
      group "MXFP4 experts" (model "gpt-oss-mxfp4" Nx.float32);
      group "float experts, bfloat16" (model "gpt-oss-bf16" Nx.bfloat16);
      group "MXFP4 experts, bfloat16" (model "gpt-oss-mxfp4" Nx.bfloat16);
    ]
  |> exit
