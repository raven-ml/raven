(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* gpt-oss against the transformers implementation, on the two tiny random
   checkpoints, one with float experts and one with MXFP4 experts, eagerly on
   the host. [gen/reference.py] recorded the fixtures at float32 on the CPU.
   The packed checkpoint's scales are tiny, so its cases are also recorded
   with 118 added to every scale byte.

   The MoE block of one layer: the router's logits, experts and weights, and
   the block's output on a batch, a ragged pair and an input scaled until the
   activation clamps. The whole model, recorded with a sliding window of 4 so
   that the window binds on the 12-token prompt: the rotary frequencies, the
   attention scale and the sinks; each layer's attention alone on its
   recorded input; the argmax of every position and the largest logits of
   the last; the residual stream after every block; a ragged batch of two
   prompts. At bfloat16 the model keeps about three digits of the float32
   reference, the gap of its weights and activations. *)

open Windtrap
open Kaun
open Fixture

let float32 shape values = Nx.create Nx.float32 shape values

let ids_tensor rows =
  let batch = Array.length rows and seq = Array.length rows.(0) in
  Nx.create Nx.int64 [| batch; seq |]
    (Array.map Int64.of_int (Array.concat (Array.to_list rows)))

let tensor archive name =
  match Nx_io.Archive.find name archive with
  | Some (Nx.P t) -> Nx.cast Nx.float32 t
  | None -> failwith (name ^ ": no entry in the weights")

let bytes archive name =
  match Nx_io.Archive.find name archive with
  | Some e -> Nx.unpack Nx.uint8 e
  | None -> failwith (name ^ ": no entry in the weights")

(* The checkpoint's [[| ...; groups; 16 |]] blocks as [[| ...; inputs / 2 |]]
   codes, each scale byte plus [offset]. *)
let packed_weight archive ~offset name =
  let blocks = bytes archive (name ^ "_blocks") in
  let s = Nx.shape blocks and r = Nx.ndim blocks in
  let codes =
    Nx.reshape
      (Array.append (Array.sub s 0 (r - 2)) [| s.(r - 2) * 16 |])
      blocks
  in
  let scales = Nx.add_s (bytes archive (name ^ "_scales")) offset in
  Moe.Quant (Nx_quant.mxfp4 ~scales codes)

let float_weight archive ~offset:_ name = Moe.Float (tensor archive name)

let moe_params archive ~layer ~weight =
  let name leaf = Printf.sprintf "model.layers.%d.mlp.%s" layer leaf in
  let router =
    {
      Linear.w = Nx.transpose (tensor archive (name "router.weight"));
      b = Some (tensor archive (name "router.bias"));
    }
  in
  let experts =
    {
      Moe.gate_up = weight (name "experts.gate_up_proj");
      gate_up_bias = tensor archive (name "experts.gate_up_proj_bias");
      down = weight (name "experts.down_proj");
      down_bias = tensor archive (name "experts.down_proj_bias");
    }
  in
  (router, experts)

(* The MoE block *)

let tol = 1e-5

let block ~k ~limit (router, p) case () =
  let shape = ints (mem "shape" case) in
  let width = shape.(Array.length shape - 1) in
  let x = float32 shape (floats (mem "hidden" case)) in
  let x = Nx.reshape [| -1; width |] x in
  let logits = Linear.apply router x in
  close ~tol "router logits" (floats (mem "router_logits" case)) (flat logits);
  let ((ids, weights) as routing) = Moe.route ~k logits in
  equal ~msg:"selected experts" (array int)
    (ints (mem "experts" case))
    (Array.map Int64.to_int (Nx.to_array (Nx.reshape [| -1 |] ids)));
  close ~tol "expert weights"
    (floats (mem "expert_weights" case))
    (flat weights);
  close ~tol "output"
    (floats (mem "output" case))
    (flat (Moe.apply ~limit p routing x))

let blocks name ~weight =
  let fx = load name in
  let archive = lazy (weights fx) in
  let config = mem "config" fx in
  let k = to_int (mem "num_experts_per_tok" config) in
  let limit = to_number (mem "swiglu_limit" config) in
  let layer = to_int (mem "layer" fx) in
  List.concat_map
    (fun (offset, cases) ->
      let p =
        lazy
          (let archive = Lazy.force archive in
           moe_params archive ~layer
             ~weight:(weight archive ~offset:(int_of_string offset)))
      in
      List.map
        (fun (case, recorded) ->
          let label = if offset = "0" then case else "scales+118 " ^ case in
          test label (fun () -> block ~k ~limit (Lazy.force p) recorded ()))
        (members cases))
    (members (mem "cases" fx))

(* The whole model *)

let with_scale_offset offset (p : _ Gpt_oss.params) =
  let shift = function
    | Moe.Quant (Nx_quant.Mxfp4 { codes; scales }) ->
        Moe.Quant (Nx_quant.mxfp4 ~scales:(Nx.add_s scales offset) codes)
    | w -> w
  in
  let block (b : _ Gpt_oss.block) =
    let moe = b.moe in
    let moe =
      { moe with gate_up = shift moe.gate_up; down = shift moe.down }
    in
    { b with moe }
  in
  { p with blocks = List.map block p.blocks }

let rotary fx (cfg : _ Gpt_oss.config) () =
  let r = mem "rotary" fx in
  close ~tol:1e-6 "rotary frequencies"
    (floats (mem "inv_freq" r))
    (Rope.frequencies cfg.rope);
  (* The reference scales its cosines and sines by a concentration, which the
     model moves to the scores as its square. *)
  let concentration = to_number (mem "attention_scaling" r) in
  close ~tol:1e-7 "attention scale"
    [| (concentration ** 2.0) /. sqrt (float_of_int cfg.head_dim) |]
    [| cfg.attention_scale |]

let sinks fx (p : _ Gpt_oss.params) () =
  List.iteri
    (fun i (b : _ Gpt_oss.block) ->
      identical
        (Printf.sprintf "sinks of block %d" i)
        (floats (mem (string_of_int i) (mem "sinks" fx)))
        (flat b.sinks))
    p.blocks

(* Each layer's attention alone on its recorded input, and, at float32, that
   the window hides something on this prompt. *)
let attention (type b) ~tol ~exact fx case cfg p (dt : (float, b) Nx.dtype)
    () =
  let n = Array.length (ints (mem "ids" fx)) in
  List.iteri
    (fun i (layer, (b : _ Gpt_oss.block)) ->
      let recorded = mem (string_of_int i) (mem "attention" case) in
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
      let x = float32 [| 1; n; cfg.dim |] (floats (mem "input" recorded)) in
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

let logits ~tol ~exact fx case cfg p () =
  let tokens = ints (mem "ids" fx) in
  let n = Array.length tokens in
  let h = Gpt_oss.hidden cfg p (ids_tensor [| tokens |]) in
  let logits = Gpt_oss.logits cfg p h in
  let argmax =
    Array.map Int64.to_int (Nx.to_array (Nx.argmax ~axis:2 logits))
  in
  let top_ids = ints (mem "last_top_ids" case) in
  let top_values = floats (mem "last_top_values" case) in
  (if exact then
     equal ~msg:"argmax of every position" (array int)
       (ints (mem "argmax_per_position" case))
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
    (floats (mem "last_first_values" case))
    (Array.sub last 0 8)

let streams ~tol fx case cfg p () =
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
        (floats (mem (string_of_int k) (mem "hidden" case)))
        (flat (Nx.slice [ I 0; I (n - 1); R (0, 8) ] h)))
    (List.init layers (fun i -> i + 1))

(* The prompt and a shorter one, left-padded, in one batch. *)
let ragged ~tol fx case cfg p dt () =
  let tokens = ints (mem "ids" fx) and short = ints (mem "short_ids" fx) in
  let n = Array.length tokens and m = Array.length short in
  let padded = Array.append (Array.make (n - m) 0) short in
  let index = Cache_index.rows ~context:n [| n; m |] in
  let caches =
    Gpt_oss.cache ~placement:(fun _ ~axis:_ -> Nx.Host.on) cfg ~slots:(2 * n)
      dt
  in
  let ids = ids_tensor [| tokens; padded |] in
  let h, _ = Gpt_oss.cached cfg p caches index ids in
  let both = flat (Gpt_oss.logits cfg p (Nx.slice [ A; I (n - 1) ] h)) in
  let vocab = Array.length both / 2 in
  let row r key =
    Array.map (fun i -> both.((r * vocab) + i)) (ints (mem key case))
  in
  close ~tol "the long row"
    (floats (mem "last_top_values" case))
    (row 0 "last_top_ids");
  close ~tol "the short row"
    (floats (mem "short_top_values" case))
    (row 1 "short_top_ids")

let model (type b) name (dt : (float, b) Nx.dtype) =
  let fx = load name in
  let exact = Nx.Dtype.equal dt Nx.float32 in
  let tol = if exact then 1e-5 else 5e-2 in
  let loaded =
    lazy
      (let window = to_int (mem "sliding_window" fx) in
       let cfg = { (config fx) with window } in
       let host _ ~axis:_ = Nx.Host.on in
       (cfg, Gpt_oss.of_hf ~placement:host cfg dt (weights fx)))
  in
  let first f () =
    let cfg, p = Lazy.force loaded in
    f cfg p ()
  in
  [
    test "rotary frequencies and attention scale"
      (first (fun cfg _ -> rotary fx cfg));
    test "sinks" (first (fun _ p -> sinks fx p));
  ]
  @ List.concat_map
      (fun (offset, case) ->
        let label s = if offset = "0" then s else "scales+118, " ^ s in
        let run f () =
          let cfg, p = Lazy.force loaded in
          f cfg (with_scale_offset (int_of_string offset) p) ()
        in
        [
          test (label "attention of each block alone")
            (run (fun cfg p -> attention ~tol ~exact fx case cfg p dt));
          test (label "logits")
            (run (fun cfg p -> logits ~tol ~exact fx case cfg p));
          test (label "residual streams")
            (run (fun cfg p -> streams ~tol fx case cfg p));
          test (label "ragged batch")
            (run (fun cfg p -> ragged ~tol fx case cfg p dt));
        ])
      (members (mem "cases" fx))

let () =
  run "gpt-oss against transformers"
    [
      group "float experts" (blocks "gpt-oss-bf16" ~weight:float_weight);
      group "MXFP4 experts" (blocks "gpt-oss-mxfp4" ~weight:packed_weight);
      group "float model" (model "gpt-oss-bf16-model" Nx.float32);
      group "MXFP4 model" (model "gpt-oss-mxfp4-model" Nx.float32);
      group "float model, bfloat16" (model "gpt-oss-bf16-model" Nx.bfloat16);
      group "MXFP4 model, bfloat16" (model "gpt-oss-mxfp4-model" Nx.bfloat16);
    ]
  |> exit
