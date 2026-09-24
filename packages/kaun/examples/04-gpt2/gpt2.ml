(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun
module Hf = Kaun_hf

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Configuration *)

type config = {
  vocab_size : int;
  n_positions : int;
  n_embd : int;
  n_layer : int;
  n_head : int;
  n_inner : int;
  layer_norm_eps : float;
}

(* Model: plain records of kaun layers, generic over the float dtype *)

type 'a block = {
  ln1 : 'a Layer_norm.t;
  attn : 'a Attention.t;
  ln2 : 'a Layer_norm.t;
  fc : 'a Linear.t;
  proj : 'a Linear.t;
}

type 'a params = {
  wte : 'a Embedding.t;
  wpe : 'a Embedding.t;
  blocks : 'a block list;
  ln_f : 'a Layer_norm.t;
}

type t = Nx.float32_t params
type role = Whole | Column | Row | Kv_heads

module Params = struct
  type nonrec 'a t = 'a params

  let map_block f b =
    let ln1 = Layer_norm.map f b.ln1 in
    let attn = Attention.map f b.attn in
    let ln2 = Layer_norm.map f b.ln2 in
    let fc = Linear.map f b.fc in
    let proj = Linear.map f b.proj in
    { ln1; attn; ln2; fc; proj }

  let map f p =
    let wte = Embedding.map f p.wte in
    let wpe = Embedding.map f p.wpe in
    let blocks = List.map (map_block f) p.blocks in
    let ln_f = Layer_norm.map f p.ln_f in
    { wte; wpe; blocks; ln_f }

  let map2_block f b b' =
    let ln1 = Layer_norm.map2 f b.ln1 b'.ln1 in
    let attn = Attention.map2 f b.attn b'.attn in
    let ln2 = Layer_norm.map2 f b.ln2 b'.ln2 in
    let fc = Linear.map2 f b.fc b'.fc in
    let proj = Linear.map2 f b.proj b'.proj in
    { ln1; attn; ln2; fc; proj }

  let map2 f p p' =
    let wte = Embedding.map2 f p.wte p'.wte in
    let wpe = Embedding.map2 f p.wpe p'.wpe in
    let blocks = List.map2 (map2_block f) p.blocks p'.blocks in
    let ln_f = Layer_norm.map2 f p.ln_f p'.ln_f in
    { wte; wpe; blocks; ln_f }

  let iter_block f b =
    Layer_norm.iter f b.ln1;
    Attention.iter f b.attn;
    Layer_norm.iter f b.ln2;
    Linear.iter f b.fc;
    Linear.iter f b.proj

  let iter f p =
    Embedding.iter f p.wte;
    Embedding.iter f p.wpe;
    List.iter (iter_block f) p.blocks;
    Layer_norm.iter f p.ln_f

  let fold_block f acc b =
    let acc = Layer_norm.fold (fun s -> f ("ln1." ^ s)) acc b.ln1 in
    let acc = Attention.fold (fun s -> f ("attn." ^ s)) acc b.attn in
    let acc = Layer_norm.fold (fun s -> f ("ln2." ^ s)) acc b.ln2 in
    let acc = Linear.fold (fun s -> f ("fc." ^ s)) acc b.fc in
    Linear.fold (fun s -> f ("proj." ^ s)) acc b.proj

  let fold f acc p =
    let acc = Embedding.fold (fun s -> f ("wte." ^ s)) acc p.wte in
    let acc = Embedding.fold (fun s -> f ("wpe." ^ s)) acc p.wpe in
    let _, acc =
      List.fold_left
        (fun (i, acc) b ->
          let pre = Printf.sprintf "blocks.%d." i in
          (i + 1, fold_block (fun s -> f (pre ^ s)) acc b))
        (0, acc) p.blocks
    in
    Layer_norm.fold (fun s -> f ("ln_f." ^ s)) acc p.ln_f

  let fold2_block f acc b b' =
    let acc = Layer_norm.fold2 (fun s -> f ("ln1." ^ s)) acc b.ln1 b'.ln1 in
    let acc = Attention.fold2 (fun s -> f ("attn." ^ s)) acc b.attn b'.attn in
    let acc = Layer_norm.fold2 (fun s -> f ("ln2." ^ s)) acc b.ln2 b'.ln2 in
    let acc = Linear.fold2 (fun s -> f ("fc." ^ s)) acc b.fc b'.fc in
    Linear.fold2 (fun s -> f ("proj." ^ s)) acc b.proj b'.proj

  let fold2 f acc p p' =
    let acc = Embedding.fold2 (fun s -> f ("wte." ^ s)) acc p.wte p'.wte in
    let acc = Embedding.fold2 (fun s -> f ("wpe." ^ s)) acc p.wpe p'.wpe in
    let _, acc =
      List.fold_left2
        (fun (i, acc) b b' ->
          let pre = Printf.sprintf "blocks.%d." i in
          (i + 1, fold2_block (fun s -> f (pre ^ s)) acc b b'))
        (0, acc) p.blocks p'.blocks
    in
    Layer_norm.fold2 (fun s -> f ("ln_f." ^ s)) acc p.ln_f p'.ln_f

  let names_block i b =
    let pre field s = Printf.sprintf "blocks.%d.%s.%s" i field s in
    {
      ln1 = Layer_norm.map (pre "ln1") (Layer_norm.names b.ln1);
      attn = Attention.map (pre "attn") (Attention.names b.attn);
      ln2 = Layer_norm.map (pre "ln2") (Layer_norm.names b.ln2);
      fc = Linear.map (pre "fc") (Linear.names b.fc);
      proj = Linear.map (pre "proj") (Linear.names b.proj);
    }

  let names p =
    let pre field s = field ^ "." ^ s in
    {
      wte = Embedding.map (pre "wte") (Embedding.names p.wte);
      wpe = Embedding.map (pre "wpe") (Embedding.names p.wpe);
      blocks = List.mapi names_block p.blocks;
      ln_f = Layer_norm.map (pre "ln_f") (Layer_norm.names p.ln_f);
    }
end

let make cfg =
  let zeros = Init.zeros in
  let linear ~inputs ~outputs =
    Linear.make ~w_init:zeros ~bias_init:zeros ~inputs ~outputs Nx.float32
  in
  let embedding ~vocab = Embedding.make ~init:zeros ~vocab ~dim:cfg.n_embd in
  let block () =
    {
      ln1 = Layer_norm.init ~dim:cfg.n_embd;
      attn =
        Attention.make ~w_init:zeros ~bias_init:zeros ~embed_dim:cfg.n_embd
          Nx.float32;
      ln2 = Layer_norm.init ~dim:cfg.n_embd;
      fc = linear ~inputs:cfg.n_embd ~outputs:cfg.n_inner;
      proj = linear ~inputs:cfg.n_inner ~outputs:cfg.n_embd;
    }
  in
  {
    wte = embedding ~vocab:cfg.vocab_size Nx.float32;
    wpe = embedding ~vocab:cfg.n_positions Nx.float32;
    blocks = List.init cfg.n_layer (fun _ -> block ());
    ln_f = Layer_norm.init ~dim:cfg.n_embd;
  }

(* Forward pass. Training passes [?dropout] — a rate and a [Nx.Rng] key — to
   enable the canonical GPT-2 dropout sites: the embedding sum and each block's
   post-attention and post-MLP projections. Every site's mask derives from the
   one key by [Nx.Rng.fold_in], so a single per-step key drives them all; under
   [Rune.jit] that key must be an input leaf of the step. Inference (the
   default) applies none. *)

let drop dropout i x =
  match dropout with
  | None -> x
  | Some (rate, key) ->
      Dropout.apply ~rate ~training:true ~key:(Nx.Rng.fold_in key i) x

let head_dim cfg = cfg.n_embd / cfg.n_head

let block cfg ?dropout b cache index x =
  let eps = cfg.layer_norm_eps in
  let a, cache =
    Attention.cached ~head_dim:(head_dim cfg) b.attn cache index
      (Layer_norm.apply ~eps b.ln1 x)
  in
  let x = Nx.add x (drop dropout 0 a) in
  ( Nx.add x
      (drop dropout 1
         (Linear.apply b.proj
            (Fn.gelu_approx (Linear.apply b.fc (Layer_norm.apply ~eps b.ln2 x))))),
    cache )

let embed p ids pos =
  Nx.add (Embedding.apply p.wte ids) (Embedding.apply p.wpe pos)

(* One fold over the blocks, which threads the caches along the index. *)

module Cache = Attention.Cache.List

let cache ?placement cfg ~slots dtype =
  let place x =
    match placement with None -> x | Some p -> Nx.place (p Kv_heads ~axis:1) x
  in
  List.init cfg.n_layer (fun _ ->
      Attention.Cache.map place
        (Attention.Cache.make ~slots ~kv_heads:cfg.n_head
           ~head_dim:(head_dim cfg) dtype))

let cached cfg ?dropout p caches index ids =
  if Cache_index.context index > cfg.n_positions then
    invalid_argf "Gpt2.cached: %d positions exceed n_positions %d"
      (Cache_index.context index)
      cfg.n_positions;
  (* Per-consumer subkeys: index 0 feeds the embedding dropout, index i + 1
     block i (which folds again per site). *)
  let sub i =
    Option.map (fun (rate, key) -> (rate, Nx.Rng.fold_in key i)) dropout
  in
  let _, x, rev_caches =
    List.fold_left2
      (fun (i, x, cs) b c ->
        let x, c = block cfg ?dropout:(sub i) b c index x in
        (i + 1, x, c :: cs))
      (1, drop dropout 0 (embed p ids (Cache_index.positions index)), [])
      p.blocks caches
  in
  (x, List.rev rev_caches)

let hidden cfg ?dropout p ids =
  let batch = Nx.dim 0 ids and seq = Nx.dim 1 ids in
  let nothing = cache cfg ~slots:0 (Nx.dtype p.ln_f.Layer_norm.gamma) in
  fst (cached cfg ?dropout p nothing (Cache_index.whole ~batch ~seq ()) ids)

let logits cfg p h =
  (* Tied LM head: logits = h @ wteᵀ. *)
  Nx.matmul
    (Layer_norm.apply ~eps:cfg.layer_norm_eps p.ln_f h)
    (Nx.transpose p.wte.table)

(* Importing a HuggingFace checkpoint.

   The file names its tensors h.{i}.attn.c_attn.weight, h.{i}.mlp.c_fc.bias, ...
   and fuses the q, k and v projections into c_attn, [n_embd; 3 * n_embd]. Its
   weights are already [inputs; outputs], so only the fused projection is cut,
   with [Nx.split], into three views. *)

let of_hf ?placement cfg dt ckpt =
  let place role ~axis x =
    match placement with None -> x | Some p -> Nx.place (p role ~axis) x
  in
  let whole x = place Whole ~axis:0 x in
  let float ~shape name = Checkpoint.to_float ~shape dt name ckpt in
  let d = cfg.n_embd in
  let layer_norm name =
    Layer_norm.map whole
      {
        Layer_norm.gamma = float ~shape:[| d |] (name ^ ".weight");
        beta = float ~shape:[| d |] (name ^ ".bias");
      }
  in
  let stored ~inputs ~outputs name =
    {
      Linear.w = float ~shape:[| inputs; outputs |] (name ^ ".weight");
      b = Some (float ~shape:[| outputs |] (name ^ ".bias"));
    }
  in
  (* A column projection is cut along its outputs, bias included; a row
     projection along its inputs, and its bias is added whole. *)
  let column { Linear.w; b } =
    {
      Linear.w = place Column ~axis:1 w;
      b = Option.map (place Column ~axis:0) b;
    }
  in
  let row { Linear.w; b } =
    { Linear.w = place Row ~axis:0 w; b = Option.map whole b }
  in
  let block i =
    let at leaf = Printf.sprintf "h.%d.%s" i leaf in
    let fused = stored ~inputs:d ~outputs:(3 * d) (at "attn.c_attn") in
    let q, k, v =
      match
        List.map2
          (fun w b -> column { Linear.w; b = Some b })
          (Nx.split ~axis:1 3 fused.w)
          (Nx.split ~axis:0 3 (Option.get fused.b))
      with
      | [ q; k; v ] -> (q, k, v)
      | _ -> assert false
    in
    {
      ln1 = layer_norm (at "ln_1");
      attn =
        { q; k; v; out = row (stored ~inputs:d ~outputs:d (at "attn.c_proj")) };
      ln2 = layer_norm (at "ln_2");
      fc = column (stored ~inputs:d ~outputs:cfg.n_inner (at "mlp.c_fc"));
      proj = row (stored ~inputs:cfg.n_inner ~outputs:d (at "mlp.c_proj"));
    }
  in
  {
    wte =
      {
        Embedding.table =
          whole (float ~shape:[| cfg.vocab_size; d |] "wte.weight");
      };
    wpe =
      {
        Embedding.table =
          whole (float ~shape:[| cfg.n_positions; d |] "wpe.weight");
      };
    blocks = List.init cfg.n_layer block;
    ln_f = layer_norm "ln_f";
  }

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let dtype_of_string = function
  | "float32" -> Dtype Nx.float32
  | "float16" -> Dtype Nx.float16
  | "bfloat16" -> Dtype Nx.bfloat16
  | d -> failwith ("--dtype must be float32, float16 or bfloat16, got " ^ d)

let stored_dtype ckpt =
  let (Rune.Ptree.P table) = Checkpoint.get "wte.weight" ckpt in
  match Nx.dtype table with
  | Nx.Float16 -> Dtype Nx.float16
  | Nx.BFloat16 -> Dtype Nx.bfloat16
  | Nx.Float32 -> Dtype Nx.float32
  | _ ->
      failwith
        "the checkpoint's embedding table is not a float16, bfloat16 or \
         float32 entry"

(* Pretrained loading *)

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

let json_int ~default json name =
  match json_mem name json with
  | Jsont.Number (f, _) -> int_of_float f
  | _ -> default ()

let config_of_json json =
  let req name =
    json_int
      ~default:(fun () -> failwith ("gpt2 config.json: missing " ^ name))
      json name
  in
  let n_embd = req "n_embd" in
  {
    vocab_size = req "vocab_size";
    n_positions = json_int ~default:(fun () -> 1024) json "n_positions";
    n_embd;
    n_layer = req "n_layer";
    n_head = req "n_head";
    n_inner = json_int ~default:(fun () -> 4 * n_embd) json "n_inner";
    layer_norm_eps =
      (match json_mem "layer_norm_epsilon" json with
      | Jsont.Number (f, _) -> f
      | _ -> 1e-5);
  }

let from_file ?placement cfg dt path =
  of_hf ?placement cfg dt (Checkpoint.load path)

let from_pretrained ?placement ?(repo_id = "gpt2") dt =
  let cfg = config_of_json (Hf.load_config repo_id) in
  (cfg, of_hf ?placement cfg dt (Hf.load_checkpoint repo_id))
