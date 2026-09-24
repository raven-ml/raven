(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun
module Hf = Kaun_hf

(* Configuration *)

type config = {
  vocab_size : int;
  dim : int;
  n_layers : int;
  n_heads : int;
  n_kv_heads : int;
  head_dim : int;
  hidden_dim : int;
  norm_eps : float;
  rope : Rope.t;
  tied : bool;
}

(* Model: plain records of kaun layers, generic over the float dtype *)

type 'a block = {
  attn_norm : 'a Rms_norm.t;
  attn : 'a Attention.t;
  ffn_norm : 'a Rms_norm.t;
  gate : 'a Linear.t;
  up : 'a Linear.t;
  down : 'a Linear.t;
}

type 'a params = {
  tok : 'a Embedding.t;
  blocks : 'a block list;
  norm : 'a Rms_norm.t;
  head : 'a Linear.t option;
}

type t = Nx.float32_t params
type role = Whole | Column | Row | Kv_heads

module Params = struct
  type nonrec 'a t = 'a params

  let map_block f b =
    let attn_norm = Rms_norm.map f b.attn_norm in
    let attn = Attention.map f b.attn in
    let ffn_norm = Rms_norm.map f b.ffn_norm in
    let gate = Linear.map f b.gate in
    let up = Linear.map f b.up in
    let down = Linear.map f b.down in
    { attn_norm; attn; ffn_norm; gate; up; down }

  let map f p =
    let tok = Embedding.map f p.tok in
    let blocks = List.map (map_block f) p.blocks in
    let norm = Rms_norm.map f p.norm in
    let head = Option.map (Linear.map f) p.head in
    { tok; blocks; norm; head }

  let map2_block f b b' =
    let attn_norm = Rms_norm.map2 f b.attn_norm b'.attn_norm in
    let attn = Attention.map2 f b.attn b'.attn in
    let ffn_norm = Rms_norm.map2 f b.ffn_norm b'.ffn_norm in
    let gate = Linear.map2 f b.gate b'.gate in
    let up = Linear.map2 f b.up b'.up in
    let down = Linear.map2 f b.down b'.down in
    { attn_norm; attn; ffn_norm; gate; up; down }

  let map2_head f h h' =
    match (h, h') with
    | None, None -> None
    | Some l, Some l' -> Some (Linear.map2 f l l')
    | _ ->
        invalid_arg "Llama.Params.map2: one model ties its head, one does not"

  let map2 f p p' =
    let tok = Embedding.map2 f p.tok p'.tok in
    let blocks = List.map2 (map2_block f) p.blocks p'.blocks in
    let norm = Rms_norm.map2 f p.norm p'.norm in
    let head = map2_head f p.head p'.head in
    { tok; blocks; norm; head }

  let iter_block f b =
    Rms_norm.iter f b.attn_norm;
    Attention.iter f b.attn;
    Rms_norm.iter f b.ffn_norm;
    Linear.iter f b.gate;
    Linear.iter f b.up;
    Linear.iter f b.down

  let iter f p =
    Embedding.iter f p.tok;
    List.iter (iter_block f) p.blocks;
    Rms_norm.iter f p.norm;
    Option.iter (Linear.iter f) p.head

  let fold_block f acc b =
    let acc = Rms_norm.fold (fun s -> f ("attn_norm." ^ s)) acc b.attn_norm in
    let acc = Attention.fold (fun s -> f ("attn." ^ s)) acc b.attn in
    let acc = Rms_norm.fold (fun s -> f ("ffn_norm." ^ s)) acc b.ffn_norm in
    let acc = Linear.fold (fun s -> f ("gate." ^ s)) acc b.gate in
    let acc = Linear.fold (fun s -> f ("up." ^ s)) acc b.up in
    Linear.fold (fun s -> f ("down." ^ s)) acc b.down

  let fold f acc p =
    let acc = Embedding.fold (fun s -> f ("tok." ^ s)) acc p.tok in
    let _, acc =
      List.fold_left
        (fun (i, acc) b ->
          let pre = Printf.sprintf "blocks.%d." i in
          (i + 1, fold_block (fun s -> f (pre ^ s)) acc b))
        (0, acc) p.blocks
    in
    let acc = Rms_norm.fold (fun s -> f ("norm." ^ s)) acc p.norm in
    match p.head with
    | None -> acc
    | Some l -> Linear.fold (fun s -> f ("head." ^ s)) acc l

  let fold2_block f acc b b' =
    let acc =
      Rms_norm.fold2
        (fun s -> f ("attn_norm." ^ s))
        acc b.attn_norm b'.attn_norm
    in
    let acc = Attention.fold2 (fun s -> f ("attn." ^ s)) acc b.attn b'.attn in
    let acc =
      Rms_norm.fold2 (fun s -> f ("ffn_norm." ^ s)) acc b.ffn_norm b'.ffn_norm
    in
    let acc = Linear.fold2 (fun s -> f ("gate." ^ s)) acc b.gate b'.gate in
    let acc = Linear.fold2 (fun s -> f ("up." ^ s)) acc b.up b'.up in
    Linear.fold2 (fun s -> f ("down." ^ s)) acc b.down b'.down

  let fold2 f acc p p' =
    let acc = Embedding.fold2 (fun s -> f ("tok." ^ s)) acc p.tok p'.tok in
    let _, acc =
      List.fold_left2
        (fun (i, acc) b b' ->
          let pre = Printf.sprintf "blocks.%d." i in
          (i + 1, fold2_block (fun s -> f (pre ^ s)) acc b b'))
        (0, acc) p.blocks p'.blocks
    in
    let acc = Rms_norm.fold2 (fun s -> f ("norm." ^ s)) acc p.norm p'.norm in
    match (p.head, p'.head) with
    | None, None -> acc
    | Some l, Some l' -> Linear.fold2 (fun s -> f ("head." ^ s)) acc l l'
    | _ ->
        invalid_arg "Llama.Params.fold2: one model ties its head, one does not"

  let names_block i b =
    let pre field s = Printf.sprintf "blocks.%d.%s.%s" i field s in
    {
      attn_norm = Rms_norm.map (pre "attn_norm") (Rms_norm.names b.attn_norm);
      attn = Attention.map (pre "attn") (Attention.names b.attn);
      ffn_norm = Rms_norm.map (pre "ffn_norm") (Rms_norm.names b.ffn_norm);
      gate = Linear.map (pre "gate") (Linear.names b.gate);
      up = Linear.map (pre "up") (Linear.names b.up);
      down = Linear.map (pre "down") (Linear.names b.down);
    }

  let names p =
    let pre field s = field ^ "." ^ s in
    {
      tok = Embedding.map (pre "tok") (Embedding.names p.tok);
      blocks = List.mapi names_block p.blocks;
      norm = Rms_norm.map (pre "norm") (Rms_norm.names p.norm);
      head =
        Option.map (fun l -> Linear.map (pre "head") (Linear.names l)) p.head;
    }
end

let make cfg =
  let zeros = Init.zeros in
  let linear ~inputs ~outputs =
    Linear.make ~bias:false ~w_init:zeros ~inputs ~outputs Nx.float32
  in
  let block () =
    {
      attn_norm = Rms_norm.init ~dim:cfg.dim;
      attn =
        Attention.make ~bias:false ~w_init:zeros
          ~q_dim:(cfg.n_heads * cfg.head_dim)
          ~kv_dim:(cfg.n_kv_heads * cfg.head_dim)
          ~embed_dim:cfg.dim Nx.float32;
      ffn_norm = Rms_norm.init ~dim:cfg.dim;
      gate = linear ~inputs:cfg.dim ~outputs:cfg.hidden_dim;
      up = linear ~inputs:cfg.dim ~outputs:cfg.hidden_dim;
      down = linear ~inputs:cfg.hidden_dim ~outputs:cfg.dim;
    }
  in
  {
    tok =
      Embedding.make ~init:zeros ~vocab:cfg.vocab_size ~dim:cfg.dim Nx.float32;
    blocks = List.init cfg.n_layers (fun _ -> block ());
    norm = Rms_norm.init ~dim:cfg.dim;
    head =
      (if cfg.tied then None
       else Some (linear ~inputs:cfg.dim ~outputs:cfg.vocab_size));
  }

(* Forward passes: one fold over the blocks, which threads the caches along the
   index. *)

let block cfg b cache index x =
  let eps = cfg.norm_eps in
  let a, cache =
    Attention.cached ~head_dim:cfg.head_dim ~rope:cfg.rope b.attn cache index
      (Rms_norm.apply ~eps b.attn_norm x)
  in
  let x = Nx.add x a in
  let h = Rms_norm.apply ~eps b.ffn_norm x in
  let mlp =
    Linear.apply b.down
      (Nx.mul (Fn.silu (Linear.apply b.gate h)) (Linear.apply b.up h))
  in
  (Nx.add x mlp, cache)

module Cache = Attention.Cache.List

let cache ?placement cfg ~slots dtype =
  let place x =
    match placement with None -> x | Some p -> Nx.place (p Kv_heads ~axis:1) x
  in
  List.init cfg.n_layers (fun _ ->
      Attention.Cache.map place
        (Attention.Cache.make ~slots ~kv_heads:cfg.n_kv_heads
           ~head_dim:cfg.head_dim dtype))

let cached cfg p caches index ids =
  let x, rev =
    List.fold_left2
      (fun (x, cs) b c ->
        let x, c = block cfg b c index x in
        (x, c :: cs))
      (Embedding.apply p.tok ids, [])
      p.blocks caches
  in
  (x, List.rev rev)

let hidden cfg p ids =
  let batch = Nx.dim 0 ids and seq = Nx.dim 1 ids in
  let nothing = cache cfg ~slots:0 (Nx.dtype p.norm.Rms_norm.gamma) in
  fst (cached cfg p nothing (Cache_index.whole ~batch ~seq ()) ids)

let logits cfg p h =
  let h = Rms_norm.apply ~eps:cfg.norm_eps p.norm h in
  match p.head with
  | Some l -> Linear.apply l h
  | None -> Nx.matmul h (Nx.transpose p.tok.Embedding.table)

(* Importing a HuggingFace checkpoint.

   The file names its tensors model.layers.{i}.self_attn.q_proj.weight, ... and
   stores every projection as [outputs; inputs], the transpose of [Linear]'s
   layout. Its q and k weights are laid out for the rotation that pairs feature
   i with i + head_dim / 2, which is [Rope]'s. A tied model has no lm_head
   entry. *)

let of_hf ?placement cfg dt ckpt =
  let place role ~axis x =
    match placement with None -> x | Some p -> Nx.place (p role ~axis) x
  in
  let weight ~shape name =
    Checkpoint.to_float ~shape dt (name ^ ".weight") ckpt
  in
  let norm name =
    { Rms_norm.gamma = place Whole ~axis:0 (weight ~shape:[| cfg.dim |] name) }
  in
  (* A column projection is cut along its outputs, a row one along its
     inputs. *)
  let linear role ~axis ~inputs ~outputs name =
    let w = Nx.matrix_transpose (weight ~shape:[| outputs; inputs |] name) in
    { Linear.w = place role ~axis w; b = None }
  in
  let column = linear Column ~axis:1 and row = linear Row ~axis:0 in
  let q_dim = cfg.n_heads * cfg.head_dim in
  let kv_dim = cfg.n_kv_heads * cfg.head_dim in
  let block i =
    let at leaf = Printf.sprintf "model.layers.%d.%s" i leaf in
    {
      attn_norm = norm (at "input_layernorm");
      attn =
        {
          q = column ~inputs:cfg.dim ~outputs:q_dim (at "self_attn.q_proj");
          k = column ~inputs:cfg.dim ~outputs:kv_dim (at "self_attn.k_proj");
          v = column ~inputs:cfg.dim ~outputs:kv_dim (at "self_attn.v_proj");
          out = row ~inputs:q_dim ~outputs:cfg.dim (at "self_attn.o_proj");
        };
      ffn_norm = norm (at "post_attention_layernorm");
      gate = column ~inputs:cfg.dim ~outputs:cfg.hidden_dim (at "mlp.gate_proj");
      up = column ~inputs:cfg.dim ~outputs:cfg.hidden_dim (at "mlp.up_proj");
      down = row ~inputs:cfg.hidden_dim ~outputs:cfg.dim (at "mlp.down_proj");
    }
  in
  {
    tok =
      {
        Embedding.table =
          place Whole ~axis:0
            (weight ~shape:[| cfg.vocab_size; cfg.dim |] "model.embed_tokens");
      };
    blocks = List.init cfg.n_layers block;
    norm = norm "model.norm";
    head =
      (if cfg.tied then None
       else Some (column ~inputs:cfg.dim ~outputs:cfg.vocab_size "lm_head"));
  }

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let dtype_of_string = function
  | "float32" -> Dtype Nx.float32
  | "float16" -> Dtype Nx.float16
  | "bfloat16" -> Dtype Nx.bfloat16
  | d -> failwith ("--dtype must be float32, float16 or bfloat16, got " ^ d)

let stored_dtype ckpt =
  let (Rune.Ptree.P table) = Checkpoint.get "model.embed_tokens.weight" ckpt in
  match Nx.dtype table with
  | Nx.Float16 -> Dtype Nx.float16
  | Nx.BFloat16 -> Dtype Nx.bfloat16
  | Nx.Float32 -> Dtype Nx.float32
  | _ ->
      failwith
        "the checkpoint's embedding table is not a float16, bfloat16 or \
         float32 entry"

(* Configuration from HuggingFace's config.json *)

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

let config_of_json json =
  let missing name = failwith ("llama config.json: missing " ^ name) in
  let number ?default name =
    match (json_mem name json, default) with
    | Jsont.Number (f, _), _ -> f
    | _, Some d -> d
    | _, None -> missing name
  in
  let int ?default name =
    int_of_float (number ?default:(Option.map float_of_int default) name)
  in
  let dim = int "hidden_size" and n_heads = int "num_attention_heads" in
  let head_dim = int ~default:(dim / n_heads) "head_dim" in
  let theta = number ~default:10000.0 "rope_theta" in
  let rope =
    match json_mem "rope_scaling" json with
    | Jsont.Null _ -> Rope.make ~theta ~head_dim ()
    | scaling -> (
        let field name =
          match json_mem name scaling with
          | Jsont.Number (f, _) -> f
          | _ -> missing ("rope_scaling." ^ name)
        in
        (* [rope_type], or [type] in files written by older tools. *)
        let kind =
          match (json_mem "rope_type" scaling, json_mem "type" scaling) with
          | Jsont.String (k, _), _ | _, Jsont.String (k, _) -> k
          | _ -> missing "rope_scaling.rope_type"
        in
        match kind with
        | "default" -> Rope.make ~theta ~head_dim ()
        | "llama3" ->
            Rope.llama3 ~theta ~head_dim ~factor:(field "factor")
              ~low_freq_factor:(field "low_freq_factor")
              ~high_freq_factor:(field "high_freq_factor")
              ~original_context:
                (int_of_float (field "original_max_position_embeddings"))
        | other -> failwith ("llama config.json: unsupported rope_type " ^ other)
        )
  in
  {
    vocab_size = int "vocab_size";
    dim;
    n_layers = int "num_hidden_layers";
    n_heads;
    n_kv_heads = int ~default:n_heads "num_key_value_heads";
    head_dim;
    hidden_dim = int "intermediate_size";
    norm_eps = number "rms_norm_eps";
    rope;
    tied =
      (match json_mem "tie_word_embeddings" json with
      | Jsont.Bool (b, _) -> b
      | _ -> false);
  }

(* Pretrained loading *)

let from_file ?placement cfg dt path =
  of_hf ?placement cfg dt (Checkpoint.load path)

(* An ungated mirror whose weight files are byte-identical to Meta's. *)
let default_repo = "NousResearch/Llama-3.2-1B"

let from_pretrained ?placement ?(repo_id = default_repo) dt =
  let cfg = config_of_json (Hf.load_config repo_id) in
  (cfg, of_hf ?placement cfg dt (Hf.load_checkpoint repo_id))
