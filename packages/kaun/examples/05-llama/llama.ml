(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun
module Hf = Kaun_hf

(* Configuration *)

type 'd config = {
  vocab_size : int;
  dim : int;
  n_layers : int;
  n_heads : int;
  n_kv_heads : int;
  head_dim : int;
  hidden_dim : int;
  norm_eps : float;
  rope : 'd Rope.t;
  tied : bool;
}

let place_config at cfg = { cfg with rope = Rope.place at cfg.rope }

(* Model: plain records of kaun layers, generic over the float dtype *)

type ('a, 'd) block = {
  attn_norm : ('a, 'd) Rms_norm.t;
  attn : ('a, 'd) Attention.t;
  ffn_norm : ('a, 'd) Rms_norm.t;
  gate : ('a, 'd) Linear.t;
  up : ('a, 'd) Linear.t;
  down : ('a, 'd) Linear.t;
}

type ('a, 'd) params = {
  tok : ('a, 'd) Embedding.t;
  blocks : ('a, 'd) block list;
  norm : ('a, 'd) Rms_norm.t;
  head : ('a, 'd) Linear.t option;
}

type 'd t = ('d Nx.float32_t, 'd) params
type role = Whole | Column | Row | Kv_heads

module Params = struct
  type nonrec ('a, 'd) t = ('a, 'd) params

  let walk_block c b =
    let open Nx.Ptree.Walk in
    let attn_norm = field c "attn_norm" Rms_norm.walk b.attn_norm in
    let attn = field c "attn" Attention.walk b.attn in
    let ffn_norm = field c "ffn_norm" Rms_norm.walk b.ffn_norm in
    let gate = field c "gate" Linear.walk b.gate in
    let up = field c "up" Linear.walk b.up in
    let down = field c "down" Linear.walk b.down in
    { attn_norm; attn; ffn_norm; gate; up; down }

  let walk c p =
    let open Nx.Ptree.Walk in
    let tok = field c "tok" Embedding.walk p.tok in
    let blocks = field c "blocks" (list walk_block) p.blocks in
    let norm = field c "norm" Rms_norm.walk p.norm in
    let head = field c "head" (option Linear.walk) p.head in
    { tok; blocks; norm; head }
end

(* Every initializer is zeros, so [key] only says where the model lies. *)
let make ~key cfg =
  let zeros = Init.zeros in
  let at = Nx.Rng.placement key in
  let linear ~inputs ~outputs =
    Linear.make ~bias:false ~w_init:zeros ~key ~inputs ~outputs Nx.float32
  in
  let block () =
    {
      attn_norm = Rms_norm.init ~dim:cfg.dim at;
      attn =
        Attention.make ~bias:false ~w_init:zeros ~key
          ~q_dim:(cfg.n_heads * cfg.head_dim)
          ~kv_dim:(cfg.n_kv_heads * cfg.head_dim)
          ~embed_dim:cfg.dim Nx.float32;
      ffn_norm = Rms_norm.init ~dim:cfg.dim at;
      gate = linear ~inputs:cfg.dim ~outputs:cfg.hidden_dim;
      up = linear ~inputs:cfg.dim ~outputs:cfg.hidden_dim;
      down = linear ~inputs:cfg.hidden_dim ~outputs:cfg.dim;
    }
  in
  {
    tok =
      Embedding.make ~init:zeros ~key ~vocab:cfg.vocab_size ~dim:cfg.dim
        Nx.float32;
    blocks = List.init cfg.n_layers (fun _ -> block ());
    norm = Rms_norm.init ~dim:cfg.dim at;
    head =
      (if cfg.tied then None
       else Some (linear ~inputs:cfg.dim ~outputs:cfg.vocab_size));
  }

(* Forward passes: one fold over the blocks, which threads the caches along the
   index. *)

(* A projection cut along its inputs asks for its sum, the devices' terms
   ({!Nx.Placement.partial}): the residual add keeps it, and one [close]
   after it closes the sublayer. Where no device cuts a projection's
   inputs, both are the identity. *)
let close x = Nx.place (Nx.Placement.closed (Nx.placement x)) x

let block cfg b cache index x =
  let eps = cfg.norm_eps in
  let a, cache =
    Attention.cached ~head_dim:cfg.head_dim ~rope:cfg.rope ~sum:true b.attn
      cache index (Rms_norm.apply ~eps b.attn_norm x)
  in
  let x = close (Nx.add x a) in
  let h = Rms_norm.apply ~eps b.ffn_norm x in
  let mlp =
    Linear.apply ~sum:true b.down
      (Nx.mul (Fn.silu (Linear.apply b.gate h)) (Linear.apply b.up h))
  in
  (close (Nx.add x mlp), cache)

(* The pools, made where the kv-heads role places them. *)
let cache ~placement cfg ~slots dtype =
  let at = placement Kv_heads ~axis:1 in
  List.init cfg.n_layers (fun _ ->
      Attention.Cache.make ~slots ~kv_heads:cfg.n_kv_heads
           ~head_dim:cfg.head_dim dtype at)

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

(* The whole index reads no cache, so its empty pools lie whole beside the ids,
   and its positions where the ids are. *)
let hidden cfg p ids =
  let batch = Nx.dim 0 ids and seq = Nx.dim 1 ids in
  let at = Nx.placement ids in
  let whole = Nx.Placement.uncut at in
  let nothing =
    cache
      ~placement:(fun _ ~axis:_ -> whole)
      cfg ~slots:0
      (Nx.dtype p.norm.Rms_norm.gamma)
  in
  let index = Cache_index.place at (Cache_index.whole ~batch ~seq ()) in
  fst (cached cfg p nothing index ids)

let logits cfg p h =
  let h = Rms_norm.apply ~eps:cfg.norm_eps p.norm h in
  match p.head with
  | Some l -> Linear.apply l h
  | None -> Nx.matmul h (Nx.transpose p.tok.Embedding.table)

(* Importing HuggingFace weights.

   The file names its tensors model.layers.{i}.self_attn.q_proj.weight, ... and
   stores every projection as [outputs; inputs], the transpose of [Linear]'s
   layout. Its q and k weights are laid out for the rotation that pairs feature
   i with i + head_dim / 2, which is [Rope]'s. A tied model has no lm_head
   entry. *)

let of_hf ~placement cfg dt weights =
  let place role ~axis x = Nx.place (placement role ~axis) x in
  let weight ~shape name =
    Nx_io.Archive.float ~shape dt (name ^ ".weight") weights
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

(* Importing GGUF weights.

   llama.cpp's converter names the tensors token_embd, blk.{i}.attn_q, ... and
   keeps HuggingFace's [outputs; inputs] layout as their logical shape, a block
   format's blocks running along the inputs. Within each head it reorders the
   rows of the q and k projections for the rotation that pairs feature 2j with
   2j + 1: its row 2j + c is HuggingFace's row c * head_dim / 2 + j. [unpair]
   restores HuggingFace's order, which is [Rope]'s. Moving whole rows moves
   whole blocks, so it applies to a block format's bytes as to floats. *)

let unpair ~heads w =
  let rows = Nx.dim 0 w and cols = Nx.dim 1 w in
  let half = rows / heads / 2 in
  Nx.copy
    (Nx.reshape [| rows; cols |]
       (Nx.transpose ~axes:[ 0; 2; 1; 3 ]
          (Nx.reshape [| heads; half; 2; cols |] w)))

(* A block format's entry, its bytes at uint8. *)
let gguf_bytes name entries : Nx.host Nx.uint8_t =
  match Nx_io.Archive.find name entries with
  | None -> failwith (name ^ ": no entry in the archive")
  | Some (Nx.P t) -> (
      match Nx.dtype t with
      | Nx.Dtype.Uint8 -> t
      | _ -> failwith (name ^ ": a block format's entry is not uint8"))

let shape_string s =
  String.concat "; " (Array.to_list (Array.map string_of_int s))

(* The entry [name] at [dt], of logical shape [shape], its rows reordered by
   [unpair ~heads] when [heads] is given. A block format is decoded by
   [Nx_quant.dequant]. *)
let gguf_float g ?heads ~shape dt name =
  let entries = Nx_io.Gguf.tensors g in
  let info = Nx_io.Gguf.info name g in
  if info.shape <> shape then
    failwith
      (Printf.sprintf "%s: shape [%s] in the file, [%s] expected" name
         (shape_string info.shape) (shape_string shape));
  let rows w = match heads with None -> w | Some heads -> unpair ~heads w in
  let blocks quant =
    Nx_quant.dequant dt (quant (rows (gguf_bytes name entries)))
  in
  match info.dtype with
  | F32 | F16 | BF16 | F64 -> rows (Nx_io.Archive.float ~shape dt name entries)
  | Q8_0 -> blocks Nx_quant.q8_0
  | Q4_K -> blocks Nx_quant.q4_k
  | Q6_K -> blocks Nx_quant.q6_k
  | _ ->
      failwith
        (name
       ^ ": not a float16, bfloat16, float32, float64, Q8_0, Q4_K or Q6_K entry"
        )

let of_gguf ~placement cfg dt g =
  let place role ~axis x = Nx.place (placement role ~axis) x in
  let weight ?heads ~shape name =
    gguf_float g ?heads ~shape dt (name ^ ".weight")
  in
  let norm name =
    { Rms_norm.gamma = place Whole ~axis:0 (weight ~shape:[| cfg.dim |] name) }
  in
  let linear role ~axis ?heads ~inputs ~outputs name =
    let w =
      Nx.matrix_transpose (weight ?heads ~shape:[| outputs; inputs |] name)
    in
    { Linear.w = place role ~axis w; b = None }
  in
  let column = linear Column ~axis:1 and row = linear Row ~axis:0 in
  let q_dim = cfg.n_heads * cfg.head_dim in
  let kv_dim = cfg.n_kv_heads * cfg.head_dim in
  let block i =
    let at leaf = Printf.sprintf "blk.%d.%s" i leaf in
    {
      attn_norm = norm (at "attn_norm");
      attn =
        {
          q =
            column ~heads:cfg.n_heads ~inputs:cfg.dim ~outputs:q_dim
              (at "attn_q");
          k =
            column ~heads:cfg.n_kv_heads ~inputs:cfg.dim ~outputs:kv_dim
              (at "attn_k");
          v = column ~inputs:cfg.dim ~outputs:kv_dim (at "attn_v");
          out = row ~inputs:q_dim ~outputs:cfg.dim (at "attn_output");
        };
      ffn_norm = norm (at "ffn_norm");
      gate = column ~inputs:cfg.dim ~outputs:cfg.hidden_dim (at "ffn_gate");
      up = column ~inputs:cfg.dim ~outputs:cfg.hidden_dim (at "ffn_up");
      down = row ~inputs:cfg.hidden_dim ~outputs:cfg.dim (at "ffn_down");
    }
  in
  {
    tok =
      {
        Embedding.table =
          place Whole ~axis:0
            (weight ~shape:[| cfg.vocab_size; cfg.dim |] "token_embd");
      };
    blocks = List.init cfg.n_layers block;
    norm = norm "output_norm";
    head =
      (if cfg.tied then None
       else Some (column ~inputs:cfg.dim ~outputs:cfg.vocab_size "output"));
  }

(* Configuration from a GGUF file's metadata, the architecture's keys under its
   name. *)

let config_of_gguf g =
  let metadata = Nx_io.Gguf.metadata g in
  let fail fmt = Printf.ksprintf (fun s -> failwith ("llama GGUF: " ^ s)) fmt in
  let find key = List.assoc_opt key metadata in
  (match find "general.architecture" with
  | Some (Nx_io.Gguf.String "llama") -> ()
  | Some (String a) -> fail "architecture %s, llama expected" a
  | _ -> fail "missing general.architecture");
  let int_opt key =
    match find ("llama." ^ key) with
    | None -> None
    | Some (Uint8 n | Int8 n | Uint16 n | Int16 n | Uint32 n | Int32 n) ->
        Some n
    | Some (Uint64 n | Int64 n) -> Some (Int64.to_int n)
    | Some _ -> fail "llama.%s is not an integer" key
  in
  let int key =
    match int_opt key with Some n -> n | None -> fail "missing llama.%s" key
  in
  let float ?default key =
    match (find ("llama." ^ key), default) with
    | Some (Float32 x | Float64 x), _ -> x
    | Some _, _ -> fail "llama.%s is not a float" key
    | None, Some d -> d
    | None, None -> fail "missing llama.%s" key
  in
  let dim = int "embedding_length" and n_heads = int "attention.head_count" in
  let head_dim =
    Option.value ~default:(dim / n_heads) (int_opt "attention.key_length")
  in
  (match int_opt "rope.dimension_count" with
  | Some d when d <> head_dim ->
      fail "llama.rope.dimension_count %d rotates part of a head of %d" d
        head_dim
  | _ -> ());
  (match find "llama.rope.scaling.type" with
  | None | Some (String "none") -> ()
  | Some (String s) -> fail "unsupported llama.rope.scaling.type %s" s
  | Some _ -> fail "llama.rope.scaling.type is not a string");
  let entries = Nx_io.Gguf.tensors g in
  let present name = Option.is_some (Nx_io.Archive.find name entries) in
  let context = int "context_length" in
  let theta = float ~default:10000.0 "rope.freq_base" in
  let rope = Rope.make ~theta ~head_dim ~context () in
  (* Llama 3's schedule is stored as each pair's divisor of the standard
     frequency. *)
  let rope =
    if not (present "rope_freqs.weight") then rope
    else
      let divisors =
        Nx.to_array
          (gguf_float g
             ~shape:[| head_dim / 2 |]
             Nx.float64 "rope_freqs.weight")
      in
      Rope.of_frequencies ~context
        (Array.map2 ( /. ) (Rope.frequencies rope) divisors)
  in
  let vocab_size =
    match int_opt "vocab_size" with
    | Some n -> n
    | None -> (Nx_io.Gguf.info "token_embd.weight" g).shape.(0)
  in
  {
    vocab_size;
    dim;
    n_layers = int "block_count";
    n_heads;
    n_kv_heads =
      Option.value ~default:n_heads (int_opt "attention.head_count_kv");
    head_dim;
    hidden_dim = int "feed_forward_length";
    norm_eps = float "attention.layer_norm_rms_epsilon";
    rope;
    tied = not (present "output.weight");
  }

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let dtype_of_string = function
  | "float32" -> Dtype Nx.float32
  | "float16" -> Dtype Nx.float16
  | "bfloat16" -> Dtype Nx.bfloat16
  | d -> failwith ("--dtype must be float32, float16 or bfloat16, got " ^ d)

let stored_dtype weights =
  let name = "model.embed_tokens.weight" in
  match Nx_io.Archive.find name weights with
  | None -> failwith (name ^ ": no entry in the archive")
  | Some (Nx.P table) -> (
      match Nx.dtype table with
      | Nx.Dtype.Float16 -> Dtype Nx.float16
      | Nx.Dtype.Bfloat16 -> Dtype Nx.bfloat16
      | Nx.Dtype.Float32 -> Dtype Nx.float32
      | _ -> failwith (name ^ " is not a float16, bfloat16 or float32 entry"))

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
  let context = int "max_position_embeddings" in
  let rope =
    match json_mem "rope_scaling" json with
    | Jsont.Null _ -> Rope.make ~theta ~head_dim ~context ()
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
        | "default" -> Rope.make ~theta ~head_dim ~context ()
        | "llama3" ->
            Rope.llama3 ~theta ~head_dim ~factor:(field "factor")
              ~low_freq_factor:(field "low_freq_factor")
              ~high_freq_factor:(field "high_freq_factor")
              ~original_context:
                (int_of_float (field "original_max_position_embeddings"))
              ~context
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

let from_file ~placement cfg dt path =
  of_hf ~placement cfg dt (Nx_io.load_safetensors path)

let from_gguf ~placement dt path =
  let g = Nx_io.load_gguf path in
  let cfg = config_of_gguf g in
  (cfg, of_gguf ~placement cfg dt g)

(* An ungated mirror whose weight files are byte-identical to Meta's. *)
let default_repo = "NousResearch/Llama-3.2-1B"

let from_pretrained ~placement ?(repo_id = default_repo) dt =
  let cfg = config_of_json (Hf.load_config repo_id) in
  (cfg, of_hf ~placement cfg dt (Hf.load_safetensors repo_id))
