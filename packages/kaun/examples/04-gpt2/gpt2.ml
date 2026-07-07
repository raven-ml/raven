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

(* Model: plain records of kaun layers *)

type block = {
  ln1 : Layer_norm.t;
  attn : Attention.t;
  ln2 : Layer_norm.t;
  fc : Linear.t;
  proj : Linear.t;
}

type t = {
  wte : Embedding.t;
  wpe : Embedding.t;
  blocks : block list;
  ln_f : Layer_norm.t;
}

module Params = struct
  type nonrec t = t

  let map_block (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) b =
    {
      ln1 = Layer_norm.map f b.ln1;
      attn = Attention.map f b.attn;
      ln2 = Layer_norm.map f b.ln2;
      fc = Linear.map f b.fc;
      proj = Linear.map f b.proj;
    }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p =
    {
      wte = Embedding.map f p.wte;
      wpe = Embedding.map f p.wpe;
      blocks = List.map (map_block f) p.blocks;
      ln_f = Layer_norm.map f p.ln_f;
    }

  let map2_block (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) b
      b' =
    {
      ln1 = Layer_norm.map2 f b.ln1 b'.ln1;
      attn = Attention.map2 f b.attn b'.attn;
      ln2 = Layer_norm.map2 f b.ln2 b'.ln2;
      fc = Linear.map2 f b.fc b'.fc;
      proj = Linear.map2 f b.proj b'.proj;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p p' =
    {
      wte = Embedding.map2 f p.wte p'.wte;
      wpe = Embedding.map2 f p.wpe p'.wpe;
      blocks = List.map2 (map2_block f) p.blocks p'.blocks;
      ln_f = Layer_norm.map2 f p.ln_f p'.ln_f;
    }

  let iter_block (f : 'a 'b. ('a, 'b) Nx.t -> unit) b =
    Layer_norm.iter f b.ln1;
    Attention.iter f b.attn;
    Layer_norm.iter f b.ln2;
    Linear.iter f b.fc;
    Linear.iter f b.proj

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) p =
    Embedding.iter f p.wte;
    Embedding.iter f p.wpe;
    List.iter (iter_block f) p.blocks;
    Layer_norm.iter f p.ln_f

  let names p =
    let pre prefix ns = List.map (fun n -> prefix ^ "." ^ n) ns in
    let block b =
      pre "ln1" (Layer_norm.names b.ln1)
      @ pre "attn" (Attention.names b.attn)
      @ pre "ln2" (Layer_norm.names b.ln2)
      @ pre "fc" (Linear.names b.fc)
      @ pre "proj" (Linear.names b.proj)
    in
    pre "wte" (Embedding.names p.wte)
    @ pre "wpe" (Embedding.names p.wpe)
    @ List.concat
        (List.mapi
           (fun i b -> pre (Printf.sprintf "blocks.%d" i) (block b))
           p.blocks)
    @ pre "ln_f" (Layer_norm.names p.ln_f)
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

(* Forward pass (inference: dropout omitted) *)

let block_apply cfg b x =
  let eps = cfg.layer_norm_eps in
  let x =
    Nx.add x
      (Attention.apply ~num_heads:cfg.n_head ~causal:true b.attn
         (Layer_norm.apply ~eps b.ln1 x))
  in
  Nx.add x
    (Linear.apply b.proj
       (Fn.gelu_approx (Linear.apply b.fc (Layer_norm.apply ~eps b.ln2 x))))

let logits cfg p ids =
  let seq = (Nx.shape ids).(1) in
  if seq > cfg.n_positions then
    invalid_argf "Gpt2.logits: seq %d exceeds n_positions %d" seq
      cfg.n_positions;
  let pos = Nx.reshape [| 1; seq |] (Nx.arange Nx.int32 0 seq 1) in
  let x = Nx.add (Embedding.apply p.wte ids) (Embedding.apply p.wpe pos) in
  let x = List.fold_left (fun x b -> block_apply cfg b x) x p.blocks in
  let h = Layer_norm.apply ~eps:cfg.layer_norm_eps p.ln_f x in
  (* Tied LM head: logits = h @ wteᵀ. *)
  Nx.matmul h (Nx.transpose p.wte.table)

(* Cached forward pass: the per-layer key-value caches make one decode step cost
   a single-position forward instead of a whole-sequence one. Shapes depend only
   on the input length and the cache length — the position is a tensor — so a
   single-token step traces once under [Rune.jit]. *)

type cache = Nx.float32_elt Attention.Cache.t list

let cache cfg ~len =
  List.init cfg.n_layer (fun _ ->
      Attention.Cache.make ~num_heads:cfg.n_head
        ~head_dim:(cfg.n_embd / cfg.n_head) ~len Nx.float32)

let block_apply_cached cfg b c ~pos x =
  let eps = cfg.layer_norm_eps in
  let attn, c =
    Attention.apply_cached ~num_heads:cfg.n_head ~pos ~cache:c b.attn
      (Layer_norm.apply ~eps b.ln1 x)
  in
  let x = Nx.add x attn in
  ( Nx.add x
      (Linear.apply b.proj
         (Fn.gelu_approx (Linear.apply b.fc (Layer_norm.apply ~eps b.ln2 x)))),
    c )

let logits_cached cfg p ~pos caches ids =
  let seq = (Nx.shape ids).(1) in
  let positions =
    Nx.add
      (Nx.reshape [| 1; 1 |] pos)
      (Nx.reshape [| 1; seq |] (Nx.arange Nx.int32 0 seq 1))
  in
  let x =
    Nx.add (Embedding.apply p.wte ids) (Embedding.apply p.wpe positions)
  in
  let x, rev_caches =
    List.fold_left2
      (fun (x, cs) b c ->
        let x, c = block_apply_cached cfg b c ~pos x in
        (x, c :: cs))
      (x, []) p.blocks caches
  in
  (* Only the last position's logits matter for decoding. *)
  let x = Nx.slice [ A; R (seq - 1, seq) ] x in
  let h = Layer_norm.apply ~eps:cfg.layer_norm_eps p.ln_f x in
  let vocab = (Nx.shape p.wte.table).(0) in
  ( Nx.reshape
      [| (Nx.shape ids).(0); vocab |]
      (Nx.matmul h (Nx.transpose p.wte.table)),
    List.rev rev_caches )

(* Greedy decoding: append the argmax of the last position's logits, re-running
   the model on the grown sequence each step (no key-value cache). *)

let generate cfg p ~max_tokens prompt =
  let n0 = Array.length prompt in
  if n0 = 0 then invalid_arg "Gpt2.generate: prompt must not be empty";
  let tokens = Array.make (n0 + max_tokens) 0l in
  Array.blit prompt 0 tokens 0 n0;
  for n = n0 to n0 + max_tokens - 1 do
    let input = Nx.create Nx.int32 [| 1; n |] (Array.sub tokens 0 n) in
    let last = Nx.slice [ I 0; I (n - 1) ] (logits cfg p input) in
    tokens.(n) <- Nx.item [] (Nx.argmax ~axis:0 last)
  done;
  tokens

(* HuggingFace checkpoint adaptation.

   HF names tensors h.{i}.attn.c_attn.weight, h.{i}.mlp.c_fc.bias, ... and fuses
   the q, k and v projections into c_attn ([n_embd; 3 * n_embd]). Its Conv1D
   weights are already [inputs; outputs], so only splits and renames are
   needed. *)

let hf_name name =
  match name with
  | "wte.weight" -> "wte.table"
  | "wpe.weight" -> "wpe.table"
  | "ln_f.weight" -> "ln_f.gamma"
  | "ln_f.bias" -> "ln_f.beta"
  | _ -> (
      match String.split_on_char '.' name with
      | "h" :: i :: rest -> (
          let ours leaf = Printf.sprintf "blocks.%s.%s" i leaf in
          match rest with
          | [ "ln_1"; "weight" ] -> ours "ln1.gamma"
          | [ "ln_1"; "bias" ] -> ours "ln1.beta"
          | [ "ln_2"; "weight" ] -> ours "ln2.gamma"
          | [ "ln_2"; "bias" ] -> ours "ln2.beta"
          | [ "attn"; "c_proj"; "weight" ] -> ours "attn.out.w"
          | [ "attn"; "c_proj"; "bias" ] -> ours "attn.out.b"
          | [ "mlp"; "c_fc"; "weight" ] -> ours "fc.w"
          | [ "mlp"; "c_fc"; "bias" ] -> ours "fc.b"
          | [ "mlp"; "c_proj"; "weight" ] -> ours "proj.w"
          | [ "mlp"; "c_proj"; "bias" ] -> ours "proj.b"
          | _ -> name (* attention mask buffers; unused *))
      | _ -> name)

let of_hf ~n_layer ckpt =
  let split_qkv ckpt i =
    let fused leaf = Printf.sprintf "h.%d.attn.c_attn.%s" i leaf in
    let ours p leaf = Printf.sprintf "blocks.%d.attn.%s.%s" i p leaf in
    ckpt
    |> Hf.split (fused "weight")
         ~into:[ ours "q" "w"; ours "k" "w"; ours "v" "w" ]
    |> Hf.split (fused "bias")
         ~into:[ ours "q" "b"; ours "k" "b"; ours "v" "b" ]
  in
  List.fold_left split_qkv ckpt (List.init n_layer Fun.id) |> Hf.rename hf_name

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

let of_checkpoint cfg ckpt =
  of_hf ~n_layer:cfg.n_layer ckpt
  |> Checkpoint.to_params (module Params) ~like:(make cfg) ~cast:true

let from_file cfg path = of_checkpoint cfg (Checkpoint.load path)

let from_pretrained ?(repo_id = "gpt2") () =
  let cfg = config_of_json (Hf.load_config repo_id) in
  (cfg, of_checkpoint cfg (Hf.load_checkpoint repo_id))
