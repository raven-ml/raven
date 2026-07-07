(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPT-2 text generation.

   Builds the 124M-parameter GPT-2 model from HuggingFace safetensors weights
   and greedily generates tokens from a prompt. Every step reruns the full
   forward pass over all tokens so far (prompt mode: concrete shapes, no
   key-value cache), so generation is quadratic in sequence length but needs
   no symbolic shapes. *)

open Tolk_frontend
module Embedding = Tolk_nn.Embedding
module Linear = Tolk_nn.Linear
module Layer_norm = Tolk_nn.Layer_norm
module State = Tolk_nn.State

let max_context = 128

(* Model *)

type attention = {
  c_attn : Linear.t;
  c_proj : Linear.t;
  n_heads : int;
  dim : int;
  head_dim : int;
}

let attention dim n_heads =
  {
    c_attn = Linear.create dim (3 * dim);
    c_proj = Linear.create dim dim;
    n_heads;
    dim;
    head_dim = dim / n_heads;
  }

let attention_apply a x mask =
  let bsz, seqlen =
    match Tensor.shape x with
    | [ b; s; _ ] -> (b, s)
    | _ -> invalid_arg "attention: input must be (batch, seq, dim)"
  in
  let xqkv =
    Movement.reshape (Linear.apply a.c_attn x)
      [ bsz; seqlen; 3; a.n_heads; a.head_dim ]
  in
  let sel i =
    Op.getitem xqkv Movement.[ All; All; I i; All; All ]
  in
  let tr t = Movement.transpose ~dim0:1 ~dim1:2 t in
  let xq, keys, values = (tr (sel 0), tr (sel 1), tr (sel 2)) in
  let out = Op.scaled_dot_product_attention ?attn_mask:mask xq keys values in
  Linear.apply a.c_proj (Movement.reshape (tr out) [ bsz; seqlen; a.dim ])

type feed_forward = { c_fc : Linear.t; c_proj : Linear.t }

let feed_forward dim hidden_dim =
  { c_fc = Linear.create dim hidden_dim; c_proj = Linear.create hidden_dim dim }

let feed_forward_apply f x =
  Linear.apply f.c_proj (Elementwise.gelu (Linear.apply f.c_fc x))

type block = {
  attn : attention;
  mlp : feed_forward;
  ln_1 : Layer_norm.t;
  ln_2 : Layer_norm.t;
}

let block dim n_heads norm_eps =
  {
    attn = attention dim n_heads;
    mlp = feed_forward dim (4 * dim);
    ln_1 = Layer_norm.create ~eps:norm_eps dim;
    ln_2 = Layer_norm.create ~eps:norm_eps dim;
  }

let block_apply b x mask =
  let h =
    Elementwise.add x
      (Dtype_ops.float (attention_apply b.attn (Layer_norm.apply b.ln_1 x) mask))
  in
  Elementwise.contiguous
    (Elementwise.add h (feed_forward_apply b.mlp (Layer_norm.apply b.ln_2 h)))

type transformer = {
  wte : Embedding.t;
  wpe : Embedding.t;
  h : block list;
  ln_f : Layer_norm.t;
  lm_head : Linear.t;
  allpos : Tensor.t;
}

let transformer ~dim ~n_heads ~n_layers ~norm_eps ~vocab_size ~max_seq_len =
  {
    wte = Embedding.create vocab_size dim;
    wpe = Embedding.create max_seq_len dim;
    h = List.init n_layers (fun _ -> block dim n_heads norm_eps);
    ln_f = Layer_norm.create ~eps:norm_eps dim;
    lm_head = Linear.create ~bias:false dim vocab_size;
    allpos = Movement.reshape (Op.arange max_context) [ 1; -1 ];
  }

(* Greedy forward: the argmax of the logits of the last position. *)
let forward m tokens =
  let seqlen = List.nth (Tensor.shape tokens) 1 in
  let tok_emb = Embedding.apply m.wte tokens in
  let pos_emb =
    Embedding.apply m.wpe (Movement.shrink m.allpos [ (0, 1); (0, seqlen) ])
  in
  let h = Elementwise.add tok_emb pos_emb in
  let mask =
    if seqlen > 1 then
      Some
        (Op.triu ~diagonal:1
           (Creation.full [ 1; 1; seqlen; seqlen ] (Tensor.Sfloat neg_infinity)))
    else None
  in
  let h = List.fold_left (fun h b -> block_apply b h mask) h m.h in
  let logits = Linear.apply m.lm_head (Layer_norm.apply m.ln_f h) in
  Op.argmax ~axis:(-1) (Op.getitem logits Movement.[ All; I (-1); All ])

(* State dict *)

let state_dict m =
  let linear prefix (l : Linear.t) =
    ((prefix ^ ".weight", l.weight)
    :: (match l.bias with Some b -> [ (prefix ^ ".bias", b) ] | None -> []))
  in
  let norm prefix (l : Layer_norm.t) =
    (match l.weight with Some w -> [ (prefix ^ ".weight", w) ] | None -> [])
    @ match l.bias with Some b -> [ (prefix ^ ".bias", b) ] | None -> []
  in
  [ ("wte.weight", m.wte.weight); ("wpe.weight", m.wpe.weight) ]
  @ List.concat
      (List.mapi
         (fun i b ->
           let p = Printf.sprintf "h.%d" i in
           linear (p ^ ".attn.c_attn") b.attn.c_attn
           @ linear (p ^ ".attn.c_proj") b.attn.c_proj
           @ linear (p ^ ".mlp.c_fc") b.mlp.c_fc
           @ linear (p ^ ".mlp.c_proj") b.mlp.c_proj
           @ norm (p ^ ".ln_1") b.ln_1
           @ norm (p ^ ".ln_2") b.ln_2)
         m.h)
  @ norm "ln_f" m.ln_f
  @ linear "lm_head" m.lm_head

(* Build *)

let cache_dir =
  Filename.concat
    (try Sys.getenv "XDG_CACHE_HOME"
     with Not_found -> Filename.concat (Sys.getenv "HOME") ".cache")
    "tolk-gpt2"

let fetch url file =
  let path = Filename.concat cache_dir file in
  if not (Sys.file_exists path) then begin
    Printf.printf "downloading %s...\n%!" url;
    let quoted = Filename.quote in
    if Sys.command (Printf.sprintf "mkdir -p %s" (quoted cache_dir)) <> 0 then
      failwith "mkdir failed";
    let cmd =
      Printf.sprintf "curl -fsSL -o %s %s" (quoted (path ^ ".tmp")) (quoted url)
    in
    if Sys.command cmd <> 0 then failwith ("download failed: " ^ url);
    Sys.rename (path ^ ".tmp") path
  end;
  path

let build () =
  let model =
    transformer ~dim:768 ~n_heads:12 ~n_layers:12 ~norm_eps:1e-5
      ~vocab_size:50257 ~max_seq_len:1024
  in
  let weights =
    State.safe_load
      (fetch "https://huggingface.co/gpt2/resolve/main/model.safetensors"
         "model.safetensors")
  in
  (* The checkpoint stores the four projection matrices in (in, out) layout;
     transpose them to the (out, in) layout of [Linear]. *)
  let transposed k =
    List.exists
      (fun suffix ->
        String.length k >= String.length suffix
        && String.equal suffix
             (String.sub k
                (String.length k - String.length suffix)
                (String.length suffix)))
      [
        "attn.c_attn.weight"; "attn.c_proj.weight"; "mlp.c_fc.weight";
        "mlp.c_proj.weight";
      ]
  in
  let weights =
    List.map
      (fun (k, v) -> (k, if transposed k then Movement.transpose v else v))
      weights
  in
  (* lm_head and wte are tied. *)
  let weights = ("lm_head.weight", List.assoc "wte.weight" weights) :: weights in
  State.load_state_dict (state_dict model) weights;
  model

(* Generation *)

let generate model tokenizer prompt count =
  let toks = ref (Array.to_list (Brot.encode_ids tokenizer prompt)) in
  let times = ref [] in
  for _ = 1 to count do
    let arr = Array.of_list !toks in
    let tokens = Run.of_int_array ~shape:[ 1; Array.length arr ] arr in
    let t0 = Unix.gettimeofday () in
    let tok = Run.item_int (forward model tokens) in
    times := (Unix.gettimeofday () -. t0) :: !times;
    toks := !toks @ [ tok ]
  done;
  let total = List.fold_left ( +. ) 0. !times in
  Printf.printf "generated %d tokens in %.2f s (%.2f tok/s)\n%!" count total
    (float_of_int count /. total);
  Brot.decode tokenizer (Array.of_list !toks)

let default_prompt = "What is the answer to life, the universe, and everything?"

(* Reference greedy generations for --count 10 used by --validate. *)
let expected =
  [
    ( default_prompt,
      "What is the answer to life, the universe, and everything?\n\n\
       The answer to life, the universe," );
    ("Hello.", "Hello.\n\nI'm not sure if you're aware");
  ]

let () =
  let prompt = ref default_prompt in
  let count = ref 10 in
  let validate = ref false in
  Arg.parse
    [
      ("--prompt", Arg.Set_string prompt, "Phrase to start with");
      ("--count", Arg.Set_int count, "Max number of tokens to generate");
      ( "--validate",
        Arg.Set validate,
        "Check the output against the reference generation (needs --count 10)"
      );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "gpt2 [--prompt P] [--count N] [--validate]";
  Printf.printf "using %s backend\n%!" (Run.device_name ());
  let tokenizer =
    Brot.from_file
      (fetch "https://huggingface.co/gpt2/resolve/main/tokenizer.json"
         "tokenizer.json")
    |> function
    | Ok t -> t
    | Error e -> failwith ("tokenizer: " ^ e)
  in
  (* The model was trained with the byte-level BPE used by the reference
     implementations; make sure our tokenizer agrees on the test prompts. *)
  assert (
    Brot.encode_ids tokenizer default_prompt
    = [| 2061; 318; 262; 3280; 284; 1204; 11; 262; 6881; 11; 290; 2279; 30 |]);
  assert (Brot.encode_ids tokenizer "Hello." = [| 15496; 13 |]);
  let t0 = Unix.gettimeofday () in
  let model = build () in
  Printf.printf "loaded weights in %.2f s\n%!" (Unix.gettimeofday () -. t0);
  let text = generate model tokenizer !prompt !count in
  print_endline "Generating text...";
  print_endline text;
  if !validate then
    match List.assoc_opt !prompt expected with
    | Some e when !count = 10 ->
        if String.equal text e then print_endline "output validated"
        else begin
          Printf.eprintf "output mismatch!\nexpected: %S\ngot:      %S\n" e text;
          exit 1
        end
    | _ -> prerr_endline "no reference output for this prompt/count"
