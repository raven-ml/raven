(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Autoregressive text generation with pretrained GPT-2.

   Downloads gpt2 from HuggingFace (~548MB on first run) and generates text
   continuations from several prompts using greedy decoding. *)

open Kaun

(* Tokenizer *)

let load_tokenizer model_id =
  let vocab = Kaun_hf.download_file ~model_id ~filename:"vocab.json" () in
  let merges = Kaun_hf.download_file ~model_id ~filename:"merges.txt" () in
  Brot.from_model_file ~vocab ~merges
    ~pre:
      (Brot.Pre_tokenizer.byte_level ~add_prefix_space:false ~use_regex:true ())
    ~decoder:(Brot.Decoder.byte_level ())
    ()

let encode tokenizer text =
  Array.map Int32.of_int (Brot.encode_ids tokenizer text)

let decode tokenizer ids = Brot.decode tokenizer (Array.map Int32.to_int ids)

(* Greedy decode: at each step pick the highest-probability next token. *)

let generate model vars ~max_tokens prompt =
  let tokens = ref (Array.to_list prompt) in
  for _ = 1 to max_tokens do
    let ids = Array.of_list !tokens in
    let n = Array.length ids in
    let input = Nx.create Nx.int32 [| 1; n |] ids in
    let logits, _ = Layer.apply model vars ~training:false input in
    let last = Nx.slice [ I 0; I (n - 1) ] logits in
    let next : int32 = Nx.item [] (Nx.argmax ~axis:0 last) in
    tokens := !tokens @ [ next ]
  done;
  Array.of_list !tokens

(* Show the model's top-k predictions at a given position. *)

let print_top_k ~k model vars input_ids ~pos =
  let logits, _ = Layer.apply model vars ~training:false input_ids in
  let row = Nx.slice [ I 0; I pos ] logits in
  let sorted = Nx.argsort ~descending:true ~axis:0 row in
  let probs = Nx.softmax ~axes:[ 0 ] row in
  for i = 0 to k - 1 do
    let idx = Int32.to_int (Nx.item [ i ] sorted) in
    let prob : float = Nx.item [ idx ] probs in
    Printf.printf "    #%d  token %-6d  p=%.4f\n" (i + 1) idx prob
  done

let () =
  let model_id = "gpt2" in
  let dtype = Nx.float32 in

  (* Load tokenizer and model *)
  Printf.printf "Loading %s...\n%!" model_id;
  let tokenizer = load_tokenizer model_id in
  let cfg, params = Gpt2.from_pretrained ~model_id () in
  Printf.printf "  vocab=%d  n_embd=%d  layers=%d  heads=%d\n\n" cfg.vocab_size
    cfg.n_embd cfg.n_layer cfg.n_head;

  let model = Gpt2.for_causal_lm cfg () in
  let vars = Layer.make_vars ~params ~state:Ptree.empty ~dtype in

  (* --- What does the model predict after "Hello world"? --- *)
  Printf.printf "=== Next-token predictions ===\n";
  Printf.printf "  Prompt: \"Hello world\"\n";
  Printf.printf "  Top 5 continuations:\n";
  let hello_ids = encode tokenizer "Hello world" in
  let hello = Nx.create Nx.int32 [| 1; Array.length hello_ids |] hello_ids in
  print_top_k ~k:5 model vars hello ~pos:(Array.length hello_ids - 1);

  (* --- Greedy generation from several prompts --- *)
  Printf.printf "\n=== Greedy generation (30 tokens each) ===\n\n";
  let prompts =
    [ "The meaning of life is"; "Once upon a time"; "The quick brown fox" ]
  in
  List.iter
    (fun text ->
      let prompt = encode tokenizer text in
      let generated = generate model vars ~max_tokens:30 prompt in
      let continuation =
        Array.sub generated (Array.length prompt)
          (Array.length generated - Array.length prompt)
      in
      Printf.printf "  \"%s\" ->\n" text;
      Printf.printf "    %s\n\n" (decode tokenizer continuation))
    prompts
