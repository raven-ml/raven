(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text generation with pretrained GPT-2.

   Downloads gpt2 from the HuggingFace Hub on first run (~548MB, cached
   afterwards), adapts the checkpoint onto kaun-next layer records, and
   generates continuations with greedy decoding. Skips gracefully when the files
   cannot be downloaded. *)

(* Tokenizer: GPT-2's byte-level BPE, from the repository's vocab and merges
   files. *)

let load_tokenizer repo_id =
  let vocab = Kaun_next_hf.download_file ~file:"vocab.json" repo_id in
  let merges = Kaun_next_hf.download_file ~file:"merges.txt" repo_id in
  Brot.from_model_file ~vocab ~merges
    ~pre:
      (Brot.Pre_tokenizer.byte_level ~add_prefix_space:false ~use_regex:true ())
    ~decoder:(Brot.Decoder.byte_level ())
    ()

let encode tokenizer text =
  Array.map Int32.of_int (Brot.encode_ids tokenizer text)

let decode tokenizer ids = Brot.decode tokenizer (Array.map Int32.to_int ids)

(* Greedy decoding: at each step append the highest-probability next token. *)

let generate cfg params ~max_tokens prompt =
  let tokens = ref (Array.to_list prompt) in
  for _ = 1 to max_tokens do
    let ids = Array.of_list !tokens in
    let n = Array.length ids in
    let input = Nx.create Nx.int32 [| 1; n |] ids in
    let logits = Gpt2.logits cfg params input in
    let last = Nx.slice [ I 0; I (n - 1) ] logits in
    let next : int32 = Nx.item [] (Nx.argmax ~axis:0 last) in
    tokens := !tokens @ [ next ]
  done;
  Array.of_list !tokens

(* The model's top-k next-token predictions after [prompt]. *)

let print_top_k ~k cfg params tokenizer prompt =
  let ids = encode tokenizer prompt in
  let n = Array.length ids in
  let input = Nx.create Nx.int32 [| 1; n |] ids in
  let logits = Gpt2.logits cfg params input in
  let row = Nx.slice [ I 0; I (n - 1) ] logits in
  let sorted = Nx.argsort ~descending:true ~axis:0 row in
  let probs = Kaun_next.Fn.softmax row in
  Printf.printf "  %S ->\n" prompt;
  for i = 0 to k - 1 do
    let id = Int32.to_int (Nx.item [ i ] sorted) in
    let prob : float = Nx.item [ id ] probs in
    let token = decode tokenizer [| Int32.of_int id |] in
    Printf.printf "    #%d  %-12S p=%.4f\n" (i + 1) token prob
  done

let run repo_id =
  Printf.printf "Loading %s...\n%!" repo_id;
  let tokenizer = load_tokenizer repo_id in
  let cfg, params = Gpt2.from_pretrained ~repo_id () in
  Printf.printf "  vocab=%d n_embd=%d layers=%d heads=%d\n\n" cfg.vocab_size
    cfg.n_embd cfg.n_layer cfg.n_head;

  Printf.printf "=== Top 5 next-token predictions ===\n";
  print_top_k ~k:5 cfg params tokenizer "Hello world";

  Printf.printf "\n=== Greedy generation (20 tokens each) ===\n\n";
  [ "The meaning of life is"; "Once upon a time" ]
  |> List.iter (fun text ->
      let prompt = encode tokenizer text in
      let generated = generate cfg params ~max_tokens:20 prompt in
      let continuation =
        Array.sub generated (Array.length prompt)
          (Array.length generated - Array.length prompt)
      in
      Printf.printf "  %S ->\n    %s\n\n%!" text (decode tokenizer continuation))

let () =
  match run "gpt2" with
  | () -> ()
  | exception Failure msg ->
      Printf.eprintf "Skipping gpt2 example: %s\n" msg;
      Printf.eprintf "(Is the network available?)\n"
