(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text generation with pretrained GPT-2.

   Loads the 124M-parameter GPT-2 checkpoint — from a local safetensors file
   when one is cached, downloading from the HuggingFace Hub otherwise (~548MB,
   cached afterwards) — and greedily generates continuations of a prompt.
   [--jit DEVICE] compiles the forward pass with [Rune.jit]. *)

let default_prompt = "What is the answer to life, the universe, and everything?"

(* Loading: prefer local files for determinism (the same cache the tolk gpt2
   example fills), fall back to the HuggingFace Hub. *)

let local_file file =
  let cache =
    try Sys.getenv "XDG_CACHE_HOME"
    with Not_found -> Filename.concat (Sys.getenv "HOME") ".cache"
  in
  let path = List.fold_left Filename.concat cache [ "tolk-gpt2"; file ] in
  if Sys.file_exists path then Some path else None

let gpt2_124m : Gpt2.config =
  {
    vocab_size = 50257;
    n_positions = 1024;
    n_embd = 768;
    n_layer = 12;
    n_head = 12;
    n_inner = 3072;
    layer_norm_eps = 1e-5;
  }

let load_model () =
  match local_file "model.safetensors" with
  | Some path -> (gpt2_124m, Gpt2.from_file gpt2_124m path)
  | None -> Gpt2.from_pretrained ()

let load_tokenizer () =
  let path =
    match local_file "tokenizer.json" with
    | Some path -> path
    | None -> Kaun_hf.download_file ~file:"tokenizer.json" "gpt2"
  in
  match Brot.from_file path with
  | Ok t -> t
  | Error e -> failwith ("tokenizer: " ^ e)

(* Jitted greedy decoding. The input window is fixed at [prompt + count]
   tokens so that a single compilation serves every step: with the causal
   mask, the logits at position n-1 are unaffected by the padding that
   follows, so each step reads them from a full-window forward pass. *)

let generate_jit ~device cfg params ~max_tokens prompt =
  let n0 = Array.length prompt in
  let window = n0 + max_tokens in
  let tokens = Array.make window 0l in
  Array.blit prompt 0 tokens 0 n0;
  let step = Rune.jit' ~device (Gpt2.logits cfg params) in
  let times = Array.make max_tokens 0. in
  for n = n0 to window - 1 do
    let t0 = Unix.gettimeofday () in
    let input = Nx.create Nx.int32 [| 1; window |] tokens in
    let last = Nx.slice [ I 0; I (n - 1) ] (step input) in
    tokens.(n) <- Nx.item [] (Nx.argmax ~axis:0 last);
    times.(n - n0) <- Unix.gettimeofday () -. t0
  done;
  if max_tokens > 1 then begin
    let rest = Array.fold_left ( +. ) (-.times.(0)) times in
    Printf.printf "first token %.2f s (compile), then %.2f tok/s\n%!" times.(0)
      (float_of_int (max_tokens - 1) /. rest)
  end;
  tokens

let () =
  let prompt = ref default_prompt in
  let count = ref 10 in
  let jit = ref "" in
  Arg.parse
    [
      ("--prompt", Arg.Set_string prompt, "Phrase to start with");
      ("--count", Arg.Set_int count, "Max number of tokens to generate");
      ( "--jit",
        Arg.Set_string jit,
        "Compile the forward pass for this device (CPU or CUDA); eager when \
         omitted" );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "gpt2 [--prompt P] [--count N] [--jit DEVICE]";
  let tokenizer = load_tokenizer () in
  let t0 = Unix.gettimeofday () in
  let cfg, params = load_model () in
  Printf.printf "loaded weights in %.2f s\n%!" (Unix.gettimeofday () -. t0);
  let ids = Array.map Int32.of_int (Brot.encode_ids tokenizer !prompt) in
  let t0 = Unix.gettimeofday () in
  let toks =
    match !jit with
    | "" -> Gpt2.generate cfg params ~max_tokens:!count ids
    | device -> generate_jit ~device cfg params ~max_tokens:!count ids
  in
  let dt = Unix.gettimeofday () -. t0 in
  Printf.printf "generated %d tokens in %.2f s (%.2f tok/s)\n%!" !count dt
    (float_of_int !count /. dt);
  let text = Brot.decode tokenizer (Array.map Int32.to_int toks) in
  print_endline "Generating text...";
  print_endline text
