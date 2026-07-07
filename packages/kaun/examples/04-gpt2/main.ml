(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text generation with pretrained GPT-2.

   Loads the 124M-parameter GPT-2 checkpoint — from a local safetensors file
   when one is cached, downloading from the HuggingFace Hub otherwise (~548MB,
   cached afterwards) — and greedily generates continuations of a prompt. [--jit
   DEVICE] compiles the forward pass with [Rune.jit]. *)

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

(* Jitted greedy decoding with a key-value cache. One jitted function serves the
   whole generation: it consumes tokens at position [pos], fills the caches, and
   returns the next token, the advanced position and the updated caches — its
   output feeds the next call directly. The position enters the graph as a
   tensor, so [Rune.jit2] compiles exactly two variants: a prefill over the
   whole prompt, and a single-token step replayed for every generated token. *)

type step = {
  token : Nx.int32_t; (* [| 1; seq |]: the prompt, then one token at a time *)
  pos : Nx.int32_t; (* [| 1 |], the position of [token]'s first entry *)
  caches : Gpt2.cache;
}

module Step = struct
  type t = step

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { token; pos; caches } =
    {
      token = f token;
      pos = f pos;
      caches = List.map (Kaun.Attention.map_cache f) caches;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      token = f a.token b.token;
      pos = f a.pos b.pos;
      caches = List.map2 (Kaun.Attention.map2_cache f) a.caches b.caches;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { token; pos; caches } =
    f token;
    f pos;
    List.iter (Kaun.Attention.iter_cache f) caches
end

let generate_jit ~device cfg params ~max_tokens prompt =
  let n0 = Array.length prompt in
  let len = n0 + max_tokens in
  let tokens = Array.make len 0l in
  Array.blit prompt 0 tokens 0 n0;
  let step_fn =
    Rune.jit2 ~device
      (module Step)
      (module Step)
      (fun { token; pos; caches } ->
        let seq = (Nx.shape token).(1) in
        let logits, caches = Gpt2.logits_cached cfg params ~pos caches token in
        {
          token = Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 logits);
          pos = Nx.add_s pos (Int32.of_int seq);
          caches;
        })
  in
  let t0 = Unix.gettimeofday () in
  let state =
    ref
      (step_fn
         {
           token = Nx.create Nx.int32 [| 1; n0 |] prompt;
           pos = Nx.zeros Nx.int32 [| 1 |];
           caches = Gpt2.cache cfg ~len;
         })
  in
  tokens.(n0) <- Nx.item [ 0; 0 ] !state.token;
  let prefill = Unix.gettimeofday () -. t0 in
  let times = Array.make (max 1 (max_tokens - 1)) 0. in
  for n = n0 + 1 to len - 1 do
    let t0 = Unix.gettimeofday () in
    let s = step_fn !state in
    tokens.(n) <- Nx.item [ 0; 0 ] s.token;
    state := s;
    times.(n - n0 - 1) <- Unix.gettimeofday () -. t0
  done;
  if max_tokens > 2 then begin
    let rest = Array.fold_left ( +. ) (-.times.(0)) times in
    Printf.printf
      "prefill %.2f s (compile), first step %.2f s (compile), then %.2f tok/s\n\
       %!"
      prefill times.(0)
      (float_of_int (max_tokens - 2) /. rest)
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
  print_endline text
