(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text generation with pretrained GPT-2.

   Loads the 124M-parameter GPT-2 checkpoint — from a local safetensors file
   when one is cached, downloading from the HuggingFace Hub otherwise (~548MB,
   cached afterwards) — and greedily generates continuations of a prompt. [--jit
   DEVICE] compiles the decode step with [Rune.jit]. *)

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

let load_checkpoint () =
  match local_file "model.safetensors" with
  | Some path -> (gpt2_124m, Kaun.Checkpoint.load path)
  | None ->
      ( Gpt2.config_of_json (Kaun_hf.load_config "gpt2"),
        Kaun_hf.load_checkpoint "gpt2" )

let load_tokenizer () =
  let path =
    match local_file "tokenizer.json" with
    | Some path -> path
    | None -> Kaun_hf.download_file ~file:"tokenizer.json" "gpt2"
  in
  match Brot.from_file path with
  | Ok t -> t
  | Error e -> failwith ("tokenizer: " ^ e)

(* Greedy decoding with a key-value cache. One step function serves the whole
   generation: it consumes the tokens its index places, fills the caches, and
   returns the next token, the advanced index and the updated caches — its output
   feeds the next call directly. Positions and slots enter as tensors, so under
   [--jit] [Rune.jit2] compiles exactly two variants: a prefill over the whole
   prompt, and a single-token step replayed for every generated token, writing
   the donated caches in place.

   The step is generic over the parameters' float dtype [b]: the key-value
   caches carry the same dtype as the weights, so [--dtype float16] decodes with
   half precision weights, activations and caches alike. *)

let generate (type b) ?device cfg (params : (float, b) Nx.t Gpt2.params)
    (dt : (float, b) Nx.dtype) ~max_tokens prompt =
  let module Step = struct
    type t = {
      token : Nx.int32_t; (* [| 1; seq |]: the prompt, then one token *)
      index : Kaun.Cache_index.t; (* where [token]'s entries sit *)
      caches : (float, b) Nx.t Gpt2.Cache.t;
    }

    let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) { token; index; caches }
        =
      {
        token = f token;
        index = Kaun.Cache_index.map f index;
        caches = Gpt2.Cache.map f caches;
      }

    let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) a b =
      {
        token = f a.token b.token;
        index = Kaun.Cache_index.map2 f a.index b.index;
        caches = Gpt2.Cache.map2 f a.caches b.caches;
      }

    let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) { token; index; caches } =
      f token;
      Kaun.Cache_index.iter f index;
      Gpt2.Cache.iter f caches
  end in
  let n0 = Array.length prompt in
  let len = n0 + max_tokens in
  let tokens = Array.make len 0l in
  Array.blit prompt 0 tokens 0 n0;
  let step { Step.token; index; caches } =
    let seq = (Nx.shape token).(1) in
    let h, caches = Gpt2.cached cfg params caches index token in
    (* Only the last position's logits matter for decoding. *)
    let last = Nx.slice [ A; I (seq - 1) ] h in
    {
      Step.token =
        Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 (Gpt2.logits cfg params last));
      index = Kaun.Cache_index.advance index;
      caches;
    }
  in
  let step_fn =
    match device with
    | None -> step
    | Some device ->
        Rune.jit2 ~device ~donate:true (module Step) (module Step) step
  in
  let t0 = Unix.gettimeofday () in
  let state =
    ref
      (step_fn
         {
           Step.token = Nx.create Nx.int32 [| 1; n0 |] prompt;
           index = Kaun.Cache_index.rows ~context:len [| n0 |];
           caches = Gpt2.cache cfg ~slots:len dt;
         })
  in
  tokens.(n0) <- Nx.item [ 0; 0 ] !state.Step.token;
  let prefill = Unix.gettimeofday () -. t0 in
  let times = Array.make (max 1 (max_tokens - 1)) 0. in
  for n = n0 + 1 to len - 1 do
    let t0 = Unix.gettimeofday () in
    let s = step_fn !state in
    tokens.(n) <- Nx.item [ 0; 0 ] s.Step.token;
    state := s;
    times.(n - n0 - 1) <- Unix.gettimeofday () -. t0
  done;
  if max_tokens > 2 then begin
    let rest = Array.fold_left ( +. ) (-.times.(0)) times in
    Printf.printf
      "prefill %.2f s, first step %.2f s (both compile under --jit), then %.2f \
       tok/s\n\
       %!"
      prefill times.(0)
      (float_of_int (max_tokens - 2) /. rest)
  end;
  tokens

(* The decode contract's law on this model and these weights: the prompt fed
   through the caches in chunks gives the logits it gives whole. *)
let check cfg params dt ids =
  let n = Array.length ids in
  let tokens = Nx.create Nx.int32 [| 1; n |] ids in
  let whole = Gpt2.logits cfg params (Gpt2.hidden cfg params tokens) in
  let slots = Nx.create Nx.int32 [| 1; n |] (Array.init n Int32.of_int) in
  let _, hs, _ =
    List.fold_left
      (fun (at, hs, caches) len ->
        let len = min len (n - at) in
        if len = 0 then (at, hs, caches)
        else
          let pos =
            Nx.create Nx.int32 [| 1; len |]
              (Array.init len (fun i -> Int32.of_int (at + i)))
          in
          let h, caches =
            Gpt2.cached cfg params caches
              (Kaun.Cache_index.make ~pos ~table:slots ())
              (Nx.slice [ A; R (at, at + len) ] tokens)
          in
          (at + len, h :: hs, caches))
      (0, [], Gpt2.cache cfg ~slots:n dt)
      [ 1; 7; n ]
  in
  let chunked =
    Gpt2.logits cfg params (Nx.concatenate ~axis:1 (List.rev hs))
  in
  let scale = Nx.item [] (Nx.max (Nx.abs whole)) in
  let worst = Nx.item [] (Nx.max (Nx.abs (Nx.sub whole chunked))) /. scale in
  Printf.printf
    "hidden against cached in chunks of 1, 7 and the rest: worst relative \
     error %.1e\n\
     %!"
    worst;
  if not (worst < 1e-4) then exit 1

let () =
  let prompt = ref default_prompt in
  let count = ref 10 in
  let check_only = ref false in
  let jit = ref "" in
  let dtype = ref "" in
  Arg.parse
    [
      ("--prompt", Arg.Set_string prompt, "Phrase to start with");
      ("--count", Arg.Set_int count, "Max number of tokens to generate");
      ( "--check",
        Arg.Set check_only,
        "Check that the cached path reproduces the whole-sequence logits on \
         the prompt, and exit" );
      ( "--jit",
        Arg.Set_string jit,
        "Compile the forward pass for this device (CPU or CUDA); eager when \
         omitted" );
      ( "--dtype",
        Arg.Set_string dtype,
        "Model dtype: float32, float16 or bfloat16 (default: the \
         checkpoint's own, which casts nothing). Weights, activations and \
         caches run at this dtype" );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "gpt2 [--prompt P] [--count N] [--jit DEVICE] [--dtype DT]";
  let tokenizer = load_tokenizer () in
  let t0 = Unix.gettimeofday () in
  let cfg, ckpt = load_checkpoint () in
  let (Gpt2.Dtype dt) =
    if !dtype = "" then Gpt2.stored_dtype ckpt else Gpt2.dtype_of_string !dtype
  in
  let device = if !jit = "" then None else Some !jit in
  let params = Gpt2.of_hf ?device cfg dt ckpt in
  Printf.printf "loaded weights in %.2f s\n%!" (Unix.gettimeofday () -. t0);
  let ids = Array.map Int32.of_int (Brot.encode_ids tokenizer !prompt) in
  if !check_only then begin
    check cfg params dt ids;
    exit 0
  end;
  let bytes = ref 0 in
  Gpt2.Params.iter (fun t -> bytes := !bytes + Nx.nbytes t) params;
  Printf.printf "weights: %.0f MB at %s\n%!"
    (float_of_int !bytes /. 1e6)
    (Nx_core.Dtype.to_string dt);
  let t0 = Unix.gettimeofday () in
  let toks = generate ?device cfg params dt ~max_tokens:!count ids in
  let dt = Unix.gettimeofday () -. t0 in
  Printf.printf "generated %d tokens in %.2f s (%.2f tok/s)\n%!" !count dt
    (float_of_int !count /. dt);
  let text = Brot.decode tokenizer (Array.map Int32.to_int toks) in
  print_endline text
