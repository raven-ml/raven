(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text generation with pretrained GPT-2.

   Loads the 124M-parameter GPT-2 weights — from a local safetensors file when
   one is cached, downloading from the HuggingFace Hub otherwise (~548MB, cached
   afterwards) — and greedily generates continuations of a prompt. [--jit
   DEVICE] places the model on DEVICE and compiles the decode step for it with
   [Rune.jit], which is how a GPU computes. *)

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

let load_weights () =
  match local_file "model.safetensors" with
  | Some path -> (gpt2_124m, Nx_io.load_safetensors path)
  | None ->
      ( Gpt2.config_of_json (Kaun_hf.load_config "gpt2"),
        Kaun_hf.load_safetensors "gpt2" )

let load_tokenizer () =
  let path =
    match local_file "tokenizer.json" with
    | Some path -> path
    | None -> Kaun_hf.download_file ~file:"tokenizer.json" "gpt2"
  in
  match Brot.from_file path with
  | Ok t -> t
  | Error e -> failwith ("tokenizer: " ^ e)

(* [bucket n] is the least power of two at or above [n]. *)
let bucket n =
  let rec up w = if w >= n then w else up (2 * w) in
  up 1

(* Greedy decoding with a key-value cache. One step function serves the whole
   generation: it reads the tokens its index places, fills the caches, and
   returns the next token and the updated caches, and the host makes the index
   of each call. Positions and slots enter as tensors, so under [--jit] a
   compiled program serves every position of one shape, and the caches are
   donated to it. The caches are made where the model lies, and the index, made
   on the host, is placed there for each call.

   The shapes are bucketed, since [Rune.jit] compiles one program per shape: the
   prefill pads the prompt on the left to a power of two, and a call reads the
   table's first [w] columns, [w] the least power of two above its last
   position. A decode of [n] tokens compiles about log2 [n] widths, and each
   call reads at most twice the live context where a fixed width would read the
   whole cache. A width compiles when the decode first reaches it, a stall of
   about 1 s on the CPU from an empty disk cache and 0.2 s from a warm one.

   The step is generic over the parameters' float dtype [b]: the key-value
   caches carry the same dtype as the weights, so [--dtype float16] decodes with
   half precision weights, activations and caches alike. *)

let generate (type b d) ~jit cfg (params : ((float, b, d) Nx.t, d) Gpt2.params)
    (dt : (float, b) Nx.dtype) ~max_tokens prompt =
  let at = Nx.placement params.ln_f.gamma in
  let n0 = Array.length prompt in
  let len = n0 + max_tokens in
  let tokens = Array.make len 0L in
  Array.blit prompt 0 tokens 0 n0;
  let step token index caches =
    let seq = (Nx.shape token).(1) in
    let h, caches = Gpt2.cached cfg params caches index token in
    (* Only the last position's logits matter for decoding. *)
    let last = Nx.slice [ A; I (seq - 1) ] h in
    ( Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 (Gpt2.logits cfg params last)),
      caches )
  in
  let step =
    if not jit then fun token index caches ->
      step token (Kaun.Cache_index.place at index) caches
    else
      let caches =
        Nx.Ptree.list (Nx.Ptree.instantiate (module Kaun.Attention.Cache))
      in
      let compiled =
        Rune.jit
          Nx.Ptree.(
            tensor @-> v Kaun.Cache_index.walk @-> caches
            @-> returns (pair tensor caches))
          step
      in
      fun token index cs ->
        compiled token
          (Kaun.Cache_index.place at index)
          (Nx.Ptree.donate caches cs)
  in
  (* The table names slot [j] at column [j] up to the decode's length, and no
     slot past it. Prefill pads the prompt on the left with tokens at position
     [-1], which store nothing and which no token sees. *)
  let width = bucket len in
  if width > cfg.Gpt2.n_positions then
    invalid_arg
      (Printf.sprintf "generate: a table of %d columns exceeds n_positions %d"
         width cfg.n_positions);
  let table =
    Nx.create Nx.int64 [| 1; width |]
      (Array.init width (fun j -> if j < len then Int64.of_int j else -1L))
  in
  let index pos =
    let seq = Array.length pos in
    let w = bucket (pos.(seq - 1) + 1) in
    Kaun.Cache_index.make
      ~pos:(Nx.create Nx.int64 [| 1; seq |] (Array.map Int64.of_int pos))
      ~table:(Nx.slice [ A; R (0, w) ] table)
      ()
  in
  let t0 = Unix.gettimeofday () in
  let seq = bucket n0 in
  let pad = seq - n0 in
  let state =
    ref
      (step
         (Nx.place at
            (Nx.create Nx.int64 [| 1; seq |]
               (Array.append (Array.make pad 0L) prompt)))
         (index (Array.init seq (fun i -> max (-1) (i - pad))))
         (Gpt2.cache ~placement:(fun _ ~axis:_ -> at) cfg ~slots:len dt))
  in
  tokens.(n0) <- Nx.item [ 0; 0 ] (fst !state);
  let prefill = Unix.gettimeofday () -. t0 in
  (* The first step at each width compiles under [--jit]: those are timed apart
     from the steps that replay a program. *)
  let widths = ref 0 and compiling = ref 0. in
  let steps = ref 0 and stepping = ref 0. in
  for n = n0 + 1 to len - 1 do
    let t0 = Unix.gettimeofday () in
    let token, caches = !state in
    state := step token (index [| n - 1 |]) caches;
    tokens.(n) <- Nx.item [ 0; 0 ] (fst !state);
    let elapsed = Unix.gettimeofday () -. t0 in
    if n = n0 + 1 || bucket n <> bucket (n - 1) then begin
      incr widths;
      compiling := !compiling +. elapsed
    end
    else begin
      incr steps;
      stepping := !stepping +. elapsed
    end
  done;
  if !steps > 0 then
    Printf.printf
      "prefill %.2f s, first steps at %d widths %.2f s (all compile under \
       --jit), then %.2f tok/s\n\
       %!"
      prefill !widths !compiling
      (float_of_int !steps /. !stepping);
  tokens

(* The decode contract's law on this model and these weights: the prompt fed
   through the caches in chunks gives the logits it gives whole. With [jit], the
   law is compiled where the weights are. *)
let check ~jit cfg params dt ids =
  let n = Array.length ids in
  let at = Nx.placement params.Gpt2.ln_f.gamma in
  (* The largest magnitude of the whole prompt's logits, and the largest
     difference of the chunked prompt's from them. The positions and slots are
     made where the tokens lie. *)
  let errors tokens =
    let here = Nx.placement tokens in
    let range lo hi =
      Nx.reshape [| 1; hi - lo |] (Nx.arange_at here Nx.int64 lo hi 1)
    in
    let whole = Gpt2.logits cfg params (Gpt2.hidden cfg params tokens) in
    let slots = range 0 n in
    let _, hs, _ =
      List.fold_left
        (fun (at, hs, caches) len ->
          let len = min len (n - at) in
          if len = 0 then (at, hs, caches)
          else
            let pos = range at (at + len) in
            let h, caches =
              Gpt2.cached cfg params caches
                (Kaun.Cache_index.make ~pos ~table:slots ())
                (Nx.slice [ A; R (at, at + len) ] tokens)
            in
            (at + len, h :: hs, caches))
        (0, [], Gpt2.cache ~placement:(fun _ ~axis:_ -> here) cfg ~slots:n dt)
        [ 1; 7; n ]
    in
    let chunked =
      Gpt2.logits cfg params (Nx.concatenate ~axis:1 (List.rev hs))
    in
    (Nx.max (Nx.abs whole), Nx.max (Nx.abs (Nx.sub whole chunked)))
  in
  let errors =
    if not jit then errors
    else Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) errors
  in
  let scale, diff = errors (Nx.place at (Nx.create Nx.int64 [| 1; n |] ids)) in
  let worst = Nx.item [] diff /. Nx.item [] scale in
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
        "Compile the forward pass for this device (CPU, METAL or CUDA:i); \
         eager when omitted" );
      ( "--dtype",
        Arg.Set_string dtype,
        "Model dtype: float32, float16 or bfloat16 (default: the weights' own, \
         which casts nothing). Weights, activations and caches run at this \
         dtype" );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "gpt2 [--prompt P] [--count N] [--jit DEVICE] [--dtype DT]";
  if !count < 1 then begin
    prerr_endline "gpt2: --count must be positive";
    exit 2
  end;
  let tokenizer = load_tokenizer () in
  let t0 = Unix.gettimeofday () in
  let cfg, weights = load_weights () in
  let (Gpt2.Dtype dt) =
    if !dtype = "" then Gpt2.stored_dtype weights
    else Gpt2.dtype_of_string !dtype
  in
  let ids = Array.map Int64.of_int (Brot.encode_ids tokenizer !prompt) in
  let run ~jit at =
    let params = Gpt2.of_hf ~placement:(fun _ ~axis:_ -> at) cfg dt weights in
    Printf.printf "loaded weights in %.2f s\n%!" (Unix.gettimeofday () -. t0);
    if !check_only then begin
      check ~jit cfg params dt ids;
      exit 0
    end;
    let bytes =
      Nx.Ptree.fold
        (Nx.Ptree.instantiate (module Gpt2.Params))
        (fun _ t n -> n + Nx.nbytes t)
        params 0
    in
    Printf.printf "weights: %.0f MB at %s\n%!"
      (float_of_int bytes /. 1e6)
      (Nx.Dtype.name dt);
    let t0 = Unix.gettimeofday () in
    let toks = generate ~jit cfg params dt ~max_tokens:!count ids in
    let dt = Unix.gettimeofday () -. t0 in
    Printf.printf "generated %d tokens in %.2f s (%.2f tok/s)\n%!" !count dt
      (float_of_int !count /. dt);
    let text = Brot.decode tokenizer (Array.map Int64.to_int toks) in
    print_endline text
  in
  if !jit = "" then run ~jit:false Nx.Host.on
  else
    let module D = (val Nx.devices [ Devices.first !jit ]) in
    run ~jit:true D.on
