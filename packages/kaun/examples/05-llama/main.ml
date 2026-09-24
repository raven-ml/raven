(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text generation with pretrained Llama 3.2 1B.

   Downloads the checkpoint and tokenizer from an ungated HuggingFace mirror
   (about 2.5 GB, cached afterwards) and samples a continuation of a prompt
   through key-value caches. [--jit DEVICE] compiles the decode step with
   [Rune.jit]; [--temperature 0] decodes greedily. *)

open Kaun

(* One step function serves the whole generation: it consumes the tokens its
   index places, fills the caches, and returns the next token, the advanced
   index, the next key and the written caches, so its output feeds the next
   call. Positions, slots, the key and the sampling parameters enter as tensors:
   [Rune.jit2] compiles a prefill and one single-token step, and a captured
   temperature would be frozen into them. *)
let generate (type b) ?device cfg (params : (float, b) Nx.t Llama.params)
    (dt : (float, b) Nx.dtype) ~temperature ~top_k ~top_p ~seed ~max_tokens
    prompt =
  let module Step = struct
    type t = {
      token : Nx.int32_t;
      index : Cache_index.t;
      key : Nx.Rng.key;
      temperature : Nx.float32_t;
      k : Nx.int32_t;
      p : Nx.float32_t;
      caches : (float, b) Nx.t Llama.Cache.t;
    }

    let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) s =
      {
        token = f s.token;
        index = Cache_index.map f s.index;
        key = f s.key;
        temperature = f s.temperature;
        k = f s.k;
        p = f s.p;
        caches = Llama.Cache.map f s.caches;
      }

    let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) a b =
      {
        token = f a.token b.token;
        index = Cache_index.map2 f a.index b.index;
        key = f a.key b.key;
        temperature = f a.temperature b.temperature;
        k = f a.k b.k;
        p = f a.p b.p;
        caches = Llama.Cache.map2 f a.caches b.caches;
      }

    let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) s =
      f s.token;
      Cache_index.iter f s.index;
      f s.key;
      f s.temperature;
      f s.k;
      f s.p;
      Llama.Cache.iter f s.caches
  end in
  let greedy = temperature <= 0.0 in
  let step (s : Step.t) =
    let seq = Nx.dim 1 s.token in
    let h, caches = Llama.cached cfg params s.caches s.index s.token in
    (* The last position's logits, at float32 for the masks and the draw. *)
    let logits =
      Nx.cast Nx.float32
        (Llama.logits cfg params (Nx.slice [ A; I (seq - 1) ] h))
    in
    let keys = Nx.Rng.split s.key in
    let next =
      if greedy then Nx.argmax ~axis:1 logits
      else
        Nx.Rng.categorical keys.(1)
          (Fn.keep_top_p ~p:s.p
             (Fn.keep_top_k ~k:s.k
                (Nx.div logits (Nx.reshape [| 1; 1 |] s.temperature))))
    in
    {
      s with
      token = Nx.reshape [| 1; 1 |] next;
      index = Cache_index.advance s.index;
      key = keys.(0);
      caches;
    }
  in
  let step =
    match device with
    | None -> step
    | Some device ->
        Rune.jit_step ~device
          (module Nx.Ptree)
          (module Step)
          (fun _ s -> step s)
          (Nx.Ptree.list [])
  in
  let n0 = Array.length prompt in
  let context = n0 + max_tokens in
  let state =
    ref
      (step
         {
           Step.token = Nx.create Nx.int32 [| 1; n0 |] prompt;
           index = Cache_index.rows ~context [| n0 |];
           key = Nx.Rng.key seed;
           temperature = Nx.scalar Nx.float32 (Float.max temperature 1e-6);
           k = Nx.scalar Nx.int32 (Int32.of_int top_k);
           p = Nx.scalar Nx.float32 top_p;
           caches = Llama.cache cfg ~slots:context dt;
         })
  in
  let out = Array.make max_tokens 0l in
  (* The first single-token step compiles under [--jit]: time from the
     second. *)
  let t0 = ref (Unix.gettimeofday ()) in
  for n = 0 to max_tokens - 1 do
    out.(n) <- Nx.item [ 0; 0 ] !state.Step.token;
    if n = 1 then t0 := Unix.gettimeofday ();
    if n < max_tokens - 1 then state := step !state
  done;
  if max_tokens > 2 then
    Printf.printf "%.2f tok/s\n%!"
      (float_of_int (max_tokens - 2) /. (Unix.gettimeofday () -. !t0));
  out

let load_tokenizer () =
  let path = Kaun_hf.download_file ~file:"tokenizer.json" Llama.default_repo in
  match Brot.from_file path with
  | Ok t -> t
  | Error e -> failwith ("tokenizer: " ^ e)

let () =
  let prompt = ref "The capital of France is" in
  let count = ref 24 and jit = ref "" in
  let dtype = ref "" and temperature = ref 0.7 and seed = ref 0 in
  let top_k = ref 50 and top_p = ref 0.9 in
  Arg.parse
    [
      ("--prompt", Arg.Set_string prompt, "Text to continue");
      ("--count", Arg.Set_int count, "Number of tokens to generate");
      ("--jit", Arg.Set_string jit, "Compile the decode step for this device");
      ( "--dtype",
        Arg.Set_string dtype,
        "float32, float16 or bfloat16 (default: the checkpoint's own)" );
      ("--temperature", Arg.Set_float temperature, "0 decodes greedily");
      ("--top-k", Arg.Set_int top_k, "Keep the k most likely tokens");
      ("--top-p", Arg.Set_float top_p, "Keep the smallest set reaching mass p");
      ("--seed", Arg.Set_int seed, "Sampling seed");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "llama [--prompt P] [--count N] [--jit DEVICE] [--dtype DT]";
  let cfg = Llama.config_of_json (Kaun_hf.load_config Llama.default_repo) in
  let ckpt = Kaun_hf.load_checkpoint Llama.default_repo in
  let tokenizer = load_tokenizer () in
  (* The tokenizer opens the ids with the begin-of-text token the model was
     trained to start from. *)
  let ids = Array.map Int32.of_int (Brot.encode_ids tokenizer !prompt) in
  let device = if !jit = "" then None else Some !jit in
  (* At the checkpoint's own dtype the import casts nothing. *)
  let (Llama.Dtype dt) =
    if !dtype = "" then Llama.stored_dtype ckpt
    else Llama.dtype_of_string !dtype
  in
  let toks =
    generate ?device cfg
      (Llama.of_hf ?device cfg dt ckpt)
      dt ~temperature:!temperature ~top_k:!top_k ~top_p:!top_p ~seed:!seed
      ~max_tokens:!count ids
  in
  print_string !prompt;
  print_endline (Brot.decode tokenizer (Array.map Int32.to_int toks))
