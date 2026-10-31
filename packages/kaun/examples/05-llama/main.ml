(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text generation with pretrained Llama 3.2 1B.

   Downloads the weights and tokenizer from an ungated HuggingFace mirror (about
   2.5 GB, cached afterwards) and samples a continuation of a prompt through
   key-value caches. Without [--devices] it decodes eagerly on the host.
   [--devices LIST] places the model on a device, tensor-parallel over several,
   and compiles the decode step for them with [Rune.jit], which is how a GPU
   computes; [--temperature 0] decodes greedily. rune compiles a program for
   one device, so over several devices the step is evaluated by nx: it runs on
   memory devices ([cpu:k]) and raises on a set of GPUs, which compute only
   compiled programs. *)

open Kaun

(* Tensor parallelism over the set [s]: the projections into the heads and the
   hidden features split by columns, those out of them by rows, the caches on
   their kv-heads, and the rest a copy on each device. On one device every leaf
   is whole there. *)
let parallel s role ~axis =
  match role with
  | Llama.Whole -> Nx.Placement.on s
  | Column | Row | Kv_heads -> Nx.Placement.split ~axis s

(* The sampling parameters are tensors, so a compiled step reads them as
   arguments: a captured temperature would be frozen into its program. *)
type 'd sampling = {
  temperature : 'd Nx.float32_t;
  k : 'd Nx.int64_t;
  p : 'd Nx.float32_t;
}

module Sampling = struct
  type (_, 'd) t = 'd sampling

  let walk c s =
    let open Nx.Ptree.Walk in
    let temperature = field c "temperature" tensor s.temperature in
    let k = field c "k" tensor s.k in
    let p = field c "p" tensor s.p in
    { temperature; k; p }
end

(* One step function serves the whole generation: it reads the tokens its index
   places, the key and the sampling parameters, is given the caches, donated,
   fills them, and returns the next token, the next key and the written caches.
   The host advances the index and places it for each call. Positions and slots
   enter as tensors, so [Rune.jit] compiles a prefill and one single-token step.
   The caches, the key and the sampling parameters are made where the
   parameters are, [placement] placing each as [Llama.of_hf] did. *)
let generate (type b d) ~jit
    ~(placement : Llama.role -> axis:int -> d Nx.Placement.t) cfg
    (params : ((float, b, d) Nx.t, d) Llama.params) (dt : (float, b) Nx.dtype)
    ~temperature ~top_k ~top_p ~seed ~max_tokens prompt =
  let whole = placement Whole ~axis:0 in
  let greedy = temperature <= 0.0 in
  let step token index key sampling caches =
    let seq = Nx.dim 1 token in
    let h, caches = Llama.cached cfg params caches index token in
    (* The last position's logits, at float32 for the masks and the draw. *)
    let logits =
      Nx.cast Nx.float32
        (Llama.logits cfg params (Nx.slice [ A; I (seq - 1) ] h))
    in
    let keys = Nx.Rng.split key in
    let next =
      if greedy then Nx.argmax ~axis:1 logits
      else
        Nx.Rng.categorical ~key:keys.(1)
          (Fn.keep_top_p ~p:sampling.p
             (Fn.keep_top_k ~k:sampling.k
                (Nx.div logits (Nx.reshape [| 1; 1 |] sampling.temperature))))
    in
    ((Nx.reshape [| 1; 1 |] next, keys.(0)), caches)
  in
  let step =
    if not jit then fun token index key sampling caches ->
      step token (Cache_index.place whole index) key sampling caches
    else
      let sampling = Nx.Ptree.instantiate (module Sampling)
      and caches = Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache))
      and key = Nx.Rng.ptree in
      let compiled =
        Rune.jit
          Nx.Ptree.(
            tensor @-> v Cache_index.walk @-> key @-> sampling @-> caches
            @-> returns (pair (pair tensor key) caches))
          step
      in
      fun token index key s cs ->
        compiled token
          (Cache_index.place whole index)
          key s
          (Nx.Ptree.donate caches cs)
  in
  let n0 = Array.length prompt in
  let context = n0 + max_tokens in
  let sampling =
    {
      temperature = Nx.scalar_at whole Nx.float32 (Float.max temperature 1e-6);
      k = Nx.scalar_at whole Nx.int64 (Int64.of_int top_k);
      p = Nx.scalar_at whole Nx.float32 top_p;
    }
  in
  let index = ref (Cache_index.rows ~context [| n0 |]) in
  let state =
    ref
      (step
         (Nx.place whole (Nx.create Nx.int64 [| 1; n0 |] prompt))
         !index
         (Nx.Rng.place whole (Nx.Rng.key seed))
         sampling
         (Llama.cache ~placement cfg ~slots:context dt))
  in
  let out = Array.make max_tokens 0L in
  (* The first single-token step compiles under [--devices]: time from the
     second. *)
  let t0 = ref (Unix.gettimeofday ()) in
  for n = 0 to max_tokens - 1 do
    let (token, key), caches = !state in
    out.(n) <- Nx.item [ 0; 0 ] token;
    if n = 1 then t0 := Unix.gettimeofday ();
    if n < max_tokens - 1 then begin
      index := Cache_index.advance !index;
      state := step token !index key sampling caches
    end
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
  let count = ref 24 and devices = ref "" in
  let dtype = ref "" and temperature = ref 0.7 and seed = ref 0 in
  let top_k = ref 50 and top_p = ref 0.9 in
  Arg.parse
    [
      ("--prompt", Arg.Set_string prompt, "Text to continue");
      ("--count", Arg.Set_int count, "Number of tokens to generate");
      ( "--devices",
        Arg.Set_string devices,
        "Compile the decode step over these devices, tensor-parallel over \
         several: a device (METAL) or a comma-separated list \
         (CPU:1,CPU:2,CPU:3,CPU:4)" );
      ( "--dtype",
        Arg.Set_string dtype,
        "float32, float16 or bfloat16 (default: the weights' own)" );
      ("--temperature", Arg.Set_float temperature, "0 decodes greedily");
      ("--top-k", Arg.Set_int top_k, "Keep the k most likely tokens");
      ("--top-p", Arg.Set_float top_p, "Keep the smallest set reaching mass p");
      ("--seed", Arg.Set_int seed, "Sampling seed");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "llama [--prompt P] [--count N] [--devices LIST] [--dtype DT]";
  let cfg = Llama.config_of_json (Kaun_hf.load_config Llama.default_repo) in
  let weights = Kaun_hf.load_safetensors Llama.default_repo in
  let tokenizer = load_tokenizer () in
  (* The tokenizer opens the ids with the begin-of-text token the model was
     trained to start from. *)
  let ids = Array.map Int64.of_int (Brot.encode_ids tokenizer !prompt) in
  (* At the weights' own dtype the import casts nothing. *)
  let (Llama.Dtype dt) =
    if !dtype = "" then Llama.stored_dtype weights
    else Llama.dtype_of_string !dtype
  in
  let run ~jit placement =
    let cfg = Llama.place_config (placement Llama.Whole ~axis:0) cfg in
    generate ~jit ~placement cfg
      (Llama.of_hf ~placement cfg dt weights)
      dt ~temperature:!temperature ~top_k:!top_k ~top_p:!top_p ~seed:!seed
      ~max_tokens:!count ids
  in
  let toks =
    if !devices = "" then run ~jit:false (fun _ ~axis:_ -> Nx.Host.on)
    else
      let module D = (val Nx.devices (Devices.all !devices)) in
      run ~jit:true (parallel D.v)
  in
  print_string !prompt;
  print_endline (Brot.decode tokenizer (Array.map Int64.to_int toks))
