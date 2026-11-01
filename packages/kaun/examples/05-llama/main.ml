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
   computes; [--temperature 0] decodes greedily. Over several devices, memory
   devices ([cpu:k]) among them, the step compiles as over one.

   [--gguf FILE] reads the configuration and the weights from a GGUF checkpoint
   instead, quantised or not, decoded at float32 unless [--dtype] says
   otherwise; the tokenizer still comes from [--repo], which must be the
   checkpoint's model: [--gguf tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --repo
   TinyLlama/TinyLlama-1.1B-Chat-v1.0]. *)

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

(* [bucket n] is the least power of two at or above [n]. *)
let bucket n =
  let rec up w = if w >= n then w else up (2 * w) in
  up 1

(* One step function serves the whole generation: it reads the tokens its index
   places, the key and the sampling parameters, is given the caches, donated,
   fills them, and returns the next token, the next key and the written caches.
   The host makes the index of each call and places it. Positions and slots
   enter as tensors, so a program [Rune.jit] compiles serves every position of
   one shape. The caches, the key and the sampling parameters are made where
   the parameters are, [placement] placing each as [Llama.of_hf] did.

   The shapes are bucketed, since [Rune.jit] compiles one program per shape: the
   prefill pads the prompt on the left to a power of two, and a call reads the
   table's first [w] columns, [w] the least power of two above its last
   position. A decode of [n] tokens compiles about log2 [n] widths, and each
   call reads at most twice the live context where a fixed width would read the
   whole cache. A width compiles when the decode first reaches it, a stall of
   about 1 s on the CPU from an empty disk cache and 0.2 s from a warm one. *)
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
  (* The table names slot [j] at column [j] up to the decode's length, and no
     slot past it. Prefill pads the prompt on the left with tokens at position
     [-1], which store nothing and which no token sees. *)
  let width = bucket context in
  let table =
    Nx.create Nx.int64 [| 1; width |]
      (Array.init width (fun j -> if j < context then Int64.of_int j else -1L))
  in
  let index pos =
    let seq = Array.length pos in
    let w = bucket (pos.(seq - 1) + 1) in
    Cache_index.make
      ~pos:(Nx.create Nx.int64 [| 1; seq |] (Array.map Int64.of_int pos))
      ~table:(Nx.slice [ A; R (0, w) ] table)
      ()
  in
  let sampling =
    {
      temperature = Nx.scalar_at whole Nx.float32 (Float.max temperature 1e-6);
      k = Nx.scalar_at whole Nx.int64 (Int64.of_int top_k);
      p = Nx.scalar_at whole Nx.float32 top_p;
    }
  in
  let seq = bucket n0 in
  let pad = seq - n0 in
  let state =
    ref
      (step
         (Nx.place whole
            (Nx.create Nx.int64 [| 1; seq |]
               (Array.append (Array.make pad 0L) prompt)))
         (index (Array.init seq (fun i -> max (-1) (i - pad))))
         (Nx.Rng.place whole (Nx.Rng.key seed))
         sampling
         (Llama.cache ~placement cfg ~slots:context dt))
  in
  let out = Array.make max_tokens 0L in
  (* The first step at each width compiles under [--devices]: the rate counts
     the steps that replay a program. *)
  let steps = ref 0 and stepping = ref 0. in
  for n = 0 to max_tokens - 1 do
    let (token, key), caches = !state in
    out.(n) <- Nx.item [ 0; 0 ] token;
    if n < max_tokens - 1 then begin
      let pos = n0 + n in
      let t0 = Unix.gettimeofday () in
      state := step token (index [| pos |]) key sampling caches;
      (* Reading the token waits for the step. *)
      ignore (Nx.item [ 0; 0 ] (fst (fst !state)) : int64);
      if n > 0 && bucket (pos + 1) = bucket pos then begin
        incr steps;
        stepping := !stepping +. (Unix.gettimeofday () -. t0)
      end
    end
  done;
  if !steps > 0 then
    Printf.printf "%.2f tok/s\n%!" (float_of_int !steps /. !stepping);
  out

(* Where the configuration and the weights come from. *)
type checkpoint = Hf of Nx.host Nx_io.Archive.t | Gguf of Nx_io.Gguf.t

let load_tokenizer repo =
  let path = Kaun_hf.download_file ~file:"tokenizer.json" repo in
  match Brot.from_file path with
  | Ok t -> t
  | Error e -> failwith ("tokenizer: " ^ e)

let () =
  let prompt = ref "The capital of France is" in
  let count = ref 24 and devices = ref "" in
  let dtype = ref "" and temperature = ref 0.7 and seed = ref 0 in
  let top_k = ref 50 and top_p = ref 0.9 in
  let repo = ref Llama.default_repo and gguf = ref "" in
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
      ( "--repo",
        Arg.Set_string repo,
        "HuggingFace repository of the model and its tokenizer (default: Llama \
         3.2 1B)" );
      ("--gguf", Arg.Set_string gguf, "Read the model from this GGUF file");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "llama [--prompt P] [--count N] [--devices LIST] [--dtype DT] [--repo R] \
     [--gguf FILE]";
  if !count < 0 then begin
    prerr_endline "llama: --count must not be negative";
    exit 2
  end;
  let checkpoint =
    if !gguf = "" then Hf (Kaun_hf.load_safetensors !repo)
    else Gguf (Nx_io.load_gguf !gguf)
  in
  let cfg =
    match checkpoint with
    | Hf _ -> Llama.config_of_json (Kaun_hf.load_config !repo)
    | Gguf g -> Llama.config_of_gguf g
  in
  let tokenizer = load_tokenizer !repo in
  (* The tokenizer opens the ids with the begin-of-text token the model was
     trained to start from. *)
  let ids = Array.map Int64.of_int (Brot.encode_ids tokenizer !prompt) in
  (* At the weights' own dtype the import casts nothing, and float32 holds a
     quantised file's decoded values exactly. *)
  let (Llama.Dtype dt) =
    match (!dtype, checkpoint) with
    | "", Hf weights -> Llama.stored_dtype weights
    | "", Gguf _ -> Llama.Dtype Nx.float32
    | d, _ -> Llama.dtype_of_string d
  in
  let run ~jit placement =
    let cfg = Llama.place_config (placement Llama.Whole ~axis:0) cfg in
    let params =
      match checkpoint with
      | Hf weights -> Llama.of_hf ~placement cfg dt weights
      | Gguf g -> Llama.of_gguf ~placement cfg dt g
    in
    generate ~jit ~placement cfg params dt ~temperature:!temperature
      ~top_k:!top_k ~top_p:!top_p ~seed:!seed ~max_tokens:!count ids
  in
  let toks =
    if !devices = "" then run ~jit:false (fun _ ~axis:_ -> Nx.Host.on)
    else
      let module D = (val Nx.devices (Devices.all !devices)) in
      run ~jit:true (parallel D.v)
  in
  (* The prompt's tokens decode with the continuation: a SentencePiece tokenizer
     drops the space that opens the text it decodes, so the continuation decoded
     alone would lose its first space. *)
  print_endline
    (Brot.decode tokenizer ~skip_special_tokens:true
       (Array.map Int64.to_int (Array.append ids toks)))
