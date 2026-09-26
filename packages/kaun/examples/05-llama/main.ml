(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text generation with pretrained Llama 3.2 1B.

   Downloads the checkpoint and tokenizer from an ungated HuggingFace mirror
   (about 2.5 GB, cached afterwards) and samples a continuation of a prompt
   through key-value caches. [--devices LIST] compiles the decode step with
   [Rune.jit] for a device, tensor-parallel over several; [--temperature 0]
   decodes greedily. *)

open Kaun

(* Tensor parallelism over [ds]: the projections into the heads and the hidden
   features split by columns, those out of them by rows, the caches on their
   kv-heads, and the rest a copy on each device. On one device every leaf is
   whole there. *)
let parallel ds role ~axis =
  match role with
  | Llama.Whole -> Nx.Placement.replicated ds
  | Column | Row | Kv_heads -> Nx.Placement.sharded ~axis ds

(* The sampling parameters are tensors, so a compiled step reads them as
   arguments: a captured temperature would be frozen into its program. *)
type sampling = { temperature : Nx.float32_t; k : Nx.int32_t; p : Nx.float32_t }

module Sampling = struct
  type _ t = sampling

  let walk c s =
    let open Nx.Ptree.Walk in
    let temperature = field c "temperature" tensor s.temperature in
    let k = field c "k" tensor s.k in
    let p = field c "p" tensor s.p in
    { temperature; k; p }
end

(* One step function serves the whole generation: it reads the tokens its index
   places, the key and the sampling parameters, consumes the caches, fills them,
   and returns the next token, the next key and the written caches. The host
   advances the index between calls. Positions and slots enter as tensors, so
   [Rune.jit] compiles a prefill and one single-token step. With [devices], the
   step compiles for them and the caches are placed there as the parameters
   are. *)
let generate (type b) ?devices cfg (params : (float, b) Nx.t Llama.params)
    (dt : (float, b) Nx.dtype) ~temperature ~top_k ~top_p ~seed ~max_tokens
    prompt =
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
        Nx.Rng.categorical keys.(1)
          (Fn.keep_top_p ~p:sampling.p
             (Fn.keep_top_k ~k:sampling.k
                (Nx.div logits (Nx.reshape [| 1; 1 |] sampling.temperature))))
    in
    ((Nx.reshape [| 1; 1 |] next, keys.(0)), caches)
  in
  let step =
    match devices with
    | None -> step
    | Some devices ->
        let sampling = Nx.Ptree.instantiate (module Sampling)
        and caches =
          Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache))
        in
        Rune.jit ~devices
          Nx.Ptree.(
            tensor @-> Cache_index.ptree @-> Nx.Rng.ptree @-> sampling
            @-> consumes caches
            @@ returns (pair (pair tensor Nx.Rng.ptree) caches))
          step
  in
  let n0 = Array.length prompt in
  let context = n0 + max_tokens in
  let sampling =
    {
      temperature = Nx.scalar Nx.float32 (Float.max temperature 1e-6);
      k = Nx.scalar Nx.int32 (Int32.of_int top_k);
      p = Nx.scalar Nx.float32 top_p;
    }
  in
  let index = ref (Cache_index.rows ~context [| n0 |]) in
  let state =
    ref
      (step
         (Nx.create Nx.int32 [| 1; n0 |] prompt)
         !index (Nx.Rng.key seed) sampling
         (Llama.cache
            ?placement:(Option.map parallel devices)
            cfg ~slots:context dt))
  in
  let out = Array.make max_tokens 0l in
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

(* A device, a CPU device count ([4] is CPU:1..CPU:4) or a comma-separated
   list. *)
let parse_devices s =
  match int_of_string_opt s with
  | Some n when n > 0 -> List.init n (fun i -> Printf.sprintf "CPU:%d" (i + 1))
  | Some _ -> failwith "--devices: the device count must be positive"
  | None -> List.map String.trim (String.split_on_char ',' s)

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
         several: a device (METAL), a CPU count (4 = CPU:1..CPU:4) or a \
         comma-separated list" );
      ( "--dtype",
        Arg.Set_string dtype,
        "float32, float16 or bfloat16 (default: the checkpoint's own)" );
      ("--temperature", Arg.Set_float temperature, "0 decodes greedily");
      ("--top-k", Arg.Set_int top_k, "Keep the k most likely tokens");
      ("--top-p", Arg.Set_float top_p, "Keep the smallest set reaching mass p");
      ("--seed", Arg.Set_int seed, "Sampling seed");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "llama [--prompt P] [--count N] [--devices LIST] [--dtype DT]";
  let cfg = Llama.config_of_json (Kaun_hf.load_config Llama.default_repo) in
  let ckpt = Kaun_hf.load_checkpoint Llama.default_repo in
  let tokenizer = load_tokenizer () in
  (* The tokenizer opens the ids with the begin-of-text token the model was
     trained to start from. *)
  let ids = Array.map Int32.of_int (Brot.encode_ids tokenizer !prompt) in
  let devices =
    if !devices = "" then None
    else Some (List.map Rune.device (parse_devices !devices))
  in
  (* At the checkpoint's own dtype the import casts nothing. *)
  let (Llama.Dtype dt) =
    if !dtype = "" then Llama.stored_dtype ckpt
    else Llama.dtype_of_string !dtype
  in
  let toks =
    generate ?devices cfg
      (Llama.of_hf ?placement:(Option.map parallel devices) cfg dt ckpt)
      dt ~temperature:!temperature ~top_k:!top_k ~top_p:!top_p ~seed:!seed
      ~max_tokens:!count ids
  in
  print_string !prompt;
  print_endline (Brot.decode tokenizer (Array.map Int32.to_int toks))
