(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text generation with pretrained Llama 3.2 1B.

   Downloads the checkpoint and tokenizer from an ungated HuggingFace mirror
   (about 2.5 GB, cached afterwards) and samples a continuation of a prompt
   through key-value caches. [--jit DEVICE] compiles the decode step with
   [Rune.jit]; [--temperature 0] decodes greedily.

   The prompt is given as token ids. Llama 3's tokenizer splits text with a
   regular expression that uses lookahead, which brot does not load yet, so this
   example cannot encode text; it decodes the generated ids itself from the
   tokenizer file's vocabulary. The default ids spell "The capital of France
   is". *)

open Kaun
module Span = Attention.Span

let default_ids = "128000,791,6864,315,9822,374"

(* One step function serves the whole generation: it consumes the tokens its
   span places, fills the caches, and returns the next token, the advanced span,
   the next key and the written caches, so its output feeds the next call.
   Positions, slots, the key and the sampling parameters enter as tensors:
   [Rune.jit2] compiles a prefill and one single-token step, and a captured
   temperature would be frozen into them. *)
let generate (type b) ?device cfg (params : (float, b) Nx.t Llama.params)
    (dt : (float, b) Nx.dtype) ~temperature ~top_k ~top_p ~seed ~max_tokens
    prompt =
  let module Step = struct
    type t = {
      token : Nx.int32_t;
      span : Span.t;
      key : Nx.Rng.key;
      temperature : Nx.float32_t;
      k : Nx.int32_t;
      p : Nx.float32_t;
      caches : (float, b) Nx.t Llama.Cache.t;
    }

    let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) s =
      {
        token = f s.token;
        span = Span.map f s.span;
        key = f s.key;
        temperature = f s.temperature;
        k = f s.k;
        p = f s.p;
        caches = Llama.Cache.map f s.caches;
      }

    let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) a b =
      {
        token = f a.token b.token;
        span = Span.map2 f a.span b.span;
        key = f a.key b.key;
        temperature = f a.temperature b.temperature;
        k = f a.k b.k;
        p = f a.p b.p;
        caches = Llama.Cache.map2 f a.caches b.caches;
      }

    let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) s =
      f s.token;
      Span.iter f s.span;
      f s.key;
      f s.temperature;
      f s.k;
      f s.p;
      Llama.Cache.iter f s.caches
  end in
  let greedy = temperature <= 0.0 in
  let step (s : Step.t) =
    let seq = Nx.dim 1 s.token in
    let h, caches = Llama.cached cfg params s.caches s.span s.token in
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
          (Fn.top_p ~p:s.p
             (Fn.top_k ~k:s.k
                (Nx.div logits (Nx.reshape [| 1; 1 |] s.temperature))))
    in
    {
      s with
      token = Nx.reshape [| 1; 1 |] next;
      span = Span.advance s.span;
      key = keys.(0);
      caches;
    }
  in
  let step =
    match device with
    | None -> step
    | Some device ->
        Rune.jit2 ~device ~donate:true (module Step) (module Step) step
  in
  let n0 = Array.length prompt in
  let context = n0 + max_tokens in
  let state =
    ref
      (step
         {
           Step.token = Nx.create Nx.int32 [| 1; n0 |] prompt;
           span = Span.rows ~context [| n0 |];
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

(* The byte-level vocabulary of the tokenizer file, by id, with the special
   tokens it adds, so an end-of-text marker shows in the output. *)
let vocabulary path =
  let ic = open_in_bin path in
  let text =
    Fun.protect
      ~finally:(fun () -> close_in ic)
      (fun () -> In_channel.input_all ic)
  in
  let mem name = function
    | Jsont.Object (mems, _) -> Option.map snd (Jsont.Json.find_mem name mems)
    | _ -> None
  in
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Error e -> failwith ("tokenizer.json: " ^ e)
  | Ok json -> (
      match Option.bind (mem "model" json) (mem "vocab") with
      | Some (Jsont.Object (mems, _)) ->
          let table = Hashtbl.create (List.length mems) in
          List.iter
            (function
              | (token, _), Jsont.Number (id, _) ->
                  Hashtbl.replace table (int_of_float id) token
              | _ -> ())
            mems;
          (match mem "added_tokens" json with
          | Some (Jsont.Array (added, _)) ->
              List.iter
                (fun t ->
                  match (mem "id" t, mem "content" t) with
                  | Some (Jsont.Number (id, _)), Some (Jsont.String (c, _)) ->
                      Hashtbl.replace table (int_of_float id) c
                  | _ -> ())
                added
          | _ -> ());
          table
      | _ -> failwith "tokenizer.json: no model.vocab")

let decode vocab ids =
  Array.to_list ids
  |> List.filter_map (fun id -> Hashtbl.find_opt vocab (Int32.to_int id))
  |> String.concat "" |> Brot.Pre_tokenizer.byte_level_decode

let () =
  let ids = ref default_ids and count = ref 24 and jit = ref "" in
  let dtype = ref "float32" and temperature = ref 0.7 and seed = ref 0 in
  let top_k = ref 50 and top_p = ref 0.9 in
  Arg.parse
    [
      ("--ids", Arg.Set_string ids, "Prompt as comma-separated token ids");
      ("--count", Arg.Set_int count, "Number of tokens to generate");
      ("--jit", Arg.Set_string jit, "Compile the decode step for this device");
      ("--dtype", Arg.Set_string dtype, "float32 (default), float16 or bfloat16");
      ("--temperature", Arg.Set_float temperature, "0 decodes greedily");
      ("--top-k", Arg.Set_int top_k, "Keep the k most likely tokens");
      ("--top-p", Arg.Set_float top_p, "Keep the smallest set reaching mass p");
      ("--seed", Arg.Set_int seed, "Sampling seed");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "llama [--ids I,J,...] [--count N] [--jit DEVICE] [--dtype DT]";
  let cfg, params = Llama.from_pretrained () in
  let vocab =
    vocabulary (Kaun_hf.download_file ~file:"tokenizer.json" Llama.default_repo)
  in
  let ids =
    String.split_on_char ',' !ids
    |> List.map (fun s -> Int32.of_string (String.trim s))
    |> Array.of_list
  in
  let device = if !jit = "" then None else Some !jit in
  let run : type b.
      (float, b) Nx.dtype -> (float, b) Nx.t Llama.params -> int32 array =
   fun dt params ->
    generate ?device cfg params dt ~temperature:!temperature ~top_k:!top_k
      ~top_p:!top_p ~seed:!seed ~max_tokens:!count ids
  in
  let toks =
    match !dtype with
    | "float32" -> run Nx.float32 params
    | "float16" -> run Nx.float16 (Llama.Params.map (Nx.cast Nx.float16) params)
    | "bfloat16" ->
        run Nx.bfloat16 (Llama.Params.map (Nx.cast Nx.bfloat16) params)
    | d -> failwith ("--dtype must be float32, float16 or bfloat16, got " ^ d)
  in
  print_string (decode vocab ids);
  print_endline (decode vocab toks)
