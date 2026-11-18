(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Chat with gpt-oss: a harmony conversation decoded greedily through
   key-value caches, eagerly on the host or compiled over [--devices]. *)

open Kaun

(* The most likely next token after [ids], and the caches with their keys and
   values written. *)
let greedy ~jit s cfg p =
  let placement = Nx.Placement.on s in
  if jit then Layer_loop.greedy ~placement cfg p
  else fun caches index ids ->
    let index = Cache_index.place placement index in
    let ids = Nx.place placement ids in
    let h, caches = Gpt_oss.cached cfg p caches index ids in
    let last = Nx.slice [ A; I (Nx.dim 1 h - 1) ] h in
    (Nx.argmax ~axis:1 (Gpt_oss.logits cfg p last), caches)

(* Decodes [prompt] and up to [count] tokens after it, giving each to
   [on_token] until it says to stop. *)
let decode step caches ~count ~on_token prompt =
  let n0 = Array.length prompt in
  let next caches index ids =
    let token, caches = step caches index ids in
    (Nx.item [ 0 ] token, caches)
  in
  let t0 = Unix.gettimeofday () in
  let index = Cache_index.rows ~context:(n0 + count) [| n0 |] in
  let token, caches =
    next caches index (Nx.create Nx.int64 [| 1; n0 |] prompt)
  in
  let t1 = Unix.gettimeofday () in
  let rec go n token caches index =
    if on_token token || n = count then n
    else
      let index = Cache_index.advance index in
      let token, caches =
        next caches index (Nx.create Nx.int64 [| 1; 1 |] [| token |])
      in
      go (n + 1) token caches index
  in
  let n = go 1 token caches index in
  let rate = float_of_int (n - 1) /. (Unix.gettimeofday () -. t1) in
  Printf.eprintf "\nprefill of %d tokens: %.2f s; %d tokens, %.1f a second\n"
    n0 (t1 -. t0) n rate

(* Prints the answer on standard output and, if [show_analysis], everything else
   the assistant writes on standard error, each channel on its own lines. Says
   whether the assistant ended its turn. *)
let printer harmony ~show_analysis =
  let parser = ref (Harmony.parser harmony) and last = ref Harmony.Final in
  fun token ->
    let p, text = Harmony.feed !parser (Int64.to_int token) in
    parser := p;
    Option.iter
      (fun (channel, text) ->
        let oc = if channel = Harmony.Final then stdout else stderr in
        if channel <> !last && !last <> Harmony.Final && show_analysis then
          prerr_newline ();
        last := channel;
        if channel = Harmony.Final || show_analysis then begin
          output_string oc text;
          flush oc
        end)
      text;
    Harmony.stopped p

let () =
  let repo = ref "tiny-random/gpt-oss-mxfp4" and devices = ref "" in
  let prompt = ref "What is the capital of France?" and count = ref 256 in
  let show_analysis = ref false in
  Arg.parse
    [
      ( "--repo",
        Arg.Set_string repo,
        "A gpt-oss repository (default tiny-random/gpt-oss-mxfp4)" );
      ( "--devices",
        Arg.Set_string devices,
        "Compile over these devices, expert-parallel over several: a device \
         (CUDA) or a comma-separated list (CPU:1,CPU:2)" );
      ("--prompt", Arg.Set_string prompt, "What the user says");
      ("--count", Arg.Set_int count, "Most tokens to generate (default 256)");
      ( "--show-analysis",
        Arg.Set show_analysis,
        "Print the model's reasoning on standard error" );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "main.exe [--repo REPO] [--devices LIST] [--prompt TEXT] [--count N] \
     [--show-analysis]";
  let cfg = Gpt_oss.config_of_json (Kaun_hf.load_config !repo) in
  let weights = Kaun_hf.load_safetensors !repo in
  let harmony =
    Harmony.of_file (Kaun_hf.download_file ~file:"tokenizer.json" !repo)
  in
  let ids = Harmony.render harmony !prompt in
  let on_token = printer harmony ~show_analysis:!show_analysis in
  (* At the weights' own dtype the import casts nothing. *)
  let (Gpt_oss.Dtype dt) = Gpt_oss.stored_dtype weights in
  let run ~jit s =
    let placement = Gpt_oss.expert_parallel s in
    let p = Gpt_oss.of_hf ~placement cfg dt weights in
    let cfg = Gpt_oss.place_config (Nx.Placement.on s) cfg in
    let caches =
      Gpt_oss.cache ~placement cfg ~slots:(Array.length ids + !count) dt
    in
    decode (greedy ~jit s cfg p) caches ~count:!count ~on_token
      (Array.map Int64.of_int ids)
  in
  if !devices = "" then run ~jit:false Nx.Host.v
  else
    let open_ name = Result.error_to_failure (Rig_gpus.open_ name) in
    let names = List.map String.trim (String.split_on_char ',' !devices) in
    let module D = (val Nx.devices (List.map open_ names)) in
    run ~jit:true D.v
