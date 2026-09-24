(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Greedy decoding with gpt-oss through key-value caches. It shows the decode
   loop and times it: one step function serves the prefill and every
   single-token step, and under [--jit DEVICE] compiles once for each of the two
   shapes.

   The compiled prefill dequantises the packed experts of every layer before the
   first layer runs: the schedule orders a kernel that reads only weights ahead
   of the ones that wait for the tokens. That is 38 GB for gpt-oss-20b at
   bfloat16. When it exceeds [prefill_budget], or under [--stepwise], the prompt
   goes through the single-token program one token at a time instead: one
   compilation, and the experts dequantised are the four a token selects.

   With [--prompt TEXT] the text becomes the user's turn of a harmony
   conversation, [--system TEXT] its instructions and [--reasoning EFFORT] how
   much the model should think first. The answer streams to standard output as
   it is generated, the model's reasoning to standard error under
   [--show-analysis], and generation ends when the model closes its turn or
   after [--count] tokens. The timings go to standard error.

   Without [--prompt] the model continues twelve fixed token ids for [--count]
   tokens and prints the ids: no tokenizer is involved.

   The default checkpoints are tiny and random. They exercise every code path,
   but what they generate is noise: it rarely opens a message, so a prompt
   usually prints nothing but the timings.

   Usage: main.exe [--repo REPO] [--jit DEVICE] [--dtype DT] [--count N]
   [--stepwise] [--prompt TEXT [--system TEXT] [--reasoning low|medium|high]
   [--show-analysis]]. *)

open Kaun

let fixed_prompt =
  [|
    200006l;
    17360l;
    200008l;
    3575l;
    553l;
    17554l;
    162016l;
    11l;
    261l;
    4410l;
    6439l;
    2359l;
  |]

let prefill_budget = 4 lsl 30

(* The bytes of dequantised experts the compiled prefill holds at once. Float
   experts are read where they are, and an eager prefill frees each layer's
   before the next. *)
let prefill_expert_bytes ?device cfg (params : _ Gpt_oss.params) =
  match (device, params.blocks) with
  | Some _, { moe = { gate_up = Moe.Mxfp4 _; _ }; _ } :: _ ->
      List.length cfg.Gpt_oss.layers
      * cfg.experts * 3 * cfg.hidden_dim * cfg.dim
      * Nx.itemsize params.norm.gamma
  | _ -> 0

(* [on_token] sees every generated token as it arrives and says whether to
   stop. *)
let generate (type b) ?device ~stepwise cfg
    (params : (float, b) Nx.t Gpt_oss.params) (dt : (float, b) Nx.dtype) ~log
    ~count ~on_token prompt =
  let module Step = struct
    type t = {
      token : Nx.int32_t;
      index : Cache_index.t;
      caches : (float, b) Nx.t Gpt_oss.Cache.t;
    }

    let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) s =
      let token = f s.token in
      let index = Cache_index.map f s.index in
      let caches = Gpt_oss.Cache.map f s.caches in
      { token; index; caches }

    let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) a b =
      let token = f a.token b.token in
      let index = Cache_index.map2 f a.index b.index in
      let caches = Gpt_oss.Cache.map2 f a.caches b.caches in
      { token; index; caches }

    let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) s =
      f s.token;
      Cache_index.iter f s.index;
      Gpt_oss.Cache.iter f s.caches
  end in
  let step (s : Step.t) =
    let seq = Nx.dim 1 s.token in
    let h, caches = Gpt_oss.cached cfg params s.caches s.index s.token in
    let logits = Gpt_oss.logits cfg params (Nx.slice [ A; I (seq - 1) ] h) in
    {
      Step.token = Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 logits);
      index = Cache_index.advance s.index;
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
  let context = n0 + count in
  let timed s =
    let t0 = Unix.gettimeofday () in
    let s = step s in
    let token = Nx.item [ 0; 0 ] s.Step.token in
    (s, token, Unix.gettimeofday () -. t0)
  in
  let caches = Gpt_oss.cache cfg ~slots:context dt in
  let state, first, prefill =
    if stepwise then (
      let token id = Nx.create Nx.int32 [| 1; 1 |] [| id |] in
      let index = Cache_index.rows ~context [| 1 |] in
      let rec feed i (s, id, total) =
        if i = n0 then (s, id, total)
        else
          let s, id, t = timed { s with Step.token = token prompt.(i) } in
          feed (i + 1) (s, id, total +. t)
      in
      let ((_, _, compiling) as first) =
        timed { Step.token = token prompt.(0); index; caches }
      in
      Printf.fprintf log "first call: %.3f s\n%!" compiling;
      feed 1 first)
    else
      timed
        {
          Step.token = Nx.create Nx.int32 [| 1; n0 |] prompt;
          index = Cache_index.rows ~context [| n0 |];
          caches;
        }
  in
  Printf.fprintf log "prefill of %d tokens%s: %.3f s\n%!" n0
    (if stepwise then ", one at a time" else "")
    prefill;
  let out = Array.make count first in
  let times = Array.make (count - 1) 0.0 in
  let state = ref state and n = ref 1 in
  let stop = ref (on_token first) in
  while (not !stop) && !n < count do
    let s, token, t = timed !state in
    state := s;
    out.(!n) <- token;
    times.(!n - 1) <- t;
    incr n;
    stop := on_token token
  done;
  if !n > 1 then begin
    Printf.fprintf log "first step: %.3f s\n" times.(0);
    let later = Array.sub times 1 (!n - 2) in
    Array.sort compare later;
    if later <> [||] then
      Printf.fprintf log "later steps: median %.1f ms\n%!"
        (1e3 *. later.(Array.length later / 2))
  end;
  Array.sub out 0 !n

let today () =
  let tm = Unix.localtime (Unix.time ()) in
  Printf.sprintf "%04d-%02d-%02d" (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday

(* Prints the answer on standard output and, if [show_analysis], everything else
   the assistant writes on standard error, each channel on its own lines. *)
let printer harmony ~show_analysis =
  let parser = ref (Harmony.parser harmony) and last = ref Harmony.Final in
  fun token ->
    let p, text = Harmony.feed !parser (Int32.to_int token) in
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

let effort_of_string = function
  | "low" -> Harmony.Low
  | "medium" -> Harmony.Medium
  | "high" -> Harmony.High
  | s -> invalid_arg ("reasoning " ^ s ^ ": expected low, medium or high")

let () =
  let repo = ref "tiny-random/gpt-oss-mxfp4" in
  let jit = ref "" and count = ref 0 and dtype = ref "" in
  let prompt = ref "" and system = ref "" and reasoning = ref "medium" in
  let show_analysis = ref false and stepwise = ref false in
  Arg.parse
    [
      ( "--repo",
        Arg.Set_string repo,
        "A gpt-oss repository (default tiny-random/gpt-oss-mxfp4)" );
      ("--jit", Arg.Set_string jit, "Compile the step for this device");
      ( "--dtype",
        Arg.Set_string dtype,
        "float32 or bfloat16 (default: the checkpoint's own)" );
      ( "--count",
        Arg.Set_int count,
        "Most tokens to generate (default: 16, or 256 for a prompt)" );
      ( "--stepwise",
        Arg.Set stepwise,
        "Feed the prompt one token at a time (default: when the compiled \
         prefill would dequantise more than 4 GiB of experts)" );
      ("--prompt", Arg.Set_string prompt, "What the user says");
      ("--system", Arg.Set_string system, "Instructions for the model");
      ("--reasoning", Arg.Set_string reasoning, "low, medium (default) or high");
      ( "--show-analysis",
        Arg.Set show_analysis,
        "Print the model's reasoning on standard error" );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "main.exe [--repo REPO] [--jit DEVICE] [--dtype DT] [--count N] \
     [--stepwise] [--prompt TEXT [--system TEXT] [--reasoning EFFORT] \
     [--show-analysis]]";
  let effort = effort_of_string !reasoning in
  let cfg = Gpt_oss.config_of_json (Kaun_hf.load_config !repo) in
  let ckpt = Kaun_hf.load_checkpoint !repo in
  (* At the checkpoint's own dtype the import casts nothing. *)
  let (Gpt_oss.Dtype dt) =
    if !dtype = "" then Gpt_oss.stored_dtype ckpt
    else Gpt_oss.dtype_of_string !dtype
  in
  let device = if !jit = "" then None else Some !jit in
  let count default = if !count > 0 then !count else default in
  let log = if !prompt = "" then stdout else stderr in
  let t0 = Unix.gettimeofday () in
  let params = Gpt_oss.of_hf ?device cfg dt ckpt in
  Printf.fprintf log "weights imported in %.1f s\n%!"
    (Unix.gettimeofday () -. t0);
  let stepwise =
    !stepwise || prefill_expert_bytes ?device cfg params > prefill_budget
  in
  if !prompt = "" then
    let out =
      generate ?device ~stepwise cfg params dt ~log ~count:(count 16)
        ~on_token:(fun _ -> false)
        fixed_prompt
    in
    print_endline
      (String.concat " " (Array.to_list (Array.map Int32.to_string out)))
  else begin
    let harmony =
      Harmony.of_file (Kaun_hf.download_file ~file:"tokenizer.json" !repo)
    in
    let instructions = if !system = "" then None else Some !system in
    let ids =
      Harmony.render harmony ~date:(today ()) ?instructions ~effort
        [ User !prompt ]
    in
    let on_token = printer harmony ~show_analysis:!show_analysis in
    let out =
      generate ?device ~stepwise cfg params dt ~log ~count:(count 256) ~on_token
        (Array.map Int32.of_int ids)
    in
    print_newline ();
    Printf.eprintf "%d tokens generated\n" (Array.length out)
  end
