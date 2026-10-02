(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Greedy decoding with gpt-oss through key-value caches. It shows the decode
   loop and times it: one step function serves the prefill and every
   single-token step. Under [--devices LIST] the step is {!Layer_loop.greedy}:
   each layer kind is compiled once per call shape and the host calls it once
   per layer, so the whole prompt goes through in one call. Over several devices
   it runs expert-parallel: each device holds an equal share of the experts, and
   the mixture sums their products across the devices.

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

   With [--profile FILE], the prompt's call is profiled and its events written
   to [FILE] in Chrome's trace format: on its first call the step is traced and
   compiled, so the profile shows what that costs.

   Usage: main.exe [--repo REPO] [--devices LIST] [--dtype DT] [--count N]
   [--profile FILE] [--prompt TEXT [--system TEXT]
   [--reasoning low|medium|high] [--show-analysis]]. *)

open Kaun

let fixed_prompt =
  [|
    200006L;
    17360L;
    200008L;
    3575L;
    553L;
    17554L;
    162016L;
    11L;
    261L;
    4410L;
    6439L;
    2359L;
  |]

(* [profiled file f] is [f ()]. Under [Some file], it writes the profile of
   [f ()] to [file]. *)
let profiled file f =
  match file with
  | None -> f ()
  | Some file ->
      let p = Nx_device.Profile.start () in
      let r = f () in
      let events = Nx_device.Profile.stop p in
      Out_channel.with_open_bin file (fun oc ->
          Nx_device.Profile.output_chrome_trace oc events);
      r

(* [on_token] sees every generated token as it arrives and says whether to stop.
   With [devices], the step compiles for them and the caches are placed there as
   the parameters are. Under [profile], the prompt's call is profiled. *)
let generate ?devices ?profile cfg params dt ~log ~count ~on_token prompt =
  let placement = Option.map Nx.Placement.on devices in
  let step = Layer_loop.greedy ?placement cfg params in
  let timed caches index ids =
    let t0 = Unix.gettimeofday () in
    let token, caches = step caches index ids in
    let token = Nx.item [ 0 ] token in
    (token, caches, Unix.gettimeofday () -. t0)
  in
  let n0 = Array.length prompt in
  let context = n0 + count in
  let index = Cache_index.rows ~context [| n0 |] in
  let first, caches, prefill =
    profiled profile (fun () ->
        timed
          (Gpt_oss.cache
             ?placement:(Option.map Gpt_oss.expert_parallel devices)
             cfg ~slots:context dt)
          index
          (Nx.create Nx.int64 [| 1; n0 |] prompt))
  in
  Printf.fprintf log "prefill of %d tokens: %.3f s\n%!" n0 prefill;
  let out = Array.make count first in
  let times = Array.make (count - 1) 0.0 in
  let state = ref (caches, Cache_index.advance index) and n = ref 1 in
  let stop = ref (on_token first) in
  while (not !stop) && !n < count do
    let caches, index = !state in
    let ids = Nx.create Nx.int64 [| 1; 1 |] [| out.(!n - 1) |] in
    let token, caches, t = timed caches index ids in
    state := (caches, Cache_index.advance index);
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

let effort_of_string = function
  | "low" -> Harmony.Low
  | "medium" -> Harmony.Medium
  | "high" -> Harmony.High
  | s -> invalid_arg ("reasoning " ^ s ^ ": expected low, medium or high")

(* The devices [s] names, as in ["CUDA:0,CUDA:1"], opened. *)
let devices_of s =
  Result.fold ~ok:Nx.Device.all ~error:failwith (Nx.Device.of_string s)

let () =
  let repo = ref "tiny-random/gpt-oss-mxfp4" in
  let devices = ref "" and count = ref 0 and dtype = ref "" in
  let profile = ref "" in
  let prompt = ref "" and system = ref "" and reasoning = ref "medium" in
  let show_analysis = ref false in
  Arg.parse
    [
      ( "--repo",
        Arg.Set_string repo,
        "A gpt-oss repository (default tiny-random/gpt-oss-mxfp4)" );
      ( "--devices",
        Arg.Set_string devices,
        "Compile the step over these devices, expert-parallel over several: a \
         device (METAL) or a comma-separated list (CPU:1,CPU:2,CPU:3,CPU:4)" );
      ( "--dtype",
        Arg.Set_string dtype,
        "float32 or bfloat16 (default: the checkpoint's own)" );
      ( "--count",
        Arg.Set_int count,
        "Most tokens to generate (default: 16, or 256 for a prompt)" );
      ( "--profile",
        Arg.Set_string profile,
        "Write the profile of the prompt's call to this file, in Chrome's \
         trace format" );
      ("--prompt", Arg.Set_string prompt, "What the user says");
      ("--system", Arg.Set_string system, "Instructions for the model");
      ("--reasoning", Arg.Set_string reasoning, "low, medium (default) or high");
      ( "--show-analysis",
        Arg.Set show_analysis,
        "Print the model's reasoning on standard error" );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "main.exe [--repo REPO] [--devices LIST] [--dtype DT] [--count N] \
     [--profile FILE] [--prompt TEXT [--system TEXT] [--reasoning EFFORT] \
     [--show-analysis]]";
  let effort = effort_of_string !reasoning in
  let cfg = Gpt_oss.config_of_json (Kaun_hf.load_config !repo) in
  let ckpt = Kaun_hf.load_checkpoint !repo in
  (* At the checkpoint's own dtype the import casts nothing. *)
  let (Gpt_oss.Dtype dt) =
    if !dtype = "" then Gpt_oss.stored_dtype ckpt
    else Gpt_oss.dtype_of_string !dtype
  in
  let devices =
    if !devices = "" then None else Some (devices_of !devices)
  in
  let count default = if !count > 0 then !count else default in
  let profile = if !profile = "" then None else Some !profile in
  let log = if !prompt = "" then stdout else stderr in
  let t0 = Unix.gettimeofday () in
  let params =
    Gpt_oss.of_hf
      ?placement:(Option.map Gpt_oss.expert_parallel devices)
      cfg dt ckpt
  in
  Printf.fprintf log "weights imported in %.1f s\n%!"
    (Unix.gettimeofday () -. t0);
  if !prompt = "" then
    let out =
      generate ?devices ?profile cfg params dt ~log ~count:(count 16)
        ~on_token:(fun _ -> false)
        fixed_prompt
    in
    print_endline
      (String.concat " " (Array.to_list (Array.map Int64.to_string out)))
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
      generate ?devices ?profile cfg params dt ~log ~count:(count 256)
        ~on_token
        (Array.map Int64.of_int ids)
    in
    print_newline ();
    Printf.eprintf "%d tokens generated\n" (Array.length out)
  end
