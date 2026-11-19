(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* PERFORMANCE.md's rows: fetch.py's clip, the first minute of its
   recording, transcribed as main.exe transcribes it (Transcript.default in
   English: greedy, the fallback ladder, timestamps), each release at
   float32, compiled on the first CUDA, NV or AMD GPU, and tiny eagerly on
   the host. A minute is three windows, the first two of 30 s, so it
   transcribes at the whole recording's rate (PERFORMANCE.md). Each row
   reports the device memory in use once loaded, or the process's peak
   resident memory on the host.

   [--warm] compiles every kernel into tolk's disk cache first, in a process
   of its own. [--recording cuda|cpu MODEL...] times the whole recording
   instead, once, outside thumper (raven.sh). Without fetch.py's files there
   is nothing to measure. *)

open Whisper

let gpu_models = [ "tiny"; "small"; "large-v3-turbo"; "large-v3" ]
let cpu_models = [ "tiny" ]

let dir =
  let cache =
    match Sys.getenv_opt "RAVEN_CACHE_ROOT" with
    | Some d -> d
    | None -> (
        match Sys.getenv_opt "XDG_CACHE_HOME" with
        | Some d -> Filename.concat d "raven"
        | None -> Filename.concat (Sys.getenv "HOME") ".cache/raven")
  in
  Filename.concat cache "whisper"

let path f = Filename.concat dir f

let fetched () =
  List.for_all (fun m -> Sys.file_exists (path (m ^ ".pt"))) gpu_models

(* The GPUs tried, in order, as gpt-oss's bench tries them. *)
let names = [ "CUDA"; "NV"; "AMD" ]

let gpu () =
  let rec go = function
    | [] -> failwith "bench_whisper: no CUDA, NV or AMD GPU"
    | name :: rest -> (
        match Rig_gpus.open_ name with Ok d -> d | Error _ -> go rest)
  in
  go names

let tokenizer (m : _ Model.t) =
  let file =
    if Model.multilingual m.dims then "multilingual.tiktoken"
    else "gpt2.tiktoken"
  in
  let ranks =
    In_channel.with_open_bin (path ("ranks/" ^ file)) In_channel.input_all
  in
  match Tokenizer.of_ranks ~languages:(Model.languages m.dims) ranks with
  | Ok t -> t
  | Error e -> failwith e

(* [transcriber at ~jit name] loads [name] at float32 on [at]: a function
   transcribing samples, its steps compiled where [jit], and its tokenizer. *)
let transcriber at ~jit name =
  let m = Model.load Nx.float32 (path (name ^ ".pt")) in
  let t = tokenizer m in
  let m = Nx.Ptree.place (module Model.Params) at m in
  let compile =
    if jit then Some { Transcript.compile = (fun s f -> Rune.jit s f) }
    else None
  in
  let steps = Transcript.steps ?compile t (Decoding.options t) in
  let o = { Transcript.default with language = Some "en" } in
  let key = Nx.Rng.key 0 in
  fun x -> Transcript.transcribe ~key steps m o x

let clip () = Audio.load (path "clip.wav")

(* [on device name] is [name]'s transcriber on [device], and a function
   stating the memory in use. *)
let on device name =
  match device with
  | "cpu" ->
      let peak () =
        In_channel.with_open_text "/proc/self/status" In_channel.input_lines
        |> List.find_map (fun l -> Scanf.sscanf_opt l "VmHWM: %d kB" Fun.id)
        |> Option.fold ~none:"" ~some:(fun kb ->
            Printf.sprintf "%.2f GB peak resident" (float kb *. 1024. /. 1e9))
      in
      (transcriber Nx.Host.on ~jit:false name, peak)
  | _ ->
      let d = gpu () in
      let module G = (val Nx.devices [ d ]) in
      (* The device memory in use as the driver reports it, every process's:
         this one's alone on an otherwise empty GPU. *)
      let in_use () =
        Option.fold ~none:""
          ~some:(fun { Rig.total; free } ->
            Printf.sprintf "%.2f GB in use" (float (total - free) /. 1e9))
          (Rig.capacity d)
      in
      (transcriber G.on ~jit:true name, in_use)

let case device name =
  let row = (if device = "cpu" then "minute-cpu/" else "minute/") ^ name in
  Thumper.bench_with_setup row
    ~setup:(fun () ->
      let transcribe, memory = on device name in
      let x = clip () in
      ignore (transcribe x);
      Printf.eprintf "whisper: %s: %s\n%!" row (memory ());
      (x, transcribe))
    (fun (x, transcribe) -> transcribe x)

(* The whole recording, once, after a call on the clip, its text written
   beside it for wer.py, in the reference scripts' format. *)
let recording device name =
  let transcribe, memory = on device name in
  ignore (transcribe (clip ()));
  let x = Audio.load (path "recording.wav") in
  let t0 = Unix.gettimeofday () in
  let r = transcribe x in
  let s = Unix.gettimeofday () -. t0 in
  let audio = float (Nx.dim 0 x) /. float Audio.rate in
  Printf.printf "%-28s %10.1f ms  %7.1fx real time  (1 call)\n"
    ("recording/" ^ name) (s *. 1e3) (audio /. s);
  Printf.printf "%-28s %s\n%!" ("memory/" ^ name) (memory ());
  Out_channel.with_open_bin
    (path (Printf.sprintf "raven-%s-%s.txt" name device))
    (fun oc -> output_string oc r.text)

let () =
  match Array.to_list Sys.argv with
  | [ _; "--warm" ] ->
      if fetched () && Result.is_ok (Rig_gpus.first ()) then
        List.iter
          (fun name ->
            ignore ((fst (on "cuda" name)) (clip ()));
            Gc.full_major ())
          gpu_models
  | _ :: "--recording" :: device :: models ->
      List.iter
        (fun name ->
          recording device name;
          Gc.full_major ())
        models
  | _ ->
      if not (fetched ()) then begin
        prerr_endline
          ("whisper: no checkpoints in " ^ dir
         ^ ", nothing to measure (apps/whisper/bench/fetch.py fetches them)");
        exit 0
      end;
      let gpu =
        if List.exists (fun n -> Result.is_ok (Rig_gpus.opens n)) names then
          List.map (case "cuda") gpu_models
        else []
      in
      exit
        (Thumper.run "whisper"
           ~config:Thumper.Config.(default |> samples 5 |> deadline 120.)
           ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
           [ Thumper.group "Whisper" (gpu @ List.map (case "cpu") cpu_models) ])
