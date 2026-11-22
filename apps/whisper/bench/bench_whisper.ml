(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* PERFORMANCE.md's rows: fetch.py's clip, the first minute of its
   recording, transcribed as main.exe transcribes it (Transcript.default in
   English: greedy, the fallback ladder, timestamps) on the device
   WHISPER_DEVICE names, as Rig_gpus.open_ reads it: by tiny eagerly on the
   host ([cpu], the default), or by tiny, small and large-v3-turbo at
   float32, compiled on a GPU ([cuda], [nv], [amd]). A minute is three
   windows, the first two of 30 s, so it transcribes at the whole
   recording's rate (PERFORMANCE.md). Each row reports the device memory in
   use once loaded, or the process's peak resident memory on the host. Each
   device's suite keeps a baseline file of its own, so that blessing one
   device's rows leaves the others': whisper.thumper for the host's,
   whisper-cuda.thumper for CUDA's.

   [--warm] compiles every kernel into tolk's disk cache first, in a process
   of its own. [--recording DEVICE MODEL...] times, outside thumper, the
   encoder alone and the whole recording, and [--profile MODEL] runs the
   encoder and the clip once each on CUDA for DEBUG=2's kernel times
   (raven.sh). Without fetch.py's files there is nothing to measure. *)

open Whisper

let gpu_models = [ "tiny"; "small"; "large-v3-turbo" ]
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

let device = Option.value ~default:"cpu" (Sys.getenv_opt "WHISPER_DEVICE")

(* Whether [d] names the host, read from the name, so that this process
   opens no driver. *)
let host d = String.lowercase_ascii d = "cpu"

(* The suite, which names the baseline file. *)
let suite =
  if host device then "whisper" else "whisper-" ^ String.lowercase_ascii device

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

let clip () = Audio.load (path "clip.wav")

(* [transcriber at ~jit name] loads [name] at float32 on [at]: a function
   transcribing samples, its steps compiled where [jit], and one running the
   encoder alone on the clip's first window, its result summed to a scalar
   read on the host. *)
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
  let encode m mel = Nx.sum (Model.encode m mel) in
  let encode =
    if not jit then encode
    else
      Nx.Ptree.(
        Rune.jit
          (instantiate (module Model.Params) @-> tensor @-> returns tensor)
          encode)
  in
  let mel =
    Audio.log_mel ~n_mels:m.dims.n_mels (Audio.pad_or_trim (clip ()))
    |> Nx.unsqueeze ~axes:[ 0 ] |> Nx.place at
  in
  ( (fun x -> Transcript.transcribe ~key steps m o x),
    fun () -> ignore (Nx.item [] (Nx.place Nx.Host.on (encode m mel))) )

(* [on device name] is [name]'s {!transcriber} on [device], and a function
   stating the memory in use. *)
let on device name =
  if host device then
    let peak () =
      In_channel.with_open_text "/proc/self/status" In_channel.input_lines
      |> List.find_map (fun l -> Scanf.sscanf_opt l "VmHWM: %d kB" Fun.id)
      |> Option.fold ~none:"" ~some:(fun kb ->
          Printf.sprintf "%.2f GB peak resident" (float kb *. 1024. /. 1e9))
    in
    (transcriber Nx.Host.on ~jit:false name, peak)
  else
    let d = Result.error_to_failure (Rig_gpus.open_ device) in
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
  let row = (if host device then "minute-cpu/" else "minute/") ^ name in
  Thumper.bench_with_setup row
    ~setup:(fun () ->
      let (transcribe, _), memory = on device name in
      let x = clip () in
      ignore (transcribe x);
      Printf.eprintf "whisper: %s: %s\n%!" row (memory ());
      (x, transcribe))
    (fun (x, transcribe) -> transcribe x)

let row name s audio n =
  Printf.printf "%-28s %10.1f ms  %7.1fx real time  (%d calls)\n%!" name
    (s *. 1e3) (audio /. s) n

let timed f =
  let t0 = Unix.gettimeofday () in
  let y = f () in
  (Unix.gettimeofday () -. t0, y)

(* The encoder alone on a window, the median of 5 calls after 1; then the
   whole recording, once, after a call on the clip; in the reference scripts'
   format. *)
let recording device name =
  let (transcribe, encode), memory = on device name in
  ignore (transcribe (clip ()));
  encode ();
  let t = List.sort compare (List.init 5 (fun _ -> fst (timed encode))) in
  row ("encoder/" ^ name) (List.nth t 2) 30. 5;
  let x = Audio.load (path "recording.wav") in
  let s, _ = timed (fun () -> transcribe x) in
  row ("recording/" ^ name) s (float (Nx.dim 0 x) /. float Audio.rate) 1;
  Printf.printf "%-28s %s\n%!" ("memory/" ^ name) (memory ())

(* After a call on the clip and one of the encoder, one more of each, each
   after a line naming it, for raven.sh to sum the kernel times that DEBUG=2
   prints after it. *)
let profile name =
  let (transcribe, encode), _ = on "cuda" name in
  let x = clip () in
  ignore (transcribe x);
  encode ();
  print_endline "profile: encoder";
  encode ();
  print_endline "profile: minute";
  ignore (transcribe x);
  print_endline "profile: end"

let () =
  match Array.to_list Sys.argv with
  | [ _; "--warm" ] ->
      if fetched () && not (host device) then
        List.iter
          (fun name ->
            let (transcribe, encode), _ = on device name in
            ignore (transcribe (clip ()));
            encode ();
            Gc.full_major ())
          gpu_models
  | [ _; "--profile"; name ] -> profile name
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
      let cases =
        if host device then List.map (case device) cpu_models
        else begin
          Result.error_to_failure (Rig_gpus.opens device);
          List.map (case device) gpu_models
        end
      in
      exit
        (Thumper.run suite
           ~config:Thumper.Config.(default |> samples 5 |> deadline 120.)
           ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
           [ Thumper.group "Whisper" cases ])
