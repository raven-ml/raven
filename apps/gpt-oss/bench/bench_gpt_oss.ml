(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* gpt-oss-20b's rows in PERFORMANCE.md, on the first CUDA, NV or AMD GPU that
   opens, or on the device GPT_OSS_DEVICE names ([cpu] for the host), compiled
   one layer at a time ([Layer_loop]):

   - [decode/512], one decode step at position 512, its token read back;
   - [prefill/512], a prompt of 512 tokens;
   - [first-token/12], a new process from its start to its first token after a
     12-token prompt, loading the weights included, over a disk cache of kernels
     that one run filled, and [first-token-cold/12] over an empty one.

   These keep the checkpoint as it is stored, MXFP4 experts and the rest at
   bfloat16, as the chat, vLLM and transformers run it. [decode-q8_0/512] and
   [prefill-q8_0/512] pack the attention projections and the head as Q8_0, as
   llama.cpp's GGUF file of gpt-oss-20b stores them. The scripts beside this
   file run the same rows with each of those.

   The weights are openai/gpt-oss-20b's own, read from raven's cache: a
   prefill's time depends on how its router spreads the tokens over the
   experts, and random weights spread them evenly. Each step case loads them in
   the worker thumper forks for it, so no GPU driver starts before the fork.
   [--warm] compiles every kernel into tolk's disk cache first, in a process of
   its own. Without such a device, or without the weights, there is nothing to
   measure. *)

open Kaun

let repo = "openai/gpt-oss-20b"
let revision = "6cee5e81ee83917806bbde320786a8fb61efebee"
let depth = 512
let context = 1024
let prompt = 512
let short_prompt = 12

(* Seconds of uncounted calls before a trial's samples, so a GPU leaves its
   idle clock first. *)
let warmup_seconds = 3.

let weights_cached () =
  Sys.file_exists
    (Kaun_hf.cache_path ~revision ~file:"model.safetensors.index.json" repo)

(* The devices tried, in order: the one GPT_OSS_DEVICE names as
   [Rig_gpus.open_] reads it ([cpu] for the host), else CUDA, NV and AMD.
   [Rig_gpus.first] would also try Metal and the PCI paths. Hiding the CUDA
   devices ([CUDA_VISIBLE_DEVICES=]) measures the card through NV. *)
let names =
  match Sys.getenv_opt "GPT_OSS_DEVICE" with
  | Some name -> [ name ]
  | None -> [ "CUDA"; "NV"; "AMD" ]

(* Whether the device is the host, where a call takes seconds, so that the
   rows take the fewest samples. *)
let host =
  List.for_all
    (fun n -> String.starts_with ~prefix:"CPU" (String.uppercase_ascii n))
    names

(* A suite's name, which names its baseline file: the host's rows keep a
   file of their own, [gpt_oss-cpu.thumper] beside the GPU's
   [gpt_oss.thumper]. *)
let suite name = if host then name ^ "-cpu" else name

let first () =
  let rec go why = function
    | [] -> Error (String.concat "; " (List.rev why))
    | name :: rest -> (
        match Rig_gpus.open_ name with
        | Ok d -> Ok d
        | Error e -> go (e :: why) rest)
  in
  go [] names

let gpu () = match first () with Ok d -> d | Error e -> failwith e

(* The step of gpt-oss-20b on the GPU, its attention projections and head
   packed as [pack], loaded and not yet called: [`Decode] feeds back the token
   it predicts at position [depth], [`Prefill n] runs [n] tokens at positions 0
   onwards. Each call takes the caches of the one before and reads its token
   back on the host. [on_loaded d] runs once the first call returns. *)
let load ?(on_loaded = ignore) pack kind =
  let d = gpu () in
  let module G = (val Nx.devices [ d ]) in
  let placement = G.on in
  let cfg =
    Gpt_oss.config_of_json (Kaun_hf.load_config ~offline:true ~revision repo)
  in
  let p =
    Gpt_oss.of_hf ~pack
      ~placement:(fun _ ~axis:_ -> placement)
      cfg Nx.bfloat16
      (Kaun_hf.load_safetensors ~offline:true ~revision repo)
  in
  let step =
    Layer_loop.greedy ~placement (Gpt_oss.place_config placement cfg) p
  in
  let caches =
    ref
      (Gpt_oss.cache
         ~placement:(fun _ ~axis:_ -> placement)
         cfg ~slots:context Nx.bfloat16)
  in
  let index, ids =
    match kind with
    | `Decode ->
        ( Cache_index.advance (Cache_index.rows ~context [| depth |]),
          ref (Nx.zeros Nx.int64 [| 1; 1 |]) )
    | `Prefill n ->
        ( Cache_index.rows ~context [| n |],
          ref
            (Nx.init Nx.int64 [| 1; n |] (fun i ->
                 Int64.of_int (17 + i.(1)))) )
  in
  let loaded = ref false in
  fun () ->
    let token, c = step !caches index !ids in
    caches := c;
    let token = Nx.item [ 0 ] token in
    (match kind with
    | `Decode -> ids := Nx.create Nx.int64 [| 1; 1 |] [| token |]
    | `Prefill _ -> ());
    if not !loaded then begin
      loaded := true;
      on_loaded d
    end

(* The device memory in use as the driver reports it, every process's: this
   one's alone on an otherwise empty GPU. *)
let report_memory name d =
  match Rig.capacity d with
  | Some { total; free } ->
      Printf.eprintf "gpt_oss: %s: %.2f GB of device memory in use\n%!" name
        (float_of_int (total - free) /. 1e9)
  | None -> ()

let steps =
  List.concat_map
    (fun (suffix, pack) ->
      [
        (Printf.sprintf "decode%s/%d" suffix depth, pack, `Decode);
        (Printf.sprintf "prefill%s/%d" suffix prompt, pack, `Prefill prompt);
      ])
    [ ("", Gpt_oss.Unpacked); ("-q8_0", Gpt_oss.Q8_0) ]

let step_case (name, pack, kind) =
  Thumper.bench_with_setup name
    ~setup:(fun () ->
      let call = load ~on_loaded:(report_memory name) pack kind in
      call ();
      call)
    (fun call -> call ())

(* A new process from its start to its first token, with [env] before its own
   environment. *)
let first_token env =
  let exe = Sys.executable_name in
  let env = Array.append (Array.of_list env) (Unix.environment ()) in
  let pid =
    Unix.create_process_env exe [| exe; "--first" |] env Unix.stdin
      Unix.stdout Unix.stderr
  in
  match Unix.waitpid [] pid with
  | _, WEXITED 0 -> ()
  | _ -> failwith "bench_gpt_oss: a first token's process failed"

(* [first_token_case name ~cold] times {!first_token} over a disk cache of
   kernels that one process filled, or under [cold] over an empty tolk cache
   and CUDA driver cache of each process's own, so that it compiles every
   kernel. *)
let first_token_case name ~cold =
  Thumper.bench_with_setup name
    ~setup:(fun () ->
      let dir = Filename.temp_dir "bench-gpt-oss" "" in
      let caches = ref 0 in
      let env () =
        let own =
          if not cold then dir
          else begin
            incr caches;
            let own = Filename.concat dir (string_of_int !caches) in
            Unix.mkdir own 0o700;
            own
          end
        in
        [
          "CACHEDB=" ^ Filename.concat own "cache.db";
          "CUDA_CACHE_PATH=" ^ Filename.concat own "cuda";
        ]
      in
      if not cold then first_token (env ());
      (dir, env))
    ~teardown:(fun (dir, _) ->
      ignore (Sys.command ("rm -rf " ^ Filename.quote dir)))
    (fun (_, env) -> first_token (env ()))

let opens () = Result.is_ok (first ())

(* Whether a GPU opens, asked of forked children: the workers forked to
   isolate the cases open it as they did. *)
let opens_in_child () =
  List.exists (fun name -> Result.is_ok (Rig_gpus.opens name)) names

let () =
  match Array.to_list Sys.argv with
  | [ _; "--first" ] -> load Gpt_oss.Unpacked (`Prefill short_prompt) ()
  | [ _; "--warm" ] ->
      if weights_cached () && opens () then
        List.iter
          (fun (_, pack, kind) ->
            load pack kind ();
            Gc.full_major ())
          steps
  | _ ->
      if not (weights_cached ()) then begin
        prerr_endline
          "gpt_oss: gpt-oss-20b's weights are not in raven's cache, nothing to \
           measure (README.md says how to fetch them)";
        exit 0
      end;
      if not (opens_in_child ()) then begin
        prerr_endline
          ("gpt_oss: no " ^ String.concat ", " names ^ ", nothing to measure");
        exit 0
      end;
      (* A trial's deadline covers its setup, which loads the 13.8 GB of
         weights onto the device. *)
      let config =
        if host then Thumper.Config.(default |> samples 3 |> warmup 0.)
        else Thumper.Config.(default |> warmup warmup_seconds)
      in
      let seconds = if host then 3600. else 120. in
      let code =
        Thumper.run (suite "gpt_oss")
          ~config:Thumper.Config.(config |> deadline seconds)
          ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
          (List.map step_case steps)
      in
      (* Each call is a process of its own, so the rows take the fewest
         samples. *)
      let first =
        Thumper.run (suite "gpt_oss_first")
          ~config:
            Thumper.Config.(
              default |> samples 3 |> warmup 0.
              |> deadline (if host then 3600. else 300.)
              |> metrics [ Thumper.Metric.wall_time ])
          ~budgets:[ Thumper.Budget.no_slower_than 0.1 ]
          [
            first_token_case
              (Printf.sprintf "first-token/%d" short_prompt)
              ~cold:false;
            first_token_case
              (Printf.sprintf "first-token-cold/%d" short_prompt)
              ~cold:true;
          ]
      in
      exit (max code first)
