(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* main.exe [OPTION...] AUDIO...: upstream's whisper command (whisper/
   transcribe.py, cli) on raven. Each WAV or FLAC file, read as Whisper hears it
   (Audio.load), is transcribed by a checkpoint over its rank file and written
   as text, subtitles (VTT, SRT), tab-separated values or JSON (Writer), named
   after the file. Options keep upstream's names and defaults; [main.exe --help]
   lists them. *)

open Whisper

let fail fmt = Printf.ksprintf failwith fmt

(* Options *)

let usage = "main.exe [OPTION...] AUDIO...\nOptions:"

(* "None" or a value, as upstream's optional_int and optional_float. *)
let optional of_string = function "None" -> None | s -> Some (of_string s)

let int_of s =
  match int_of_string_opt s with
  | Some i -> i
  | None -> raise (Arg.Bad ("not an integer: " ^ s))

let float_of s =
  match float_of_string_opt s with
  | Some x -> x
  | None -> raise (Arg.Bad ("not a number: " ^ s))

(* An English-only model's languages (Tokenizer). *)
let english_languages = 99
let formats = [ "txt"; "vtt"; "srt"; "tsv"; "json" ]
let model = ref ""
let ranks = ref ""
let output_dir = ref "."
let output_format = ref "all"
let verbose = ref true
let device = ref "cpu"
let task = ref Tokenizer.Transcribe
let language = ref None
let temperature = ref 0.
let increment = ref (Some 0.2)
let best_of = ref (Some 5)
let beam_size = ref (Some 5)
let patience = ref None
let length_penalty = ref None
let suppress = ref "-1"
let initial_prompt = ref None
let carry = ref false
let condition = ref true
let ratio_threshold = ref (Some 2.4)
let logprob_threshold = ref (Some (-1.))
let no_speech_threshold = ref (Some 0.6)
let clip_timestamps = ref "0"
let seed = ref None
let audio = ref []
let set_optional r of_string = Arg.String (fun s -> r := optional of_string s)

let specs =
  [
    ("--model", Arg.Set_string model, "FILE a checkpoint, as tiny.en.pt");
    ( "--tokenizer",
      Arg.Set_string ranks,
      "DIR the directory of gpt2.tiktoken, an English-only model's rank file, \
       and multilingual.tiktoken, the others'" );
    ("--output_dir", Arg.Set_string output_dir, "DIR where to write (.)");
    ("-o", Arg.Set_string output_dir, "DIR --output_dir");
    ( "--output_format",
      Arg.Symbol ("all" :: formats, fun s -> output_format := s),
      " the format, every one by default" );
    ( "-f",
      Arg.Symbol ("all" :: formats, fun s -> output_format := s),
      " --output_format" );
    ( "--verbose",
      Arg.Bool (fun b -> verbose := b),
      "BOOL print each segment as it is decided (true)" );
    ( "--device",
      Arg.Set_string device,
      "D cpu (eager, the default), cpu-jit (compiled on the host) or a GPU by \
       name, compiled: cuda, nv, amd or metal" );
    ( "--task",
      Arg.Symbol
        ( [ "transcribe"; "translate" ],
          fun s ->
            task := if s = "translate" then Tokenizer.Translate else Transcribe
        ),
      " transcribe, or translate into English" );
    ( "--language",
      Arg.String (fun s -> language := optional Fun.id s),
      "CODE the spoken language, detected by default" );
    ( "--temperature",
      Arg.String (fun s -> temperature := float_of s),
      "T the first temperature (0)" );
    ( "--temperature_increment_on_fallback",
      set_optional increment float_of,
      "T the temperatures' step to 1, or None (0.2)" );
    ( "--best_of",
      set_optional best_of int_of,
      "N draws at a positive temperature (5)" );
    ( "--beam_size",
      set_optional beam_size int_of,
      "N beams at temperature 0, or None for greedy (5)" );
    ( "--patience",
      set_optional patience float_of,
      "P a beam's patience (None: 1)" );
    ( "--length_penalty",
      set_optional length_penalty float_of,
      "A candidates over ((5 + length) / 6) ** A (None: over their length)" );
    ( "--suppress_tokens",
      Arg.Set_string suppress,
      "IDS ids to suppress, comma-separated; -1 for the non-speech ones (-1)" );
    ( "--initial_prompt",
      Arg.String (fun s -> initial_prompt := Some s),
      "TEXT text before the recording" );
    ( "--carry_initial_prompt",
      Arg.Bool (fun b -> carry := b),
      "BOOL begin every window's prompt with it (false)" );
    ( "--condition_on_previous_text",
      Arg.Bool (fun b -> condition := b),
      "BOOL prompt each window with the transcript before it (true)" );
    ( "--compression_ratio_threshold",
      set_optional ratio_threshold float_of,
      "X fall back above this compression ratio, or None (2.4)" );
    ( "--logprob_threshold",
      set_optional logprob_threshold float_of,
      "X fall back below this average log probability, or None (-1)" );
    ( "--no_speech_threshold",
      set_optional no_speech_threshold float_of,
      "X silence above this no-speech probability, or None (0.6)" );
    ( "--clip_timestamps",
      Arg.Set_string clip_timestamps,
      "S start,end,start,end,... in seconds, the last end the file's (0)" );
    ( "--seed",
      Arg.String (fun s -> seed := Some (int_of s)),
      "N the draws' key, from the system's entropy by default" );
  ]

let parse () =
  let anon a = audio := a :: !audio in
  (try Arg.parse_argv Sys.argv (Arg.align specs) anon usage with
  | Arg.Help m ->
      print_string m;
      exit 0
  | Arg.Bad m ->
      prerr_string m;
      exit 2);
  let missing what =
    prerr_string ("main.exe: " ^ what ^ " is required\n");
    exit 2
  in
  if !model = "" then missing "--model";
  if !ranks = "" then missing "--tokenizer";
  if !audio = [] then missing "an audio file";
  audio := List.rev !audio

(* Python's round, half to even. *)
let round_even x =
  let r = Float.round x in
  if Float.abs (x -. Float.trunc x) = 0.5 && Float.rem r 2. <> 0. then
    r -. Float.copy_sign 1. x
  else r

(* The clips of upstream's --clip_timestamps: seconds to frames as [round (ts *
   100)], paired, an odd last start running to the end. *)
let clips s =
  let frames =
    if s = "" then [ 0 ]
    else
      List.map
        (fun ts -> int_of_float (round_even (float_of ts *. 100.)))
        (String.split_on_char ',' s)
  in
  let rec pair = function
    | [] -> []
    | [ start ] -> [ (start, max_int) ]
    | start :: stop :: rest -> (start, stop) :: pair rest
  in
  pair frames

(* Upstream's temperatures, np.arange (t, 1 + 1e-6, increment): [t + i * inc]
   for each [i] below [ceil ((1 + 1e-6 - t) / inc)]. *)
let temperatures () =
  match !increment with
  | None -> [ !temperature ]
  | Some inc ->
      let n = int_of_float (Float.ceil ((1. +. 1e-6 -. !temperature) /. inc)) in
      List.init (max 0 n) (fun i -> !temperature +. (float i *. inc))

(* A temperature of 0 searches by beam, or takes the likeliest token; above it,
   draws. *)
let ladder () =
  List.map
    (fun t : Transcript.rung ->
      if t > 0. then
        Sample { best_of = Option.value ~default:1 !best_of; temperature = t }
      else
        match !beam_size with
        | Some size ->
            Beam { size; patience = Option.value ~default:1. !patience }
        | None -> Greedy)
    (temperatures ())

let suppressed t =
  let ids =
    if !suppress = "" then []
    else List.map int_of (String.split_on_char ',' !suppress)
  in
  let given = Array.of_list (List.filter (fun i -> i >= 0) ids) in
  if List.mem (-1) ids then Array.append (Tokenizer.non_speech t) given
  else given

(* Output *)

(* utils.py, format_timestamp, from a frame of 10 ms, as the verbose lines write
   it. *)
let timestamp f =
  let ms = 10 * f in
  let h = ms / 3_600_000 and ms = ms mod 3_600_000 in
  let m = ms / 60_000 and ms = ms mod 60_000 in
  let s = ms / 1000 and ms = ms mod 1000 in
  let h = if h > 0 then Printf.sprintf "%02d:" h else "" in
  Printf.sprintf "%s%02d:%02d.%03d" h m s ms

let writers =
  Writer.
    [ ("txt", txt); ("vtt", vtt); ("srt", srt); ("tsv", tsv); ("json", json) ]

let write path r =
  let base = Filename.remove_extension (Filename.basename path) in
  List.iter
    (fun (ext, w) ->
      if !output_format = "all" || !output_format = ext then
        Out_channel.with_open_bin
          (Filename.concat !output_dir (base ^ "." ^ ext))
          (fun oc -> output_string oc (w r)))
    writers

(* Devices *)

type 'd target = { at : 'd Nx.Placement.t; jit : bool }
type run = { run : 'd. 'd target -> unit }

let on device k =
  match device with
  | "cpu" -> k.run { at = Nx.Host.on; jit = false }
  | "cpu-jit" -> k.run { at = Nx.Host.on; jit = true }
  | name ->
      let gpu = Result.error_to_failure (Rig_gpus.open_ name) in
      let module D = (val Nx.devices [ gpu ]) in
      k.run { at = D.on; jit = true }

let rec mkdir_p d =
  if not (Sys.file_exists d) then begin
    mkdir_p (Filename.dirname d);
    Sys.mkdir d 0o755
  end

let print (w : Transcript.window) =
  List.iter
    (fun (s : Transcript.segment) ->
      Printf.printf "[%s --> %s] %s\n%!" (timestamp s.start) (timestamp s.stop)
        s.text)
    w.segments

let () =
  parse ();
  let m = Model.load Nx.float32 !model in
  (* The rank file of the model's kind, as upstream's get_tokenizer picks it.
     The model's vocabulary is the ranks, then the specials, one per language:
     99 for an English-only model ({!Tokenizer}), the release's for a
     multilingual one. *)
  let t =
    let multilingual = Model.multilingual m.dims in
    let file =
      Filename.concat !ranks
        (if multilingual then "multilingual.tiktoken" else "gpt2.tiktoken")
    in
    let text = In_channel.with_open_bin file In_channel.input_all in
    let of_ranks languages =
      match Tokenizer.of_ranks ~languages text with
      | Ok t -> t
      | Error e -> fail "%s: %s" file e
    in
    let languages = m.dims.n_vocab - Tokenizer.vocab_size (of_ranks 0) in
    let want =
      if multilingual then Model.languages m.dims else english_languages
    in
    if languages <> want then
      fail "%s: its ranks leave %d languages in the model's %d tokens, not %d"
        file languages m.dims.n_vocab want;
    of_ranks languages
  in
  let d =
    {
      (Decoding.options t) with
      suppress = suppressed t;
      length_penalty = !length_penalty;
    }
  in
  let prompt =
    match !initial_prompt with None -> [||] | Some s -> Decoding.prompt t s
  in
  let o =
    {
      Transcript.language = !language;
      task = !task;
      ladder = ladder ();
      compression_ratio_threshold = !ratio_threshold;
      logprob_threshold = !logprob_threshold;
      no_speech_threshold = !no_speech_threshold;
      condition = !condition;
      prompt;
      carry_prompt = !carry;
      clips = clips !clip_timestamps;
    }
  in
  mkdir_p !output_dir;
  let failed = ref false in
  on !device
    {
      run =
        (fun target ->
          let m = Nx.Ptree.place (module Model.Params) target.at m in
          let compile =
            if target.jit then
              Some { Transcript.compile = (fun s f -> Rune.jit s f) }
            else None
          in
          let steps = Transcript.steps ?compile t d in
          List.iter
            (fun path ->
              try
                let x = Audio.load path in
                let key = Option.map Nx.Rng.key !seed in
                let each = if !verbose then print else ignore in
                write path (Transcript.transcribe ?key ~each steps m o x)
              with Failure e | Invalid_argument e | Sys_error e ->
                failed := true;
                Printf.printf "Skipping %s due to %s\n%!" path e)
            !audio);
    };
  if !failed then exit 1
