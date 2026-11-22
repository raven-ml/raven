(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Transcription against upstream's transcribe loop.

   upstream/transcript.json holds cases of upstream's transcribe over a model
   whose decode returns scripted results in call order, over the trimmed gpt2
   tokenizer: each case's options, its content's frames, each call's seek, size,
   temperature, initial tokens and scripted result, and upstream's segments, in
   frames, and text. upstream/transcript.pt is a small randomly initialised
   Whisper of 1500 audio positions over the same tokenizer, which the laws
   transcribe. *)

open Windtrap
open Whisper_support
module T = Whisper.Tokenizer
module D = Whisper.Decoding
module Tr = Whisper.Transcript

let gpt2 =
  let ranks =
    In_channel.with_open_bin "upstream/gpt2-1000.tiktoken" In_channel.input_all
  in
  match T.of_ranks ~languages:99 ranks with Ok t -> t | Error e -> failwith e

(* Loop *)

let golden = Json.read "upstream/transcript.json"

(* An English-only model's dimensions: the loop reads [n_text_ctx] and whether
   the model is multilingual. *)
let dims =
  {
    Whisper.Model.n_mels = 80;
    n_audio_ctx = 1500;
    n_audio_state = 384;
    n_audio_head = 6;
    n_audio_layer = 4;
    n_vocab = T.vocab_size gpt2;
    n_text_ctx = Json.int (Json.mem "n_text_ctx" golden);
    n_text_state = 384;
    n_text_head = 6;
    n_text_layer = 4;
  }

let opt f j = if Json.is_null j then None else Some (f j)

let rung j : Tr.rung =
  match Json.string (Json.mem "kind" j) with
  | "greedy" -> Greedy
  | "beam" ->
      Beam
        {
          size = Json.int (Json.mem "size" j);
          patience = Json.float (Json.mem "patience" j);
        }
  | "sample" ->
      Sample
        {
          best_of = Json.int (Json.mem "best_of" j);
          temperature = Json.float (Json.mem "temperature" j);
        }
  | k -> failwith k

let clip j =
  match Json.list j with
  | [ a; b ] -> (Json.int a, Option.value ~default:max_int (opt Json.int b))
  | _ -> failwith "clip"

let options c : Tr.options =
  {
    language = None;
    task = Transcribe;
    ladder = List.map rung (Json.list (Json.mem "ladder" c));
    compression_ratio_threshold =
      opt Json.float (Json.mem "compression_ratio_threshold" c);
    logprob_threshold = opt Json.float (Json.mem "logprob_threshold" c);
    no_speech_threshold = opt Json.float (Json.mem "no_speech_threshold" c);
    condition = Json.bool (Json.mem "condition" c);
    prompt = Json.ints (Json.mem "prompt" c);
    carry_prompt = Json.bool (Json.mem "carry_prompt" c);
    clips = List.map clip (Json.list (Json.mem "clips" c));
  }

let decoding c =
  { (D.options gpt2) with timestamps = Json.bool (Json.mem "timestamps" c) }

let temperature : Tr.rung -> float = function
  | Greedy | Beam _ -> 0.
  | Sample { temperature; _ } -> temperature

(* The loop with each request answered by upstream's call at its place, which
   must be the same request. *)
let replay c =
  let o = options c and d = decoding c in
  let calls = Array.of_list (Json.list (Json.mem "calls" c)) in
  let departure = not (Json.is_null (Json.mem "departure" c)) in
  let k = ref 0 in
  let decode (r : Tr.request) =
    if !k >= Array.length calls then
      failf "request %d at seek %d past upstream's %d calls" !k r.seek
        (Array.length calls);
    let call = calls.(!k) in
    let msg = Printf.sprintf "call %d" !k in
    incr k;
    equal ~msg:(msg ^ " seek") int (Json.int (Json.mem "seek" call)) r.seek;
    equal ~msg:(msg ^ " size") int (Json.int (Json.mem "size" call)) r.size;
    equal ~msg:(msg ^ " temperature") float_exact
      (Json.float (Json.mem "temperature" call))
      (temperature (List.nth o.ladder r.rung));
    (* A carried prompt of 223 ids or more: the port keeps its last 223 ids
       whole, where upstream lets the transcript push them out. *)
    let want =
      if departure then (D.start ~prompt:o.prompt gpt2 dims d None).tokens
      else Json.ints (Json.mem "initial_tokens" call)
    in
    equal ~msg:(msg ^ " initial tokens") (array int) want r.start.tokens;
    {
      D.tokens = Json.ints (Json.mem "tokens" call);
      text = Json.string (Json.mem "text" call);
      avg_logprob = Json.float (Json.mem "avg_logprob" call);
      no_speech_prob = Json.float (Json.mem "no_speech_prob" call);
    }
  in
  let frames = Json.int (Json.mem "frames" c) in
  let windows = Tr.loop gpt2 d dims o ~language:"en" ~frames decode in
  equal ~msg:"requests" int (Array.length calls) !k;
  (calls, windows)

let loop_upstream =
  cases "the loop makes upstream's calls, segments and text"
    ~id:(fun c -> Json.string (Json.mem "name" c))
    (Json.list (Json.mem "cases" golden))
    (fun c ->
      let calls, windows = replay c in
      let segments = Json.list (Json.mem "segments" c) in
      let got =
        List.concat_map
          (fun (w : Tr.window) ->
            List.map (fun (s : Tr.segment) -> (w.seek, s)) w.segments)
          windows
      in
      equal ~msg:"segments" int (List.length segments) (List.length got);
      List.iteri
        (fun i (j, (seek, (s : Tr.segment))) ->
          let msg what = Printf.sprintf "segment %d %s" i what in
          equal ~msg:(msg "seek") int (Json.int (Json.mem "seek" j)) seek;
          equal ~msg:(msg "start") int (Json.int (Json.mem "start" j)) s.start;
          equal ~msg:(msg "stop") int (Json.int (Json.mem "end" j)) s.stop;
          equal ~msg:(msg "tokens") (array int)
            (Json.ints (Json.mem "tokens" j))
            s.tokens;
          equal ~msg:(msg "text") string
            (Json.string (Json.mem "text" j))
            s.text)
        (List.combine segments got);
      (* Each window: its seek, its attempts and whether it was silent, a window
         with no segment upstream. *)
      let seeks =
        Array.fold_left
          (fun acc call ->
            let s = Json.int (Json.mem "seek" call) in
            match acc with
            | (s', n) :: rest when s = s' -> (s, n + 1) :: rest
            | _ -> (s, 1) :: acc)
          [] calls
        |> List.rev
      in
      equal ~msg:"windows"
        (list (pair int int))
        seeks
        (List.map
           (fun (w : Tr.window) -> (w.seek, List.length w.rejected + 1))
           windows);
      List.iter
        (fun (w : Tr.window) ->
          let silent =
            not
              (List.exists
                 (fun j -> Json.int (Json.mem "seek" j) = w.seek)
                 segments)
          in
          equal
            ~msg:(Printf.sprintf "window %d silent" w.seek)
            bool silent w.silent)
        windows;
      let tokens =
        Array.concat
          (List.concat_map
             (fun (w : Tr.window) ->
               List.map (fun (s : Tr.segment) -> s.tokens) w.segments)
             windows)
      in
      equal ~msg:"text" string
        (Json.string (Json.mem "text" c))
        (T.decode gpt2 tokens))

(* Without timestamps the window is one segment, whatever timestamps the decode
   sampled, and the seek advances by its content: [<|0.00|>] twice, which
   upstream would cut as a pair closing at 0, leaves no window repeated. *)
let untimed =
  test "without timestamps a window is one segment, the seek past it" (fun () ->
      let d = { (D.options gpt2) with timestamps = false } in
      let ts0 = T.special gpt2 (Timestamp 0) in
      let tokens = [| 262; ts0; ts0 |] in
      let requests = ref 0 in
      let decode (r : Tr.request) =
        incr requests;
        if !requests > 10 then failf "request %d at seek %d" !requests r.seek;
        {
          D.tokens;
          text = String.trim (T.decode gpt2 tokens);
          avg_logprob = -0.1;
          no_speech_prob = 0.01;
        }
      in
      let o = { Tr.default with ladder = [ Greedy ] } in
      let windows = Tr.loop gpt2 d dims o ~language:"en" ~frames:4000 decode in
      equal
        (list (pair int int))
        [ (0, 3000); (3000, 1000) ]
        (List.map (fun (w : Tr.window) -> (w.seek, w.size)) windows);
      List.iter
        (fun (w : Tr.window) ->
          equal ~msg:"segments"
            (list (pair int int))
            [ (w.seek, w.seek + w.size) ]
            (List.map (fun (s : Tr.segment) -> (s.start, s.stop)) w.segments))
        windows)

(* Transcription *)

let model = Whisper.Model.load Nx.float32 "upstream/transcript.pt"
let eager = Tr.steps gpt2 (D.options gpt2)

(* Noise of [n] samples. *)
let noise n =
  Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key n) Nx.float32 [| n |]) 0.1

(* Recordings of no frame, of part of a window, of a window and of a little more
   than two. *)
let length_gen =
  Gen.frequency
    [
      (2, Gen.int_range 0 159);
      (2, Gen.int_range 160 (Whisper.Audio.length - 1));
      (1, Gen.constant Whisper.Audio.length);
      ( 2,
        Gen.int_range (Whisper.Audio.length + 1)
          ((2 * Whisper.Audio.length) + 9600) );
    ]

let options_gen =
  let open Gen in
  let* seed = int_range 0 1_000_000 in
  let+ ladder =
    of_list
      [
        Tr.default.ladder;
        [ Tr.Greedy ];
        [ Tr.Sample { best_of = 2; temperature = 0.7 }; Tr.Greedy ];
      ]
  in
  (seed, { Tr.default with ladder })

let transcribe_law =
  prop ~count:40 "a transcription's windows, segments, text and draws"
    Gen.(pair length_gen options_gen)
    (fun (n, (seed, o)) ->
      let x = noise n in
      let key = Nx.Rng.key seed in
      let seen = ref [] in
      let each w = seen := w :: !seen in
      let r = Tr.transcribe ~key ~each eager model o x in
      cover "no window" (r.windows = []);
      cover "two windows or more" (List.length r.windows >= 2);
      cover "a fallback"
        (List.exists (fun (w : Tr.window) -> w.rejected <> []) r.windows);
      let seek (w : Tr.window) = w.seek in
      equal ~msg:"each, in order" (list int) (List.map seek r.windows)
        (List.rev_map seek !seen);
      let frames = n / 160 in
      ignore
        (List.fold_left
           (fun prev (w : Tr.window) ->
             greater ~msg:"seeks increase" int ~than:prev w.seek;
             equal ~msg:"size" int (min 3000 (frames - w.seek)) w.size;
             w.seek)
           (-1) r.windows);
      let tokens =
        List.concat_map
          (fun (w : Tr.window) ->
            List.map (fun (s : Tr.segment) -> s.tokens) w.segments)
          r.windows
      in
      equal ~msg:"text" string (T.decode gpt2 (Array.concat tokens)) r.text;
      equal ~msg:"English-only" string "en" r.language;
      let again = Tr.transcribe ~key eager model o x in
      equal ~msg:"same key, same windows" int (List.length r.windows)
        (List.length again.windows);
      List.iter2
        (fun (w : Tr.window) (v : Tr.window) ->
          equal ~msg:"same key, same tokens" (array int) w.choice.result.tokens
            v.choice.result.tokens)
        r.windows again.windows)

(* With [condition = false] every window's prompt is empty, so each attempt
   decodes alone from its window, its rung and its key. *)
let draws_law =
  test "each attempt draws from the key folded with its seek and rung"
    (fun () ->
      let o =
        {
          Tr.default with
          condition = false;
          ladder =
            [
              Tr.Sample { best_of = 2; temperature = 0.5 };
              Tr.Sample { best_of = 1; temperature = 1. };
            ];
          compression_ratio_threshold = Some 0.;
        }
      in
      let x = noise (Whisper.Audio.length + 48_000) in
      let key = Nx.Rng.key 7 in
      let r = Tr.transcribe ~key eager model o x in
      let mel =
        Whisper.Audio.log_mel ~padding:Whisper.Audio.length ~n_mels:80 x
      in
      let d = D.options gpt2 in
      let start = D.start gpt2 model.dims d None in
      let width = Array.length start.tokens in
      let attempts = ref 0 in
      List.iter
        (fun (w : Tr.window) ->
          let window =
            Nx.pad
              [| (0, 3000 - w.size); (0, 0) |]
              0.
              (Nx.slice [ R (w.seek, w.seek + w.size) ] mel)
          in
          let f =
            Whisper.Model.encode model (Nx.unsqueeze ~axes:[ 0 ] window)
          in
          List.iteri
            (fun i (a : Tr.attempt) ->
              incr attempts;
              let best_of, t =
                match a.rung with
                | Sample { best_of; temperature } -> (best_of, temperature)
                | _ -> assert false
              in
              let search =
                D.Sample
                  {
                    best_of;
                    temperature = Nx.scalar Nx.float32 t;
                    key = Nx.Rng.fold_in (Nx.Rng.fold_in key w.seek) i;
                  }
              in
              (* Eager steps place every start in rows of its own. *)
              let out =
                D.decode model gpt2 d search (D.starts ~width [| start |]) f
              in
              equal ~msg:"tokens" (array int) (D.results gpt2 d out).(0).tokens
                a.result.tokens)
            (w.rejected @ [ w.choice ]))
        r.windows;
      greater ~msg:"attempts" int ~than:2 !attempts)

(* Compiled steps: a program for the features and one for decoding per kind of
   rung, Greedy then Sample, whatever the temperature; later recordings compile
   nothing. With one key a compiled transcription draws the same tokens
   again. *)
let compiled =
  test "compiled steps compile once per step and kind of rung" (fun () ->
      let traces = ref 0 in
      let compile =
        {
          Tr.compile =
            (fun s f ->
              Rune.jit s (fun x ->
                  incr traces;
                  f x));
        }
      in
      let jitted = Tr.steps ~compile gpt2 (D.options gpt2) in
      let tokens r =
        List.map (fun (w : Tr.window) -> w.choice.result.tokens) r.Tr.windows
      in
      let run o n =
        Tr.transcribe ~key:(Nx.Rng.key n) jitted model o (noise n)
      in
      let greedy = { Tr.default with ladder = [ Tr.Greedy ] } in
      ignore (run greedy (Whisper.Audio.length + 16_000));
      equal ~msg:"a greedy recording" int 2 !traces;
      ignore (run greedy 200_000);
      equal ~msg:"another greedy recording" int 2 !traces;
      let sampled =
        {
          Tr.default with
          ladder =
            [
              Tr.Sample { best_of = 1; temperature = 0.5 };
              Tr.Sample { best_of = 1; temperature = 1. };
            ];
          compression_ratio_threshold = Some 0.;
        }
      in
      let first = run sampled (Whisper.Audio.length + 48_000) in
      equal ~msg:"a sampled recording" int 3 !traces;
      let again = run sampled (Whisper.Audio.length + 48_000) in
      equal ~msg:"the same key" (list (array int)) (tokens first) (tokens again);
      ignore (run sampled 300_000);
      equal ~msg:"another sampled recording" int 3 !traces)

(* Refusals *)

let refusals =
  let x = noise 16_000 in
  let o = Tr.default in
  let rung r = { o with ladder = [ r ] } in
  cases "transcribe refuses what it cannot do" ~id:fst
    [
      ("samples of two axes", (o, Nx.reshape [| 2; 8000 |] x));
      ("no rung", ({ o with ladder = [] }, x));
      ("a beam of no lane", (rung (Beam { size = 0; patience = 1. }), x));
      ("a beam of no candidate", (rung (Beam { size = 1; patience = 0.5 }), x));
      ( "a sample of no lane",
        (rung (Sample { best_of = 0; temperature = 1. }), x) );
      ( "a temperature of 0",
        (rung (Sample { best_of = 1; temperature = 0. }), x) );
      ( "a negative temperature",
        (rung (Sample { best_of = 1; temperature = -1. }), x) );
      ( "an infinite temperature",
        (rung (Sample { best_of = 1; temperature = Float.infinity }), x) );
      ( "a NaN temperature",
        (rung (Sample { best_of = 1; temperature = Float.nan }), x) );
      ( "a NaN threshold",
        ({ o with compression_ratio_threshold = Some Float.nan }, x) );
      ( "a NaN log probability",
        ({ o with logprob_threshold = Some Float.nan }, x) );
      ("a NaN no-speech", ({ o with no_speech_threshold = Some Float.nan }, x));
      ("a clip below 0", ({ o with clips = [ (-1, 50) ] }, x));
      ("an empty clip", ({ o with clips = [ (50, 50) ] }, x));
      ("a reversed clip", ({ o with clips = [ (60, 50) ] }, x));
      ("overlapping clips", ({ o with clips = [ (0, 60); (50, 80) ] }, x));
      ("clips out of order", ({ o with clips = [ (60, 80); (0, 50) ] }, x));
      ("a language other than English", ({ o with language = Some "fr" }, x));
      ("a translation", ({ o with task = Translate }, x));
    ]
    (fun (_, (o, x)) ->
      raises_match Exn.invalid_arg (fun () -> Tr.transcribe eager model o x))

let english =
  test "an English-only model takes English" (fun () ->
      let r =
        Tr.transcribe ~key:(Nx.Rng.key 0) eager model
          { Tr.default with language = Some "en"; ladder = [ Greedy ] }
          (noise 16_000)
      in
      equal string "en" r.language)

let () =
  exit
    (run "whisper.transcript"
       [
         group "loop" [ loop_upstream; untimed ];
         group "transcribe" [ transcribe_law; draws_law; english ];
         group "compiled" [ compiled ];
         group "refusals" [ refusals ];
       ])
