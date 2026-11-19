(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* whisper/audio.py: HOP_LENGTH samples a frame, N_FRAMES frames a window; a
   timestamp step, 0.02 s, is two frames. *)
let hop = 160
let window = Audio.length / hop
let frames_per_timestamp = 2

(* Steps *)

type compile = {
  compile : 'a 'r 'd. ('a -> 'r, 'd) Nx.Ptree.fn -> ('a -> 'r) -> 'a -> 'r;
}

type ('b, 'd) model = ((float, 'b, 'd) Nx.t, 'd) Model.t

type ('b, 'd) steps = {
  tokenizer : Tokenizer.t;
  options : Decoding.options;
  encode : ('b, 'd) model -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t;
  languages : ('b, 'd) model -> (float, 'b, 'd) Nx.t -> 'd Nx.float32_t;
  decode :
    ('b, 'd) model ->
    'd Decoding.search ->
    'd Decoding.starts ->
    (float, 'b, 'd) Nx.t ->
    'd Decoding.outcome;
}

let eager = { compile = (fun _ f -> f) }

let steps ?(compile = eager) t o =
  ignore (Decoding.suppressed t o);
  let open Nx.Ptree in
  let params () = instantiate (module Model.Params) in
  {
    tokenizer = t;
    options = o;
    encode =
      compile.compile (params () @-> tensor @-> returns tensor) Model.encode;
    languages =
      compile.compile
        (params () @-> tensor @-> returns tensor)
        (fun m f -> Decoding.languages m t f);
    decode =
      compile.compile
        (params () @-> v Decoding.Search.walk @-> v Decoding.Starts.walk
       @-> tensor
        @-> returns (v Decoding.Outcome.walk))
        (fun m search starts f -> Decoding.decode m t o search starts f);
  }

(* Options *)

type rung =
  | Greedy
  | Beam of { size : int; patience : float }
  | Sample of { best_of : int; temperature : float }

type options = {
  language : string option;
  task : Tokenizer.task;
  ladder : rung list;
  compression_ratio_threshold : float option;
  logprob_threshold : float option;
  no_speech_threshold : float option;
  condition : bool;
  prompt : int array;
  carry_prompt : bool;
  clips : (int * int) list;
}

(* whisper/transcribe.py, transcribe's defaults: temperatures 0 to 1 by 0.2, and
   the three thresholds. *)
let temperatures = [ 0.2; 0.4; 0.6; 0.8; 1. ]
let compression_ratio_threshold = 2.4
let logprob_threshold = -1.
let no_speech_threshold = 0.6

(* A rung above this temperature empties the context (transcribe.py). *)
let reset_temperature = 0.5

let default =
  {
    language = None;
    task = Transcribe;
    ladder =
      Greedy
      :: List.map
           (fun temperature -> Sample { best_of = 1; temperature })
           temperatures;
    compression_ratio_threshold = Some compression_ratio_threshold;
    logprob_threshold = Some logprob_threshold;
    no_speech_threshold = Some no_speech_threshold;
    condition = true;
    prompt = [||];
    carry_prompt = false;
    clips = [ (0, max_int) ];
  }

let temperature = function
  | Greedy | Beam _ -> 0.
  | Sample { temperature; _ } -> temperature

(* A beam keeps [size * patience] candidates rounded half to even
   ({!Decoding.Beam}), at least one exactly when the product is above 0.5. *)
let check_rung = function
  | Greedy -> ()
  | Beam { size; patience } ->
      if size < 1 || not (float size *. patience > 0.5) then
        invalid_argf
          "Whisper.Transcript: a beam of %d at patience %g keeps no candidate"
          size patience
  | Sample { best_of; temperature } ->
      if best_of < 1 then
        invalid_argf "Whisper.Transcript: a sample of %d lanes" best_of;
      if not (temperature > 0. && Float.is_finite temperature) then
        invalid_argf "Whisper.Transcript: a temperature of %g" temperature

let check_options o =
  if o.ladder = [] then invalid_arg "Whisper.Transcript: no rung";
  List.iter check_rung o.ladder;
  let threshold what = function
    | Some x when Float.is_nan x ->
        invalid_argf "Whisper.Transcript: a %s threshold of NaN" what
    | _ -> ()
  in
  threshold "compression ratio" o.compression_ratio_threshold;
  threshold "log probability" o.logprob_threshold;
  threshold "no-speech" o.no_speech_threshold;
  ignore
    (List.fold_left
       (fun before (start, stop) ->
         if start < before || start >= stop then
           invalid_argf
             "Whisper.Transcript: clip (%d, %d) after a clip ending at %d" start
             stop before;
         stop)
       0 o.clips)

(* The language and the task a model of [dims] decodes: [None] for an
   English-only model, which transcribes English only. *)
let sot_language t (dims : Model.dims) o language =
  if Model.multilingual dims then (
    if not (Array.mem language (Tokenizer.languages t)) then
      invalid_argf "Whisper.Transcript: no language %S" language;
    Some (language, o.task))
  else begin
    if language <> "en" then
      invalid_argf "Whisper.Transcript: an English-only model takes no %S"
        language;
    if o.task <> Transcribe then
      invalid_arg "Whisper.Transcript: an English-only model does not translate";
    None
  end

(* Transcripts *)

type segment = { start : int; stop : int; tokens : int array; text : string }

type attempt = {
  rung : rung;
  result : Decoding.result;
  compression_ratio : float;
}

type window = {
  seek : int;
  size : int;
  choice : attempt;
  rejected : attempt list;
  silent : bool;
  segments : segment list;
}

type t = { language : string; text : string; windows : window list }
type request = { seek : int; size : int; start : Decoding.start; rung : int }

(* The fallback: whisper/transcribe.py, decode_with_fallback. *)
let fails o (a : attempt) =
  let r = a.result in
  let above th x = match th with Some th -> x > th | None -> false in
  let below th x = match th with Some th -> x < th | None -> false in
  let failed =
    above o.compression_ratio_threshold a.compression_ratio
    || below o.logprob_threshold r.avg_logprob
  in
  let silence =
    match (o.no_speech_threshold, o.logprob_threshold) with
    | Some ns, Some lp -> r.no_speech_prob > ns && r.avg_logprob < lp
    | _ -> false
  in
  failed && not silence

(* The skip: whisper/transcribe.py's no voice activity check. *)
let silent o (a : attempt) =
  let r = a.result in
  match o.no_speech_threshold with
  | None -> false
  | Some ns -> (
      r.no_speech_prob > ns
      &&
      match o.logprob_threshold with
      | Some lp -> not (r.avg_logprob > lp)
      | None -> true)

(* A segment of [toks] from [start] to [stop]. Its text is that of the tokens
   below the end of text; an instantaneous or blank segment keeps its frames
   alone (transcribe.py, new_segment). *)
let segment t start stop toks =
  let eot = Tokenizer.special t End_of_text in
  let text =
    Tokenizer.decode t
      (Array.of_list (List.filter (fun x -> x < eot) (Array.to_list toks)))
  in
  if start = stop || Space.strip text = "" then
    { start; stop; tokens = [||]; text = "" }
  else { start; stop; tokens = toks; text }

(* An attempt's tokens cut into segments, and the seek's advance:
   whisper/transcribe.py's segmentation. Without timestamps the window is one
   segment and the seek moves past it, whatever timestamps were sampled:
   upstream cuts at them, and a pair closing at [<|0.00|>] keeps its seek. *)
let segments t ~timestamps ~seek ~size tokens =
  let ts0 = Tokenizer.special t (Timestamp 0) in
  let n = Array.length tokens in
  let is_ts i = tokens.(i) >= ts0 in
  let frame i = seek + (frames_per_timestamp * (tokens.(i) - ts0)) in
  let segment = segment t in
  (* The positions of each timestamp that follows a timestamp. *)
  let pairs =
    List.filter
      (fun i -> is_ts (i - 1) && is_ts i)
      (List.init (max 0 (n - 1)) (fun i -> i + 1))
  in
  let single = n >= 2 && (not (is_ts (n - 2))) && is_ts (n - 1) in
  if not timestamps then ([ segment seek (seek + size) tokens ], size)
  else if pairs = [] then
    let rec last i =
      if i < 0 then None else if is_ts i then Some i else last (i - 1)
    in
    let stop =
      match last (n - 1) with
      | Some i when tokens.(i) <> ts0 -> frame i
      | _ -> seek + size
    in
    ([ segment seek stop tokens ], size)
  else
    let cuts = if single then pairs @ [ n ] else pairs in
    let rec cut from acc = function
      | [] -> (List.rev acc, from)
      | c :: rest ->
          let s =
            segment (frame from)
              (frame (c - 1))
              (Array.sub tokens from (c - from))
          in
          cut c (s :: acc) rest
    in
    let segments, last = cut 0 [] cuts in
    let advance =
      if single then size else frames_per_timestamp * (tokens.(last - 1) - ts0)
    in
    (segments, advance)

let last k a =
  let n = Array.length a in
  if n <= k then a else Array.sub a (n - k) k

(* The loop: whisper/transcribe.py, transcribe's seek loop over the clips. The
   context is kept to its last [n_text_ctx / 2 - 1] ids, all a prompt reads. *)
let run ?each t d (dims : Model.dims) o ~language ~frames decode =
  check_options o;
  let sot = sot_language t dims o language in
  let half = (dims.n_text_ctx / 2) - 1 in
  let initial = last half o.prompt in
  let context = ref (if o.carry_prompt then [||] else initial) in
  let prompt () =
    if not o.carry_prompt then !context
    else Array.append initial (last (half - Array.length initial) !context)
  in
  let windows = ref [] in
  let clip (start, stop) =
    let stop = min stop frames in
    let seek = ref start in
    while !seek < stop do
      let s = !seek in
      let size = min window (stop - s) in
      let st = Decoding.start ~prompt:(prompt ()) t dims d sot in
      let attempt i rung =
        let result = decode { seek = s; size; start = st; rung = i } in
        {
          rung;
          result;
          compression_ratio = Decoding.compression_ratio result.text;
        }
      in
      let rec climb i rejected = function
        | [] -> assert false
        | [ rung ] -> (attempt i rung, List.rev rejected)
        | rung :: rest ->
            let a = attempt i rung in
            if fails o a then climb (i + 1) (a :: rejected) rest
            else (a, List.rev rejected)
      in
      let choice, rejected = climb 0 [] o.ladder in
      let silent = silent o choice in
      let segments, advance =
        if silent then ([], size)
        else
          segments t ~timestamps:d.Decoding.timestamps ~seek:s ~size
            choice.result.tokens
      in
      if not silent then begin
        let tokens = Array.concat (List.map (fun x -> x.tokens) segments) in
        context := last half (Array.append !context tokens);
        if (not o.condition) || temperature choice.rung > reset_temperature then
          context := [||]
      end;
      seek := s + advance;
      let w = { seek = s; size; choice; rejected; silent; segments } in
      windows := w :: !windows;
      Option.iter (fun f -> f w) each
    done
  in
  List.iter clip o.clips;
  List.rev !windows

let loop t d dims o ~language ~frames decode =
  run t d dims o ~language ~frames decode

(* Transcription *)

let search at key (r : request) = function
  | Greedy -> Decoding.Greedy
  | Beam { size; patience } -> Decoding.Beam { size; patience }
  | Sample { best_of; temperature } ->
      let key = Nx.Rng.fold_in (Nx.Rng.fold_in key r.seek) r.rung in
      Decoding.Sample
        {
          best_of;
          temperature = Nx.place at (Nx.scalar Nx.float32 temperature);
          key = Nx.Rng.place at key;
        }

(* The least power of two at or above the longest start: [<|startofprev|>], a
   prompt of [n_text_ctx / 2 - 1] ids, a language's start sequence and
   [<|notimestamps|>]. *)
let width (dims : Model.dims) =
  let longest = 1 + ((dims.n_text_ctx / 2) - 1) + 3 + 1 in
  let rec up w = if w >= longest then w else up (2 * w) in
  up 1

let transcribe ?key ?each steps (m : _ model) o x =
  if Nx.ndim x <> 1 then
    invalid_argf "Whisper.Transcript.transcribe: samples of %d axes" (Nx.ndim x);
  check_options o;
  let key = match key with Some k -> k | None -> Nx.Rng.next_key () in
  let t = steps.tokenizer and d = steps.options and dims = m.dims in
  let at = Nx.beside m.encoder.positions in
  let dtype = Nx.dtype m.encoder.positions in
  let mel = Audio.log_mel ~padding:Audio.length ~n_mels:dims.n_mels x in
  let frames = Nx.dim 0 x / hop in
  (* A window: [size] frames from [seek], then zeros, at the model's dtype on
     its set. *)
  let window_at seek size =
    let w =
      Nx.pad
        [| (0, window - size); (0, 0) |]
        0.
        (Nx.slice [ R (seek, seek + size) ] mel)
    in
    Nx.place at (Nx.cast dtype (Nx.unsqueeze ~axes:[ 0 ] w))
  in
  let language =
    match o.language with
    | Some l -> l
    | None when not (Model.multilingual dims) -> "en"
    | None ->
        let f = steps.encode m (window_at 0 window) in
        let p = Nx.place Nx.Host.on (steps.languages m f) in
        let best = Nx.item [ 0 ] (Nx.cast Nx.int32 (Nx.argmax ~axis:1 p)) in
        (Tokenizer.languages t).(Int32.to_int best)
  in
  ignore (sot_language t dims o language);
  let width = width dims in
  (* The features of the last window encoded, each window's attempts reading
     them. *)
  let features = ref None in
  let decode (r : request) =
    let f =
      match !features with
      | Some (s, f) when s = r.seek -> f
      | _ ->
          let f = steps.encode m (window_at r.seek r.size) in
          features := Some (r.seek, f);
          f
    in
    let search = search at key r (List.nth o.ladder r.rung) in
    let starts =
      Nx.Ptree.place
        (module Decoding.Starts)
        at
        (Decoding.starts ~width [| r.start |])
    in
    let out = steps.decode m search starts f in
    (Decoding.results t d
       (Nx.Ptree.place (module Decoding.Outcome) Nx.Host.on out)).(0)
  in
  let windows = run ?each t d dims o ~language ~frames decode in
  let tokens =
    List.concat_map (fun w -> List.map (fun s -> s.tokens) w.segments) windows
  in
  { language; text = Tokenizer.decode t (Array.concat tokens); windows }
