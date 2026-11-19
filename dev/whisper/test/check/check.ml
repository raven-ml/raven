(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The port against the oracle, on the full rank files and the released
   checkpoints: each command prints one line per difference and a count, and
   exits 1 if there is one. *)

open Whisper_support
module T = Whisper.Tokenizer

let diffs = ref 0

let diff fmt =
  incr diffs;
  Printf.printf ("DIFF " ^^ fmt ^^ "\n%!")

let ids a = String.concat " " (Array.to_list (Array.map string_of_int a))

let check_ids what want got =
  if want <> got then diff "%s: want [%s] got [%s]" what (ids want) (ids got)

let check_int what want got =
  if want <> got then diff "%s: want %d got %d" what want got

let check_string what want got =
  if want <> got then diff "%s: want %S got %S" what want got

(* Tokenizer *)

let configs =
  [
    ("gpt2", "gpt2.tiktoken", 99);
    ("multilingual99", "multilingual.tiktoken", 99);
    ("multilingual100", "multilingual.tiktoken", 100);
  ]

let read path = In_channel.with_open_bin path In_channel.input_all

let tokenizer assets (_, file, languages) =
  match T.of_ranks ~languages (read (Filename.concat assets file)) with
  | Ok t -> t
  | Error e -> failwith e

let tables name t oracle =
  let o = Json.read (Filename.concat oracle (name ^ ".json")) in
  let field k = Json.int (Json.mem k o) in
  let at what = name ^ " " ^ what in
  check_int (at "n_vocab") (field "n_vocab") (T.vocab_size t);
  List.iter
    (fun (k, s) -> check_int (at k) (field k) (T.special t s))
    [
      ("eot", T.End_of_text);
      ("sot", Start_of_transcript);
      ("sot_prev", Start_of_prev);
      ("sot_lm", Start_of_lm);
      ("no_speech", No_speech);
      ("no_timestamps", No_timestamps);
      ("timestamp_begin", Timestamp 0);
      ("timestamp_last", Timestamp 1500);
      ("transcribe", Task Transcribe);
      ("translate", Task Translate);
    ];
  (* Every special: its id decodes to its text. *)
  let specials = Json.list (Json.mem "special_tokens" o) in
  check_int (at "special count") (List.length specials)
    (T.vocab_size t - T.special t End_of_text);
  List.iter
    (fun p ->
      match Json.list p with
      | [ s; id ] ->
          let s = Json.string s and id = Json.int id in
          check_string
            (at "special " ^ s)
            s
            (T.decode_with_timestamps t [| id |])
      | _ -> failwith "special_tokens")
    specials;
  check_ids (at "language tokens")
    (Json.ints (Json.mem "all_language_tokens_sorted" o))
    (Array.map (fun c -> T.special t (Language c)) (T.languages t));
  List.iter
    (fun (key, v) ->
      let prompt =
        match String.split_on_char '/' key with
        | [ "default" ] -> None
        | [ code; "transcribe" ] -> Some (code, T.Transcribe)
        | [ code; "translate" ] -> Some (code, T.Translate)
        | _ -> failwith key
      in
      let seq = T.sot_sequence t prompt in
      check_ids (at "sot " ^ key) (Json.ints (Json.mem "sot_sequence" v)) seq;
      check_ids
        (at "sot+notimestamps " ^ key)
        (Json.ints (Json.mem "including_notimestamps" v))
        (Array.append seq [| T.special t No_timestamps |]))
    (Json.mems (Json.mem "sot_sequences" o));
  check_ids (at "non_speech")
    (Json.ints (Json.mem "non_speech_tokens" o))
    (T.non_speech t);
  List.iter
    (fun (s, v) -> check_ids (at "encode " ^ s) (Json.ints v) (T.encode t s))
    (Json.mems (Json.mem "probe_encodings" o))

(* Where upstream's encode refuses special-token text, ours encodes it as text:
   tiktoken's encode_ordinary, from ordinary.json. *)
let corpus tokenizers oracle ordinary =
  let n = ref 0 in
  In_channel.with_open_bin (Filename.concat oracle "corpus.jsonl") (fun ic ->
      In_channel.input_lines ic
      |> List.iter (fun line ->
          let r = Json.of_string line in
          let text = Json.string (Json.mem "text" r) in
          List.iter
            (fun (name, t) ->
              incr n;
              let u = Json.mem name r in
              let e = Json.mem "encode" u in
              let want, decoded =
                match Json.mems e with
                | ("ids", ids) :: _ ->
                    (Json.ints ids, Json.string (Json.mem "decode" u))
                | _ ->
                    let o = Json.list (Json.mem name ordinary) in
                    let row =
                      List.find
                        (fun p -> Json.string (List.hd (Json.list p)) = text)
                        o
                    in
                    (Json.ints (List.nth (Json.list row) 1), text)
              in
              let got = T.encode t text in
              check_ids (Printf.sprintf "%s encode %S" name text) want got;
              check_string
                (Printf.sprintf "%s decode %S" name text)
                decoded (T.decode t got))
            tokenizers));
  !n

let decodes tokenizers oracle =
  let n = ref 0 in
  In_channel.with_open_bin (Filename.concat oracle "decode.jsonl") (fun ic ->
      In_channel.input_lines ic
      |> List.iter (fun line ->
          incr n;
          let r = Json.of_string line in
          let t =
            List.assoc (Json.string (Json.mem "tokenizer" r)) tokenizers
          in
          let ids = Json.ints (Json.mem "ids" r) in
          let what =
            Json.string (Json.mem "tokenizer" r)
            ^ " "
            ^ Json.string (Json.mem "case" r)
          in
          check_string (what ^ " decode")
            (Json.string (Json.mem "decode" r))
            (T.decode t ids);
          check_string
            (what ^ " decode_with_timestamps")
            (Json.string (Json.mem "decode_with_timestamps" r))
            (T.decode_with_timestamps t ids)));
  !n

let check_tokenizer assets oracle ordinary =
  let tokenizers =
    List.map (fun ((name, _, _) as c) -> (name, tokenizer assets c)) configs
  in
  List.iter (fun (name, t) -> tables name t oracle) tokenizers;
  let n = corpus tokenizers oracle (Json.read ordinary) in
  let m = decodes tokenizers oracle in
  Printf.printf
    "tokenizer: 3 tables, %d encodings, %d decodings: %d differences\n" n m
    !diffs

(* Model *)

let f64 x = Nx.cast Nx.float64 x
let max_abs x = Nx.item [] (Nx.max (Nx.abs (f64 x)))
let max_diff a b = Nx.item [] (Nx.max (Nx.abs (Nx.sub (f64 a) (f64 b))))

let entry name a =
  match Nx_io.Archive.find name a with
  | Some p -> p
  | None -> failwith ("no entry " ^ name)

let float32 name a = Nx.unpack Nx.float32 (entry name a)
let float64 name a = Nx.unpack Nx.float64 (entry name a)

(* The two comparisons of {!Tolerance}, each a difference when it fails and
   [(distance, allowed)]: [same] is the port at float64 against upstream's
   float64 output; [accurate] the port at float32 against it, at most
   {!Tolerance.c} times upstream's own float32 distance. *)
let same what depths up64 got64 =
  let tol = Tolerance.same depths ~n:(Nx.numel up64) (max_abs up64) in
  let e = max_diff up64 got64 in
  if e > tol then diff "%s: %.3g from upstream, allowed %.3g" what e tol;
  (e, tol)

let accurate what ~up32 ~up64 got =
  let bound = Tolerance.c *. max_diff up32 up64 in
  let e = max_diff up64 got in
  if e > bound then diff "%s: %.3g from float64, allowed %.3g" what e bound;
  (e, bound)

(* The first greedy token after upstream's filters at the first step
   (whisper/decoding.py): SuppressBlank bans [encode " "] and the end of text,
   SuppressTokens the non-speech ids and the specials no transcript holds. With
   timestamps, ApplyTimestampRules leaves only the timestamps of the first
   second ([max_initial_timestamp] 1.0 s, 50 steps of 0.02 s). *)
let first_token t ~timestamps logits =
  let banned =
    Array.concat
      [
        T.encode t " ";
        [| T.special t End_of_text |];
        T.non_speech t;
        Array.map (T.special t)
          [|
            Task Transcribe;
            Task Translate;
            Start_of_transcript;
            Start_of_prev;
            Start_of_lm;
            No_speech;
          |];
      ]
  in
  let l = Array.copy logits in
  Array.iter (fun id -> l.(id) <- Float.neg_infinity) banned;
  let ts = T.special t (Timestamp 0) in
  if timestamps then
    Array.iteri
      (fun id _ -> if id < ts || id > ts + 50 then l.(id) <- Float.neg_infinity)
      l;
  let best = ref 0 in
  Array.iteri (fun i x -> if x > l.(!best) then best := i) l;
  !best

let argmax a =
  let best = ref 0 in
  Array.iteri (fun i x -> if x > a.(!best) then best := i) a;
  !best

let greedy_first dir name protocol id =
  let file = Filename.concat dir (Printf.sprintf "%s-%s.jsonl" name protocol) in
  In_channel.with_open_bin file (fun ic ->
      let rec find () =
        match In_channel.input_line ic with
        | None -> failwith (file ^ ": no " ^ id)
        | Some line ->
            let r = Json.of_string line in
            if Json.string (Json.mem "id" r) = id then
              Json.int
                (Json.mem "token" (List.hd (Json.list (Json.mem "steps" r))))
            else find ()
      in
      find ())

let row logits i =
  Array.init (Nx.dim 2 logits) (fun j -> Nx.item [ 0; i; j ] logits)

let check_model weights assets oracle features greedy name ids =
  let path = Filename.concat weights (name ^ ".pt") in
  let m = Whisper.Model.load Nx.float32 path in
  let m64 = Whisper.Model.load Nx.float64 path in
  let d = m.dims in
  let multilingual = Whisper.Model.multilingual d in
  let t =
    let file =
      if multilingual then "multilingual.tiktoken" else "gpt2.tiktoken"
    in
    match
      T.of_ranks
        ~languages:(Whisper.Model.languages d)
        (read (Filename.concat assets file))
    with
    | Ok t -> t
    | Error e -> failwith e
  in
  let sot =
    T.sot_sequence t (if multilingual then Some ("en", Transcribe) else None)
  in
  let tokens = Array.append sot [| T.special t No_timestamps |] in
  let n = Array.length tokens in
  List.iter
    (fun id ->
      let o =
        Nx_io.load_npz
          (Filename.concat oracle (Printf.sprintf "%s.%s.npz" name id))
      in
      let mel =
        Nx.unpack Nx.float32
          (Nx_io.load_npy
             (Filename.concat features
                (Printf.sprintf "%s.mel%d.short.npy" id d.n_mels)))
      in
      let mel = Nx.unsqueeze ~axes:[ 0 ] (Nx.transpose mel) in
      let at what = Printf.sprintf "%s %s %s" name id what in
      (* Encoder *)
      let up = float32 "encoder" o and up64 = float64 "encoder.f64" o in
      let enc = Whisper.Model.encode m mel in
      let enc64 = Whisper.Model.encode m64 (f64 mel) in
      let flat x = Nx.reshape [| d.n_audio_ctx; d.n_audio_state |] x in
      let e64, tol64 =
        same (at "encoder f64") (Tolerance.encoder d) up64 (flat enc64)
      in
      let e, bound = accurate (at "encoder") ~up32:up ~up64 (flat enc) in
      (* Upstream's manual path scales q and k by d^-1/4 each; its default,
         SDPA, the scores by d^-1/2. *)
      let manual = float32 "encoder.manual" o in
      Printf.printf
        "%s: encoder |max| %.3g; at float64 %.3g from upstream (allowed %.3g); \
         at float32 %.3g from upstream's float64 (allowed %.3g, upstream's own \
         %.3g), %.3g from upstream's float32, %.3g from its manual path\n\
         %!"
        (name ^ " " ^ id)
        (max_abs up) e64 tol64 e bound (max_diff up up64)
        (max_diff up (flat enc))
        (max_diff manual (flat enc));
      (* Decoder: the notimestamps protocol's initial tokens fed at once. *)
      check_ids (at "initial tokens")
        (Array.map Int64.to_int
           (Nx.to_array (Nx.unpack Nx.int64 (entry "tokens" o))))
        tokens;
      let ids = Nx.create Nx.int64 [| 1; n |] (Array.map Int64.of_int tokens) in
      let logits m enc =
        let index = Kaun.Cache_index.rows ~context:d.n_text_ctx [| n |] in
        let caches = Whisper.Model.cache m ~slots:d.n_text_ctx Nx.Host.on in
        Whisper.Model.logits m
          (fst
             (Whisper.Model.decode m
                (Whisper.Model.audio m enc)
                caches index ids))
      in
      let got = logits m enc and got64 = logits m64 enc64 in
      let up = Nx.unsqueeze ~axes:[ 0 ] (float32 "logits" o) in
      let up64 = Nx.unsqueeze ~axes:[ 0 ] (float32 "logits.f64" o) in
      (* Both convert float64 logits to float32 (upstream's [.float()]), so at
         float64 the two are one float32 step apart at most. *)
      let tol64 = 2. *. Float.ldexp 1. (-24) *. max_abs up64 in
      let e64 = max_diff up64 got64 in
      if e64 > tol64 then
        diff "%s: %.3g from upstream, allowed %.3g" (at "logits f64") e64 tol64;
      let e, bound = accurate (at "logits") ~up32:up ~up64 got in
      Printf.printf
        "%s: logits |max| %.3g; at float64 %.3g from upstream (allowed %.3g); \
         at float32 %.3g from upstream's float64 (allowed %.3g, upstream's own \
         %.3g), %.3g from upstream's float32\n\
         %!"
        (name ^ " " ^ id)
        (max_abs up) e64 tol64 e bound (max_diff up up64) (max_diff up got);
      (* First tokens: raw argmax, and greedy after the first step's filters. *)
      List.iter
        (fun (protocol, i, timestamps) ->
          let ours = row got i and theirs = row up i in
          check_int (at (protocol ^ " argmax")) (argmax theirs) (argmax ours);
          let want = greedy_first greedy name protocol id in
          let tok = first_token t ~timestamps ours in
          check_int (at (protocol ^ " first token")) want tok;
          Printf.printf
            "%s: %s first token %d (upstream %d), argmax %d (upstream %d)\n%!"
            (name ^ " " ^ id)
            protocol tok want (argmax ours) (argmax theirs))
        [ ("notimestamps", n - 1, false); ("timestamps", n - 2, true) ])
    ids

(* Greedy decoding *)

(* A TSV file's rows, each as its (column, value) pairs. *)
let tsv path =
  match In_channel.with_open_bin path In_channel.input_lines with
  | [] -> []
  | header :: rows ->
      let cols = String.split_on_char '\t' header in
      List.filter_map
        (fun row ->
          if row = "" then None
          else Some (List.combine cols (String.split_on_char '\t' row)))
        rows

(* A JSON lines file's records by their "id". *)
let by_id path =
  let t = Hashtbl.create 256 in
  In_channel.with_open_bin path In_channel.input_lines
  |> List.iter (fun l ->
      if l <> "" then
        let r = Json.of_string l in
        Hashtbl.replace t (Json.string (Json.mem "id" r)) r);
  t

(* Our raw logits at step [i] of a decoding along [tokens]: the initial tokens
   at once, then one token a call, as {!Whisper.Decoding.decode} feeds them. *)
let step_row m (start : Whisper.Decoding.start) features tokens i =
  let context = m.Whisper.Model.dims.n_text_ctx in
  let audio = Whisper.Model.audio m features in
  let caches = Whisper.Model.cache m ~slots:context Nx.Host.on in
  let index = Kaun.Cache_index.rows ~context [| Array.length start.tokens |] in
  let ids a =
    Nx.create Nx.int64 [| 1; Array.length a |] (Array.map Int64.of_int a)
  in
  let h, caches =
    Whisper.Model.decode m audio caches index (ids start.tokens)
  in
  let rec go k index caches h =
    if k = i then
      Whisper.Model.logits m (Nx.slice [ I 0; I (Nx.dim 1 h - 1) ] h)
    else
      let index = Kaun.Cache_index.advance index in
      let h, caches =
        Whisper.Model.decode m audio caches index (ids [| tokens.(k) |])
      in
      go (k + 1) index caches h
  in
  go 0 index caches h

(* A released model, its tokenizer and its language for English. *)
let release weights assets name =
  let m =
    Whisper.Model.load Nx.float32 (Filename.concat weights (name ^ ".pt"))
  in
  let d = m.dims in
  let multilingual = Whisper.Model.multilingual d in
  let t =
    let file =
      if multilingual then "multilingual.tiktoken" else "gpt2.tiktoken"
    in
    match
      T.of_ranks
        ~languages:(Whisper.Model.languages d)
        (read (Filename.concat assets file))
    with
    | Ok t -> t
    | Error e -> failwith e
  in
  (m, t, if multilingual then Some ("en", T.Transcribe) else None)

type outcome = {
  set : string;
  same : bool; (* The tokens are upstream's. *)
  got : Whisper.Decoding.result;
  sum : float; (* Our float32 sum of the chosen candidate. *)
  sum_logprob : float * float; (* Upstream's, and its float64 reference. *)
  avg_logprob : float * float;
  no_speech_prob : float * float;
  margin : float;
      (* The least margin upstream's steps came within: the gap of its two
         greatest filtered logits and the timestamp rule's [|t - x|], or a
         beam's selection, finish and proposal gaps. *)
}

let float_or_inf j = if Json.is_null j then Float.infinity else Json.float j

(* The step at which [got] first differs from [want], reported with upstream's
   gap there between its two greatest filtered logits and its timestamp rule's
   [t - x]. Our raw logits at that step go to [out]/[file]-STEP.npy, for
   {!excuse}. *)
let mismatch ~at ~out ~file m t start features r got =
  let want = Json.ints (Json.mem "tokens" r) in
  let rec first i =
    if i < Array.length got && i < Array.length want && got.(i) = want.(i) then
      first (i + 1)
    else i
  in
  let i = first 0 in
  let token a = if i < Array.length a then a.(i) else T.special t End_of_text in
  let s = List.nth (Json.list (Json.mem "steps" r)) i in
  let margin =
    match List.assoc_opt "ts_rule_margin" (Json.mems s) with
    | Some j when not (Json.is_null j) -> Printf.sprintf "%.6g" (Json.float j)
    | _ -> "none"
  in
  diff
    "%s: step %d samples %d, upstream %d (upstream's top two %d %d, gap %.6g, \
     t - x %s)"
    (at "tokens") i (token got) (token want)
    (Json.int (Json.mem "top1" s))
    (Json.int (Json.mem "top2" s))
    (float_or_inf (Json.mem "gap" s))
    margin;
  Printf.printf "STEP %s %d\n%!" file i;
  Nx_io.save_npy
    (Filename.concat out (Printf.sprintf "%s-%d.npy" file i))
    (step_row m start features want i)

let protocols = [ ("notimestamps", false); ("timestamps", true) ]
let sets = [ "test-clean"; "test-other" ]

(* The corpus WER of [outcomes]' texts after English.normalize. *)
let corpus ~refs texts =
  Whisper.Wer.corpus
    (List.map
       (fun (id, text) ->
         ( Whisper.English.normalize (List.assoc id refs),
           Whisper.English.normalize text ))
       texts)

let summary ~at ~refs outcomes =
  let same = List.filter (fun (_, o) -> o.same) outcomes in
  (* [max |ours - up64|] and [max |up32 - up64|] over the utterances with
     upstream's tokens. *)
  let accurate what field value =
    let ours, up =
      List.fold_left
        (fun (ours, up) (_, o) ->
          let up32, up64 = field o in
          ( Float.max ours (Float.abs (value o -. up64)),
            Float.max up (Float.abs (up32 -. up64)) ))
        (0., 0.) same
    in
    let allowed = Tolerance.c *. up in
    if ours > allowed then
      diff "%s: %.3g from float64, allowed %.3g" (at what) ours allowed;
    Printf.sprintf "%s %.3g from float64 (upstream's own %.3g, allowed %.3g)"
      what ours up allowed
  in
  let sum = accurate "sum_logprob" (fun o -> o.sum_logprob) (fun o -> o.sum) in
  let avg =
    accurate "avg_logprob" (fun o -> o.avg_logprob) (fun o -> o.got.avg_logprob)
  in
  let ns =
    accurate "no_speech_prob"
      (fun o -> o.no_speech_prob)
      (fun o -> o.got.no_speech_prob)
  in
  let margin =
    List.fold_left
      (fun m (_, o) -> Float.min m o.margin)
      Float.infinity outcomes
  in
  let c = corpus ~refs (List.map (fun (id, o) -> (id, o.got.text)) outcomes) in
  ( c,
    Printf.sprintf
      "%d utterances, %d with upstream's tokens; upstream's least margin %.3g; \
       %s; %s; %s"
      (List.length outcomes) (List.length same) margin sum avg ns )

let wer_line (c : Whisper.Wer.counts) =
  Printf.sprintf "WER %.2f%%, S %d D %d I %d"
    (100. *. Whisper.Wer.rate c)
    c.substitutions c.deletions c.insertions

(* Upstream's log-mel of an utterance, [[| 3000; n |]]. *)
let upstream_mel features n id =
  Nx.transpose
    (Nx.unpack Nx.float32
       (Nx_io.load_npy
          (Filename.concat features (Printf.sprintf "%s.mel%d.short.npy" id n))))

let encode_mel m mel = Whisper.Model.encode m (Nx.unsqueeze ~axes:[ 0 ] mel)

(* Each utterance's audio features under [m], from upstream's log-mel. *)
let encode m features id =
  encode_mel m (upstream_mel features m.Whisper.Model.dims.n_mels id)

let decode1 m t o search start features =
  let module D = Whisper.Decoding in
  let out = D.decode m t o search (D.starts [| start |]) features in
  (out, (D.results t o out).(0))

(* The least of a greedy record's steps' gaps and timestamp margins. *)
let greedy_margin r =
  List.fold_left
    (fun m s ->
      let ts =
        match List.assoc_opt "ts_rule_margin" (Json.mems s) with
        | Some j -> Float.abs (float_or_inf j)
        | None -> Float.infinity
      in
      Float.min m (Float.min ts (float_or_inf (Json.mem "gap" s))))
    Float.infinity
    (Json.list (Json.mem "steps" r))

(* Greedy decoding of each utterance, its audio features [encode m id], both
   protocols, against upstream's records. *)
let greedy_run ~what ~encode weights assets greedy greedy64 subset wer out name
    =
  let m, t, language = release weights assets name in
  let d = m.dims in
  let refs =
    List.map
      (fun r -> (List.assoc "id" r, List.assoc "transcript" r))
      (tsv subset)
  in
  let records protocol dir =
    by_id (Filename.concat dir (Printf.sprintf "%s-%s.jsonl" name protocol))
  in
  let records =
    List.map
      (fun (p, _) -> (p, (records p greedy, records p greedy64)))
      protocols
  in
  let outcomes = Hashtbl.create 512 in
  List.iteri
    (fun k (id, _) ->
      let features = encode m id in
      List.iter
        (fun (protocol, timestamps) ->
          let oracle, oracle64 = List.assoc protocol records in
          let r = Hashtbl.find oracle id and r64 = Hashtbl.find oracle64 id in
          let at what = Printf.sprintf "%s %s %s %s" name protocol id what in
          let o = { (Whisper.Decoding.options t language) with timestamps } in
          let start = Whisper.Decoding.start t d o in
          check_ids (at "initial tokens")
            (Json.ints (Json.mem "initial_tokens" r))
            start.tokens;
          let decoded, got = decode1 m t o Greedy start features in
          let same = got.tokens = Json.ints (Json.mem "tokens" r) in
          if same then
            check_string (at "text") (Json.string (Json.mem "text" r)) got.text
          else
            mismatch ~at ~out
              ~file:(Printf.sprintf "%s-%s-%s" name protocol id)
              m t start features r got.tokens;
          let both k =
            (Json.float (Json.mem k r), Json.float (Json.mem k r64))
          in
          let up32, up64 = both "avg_logprob" in
          let length =
            float (Array.length (Json.ints (Json.mem "tokens" r)) + 1)
          in
          Hashtbl.replace outcomes (protocol, id)
            {
              set = Json.string (Json.mem "set" r);
              same;
              got;
              sum = Nx.item [ 0; 0 ] decoded.sums;
              sum_logprob =
                (Json.float (Json.mem "sum_logprob" r), up64 *. length);
              avg_logprob = (up32, up64);
              no_speech_prob = both "no_speech_prob";
              margin = greedy_margin r;
            })
        protocols;
      if (k + 1) mod 20 = 0 then
        Printf.printf "%s: %d utterances\n%!" name (k + 1))
    refs;
  let wer = tsv wer in
  List.iter
    (fun (protocol, _) ->
      List.iter
        (fun set ->
          let at what = Printf.sprintf "%s %s %s %s" name protocol set what in
          let outcomes =
            List.filter_map
              (fun (id, _) ->
                match Hashtbl.find_opt outcomes (protocol, id) with
                | Some o when o.set = set -> Some (id, o)
                | _ -> None)
              refs
          in
          let c, line = summary ~at ~refs outcomes in
          let row =
            List.find
              (fun r ->
                List.assoc "model" r = name
                && List.assoc "protocol" r = protocol
                && List.assoc "set" r = set)
              wer
          in
          let field k = List.assoc k row in
          check_string (at "wer_pct") (field "wer_pct")
            (Printf.sprintf "%.2f" (100. *. Whisper.Wer.rate c));
          List.iter
            (fun (k, v) -> check_int (at k) (int_of_string (field k)) v)
            [
              ("ref_words", c.hits + c.substitutions + c.deletions);
              ("subs", c.substitutions);
              ("dels", c.deletions);
              ("ins", c.insertions);
            ];
          Printf.printf "%s %s %s: %s (upstream %s%%); %s\n%!" name protocol set
            (wer_line c) (field "wer_pct") line)
        sets)
    protocols;
  Printf.printf "%s %s: %d differences\n" name what !diffs

let check_greedy weights assets features =
  greedy_run ~what:"greedy"
    ~encode:(fun m id -> encode m features id)
    weights assets

(* End to end *)

(* [x]'s frames, its last repeated up to 3000: P3-impl/e2e_oracle.py keeps a
   reference's frames up to the first of the floor frames past the audio, which
   all equal it. *)
let frames x =
  let k = Nx.dim 0 x in
  Nx.concatenate ~axis:0
    [
      x;
      Nx.broadcast_to [| 3000 - k; Nx.dim 1 x |] (Nx.slice [ R (k - 1, k) ] x);
    ]

(* The features' gates: within 1e-4 of the float64 reference, and so within 2e-4
   of upstream's float32 features, which lie up to about 6e-5 from it. *)
let gate64 = 1e-4
let gate32 = 2e-4

(* The samples nx.io decodes from each utterance's FLAC file against those
   upstream's load_audio gets from ffmpeg. *)
let check_samples dir paths =
  let same =
    List.fold_left
      (fun same (id, path) ->
        let ffmpeg =
          Nx.unpack Nx.int16
            (Nx_io.load_npy (dir (Printf.sprintf "samples/%s.npy" id)))
        in
        let ours =
          Nx.unpack Nx.int16 (Nx_io.Audio.samples (Nx_io.load_audio path))
        in
        let n = Nx.dim 0 ffmpeg in
        if
          Nx.shape ours = [| n; 1 |]
          && Nx.to_array (Nx.reshape [| n |] ours) = Nx.to_array ffmpeg
        then same + 1
        else begin
          diff "%s: nx.io's samples are not ffmpeg's" id;
          same
        end)
      0 paths
  in
  Printf.printf
    "nx.io's int16 samples equal ffmpeg's on %d of %d utterances\n%!" same
    (List.length paths)

(* Our features of each utterance from its file at [n] bands against the float64
   reference and upstream's float32 features. *)
let check_features dir paths n =
  let worst () = ref (0., "") in
  let ours64 = worst () and ours32 = worst () and up = worst () in
  let note r id e = if e > fst !r then r := (e, id) in
  List.iter
    (fun (id, path) ->
      let ours = Whisper.Audio.(log_mel ~n_mels:n (pad_or_trim (load path))) in
      let ref64 =
        frames
          (Nx.unpack Nx.float64
             (Nx_io.load_npy (dir (Printf.sprintf "ref64/%s.mel%d.npy" id n))))
      in
      let up32 = upstream_mel (dir "features") n id in
      note ours64 id (max_diff ours ref64);
      note ours32 id (max_diff ours up32);
      note up id (max_diff up32 ref64))
    paths;
  let e64, at64 = !ours64 and e32, at32 = !ours32 and eup, atup = !up in
  if e64 > gate64 then
    diff "%d bands: %.3g from float64 at %s, allowed %g" n e64 at64 gate64;
  if e32 > gate32 then
    diff "%d bands: %.3g from upstream's float32 at %s, allowed %g" n e32 at32
      gate32;
  Printf.printf
    "features, %d bands: ours %.3g from float64 (%s; allowed %g), %.3g from \
     upstream's float32 (%s; allowed %g); upstream's float32 %.3g from float64 \
     (%s)\n\
     %!"
    n e64 at64 gate64 e32 at32 gate32 eup atup

(* Each utterance from its FLAC file through {!Whisper.Audio} as one window
   (pad_or_trim, then log_mel), the encoder and greedy decoding, against
   upstream's records, which decode upstream's log-mel of the samples load_audio
   gets from ffmpeg. [oracle] holds greedy/ (MODEL-PROTOCOL.jsonl and wer.tsv),
   greedy64/, features/, and samples/ and ref64/ from P3-impl/e2e_oracle.py; our
   logits at a differing step go to steps/. *)
let check_e2e weights assets data subset oracle name =
  let dir = Filename.concat oracle in
  let paths =
    List.map
      (fun r -> (List.assoc "id" r, Filename.concat data (List.assoc "path" r)))
      (tsv subset)
  in
  check_samples dir paths;
  List.iter (check_features dir paths) [ 80; 128 ];
  let encode m id =
    encode_mel m
      (Whisper.Audio.log_mel ~n_mels:m.Whisper.Model.dims.n_mels
         (Whisper.Audio.pad_or_trim (Whisper.Audio.load (List.assoc id paths))))
  in
  greedy_run ~what:"e2e" ~encode weights assets (dir "greedy") (dir "greedy64")
    subset (dir "greedy/wer.tsv") (dir "steps") name

(* Beam search *)

(* A row of a host array of shape [[| _; _; len |]] at [i; j], up to its first
   [-1]. *)
let host_row x i j =
  let len = Nx.dim 2 x in
  let a = Nx.to_array (Nx.slice [ I i; I j ] x) in
  let rec stop k = if k = len || a.(k) < 0L then k else stop (k + 1) in
  Array.init (stop 0) (fun k -> Int64.to_int a.(k))

(* The least of a beam record's steps' selection, finish and proposal gaps, and
   the step where it lies. *)
let beam_margin r =
  List.fold_left
    (fun (m, at) (i, s) ->
      let v =
        List.fold_left
          (fun v k -> Float.min v (float_or_inf (Json.mem k s)))
          Float.infinity
          [ "selection"; "finish"; "proposal" ]
      in
      if v < m then (v, i) else (m, at))
    (Float.infinity, -1)
    (List.mapi (fun i s -> (i, s)) (Json.list (Json.mem "steps" r)))

(* Beam search of each utterance, without timestamps, at the oracle's beam size,
   patience 1 and no length penalty: upstream's finished candidates in order,
   its live beams, its result and its sums. *)
let check_beam weights assets features beam subset name =
  let m, t, language = release weights assets name in
  let d = m.dims in
  let refs =
    List.map
      (fun r -> (List.assoc "id" r, List.assoc "transcript" r))
      (tsv subset)
  in
  let oracle = by_id beam in
  let o = { (Whisper.Decoding.options t language) with timestamps = false } in
  let start = Whisper.Decoding.start t d o in
  let scores = ref 0. in
  let outcomes =
    List.mapi
      (fun k (id, _) ->
        let r = Hashtbl.find oracle id in
        let at what = Printf.sprintf "%s beam %s %s" name id what in
        check_ids (at "initial tokens")
          (Json.ints (Json.mem "initial_tokens" r))
          start.tokens;
        let size = Json.int (Json.mem "beam_size" r) in
        let out, got =
          decode1 m t o
            (Beam { size; patience = 1. })
            start (encode m features id)
        in
        let fin = Json.list (Json.mem "finished" r) in
        let count = Int64.to_int (Nx.item [ 0 ] out.count) in
        let same_store =
          count = List.length fin
          && List.for_all2
               (fun j f ->
                 scores :=
                   Float.max !scores
                     (Float.abs
                        (Json.float (Json.mem "score" f)
                        -. Nx.item [ 0; j ] out.scores));
                 host_row out.finished 0 j = Json.ints (Json.mem "tokens" f))
               (List.init count Fun.id) fin
        in
        let same_live =
          List.for_all2
            (fun j l -> host_row out.tokens 0 j = Json.ints l)
            (List.init size Fun.id)
            (Json.list (Json.mem "live" r))
        in
        let same = got.tokens = Json.ints (Json.mem "tokens" r) in
        let margin, step = beam_margin r in
        if not (same && same_store && same_live) then
          diff
            "%s: tokens %s, finished %s, live %s (upstream's least margin %.3g \
             at step %d of %d)"
            (at "search") (string_of_bool same)
            (string_of_bool same_store)
            (string_of_bool same_live) margin step
            (Json.int (Json.mem "n_steps" r))
        else check_string (at "text") (Json.string (Json.mem "text" r)) got.text;
        (* The chosen sum: a finished candidate's, or a live beam's, as
           upstream's finalize takes it. *)
        let sum =
          let toks = Json.ints (Json.mem "tokens" r) in
          let rec find j =
            if j = count then
              let rec lane j =
                if j = size then nan
                else if host_row out.tokens 0 j = toks then
                  Nx.item [ 0; j ] out.sums
                else lane (j + 1)
              in
              lane 0
            else if
              host_row out.finished 0 j
              = Array.append toks [| T.special t End_of_text |]
            then Nx.item [ 0; j ] out.scores
            else find (j + 1)
          in
          find 0
        in
        let both k =
          (Json.float (Json.mem k r), Json.float (Json.mem (k ^ ".f64") r))
        in
        let length =
          float (Array.length (Json.ints (Json.mem "tokens" r)) + 1)
        in
        if (k + 1) mod 20 = 0 then
          Printf.printf "%s: %d utterances\n%!" name (k + 1);
        ( id,
          ( Json.string (Json.mem "text" r),
            {
              set = Json.string (Json.mem "set" r);
              same;
              got;
              sum;
              sum_logprob = both "sum_logprob";
              avg_logprob =
                ( Json.float (Json.mem "avg_logprob" r),
                  Json.float (Json.mem "sum_logprob.f64" r) /. length );
              no_speech_prob = both "no_speech_prob";
              margin;
            } ) ))
      refs
  in
  List.iter
    (fun set ->
      let at what = Printf.sprintf "%s beam %s %s" name set what in
      let these = List.filter (fun (_, (_, o)) -> o.set = set) outcomes in
      let c, line =
        summary ~at ~refs (List.map (fun (id, (_, o)) -> (id, o)) these)
      in
      let up =
        corpus ~refs (List.map (fun (id, (text, _)) -> (id, text)) these)
      in
      if List.for_all (fun (_, (_, o)) -> o.same) these && c <> up then
        diff "%s: %s, upstream's %s" (at "wer") (wer_line c) (wer_line up);
      Printf.printf "%s beam %s: %s (upstream %s); %s\n%!" name set (wer_line c)
        (wer_line up) line)
    sets;
  Printf.printf "%s beam: finished scores at most %.3g from upstream's\n" name
    !scores;
  Printf.printf "%s beam: %d differences\n" name !diffs

(* The excuse rule: a token that differs from upstream's at a step is excused
   only where rounding could have moved upstream's decision there. With [δ] the
   greatest distance between our raw logits and upstream's at the step, moving
   each logit by at most [δ] moves the gap between the two greatest filtered
   logits by at most [2 δ], and the timestamp rule's [t - x] (whose normaliser
   cancels) by at most [2 δ]; the rule's two reductions add [ρ]: twice the bound
   nx.cpu states for a Logsumexp of the [n] timestamps, [(L + D + 5 + 6 ln n) u
   + u |t|], plus [u (|t| + |x|)], [u] float32's unit roundoff. A step is
   excused iff upstream's gap is below [2 δ] or its [|t - x|] below [2 δ + ρ].
   [t] and [x] are computed at float64 from upstream's logits under our filter's
   bans before the rule. [dir] holds ours (NAME-PROTOCOL-ID-STEP.npy) and
   upstream's (NAME-PROTOCOL-ID-STEP.upstream.npy, P2-impl/step_logits.py). *)
let excuse weights assets greedy dir =
  let suffix = ".upstream.npy" in
  let releases = Hashtbl.create 4 in
  let u = Float.ldexp 1. (-24) in
  Sys.readdir dir |> Array.to_list
  |> List.filter (String.ends_with ~suffix)
  |> List.sort compare
  |> List.iter (fun f ->
      let key = String.sub f 0 (String.length f - String.length suffix) in
      let row path =
        Nx.unpack Nx.float32 (Nx_io.load_npy (Filename.concat dir path))
      in
      let ours = row (key ^ ".npy") and up = row f in
      let dist = max_diff ours up in
      (* NAME-PROTOCOL-ID-STEP, where an id holds dashes. *)
      let parts = String.split_on_char '-' key in
      let n = List.length parts in
      let step = int_of_string (List.nth parts (n - 1)) in
      let name = List.hd parts and protocol = List.nth parts 1 in
      let id =
        String.concat "-" (List.filteri (fun i _ -> i > 1 && i < n - 1) parts)
      in
      let r =
        Hashtbl.find
          (by_id
             (Filename.concat greedy
                (Printf.sprintf "%s-%s.jsonl" name protocol)))
          id
      in
      let s = List.nth (Json.list (Json.mem "steps" r)) step in
      let gap = float_or_inf (Json.mem "gap" s) in
      let timestamps = protocol = "timestamps" in
      let rho, tx =
        if not timestamps then (0., Float.infinity)
        else begin
          let _, t, language =
            match Hashtbl.find_opt releases name with
            | Some x -> x
            | None ->
                let x = release weights assets name in
                Hashtbl.replace releases name x;
                x
          in
          let module D = Whisper.Decoding in
          let o = { (D.options t language) with timestamps } in
          let ts0 = T.special t (Timestamp 0) and v = T.vocab_size t in
          let sampled =
            Nx.create Nx.int64 [| 1; step |]
              (Array.map Int64.of_int
                 (Array.sub (Json.ints (Json.mem "tokens" r)) 0 step))
          in
          (* The bans before the rule: the rule never fires without a finite
             timestamp, and leaves the timestamps as they are. *)
          let up = Nx.reshape [| 1; v |] up in
          let ids = Nx.reshape [| 1; v |] (Nx.arange Nx.int64 0 v 1) in
          let is_ts = Nx.greater_equal_s ids (Int64.of_int ts0) in
          let no_ts =
            Nx.where is_ts (Nx.scalar Nx.float32 Float.neg_infinity) up
          in
          let pre =
            Nx.where is_ts (D.filter t o ~sampled up)
              (D.filter t o ~sampled no_ts)
          in
          let lp = Nx.log_softmax ~axes:[ 1 ] (Nx.cast Nx.float64 pre) in
          let tt =
            Nx.item [] (Nx.logsumexp (Nx.slice [ I 0; R (ts0, v) ] lp))
          in
          let x = Nx.item [] (Nx.max (Nx.slice [ I 0; R (0, ts0) ] lp)) in
          let terms = v - ts0 in
          let l = ((min terms 1024 + 15) / 16) + min 3 (terms - 1) in
          let dd =
            let blocks = (terms + 1023) / 1024 in
            let rec log2 k p =
              if p >= blocks then k else log2 (k + 1) (2 * p)
            in
            log2 0 1
          in
          let bound =
            ((float (l + dd + 5) +. (6. *. Float.log (float terms))) *. u)
            +. (u *. Float.abs tt)
          in
          ((2. *. bound) +. (u *. (Float.abs tt +. Float.abs x)), tt -. x)
        end
      in
      let margin =
        match List.assoc_opt "ts_rule_margin" (Json.mems s) with
        | Some j -> Float.abs (float_or_inf j)
        | None -> Float.infinity
      in
      let excused = gap < 2. *. dist || margin < (2. *. dist) +. rho in
      if not excused then diff "%s: not excused" key;
      Printf.printf
        "%s %s %s step %d: upstream's gap %.6g, |t - x| %.6g (ours at float64 \
         over upstream's logits %.6g), max |ours - upstream| %.3g over the \
         step's logits, rho %.3g: %s\n\
         %!"
        name protocol id step gap margin tx dist rho
        (if excused then "excused" else "NOT excused"))

let usage () =
  prerr_endline
    "usage: check.exe tokenizer ASSETS ORACLE_TOKENIZER ORDINARY_JSON\n\
    \  ASSETS holds gpt2.tiktoken and multilingual.tiktoken; ORACLE_TOKENIZER\n\
    \  the oracle's tokenizer directory; ORDINARY_JSON P1-impl/tok_tables.py's\n\
    \  ordinary.json.\n\
     usage: check.exe model WEIGHTS ASSETS ORACLE_P1 FEATURES GREEDY MODEL ID...\n\
    \  WEIGHTS holds MODEL.pt; ASSETS the rank files; ORACLE_P1\n\
    \  P1-impl/real_model.py's MODEL.ID.npz; FEATURES the oracle's\n\
    \  ID.melN.short.npy; GREEDY the oracle's greedy MODEL-PROTOCOL.jsonl.\n\
     usage: check.exe greedy WEIGHTS ASSETS FEATURES GREEDY GREEDY64 SUBSET \
     WER_TSV OUT MODEL\n\
    \  GREEDY64 P2-impl/greedy64.py's MODEL-PROTOCOL.jsonl; SUBSET the\n\
    \  oracle's subset.tsv; WER_TSV its greedy/wer.tsv; OUT where our logits\n\
    \  at a step whose token differs from upstream's are written.\n\
     usage: check.exe e2e WEIGHTS ASSETS DATA SUBSET ORACLE MODEL\n\
    \  DATA the LibriSpeech directory SUBSET's paths start from; ORACLE holds\n\
    \  greedy/ (the greedy records and wer.tsv), greedy64/, features/, and\n\
    \  P3-impl/e2e_oracle.py's samples/ and ref64/; steps/ there receives our\n\
    \  logits at a step whose token differs from upstream's.\n\
     usage: check.exe beam WEIGHTS ASSETS FEATURES BEAM SUBSET MODEL\n\
    \  BEAM P2b-impl/beam_oracle.py's MODEL's records.\n\
     usage: check.exe excuse WEIGHTS ASSETS GREEDY DIR\n\
    \  DIR holds greedy's logits at differing steps and upstream's, from\n\
    \  P2-impl/step_logits.py.";
  exit 2

let () =
  (match Array.to_list Sys.argv |> List.tl with
  | [ "tokenizer"; assets; oracle; ordinary ] ->
      check_tokenizer assets oracle ordinary
  | "model" :: weights :: assets :: oracle :: features :: greedy :: name :: ids
    ->
      check_model weights assets oracle features greedy name ids
  | [
   "greedy"; weights; assets; features; greedy; greedy64; subset; wer; out; name;
  ] ->
      check_greedy weights assets features greedy greedy64 subset wer out name
  | [ "e2e"; weights; assets; data; subset; oracle; name ] ->
      check_e2e weights assets data subset oracle name
  | [ "beam"; weights; assets; features; beam; subset; name ] ->
      check_beam weights assets features beam subset name
  | [ "excuse"; weights; assets; greedy; dir ] ->
      excuse weights assets greedy dir
  | _ -> usage ());
  exit (if !diffs = 0 then 0 else 1)
