(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Decoding against upstream's DecodingTask over trimmed tokenizers.

   upstream/options.json holds, for a set of upstream's options, the initial
   tokens, where sampling starts, where the start of transcript sits and the
   suppressed ids that DecodingTask computes, over the tokenizers of
   upstream/gpt2-1000.tiktoken and upstream/multilingual-1000.tiktoken with 99
   languages. upstream/filters.npz holds logits recorded from greedy decodes of
   a small model over the gpt2 tokenizer, at float16, and upstream/filters.json
   the tokens sampled before them, some of them crafted, and the ids that
   SuppressBlank, SuppressTokens and ApplyTimestampRules each set to -inf, and
   all three in upstream's order. upstream/decode.pt is a small randomly
   initialised Whisper over the gpt2 tokenizer, saved as OpenAI saves its
   releases; upstream/decode.npz holds audio features, and upstream/decode.json
   upstream's greedy decodes of them at float32 with float64 references of their
   avg_logprob and no_speech_prob. *)

open Windtrap
open Whisper_support
module T = Whisper.Tokenizer
module D = Whisper.Decoding

let tokenizer file =
  let ranks = In_channel.with_open_bin file In_channel.input_all in
  match T.of_ranks ~languages:99 ranks with Ok t -> t | Error e -> failwith e

let gpt2 = tokenizer "upstream/gpt2-1000.tiktoken"
let multilingual = tokenizer "upstream/multilingual-1000.tiktoken"

(* A model's dimensions: only [n_vocab], which says whether it is multilingual,
   and [n_text_ctx] matter to decoding's options. *)
let dims ~multilingual n_text_ctx =
  {
    Whisper.Model.n_mels = 80;
    n_audio_ctx = 1500;
    n_audio_state = 384;
    n_audio_head = 6;
    n_audio_layer = 4;
    n_vocab = (if multilingual then 51865 else 51864);
    n_text_ctx;
    n_text_state = 384;
    n_text_head = 6;
    n_text_layer = 4;
  }

(* Options *)

let task = function
  | "transcribe" -> T.Transcribe
  | "translate" -> T.Translate
  | s -> failwith s

let golden = Json.read "upstream/options.json"
let n_text_ctx = Json.int (Json.mem "n_text_ctx" golden)

(* Upstream's options of a case: its suppress_tokens "-1" stands for the
   non-speech ids. *)
let options_of c =
  let multi = Json.bool (Json.mem "multilingual" c) in
  let t = if multi then multilingual else gpt2 in
  let language =
    if multi then
      Some
        ( Json.string (Json.mem "language" c),
          task (Json.string (Json.mem "task" c)) )
    else None
  in
  let given = Json.ints (Json.mem "suppress" c) in
  let suppress =
    if Json.bool (Json.mem "suppress_non_speech" c) then
      Array.append (T.non_speech t) given
    else given
  in
  ( t,
    dims ~multilingual:multi n_text_ctx,
    {
      (D.options t) with
      timestamps = Json.bool (Json.mem "timestamps" c);
      suppress;
    },
    language )

let upstream =
  cases "start and suppressed are upstream's"
    ~id:(fun c -> Json.string (Json.mem "name" c))
    (Json.list (Json.mem "cases" golden))
    (fun c ->
      let t, dims, o, language = options_of c in
      let s =
        D.start
          ~prompt:(Json.ints (Json.mem "prompt" c))
          ~prefix:(Json.ints (Json.mem "prefix" c))
          t dims o language
      in
      equal (array int) (Json.ints (Json.mem "initial_tokens" c)) s.tokens;
      equal ~msg:"sample_begin" int
        (Json.int (Json.mem "sample_begin" c))
        (Array.length s.tokens);
      equal ~msg:"sot_index" int (Json.int (Json.mem "sot_index" c)) s.sot;
      equal ~msg:"suppressed" (array int)
        (Json.ints (Json.mem "suppress_tokens" c))
        (D.suppressed t o))

(* Initial tokens *)

let sot = T.special gpt2 Start_of_transcript
let sot_prev = T.special gpt2 Start_of_prev
let no_timestamps = T.special gpt2 No_timestamps

(* Prompts of every length about the cut at [n_text_ctx / 2 - 1] ids. *)
let prompt_gen ctx =
  let open Gen in
  let half = (ctx / 2) - 1 in
  let* n =
    frequency
      [
        (1, constant 0);
        (1, constant 1);
        (2, int_range (half - 1) (half + 1));
        (2, int_range 2 (2 * ctx));
      ]
  in
  array ~size:(constant n) (int_range 0 999)

let layout =
  prop "the initial tokens are the prompt's last ids, the start, the prefix"
    Gen.(
      let* ctx = int_range 8 64 in
      let* prompt = prompt_gen ctx in
      let* prefix = array ~size:(int_range 0 3) (int_range 0 999) in
      let+ timestamps = bool in
      (ctx, prompt, prefix, timestamps))
    (fun (ctx, prompt, prefix, timestamps) ->
      let o = { (D.options gpt2) with timestamps } in
      let half = (ctx / 2) - 1 and n = Array.length prompt in
      cover "no prompt" (n = 0);
      cover "a prompt cut" (n > half);
      cover "a prompt of the cut's length" (n = half);
      let kept = Array.sub prompt (max 0 (n - half)) (min n half) in
      let head = if n = 0 then [||] else Array.append [| sot_prev |] kept in
      let start = if timestamps then [| sot |] else [| sot; no_timestamps |] in
      let want = Array.concat [ head; start; prefix ] in
      assume (Array.length want <= ctx);
      let s =
        D.start ~prompt ~prefix gpt2 (dims ~multilingual:false ctx) o None
      in
      equal (array int) want s.tokens;
      equal int sot s.tokens.(s.sot))

let context =
  test "start refuses initial tokens past n_text_ctx, and takes them at it"
    (fun () ->
      let start n =
        D.start ~prefix:(Array.make n 7) gpt2
          (dims ~multilingual:false 16)
          (D.options gpt2) None
      in
      equal int 16 (Array.length (start 15).tokens);
      raises_match Exn.invalid_arg (fun () -> start 16))

let languages =
  test "start needs a language exactly for a multilingual model" (fun () ->
      let en = Some ("en", T.Transcribe) in
      raises_match Exn.invalid_arg (fun () ->
          D.start multilingual
            (dims ~multilingual:true 448)
            (D.options multilingual) None);
      raises_match Exn.invalid_arg (fun () ->
          D.start gpt2 (dims ~multilingual:false 448) (D.options gpt2) en))

(* A text prompt is [" "] and the text without the white space Python's
   [str.strip] removes at its ends: Unicode's, U+3000 and U+00A0 included. *)
let prompt =
  cases "prompt encodes the text stripped as Python strips it"
    ~id:(fun (s, _) -> Printf.sprintf "%S" s)
    [
      ("hello", " hello");
      ("  hello\n", " hello");
      ("\x0bhello\x1c", " hello");
      ("\u{3000}\u{00A0}hello world\u{2028}", " hello world");
      ("", " ");
      ("\u{3000}", " ");
    ]
    (fun (s, want) -> equal (array int) (T.encode gpt2 want) (D.prompt gpt2 s))

let ids =
  cases "start and suppressed refuse an id outside the vocabulary"
    ~id:string_of_int
    [ -1; T.vocab_size gpt2 ]
    (fun id ->
      let o = D.options gpt2 in
      let d = dims ~multilingual:false 448 in
      raises_match Exn.invalid_arg (fun () ->
          D.start ~prompt:[| id |] gpt2 d o None);
      raises_match Exn.invalid_arg (fun () ->
          D.start ~prefix:[| id |] gpt2 d o None);
      raises_match Exn.invalid_arg (fun () ->
          D.suppressed gpt2 { o with suppress = [| id |] }))

(* Suppressed tokens *)

let specials =
  List.map (T.special gpt2)
    [
      Task Transcribe;
      Task Translate;
      Start_of_transcript;
      Start_of_prev;
      Start_of_lm;
      No_speech;
    ]

let suppressed =
  prop "suppressed is the given ids and the specials, in order, each once"
    Gen.(list ~size:(int_range 0 8) (int_range 0 (T.vocab_size gpt2 - 1)))
    (fun given ->
      let o = { (D.options gpt2) with suppress = Array.of_list given } in
      let want = List.sort_uniq Int.compare (given @ specials) in
      equal (list int) want (Array.to_list (D.suppressed gpt2 o)))

(* Logit filter *)

let vocab = T.vocab_size gpt2
let first = T.special gpt2 (Timestamp 0)
let eot = T.special gpt2 End_of_text
let filters = Json.read "upstream/filters.json"

let recorded =
  match Nx_io.Archive.find "logits" (Nx_io.load_npz "upstream/filters.npz") with
  | Some p -> Nx.cast Nx.float32 (Nx.unpack Nx.float16 p)
  | None -> failwith "upstream/filters.npz: no logits"

(* [input] with the ids of [ranges], [[start, stop)] pairs, at [-∞]. *)
let banned_in ranges input =
  let out = Array.copy input in
  List.iter
    (fun r ->
      match Json.ints r with
      | [| a; b |] -> Array.fill out a (b - a) Float.neg_infinity
      | _ -> failwith "range")
    (Json.list ranges);
  out

(* Histories as a filter reads them: row [k] holds [h.(k)], then [-1]. *)
let history hs =
  let w = Array.fold_left (fun m h -> max m (Array.length h)) 0 hs in
  Nx.init Nx.int64
    [| Array.length hs; w |]
    (fun i ->
      let h = hs.(i.(0)) in
      if i.(1) < Array.length h then Int64.of_int h.(i.(1)) else -1L)

(* Upstream's rows, each group of one options filtered in one call of several
   lanes. *)
let filter_upstream =
  let rows =
    List.mapi (fun i c -> (i, c)) (Json.list (Json.mem "cases" filters))
  in
  let key (_, c) =
    (Json.bool (Json.mem "timestamps" c), Json.ints (Json.mem "suppress" c))
  in
  let groups =
    List.fold_left
      (fun gs r ->
        let k = key r in
        match List.assoc_opt k gs with
        | Some l -> (k, r :: l) :: List.remove_assoc k gs
        | None -> (k, [ r ]) :: gs)
      [] rows
    |> List.rev_map (fun (k, l) -> (k, List.rev l))
  in
  cases "filter bans upstream's ids"
    ~id:(fun ((ts, ids), rows) ->
      Printf.sprintf "%s, %d suppressed, %d rows"
        (if ts then "timestamps" else "no timestamps")
        (Array.length ids) (List.length rows))
    groups
    (fun ((timestamps, suppress), rows) ->
      let idx = Array.of_list (List.map fst rows) in
      let x =
        Nx.take ~axis:0
          (Nx.create Nx.int64
             [| Array.length idx |]
             (Array.map Int64.of_int idx))
          recorded
      in
      let sampled =
        history
          (Array.of_list
             (List.map (fun (_, c) -> Json.ints (Json.mem "sampled" c)) rows))
      in
      let o = { (D.options gpt2) with timestamps; suppress } in
      let got = D.filter gpt2 o ~sampled x in
      List.iteri
        (fun k (i, c) ->
          let input = Nx.to_array (Nx.slice [ I i; A ] recorded) in
          equal
            ~msg:(Json.string (Json.mem "name" c))
            (array float_exact)
            (banned_in (Json.mem "all" c) input)
            (Nx.to_array (Nx.slice [ I k; A ] got)))
        rows)

(* Logits about [0.], the timestamps shifted together so that their total
   probability falls on either side of the likeliest text token's, and histories
   of text, timestamps and the end of text. *)
let logits_gen =
  let open Gen in
  let* xs = array ~size:(constant vocab) (float_range (-8.) 8.) in
  let+ shift = float_range (-12.) 4. in
  for id = first to vocab - 1 do
    xs.(id) <- xs.(id) +. shift
  done;
  xs

let history_gen =
  let open Gen in
  let token =
    frequency
      [
        (4, int_range 0 999);
        (1, constant eot);
        (3, int_range first (vocab - 1));
        (2, int_range first (first + 60));
      ]
  in
  list ~size:(int_range 0 5) token |> map Array.of_list

(* Lanes of logits and histories, filtered in one call. *)
let lanes_gen =
  Gen.(list ~size:(int_range 1 4) (pair logits_gen history_gen))
  |> Gen.map Array.of_list

let filtered o lanes =
  let n = Array.length lanes in
  let x =
    Nx.create Nx.float32 [| n; vocab |]
      (Array.concat (Array.to_list (Array.map fst lanes)))
  in
  let sampled = history (Array.map snd lanes) in
  (Nx.to_array x, Nx.to_array (D.filter gpt2 o ~sampled x))

let finite x = Float.is_finite x

let input_or_banned =
  prop "a filtered logit is its input or -inf" lanes_gen (fun lanes ->
      let input, got = filtered (D.options gpt2) lanes in
      Array.iteri
        (fun i y ->
          if y <> input.(i) && y <> Float.neg_infinity then
            failf "lane %d, id %d is %h, from %h" (i / vocab) (i mod vocab) y
              input.(i))
        got)

(* log Σ exp over [ids], at float64. *)
let logsumexp xs ids =
  let m =
    List.fold_left (fun m id -> Float.max m xs.(id)) Float.neg_infinity ids
  in
  if m = Float.neg_infinity then m
  else
    m
    +. Float.log
         (List.fold_left (fun s id -> s +. Float.exp (xs.(id) -. m)) 0. ids)

let range a b = List.init (b - a) (fun i -> a + i)

(* The timestamp rules alone: no blank suppression, and no ids suppressed but
   the specials, which are no text token's. *)
let rules =
  let o = { (D.options gpt2) with suppress = [||]; suppress_blank = false } in
  let specials = Array.to_list (D.suppressed gpt2 o) in
  prop "the timestamp rules ban what their contract states" lanes_gen
    (fun lanes ->
      let input, got = filtered o lanes in
      Array.iteri
        (fun k (_, sampled) ->
          let input = Array.sub input (k * vocab) vocab in
          let out = Array.sub got (k * vocab) vocab in
          let n = Array.length sampled in
          let is_ts id = id >= first in
          let last = n >= 1 && is_ts sampled.(n - 1) in
          let penultimate = n < 2 || is_ts sampled.(n - 2) in
          let all_banned ids what =
            List.iter
              (fun id ->
                if out.(id) <> Float.neg_infinity then
                  failf "lane %d, %s: id %d is not banned" k what id)
              ids
          in
          all_banned [ T.special gpt2 No_timestamps ] "no timestamps";
          cover "first step" (n = 0);
          cover "a timestamp after a timestamp" (n >= 2 && last && penultimate);
          cover "a timestamp after text" (last && not penultimate);
          if last && penultimate then all_banned (range first vocab) "pair";
          if last && not penultimate then all_banned (range 0 eot) "after text";
          if n = 0 then begin
            all_banned (range 0 first) "first step, text";
            all_banned (range (first + 51) vocab) "first step, late"
          end;
          (* The last timestamp sampled, which need not be the greatest. *)
          let stop =
            match
              Array.fold_left
                (fun l id -> if is_ts id then Some id else l)
                None sampled
            with
            | None -> first
            | Some ts -> if last && not penultimate then ts else ts + 1
          in
          all_banned (range first stop) "decreasing";
          (* Text and the end of text lie below every timestamp. *)
          let below i =
            is_ts sampled.(i)
            && Array.exists (fun id -> id > sampled.(i)) (Array.sub sampled 0 i)
          in
          cover "a timestamp below an earlier one"
            (List.exists below (range 0 n));
          (* From [stop] on, only the pair and the first step ban timestamps. *)
          if n > 0 && not (last && penultimate) then
            List.iter
              (fun id ->
                if finite input.(id) && not (finite out.(id)) then
                  failf "lane %d: timestamp %d is banned" k id)
              (range stop vocab);
          (* Text survives only where the timestamps' total probability is at
             most the likeliest text token's, up to float32's rounding of the
             two. *)
          let text = List.filter (fun id -> finite out.(id)) (range 0 first) in
          let ts = logsumexp out (range first vocab) in
          let text_max =
            List.fold_left
              (fun m id -> Float.max m out.(id))
              Float.neg_infinity text
          in
          cover "text kept" (text <> []);
          if text <> [] then
            at_most ~msg:"text kept" (float 1e-4) ~than:text_max ts;
          (* Where the masks keep text, the comparison alone banned it. *)
          let masked = n = 0 || (last && not penultimate) in
          let none = T.special gpt2 No_timestamps in
          let kept =
            List.filter
              (fun id ->
                id <> none && (not (List.mem id specials)) && finite input.(id))
              (range 0 first)
          in
          if text = [] && (not masked) && kept <> [] then begin
            cover "text banned by the comparison" true;
            let m =
              List.fold_left
                (fun m id -> Float.max m input.(id))
                Float.neg_infinity kept
            in
            at_least ~msg:"text banned" (float 1e-4) ~than:m ts
          end
          else cover "text banned by the comparison" false)
        lanes)

let filter_shape =
  test "filter refuses logits of another shape" (fun () ->
      let o = D.options gpt2 in
      let sampled = Nx.full Nx.int64 [| 2; 1 |] (-1L) in
      raises_match Exn.invalid_arg (fun () ->
          D.filter gpt2 o ~sampled (Nx.zeros Nx.float32 [| 2; vocab + 1 |]));
      raises_match Exn.invalid_arg (fun () ->
          D.filter gpt2 o ~sampled (Nx.zeros Nx.float32 [| vocab |]));
      raises_match Exn.invalid_arg (fun () ->
          D.filter gpt2 o ~sampled (Nx.zeros Nx.float32 [| 3; vocab |])))

(* Greedy decoding *)

let model = Whisper.Model.load Nx.float32 "upstream/decode.pt"
let decodes = Json.read "upstream/decode.json"

let features =
  match
    Nx_io.Archive.find "features" (Nx_io.load_npz "upstream/decode.npz")
  with
  | Some p -> Nx.unpack Nx.float32 p
  | None -> failwith "upstream/decode.npz: no features"

let case_options c =
  let given = Json.ints (Json.mem "suppress" c) in
  {
    (D.options gpt2) with
    timestamps = Json.bool (Json.mem "timestamps" c);
    suppress_blank = Json.bool (Json.mem "suppress_blank" c);
    suppress =
      (if Json.bool (Json.mem "suppress_non_speech" c) then
         Array.append (T.non_speech gpt2) given
       else given);
    max_initial_timestamp =
      Some (Json.int (Json.mem "max_initial_timestamp" decodes));
  }

(* A case's initial tokens under [o], after its prompt and prefix. *)
let case_start c o =
  D.start
    ~prompt:(Json.ints (Json.mem "prompt" c))
    ~prefix:(Json.ints (Json.mem "prefix" c))
    gpt2 model.dims o None

(* Each case's audios, decoded together, each from the case's start. *)
let decode_case ?(decode = D.decode) search o c =
  let f = Nx.slice [ I (Json.int (Json.mem "features" c)); A; A; A ] features in
  let s = case_start c o in
  let out =
    decode model gpt2 o search (D.starts (Array.make (Nx.dim 0 f) s)) f
  in
  (out, D.results gpt2 o out)

let decode_cases = Json.list (Json.mem "cases" decodes)

let greedy_upstream =
  cases "greedy samples upstream's tokens and text"
    ~id:(fun c -> Json.string (Json.mem "name" c))
    decode_cases
    (fun c ->
      let o = case_options c in
      let s = case_start c o in
      equal ~msg:"initial tokens" (array int)
        (Json.ints (Json.mem "initial_tokens" c))
        s.tokens;
      Array.iteri
        (fun k (r : D.result) ->
          let lane = List.nth (Json.list (Json.mem "lanes" c)) k in
          let msg = Printf.sprintf "lane %d" k in
          equal ~msg (array int) (Json.ints (Json.mem "tokens" lane)) r.tokens;
          equal ~msg string (Json.string (Json.mem "text" lane)) r.text)
        (snd (decode_case D.Greedy o c)))

(* Over every lane of every decode, the port is as accurate as upstream
   ({!Tolerance}): its distance to the float64 reference is at most
   {!Tolerance.c} times upstream's own float32 distance. *)
let accurate ?decode () =
  let rows =
    List.concat_map
      (fun c ->
        List.combine
          (Json.list (Json.mem "lanes" c))
          (Array.to_list
             (snd (decode_case ?decode D.Greedy (case_options c) c))))
      decode_cases
  in
  let check what (got : D.result -> float) =
    let distance f = List.fold_left (fun m r -> Float.max m (f r)) 0. rows in
    let up32 l = Json.float (Json.mem what l)
    and up64 l = Json.float (Json.mem (what ^ ".f64") l) in
    let upstream = distance (fun (l, _) -> Float.abs (up32 l -. up64 l)) in
    let ours = distance (fun (l, r) -> Float.abs (got r -. up64 l)) in
    at_most ~msg:what float_exact ~than:(Tolerance.c *. upstream) ours
  in
  check "avg_logprob" (fun r -> r.avg_logprob);
  check "no_speech_prob" (fun r -> r.no_speech_prob)

let greedy_accurate =
  test "greedy's avg_logprob and no_speech_prob are as accurate as upstream"
    accurate

let decode_refuses =
  test "decode refuses features and starts of other shapes" (fun () ->
      let o = D.options gpt2 in
      let s = D.starts [| D.start gpt2 model.dims o None |] in
      raises_match Exn.invalid_arg (fun () ->
          D.decode model gpt2 o Greedy s (Nx.zeros Nx.float32 [| 1; 23; 32 |]));
      raises_match Exn.invalid_arg (fun () ->
          D.decode model gpt2 o Greedy s (Nx.slice [ I 0; A; A; A ] features)))

(* Beam search *)

let beams = Json.read "upstream/beam.json"

let beam_case c =
  let size = Json.int (Json.mem "size" c) in
  let patience =
    let p = Json.mem "patience" c in
    if Json.is_null p then 1. else Json.float p
  in
  let length_penalty =
    let a = Json.mem "length_penalty" c in
    if Json.is_null a then None else Some (Json.float a)
  in
  ({ (case_options c) with length_penalty }, D.Beam { size; patience })

(* A row of a host array of shape [[| _; len |]], up to its first [-1]. *)
let row a len k =
  let rec stop i =
    if i = len || a.((k * len) + i) < 0L then i else stop (i + 1)
  in
  Array.init (stop 0) (fun i -> Int64.to_int a.((k * len) + i))

(* Sums of at most [n_text_ctx / 2 = 8] log probabilities, ours and upstream's
   at float32 from logits that differ by about a float32 step: about [1e-5]
   apart, where a wrong log probability (unfiltered, tempered) moves a sum by
   [1e-2] and more. *)
let sum_tolerance = 1e-4

let beam_upstream ?decode name =
  cases name
    ~id:(fun c -> Json.string (Json.mem "name" c))
    (Json.list (Json.mem "cases" beams))
    (fun c ->
      let o, search = beam_case c in
      let out, results = decode_case ?decode search o c in
      let len = Nx.dim 2 out.tokens and g = Nx.dim 1 out.tokens in
      let mc = Nx.dim 1 out.finished in
      equal ~msg:"m" int (Json.int (Json.mem "max_candidates" c)) mc;
      let tokens = Nx.to_array out.tokens and sums = Nx.to_array out.sums in
      let finished = Nx.to_array out.finished in
      let scores = Nx.to_array out.scores and count = Nx.to_array out.count in
      let close what want got =
        equal ~msg:what (float sum_tolerance) want got
      in
      List.iteri
        (fun i a ->
          let msg what = Printf.sprintf "audio %d %s" i what in
          let fin = Json.list (Json.mem "finished" a) in
          equal ~msg:(msg "finished") int (List.length fin)
            (Int64.to_int count.(i));
          List.iteri
            (fun j f ->
              let what = msg (Printf.sprintf "finished %d" j) in
              equal ~msg:what (array int)
                (Json.ints (Json.mem "tokens" f))
                (row finished len ((i * mc) + j));
              close what (Json.float (Json.mem "score" f)) scores.((i * mc) + j))
            fin;
          let sums_up = Json.list (Json.mem "sums" a) in
          List.iteri
            (fun j l ->
              let what = msg (Printf.sprintf "beam %d" j) in
              equal ~msg:what (array int) (Json.ints l)
                (row tokens len ((i * g) + j));
              close what (Json.float (List.nth sums_up j)) sums.((i * g) + j))
            (Json.list (Json.mem "live" a));
          let r = results.(i) in
          equal ~msg:(msg "tokens") (array int)
            (Json.ints (Json.mem "tokens" a))
            r.tokens;
          equal ~msg:(msg "text") string
            (Json.string (Json.mem "text" a))
            r.text;
          close (msg "avg_logprob")
            (Json.float (Json.mem "avg_logprob" a))
            r.avg_logprob)
        (Json.list (Json.mem "audios" c)))

(* A beam keeps [size * patience] finished candidates, rounded half to even as
   Python's round: 2.5 to 2, 1.5 and 3.5 to the even 2 and 4. *)
let candidates =
  cases "a beam keeps size * patience candidates, rounded half to even"
    ~id:(fun (size, patience, _) -> Printf.sprintf "%d × %g" size patience)
    [
      (5, 0.5, Some 2);
      (3, 0.5, Some 2);
      (5, 0.7, Some 4);
      (1, 1.5, Some 2);
      (2, 1.25, Some 2);
      (4, 1., Some 4);
      (1, 0.5, None);
      (3, 0., None);
      (2, -1., None);
    ]
    (fun (size, patience, want) ->
      let o = D.options gpt2 in
      let s = D.starts [| D.start gpt2 model.dims o None |] in
      let f = Nx.slice [ I 0; R (0, 1); A; A ] features in
      let decode () = D.decode model gpt2 o (Beam { size; patience }) s f in
      match want with
      | Some m -> equal int m (Nx.dim 1 (decode ()).finished)
      | None -> raises_match Exn.invalid_arg (fun () -> ignore (decode ())))

let search_refuses =
  test "decode refuses an empty group and a length penalty outside [0, 1]"
    (fun () ->
      let o = D.options gpt2 in
      let s = D.starts [| D.start gpt2 model.dims o None |] in
      let f = Nx.slice [ I 0; R (0, 1); A; A ] features in
      let decode o search = ignore (D.decode model gpt2 o search s f) in
      raises_match Exn.invalid_arg (fun () ->
          decode o (Beam { size = 0; patience = 1. }));
      raises_match Exn.invalid_arg (fun () ->
          decode o
            (Sample
               {
                 best_of = 0;
                 temperature = Nx.scalar Nx.float32 1.;
                 key = Nx.Rng.key 0;
               }));
      List.iter
        (fun a ->
          raises_match Exn.invalid_arg (fun () ->
              decode { o with length_penalty = Some a } Greedy))
        [ -0.1; 1.1; Float.nan ])

(* Sampling: over [n] lanes of one audio, each first token is drawn from the
   softmax of the first step's filtered logits over the temperature. A banned
   token is never drawn, and each likely one's count lies within four standard
   deviations of its binomial mean. The key is fixed, so the draws are too. *)
let sample_law =
  test "sampling draws the first token with its filtered probability" (fun () ->
      let n = 2000 and temperature = 0.7 in
      let o = { (D.options gpt2) with timestamps = false } in
      let s = D.start gpt2 model.dims o None in
      let f = Nx.slice [ I 0; R (0, 1); A; A ] features in
      let search =
        D.Sample
          {
            best_of = n;
            temperature = Nx.scalar Nx.float32 temperature;
            key = Nx.Rng.key 11;
          }
      in
      let out = D.decode model gpt2 o search (D.starts [| s |]) f in
      let p =
        let k = Array.length s.tokens in
        let index = Kaun.Cache_index.whole ~batch:1 ~seq:k () in
        let ids =
          Nx.create Nx.int64 [| 1; k |] (Array.map Int64.of_int s.tokens)
        in
        let h, _ =
          Whisper.Model.decode model
            (Whisper.Model.audio model f)
            (Whisper.Model.cache model ~slots:0 Nx.Host.on)
            index ids
        in
        let logits = Whisper.Model.logits model (Nx.slice [ A; I (k - 1) ] h) in
        let sampled = Nx.full Nx.int64 [| 1; 0 |] (-1L) in
        Nx.to_array
          (Nx.softmax ~axes:[ 1 ]
             (Nx.div_s (D.filter gpt2 o ~sampled logits) temperature))
      in
      let counts = Array.make vocab 0 in
      Array.iter
        (fun id -> counts.(Int64.to_int id) <- counts.(Int64.to_int id) + 1)
        (Nx.to_array (Nx.slice [ I 0; A; I 0 ] out.tokens));
      let likely = ref 0 in
      Array.iteri
        (fun id c ->
          let msg = Printf.sprintf "id %d" id in
          if p.(id) = 0. then equal ~msg int 0 c
          else if p.(id) > 0.02 then begin
            incr likely;
            let mean = Float.of_int n *. p.(id) in
            let sd = Float.sqrt (mean *. (1. -. p.(id))) in
            equal ~msg (float (4. *. sd)) mean (Float.of_int c)
          end)
        counts;
      at_least ~msg:"likely tokens" int ~than:2 !likely)

(* Results *)

(* A host outcome of [n = 1] audio from rows of tokens and sums. *)
let outcome ?(finished = []) ~count lanes =
  let len = 4 in
  let rows l =
    Nx.create Nx.int64
      [| 1; List.length l; len |]
      (Array.concat
         (List.map
            (fun (t, _) ->
              Array.init len (fun i ->
                  if i < Array.length t then Int64.of_int t.(i) else -1L))
            l))
  in
  let sums l =
    Nx.create Nx.float32 [| 1; List.length l |] (Array.of_list (List.map snd l))
  in
  {
    D.tokens = rows lanes;
    sums = sums lanes;
    finished = rows finished;
    scores = sums finished;
    count = Nx.create Nx.int64 [| 1 |] [| Int64.of_int count |];
    no_speech = Nx.create Nx.float32 [| 1 |] [| 0.25 |];
  }

let results_rank =
  test "results scores each candidate by its sum over its length" (fun () ->
      let o = D.options gpt2 in
      let chosen ?(o = o) out = (D.results gpt2 o out).(0) in
      (* An empty candidate scores -∞ without a penalty, its sum with one. *)
      let lanes = [ ([| eot |], -0.125); ([| 5; eot |], -3.) ] in
      let r = chosen (outcome ~count:0 lanes) in
      equal (array int) [| 5 |] r.tokens;
      equal float_exact (-1.5) r.avg_logprob;
      equal float_exact 0.25 r.no_speech_prob;
      let r =
        chosen ~o:{ o with length_penalty = Some 0. } (outcome ~count:0 lanes)
      in
      equal (array int) [||] r.tokens;
      equal float_exact (-0.125) r.avg_logprob;
      (* A beam fills its candidates with its lanes by decreasing sum, the lower
         lane first among equal ones, until it holds [size]. *)
      let fin = [ ([| 6; eot; -1; -1 |], -1.25); ([| 5; 5; eot; -1 |], -3.) ] in
      let tied = [ ([| 7; 7 |], -0.5); ([| 8; 9; 10 |], -0.5) ] in
      let r = chosen (outcome ~finished:fin ~count:1 tied) in
      equal ~msg:"tied lanes" (array int) [| 7; 7 |] r.tokens;
      let ordered = [ ([| 7; 7 |], -3.); ([| 8; 9; 10 |], -0.5) ] in
      let r = chosen (outcome ~finished:fin ~count:1 ordered) in
      equal ~msg:"by sum" (array int) [| 8; 9; 10 |] r.tokens;
      let r = chosen (outcome ~finished:fin ~count:2 ordered) in
      equal ~msg:"held" (array int) [| 6 |] r.tokens;
      (* The first of equal scores wins. *)
      let r =
        chosen
          ~o:{ o with length_penalty = Some 1. }
          (outcome ~count:0 [ ([| 1; 2 |], -1.); ([| 3; 4 |], -1.) ])
      in
      equal (array int) [| 1; 2 |] r.tokens)

(* Starts *)

let starts_layout =
  prop "starts right-aligns each start in rows of a power of two"
    Gen.(list ~size:(int_range 1 4) (int_range 1 20))
    (fun lens ->
      let s =
        Array.of_list
          (List.map
             (fun n ->
               ({ tokens = Array.init n (fun i -> 100 + i); sot = n / 2 }
                 : D.start))
             lens)
      in
      let st = D.starts s in
      let longest = List.fold_left max 0 lens in
      let w = Nx.dim 1 st.tokens in
      cover "a power of two above the longest" (w > longest);
      is_true ~msg:"power of two" (w land (w - 1) = 0);
      at_least int ~than:longest w;
      if w > 1 then less int ~than:longest (w / 2);
      Array.iteri
        (fun i (x : D.start) ->
          let n = Array.length x.tokens in
          let row =
            Array.map Int64.to_int (Nx.to_array (Nx.slice [ I i ] st.tokens))
          in
          equal (array int)
            (Array.append (Array.make (w - n) (-1)) x.tokens)
            row;
          equal int (w - n + x.sot) (Int64.to_int (Nx.item [ i ] st.sot)))
        s;
      let wide = D.starts ~width:(longest + 3) s in
      equal int (longest + 3) (Nx.dim 1 wide.tokens))

let starts_refuses =
  test "starts refuses no start and a start wider than its width" (fun () ->
      raises_match Exn.invalid_arg (fun () -> D.starts [||]);
      raises_match Exn.invalid_arg (fun () ->
          D.starts ~width:2
            [| ({ tokens = [| 1; 2; 3 |]; sot = 0 } : D.start) |]))

(* Batches *)

(* Three audios, each after its own prompt and prefix: decoded together, each
   gives the tokens it gives alone. nx's kernels associate a sum as its shape
   says, so a batch's logits round otherwise, and its sums lie within
   {!sum_tolerance}. *)
let batch_law =
  let prompts =
    let c = List.nth decode_cases 3 in
    let p = Json.ints (Json.mem "prompt" c) in
    [|
      ([||], [||]); (p, Json.ints (Json.mem "prefix" c)); (Array.sub p 0 3, [||]);
    |]
  in
  let f =
    Nx.stack ~axis:0
      [
        Nx.slice [ I 0; I 0; A; A ] features;
        Nx.slice [ I 3; I 1; A; A ] features;
        Nx.slice [ I 4; I 0; A; A ] features;
      ]
  in
  cases "a batch of audios decodes as each audio alone"
    ~id:(fun (ts, name, _) ->
      Printf.sprintf "%s, %s" name
        (if ts then "timestamps" else "no timestamps"))
    (List.concat_map
       (fun ts ->
         [
           (ts, "greedy", D.Greedy);
           (ts, "beam 3, patience 1.5", D.Beam { size = 3; patience = 1.5 });
         ])
       [ false; true ])
    (fun (timestamps, _, search) ->
      let o = { (D.options gpt2) with timestamps } in
      let start (prompt, prefix) =
        D.start ~prompt ~prefix gpt2 model.dims o None
      in
      let all =
        D.decode model gpt2 o search (D.starts (Array.map start prompts)) f
      in
      Array.iteri
        (fun i p ->
          let one =
            D.decode model gpt2 o search
              (D.starts [| start p |])
              (Nx.slice [ R (i, i + 1); A; A ] f)
          in
          let msg what = Printf.sprintf "audio %d %s" i what in
          let part x = Nx.to_array (Nx.slice [ I i ] x) in
          let one_part x = Nx.to_array (Nx.slice [ I 0 ] x) in
          equal ~msg:(msg "tokens") (array int64) (one_part one.tokens)
            (part all.tokens);
          equal ~msg:(msg "finished") (array int64) (one_part one.finished)
            (part all.finished);
          equal ~msg:(msg "count") int64 (Nx.item [ 0 ] one.count)
            (Nx.item [ i ] all.count);
          let near what want got =
            Array.iteri
              (fun j w ->
                if Float.is_finite w then
                  equal ~msg:(msg what) (float sum_tolerance) w got.(j)
                else equal ~msg:(msg what) float_exact w got.(j))
              want
          in
          near "sums" (one_part one.sums) (part all.sums);
          near "scores" (one_part one.scores) (part all.scores);
          near "no_speech"
            [| Nx.item [ 0 ] one.no_speech |]
            [| Nx.item [ i ] all.no_speech |])
        prompts)

(* Languages

   upstream/languages.pt is a small randomly initialised multilingual Whisper
   (51865 tokens), upstream/languages.npz audio features of four audios and the
   probabilities upstream's detect_language gives them over the trimmed
   multilingual tokenizer at float32, with the float64 model's as the reference,
   and upstream/languages.json the languages' codes in their tokens' order and
   each audio's detected language. *)

let lang_model = Whisper.Model.load Nx.float32 "upstream/languages.pt"
let lang_golden = Json.read "upstream/languages.json"

let lang_entry name =
  match Nx_io.Archive.find name (Nx_io.load_npz "upstream/languages.npz") with
  | Some p -> p
  | None -> failwith ("upstream/languages.npz: no " ^ name)

let strings j = Array.of_list (List.map Json.string (Json.list j))

let languages_upstream =
  test "languages are upstream's probabilities, as accurately" (fun () ->
      let features = Nx.unpack Nx.float32 (lang_entry "features") in
      let got = D.languages lang_model multilingual features in
      let up32 = Nx.unpack Nx.float32 (lang_entry "probs32") in
      let up64 = Nx.unpack Nx.float64 (lang_entry "probs64") in
      let codes = T.languages multilingual in
      equal ~msg:"codes" (array string)
        (strings (Json.mem "codes" lang_golden))
        codes;
      equal ~msg:"shape" (array int) (Nx.shape up32) (Nx.shape got);
      let distance a =
        Nx.item [] (Nx.max (Nx.abs (Nx.sub (Nx.cast Nx.float64 a) up64)))
      in
      at_most float_exact ~than:(Tolerance.c *. distance up32) (distance got);
      (* The likeliest, the first of equal ones, as upstream's argmax. *)
      let detected =
        Array.map
          (fun i -> codes.(Int32.to_int i))
          (Nx.to_array (Nx.cast Nx.int32 (Nx.argmax ~axis:1 got)))
      in
      equal ~msg:"detected" (array string)
        (strings (Json.mem "detected" lang_golden))
        detected)

let languages_refuses =
  test "languages refuses an English-only model" (fun () ->
      raises_match Exn.invalid_arg (fun () ->
          D.languages model gpt2
            (Nx.zeros Nx.float32
               [| 1; model.dims.n_audio_ctx; model.dims.n_audio_state |])))

(* Compression ratio

   upstream/ratio.json holds texts with their length in bytes, the length of
   zlib.compress's stream of them under the zlib upstream's Python links, and
   upstream's compression_ratio: decodes of the oracle's utterances, long-form
   decodes near 2.4, loops near 2.4, the runs of at most 31 bytes that zlib
   scores above 2.4, Hebrew and Hindi sentences repeated, and texts ending in
   U+FFFD. *)

let ratios = Json.read "upstream/ratio.json"
let ratio_texts = Json.list (Json.mem "texts" ratios)
let lz_length s = String.length (Lz.Zlib.encode_string ~level:6 s)

(* Upstream's threshold, transcribe.py's compression_ratio_threshold. *)
let threshold = 2.4

let ratio_upstream =
  cases "compression_ratio is the text's bytes over lz's zlib stream's"
    ~id:(fun c -> Printf.sprintf "%S" (Json.string (Json.mem "text" c)))
    ratio_texts
    (fun c ->
      let s = Json.string (Json.mem "text" c) in
      let n = Json.int (Json.mem "n" c) and z = Json.int (Json.mem "zlib" c) in
      let got = D.compression_ratio s in
      equal float_exact (Float.of_int n /. Float.of_int (lz_length s)) got;
      if lz_length s = z then
        equal ~msg:"upstream's, where the streams' lengths agree" float_exact
          (Json.float (Json.mem "ratio" c))
          got)

(* Baseline: each golden's index, kind, length, zlib's and lz's stream lengths,
   and whether upstream's ratio and ours exceed 2.4. *)
let ratio_decisions =
  test "compression_ratio's decisions at 2.4 against upstream's" (fun () ->
      let mark r = if r > threshold then "F" else "-" in
      List.iteri
        (fun i c ->
          let s = Json.string (Json.mem "text" c) in
          Printf.printf "%2d %-11s %3d %3d %3d %s %s\n" i
            (Json.string (Json.mem "kind" c))
            (String.length s)
            (Json.int (Json.mem "zlib" c))
            (lz_length s)
            (mark (Json.float (Json.mem "ratio" c)))
            (mark (D.compression_ratio s)))
        ratio_texts;
      expect (output ())
      @@ __POS_OF__
           {|
         0 oracle       98  86  86 - -
         1 oracle       50  54  54 - -
         2 oracle       34  42  42 - -
         3 oracle       52  58  58 - -
         4 near        266 113 113 - -
         5 near        333 138 139 F -
         6 loop         65  28  27 - F
         7 loop         69  28  27 F F
         8 loop         80  34  33 - F
         9 loop         82  34  33 F F
        10 loop         67  29  28 - -
        11 loop         70  29  28 F F
        12 loop         49  21  20 - F
        13 loop         51  21  20 F F
        14 loop         81  35  34 - -
        15 loop         85  35  34 F F
        16 loop         46  20  19 - F
        17 loop         49  20  19 F F
        18 short        27  11  22 F -
        19 short        28  11  22 F -
        20 short        29  11  23 F -
        21 short        30  11  23 F -
        22 short        31  11  23 F -
        23 short        27  11  22 F -
        24 short        28  11  22 F -
        25 short        29  11  23 F -
        26 short        30  11  23 F -
        27 short        31  11  23 F -
        28 short        27  11  22 F -
        29 short        28  11  22 F -
        30 short        29  11  23 F -
        31 short        30  11  23 F -
        32 short        31  11  23 F -
        33 short        30  12  26 F -
        34 short        29  12  26 F -
        35 short        31  12  27 F -
        36 short        27  11  22 F -
        37 short        28  11  22 F -
        38 short        29  11  23 F -
        39 short        30  11  23 F -
        40 short        31  11  23 F -
        41 short        30  12  26 F -
        42 empty         0   8   8 - -
        43 written      69  65  65 - -
        44 written     139  73  71 - -
        45 written     209  73  71 F F
        46 written      51  54  54 - -
        47 written     103  58  57 - -
        48 written     155  58  57 F F
        49 written     113  90  90 - -
        50 written     227  96  95 - -
        51 written     341  96  95 F F
        52 written      91  79  79 - -
        53 written     183  85  83 - -
        54 written     275  85  83 F F
        55 replacement  18  26  26 - -
        56 replacement   9  14  18 - -
        |})

let ratio_text_gen =
  let open Gen in
  let utf8 =
    map
      (fun us ->
        let b = Buffer.create 16 in
        List.iter (Buffer.add_utf_8_uchar b) us;
        Buffer.contents b)
      (list ~size:(int_range 0 300) uchar)
  in
  let run =
    let* unit = string_of ~size:(int_range 1 2) char in
    let+ n = int_range 19 33 in
    String.sub (String.concat "" (List.init n (fun _ -> unit))) 0 n
  in
  frequency
    [
      (1, constant "");
      (2, map (String.make 1) char);
      (4, run);
      (3, string_of ~size:(int_range 0 4096) char);
      (3, utf8);
    ]

let ratio_law =
  prop "compression_ratio is the length over lz's zlib stream's at level 6"
    ratio_text_gen (fun s ->
      cover "empty" (s = "");
      cover "at most 31 bytes" (String.length s <= 31);
      cover "past 1 KiB" (String.length s > 1024);
      equal float_exact
        (Float.of_int (String.length s) /. Float.of_int (lz_length s))
        (D.compression_ratio s))

let ratio_empty =
  test "compression_ratio of the empty text is 0" (fun () ->
      equal float_exact 0. (D.compression_ratio ""))

(* Compiled *)

let jit_decode =
  let open Nx.Ptree in
  let params :
      ((Nx.host Nx.float32_t, Nx.host) Whisper.Model.t, Nx.host) Nx.Ptree.t =
    instantiate (module Whisper.Model.Params)
  in
  let jitted = Hashtbl.create 4 in
  fun m t o search starts f ->
    (* One compiled function per options, which it captures. *)
    let fn =
      match Hashtbl.find_opt jitted o with
      | Some fn -> fn
      | None ->
          let fn =
            Rune.jit
              (params @-> v D.walk_search @-> v D.walk_starts @-> tensor
              @-> returns (v D.walk_outcome))
              (fun m search starts f -> D.decode m t o search starts f)
          in
          Hashtbl.replace jitted o fn;
          fn
    in
    fn m search starts f

let greedy_jit =
  test "Rune.jit of greedy decoding samples upstream's tokens, as accurately"
    (fun () ->
      List.iter
        (fun c ->
          Array.iteri
            (fun k (r : D.result) ->
              let lane = List.nth (Json.list (Json.mem "lanes" c)) k in
              equal
                ~msg:
                  (Printf.sprintf "%s lane %d"
                     (Json.string (Json.mem "name" c))
                     k)
                (array int)
                (Json.ints (Json.mem "tokens" lane))
                r.tokens)
            (snd (decode_case ~decode:jit_decode D.Greedy (case_options c) c)))
        decode_cases;
      accurate ~decode:jit_decode ())

let () =
  exit
    (run "whisper.decoding"
       [
         group "start" [ upstream; layout; context; languages; prompt; ids ];
         group "suppressed" [ suppressed ];
         group "filter"
           [ filter_upstream; input_or_banned; rules; filter_shape ];
         group "greedy" [ greedy_upstream; greedy_accurate; decode_refuses ];
         group "beam"
           [
             beam_upstream
               "beam search finishes, keeps and chooses upstream's candidates";
           ];
         group "search" [ candidates; search_refuses; sample_law ];
         group "results" [ results_rank ];
         group "starts" [ starts_layout; starts_refuses ];
         group "batch" [ batch_law ];
         group "languages" [ languages_upstream; languages_refuses ];
         group "compression ratio"
           [ ratio_upstream; ratio_decisions; ratio_law; ratio_empty ];
         group "compiled"
           [
             greedy_jit;
             beam_upstream ~decode:jit_decode
               "Rune.jit of beam search finishes, keeps and chooses upstream's \
                candidates";
           ];
       ])
