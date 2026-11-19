(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Options *)

type options = {
  timestamps : bool;
  suppress : int array;
  suppress_blank : bool;
  max_initial_timestamp : int option;
  length_penalty : float option;
}

(* whisper/decoding.py, DecodingOptions: a max_initial_timestamp of 1.0 s, the
   50th timestamp of 0.02 s. *)
let options t =
  {
    timestamps = true;
    suppress = Tokenizer.non_speech t;
    suppress_blank = true;
    max_initial_timestamp = Some 50;
    length_penalty = None;
  }

let check_ids t what ids =
  let n = Tokenizer.vocab_size t in
  Array.iter
    (fun id ->
      if id < 0 || id >= n then
        invalid_argf "Whisper.Decoding: %s id %d outside [0, %d]" what id (n - 1))
    ids

(* Initial tokens

   whisper/decoding.py, DecodingTask._get_initial_tokens. Upstream also cuts the
   prefix to its last [n_text_ctx / 2 - sample_len] ids, which at its default
   [sample_len] of [n_text_ctx / 2] is Python's [prefix[-0:]], the whole
   prefix. *)

type start = { tokens : int array; sot : int }

let start ?(prompt = [||]) ?(prefix = [||]) t (dims : Model.dims) o language =
  (match (language, Model.multilingual dims) with
  | None, true ->
      invalid_arg
        "Whisper.Decoding.start: a multilingual model needs a language"
  | Some _, false ->
      invalid_arg
        "Whisper.Decoding.start: an English-only model has no language"
  | _ -> ());
  check_ids t "prompt" prompt;
  check_ids t "prefix" prefix;
  let sot_sequence = Tokenizer.sot_sequence t language in
  let sot_sequence =
    if o.timestamps then sot_sequence
    else Array.append sot_sequence [| Tokenizer.special t No_timestamps |]
  in
  let prompt =
    let n = Array.length prompt in
    if n = 0 then [||]
    else
      let keep = max 0 (min n ((dims.n_text_ctx / 2) - 1)) in
      Array.append
        [| Tokenizer.special t Start_of_prev |]
        (Array.sub prompt (n - keep) keep)
  in
  let tokens = Array.concat [ prompt; sot_sequence; prefix ] in
  if Array.length tokens > dims.n_text_ctx then
    invalid_argf
      "Whisper.Decoding.start: %d initial tokens exceed n_text_ctx %d"
      (Array.length tokens) dims.n_text_ctx;
  { tokens; sot = Array.length prompt }

(* whisper/transcribe.py: tokenizer.encode(" " + initial_prompt.strip()). *)
let prompt t s = Tokenizer.encode t (" " ^ Space.strip s)

(* whisper/decoding.py, DecodingTask._get_suppress_tokens. Upstream suppresses
   none of the specials when it is given no ids to suppress; here they are
   suppressed whatever the ids. *)
let suppressed t o =
  check_ids t "suppress" o.suppress;
  let specials =
    List.map (Tokenizer.special t)
      [
        Task Transcribe;
        Task Translate;
        Start_of_transcript;
        Start_of_prev;
        Start_of_lm;
        No_speech;
      ]
  in
  Array.of_list
    (List.sort_uniq Int.compare (Array.to_list o.suppress @ specials))

(* Logit filter

   whisper/decoding.py, SuppressBlank, SuppressTokens and ApplyTimestampRules,
   over lanes. Upstream's [tokens.shape[1] == sample_begin] is a lane that has
   sampled nothing. Each ban is a comparison of the ids against per-lane bounds
   read from the history, so the filter is one elementwise expression and a
   [where]; only the timestamps' last rule reduces. *)

let shape_string s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* What a filter reads besides the logits and the history, placed at [at]: the
   suppressed ids and the first step's blank ids as rows of the vocabulary. *)
type 'd rules = {
  vocab : int;
  suppressed : 'd Nx.bool_t;
  blank : 'd Nx.bool_t option;
  timestamps : bool;
  ts0 : int;
  eot : int;
  none : int;
  initial : int option;
}

let rules t o at =
  let vocab = Tokenizer.vocab_size t in
  let row ids =
    let b = Array.make vocab false in
    Array.iter (fun id -> b.(id) <- true) ids;
    Nx.place at (Nx.create Nx.bool [| 1; vocab |] b)
  in
  let eot = Tokenizer.special t End_of_text in
  let ts0 = Tokenizer.special t (Timestamp 0) in
  {
    vocab;
    suppressed = row (suppressed t o);
    blank =
      (if o.suppress_blank then
         Some (row (Array.append (Tokenizer.encode t " ") [| eot |]))
       else None);
    timestamps = o.timestamps;
    ts0;
    eot;
    none = Tokenizer.special t No_timestamps;
    initial = Option.map (fun m -> ts0 + m) o.max_initial_timestamp;
  }

let check_filter r ~sampled logits =
  let s = Nx.shape logits and h = Nx.shape sampled in
  if Array.length s <> 2 || s.(1) <> r.vocab then
    invalid_argf "Whisper.Decoding.filter: logits of shape %s, not [lanes; %d]"
      (shape_string s) r.vocab;
  if Array.length h <> 2 || h.(0) <> s.(0) then
    invalid_argf
      "Whisper.Decoding.filter: sampled of shape %s for logits of shape %s"
      (shape_string h) (shape_string s)

let apply r ~sampled logits =
  let lanes = Nx.dim 0 logits in
  let at = Nx.beside logits in
  let int x = Nx.scalar_at at Nx.int64 (Int64.of_int x) in
  let ( ||| ) = Nx.logical_or and ( &&& ) = Nx.logical_and in
  let ids =
    Nx.reshape [| 1; r.vocab |] (Nx.arange_at at Nx.int64 0 r.vocab 1)
  in
  (* A history of no column reads as one of [-1]. *)
  let sampled =
    if Nx.dim 1 sampled = 0 then Nx.full_at at Nx.int64 [| lanes; 1 |] (-1L)
    else sampled
  in
  let count =
    Nx.count ~axes:[ 1 ] ~keepdims:true (Nx.greater_equal_s sampled 0L)
  in
  let first = Nx.equal_s count 0L in
  let ban =
    match r.blank with
    | Some blank -> r.suppressed ||| (first &&& blank)
    | None -> r.suppressed
  in
  let neg_inf = Nx.scalar_at at Nx.float32 Float.neg_infinity in
  if not r.timestamps then Nx.where ban neg_inf logits
  else
    (* The token [k] places before the next, read where there is one. *)
    let back k =
      Nx.take_along_axis ~axis:1
        (Nx.maximum_s (Nx.sub_s count (Int64.of_int k)) 0L)
        sampled
    in
    let is_ts x = Nx.greater_equal_s x (Int64.of_int r.ts0) in
    let last = Nx.greater_equal_s count 1L &&& is_ts (back 1) in
    let penultimate = Nx.less_s count 2L ||| is_ts (back 2) in
    (* The last timestamp sampled is the one at the greatest column, as
       upstream's [timestamps[-1]]. *)
    let cols =
      Nx.reshape
        [| 1; Nx.dim 1 sampled |]
        (Nx.arange_at at Nx.int64 0 (Nx.dim 1 sampled) 1)
    in
    let col =
      Nx.max ~axes:[ 1 ] ~keepdims:true
        (Nx.where (is_ts sampled) cols (int (-1)))
    in
    let ts = Nx.take_along_axis ~axis:1 (Nx.maximum_s col 0L) sampled in
    (* Timestamps below [from] are banned: before the last one sampled, and that
       one too unless it closes a segment, which the next may open with it. *)
    let from =
      Nx.where (Nx.less_s col 0L) (int r.ts0)
        (Nx.where (last &&& Nx.logical_not penultimate) ts (Nx.add_s ts 1L))
    in
    let id_ts = is_ts ids in
    let initial =
      match r.initial with
      | Some m -> Nx.logical_not id_ts ||| Nx.greater_s ids (Int64.of_int m)
      | None -> Nx.logical_not id_ts
    in
    let ban =
      ban
      ||| Nx.equal_s ids (Int64.of_int r.none)
      ||| (last &&& penultimate &&& id_ts)
      ||| (last &&& Nx.logical_not penultimate
          &&& Nx.less_s ids (Int64.of_int r.eot))
      ||| (id_ts &&& Nx.less ids from)
      ||| (first &&& initial)
    in
    let x = Nx.where ban neg_inf logits in
    let lp = Nx.log_softmax ~axes:[ 1 ] x in
    let timestamps =
      Nx.logsumexp ~axes:[ 1 ] ~keepdims:true
        (Nx.slice [ A; R (r.ts0, r.vocab) ] lp)
    in
    let text =
      Nx.max ~axes:[ 1 ] ~keepdims:true (Nx.slice [ A; R (0, r.ts0) ] lp)
    in
    Nx.where (Nx.greater timestamps text &&& Nx.logical_not id_ts) neg_inf x

let filter t o ~sampled logits =
  let r = rules t o (Nx.beside logits) in
  check_filter r ~sampled logits;
  apply r ~sampled logits

(* Search *)

type 'd search =
  | Greedy
  | Sample of {
      best_of : int;
      temperature : 'd Nx.float32_t;
      key : 'd Nx.Rng.key;
    }
  | Beam of { size : int; patience : float }

(* whisper/decoding.py, BeamSearchDecoder: [round (beam_size * patience)],
   Python's round, which takes the even neighbour of a half. *)
let candidates size patience =
  let x = float size *. patience in
  let r = Float.round x in
  let r =
    if Float.abs (x -. Float.trunc x) = 0.5 && Float.rem r 2. <> 0. then
      r -. Float.copy_sign 1. x
    else r
  in
  int_of_float r

module Search = struct
  type ('a, 'd) t = 'd search

  let walk c s =
    let open Nx.Ptree.Walk in
    match s with
    | Greedy ->
        case c "greedy";
        Greedy
    | Sample { best_of; temperature; key } ->
        case c "sample";
        let best_of = field c "best_of" int best_of in
        let temperature = field c "temperature" tensor temperature in
        let key = field c "key" Nx.Rng.walk key in
        Sample { best_of; temperature; key }
    | Beam { size; patience } ->
        case c "beam";
        let size = field c "size" int size in
        ignore (field c "candidates" int (candidates size patience));
        Beam { size; patience }
end

(* Starts *)

type 'd starts = { tokens : 'd Nx.int64_t; sot : 'd Nx.int64_t }

module Starts = struct
  type ('a, 'd) t = 'd starts

  let walk c s =
    let open Nx.Ptree.Walk in
    let tokens = field c "tokens" tensor s.tokens in
    let sot = field c "sot" tensor s.sot in
    { tokens; sot }
end

let starts ?width (s : start array) =
  let n = Array.length s in
  if n = 0 then invalid_arg "Whisper.Decoding.starts: no start";
  let longest =
    Array.fold_left (fun m (x : start) -> max m (Array.length x.tokens)) 0 s
  in
  let width =
    match width with
    | Some w -> w
    | None ->
        let rec up w = if w >= longest then w else up (2 * w) in
        up 1
  in
  Array.iter
    (fun (x : start) ->
      if Array.length x.tokens > width then
        invalid_argf
          "Whisper.Decoding.starts: %d initial tokens exceed width %d"
          (Array.length x.tokens) width)
    s;
  let pad (x : start) = width - Array.length x.tokens in
  let tokens =
    Nx.init Nx.int64 [| n; width |] (fun i ->
        let x = s.(i.(0)) and c = i.(1) - pad s.(i.(0)) in
        if c < 0 then -1L else Int64.of_int x.tokens.(c))
  in
  let sot =
    Nx.init Nx.int64 [| n |] (fun i ->
        let x = s.(i.(0)) in
        Int64.of_int (pad x + x.sot))
  in
  { tokens; sot }

(* Outcome *)

type 'd outcome = {
  tokens : 'd Nx.int64_t;
  sums : 'd Nx.float32_t;
  finished : 'd Nx.int64_t;
  scores : 'd Nx.float32_t;
  count : 'd Nx.int64_t;
  no_speech : 'd Nx.float32_t;
}

module Outcome = struct
  type ('a, 'd) t = 'd outcome

  let walk c o =
    let open Nx.Ptree.Walk in
    let tokens = field c "tokens" tensor o.tokens in
    let sums = field c "sums" tensor o.sums in
    let finished = field c "finished" tensor o.finished in
    let scores = field c "scores" tensor o.scores in
    let count = field c "count" tensor o.count in
    let no_speech = field c "no_speech" tensor o.no_speech in
    { tokens; sums; finished; scores; count; no_speech }
end

(* Decoding

   whisper/decoding.py, DecodingTask._main_loop, GreedyDecoder and
   BeamSearchDecoder.update. The [n] audios decode in groups of [g] lanes: lane
   [l] is the [l mod g]-th of audio [l / g].

   Audio [i] owns the slots from [i * per], [per = w + g * len]: its initial
   tokens take its first [w], fed once for the group, and each lane a fresh slot
   per step after them. A beam continues another lane's history by taking that
   lane's row of the table, so the pools are never copied, and two lanes share a
   slot only at the same column after the same tokens.

   A step decodes each live lane's last token, filters the logits and chooses.
   The prefill and the first choice run before the loop, so the loop's last step
   decodes nothing it does not choose from. *)

type ('a, 'd) state = {
  step : 'd Nx.int64_t; (* The steps taken: the next token's column. *)
  sampled : 'd Nx.int64_t;
      (* [[| lanes; len |]]: each lane's sampled tokens, then [-1]. *)
  sums : 'd Nx.float32_t; (* [[| lanes |]], at float32 as upstream sums. *)
  live : 'd Nx.bool_t;
      (* [[| lanes |]]: whether the lane samples at the next step. A beam's
         lanes stop with their audio. *)
  table : 'd Nx.int64_t;
      (* [[| lanes; n_text_ctx |]]: the slot of each of a lane's positions. *)
  finished : 'd Nx.int64_t;
      (* A beam's finished candidates, written by a scatter whose slots past [m]
         write nothing, as upstream stops adding once an audio holds [m]. *)
  scores : 'd Nx.float32_t;
  count : 'd Nx.int64_t;
  caches : ('a, 'd) Kaun.Attention.Cache.t list;
}

module State = struct
  type ('a, 'd) t = ('a, 'd) state

  let walk c s =
    let open Nx.Ptree.Walk in
    let step = field c "step" tensor s.step in
    let sampled = field c "sampled" tensor s.sampled in
    let sums = field c "sums" tensor s.sums in
    let live = field c "live" tensor s.live in
    let table = field c "table" tensor s.table in
    let finished = field c "finished" tensor s.finished in
    let scores = field c "scores" tensor s.scores in
    let count = field c "count" tensor s.count in
    let caches = field c "caches" (list Kaun.Attention.Cache.walk) s.caches in
    { step; sampled; sums; live; table; finished; scores; count; caches }
end

let check_penalty fn o =
  match o.length_penalty with
  | Some a when not (a >= 0. && a <= 1.) ->
      invalid_argf "Whisper.Decoding.%s: length penalty %g outside [0, 1]" fn a
  | _ -> ()

let decode m t o search (starts : _ starts) features =
  let d = m.Model.dims in
  let n = Nx.dim 0 starts.tokens and w = Nx.dim 1 starts.tokens in
  check_penalty "decode" o;
  if Nx.dim 0 features <> n || Nx.shape starts.sot <> [| n |] then
    invalid_argf
      "Whisper.Decoding.decode: starts of shapes %s and %s for features of \
       shape %s"
      (shape_string (Nx.shape starts.tokens))
      (shape_string (Nx.shape starts.sot))
      (shape_string (Nx.shape features));
  let g, mc =
    match search with
    | Greedy -> (1, 0)
    | Sample { best_of; _ } -> (best_of, 0)
    | Beam { size; patience } ->
        let mc = candidates size patience in
        if size >= 1 && mc < 1 then
          invalid_argf
            "Whisper.Decoding.decode: beam of %d at patience %g keeps %d \
             candidates"
            size patience mc;
        (size, mc)
  in
  if g < 1 then invalid_argf "Whisper.Decoding.decode: %d lanes per audio" g;
  let audio = Model.audio m features in
  let at = Nx.beside starts.tokens in
  let r = rules t o at in
  let lanes = n * g and ctx = d.n_text_ctx in
  let len = ctx / 2 in
  let per = w + (g * len) in
  let eot = Int64.of_int r.eot in
  let ( &&& ) = Nx.logical_and in
  let int x = Nx.scalar_at at Nx.int64 (Int64.of_int x) in
  let host_lanes f =
    Nx.place at
      (Nx.init Nx.int64 [| lanes; 1 |] (fun i -> Int64.of_int (f i.(0))))
  in
  (* Each lane's audio, its rank in its group, and its first slot. *)
  let audio_of = host_lanes (fun l -> l / g) in
  let base = host_lanes (fun l -> l / g * per) in
  let rank = host_lanes (fun l -> l mod g) in
  let group x =
    Nx.reshape [| lanes; 1 |]
      (Nx.take ~axis:0 (Nx.reshape [| lanes |] audio_of) x)
  in
  (* Initial tokens and the sampling limit: a lane samples at most [len] tokens,
     and stops once its tokens number more than [ctx]. *)
  let first = Nx.count ~axes:[ 1 ] (Nx.greater_equal_s starts.tokens 0L) in
  let lane_first = group first in
  let limit =
    Nx.minimum_s (Nx.sub (int (ctx + 1)) lane_first) (Int64.of_int len)
  in
  (* The table: a lane's initial tokens at its audio's first slots, then a fresh
     slot per step. *)
  let cols = Nx.reshape [| 1; ctx |] (Nx.arange_at at Nx.int64 0 ctx 1) in
  let own =
    let sampled_col = Nx.sub cols lane_first in
    Nx.where (Nx.less cols lane_first) (Nx.add base cols)
      (Nx.where
         (Nx.less sampled_col (int len))
         (Nx.add
            (Nx.add base (int w))
            (Nx.add (Nx.mul sampled_col (int g)) rank))
         (int (-1)))
  in
  (* The prefill: each audio's initial tokens once, through its group's first
     lane. *)
  let caches = Model.cache m ~slots:(n * per) at in
  let h, caches =
    let valid = Nx.greater_equal_s starts.tokens 0L in
    let wcols = Nx.reshape [| 1; w |] (Nx.arange_at at Nx.int64 0 w 1) in
    let pos =
      Nx.where valid
        (Nx.add wcols (Nx.sub (Nx.reshape [| n; 1 |] first) (int w)))
        (int (-1))
    in
    let row =
      Nx.place at (Nx.init Nx.int64 [| n |] (fun i -> Int64.of_int (i.(0) * g)))
    in
    let index = Kaun.Cache_index.make ~row ~pos ~table:own () in
    Model.decode m audio caches index (Nx.where valid starts.tokens (int 0))
  in
  let width = Nx.dim 2 h in
  let no_speech =
    let at_sot =
      Nx.take_along_axis ~axis:1 (Nx.reshape [| n; 1; 1 |] starts.sot) h
    in
    Nx.slice
      [ A; I (Tokenizer.special t No_speech) ]
      (Nx.softmax ~axes:[ 1 ]
         (Model.logits m (Nx.reshape [| n; width |] at_sot)))
  in
  let logits =
    let last = Model.logits m (Nx.slice [ A; I (w - 1) ] h) in
    Nx.reshape [| lanes; r.vocab |]
      (Nx.broadcast_to [| n; g; r.vocab |]
         (Nx.reshape [| n; 1; r.vocab |] last))
  in
  let flat x = Nx.reshape [| lanes |] x in
  let column x = Nx.reshape [| lanes; 1 |] x in
  (* Greedy and sampling: upstream's GreedyDecoder. A lane that has stopped
     keeps its history and its sum. *)
  let independent st f next =
    let lp =
      Nx.take_along_axis ~axis:1 (column next) (Nx.log_softmax ~axes:[ 1 ] f)
    in
    let live = column st.live in
    let sums = Nx.where st.live (Nx.add st.sums (flat lp)) st.sums in
    let at_col =
      Nx.where live (Nx.broadcast_to [| lanes; 1 |] st.step) (int len)
    in
    let sampled = Nx.scatter ~axis:1 at_col (column next) st.sampled in
    let live =
      st.live &&& Nx.not_equal_s next eot
      &&& flat (Nx.less (Nx.add_s st.step 1L) limit)
    in
    { st with step = Nx.add_s st.step 1L; sampled; sums; live }
  in
  (* Beam search: upstream's BeamSearchDecoder.update, per audio over the
     candidates its beams propose, [g + 1] each, in the order upstream's stable
     sort of its dict gives them. Beams [1..g-1] start at [-∞], so the first
     step chooses among beam 0's candidates alone, as upstream's dict, which
     merges the [g] equal beams' candidates, does. *)
  let beam st f =
    let k = g + 1 and c = g * (g + 1) in
    let lp = Nx.log_softmax ~axes:[ 1 ] f in
    let v, tok = Nx.top_k ~k lp in
    let cand = Nx.reshape [| n; c |] (Nx.add (column st.sums) v) in
    let tok = Nx.reshape [| n; c |] tok in
    let order = Nx.argsort ~descending:true ~axis:1 cand in
    let score = Nx.take_along_axis ~axis:1 order cand in
    let tok = Nx.take_along_axis ~axis:1 order tok in
    (* The lane each candidate extends. *)
    let src =
      Nx.add
        (Nx.div order (int k))
        (Nx.reshape [| n; 1 |] (Nx.arange_at at Nx.int64 0 lanes g))
    in
    let ends = Nx.equal_s tok eot in
    let continues = Nx.cast Nx.int64 (Nx.logical_not ends) in
    let seen = Nx.cumsum ~axis:1 continues in
    let keep = Nx.logical_not ends &&& Nx.less_equal_s seen (Int64.of_int g) in
    let fin = ends &&& Nx.less_s seen (Int64.of_int g) in
    (* The first [g] kept, in order: a stable sort puts them first. *)
    let pick =
      Nx.slice
        [ A; R (0, g) ]
        (Nx.argsort ~axis:1 (Nx.cast Nx.int64 (Nx.logical_not keep)))
    in
    let next = flat (Nx.take_along_axis ~axis:1 pick tok) in
    let sums = flat (Nx.take_along_axis ~axis:1 pick score) in
    let source = flat (Nx.take_along_axis ~axis:1 pick src) in
    let active = st.live in
    let active_n =
      Nx.reshape [| n; 1 |] (Nx.slice [ Rs (0, lanes, g) ] active)
    in
    let at_col = Nx.broadcast_to [| lanes; 1 |] st.step in
    let sampled =
      Nx.scatter ~axis:1 at_col (column next)
        (Nx.take ~axis:0 source st.sampled)
    in
    let table =
      Nx.where
        (Nx.less cols (Nx.add lane_first at_col))
        (Nx.take ~axis:0 source st.table)
        own
    in
    (* Newly finished candidates, in rank order, at the next free slots. *)
    let count = Nx.reshape [| n; 1 |] st.count in
    let slot =
      Nx.where (fin &&& active_n)
        (Nx.add count (Nx.sub_s (Nx.cumsum ~axis:1 (Nx.cast Nx.int64 fin)) 1L))
        (int mc)
    in
    let rows =
      let h = Nx.take ~axis:0 (Nx.reshape [| n * c |] src) st.sampled in
      let h =
        Nx.scatter ~axis:1
          (Nx.broadcast_to [| n * c; 1 |] st.step)
          (Nx.full_at at Nx.int64 [| n * c; 1 |] eot)
          h
      in
      Nx.reshape [| n; c; len |] h
    in
    let finished =
      Nx.scatter ~axis:1
        (Nx.broadcast_to [| n; c; len |] (Nx.reshape [| n; c; 1 |] slot))
        rows st.finished
    in
    let scores = Nx.scatter ~axis:1 slot score st.scores in
    let count =
      Nx.where active_n
        (Nx.minimum_s
           (Nx.add count (Nx.count ~axes:[ 1 ] ~keepdims:true fin))
           (Int64.of_int mc))
        count
    in
    let count = Nx.reshape [| n |] count in
    let keep_old x y = Nx.where (column active) x y in
    let live =
      active
      &&& flat (group (Nx.less_s count (Int64.of_int mc)))
      &&& flat (Nx.less (Nx.add_s st.step 1L) limit)
    in
    {
      st with
      step = Nx.add_s st.step 1L;
      sampled = keep_old sampled st.sampled;
      sums = Nx.where active sums st.sums;
      table = keep_old table st.table;
      finished;
      scores;
      count;
      live;
    }
  in
  let choose st logits =
    let f = apply r ~sampled:st.sampled logits in
    match search with
    | Greedy -> independent st f (Nx.argmax ~axis:1 f)
    | Sample { temperature; key; _ } ->
        let key = Nx.Rng.fold_in_tensor key st.step in
        independent st f
          (Nx.Rng.categorical ~key ~axis:1 (Nx.div f temperature))
    | Beam _ -> beam st f
  in
  let sums =
    match search with
    | Beam _ ->
        Nx.place at
          (Nx.init Nx.float32 [| lanes |] (fun i ->
               if i.(0) mod g = 0 then 0. else Float.neg_infinity))
    | Greedy | Sample _ -> Nx.zeros_at at Nx.float32 [| lanes |]
  in
  let init =
    {
      step = int 0;
      sampled = Nx.full_at at Nx.int64 [| lanes; len |] (-1L);
      sums;
      live = Nx.full_at at Nx.bool [| lanes |] true;
      table = own;
      finished = Nx.full_at at Nx.int64 [| n; mc; len |] (-1L);
      scores = Nx.full_at at Nx.float32 [| n; mc |] Float.neg_infinity;
      count = Nx.zeros_at at Nx.int64 [| n |];
      caches;
    }
  in
  (* A step: decode each live lane's last token at its position, then choose. *)
  let step st =
    let last = Nx.sub_s st.step 1L in
    let ids =
      Nx.take_along_axis ~axis:1
        (Nx.broadcast_to [| lanes; 1 |] last)
        st.sampled
    in
    let live = column st.live in
    let pos = Nx.where live (Nx.add lane_first last) (int (-1)) in
    let index = Kaun.Cache_index.make ~pos ~table:st.table () in
    let h, caches =
      Model.decode m audio st.caches index (Nx.where live ids (int 0))
    in
    let logits = Model.logits m (Nx.reshape [| lanes; width |] h) in
    choose { st with caches } logits
  in
  let st =
    Rune.iterate
      (Nx.Ptree.instantiate (module State))
      ~max:(len - 1)
      ~until:(fun st -> Nx.logical_not (Nx.any st.live))
      step (choose init logits)
  in
  {
    tokens = Nx.reshape [| n; g; len |] st.sampled;
    sums = Nx.reshape [| n; g |] st.sums;
    finished = st.finished;
    scores = st.scores;
    count = st.count;
    no_speech;
  }

(* Results

   whisper/decoding.py, BeamSearchDecoder.finalize, MaximumLikelihoodRanker and
   DecodingTask.run. Upstream divides by a zero length, which raises; an empty
   candidate scores [-∞] here. *)

type result = {
  tokens : int array;
  text : string;
  avg_logprob : float;
  no_speech_prob : float;
}

let results t o (out : Nx.host outcome) =
  check_penalty "results" o;
  let n = Nx.dim 0 out.tokens and g = Nx.dim 1 out.tokens in
  let len = Nx.dim 2 out.tokens and mc = Nx.dim 1 out.finished in
  let eot = Tokenizer.special t End_of_text in
  let tokens = Nx.to_array out.tokens and sums = Nx.to_array out.sums in
  let finished = Nx.to_array out.finished in
  let scores = Nx.to_array out.scores and count = Nx.to_array out.count in
  let no_speech = Nx.to_array out.no_speech in
  (* A row's tokens before the end of text. *)
  let row a k =
    let rec stop i =
      if i = len then i
      else
        let x = Int64.to_int a.((k * len) + i) in
        if x = eot || x < 0 then i else stop (i + 1)
    in
    Array.init (stop 0) (fun i -> Int64.to_int a.((k * len) + i))
  in
  let score (toks, sum) =
    let length = Array.length toks in
    match o.length_penalty with
    | None -> if length = 0 then Float.neg_infinity else sum /. float length
    | Some a -> sum /. Float.pow ((5. +. float length) /. 6.) a
  in
  Array.init n (fun i ->
      let lanes =
        List.init g (fun j -> (row tokens ((i * g) + j), sums.((i * g) + j)))
      in
      let candidates =
        if mc = 0 then lanes
        else
          let held = Int64.to_int count.(i) in
          let done_ =
            List.init held (fun j ->
                (row finished ((i * mc) + j), scores.((i * mc) + j)))
          in
          if held >= g then done_
          else
            let by_sum =
              List.stable_sort (fun (_, a) (_, b) -> Float.compare b a) lanes
            in
            done_ @ List.filteri (fun j _ -> j < g - held) by_sum
      in
      let best =
        List.fold_left
          (fun best c ->
            match best with
            | Some b when score b >= score c -> best
            | _ -> Some c)
          None candidates
      in
      let toks, sum = Option.get best in
      {
        tokens = toks;
        text = Space.strip (Tokenizer.decode t toks);
        avg_logprob = sum /. float (Array.length toks + 1);
        no_speech_prob = no_speech.(i);
      })

(* Languages

   whisper/decoding.py, detect_language: the logits after
   [<|startoftranscript|>] alone, every token but the languages' at [-∞], under
   a softmax. The languages' tokens are numbered in order, so they are one slice
   of the logits. *)

let languages m t features =
  if not (Model.multilingual m.Model.dims) then
    invalid_arg "Whisper.Decoding.languages: the model is English-only";
  let audio = Model.audio m features in
  let n = Nx.dim 0 features in
  let at = Nx.beside features in
  let caches = Model.cache m ~slots:n at in
  let index =
    Kaun.Cache_index.place at
      (Kaun.Cache_index.rows ~context:1 (Array.make n 1))
  in
  let sot = Tokenizer.special t Start_of_transcript in
  let h, _ =
    Model.decode m audio caches index
      (Nx.full_at at Nx.int64 [| n; 1 |] (Int64.of_int sot))
  in
  let logits = Model.logits m (Nx.reshape [| n; Nx.dim 2 h |] h) in
  let codes = Tokenizer.languages t in
  let first = Tokenizer.special t (Language codes.(0)) in
  let k = Array.length codes in
  Nx.softmax ~axes:[ 1 ] (Nx.slice [ A; R (first, first + k) ] logits)

(* Compression ratio

   whisper/utils.py, compression_ratio: the UTF-8 bytes over the length of
   zlib.compress's stream, at zlib's default level. *)

let zlib_level = 6

let compression_ratio s =
  float (String.length s)
  /. float (String.length (Lz.Zlib.encode_string ~level:zlib_level s))
