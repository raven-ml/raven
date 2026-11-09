(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Normalizer = Normalizer
module Pre_tokenizer = Pre_tokenizer
module Post_processor = Post_processor
module Decoder = Decoder
module Encoding = Encoding

let strf = Printf.sprintf

(* Error messages *)

let err_pair_no_post = "pair sequences require a configured post-processor"
let err_no_pad_token = "padding requested but no pad token configured"
let err_pad_not_in_vocab tok = strf "pad token '%s' not in vocabulary" tok
let err_export_tiktoken = "only supported for BPE models"
let err_infer_type = "unable to infer model type from JSON"

(* Types *)

type direction = [ `Left | `Right ]

type added_token = {
  content : string;
  special : bool;
  single_word : bool;
  lstrip : bool;
  rstrip : bool;
  normalized : bool;
}

type pad_length = [ `Batch_longest | `Fixed of int | `To_multiple of int ]

type padding = {
  length : pad_length;
  direction : direction;
  pad_id : int option;
  pad_type_id : int option;
  pad_token : string option;
}

type truncation = { max_length : int; stride : int; direction : direction }
type data = [ `Files of string list | `Seq of string Seq.t ]
type sequence = { text : string; pair : string option }

type algorithm =
  | Alg_bpe of Bpe.t
  | Alg_wordpiece of Wordpiece.t
  | Alg_wordlevel of Word_level.t
  | Alg_unigram of Unigram.t
  | Alg_chars of Chars.t

(* The rewrite a text goes through before a walker sees it, as the normalizer
   that performs it, so that its alignment comes from the same place as the
   pipeline normalizer's; [None] when it leaves the text as it is. *)
type rewrite = first:bool Lazy.t -> string -> Normalizer.t option

(* How the pipeline cuts a stretch of normalized text into pretokens. *)
type cut =
  | Whole  (** No pre-tokenizer: the stretch is one pretoken. *)
  | Walk of Pre_tokenizer.t
      (** Byte ranges of the stretch as it is, through {!Pre_tokenizer.fill},
          which is what spares the walked path a substring. *)
  | Rewritten of { pre : Pre_tokenizer.t; rewrite : rewrite }
      (** Byte ranges of the stretch once rewritten. *)
  | Segmented of {
      outer : Pre_tokenizer.t;
      rewrite : rewrite;
      inner : Pre_tokenizer.t;
    }
      (** [outer] cuts the stretch into segments, and [inner] walks each once
          rewritten. *)
  | Pieces of Pre_tokenizer.t
      (** Strings, through {!Pre_tokenizer.pre_tokenize}: the pretokens are not
          ranges of the text they were cut from. *)

(* The ids buffers the batches of a tokenizer borrow, kept between batches and
   collected with the tokenizer: [free] holds them, [kept] counts them. *)
type batch_buffers = { free : Bpe.sink list Atomic.t; kept : int Atomic.t }

type t = {
  algorithm : algorithm;
  normalizer : Normalizer.t option;
  (* The normalizer encoding runs: [normalizer], or its spaced form in front of
     a spaced model (see [spaces]), which [spaced_text] says. *)
  normalize : Normalizer.t option;
  spaced_text : bool;
  pre_tokenizer : Pre_tokenizer.t option;
  cut : cut;
  (* The walker documents walk and encode with through the fused C kernel, if
     they do: a walked byte-level pre-tokenizer the kernel has a walker for,
     over a BPE model it covers, decided once here. *)
  fused : Pre_tokenizer.walker option;
  post_processor : Post_processor.t option;
  decoder : Decoder.t option;
  added : Added_tokens.t;
  (* An identifier to its token string and to the source bytes it accounts for,
     over the model vocabulary and the added tokens together. *)
  token_table : string array;
  len_table : int array;
  bos_token : string option;
  eos_token : string option;
  pad_token : string option;
  pad_id : int option;
  pad_type_id : int;
  unk_token : string option;
  batch_buffers : batch_buffers;
}

let added_token ?(special = true) ?(single_word = false) ?(lstrip = false)
    ?(rstrip = false) ?normalized content =
  let normalized = Option.value normalized ~default:(not special) in
  { content; special; single_word; lstrip; rstrip; normalized }

let padding ?(direction = `Right) ?pad_id ?pad_type_id ?pad_token length =
  { length; direction; pad_id; pad_type_id; pad_token }

let truncation ?(stride = 0) ?(direction = `Right) max_length =
  if stride < 0 || (max_length > 0 && stride >= max_length) then
    invalid_arg
      "truncation: stride must be non-negative and less than max_length";
  { max_length; stride; direction }

(* Algorithm dispatch *)

let alg_token_to_id algorithm token =
  match algorithm with
  | Alg_bpe m -> Bpe.token_to_id m token
  | Alg_wordpiece m -> Wordpiece.token_to_id m token
  | Alg_wordlevel m -> Word_level.token_to_id m token
  | Alg_unigram m -> Unigram.token_to_id m token
  | Alg_chars m -> Chars.token_to_id m token

let alg_id_to_token algorithm id =
  match algorithm with
  | Alg_bpe m -> Bpe.id_to_token m id
  | Alg_wordpiece m -> Wordpiece.id_to_token m id
  | Alg_wordlevel m -> Word_level.id_to_token m id
  | Alg_unigram m -> Unigram.id_to_token m id
  | Alg_chars m -> Chars.id_to_token m id

let alg_vocab algorithm =
  match algorithm with
  | Alg_bpe m -> Bpe.get_vocab m
  | Alg_wordpiece m -> Wordpiece.get_vocab m
  | Alg_wordlevel m -> Word_level.get_vocab m
  | Alg_unigram m ->
      Unigram.get_vocab m |> List.mapi (fun i (token, _) -> (token, i))
  | Alg_chars m -> Chars.get_vocab m

let alg_vocab_size algorithm =
  match algorithm with
  | Alg_bpe m -> Bpe.get_vocab_size m
  | Alg_wordpiece m -> Wordpiece.get_vocab_size m
  | Alg_wordlevel m -> Word_level.get_vocab_size m
  | Alg_unigram m -> Unigram.get_vocab_size m
  | Alg_chars m -> Chars.get_vocab_size m

let alg_save algorithm ~folder ?prefix () =
  match algorithm with
  | Alg_bpe m ->
      Bpe.save m ~path:folder ?name:prefix ();
      let name base ext =
        match prefix with
        | Some n -> Filename.concat folder (strf "%s-%s.%s" n base ext)
        | None -> Filename.concat folder (strf "%s.%s" base ext)
      in
      [ name "vocab" "json"; name "merges" "txt" ]
  | Alg_wordpiece m -> [ Wordpiece.save m ~path:folder ?name:prefix () ]
  | Alg_wordlevel m -> Word_level.save m ~folder ()
  | Alg_unigram m -> Unigram.save m ~folder ()
  | Alg_chars m -> Chars.save m ~folder ()

let alg_token_table = function
  | Alg_bpe m -> Bpe.token_table m
  | Alg_wordpiece m -> Wordpiece.token_table m
  | Alg_wordlevel m -> Word_level.token_table m
  | Alg_unigram m -> Unigram.token_table m
  | Alg_chars m -> Chars.token_table m

let alg_len_table = function
  | Alg_bpe m -> Bpe.len_table m
  | Alg_wordpiece m -> Wordpiece.len_table m
  | Alg_wordlevel m -> Word_level.len_table m
  | Alg_unigram m -> Unigram.len_table m
  | Alg_chars m -> Chars.len_table m

(* The model's span encoder for a document. A BPE model's merge buffers and
   pretoken cache are claimed once for the document rather than once per span,
   and the closure keeps the dispatch out of the span loop. A fused BPE also
   hands out its walking encoder, which cuts and encodes in one pass. Handed a
   batch sink, a BPE's encoders write the kernels' ids into it and treat the ids
   buffer they are given as the spill; the others keep writing to that buffer,
   which the batch driver flushes at document end either way. A WordPiece or
   Unigram model's cached encoder is claimed the same way. *)
let with_span_encoder t ?ids32 f =
  match t.algorithm with
  | Alg_bpe m ->
      Bpe.with_state m (fun st ->
          match ids32 with
          | Some sink ->
              (* The walking encoder exists for a fused model alone: an
                 unfused one allocates no closure for it per document. *)
              let walk =
                match t.fused with
                | None -> None
                | Some walker ->
                    Some
                      (fun spill ~opaque ~marks:_ _ text ~pos ~stop ->
                        Bpe.encode_walk_ids32 m st ~walker sink spill ~opaque
                          text ~pos ~stop)
              in
              f (Bpe.encode_into_ids32 m st sink) walk
          | None ->
              f (Bpe.encode_into m st)
                (Option.map
                   (fun walker -> Bpe.encode_walk m st ~walker)
                   t.fused))
  | Alg_wordpiece m -> Wordpiece.with_encoder m (fun encoder -> f encoder None)
  | Alg_wordlevel m -> f (Word_level.encode_into m) None
  | Alg_unigram m -> Unigram.with_encoder m (fun encoder -> f encoder None)
  | Alg_chars m -> f (Chars.encode_into m) None

let alg_name = function
  | Alg_bpe _ -> "BPE"
  | Alg_wordpiece _ -> "WordPiece"
  | Alg_wordlevel _ -> "WordLevel"
  | Alg_unigram _ -> "Unigram"
  | Alg_chars _ -> "Chars"

let vocab_to_hashtbl vocab =
  let tbl = Hashtbl.create (List.length vocab) in
  List.iter (fun (token, id) -> Hashtbl.replace tbl token id) vocab;
  tbl

(* Special tokens *)

let dedup_by key items =
  let seen = Hashtbl.create 16 in
  let acc = ref [] in
  List.iter
    (fun item ->
      let k = key item in
      if not (Hashtbl.mem seen k) then (
        Hashtbl.replace seen k ();
        acc := item :: !acc))
    items;
  List.rev !acc

(* A token named for the beginning, end or padding role is a special token in
   its own right. The unknown token is not: the model emits it, and nothing
   matches it in the input. A role whose content is already among the given
   tokens keeps the flags given there. *)
let role_tokens ~bos_token ~eos_token ~pad_token given =
  let named content =
    List.exists (fun (a : added_token) -> a.content = content) given
  in
  List.filter_map Fun.id [ bos_token; eos_token; pad_token ]
  |> List.filter (fun content -> not (named content))
  |> List.map added_token

(* Identifiers follow HuggingFace: a token the model holds keeps the model's
   identifier, the others are numbered from the end of the model vocabulary.
   Repeated content is one token and takes one identifier. *)
let added_tokens_of algorithm tokens =
  let vocab_size = alg_vocab_size algorithm in
  let assigned = Hashtbl.create 16 in
  let number (a : added_token) highest =
    match Hashtbl.find_opt assigned a.content with
    | Some id -> id
    | None -> (
        match alg_token_to_id algorithm a.content with
        | Some id -> id
        | None -> (
            match highest with
            | Some id when id >= vocab_size || vocab_size = 0 -> id + 1
            | _ -> vocab_size))
  in
  let tokens, _ =
    List.fold_left
      (fun (acc, highest) (a : added_token) ->
        let id = number a highest in
        Hashtbl.replace assigned a.content id;
        let token =
          {
            Added_tokens.content = a.content;
            id;
            special = a.special;
            single_word = a.single_word;
            lstrip = a.lstrip;
            rstrip = a.rstrip;
            normalized = a.normalized;
          }
        in
        let highest =
          match highest with
          | Some other -> Some (max other id)
          | None -> Some id
        in
        (token :: acc, highest))
      ([], None) tokens
  in
  List.rev tokens

(* Construction *)

(* Spans index the text as it is, so a walked byte-level pre-tokenizer hands the
   model the raw bytes of a pretoken rather than their encoded form; only a BPE
   model can match its vocabulary against those (it is flipped below), so any
   other model behind one is handed pieces, which come encoded. A rewrite in
   front of a spaced model writes the spaces as they are. *)
let cut_of ~algorithm ~spaced = function
  | None -> Whole
  | Some pre -> (
      let matches_bytes =
        match algorithm with Alg_bpe _ -> true | _ -> false
      in
      match Pre_tokenizer.plan pre with
      | (Pre_tokenizer.Walk _ | Pre_tokenizer.Segmented _)
        when Pre_tokenizer.encodes_bytes pre && not matches_bytes ->
          Pieces pre
      | Pre_tokenizer.Walk { rewrite = Verbatim; _ } -> Walk pre
      | Pre_tokenizer.Walk { rewrite; _ } ->
          Rewritten { pre; rewrite = Pre_tokenizer.rewriter ~spaced rewrite }
      | Pre_tokenizer.Segmented { outer; rewrite; inner; _ } ->
          Segmented
            { outer; rewrite = Pre_tokenizer.rewriter ~spaced rewrite; inner }
      | Pre_tokenizer.Pieces -> Pieces pre)

(* An added token's identifier may lie past the model vocabulary. How many
   source bytes it accounts for is a property of the text it matched rather than
   of the identifier, so it reads as zero: the literal frame it stands in places
   it by its span. *)
let id_tables algorithm added =
  let tokens = alg_token_table algorithm and lens = alg_len_table algorithm in
  let size =
    List.fold_left
      (fun acc (tok : Added_tokens.token) -> max acc (tok.id + 1))
      (Array.length tokens)
      (Added_tokens.tokens added)
  in
  if size = Array.length tokens then (tokens, lens)
  else begin
    let token_table = Array.make size "" in
    Array.blit tokens 0 token_table 0 (Array.length tokens);
    let len_table = Array.make size 0 in
    Array.blit lens 0 len_table 0 (Array.length lens);
    List.iter
      (fun (tok : Added_tokens.token) ->
        if tok.id >= Array.length tokens then
          token_table.(tok.id) <- tok.content)
      (Added_tokens.tokens added);
    (token_table, len_table)
  end

(* Whether a BPE model is to match raw bytes: behind a walked byte-level
   pre-tokenizer, which hands it the bytes of a pretoken rather than their
   encoded form. The flip is derived from the pipeline and is never a knob of
   its own; a pre-tokenizer that hands back pieces has already encoded them, and
   a model flipped to raw bytes would encode them twice. *)
let byte_level_pipeline pre =
  match pre with
  | None -> false
  | Some pre -> (
      Pre_tokenizer.encodes_bytes pre
      &&
      match Pre_tokenizer.plan pre with
      | Pre_tokenizer.Walk _ | Pre_tokenizer.Segmented _ -> true
      | Pre_tokenizer.Pieces -> false)

(* A SentencePiece pipeline writes ▁ for every space just before the model: in
   the last stages of its normalizer when no pre-tokenizer follows
   ([Normalizer.spaced]), or in a metaspace that does not split
   ([Pre_tokenizer.marks_spaces]). A BPE model can read the spaces themselves
   instead ([Bpe.create]'s [spaced]): the pipeline then writes its text with
   every ▁ a space, which takes no pass to mark and two bytes less for every
   space. Like the byte-level flip, this is derived from the pipeline and is
   never a knob of its own. A normalized added token that takes in the white
   space beside it would take in those spaces too, so its pipeline is left as
   it is. *)
let spaces ~normalizer ~pre ~(added_tokens : added_token list) =
  match pre with
  | Some pre -> Pre_tokenizer.marks_spaces pre
  | None ->
      Option.is_some (Option.bind normalizer Normalizer.spaced)
      && List.for_all
           (fun (a : added_token) ->
             not (a.normalized && (a.lstrip || a.rstrip)))
           added_tokens

(* What a spaced normalizer leaves out. *)
let marking = Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81"

(* Whether a BPE model of [vocab] in a pipeline that [spaces] is made spaced. *)
let spaced_model ~spaces ~byte_level ?continuing_subword_prefix
    ?end_of_word_suffix vocab =
  let none = function None | Some "" -> true | Some _ -> false in
  spaces && (not byte_level)
  && none continuing_subword_prefix
  && none end_of_word_suffix && Bpe.spaceable vocab

(* A model built for another pipeline — one just trained, say — is rebuilt on
   the same vocabulary and merges. The constructors that hold those already flip
   the model as they build it, so this only ever fires for the others. *)
let fit_to_pipeline ~spaces pre algorithm =
  match algorithm with
  | Alg_bpe model ->
      let byte_level = byte_level_pipeline pre || Bpe.get_byte_level model in
      let prefix = Bpe.get_continuing_subword_prefix model in
      let suffix = Bpe.get_end_of_word_suffix model in
      let spaced =
        if Bpe.get_spaced model then spaces && not byte_level
        else
          spaces
          && spaced_model ~spaces ~byte_level ?continuing_subword_prefix:prefix
               ?end_of_word_suffix:suffix
               (vocab_to_hashtbl (Bpe.get_vocab model))
      in
      if
        byte_level = Bpe.get_byte_level model
        && spaced = Bpe.get_spaced model
      then algorithm
      else
        Alg_bpe
          (Bpe.create
             ~vocab:(vocab_to_hashtbl (Bpe.get_vocab model))
             ~merges:(Bpe.get_merges model) ~byte_level ~spaced
             ?continuing_subword_prefix:(if byte_level then None else prefix)
             ?end_of_word_suffix:(if byte_level then None else suffix)
             ~cache_capacity:(Bpe.get_cache_capacity model)
             ?dropout:(Bpe.get_dropout model)
             ?unk_token:(Bpe.get_unk_token model)
             ~fuse_unk:(Bpe.get_fuse_unk model)
             ~byte_fallback:(Bpe.get_byte_fallback model)
             ~ignore_merges:(Bpe.get_ignore_merges model)
             ())
  | _ -> algorithm

let create ?normalizer ?pre ?post ?decoder ?(added_tokens = []) ?bos_token
    ?eos_token ?pad_token ?unk_token algorithm =
  let given =
    List.filter (fun (a : added_token) -> a.content <> "") added_tokens
  in
  let requested = given @ role_tokens ~bos_token ~eos_token ~pad_token given in
  let algorithm =
    fit_to_pipeline
      ~spaces:(spaces ~normalizer ~pre ~added_tokens:requested)
      pre algorithm
  in
  let spaced =
    match algorithm with Alg_bpe m -> Bpe.get_spaced m | _ -> false
  in
  let spaced_text = spaced && Option.is_none pre in
  let normalize =
    if spaced_text then Option.bind normalizer Normalizer.spaced else normalizer
  in
  let added =
    Added_tokens.make
      ~normalize:
        (match normalize with Some n -> Normalizer.apply n | None -> Fun.id)
      (added_tokens_of algorithm requested)
  in
  let token_id token =
    match Added_tokens.token_to_id added token with
    | Some _ as id -> id
    | None -> alg_token_to_id algorithm token
  in
  let pad_id = Option.bind pad_token token_id in
  let token_table, len_table = id_tables algorithm added in
  let cut = cut_of ~algorithm ~spaced pre in
  let fused =
    match (cut, algorithm) with
    | Walk pre, Alg_bpe m when Bpe.fused m ->
        Pre_tokenizer.byte_level_walker pre
    | _ -> None
  in
  {
    algorithm;
    normalizer;
    normalize;
    spaced_text;
    pre_tokenizer = pre;
    cut;
    fused;
    post_processor = post;
    decoder;
    added;
    token_table;
    len_table;
    bos_token;
    eos_token;
    pad_token;
    pad_id;
    pad_type_id = 0;
    unk_token;
    batch_buffers = { free = Atomic.make []; kept = Atomic.make 0 };
  }

(* Accessors *)

let normalizer t = t.normalizer
let pre_tokenizer t = t.pre_tokenizer
let post_processor t = t.post_processor
let decoder t = t.decoder

let added_tokens t =
  Added_tokens.tokens t.added
  |> List.map (fun (tok : Added_tokens.token) ->
      {
        content = tok.content;
        special = tok.special;
        single_word = tok.single_word;
        lstrip = tok.lstrip;
        rstrip = tok.rstrip;
        normalized = tok.normalized;
      })

let bos_token t = t.bos_token
let eos_token t = t.eos_token
let pad_token t = t.pad_token
let unk_token t = t.unk_token

(* The added tokens that the model does not already hold. *)
let added_vocab t =
  Added_tokens.tokens t.added
  |> List.filter_map (fun (tok : Added_tokens.token) ->
      match alg_token_to_id t.algorithm tok.content with
      | Some _ -> None
      | None -> Some (tok.content, tok.id))

let vocab t = alg_vocab t.algorithm @ added_vocab t
let vocab_size t = alg_vocab_size t.algorithm + List.length (added_vocab t)

let token_to_id t token =
  match Added_tokens.token_to_id t.added token with
  | Some _ as id -> id
  | None -> alg_token_to_id t.algorithm token

let id_to_token t id =
  match Added_tokens.id_to_token t.added id with
  | Some _ as token -> token
  | None -> alg_id_to_token t.algorithm id

(* Algorithm constructors *)

let bpe ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token ?eos_token
    ?pad_token ?unk_token ?vocab ?merges ?cache_capacity ?dropout
    ?continuing_subword_prefix ?end_of_word_suffix ?fuse_unk ?byte_fallback
    ?ignore_merges () =
  let vocab_tbl =
    match vocab with None -> Hashtbl.create 100 | Some v -> vocab_to_hashtbl v
  in
  let byte_level = byte_level_pipeline pre in
  let spaced =
    spaced_model
      ~spaces:
        (spaces ~normalizer ~pre
           ~added_tokens:(Option.value added_tokens ~default:[]))
      ~byte_level ?continuing_subword_prefix ?end_of_word_suffix vocab_tbl
  in
  let algorithm =
    Alg_bpe
      (Bpe.create ~vocab:vocab_tbl
         ~merges:(Option.value merges ~default:[])
         ~byte_level ~spaced ?cache_capacity ?dropout ?unk_token
         ?continuing_subword_prefix ?end_of_word_suffix ?fuse_unk
         ?byte_fallback ?ignore_merges ())
  in
  create ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let wordpiece ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token
    ?eos_token ?pad_token ?unk_token ?vocab ?continuing_subword_prefix
    ?max_input_chars_per_word ?cache_capacity () =
  let vocab_tbl =
    match vocab with None -> Hashtbl.create 100 | Some v -> vocab_to_hashtbl v
  in
  let algorithm =
    Alg_wordpiece
      (Wordpiece.create ~vocab:vocab_tbl ?unk_token ?continuing_subword_prefix
         ?max_input_chars_per_word ?cache_capacity ())
  in
  create ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let word_level ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token
    ?eos_token ?pad_token ?unk_token ?vocab () =
  let pre =
    match pre with Some _ -> pre | None -> Some Pre_tokenizer.whitespace
  in
  let algorithm = Alg_wordlevel (Word_level.create ?vocab ?unk_token ()) in
  create ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let unigram ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token ?eos_token
    ?pad_token ?unk_token ?vocab ?unk_id ?byte_fallback ?cache_capacity () =
  let vocab = Option.value vocab ~default:[] in
  let unk_id =
    match unk_id with
    | Some _ -> unk_id
    | None ->
        Option.bind unk_token (fun unk ->
            List.find_index (fun (token, _) -> token = unk) vocab)
  in
  let algorithm =
    Alg_unigram (Unigram.create ?unk_id ?byte_fallback ?cache_capacity vocab)
  in
  create ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let chars ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token ?eos_token
    ?pad_token ?unk_token () =
  let algorithm = Alg_chars (Chars.create ()) in
  create ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let from_model_file ~vocab ?merges ?normalizer ?pre ?post ?decoder ?added_tokens
    ?bos_token ?eos_token ?pad_token ?unk_token () =
  let algorithm =
    match merges with
    | Some merges_file ->
        Alg_bpe
          (Bpe.from_files ~byte_level:(byte_level_pipeline pre)
             ~vocab_file:vocab ~merges_file ())
    | None -> Alg_wordpiece (Wordpiece.from_file ~vocab_file:vocab)
  in
  create ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let add_tokens t tokens =
  create ?normalizer:t.normalizer ?pre:t.pre_tokenizer ?post:t.post_processor
    ?decoder:t.decoder
    ~added_tokens:(added_tokens t @ tokens)
    ?bos_token:t.bos_token ?eos_token:t.eos_token ?pad_token:t.pad_token
    ?unk_token:t.unk_token t.algorithm

(* Encoding *)

let slice text start stop =
  if start = 0 && stop = String.length text then text
  else String.sub text start (stop - start)

(* Whether [piece] is the bytes of [text] at [at]: what tells a pre-tokenizer
   that only cut its input from one that handed back text of its own. *)
let is_slice text ~at piece =
  let len = String.length piece in
  at + len <= String.length text
  &&
  let rec same i =
    i = len
    || String.unsafe_get text (at + i) = String.unsafe_get piece i
       && same (i + 1)
  in
  same 0

(* The buffers a document is encoded through, one set per domain and claimed for
   the duration of the document: the ids, and the bounds and marks of the spans
   they came from. [batch] is the ids buffer a batch's chunks append to on this
   domain, borrowed from the tokenizer and held longer: by the batch whose
   serial number is in [holder], 0 when free, until its result is copied out. *)
type scratch = {
  mutable spans : Spans.t;
  mutable segments : Spans.t;
  ids : Ints.t;
  opaque : Ints.t;
  span_start : Ints.t;
  span_stop : Ints.t;
  marks : Ints.t;
  mutable batch : Bpe.sink;
  holder : int Atomic.t;
  busy : bool Atomic.t;
}

let span_chunk = 1024

(* The [batch] of a scratch no batch holds, which nothing writes to. *)
let no_batch =
  {
    Bpe.ba = Bigarray.Array1.create Bigarray.Int32 Bigarray.C_layout 0;
    used = 0;
  }

let new_scratch () =
  {
    spans = Spans.create ~capacity:span_chunk;
    segments = Spans.create ~capacity:span_chunk;
    ids = Ints.create ();
    opaque = Ints.create ();
    span_start = Ints.create ();
    span_stop = Ints.create ();
    marks = Ints.create ();
    batch = no_batch;
    holder = Atomic.make 0;
    busy = Atomic.make false;
  }

let scratch_key = Domain.DLS.new_key new_scratch

(* Threads of one domain share its buffers, each of which has a single writer,
   so the second to ask gets buffers of its own instead. The claim is given back
   even when [f] raises. *)
let with_scratch f =
  let sc = Domain.DLS.get scratch_key in
  let sc =
    if Atomic.compare_and_set sc.busy false true then sc else new_scratch ()
  in
  Ints.clear sc.ids;
  Ints.clear sc.span_start;
  Ints.clear sc.span_stop;
  Ints.clear sc.marks;
  match f sc with
  | value ->
      Atomic.set sc.busy false;
      value
  | exception e ->
      let backtrace = Printexc.get_raw_backtrace () in
      Atomic.set sc.busy false;
      Printexc.raise_with_backtrace e backtrace

let opening = Lazy.from_val true
let continuing = Lazy.from_val false

(* A document, encoded into [sc.ids]. [record] keeps what an encoding derives
   its tokens, offsets and word ids from: the bounds of every span, the id
   cursor after each of them, and the frames that place the spans back in
   [text]. A frame is the only thing that needs to know how the text was
   normalized, so that is handed back exactly when frames are being built, which
   is why a [Some] map and [record] say the same thing.

   Added tokens matched against raw text cut the input first, then, in each
   remaining stretch once normalized, the ones matched against normalized text.
   They never reach the walker or the model, so they cannot be split.

   [ids32] hands the span encoders a batch sink; [sc.ids] is then the spill of
   whatever OCaml encodes, added-token literals here included, and the caller
   flushes it at document end. Only the ids-only batch driver passes it, so
   [record] is false whenever it is given. *)
let encode_document t sc ?ids32 text ~record =
  Ints.clear sc.opaque;
  let frames = ref [] and frame_stops = ref [] in
  let raw_align = lazy (Run.Known (Normalizer.identity text)) in
  let close ~literal ~walked ~place ~rewrite ~base = function
    | None -> ()
    | Some align ->
        frames :=
          { Run.text = walked; literal; place; rewrite; align; base } :: !frames;
        frame_stops := Ints.length sc.marks :: !frame_stops
  in
  let verbatim = Run.Shifted 0 in
  (* The alignment of a normalization is left for the offsets to ask for, so a
     document normalizes once whether or not they are ever read. *)
  let apply n source =
    ( Normalizer.apply n source,
      if record then
        Some (Run.Deferred { normalizer = n; source; alignment = None })
      else None )
  in
  let normalized raw =
    match t.normalize with
    | Some n -> apply n raw
    | None ->
        ( raw,
          if record then Some (Run.Known (Normalizer.identity raw)) else None )
  in
  (* Whether the first stretch still opens the document once normalized by [n],
     which is what a marker prepended on [`First] asks: it does unless the
     normalizer took away the bytes that opened it, which only its alignment
     tells. Forced at most once per document, and only by such a marker. *)
  let opens_document n norm align raw =
    let align =
      match align with
      | Some align -> Run.force align
      | None -> snd (Normalizer.apply_aligned n raw)
    in
    String.length norm > 0
    && fst (Normalizer.original_span align ~start:0 ~stop:1) = 0
  in
  with_span_encoder t ?ids32 (fun encode encode_walk ->
      let[@inline] encode_span walked start finish =
        encode sc.ids ~opaque:sc.opaque walked ~pos:start ~len:(finish - start);
        if record then begin
          Ints.add sc.span_start start;
          Ints.add sc.span_stop finish;
          Ints.add sc.marks (Ints.length sc.ids)
        end
      in
      (* The spans of [pre] over [walked] between [pos] and [stop], each encoded
         as it is found, a buffer at a time; the buffer grows when a single span
         does not fit it. A fused encoder walks and encodes in one pass and
         appends the marks itself, hand-backs included, so on the record path
         they stay aligned with the span bounds read off the buffer; on the
         ids-only path they are dropped chunk by chunk. *)
      let walk pre walked ~pos ~stop =
        match encode_walk with
        | Some encode_walk ->
            let p = ref pos in
            while !p < stop do
              let spans = sc.spans in
              Spans.clear spans;
              if not record then Ints.clear sc.marks;
              let resume =
                encode_walk sc.ids ~opaque:sc.opaque ~marks:sc.marks spans
                  walked ~pos:!p ~stop
              in
              let n = Spans.count spans in
              if n = 0 && resume = !p then
                sc.spans <- Spans.create ~capacity:(2 * Spans.capacity spans)
              else begin
                if record then
                  for k = 0 to n - 1 do
                    Ints.add sc.span_start (Spans.start spans k);
                    Ints.add sc.span_stop (Spans.stop spans k)
                  done;
                p := resume
              end
            done
        | None ->
            let p = ref pos in
            while !p < stop do
              let spans = sc.spans in
              Spans.clear spans;
              let resume = Pre_tokenizer.fill pre walked ~pos:!p ~stop spans in
              let n = Spans.count spans in
              if n = 0 && resume = !p then
                sc.spans <- Spans.create ~capacity:(2 * Spans.capacity spans)
              else begin
                for k = 0 to n - 1 do
                  encode_span walked (Spans.start spans k) (Spans.stop spans k)
                done;
                p := resume
              end
            done
      in
      (* [norm] from [start] to [stop], rewritten and walked by [pre] as one
         frame placed at [start]. [first] says whether it opens the document. *)
      let rewritten pre rewrite norm align ~base ~start ~stop ~first =
        let source = slice norm start stop in
        let walked, alignment =
          match rewrite ~first source with
          | Some n -> apply n source
          | None -> (source, None)
        in
        walk pre walked ~pos:0 ~stop:(String.length walked);
        close ~literal:false ~walked ~place:(Run.Shifted start)
          ~rewrite:alignment ~base align
      in
      (* A stretch of normalized text with no added token left in it. [opens]
         says whether the stretch opens the document. *)
      let segment norm align ~base ~start ~stop ~opens =
        if start < stop then
          match t.cut with
          | Whole ->
              encode_span norm start stop;
              close ~literal:false ~walked:norm ~place:verbatim ~rewrite:None
                ~base align
          | Walk pre ->
              walk pre norm ~pos:start ~stop;
              close ~literal:false ~walked:norm ~place:verbatim ~rewrite:None
                ~base align
          | Rewritten { pre; rewrite } ->
              rewritten pre rewrite norm align ~base ~start ~stop
                ~first:(if start = 0 then opens else continuing)
          | Segmented { outer; rewrite; inner } ->
              let p = ref start in
              while !p < stop do
                let spans = sc.segments in
                Spans.clear spans;
                let resume =
                  Pre_tokenizer.fill outer norm ~pos:!p ~stop spans
                in
                let n = Spans.count spans in
                if n = 0 && resume = !p then
                  sc.segments <-
                    Spans.create ~capacity:(2 * Spans.capacity spans)
                else begin
                  for k = 0 to n - 1 do
                    let start = Spans.start spans k in
                    rewritten inner rewrite norm align ~base ~start
                      ~stop:(Spans.stop spans k)
                      ~first:(if start = 0 then opens else continuing)
                  done;
                  p := resume
                end
              done
          | Pieces pre ->
              let source = slice norm start stop in
              let length = String.length source in
              let first = if start = 0 then opens else continuing in
              List.iter
                (fun (piece, (at, finish)) ->
                  encode_span piece 0 (String.length piece);
                  (* A piece the pre-tokenizer cut from the text is that text
                     byte for byte, and its tokens are placed inside it; one it
                     rewrote can only be placed whole. The range it reports is
                     taken on trust no further than the text it was cut from,
                     since a pre-tokenizer that rewrites may report the range of
                     what it produced instead. *)
                  let at = min (max 0 at) length in
                  let finish = min (max at finish) length in
                  let place =
                    if is_slice source ~at piece then Run.Shifted (start + at)
                    else Run.Fixed { start = start + at; stop = start + finish }
                  in
                  close ~literal:false ~walked:piece ~place ~rewrite:None ~base
                    align)
                (Pre_tokenizer.pieces pre ~first source)
      in
      (* An added token is one span of one id, and its token string is the text
         it matched: [lstrip] and [rstrip] take in the white space beside it,
         which the identifier alone does not describe. A spaced normalizer
         leaves the spaces its pipeline marks, so a token matched in its text
         reports that text marked, as the pipeline writes it: a frame of its own
         over the marked text. *)
      let literal ~marked walked align ~base ~start ~stop ~id =
        Ints.add sc.ids id;
        if record then
          if marked then begin
            let source = slice walked start stop in
            let walked = Normalizer.apply marking source in
            let rewrite =
              Some
                (Run.Deferred { normalizer = marking; source; alignment = None })
            in
            Ints.add sc.span_start 0;
            Ints.add sc.span_stop (String.length walked);
            Ints.add sc.marks (Ints.length sc.ids);
            close ~literal:true ~walked ~place:(Run.Shifted start) ~rewrite
              ~base align
          end
          else begin
            Ints.add sc.span_start start;
            Ints.add sc.span_stop stop;
            Ints.add sc.marks (Ints.length sc.ids);
            close ~literal:true ~walked ~place:verbatim ~rewrite:None ~base
              align
          end
      in
      let stretch ~base raw =
        let norm, align = normalized raw in
        let opens =
          if base > 0 then continuing
          else
            match t.normalize with
            | None -> opening
            | Some n -> lazy (opens_document n norm align raw)
        in
        let stop = String.length norm in
        let pos = ref 0 and scanning = ref true in
        while !scanning do
          match Added_tokens.find_normalized t.added norm ~pos:!pos with
          | None ->
              segment norm align ~base ~start:!pos ~stop ~opens;
              scanning := false
          | Some (start, finish, id) ->
              segment norm align ~base ~start:!pos ~stop:start ~opens;
              literal ~marked:t.spaced_text norm align ~base ~start
                ~stop:finish ~id;
              pos := finish
        done
      in
      if Added_tokens.is_empty t.added then stretch ~base:0 text
      else begin
        let len = String.length text in
        let pos = ref 0 and scanning = ref true in
        while !scanning do
          match Added_tokens.find_raw t.added text ~pos:!pos with
          | None ->
              if !pos < len then stretch ~base:!pos (slice text !pos len);
              scanning := false
          | Some (start, finish, id) ->
              if !pos < start then stretch ~base:!pos (slice text !pos start);
              literal ~marked:false text
                (if record then Some (Lazy.force raw_align) else None)
                ~base:0 ~start ~stop:finish ~id;
              pos := finish
        done
      end;
      if not record then None
      else
        Some
          {
            Run.frames = Array.of_list (List.rev !frames);
            frame_stop = Array.of_list (List.rev !frame_stops);
            span_start = Ints.to_array sc.span_start;
            span_stop = Ints.to_array sc.span_stop;
            marks = Ints.to_array sc.marks;
            opaque = Ints.to_array sc.opaque;
            token_table = t.token_table;
            len_table = t.len_table;
          })

let encode_text t text =
  with_scratch (fun sc ->
      let run = encode_document t sc text ~record:true in
      let ids = Ints.to_array sc.ids in
      match run with
      | Some run -> Encoding.of_run run ~ids
      | None -> Encoding.empty)

let ids_of_text t text =
  with_scratch (fun sc ->
      let (_ : Run.t option) = encode_document t sc text ~record:false in
      Ints.to_array sc.ids)

let post_process t ~add_special primary pair =
  match t.post_processor with
  | None ->
      if Option.is_some pair then invalid_arg err_pair_no_post else primary
  | Some processor ->
      Post_processor.process processor ?pair primary
        ~add_special_tokens:add_special

(* Truncation happens before the post-processor runs, on a budget the tokens it
   will add are taken out of, so that a special token never pushes content past
   [max_length]. A pair gives up tokens from whichever of the two is longer at
   the time, one at a time, until the budget is met. *)
(* How HuggingFace divides a budget between the two sequences of a pair,
   transcribed from the longest-first branch of [utils/truncation.rs] in
   tokenizers 0.23.1: the shorter sequence keeps what it has while the longer
   can give up the difference, and otherwise the budget is halved, the odd token
   going to the longer of the two. *)
let split_budget ~budget first second =
  let swap = first > second in
  let short = if swap then second else first in
  let long = if short > budget then short else max short (budget - short) in
  let short, long =
    if short + long > budget then (budget / 2, (budget / 2) + (budget mod 2))
    else (short, long)
  in
  if swap then (long, short) else (short, long)

(* HuggingFace tokenizes only as many pretokens as it takes to reach
   [max_length] tokens when truncation is configured, keeping whole the pretoken
   that crosses the limit; the excess past it never reaches the overflow
   windows. The kept tokens are not affected — the budget is at most
   [max_length] — so only what lands in {!Encoding.overflowing} sees the
   difference. *)
let limit_to_pretokens encoding ~max_length ~direction =
  let len = Encoding.length encoding in
  if max_length <= 0 || len <= max_length then encoding
  else
    let words = Encoding.word_ids encoding in
    let keep =
      match direction with
      | `Right ->
          let last = words.(max_length - 1) in
          let stop = ref max_length in
          if Option.is_some last then
            while !stop < len && words.(!stop) = last do
              incr stop
            done;
          !stop
      | `Left ->
          let first = words.(len - max_length) in
          let start = ref (len - max_length) in
          if Option.is_some first then
            while !start > 0 && words.(!start - 1) = first do
              decr start
            done;
          len - !start
    in
    if keep = len then encoding
    else
      Encoding.with_overflowing
        (Encoding.truncate encoding ~max_length:keep ~direction)
        []

let truncate_before_post t ~add_special_tokens ~truncation primary pair =
  match truncation with
  | None -> (primary, pair)
  | Some { max_length; stride; direction } -> (
      let primary = limit_to_pretokens primary ~max_length ~direction in
      let pair = Option.map (limit_to_pretokens ~max_length ~direction) pair in
      let added =
        match t.post_processor with
        | Some processor when add_special_tokens ->
            Post_processor.added_tokens processor ~is_pair:(Option.is_some pair)
        | _ -> 0
      in
      let budget = max 0 (max_length - added) in
      let truncate encoding length =
        if Encoding.length encoding <= length then encoding
        else Encoding.truncate encoding ~max_length:length ~stride ~direction
      in
      match pair with
      | None -> (truncate primary budget, None)
      | Some pair ->
          let first, second =
            split_budget ~budget (Encoding.length primary)
              (Encoding.length pair)
          in
          (truncate primary first, Some (truncate pair second)))

(* An overflowing window is a sequence in its own right, so the post-processor
   runs over it too and it carries the same special tokens. On a pair,
   HuggingFace pairs the windows of the two sequences with each other and with
   the kept parts: each window of the first against the second's kept part and
   then against each of its windows, and last the first's kept part against each
   window of the second. *)
let encode_single t ~add_special_tokens ~truncation seq =
  let primary = encode_text t seq.text in
  let pair = Option.map (encode_text t) seq.pair in
  let primary, pair =
    truncate_before_post t ~add_special_tokens ~truncation primary pair
  in
  let processed = post_process t ~add_special:add_special_tokens primary pair in
  match pair with
  | None -> (
      match Encoding.overflowing primary with
      | [] -> processed
      | windows ->
          Encoding.with_overflowing processed
            (List.map
               (fun window ->
                 post_process t ~add_special:add_special_tokens window None)
               windows))
  | Some pair -> (
      match (Encoding.overflowing primary, Encoding.overflowing pair) with
      | [], [] -> processed
      | primary_windows, pair_windows ->
          let process p q =
            post_process t ~add_special:add_special_tokens p (Some q)
          in
          let kept_primary = Encoding.with_overflowing primary [] in
          let kept_pair = Encoding.with_overflowing pair [] in
          let windows =
            List.concat_map
              (fun window ->
                process window kept_pair
                :: List.map (process window) pair_windows)
              primary_windows
            @ List.map (process kept_primary) pair_windows
          in
          Encoding.with_overflowing processed windows)

(* Padding *)

let resolve_pad t (cfg : padding) =
  let token =
    match cfg.pad_token with Some _ as v -> v | None -> t.pad_token
  in
  let token =
    match token with
    | Some token -> token
    | None -> invalid_arg err_no_pad_token
  in
  let id = match cfg.pad_id with Some _ as v -> v | None -> t.pad_id in
  let id =
    match id with
    | Some id -> id
    | None -> (
        match token_to_id t token with
        | Some id -> id
        | None -> invalid_arg (err_pad_not_in_vocab token))
  in
  let type_id = Option.value cfg.pad_type_id ~default:t.pad_type_id in
  (token, id, type_id)

let round_up_to_multiple n m = if n mod m = 0 then n else (n + m - 1) / m * m

(* How long a sequence of [length] tokens is padded to. Only `Batch_longest
   looks past the sequence itself, so it is the caller's to resolve. *)
let pad_target cfg length =
  match cfg.length with
  | `Fixed n -> n
  | `Batch_longest -> length
  | `To_multiple m -> if m <= 0 then length else round_up_to_multiple length m

let apply_padding t encodings = function
  | None -> encodings
  | Some cfg ->
      let pad_token, pad_id, pad_type_id = resolve_pad t cfg in
      let direction = cfg.direction in
      let pad enc target =
        if Encoding.length enc >= target then enc
        else
          Encoding.pad enc ~target_length:target ~pad_id ~pad_type_id ~pad_token
            ~direction
      in
      let longest =
        List.fold_left
          (fun acc enc -> max acc (Encoding.length enc))
          0 encodings
      in
      List.map
        (fun enc ->
          match cfg.length with
          | `Batch_longest -> pad enc longest
          | _ -> pad enc (pad_target cfg (Encoding.length enc)))
        encodings

(* Batch driver *)

(* Work is handed out by bytes rather than by document, so that a batch of one
   long text and a thousand short ones spreads just as evenly. A chunk is a run
   of consecutive documents holding about a [2 * width]th of the batch, so that
   each of [width] domains has a couple to balance over. Under an executor
   [width] is the most domains a program should run at once, whatever the
   executor runs; on the calling domain alone it is one, since smaller chunks
   only add buffers. A chunk holds no less than [min_chunk_bytes], below which
   handing it to another domain costs about what encoding it does, and no more
   than [max_chunk_bytes], past which a large batch balances too coarsely. The
   last chunk of a batch holds whatever is left. *)
let min_chunk_bytes = 1 lsl 16
let max_chunk_bytes = 1 lsl 18

let chunk_size ~exec bytes =
  let width =
    match exec with None -> 1 | Some _ -> Domain.recommended_domain_count ()
  in
  max min_chunk_bytes (min max_chunk_bytes (bytes / (2 * width)))

type chunk = { first : int; stop : int; bytes : int }

let chunks_of ~chunk ~bytes count =
  let chunks = ref [] and first = ref 0 and acc = ref 0 in
  for i = 0 to count - 1 do
    acc := !acc + bytes i;
    if !acc >= chunk then begin
      chunks := { first = !first; stop = i + 1; bytes = !acc } :: !chunks;
      first := i + 1;
      acc := 0
    end
  done;
  if !first < count then
    chunks := { first = !first; stop = count; bytes = !acc } :: !chunks;
  Array.of_list (List.rev !chunks)

type exec = int -> (int -> unit) -> unit

(* [work] over every chunk through [exec], in as many tasks as there are chunks
   but no more than the domains a program should run at once, so that an
   executor starting a domain per task stays within the runtime's limit. A task
   claims chunks one at a time through a single counter, largest first, until
   none is left, so the order holds whatever runs the tasks and however many
   run at once. The first chunk to raise stops the handout: every task finishes
   the chunk it holds and claims no other, and the failure is re-raised once
   [exec] returns. When [exec] returns or raises the handout stops too, so that
   a task it left running claims nothing more. Whether every chunk was encoded
   is a count of chunks finished, not of tasks called: an [exec] that returns
   while a task still runs would otherwise hand back chunks still being
   written. *)
let spread exec chunks work =
  let count = Array.length chunks in
  let order = Array.init count Fun.id in
  Array.sort (fun a b -> Int.compare chunks.(b).bytes chunks.(a).bytes) order;
  let cursor = Atomic.make 0 and finished = Atomic.make 0 in
  let failure = Atomic.make None in
  let stop () = Atomic.set cursor count in
  let task _ =
    try
      let claim = ref (Atomic.fetch_and_add cursor 1) in
      while !claim < count do
        work order.(!claim);
        Atomic.incr finished;
        claim := Atomic.fetch_and_add cursor 1
      done
    with e ->
      let backtrace = Printexc.get_raw_backtrace () in
      ignore (Atomic.compare_and_set failure None (Some (e, backtrace)) : bool);
      stop ()
  in
  (match exec (min count (Domain.recommended_domain_count ())) task with
  | () -> stop ()
  | exception e ->
      let backtrace = Printexc.get_raw_backtrace () in
      stop ();
      Printexc.raise_with_backtrace e backtrace);
  match Atomic.get failure with
  | Some (e, backtrace) -> Printexc.raise_with_backtrace e backtrace
  | None ->
      if Atomic.get finished < count then
        invalid_arg "exec returned before every chunk was encoded"

(* Without an executor, or with a single chunk, the batch is encoded on the
   calling domain. *)
let run_chunks ?exec chunks work =
  match exec with
  | Some exec when Array.length chunks > 1 -> spread exec chunks work
  | _ -> Array.iteri (fun c _ -> work c) chunks

let encode_sequences t sequences ~add_special_tokens ~padding ~truncation
    ?exec () =
  match sequences with
  | [] -> []
  | _ ->
      let sequences = Array.of_list sequences in
      let count = Array.length sequences in
      let results = Array.make count Encoding.empty in
      let bytes i =
        let seq = sequences.(i) in
        String.length seq.text
        + match seq.pair with Some pair -> String.length pair | None -> 0
      in
      let total = ref 0 in
      for i = 0 to count - 1 do
        total := !total + bytes i
      done;
      let chunk = chunk_size ~exec !total in
      let chunks = chunks_of ~chunk ~bytes count in
      run_chunks ?exec chunks (fun c ->
          for i = chunks.(c).first to chunks.(c).stop - 1 do
            results.(i) <-
              encode_single t ~add_special_tokens ~truncation sequences.(i)
          done);
      apply_padding t (Array.to_list results) padding

(* [t] with its special tokens left unmatched when the caller asks for their
   text to be encoded like any other. *)
let matching t ~match_special_tokens =
  if match_special_tokens then t
  else { t with added = Added_tokens.without_special t.added }

let encode t ?pair ?(add_special_tokens = true) ?(match_special_tokens = true)
    ?padding ?truncation text =
  let t = matching t ~match_special_tokens in
  match
    apply_padding t
      [ encode_single t ~add_special_tokens ~truncation { text; pair } ]
      padding
  with
  | [ encoding ] -> encoding
  | _ -> assert false

let encode_batch t ?(add_special_tokens = true) ?(match_special_tokens = true)
    ?padding ?truncation ?exec = function
  | [] -> []
  | texts ->
      let t = matching t ~match_special_tokens in
      let sequences = List.map (fun text -> { text; pair = None }) texts in
      encode_sequences t sequences ~add_special_tokens ~padding ~truncation
        ?exec ()

let encode_batch_pairs t ?(add_special_tokens = true)
    ?(match_special_tokens = true) ?padding ?truncation ?exec = function
  | [] -> []
  | pairs ->
      let t = matching t ~match_special_tokens in
      let sequences =
        List.map (fun (text, pair) -> { text; pair = Some pair }) pairs
      in
      encode_sequences t sequences ~add_special_tokens ~padding ~truncation
        ?exec ()

(* Asked for the ids alone, there is no encoding to build whenever the
   post-processor only wraps the sequence: what is left is the body, a prefix
   and a suffix, with truncation and padding over the three. A pair genuinely
   interleaves two sequences, so it goes the long way, and so does anything the
   post-processor does not describe as affixes. Overflowing windows have nowhere
   to go in an [int array] and are dropped. *)
let encode_ids t ?pair ?(add_special_tokens = true)
    ?(match_special_tokens = true) ?padding ?truncation text =
  let t = matching t ~match_special_tokens in
  let affixes =
    match pair with
    | Some _ -> None
    | None -> (
        match t.post_processor with
        | None -> Some ([||], [||])
        | Some processor -> Post_processor.affixes processor ~add_special_tokens
        )
  in
  match affixes with
  | None ->
      Encoding.ids
        (encode t ?pair ~add_special_tokens ?padding ?truncation text)
  | Some (prefix, suffix) -> (
      let body = ids_of_text t text in
      let body =
        match truncation with
        | None -> body
        | Some { max_length; direction; _ } -> (
            let budget =
              max 0 (max_length - Array.length prefix - Array.length suffix)
            in
            let length = Array.length body in
            if length <= budget then body
            else
              match direction with
              | `Right -> Array.sub body 0 budget
              | `Left -> Array.sub body (length - budget) budget)
      in
      let ids =
        if Array.length prefix = 0 && Array.length suffix = 0 then body
        else Array.concat [ prefix; body; suffix ]
      in
      let length = Array.length ids in
      match padding with
      | None -> ids
      | Some cfg ->
          let target = pad_target cfg length in
          if target <= length then ids
          else begin
            let _, pad_id, _ = resolve_pad t cfg in
            let padded = Array.make target pad_id in
            let at =
              match cfg.direction with `Left -> target - length | `Right -> 0
            in
            Array.blit ids 0 padded at length;
            padded
          end)

(* Cutting a document *)

(* A document at least twice a chunk long is encoded in pieces of about a chunk,
   so that one document can occupy more than one domain — when it can be cut at
   all: at a space with an alphanumeric before it and a letter after it, that no
   added token touches, in a pipeline whose walker promises that such a cut
   leaves its spans as they were and that has no normalizer, since a normalizer
   is free to act across the cut and none is asked whether it does. [cuttable]
   is the pipeline's part of the rule, [cut_document] the text's. *)
let cuttable t =
  let splittable pre =
    match Pre_tokenizer.plan pre with
    | Pre_tokenizer.Walk { splittable; _ }
    | Pre_tokenizer.Segmented { splittable; _ } ->
        splittable
    | Pre_tokenizer.Pieces -> false
  in
  Option.is_none t.normalizer
  &&
  match t.cut with
  | Whole | Pieces _ -> false
  | Walk pre | Rewritten { pre; _ } -> splittable pre
  | Segmented { outer; _ } -> splittable outer

let ascii_letter c = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
let ascii_alnum c = ascii_letter c || (c >= '0' && c <= '9')

(* Whether the content of [token] lies over the byte before [at], the one at it
   or the one after it. Added tokens are matched by one scan of the text, and
   the scan of a piece agrees with the scan of the whole as long as the whole
   scan reaches the cut and no match — kept, or refused for not standing alone
   as a word — covers it or takes in the space at it. Every such match puts a
   content over one of those three bytes: an alphanumeric on either side of the
   space stops the white space an [lstrip] or [rstrip] token takes in from
   reaching any further. *)
let touches text ~at (token : Added_tokens.token) =
  let content = token.content in
  let rec from p =
    p <= at + 1 && (is_slice text ~at:p content || from (p + 1))
  in
  from (max 0 (at - String.length content))

(* [text] in pieces of about [chunk] bytes, cut on a space between an ASCII
   alphanumeric and an ASCII letter — the space is then a whole character, and
   so are its neighbours — that no added token touches. The space opens the
   piece that follows it, so that a pipeline prepending a space or a marker to a
   text not starting with one leaves every piece but the first as it found it. A
   boundary with no such space near it is not cut. *)
let cut_document t ~chunk text =
  let len = String.length text in
  let count = len / chunk in
  let tokens = Added_tokens.tokens t.added in
  let safe at =
    String.unsafe_get text at = ' '
    && ascii_alnum (String.unsafe_get text (at - 1))
    && ascii_letter (String.unsafe_get text (at + 1))
    && not (List.exists (touches text ~at) tokens)
  in
  let pieces = ref [] and start = ref 0 in
  for k = 1 to count - 1 do
    let target = max (!start + 1) (k * len / count) in
    let limit = min (len - 1) (target + (len / count)) in
    let at = ref target in
    while !at < limit && not (safe !at) do
      incr at
    done;
    if !at < limit then begin
      pieces := String.sub text !start (!at - !start) :: !pieces;
      start := !at
    end
  done;
  List.rev (String.sub text !start (len - !start) :: !pieces)

(* Flat batches *)

type ids = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t

let create_ids n = Bigarray.Array1.create Bigarray.Int32 Bigarray.C_layout n

(* The pieces of a batch — a document is one piece unless it was cut — as the
   text of each and the row it belongs to. The pieces of a row are consecutive.
   A batch with nothing to cut is its own pieces. *)
let pieces_of t rows ~cut ~chunk =
  let nrows = Array.length rows in
  let long text = cut && String.length text >= 2 * chunk in
  if not (Array.exists long rows) then (rows, Array.init nrows Fun.id)
  else begin
    let texts = ref [] and owners = ref [] in
    Array.iteri
      (fun row text ->
        List.iter
          (fun piece ->
            texts := piece :: !texts;
            owners := row :: !owners)
          (if long text then cut_document t ~chunk text else [ text ]))
      rows;
    (Array.of_list (List.rev !texts), Array.of_list (List.rev !owners))
  end

(* A batch's chunks append their ids to a buffer of their domain's, which the
   batch borrows from the tokenizer and holds until its result is copied out: a
   batch then allocates its ids once, in the result, and the collector, which
   counts the bytes of every buffer against its pace, runs that much less. The
   tokenizer keeps at most [Domain.recommended_domain_count ()] buffers between
   batches, of at most [batch_keep] ids (8 MB) each, as many as caches of the
   default size, and they go with it. *)
let batch_keep = 1 lsl 21

let max_buffers = Domain.recommended_domain_count ()
let batch_serial = Atomic.make 0

(* Called by a batch's chunk before each document it encodes into its ids
   buffer. For tests. *)
let batch_document_hook = Atomic.make (fun () -> ())

let rec push list x =
  let seen = Atomic.get list in
  if not (Atomic.compare_and_set list seen (x :: seen)) then push list x

(* The scratches whose buffers a batch holds, until the batch ends. *)
type held = Holding of scratch list | Ended

(* Adds [sc] to [held], unless the batch has ended: a chunk an executor left
   running past the batch's end takes no buffer for it. *)
let rec hold held sc =
  match Atomic.get held with
  | Ended -> false
  | Holding scs as seen ->
      Atomic.compare_and_set held seen (Holding (sc :: scs)) || hold held sc

let rec borrow b =
  match Atomic.get b.free with
  | [] -> { Bpe.ba = create_ids 0; used = 0 }
  | sink :: rest as seen ->
      if Atomic.compare_and_set b.free seen rest then begin
        Atomic.decr b.kept;
        sink
      end
      else borrow b

let give_back b sink =
  if Bigarray.Array1.dim sink.Bpe.ba <= batch_keep then
    if Atomic.fetch_and_add b.kept 1 < max_buffers then push b.free sink
    else Atomic.decr b.kept

(* The sink of a chunk of batch [call] needing [room] ids, encoded with [sc]:
   the buffer [sc] holds for [call], borrowed from [b] if no other batch holds
   [sc] and [call] has not ended, and otherwise one of the chunk's own. A
   buffer without the room is left to the chunks that wrote it and replaced: by
   one of the [share] of the batch a domain should take when the batch takes
   [sc], by one twice its size after, up to [batch_keep] either way. A chunk's
   ids are never moved for another's. *)
let batch_sink b sc ~call ~held ~room ~share =
  let owner = Atomic.get sc.holder in
  let taken =
    owner = 0
    && Atomic.compare_and_set sc.holder 0 call
    &&
    if hold held sc then begin
      let sink = borrow b in
      sink.Bpe.used <- 0;
      sc.batch <- sink;
      true
    end
    else begin
      Atomic.set sc.holder 0;
      false
    end
  in
  if taken || owner = call then begin
    (* Read once: a batch ending meanwhile puts [no_batch] there, which has no
       room, so a chunk never writes into it. *)
    let sink = sc.batch in
    let size = Bigarray.Array1.dim sink.Bpe.ba in
    if size - sink.Bpe.used >= room then sink
    else begin
      let grown = min batch_keep (if taken then share else 2 * size) in
      let sink = { Bpe.ba = create_ids (max room grown); used = 0 } in
      sc.batch <- sink;
      sink
    end
  end
  else { Bpe.ba = create_ids room; used = 0 }

(* Ends batch [call]: the scratches it holds are freed, and their buffers go
   back to [b] when [give] holds — only once every chunk has returned, since a
   chunk an executor left running would write into the buffer the next batch
   borrows. Otherwise they are dropped, and the collector takes them once the
   chunks writing them are done. *)
let end_batch b ~call ~give held =
  match Atomic.exchange held Ended with
  | Ended -> ()
  | Holding scs ->
      List.iter
        (fun sc ->
          let sink = sc.batch in
          sc.batch <- no_batch;
          ignore (Atomic.compare_and_set sc.holder call 0 : bool);
          if give then give_back b sink)
        scs

(* The throughput path: {!encode_ids} over a batch, with the ids of every
   document in one buffer. A chunk of pieces is encoded into the batch buffer of
   the domain encoding it ([batch_sink]), so that a domain writes nothing
   another one reads, and the buffers are copied into the result at offsets
   that are known only once every chunk has been encoded — a row's length is the
   whole batch's business as soon as padding is [`Batch_longest]. Affixes,
   truncation and padding are applied during that copy, piece by piece: they
   move no ids the model produced. The buffers are [ids] rather than [Ints.t]:
   they hold the bulk of a batch, and outside the heap the collector neither
   scans nor sweeps them. *)
let encode_batch_ids t ?(add_special_tokens = true)
    ?(match_special_tokens = true) ?padding ?truncation ?exec texts =
  let t = matching t ~match_special_tokens in
  let rows = Array.of_list texts in
  let nrows = Array.length rows in
  let affixes =
    match t.post_processor with
    | None -> Some ([||], [||])
    | Some processor -> Post_processor.affixes processor ~add_special_tokens
  in
  let wrapped = Option.is_some affixes in
  let prefix, suffix = Option.value affixes ~default:([||], [||]) in
  let np = Array.length prefix and ns = Array.length suffix in
  let chunk =
    chunk_size ~exec
      (Array.fold_left (fun total text -> total + String.length text) 0 rows)
  in
  let texts, owner = pieces_of t rows ~cut:(wrapped && cuttable t) ~chunk in
  let npieces = Array.length texts in
  let chunks =
    chunks_of ~chunk ~bytes:(fun i -> String.length texts.(i)) npieces
  in
  (* Where a piece's ids lie: in the buffer of its chunk, which the worker puts
     in place of the placeholder, from [start] to [stop]. *)
  let buffers = Array.make (Array.length chunks) (create_ids 0) in
  let chunk_of = Array.make npieces 0 in
  let start = Array.make npieces 0 and stop = Array.make npieces 0 in
  let call = 1 + Atomic.fetch_and_add batch_serial 1 in
  let held = Atomic.make (Holding []) in
  (* The ids a domain should take of the batch, as [batch_sink] reserves room:
     the batch's over the domains that can share it. *)
  let share =
    let width =
      match exec with
      | None -> 1
      | Some _ -> min (Array.length chunks) (Domain.recommended_domain_count ())
    in
    let bytes = Array.fold_left (fun n c -> n + c.bytes) 0 chunks in
    65536 + (bytes / 3 / max 1 width)
  in
  let end_batch ~give = end_batch t.batch_buffers ~call ~give held in
  let work c =
    let chunk = chunks.(c) in
    if wrapped then
      (* The fused kernels write their ids into the batch buffer as int32
         directly; what OCaml encodes lands in [sc.ids], the spill, moved into
         place at document end. The room asked for is what a kernel call
         reserves and a third of the chunk's bytes, more than byte-level
         models need on prose; a chunk needing more grows the buffer. *)
      with_scratch (fun sc ->
          let sink =
            batch_sink t.batch_buffers sc ~call ~held ~share
              ~room:(65536 + (chunk.bytes / 3))
          in
          for i = chunk.first to chunk.stop - 1 do
            let first = sink.Bpe.used in
            (Atomic.get batch_document_hook) ();
            let (_ : Run.t option) =
              encode_document t sc ~ids32:sink texts.(i) ~record:false
            in
            Bpe.flush_spill sink sc.ids;
            chunk_of.(i) <- c;
            start.(i) <- first;
            stop.(i) <- sink.Bpe.used
          done;
          buffers.(c) <- sink.Bpe.ba)
    else begin
      (* The post-processor does more than wrap: a piece is then a whole
         document, encoded the long way and truncated already, its ids moved
         to the chunk's buffer once done. *)
      let buffer = ref (create_ids (max 64 (chunk.bytes / 3))) in
      let used = ref 0 in
      let into = Ints.create () in
      for i = chunk.first to chunk.stop - 1 do
        let ids =
          Encoding.ids
            (encode_single t ~add_special_tokens ~truncation
               { text = texts.(i); pair = None })
        in
        for k = 0 to Array.length ids - 1 do
          Ints.add into (Array.unsafe_get ids k)
        done;
        let n = Ints.length into in
        let held = Bigarray.Array1.dim !buffer in
        if !used + n > held then begin
          let grown = create_ids (2 * max held (!used + n)) in
          Bigarray.Array1.blit
            (Bigarray.Array1.sub !buffer 0 !used)
            (Bigarray.Array1.sub grown 0 !used);
          buffer := grown
        end;
        Ints.blit_to_int32 into ~pos:0 ~len:n !buffer ~at:!used;
        Ints.clear into;
        chunk_of.(i) <- c;
        start.(i) <- !used;
        used := !used + n;
        stop.(i) <- !used
      done;
      buffers.(c) <- !buffer
    end
  in
  (match run_chunks ?exec chunks work with
  | () -> ()
  | exception e ->
      let backtrace = Printexc.get_raw_backtrace () in
      end_batch ~give:false;
      Printexc.raise_with_backtrace e backtrace);
  Fun.protect ~finally:(fun () -> end_batch ~give:true) @@ fun () ->
  (* A row's body is its pieces' ids end to end; [place] is where each of them
     begins in it. *)
  let body = Array.make nrows 0 and place = Array.make npieces 0 in
  for i = 0 to npieces - 1 do
    let row = owner.(i) in
    place.(i) <- body.(row);
    body.(row) <- body.(row) + stop.(i) - start.(i)
  done;
  let budget =
    match truncation with
    | Some { max_length; _ } when wrapped -> max 0 (max_length - np - ns)
    | _ -> max_int
  in
  let content = Array.init nrows (fun row -> np + min body.(row) budget + ns) in
  let lengths =
    match padding with
    | None -> content
    | Some cfg ->
        let longest = Array.fold_left max 0 content in
        Array.map
          (fun size ->
            max size
              (match cfg.length with
              | `Batch_longest -> longest
              | _ -> pad_target cfg size))
          content
  in
  let offset = Array.make (nrows + 1) 0 in
  for row = 0 to nrows - 1 do
    offset.(row + 1) <- offset.(row) + lengths.(row)
  done;
  let result = create_ids offset.(nrows) in
  (* The pad token is looked up only when a row needs it, as {!encode_ids} looks
     it up. *)
  let pad_id =
    match padding with
    | Some cfg when offset.(nrows) > Array.fold_left ( + ) 0 content ->
        let _, pad_id, _ = resolve_pad t cfg in
        Int32.of_int pad_id
    | _ -> 0l
  in
  let pad_left =
    match padding with Some { direction = `Left; _ } -> true | _ -> false
  in
  let trim_left =
    match truncation with
    | Some { direction = `Left; _ } -> wrapped
    | _ -> false
  in
  let write ids ~at =
    for k = 0 to Array.length ids - 1 do
      Bigarray.Array1.unsafe_set result (at + k)
        (Int32.of_int (Array.unsafe_get ids k))
    done
  in
  let fill ~at ~len =
    if len > 0 then
      Bigarray.Array1.fill (Bigarray.Array1.sub result at len) pad_id
  in
  (* The copy runs on the calling domain, so that a batch calls its executor
     once. A piece copies the part of itself that truncation keeps; pieces that
     follow each other in a buffer and in the result, as whole rows without
     affixes do, copy as one blit. The first piece of a row lays out the row
     around the body. *)
  let run_src = ref result and run_at = ref 0 and run_to = ref 0 in
  let run_len = ref 0 in
  let flush () =
    if !run_len > 0 then
      Bigarray.Array1.blit
        (Bigarray.Array1.sub !run_src !run_at !run_len)
        (Bigarray.Array1.sub result !run_to !run_len)
  in
  for i = 0 to npieces - 1 do
    let row = owner.(i) in
    let kept = min body.(row) budget in
    let skip = if trim_left then body.(row) - kept else 0 in
    let padded = lengths.(row) - content.(row) in
    let front = if pad_left then padded else 0 in
    let base = offset.(row) + front + np in
    let lo = max place.(i) skip
    and hi = min (place.(i) + stop.(i) - start.(i)) (skip + kept) in
    if lo < hi then begin
      let src = buffers.(chunk_of.(i)) in
      let at = start.(i) + lo - place.(i) and dst = base + lo - skip in
      if
        src == !run_src
        && at = !run_at + !run_len
        && dst = !run_to + !run_len
      then run_len := !run_len + (hi - lo)
      else begin
        flush ();
        run_src := src;
        run_at := at;
        run_to := dst;
        run_len := hi - lo
      end
    end;
    if i = 0 || owner.(i - 1) <> row then begin
      fill ~at:offset.(row) ~len:front;
      fill ~at:(base + kept + ns) ~len:(padded - front);
      write prefix ~at:(offset.(row) + front);
      write suffix ~at:(base + kept)
    end
  done;
  flush ();
  (result, lengths)

(* Decoding *)

let decode t ?(skip_special_tokens = false) ids =
  let tokens =
    Array.to_list ids
    |> List.filter_map (fun id ->
        if skip_special_tokens && Added_tokens.is_special t.added id then None
        else id_to_token t id)
  in
  match t.decoder with
  | Some decoder -> Decoder.decode decoder tokens
  | None -> String.concat " " tokens

(* A tokenizer without a decoder joins its tokens with spaces, so the text of
   each token is known as it arrives. *)
type decoding = Decoding of Decoder.stream | Joining of { first : bool }

type stream = { tokenizer : t; skip_special_tokens : bool; decoding : decoding }

let decode_stream ?(skip_special_tokens = false) t =
  let decoding =
    match t.decoder with
    | Some decoder -> Decoding (Decoder.stream decoder)
    | None -> Joining { first = true }
  in
  { tokenizer = t; skip_special_tokens; decoding }

let decode_step s id =
  let t = s.tokenizer in
  let token =
    if s.skip_special_tokens && Added_tokens.is_special t.added id then None
    else id_to_token t id
  in
  match (token, s.decoding) with
  | None, _ -> ("", s)
  | Some token, Decoding stream ->
      let text, stream = Decoder.step stream token in
      (text, { s with decoding = Decoding stream })
  | Some token, Joining { first } ->
      let text = if first then token else " " ^ token in
      (text, { s with decoding = Joining { first = false } })

let decode_finish s =
  match s.decoding with
  | Decoding stream -> Decoder.finish stream
  | Joining _ -> ""

(* Training *)

let special_tokens_for_training init requested =
  let items =
    (match requested with
      | Some sl -> List.map (fun (a : added_token) -> a.content) sl
      | None -> [])
    @
    match init with
    | Some tok ->
        List.map (fun (a : added_token) -> a.content) (added_tokens tok)
    | None -> []
  in
  dedup_by Fun.id items

let merge_added_tokens_from_training ~requested ~trained_tokens =
  let items =
    (match requested with Some tokens -> tokens | None -> [])
    @ List.map added_token trained_tokens
  in
  dedup_by (fun (a : added_token) -> a.content) items

(* A line keeps the newline that ends it, so a corpus file's line breaks reach
   the pre-tokenizer and a byte-level model learns a token for them. *)
let iter_lines ic f =
  let buf = Buffer.create 4096 in
  let chunk = Bytes.create 65536 in
  let rec loop () =
    let n = input ic chunk 0 (Bytes.length chunk) in
    if n > 0 then begin
      let start = ref 0 in
      for i = 0 to n - 1 do
        if Bytes.unsafe_get chunk i = '\n' then begin
          Buffer.add_subbytes buf chunk !start (i - !start + 1);
          f (Buffer.contents buf);
          Buffer.clear buf;
          start := i + 1
        end
      done;
      Buffer.add_subbytes buf chunk !start (n - !start);
      loop ()
    end
  in
  loop ();
  if Buffer.length buf > 0 then f (Buffer.contents buf)

let iter_texts data f =
  match data with
  | `Files files ->
      List.iter
        (fun file ->
          let ic = open_in_bin file in
          Fun.protect
            ~finally:(fun () -> close_in ic)
            (fun () -> iter_lines ic f))
        files
  | `Seq seq -> Seq.iter f seq

(* The words a model is trained on are the pre-tokens the pipeline would hand
   it: every text is normalized, then cut by the pre-tokenizer. With no
   pre-tokenizer a text is one word. *)
let training_words ?normalizer ?pre data =
  let counts = Hashtbl.create 10000 in
  let add word =
    if word <> "" then
      Hashtbl.replace counts word
        (1 + try Hashtbl.find counts word with Not_found -> 0)
  in
  iter_texts data (fun text ->
      let normalized =
        match normalizer with Some n -> Normalizer.apply n text | None -> text
      in
      match pre with
      | Some pre ->
          List.iter
            (fun (piece, _) -> add piece)
            (Pre_tokenizer.pre_tokenize pre normalized)
      | None -> add normalized);
  Hashtbl.fold (fun word count acc -> (word, count) :: acc) counts []

(* Each entry of the initial alphabet stands for the code point it starts with;
   an entry that holds none is dropped. *)
let initial_alphabet_of strs =
  List.filter_map
    (fun s ->
      if String.length s = 0 then None
      else
        let d = String.get_utf_8_uchar s 0 in
        if Uchar.utf_decode_is_valid d then
          Some (String.sub s 0 (Uchar.utf_decode_length d))
        else None)
    strs

let train_bpe ?init ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token
    ?eos_token ?pad_token ?unk_token ?(vocab_size = 30000) ?(min_frequency = 0)
    ?limit_alphabet ?initial_alphabet ?continuing_subword_prefix
    ?end_of_word_suffix ?max_token_length data =
  let special_tokens = special_tokens_for_training init added_tokens in
  let initial_alphabet =
    Option.value initial_alphabet ~default:[] |> initial_alphabet_of
  in
  let words = training_words ?normalizer ?pre data in
  let trained_model, trained_tokens =
    Bpe.train ~min_frequency ~vocab_size ~show_progress:false ~special_tokens
      ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
      ~end_of_word_suffix ~max_token_length words
  in
  let all_added =
    merge_added_tokens_from_training ~requested:added_tokens ~trained_tokens
  in
  create ?normalizer ?pre ?post ?decoder ~added_tokens:all_added ?bos_token
    ?eos_token ?pad_token ?unk_token (Alg_bpe trained_model)

let train_wordpiece ?init ?normalizer ?pre ?post ?decoder ?added_tokens
    ?bos_token ?eos_token ?pad_token ?unk_token ?(vocab_size = 30000)
    ?(min_frequency = 0) ?limit_alphabet ?initial_alphabet
    ?(continuing_subword_prefix = "##") ?end_of_word_suffix data =
  let special_tokens = special_tokens_for_training init added_tokens in
  let initial_alphabet =
    Option.value initial_alphabet ~default:[] |> initial_alphabet_of
  in
  let words = training_words ?normalizer ?pre data in
  let trained_model, trained_tokens =
    Wordpiece.train ~min_frequency ~vocab_size ~show_progress:false
      ~special_tokens ~limit_alphabet ~initial_alphabet
      ~continuing_subword_prefix ~end_of_word_suffix words
  in
  let all_added =
    merge_added_tokens_from_training ~requested:added_tokens ~trained_tokens
  in
  create ?normalizer ?pre ?post ?decoder ~added_tokens:all_added ?bos_token
    ?eos_token ?pad_token ?unk_token (Alg_wordpiece trained_model)

let train_word_level ?init ?normalizer ?pre ?post ?decoder ?added_tokens
    ?bos_token ?eos_token ?pad_token ?unk_token ?(vocab_size = 30000)
    ?(min_frequency = 0) data =
  let special_tokens = special_tokens_for_training init added_tokens in
  let words = training_words ?normalizer ?pre data in
  let trained_model, trained_tokens =
    Word_level.train ~vocab_size ~min_frequency ~show_progress:false
      ~special_tokens words
  in
  let all_added =
    merge_added_tokens_from_training ~requested:added_tokens ~trained_tokens
  in
  create ?normalizer ?pre ?post ?decoder ~added_tokens:all_added ?bos_token
    ?eos_token ?pad_token ?unk_token (Alg_wordlevel trained_model)

let train_unigram ?init ?normalizer ?pre ?post ?decoder ?added_tokens ?bos_token
    ?eos_token ?pad_token ?unk_token ?(vocab_size = 8000) data =
  let special_tokens = special_tokens_for_training init added_tokens in
  let words = training_words ?normalizer ?pre data in
  let trained_model, trained_tokens =
    Unigram.train ~vocab_size ~show_progress:false ~special_tokens
      ~shrinking_factor:0.75 ~unk_token ~max_piece_length:16 ~n_sub_iterations:2
      words
  in
  let all_added =
    merge_added_tokens_from_training ~requested:added_tokens ~trained_tokens
  in
  create ?normalizer ?pre ?post ?decoder ~added_tokens:all_added ?bos_token
    ?eos_token ?pad_token ?unk_token (Alg_unigram trained_model)

(* JSON serialization *)

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

let json_string_or_null = function Jsont.String (s, _) -> Some s | _ -> None
let json_option_of f = function None -> Jsont.Json.null () | Some v -> f v

let added_token_of_json json =
  let mem name = json_mem name json in
  let bool_or default = function Jsont.Bool (b, _) -> b | _ -> default in
  let to_str = function
    | Jsont.String (s, _) -> s
    | _ -> failwith "expected string"
  in
  let special = bool_or true (mem "special") in
  {
    content = to_str (mem "content");
    special;
    single_word = bool_or false (mem "single_word");
    lstrip = bool_or false (mem "lstrip");
    rstrip = bool_or false (mem "rstrip");
    normalized = bool_or (not special) (mem "normalized");
  }

let added_token_to_json (tok : Added_tokens.token) =
  json_obj
    [
      ("id", Jsont.Json.int tok.id);
      ("content", Jsont.Json.string tok.content);
      ("single_word", Jsont.Json.bool tok.single_word);
      ("lstrip", Jsont.Json.bool tok.lstrip);
      ("rstrip", Jsont.Json.bool tok.rstrip);
      ("normalized", Jsont.Json.bool tok.normalized);
      ("special", Jsont.Json.bool tok.special);
    ]

let vocab_to_json vocab =
  json_obj (List.map (fun (token, id) -> (token, Jsont.Json.int id)) vocab)

let alg_to_json = function
  | Alg_bpe bpe ->
      let vocab_json = vocab_to_json (Bpe.get_vocab bpe) in
      let merges_json =
        Bpe.get_merges bpe
        |> List.map (fun (a, b) ->
            Jsont.Json.list [ Jsont.Json.string a; Jsont.Json.string b ])
        |> Jsont.Json.list
      in
      json_obj
        [
          ("type", Jsont.Json.string "BPE");
          ("dropout", json_option_of Json_text.f32 (Bpe.get_dropout bpe));
          ("unk_token", json_option_of Jsont.Json.string (Bpe.get_unk_token bpe));
          ( "continuing_subword_prefix",
            json_option_of Jsont.Json.string
              (Bpe.get_continuing_subword_prefix bpe) );
          ( "end_of_word_suffix",
            json_option_of Jsont.Json.string (Bpe.get_end_of_word_suffix bpe) );
          ("fuse_unk", Jsont.Json.bool (Bpe.get_fuse_unk bpe));
          ("byte_fallback", Jsont.Json.bool (Bpe.get_byte_fallback bpe));
          ("ignore_merges", Jsont.Json.bool (Bpe.get_ignore_merges bpe));
          ("vocab", vocab_json);
          ("merges", merges_json);
        ]
  | Alg_wordpiece wp ->
      json_obj
        [
          ("type", Jsont.Json.string "WordPiece");
          ("unk_token", Jsont.Json.string (Wordpiece.get_unk_token wp));
          ( "continuing_subword_prefix",
            Jsont.Json.string (Wordpiece.get_continuing_subword_prefix wp) );
          ("max_input_chars_per_word", Jsont.Json.int 100);
          ("vocab", vocab_to_json (Wordpiece.get_vocab wp));
        ]
  | Alg_wordlevel wl ->
      json_obj
        [
          ("type", Jsont.Json.string "WordLevel");
          ("unk_token", Jsont.Json.string "[UNK]");
          ("vocab", vocab_to_json (Word_level.get_vocab wl));
        ]
  | Alg_unigram ug ->
      let vocab_json =
        Unigram.get_vocab ug
        |> List.map (fun (token, score) ->
            Jsont.Json.list [ Jsont.Json.string token; Json_text.f64 score ])
        |> Jsont.Json.list
      in
      json_obj
        [
          ("type", Jsont.Json.string "Unigram");
          ("unk_id", json_option_of Jsont.Json.int (Unigram.get_unk_id ug));
          ("vocab", vocab_json);
          ("byte_fallback", Jsont.Json.bool (Unigram.get_byte_fallback ug));
        ]
  | Alg_chars _ ->
      json_obj [ ("type", Jsont.Json.string "Chars"); ("vocab", json_obj []) ]

let to_json (t : t) =
  let added_tokens =
    Added_tokens.tokens t.added
    |> List.sort (fun (a : Added_tokens.token) b -> Int.compare a.id b.id)
    |> List.map added_token_to_json
  in
  json_obj
    [
      ("version", Jsont.Json.string "1.0");
      ("truncation", Jsont.Json.null ());
      ("padding", Jsont.Json.null ());
      ("added_tokens", Jsont.Json.list added_tokens);
      ("normalizer", json_option_of Normalizer.to_json t.normalizer);
      ("pre_tokenizer", json_option_of Pre_tokenizer.to_json t.pre_tokenizer);
      ("post_processor", json_option_of Post_processor.to_json t.post_processor);
      ("decoder", json_option_of Decoder.to_json t.decoder);
      ("model", alg_to_json t.algorithm);
    ]

(* JSON deserialization helpers *)

let json_to_assoc = function
  | Jsont.Object (mems, _) ->
      List.map
        (fun ((k, _), v) ->
          match v with
          | Jsont.Number (f, _) -> (k, int_of_float f)
          | _ -> failwith ("Expected number for vocab entry: " ^ k))
        mems
  | _ -> failwith "Expected object for vocab"

let json_to_list = function
  | Jsont.Array (l, _) -> l
  | _ -> failwith "Expected array"

let json_to_string = function
  | Jsont.String (s, _) -> s
  | _ -> failwith "Expected string"

(* A number as tokenizers reads its text, in [text] where the tree records the
   number's bytes, else as given. tokenizers' reading can differ in the last
   place from the double the text names, and a Unigram segmentation that ties
   on the written scores is decided by that place. *)
let json_to_float ?text = function
  | Jsont.Number (f, meta) -> (
      let loc = Jsont.Meta.textloc meta in
      match text with
      | Some text when not (Jsont.Textloc.is_none loc) ->
          Hf_number.read text ~first:(Jsont.Textloc.first_byte loc)
            ~last:(Jsont.Textloc.last_byte loc)
      | _ -> f)
  | _ -> failwith "Expected number"

(* tokenizers holds a BPE dropout in single precision. *)
let single x = Int32.float_of_bits (Int32.bits_of_float x)

let json_has_field name j =
  match json_mem name j with Jsont.Null _ -> false | _ -> true

let json_result_to_option of_json = function
  | Jsont.Null _ -> None
  | j -> ( match of_json j with Ok v -> Some v | Error msg -> failwith msg)

let infer_model_type mj =
  match json_string_or_null (json_mem "type" mj) with
  | Some s -> s
  | None ->
      if json_has_field "merges" mj then "BPE"
      else if json_has_field "unk_id" mj then "Unigram"
      else if
        json_has_field "continuing_subword_prefix" mj
        || json_has_field "max_input_chars_per_word" mj
      then "WordPiece"
      else if json_has_field "vocab" mj then "WordLevel"
      else failwith err_infer_type

let parse_merge = function
  | Jsont.Array ([ a; b ], _) -> (json_to_string a, json_to_string b)
  | Jsont.String (s, _) -> (
      match String.split_on_char ' ' s with
      | [ a; b ] -> (a, b)
      | _ -> failwith "Invalid merge string format")
  | _ -> failwith "Invalid merge entry"

let alg_of_json ?text ~byte_level ~spaces mj =
  let mem name = json_mem name mj in
  let str name = json_string_or_null (mem name) in
  let flag name = match mem name with Jsont.Bool (b, _) -> b | _ -> false in
  match infer_model_type mj with
  | "BPE" ->
      let vocab_list = json_to_assoc (mem "vocab") in
      let merges = json_to_list (mem "merges") |> List.map parse_merge in
      let dropout =
        match mem "dropout" with
        | Jsont.Number _ as f -> Some (single (json_to_float ?text f))
        | _ -> None
      in
      let vocab = vocab_to_hashtbl vocab_list in
      let continuing_subword_prefix = str "continuing_subword_prefix" in
      let end_of_word_suffix = str "end_of_word_suffix" in
      let spaced =
        spaced_model ~spaces ~byte_level ?continuing_subword_prefix
          ?end_of_word_suffix vocab
      in
      Alg_bpe
        (Bpe.create ~vocab ~merges ~byte_level ~spaced ?dropout
           ?unk_token:(str "unk_token") ?continuing_subword_prefix
           ?end_of_word_suffix
           ~fuse_unk:(flag "fuse_unk") ~byte_fallback:(flag "byte_fallback")
           ~ignore_merges:(flag "ignore_merges") ())
  | "WordPiece" ->
      let vocab_list = json_to_assoc (mem "vocab") in
      let unk_token = str "unk_token" |> Option.value ~default:"[UNK]" in
      let continuing_subword_prefix =
        str "continuing_subword_prefix" |> Option.value ~default:"##"
      in
      let max_input_chars_per_word =
        match mem "max_input_chars_per_word" with
        | Jsont.Number (f, _) -> int_of_float f
        | _ -> 100
      in
      Alg_wordpiece
        (Wordpiece.create
           ~vocab:(vocab_to_hashtbl vocab_list)
           ~unk_token ~continuing_subword_prefix ~max_input_chars_per_word ())
  | "WordLevel" ->
      let vocab_list = json_to_assoc (mem "vocab") in
      let unk_token = str "unk_token" |> Option.value ~default:"[UNK]" in
      Alg_wordlevel (Word_level.create ~vocab:vocab_list ~unk_token ())
  | "Unigram" ->
      let vocab =
        json_to_list (mem "vocab")
        |> List.map (fun arr ->
            match json_to_list arr with
            | [ token; score ] ->
                (json_to_string token, json_to_float ?text score)
            | _ -> failwith "Invalid unigram vocab format")
      in
      let unk_id =
        match mem "unk_id" with
        | Jsont.Number (f, _) -> Some (int_of_float f)
        | _ -> None
      in
      Alg_unigram
        (Unigram.create ?unk_id ~byte_fallback:(flag "byte_fallback") vocab)
  | "Chars" -> Alg_chars (Chars.create ())
  | s -> failwith (strf "Unsupported model type: %s" s)

let of_json_in ?text json =
  try
    let mem name = json_mem name json in
    let normalizer =
      json_result_to_option Normalizer.of_json (mem "normalizer")
    in
    let pre =
      json_result_to_option Pre_tokenizer.of_json (mem "pre_tokenizer")
    in
    let post =
      json_result_to_option Post_processor.of_json (mem "post_processor")
    in
    let decoder = json_result_to_option Decoder.of_json (mem "decoder") in
    let added_tokens =
      match mem "added_tokens" with
      | Jsont.Array (l, _) -> List.map added_token_of_json l
      | _ -> []
    in
    let algorithm =
      alg_of_json ?text ~byte_level:(byte_level_pipeline pre)
        ~spaces:(spaces ~normalizer ~pre ~added_tokens)
        (mem "model")
    in
    Ok (create ?normalizer ?pre ?post ?decoder ~added_tokens algorithm)
  with
  | Failure msg -> Error msg
  | exn -> Error (Printexc.to_string exn)

let of_json json = of_json_in json

let is_unigram json =
  match infer_model_type (json_mem "model" json) with
  | model -> model = "Unigram"
  | exception Failure _ -> false

(* File I/O *)

let write_string_to_file path s =
  let oc = open_out path in
  Fun.protect ~finally:(fun () -> close_out oc) (fun () -> output_string oc s)

let from_file path =
  try
    let ic = open_in path in
    let s =
      Fun.protect
        ~finally:(fun () -> close_in ic)
        (fun () -> really_input_string ic (in_channel_length ic))
    in
    match Jsont_bytesrw.decode_string Jsont.json s with
    | Error e -> Error e
    | Ok json when not (is_unigram json) -> of_json json
    | Ok _ -> (
        (* A Unigram model reads its scores from their text, whose bytes the
           tree records only when decoded with locations. Locations nearly
           double the decoding of a large BPE file, so only a Unigram file is
           decoded again with them. *)
        match Jsont_bytesrw.decode_string ~locs:true Jsont.json s with
        | Error e -> Error e
        | Ok json -> of_json_in ~text:s json)
  with
  | Sys_error msg -> Error ("File error: " ^ msg)
  | exn -> Error (Printexc.to_string exn)

let save_pretrained t ~path =
  (try Sys.mkdir path 0o755 with Sys_error _ -> ());
  write_string_to_file
    (Filename.concat path "tokenizer.json")
    (Json_text.to_string (to_json t))

let export_tiktoken t ~merges_path ~vocab_path =
  match t.algorithm with
  | Alg_bpe bpe ->
      let vocab =
        alg_vocab t.algorithm
        |> List.sort (fun (_, id1) (_, id2) -> Int.compare id1 id2)
      in
      let json_str =
        match
          Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json
            (vocab_to_json vocab)
        with
        | Ok s -> s
        | Error e -> failwith ("export_tiktoken: failed to encode vocab: " ^ e)
      in
      write_string_to_file vocab_path json_str;
      let oc = open_out merges_path in
      Fun.protect
        ~finally:(fun () -> close_out oc)
        (fun () ->
          output_string oc "#version: 0.2\n";
          List.iter
            (fun (a, b) -> Printf.fprintf oc "%s %s\n" a b)
            (Bpe.get_merges bpe))
  | _ -> invalid_arg err_export_tiktoken

let save_model_files ?prefix t ~folder = alg_save t.algorithm ~folder ?prefix ()

(* Formatting *)

let pp ppf t =
  let yes_no = function Some _ -> "yes" | None -> "no" in
  Format.fprintf ppf
    "@[<1><brot %s@ vocab=%d@ normalizer=%s@ pre=%s@ post=%s@ decoder=%s>@]"
    (alg_name t.algorithm)
    (alg_vocab_size t.algorithm)
    (yes_no t.normalizer) (yes_no t.pre_tokenizer) (yes_no t.post_processor)
    (yes_no t.decoder)

(* Testing *)

let hold_cache t f =
  match t.algorithm with
  | Alg_bpe m -> Bpe.with_state m (fun _ -> f ())
  | Alg_wordpiece m -> Wordpiece.with_encoder m (fun _ -> f ())
  | Alg_unigram m -> Unigram.with_encoder m (fun _ -> f ())
  | Alg_wordlevel _ | Alg_chars _ -> f ()

let cache_set = Pretoken_cache.set_index

let cache_key = Pretoken_cache.key_words
let cache_hashes k0 k1 = (Pretoken_cache.hash k0 k1, Kernel.cache_hash k0 k1)

let run_chunks ?exec sizes work =
  run_chunks ?exec
    (Array.mapi (fun i bytes -> { first = i; stop = i + 1; bytes }) sizes)
    work

module Float_text = Float_text
