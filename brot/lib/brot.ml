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
let err_add_tokens = "only supported for word-level tokenizers"
let err_export_tiktoken = "only supported for BPE models"
let err_infer_type = "unable to infer model type from JSON"

(* Types *)

type direction = [ `Left | `Right ]

type special = {
  token : string;
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

type truncation = { max_length : int; direction : direction }
type data = [ `Files of string list | `Seq of string Seq.t ]
type sequence = { text : string; pair : string option }

type algorithm =
  | Alg_bpe of Bpe.t
  | Alg_wordpiece of Wordpiece.t
  | Alg_wordlevel of Word_level.t
  | Alg_unigram of Unigram.t
  | Alg_chars of Chars.t

type t = {
  algorithm : algorithm;
  normalizer : Normalizer.t option;
  pre_tokenizer : Pre_tokenizer.t option;
  post_processor : Post_processor.t option;
  decoder : Decoder.t option;
  specials : special list;
  special_lookup : (string, unit) Hashtbl.t;
  bos_token : string option;
  eos_token : string option;
  pad_token : string option;
  pad_id : int option;
  pad_type_id : int;
  unk_token : string option;
}

let special ?(single_word = false) ?(lstrip = false) ?(rstrip = false)
    ?(normalized = false) token =
  { token; single_word; lstrip; rstrip; normalized }

let padding ?(direction = `Right) ?pad_id ?pad_type_id ?pad_token length =
  { length; direction; pad_id; pad_type_id; pad_token }

let truncation ?(direction = `Right) max_length = { max_length; direction }

(* Algorithm dispatch *)

let alg_add_tokens algorithm tokens =
  match algorithm with
  | Alg_wordlevel model ->
      ignore (Word_level.add_tokens model tokens);
      algorithm
  | Alg_bpe _ | Alg_wordpiece _ | Alg_unigram _ | Alg_chars _ -> algorithm

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

let alg_tokenize algorithm text =
  match algorithm with
  | Alg_bpe m ->
      Bpe.tokenize m text
      |> List.map (fun (tok : Bpe.token) -> (tok.id, tok.value, tok.offsets))
  | Alg_wordpiece m ->
      Wordpiece.tokenize m text
      |> List.map (fun (tok : Wordpiece.token) ->
          (tok.id, tok.value, tok.offsets))
  | Alg_wordlevel m -> Word_level.tokenize m text
  | Alg_unigram m -> Unigram.tokenize m text
  | Alg_chars m -> Chars.tokenize m text

let alg_tokenize_ids algorithm text =
  match algorithm with
  | Alg_bpe m -> Bpe.tokenize_ids m text
  | Alg_wordpiece m -> Wordpiece.tokenize_ids m text
  | Alg_wordlevel m -> Word_level.tokenize_ids m text
  | Alg_unigram m ->
      Unigram.tokenize m text
      |> List.map (fun (id, _, _) -> id)
      |> Array.of_list
  | Alg_chars m ->
      Chars.tokenize m text |> List.map (fun (id, _, _) -> id) |> Array.of_list

let alg_name = function
  | Alg_bpe _ -> "BPE"
  | Alg_wordpiece _ -> "WordPiece"
  | Alg_wordlevel _ -> "WordLevel"
  | Alg_unigram _ -> "Unigram"
  | Alg_chars _ -> "Chars"

let vocab_to_hashtbl vocab =
  let tbl = Hashtbl.create (List.length vocab) in
  List.iter (fun (token, id) -> Hashtbl.add tbl token id) vocab;
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

let collect_unique_tokens specials ~bos_token ~eos_token ~pad_token ~unk_token =
  let items =
    List.map (fun (s : special) -> s.token) specials
    @ List.filter_map Fun.id [ bos_token; eos_token; pad_token; unk_token ]
  in
  dedup_by Fun.id items

let build_special_lookup specials ~bos_token ~eos_token ~pad_token ~unk_token =
  let tokens =
    collect_unique_tokens specials ~bos_token ~eos_token ~pad_token ~unk_token
  in
  let table = Hashtbl.create (List.length tokens) in
  List.iter (fun t -> Hashtbl.replace table t ()) tokens;
  table

(* Construction *)

let create ?normalizer ?pre ?post ?decoder ?(specials = []) ?bos_token
    ?eos_token ?pad_token ?unk_token algorithm =
  let all_tokens =
    collect_unique_tokens specials ~bos_token ~eos_token ~pad_token ~unk_token
  in
  let algorithm = alg_add_tokens algorithm all_tokens in
  let special_lookup =
    build_special_lookup specials ~bos_token ~eos_token ~pad_token ~unk_token
  in
  let pad_id = Option.bind pad_token (alg_token_to_id algorithm) in
  {
    algorithm;
    normalizer;
    pre_tokenizer = pre;
    post_processor = post;
    decoder;
    specials;
    special_lookup;
    bos_token;
    eos_token;
    pad_token;
    pad_id;
    pad_type_id = 0;
    unk_token;
  }

(* Accessors *)

let normalizer t = t.normalizer
let pre_tokenizer t = t.pre_tokenizer
let post_processor t = t.post_processor
let decoder t = t.decoder
let specials t = t.specials
let bos_token t = t.bos_token
let eos_token t = t.eos_token
let pad_token t = t.pad_token
let unk_token t = t.unk_token
let vocab t = alg_vocab t.algorithm
let vocab_size t = alg_vocab_size t.algorithm
let token_to_id t token = alg_token_to_id t.algorithm token
let id_to_token t id = alg_id_to_token t.algorithm id

(* Algorithm constructors *)

let bpe ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token ?vocab ?merges ?cache_capacity ?dropout
    ?continuing_subword_prefix ?end_of_word_suffix ?fuse_unk ?byte_fallback
    ?ignore_merges () =
  let vocab_tbl =
    match vocab with None -> Hashtbl.create 100 | Some v -> vocab_to_hashtbl v
  in
  let algorithm =
    Alg_bpe
      (Bpe.create ~vocab:vocab_tbl
         ~merges:(Option.value merges ~default:[])
         ?cache_capacity ?dropout ?unk_token ?continuing_subword_prefix
         ?end_of_word_suffix ?fuse_unk ?byte_fallback ?ignore_merges ())
  in
  create ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let wordpiece ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token ?vocab ?continuing_subword_prefix
    ?max_input_chars_per_word () =
  let vocab_tbl =
    match vocab with None -> Hashtbl.create 100 | Some v -> vocab_to_hashtbl v
  in
  let algorithm =
    Alg_wordpiece
      (Wordpiece.create ~vocab:vocab_tbl ?unk_token ?continuing_subword_prefix
         ?max_input_chars_per_word ())
  in
  create ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let word_level ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token ?vocab () =
  let pre =
    match pre with Some _ -> pre | None -> Some (Pre_tokenizer.whitespace ())
  in
  let algorithm = Alg_wordlevel (Word_level.create ?vocab ?unk_token ()) in
  create ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let unigram ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token ?vocab () =
  let algorithm =
    Alg_unigram (Unigram.create (Option.value vocab ~default:[]))
  in
  create ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let chars ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token () =
  let algorithm = Alg_chars (Chars.create ()) in
  create ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let from_model_file ~vocab ?merges ?normalizer ?pre ?post ?decoder ?specials
    ?bos_token ?eos_token ?pad_token ?unk_token () =
  let algorithm =
    match merges with
    | Some merges_file ->
        Alg_bpe (Bpe.from_files ~vocab_file:vocab ~merges_file)
    | None -> Alg_wordpiece (Wordpiece.from_file ~vocab_file:vocab)
  in
  create ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
    ?pad_token ?unk_token algorithm

let add_tokens t tokens =
  match t.algorithm with
  | Alg_wordlevel model ->
      let vocab = Word_level.get_vocab model in
      let new_model = Word_level.create ~vocab ?unk_token:t.unk_token () in
      ignore (Word_level.add_tokens new_model tokens);
      { t with algorithm = Alg_wordlevel new_model }
  | Alg_bpe _ | Alg_wordpiece _ | Alg_unigram _ | Alg_chars _ ->
      invalid_arg err_add_tokens

(* Encoding *)

let encode_text t text =
  let normalized =
    match t.normalizer with Some n -> Normalizer.apply n text | None -> text
  in
  let pre_tokens =
    match t.pre_tokenizer with
    | Some pre -> Pre_tokenizer.pre_tokenize pre normalized
    | None -> [ (normalized, (0, String.length normalized)) ]
  in
  match (t.algorithm, pre_tokens) with
  | Alg_bpe m, [ (fragment, _) ] -> Bpe.tokenize_encoding m fragment ~type_id:0
  | Alg_wordpiece m, _ ->
      Wordpiece.tokenize_spans_encoding m pre_tokens ~type_id:0
  | _ ->
      pre_tokens
      |> List.concat_map (fun (fragment, _) ->
          alg_tokenize t.algorithm fragment)
      |> Encoding.from_tokens ~type_id:0

let post_process t ~add_special primary pair =
  match t.post_processor with
  | None ->
      if Option.is_some pair then invalid_arg err_pair_no_post else primary
  | Some processor ->
      Post_processor.process processor ?pair primary
        ~add_special_tokens:add_special

let encode_single t ~add_special_tokens ~truncation seq =
  let primary = encode_text t seq.text in
  let pair = Option.map (encode_text t) seq.pair in
  let processed = post_process t ~add_special:add_special_tokens primary pair in
  match truncation with
  | None -> processed
  | Some { max_length; direction } ->
      Encoding.truncate processed ~max_length ~stride:0 ~direction

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
        match alg_token_to_id t.algorithm token with
        | Some id -> id
        | None -> invalid_arg (err_pad_not_in_vocab token))
  in
  let type_id = Option.value cfg.pad_type_id ~default:t.pad_type_id in
  (token, id, type_id)

let round_up_to_multiple n m = if n mod m = 0 then n else (n + m - 1) / m * m

let apply_padding t encodings = function
  | None -> encodings
  | Some cfg -> (
      let pad_token, pad_id, pad_type_id = resolve_pad t cfg in
      let direction = cfg.direction in
      let pad enc target =
        if Encoding.length enc >= target then enc
        else
          Encoding.pad enc ~target_length:target ~pad_id ~pad_type_id ~pad_token
            ~direction
      in
      match cfg.length with
      | `Fixed n -> List.map (fun enc -> pad enc n) encodings
      | `Batch_longest ->
          let max_len =
            List.fold_left
              (fun acc enc -> max acc (Encoding.length enc))
              0 encodings
          in
          List.map (fun enc -> pad enc max_len) encodings
      | `To_multiple m ->
          if m <= 0 then encodings
          else
            List.map
              (fun enc ->
                pad enc (round_up_to_multiple (Encoding.length enc) m))
              encodings)

(* Parallel batch encoding *)

let encode_parallel t sequences ~add_special_tokens ~truncation =
  let arr = Array.of_list sequences in
  let n = Array.length arr in
  let results =
    Array.make n (encode_single t ~add_special_tokens ~truncation arr.(0))
  in
  let num_domains = min n (Domain.recommended_domain_count ()) in
  if num_domains <= 1 then
    for i = 1 to n - 1 do
      results.(i) <- encode_single t ~add_special_tokens ~truncation arr.(i)
    done
  else begin
    let chunk_size = n / num_domains in
    let remainder = n mod num_domains in
    let domains =
      Array.init (num_domains - 1) (fun d ->
          let start = ((d + 1) * chunk_size) + min (d + 1) remainder in
          let len = chunk_size + if d + 1 < remainder then 1 else 0 in
          Domain.spawn (fun () ->
              for i = start to start + len - 1 do
                results.(i) <-
                  encode_single t ~add_special_tokens ~truncation arr.(i)
              done))
    in
    let main_len = chunk_size + if 0 < remainder then 1 else 0 in
    for i = 1 to main_len - 1 do
      results.(i) <- encode_single t ~add_special_tokens ~truncation arr.(i)
    done;
    Array.iter Domain.join domains
  end;
  Array.to_list results

let encode_sequences t sequences ~add_special_tokens ~padding ~truncation =
  let n = List.length sequences in
  let raw =
    if n >= 4 then encode_parallel t sequences ~add_special_tokens ~truncation
    else List.map (encode_single t ~add_special_tokens ~truncation) sequences
  in
  apply_padding t raw padding

let encode t ?pair ?(add_special_tokens = true) ?padding ?truncation text =
  match
    encode_sequences t
      [ { text; pair } ]
      ~add_special_tokens ~padding ~truncation
  with
  | [ encoding ] -> encoding
  | _ -> assert false

let encode_batch t ?(add_special_tokens = true) ?padding ?truncation = function
  | [] -> []
  | texts ->
      let sequences = List.map (fun text -> { text; pair = None }) texts in
      encode_sequences t sequences ~add_special_tokens ~padding ~truncation

let encode_pairs_batch t ?(add_special_tokens = true) ?padding ?truncation =
  function
  | [] -> []
  | pairs ->
      let sequences =
        List.map (fun (text, pair) -> { text; pair = Some pair }) pairs
      in
      encode_sequences t sequences ~add_special_tokens ~padding ~truncation

let encode_ids t ?pair ?add_special_tokens ?padding ?truncation text =
  let use_fast_path =
    Option.is_none pair
    && (add_special_tokens = None || add_special_tokens = Some false)
    && Option.is_none padding && Option.is_none truncation
    && Option.is_none t.post_processor
  in
  if not use_fast_path then
    Encoding.ids (encode t ?pair ?add_special_tokens ?padding ?truncation text)
  else
    let normalized =
      match t.normalizer with Some n -> Normalizer.apply n text | None -> text
    in
    let pre_tokens =
      match t.pre_tokenizer with
      | Some pre -> Pre_tokenizer.pre_tokenize pre normalized
      | None -> [ (normalized, (0, String.length normalized)) ]
    in
    let id_arrays =
      List.map
        (fun (fragment, _) -> alg_tokenize_ids t.algorithm fragment)
        pre_tokens
    in
    let total_len =
      List.fold_left (fun acc a -> acc + Array.length a) 0 id_arrays
    in
    let result = Array.make total_len 0 in
    let pos = ref 0 in
    List.iter
      (fun a ->
        let len = Array.length a in
        Array.blit a 0 result !pos len;
        pos := !pos + len)
      id_arrays;
    result

(* Decoding *)

let decode t ?(skip_special_tokens = false) ids =
  let tokens =
    Array.to_list ids
    |> List.filter_map (fun id ->
        match alg_id_to_token t.algorithm id with
        | None -> None
        | Some token
          when skip_special_tokens && Hashtbl.mem t.special_lookup token ->
            None
        | Some token -> Some token)
  in
  match t.decoder with
  | Some decoder -> Decoder.decode decoder tokens
  | None -> (
      match t.algorithm with
      | Alg_wordlevel _ -> String.concat " " tokens
      | _ -> String.concat "" tokens)

let decode_batch t ?(skip_special_tokens = false) id_lists =
  List.map (decode t ~skip_special_tokens) id_lists

(* Training *)

let special_tokens_for_training init specials =
  let items =
    (match specials with
      | Some sl -> List.map (fun (s : special) -> s.token) sl
      | None -> [])
    @
    match init with
    | Some tok -> List.map (fun (s : special) -> s.token) tok.specials
    | None -> []
  in
  dedup_by Fun.id items

let merge_specials_from_training ~user_specials ~trained_tokens =
  let items =
    (match user_specials with Some sl -> sl | None -> [])
    @ List.map special trained_tokens
  in
  dedup_by (fun (s : special) -> s.token) items

let data_to_strings = function
  | `Files files ->
      let lines = ref [] in
      List.iter
        (fun file ->
          let ic = open_in file in
          (try
             while true do
               lines := input_line ic :: !lines
             done
           with End_of_file -> ());
          close_in ic)
        files;
      List.rev !lines
  | `Seq seq -> List.of_seq seq

let initial_alphabet_of strs =
  List.map (fun s -> if String.length s > 0 then s.[0] else ' ') strs

let train_bpe ?init ?normalizer ?pre ?post ?decoder ?specials ?bos_token
    ?eos_token ?pad_token ?unk_token ?(vocab_size = 30000) ?(min_frequency = 0)
    ?limit_alphabet ?initial_alphabet ?continuing_subword_prefix
    ?end_of_word_suffix ?(show_progress = true) ?max_token_length data =
  let special_tokens = special_tokens_for_training init specials in
  let initial_alphabet =
    Option.value initial_alphabet ~default:[] |> initial_alphabet_of
  in
  let limit_alphabet = Some (Option.value limit_alphabet ~default:1000) in
  let texts = data_to_strings data in
  let existing_bpe =
    Option.bind init (fun t ->
        match t.algorithm with Alg_bpe m -> Some m | _ -> None)
  in
  let trained_model, result_specials =
    Bpe.train ~min_frequency ~vocab_size ~show_progress ~special_tokens
      ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
      ~end_of_word_suffix ~max_token_length texts existing_bpe
  in
  let all_specials =
    merge_specials_from_training ~user_specials:specials
      ~trained_tokens:result_specials
  in
  create ?normalizer ?pre ?post ?decoder ~specials:all_specials ?bos_token
    ?eos_token ?pad_token ?unk_token (Alg_bpe trained_model)

let train_wordpiece ?init ?normalizer ?pre ?post ?decoder ?specials ?bos_token
    ?eos_token ?pad_token ?unk_token ?(vocab_size = 30000) ?(min_frequency = 0)
    ?limit_alphabet ?initial_alphabet ?(continuing_subword_prefix = "##")
    ?end_of_word_suffix ?(show_progress = true) data =
  let special_tokens = special_tokens_for_training init specials in
  let initial_alphabet =
    Option.value initial_alphabet ~default:[] |> initial_alphabet_of
  in
  let limit_alphabet = Some (Option.value limit_alphabet ~default:1000) in
  let texts = data_to_strings data in
  let existing_wp =
    Option.bind init (fun t ->
        match t.algorithm with Alg_wordpiece m -> Some m | _ -> None)
  in
  let trained_model, result_specials =
    Wordpiece.train ~min_frequency ~vocab_size ~show_progress ~special_tokens
      ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
      ~end_of_word_suffix texts existing_wp
  in
  let all_specials =
    merge_specials_from_training ~user_specials:specials
      ~trained_tokens:result_specials
  in
  create ?normalizer ?pre ?post ?decoder ~specials:all_specials ?bos_token
    ?eos_token ?pad_token ?unk_token (Alg_wordpiece trained_model)

let train_wordlevel ?init ?normalizer ?pre ?post ?decoder ?specials ?bos_token
    ?eos_token ?pad_token ?unk_token ?(vocab_size = 30000) ?(min_frequency = 0)
    ?(show_progress = true) data =
  let special_tokens = special_tokens_for_training init specials in
  let texts = data_to_strings data in
  let existing_wl =
    Option.bind init (fun t ->
        match t.algorithm with Alg_wordlevel m -> Some m | _ -> None)
  in
  let trained_model, result_specials =
    Word_level.train ~vocab_size ~min_frequency ~show_progress ~special_tokens
      texts existing_wl
  in
  let all_specials =
    merge_specials_from_training ~user_specials:specials
      ~trained_tokens:result_specials
  in
  create ?normalizer ?pre ?post ?decoder ~specials:all_specials ?bos_token
    ?eos_token ?pad_token ?unk_token (Alg_wordlevel trained_model)

let train_unigram ?init ?normalizer ?pre ?post ?decoder ?specials ?bos_token
    ?eos_token ?pad_token ?unk_token ?(vocab_size = 8000)
    ?(show_progress = true) ?(shrinking_factor = 0.75) ?(max_piece_length = 16)
    ?(n_sub_iterations = 2) data =
  let special_tokens = special_tokens_for_training init specials in
  let texts = data_to_strings data in
  let existing_ug =
    Option.bind init (fun t ->
        match t.algorithm with Alg_unigram m -> Some m | _ -> None)
  in
  let trained_model, result_specials =
    Unigram.train ~vocab_size ~show_progress ~special_tokens ~shrinking_factor
      ~unk_token ~max_piece_length ~n_sub_iterations texts existing_ug
  in
  let all_specials =
    merge_specials_from_training ~user_specials:specials
      ~trained_tokens:result_specials
  in
  create ?normalizer ?pre ?post ?decoder ~specials:all_specials ?bos_token
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

let special_of_json json =
  let mem name = json_mem name json in
  let to_bool = function Jsont.Bool (b, _) -> b | _ -> false in
  let to_str = function
    | Jsont.String (s, _) -> s
    | _ -> failwith "expected string"
  in
  {
    token = to_str (mem "content");
    single_word = to_bool (mem "single_word");
    lstrip = to_bool (mem "lstrip");
    rstrip = to_bool (mem "rstrip");
    normalized = to_bool (mem "normalized");
  }

let added_token_to_json ~id (s : special) =
  json_obj
    [
      ("id", Jsont.Json.int id);
      ("content", Jsont.Json.string s.token);
      ("single_word", Jsont.Json.bool s.single_word);
      ("lstrip", Jsont.Json.bool s.lstrip);
      ("rstrip", Jsont.Json.bool s.rstrip);
      ("normalized", Jsont.Json.bool s.normalized);
      ("special", Jsont.Json.bool true);
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
          ("dropout", Jsont.Json.null ());
          ("unk_token", json_option_of Jsont.Json.string (Bpe.get_unk_token bpe));
          ( "continuing_subword_prefix",
            json_option_of Jsont.Json.string
              (Bpe.get_continuing_subword_prefix bpe) );
          ( "end_of_word_suffix",
            json_option_of Jsont.Json.string (Bpe.get_end_of_word_suffix bpe) );
          ("fuse_unk", Jsont.Json.bool false);
          ("byte_fallback", Jsont.Json.bool false);
          ("ignore_merges", Jsont.Json.bool false);
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
            Jsont.Json.list [ Jsont.Json.string token; Jsont.Json.number score ])
        |> Jsont.Json.list
      in
      json_obj
        [
          ("type", Jsont.Json.string "Unigram");
          ("unk_id", Jsont.Json.null ());
          ("vocab", vocab_json);
        ]
  | Alg_chars _ ->
      json_obj [ ("type", Jsont.Json.string "Chars"); ("vocab", json_obj []) ]

let to_json (t : t) =
  let vocab_list = alg_vocab t.algorithm in
  let added_tokens =
    t.specials
    |> List.filter_map (fun spec ->
        List.find_opt (fun (token, _) -> token = spec.token) vocab_list
        |> Option.map (fun (_, id) -> added_token_to_json ~id spec))
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

let json_to_float = function
  | Jsont.Number (f, _) -> f
  | _ -> failwith "Expected number"

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

let alg_of_json mj =
  let mem name = json_mem name mj in
  let str name = json_string_or_null (mem name) in
  match infer_model_type mj with
  | "BPE" ->
      let vocab_list = json_to_assoc (mem "vocab") in
      let merges = json_to_list (mem "merges") |> List.map parse_merge in
      Alg_bpe
        (Bpe.create
           ~vocab:(vocab_to_hashtbl vocab_list)
           ~merges ?unk_token:(str "unk_token")
           ?continuing_subword_prefix:(str "continuing_subword_prefix")
           ?end_of_word_suffix:(str "end_of_word_suffix") ())
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
            | [ token; score ] -> (json_to_string token, json_to_float score)
            | _ -> failwith "Invalid unigram vocab format")
      in
      Alg_unigram (Unigram.create vocab)
  | "Chars" -> Alg_chars (Chars.create ())
  | s -> failwith (strf "Unsupported model type: %s" s)

let from_json json =
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
    let algorithm = alg_of_json (mem "model") in
    let added_tokens =
      match mem "added_tokens" with
      | Jsont.Array (l, _) -> List.map special_of_json l
      | _ -> []
    in
    Ok (create ?normalizer ?pre ?post ?decoder ~specials:added_tokens algorithm)
  with
  | Failure msg -> Error msg
  | exn -> Error (Printexc.to_string exn)

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
    | Ok json -> from_json json
    | Error e -> Error e
  with
  | Sys_error msg -> Error ("File error: " ^ msg)
  | exn -> Error (Printexc.to_string exn)

let save_pretrained t ~path =
  (try Sys.mkdir path 0o755 with Sys_error _ -> ());
  let json_str =
    match
      Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json (to_json t)
    with
    | Ok s -> s
    | Error e -> failwith ("save_pretrained: failed to encode JSON: " ^ e)
  in
  write_string_to_file (Filename.concat path "tokenizer.json") json_str

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

let save_model_files t ~folder ?prefix () =
  alg_save t.algorithm ~folder ?prefix ()

(* Formatting *)

let pp ppf t =
  let yes_no = function Some _ -> "yes" | None -> "no" in
  Format.fprintf ppf
    "@[<1><brot %s@ vocab=%d@ normalizer=%s@ pre=%s@ post=%s@ decoder=%s>@]"
    (alg_name t.algorithm)
    (alg_vocab_size t.algorithm)
    (yes_no t.normalizer) (yes_no t.pre_tokenizer) (yes_no t.post_processor)
    (yes_no t.decoder)
