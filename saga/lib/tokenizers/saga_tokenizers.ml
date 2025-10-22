(** Ergonomic tokenizer API built on Saga's primitives. *)

module Unicode = Unicode
module Normalizers = Normalizers
module Pre_tokenizers = Pre_tokenizers
module Processors = Processors
module Decoders = Decoders
module Encoding = Encoding

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

type data =
  [ `Files of string list
  | `Seq of string Seq.t
  | `Iterator of unit -> string option ]

type sequence = { text : string; pair : string option }
type pad_spec = { token : string; id : int option; type_id : int option }

(* Special token constructors *)
module Special = struct
  let make ?(single_word = false) ?(lstrip = false) ?(rstrip = false)
      ?(normalized = false) token =
    { token; single_word; lstrip; rstrip; normalized }

  let pad token = make token
  let unk token = make token
  let bos token = make token
  let eos token = make token
  let cls token = make token
  let sep token = make token
  let mask token = make token
end

type algorithm =
  | Alg_bpe of Bpe.t
  | Alg_wordpiece of Wordpiece.t
  | Alg_wordlevel of Word_level.t
  | Alg_unigram of Unigram.t
  | Alg_chars of Chars.t

type special_config = {
  bos : string option;
  eos : string option;
  pad : pad_spec option;
  unk : string option;
  extras : string list;
}

let add_tokens_to_algorithm algorithm tokens =
  match algorithm with
  | Alg_bpe _model ->
      (* BPE doesn't support dynamically adding tokens *)
      algorithm
  | Alg_wordpiece _model ->
      (* WordPiece doesn't support dynamically adding tokens *)
      algorithm
  | Alg_wordlevel model ->
      ignore (Word_level.add_tokens model tokens);
      algorithm
  | Alg_unigram _ -> algorithm
  | Alg_chars _ ->
      (* Chars doesn't have a vocabulary to add to *)
      algorithm

let token_to_id_algorithm algorithm token =
  match algorithm with
  | Alg_bpe model -> Bpe.token_to_id model token
  | Alg_wordpiece model -> Wordpiece.token_to_id model token
  | Alg_wordlevel model -> Word_level.token_to_id model token
  | Alg_unigram model -> Unigram.token_to_id model token
  | Alg_chars model -> Chars.token_to_id model token

let id_to_token_algorithm algorithm id =
  match algorithm with
  | Alg_bpe model -> Bpe.id_to_token model id
  | Alg_wordpiece model -> Wordpiece.id_to_token model id
  | Alg_wordlevel model -> Word_level.id_to_token model id
  | Alg_unigram model -> Unigram.id_to_token model id
  | Alg_chars model -> Chars.id_to_token model id

let vocab_algorithm algorithm =
  match algorithm with
  | Alg_bpe model -> Bpe.get_vocab model
  | Alg_wordpiece model -> Wordpiece.get_vocab model
  | Alg_wordlevel model -> Word_level.get_vocab model
  | Alg_unigram model ->
      Unigram.get_vocab model |> List.mapi (fun idx (token, _) -> (token, idx))
  | Alg_chars model -> Chars.get_vocab model

let vocab_size_algorithm algorithm =
  match algorithm with
  | Alg_bpe model -> Bpe.get_vocab_size model
  | Alg_wordpiece model -> Wordpiece.get_vocab_size model
  | Alg_wordlevel model -> Word_level.get_vocab_size model
  | Alg_unigram model -> Unigram.get_vocab_size model
  | Alg_chars model -> Chars.get_vocab_size model

let save_algorithm algorithm ~folder ?prefix () =
  match algorithm with
  | Alg_bpe model ->
      Bpe.save model ~path:folder ?name:prefix ();
      let vocab_file =
        match prefix with
        | Some n -> Filename.concat folder (Printf.sprintf "%s-vocab.json" n)
        | None -> Filename.concat folder "vocab.json"
      in
      let merges_file =
        match prefix with
        | Some n -> Filename.concat folder (Printf.sprintf "%s-merges.txt" n)
        | None -> Filename.concat folder "merges.txt"
      in
      [ vocab_file; merges_file ]
  | Alg_wordpiece model -> [ Wordpiece.save model ~path:folder ?name:prefix () ]
  | Alg_wordlevel model -> Word_level.save model ~folder ()
  | Alg_unigram model -> Unigram.save model ~folder ()
  | Alg_chars model -> Chars.save model ~folder ()

let tokenize_algorithm algorithm text =
  match algorithm with
  | Alg_bpe model ->
      Bpe.tokenize model text
      |> List.map (fun (tok : Bpe.token) -> (tok.id, tok.value, tok.offsets))
  | Alg_wordpiece model ->
      Wordpiece.tokenize model text
      |> List.map (fun (tok : Wordpiece.token) ->
             (tok.id, tok.value, tok.offsets))
  | Alg_wordlevel model -> Word_level.tokenize model text
  | Alg_unigram model -> Unigram.tokenize model text
  | Alg_chars model -> Chars.tokenize model text

type pad_runtime = { token : string; id : int; type_id : int }

let empty_special_config =
  { bos = None; eos = None; pad = None; unk = None; extras = [] }

let special_list_to_config (specials : special list) =
  let add_extra extras token =
    if List.exists (( = ) token) extras then extras else extras @ [ token ]
  in
  List.fold_left
    (fun acc (sp : special) ->
      { acc with extras = add_extra acc.extras sp.token })
    empty_special_config specials

let merge_configs base extra =
  let merge_opt prefer new_value =
    match new_value with None -> prefer | Some _ -> new_value
  in
  let merged_pad =
    match (base.pad, extra.pad) with
    | None, None -> None
    | Some pad, None -> Some pad
    | None, Some pad -> Some pad
    | Some base_pad, Some new_pad ->
        let token_changed = not (String.equal base_pad.token new_pad.token) in
        let resolved_id =
          if token_changed then new_pad.id else merge_opt base_pad.id new_pad.id
        in
        let resolved_type_id =
          if token_changed then
            match new_pad.type_id with
            | Some _ as t -> t
            | None -> base_pad.type_id
          else merge_opt base_pad.type_id new_pad.type_id
        in
        Some
          {
            token = new_pad.token;
            id = resolved_id;
            type_id = resolved_type_id;
          }
  in
  let merged_extras =
    List.fold_left
      (fun acc token ->
        if List.exists (( = ) token) acc then acc else acc @ [ token ])
      base.extras extra.extras
  in
  {
    bos = merge_opt base.bos extra.bos;
    eos = merge_opt base.eos extra.eos;
    pad = merged_pad;
    unk = merge_opt base.unk extra.unk;
    extras = merged_extras;
  }

let config_to_special_list config =
  let acc = ref [] in
  let add_special token = acc := Special.make token :: !acc in
  (match config.bos with Some token -> add_special token | None -> ());
  (match config.eos with Some token -> add_special token | None -> ());
  (match config.pad with
  | Some { token; id = _; type_id = _ } -> add_special token
  | None -> ());
  (match config.unk with Some token -> add_special token | None -> ());
  List.iter add_special config.extras;
  List.rev !acc

let tokens_of_config config =
  let tokens = ref [] in
  (match config.bos with
  | Some token -> tokens := token :: !tokens
  | None -> ());
  (match config.eos with
  | Some token -> tokens := token :: !tokens
  | None -> ());
  (match config.pad with
  | Some { token; _ } -> tokens := token :: !tokens
  | None -> ());
  (match config.unk with
  | Some token -> tokens := token :: !tokens
  | None -> ());
  tokens := List.rev_append config.extras !tokens;
  !tokens

let build_special_lookup config =
  let table = Hashtbl.create 16 in
  (match config.bos with
  | Some token -> Hashtbl.replace table token ()
  | None -> ());
  (match config.eos with
  | Some token -> Hashtbl.replace table token ()
  | None -> ());
  (match config.pad with
  | Some { token; _ } -> Hashtbl.replace table token ()
  | None -> ());
  (match config.unk with
  | Some token -> Hashtbl.replace table token ()
  | None -> ());
  List.iter (fun token -> Hashtbl.replace table token ()) config.extras;
  table

let ensure_specials algorithm config =
  let tokens = tokens_of_config config in
  let algorithm = add_tokens_to_algorithm algorithm tokens in
  let resolve_pad (pad_opt : pad_spec option) =
    match pad_opt with
    | None -> (None, None)
    | Some (ps : pad_spec) ->
        let id =
          match ps.id with
          | Some id_val -> id_val
          | None -> (
              match token_to_id_algorithm algorithm ps.token with
              | Some id_val -> id_val
              | None ->
                  invalid_arg
                    (Printf.sprintf "Pad token '%s' not present in vocabulary"
                       ps.token))
        in
        let type_id = Option.value ps.type_id ~default:0 in
        let pad_spec_result : pad_spec =
          { token = ps.token; id = Some id; type_id = Some type_id }
        in
        let pad_runtime_result : pad_runtime =
          { token = ps.token; id; type_id }
        in
        (Some pad_spec_result, Some pad_runtime_result)
  in
  let pad_spec, pad_runtime = resolve_pad config.pad in
  (algorithm, { config with pad = pad_spec }, pad_runtime)

let direction_to_trunc : direction -> Encoding.truncation_direction = function
  | `Left -> Encoding.Left
  | `Right -> Encoding.Right

let direction_to_pad : direction -> Encoding.padding_direction = function
  | `Left -> Encoding.Left
  | `Right -> Encoding.Right

let tokens_to_encoding tokens = Encoding.from_tokens tokens ~type_id:0

let encoding_to_processor (e : Encoding.t) : Processors.encoding =
  {
    ids = Encoding.get_ids e;
    type_ids = Encoding.get_type_ids e;
    tokens = Encoding.get_tokens e;
    offsets = Encoding.get_offsets e;
    special_tokens_mask = Encoding.get_special_tokens_mask e;
    attention_mask = Encoding.get_attention_mask e;
    overflowing = [];
    sequence_ranges = [];
  }

let processor_to_encoding (pe : Processors.encoding) : Encoding.t =
  let seq = Hashtbl.create 1 in
  Encoding.create ~ids:pe.ids ~type_ids:pe.type_ids ~tokens:pe.tokens
    ~words:(Array.make (Array.length pe.ids) None)
    ~offsets:pe.offsets ~special_tokens_mask:pe.special_tokens_mask
    ~attention_mask:pe.attention_mask ~overflowing:[] ~sequence_ranges:seq

module Tokenizer = struct
  type t = {
    mutable algorithm : algorithm;
    mutable normalizer : Normalizers.t option;
    mutable pre_tokenizer : Pre_tokenizers.t option;
    mutable post_processor : Processors.t option;
    mutable decoder : Decoders.t option;
    mutable specials_config : special_config;
    mutable pad_runtime : pad_runtime option;
    special_lookup : (string, unit) Hashtbl.t;
  }

  let create ?normalizer ?pre ?post ?decoder ?(specials = []) algorithm =
    let config = special_list_to_config specials in
    let algorithm, config, pad_runtime = ensure_specials algorithm config in
    let lookup = build_special_lookup config in
    {
      algorithm;
      normalizer;
      pre_tokenizer = pre;
      post_processor = post;
      decoder;
      specials_config = config;
      pad_runtime;
      special_lookup = lookup;
    }

  let set_special_lookup t =
    Hashtbl.clear t.special_lookup;
    Hashtbl.iter
      (fun token () -> Hashtbl.replace t.special_lookup token ())
      (build_special_lookup t.specials_config)

  let normalizer t = t.normalizer

  let with_normalizer t normalizer =
    t.normalizer <- normalizer;
    t

  let pre_tokenizer t = t.pre_tokenizer

  let with_pre_tokenizer t pre_tokenizer =
    t.pre_tokenizer <- pre_tokenizer;
    t

  let post_processor t = t.post_processor

  let with_post_processor t post_processor =
    t.post_processor <- post_processor;
    t

  let decoder t = t.decoder

  let with_decoder t decoder =
    t.decoder <- decoder;
    t

  let specials t = config_to_special_list t.specials_config

  let with_specials t specials =
    let config = special_list_to_config specials in
    let algorithm, config, pad_runtime = ensure_specials t.algorithm config in
    t.algorithm <- algorithm;
    t.specials_config <- config;
    t.pad_runtime <- pad_runtime;
    set_special_lookup t;
    t

  let add_specials t specials =
    if specials = [] then t
    else
      let new_config =
        merge_configs t.specials_config (special_list_to_config specials)
      in
      let algorithm, config, pad_runtime =
        ensure_specials t.algorithm new_config
      in
      t.algorithm <- algorithm;
      t.specials_config <- config;
      t.pad_runtime <- pad_runtime;
      set_special_lookup t;
      t

  (* Special token role accessors *)
  let bos_token t = t.specials_config.bos

  let set_bos_token t bos =
    t.specials_config <- { t.specials_config with bos };
    t

  let eos_token t = t.specials_config.eos

  let set_eos_token t eos =
    t.specials_config <- { t.specials_config with eos };
    t

  let pad_token t =
    match t.specials_config.pad with
    | None -> None
    | Some { token; _ } -> Some token

  let set_pad_token t pad =
    let pad_spec : pad_spec option =
      match pad with
      | None -> None
      | Some token ->
          Some
            { token; id = (None : int option); type_id = (None : int option) }
    in
    let new_config = { t.specials_config with pad = pad_spec } in
    let algorithm, config, pad_runtime =
      ensure_specials t.algorithm new_config
    in
    t.algorithm <- algorithm;
    t.specials_config <- config;
    t.pad_runtime <- pad_runtime;
    t

  let unk_token t = t.specials_config.unk

  let set_unk_token t unk =
    t.specials_config <- { t.specials_config with unk };
    t

  let vocab t = vocab_algorithm t.algorithm
  let vocab_size t = vocab_size_algorithm t.algorithm
  let token_to_id t token = token_to_id_algorithm t.algorithm token
  let id_to_token t id = id_to_token_algorithm t.algorithm id

  let bpe ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
      ?pad_token ?unk_token ?vocab ?merges ?cache_capacity ?dropout
      ?continuing_subword_prefix ?end_of_word_suffix ?fuse_unk ?byte_fallback
      ?ignore_merges () =
    let vocab_tbl =
      match vocab with
      | None -> Hashtbl.create 100
      | Some vocab_list ->
          let tbl = Hashtbl.create (List.length vocab_list) in
          List.iter (fun (token, id) -> Hashtbl.add tbl token id) vocab_list;
          tbl
    in
    let merges = Option.value merges ~default:[] in
    let cache_capacity = Option.value cache_capacity ~default:10000 in
    let fuse_unk = Option.value fuse_unk ~default:false in
    let byte_fallback = Option.value byte_fallback ~default:false in
    let ignore_merges = Option.value ignore_merges ~default:false in
    let algorithm =
      Alg_bpe
        (Bpe.create
           {
             vocab = vocab_tbl;
             merges;
             cache_capacity;
             dropout;
             unk_token;
             continuing_subword_prefix;
             end_of_word_suffix;
             fuse_unk;
             byte_fallback;
             ignore_merges;
           })
    in
    let tok = create ?normalizer ?pre ?post ?decoder ?specials algorithm in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let wordpiece ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
      ?pad_token ?unk_token ?vocab ?continuing_subword_prefix
      ?max_input_chars_per_word () =
    let vocab_tbl =
      match vocab with
      | None -> Hashtbl.create 100
      | Some vocab_list ->
          let tbl = Hashtbl.create (List.length vocab_list) in
          List.iter (fun (token, id) -> Hashtbl.add tbl token id) vocab_list;
          tbl
    in
    let unk_token_model = Option.value unk_token ~default:"[UNK]" in
    let continuing_subword_prefix =
      Option.value continuing_subword_prefix ~default:"##"
    in
    let max_input_chars_per_word =
      Option.value max_input_chars_per_word ~default:100
    in
    let algorithm =
      Alg_wordpiece
        (Wordpiece.create
           {
             vocab = vocab_tbl;
             unk_token = unk_token_model;
             continuing_subword_prefix;
             max_input_chars_per_word;
           })
    in
    let tok = create ?normalizer ?pre ?post ?decoder ?specials algorithm in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let word_level ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
      ?pad_token ?unk_token ?vocab () =
    (* Default to whitespace pre-tokenizer for word-level tokenization *)
    let pre =
      match pre with
      | Some _ -> pre
      | None -> Some (Pre_tokenizers.whitespace ())
    in
    let algorithm = Alg_wordlevel (Word_level.create ?vocab ?unk_token ()) in
    let tok = create ?normalizer ?pre ?post ?decoder ?specials algorithm in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let unigram ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
      ?pad_token ?unk_token ?vocab ?byte_fallback ?max_piece_length
      ?n_sub_iterations ?shrinking_factor () =
    let _ =
      ( byte_fallback,
        max_piece_length,
        n_sub_iterations,
        shrinking_factor,
        unk_token )
    in
    let vocab = Option.value vocab ~default:[] in
    let algorithm = Alg_unigram (Unigram.create vocab) in
    let tok = create ?normalizer ?pre ?post ?decoder ?specials algorithm in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let chars ?normalizer ?pre ?post ?decoder ?specials ?bos_token ?eos_token
      ?pad_token ?unk_token () =
    (* Character-level tokenization using ASCII codes as token IDs *)
    let algorithm = Alg_chars (Chars.create ()) in
    let tok = create ?normalizer ?pre ?post ?decoder ?specials algorithm in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let regex _pattern ?normalizer ?pre ?post ?decoder ?specials ?bos_token
      ?eos_token ?pad_token ?unk_token () =
    (* Regex tokenization - simplified to word_level for now *)
    let algorithm = Alg_wordlevel (Word_level.create ()) in
    let tok = create ?normalizer ?pre ?post ?decoder ?specials algorithm in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let from_model_file ~vocab ?merges ?normalizer ?pre ?post ?decoder ?specials
      ?bos_token ?eos_token ?pad_token ?unk_token () =
    let tok =
      match merges with
      | Some merges_file ->
          (* Load BPE model *)
          let algorithm =
            Alg_bpe (Bpe.from_files ~vocab_file:vocab ~merges_file)
          in
          create ?normalizer ?pre ?post ?decoder ?specials algorithm
      | None ->
          (* Load WordPiece model *)
          let algorithm =
            Alg_wordpiece (Wordpiece.from_file ~vocab_file:vocab)
          in
          create ?normalizer ?pre ?post ?decoder ?specials algorithm
    in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let add_tokens t tokens =
    t.algorithm <- add_tokens_to_algorithm t.algorithm tokens;
    t

  let encode_text_internal t text =
    let normalized =
      match t.normalizer with
      | Some normalizer -> Normalizers.normalize_str normalizer text
      | None -> text
    in
    let pre_tokens =
      match t.pre_tokenizer with
      | Some pre -> pre normalized
      | None -> [ (normalized, (0, String.length normalized)) ]
    in
    pre_tokens
    |> List.concat_map (fun (fragment, _) ->
           tokenize_algorithm t.algorithm fragment)
    |> tokens_to_encoding

  let apply_post_processor t ~add_special primary pair =
    match t.post_processor with
    | None ->
        if Option.is_some pair then
          invalid_arg
            "Tokenizer.encode: pair sequences require a configured \
             post-processor"
        else primary
    | Some processor -> (
        let inputs =
          match pair with None -> [ primary ] | Some enc -> [ primary; enc ]
        in
        let processed =
          Processors.process processor
            (List.map encoding_to_processor inputs)
            ~add_special_tokens:add_special
        in
        match processed with
        | [] -> primary
        | encoding :: _ -> processor_to_encoding encoding)

  let encode_single t ~add_special_tokens ~truncation sequence =
    let primary = encode_text_internal t sequence.text in
    let pair = Option.map (encode_text_internal t) sequence.pair in
    let processed =
      apply_post_processor t ~add_special:add_special_tokens primary pair
    in
    match truncation with
    | None -> processed
    | Some { max_length; direction } ->
        Encoding.truncate processed ~max_length ~stride:0
          ~direction:(direction_to_trunc direction)

  let resolve_pad t cfg =
    let resolve_token () =
      match cfg.pad_token with
      | Some token -> token
      | None -> (
          match t.pad_runtime with
          | Some runtime -> runtime.token
          | None ->
              invalid_arg
                "Tokenizer.encode: padding requested but no pad token \
                 configured")
    in
    let token = resolve_token () in
    let resolve_id () =
      match cfg.pad_id with
      | Some id -> id
      | None -> (
          match t.pad_runtime with
          | Some runtime when runtime.token = token -> runtime.id
          | _ -> (
              match token_to_id_algorithm t.algorithm token with
              | Some id -> id
              | None ->
                  invalid_arg
                    (Printf.sprintf
                       "Tokenizer.encode: pad token '%s' not present in \
                        vocabulary"
                       token)))
    in
    let id = resolve_id () in
    let type_id =
      match cfg.pad_type_id with
      | Some type_id -> type_id
      | None -> (
          match t.pad_runtime with
          | Some runtime when runtime.token = token -> runtime.type_id
          | _ -> 0)
    in
    (token, id, type_id)

  let pad_encoding encoding ~target_length ~pad_id ~pad_type_id ~pad_token
      ~direction =
    if Encoding.length encoding >= target_length then encoding
    else
      Encoding.pad encoding ~target_length ~pad_id ~pad_type_id ~pad_token
        ~direction

  let apply_padding t encodings padding =
    match padding with
    | None -> encodings
    | Some cfg -> (
        let pad_token, pad_id, pad_type_id = resolve_pad t cfg in
        let pad_dir = direction_to_pad cfg.direction in
        let pad_to_length encoding target =
          pad_encoding encoding ~target_length:target ~pad_id ~pad_type_id
            ~pad_token ~direction:pad_dir
        in
        match cfg.length with
        | `Fixed n ->
            List.map (fun encoding -> pad_to_length encoding n) encodings
        | `Batch_longest ->
            let max_len =
              List.fold_left
                (fun acc encoding -> max acc (Encoding.length encoding))
                0 encodings
            in
            List.map (fun encoding -> pad_to_length encoding max_len) encodings
        | `To_multiple m ->
            if m <= 0 then encodings
            else
              List.map
                (fun encoding ->
                  let current_len = Encoding.length encoding in
                  let target =
                    if current_len mod m = 0 then current_len
                    else (current_len + m - 1) / m * m
                  in
                  pad_to_length encoding target)
                encodings)

  let encode_sequences t sequences ~add_special_tokens ~padding ~truncation =
    let raw =
      List.map (encode_single t ~add_special_tokens ~truncation) sequences
    in
    apply_padding t raw padding

  let encode t ?pair ?(add_special_tokens = true) ?padding ?truncation text =
    match
      encode_sequences t
        [ { text; pair } ]
        ~add_special_tokens ~padding ~truncation
    with
    | [ encoding ] -> encoding
    | _ -> invalid_arg "Tokenizer.encode: unexpected internal state"

  let encode_batch t ?pairs ?(add_special_tokens = true) ?padding ?truncation
      texts =
    match texts with
    | [] -> []
    | _ ->
        let sequences =
          match pairs with
          | None -> List.map (fun text -> { text; pair = None }) texts
          | Some pair_list ->
              let len_texts = List.length texts in
              if List.length pair_list <> len_texts then
                invalid_arg
                  "Tokenizer.encode_batch: pairs length must match texts";
              List.map2 (fun text pair -> { text; pair }) texts pair_list
        in
        encode_sequences t sequences ~add_special_tokens ~padding ~truncation

  let encode_ids t ?pair ?add_special_tokens ?padding ?truncation text =
    let encoding =
      encode t ?pair ?add_special_tokens ?padding ?truncation text
    in
    Array.copy (Encoding.get_ids encoding)

  let decode t ?(skip_special_tokens = false) ids =
    let tokens =
      Array.to_list ids
      |> List.filter_map (fun id ->
             match id_to_token_algorithm t.algorithm id with
             | None -> None
             | Some token
               when skip_special_tokens && Hashtbl.mem t.special_lookup token ->
                 None
             | Some token -> Some token)
    in
    match t.decoder with
    | Some decoder -> Decoders.decode decoder tokens
    | None -> (
        match t.algorithm with
        | Alg_wordlevel _ -> String.concat " " tokens
        | Alg_wordpiece _ -> String.concat "" tokens
        | _ -> String.concat "" tokens)

  let decode_batch t ?(skip_special_tokens = false) id_lists =
    List.map (decode t ~skip_special_tokens) id_lists

  let special_strings_of_list specials =
    tokens_of_config (special_list_to_config specials)

  let merge_string_lists lists =
    let table = Hashtbl.create 16 in
    let add acc token =
      if Hashtbl.mem table token then acc
      else (
        Hashtbl.add table token ();
        token :: acc)
    in
    List.fold_left (fun acc lst -> List.fold_left add acc lst) [] lists
    |> List.rev

  let existing_special_tokens t = tokens_of_config t.specials_config

  let special_tokens_for_training init specials =
    let from_init =
      match init with Some tok -> existing_special_tokens tok | None -> []
    in
    let from_param =
      match specials with Some s -> special_strings_of_list s | None -> []
    in
    merge_string_lists [ from_param; from_init ]

  let data_to_strings = function
    | `Files files ->
        let lines = ref [] in
        List.iter
          (fun file ->
            let ic = open_in file in
            try
              while true do
                lines := input_line ic :: !lines
              done
            with End_of_file -> close_in ic)
          files;
        List.rev !lines
    | `Iterator iterator ->
        let lines = ref [] in
        let rec loop () =
          match iterator () with
          | None -> ()
          | Some line ->
              lines := line :: !lines;
              loop ()
        in
        loop ();
        List.rev !lines
    | `Seq seq ->
        let state = ref seq in
        let lines = ref [] in
        let rec loop () =
          match !state () with
          | Seq.Nil -> ()
          | Seq.Cons (x, next) ->
              state := next;
              lines := x :: !lines;
              loop ()
        in
        loop ();
        List.rev !lines

  let train_bpe ?init ?normalizer ?pre ?post ?decoder ?specials ?bos_token
      ?eos_token ?pad_token ?unk_token ?(vocab_size = 30000)
      ?(min_frequency = 0) ?limit_alphabet ?initial_alphabet
      ?continuing_subword_prefix ?end_of_word_suffix ?(show_progress = true)
      ?max_token_length data =
    let special_tokens = special_tokens_for_training init specials in
    let initial_alphabet =
      match initial_alphabet with
      | None -> []
      | Some strs ->
          List.map (fun s -> if String.length s > 0 then s.[0] else ' ') strs
    in
    let limit_alphabet =
      match limit_alphabet with Some n -> Some n | None -> Some 1000
    in
    let texts = data_to_strings data in
    let existing_bpe =
      match init with
      | Some t -> ( match t.algorithm with Alg_bpe m -> Some m | _ -> None)
      | None -> None
    in
    let trained_model, result_specials =
      Bpe.train ~min_frequency ~vocab_size ~show_progress ~special_tokens
        ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
        ~end_of_word_suffix ~max_token_length texts existing_bpe
    in
    let algorithm = Alg_bpe trained_model in
    let tok = create ?normalizer ?pre ?post ?decoder algorithm in
    let tok =
      if result_specials <> [] then
        let special_list : special list =
          List.map Special.make result_specials
        in
        add_specials tok special_list
      else tok
    in
    let tok =
      match specials with Some s -> with_specials tok s | None -> tok
    in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let train_wordpiece ?init ?normalizer ?pre ?post ?decoder ?specials ?bos_token
      ?eos_token ?pad_token ?unk_token ?(vocab_size = 30000)
      ?(min_frequency = 0) ?limit_alphabet ?initial_alphabet
      ?(continuing_subword_prefix = "##") ?end_of_word_suffix
      ?(show_progress = true) data =
    let special_tokens = special_tokens_for_training init specials in
    let initial_alphabet =
      match initial_alphabet with
      | None -> []
      | Some strs ->
          List.map (fun s -> if String.length s > 0 then s.[0] else ' ') strs
    in
    let limit_alphabet =
      match limit_alphabet with Some n -> Some n | None -> Some 1000
    in
    let texts = data_to_strings data in
    let existing_wp =
      match init with
      | Some t -> (
          match t.algorithm with Alg_wordpiece m -> Some m | _ -> None)
      | None -> None
    in
    let trained_model, result_specials =
      Wordpiece.train ~min_frequency ~vocab_size ~show_progress ~special_tokens
        ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
        ~end_of_word_suffix texts existing_wp
    in
    let algorithm = Alg_wordpiece trained_model in
    let tok = create ?normalizer ?pre ?post ?decoder algorithm in
    let tok =
      if result_specials <> [] then
        add_specials tok
          (List.map (fun token -> Special.make token) result_specials)
      else tok
    in
    let tok =
      match specials with Some s -> with_specials tok s | None -> tok
    in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let train_wordlevel ?init ?normalizer ?pre ?post ?decoder ?specials ?bos_token
      ?eos_token ?pad_token ?unk_token ?(vocab_size = 30000)
      ?(min_frequency = 0) ?(show_progress = true) data =
    let special_tokens = special_tokens_for_training init specials in
    let texts = data_to_strings data in
    let existing_wl =
      match init with
      | Some t -> (
          match t.algorithm with Alg_wordlevel m -> Some m | _ -> None)
      | None -> None
    in
    let trained_model, result_specials =
      Word_level.train ~vocab_size ~min_frequency ~show_progress ~special_tokens
        texts existing_wl
    in
    let algorithm = Alg_wordlevel trained_model in
    let tok = create ?normalizer ?pre ?post ?decoder algorithm in
    let tok =
      if result_specials <> [] then
        add_specials tok
          (List.map (fun token -> Special.make token) result_specials)
      else tok
    in
    let tok =
      match specials with Some s -> with_specials tok s | None -> tok
    in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let train_unigram ?init ?normalizer ?pre ?post ?decoder ?specials ?bos_token
      ?eos_token ?pad_token ?unk_token ?(vocab_size = 8000)
      ?(show_progress = true) ?(shrinking_factor = 0.75)
      ?(max_piece_length = 16) ?(n_sub_iterations = 2) data =
    let special_tokens = special_tokens_for_training init specials in
    let texts = data_to_strings data in
    let existing_ug =
      match init with
      | Some t -> (
          match t.algorithm with Alg_unigram m -> Some m | _ -> None)
      | None -> None
    in
    let trained_model, result_specials =
      Unigram.train ~vocab_size ~show_progress ~special_tokens ~shrinking_factor
        ~unk_token ~max_piece_length ~n_sub_iterations texts existing_ug
    in
    let algorithm = Alg_unigram trained_model in
    let tok = create ?normalizer ?pre ?post ?decoder algorithm in
    let tok =
      if result_specials <> [] then
        add_specials tok
          (List.map (fun token -> Special.make token) result_specials)
      else tok
    in
    let tok =
      match specials with Some s -> with_specials tok s | None -> tok
    in
    let tok =
      match bos_token with Some t -> set_bos_token tok (Some t) | None -> tok
    in
    let tok =
      match eos_token with Some t -> set_eos_token tok (Some t) | None -> tok
    in
    let tok =
      match pad_token with Some t -> set_pad_token tok (Some t) | None -> tok
    in
    let tok =
      match unk_token with Some t -> set_unk_token tok (Some t) | None -> tok
    in
    tok

  let export_tiktoken t ~merges_path ~vocab_path =
    match t.algorithm with
    | Alg_bpe _bpe ->
        let vocab =
          vocab_algorithm t.algorithm
          |> List.sort (fun (_, id1) (_, id2) -> Int.compare id1 id2)
        in
        let vocab_json =
          `Assoc (List.map (fun (token, id) -> (token, `Int id)) vocab)
        in
        Yojson.Basic.to_file vocab_path vocab_json;
        let oc = open_out merges_path in
        output_string oc "#version: 0.2\n";
        (* TODO: Export merges - BPE model doesn't expose merges list *)
        close_out oc
    | _ ->
        invalid_arg
          "Tokenizer.export_tiktoken: export is only supported for BPE models"

  let save_model_files t ~folder ?prefix () =
    save_algorithm t.algorithm ~folder ?prefix ()

  (* JSON serialization helpers *)
  module Json_helpers = struct
    open Yojson.Basic.Util

    let option_to_json f = function None -> `Null | Some v -> f v

    let string_or_null = function
      | `Null -> None
      | json -> Some (to_string json)

    let special_of_json json : special =
      {
        token = json |> member "token" |> to_string;
        single_word =
          json |> member "single_word" |> to_bool_option
          |> Option.value ~default:false;
        lstrip =
          json |> member "lstrip" |> to_bool_option
          |> Option.value ~default:false;
        rstrip =
          json |> member "rstrip" |> to_bool_option
          |> Option.value ~default:false;
        normalized =
          json |> member "normalized" |> to_bool_option
          |> Option.value ~default:false;
      }

    let added_token_to_json ~id (s : special) : Yojson.Basic.t =
      `Assoc
        [
          ("id", `Int id);
          ("content", `String s.token);
          ("single_word", `Bool s.single_word);
          ("lstrip", `Bool s.lstrip);
          ("rstrip", `Bool s.rstrip);
          ("normalized", `Bool s.normalized);
          ("special", `Bool true);
        ]

    let normalizer_to_json = function
      | None -> `Null
      | Some norm -> Normalizers.to_json norm

    let normalizer_of_json = function
      | `Null -> Ok None
      | json -> Ok (Some (Normalizers.of_json json))

    let pre_tokenizer_to_json = function
      | None -> `Null
      | Some _pre -> `Null (* TODO: Add Pre_tokenizers.to_json *)

    let pre_tokenizer_of_json = function
      | `Null -> Ok None
      | _json -> Ok None (* TODO: Add Pre_tokenizers.of_json *)

    let post_processor_to_json = function
      | None -> `Null
      | Some post -> Processors.to_json post

    let post_processor_of_json = function
      | `Null -> Ok None
      | json -> Ok (Some (Processors.of_json json))

    let decoder_to_json = function
      | None -> `Null
      | Some dec -> Decoders.to_json dec

    let decoder_of_json = function
      | `Null -> Ok None
      | json -> Ok (Some (Decoders.of_json json))
  end

  let to_json (t : t) : Yojson.Basic.t =
    let open Json_helpers in
    (* Collect all added tokens with their IDs *)
    let vocab_list = vocab_algorithm t.algorithm in
    let added_tokens =
      config_to_special_list t.specials_config
      |> List.filter_map (fun spec ->
             List.find_opt (fun (token, _) -> token = spec.token) vocab_list
             |> Option.map (fun (_, id) -> added_token_to_json ~id spec))
    in
    (* Serialize model based on algorithm type *)
    let model_json =
      match t.algorithm with
      | Alg_bpe bpe ->
          let vocab = Bpe.get_vocab bpe in
          let vocab_json =
            `Assoc (List.map (fun (token, id) -> (token, `Int id)) vocab)
          in
          (* TODO: Need to add get_merges to BPE module *)
          let merges_json = `List [] in
          `Assoc
            [
              ("type", `String "BPE");
              ("dropout", option_to_json (fun f -> `Float f) None);
              ( "unk_token",
                option_to_json (fun s -> `String s) (Bpe.get_unk_token bpe) );
              ( "continuing_subword_prefix",
                option_to_json
                  (fun s -> `String s)
                  (Bpe.get_continuing_subword_prefix bpe) );
              ( "end_of_word_suffix",
                option_to_json
                  (fun s -> `String s)
                  (Bpe.get_end_of_word_suffix bpe) );
              ("fuse_unk", `Bool false);
              ("byte_fallback", `Bool false);
              ("ignore_merges", `Bool false);
              ("vocab", vocab_json);
              ("merges", merges_json);
            ]
      | Alg_wordpiece wp ->
          let vocab = Wordpiece.get_vocab wp in
          let vocab_json =
            `Assoc (List.map (fun (token, id) -> (token, `Int id)) vocab)
          in
          let unk_token = Wordpiece.get_unk_token wp in
          let continuing_subword_prefix =
            Wordpiece.get_continuing_subword_prefix wp
          in
          `Assoc
            [
              ("type", `String "WordPiece");
              ("unk_token", `String unk_token);
              ("continuing_subword_prefix", `String continuing_subword_prefix);
              ("max_input_chars_per_word", `Int 100);
              ("vocab", vocab_json);
            ]
      | Alg_wordlevel wl ->
          let vocab = Word_level.get_vocab wl in
          let vocab_json =
            `Assoc (List.map (fun (token, id) -> (token, `Int id)) vocab)
          in
          `Assoc
            [
              ("type", `String "WordLevel");
              ("unk_token", `String "[UNK]");
              ("vocab", vocab_json);
            ]
      | Alg_unigram ug ->
          let vocab = Unigram.get_vocab ug in
          let vocab_json =
            `List
              (List.map
                 (fun (token, score) -> `List [ `String token; `Float score ])
                 vocab)
          in
          `Assoc
            [
              ("type", `String "Unigram");
              ("unk_id", `Null);
              ("vocab", vocab_json);
            ]
      | Alg_chars _chars ->
          `Assoc [ ("type", `String "Chars"); ("vocab", `Assoc []) ]
    in
    `Assoc
      [
        ("version", `String "1.0");
        ("truncation", `Null);
        ("padding", `Null);
        ("added_tokens", `List added_tokens);
        ("normalizer", normalizer_to_json t.normalizer);
        ("pre_tokenizer", pre_tokenizer_to_json t.pre_tokenizer);
        ("post_processor", post_processor_to_json t.post_processor);
        ("decoder", decoder_to_json t.decoder);
        ("model", model_json);
      ]

  let from_json (json : Yojson.Basic.t) : (t, exn) result =
    let open Yojson.Basic.Util in
    let open Json_helpers in
    try
      (* Parse normalizer *)
      let normalizer_result =
        normalizer_of_json (json |> member "normalizer")
      in
      let normalizer =
        match normalizer_result with Ok n -> n | Error e -> raise e
      in
      (* Parse pre-tokenizer *)
      let pre_result = pre_tokenizer_of_json (json |> member "pre_tokenizer") in
      let pre = match pre_result with Ok p -> p | Error e -> raise e in
      (* Parse post-processor *)
      let post_result =
        post_processor_of_json (json |> member "post_processor")
      in
      let post = match post_result with Ok p -> p | Error e -> raise e in
      (* Parse decoder *)
      let decoder_result = decoder_of_json (json |> member "decoder") in
      let decoder =
        match decoder_result with Ok d -> d | Error e -> raise e
      in
      (* Parse model *)
      let model_json = json |> member "model" in
      let model_type = model_json |> member "type" |> to_string in
      let algorithm =
        match model_type with
        | "BPE" ->
            let vocab_json = model_json |> member "vocab" |> to_assoc in
            let vocab_list =
              List.map (fun (token, id) -> (token, to_int id)) vocab_json
            in
            let merges_json =
              model_json |> member "merges" |> to_list
              |> List.map (fun arr ->
                     match to_list arr with
                     | [ a; b ] -> (to_string a, to_string b)
                     | _ -> failwith "Invalid merge format")
            in
            let unk_token =
              model_json |> member "unk_token" |> string_or_null
            in
            let continuing_subword_prefix =
              model_json |> member "continuing_subword_prefix" |> string_or_null
            in
            let end_of_word_suffix =
              model_json |> member "end_of_word_suffix" |> string_or_null
            in
            let vocab_ht = Hashtbl.create (List.length vocab_list) in
            List.iter
              (fun (token, id) -> Hashtbl.add vocab_ht token id)
              vocab_list;
            let bpe =
              Bpe.create
                {
                  vocab = vocab_ht;
                  merges = merges_json;
                  cache_capacity = 10000;
                  dropout = None;
                  unk_token;
                  continuing_subword_prefix;
                  end_of_word_suffix;
                  fuse_unk = false;
                  byte_fallback = false;
                  ignore_merges = false;
                }
            in
            Alg_bpe bpe
        | "WordPiece" ->
            let vocab_json = model_json |> member "vocab" |> to_assoc in
            let vocab_list =
              List.map (fun (token, id) -> (token, to_int id)) vocab_json
            in
            let unk_token =
              model_json |> member "unk_token" |> string_or_null
              |> Option.value ~default:"[UNK]"
            in
            let continuing_subword_prefix =
              model_json
              |> member "continuing_subword_prefix"
              |> string_or_null |> Option.value ~default:"##"
            in
            let max_input_chars_per_word =
              try model_json |> member "max_input_chars_per_word" |> to_int
              with _ -> 100
            in
            let vocab_ht = Hashtbl.create (List.length vocab_list) in
            List.iter
              (fun (token, id) -> Hashtbl.add vocab_ht token id)
              vocab_list;
            let wp =
              Wordpiece.create
                {
                  vocab = vocab_ht;
                  unk_token;
                  continuing_subword_prefix;
                  max_input_chars_per_word;
                }
            in
            Alg_wordpiece wp
        | "WordLevel" ->
            let vocab_json = model_json |> member "vocab" |> to_assoc in
            let vocab_list =
              List.map (fun (token, id) -> (token, to_int id)) vocab_json
            in
            let unk_token =
              model_json |> member "unk_token" |> string_or_null
              |> Option.value ~default:"[UNK]"
            in
            let wl = Word_level.create ~vocab:vocab_list ~unk_token () in
            Alg_wordlevel wl
        | "Unigram" ->
            let vocab_json = model_json |> member "vocab" |> to_list in
            let vocab_list =
              List.map
                (fun arr ->
                  match to_list arr with
                  | [ token; score ] -> (to_string token, to_float score)
                  | _ -> failwith "Invalid unigram vocab format")
                vocab_json
            in
            let ug = Unigram.create vocab_list in
            Alg_unigram ug
        | "Chars" ->
            let chars = Chars.create () in
            Alg_chars chars
        | _ -> failwith (Printf.sprintf "Unsupported model type: %s" model_type)
      in
      (* Parse added tokens *)
      let added_tokens =
        try json |> member "added_tokens" |> to_list |> List.map special_of_json
        with _ -> []
      in
      (* Create tokenizer *)
      let tok =
        create ?normalizer ?pre ?post ?decoder ~specials:added_tokens algorithm
      in
      Ok tok
    with
    | Yojson.Json_error msg -> Error (Failure ("JSON parse error: " ^ msg))
    | Type_error (msg, _) -> Error (Failure ("JSON type error: " ^ msg))
    | e -> Error e

  let from_file (path : string) : (t, exn) result =
    try
      let json = Yojson.Basic.from_file path in
      from_json json
    with
    | Sys_error msg -> Error (Failure ("File error: " ^ msg))
    | e -> Error e

  let save_pretrained (t : t) ~path : unit =
    (* Create directory if it doesn't exist *)
    (try Sys.mkdir path 0o755 with Sys_error _ -> ());
    (* Save tokenizer.json *)
    let json = to_json t in
    let tokenizer_path = Filename.concat path "tokenizer.json" in
    Yojson.Basic.to_file tokenizer_path json
end
