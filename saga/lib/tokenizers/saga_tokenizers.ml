(** Main tokenizers module that aggregates all tokenization functionality *)

module Either = struct
  type ('a, 'b) t = Left of 'a | Right of 'b
end

module Models = Models
(** Re-export all the submodules *)

module Normalizers = Normalizers
module Pre_tokenizers = Pre_tokenizers
module Processors = Processors
module Decoders = Decoders
module Trainers = Trainers
module Unicode = Unicode
module Encoding = Encoding
module Bpe = Bpe
module Wordpiece = Wordpiece

(* Compatibility layer implementation *)

type split_delimiter_behavior =
  [ `Removed
  | `Isolated
  | `Merged_with_previous
  | `Merged_with_next
  | `Contiguous ]
(** Type definitions *)

type strategy = [ `Longest_first | `Only_first | `Only_second ]
type prepend_scheme = [ `Always | `Never | `First ]

(** Added token type *)
module Added_token = struct
  type t = {
    content : string;
    single_word : bool;
    lstrip : bool;
    rstrip : bool;
    normalized : bool;
    special : bool;
  }

  let create ?(content = "") ?(single_word = false) ?(lstrip = false)
      ?(rstrip = false) ?(normalized = true) ?(special = false) () =
    { content; single_word; lstrip; rstrip; normalized; special }

  let content t = t.content
  let lstrip t = t.lstrip
  let normalized t = t.normalized
  let rstrip t = t.rstrip
  let single_word t = t.single_word
  let special t = t.special
  let _ = special (* Suppress unused warning *)
end

type direction = [ `Left | `Right ]
(** Direction type for padding *)

(** Main Tokenizer module *)
module Tokenizer = struct
  type padding_config = {
    direction : direction;
    pad_id : int;
    pad_type_id : int;
    pad_token : string;
    length : int option;
    pad_to_multiple_of : int option;
  }

  type truncation_config = {
    max_length : int;
    stride : int;
    strategy : strategy;
    direction : direction;
  }

  type t = {
    mutable model : Models.t;
    mutable normalizer : Normalizers.t option;
    mutable pre_tokenizer : Pre_tokenizers.t option;
    mutable post_processor : Processors.t option;
    mutable decoder : Decoders.t option;
    mutable padding : padding_config option;
    mutable truncation : truncation_config option;
    added_tokens : (string, Added_token.t) Hashtbl.t;
    special_tokens : (string, int) Hashtbl.t;
  }

  let create ~model =
    {
      model;
      normalizer = None;
      pre_tokenizer = None;
      post_processor = None;
      decoder = None;
      padding = None;
      truncation = None;
      added_tokens = Hashtbl.create 16;
      special_tokens = Hashtbl.create 16;
    }

  let from_file path =
    try
      let json = Yojson.Basic.from_file path in
      (* Parse tokenizer.json structure *)
      let t =
        match json with
        | `Assoc fields ->
            let model =
              match List.assoc_opt "model" fields with
              | Some model_json -> Models.of_json model_json
              | None -> Models.word_level ()
            in
            let tok = create ~model in
            (* Optional components *)
            (match List.assoc_opt "normalizer" fields with
            | Some `Null | None -> ()
            | Some njson -> tok.normalizer <- Some (Normalizers.of_json njson));
            (match List.assoc_opt "post_processor" fields with
            | Some `Null | None -> ()
            | Some pjson ->
                tok.post_processor <- Some (Processors.of_json pjson));
            (match List.assoc_opt "truncation" fields with
            | Some (`Assoc tfs) ->
                let max_length =
                  match List.assoc_opt "max_length" tfs with
                  | Some (`Int i) -> i
                  | _ -> 0
                in
                let stride =
                  match List.assoc_opt "stride" tfs with
                  | Some (`Int i) -> i
                  | _ -> 0
                in
                let direction =
                  match List.assoc_opt "direction" tfs with
                  | Some (`String "Left") -> `Left
                  | _ -> `Right
                in
                let strategy =
                  match List.assoc_opt "strategy" tfs with
                  | Some (`String "OnlyFirst") -> `Only_first
                  | Some (`String "OnlySecond") -> `Only_second
                  | _ -> `Longest_first
                in
                tok.truncation <-
                  Some { max_length; stride; strategy; direction }
            | _ -> ());
            (match List.assoc_opt "padding" fields with
            | Some (`Assoc pfs) ->
                let direction =
                  match List.assoc_opt "direction" pfs with
                  | Some (`String "Left") -> `Left
                  | _ -> `Right
                in
                let pad_id =
                  match List.assoc_opt "pad_id" pfs with
                  | Some (`Int i) -> i
                  | _ -> 0
                in
                let pad_type_id =
                  match List.assoc_opt "pad_type_id" pfs with
                  | Some (`Int i) -> i
                  | _ -> 0
                in
                let pad_token =
                  match List.assoc_opt "pad_token" pfs with
                  | Some (`String s) -> s
                  | _ -> "<pad>"
                in
                let length =
                  match List.assoc_opt "length" pfs with
                  | Some (`Int i) -> Some i
                  | _ -> None
                in
                let pad_to_multiple_of =
                  match List.assoc_opt "pad_to_multiple_of" pfs with
                  | Some (`Int i) -> Some i
                  | _ -> None
                in
                tok.padding <-
                  Some
                    {
                      direction;
                      pad_id;
                      pad_type_id;
                      pad_token;
                      length;
                      pad_to_multiple_of;
                    }
            | _ -> ());
            tok
        | _ -> create ~model:(Models.word_level ())
      in
      Ok t
    with e -> Error e

  let from_str str =
    try
      let bytes = Bytes.of_string str in
      let tmp = Filename.temp_file "saga_tokenizer" ".json" in
      let oc = open_out tmp in
      output_string oc (Bytes.to_string bytes);
      close_out oc;
      let res = from_file tmp in
      (try Sys.remove tmp with _ -> ());
      res
    with e -> Error e

  let from_pretrained identifier ?revision ?token () =
    (* Would need to download from HF hub *)
    let _ = (identifier, revision, token) in
    let model = Models.word_level () in
    Ok (create ~model)

  let from_buffer bytes = from_str (Bytes.to_string bytes)

  let save t ~path ?pretty () =
    let _ = pretty in
    (* Create a basic JSON structure *)
    let json =
      `Assoc
        [
          ("version", `String "1.0");
          ("truncation", `Null);
          ("padding", `Null);
          ("added_tokens", `List []);
          ("normalizer", `Null);
          ("pre_tokenizer", `Null);
          ("post_processor", `Null);
          ("decoder", `Null);
          ("model", Models.to_json t.model);
        ]
    in
    let oc = open_out path in
    Yojson.Basic.to_channel oc json;
    close_out oc

  let to_str t ?pretty () =
    let int_opt = function None -> `Null | Some i -> `Int i in
    let json =
      `Assoc
        [
          ("version", `String "1.0");
          ( "truncation",
            match t.truncation with
            | None -> `Null
            | Some cfg ->
                `Assoc
                  [
                    ("max_length", `Int cfg.max_length);
                    ("stride", `Int cfg.stride);
                    ( "strategy",
                      `String
                        (match cfg.strategy with
                        | `Longest_first -> "LongestFirst"
                        | `Only_first -> "OnlyFirst"
                        | `Only_second -> "OnlySecond") );
                    ( "direction",
                      `String
                        (match cfg.direction with
                        | `Left -> "Left"
                        | `Right -> "Right") );
                  ] );
          ( "padding",
            match t.padding with
            | None -> `Null
            | Some cfg ->
                `Assoc
                  [
                    ( "direction",
                      `String
                        (match cfg.direction with
                        | `Left -> "Left"
                        | `Right -> "Right") );
                    ("pad_id", `Int cfg.pad_id);
                    ("pad_type_id", `Int cfg.pad_type_id);
                    ("pad_token", `String cfg.pad_token);
                    ("length", int_opt cfg.length);
                    ("pad_to_multiple_of", int_opt cfg.pad_to_multiple_of);
                  ] );
          ( "added_tokens",
            `List
              (Hashtbl.fold
                 (fun _ tok acc ->
                   `Assoc
                     [
                       ("content", `String (Added_token.content tok));
                       ("single_word", `Bool (Added_token.single_word tok));
                       ("lstrip", `Bool (Added_token.lstrip tok));
                       ("rstrip", `Bool (Added_token.rstrip tok));
                       ("normalized", `Bool (Added_token.normalized tok));
                       ("special", `Bool (Added_token.special tok));
                     ]
                   :: acc)
                 t.added_tokens []) );
          ( "normalizer",
            match t.normalizer with
            | None -> `Null
            | Some n -> Normalizers.to_json n );
          ( "pre_tokenizer",
            match t.pre_tokenizer with
            | None -> `Null
            | Some _ ->
                (* We do not serialize function values; represent symbolic
                   name *)
                `Assoc [ ("type", `String "Custom") ] );
          ( "post_processor",
            match t.post_processor with
            | None -> `Null
            | Some p -> Processors.to_json p );
          ( "decoder",
            match t.decoder with
            | None -> `Null
            | Some _ -> `Assoc [ ("type", `String "Custom") ] );
          ("model", Models.to_json t.model);
        ]
    in
    match pretty with
    | Some true -> Yojson.Basic.pretty_to_string json
    | _ -> Yojson.Basic.to_string json

  let get_model t = t.model
  let set_model t model = t.model <- model
  let get_normalizer t = t.normalizer
  let set_normalizer t normalizer = t.normalizer <- normalizer
  let get_pre_tokenizer t = t.pre_tokenizer
  let set_pre_tokenizer t pre_tokenizer = t.pre_tokenizer <- pre_tokenizer
  let get_post_processor t = t.post_processor
  let set_post_processor t post_processor = t.post_processor <- post_processor
  let get_decoder t = t.decoder
  let set_decoder t decoder = t.decoder <- decoder
  let get_padding t = t.padding
  let _set_padding t padding = t.padding <- Some padding
  let no_padding t = t.padding <- None
  let get_truncation t = t.truncation
  let _set_truncation t truncation = t.truncation <- Some truncation
  let no_truncation t = t.truncation <- None

  let add_special_tokens t tokens =
    let count = ref 0 in
    let token_strings = ref [] in
    List.iter
      (fun token_either ->
        let added_token =
          match token_either with
          | Either.Left str -> Added_token.create ~content:str ~special:true ()
          | Either.Right tok -> tok
        in
        if not (Hashtbl.mem t.added_tokens (Added_token.content added_token))
        then (
          Hashtbl.add t.added_tokens
            (Added_token.content added_token)
            added_token;
          (* TODO: properly assign IDs to special tokens *)
          Hashtbl.add t.special_tokens
            (Added_token.content added_token)
            (Hashtbl.length t.special_tokens);
          token_strings := Added_token.content added_token :: !token_strings;
          incr count))
      tokens;
    (* Also add special tokens to the model's vocabulary *)
    let model_count = Models.add_tokens t.model (List.rev !token_strings) in
    let _ = model_count in
    !count

  let add_tokens t tokens =
    let count = ref 0 in
    let token_strings = ref [] in
    List.iter
      (fun token_either ->
        let added_token =
          match token_either with
          | Either.Left str -> Added_token.create ~content:str ()
          | Either.Right tok -> tok
        in
        if not (Hashtbl.mem t.added_tokens (Added_token.content added_token))
        then (
          Hashtbl.add t.added_tokens
            (Added_token.content added_token)
            added_token;
          token_strings := Added_token.content added_token :: !token_strings;
          incr count))
      tokens;
    (* Also add tokens to the model's vocabulary *)
    let model_count = Models.add_tokens t.model (List.rev !token_strings) in
    let _ = model_count in
    (* Could verify this matches count *)
    !count

  let get_vocab t ?with_added_tokens () =
    let _with_added = Option.value with_added_tokens ~default:true in
    (* Added tokens are injected into model vocab via Models.add_tokens, so
       model vocab is authoritative *)
    Models.get_vocab t.model

  let get_vocab_size t ?with_added_tokens () =
    let with_added = Option.value with_added_tokens ~default:true in
    let base_size = Models.get_vocab_size t.model in
    if with_added then
      base_size
      + Hashtbl.length t.added_tokens
      + Hashtbl.length t.special_tokens
    else base_size

  let enable_padding t config = t.padding <- Some config
  let enable_truncation t config = t.truncation <- Some config

  (* Conversion helpers between Encoding.t and Processors.encoding *)
  let encoding_to_processor (e : Encoding.t) : Processors.encoding =
    {
      ids = Encoding.get_ids e;
      type_ids = Encoding.get_type_ids e;
      tokens = Encoding.get_tokens e;
      offsets = Encoding.get_offsets e;
      special_tokens_mask = Encoding.get_special_tokens_mask e;
      attention_mask = Encoding.get_attention_mask e;
      overflowing = [];
      (* Encoding has overflowing; ignore in processor for now *)
      sequence_ranges = [];
    }

  let processor_to_encoding (pe : Processors.encoding) : Encoding.t =
    (* Rebuild Encoding.t; words are None; sequence ranges empty *)
    let len = Array.length pe.ids in
    let words = Array.make len None in
    let seq = Hashtbl.create 1 in
    Encoding.create ~ids:pe.ids ~type_ids:pe.type_ids ~tokens:pe.tokens ~words
      ~offsets:pe.offsets ~special_tokens_mask:pe.special_tokens_mask
      ~attention_mask:pe.attention_mask ~overflowing:[] ~sequence_ranges:seq

  let encode t ~sequence ?pair ?is_pretokenized ?add_special_tokens () =
    let is_pretokenized = Option.value is_pretokenized ~default:false in
    let add_special_tokens = Option.value add_special_tokens ~default:true in
    let _ = (pair, is_pretokenized, add_special_tokens) in

    (* Handle Either type for sequence *)
    let text =
      match sequence with
      | Either.Left s -> s
      | Either.Right lst -> String.concat " " lst
    in

    (* Normalize *)
    let normalized =
      match t.normalizer with
      | Some n -> Normalizers.normalize_str n text
      | None -> text
    in

    (* Pre-tokenize *)
    let pre_tokens =
      match t.pre_tokenizer with
      | Some pt -> pt normalized (* Pre_tokenizers.t is already a function *)
      | None -> [ (normalized, (0, String.length normalized)) ]
    in

    (* Tokenize *)
    let tokens =
      List.concat_map (fun (text, _) -> Models.tokenize t.model text) pre_tokens
    in

    (* Convert to encoding using the new Encoding module *)
    let token_list =
      List.map
        (fun (tok : Models.token) -> (tok.id, tok.value, tok.offsets))
        tokens
    in
    let enc = Encoding.from_tokens token_list ~type_id:0 in
    (* Post-process if configured *)
    let enc =
      match t.post_processor with
      | None -> enc
      | Some proc -> (
          let proc_enc = encoding_to_processor enc in
          let outs = Processors.process proc [ proc_enc ] ~add_special_tokens in
          match outs with [] -> enc | x :: _ -> processor_to_encoding x)
    in
    (* Apply truncation if configured *)
    let enc =
      match t.truncation with
      | None -> enc
      | Some cfg ->
          let t_dir : Encoding.truncation_direction =
            match cfg.direction with
            | `Left -> Encoding.Left
            | `Right -> Encoding.Right
          in
          Encoding.truncate enc ~max_length:cfg.max_length ~stride:cfg.stride
            ~direction:t_dir
    in
    (* Apply padding if configured and length provided *)
    let enc =
      match t.padding with
      | None -> enc
      | Some cfg -> (
          let p_dir : Encoding.padding_direction =
            match cfg.direction with
            | `Left -> Encoding.Left
            | `Right -> Encoding.Right
          in
          match (cfg.length, cfg.pad_to_multiple_of) with
          | Some target, _ ->
              Encoding.pad enc ~target_length:target ~pad_id:cfg.pad_id
                ~pad_type_id:cfg.pad_type_id ~pad_token:cfg.pad_token
                ~direction:p_dir
          | None, Some m when m > 0 ->
              let cur = Encoding.length enc in
              let target = (cur + m - 1) / m * m in
              Encoding.pad enc ~target_length:target ~pad_id:cfg.pad_id
                ~pad_type_id:cfg.pad_type_id ~pad_token:cfg.pad_token
                ~direction:p_dir
          | None, _ -> enc)
    in
    enc

  let encode_batch t ~input ?is_pretokenized ?add_special_tokens () =
    let encs =
      List.map
        (fun item ->
          match item with
          | Either.Left sequence ->
              encode t ~sequence ?pair:None ?is_pretokenized ?add_special_tokens
                ()
          | Either.Right (seq1, seq2) ->
              encode t ~sequence:seq1 ~pair:seq2 ?is_pretokenized
                ?add_special_tokens ())
        input
    in
    (* If padding enabled without fixed length, pad to the max in the batch *)
    match t.padding with
    | None -> encs
    | Some cfg ->
        let p_dir : Encoding.padding_direction =
          match cfg.direction with
          | `Left -> Encoding.Left
          | `Right -> Encoding.Right
        in
        let max_len =
          List.fold_left (fun acc e -> max acc (Encoding.length e)) 0 encs
        in
        let target =
          match (cfg.length, cfg.pad_to_multiple_of) with
          | Some l, _ -> l
          | None, Some m when m > 0 -> (max_len + m - 1) / m * m
          | _ -> max_len
        in
        List.map
          (fun e ->
            if Encoding.length e >= target then e
            else
              Encoding.pad e ~target_length:target ~pad_id:cfg.pad_id
                ~pad_type_id:cfg.pad_type_id ~pad_token:cfg.pad_token
                ~direction:p_dir)
          encs

  let decode t ids ?skip_special_tokens ?clean_up_tokenization_spaces () =
    let skip_special_tokens = Option.value skip_special_tokens ~default:false in
    let clean_up_tokenization_spaces =
      Option.value clean_up_tokenization_spaces ~default:true
    in
    let _ = clean_up_tokenization_spaces in

    (* Get tokens from IDs and filter special tokens if requested *)
    let tokens =
      List.filter_map
        (fun id ->
          match Models.id_to_token t.model id with
          | Some token
            when skip_special_tokens && Hashtbl.mem t.special_tokens token ->
              None
          | Some token -> Some token
          | None -> None)
        ids
    in

    (* Apply decoder if present *)
    match t.decoder with
    | Some d -> Decoders.decode d tokens
    | None -> (
        (* Default decoding behavior depends on model type *)
        match t.model with
        | Models.WordLevel _ ->
            (* WordLevel adds spaces between tokens by default *)
            String.concat " " tokens
        | Models.WordPiece _ ->
            (* WordPiece concatenates without spaces but handles ## prefix *)
            String.concat "" tokens
        | _ ->
            (* Other models concatenate without spaces *)
            String.concat "" tokens)

  let decode_batch t id_lists ?skip_special_tokens ?clean_up_tokenization_spaces
      () =
    List.map
      (fun ids ->
        decode t ids ?skip_special_tokens ?clean_up_tokenization_spaces ())
      id_lists

  let token_to_id t token = Models.token_to_id t.model token
  let id_to_token t id = Models.id_to_token t.model id

  let get_added_tokens_decoder t =
    (* Return mapping of added token ids to their Added_token.t when present in
       model vocab *)
    Hashtbl.fold
      (fun _ tok acc ->
        match Models.token_to_id t.model (Added_token.content tok) with
        | Some id -> (id, tok) :: acc
        | None -> acc)
      t.added_tokens []

  let train t ~files ?trainer () =
    let trainer =
      match trainer with Some tr -> tr | None -> Trainers.word_level ()
    in
    let res = Trainers.train trainer ~files ?model:(Some t.model) () in
    (* Update model and register special tokens *)
    t.model <- res.model;
    let _ =
      if res.special_tokens <> [] then
        add_special_tokens t
          (List.map (fun s -> Either.Left s) res.special_tokens)
      else 0
    in
    ()

  let train_from_iterator t (seq : string Seq.t) ?trainer ?length () =
    let _ = length in
    let trainer =
      match trainer with Some tr -> tr | None -> Trainers.word_level ()
    in
    (* Convert Seq.t to iterator (unit -> string option) *)
    let state = ref seq in
    let iterator () =
      match !state () with
      | Seq.Nil -> None
      | Seq.Cons (x, next) ->
          state := next;
          Some x
    in
    let res =
      Trainers.train_from_iterator trainer ~iterator ?model:(Some t.model) ()
    in
    t.model <- res.model;
    let _ =
      if res.special_tokens <> [] then
        add_special_tokens t
          (List.map (fun s -> Either.Left s) res.special_tokens)
      else 0
    in
    ()

  let post_process _t ~encoding ?pair ?add_special_tokens () =
    let _add_special_tokens = Option.value add_special_tokens ~default:true in
    let _ = pair in
    (* Not fully wired to Processors.encoding; return identity for now *)
    encoding

  let num_special_tokens_to_add t ~is_pair =
    match t.post_processor with
    | None -> 0
    | Some p -> Processors.added_tokens p ~is_pair

  let save_pretrained t ~path =
    (* Save tokenizer.json and model files *)
    let tok_file = Filename.concat path "tokenizer.json" in
    let oc = open_out tok_file in
    output_string oc (to_str t ~pretty:true ());
    close_out oc;
    let _files = Models.save t.model ~folder:path () in
    ()
end
