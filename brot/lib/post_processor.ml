(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let err_type_id tok = strf "expected integer type id after ':' in '%s'" tok
let err_piece tok = strf "expected 'id' or 'id:type_id', got '%s'" tok
let err_unknown_special tok = strf "unknown special token '%s'" tok
let err_mismatch tok = strf "ids and tokens differ in length for '%s'" tok
let err_expected what v = strf "expected %s, got %s" what v
let err_seq_id = "sequence id must be \"A\", \"B\", 0 or 1"
let err_type_id_field = "expected number for 'type_id'"
let err_missing_sequence = "template references a sequence not provided"
let err_pair_required = "pair template required when two sequences are provided"
let err_pair_must_ref_both = "pair template must reference both $A and $B"
let err_template_def = "expected string, array or null for template"
let err_unsupported_piece = "expected Sequence or SpecialToken object"
let err_special_missing_id = "missing 'id' in SpecialToken"
let err_special_missing_ids = "missing 'ids' in special token"
let err_special_entry = "expected object for special token entry"

(* Types *)

type sequence_id = Sequence_a | Sequence_b

type template_piece =
  | Piece_sequence of { id : sequence_id; type_id : int }
  | Piece_special of { key : string; type_id : int }

type template = template_piece list

type special_token = {
  key : string;
  value_ids : int list;
  value_tokens : string list;
}

type token = string * int

type t =
  | Bert of { sep : token; cls : token }
  | Roberta of {
      sep : token;
      cls : token;
      pad : token;
      trim_offsets : bool;
      add_prefix_space : bool;
    }
  | ByteLevel of { trim_offsets : bool }
  | Template of {
      single : template;
      pair : template option;
      special_tokens : special_token list;
    }
  | Sequence of t list

(* Helpers *)

let special_token ~id ~token ~type_id =
  Encoding.token ~id ~token ~offset:(0, 0) ~type_id ~special:true

let with_type_id enc type_id =
  Encoding.create ~ids:(Encoding.ids enc)
    ~type_ids:(Array.make (Encoding.length enc) type_id)
    ~tokens:(Encoding.tokens enc) ~words:(Encoding.word_ids enc)
    ~offsets:(Encoding.offsets enc)
    ~special_tokens_mask:(Encoding.special_tokens_mask enc)
    ~attention_mask:(Encoding.attention_mask enc)
    ()

let is_ws = function
  | ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c' -> true
  | _ -> false

let build_special_lookup special_tokens =
  let tbl = Hashtbl.create (List.length special_tokens + 1) in
  List.iter (fun tok -> Hashtbl.replace tbl tok.key tok) special_tokens;
  tbl

let string_is_int s =
  let len = String.length s in
  let rec loop i =
    if i >= len then true
    else match s.[i] with '0' .. '9' -> loop (i + 1) | _ -> false
  in
  len > 0 && loop 0

let sequence_id_to_label = function Sequence_a -> "A" | Sequence_b -> "B"
let sequence_id_to_index = function Sequence_a -> 0 | Sequence_b -> 1

(* JSON helpers *)

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let json_find name fields =
  match Jsont.Json.find_mem name fields with
  | Some (_, v) -> Some v
  | None -> None

let json_bool_field fields name ~default =
  match json_find name fields with
  | Some (Jsont.Bool (b, _)) -> b
  | _ -> default

let json_str_int_pair fields name ~default =
  match json_find name fields with
  | Some (Jsont.Array ([ Jsont.String (s, _); Jsont.Number (f, _) ], _)) ->
      (s, int_of_float f)
  | _ -> default

(* Processors *)

let process_bert ~sep ~cls encodings ~add_special_tokens =
  if not add_special_tokens then encodings
  else
    let cls_str, cls_id = cls in
    let sep_str, sep_id = sep in
    let cls_tok tid = special_token ~id:cls_id ~token:cls_str ~type_id:tid in
    let sep_tok tid = special_token ~id:sep_id ~token:sep_str ~type_id:tid in
    match encodings with
    | [] -> []
    | [ encoding ] ->
        [
          Encoding.concat_list [ cls_tok 0; with_type_id encoding 0; sep_tok 0 ];
        ]
    | [ enc1; enc2 ] ->
        [
          Encoding.concat_list
            [
              cls_tok 0;
              with_type_id enc1 0;
              sep_tok 0;
              with_type_id enc2 1;
              sep_tok 1;
            ];
        ]
    | _ -> encodings

let process_roberta ~sep ~cls ~pad:_ ~trim_offsets:_ ~add_prefix_space:_
    encodings ~add_special_tokens =
  if not add_special_tokens then encodings
  else
    let cls_str, cls_id = cls in
    let sep_str, sep_id = sep in
    let cls_tok = special_token ~id:cls_id ~token:cls_str ~type_id:0 in
    let sep_tok = special_token ~id:sep_id ~token:sep_str ~type_id:0 in
    match encodings with
    | [] -> []
    | [ encoding ] ->
        [ Encoding.concat_list [ cls_tok; with_type_id encoding 0; sep_tok ] ]
    | [ enc1; enc2 ] ->
        [
          Encoding.concat_list
            [
              cls_tok;
              with_type_id enc1 0;
              sep_tok;
              sep_tok;
              with_type_id enc2 0;
              sep_tok;
            ];
        ]
    | _ -> encodings

let trim_offset enc_tokens idx (start, stop) =
  if start >= stop then (start, stop)
  else
    let token =
      if idx < Array.length enc_tokens then enc_tokens.(idx) else ""
    in
    let decoded = Pre_tokenizer.byte_level_decode token in
    let len = String.length decoded in
    let rec leading i =
      if i >= len then len else if is_ws decoded.[i] then leading (i + 1) else i
    in
    let rec trailing i =
      if i <= 0 then len
      else if is_ws decoded.[i - 1] then trailing (i - 1)
      else i
    in
    let lead = leading 0 in
    let trail = trailing len in
    let trimmed_lead = min (stop - start) lead in
    let trimmed_trail = min (stop - start - trimmed_lead) (len - trail) in
    let new_start = start + trimmed_lead in
    let new_stop = max new_start (stop - trimmed_trail) in
    (new_start, new_stop)

let process_byte_level ~trim_offsets encodings ~add_special_tokens:_ =
  if not trim_offsets then encodings
  else
    List.map
      (fun encoding ->
        let enc_tokens = Encoding.tokens encoding in
        let new_offsets =
          Array.mapi (trim_offset enc_tokens) (Encoding.offsets encoding)
        in
        Encoding.create ~ids:(Encoding.ids encoding)
          ~type_ids:(Encoding.type_ids encoding)
          ~tokens:enc_tokens
          ~words:(Encoding.word_ids encoding)
          ~offsets:new_offsets
          ~special_tokens_mask:(Encoding.special_tokens_mask encoding)
          ~attention_mask:(Encoding.attention_mask encoding)
          ~overflowing:(Encoding.overflowing encoding)
          ())
      encodings

(* Template parsing *)

let split_template_string str =
  let len = String.length str in
  let rec skip_ws i =
    if i >= len then len
    else match str.[i] with ' ' | '\t' -> skip_ws (i + 1) | _ -> i
  in
  let rec find_end i =
    if i >= len then len
    else match str.[i] with ' ' | '\t' -> i | _ -> find_end (i + 1)
  in
  let rec loop i acc =
    let i = skip_ws i in
    if i >= len then List.rev acc
    else
      let j = find_end i in
      loop j (String.sub str i (j - i) :: acc)
  in
  loop 0 []

let parse_sequence_base base =
  let lower = String.lowercase_ascii base in
  if lower = "$" || lower = "$a" then Some (Sequence_a, 0)
  else if lower = "$b" then Some (Sequence_b, 0)
  else if String.length base > 0 && base.[0] = '$' then
    let rest = String.sub base 1 (String.length base - 1) in
    if string_is_int rest then Some (Sequence_a, int_of_string rest) else None
  else None

let parse_template_piece_from_string ~special_lookup token =
  let parts = String.split_on_char ':' token in
  let base, explicit_type =
    match parts with
    | [ id; type_part ] when string_is_int type_part ->
        (id, Some (int_of_string type_part))
    | [ _; _ ] -> invalid_arg (err_type_id token)
    | [ id ] -> (id, None)
    | _ -> invalid_arg (err_piece token)
  in
  match parse_sequence_base base with
  | Some (seq_id, default_type) ->
      let type_id = Option.value ~default:default_type explicit_type in
      Piece_sequence { id = seq_id; type_id }
  | None ->
      if Hashtbl.mem special_lookup base then
        let type_id = Option.value ~default:0 explicit_type in
        Piece_special { key = base; type_id }
      else invalid_arg (err_unknown_special token)

let parse_template_string ~special_lookup str =
  List.map
    (parse_template_piece_from_string ~special_lookup)
    (split_template_string str)

let parse_sequence_id_json fields =
  match json_find "id" fields with
  | Some (Jsont.String (s, _)) -> (
      match String.lowercase_ascii s with
      | "a" -> Sequence_a
      | "b" -> Sequence_b
      | _ -> invalid_arg err_seq_id)
  | Some (Jsont.Number (v, _)) -> (
      match int_of_float v with
      | 0 -> Sequence_a
      | 1 -> Sequence_b
      | _ -> invalid_arg err_seq_id)
  | None -> Sequence_a
  | _ -> invalid_arg err_seq_id

let json_type_id fields =
  match json_find "type_id" fields with
  | Some (Jsont.Number (v, _)) -> int_of_float v
  | None -> 0
  | _ -> invalid_arg err_type_id_field

let parse_template_piece_from_json ~special_lookup json =
  match json with
  | Jsont.Object (outer_fields, _) -> (
      match json_find "Sequence" outer_fields with
      | Some (Jsont.Object (fields, _)) ->
          let id = parse_sequence_id_json fields in
          let type_id = json_type_id fields in
          Piece_sequence { id; type_id }
      | _ -> (
          match json_find "SpecialToken" outer_fields with
          | Some (Jsont.Object (fields, _)) ->
              let key =
                match json_find "id" fields with
                | Some (Jsont.String (s, _)) -> s
                | _ -> invalid_arg err_special_missing_id
              in
              if not (Hashtbl.mem special_lookup key) then
                invalid_arg (err_unknown_special key);
              let type_id = json_type_id fields in
              Piece_special { key; type_id }
          | _ -> invalid_arg err_unsupported_piece))
  | _ -> invalid_arg err_unsupported_piece

let parse_template_definition ~special_lookup = function
  | Jsont.String (s, _) -> parse_template_string ~special_lookup s
  | Jsont.Array (l, _) ->
      List.map (parse_template_piece_from_json ~special_lookup) l
  | Jsont.Null _ -> []
  | _ -> invalid_arg err_template_def

(* Template encoding *)

let build_encoding_from_pieces pieces source_encodings special_lookup =
  let ids_rev = ref [] in
  let type_ids_rev = ref [] in
  let tokens_rev = ref [] in
  let words_rev = ref [] in
  let offsets_rev = ref [] in
  let special_mask_rev = ref [] in
  let attention_rev = ref [] in
  let append ~id ~token ~word ~type_id ~offset ~special ~attention =
    ids_rev := id :: !ids_rev;
    type_ids_rev := type_id :: !type_ids_rev;
    tokens_rev := token :: !tokens_rev;
    words_rev := word :: !words_rev;
    offsets_rev := offset :: !offsets_rev;
    special_mask_rev := special :: !special_mask_rev;
    attention_rev := attention :: !attention_rev
  in
  let append_sequence seq_id type_id =
    let index = sequence_id_to_index seq_id in
    if index >= Array.length source_encodings then
      invalid_arg err_missing_sequence;
    let src = source_encodings.(index) in
    let src_ids = Encoding.ids src in
    let src_tokens = Encoding.tokens src in
    let src_words = Encoding.word_ids src in
    let src_offsets = Encoding.offsets src in
    let src_special = Encoding.special_tokens_mask src in
    let src_attention = Encoding.attention_mask src in
    let len = Array.length src_ids in
    for i = 0 to len - 1 do
      let token = if i < Array.length src_tokens then src_tokens.(i) else "" in
      let word = if i < Array.length src_words then src_words.(i) else None in
      let offset =
        if i < Array.length src_offsets then src_offsets.(i) else (0, 0)
      in
      let special =
        if i < Array.length src_special && src_special.(i) <> 0 then 1 else 0
      in
      let attention =
        if i < Array.length src_attention && src_attention.(i) <> 0 then 1
        else 0
      in
      append ~id:src_ids.(i) ~token ~word ~type_id ~offset ~special ~attention
    done
  in
  let append_special key type_id =
    match Hashtbl.find_opt special_lookup key with
    | None -> invalid_arg (err_unknown_special key)
    | Some special ->
        let rec loop ids tokens =
          match (ids, tokens) with
          | id :: rest_ids, token :: rest_tokens ->
              append ~id ~token ~word:None ~type_id ~offset:(0, 0) ~special:1
                ~attention:1;
              loop rest_ids rest_tokens
          | [], [] -> ()
          | _ -> invalid_arg (err_mismatch key)
        in
        loop special.value_ids special.value_tokens
  in
  List.iter
    (function
      | Piece_sequence { id; type_id } -> append_sequence id type_id
      | Piece_special { key; type_id } -> append_special key type_id)
    pieces;
  let to_array r = Array.of_list (List.rev !r) in
  Encoding.create ~ids:(to_array ids_rev) ~type_ids:(to_array type_ids_rev)
    ~tokens:(to_array tokens_rev) ~words:(to_array words_rev)
    ~offsets:(to_array offsets_rev)
    ~special_tokens_mask:(to_array special_mask_rev)
    ~attention_mask:(to_array attention_rev) ()

let process_template ~single ~pair ~special_tokens encodings ~add_special_tokens
    =
  if not add_special_tokens then encodings
  else
    let special_lookup = build_special_lookup special_tokens in
    let source = Array.of_list encodings in
    match Array.length source with
    | 0 -> []
    | 1 -> [ build_encoding_from_pieces single source special_lookup ]
    | 2 ->
        let pair =
          match pair with Some p -> p | None -> invalid_arg err_pair_required
        in
        [ build_encoding_from_pieces pair source special_lookup ]
    | _ -> encodings

(* Processing *)

let rec process_list processor encodings ~add_special_tokens =
  match processor with
  | Bert { sep; cls } -> process_bert ~sep ~cls encodings ~add_special_tokens
  | Roberta { sep; cls; pad; trim_offsets; add_prefix_space } ->
      process_roberta ~sep ~cls ~pad ~trim_offsets ~add_prefix_space encodings
        ~add_special_tokens
  | ByteLevel { trim_offsets } ->
      process_byte_level ~trim_offsets encodings ~add_special_tokens
  | Template { single; pair; special_tokens } ->
      process_template ~single ~pair ~special_tokens encodings
        ~add_special_tokens
  | Sequence processors ->
      List.fold_left
        (fun encs proc -> process_list proc encs ~add_special_tokens)
        encodings processors

let process processor ?pair enc ~add_special_tokens =
  let encodings = match pair with None -> [ enc ] | Some p -> [ enc; p ] in
  match process_list processor encodings ~add_special_tokens with
  | [ r ] -> r
  | r :: _ -> r
  | [] -> enc

let rec added_tokens processor ~is_pair =
  match processor with
  | Bert _ -> if is_pair then 3 else 2
  | Roberta _ -> if is_pair then 4 else 2
  | ByteLevel _ -> 0
  | Template { single; pair; special_tokens } ->
      let lookup = build_special_lookup special_tokens in
      let count_special pieces =
        List.fold_left
          (fun acc piece ->
            match piece with
            | Piece_special { key; _ } -> (
                match Hashtbl.find_opt lookup key with
                | Some tok -> acc + List.length tok.value_ids
                | None -> acc)
            | _ -> acc)
          0 pieces
      in
      if is_pair then
        match pair with
        | Some p -> count_special p
        | None -> count_special single
      else count_special single
  | Sequence processors ->
      List.fold_left
        (fun acc proc -> acc + added_tokens proc ~is_pair)
        0 processors

(* Constructors *)

let bert ~sep ~cls () = Bert { sep; cls }

let roberta ~sep ~cls ?(trim_offsets = true) ?(add_prefix_space = true) () =
  let pad = ("<pad>", 1) in
  Roberta { sep; cls; pad; trim_offsets; add_prefix_space }

let byte_level ?(trim_offsets = true) () = ByteLevel { trim_offsets }

let template ~single ?pair ?(special_tokens = []) () =
  let specials =
    List.map
      (fun (token, id) ->
        { key = token; value_ids = [ id ]; value_tokens = [ token ] })
      special_tokens
  in
  let lookup = build_special_lookup specials in
  let single = parse_template_string ~special_lookup:lookup single in
  let has_sequence pieces seq =
    List.exists
      (function Piece_sequence { id; _ } when id = seq -> true | _ -> false)
      pieces
  in
  let pair =
    match pair with
    | None -> None
    | Some p ->
        let tpl = parse_template_string ~special_lookup:lookup p in
        if not (has_sequence tpl Sequence_a && has_sequence tpl Sequence_b) then
          invalid_arg err_pair_must_ref_both;
        Some tpl
  in
  Template { single; pair; special_tokens = specials }

let sequence processors = Sequence processors

(* Formatting *)

let rec pp ppf = function
  | Bert { sep = sep_s, _; cls = cls_s, _ } ->
      Format.fprintf ppf "@[<2>Bert@ ~cls:%S@ ~sep:%S@]" cls_s sep_s
  | Roberta { sep = sep_s, _; cls = cls_s, _; _ } ->
      Format.fprintf ppf "@[<2>Roberta@ ~cls:%S@ ~sep:%S@]" cls_s sep_s
  | ByteLevel { trim_offsets } ->
      Format.fprintf ppf "@[<2>ByteLevel@ ~trim_offsets:%b@]" trim_offsets
  | Template _ -> Format.fprintf ppf "Template"
  | Sequence processors ->
      Format.fprintf ppf "@[<2>Sequence[@,%a]@]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
           pp)
        processors

(* Serialization *)

let token_pair_to_json (s, id) =
  Jsont.Json.list [ Jsont.Json.string s; Jsont.Json.int id ]

let template_to_json pieces =
  let piece_json tag id type_id =
    json_obj
      [ (tag, json_obj [ ("id", id); ("type_id", Jsont.Json.int type_id) ]) ]
  in
  Jsont.Json.list
    (List.map
       (function
         | Piece_sequence { id; type_id } ->
             piece_json "Sequence"
               (Jsont.Json.string (sequence_id_to_label id))
               type_id
         | Piece_special { key; type_id } ->
             piece_json "SpecialToken" (Jsont.Json.string key) type_id)
       pieces)

let rec to_json = function
  | Bert { sep; cls } ->
      json_obj
        [
          ("type", Jsont.Json.string "BertProcessing");
          ("sep", token_pair_to_json sep);
          ("cls", token_pair_to_json cls);
        ]
  | Roberta { sep; cls; pad; trim_offsets; add_prefix_space } ->
      json_obj
        [
          ("type", Jsont.Json.string "RobertaProcessing");
          ("sep", token_pair_to_json sep);
          ("cls", token_pair_to_json cls);
          ("pad", token_pair_to_json pad);
          ("trim_offsets", Jsont.Json.bool trim_offsets);
          ("add_prefix_space", Jsont.Json.bool add_prefix_space);
        ]
  | ByteLevel { trim_offsets } ->
      json_obj
        [
          ("type", Jsont.Json.string "ByteLevel");
          ("trim_offsets", Jsont.Json.bool trim_offsets);
        ]
  | Template { single; pair; special_tokens } ->
      let pair_json =
        match pair with
        | None -> Jsont.Json.null ()
        | Some p -> template_to_json p
      in
      let special_token_json tok =
        let ids = Jsont.Json.list (List.map Jsont.Json.int tok.value_ids) in
        let tokens =
          Jsont.Json.list (List.map Jsont.Json.string tok.value_tokens)
        in
        ( Jsont.Json.name tok.key,
          json_obj
            [
              ("id", Jsont.Json.string tok.key); ("ids", ids); ("tokens", tokens);
            ] )
      in
      let special_json =
        Jsont.Json.object' (List.map special_token_json special_tokens)
      in
      json_obj
        [
          ("type", Jsont.Json.string "TemplateProcessing");
          ("single", template_to_json single);
          ("pair", pair_json);
          ("special_tokens", special_json);
        ]
  | Sequence processors ->
      json_obj
        [
          ("type", Jsont.Json.string "Sequence");
          ("processors", Jsont.Json.list (List.map to_json processors));
        ]

(* Deserialization *)

let parse_special_token_json fields alias =
  let key =
    match json_find "id" fields with
    | Some (Jsont.String (s, _)) -> s
    | _ -> alias
  in
  let value_ids =
    match json_find "ids" fields with
    | Some (Jsont.Array (lst, _)) ->
        List.map
          (function
            | Jsont.Number (f, _) -> int_of_float f
            | v ->
                invalid_arg
                  (err_expected "number" (Format.asprintf "%a" Jsont.pp_json v)))
          lst
    | _ -> invalid_arg err_special_missing_ids
  in
  let value_tokens =
    match json_find "tokens" fields with
    | Some (Jsont.Array (lst, _)) ->
        List.map
          (function
            | Jsont.String (s, _) -> s
            | v ->
                invalid_arg
                  (err_expected "string" (Format.asprintf "%a" Jsont.pp_json v)))
          lst
    | _ -> [ key ]
  in
  if List.length value_ids <> List.length value_tokens then
    invalid_arg (err_mismatch key);
  { key; value_ids; value_tokens }

let parse_special_tokens_json fields =
  match json_find "special_tokens" fields with
  | Some (Jsont.Object (tokens, _)) ->
      List.map
        (fun ((alias, _), value) ->
          match value with
          | Jsont.Object (token_fields, _) ->
              parse_special_token_json token_fields alias
          | _ -> invalid_arg err_special_entry)
        tokens
  | Some v ->
      invalid_arg
        (err_expected "object for 'special_tokens'"
           (Format.asprintf "%a" Jsont.pp_json v))
  | None -> []

let rec of_json_exn json =
  match json with
  | Jsont.Object (fields, _) -> (
      match json_find "type" fields with
      | Some (Jsont.String ("BertProcessing", _)) ->
          let sep = json_str_int_pair fields "sep" ~default:("[SEP]", 102) in
          let cls = json_str_int_pair fields "cls" ~default:("[CLS]", 101) in
          Bert { sep; cls }
      | Some (Jsont.String ("RobertaProcessing", _)) ->
          let sep = json_str_int_pair fields "sep" ~default:("</s>", 2) in
          let cls = json_str_int_pair fields "cls" ~default:("<s>", 0) in
          let pad = json_str_int_pair fields "pad" ~default:("<pad>", 1) in
          let trim_offsets =
            json_bool_field fields "trim_offsets" ~default:true
          in
          let add_prefix_space =
            json_bool_field fields "add_prefix_space" ~default:true
          in
          Roberta { sep; cls; pad; trim_offsets; add_prefix_space }
      | Some (Jsont.String ("ByteLevel", _)) ->
          let trim_offsets =
            json_bool_field fields "trim_offsets" ~default:true
          in
          ByteLevel { trim_offsets }
      | Some (Jsont.String ("TemplateProcessing", _)) ->
          let special_tokens = parse_special_tokens_json fields in
          let lookup = build_special_lookup special_tokens in
          let single =
            match json_find "single" fields with
            | Some json -> parse_template_definition ~special_lookup:lookup json
            | None -> parse_template_string ~special_lookup:lookup "$A"
          in
          let pair =
            match json_find "pair" fields with
            | Some (Jsont.Null _) | None -> None
            | Some json ->
                Some (parse_template_definition ~special_lookup:lookup json)
          in
          Template { single; pair; special_tokens }
      | Some (Jsont.String ("Sequence", _)) -> (
          match json_find "processors" fields with
          | Some (Jsont.Array (procs, _)) ->
              Sequence (List.map of_json_exn procs)
          | _ -> failwith "expected array for 'processors'")
      | _ -> failwith "unsupported processor type")
  | _ -> failwith "expected JSON object"

let of_json json =
  try Ok (of_json_exn json) with
  | Failure msg -> Error msg
  | Invalid_argument msg -> Error msg
