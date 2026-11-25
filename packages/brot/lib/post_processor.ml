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
let err_missing_sequence = "template references a sequence not provided"
let err_pair_must_ref_both = "pair template must reference both $A and $B"
let err_sequences n = strf "a template receives %d sequences, not 1 or 2" n
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
      trim_offsets : bool;
      add_prefix_space : bool;
    }
  | ByteLevel of { add_prefix_space : bool; trim_offsets : bool }
  | Template of {
      single : template;
      pair : template;
      special_tokens : special_token list;
    }
  | Sequence of t list

(* Helpers *)

let special_token ~id ~token ~type_id =
  Encoding.token ~id ~token ~offset:(0, 0) ~type_id ~special:true

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

(* The readers here raise; [of_json] turns what they raise into an [Error]. *)
let ok = function Ok v -> v | Error msg -> failwith msg

(* A special token as ["[SEP]", 102], which has no default. *)
let json_token fields name =
  match json_find name fields with
  | Some (Jsont.Array ([ Jsont.String (s, _); id ], _)) ->
      (s, ok (Json_text.int_of name id))
  | Some _ -> failwith (strf "member '%s': expected [token, id]" name)
  | None -> failwith (strf "missing member '%s'" name)

(* Processors *)

(* The characters trimmed off each end of a byte-level token are the space
   marker U+0120 and those with the White_Space property. A tab or a newline
   encodes to U+0109 or U+010A, which are letters, so a token made of them keeps
   its offsets. [chars] is how many characters the token has in all, which is
   what says that the whole of it is trimmable. *)
let count_trimmable token =
  let stop = String.length token in
  let rec loop i leading trailing chars at_start =
    if i >= stop then (leading, trailing, chars)
    else
      let c = Char_class.at token i ~stop in
      let len = Char_class.at_len c in
      let trimmable =
        (len = 2 && token.[i] = '\xc4' && token.[i + 1] = '\xa0')
        || Char_class.at_category c = Char_class.whitespace
      in
      if trimmable then
        loop (i + len)
          (if at_start then leading + 1 else leading)
          (trailing + 1) (chars + 1) at_start
      else loop (i + len) leading 0 (chars + 1) false
  in
  loop 0 0 0 0 true

let trim_offset ~add_prefix_space enc_tokens idx (start, stop) =
  if start >= stop then (start, stop)
  else
    let token =
      if idx < Array.length enc_tokens then enc_tokens.(idx) else ""
    in
    let leading, trailing, chars = count_trimmable token in
    if leading = 0 && trailing = 0 then (start, stop)
    else
      (* The space a byte-level pre-tokenizer prepends is not in the input, so
         trimming it would move the offset past the first real character. Two of
         them it did not add, and both go. *)
      let leading =
        if add_prefix_space && leading = 1 && (idx = 0 || start = 0) then 0
        else leading
      in
      if trailing >= chars then
        (* Nothing but white space: the token stands for no text, so its span
           closes rather than losing a byte per character. The counts are
           characters of the token and the span is bytes of the input, and the
           two only agree when the token spells the bytes it stands for — a
           prepended space, which is in no input, is exactly when they do
           not. *)
        let at = if leading = 0 then start else stop in
        (at, at)
      else
        let start = if leading = 0 then start else min (start + leading) stop in
        let stop =
          if trailing > 0 && stop >= trailing then max (stop - trailing) start
          else stop
        in
        (start, stop)

let trim_encoding ~add_prefix_space encoding =
  let enc_tokens = Encoding.tokens encoding in
  let new_offsets =
    Array.mapi
      (trim_offset ~add_prefix_space enc_tokens)
      (Encoding.offsets encoding)
  in
  Encoding.create ~ids:(Encoding.ids encoding)
    ~type_ids:(Encoding.type_ids encoding)
    ~tokens:enc_tokens
    ~words:(Encoding.word_ids encoding)
    ~offsets:new_offsets
    ~special_tokens_mask:(Encoding.special_tokens_mask encoding)
    ~attention_mask:(Encoding.attention_mask encoding)
    ~overflowing:(Encoding.overflowing encoding)
    ()

(* A processor maps the sequences it receives to a list of sequences, which
   [process] merges once, after the last processor of a [sequence]. [bert] and
   [roberta] read the first sequence they receive as their single one and each
   later one as a second, and a template gives each of its pieces as a sequence
   of its own. So after a template of [[X] $A], [bert] receives [[X]] and [A]
   and gives [[CLS] [X] [SEP]] and [A [SEP]].

   A sequence is held as the encodings it is made of, joined only to trim its
   offsets and at the end, so that its tokens are copied once. *)

let trim_sequences ~add_prefix_space sequences =
  List.map
    (fun parts -> [ trim_encoding ~add_prefix_space (Encoding.concat parts) ])
    sequences

let retype parts type_id =
  List.map (fun part -> Encoding.with_type_id part type_id) parts

let special (token, id) ~type_id = special_token ~id ~token ~type_id

(* [bert] puts [cls] before its first sequence and [sep] after each, typed 0 in
   the first sequence and 1 in the others. *)
let process_bert ~sep ~cls sequences ~add_special_tokens =
  if not add_special_tokens then sequences
  else
    List.mapi
      (fun i parts ->
        let parts = if i = 0 then special cls ~type_id:0 :: parts else parts in
        parts @ [ special sep ~type_id:(min i 1) ])
      sequences

let process_roberta ~sep ~cls ~trim_offsets ~add_prefix_space sequences
    ~add_special_tokens =
  (* RoBERTa has a single segment: the second sequence of a pair keeps type id
     0, with or without special tokens. *)
  let sequences =
    List.map
      (fun parts ->
        if trim_offsets then
          [
            Encoding.with_type_id
              (trim_encoding ~add_prefix_space (Encoding.concat parts))
              0;
          ]
        else retype parts 0)
      sequences
  in
  (* [roberta] puts [cls] before its first sequence and [sep] before each other
     one, and [sep] after each. *)
  if not add_special_tokens then sequences
  else
    let cls = special cls ~type_id:0 and sep = special sep ~type_id:0 in
    List.mapi
      (fun i parts -> ((if i = 0 then cls else sep) :: parts) @ [ sep ])
      sequences

let process_byte_level ~add_prefix_space ~trim_offsets sequences
    ~add_special_tokens:_ =
  if trim_offsets then trim_sequences ~add_prefix_space sequences else sequences

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
  | Some (Jsont.Number _ as v) -> (
      match ok (Json_text.int_of "id" v) with
      | 0 -> Sequence_a
      | 1 -> Sequence_b
      | _ -> invalid_arg err_seq_id)
  | None -> Sequence_a
  | _ -> invalid_arg err_seq_id

let json_type_id fields = ok (Json_text.int fields "type_id" ~default:0)

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

let sequence_at source index =
  if index >= Array.length source then invalid_arg err_missing_sequence;
  source.(index)

let find_special special_tokens key =
  match List.find_opt (fun tok -> String.equal tok.key key) special_tokens with
  | Some special -> special
  | None -> invalid_arg (err_unknown_special key)

(* A piece keeps the arrays of what it was built from, so a sequence that has
   not worked out its tokens, offsets or word ids still has not. *)
let sequence_of_piece source special_tokens = function
  | Piece_sequence { id; type_id } ->
      retype (sequence_at source (sequence_id_to_index id)) type_id
  | Piece_special { key; type_id } ->
      let special = find_special special_tokens key in
      if List.compare_lengths special.value_ids special.value_tokens <> 0 then
        invalid_arg (err_mismatch key);
      List.map2
        (fun id token -> special_token ~id ~token ~type_id)
        special.value_ids special.value_tokens

(* A template reads the first sequence it receives as [$A] and the second as
   [$B]. More it cannot place. *)
let template_for ~single ~pair = function
  | 1 -> single
  | 2 -> pair
  | n -> invalid_arg (err_sequences n)

(* Without special tokens the template still decides the order and type ids of
   the sequences; only its special pieces are dropped. *)
let process_template ~single ~pair ~special_tokens sequences ~add_special_tokens
    =
  match sequences with
  | [] -> []
  | _ ->
      let source = Array.of_list sequences in
      List.filter_map
        (function
          | Piece_special _ when not add_special_tokens -> None
          | piece -> Some (sequence_of_piece source special_tokens piece))
        (template_for ~single ~pair (Array.length source))

(* Processing *)

let rec process_list processor sequences ~add_special_tokens =
  match processor with
  | Bert { sep; cls } -> process_bert ~sep ~cls sequences ~add_special_tokens
  | Roberta { sep; cls; trim_offsets; add_prefix_space } ->
      process_roberta ~sep ~cls ~trim_offsets ~add_prefix_space sequences
        ~add_special_tokens
  | ByteLevel { add_prefix_space; trim_offsets } ->
      process_byte_level ~add_prefix_space ~trim_offsets sequences
        ~add_special_tokens
  | Template { single; pair; special_tokens } ->
      process_template ~single ~pair ~special_tokens sequences
        ~add_special_tokens
  | Sequence processors ->
      List.fold_left
        (fun sequences proc -> process_list proc sequences ~add_special_tokens)
        sequences processors

(* Each sequence enters typed by its place, as tokenizers' [process] types it:
   the first 0 and the second 1. *)
let process processor ?pair enc ~add_special_tokens =
  let sequences =
    match pair with
    | None -> [ [ Encoding.with_type_id enc 0 ] ]
    | Some p ->
        [ [ Encoding.with_type_id enc 0 ]; [ Encoding.with_type_id p 1 ] ]
  in
  Encoding.concat
    (List.concat (process_list processor sequences ~add_special_tokens))

(* Inserted tokens are the ones [process] masks as special, so what it does to
   empty sequences counts them, and what it does to one token of input shows
   what it puts around a sequence. *)

let added_tokens processor ~is_pair =
  let pair = if is_pair then Some Encoding.empty else None in
  Encoding.length
    (process processor ?pair Encoding.empty ~add_special_tokens:true)

let affixes processor ~add_special_tokens =
  let input =
    Encoding.create ~ids:[| 0 |] ~type_ids:[| 0 |] ~tokens:[| "a" |]
      ~words:[| Some 0 |]
      ~offsets:[| (0, 1) |]
      ~special_tokens_mask:[| 0 |] ~attention_mask:[| 1 |] ()
  in
  let enc = process processor input ~add_special_tokens in
  let ids = Encoding.ids enc and mask = Encoding.special_tokens_mask enc in
  let n = Array.length ids in
  let inputs = List.filter (fun i -> mask.(i) = 0) (List.init n Fun.id) in
  match inputs with
  | [ i ] -> Some (Array.sub ids 0 i, Array.sub ids (i + 1) (n - i - 1))
  | _ -> None

(* Constructors *)

let bert ~sep ~cls = Bert { sep; cls }

let roberta ~sep ~cls ?(trim_offsets = true) ?(add_prefix_space = true) () =
  Roberta { sep; cls; trim_offsets; add_prefix_space }

let byte_level ?(add_prefix_space = true) ?(trim_offsets = true) () =
  ByteLevel { add_prefix_space; trim_offsets }

let default_pair =
  [
    Piece_sequence { id = Sequence_a; type_id = 0 };
    Piece_sequence { id = Sequence_b; type_id = 1 };
  ]

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
    | None -> default_pair
    | Some p ->
        let tpl = parse_template_string ~special_lookup:lookup p in
        if not (has_sequence tpl Sequence_a && has_sequence tpl Sequence_b) then
          invalid_arg err_pair_must_ref_both;
        tpl
  in
  Template { single; pair; special_tokens = specials }

let sequence processors = Sequence processors

(* Formatting *)

let rec pp ppf = function
  | Bert { sep = sep_s, _; cls = cls_s, _ } ->
      Format.fprintf ppf "@[<2>Bert@ ~cls:%S@ ~sep:%S@]" cls_s sep_s
  | Roberta { sep = sep_s, _; cls = cls_s, _; trim_offsets; add_prefix_space }
    ->
      Format.fprintf ppf
        "@[<2>Roberta@ ~cls:%S@ ~sep:%S@ ~trim_offsets:%b@ \
         ~add_prefix_space:%b@]"
        cls_s sep_s trim_offsets add_prefix_space
  | ByteLevel { add_prefix_space; trim_offsets } ->
      Format.fprintf ppf
        "@[<2>ByteLevel@ ~add_prefix_space:%b@ ~trim_offsets:%b@]"
        add_prefix_space trim_offsets
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
  | Roberta { sep; cls; trim_offsets; add_prefix_space } ->
      json_obj
        [
          ("type", Jsont.Json.string "RobertaProcessing");
          ("sep", token_pair_to_json sep);
          ("cls", token_pair_to_json cls);
          ("trim_offsets", Jsont.Json.bool trim_offsets);
          ("add_prefix_space", Jsont.Json.bool add_prefix_space);
        ]
  | ByteLevel { add_prefix_space; trim_offsets } ->
      json_obj
        [
          ("type", Jsont.Json.string "ByteLevel");
          ("add_prefix_space", Jsont.Json.bool add_prefix_space);
          ("trim_offsets", Jsont.Json.bool trim_offsets);
        ]
  | Template { single; pair; special_tokens } ->
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
          ("pair", template_to_json pair);
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
        List.map (fun id -> ok (Json_text.int_of "ids" id)) lst
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
          let sep = json_token fields "sep" in
          let cls = json_token fields "cls" in
          Bert { sep; cls }
      | Some (Jsont.String ("RobertaProcessing", _)) ->
          let sep = json_token fields "sep" in
          let cls = json_token fields "cls" in
          let trim_offsets =
            ok (Json_text.bool fields "trim_offsets" ~default:true)
          in
          let add_prefix_space =
            ok (Json_text.bool fields "add_prefix_space" ~default:true)
          in
          Roberta { sep; cls; trim_offsets; add_prefix_space }
      | Some (Jsont.String ("ByteLevel", _)) ->
          let add_prefix_space =
            ok (Json_text.bool fields "add_prefix_space" ~default:true)
          in
          let trim_offsets =
            ok (Json_text.bool fields "trim_offsets" ~default:true)
          in
          ByteLevel { add_prefix_space; trim_offsets }
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
            | Some (Jsont.Null _) | None -> default_pair
            | Some json -> parse_template_definition ~special_lookup:lookup json
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
