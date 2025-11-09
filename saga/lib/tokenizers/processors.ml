(** Post-processing module for tokenization output. *)

type encoding = {
  ids : int array;
  type_ids : int array;
  tokens : string array;
  offsets : (int * int) array;
  special_tokens_mask : int array;
  attention_mask : int array;
  overflowing : encoding list;
  sequence_ranges : (int * int * int) list;
}
(** Type representing an encoding to be processed *)

(** Main post-processor type *)
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

type t =
  | Bert of { sep : string * int; cls : string * int }
  | Roberta of {
      sep : string * int;
      cls : string * int;
      pad : string * int;
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

(** Helper functions *)
let create_empty_encoding () =
  {
    ids = [||];
    type_ids = [||];
    tokens = [||];
    offsets = [||];
    special_tokens_mask = [||];
    attention_mask = [||];
    overflowing = [];
    sequence_ranges = [];
  }

let add_token encoding token_str token_id type_id is_special =
  {
    encoding with
    ids = Array.append encoding.ids [| token_id |];
    tokens = Array.append encoding.tokens [| token_str |];
    type_ids = Array.append encoding.type_ids [| type_id |];
    special_tokens_mask =
      Array.append encoding.special_tokens_mask
        [| (if is_special then 1 else 0) |];
    attention_mask = Array.append encoding.attention_mask [| 1 |];
    offsets = Array.append encoding.offsets [| (0, 0) |];
  }

(** Process BERT encoding *)
let process_bert ~sep ~cls encodings ~add_special_tokens =
  if not add_special_tokens then encodings
  else
    let cls_str, cls_id = cls in
    let sep_str, sep_id = sep in

    match encodings with
    | [] -> []
    | [ encoding ] ->
        (* Single sequence: [CLS] seq [SEP] *)
        let result = create_empty_encoding () in
        let result = add_token result cls_str cls_id 0 true in
        let result =
          {
            result with
            ids = Array.append result.ids encoding.ids;
            tokens = Array.append result.tokens encoding.tokens;
            type_ids =
              Array.append result.type_ids
                (Array.make (Array.length encoding.ids) 0);
            special_tokens_mask =
              Array.append result.special_tokens_mask
                (Array.make (Array.length encoding.ids) 0);
            attention_mask =
              Array.append result.attention_mask
                (Array.make (Array.length encoding.ids) 1);
            offsets = Array.append result.offsets encoding.offsets;
          }
        in
        let result = add_token result sep_str sep_id 0 true in
        [ result ]
    | [ enc1; enc2 ] ->
        (* Pair of sequences: [CLS] seq1 [SEP] seq2 [SEP] *)
        let result = create_empty_encoding () in
        let result = add_token result cls_str cls_id 0 true in

        (* Add first sequence *)
        let result =
          {
            result with
            ids = Array.append result.ids enc1.ids;
            tokens = Array.append result.tokens enc1.tokens;
            type_ids =
              Array.append result.type_ids
                (Array.make (Array.length enc1.ids) 0);
            special_tokens_mask =
              Array.append result.special_tokens_mask
                (Array.make (Array.length enc1.ids) 0);
            attention_mask =
              Array.append result.attention_mask
                (Array.make (Array.length enc1.ids) 1);
            offsets = Array.append result.offsets enc1.offsets;
          }
        in
        let result = add_token result sep_str sep_id 0 true in

        (* Add second sequence *)
        let result =
          {
            result with
            ids = Array.append result.ids enc2.ids;
            tokens = Array.append result.tokens enc2.tokens;
            type_ids =
              Array.append result.type_ids
                (Array.make (Array.length enc2.ids) 1);
            special_tokens_mask =
              Array.append result.special_tokens_mask
                (Array.make (Array.length enc2.ids) 0);
            attention_mask =
              Array.append result.attention_mask
                (Array.make (Array.length enc2.ids) 1);
            offsets = Array.append result.offsets enc2.offsets;
          }
        in
        let result = add_token result sep_str sep_id 1 true in
        [ result ]
    | _ -> encodings (* More than 2 sequences not supported *)

(** Process RoBERTa encoding *)
let process_roberta ~sep ~cls ~pad:_ ~trim_offsets:_ ~add_prefix_space:_
    encodings ~add_special_tokens =
  if not add_special_tokens then encodings
  else
    (* RoBERTa uses: <s> seq </s> for single, <s> seq1 </s></s> seq2 </s> for
       pairs *)
    let cls_str, cls_id = cls in
    let sep_str, sep_id = sep in

    match encodings with
    | [] -> []
    | [ encoding ] ->
        (* Single sequence: <s> seq </s> *)
        let result = create_empty_encoding () in
        let result = add_token result cls_str cls_id 0 true in
        let result =
          {
            result with
            ids = Array.append result.ids encoding.ids;
            tokens = Array.append result.tokens encoding.tokens;
            type_ids =
              Array.append result.type_ids
                (Array.make (Array.length encoding.ids) 0);
            special_tokens_mask =
              Array.append result.special_tokens_mask
                (Array.make (Array.length encoding.ids) 0);
            attention_mask =
              Array.append result.attention_mask
                (Array.make (Array.length encoding.ids) 1);
            offsets = Array.append result.offsets encoding.offsets;
          }
        in
        let result = add_token result sep_str sep_id 0 true in
        [ result ]
    | [ enc1; enc2 ] ->
        (* Pair: <s> seq1 </s></s> seq2 </s> *)
        let result = create_empty_encoding () in
        let result = add_token result cls_str cls_id 0 true in

        (* Add first sequence *)
        let result =
          {
            result with
            ids = Array.append result.ids enc1.ids;
            tokens = Array.append result.tokens enc1.tokens;
            type_ids =
              Array.append result.type_ids
                (Array.make (Array.length enc1.ids) 0);
            special_tokens_mask =
              Array.append result.special_tokens_mask
                (Array.make (Array.length enc1.ids) 0);
            attention_mask =
              Array.append result.attention_mask
                (Array.make (Array.length enc1.ids) 1);
            offsets = Array.append result.offsets enc1.offsets;
          }
        in
        let result = add_token result sep_str sep_id 0 true in
        let result = add_token result sep_str sep_id 0 true in

        (* Add second sequence *)
        let result =
          {
            result with
            ids = Array.append result.ids enc2.ids;
            tokens = Array.append result.tokens enc2.tokens;
            type_ids =
              Array.append result.type_ids
                (Array.make (Array.length enc2.ids) 0);
            special_tokens_mask =
              Array.append result.special_tokens_mask
                (Array.make (Array.length enc2.ids) 0);
            attention_mask =
              Array.append result.attention_mask
                (Array.make (Array.length enc2.ids) 1);
            offsets = Array.append result.offsets enc2.offsets;
          }
        in
        let result = add_token result sep_str sep_id 0 true in
        [ result ]
    | _ -> encodings

(** Process byte-level encoding *)
let process_byte_level ~trim_offsets encodings ~add_special_tokens:_ =
  if not trim_offsets then encodings
  else
    (* Trim whitespace from offsets *)
    List.map
      (fun encoding ->
        {
          encoding with
          offsets =
            Array.mapi
              (fun idx (start, stop) ->
                if start >= stop then (start, stop)
                else
                  let token =
                    if idx < Array.length encoding.tokens then
                      encoding.tokens.(idx)
                    else ""
                  in
                  let decoded = Pre_tokenizers.byte_level_decode token in
                  let len = String.length decoded in
                  let is_ws c =
                    match c with
                    | ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c' -> true
                    | _ -> false
                  in
                  let rec leading i =
                    if i >= len then len
                    else if is_ws decoded.[i] then leading (i + 1)
                    else i
                  in
                  let rec trailing i =
                    if i <= 0 then len
                    else if is_ws decoded.[i - 1] then trailing (i - 1)
                    else i
                  in
                  let lead = leading 0 in
                  let trail = trailing len in
                  let trimmed_lead = min (stop - start) lead in
                  let trimmed_trail =
                    min (stop - start - trimmed_lead) (len - trail)
                  in
                  let new_start = start + trimmed_lead in
                  let new_stop = max new_start (stop - trimmed_trail) in
                  (new_start, new_stop))
              encoding.offsets;
        })
      encodings

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
  let rec aux i acc =
    let i = skip_ws i in
    if i >= len then List.rev acc
    else
      let j = find_end i in
      let token = String.sub str i (j - i) in
      aux j (token :: acc)
  in
  aux 0 []

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
    | [ _; type_part ] when type_part = "" ->
        invalid_arg
          (Printf.sprintf "Processors.template: invalid type id in '%s'" token)
    | [ _; _ ] ->
        invalid_arg
          (Printf.sprintf "Processors.template: invalid type id in '%s'" token)
    | [ id ] -> (id, None)
    | _ ->
        invalid_arg
          (Printf.sprintf "Processors.template: invalid piece syntax '%s'" token)
  in
  match parse_sequence_base base with
  | Some (seq_id, default_type) ->
      let type_id = Option.value ~default:default_type explicit_type in
      Piece_sequence { id = seq_id; type_id }
  | None ->
      let key = base in
      if Hashtbl.mem special_lookup key then
        let type_id = Option.value ~default:0 explicit_type in
        Piece_special { key; type_id }
      else
        invalid_arg
          (Printf.sprintf
             "Processors.template: unknown special token '%s'. Ensure the \
              token is listed in special_tokens."
             token)

let parse_template_string ~special_lookup str =
  split_template_string str
  |> List.map (parse_template_piece_from_string ~special_lookup)

let sequence_id_to_label = function Sequence_a -> "A" | Sequence_b -> "B"
let sequence_id_to_index = function Sequence_a -> 0 | Sequence_b -> 1

let parse_template_piece_from_json ~special_lookup json =
  match json with
  | `Assoc [ ("Sequence", `Assoc fields) ] ->
      let id =
        match List.assoc_opt "id" fields with
        | Some (`String s) -> (
            match String.lowercase_ascii s with
            | "a" -> Sequence_a
            | "b" -> Sequence_b
            | _ ->
                invalid_arg "Processors.template: invalid sequence identifier")
        | Some (`Int v) -> (
            match v with
            | 0 -> Sequence_a
            | 1 -> Sequence_b
            | _ -> invalid_arg "Processors.template: sequence id must be 0 or 1"
            )
        | None -> Sequence_a
        | _ -> invalid_arg "Processors.template: invalid sequence identifier"
      in
      let type_id =
        match List.assoc_opt "type_id" fields with
        | Some (`Int v) -> v
        | None -> 0
        | _ -> invalid_arg "Processors.template: invalid type id for Sequence"
      in
      Piece_sequence { id; type_id }
  | `Assoc [ ("SpecialToken", `Assoc fields) ] ->
      let token_key =
        match List.assoc_opt "id" fields with
        | Some (`String s) -> s
        | _ ->
            invalid_arg "Processors.template: SpecialToken missing identifier"
      in
      let () =
        if not (Hashtbl.mem special_lookup token_key) then
          invalid_arg
            (Printf.sprintf "Processors.template: unknown special token '%s'"
               token_key)
      in
      let type_id =
        match List.assoc_opt "type_id" fields with
        | Some (`Int v) -> v
        | None -> 0
        | _ ->
            invalid_arg "Processors.template: invalid type id for SpecialToken"
      in
      Piece_special { key = token_key; type_id }
  | _ -> invalid_arg "Processors.template: unsupported template piece format"

let parse_template_definition ~special_lookup = function
  | `String s -> parse_template_string ~special_lookup s
  | `List l -> List.map (parse_template_piece_from_json ~special_lookup) l
  | `Null -> []
  | _ -> invalid_arg "Processors.template: invalid template definition"

let template_to_json pieces =
  `List
    (List.map
       (function
         | Piece_sequence { id; type_id } ->
             let fields =
               [
                 ("id", `String (sequence_id_to_label id));
                 ("type_id", `Int type_id);
               ]
             in
             `Assoc [ ("Sequence", `Assoc fields) ]
         | Piece_special { key; type_id } ->
             let fields = [ ("id", `String key); ("type_id", `Int type_id) ] in
             `Assoc [ ("SpecialToken", `Assoc fields) ])
       pieces)

let build_encoding_from_pieces pieces source_encodings special_lookup =
  let ids_rev = ref [] in
  let type_ids_rev = ref [] in
  let tokens_rev = ref [] in
  let offsets_rev = ref [] in
  let special_mask_rev = ref [] in
  let attention_rev = ref [] in
  let append_entry ~id ~token ~type_id ~offset ~special ~attention =
    ids_rev := id :: !ids_rev;
    type_ids_rev := type_id :: !type_ids_rev;
    tokens_rev := token :: !tokens_rev;
    offsets_rev := offset :: !offsets_rev;
    special_mask_rev := special :: !special_mask_rev;
    attention_rev := attention :: !attention_rev
  in
  let append_sequence seq_id type_id =
    let index = sequence_id_to_index seq_id in
    if index >= Array.length source_encodings then
      invalid_arg "Processors.template: missing input sequence"
    else
      let src = source_encodings.(index) in
      let len = Array.length src.ids in
      for i = 0 to len - 1 do
        let token =
          if i < Array.length src.tokens then src.tokens.(i) else ""
        in
        let offset =
          if i < Array.length src.offsets then src.offsets.(i) else (0, 0)
        in
        let special =
          if i < Array.length src.special_tokens_mask then
            src.special_tokens_mask.(i)
          else 0
        in
        let attention =
          if i < Array.length src.attention_mask then src.attention_mask.(i)
          else 1
        in
        let id =
          if i < Array.length src.ids then src.ids.(i)
          else
            invalid_arg
              "Processors.template: encoding ids and tokens length mismatch"
        in
        append_entry ~id ~token ~type_id ~offset
          ~special:(if special <> 0 then 1 else 0)
          ~attention:(if attention <> 0 then 1 else 0)
      done
  in
  List.iter
    (function
      | Piece_sequence { id; type_id } -> append_sequence id type_id
      | Piece_special { key; type_id } -> (
          match Hashtbl.find_opt special_lookup key with
          | None ->
              invalid_arg
                (Printf.sprintf
                   "Processors.template: unknown special token '%s'" key)
          | Some special ->
              let rec append ids tokens =
                match (ids, tokens) with
                | id :: rest_ids, token :: rest_tokens ->
                    append_entry ~id ~token ~type_id ~offset:(0, 0) ~special:1
                      ~attention:1;
                    append rest_ids rest_tokens
                | [], [] -> ()
                | _ ->
                    invalid_arg
                      (Printf.sprintf
                         "Processors.template: mismatched ids/tokens length \
                          for special token '%s'"
                         key)
              in
              append special.value_ids special.value_tokens))
    pieces;
  let to_array lst_ref = Array.of_list (List.rev !lst_ref) in
  {
    ids = to_array ids_rev;
    type_ids = to_array type_ids_rev;
    tokens = to_array tokens_rev;
    offsets = to_array offsets_rev;
    special_tokens_mask = to_array special_mask_rev;
    attention_mask = to_array attention_rev;
    overflowing = [];
    sequence_ranges = [];
  }

(** Process template encoding *)
let process_template ~single ~pair ~special_tokens encodings ~add_special_tokens
    =
  if not add_special_tokens then encodings
  else
    let special_lookup = build_special_lookup special_tokens in
    let source = Array.of_list encodings in
    match Array.length source with
    | 0 -> []
    | 1 ->
        let encoding =
          build_encoding_from_pieces single source special_lookup
        in
        [ encoding ]
    | 2 ->
        let template =
          match pair with
          | Some p -> p
          | None ->
              invalid_arg
                "Processors.template: pair template required for two sequences"
        in
        let encoding =
          build_encoding_from_pieces template source special_lookup
        in
        [ encoding ]
    | _ -> encodings

(** Main process function *)
let rec process processor encodings ~add_special_tokens =
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
        (fun encs proc -> process proc encs ~add_special_tokens)
        encodings processors

(** Get number of added tokens *)
let rec added_tokens processor ~is_pair =
  match processor with
  | Bert _ ->
      if is_pair then 3
      else 2 (* [CLS] ... [SEP] or [CLS] ... [SEP] ... [SEP] *)
  | Roberta _ ->
      if is_pair then 4 else 2 (* <s> ... </s> or <s> ... </s></s> ... </s> *)
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

(** Constructors *)
let bert ~sep ~cls () = Bert { sep; cls }

let roberta ~sep ~cls ?(trim_offsets = true) ?(add_prefix_space = true) () =
  let pad = ("<pad>", 1) in
  (* Default pad token *)
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
          invalid_arg
            "Processors.template: pair template must reference both $A and $B";
        Some tpl
  in
  Template { single; pair; special_tokens = specials }

let sequence processors = Sequence processors

(** Serialization *)
let rec to_json = function
  | Bert { sep = sep_str, sep_id; cls = cls_str, cls_id } ->
      `Assoc
        [
          ("type", `String "BertProcessing");
          ("sep", `List [ `String sep_str; `Int sep_id ]);
          ("cls", `List [ `String cls_str; `Int cls_id ]);
        ]
  | Roberta
      {
        sep = sep_str, sep_id;
        cls = cls_str, cls_id;
        pad = pad_str, pad_id;
        trim_offsets;
        add_prefix_space;
      } ->
      `Assoc
        [
          ("type", `String "RobertaProcessing");
          ("sep", `List [ `String sep_str; `Int sep_id ]);
          ("cls", `List [ `String cls_str; `Int cls_id ]);
          ("pad", `List [ `String pad_str; `Int pad_id ]);
          ("trim_offsets", `Bool trim_offsets);
          ("add_prefix_space", `Bool add_prefix_space);
        ]
  | ByteLevel { trim_offsets } ->
      `Assoc
        [ ("type", `String "ByteLevel"); ("trim_offsets", `Bool trim_offsets) ]
  | Template { single; pair; special_tokens } ->
      `Assoc
        [
          ("type", `String "TemplateProcessing");
          ("single", template_to_json single);
          ( "pair",
            match pair with None -> `Null | Some p -> template_to_json p );
          ( "special_tokens",
            `Assoc
              (List.map
                 (fun tok ->
                   ( tok.key,
                     `Assoc
                       [
                         ("id", `String tok.key);
                         ( "ids",
                           `List (List.map (fun id -> `Int id) tok.value_ids) );
                         ( "tokens",
                           `List
                             (List.map (fun s -> `String s) tok.value_tokens) );
                       ] ))
                 special_tokens) );
        ]
  | Sequence processors ->
      `Assoc
        [
          ("type", `String "Sequence");
          ("processors", `List (List.map to_json processors));
        ]

let rec of_json = function
  | `Assoc fields -> (
      match List.assoc_opt "type" fields with
      | Some (`String "BertProcessing") ->
          let sep =
            match List.assoc_opt "sep" fields with
            | Some (`List [ `String s; `Int i ]) -> (s, i)
            | _ -> ("[SEP]", 102)
          in
          let cls =
            match List.assoc_opt "cls" fields with
            | Some (`List [ `String s; `Int i ]) -> (s, i)
            | _ -> ("[CLS]", 101)
          in
          Bert { sep; cls }
      | Some (`String "RobertaProcessing") ->
          let sep =
            match List.assoc_opt "sep" fields with
            | Some (`List [ `String s; `Int i ]) -> (s, i)
            | _ -> ("</s>", 2)
          in
          let cls =
            match List.assoc_opt "cls" fields with
            | Some (`List [ `String s; `Int i ]) -> (s, i)
            | _ -> ("<s>", 0)
          in
          let pad =
            match List.assoc_opt "pad" fields with
            | Some (`List [ `String s; `Int i ]) -> (s, i)
            | _ -> ("<pad>", 1)
          in
          let trim_offsets =
            match List.assoc_opt "trim_offsets" fields with
            | Some (`Bool b) -> b
            | _ -> true
          in
          let add_prefix_space =
            match List.assoc_opt "add_prefix_space" fields with
            | Some (`Bool b) -> b
            | _ -> true
          in
          Roberta { sep; cls; pad; trim_offsets; add_prefix_space }
      | Some (`String "ByteLevel") ->
          let trim_offsets =
            match List.assoc_opt "trim_offsets" fields with
            | Some (`Bool b) -> b
            | _ -> true
          in
          ByteLevel { trim_offsets }
      | Some (`String "TemplateProcessing") ->
          let special_tokens =
            match List.assoc_opt "special_tokens" fields with
            | Some (`Assoc tokens) ->
                List.map
                  (fun (alias, value) ->
                    match value with
                    | `Assoc token_fields ->
                        let key =
                          match List.assoc_opt "id" token_fields with
                          | Some (`String s) -> s
                          | _ -> alias
                        in
                        let value_ids =
                          match List.assoc_opt "ids" token_fields with
                          | Some (`List lst) ->
                              List.map
                                (function
                                  | `Int v -> v
                                  | json ->
                                      invalid_arg
                                        (Printf.sprintf
                                           "Processors.template: invalid id \
                                            value %s for special token"
                                           (Yojson.Basic.to_string json)))
                                lst
                          | _ ->
                              invalid_arg
                                "Processors.template: special token missing ids"
                        in
                        let value_tokens =
                          match List.assoc_opt "tokens" token_fields with
                          | Some (`List lst) ->
                              List.map
                                (function
                                  | `String s -> s
                                  | json ->
                                      invalid_arg
                                        (Printf.sprintf
                                           "Processors.template: invalid token \
                                            value %s for special token"
                                           (Yojson.Basic.to_string json)))
                                lst
                          | _ -> [ key ]
                        in
                        if List.length value_ids <> List.length value_tokens
                        then
                          invalid_arg
                            (Printf.sprintf
                               "Processors.template: mismatched ids/tokens for \
                                special token '%s'"
                               key);
                        { key; value_ids; value_tokens }
                    | _ ->
                        invalid_arg
                          "Processors.template: invalid special token entry")
                  tokens
            | Some json ->
                invalid_arg
                  (Printf.sprintf
                     "Processors.template: invalid special_tokens definition %s"
                     (Yojson.Basic.to_string json))
            | None -> []
          in
          let lookup = build_special_lookup special_tokens in
          let single =
            match List.assoc_opt "single" fields with
            | Some json -> parse_template_definition ~special_lookup:lookup json
            | None -> parse_template_string ~special_lookup:lookup "$A"
          in
          let pair =
            match List.assoc_opt "pair" fields with
            | Some `Null -> None
            | Some json ->
                Some (parse_template_definition ~special_lookup:lookup json)
            | None -> None
          in
          Template { single; pair; special_tokens }
      | Some (`String "Sequence") -> (
          match List.assoc_opt "processors" fields with
          | Some (`List procs) -> Sequence (List.map of_json procs)
          | _ -> failwith "Invalid Sequence processor")
      | _ -> failwith "Unknown processor type")
  | _ -> failwith "Invalid processor JSON"
