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
      single : string;
      pair : string option;
      special_tokens : (string * int) list;
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
            Array.map
              (fun (start, stop) ->
                (* TODO: Implement proper offset trimming *)
                (start, stop))
              encoding.offsets;
        })
      encodings

(** Process template encoding *)
let process_template ~single ~pair:_ ~special_tokens:_ encodings
    ~add_special_tokens =
  if not add_special_tokens then encodings
  else
    (* TODO: Implement template processing *)
    let _ = single in
    encodings

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
  | Template _ -> 0 (* TODO: Calculate from template *)
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
  Template { single; pair; special_tokens }

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
          ("single", `String single);
          ("pair", match pair with None -> `Null | Some p -> `String p);
          ( "special_tokens",
            `List
              (List.map
                 (fun (s, i) -> `List [ `String s; `Int i ])
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
          (* HuggingFace uses structured template format with SpecialToken/Sequence *)
          (* For now, we'll create a simplified string template from the structure *)
          (* TODO: Implement full Template parsing compatible with HF format *)
          let single =
            match List.assoc_opt "single" fields with
            | Some (`String s) -> s
            | Some (`List _pieces) -> "$0" (* Simplified default *)
            | _ -> "$0"
          in
          let pair =
            match List.assoc_opt "pair" fields with
            | Some (`String p) -> Some p
            | Some (`List _pieces) -> Some "$A $B" (* Simplified default *)
            | Some `Null -> None
            | _ -> None
          in
          let special_tokens =
            match List.assoc_opt "special_tokens" fields with
            | Some (`Assoc tokens) ->
                (* HF format: {"[CLS]": {"id": "[CLS]", "ids": [101], "tokens":
                   ["[CLS]"]}} *)
                List.filter_map
                  (fun (_key, value) ->
                    match value with
                    | `Assoc fields -> (
                        match
                          ( List.assoc_opt "ids" fields,
                            List.assoc_opt "tokens" fields )
                        with
                        | ( Some (`List (`Int id :: _)),
                            Some (`List (`String tok :: _)) ) ->
                            Some (tok, id)
                        | _ -> None)
                    | _ -> None)
                  tokens
            | Some (`List tokens) ->
                (* Simplified list format *)
                List.filter_map
                  (fun t ->
                    match t with
                    | `List [ `String s; `Int i ] -> Some (s, i)
                    | _ -> None)
                  tokens
            | _ -> []
          in
          Template { single; pair; special_tokens }
      | Some (`String "Sequence") -> (
          match List.assoc_opt "processors" fields with
          | Some (`List procs) -> Sequence (List.map of_json procs)
          | _ -> failwith "Invalid Sequence processor")
      | _ -> failwith "Unknown processor type")
  | _ -> failwith "Invalid processor JSON"
