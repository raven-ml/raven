(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

exception Error of string

type vocab = (string, int) Hashtbl.t
type vocab_r = (int, string) Hashtbl.t
type token = { id : int; value : string; offsets : int * int }

type config = {
  vocab : vocab;
  unk_token : string;
  continuing_subword_prefix : string;
  max_input_chars_per_word : int;
}

type t = {
  vocab : vocab;
  vocab_r : vocab_r;
  unk_token : string;
  continuing_subword_prefix : string;
  max_input_chars_per_word : int;
}

let create_internal vocab unk_token continuing_subword_prefix
    max_input_chars_per_word =
  let vocab_r = Hashtbl.create (Hashtbl.length vocab) in
  Hashtbl.iter (fun k v -> Hashtbl.add vocab_r v k) vocab;
  (* Only raise error if vocabulary is non-empty but missing UNK token *)
  if Hashtbl.length vocab > 0 && not (Hashtbl.mem vocab unk_token) then
    raise (Error "WordPiece error: Missing [UNK] token from the vocabulary");
  {
    vocab;
    vocab_r;
    unk_token;
    continuing_subword_prefix;
    max_input_chars_per_word;
  }

let create (cfg : config) =
  create_internal cfg.vocab cfg.unk_token cfg.continuing_subword_prefix
    cfg.max_input_chars_per_word

let default () = create_internal (Hashtbl.create 0) "[UNK]" "##" 100

let read_file ~vocab_file =
  let vocab = Hashtbl.create 10000 in
  let ic = open_in vocab_file in
  Fun.protect ~finally:(fun () -> close_in ic) (fun () ->
      let index = ref 0 in
      (try
         while true do
           let line = input_line ic in
           let token = String.trim line in
           if token <> "" then (
             Hashtbl.add vocab token !index;
             incr index)
         done
       with End_of_file -> ());
      vocab)

let read_bytes bytes =
  let vocab = Hashtbl.create 10000 in
  let str = Bytes.to_string bytes in
  let lines = String.split_on_char '\n' str in
  List.iteri
    (fun index line ->
      let token = String.trim line in
      if token <> "" then Hashtbl.add vocab token index)
    lines;
  vocab

let from_file ~vocab_file =
  let vocab = read_file ~vocab_file in
  (* Use default values for BERT-style WordPiece *)
  create_internal vocab "[UNK]" "##" 100

let from_file_with_config ~vocab_file ~unk_token ~continuing_subword_prefix
    ~max_input_chars_per_word =
  let vocab = read_file ~vocab_file in
  create_internal vocab unk_token continuing_subword_prefix
    max_input_chars_per_word

let tokenize model sequence =
  (* Handle empty vocabulary case *)
  if Hashtbl.length model.vocab = 0 then []
  else
    let char_count =
      let len = String.length sequence in
      let count = ref 0 in
      let rec loop i =
        if i >= len then !count
        else
          let d = String.get_utf_8_uchar sequence i in
          incr count;
          loop (i + Uchar.utf_decode_length d)
      in
      loop 0
    in
    if char_count > model.max_input_chars_per_word then
      let id = Hashtbl.find model.vocab model.unk_token in
      [ { id; value = model.unk_token; offsets = (0, String.length sequence) } ]
    else
      let rec tokenize_greedy start acc =
        if start >= String.length sequence then List.rev acc
        else
          let rec find_longest_match end_pos =
            if end_pos <= start then None
            else
              let substr = String.sub sequence start (end_pos - start) in
              let token_str =
                if start > 0 then model.continuing_subword_prefix ^ substr
                else substr
              in
              match Hashtbl.find_opt model.vocab token_str with
              | Some id ->
                  Some { id; value = token_str; offsets = (start, end_pos) }
              | None ->
                  let new_end =
                    let rec find_char_start pos =
                      if pos <= start then start
                      else if Char.code sequence.[pos - 1] land 0xC0 <> 0x80
                      then pos - 1
                      else find_char_start (pos - 1)
                    in
                    find_char_start end_pos
                  in
                  if new_end <= start then None else find_longest_match new_end
          in
          match find_longest_match (String.length sequence) with
          | Some token -> tokenize_greedy (snd token.offsets) (token :: acc)
          | None ->
              let id = Hashtbl.find model.vocab model.unk_token in
              [
                {
                  id;
                  value = model.unk_token;
                  offsets = (0, String.length sequence);
                };
              ]
      in
      tokenize_greedy 0 []

let token_to_id model token = Hashtbl.find_opt model.vocab token
let id_to_token model id = Hashtbl.find_opt model.vocab_r id
let get_vocab model = Hashtbl.fold (fun k v acc -> (k, v) :: acc) model.vocab []
let get_vocab_size model = Hashtbl.length model.vocab
let get_unk_token model = model.unk_token
let get_continuing_subword_prefix model = model.continuing_subword_prefix
let get_max_input_chars_per_word model = model.max_input_chars_per_word

let save model ~path ?name () =
  let vocab_file =
    match name with
    | Some n -> Filename.concat path (n ^ "-vocab.txt")
    | None -> Filename.concat path "vocab.txt"
  in
  let vocab_list =
    Hashtbl.fold (fun k v acc -> (v, k) :: acc) model.vocab []
    |> List.sort compare
    |> List.map (fun (_, k) -> k)
  in
  let oc = open_out vocab_file in
  Fun.protect ~finally:(fun () -> close_out oc) (fun () ->
      List.iter
        (fun token ->
          output_string oc token;
          output_char oc '\n')
        vocab_list);
  vocab_file

let from_bpe bpe =
  let vocab = Hashtbl.create (Bpe.get_vocab_size bpe) in
  List.iter (fun (k, id) -> Hashtbl.add vocab k id) (Bpe.get_vocab bpe);
  let unk_token =
    match Bpe.get_unk_token bpe with Some u -> u | None -> "[UNK]"
  in
  let continuing_subword_prefix =
    match Bpe.get_continuing_subword_prefix bpe with
    | Some p -> p
    | None -> "##"
  in
  create_internal vocab unk_token continuing_subword_prefix 100

(* ───── JSON helpers ───── *)

let json_of_string s =
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> v
  | Error e -> failwith e

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

(* ───── Serialization ───── *)

let to_json model =
  let vocab_list =
    Hashtbl.fold (fun k v acc -> (v, k) :: acc) model.vocab []
    |> List.sort compare
    |> List.map (fun (_, k) ->
        (Jsont.Json.name k, Jsont.Json.int (Hashtbl.find model.vocab k)))
  in
  json_obj
    [
      ("type", Jsont.Json.string "WordPiece");
      ("unk_token", Jsont.Json.string model.unk_token);
      ( "continuing_subword_prefix",
        Jsont.Json.string model.continuing_subword_prefix );
      ("max_input_chars_per_word", Jsont.Json.int model.max_input_chars_per_word);
      ("vocab", Jsont.Json.object' vocab_list);
    ]

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

let of_json json =
  match json with
  | Jsont.Object (fields, _) ->
      let get_field name =
        match Jsont.Json.find_mem name fields with
        | Some (_, v) -> v
        | None -> raise (Error ("Missing field: " ^ name))
      in
      let () =
        match get_field "type" with
        | Jsont.String ("WordPiece", _) -> ()
        | _ -> raise (Error "Invalid type")
        | exception _ -> ()
      in
      let unk_token =
        match get_field "unk_token" with
        | Jsont.String (s, _) -> s
        | _ -> raise (Error "Invalid unk_token")
      in
      let continuing_subword_prefix =
        match get_field "continuing_subword_prefix" with
        | Jsont.String (s, _) -> s
        | _ -> raise (Error "Invalid continuing_subword_prefix")
      in
      let max_input_chars_per_word =
        match get_field "max_input_chars_per_word" with
        | Jsont.Number (f, _) -> int_of_float f
        | _ -> raise (Error "Invalid max_input_chars_per_word")
      in
      let vocab_json = json_mem "vocab" json in
      let vocab =
        match vocab_json with
        | Jsont.Object (pairs, _) ->
            let h = Hashtbl.create (List.length pairs) in
            List.iter
              (fun ((k, _), v) ->
                match v with
                | Jsont.Number (f, _) -> Hashtbl.add h k (int_of_float f)
                | _ -> raise (Error "Invalid vocab entry"))
              pairs;
            h
        | _ -> raise (Error "Invalid vocab")
      in
      create_internal vocab unk_token continuing_subword_prefix
        max_input_chars_per_word
  | _ -> raise (Error "Invalid JSON structure")

let from_bytes bytes =
  let str = Bytes.to_string bytes in
  of_json (json_of_string str)

(* ───── Trainer ───── *)

let train ~min_frequency ~vocab_size ~show_progress ~special_tokens
    ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
    ~end_of_word_suffix texts existing =
  let _ = existing in
  (* WordPiece training uses BPE algorithm internally *)
  let bpe_trained, result_tokens =
    Bpe.train ~min_frequency ~vocab_size ~show_progress ~special_tokens
      ~limit_alphabet ~initial_alphabet
      ~continuing_subword_prefix:(Some continuing_subword_prefix)
      ~end_of_word_suffix ~max_token_length:None texts None
  in
  let wordpiece_model = from_bpe bpe_trained in
  (wordpiece_model, result_tokens)
