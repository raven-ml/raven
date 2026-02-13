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
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
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
      let seq_len = String.length sequence in
      let prefix = model.continuing_subword_prefix in
      let prefix_len = String.length prefix in
      let rec tokenize_greedy start acc =
        if start >= seq_len then List.rev acc
        else
          (* Build the full candidate string once: prefix ^ sequence[start..] *)
          let remainder = seq_len - start in
          let full_candidate =
            if start > 0 then prefix ^ String.sub sequence start remainder
            else String.sub sequence start remainder
          in
          let candidate_len = String.length full_candidate in
          let rec find_longest_match cand_end =
            let token_end_in_seq =
              if start > 0 then cand_end - prefix_len else cand_end
            in
            if token_end_in_seq <= 0 then None
            else
              let token_str = String.sub full_candidate 0 cand_end in
              match Hashtbl.find_opt model.vocab token_str with
              | Some id ->
                  let end_pos = start + token_end_in_seq in
                  Some { id; value = token_str; offsets = (start, end_pos) }
              | None ->
                  (* Step back one UTF-8 character *)
                  let new_cand_end =
                    let rec find_char_boundary pos =
                      if pos <= if start > 0 then prefix_len else 0 then 0
                      else if
                        Char.code full_candidate.[pos - 1] land 0xC0 <> 0x80
                      then pos - 1
                      else find_char_boundary (pos - 1)
                    in
                    find_char_boundary cand_end
                  in
                  if new_cand_end <= if start > 0 then prefix_len else 0 then
                    None
                  else find_longest_match new_cand_end
          in
          match find_longest_match candidate_len with
          | Some token -> tokenize_greedy (snd token.offsets) (token :: acc)
          | None ->
              let id = Hashtbl.find model.vocab model.unk_token in
              [ { id; value = model.unk_token; offsets = (0, seq_len) } ]
      in
      tokenize_greedy 0 []

let tokenize_ids model sequence =
  if Hashtbl.length model.vocab = 0 then [||]
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
      [| id |]
    else
      let seq_len = String.length sequence in
      let prefix = model.continuing_subword_prefix in
      let prefix_len = String.length prefix in
      let ids = ref [] in
      let n = ref 0 in
      let rec greedy start =
        if start >= seq_len then ()
        else
          let remainder = seq_len - start in
          let full_candidate =
            if start > 0 then prefix ^ String.sub sequence start remainder
            else String.sub sequence start remainder
          in
          let candidate_len = String.length full_candidate in
          let rec find cand_end =
            let token_end =
              if start > 0 then cand_end - prefix_len else cand_end
            in
            if token_end <= 0 then None
            else
              let token_str = String.sub full_candidate 0 cand_end in
              match Hashtbl.find_opt model.vocab token_str with
              | Some id -> Some (id, start + token_end)
              | None ->
                  let new_end =
                    let rec boundary pos =
                      if pos <= if start > 0 then prefix_len else 0 then 0
                      else if
                        Char.code full_candidate.[pos - 1] land 0xC0 <> 0x80
                      then pos - 1
                      else boundary (pos - 1)
                    in
                    boundary cand_end
                  in
                  if new_end <= if start > 0 then prefix_len else 0 then None
                  else find new_end
          in
          match find candidate_len with
          | Some (id, next_start) ->
              ids := id :: !ids;
              incr n;
              greedy next_start
          | None ->
              let unk_id = Hashtbl.find model.vocab model.unk_token in
              ids := [ unk_id ];
              n := 1
      in
      greedy 0;
      let result = Array.make !n 0 in
      List.iteri (fun i id -> result.(!n - 1 - i) <- id) !ids;
      result

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
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () ->
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
