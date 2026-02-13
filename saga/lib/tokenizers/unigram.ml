(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type vocab_entry = string * float
type token_map = (string, int) Hashtbl.t
type vocab = vocab_entry array
type t = { vocab : vocab; token_to_ids : token_map }

let create vocab_list =
  let vocab = Array.of_list vocab_list in
  let token_to_ids = Hashtbl.create (Array.length vocab) in
  Array.iteri
    (fun idx (token, _) -> Hashtbl.replace token_to_ids token idx)
    vocab;
  { vocab; token_to_ids }

let token_to_id model token = Hashtbl.find_opt model.token_to_ids token

let id_to_token model id =
  if id >= 0 && id < Array.length model.vocab then
    let token, _ = model.vocab.(id) in
    Some token
  else None

let get_vocab model = Array.to_list model.vocab
let get_vocab_size model = Array.length model.vocab

let tokenize model text =
  let len = String.length text in
  let rec consume pos acc =
    if pos >= len then List.rev acc
    else if
      text.[pos] = ' '
      || text.[pos] = '\n'
      || text.[pos] = '\t'
      || text.[pos] = '\r'
    then consume (pos + 1) acc
    else
      let rec find_best_length length =
        if length = 0 then None
        else
          let s = String.sub text pos length in
          match token_to_id model s with
          | Some id -> Some (id, s, (pos, pos + length))
          | None -> find_best_length (length - 1)
      in
      match find_best_length (len - pos) with
      | Some token ->
          let _, _, (_, next_pos) = token in
          consume next_pos (token :: acc)
      | None ->
          let s = String.sub text pos 1 in
          let id = match token_to_id model s with Some id -> id | None -> 0 in
          consume (pos + 1) ((id, s, (pos, pos + 1)) :: acc)
  in
  consume 0 []

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let json_to_string j =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json j with
  | Ok s -> s
  | Error e -> failwith e

let save model ~folder () =
  let json_vocab =
    Array.to_list model.vocab
    |> List.mapi (fun id (token, prob) ->
        json_obj
          [
            ("id", Jsont.Json.int id);
            ("token", Jsont.Json.string token);
            ("prob", Jsont.Json.number prob);
          ])
  in
  let json =
    json_obj
      [
        ("type", Jsont.Json.string "Unigram");
        ("vocab", Jsont.Json.list json_vocab);
      ]
  in
  let path = Filename.concat folder "unigram.json" in
  let oc = open_out path in
  output_string oc (json_to_string json);
  close_out oc;
  [ "unigram.json" ]

let train ~vocab_size ~show_progress ~special_tokens ~shrinking_factor
    ~unk_token ~max_piece_length ~n_sub_iterations texts existing =
  let _ =
    ( show_progress,
      shrinking_factor,
      unk_token,
      max_piece_length,
      n_sub_iterations,
      existing )
  in
  let counts = Hashtbl.create 10000 in
  List.iter
    (fun line ->
      let words = Str.split (Str.regexp "[ \t\n\r]+") line in
      List.iter
        (fun word ->
          if word <> "" then
            Hashtbl.replace counts word
              (1 + Option.value ~default:0 (Hashtbl.find_opt counts word)))
        words)
    texts;

  let total =
    Hashtbl.fold (fun _ count acc -> acc + count) counts 0 |> float_of_int
  in
  let sorted =
    Hashtbl.fold (fun token count acc -> (token, count) :: acc) counts []
    |> List.sort (fun (_, c1) (_, c2) -> compare c2 c1)
  in

  let take_first n lst =
    let rec aux i = function
      | [] -> []
      | _ when i = 0 -> []
      | x :: xs -> x :: aux (i - 1) xs
    in
    aux n lst
  in

  let selected = take_first vocab_size sorted in
  let vocab_with_probs =
    special_tokens
    |> List.map (fun token -> (token, 1.0 /. float_of_int (vocab_size + 1)))
    |> fun specials ->
    specials
    @ List.map
        (fun (token, count) ->
          let prob = if total = 0. then 0. else float_of_int count /. total in
          (token, prob))
        selected
  in
  let model = create vocab_with_probs in
  (model, special_tokens)
