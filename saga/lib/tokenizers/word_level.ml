(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type vocab = (string, int) Hashtbl.t
type vocab_r = (int, string) Hashtbl.t
type t = { vocab : vocab; vocab_r : vocab_r; unk_token : string }

let create ?(vocab = []) ?(unk_token = "<unk>") () =
  let size = max 1 (List.length vocab) in
  let vocab_tbl = Hashtbl.create size in
  let vocab_r_tbl = Hashtbl.create size in
  List.iter
    (fun (token, id) ->
      Hashtbl.replace vocab_tbl token id;
      Hashtbl.replace vocab_r_tbl id token)
    vocab;
  { vocab = vocab_tbl; vocab_r = vocab_r_tbl; unk_token }

let add_token vocab vocab_r token id =
  Hashtbl.replace vocab token id;
  Hashtbl.replace vocab_r id token

let tokenize model text =
  if String.length text = 0 then []
  else
    (* Match HuggingFace tokenizers semantics exactly: 1. Try to find token in
       vocab 2. Fall back to UNK token if available 3. Return empty list if
       neither exists (error case) *)
    match Hashtbl.find_opt model.vocab text with
    | Some id -> [ (id, text, (0, String.length text)) ]
    | None -> (
        match Hashtbl.find_opt model.vocab model.unk_token with
        | Some unk_id -> [ (unk_id, model.unk_token, (0, String.length text)) ]
        | None -> [] (* Token not found and no UNK token - return empty *))

let token_to_id model token = Hashtbl.find_opt model.vocab token
let id_to_token model id = Hashtbl.find_opt model.vocab_r id

let get_vocab model =
  Hashtbl.fold (fun token id acc -> (token, id) :: acc) model.vocab []

let get_vocab_size model = Hashtbl.length model.vocab

let add_tokens model tokens =
  let start_id = Hashtbl.length model.vocab in
  let count = ref 0 in
  List.iteri
    (fun i token ->
      if not (Hashtbl.mem model.vocab token) then (
        add_token model.vocab model.vocab_r token (start_id + i);
        incr count))
    tokens;
  !count

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let json_to_string j =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json j with
  | Ok s -> s
  | Error e -> failwith e

let save model ~folder () =
  let vocab_items =
    get_vocab model
    |> List.sort (fun (_, id1) (_, id2) -> compare id1 id2)
    |> List.map (fun (token, id) ->
        json_obj [ ("token", Jsont.Json.string token); ("id", Jsont.Json.int id) ])
  in
  let json =
    json_obj
      [
        ("type", Jsont.Json.string "WordLevel");
        ("unk_token", Jsont.Json.string model.unk_token);
        ("vocab", Jsont.Json.list vocab_items);
      ]
  in
  let path = Filename.concat folder "wordlevel.json" in
  let oc = open_out path in
  output_string oc (json_to_string json);
  close_out oc;
  [ "wordlevel.json" ]

let train ~vocab_size ~min_frequency ~show_progress ~special_tokens texts
    existing =
  let _ = show_progress in
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

  let items =
    Hashtbl.fold
      (fun word count acc ->
        if count >= min_frequency then (word, count) :: acc else acc)
      counts []
    |> List.sort (fun (_, c1) (_, c2) -> compare c2 c1)
  in
  let vocab_items = ref [] in
  let idx = ref 0 in
  List.iter
    (fun token ->
      if !idx < vocab_size then (
        vocab_items := (fst token, !idx) :: !vocab_items;
        incr idx))
    items;
  let vocab_items = List.rev !vocab_items in

  let specials = List.mapi (fun i token -> (token, i)) special_tokens in
  let vocab = specials @ vocab_items in
  let model =
    match existing with
    | Some model ->
        model.vocab |> Hashtbl.clear;
        model.vocab_r |> Hashtbl.clear;
        List.iter
          (fun (token, id) -> add_token model.vocab model.vocab_r token id)
          vocab;
        model
    | None -> create ~vocab ()
  in
  (model, special_tokens)
