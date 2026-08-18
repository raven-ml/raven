(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  vocab : (string, int) Hashtbl.t;
  vocab_r : (int, string) Hashtbl.t;
  token_table : string array;
  len_table : int array;
  unk_token : string;
  unk_id : int;
}

let create ?(vocab = []) ?(unk_token = "<unk>") () =
  let size = max 1 (List.length vocab) in
  let vocab_tbl = Hashtbl.create size in
  let vocab_r_tbl = Hashtbl.create size in
  List.iter
    (fun (token, id) ->
      Hashtbl.replace vocab_tbl token id;
      Hashtbl.replace vocab_r_tbl id token)
    vocab;
  let max_id = List.fold_left (fun acc (_, id) -> max acc id) (-1) vocab in
  let token_table = Array.make (max_id + 1) "" in
  List.iter (fun (token, id) -> token_table.(id) <- token) vocab;
  let len_table = Array.map String.length token_table in
  let unk_id =
    match Hashtbl.find_opt vocab_tbl unk_token with Some id -> id | None -> -1
  in
  if unk_id >= 0 then len_table.(unk_id) <- 0;
  {
    vocab = vocab_tbl;
    vocab_r = vocab_r_tbl;
    token_table;
    len_table;
    unk_token;
    unk_id;
  }

let add_token vocab vocab_r token id =
  Hashtbl.replace vocab token id;
  Hashtbl.replace vocab_r id token

(* A word absent from the vocabulary is the unknown token, and nothing at all
   when the vocabulary does not hold that either. *)
let encode_into model ids text ~pos ~len =
  if len > 0 then begin
    let word =
      if pos = 0 && len = String.length text then text
      else String.sub text pos len
    in
    match Hashtbl.find_opt model.vocab word with
    | Some id -> Ints.add ids id
    | None -> if model.unk_id >= 0 then Ints.add ids model.unk_id
  end

let token_table model = model.token_table
let len_table model = model.len_table
let token_to_id model token = Hashtbl.find_opt model.vocab token
let id_to_token model id = Hashtbl.find_opt model.vocab_r id

let get_vocab model =
  Hashtbl.fold (fun token id acc -> (token, id) :: acc) model.vocab []

let get_vocab_size model = Hashtbl.length model.vocab

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
        json_obj
          [ ("token", Jsont.Json.string token); ("id", Jsont.Json.int id) ])
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
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc (json_to_string json));
  [ "wordlevel.json" ]

let train ~vocab_size ~min_frequency ~show_progress ~special_tokens word_counts
    =
  let _ = show_progress in
  let items =
    List.filter (fun (_, count) -> count >= min_frequency) word_counts
    |> List.sort (fun (_, c1) (_, c2) -> compare c2 c1)
  in
  let specials = List.mapi (fun i token -> (token, i)) special_tokens in
  (* The words are numbered after the special tokens, and count against
     [vocab_size] together with them. *)
  let vocab_items = ref [] in
  let idx = ref (List.length special_tokens) in
  List.iter
    (fun (word, _) ->
      if !idx < vocab_size then (
        vocab_items := (word, !idx) :: !vocab_items;
        incr idx))
    items;
  (create ~vocab:(specials @ List.rev !vocab_items) (), special_tokens)
