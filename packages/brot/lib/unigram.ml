(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compact trie for longest-prefix matching *)

type trie = {
  trie_ids : int array;
  child_starts : int array;
  edge_bytes : bytes;
  edge_targets : int array;
}

let build_trie token_to_ids =
  if Hashtbl.length token_to_ids = 0 then
    {
      trie_ids = [||];
      child_starts = [| 0 |];
      edge_bytes = Bytes.empty;
      edge_targets = [||];
    }
  else
    let cap = ref 256 in
    let ids = ref (Array.make !cap (-1)) in
    let ch = ref (Array.init !cap (fun _ -> Hashtbl.create 0)) in
    let n = ref 1 in
    !ch.(0) <- Hashtbl.create 64;
    let grow () =
      let new_cap = !cap * 2 in
      let new_ids = Array.make new_cap (-1) in
      Array.blit !ids 0 new_ids 0 !n;
      ids := new_ids;
      let new_ch =
        Array.init new_cap (fun i ->
            if i < !n then !ch.(i) else Hashtbl.create 0)
      in
      ch := new_ch;
      cap := new_cap
    in
    Hashtbl.iter
      (fun key id ->
        let cur = ref 0 in
        for i = 0 to String.length key - 1 do
          let byte = Char.code (String.unsafe_get key i) in
          let child =
            match Hashtbl.find_opt !ch.(!cur) byte with
            | Some c -> c
            | None ->
                if !n >= !cap then grow ();
                let c = !n in
                incr n;
                !ch.(c) <- Hashtbl.create 4;
                Hashtbl.add !ch.(!cur) byte c;
                c
          in
          cur := child
        done;
        !ids.(!cur) <- id)
      token_to_ids;
    let node_count = !n in
    let trie_ids = Array.init node_count (fun i -> !ids.(i)) in
    let child_starts = Array.make (node_count + 1) 0 in
    let total = ref 0 in
    for i = 0 to node_count - 1 do
      child_starts.(i) <- !total;
      total := !total + Hashtbl.length !ch.(i)
    done;
    child_starts.(node_count) <- !total;
    let edge_bytes = Bytes.create !total in
    let edge_targets = Array.make !total 0 in
    let pos = ref 0 in
    for i = 0 to node_count - 1 do
      Hashtbl.iter
        (fun byte child ->
          Bytes.unsafe_set edge_bytes !pos (Char.unsafe_chr byte);
          edge_targets.(!pos) <- child;
          incr pos)
        !ch.(i)
    done;
    for i = 0 to node_count - 1 do
      let start = child_starts.(i) in
      let stop = child_starts.(i + 1) in
      for j = start + 1 to stop - 1 do
        let kb = Bytes.unsafe_get edge_bytes j in
        let kt = edge_targets.(j) in
        let k = ref (j - 1) in
        while !k >= start && Bytes.unsafe_get edge_bytes !k > kb do
          Bytes.unsafe_set edge_bytes (!k + 1) (Bytes.unsafe_get edge_bytes !k);
          edge_targets.(!k + 1) <- edge_targets.(!k);
          decr k
        done;
        Bytes.unsafe_set edge_bytes (!k + 1) kb;
        edge_targets.(!k + 1) <- kt
      done
    done;
    { trie_ids; child_starts; edge_bytes; edge_targets }

let[@inline] trie_step trie node byte =
  let lo = ref (Array.unsafe_get trie.child_starts node) in
  let hi = ref (Array.unsafe_get trie.child_starts (node + 1) - 1) in
  let result = ref (-1) in
  while !lo <= !hi do
    let mid = !lo + ((!hi - !lo) asr 1) in
    let mid_byte = Char.code (Bytes.unsafe_get trie.edge_bytes mid) in
    if mid_byte = byte then (
      result := Array.unsafe_get trie.edge_targets mid;
      lo := !hi + 1)
    else if mid_byte < byte then lo := mid + 1
    else hi := mid - 1
  done;
  !result

let trie_longest_match trie text ~start =
  if Array.length trie.trie_ids = 0 then None
  else
    let text_len = String.length text in
    let last_id = ref (-1) in
    let last_end = ref start in
    let current = ref 0 in
    let stopped = ref false in
    let j = ref start in
    while !j < text_len && not !stopped do
      let child =
        trie_step trie !current (Char.code (String.unsafe_get text !j))
      in
      if child < 0 then stopped := true
      else (
        current := child;
        incr j;
        let tid = Array.unsafe_get trie.trie_ids child in
        if tid >= 0 then (
          last_id := tid;
          last_end := !j))
    done;
    if !last_id >= 0 then Some (!last_id, !last_end) else None

(* Model type *)

type t = {
  vocab : (string * float) array;
  token_to_ids : (string, int) Hashtbl.t;
  trie : trie;
}

let create vocab_list =
  let vocab = Array.of_list vocab_list in
  let token_to_ids = Hashtbl.create (Array.length vocab) in
  Array.iteri
    (fun idx (token, _) -> Hashtbl.replace token_to_ids token idx)
    vocab;
  let trie = build_trie token_to_ids in
  { vocab; token_to_ids; trie }

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
      match trie_longest_match model.trie text ~start:pos with
      | Some (id, end_pos) ->
          let s = String.sub text pos (end_pos - pos) in
          consume end_pos ((id, s, (pos, end_pos)) :: acc)
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
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc (json_to_string json));
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
      let words = Re.split (Re.compile (Re.rep1 (Re.set " \t\n\r"))) line in
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
