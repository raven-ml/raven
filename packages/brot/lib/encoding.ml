(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  ids : int array;
  type_ids : int array;
  tokens : string array;
  words : int option array;
  offsets : (int * int) array;
  special_tokens_mask : int array;
  attention_mask : int array;
  mutable overflowing : t list;
  sequence_ranges : (int, int * int) Hashtbl.t;
}

(* Constructors *)

let empty_ranges : (int, int * int) Hashtbl.t = Hashtbl.create 0

let empty =
  {
    ids = [||];
    type_ids = [||];
    tokens = [||];
    words = [||];
    offsets = [||];
    special_tokens_mask = [||];
    attention_mask = [||];
    overflowing = [];
    sequence_ranges = empty_ranges;
  }

let create ~ids ~type_ids ~tokens ~words ~offsets ~special_tokens_mask
    ~attention_mask ?(overflowing = []) () =
  {
    ids;
    type_ids;
    tokens;
    words;
    offsets;
    special_tokens_mask;
    attention_mask;
    overflowing;
    sequence_ranges = empty_ranges;
  }

let token ~id ~token ~offset ~type_id ~special =
  {
    ids = [| id |];
    type_ids = [| type_id |];
    tokens = [| token |];
    words = [| None |];
    offsets = [| offset |];
    special_tokens_mask = [| (if special then 1 else 0) |];
    attention_mask = [| 1 |];
    overflowing = [];
    sequence_ranges = empty_ranges;
  }

let from_tokens tokens ~type_id =
  let n = List.length tokens in
  let ids = Array.make n 0 in
  let token_strs = Array.make n "" in
  let offsets = Array.make n (0, 0) in
  List.iteri
    (fun i (id, tok, off) ->
      ids.(i) <- id;
      token_strs.(i) <- tok;
      offsets.(i) <- off)
    tokens;
  {
    ids;
    tokens = token_strs;
    offsets;
    words = Array.make n None;
    type_ids = Array.make n type_id;
    attention_mask = Array.make n 1;
    special_tokens_mask = Array.make n 0;
    overflowing = [];
    sequence_ranges = empty_ranges;
  }

let concat a b =
  {
    ids = Array.append a.ids b.ids;
    type_ids = Array.append a.type_ids b.type_ids;
    tokens = Array.append a.tokens b.tokens;
    words = Array.append a.words b.words;
    offsets = Array.append a.offsets b.offsets;
    special_tokens_mask =
      Array.append a.special_tokens_mask b.special_tokens_mask;
    attention_mask = Array.append a.attention_mask b.attention_mask;
    overflowing = a.overflowing;
    sequence_ranges = a.sequence_ranges;
  }

let concat_list encodings =
  match encodings with
  | [] -> empty
  | [ single ] -> single
  | first :: _ ->
      let total =
        List.fold_left (fun acc t -> acc + Array.length t.ids) 0 encodings
      in
      let ids = Array.make total 0 in
      let type_ids = Array.make total 0 in
      let tokens = Array.make total "" in
      let words = Array.make total None in
      let offsets = Array.make total (0, 0) in
      let special_tokens_mask = Array.make total 0 in
      let attention_mask = Array.make total 0 in
      let pos = ref 0 in
      List.iter
        (fun t ->
          let n = Array.length t.ids in
          Array.blit t.ids 0 ids !pos n;
          Array.blit t.type_ids 0 type_ids !pos n;
          Array.blit t.tokens 0 tokens !pos n;
          Array.blit t.words 0 words !pos n;
          Array.blit t.offsets 0 offsets !pos n;
          Array.blit t.special_tokens_mask 0 special_tokens_mask !pos n;
          Array.blit t.attention_mask 0 attention_mask !pos n;
          pos := !pos + n)
        encodings;
      {
        ids;
        type_ids;
        tokens;
        words;
        offsets;
        special_tokens_mask;
        attention_mask;
        overflowing = first.overflowing;
        sequence_ranges = first.sequence_ranges;
      }

(* Accessors *)

let is_empty t = Array.length t.ids = 0
let length t = Array.length t.ids
let ids t = t.ids
let type_ids t = t.type_ids
let tokens t = t.tokens
let word_ids t = t.words
let offsets t = t.offsets
let special_tokens_mask t = t.special_tokens_mask
let attention_mask t = t.attention_mask
let overflowing t = t.overflowing

(* Truncation *)

let slice t start len =
  {
    ids = Array.sub t.ids start len;
    type_ids = Array.sub t.type_ids start len;
    tokens = Array.sub t.tokens start len;
    words = Array.sub t.words start len;
    offsets = Array.sub t.offsets start len;
    special_tokens_mask = Array.sub t.special_tokens_mask start len;
    attention_mask = Array.sub t.attention_mask start len;
    overflowing = [];
    sequence_ranges = empty_ranges;
  }

let truncate t ~max_length ~stride ~direction =
  let encoding_len = length t in
  if max_length >= encoding_len then t
  else if max_length = 0 then { empty with overflowing = [ t ] }
  else begin
    assert (stride < max_length);
    let step = max_length - stride in
    let ranges =
      match direction with
      | `Right ->
          let rec loop start acc =
            if start >= encoding_len then List.rev acc
            else
              let stop = min (start + max_length) encoding_len in
              loop (start + step) ((start, stop) :: acc)
          in
          loop 0 []
      | `Left ->
          let rec loop stop acc =
            if stop <= 0 then acc
            else
              let start = max 0 (stop - max_length) in
              loop (stop - step) ((start, stop) :: acc)
          in
          loop encoding_len []
    in
    match ranges with
    | [] -> empty
    | (start, stop) :: rest ->
        let enc = slice t start (stop - start) in
        enc.overflowing <-
          List.map (fun (start, stop) -> slice t start (stop - start)) rest;
        enc
  end

(* Pad *)

let pad_array src n fill direction =
  let src_len = Array.length src in
  let dst = Array.make (src_len + n) fill in
  let off = match direction with `Left -> n | `Right -> 0 in
  Array.blit src 0 dst off src_len;
  dst

let rec pad t ~target_length ~pad_id ~pad_type_id ~pad_token ~direction =
  let overflowing =
    List.map
      (fun e -> pad e ~target_length ~pad_id ~pad_type_id ~pad_token ~direction)
      t.overflowing
  in
  let current_len = length t in
  if current_len >= target_length then { t with overflowing }
  else
    let n = target_length - current_len in
    let pad_a arr fill = pad_array arr n fill direction in
    let sequence_ranges =
      match direction with
      | `Right -> t.sequence_ranges
      | `Left ->
          if Hashtbl.length t.sequence_ranges = 0 then empty_ranges
          else begin
            let tbl = Hashtbl.create (Hashtbl.length t.sequence_ranges) in
            Hashtbl.iter
              (fun seq_id (start, stop) ->
                Hashtbl.add tbl seq_id (start + n, stop + n))
              t.sequence_ranges;
            tbl
          end
    in
    {
      ids = pad_a t.ids pad_id;
      type_ids = pad_a t.type_ids pad_type_id;
      tokens = pad_a t.tokens pad_token;
      words = pad_a t.words None;
      offsets = pad_a t.offsets (0, 0);
      special_tokens_mask = pad_a t.special_tokens_mask 1;
      attention_mask = pad_a t.attention_mask 0;
      overflowing;
      sequence_ranges;
    }
