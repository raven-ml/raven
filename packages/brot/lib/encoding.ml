(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An array a caller may never ask for: what it takes to work it out until it is
   asked for, and then the array. Whatever the three hold on to — the run, the
   encodings a post-processor put together — is collected as soon as they have
   all been asked for. *)
type 'a derived = Kept of 'a | Derive of (unit -> 'a)

type t = {
  ids : int array;
  type_ids : int array;
  special_tokens_mask : int array;
  attention_mask : int array;
  mutable tokens : string array derived;
  mutable offsets : (int * int) array derived;
  mutable words : int option array derived;
  mutable overflowing : t list;
  sequence_ranges : (int, int * int) Hashtbl.t;
}

(* Constructors *)

let empty_ranges : (int, int * int) Hashtbl.t = Hashtbl.create 0

let empty =
  {
    ids = [||];
    type_ids = [||];
    tokens = Kept [||];
    words = Kept [||];
    offsets = Kept [||];
    special_tokens_mask = [||];
    attention_mask = [||];
    overflowing = [];
    sequence_ranges = empty_ranges;
  }

let create ~ids ~type_ids ~tokens ~words ~offsets ~special_tokens_mask
    ~attention_mask ?(overflowing = []) () =
  let n = Array.length ids in
  if
    Array.length type_ids <> n
    || Array.length tokens <> n
    || Array.length words <> n
    || Array.length offsets <> n
    || Array.length special_tokens_mask <> n
    || Array.length attention_mask <> n
  then invalid_arg "Encoding.create: arrays must all have the same length";
  {
    ids;
    type_ids;
    tokens = Kept tokens;
    words = Kept words;
    offsets = Kept offsets;
    special_tokens_mask;
    attention_mask;
    overflowing;
    sequence_ranges = empty_ranges;
  }

let token ~id ~token ~offset ~type_id ~special =
  {
    ids = [| id |];
    type_ids = [| type_id |];
    tokens = Kept [| token |];
    words = Kept [| None |];
    offsets = Kept [| offset |];
    special_tokens_mask = [| (if special then 1 else 0) |];
    attention_mask = [| 1 |];
    overflowing = [];
    sequence_ranges = empty_ranges;
  }

let of_run run ~ids =
  let n = Array.length ids in
  {
    ids;
    type_ids = Array.make n 0;
    tokens = Derive (fun () -> Run.tokens run ~ids);
    words = Derive (fun () -> Run.words run ~ids);
    offsets = Derive (fun () -> Run.offsets run ~ids);
    special_tokens_mask = Array.make n 0;
    attention_mask = Array.make n 1;
    overflowing = [];
    sequence_ranges = empty_ranges;
  }

(* Accessors *)

let is_empty t = Array.length t.ids = 0
let length t = Array.length t.ids
let ids t = t.ids
let type_ids t = t.type_ids
let special_tokens_mask t = t.special_tokens_mask
let attention_mask t = t.attention_mask
let overflowing t = t.overflowing

let tokens t =
  match t.tokens with
  | Kept tokens -> tokens
  | Derive derive ->
      let tokens = derive () in
      t.tokens <- Kept tokens;
      tokens

let offsets t =
  match t.offsets with
  | Kept offsets -> offsets
  | Derive derive ->
      let offsets = derive () in
      t.offsets <- Kept offsets;
      offsets

let word_ids t =
  match t.words with
  | Kept words -> words
  | Derive derive ->
      let words = derive () in
      t.words <- Kept words;
      words

(* Concatenation *)

(* The derived arrays of a concatenation are the derived arrays of its parts,
   which is why they are still worth deferring: a post-processor puts a document
   between special tokens without forcing it. *)
let gather total encodings read fill =
  let dst = Array.make total fill in
  let pos = ref 0 in
  List.iter
    (fun t ->
      let src = read t in
      let n = Array.length src in
      Array.blit src 0 dst !pos n;
      pos := !pos + n)
    encodings;
  dst

let concat encodings =
  match encodings with
  | [] -> empty
  | [ single ] -> single
  | first :: _ ->
      let total =
        List.fold_left (fun acc t -> acc + Array.length t.ids) 0 encodings
      in
      let ids = Array.make total 0 in
      let type_ids = Array.make total 0 in
      let special_tokens_mask = Array.make total 0 in
      let attention_mask = Array.make total 0 in
      let pos = ref 0 in
      List.iter
        (fun t ->
          let n = Array.length t.ids in
          Array.blit t.ids 0 ids !pos n;
          Array.blit t.type_ids 0 type_ids !pos n;
          Array.blit t.special_tokens_mask 0 special_tokens_mask !pos n;
          Array.blit t.attention_mask 0 attention_mask !pos n;
          pos := !pos + n)
        encodings;
      {
        ids;
        type_ids;
        tokens = Derive (fun () -> gather total encodings tokens "");
        words = Derive (fun () -> gather total encodings word_ids None);
        offsets = Derive (fun () -> gather total encodings offsets (0, 0));
        special_tokens_mask;
        attention_mask;
        overflowing = first.overflowing;
        sequence_ranges = first.sequence_ranges;
      }

let with_type_id t type_id =
  { t with type_ids = Array.make (Array.length t.ids) type_id }

let with_overflowing t overflowing = { t with overflowing }

(* Truncation *)

let slice t start len =
  {
    ids = Array.sub t.ids start len;
    type_ids = Array.sub t.type_ids start len;
    tokens = Kept (Array.sub (tokens t) start len);
    words = Kept (Array.sub (word_ids t) start len);
    offsets = Kept (Array.sub (offsets t) start len);
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
          (* The window kept is the rightmost one, and the walk stops with the
             window that reaches the start of the encoding. *)
          let rec loop stop =
            if stop <= 0 then []
            else
              let start = max 0 (stop - max_length) in
              if start = 0 then [ (start, stop) ]
              else (start, stop) :: loop (stop - step)
          in
          loop encoding_len
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
      tokens = Kept (pad_a (tokens t) pad_token);
      words = Kept (pad_a (word_ids t) None);
      offsets = Kept (pad_a (offsets t) (0, 0));
      special_tokens_mask = pad_a t.special_tokens_mask 1;
      attention_mask = pad_a t.attention_mask 0;
      overflowing;
      sequence_ranges;
    }

(* Formatting *)

let pp ppf t =
  let tokens = tokens t and offsets = offsets t and words = word_ids t in
  let token_width =
    Array.fold_left
      (fun acc token -> max acc (String.length token))
      (String.length "Token") tokens
  in
  Format.fprintf ppf "@[<v>%-5s %-*s %6s %-12s %5s %5s %5s %8s" "Index"
    token_width "Token" "ID" "Offsets" "Word" "Type" "Attn" "Special";
  for i = 0 to length t - 1 do
    let start, stop = offsets.(i) in
    let word = match words.(i) with Some w -> string_of_int w | None -> "-" in
    Format.fprintf ppf "@,%-5d %-*s %6d (%4d,%4d) %5s %5d %5d %8d" i token_width
      tokens.(i) t.ids.(i) start stop word t.type_ids.(i) t.attention_mask.(i)
      t.special_tokens_mask.(i)
  done;
  Format.fprintf ppf "@]"
