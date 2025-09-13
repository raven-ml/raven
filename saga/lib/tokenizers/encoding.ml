(** Encoding module - represents the output of a tokenizer *)

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
(** Internal type definition *)

(** Truncation direction *)
type truncation_direction = Left | Right

(** Padding direction *)
type padding_direction = Left | Right

(** Create a new encoding *)
let create ~ids ~type_ids ~tokens ~words ~offsets ~special_tokens_mask
    ~attention_mask ~overflowing ~sequence_ranges =
  {
    ids;
    type_ids;
    tokens;
    words;
    offsets;
    special_tokens_mask;
    attention_mask;
    overflowing;
    sequence_ranges;
  }

(** Create an empty encoding with given capacity *)
let with_capacity len =
  {
    ids = Array.make len 0;
    type_ids = Array.make len 0;
    tokens = Array.make len "";
    words = Array.make len None;
    offsets = Array.make len (0, 0);
    special_tokens_mask = Array.make len 0;
    attention_mask = Array.make len 0;
    overflowing = [];
    sequence_ranges = Hashtbl.create 1;
  }

(** Create encoding from tokens *)
let from_tokens tokens ~type_id =
  let length = List.length tokens in
  let ids = Array.make length 0 in
  let token_strs = Array.make length "" in
  let offsets = Array.make length (0, 0) in

  List.iteri
    (fun i (id, token, offset) ->
      ids.(i) <- id;
      token_strs.(i) <- token;
      offsets.(i) <- offset)
    tokens;

  {
    ids;
    tokens = token_strs;
    offsets;
    words = Array.make length None;
    type_ids = Array.make length type_id;
    attention_mask = Array.make length 1;
    special_tokens_mask = Array.make length 0;
    overflowing = [];
    sequence_ranges = Hashtbl.create 1;
  }

(** Check if encoding is empty *)
let is_empty t = Array.length t.ids = 0

(** Get the length of the encoding *)
let length t = Array.length t.ids

(** Get the number of sequences in the encoding *)
let n_sequences t =
  if Hashtbl.length t.sequence_ranges = 0 then 1
  else Hashtbl.length t.sequence_ranges

(** Set sequence id for the whole encoding *)
let set_sequence_id t sequence_id =
  let new_ranges = Hashtbl.copy t.sequence_ranges in
  Hashtbl.replace new_ranges sequence_id (0, length t);
  { t with sequence_ranges = new_ranges }

(** Getters *)
let get_ids t = t.ids

let get_type_ids t = t.type_ids
let get_tokens t = t.tokens
let get_word_ids t = t.words
let get_offsets t = t.offsets
let get_special_tokens_mask t = t.special_tokens_mask
let get_attention_mask t = t.attention_mask
let get_overflowing t = t.overflowing

(** Set type IDs *)
let set_type_ids t type_ids = { t with type_ids }

(** Set overflowing *)
let set_overflowing t overflowing = { t with overflowing }

(** Take overflowing *)
let take_overflowing t =
  let overflowing = t.overflowing in
  t.overflowing <- [];
  (t, overflowing)

(** Get sequence IDs for each token *)
let get_sequence_ids t =
  let sequences = Array.make (length t) None in
  for seq_id = 0 to n_sequences t - 1 do
    match Hashtbl.find_opt t.sequence_ranges seq_id with
    | Some (start, stop) ->
        for i = start to stop - 1 do
          if i < Array.length sequences then sequences.(i) <- Some seq_id
        done
    | None -> ()
  done;
  sequences

(** Get the range for a given sequence *)
let sequence_range t sequence_id =
  match Hashtbl.find_opt t.sequence_ranges sequence_id with
  | Some range -> range
  | None -> (0, length t)

(** Get the sequence index containing the given token *)
let token_to_sequence t token =
  if token >= length t then None
  else if Hashtbl.length t.sequence_ranges = 0 then Some 0
  else
    Hashtbl.fold
      (fun seq_id (start, stop) acc ->
        match acc with
        | Some _ -> acc
        | None -> if token >= start && token < stop then Some seq_id else None)
      t.sequence_ranges None

(** Get the tokens corresponding to the given word *)
let word_to_tokens t ~word ~sequence_id =
  let start_ref = ref None in
  let end_ref = ref None in
  let seq_start, seq_end = sequence_range t sequence_id in

  for i = seq_start to seq_end - 1 do
    match t.words.(i) with
    | Some w when w = word ->
        if !start_ref = None || Some i < !start_ref then start_ref := Some i;
        if !end_ref = None || Some i >= !end_ref then end_ref := Some (i + 1)
    | _ -> ()
  done;

  match (!start_ref, !end_ref) with
  | Some start, Some end_pos -> Some (start, end_pos)
  | _ -> None

(** Get the character offsets of the given word *)
let word_to_chars t ~word ~sequence_id =
  match word_to_tokens t ~word ~sequence_id with
  | Some (start, end_pos) when end_pos > 0 ->
      let start_offset, _ = t.offsets.(start) in
      let _, end_offset = t.offsets.(end_pos - 1) in
      Some (start_offset, end_offset)
  | _ -> None

(** Get the character offsets of the given token *)
let token_to_chars t token =
  match token_to_sequence t token with
  | Some seq_id when token < Array.length t.offsets ->
      Some (seq_id, t.offsets.(token))
  | _ -> None

(** Get the word containing the given token *)
let token_to_word t token =
  match token_to_sequence t token with
  | Some seq_id -> (
      match t.words.(token) with
      | Some word -> Some (seq_id, word)
      | None -> None)
  | None -> None

(** Get the token containing the given character position *)
let char_to_token t ~pos ~sequence_id =
  let seq_start, seq_end = sequence_range t sequence_id in
  let rec find_token i =
    if i >= seq_end then None
    else
      let start_offset, end_offset = t.offsets.(i) in
      if pos >= start_offset && pos < end_offset then Some i
      else find_token (i + 1)
  in
  find_token seq_start

(** Get the word containing the given character position *)
let char_to_word t ~pos ~sequence_id =
  match char_to_token t ~pos ~sequence_id with
  | Some token -> Option.bind t.words.(token) (fun w -> Some w)
  | None -> None

(** Helper to copy array slice *)
let array_slice arr start stop =
  Array.init (stop - start) (fun i -> arr.(start + i))

(** Truncate the encoding *)
let truncate t ~max_length ~stride ~(direction : truncation_direction) =
  let encoding_len = length t in
  if max_length >= encoding_len then t
  else if max_length = 0 then (
    (* Move everything to overflowing *)
    let empty = with_capacity 0 in
    empty.overflowing <- [ t ];
    empty)
  else (
    assert (stride < max_length);

    (* Clear sequence ranges when truncating *)
    Hashtbl.clear t.sequence_ranges;

    let offset = max_length - stride in

    (* Calculate parts ranges *)
    let parts_ranges =
      match direction with
      | Right ->
          let rec collect start acc =
            if start >= encoding_len then List.rev acc
            else
              let stop = min (start + max_length) encoding_len in
              collect (start + offset) ((start, stop) :: acc)
          in
          collect 0 []
      | Left ->
          let rec collect stop acc =
            if stop <= 0 then acc
            else
              let start = max 0 (stop - max_length) in
              collect (stop - offset) ((start, stop) :: acc)
          in
          collect encoding_len []
    in

    match parts_ranges with
    | [] -> with_capacity 0
    | (start, stop) :: rest ->
        (* Create main encoding from first part *)
        let new_encoding =
          {
            ids = array_slice t.ids start stop;
            type_ids = array_slice t.type_ids start stop;
            tokens = array_slice t.tokens start stop;
            words = array_slice t.words start stop;
            offsets = array_slice t.offsets start stop;
            special_tokens_mask = array_slice t.special_tokens_mask start stop;
            attention_mask = array_slice t.attention_mask start stop;
            overflowing = [];
            sequence_ranges = Hashtbl.create 1;
          }
        in

        (* Create overflowing encodings *)
        new_encoding.overflowing <-
          List.map
            (fun (start, stop) ->
              {
                ids = array_slice t.ids start stop;
                type_ids = array_slice t.type_ids start stop;
                tokens = array_slice t.tokens start stop;
                words = array_slice t.words start stop;
                offsets = array_slice t.offsets start stop;
                special_tokens_mask =
                  array_slice t.special_tokens_mask start stop;
                attention_mask = array_slice t.attention_mask start stop;
                overflowing = [];
                sequence_ranges = Hashtbl.create 1;
              })
            rest;

        new_encoding)

(** Merge multiple encodings *)
let rec merge encodings ~growing_offsets =
  let rec merge_list acc = function
    | [] -> acc
    | e :: rest -> merge_list (merge_with acc e ~growing_offsets) rest
  in
  match encodings with [] -> with_capacity 0 | e :: rest -> merge_list e rest

(** Merge with another encoding *)
and merge_with t1 t2 ~growing_offsets =
  let original_len = length t1 in

  (* Merge overflowing encodings *)
  let new_overflowing = ref [] in

  (* Merge t1's overflowing with t2 *)
  List.iter
    (fun o1 ->
      new_overflowing := merge_with o1 t2 ~growing_offsets :: !new_overflowing;
      List.iter
        (fun o2 ->
          new_overflowing :=
            merge_with o1 o2 ~growing_offsets :: !new_overflowing)
        t2.overflowing)
    t1.overflowing;

  (* Merge t1 with t2's overflowing *)
  List.iter
    (fun o2 ->
      new_overflowing := merge_with t1 o2 ~growing_offsets :: !new_overflowing)
    t2.overflowing;

  (* Update sequence ranges *)
  let new_ranges = Hashtbl.copy t1.sequence_ranges in
  Hashtbl.iter
    (fun seq_id (start, stop) ->
      Hashtbl.replace new_ranges seq_id
        (original_len + start, original_len + stop))
    t2.sequence_ranges;

  (* Calculate offset for growing offsets *)
  let starting_offset =
    if growing_offsets && Array.length t1.offsets > 0 then
      snd t1.offsets.(Array.length t1.offsets - 1)
    else 0
  in

  (* Merge arrays *)
  let merged_offsets =
    if growing_offsets then
      Array.map
        (fun (start, stop) -> (start + starting_offset, stop + starting_offset))
        t2.offsets
    else t2.offsets
  in

  {
    ids = Array.append t1.ids t2.ids;
    type_ids = Array.append t1.type_ids t2.type_ids;
    tokens = Array.append t1.tokens t2.tokens;
    words = Array.append t1.words t2.words;
    offsets = Array.append t1.offsets merged_offsets;
    special_tokens_mask =
      Array.append t1.special_tokens_mask t2.special_tokens_mask;
    attention_mask = Array.append t1.attention_mask t2.attention_mask;
    overflowing = List.rev !new_overflowing;
    sequence_ranges = new_ranges;
  }

(** Pad the encoding *)
let rec pad t ~target_length ~pad_id ~pad_type_id ~pad_token ~direction =
  (* First pad all overflowing encodings *)
  let padded_overflowing =
    List.map
      (fun e -> pad e ~target_length ~pad_id ~pad_type_id ~pad_token ~direction)
      t.overflowing
  in

  let current_len = length t in
  if current_len >= target_length then
    { t with overflowing = padded_overflowing }
  else
    let pad_length = target_length - current_len in

    match direction with
    | Left ->
        (* Update sequence ranges *)
        let new_ranges = Hashtbl.create (Hashtbl.length t.sequence_ranges) in
        Hashtbl.iter
          (fun seq_id (start, stop) ->
            Hashtbl.add new_ranges seq_id (start + pad_length, stop + pad_length))
          t.sequence_ranges;

        {
          ids = Array.append (Array.make pad_length pad_id) t.ids;
          type_ids = Array.append (Array.make pad_length pad_type_id) t.type_ids;
          tokens = Array.append (Array.make pad_length pad_token) t.tokens;
          words = Array.append (Array.make pad_length None) t.words;
          offsets = Array.append (Array.make pad_length (0, 0)) t.offsets;
          special_tokens_mask =
            Array.append (Array.make pad_length 1) t.special_tokens_mask;
          attention_mask =
            Array.append (Array.make pad_length 0) t.attention_mask;
          overflowing = padded_overflowing;
          sequence_ranges = new_ranges;
        }
    | Right ->
        {
          ids = Array.append t.ids (Array.make pad_length pad_id);
          type_ids = Array.append t.type_ids (Array.make pad_length pad_type_id);
          tokens = Array.append t.tokens (Array.make pad_length pad_token);
          words = Array.append t.words (Array.make pad_length None);
          offsets = Array.append t.offsets (Array.make pad_length (0, 0));
          special_tokens_mask =
            Array.append t.special_tokens_mask (Array.make pad_length 1);
          attention_mask =
            Array.append t.attention_mask (Array.make pad_length 0);
          overflowing = padded_overflowing;
          sequence_ranges = t.sequence_ranges;
        }

type encoding_data = {
  ids : int array;
  type_ids : int array;
  tokens : string array;
  offsets : (int * int) array;
  attention_mask : int array;
  special_tokens_mask : int array;
  overflowing : t list;
  word_ids : int option array;
  sequence_ids : int option array;
  n_sequences : int;
}
(** Encoding data for serialization - currently unused but may be needed for
    JSON serialization *)

let _to_data (t : t) : encoding_data =
  {
    ids = t.ids;
    type_ids = t.type_ids;
    tokens = t.tokens;
    offsets = t.offsets;
    attention_mask = t.attention_mask;
    special_tokens_mask = t.special_tokens_mask;
    overflowing = t.overflowing;
    word_ids = t.words;
    sequence_ids = get_sequence_ids t;
    n_sequences = n_sequences t;
  }

let _from_data (d : encoding_data) : t =
  let t =
    {
      ids = d.ids;
      type_ids = d.type_ids;
      tokens = d.tokens;
      words = d.word_ids;
      offsets = d.offsets;
      special_tokens_mask = d.special_tokens_mask;
      attention_mask = d.attention_mask;
      overflowing = d.overflowing;
      sequence_ranges = Hashtbl.create 1;
    }
  in

  (* Reconstruct sequence ranges from sequence_ids if needed *)
  if d.n_sequences > 1 then (
    let current_seq = ref None in
    let start = ref 0 in
    Array.iteri
      (fun i seq_opt ->
        match (seq_opt, !current_seq) with
        | Some seq, None ->
            current_seq := Some seq;
            start := i
        | Some seq, Some curr_seq when seq <> curr_seq ->
            Hashtbl.add t.sequence_ranges curr_seq (!start, i);
            current_seq := Some seq;
            start := i
        | None, Some curr_seq ->
            Hashtbl.add t.sequence_ranges curr_seq (!start, i);
            current_seq := None
        | _ -> ())
      d.sequence_ids;

    (* Handle the last sequence *)
    match !current_seq with
    | Some seq ->
        Hashtbl.add t.sequence_ranges seq (!start, Array.length d.sequence_ids)
    | None -> ());

  t
