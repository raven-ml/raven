(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Double-array trie over the vocabulary. A node is an index; the child of
   [node] along [byte] is [base.(node) + byte] when [check] there names [node],
   so a step is two loads and a compare, whatever the fanout. [ids.(node)] is
   the entry the path to [node] spells, or [-1]. *)

type trie = { base : int array; check : int array; ids : int array }

let[@inline] trie_step trie node byte =
  let next = Array.unsafe_get trie.base node + byte in
  if Array.unsafe_get trie.check next = node then next else -1

(* Nodes are placed depth first over the entries in byte order, where a node is
   a run of entries sharing a prefix and its children are the runs one byte
   longer. Each node's base is the first that puts every child on a free slot,
   found by walking the list of free slots, which packs the array densely:
   nearly every node has one child. Every base leaves room for a full alphabet
   past it, so a step never needs a bounds check; a leaf's base is [0], and slot
   [0] belongs to no one, so a leaf never steps anywhere. An entry appearing
   twice keeps the later identifier. *)
let build_trie tokens =
  let n = Array.length tokens in
  let order = Array.init n Fun.id in
  Array.stable_sort (fun a b -> String.compare tokens.(a) tokens.(b)) order;
  let capacity = ref (2 * (n + 257)) in
  let base = ref (Array.make !capacity 0) in
  let check = ref (Array.make !capacity (-1)) in
  let ids = ref (Array.make !capacity (-1)) in
  (* Free slots, doubly linked in increasing order from [head]; [prev] is [0] at
     the head and [next] is [capacity] at the tail. The last slot is always
     free, so growing appends to the list. *)
  let next = ref (Array.init !capacity (fun i -> i + 1)) in
  let prev = ref (Array.init !capacity (fun i -> i - 1)) in
  let head = ref 1 in
  let ensure limit =
    if limit > !capacity then begin
      let capacity' = max limit (2 * !capacity) in
      let extend a fill =
        let a' = Array.make capacity' fill in
        Array.blit a 0 a' 0 !capacity;
        a'
      in
      base := extend !base 0;
      check := extend !check (-1);
      ids := extend !ids (-1);
      next := extend !next 0;
      prev := extend !prev 0;
      for i = !capacity to capacity' - 1 do
        !next.(i) <- i + 1;
        !prev.(i) <- i - 1
      done;
      capacity := capacity'
    end
  in
  let take slot =
    let p = !prev.(slot) and nx = !next.(slot) in
    if p = 0 then head := nx else !next.(p) <- nx;
    if nx < !capacity then !prev.(nx) <- p
  in
  let extent = ref 257 in
  (* The children of a node: one per distinct byte at [depth], as the bytes and
     the run of entries each covers. *)
  let bytes = Array.make 256 0
  and lo = Array.make 256 0
  and hi = Array.make 256 0 in
  let stack = Stack.create () in
  Stack.push (0, n, 0, 0) stack;
  while not (Stack.is_empty stack) do
    let from, until, depth, node = Stack.pop stack in
    let i = ref from in
    while !i < until && String.length tokens.(order.(!i)) = depth do
      !ids.(node) <- order.(!i);
      incr i
    done;
    let count = ref 0 in
    while !i < until do
      let byte = Char.code (String.unsafe_get tokens.(order.(!i)) depth) in
      let start = !i in
      incr i;
      while
        !i < until
        && Char.code (String.unsafe_get tokens.(order.(!i)) depth) = byte
      do
        incr i
      done;
      bytes.(!count) <- byte;
      lo.(!count) <- start;
      hi.(!count) <- !i;
      incr count
    done;
    if !count > 0 then begin
      let first = bytes.(0) in
      let fits b =
        let ok = ref true and k = ref 1 in
        while !ok && !k < !count do
          if !check.(b + bytes.(!k)) >= 0 then ok := false;
          incr k
        done;
        !ok
      in
      let rec place slot =
        let b = slot - first in
        if b < 1 then place !next.(slot)
        else begin
          ensure (b + 257);
          if fits b then b else place !next.(slot)
        end
      in
      let b = place !head in
      !base.(node) <- b;
      extent := max !extent (b + 257);
      for k = 0 to !count - 1 do
        let child = b + bytes.(k) in
        take child;
        !check.(child) <- node;
        Stack.push (lo.(k), hi.(k), depth + 1, child) stack
      done
    end
  done;
  {
    base = Array.sub !base 0 !extent;
    check = Array.sub !check 0 !extent;
    ids = Array.sub !ids 0 !extent;
  }

(* Model type *)

(* How far below the rarest entry of the vocabulary a character it does not hold
   scores. SentencePiece's constant, and the one HuggingFace keeps. *)
let unk_penalty = 10.0

type t = {
  token_table : string array;
  scores : float array;
  token_to_ids : (string, int) Hashtbl.t;
  len_table : int array;
  trie : trie;
  min_score : float;
  unk_id : int; (* [-1] when the model has none. *)
  byte_fallback : bool;
  byte_ids : int array; (* A byte to the id of its ["<0xXX>"] entry, or [-1]. *)
}

let create ?unk_id ?(byte_fallback = false) vocab_list =
  let vocab = Array.of_list vocab_list in
  let size = Array.length vocab in
  let unk_id =
    match unk_id with
    | None -> -1
    | Some id ->
        if id < 0 || id >= size then
          invalid_arg
            (Printf.sprintf
               "Unigram.create: unk_id %d is outside a vocabulary of %d entries"
               id size);
        id
  in
  let token_table = Array.map fst vocab in
  let scores = Array.map snd vocab in
  let token_to_ids = Hashtbl.create (max 1 size) in
  Array.iteri
    (fun id token -> Hashtbl.replace token_to_ids token id)
    token_table;
  let byte_ids = Array.make 256 (-1) in
  for b = 0 to 255 do
    match Hashtbl.find_opt token_to_ids (Printf.sprintf "<0x%02X>" b) with
    | Some id -> byte_ids.(b) <- id
    | None -> ()
  done;
  (* An identifier accounts for the bytes of its entry, except the byte a
     fallback token spells out and the unknown token, which stands for a stretch
     of text no entry covers and so for no fixed number of bytes at all. *)
  let len_table = Array.map String.length token_table in
  if byte_fallback then
    Array.iter (fun id -> if id >= 0 then len_table.(id) <- 1) byte_ids;
  if unk_id >= 0 then len_table.(unk_id) <- 0;
  {
    token_table;
    scores;
    token_to_ids;
    len_table;
    trie = build_trie token_table;
    min_score = Array.fold_left min infinity scores;
    unk_id;
    byte_fallback;
    byte_ids;
  }

let token_to_id model token = Hashtbl.find_opt model.token_to_ids token

let id_to_token model id =
  if id >= 0 && id < Array.length model.token_table then
    Some (Array.unsafe_get model.token_table id)
  else None

let get_vocab model =
  List.init (Array.length model.token_table) (fun id ->
      (model.token_table.(id), model.scores.(id)))

let get_vocab_size model = Array.length model.token_table
let get_unk_id model = if model.unk_id < 0 then None else Some model.unk_id
let get_byte_fallback model = model.byte_fallback
let token_table model = model.token_table
let len_table model = model.len_table

(* Tokenization *)

type state = {
  mutable score : float array;
  mutable back : int array;
  mutable piece : int array;
  mutable path : int array;
  busy : bool Atomic.t;
}

(* Pretokens run to a handful of bytes, and a state that has met a long one
   keeps the room for the next. *)
let fresh_state () =
  {
    score = Array.make 256 0.0;
    back = Array.make 256 (-1);
    piece = Array.make 256 0;
    path = Array.make 256 0;
    busy = Atomic.make false;
  }

let state_key = Domain.DLS.new_key fresh_state

(* The domain's state, held for the duration of [f]. Its arrays have a single
   writer, so a second thread of the domain asking while it is held gets a state
   of its own. The claim is given back even when [f] raises. *)
let with_state f =
  let st = Domain.DLS.get state_key in
  let st =
    if Atomic.compare_and_set st.busy false true then st else fresh_state ()
  in
  match f st with
  | v ->
      Atomic.set st.busy false;
      v
  | exception e ->
      let backtrace = Printexc.get_raw_backtrace () in
      Atomic.set st.busy false;
      Printexc.raise_with_backtrace e backtrace

let grow st needed =
  if Array.length st.score < needed then begin
    let capacity = max needed (2 * Array.length st.score) in
    st.score <- Array.make capacity 0.0;
    st.back <- Array.make capacity (-1);
    st.piece <- Array.make capacity 0;
    st.path <- Array.make capacity 0
  end

let utf8_len_table =
  Array.init 256 (fun b ->
      if b land 0x80 = 0 then 1
      else if b land 0xE0 = 0xC0 then 2
      else if b land 0xF0 = 0xE0 then 3
      else if b land 0xF8 = 0xF0 then 4
      else 1)

let err_missing_unk =
  "Unigram.encode_into: the text holds a character the vocabulary does not \
   have, and the model has no unk_id to stand in for it"

let all_bytes_known model text ~first ~last =
  let known = ref true in
  for i = first to last - 1 do
    if
      Array.unsafe_get model.byte_ids (Char.code (String.unsafe_get text i)) < 0
    then known := false
  done;
  !known

(* A stretch of text no entry covers: the entry it spells as a whole, should the
   vocabulary hold one, else its bytes one token each when byte fallback is on
   and the vocabulary holds every one of them, else the unknown token. Only a
   run of several pieces can spell an entry: a single one would have been
   matched. *)
let add_unknown model ids text ~first ~last ~fused =
  let entry =
    if fused then
      Hashtbl.find_opt model.token_to_ids (String.sub text first (last - first))
    else None
  in
  match entry with
  | Some id -> Ints.add ids id
  | None ->
      if model.byte_fallback && all_bytes_known model text ~first ~last then
        for i = first to last - 1 do
          Ints.add ids
            (Array.unsafe_get model.byte_ids
               (Char.code (String.unsafe_get text i)))
        done
      else Ints.add ids model.unk_id

(* The best path read back from the end of the pretoken and walked forward
   again. A run of characters the vocabulary does not hold is one unknown token
   rather than one each — or the entry the run spells, should the vocabulary
   hold that — and byte fallback spells out the whole run. *)
let emit model st ids text ~pos ~len =
  let back = st.back and piece = st.piece and path = st.path in
  let count = ref 0 in
  let node = ref len in
  while !node > 0 do
    Array.unsafe_set path !count !node;
    incr count;
    node := Array.unsafe_get back !node
  done;
  let unk_id = model.unk_id in
  let k = ref (!count - 1) in
  while !k >= 0 do
    let id = Array.unsafe_get piece (Array.unsafe_get path !k) in
    let last =
      if unk_id >= 0 && (id < 0 || id = unk_id) then begin
        let last = ref !k in
        while
          !last > 0
          &&
          let next =
            Array.unsafe_get piece (Array.unsafe_get path (!last - 1))
          in
          next < 0 || next = unk_id
        do
          decr last
        done;
        !last
      end
      else !k
    in
    (* A single piece the trie matched is a token of the vocabulary whatever its
       identifier; only a run, or a character no entry covers, is unknown. *)
    if last = !k && id >= 0 then Ints.add ids id
    else begin
      let first =
        if !k = !count - 1 then 0 else Array.unsafe_get path (!k + 1)
      in
      add_unknown model ids text ~first:(pos + first)
        ~last:(pos + Array.unsafe_get path last)
        ~fused:(last < !k)
    end;
    k := last - 1
  done

let encode_into model st ids text ~pos ~len =
  if len > 0 then begin
    grow st (len + 1);
    let score = st.score and back = st.back and piece = st.piece in
    let trie = model.trie in
    let trie_ids = trie.ids in
    let scores = model.scores in
    let unk_id = model.unk_id in
    let unk_score = model.min_score -. unk_penalty in
    Array.fill back 0 (len + 1) (-1);
    Array.unsafe_set score 0 0.0;
    let at = ref 0 in
    while !at < len do
      let here = Array.unsafe_get score !at in
      let width =
        let w =
          Array.unsafe_get utf8_len_table
            (Char.code (String.unsafe_get text (pos + !at)))
        in
        if !at + w <= len then w else 1
      in
      (* Every entry of the vocabulary that starts here, scored against the best
         path into this position. *)
      let covered = ref false in
      let node = ref 0 and stop = ref !at and walking = ref true in
      while !walking && !stop < len do
        let child =
          trie_step trie !node
            (Char.code (String.unsafe_get text (pos + !stop)))
        in
        if child < 0 then walking := false
        else begin
          node := child;
          incr stop;
          let id = Array.unsafe_get trie_ids child in
          if id >= 0 then begin
            let candidate = here +. Array.unsafe_get scores id in
            if
              Array.unsafe_get back !stop < 0
              || candidate > Array.unsafe_get score !stop
            then begin
              Array.unsafe_set score !stop candidate;
              Array.unsafe_set back !stop !at;
              Array.unsafe_set piece !stop id
            end;
            if !stop - !at = width then covered := true
          end
        end
      done;
      (* No entry is this character on its own, so the path may have to spend an
         unknown token on it; the model must have one to spend only when that is
         the best way into the character. *)
      if not !covered then begin
        let stop = !at + width in
        let candidate = here +. unk_score in
        if
          Array.unsafe_get back stop < 0
          || candidate > Array.unsafe_get score stop
        then begin
          if unk_id < 0 then failwith err_missing_unk;
          Array.unsafe_set score stop candidate;
          Array.unsafe_set back stop !at;
          Array.unsafe_set piece stop (-1)
        end
      end;
      at := !at + width
    done;
    emit model st ids text ~pos ~len
  end

(* Serialization *)

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let json_to_string j =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json j with
  | Ok s -> s
  | Error e -> failwith e

let save model ~folder () =
  let json_vocab =
    get_vocab model
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
        ( "unk_id",
          match get_unk_id model with
          | None -> Jsont.Json.null ()
          | Some id -> Jsont.Json.int id );
        ("byte_fallback", Jsont.Json.bool model.byte_fallback);
        ("vocab", Jsont.Json.list json_vocab);
      ]
  in
  let path = Filename.concat folder "unigram.json" in
  let oc = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc (json_to_string json));
  [ "unigram.json" ]

(* Training *)

let train ~vocab_size ~show_progress ~special_tokens ~shrinking_factor
    ~unk_token ~max_piece_length ~n_sub_iterations word_counts =
  let _ =
    (show_progress, shrinking_factor, max_piece_length, n_sub_iterations)
  in
  let total =
    List.fold_left (fun acc (_, count) -> acc + count) 0 word_counts
    |> float_of_int
  in
  let sorted = List.sort (fun (_, c1) (_, c2) -> compare c2 c1) word_counts in

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
  let unk_id =
    match unk_token with
    | None -> None
    | Some unk ->
        List.find_index (fun (token, _) -> token = unk) vocab_with_probs
  in
  let model = create ?unk_id vocab_with_probs in
  (model, special_tokens)
