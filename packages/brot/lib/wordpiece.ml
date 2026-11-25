(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compact trie for zero-allocation longest-prefix matching *)

type trie = {
  trie_ids : int array;
  child_starts : int array;
  edge_bytes : bytes;
  edge_targets : int array;
  (* Flat 256-element arrays for dense nodes (>16 children) — O(1) lookup *)
  flat_nodes : int array array;
}

let build_trie vocab =
  if Hashtbl.length vocab = 0 then
    {
      trie_ids = [||];
      child_starts = [| 0 |];
      edge_bytes = Bytes.empty;
      edge_targets = [||];
      flat_nodes = [||];
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
      vocab;
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
    (* Sort each node's children by byte value for binary search *)
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
    (* Build flat 256-element arrays for dense nodes (>16 children) *)
    let flat_nodes = Array.make node_count [||] in
    for i = 0 to node_count - 1 do
      let start = child_starts.(i) in
      let count = child_starts.(i + 1) - start in
      if count > 16 then begin
        let flat = Array.make 256 (-1) in
        for j = start to start + count - 1 do
          let b = Char.code (Bytes.unsafe_get edge_bytes j) in
          flat.(b) <- Array.unsafe_get edge_targets j
        done;
        flat_nodes.(i) <- flat
      end
    done;
    { trie_ids; child_starts; edge_bytes; edge_targets; flat_nodes }

let[@inline] trie_step trie node byte =
  let flat = Array.unsafe_get trie.flat_nodes node in
  if Array.length flat > 0 then Array.unsafe_get flat byte
  else
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

(* Model type *)

type t = {
  encoders : Pretoken_cache.encoder Pretoken_cache.states;
  cache_capacity : int;
  vocab : (string, int) Hashtbl.t;
  tokens : string Id_table.t;
  (* How many bytes of a word an id accounts for: the entry, stripped of the
     prefix a continuation subword carries. The unknown token spent on a whole
     word is recorded as it is emitted. *)
  lens : int Id_table.t;
  trie : trie;
  unk_token : string;
  unk_id : int;
  continuing_subword_prefix : string;
  max_input_chars_per_word : int;
}

let create ~vocab ?(unk_token = "[UNK]") ?(continuing_subword_prefix = "##")
    ?(max_input_chars_per_word = 100)
    ?(cache_capacity = Pretoken_cache.default_capacity) () =
  let tokens = Id_table.of_vocab ~who:"Wordpiece.create" vocab in
  let prefix_len = String.length continuing_subword_prefix in
  let lens =
    Id_table.mapi ~absent:0
      (fun _ token ->
        let n = String.length token in
        if
          prefix_len > 0
          && String.starts_with ~prefix:continuing_subword_prefix token
        then n - prefix_len
        else n)
      tokens
  in
  let unk_id =
    match Hashtbl.find_opt vocab unk_token with Some id -> id | None -> -1
  in
  let trie = build_trie vocab in
  {
    encoders = Pretoken_cache.states ();
    cache_capacity;
    vocab;
    tokens;
    lens;
    trie;
    unk_token;
    unk_id;
    continuing_subword_prefix;
    max_input_chars_per_word;
  }

(* [line] without the White_Space characters that end it. *)
let trim_end line =
  let stop = String.length line in
  let rec kept i keep =
    if i >= stop then keep
    else
      let d = Char_class.at line i ~stop in
      let next = i + Char_class.at_len d in
      if Char_class.at_category d = Char_class.whitespace then kept next keep
      else kept next next
  in
  let keep = kept 0 0 in
  if keep = stop then line else String.sub line 0 keep

let read_file ~vocab_file =
  let vocab = Hashtbl.create 10000 in
  let ic = open_in_bin vocab_file in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
      let index = ref 0 in
      (try
         while true do
           (* A blank line takes an id too: an id is the line's number. *)
           Hashtbl.replace vocab (trim_end (input_line ic)) !index;
           incr index
         done
       with End_of_file -> ());
      vocab)

let from_file ?unk_token ~vocab_file () =
  let vocab = read_file ~vocab_file in
  create ~vocab ?unk_token ()

let missing_unk model =
  failwith
    (Printf.sprintf "WordPiece: the unknown token %S is not in the vocabulary"
       model.unk_token)

(* The unknown token, standing for the whole word. *)
let add_unknown model ids ~opaque ~len =
  if model.unk_id < 0 then missing_unk model;
  Ints.add opaque (Ints.length ids);
  Ints.add opaque 1;
  Ints.add opaque len;
  Ints.add ids model.unk_id

(* The ids of [text.\[pos..pos+len)], appended to [ids]. A word no run of
   subwords covers is one unknown token, so the ids already written for it are
   dropped. The trie cursors are held outside the walk: a [ref] taken per
   position would allocate once per subword. *)
let encode_into model ids ~opaque text ~pos ~len =
  (* An empty vocabulary holds no piece, nor the unknown token. *)
  if len > 0 && Hashtbl.length model.vocab = 0 then missing_unk model
  else if len > 0 then begin
    let stop = pos + len in
    let chars = ref 0 and k = ref pos in
    while !k < stop do
      k := !k + Char_class.length text !k ~stop;
      incr chars
    done;
    if !chars > model.max_input_chars_per_word then
      add_unknown model ids ~opaque ~len
    else begin
      let trie = model.trie in
      let prefix = model.continuing_subword_prefix in
      let prefix_len = String.length prefix in
      let written = Ints.length ids in
      let p = ref pos and unknown = ref false in
      let node = ref 0 and stopped = ref false in
      let last_id = ref 0 and last_end = ref 0 and i = ref 0 in
      while !p < stop && not !unknown do
        node := 0;
        stopped := false;
        last_id := -1;
        last_end := !p;
        if !p > pos then begin
          i := 0;
          while !i < prefix_len && not !stopped do
            let child =
              trie_step trie !node (Char.code (String.unsafe_get prefix !i))
            in
            if child < 0 then stopped := true
            else begin
              node := child;
              incr i
            end
          done
        end;
        if not !stopped then begin
          i := !p;
          while !i < stop && not !stopped do
            let child =
              trie_step trie !node (Char.code (String.unsafe_get text !i))
            in
            if child < 0 then stopped := true
            else begin
              node := child;
              incr i;
              let id = Array.unsafe_get trie.trie_ids child in
              if id >= 0 then begin
                last_id := id;
                last_end := !i
              end
            end
          done
        end;
        if !last_id >= 0 then begin
          Ints.add ids !last_id;
          p := !last_end
        end
        else unknown := true
      done;
      if !unknown then begin
        Ints.truncate ids written;
        add_unknown model ids ~opaque ~len
      end
    end
  end

let cached_encoder model =
  Pretoken_cache.cached
    (Pretoken_cache.create model.cache_capacity)
    ~len_table:(Id_table.dense model.lens) (encode_into model)

let with_encoder model f =
  Pretoken_cache.with_state model.encoders ~make:cached_encoder
    ~spare:encode_into model f

let cache_capacity model = model.cache_capacity

let token_table model = model.tokens
let len_table model = model.lens
let token_to_id model token = Hashtbl.find_opt model.vocab token

let id_to_token model id =
  let token = Id_table.get model.tokens id in
  (* An id that no pair holds is [""] in the table. *)
  if token = "" && Hashtbl.find_opt model.vocab "" <> Some id then None
  else Some token

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
  let b = Buffer.create 4096 in
  List.iter
    (fun token ->
      Buffer.add_string b token;
      Buffer.add_char b '\n')
    vocab_list;
  File_out.write vocab_file (Buffer.contents b);
  vocab_file

let from_bpe ~unk_token bpe =
  let vocab = Hashtbl.create (Bpe.get_vocab_size bpe) in
  List.iter (fun (k, id) -> Hashtbl.add vocab k id) (Bpe.get_vocab bpe);
  let continuing_subword_prefix =
    match Bpe.get_continuing_subword_prefix bpe with
    | Some p -> p
    | None -> "##"
  in
  create ~vocab ?unk_token ~continuing_subword_prefix ()

(* Trainer *)

let train ~min_frequency ~vocab_size ~show_progress ~special_tokens
    ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix ~unk_token
    word_counts =
  from_bpe ~unk_token
    (Bpe.train ~min_frequency ~vocab_size ~show_progress ~special_tokens
       ~limit_alphabet ~initial_alphabet
       ~continuing_subword_prefix:(Some continuing_subword_prefix)
       ~end_of_word_suffix:None ~max_token_length:None ~unk_token:None
       word_counts)
