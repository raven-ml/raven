(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let list_drop n l =
  let rec aux i = function
    | _ :: l when i < n -> aux (i + 1) l
    | rest -> rest
  in
  if n <= 0 then l else aux 0 l

module IntPair = struct
  type t = int * int

  let compare = compare
end

module IntPairMap = Map.Make (IntPair)
module IntPairSet = Set.Make (IntPair)

module IntSet = Set.Make (struct
  type t = int

  let compare = compare
end)

module StringMap = Map.Make (String)

type vocab = (string, int) Hashtbl.t
type vocab_r = (int, string) Hashtbl.t
type merges = (string * string) list
type merge_map = (int * int) IntPairMap.t

type symbol = {
  mutable c : int;
  mutable prev : int;
  mutable next : int;
  mutable len : int;
}

type word = { mutable symbols : symbol array; mutable size : int }
type token = { id : int; value : string; offsets : int * int }
type cache_entry = word

type config = {
  vocab : vocab;
  merges : merges;
  cache_capacity : int;
  dropout : float option;
  unk_token : string option;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
  fuse_unk : bool;
  byte_fallback : bool;
  ignore_merges : bool;
}

type t = {
  vocab : vocab;
  vocab_r : vocab_r;
  merges : merge_map;
  cache : (string, cache_entry) Hashtbl.t option;
  dropout : float option;
  unk_token : string option;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
  fuse_unk : bool;
  byte_fallback : bool;
  ignore_merges : bool;
}

let create_word capacity =
  {
    symbols = Array.make capacity { c = -1; prev = -1; next = -1; len = 0 };
    size = 0;
  }

let add_symbol word c byte_len =
  if word.size >= Array.length word.symbols then
    failwith "Word capacity exceeded";
  let prev = if word.size > 0 then word.size - 1 else -1 in
  let symbol = { c; prev; next = -1; len = byte_len } in
  if prev >= 0 then word.symbols.(prev).next <- word.size;
  word.symbols.(word.size) <- symbol;
  word.size <- word.size + 1

module PQueue = struct
  (* Use a binary heap for O(log n) push/pop instead of O(n log n) sort *)
  type 'a t = {
    mutable arr : 'a array;
    mutable size : int;
    cmp : 'a -> 'a -> int;
  }

  let create cmp = { arr = [||]; size = 0; cmp }
  let parent i = (i - 1) / 2
  let left i = (2 * i) + 1
  let right i = (2 * i) + 2

  let swap t i j =
    let temp = t.arr.(i) in
    t.arr.(i) <- t.arr.(j);
    t.arr.(j) <- temp

  let rec sift_up t i =
    if i > 0 then
      let p = parent i in
      if t.cmp t.arr.(i) t.arr.(p) < 0 then (
        swap t i p;
        sift_up t p)

  let rec sift_down t i =
    let l = left i in
    let r = right i in
    let smallest = ref i in
    if l < t.size && t.cmp t.arr.(l) t.arr.(!smallest) < 0 then smallest := l;
    if r < t.size && t.cmp t.arr.(r) t.arr.(!smallest) < 0 then smallest := r;
    if !smallest <> i then (
      swap t i !smallest;
      sift_down t !smallest)

  let push t x =
    if t.size = Array.length t.arr then
      t.arr <- Array.append t.arr (Array.make (max 16 t.size) x);
    t.arr.(t.size) <- x;
    sift_up t t.size;
    t.size <- t.size + 1

  let pop t =
    if t.size = 0 then None
    else
      let result = t.arr.(0) in
      t.size <- t.size - 1;
      if t.size > 0 then (
        t.arr.(0) <- t.arr.(t.size);
        sift_down t 0);
      Some result
end

let apply_merges model dropout word =
  let p = match dropout with Some p -> p | None -> 0.0 in
  let use_dropout = p > 0.0 in
  let cmp (r1, p1, _) (r2, p2, _) =
    let c = r1 - r2 in
    if c = 0 then p1 - p2 else c
  in
  let queue = PQueue.create cmp in
  for i = 0 to word.size - 2 do
    if word.symbols.(i).len > 0 && word.symbols.(i + 1).len > 0 then
      let pair = (word.symbols.(i).c, word.symbols.(i + 1).c) in
      match IntPairMap.find_opt pair model.merges with
      | Some (rank, new_id) -> PQueue.push queue (rank, i, new_id)
      | None -> ()
  done;
  let skips = ref [] in
  let rec process_queue () =
    match PQueue.pop queue with
    | None -> ()
    | Some top -> (
        let rank, pos, new_id = top in
        if word.symbols.(pos).len = 0 then process_queue ()
        else
          let next_pos = word.symbols.(pos).next in
          if next_pos = -1 then process_queue ()
          else
            let next_pos = next_pos in
            let cur_pair = (word.symbols.(pos).c, word.symbols.(next_pos).c) in
            match IntPairMap.find_opt cur_pair model.merges with
            | Some (r, nid) when r = rank && nid = new_id ->
                if use_dropout && Random.float 1.0 < p then
                  skips := top :: !skips
                else (
                  List.iter (PQueue.push queue) !skips;
                  skips := [];
                  word.symbols.(pos).c <- new_id;
                  word.symbols.(pos).len <-
                    word.symbols.(pos).len + word.symbols.(next_pos).len;
                  word.symbols.(pos).next <- word.symbols.(next_pos).next;
                  word.symbols.(next_pos).len <- 0;
                  if word.symbols.(pos).next >= 0 then
                    word.symbols.(word.symbols.(pos).next).prev <- pos;
                  (if word.symbols.(pos).prev >= 0 then
                     let prev = word.symbols.(pos).prev in
                     let pair = (word.symbols.(prev).c, word.symbols.(pos).c) in
                     match IntPairMap.find_opt pair model.merges with
                     | Some (r, nid) -> PQueue.push queue (r, prev, nid)
                     | None -> ());
                  let next = word.symbols.(pos).next in
                  if next >= 0 then
                    let pair = (word.symbols.(pos).c, word.symbols.(next).c) in
                    match IntPairMap.find_opt pair model.merges with
                    | Some (r, nid) -> PQueue.push queue (r, pos, nid)
                    | None -> ());
                process_queue () (* Continue processing the queue *)
            | _ -> process_queue ())
  in
  process_queue ();
  let new_symbols = Array.make word.size word.symbols.(0) in
  let j = ref 0 in
  for k = 0 to word.size - 1 do
    if word.symbols.(k).len > 0 then (
      new_symbols.(!j) <- word.symbols.(k);
      incr j)
  done;
  word.symbols <- Array.sub new_symbols 0 !j;
  word.size <- !j

let merge_word model text =
  let len = String.length text in
  let word = create_word len in
  let decoder = Uutf.decoder (`String text) in
  let i = ref 0 in
  let pending_unk = ref None in
  (* Reuse a single buffer for all UTF-8 encoding *)
  let char_buf = Buffer.create 4 in
  let flush_unk () =
    match !pending_unk with
    | Some (unk_id, unk_len) ->
        add_symbol word unk_id unk_len;
        pending_unk := None
    | None -> ()
  in
  let rec process_chars () =
    match Uutf.decode decoder with
    | `Uchar u ->
        let start = !i in
        (* Reuse buffer - just reset it instead of creating new *)
        Buffer.clear char_buf;
        Uutf.Buffer.add_utf_8 char_buf u;
        let char_str = Buffer.contents char_buf in
        let byte_len = String.length char_str in
        i := !i + byte_len;
        let is_first = start = 0 in
        let is_last = !i >= len in
        (* Build token_str with minimal allocations *)
        let token_str =
          match
            ( is_first,
              is_last,
              model.continuing_subword_prefix,
              model.end_of_word_suffix )
          with
          | true, true, _, _ -> char_str
          | true, false, _, Some suffix -> char_str ^ suffix
          | true, false, _, None -> char_str
          | false, true, Some prefix, Some suffix -> prefix ^ char_str ^ suffix
          | false, true, Some prefix, None -> prefix ^ char_str
          | false, true, None, Some suffix -> char_str ^ suffix
          | false, true, None, None -> char_str
          | false, false, Some prefix, _ -> prefix ^ char_str
          | false, false, None, _ -> char_str
        in
        let unk_handling () =
          match model.unk_token with
          | Some unk -> (
              match Hashtbl.find_opt model.vocab unk with
              | Some unk_id ->
                  if model.fuse_unk then
                    pending_unk :=
                      Some
                        (match !pending_unk with
                        | Some (id, len) -> (id, len + byte_len)
                        | None -> (unk_id, byte_len))
                  else (
                    flush_unk ();
                    add_symbol word unk_id byte_len)
              | None ->
                  failwith
                    (Printf.sprintf "Unknown token '%s' not in vocabulary" unk))
          | None -> ()
        in
        (match Hashtbl.find_opt model.vocab token_str with
        | Some id ->
            flush_unk ();
            add_symbol word id byte_len
        | None ->
            if model.byte_fallback then
              let byte_ids_opt =
                let rec loop acc idx =
                  if idx = byte_len then Some (List.rev acc)
                  else
                    let byte = char_str.[idx] in
                    let hex = Printf.sprintf "<0x%02X>" (Char.code byte) in
                    match Hashtbl.find_opt model.vocab hex with
                    | Some id -> loop (id :: acc) (idx + 1)
                    | None -> None
                in
                loop [] 0
              in
              match byte_ids_opt with
              | Some ids ->
                  flush_unk ();
                  List.iter (fun id -> add_symbol word id 1) ids
              | None -> unk_handling ()
            else unk_handling ());
        process_chars ()
    | `End -> flush_unk ()
    | `Malformed _ -> process_chars ()
    | `Await -> assert false
  in
  process_chars ();
  apply_merges model model.dropout word;
  word

let word_to_tokens model word =
  let tokens = ref [] in
  let offset = ref 0 in
  for i = 0 to word.size - 1 do
    if word.symbols.(i).len > 0 then (
      let id = word.symbols.(i).c in
      let value =
        match Hashtbl.find_opt model.vocab_r id with
        | Some v -> v
        | None -> "<unk>"
      in
      let start = !offset in
      let end_ = !offset + word.symbols.(i).len in
      tokens := { id; value; offsets = (start, end_) } :: !tokens;
      offset := end_)
  done;
  List.rev !tokens

let tokenize model text =
  if String.length text = 0 then []
  else
    (* First check if the entire text is in the vocabulary *)
    match Hashtbl.find_opt model.vocab text with
    | Some id -> [ { id; value = text; offsets = (0, String.length text) } ]
    | None -> (
        if
          (* If not, apply BPE merges *)
          model.ignore_merges
        then word_to_tokens model (merge_word model text)
        else
          match model.cache with
          | Some cache when String.length text < 1000 -> (
              match Hashtbl.find_opt cache text with
              | Some word -> word_to_tokens model word
              | None ->
                  let word = merge_word model text in
                  Hashtbl.add cache text word;
                  word_to_tokens model word)
          | _ ->
              let word = merge_word model text in
              word_to_tokens model word)

let token_to_id model token = Hashtbl.find_opt model.vocab token
let id_to_token model id = Hashtbl.find_opt model.vocab_r id
let get_vocab model = Hashtbl.fold (fun k v acc -> (k, v) :: acc) model.vocab []
let get_vocab_size model = Hashtbl.length model.vocab
let get_unk_token model = model.unk_token
let get_continuing_subword_prefix model = model.continuing_subword_prefix
let get_end_of_word_suffix model = model.end_of_word_suffix

let get_merges model =
  IntPairMap.fold
    (fun (a_id, b_id) (rank, _) acc ->
      match
        ( Hashtbl.find_opt model.vocab_r a_id,
          Hashtbl.find_opt model.vocab_r b_id )
      with
      | Some a, Some b -> (rank, (a, b)) :: acc
      | _ -> acc)
    model.merges []
  |> List.sort (fun (r1, _) (r2, _) -> Int.compare r1 r2)
  |> List.map snd

let clear_cache model =
  match model.cache with Some cache -> Hashtbl.clear cache | None -> ()

let resize_cache model _capacity =
  match model.cache with Some cache -> Hashtbl.clear cache | None -> ()

let convert_merges_to_merge_map vocab merges continuing_subword_prefix =
  let prefix_len =
    match continuing_subword_prefix with Some p -> String.length p | None -> 0
  in
  List.mapi
    (fun rank (a, b) ->
      match (Hashtbl.find_opt vocab a, Hashtbl.find_opt vocab b) with
      | Some a_id, Some b_id -> (
          let new_token =
            if prefix_len > 0 && String.length b > prefix_len then
              a ^ String.sub b prefix_len (String.length b - prefix_len)
            else a ^ b
          in
          match Hashtbl.find_opt vocab new_token with
          | Some new_id -> Some ((a_id, b_id), (rank, new_id))
          | None ->
              failwith
                (Printf.sprintf "Merge token '%s' not in vocabulary" new_token))
      | _ -> failwith (Printf.sprintf "Merge tokens not in vocabulary"))
    merges
  |> List.filter_map (fun x -> x)
  |> List.fold_left (fun acc (k, v) -> IntPairMap.add k v acc) IntPairMap.empty

let create (cfg : config) : t =
  let vocab_r = Hashtbl.create (Hashtbl.length cfg.vocab) in
  Hashtbl.iter (fun k v -> Hashtbl.add vocab_r v k) cfg.vocab;
  let cache =
    if cfg.cache_capacity = 0 then None
    else Some (Hashtbl.create cfg.cache_capacity)
  in
  let merges =
    convert_merges_to_merge_map cfg.vocab cfg.merges
      cfg.continuing_subword_prefix
  in
  {
    vocab = cfg.vocab;
    vocab_r;
    merges;
    cache;
    dropout = cfg.dropout;
    unk_token = cfg.unk_token;
    continuing_subword_prefix = cfg.continuing_subword_prefix;
    end_of_word_suffix = cfg.end_of_word_suffix;
    fuse_unk = cfg.fuse_unk;
    byte_fallback = cfg.byte_fallback;
    ignore_merges = cfg.ignore_merges;
  }

let read_files ~vocab_file ~merges_file =
  let vocab_json =
    let ic = open_in vocab_file in
    let content = really_input_string ic (in_channel_length ic) in
    close_in ic;
    Yojson.Basic.from_string content
  in
  let vocab = Hashtbl.create 1024 in
  (match vocab_json with
  | `Assoc items ->
      List.iter
        (fun (k, v) ->
          match v with
          | `Int id -> Hashtbl.add vocab k id
          | `Float f -> Hashtbl.add vocab k (int_of_float f)
          | _ -> failwith "Invalid vocab format")
        items
  | _ -> failwith "Invalid vocab.json format");
  let merges =
    let ic = open_in merges_file in
    let merges = ref [] in
    (try
       while true do
         let line = input_line ic in
         (* Skip empty lines and comment lines that start with #version *)
         if
           String.length line > 0
           && not (String.starts_with ~prefix:"#version" line)
         then
           match String.split_on_char ' ' line with
           | [ a; b ] -> merges := (a, b) :: !merges
           | _ -> failwith (Printf.sprintf "Invalid merge line: %s" line)
       done
     with End_of_file -> ());
    close_in ic;
    List.rev !merges
  in
  (vocab, merges)

let from_files ~vocab_file ~merges_file =
  let vocab, merges = read_files ~vocab_file ~merges_file in
  create
    {
      vocab;
      merges;
      cache_capacity = 10000;
      dropout = None;
      unk_token = None;
      continuing_subword_prefix = None;
      end_of_word_suffix = None;
      fuse_unk = false;
      byte_fallback = false;
      ignore_merges = false;
    }

let default () =
  create
    {
      vocab = Hashtbl.create 0;
      merges = [];
      cache_capacity = 10000;
      dropout = None;
      unk_token = None;
      continuing_subword_prefix = None;
      end_of_word_suffix = None;
      fuse_unk = false;
      byte_fallback = false;
      ignore_merges = false;
    }

let save model ~path ?name () =
  let vocab_file =
    match name with
    | Some n -> Filename.concat path (Printf.sprintf "%s-vocab.json" n)
    | None -> Filename.concat path "vocab.json"
  in
  let merges_file =
    match name with
    | Some n -> Filename.concat path (Printf.sprintf "%s-merges.txt" n)
    | None -> Filename.concat path "merges.txt"
  in
  let vocab_items =
    Hashtbl.fold
      (fun k v acc -> (k, (`Int v : Yojson.Basic.t)) :: acc)
      model.vocab []
    |> List.sort (fun (_, a) (_, b) ->
        match (a, b) with `Int x, `Int y -> compare x y | _ -> 0)
  in
  let vocab_json = `Assoc vocab_items in
  let oc = open_out vocab_file in
  output_string oc (Yojson.Basic.to_string vocab_json);
  close_out oc;
  let oc = open_out merges_file in
  output_string oc "#version: 0.2\n";
  let merges_list =
    IntPairMap.fold
      (fun (a_id, b_id) (rank, _) acc ->
        match
          ( Hashtbl.find_opt model.vocab_r a_id,
            Hashtbl.find_opt model.vocab_r b_id )
        with
        | Some a, Some b -> (rank, a, b) :: acc
        | _ -> acc)
      model.merges []
    |> List.sort (fun (r1, _, _) (r2, _, _) -> compare r1 r2)
  in
  List.iter (fun (_, a, b) -> Printf.fprintf oc "%s %s\n" a b) merges_list;
  close_out oc

let train ~min_frequency ~vocab_size ~show_progress ~special_tokens
    ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
    ~end_of_word_suffix ~max_token_length texts existing =
  let _ = (show_progress, existing) in

  (* Count words from texts *)
  let word_counts = Hashtbl.create 10000 in
  List.iter
    (fun text ->
      let words = String.split_on_char ' ' text in
      List.iter
        (fun word ->
          if String.length word > 0 then
            Hashtbl.replace word_counts word
              (1 + try Hashtbl.find word_counts word with Not_found -> 0))
        words)
    texts;

  let compute_pair_counts words_copy =
    let pair_counts = Hashtbl.create 10000 in
    Hashtbl.iter
      (fun word count ->
        let chars = String.split_on_char ' ' word in
        for i = 0 to List.length chars - 2 do
          let a = List.nth chars i in
          let b = List.nth chars (i + 1) in
          let pair = (a, b) in
          Hashtbl.replace pair_counts pair
            (count + try Hashtbl.find pair_counts pair with Not_found -> 0)
        done)
      words_copy;
    pair_counts
  in

  (* Build vocabulary *)
  let vocab = Hashtbl.create 10000 in
  let vocab_size_ref = ref 0 in
  List.iter
    (fun token ->
      if not (Hashtbl.mem vocab token) then (
        Hashtbl.add vocab token !vocab_size_ref;
        incr vocab_size_ref))
    special_tokens;

  (* Build alphabet *)
  let alphabet = Hashtbl.create 10000 in
  Hashtbl.iter
    (fun word count ->
      let decoder = Uutf.decoder (`String word) in
      let rec loop () =
        match Uutf.decode decoder with
        | `Uchar u ->
            let buf = Buffer.create 4 in
            Uutf.Buffer.add_utf_8 buf u;
            let char_str = Buffer.contents buf in
            Hashtbl.replace alphabet char_str
              (count + try Hashtbl.find alphabet char_str with Not_found -> 0);
            loop ()
        | `End -> ()
        | _ -> loop ()
      in
      loop ())
    word_counts;

  List.iter
    (fun c ->
      let char_str = String.make 1 c in
      Hashtbl.replace alphabet char_str max_int)
    initial_alphabet;

  let kept = Hashtbl.fold (fun k v acc -> (k, v) :: acc) alphabet [] in
  let kept = List.sort (fun (_, v1) (_, v2) -> compare v1 v2) kept in
  let to_remove =
    match limit_alphabet with
    | Some limit -> max 0 (List.length kept - limit)
    | None -> 0
  in
  let kept = list_drop to_remove kept in
  let kept = List.sort (fun (k1, _) (k2, _) -> compare k1 k2) kept in
  List.iter
    (fun (c, _) ->
      if not (Hashtbl.mem vocab c) then (
        Hashtbl.add vocab c !vocab_size_ref;
        incr vocab_size_ref))
    kept;

  (* Learn merges *)
  let merges = ref [] in
  let words_copy = ref (Hashtbl.create (Hashtbl.length word_counts)) in
  Hashtbl.iter
    (fun word count ->
      let decoder = Uutf.decoder (`String word) in
      let chars = ref [] in
      let rec loop () =
        match Uutf.decode decoder with
        | `Uchar u ->
            let buf = Buffer.create 4 in
            Uutf.Buffer.add_utf_8 buf u;
            chars := Buffer.contents buf :: !chars;
            loop ()
        | `End -> ()
        | _ -> loop ()
      in
      loop ();
      let separated = String.concat " " (List.rev !chars) in
      Hashtbl.add !words_copy separated count)
    word_counts;

  while !vocab_size_ref < vocab_size do
    let pair_counts = compute_pair_counts !words_copy in
    let best_pair = ref None in
    let best_count = ref (-1) in
    let best_pair_tie = ref ("", "") in
    Hashtbl.iter
      (fun pair count ->
        if count > !best_count then (
          best_count := count;
          best_pair := Some pair;
          best_pair_tie := pair)
        else if count = !best_count then
          if compare pair !best_pair_tie < 0 then best_pair_tie := pair)
      pair_counts;
    match !best_pair with
    | None -> vocab_size_ref := vocab_size
    | Some (a, b) ->
        if !best_count < min_frequency then vocab_size_ref := vocab_size
        else
          let new_token = a ^ b in
          let skip =
            match max_token_length with
            | Some l when String.length new_token > l -> true
            | _ -> false
          in
          if not skip then (
            if not (Hashtbl.mem vocab new_token) then (
              Hashtbl.add vocab new_token !vocab_size_ref;
              incr vocab_size_ref);
            merges := (a, b) :: !merges;
            let new_words = Hashtbl.create (Hashtbl.length !words_copy) in
            Hashtbl.iter
              (fun word count ->
                let merged =
                  Str.global_replace
                    (Str.regexp_string (a ^ " " ^ b))
                    new_token word
                in
                Hashtbl.add new_words merged count)
              !words_copy;
            words_copy := new_words)
  done;

  let bpe_config : config =
    {
      vocab;
      merges = List.rev !merges;
      cache_capacity = 10000;
      dropout = None;
      unk_token = None;
      continuing_subword_prefix;
      end_of_word_suffix;
      fuse_unk = false;
      byte_fallback = false;
      ignore_merges = false;
    }
  in
  let trained_model = create bpe_config in
  (trained_model, special_tokens)
