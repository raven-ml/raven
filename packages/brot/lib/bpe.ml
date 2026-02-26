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

type vocab = (string, int) Hashtbl.t
type merges = (string * string) list

(* Open-addressing hash table for merge lookups. Returns int directly (no option
   allocation). -1 = not found. *)
module Merge_map = struct
  type t = { keys : int array; values : int array; mask : int }

  let[@inline] hash key =
    let h = key * 0x517CC1B727220A95 in
    h lxor (h lsr 29)

  let create entries =
    let n = List.length entries in
    let cap = ref 16 in
    while !cap < n * 4 do
      cap := !cap * 2
    done;
    let mask = !cap - 1 in
    let keys = Array.make !cap (-1) in
    let values = Array.make !cap 0 in
    List.iter
      (fun (key, value) ->
        let h = ref (hash key land mask) in
        while Array.unsafe_get keys !h >= 0 do
          h := (!h + 1) land mask
        done;
        Array.unsafe_set keys !h key;
        Array.unsafe_set values !h value)
      entries;
    { keys; values; mask }

  let[@inline] find t key =
    let mask = t.mask in
    let keys = t.keys in
    let h = ref (hash key land mask) in
    let k = ref (Array.unsafe_get keys !h) in
    while !k <> key && !k >= 0 do
      h := (!h + 1) land mask;
      k := Array.unsafe_get keys !h
    done;
    if !k = key then Array.unsafe_get t.values !h else -1

  let fold f t acc =
    let keys = t.keys in
    let values = t.values in
    let len = Array.length keys in
    let acc = ref acc in
    for i = 0 to len - 1 do
      let k = Array.unsafe_get keys i in
      if k >= 0 then acc := f k (Array.unsafe_get values i) !acc
    done;
    !acc
end

let[@inline] merge_key a b = (a lsl 21) lor b
let[@inline] pack_merge rank new_id = (rank lsl 21) lor new_id
let[@inline] merge_rank v = v lsr 21
let[@inline] merge_new_id v = v land 0x1FFFFF

type word = {
  sym_c : int array;
  sym_prev : int array;
  sym_next : int array;
  sym_len : int array;
  mutable size : int;
}

(* Specialized min-heap for BPE merges using parallel arrays (no tuple allocation).
   Ordered by (rank, position) — lower rank first, then lower position. *)
(* Min-heap with packed comparison key: (rank lsl 21) lor pos.
   Single int comparison for sift operations, 2 arrays instead of 3. *)
module Merge_queue = struct
  type t = {
    mutable keys : int array;
    mutable new_ids : int array;
    mutable size : int;
    mutable pop_key : int;
    mutable pop_new_id : int;
    mutable skip_keys : int array;
    mutable skip_new_ids : int array;
    mutable skip_size : int;
  }

  let create cap =
    let cap = max 16 cap in
    {
      keys = Array.make cap 0;
      new_ids = Array.make cap 0;
      size = 0;
      pop_key = 0;
      pop_new_id = 0;
      skip_keys = [||];
      skip_new_ids = [||];
      skip_size = 0;
    }

  let[@inline] pack_key rank pos = (rank lsl 21) lor pos

  let sift_up t idx =
    let keys = t.keys in
    let new_ids = t.new_ids in
    let key = Array.unsafe_get keys idx in
    let nid = Array.unsafe_get new_ids idx in
    let i = ref idx in
    let cont = ref (!i > 0) in
    while !cont do
      let p = (!i - 1) asr 1 in
      if key < Array.unsafe_get keys p then (
        Array.unsafe_set keys !i (Array.unsafe_get keys p);
        Array.unsafe_set new_ids !i (Array.unsafe_get new_ids p);
        i := p;
        cont := !i > 0)
      else cont := false
    done;
    Array.unsafe_set keys !i key;
    Array.unsafe_set new_ids !i nid

  let sift_down t idx =
    let keys = t.keys in
    let new_ids = t.new_ids in
    let size = t.size in
    let key = Array.unsafe_get keys idx in
    let nid = Array.unsafe_get new_ids idx in
    let i = ref idx in
    let continue_ = ref true in
    while !continue_ do
      let l = (2 * !i) + 1 in
      if l >= size then continue_ := false
      else begin
        let r = l + 1 in
        let smallest =
          if r < size && Array.unsafe_get keys r < Array.unsafe_get keys l then
            r
          else l
        in
        if Array.unsafe_get keys smallest < key then (
          Array.unsafe_set keys !i (Array.unsafe_get keys smallest);
          Array.unsafe_set new_ids !i (Array.unsafe_get new_ids smallest);
          i := smallest)
        else continue_ := false
      end
    done;
    Array.unsafe_set keys !i key;
    Array.unsafe_set new_ids !i nid

  let push t rank pos new_id =
    let s = t.size in
    if s = Array.length t.keys then begin
      let new_cap = max 16 (s * 2) in
      let grow a =
        let b = Array.make new_cap 0 in
        Array.blit a 0 b 0 s;
        b
      in
      t.keys <- grow t.keys;
      t.new_ids <- grow t.new_ids
    end;
    Array.unsafe_set t.keys s (pack_key rank pos);
    Array.unsafe_set t.new_ids s new_id;
    t.size <- s + 1;
    sift_up t s

  let pop t =
    if t.size = 0 then false
    else begin
      t.pop_key <- Array.unsafe_get t.keys 0;
      t.pop_new_id <- Array.unsafe_get t.new_ids 0;
      t.size <- t.size - 1;
      if t.size > 0 then begin
        Array.unsafe_set t.keys 0 (Array.unsafe_get t.keys t.size);
        Array.unsafe_set t.new_ids 0 (Array.unsafe_get t.new_ids t.size);
        sift_down t 0
      end;
      true
    end
end

type token = { id : int; value : string; offsets : int * int }

(* Direct-mapped bounded cache: hash key to slot, newest entry wins. Fixed
   memory, no eviction logic, no unbounded growth. *)
type cache = {
  cache_keys : string array;
  cache_vals : word array;
  cache_mask : int;
}

let empty_word =
  { sym_c = [||]; sym_prev = [||]; sym_next = [||]; sym_len = [||]; size = 0 }

let create_cache capacity =
  (* Round up to power of 2 *)
  let cap = ref 16 in
  while !cap < capacity do
    cap := !cap * 2
  done;
  {
    cache_keys = Array.make !cap "";
    cache_vals = Array.make !cap empty_word;
    cache_mask = !cap - 1;
  }

let[@inline] cache_find c key =
  let h = Hashtbl.hash key land c.cache_mask in
  if String.equal (Array.unsafe_get c.cache_keys h) key then
    Array.unsafe_get c.cache_vals h
  else empty_word

let[@inline] cache_add c key value =
  let h = Hashtbl.hash key land c.cache_mask in
  Array.unsafe_set c.cache_keys h key;
  Array.unsafe_set c.cache_vals h value

type t = {
  vocab : vocab;
  vocab_r : string array;
  merges : Merge_map.t;
  cache : cache option;
  dropout : float option;
  unk_token : string option;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
  fuse_unk : bool;
  byte_fallback : bool;
  ignore_merges : bool;
  ascii_to_id : int array;
  byte_fallback_ids : int array;
  char_to_id : Merge_map.t;
  prefixed_ascii_to_id : int array;
  prefixed_char_to_id : Merge_map.t;
  unk_id : int;
  mutable work_word : word;
  mutable work_queue : Merge_queue.t;
  work_in_use : bool Atomic.t;
}

let create_word capacity =
  let cap = max 16 capacity in
  {
    sym_c = Array.make cap 0;
    sym_prev = Array.make cap 0;
    sym_next = Array.make cap 0;
    sym_len = Array.make cap 0;
    size = 0;
  }

let ensure_word_capacity word capacity =
  if Array.length word.sym_c >= capacity then begin
    word.size <- 0;
    word
  end
  else create_word capacity

let ensure_queue_capacity queue capacity =
  let cap = max 16 capacity in
  if Array.length queue.Merge_queue.keys >= cap then begin
    queue.Merge_queue.size <- 0;
    queue
  end
  else Merge_queue.create cap

let[@inline] add_symbol word c byte_len =
  let s = word.size in
  let prev = if s > 0 then s - 1 else -1 in
  Array.unsafe_set word.sym_c s c;
  Array.unsafe_set word.sym_prev s prev;
  Array.unsafe_set word.sym_next s (-1);
  Array.unsafe_set word.sym_len s byte_len;
  if prev >= 0 then Array.unsafe_set word.sym_next prev s;
  word.size <- s + 1

let apply_merges model dropout word queue =
  let p = match dropout with Some p -> p | None -> 0.0 in
  let use_dropout = p > 0.0 in
  let merges = model.merges in
  let sym_c = word.sym_c in
  let sym_prev = word.sym_prev in
  let sym_next = word.sym_next in
  let sym_len = word.sym_len in
  for i = 0 to word.size - 2 do
    let key =
      merge_key (Array.unsafe_get sym_c i) (Array.unsafe_get sym_c (i + 1))
    in
    let packed = Merge_map.find merges key in
    if packed >= 0 then
      Merge_queue.push queue (merge_rank packed) i (merge_new_id packed)
  done;
  queue.skip_size <- 0;
  while Merge_queue.pop queue do
    let pkey = queue.pop_key in
    let pos = pkey land 0x1FFFFF in
    let new_id = queue.pop_new_id in
    if Array.unsafe_get sym_len pos > 0 then begin
      let next_pos = Array.unsafe_get sym_next pos in
      if next_pos >= 0 then begin
        let key =
          merge_key
            (Array.unsafe_get sym_c pos)
            (Array.unsafe_get sym_c next_pos)
        in
        let packed = Merge_map.find merges key in
        if packed >= 0 && merge_new_id packed = new_id then begin
          if use_dropout && Random.float 1.0 < p then begin
            let s = queue.skip_size in
            if s = Array.length queue.skip_keys then begin
              let new_cap = max 8 (s * 2) in
              let grow old =
                let a = Array.make new_cap 0 in
                if s > 0 then Array.blit old 0 a 0 s;
                a
              in
              queue.skip_keys <- grow queue.skip_keys;
              queue.skip_new_ids <- grow queue.skip_new_ids
            end;
            Array.unsafe_set queue.skip_keys s pkey;
            Array.unsafe_set queue.skip_new_ids s new_id;
            queue.skip_size <- s + 1
          end
          else begin
            for i = 0 to queue.skip_size - 1 do
              Merge_queue.push queue
                (Array.unsafe_get queue.skip_keys i lsr 21)
                (Array.unsafe_get queue.skip_keys i land 0x1FFFFF)
                (Array.unsafe_get queue.skip_new_ids i)
            done;
            queue.skip_size <- 0;
            Array.unsafe_set sym_c pos new_id;
            Array.unsafe_set sym_len pos
              (Array.unsafe_get sym_len pos + Array.unsafe_get sym_len next_pos);
            Array.unsafe_set sym_next pos (Array.unsafe_get sym_next next_pos);
            Array.unsafe_set sym_len next_pos 0;
            let new_next = Array.unsafe_get sym_next pos in
            if new_next >= 0 then Array.unsafe_set sym_prev new_next pos;
            let prev = Array.unsafe_get sym_prev pos in
            if prev >= 0 then begin
              let k =
                merge_key
                  (Array.unsafe_get sym_c prev)
                  (Array.unsafe_get sym_c pos)
              in
              let v = Merge_map.find merges k in
              if v >= 0 then
                Merge_queue.push queue (merge_rank v) prev (merge_new_id v)
            end;
            let next = Array.unsafe_get sym_next pos in
            if next >= 0 then begin
              let k =
                merge_key
                  (Array.unsafe_get sym_c pos)
                  (Array.unsafe_get sym_c next)
              in
              let v = Merge_map.find merges k in
              if v >= 0 then
                Merge_queue.push queue (merge_rank v) pos (merge_new_id v)
            end
          end
        end
      end
    end
  done;
  (* Compact using linked-list traversal: O(N_final) instead of O(N_original) *)
  let j = ref 0 in
  let cur = ref 0 in
  while !cur >= 0 do
    if !j <> !cur then begin
      Array.unsafe_set sym_c !j (Array.unsafe_get sym_c !cur);
      Array.unsafe_set sym_len !j (Array.unsafe_get sym_len !cur)
    end;
    incr j;
    cur := Array.unsafe_get sym_next !cur
  done;
  word.size <- !j

let utf8_byte_len_table =
  Array.init 256 (fun b ->
      if b land 0x80 = 0 then 1
      else if b land 0xE0 = 0xC0 then 2
      else if b land 0xF0 = 0xE0 then 3
      else if b land 0xF8 = 0xF0 then 4
      else 1)

let[@inline] utf8_byte_len b = Array.unsafe_get utf8_byte_len_table b

let[@inline] pack_char_key text pos byte_len =
  let b0 = Char.code (String.unsafe_get text pos) in
  match byte_len with
  | 1 -> b0
  | 2 -> (b0 lsl 8) lor Char.code (String.unsafe_get text (pos + 1))
  | 3 ->
      (b0 lsl 16)
      lor (Char.code (String.unsafe_get text (pos + 1)) lsl 8)
      lor Char.code (String.unsafe_get text (pos + 2))
  | _ ->
      (b0 lsl 24)
      lor (Char.code (String.unsafe_get text (pos + 1)) lsl 16)
      lor (Char.code (String.unsafe_get text (pos + 2)) lsl 8)
      lor Char.code (String.unsafe_get text (pos + 3))

(* Try emitting byte fallback tokens for [byte_len] bytes starting at [src]
   offset [offset]. Returns true if all bytes had fallback IDs. *)
let try_byte_fallback model word flush_unk src offset byte_len =
  let all_found = ref true in
  for i = 0 to byte_len - 1 do
    if
      Array.unsafe_get model.byte_fallback_ids
        (Char.code (String.unsafe_get src (offset + i)))
      < 0
    then all_found := false
  done;
  if !all_found then begin
    flush_unk ();
    for i = 0 to byte_len - 1 do
      add_symbol word
        (Array.unsafe_get model.byte_fallback_ids
           (Char.code (String.unsafe_get src (offset + i))))
        1
    done;
    true
  end
  else false

(* No prefix/suffix — avoids all per-character string allocation for ASCII via
   pre-computed lookup tables. *)
let init_word_fast model word text text_len =
  let pos = ref 0 in
  let pending_unk_id = ref (-1) in
  let pending_unk_len = ref 0 in
  let flush_unk () =
    if !pending_unk_id >= 0 then begin
      add_symbol word !pending_unk_id !pending_unk_len;
      pending_unk_id := -1;
      pending_unk_len := 0
    end
  in
  let handle_unk byte_len =
    if model.unk_id >= 0 then begin
      if model.fuse_unk then begin
        if !pending_unk_id >= 0 then
          pending_unk_len := !pending_unk_len + byte_len
        else begin
          pending_unk_id := model.unk_id;
          pending_unk_len := byte_len
        end
      end
      else begin
        flush_unk ();
        add_symbol word model.unk_id byte_len
      end
    end
  in
  while !pos < text_len do
    let b = Char.code (String.unsafe_get text !pos) in
    if b < 128 then begin
      let id = Array.unsafe_get model.ascii_to_id b in
      if id >= 0 then begin
        flush_unk ();
        add_symbol word id 1
      end
      else if model.byte_fallback then begin
        let fbid = Array.unsafe_get model.byte_fallback_ids b in
        if fbid >= 0 then begin
          flush_unk ();
          add_symbol word fbid 1
        end
        else handle_unk 1
      end
      else handle_unk 1;
      incr pos
    end
    else begin
      let byte_len = utf8_byte_len b in
      let key = pack_char_key text !pos byte_len in
      let id = Merge_map.find model.char_to_id key in
      if id >= 0 then begin
        flush_unk ();
        add_symbol word id byte_len
      end
      else if model.byte_fallback then begin
        if not (try_byte_fallback model word flush_unk text !pos byte_len) then
          handle_unk byte_len
      end
      else handle_unk byte_len;
      pos := !pos + byte_len
    end
  done;
  flush_unk ()

(* Models with continuing_subword_prefix or end_of_word_suffix *)
let init_word_slow model word text text_len =
  let pos = ref 0 in
  let pending_unk_id = ref (-1) in
  let pending_unk_len = ref 0 in
  let flush_unk () =
    if !pending_unk_id >= 0 then begin
      add_symbol word !pending_unk_id !pending_unk_len;
      pending_unk_id := -1;
      pending_unk_len := 0
    end
  in
  let handle_unk byte_len =
    if model.unk_id >= 0 then begin
      if model.fuse_unk then begin
        if !pending_unk_id >= 0 then
          pending_unk_len := !pending_unk_len + byte_len
        else begin
          pending_unk_id := model.unk_id;
          pending_unk_len := byte_len
        end
      end
      else begin
        flush_unk ();
        add_symbol word model.unk_id byte_len
      end
    end
  in
  let has_prefix = model.continuing_subword_prefix <> None in
  let has_suffix = model.end_of_word_suffix <> None in
  while !pos < text_len do
    let b = Char.code (String.unsafe_get text !pos) in
    let byte_len = utf8_byte_len b in
    if b land 0xC0 = 0x80 then pos := !pos + 1
    else begin
      let start = !pos in
      let is_first = start = 0 in
      let is_last = !pos + byte_len >= text_len in
      pos := !pos + byte_len;
      (* Suffix only applies at word boundaries (first-not-last or
         last-not-first), never to middle chars and never to single-char
         words *)
      let needs_string = has_suffix && is_first <> is_last in
      if needs_string then begin
        (* Slow path: suffix involved, at most 2x per word *)
        let char_str = String.sub text start byte_len in
        let token_str =
          match
            ( is_first,
              is_last,
              model.continuing_subword_prefix,
              model.end_of_word_suffix )
          with
          | true, false, _, Some suffix -> char_str ^ suffix
          | true, false, _, None -> char_str
          | false, true, Some prefix, Some suffix -> prefix ^ char_str ^ suffix
          | false, true, Some prefix, None -> prefix ^ char_str
          | false, true, None, Some suffix -> char_str ^ suffix
          | false, true, None, None -> char_str
          | _, _, _, _ -> char_str
        in
        match Hashtbl.find_opt model.vocab token_str with
        | Some id ->
            flush_unk ();
            add_symbol word id byte_len
        | None ->
            if model.byte_fallback then begin
              if
                not (try_byte_fallback model word flush_unk text start byte_len)
              then handle_unk byte_len
            end
            else handle_unk byte_len
      end
      else begin
        (* Fast path: no suffix, use packed-int lookup (zero allocation) *)
        let needs_prefix = has_prefix && not is_first in
        let id =
          if needs_prefix then
            if b < 128 then Array.unsafe_get model.prefixed_ascii_to_id b
            else
              Merge_map.find model.prefixed_char_to_id
                (pack_char_key text start byte_len)
          else if b < 128 then Array.unsafe_get model.ascii_to_id b
          else
            Merge_map.find model.char_to_id (pack_char_key text start byte_len)
        in
        if id >= 0 then begin
          flush_unk ();
          add_symbol word id byte_len
        end
        else if model.byte_fallback then begin
          if not (try_byte_fallback model word flush_unk text start byte_len)
          then handle_unk byte_len
        end
        else handle_unk byte_len
      end
    end
  done;
  flush_unk ()

let merge_word model text =
  let text_len = String.length text in
  let owned = Atomic.compare_and_set model.work_in_use false true in
  let word, queue =
    if owned then begin
      let w = ensure_word_capacity model.work_word text_len in
      model.work_word <- w;
      let q = ensure_queue_capacity model.work_queue text_len in
      model.work_queue <- q;
      (w, q)
    end
    else (create_word text_len, Merge_queue.create text_len)
  in
  if model.continuing_subword_prefix = None && model.end_of_word_suffix = None
  then init_word_fast model word text text_len
  else init_word_slow model word text text_len;
  apply_merges model model.dropout word queue;
  if owned then begin
    let n = word.size in
    let sym_c = Array.make n 0 in
    let sym_len = Array.make n 0 in
    Array.blit word.sym_c 0 sym_c 0 n;
    Array.blit word.sym_len 0 sym_len 0 n;
    Atomic.set model.work_in_use false;
    { sym_c; sym_prev = [||]; sym_next = [||]; sym_len; size = n }
  end
  else word

let word_to_tokens model word =
  let offset = ref 0 in
  List.init word.size (fun i ->
      let id = Array.unsafe_get word.sym_c i in
      let vr = model.vocab_r in
      let value =
        if id >= 0 && id < Array.length vr then Array.unsafe_get vr id
        else "<unk>"
      in
      let start = !offset in
      let end_ = start + Array.unsafe_get word.sym_len i in
      offset := end_;
      { id; value; offsets = (start, end_) })

let word_to_ids word =
  Array.init word.size (fun i -> Array.unsafe_get word.sym_c i)

let word_to_encoding model word ~type_id =
  let n = word.size in
  let ids = Array.make n 0 in
  let tokens = Array.make n "" in
  let offsets = Array.make n (0, 0) in
  let offset = ref 0 in
  for i = 0 to n - 1 do
    let id = Array.unsafe_get word.sym_c i in
    Array.unsafe_set ids i id;
    let vr = model.vocab_r in
    Array.unsafe_set tokens i
      (if id >= 0 && id < Array.length vr then Array.unsafe_get vr id
       else "<unk>");
    let start = !offset in
    let end_ = start + Array.unsafe_get word.sym_len i in
    Array.unsafe_set offsets i (start, end_);
    offset := end_
  done;
  Encoding.create ~ids ~type_ids:(Array.make n type_id) ~tokens
    ~words:(Array.make n None) ~offsets ~special_tokens_mask:(Array.make n 0)
    ~attention_mask:(Array.make n 1) ()

let get_word model text =
  if model.ignore_merges then merge_word model text
  else
    match model.cache with
    | Some cache when String.length text < 4096 ->
        let cached = cache_find cache text in
        if cached.size > 0 then cached
        else
          let word = merge_word model text in
          cache_add cache text word;
          word
    | _ -> merge_word model text

let tokenize model text =
  if String.length text = 0 then []
  else
    match Hashtbl.find_opt model.vocab text with
    | Some id -> [ { id; value = text; offsets = (0, String.length text) } ]
    | None -> word_to_tokens model (get_word model text)

let tokenize_ids model text =
  if String.length text = 0 then [||]
  else
    match Hashtbl.find_opt model.vocab text with
    | Some id -> [| id |]
    | None -> word_to_ids (get_word model text)

let tokenize_encoding model text ~type_id =
  if String.length text = 0 then Encoding.empty
  else
    match Hashtbl.find_opt model.vocab text with
    | Some id ->
        Encoding.token ~id ~token:text
          ~offset:(0, String.length text)
          ~type_id ~special:false
    | None -> word_to_encoding model (get_word model text) ~type_id

let token_to_id model token = Hashtbl.find_opt model.vocab token

let id_to_token model id =
  if id >= 0 && id < Array.length model.vocab_r then
    Some (Array.unsafe_get model.vocab_r id)
  else None

let get_vocab model = Hashtbl.fold (fun k v acc -> (k, v) :: acc) model.vocab []
let get_vocab_size model = Hashtbl.length model.vocab
let get_unk_token model = model.unk_token
let get_continuing_subword_prefix model = model.continuing_subword_prefix
let get_end_of_word_suffix model = model.end_of_word_suffix

let get_merges model =
  Merge_map.fold
    (fun key packed acc ->
      let a_id = key lsr 21 in
      let b_id = key land 0x1FFFFF in
      let rank = merge_rank packed in
      let vr = model.vocab_r in
      let vr_len = Array.length vr in
      if a_id >= 0 && a_id < vr_len && b_id >= 0 && b_id < vr_len then
        (rank, (Array.unsafe_get vr a_id, Array.unsafe_get vr b_id)) :: acc
      else acc)
    model.merges []
  |> List.sort (fun (r1, _) (r2, _) -> Int.compare r1 r2)
  |> List.map snd

let convert_merges_to_merge_map vocab merges continuing_subword_prefix =
  let csp_str =
    match continuing_subword_prefix with Some p -> p | None -> ""
  in
  let csp_len = String.length csp_str in
  List.mapi
    (fun rank (a, b) ->
      match (Hashtbl.find_opt vocab a, Hashtbl.find_opt vocab b) with
      | Some a_id, Some b_id -> (
          let alen = String.length a in
          let blen = String.length b in
          let new_token =
            if
              csp_len > 0 && blen > csp_len
              && String.starts_with ~prefix:csp_str b
            then (
              let brest = blen - csp_len in
              let s = Bytes.create (alen + brest) in
              Bytes.blit_string a 0 s 0 alen;
              Bytes.blit_string b csp_len s alen brest;
              Bytes.unsafe_to_string s)
            else
              let s = Bytes.create (alen + blen) in
              Bytes.blit_string a 0 s 0 alen;
              Bytes.blit_string b 0 s alen blen;
              Bytes.unsafe_to_string s
          in
          match Hashtbl.find_opt vocab new_token with
          | Some new_id -> Some ((a_id, b_id), pack_merge rank new_id)
          | None ->
              failwith
                (Printf.sprintf "Merge token '%s' not in vocabulary" new_token))
      | _ ->
          failwith
            (Printf.sprintf "Merge tokens ('%s', '%s') not in vocabulary" a b))
    merges
  |> List.filter_map Fun.id
  |> fun entries ->
  Merge_map.create
    (List.map
       (fun ((a_id, b_id), packed) -> (merge_key a_id b_id, packed))
       entries)

let create ~vocab ~merges ?(cache_capacity = 10000) ?dropout ?unk_token
    ?continuing_subword_prefix ?end_of_word_suffix ?(fuse_unk = false)
    ?(byte_fallback = false) ?(ignore_merges = false) () : t =
  let max_id = Hashtbl.fold (fun _ id acc -> max id acc) vocab (-1) in
  let vocab_r = Array.make (max_id + 1) "" in
  Hashtbl.iter (fun k v -> Array.unsafe_set vocab_r v k) vocab;
  let cache =
    if cache_capacity = 0 then None else Some (create_cache cache_capacity)
  in
  let merges =
    convert_merges_to_merge_map vocab merges continuing_subword_prefix
  in
  let ascii_to_id = Array.make 128 (-1) in
  for i = 0 to 127 do
    let s = String.make 1 (Char.chr i) in
    match Hashtbl.find_opt vocab s with
    | Some id -> ascii_to_id.(i) <- id
    | None -> ()
  done;
  let byte_fallback_ids = Array.make 256 (-1) in
  for i = 0 to 255 do
    let hex = Printf.sprintf "<0x%02X>" i in
    match Hashtbl.find_opt vocab hex with
    | Some id -> byte_fallback_ids.(i) <- id
    | None -> ()
  done;
  (* Build packed-int char lookup table for zero-allocation multi-byte lookup *)
  let char_entries = ref [] in
  Hashtbl.iter
    (fun key id ->
      let len = String.length key in
      if len >= 1 && len <= 4 then begin
        let b0 = Char.code (String.unsafe_get key 0) in
        let expected_len = utf8_byte_len b0 in
        if expected_len = len then
          let packed =
            match len with
            | 1 -> b0
            | 2 -> (b0 lsl 8) lor Char.code (String.unsafe_get key 1)
            | 3 ->
                (b0 lsl 16)
                lor (Char.code (String.unsafe_get key 1) lsl 8)
                lor Char.code (String.unsafe_get key 2)
            | _ ->
                (b0 lsl 24)
                lor (Char.code (String.unsafe_get key 1) lsl 16)
                lor (Char.code (String.unsafe_get key 2) lsl 8)
                lor Char.code (String.unsafe_get key 3)
          in
          char_entries := (packed, id) :: !char_entries
      end)
    vocab;
  let char_to_id = Merge_map.create !char_entries in
  (* Build prefixed char lookup tables for zero-allocation init_word_slow *)
  let prefixed_ascii_to_id = Array.make 128 (-1) in
  let prefixed_char_entries = ref [] in
  (match continuing_subword_prefix with
  | Some prefix ->
      for i = 0 to 127 do
        let s = prefix ^ String.make 1 (Char.chr i) in
        match Hashtbl.find_opt vocab s with
        | Some id -> prefixed_ascii_to_id.(i) <- id
        | None -> ()
      done;
      Hashtbl.iter
        (fun key id ->
          let plen = String.length prefix in
          let klen = String.length key in
          if klen > plen && String.sub key 0 plen = prefix then begin
            let rest_len = klen - plen in
            if rest_len >= 2 && rest_len <= 4 then begin
              let b0 = Char.code (String.unsafe_get key plen) in
              let expected = utf8_byte_len b0 in
              if expected = rest_len then
                let packed = pack_char_key key plen rest_len in
                prefixed_char_entries := (packed, id) :: !prefixed_char_entries
            end
          end)
        vocab
  | None -> ());
  let prefixed_char_to_id = Merge_map.create !prefixed_char_entries in
  let unk_id =
    match unk_token with
    | Some unk -> (
        match Hashtbl.find_opt vocab unk with Some id -> id | None -> -1)
    | None -> -1
  in
  {
    vocab;
    vocab_r;
    merges;
    cache;
    dropout;
    unk_token;
    continuing_subword_prefix;
    end_of_word_suffix;
    fuse_unk;
    byte_fallback;
    ignore_merges;
    ascii_to_id;
    byte_fallback_ids;
    char_to_id;
    prefixed_ascii_to_id;
    prefixed_char_to_id;
    unk_id;
    work_word = create_word 16;
    work_queue = Merge_queue.create 16;
    work_in_use = Atomic.make false;
  }

let json_of_string s =
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> v
  | Error e -> failwith e

let json_to_string j =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json j with
  | Ok s -> s
  | Error e -> failwith e

let read_files ~vocab_file ~merges_file =
  let vocab_json =
    let ic = open_in vocab_file in
    let content =
      Fun.protect
        ~finally:(fun () -> close_in ic)
        (fun () -> really_input_string ic (in_channel_length ic))
    in
    json_of_string content
  in
  let vocab = Hashtbl.create 1024 in
  (match vocab_json with
  | Jsont.Object (mems, _) ->
      List.iter
        (fun ((k, _), v) ->
          match v with
          | Jsont.Number (f, _) -> Hashtbl.add vocab k (int_of_float f)
          | _ -> failwith "Invalid vocab format")
        mems
  | _ -> failwith "Invalid vocab.json format");
  let merges =
    let ic = open_in merges_file in
    Fun.protect
      ~finally:(fun () -> close_in ic)
      (fun () ->
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
        List.rev !merges)
  in
  (vocab, merges)

let from_files ~vocab_file ~merges_file =
  let vocab, merges = read_files ~vocab_file ~merges_file in
  create ~vocab ~merges ()

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
    Hashtbl.fold (fun k v acc -> (k, v) :: acc) model.vocab []
    |> List.sort (fun (_, a) (_, b) -> compare a b)
  in
  let vocab_json =
    Jsont.Json.object'
      (List.map
         (fun (k, v) -> (Jsont.Json.name k, Jsont.Json.int v))
         vocab_items)
  in
  let oc = open_out vocab_file in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc (json_to_string vocab_json));
  let oc = open_out merges_file in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () ->
      output_string oc "#version: 0.2\n";
      let merges_list =
        Merge_map.fold
          (fun key packed acc ->
            let a_id = key lsr 21 in
            let b_id = key land 0x1FFFFF in
            let rank = merge_rank packed in
            let vr = model.vocab_r in
            let vr_len = Array.length vr in
            if a_id >= 0 && a_id < vr_len && b_id >= 0 && b_id < vr_len then
              (rank, Array.unsafe_get vr a_id, Array.unsafe_get vr b_id) :: acc
            else acc)
          model.merges []
        |> List.sort (fun (r1, _, _) (r2, _, _) -> compare r1 r2)
      in
      List.iter (fun (_, a, b) -> Printf.fprintf oc "%s %s\n" a b) merges_list)

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
      let len = String.length word in
      let buf = Buffer.create 4 in
      let rec loop i =
        if i >= len then ()
        else
          let d = String.get_utf_8_uchar word i in
          let n = Uchar.utf_decode_length d in
          if Uchar.utf_decode_is_valid d then (
            let u = Uchar.utf_decode_uchar d in
            Buffer.clear buf;
            Buffer.add_utf_8_uchar buf u;
            let char_str = Buffer.contents buf in
            Hashtbl.replace alphabet char_str
              (count + try Hashtbl.find alphabet char_str with Not_found -> 0));
          loop (i + n)
      in
      loop 0)
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
  let csp_str =
    match continuing_subword_prefix with Some p -> p | None -> ""
  in
  let csp_len = String.length csp_str in
  List.iter
    (fun (c, _) ->
      if not (Hashtbl.mem vocab c) then (
        Hashtbl.add vocab c !vocab_size_ref;
        incr vocab_size_ref);
      if csp_len > 0 then (
        let clen = String.length c in
        let s = Bytes.create (csp_len + clen) in
        Bytes.blit_string csp_str 0 s 0 csp_len;
        Bytes.blit_string c 0 s csp_len clen;
        let prefixed = Bytes.unsafe_to_string s in
        if not (Hashtbl.mem vocab prefixed) then (
          Hashtbl.add vocab prefixed !vocab_size_ref;
          incr vocab_size_ref)))
    kept;

  (* Learn merges *)
  let merges = ref [] in
  let words_copy = ref (Hashtbl.create (Hashtbl.length word_counts)) in
  Hashtbl.iter
    (fun word count ->
      let len = String.length word in
      let chars = ref [] in
      let buf = Buffer.create 8 in
      let is_first = ref true in
      let rec loop i =
        if i >= len then ()
        else
          let d = String.get_utf_8_uchar word i in
          let n = Uchar.utf_decode_length d in
          if Uchar.utf_decode_is_valid d then (
            let u = Uchar.utf_decode_uchar d in
            Buffer.clear buf;
            if csp_len > 0 && not !is_first then Buffer.add_string buf csp_str;
            Buffer.add_utf_8_uchar buf u;
            is_first := false;
            chars := Buffer.contents buf :: !chars);
          loop (i + n)
      in
      loop 0;
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
          let blen = String.length b in
          let new_token =
            if
              csp_len > 0 && blen > csp_len
              && String.starts_with ~prefix:csp_str b
            then (
              let alen = String.length a in
              let brest = blen - csp_len in
              let s = Bytes.create (alen + brest) in
              Bytes.blit_string a 0 s 0 alen;
              Bytes.blit_string b csp_len s alen brest;
              Bytes.unsafe_to_string s)
            else a ^ b
          in
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
            let pat = a ^ " " ^ b in
            let pat_len = String.length pat in
            Hashtbl.iter
              (fun word count ->
                let wlen = String.length word in
                if wlen < pat_len then Hashtbl.add new_words word count
                else
                  let buf = Buffer.create wlen in
                  let pos = ref 0 in
                  let changed = ref false in
                  while !pos <= wlen - pat_len do
                    let at_boundary =
                      (!pos = 0
                      || Char.equal (String.unsafe_get word (!pos - 1)) ' ')
                      && (!pos + pat_len = wlen
                         || Char.equal
                              (String.unsafe_get word (!pos + pat_len))
                              ' ')
                    in
                    if at_boundary && String.sub word !pos pat_len = pat then (
                      Buffer.add_string buf new_token;
                      pos := !pos + pat_len;
                      changed := true)
                    else (
                      Buffer.add_char buf (String.unsafe_get word !pos);
                      incr pos)
                  done;
                  if !changed then (
                    Buffer.add_substring buf word !pos (wlen - !pos);
                    Hashtbl.add new_words (Buffer.contents buf) count)
                  else Hashtbl.add new_words word count)
              !words_copy;
            words_copy := new_words)
  done;

  let trained_model =
    create ~vocab ~merges:(List.rev !merges) ?continuing_subword_prefix
      ?end_of_word_suffix ()
  in
  (trained_model, special_tokens)
