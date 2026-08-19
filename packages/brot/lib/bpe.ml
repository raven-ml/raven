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

let affix = function Some s -> s | None -> ""

type vocab = (string, int) Hashtbl.t
type merges = (string * string) list

(* Open-addressing hash table for merge lookups. Returns int directly (no option
   allocation). -1 = not found. *)
module Merge_map = struct
  type t = { keys : int array; values : int array; mask : int }

  let[@inline] hash key =
    let h = key * 0x1B873593 in
    h lxor (h lsr 16)

  (* A key given twice keeps the value it was given last. *)
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
        while
          Array.unsafe_get keys !h >= 0 && Array.unsafe_get keys !h <> key
        do
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

(* Pretoken cache.

   Two-way set-associative: the table is 64-byte sets of two 32-byte ways, a
   pretoken hashes to exactly one set and may sit in either way. A miss stores
   into way 0 after shifting way 0 into way 1, so the older of the two goes and
   no state bits are kept. Nothing is ever probed beyond the set or grown.

   Way 0 is at byte 0 of its set and way 1 at byte 32. Within a way:

   [0..7] k0: bytes 0..7 of the pretoken, zero-padded;

   [8..15] k1: bytes 8..14 zero-padded, the length 1..15 in the top byte;

   [16..23] ids 0 and 1 as 24-bit fields at bits 0 and 32, the token count in
   bits 24..31;

   [24..31] ids 2 and 3 in the same two fields.

   Every word is stored native-endian, so a table has one format wherever it is
   read. Carrying the length in the key means keys of different lengths never
   collide and a live key is never zero, so a zeroed table reads as empty. A hit
   is at most four 64-bit compares within one set and two 64-bit loads.

   A table is written by whichever thread holds the claim on the state that owns
   it, so there is never a second writer. The value words are still published
   before the key words, and [k0] last of all — for the way being shifted as for
   the way being filled — so that a read torn by anything at all comes out a
   miss rather than a wrong hit. *)

external string_get64u : string -> int -> int64 = "%caml_string_get64u"
external bytes_get64u : Bytes.t -> int -> int64 = "%caml_bytes_get64u"
external bytes_set64u : Bytes.t -> int -> int64 -> unit = "%caml_bytes_set64u"

let max_key_len = 15
let max_lanes = 4
let entry_bytes = 32
let set_bytes = 2 * entry_bytes
let cached_id_limit = 1 lsl 24

(* For each length, the mask keeping the bytes each key word holds. *)
let key_masks =
  let ones n =
    if n >= 8 then -1L else Int64.sub (Int64.shift_left 1L (8 * n)) 1L
  in
  let b = Bytes.make ((max_key_len + 1) * 16) '\000' in
  for len = 0 to max_key_len do
    bytes_set64u b (len * 16) (ones (min len 8));
    bytes_set64u b ((len * 16) + 8) (ones (max 0 (len - 8)))
  done;
  b

let[@inline] byte_word text pos count =
  let w = ref 0L in
  for i = 0 to count - 1 do
    w :=
      Int64.logor !w
        (Int64.shift_left
           (Int64.of_int (Char.code (String.unsafe_get text (pos + i))))
           (8 * i))
  done;
  !w

(* A key word is one eight-byte read masked to the bytes it keeps. The read has
   to stay inside the string — being unchecked is a property of native code
   alone, and bytecode raises on it — so a pretoken that ends within eight bytes
   of the end of the text is read from the last eight bytes of the text and
   shifted down instead. Only a text shorter than eight bytes has no such read,
   and gathers its bytes one at a time. [n] is the length of [text]. *)

let[@inline] key0 text n pos len =
  let w =
    if pos + 8 <= n then string_get64u text pos
    else if n >= 8 then
      Int64.shift_right_logical (string_get64u text (n - 8)) ((pos - n + 8) * 8)
    else byte_word text pos (n - pos)
  in
  Int64.logand w (bytes_get64u key_masks (len lsl 4))

let[@inline] key1 text n pos len =
  let tag = Int64.of_int (len lsl 56) in
  if len <= 8 then tag
  else
    let w =
      if pos + 16 <= n then string_get64u text (pos + 8)
      else
        Int64.shift_right_logical
          (string_get64u text (n - 8))
          ((pos + 16 - n) * 8)
    in
    Int64.logor tag (Int64.logand w (bytes_get64u key_masks ((len lsl 4) + 8)))

(* [mask] is the set index mask: the table holds [mask + 1] sets. *)
type cache = { table : Bytes.t; mask : int }

let no_cache = { table = Bytes.empty; mask = -1 }

(* The byte offset of the set a key belongs to. *)
let[@inline] set_of cache k0 k1 =
  let h =
    Int64.mul
      (Int64.logxor k0 (Int64.mul k1 0x9E3779B97F4A7C15L))
      0xD6E8FEB86659FD93L
  in
  (Int64.to_int (Int64.shift_right_logical h 24) land cache.mask) lsl 6

(* [entries] rounded up to a power of two, two per set; at least one set. *)
let create_cache entries =
  let sets = ref 1 in
  while !sets * 2 < entries do
    sets := !sets * 2
  done;
  { table = Bytes.make (!sets * set_bytes) '\000'; mask = !sets - 1 }

(* Pretokens above [max_key_len] bytes are keyed by their bytes; they are a
   percent or two of a corpus, so one substring per lookup is affordable. The
   table is dropped whole once it grows past [long_capacity] rather than
   evicting, and pretokens beyond [long_max_len] are not kept at all. *)
let long_capacity = 65536
let long_max_len = 4096

(* Merging needs a word, a queue, a rank scan and the caches to work in. They
   are held per (model, domain) and grow to the longest pretoken merged there;
   within a domain the first thread to claim them wins and the others work in
   buffers of their own. *)
type state = {
  mutable st_word : word;
  mutable st_queue : Merge_queue.t;
  st_rank : int array;
  st_cache : cache;
  st_long : (string, int array) Hashtbl.t;
  st_busy : bool Atomic.t;
  st_kernel : Kernel.byte_level;
  st_cursor : Bytes.t;
}

(* Vocabulary ids of single characters in one affixed form, keyed by the
   character alone: [ascii] indexes a single byte, [multi] the 1 to 4 bytes of a
   character packed into an int. *)
type char_table = { ascii : int array; multi : Merge_map.t }

type t = {
  stamp : int;
  vocab : vocab;
  vocab_r : string array;
  merges : Merge_map.t;
  cache_slots : int;
  (* Every vocabulary entry short enough to be keyed, tokenized once at creation
     and laid out as cache entries, so that each domain seeds its own table by
     replaying them instead of merging them again. *)
  seeds : Bytes.t;
  dropout : float option;
  unk_token : string option;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
  fuse_unk : bool;
  byte_fallback : bool;
  ignore_merges : bool;
  (* Byte-level models match raw bytes: [source] holds every entry decoded
     through the byte-to-unicode table, [source_vocab] inverts it and [byte_ids]
     gives the id of each single byte. The vocabulary itself keeps the encoded
     form, which is what serialization and decoding speak. *)
  byte_level : bool;
  source : string array;
  source_vocab : vocab;
  byte_ids : int array;
  (* How many bytes of a pretoken each id accounts for. *)
  len_table : int array;
  (* A character is looked up in the form its position gives it: the prefix on
     every character but the first, the suffix on the last. *)
  chars : char_table;
  prefixed_chars : char_table;
  suffixed_chars : char_table;
  prefixed_suffixed_chars : char_table;
  byte_fallback_ids : int array;
  unk_id : int;
  (* Byte fallback spells a character out in the form its position gives it, so
     one source byte can become as many as [1 + prefix + suffix] symbols and the
     merge buffers have to be sized for that rather than for the bytes. *)
  sym_per_byte : int;
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

(* Above this many symbols the linear rank scan is dropped for the heap. *)
let max_linear = 32

(* A domain holds the states of the models it has encoded with, most recent
   first, and forgets the oldest past this many: a state carries a cache of
   several megabytes. *)
let max_states = 8

let states_key : (int * state) list ref Domain.DLS.key =
  Domain.DLS.new_key (fun () -> ref [])

let[@inline] add_symbol word c byte_len =
  let s = word.size in
  let prev = if s > 0 then s - 1 else -1 in
  Array.unsafe_set word.sym_c s c;
  Array.unsafe_set word.sym_prev s prev;
  Array.unsafe_set word.sym_next s (-1);
  Array.unsafe_set word.sym_len s byte_len;
  if prev >= 0 then Array.unsafe_set word.sym_next prev s;
  word.size <- s + 1

(* Compact using linked-list traversal: O(N_final) instead of O(N_original). A
   word with no symbol has no list to walk: the links left in the reused buffers
   belong to the word merged before it. *)
let compact word =
  if word.size > 0 then begin
    let sym_c = word.sym_c in
    let sym_len = word.sym_len in
    let sym_next = word.sym_next in
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
  end

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
        if packed >= 0 && merge_new_id packed = new_id then
          begin if use_dropout && Random.float 1.0 < p then begin
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
  compact word

(* A word with few symbols keeps, for every position, the merge its pair opens —
   the packed (rank, id) the map gives it, [max_int] for none — and takes the
   minimum by scanning them. Rank leads the packed value so the scan orders by
   rank and, at equal rank, by position, which is the order HuggingFace merges
   in. A merge is O(1) surgery on the linked list and touches the ranks of the
   two neighbours alone, and a position the surgery drops is ranked [max_int] so
   it never wins again. The scan walks dead positions too: below [max_linear]
   symbols that is cheaper than the heap it replaces. *)
let[@inline] rank_of merges sym_c a b =
  let v =
    Merge_map.find merges
      (merge_key (Array.unsafe_get sym_c a) (Array.unsafe_get sym_c b))
  in
  if v >= 0 then v else max_int

let merge_linear model word rank =
  let merges = model.merges in
  let sym_c = word.sym_c in
  let sym_prev = word.sym_prev in
  let sym_next = word.sym_next in
  let sym_len = word.sym_len in
  let n = word.size in
  if n > 0 then begin
    for i = 0 to n - 2 do
      Array.unsafe_set rank i (rank_of merges sym_c i (i + 1))
    done;
    Array.unsafe_set rank (n - 1) max_int;
    let merging = ref true in
    while !merging do
      let best = ref max_int in
      let best_pos = ref (-1) in
      for i = 0 to n - 1 do
        if Array.unsafe_get rank i < !best then begin
          best := Array.unsafe_get rank i;
          best_pos := i
        end
      done;
      if !best_pos < 0 then merging := false
      else begin
        let pos = !best_pos in
        let gone = Array.unsafe_get sym_next pos in
        Array.unsafe_set sym_c pos (merge_new_id !best);
        Array.unsafe_set sym_len pos
          (Array.unsafe_get sym_len pos + Array.unsafe_get sym_len gone);
        Array.unsafe_set sym_next pos (Array.unsafe_get sym_next gone);
        Array.unsafe_set sym_len gone 0;
        Array.unsafe_set rank gone max_int;
        let next = Array.unsafe_get sym_next pos in
        if next >= 0 then begin
          Array.unsafe_set sym_prev next pos;
          Array.unsafe_set rank pos (rank_of merges sym_c pos next)
        end
        else Array.unsafe_set rank pos max_int;
        let prev = Array.unsafe_get sym_prev pos in
        if prev >= 0 then
          Array.unsafe_set rank prev (rank_of merges sym_c prev pos)
      end
    done
  end;
  compact word

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

let build_char_table vocab ~prefix ~suffix =
  let ascii = Array.make 128 (-1) in
  for i = 0 to 127 do
    match
      Hashtbl.find_opt vocab (prefix ^ String.make 1 (Char.chr i) ^ suffix)
    with
    | Some id -> Array.unsafe_set ascii i id
    | None -> ()
  done;
  let plen = String.length prefix in
  let slen = String.length suffix in
  let entries = ref [] in
  Hashtbl.iter
    (fun key id ->
      let char_len = String.length key - plen - slen in
      if
        char_len >= 1 && char_len <= 4
        && String.starts_with ~prefix key
        && String.ends_with ~suffix key
        && utf8_byte_len (Char.code (String.unsafe_get key plen)) = char_len
      then entries := (pack_char_key key plen char_len, id) :: !entries)
    vocab;
  { ascii; multi = Merge_map.create !entries }

(* Emits one byte fallback token for each of the [byte_len] bytes at [src]
   offset [offset], and is [false], having emitted nothing, if any of them has
   no fallback token. A pending unknown token stays pending across a fallback:
   HuggingFace lets it out at the next vocabulary hit, or at the end of the
   word. *)
let try_byte_fallback model word src offset byte_len =
  let all_found = ref true in
  for i = 0 to byte_len - 1 do
    if
      Array.unsafe_get model.byte_fallback_ids
        (Char.code (String.unsafe_get src (offset + i)))
      < 0
    then all_found := false
  done;
  if !all_found then begin
    for i = 0 to byte_len - 1 do
      add_symbol word
        (Array.unsafe_get model.byte_fallback_ids
           (Char.code (String.unsafe_get src (offset + i))))
        1
    done;
    true
  end
  else false

(* Byte-level models match raw bytes, so every symbol is one byte and the
   character tables never come into it. *)
let init_word_bytes model word text pos stop =
  let byte_ids = model.byte_ids in
  let fallback_ids = model.byte_fallback_ids in
  let unk_id = model.unk_id in
  let byte_fallback = model.byte_fallback in
  let fuse_unk = model.fuse_unk in
  let pending = ref 0 in
  for i = pos to stop - 1 do
    let b = Char.code (String.unsafe_get text i) in
    let id = Array.unsafe_get byte_ids b in
    if id >= 0 then begin
      if !pending > 0 then begin
        add_symbol word unk_id !pending;
        pending := 0
      end;
      add_symbol word id 1
    end
    else begin
      let fb = if byte_fallback then Array.unsafe_get fallback_ids b else -1 in
      if fb >= 0 then add_symbol word fb 1
      else if unk_id >= 0 then
        if fuse_unk then incr pending
        else begin
          if !pending > 0 then add_symbol word unk_id !pending;
          pending := 1
        end
    end
  done;
  if !pending > 0 then add_symbol word unk_id !pending

(* No prefix and no suffix: every character is looked up in the same table, so
   the position it holds in the word never has to be worked out.

   An unknown character is held back rather than emitted, because a byte
   fallback coming after it goes out first; the unknown token follows at the
   next vocabulary hit or at the end of the word. [pending] is the bytes it
   stands for so far, zero for none. Neither it nor the cursor is closed over,
   which is what lets the compiler keep them in registers. *)
let init_word_fast model word text pos stop =
  let ascii = model.chars.ascii in
  let multi = model.chars.multi in
  let unk_id = model.unk_id in
  let byte_fallback = model.byte_fallback in
  let fuse_unk = model.fuse_unk in
  let i = ref pos in
  let pending = ref 0 in
  while !i < stop do
    let b = Char.code (String.unsafe_get text !i) in
    (* A truncated character is taken as the bytes that are left of it: the key
       it packs is not the key of any whole character, so it falls through to
       the fallback or to the unknown token. *)
    let byte_len =
      if b < 128 then 1
      else
        let width = utf8_byte_len b in
        if width > stop - !i then stop - !i else width
    in
    let id =
      if b < 128 then Array.unsafe_get ascii b
      else Merge_map.find multi (pack_char_key text !i byte_len)
    in
    if id >= 0 then begin
      if !pending > 0 then begin
        add_symbol word unk_id !pending;
        pending := 0
      end;
      add_symbol word id byte_len
    end
    else if
      (not (byte_fallback && try_byte_fallback model word text !i byte_len))
      && unk_id >= 0
    then
      if fuse_unk then pending := !pending + byte_len
      else begin
        if !pending > 0 then add_symbol word unk_id !pending;
        pending := byte_len
      end;
    i := !i + byte_len
  done;
  if !pending > 0 then add_symbol word unk_id !pending

(* The character at [start] spelled the way the vocabulary holds it. *)
let affixed_char model text start byte_len ~is_first ~is_last =
  let prefix = if is_first then "" else affix model.continuing_subword_prefix in
  let suffix = if is_last then affix model.end_of_word_suffix else "" in
  prefix ^ String.sub text start byte_len ^ suffix

(* Models with continuing_subword_prefix or end_of_word_suffix *)
let init_word_slow model word text pos stop =
  let i = ref pos in
  let pending = ref 0 in
  let flush_unk () =
    if !pending > 0 then begin
      add_symbol word model.unk_id !pending;
      pending := 0
    end
  in
  let handle_unk byte_len =
    if model.unk_id >= 0 then
      if model.fuse_unk then pending := !pending + byte_len
      else begin
        if !pending > 0 then add_symbol word model.unk_id !pending;
        pending := byte_len
      end
  in
  while !i < stop do
    let b = Char.code (String.unsafe_get text !i) in
    if b land 0xC0 = 0x80 then incr i
    else begin
      let start = !i in
      let width = utf8_byte_len b in
      let byte_len = if width > stop - start then stop - start else width in
      let is_first = start = pos in
      let is_last = start + byte_len >= stop in
      i := start + byte_len;
      let table =
        if is_first then if is_last then model.suffixed_chars else model.chars
        else if is_last then model.prefixed_suffixed_chars
        else model.prefixed_chars
      in
      let id =
        if b < 128 then Array.unsafe_get table.ascii b
        else Merge_map.find table.multi (pack_char_key text start byte_len)
      in
      if id >= 0 then begin
        flush_unk ();
        add_symbol word id byte_len
      end
      else if model.byte_fallback then begin
        let s = affixed_char model text start byte_len ~is_first ~is_last in
        if not (try_byte_fallback model word s 0 (String.length s)) then
          handle_unk byte_len
      end
      else handle_unk byte_len
    end
  done;
  flush_unk ()

let[@inline] uses_dropout model =
  match model.dropout with Some p -> p > 0.0 | None -> false

(* Under [ignore_merges] a pretoken that is itself in the vocabulary is emitted
   as that single token. Otherwise the merges decide, and a vocabulary entry
   that no merge builds comes out as its decomposition. Dropout is drawn per
   occurrence, so it can take neither this shortcut nor the cache. *)
let whole_word_id model text pos len =
  if model.ignore_merges && not (uses_dropout model) then
    Hashtbl.find_opt model.source_vocab
      (if pos = 0 && len = String.length text then text
       else String.sub text pos len)
  else None

let run_merge model st text pos len =
  let word = ensure_word_capacity st.st_word (len * model.sym_per_byte) in
  st.st_word <- word;
  if model.byte_level then init_word_bytes model word text pos (pos + len)
  else if
    model.continuing_subword_prefix = None && model.end_of_word_suffix = None
  then init_word_fast model word text pos (pos + len)
  else init_word_slow model word text pos (pos + len);
  if word.size <= max_linear && not (uses_dropout model) then
    merge_linear model word st.st_rank
  else begin
    let queue = ensure_queue_capacity st.st_queue len in
    st.st_queue <- queue;
    apply_merges model model.dropout word queue
  end

(* Leaves the tokenization of [text.\[pos..pos+len)] in [st.st_word]. *)
let build_word model st text pos len =
  match whole_word_id model text pos len with
  | Some id ->
      let word = ensure_word_capacity st.st_word 1 in
      st.st_word <- word;
      add_symbol word id len
  | None -> run_merge model st text pos len

(* A cached word is handed back as ids alone and its token lengths are read off
   [len_table], so only a word whose every id accounts for exactly the bytes the
   merge gave it may be kept: an unknown token, a character no symbol stands
   for, or a byte fallback spelling out an affix all fail here and are merged
   again every time. *)
let exact model word len =
  let len_table = model.len_table in
  let limit = Array.length len_table in
  let ok = ref true in
  let total = ref 0 in
  for i = 0 to word.size - 1 do
    let id = Array.unsafe_get word.sym_c i in
    let l = Array.unsafe_get word.sym_len i in
    if id < 0 || id >= limit || Array.unsafe_get len_table id <> l then
      ok := false;
    total := !total + l
  done;
  !ok && !total = len

let in_lanes word =
  let n = word.size in
  if n < 1 || n > max_lanes then false
  else begin
    let ok = ref true in
    for i = 0 to n - 1 do
      if Array.unsafe_get word.sym_c i >= cached_id_limit then ok := false
    done;
    !ok
  end

let[@inline] cacheable model word len = in_lanes word && exact model word len

(* Writes the entry [(k0, k1, v0, v1)] into way 0 of the set at [set], after
   shifting what way 0 held into way 1. *)
let[@inline] store cache set k0 k1 v0 v1 =
  let table = cache.table in
  bytes_set64u table (set + 48) (bytes_get64u table (set + 16));
  bytes_set64u table (set + 56) (bytes_get64u table (set + 24));
  bytes_set64u table (set + 40) (bytes_get64u table (set + 8));
  bytes_set64u table (set + 32) (bytes_get64u table set);
  bytes_set64u table (set + 16) v0;
  bytes_set64u table (set + 24) v1;
  bytes_set64u table (set + 8) k1;
  bytes_set64u table set k0

(* The two value words of a cacheable word. *)
let[@inline] value_lo word =
  let n = word.size in
  let c = word.sym_c in
  let id1 = if n > 1 then Array.unsafe_get c 1 else 0 in
  Int64.of_int (Array.unsafe_get c 0 lor (n lsl 24) lor (id1 lsl 32))

let[@inline] value_hi word =
  let n = word.size in
  let c = word.sym_c in
  let id2 = if n > 2 then Array.unsafe_get c 2 else 0 in
  let id3 = if n > 3 then Array.unsafe_get c 3 else 0 in
  Int64.of_int (id2 lor (id3 lsl 32))

(* The tables the C kernel reads, in the field order it pins. A miss may be
   merged in C unless [ignore_merges] asks for the whole-word lookup, which
   needs [source_vocab]; dropout rules the kernel out at selection instead. *)
let kernel_tables model cache =
  {
    Kernel.lead = Pre_tokenizer.lead_class;
    cache = cache.table;
    cache_mask = cache.mask;
    byte_ids = model.byte_ids;
    merge_keys = model.merges.Merge_map.keys;
    merge_values = model.merges.Merge_map.values;
    merge_mask = model.merges.Merge_map.mask;
    len_table = model.len_table;
    merge = not model.ignore_merges;
  }

(* Buffers of its own and no cache, for the thread that does not claim the
   domain's state and for building the seed. *)
let private_state model =
  {
    st_word = create_word 16;
    st_queue = Merge_queue.create 16;
    st_rank = Array.make max_linear 0;
    st_cache = no_cache;
    st_long = Hashtbl.create 16;
    st_busy = Atomic.make true;
    st_kernel = kernel_tables model no_cache;
    st_cursor = Kernel.cursor ();
  }

(* Every vocabulary entry short enough to be keyed, tokenized by the very
   function a miss runs so that a seeded answer can never disagree with a merged
   one. The entry is seeded with its tokenization and not with its own id: a
   vocabulary holds entries no merge builds, and HuggingFace spells those
   out. *)
let build_seeds model =
  let st = private_state model in
  let buf = Buffer.create (entry_bytes * 1024) in
  Hashtbl.iter
    (fun text _ ->
      let len = String.length text in
      if len >= 1 && len <= max_key_len then begin
        build_word model st text 0 len;
        let word = st.st_word in
        if cacheable model word len then begin
          Buffer.add_int64_ne buf (key0 text len 0 len);
          Buffer.add_int64_ne buf (key1 text len 0 len);
          Buffer.add_int64_ne buf (value_lo word);
          Buffer.add_int64_ne buf (value_hi word)
        end
      end)
    model.source_vocab;
  Buffer.to_bytes buf

let seed_cache model cache =
  let seeds = model.seeds in
  for i = 0 to (Bytes.length seeds / entry_bytes) - 1 do
    let o = i * entry_bytes in
    let k0 = bytes_get64u seeds o in
    let k1 = bytes_get64u seeds (o + 8) in
    store cache (set_of cache k0 k1) k0 k1
      (bytes_get64u seeds (o + 16))
      (bytes_get64u seeds (o + 24))
  done

let make_state model =
  let cache =
    if model.cache_slots = 0 then no_cache else create_cache model.cache_slots
  in
  if cache.mask >= 0 then seed_cache model cache;
  {
    st_word = create_word 16;
    st_queue = Merge_queue.create 16;
    st_rank = Array.make max_linear 0;
    st_cache = cache;
    st_long = Hashtbl.create 512;
    st_busy = Atomic.make false;
    st_kernel = kernel_tables model cache;
    st_cursor = Kernel.cursor ();
  }

(* Two threads of one domain that both find no state for the model build one
   each and one of the two entries is lost; the loser is collected and its owner
   works on unshared buffers, so the race costs a table and nothing else. *)
let state model =
  let states = Domain.DLS.get states_key in
  let rec find = function
    | (stamp, st) :: _ when stamp = model.stamp -> st
    | _ :: rest -> find rest
    | [] ->
        let st = make_state model in
        states :=
          (model.stamp, st)
          :: List.filteri (fun i _ -> i < max_states - 1) !states;
        st
  in
  find !states

(* The domain's state, held for the duration of [f]. Threads of one domain share
   the state it holds — its word, queue and cache are all single-writer — so the
   second to ask gets buffers of its own instead. The claim is given back even
   when [f] raises: otherwise one exception would cost that domain its cache for
   the rest of the program. *)
let with_state model f =
  let st = state model in
  let st =
    if Atomic.compare_and_set st.st_busy false true then st
    else private_state model
  in
  match f st with
  | v ->
      Atomic.set st.st_busy false;
      v
  | exception e ->
      let backtrace = Printexc.get_raw_backtrace () in
      Atomic.set st.st_busy false;
      Printexc.raise_with_backtrace e backtrace

(* The ids of a merged word. A symbol whose bytes are not what its id accounts
   for — the unknown token, a fallback token the text spelled by name — is
   recorded with them, which is the same test the cache applies: a cached word
   is one that never needs recording. *)
let emit_word model word ids ~opaque =
  let len_table = model.len_table in
  let limit = Array.length len_table in
  for i = 0 to word.size - 1 do
    let id = Array.unsafe_get word.sym_c i in
    let l = Array.unsafe_get word.sym_len i in
    if id >= limit || Array.unsafe_get len_table id <> l then begin
      Ints.add opaque (Ints.length ids);
      Ints.add opaque 1;
      Ints.add opaque l
    end;
    Ints.add ids id
  done

let encode_into model st ids ~opaque text ~pos ~len =
  if len > 0 then begin
    let cache = st.st_cache in
    if cache.mask >= 0 && len <= max_key_len then begin
      let n = String.length text in
      let k0 = key0 text n pos len in
      let k1 = key1 text n pos len in
      let set = set_of cache k0 k1 in
      let table = cache.table in
      let way =
        if
          (bytes_get64u table set : int64) = k0
          && (bytes_get64u table (set + 8) : int64) = k1
        then set
        else if
          (bytes_get64u table (set + 32) : int64) = k0
          && (bytes_get64u table (set + 40) : int64) = k1
        then set + 32
        else -1
      in
      if way >= 0 then begin
        let v0 = Int64.to_int (bytes_get64u table (way + 16)) in
        let v1 = Int64.to_int (bytes_get64u table (way + 24)) in
        Ints.add4 ids (v0 land 0xFFFFFF)
          ((v0 lsr 32) land 0xFFFFFF)
          (v1 land 0xFFFFFF)
          ((v1 lsr 32) land 0xFFFFFF)
          ~count:((v0 lsr 24) land 0xFF)
      end
      else begin
        build_word model st text pos len;
        let word = st.st_word in
        emit_word model word ids ~opaque;
        if cacheable model word len then
          store cache set k0 k1 (value_lo word) (value_hi word)
      end
    end
    else if cache.mask >= 0 && len <= long_max_len then begin
      let long = st.st_long in
      let key =
        if pos = 0 && len = String.length text then text
        else String.sub text pos len
      in
      match Hashtbl.find_opt long key with
      | Some a ->
          for i = 0 to Array.length a - 1 do
            Ints.add ids (Array.unsafe_get a i)
          done
      | None ->
          build_word model st text pos len;
          let word = st.st_word in
          emit_word model word ids ~opaque;
          if exact model word len then begin
            if Hashtbl.length long >= long_capacity then Hashtbl.reset long;
            Hashtbl.replace long key (Array.sub word.sym_c 0 word.size)
          end
    end
    else begin
      build_word model st text pos len;
      emit_word model st.st_word ids ~opaque
    end
  end

let fused model =
  Kernel.available && model.byte_level && not (uses_dropout model)

let err_range = "range is not within the text"

(* The fused walk: the C kernel cuts spans, probes the cache and merges short
   misses, and everything it refuses comes back here with a resume position — an
   unclassified code point is classified and the call re-entered, a full ids
   buffer is grown, and an [Encode] span (over 15 bytes, a byte without an id,
   [ignore_merges], a merge result [emit_word] would record) goes through
   [encode_into], which owns the miss reference. The contract is
   {!Pre_tokenizer.fill}'s: [stop] once the range is exhausted, otherwise the
   start of the first span that did not fit, and a call that appends nothing and
   returns [pos] means the span buffer is too small. *)
let encode_walk model st ids ~opaque ~marks spans text ~pos ~stop =
  if pos < 0 || stop < pos || stop > String.length text || stop > 0xFFFF_FFFF
  then invalid_arg err_range;
  let cur = st.st_cursor and t = st.st_kernel in
  let rec go p =
    if p >= stop then stop
    else begin
      Ints.reserve ids (max 4096 (min (stop - p) 65536));
      Ints.reserve marks (Spans.capacity spans - Spans.count spans);
      Kernel.set cur ~spans:(Spans.count spans) ~ids:(Ints.length ids)
        ~marks:(Ints.length marks);
      let r =
        Kernel.byte_level_encode text p stop (Spans.buffer spans)
          (Ints.buffer ids) (Ints.buffer marks) cur
          (Char_class.unicode_table ())
          t
      in
      Spans.set_count spans (Kernel.spans cur);
      Ints.set_length ids (Kernel.ids cur);
      Ints.set_length marks (Kernel.marks cur);
      let resume = Kernel.resume cur in
      match r with
      | Kernel.Done | Kernel.Spans_full -> resume
      | Kernel.Ids_full ->
          if resume = p then Ints.reserve ids (2 * Ints.capacity ids);
          go resume
      | Kernel.Class ->
          ignore (Char_class.category (Kernel.code_point cur) : int);
          go resume
      | Kernel.Encode ->
          let k = Spans.count spans - 1 in
          let start = Spans.start spans k and finish = Spans.stop spans k in
          encode_into model st ids ~opaque text ~pos:start ~len:(finish - start);
          Ints.add marks (Ints.length ids);
          go finish
    end
  in
  go pos

let len_table model = model.len_table
let token_table model = model.vocab_r
let get_byte_level model = model.byte_level
let get_cache_capacity model = model.cache_slots
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
let get_dropout model = model.dropout
let get_fuse_unk model = model.fuse_unk
let get_byte_fallback model = model.byte_fallback
let get_ignore_merges model = model.ignore_merges

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
  let csp_str = affix continuing_subword_prefix in
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

let next_stamp = Atomic.make 0

(* How many bytes of a pretoken an id accounts for: the entry it stands for,
   stripped of the affixes its position gives it, or the byte a fallback token
   spells out. A symbol that stands for anything else — the unknown token for
   the run of bytes no entry covers, a fallback token merged from the text that
   spells its name — is recorded as it is emitted. *)
let build_len_table ~source ~byte_level ~prefix ~suffix ~byte_fallback
    ~byte_fallback_ids =
  let plen = String.length prefix in
  let slen = String.length suffix in
  let table =
    Array.map
      (fun s ->
        if byte_level then String.length s
        else
          let n = String.length s in
          let n =
            if plen > 0 && String.starts_with ~prefix s then n - plen else n
          in
          let n =
            if slen > 0 && String.ends_with ~suffix s then n - slen else n
          in
          max 0 n)
      source
  in
  if byte_fallback then
    Array.iter
      (fun id -> if id >= 0 && id < Array.length table then table.(id) <- 1)
      byte_fallback_ids;
  table

let create ~vocab ~merges ?(byte_level = false) ?(cache_capacity = 262144)
    ?dropout ?unk_token ?continuing_subword_prefix ?end_of_word_suffix
    ?(fuse_unk = false) ?(byte_fallback = false) ?(ignore_merges = false) () : t
    =
  (* Tokenizer files spell "no prefix" and "no suffix" as [""]. *)
  let non_empty = function Some "" -> None | given -> given in
  let continuing_subword_prefix = non_empty continuing_subword_prefix in
  let end_of_word_suffix = non_empty end_of_word_suffix in
  if
    byte_level
    && (continuing_subword_prefix <> None || end_of_word_suffix <> None)
  then
    invalid_arg
      "Bpe.create: byte_level is incompatible with continuing_subword_prefix \
       and end_of_word_suffix";
  let max_id = Hashtbl.fold (fun _ id acc -> max id acc) vocab (-1) in
  let vocab_r = Array.make (max_id + 1) "" in
  Hashtbl.iter (fun k v -> Array.unsafe_set vocab_r v k) vocab;
  let merges =
    convert_merges_to_merge_map vocab merges continuing_subword_prefix
  in
  let byte_fallback_ids = Array.make 256 (-1) in
  for i = 0 to 255 do
    let hex = Printf.sprintf "<0x%02X>" i in
    match Hashtbl.find_opt vocab hex with
    | Some id -> byte_fallback_ids.(i) <- id
    | None -> ()
  done;
  let prefix = affix continuing_subword_prefix in
  let suffix = affix end_of_word_suffix in
  let chars = build_char_table vocab ~prefix:"" ~suffix:"" in
  let prefixed_chars =
    if prefix = "" then chars else build_char_table vocab ~prefix ~suffix:""
  in
  let suffixed_chars =
    if suffix = "" then chars else build_char_table vocab ~prefix:"" ~suffix
  in
  let prefixed_suffixed_chars =
    if prefix = "" then suffixed_chars
    else if suffix = "" then prefixed_chars
    else build_char_table vocab ~prefix ~suffix
  in
  let unk_id =
    match unk_token with
    | Some unk -> (
        match Hashtbl.find_opt vocab unk with Some id -> id | None -> -1)
    | None -> -1
  in
  (* The source form of an entry is what a pretoken is matched against: the
     entry itself, or, for a byte-level model, the bytes it stands for. Ids
     ascend so the lowest one wins whenever two entries share a source. *)
  let source =
    if byte_level then Array.map Pre_tokenizer.byte_level_decode vocab_r
    else vocab_r
  in
  let source_vocab =
    if not byte_level then vocab
    else begin
      let tbl = Hashtbl.create (Array.length source) in
      Array.iteri
        (fun id s ->
          if s <> "" && not (Hashtbl.mem tbl s) then Hashtbl.add tbl s id)
        source;
      tbl
    end
  in
  let byte_ids = Array.make 256 (-1) in
  if byte_level then
    Array.iteri
      (fun id s ->
        if String.length s = 1 then begin
          let b = Char.code (String.unsafe_get s 0) in
          if Array.unsafe_get byte_ids b < 0 then Array.unsafe_set byte_ids b id
        end)
      source;
  let model =
    {
      stamp = Atomic.fetch_and_add next_stamp 1;
      vocab;
      vocab_r;
      merges;
      (* Dropout is drawn per occurrence, so a dropped-out model can reuse
         nothing and keeps no cache at all. *)
      cache_slots =
        (match dropout with
        | Some p when p > 0.0 -> 0
        | _ -> max 0 cache_capacity);
      seeds = Bytes.empty;
      dropout;
      unk_token;
      continuing_subword_prefix;
      end_of_word_suffix;
      fuse_unk;
      byte_fallback;
      ignore_merges;
      byte_level;
      source;
      source_vocab;
      byte_ids;
      len_table =
        build_len_table ~source ~byte_level ~prefix ~suffix ~byte_fallback
          ~byte_fallback_ids;
      chars;
      prefixed_chars;
      suffixed_chars;
      prefixed_suffixed_chars;
      byte_fallback_ids;
      unk_id;
      sym_per_byte =
        (if byte_fallback && not byte_level then
           1 + String.length prefix + String.length suffix
         else 1);
    }
  in
  (* A vocabulary entry carries the affixes its position gave it — [low</w>],
     [##ing] — and no pretoken ever looks like that, so seeding an affixed model
     would fill the table with entries nothing can hit. *)
  if model.cache_slots > 0 && prefix = "" && suffix = "" then
    { model with seeds = build_seeds model }
  else model

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

let from_files ?byte_level ~vocab_file ~merges_file () =
  let vocab, merges = read_files ~vocab_file ~merges_file in
  create ?byte_level ~vocab ~merges ()

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

(* Training *)

(* A pair of symbol ids packed into one int, which orders pairs the way ties
   between equally frequent pairs are settled: by left id, then by right. *)
let[@inline] pair_key a b = (a lsl 31) lor b
let[@inline] pair_left key = key lsr 31
let[@inline] pair_right key = key land 0x7FFFFFFF

(* Merge candidates, most frequent first, ties going to the lowest pair. An
   entry holds the count its pair had when it was pushed, and the words the pair
   turned up in since the pair was last queued; a count that moved since is
   corrected when the entry comes out. *)
module Pair_queue = struct
  type t = {
    mutable counts : int array;
    mutable keys : int array;
    mutable words : int list array;
    mutable size : int;
  }

  let create () =
    {
      counts = Array.make 1024 0;
      keys = Array.make 1024 0;
      words = Array.make 1024 [];
      size = 0;
    }

  let[@inline] precedes c1 k1 c2 k2 = c1 > c2 || (c1 = c2 && k1 < k2)

  let push t count key words =
    let s = t.size in
    if s = Array.length t.counts then begin
      let cap = s * 2 in
      let grow a =
        let b = Array.make cap 0 in
        Array.blit a 0 b 0 s;
        b
      in
      t.counts <- grow t.counts;
      t.keys <- grow t.keys;
      let b = Array.make cap [] in
      Array.blit t.words 0 b 0 s;
      t.words <- b
    end;
    let counts = t.counts and keys = t.keys and where = t.words in
    let i = ref s in
    let cont = ref (s > 0) in
    while !cont do
      let p = (!i - 1) / 2 in
      if
        precedes count key (Array.unsafe_get counts p) (Array.unsafe_get keys p)
      then begin
        Array.unsafe_set counts !i (Array.unsafe_get counts p);
        Array.unsafe_set keys !i (Array.unsafe_get keys p);
        Array.unsafe_set where !i (Array.unsafe_get where p);
        i := p;
        cont := !i > 0
      end
      else cont := false
    done;
    Array.unsafe_set counts !i count;
    Array.unsafe_set keys !i key;
    Array.unsafe_set where !i words;
    t.size <- s + 1

  let pop t =
    if t.size = 0 then None
    else begin
      let counts = t.counts and keys = t.keys and where = t.words in
      let top =
        ( Array.unsafe_get counts 0,
          Array.unsafe_get keys 0,
          Array.unsafe_get where 0 )
      in
      let size = t.size - 1 in
      t.size <- size;
      if size > 0 then begin
        let count = Array.unsafe_get counts size in
        let key = Array.unsafe_get keys size in
        let words = Array.unsafe_get where size in
        Array.unsafe_set where size [];
        let i = ref 0 in
        let cont = ref true in
        while !cont do
          let l = (2 * !i) + 1 in
          if l >= size then cont := false
          else begin
            let r = l + 1 in
            let c =
              if
                r < size
                && precedes
                     (Array.unsafe_get counts r)
                     (Array.unsafe_get keys r)
                     (Array.unsafe_get counts l)
                     (Array.unsafe_get keys l)
              then r
              else l
            in
            if
              precedes
                (Array.unsafe_get counts c)
                (Array.unsafe_get keys c) count key
            then begin
              Array.unsafe_set counts !i (Array.unsafe_get counts c);
              Array.unsafe_set keys !i (Array.unsafe_get keys c);
              Array.unsafe_set where !i (Array.unsafe_get where c);
              i := c
            end
            else cont := false
          end
        done;
        Array.unsafe_set counts !i count;
        Array.unsafe_set keys !i key;
        Array.unsafe_set where !i words
      end
      else Array.unsafe_set where 0 [];
      Some top
    end
end

(* What the trainer knows about a pair: how often it occurs, the words it has
   turned up in since it was last queued, and whether the count has risen since
   then, which is what earns it a fresh queue entry. *)
type pair_state = {
  mutable pair_count : int;
  mutable pair_words : int list;
  mutable pair_raised : bool;
}

(* Calls [f] on each character of [word] with the place it holds in it, which is
   what decides the affixes it carries. *)
let iter_chars word f =
  let len = String.length word in
  let pos = ref 0 in
  while !pos < len do
    let d = String.get_utf_8_uchar word !pos in
    let n = Uchar.utf_decode_length d in
    if Uchar.utf_decode_is_valid d then
      f (String.sub word !pos n) ~is_first:(!pos = 0) ~is_last:(!pos + n >= len);
    pos := !pos + n
  done

let train ~min_frequency ~vocab_size ~show_progress ~special_tokens
    ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
    ~end_of_word_suffix ~max_token_length word_counts =
  let _ = show_progress in
  let vocab = Hashtbl.create 10000 in
  let tokens = Dynarray.create () in
  let intern token =
    match Hashtbl.find_opt vocab token with
    | Some id -> id
    | None ->
        let id = Dynarray.length tokens in
        Hashtbl.add vocab token id;
        Dynarray.add_last tokens token;
        id
  in
  List.iter (fun token -> ignore (intern token)) special_tokens;
  let alphabet = Hashtbl.create 1024 in
  List.iter
    (fun (word, count) ->
      iter_chars word (fun c ~is_first:_ ~is_last:_ ->
          Hashtbl.replace alphabet c
            (count + try Hashtbl.find alphabet c with Not_found -> 0)))
    word_counts;
  List.iter (fun c -> Hashtbl.replace alphabet c max_int) initial_alphabet;
  let kept =
    Hashtbl.fold (fun c count acc -> (c, count) :: acc) alphabet []
    |> List.sort (fun (c1, n1) (c2, n2) ->
        if n1 <> n2 then compare n1 n2 else String.compare c1 c2)
  in
  let to_remove =
    match limit_alphabet with
    | Some limit -> max 0 (List.length kept - limit)
    | None -> 0
  in
  List.iter
    (fun (c, _) -> ignore (intern c))
    (list_drop to_remove kept
    |> List.sort (fun (c1, _) (c2, _) -> String.compare c1 c2));
  (* Words are laid out as arrays of vocabulary ids, each character in the form
     its place gives it. A character the alphabet limit left out is dropped from
     the word rather than merged. *)
  let prefix = affix continuing_subword_prefix in
  let suffix = affix end_of_word_suffix in
  let corpus =
    List.sort (fun (w1, _) (w2, _) -> String.compare w1 w2) word_counts
  in
  let word_total = List.length corpus in
  let words = Array.make word_total [||] in
  (* How many characters of the word each symbol spans, which is what
     [max_token_length] caps. It belongs to the occurrence, not to the token: a
     merge can arrive at a token a character is already spelled as, and the two
     span a different number of characters. *)
  let lengths = Array.make word_total [||] in
  let word_size = Array.make word_total 0 in
  let counts = Array.make word_total 0 in
  let symbols = ref (Array.make 64 0) in
  List.iteri
    (fun i (word, count) ->
      if Array.length !symbols < String.length word then
        symbols := Array.make (String.length word) 0;
      let n = ref 0 in
      iter_chars word (fun c ~is_first ~is_last ->
          if Hashtbl.mem vocab c then begin
            let token =
              (if is_first then "" else prefix)
              ^ c
              ^ if is_last then suffix else ""
            in
            !symbols.(!n) <- intern token;
            incr n
          end);
      words.(i) <- Array.sub !symbols 0 !n;
      lengths.(i) <- Array.make !n 1;
      word_size.(i) <- !n;
      counts.(i) <- count)
    corpus;
  let max_chars = match max_token_length with Some l -> l | None -> max_int in
  let pairs = Hashtbl.create 10000 in
  let queue = Pair_queue.create () in
  let raised = ref [] in
  let state key =
    match Hashtbl.find_opt pairs key with
    | Some s -> s
    | None ->
        let s = { pair_count = 0; pair_words = []; pair_raised = false } in
        Hashtbl.add pairs key s;
        s
  in
  let record s word =
    match s.pair_words with
    | w :: _ when w = word -> ()
    | l -> s.pair_words <- word :: l
  in
  let drop_pair a b count =
    let s = state (pair_key a b) in
    s.pair_count <- s.pair_count - count
  in
  (* A pair only counts while the run of characters it would join stays under
     [max_token_length], so an over-long merge never becomes a candidate. The
     single characters counted below are the exception: they are what the first
     merges are drawn from. *)
  let bump_pair a b chars word count =
    if chars < max_chars then begin
      let key = pair_key a b in
      let s = state key in
      s.pair_count <- s.pair_count + count;
      record s word;
      if not s.pair_raised then begin
        s.pair_raised <- true;
        raised := key :: !raised
      end
    end
  in
  for i = 0 to word_total - 1 do
    let syms = words.(i) in
    let count = counts.(i) in
    for j = 0 to word_size.(i) - 2 do
      let s =
        state
          (pair_key (Array.unsafe_get syms j) (Array.unsafe_get syms (j + 1)))
      in
      s.pair_count <- s.pair_count + count;
      record s i
    done
  done;
  Hashtbl.iter
    (fun key s ->
      Pair_queue.push queue s.pair_count key s.pair_words;
      s.pair_words <- [])
    pairs;
  let plen = String.length prefix in
  let merges = ref [] in
  let stop = ref false in
  while (not !stop) && Dynarray.length tokens < vocab_size do
    match Pair_queue.pop queue with
    | None -> stop := true
    | Some (count, key, where) ->
        let s = Hashtbl.find pairs key in
        if s.pair_count <> count then
          begin if s.pair_count > 0 then
            Pair_queue.push queue s.pair_count key where
          end
        else if count < 1 || count < min_frequency then stop := true
        else begin
          let a = pair_left key and b = pair_right key in
          let left = Dynarray.get tokens a and right = Dynarray.get tokens b in
          let tail =
            if plen > 0 && String.starts_with ~prefix right then
              String.sub right plen (String.length right - plen)
            else right
          in
          let new_id = intern (left ^ tail) in
          merges := (left, right) :: !merges;
          List.iter
            (fun i ->
              let syms = words.(i) in
              let lens = lengths.(i) in
              let n = word_size.(i) in
              let count = counts.(i) in
              let j = ref 0 in
              let w = ref 0 in
              while !j < n do
                if
                  !j + 1 < n
                  && Array.unsafe_get syms !j = a
                  && Array.unsafe_get syms (!j + 1) = b
                then begin
                  let chars =
                    Array.unsafe_get lens !j + Array.unsafe_get lens (!j + 1)
                  in
                  if !w > 0 then begin
                    let prev = Array.unsafe_get syms (!w - 1) in
                    drop_pair prev a count;
                    bump_pair prev new_id
                      (Array.unsafe_get lens (!w - 1) + chars)
                      i count
                  end;
                  if !j + 2 < n then begin
                    let next = Array.unsafe_get syms (!j + 2) in
                    drop_pair b next count;
                    bump_pair new_id next
                      (chars + Array.unsafe_get lens (!j + 2))
                      i count
                  end;
                  Array.unsafe_set syms !w new_id;
                  Array.unsafe_set lens !w chars;
                  incr w;
                  j := !j + 2
                end
                else begin
                  Array.unsafe_set syms !w (Array.unsafe_get syms !j);
                  Array.unsafe_set lens !w (Array.unsafe_get lens !j);
                  incr w;
                  incr j
                end
              done;
              word_size.(i) <- !w)
            where;
          (* The pair keeps whatever occurrences this entry did not carry, and a
             later merge can put the same two symbols side by side again, so a
             pair can be merged more than once. The merge map settles a repeat
             by keeping the rank the pair was given last. *)
          List.iter
            (fun key ->
              let s = Hashtbl.find pairs key in
              s.pair_raised <- false;
              if s.pair_count > 0 then
                Pair_queue.push queue s.pair_count key s.pair_words;
              s.pair_words <- [])
            !raised;
          raised := []
        end
  done;
  let trained_model =
    create ~vocab ~merges:(List.rev !merges) ?continuing_subword_prefix
      ?end_of_word_suffix ()
  in
  (trained_model, special_tokens)
