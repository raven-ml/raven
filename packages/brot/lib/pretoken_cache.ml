(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The layout is {!Pretoken_cache}'s. Carrying the length in the key means
   keys of different lengths never collide and a live key is never zero, so a
   zeroed table reads as empty. A hit is at most four 64-bit compares within
   one set and two 64-bit loads.

   No probe races a store: a table has one claimant at a time, and that rule,
   not the order of the writes, keeps it safe. [store] and [promote] still
   publish values before keys and [k0] last of all, which narrows what a reader
   outside the claim could misread, but does not protect one — values are
   overwritten while the old keys still stand. *)

external string_get64u : string -> int -> int64 = "%caml_string_get64u"
external bytes_get64u : Bytes.t -> int -> int64 = "%caml_bytes_get64u"
external bytes_set64u : Bytes.t -> int -> int64 -> unit = "%caml_bytes_set64u"
external swap64 : int64 -> int64 = "%bswap_int64"

let max_key_len = 15
let max_lanes = 4
let entry_bytes = 32
let set_bytes = 2 * entry_bytes
let id_limit = 1 lsl 24

(* Eight bytes of [text] in little-endian order, so that a key holds the same
   bytes, first byte lowest, on every machine. *)
let[@inline] load text pos =
  let w = string_get64u text pos in
  if Sys.big_endian then swap64 w else w

(* The front table's entry count (128 KB at 32 bytes each). Measured on a 290 MB
   OpenWebText sample with the GPT-2 vocabulary: the 4096 most frequent
   pretokens cover 82 % of short-pretoken occurrences, and a direct-mapped table
   this size answers 75 % of probes — 79 % of all cache hits — with 2^14 entries
   adding only 8 points for four times the footprint. *)
let front_entries = 4096

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
    if pos + 8 <= n then load text pos
    else if n >= 8 then
      Int64.shift_right_logical (load text (n - 8)) ((pos - n + 8) * 8)
    else byte_word text pos (n - pos)
  in
  Int64.logand w (bytes_get64u key_masks (len lsl 4))

let[@inline] key1 text n pos len =
  let tag = Int64.of_int (len lsl 56) in
  if len <= 8 then tag
  else
    let w =
      if pos + 16 <= n then load text (pos + 8)
      else
        Int64.shift_right_logical
          (load text (n - 8))
          ((pos + 16 - n) * 8)
    in
    Int64.logor tag (Int64.logand w (bytes_get64u key_masks ((len lsl 4) + 8)))

(* The back table starts at this many sets, 16 KB, and doubles once half its
   entries are live, up to its limit: a two-way table fuller than that evicts
   entries it could have kept. A store that evicts
   leaves the count as it was, so a working set the table holds, but whose
   entries collide, does not grow it. *)
let initial_sets = 256
let default_capacity = 262144

type t = {
  front : Bytes.t;
  fmask : int;
  mutable table : Bytes.t;
  mutable mask : int;
  limit : int;
  mutable live : int;
}

let none =
  {
    front = Bytes.empty;
    fmask = -1;
    table = Bytes.empty;
    mask = -1;
    limit = 0;
    live = 0;
  }

(* A product only carries upward, and the index is the low bits of
   [product lsr 24]: without the fold, keys that agree in their first five
   bytes, [▁the] and [▁their], share a set and a front slot. The fold brings
   the high half of the key down first. *)
let[@inline] hash_of k0 k1 =
  let x = Int64.logxor k0 (Int64.mul k1 0x9E3779B97F4A7C15L) in
  let x = Int64.logxor x (Int64.shift_right_logical x 32) in
  Int64.to_int (Int64.shift_right_logical (Int64.mul x 0xD6E8FEB86659FD93L) 24)

(* The byte offsets of a hash's back set and front slot. *)
let[@inline] set_at h mask = (h land mask) lsl 6
let[@inline] front_at h fmask = (h land fmask) lsl 5

(* [entries] rounded up to a power of two, two per set; at least one set. *)
let sets_for entries =
  let sets = ref 1 in
  while !sets * 2 < entries do
    sets := !sets * 2
  done;
  !sets

let create ?(grow = true) capacity =
  if capacity <= 0 then { none with live = 0 }
  else
    let limit = sets_for capacity in
    let sets = if grow then min limit initial_sets else limit in
    {
      front = Bytes.make (front_entries * entry_bytes) '\000';
      fmask = front_entries - 1;
      table = Bytes.make (sets * set_bytes) '\000';
      mask = sets - 1;
      limit;
      live = 0;
    }

(* Writes the entry [(k0, k1, v0, v1)] into way 0 of the set at [set], after
   shifting what way 0 held into way 1. *)
let[@inline] store_at table set k0 k1 v0 v1 =
  bytes_set64u table (set + 48) (bytes_get64u table (set + 16));
  bytes_set64u table (set + 56) (bytes_get64u table (set + 24));
  bytes_set64u table (set + 40) (bytes_get64u table (set + 8));
  bytes_set64u table (set + 32) (bytes_get64u table set);
  bytes_set64u table (set + 16) v0;
  bytes_set64u table (set + 24) v1;
  bytes_set64u table (set + 8) k1;
  bytes_set64u table set k0

(* Writes the entry into the front slot at [fslot], in [store_at]'s order. *)
let[@inline] promote t fslot k0 k1 v0 v1 =
  let front = t.front in
  bytes_set64u front (fslot + 16) v0;
  bytes_set64u front (fslot + 24) v1;
  bytes_set64u front (fslot + 8) k1;
  bytes_set64u front fslot k0

(* The back table at twice its sets, every entry moved to its new set, the
   older way first so that each set keeps its order. *)
let[@inline never] grow t =
  let old = t.table and sets = 2 * (t.mask + 1) in
  let table = Bytes.make (sets * set_bytes) '\000' in
  let mask = sets - 1 in
  for way = 1 downto 0 do
    for set = 0 to t.mask do
      let o = (set lsl 6) + (way * entry_bytes) in
      let k1 = bytes_get64u old (o + 8) in
      if k1 <> 0L then begin
        let k0 = bytes_get64u old o in
        store_at table
          (set_at (hash_of k0 k1) mask)
          k0 k1
          (bytes_get64u old (o + 16))
          (bytes_get64u old (o + 24))
      end
    done
  done;
  t.table <- table;
  t.mask <- mask

(* Inlined, so that the value words reach the table unboxed. *)
let[@inline] store t k0 k1 v0 v1 =
  if t.mask < t.limit - 1 && t.live > t.mask then grow t;
  let set = set_at (hash_of k0 k1) t.mask in
  if bytes_get64u t.table (set + 40) = 0L then t.live <- t.live + 1;
  store_at t.table set k0 k1 v0 v1

(* Value words: two 24-bit ids each, at bits 0 and 32, the id count in bits 24
   to 31 of the first. Built and read with [Int64] operations, so that they are
   right where an [int] has 31 or 32 bits. *)

let[@inline] lane id shift = Int64.shift_left (Int64.of_int id) shift

let[@inline] value_lo ids first count =
  let id1 = if count > 1 then Ints.get ids (first + 1) else 0 in
  Int64.logor
    (Int64.logor (lane (Ints.get ids first) 0) (lane count 24))
    (lane id1 32)

let[@inline] value_hi ids first count =
  let id2 = if count > 2 then Ints.get ids (first + 2) else 0 in
  let id3 = if count > 3 then Ints.get ids (first + 3) else 0 in
  Int64.logor (lane id2 0) (lane id3 32)

let[@inline] field v shift bits =
  Int64.to_int
    (Int64.logand (Int64.shift_right_logical v shift)
       (Int64.sub (Int64.shift_left 1L bits) 1L))

let[@inline] add_lanes ids v0 v1 =
  Ints.add4 ids (field v0 0 24) (field v0 32 24) (field v1 0 24)
    (field v1 32 24) ~count:(field v0 24 8)

(* Whether the [count] ids of [ids] from [first] can be handed back from the
   cache: they fit its lanes, and each accounts for exactly the bytes its
   entry in [len_table] says, which is what the offsets of a cached word are
   read off. *)
let keepable ~len_table ids first count len =
  count >= 1 && count <= max_lanes
  &&
  let limit = Array.length len_table in
  let total = ref 0 and ok = ref true in
  for k = first to first + count - 1 do
    let id = Ints.get ids k in
    if id < 0 || id >= id_limit || id >= limit then ok := false
    else total := !total + Array.unsafe_get len_table id
  done;
  !ok && !total = len

type encoder = Ints.t -> opaque:Ints.t -> string -> pos:int -> len:int -> unit

let encode t ~len_table (miss : encoder) ids ~opaque text ~pos ~len =
  if t.mask < 0 || len < 1 || len > max_key_len then
    miss ids ~opaque text ~pos ~len
  else begin
    let n = String.length text in
    let k0 = key0 text n pos len in
    let k1 = key1 text n pos len in
    let h = hash_of k0 k1 in
    let fslot = front_at h t.fmask in
    let front = t.front in
    if
      (bytes_get64u front fslot : int64) = k0
      && (bytes_get64u front (fslot + 8) : int64) = k1
    then
      add_lanes ids
        (bytes_get64u front (fslot + 16))
        (bytes_get64u front (fslot + 24))
    else begin
      let set = set_at h t.mask in
      let table = t.table in
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
        let v0 = bytes_get64u table (way + 16) in
        let v1 = bytes_get64u table (way + 24) in
        promote t fslot k0 k1 v0 v1;
        add_lanes ids v0 v1
      end
      else begin
        let first = Ints.length ids and marks = Ints.length opaque in
        miss ids ~opaque text ~pos ~len;
        let count = Ints.length ids - first in
        if
          Ints.length opaque = marks
          && keepable ~len_table ids first count len
        then
          store t k0 k1 (value_lo ids first count) (value_hi ids first count)
      end
    end
  end

let cached t ~len_table miss : encoder =
  if t.mask < 0 then miss
  else fun ids ~opaque text ~pos ~len ->
    encode t ~len_table miss ids ~opaque text ~pos ~len

(* Seeds *)

let set_index key ~sets =
  let n = String.length key in
  hash_of (key0 key n 0 n) (key1 key n 0 n) land (sets - 1)

let seed buf text ids count =
  let len = String.length text in
  let fits = ref (count >= 1 && count <= max_lanes) in
  for k = 0 to count - 1 do
    if ids.(k) < 0 || ids.(k) >= id_limit then fits := false
  done;
  if len >= 1 && len <= max_key_len && !fits then begin
    let id k = if k < count then ids.(k) else 0 in
    Buffer.add_int64_ne buf (key0 text len 0 len);
    Buffer.add_int64_ne buf (key1 text len 0 len);
    Buffer.add_int64_ne buf
      (Int64.logor (Int64.logor (lane (id 0) 0) (lane count 24)) (lane (id 1) 32));
    Buffer.add_int64_ne buf (Int64.logor (lane (id 2) 0) (lane (id 3) 32))
  end

let load_seeds t seeds =
  if t.mask >= 0 then
    for i = 0 to (Bytes.length seeds / entry_bytes) - 1 do
      let o = i * entry_bytes in
      store t (bytes_get64u seeds o)
        (bytes_get64u seeds (o + 8))
        (bytes_get64u seeds (o + 16))
        (bytes_get64u seeds (o + 24))
    done

(* Claims *)

let next_stamp = Atomic.make 0
let stamp () = Atomic.fetch_and_add next_stamp 1

type 'a slot = { value : 'a; busy : bool Atomic.t }
type 'a registry = (int * 'a slot) list ref Domain.DLS.key

(* A domain keeps the slots of the last [max_slots] models it began encoding
   with, newest first, and drops the oldest of them past that: a slot can hold
   megabytes. *)
let max_slots = 8

let registry () = Domain.DLS.new_key (fun () -> ref [])

let rec find stamp = function
  | (s, slot) :: _ when s = stamp -> slot
  | _ :: rest -> find stamp rest
  | [] -> raise_notrace Not_found

let slot registry ~stamp make arg =
  let slots = Domain.DLS.get registry in
  match find stamp !slots with
  | slot -> slot
  | exception Not_found ->
      let slot = { value = make arg; busy = Atomic.make false } in
      slots :=
        (stamp, slot) :: List.filteri (fun i _ -> i < max_slots - 1) !slots;
      slot

let take slot = Atomic.compare_and_set slot.busy false true
let give slot = Atomic.set slot.busy false
