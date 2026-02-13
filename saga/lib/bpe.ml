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

module StringMap = Map.Make (String)

type vocab = (string, int) Hashtbl.t
type vocab_r = (int, string) Hashtbl.t
type merges = (string * string) list
(* Open-addressing hash table for merge lookups.
   Returns int directly (no option allocation). -1 = not found. *)
module MergeMap = struct
  type t = {
    keys : int array;
    values : int array;
    mask : int;
  }

  let[@inline] hash key =
    let h = key * 0x517CC1B727220A95 in
    h lxor (h lsr 29)

  let create entries =
    let n = List.length entries in
    let cap = ref 16 in
    while !cap < n * 2 do
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

type merge_map = MergeMap.t

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
  ascii_to_id : int array;
  byte_fallback_ids : int array;
  char_to_id : (int, int) Hashtbl.t;
  unk_id : int;
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

let[@inline] add_symbol word c byte_len =
  let s = word.size in
  let prev = if s > 0 then s - 1 else -1 in
  Array.unsafe_set word.sym_c s c;
  Array.unsafe_set word.sym_prev s prev;
  Array.unsafe_set word.sym_next s (-1);
  Array.unsafe_set word.sym_len s byte_len;
  if prev >= 0 then Array.unsafe_set word.sym_next prev s;
  word.size <- s + 1

(* Specialized min-heap for BPE merges using parallel arrays (no tuple allocation).
   Ordered by (rank, position) — lower rank first, then lower position. *)
module MergeQueue = struct
  type t = {
    mutable ranks : int array;
    mutable positions : int array;
    mutable new_ids : int array;
    mutable size : int;
    mutable pop_rank : int;
    mutable pop_pos : int;
    mutable pop_new_id : int;
  }

  let create cap =
    let cap = max 16 cap in
    {
      ranks = Array.make cap 0;
      positions = Array.make cap 0;
      new_ids = Array.make cap 0;
      size = 0;
      pop_rank = 0;
      pop_pos = 0;
      pop_new_id = 0;
    }

  let[@inline] cmp_lt_rv t i rank pos =
    let ri = Array.unsafe_get t.ranks i in
    ri < rank || (ri = rank && Array.unsafe_get t.positions i < pos)

  let[@inline] cmp_lt t i j =
    let ri = Array.unsafe_get t.ranks i in
    let rj = Array.unsafe_get t.ranks j in
    ri < rj || (ri = rj && Array.unsafe_get t.positions i < Array.unsafe_get t.positions j)

  let sift_up t idx =
    let i = ref idx in
    let rank = Array.unsafe_get t.ranks idx in
    let pos = Array.unsafe_get t.positions idx in
    let nid = Array.unsafe_get t.new_ids idx in
    while !i > 0 && (let p = (!i - 1) asr 1 in
                      let rp = Array.unsafe_get t.ranks p in
                      rank < rp || (rank = rp && pos < Array.unsafe_get t.positions p))
    do
      let p = (!i - 1) asr 1 in
      Array.unsafe_set t.ranks !i (Array.unsafe_get t.ranks p);
      Array.unsafe_set t.positions !i (Array.unsafe_get t.positions p);
      Array.unsafe_set t.new_ids !i (Array.unsafe_get t.new_ids p);
      i := p
    done;
    Array.unsafe_set t.ranks !i rank;
    Array.unsafe_set t.positions !i pos;
    Array.unsafe_set t.new_ids !i nid

  let sift_down t idx =
    let i = ref idx in
    let rank = Array.unsafe_get t.ranks idx in
    let pos = Array.unsafe_get t.positions idx in
    let nid = Array.unsafe_get t.new_ids idx in
    let continue_ = ref true in
    while !continue_ do
      let l = (2 * !i) + 1 in
      if l >= t.size then
        continue_ := false
      else begin
        let r = l + 1 in
        let smallest = if r < t.size && cmp_lt t r l then r else l in
        if cmp_lt_rv t smallest rank pos then (
          Array.unsafe_set t.ranks !i (Array.unsafe_get t.ranks smallest);
          Array.unsafe_set t.positions !i (Array.unsafe_get t.positions smallest);
          Array.unsafe_set t.new_ids !i (Array.unsafe_get t.new_ids smallest);
          i := smallest)
        else
          continue_ := false
      end
    done;
    Array.unsafe_set t.ranks !i rank;
    Array.unsafe_set t.positions !i pos;
    Array.unsafe_set t.new_ids !i nid

  let push t rank pos new_id =
    let s = t.size in
    if s = Array.length t.ranks then begin
      let new_cap = max 16 (s * 2) in
      let grow a =
        let b = Array.make new_cap 0 in
        Array.blit a 0 b 0 s;
        b
      in
      t.ranks <- grow t.ranks;
      t.positions <- grow t.positions;
      t.new_ids <- grow t.new_ids
    end;
    Array.unsafe_set t.ranks s rank;
    Array.unsafe_set t.positions s pos;
    Array.unsafe_set t.new_ids s new_id;
    t.size <- s + 1;
    sift_up t s

  let pop t =
    if t.size = 0 then false
    else begin
      t.pop_rank <- Array.unsafe_get t.ranks 0;
      t.pop_pos <- Array.unsafe_get t.positions 0;
      t.pop_new_id <- Array.unsafe_get t.new_ids 0;
      t.size <- t.size - 1;
      if t.size > 0 then begin
        Array.unsafe_set t.ranks 0 (Array.unsafe_get t.ranks t.size);
        Array.unsafe_set t.positions 0 (Array.unsafe_get t.positions t.size);
        Array.unsafe_set t.new_ids 0 (Array.unsafe_get t.new_ids t.size);
        sift_down t 0
      end;
      true
    end
end

let apply_merges model dropout word =
  let p = match dropout with Some p -> p | None -> 0.0 in
  let use_dropout = p > 0.0 in
  let queue = MergeQueue.create word.size in
  let merges = model.merges in
  let sym_c = word.sym_c in
  let sym_prev = word.sym_prev in
  let sym_next = word.sym_next in
  let sym_len = word.sym_len in
  for i = 0 to word.size - 2 do
    if Array.unsafe_get sym_len i > 0 && Array.unsafe_get sym_len (i + 1) > 0
    then begin
      let key =
        merge_key (Array.unsafe_get sym_c i) (Array.unsafe_get sym_c (i + 1))
      in
      let packed = MergeMap.find merges key in
      if packed >= 0 then
        MergeQueue.push queue (merge_rank packed) i (merge_new_id packed)
    end
  done;
  let skip_ranks = ref [||] in
  let skip_positions = ref [||] in
  let skip_new_ids = ref [||] in
  let skip_size = ref 0 in
  let skip_cap = ref 0 in
  let add_skip rank pos new_id =
    if !skip_size = !skip_cap then begin
      let new_cap = max 8 (!skip_cap * 2) in
      let grow old =
        let a = Array.make new_cap 0 in
        if !skip_size > 0 then Array.blit old 0 a 0 !skip_size;
        a
      in
      skip_ranks := grow !skip_ranks;
      skip_positions := grow !skip_positions;
      skip_new_ids := grow !skip_new_ids;
      skip_cap := new_cap
    end;
    let s = !skip_size in
    Array.unsafe_set !skip_ranks s rank;
    Array.unsafe_set !skip_positions s pos;
    Array.unsafe_set !skip_new_ids s new_id;
    skip_size := s + 1
  in
  let flush_skips () =
    for i = 0 to !skip_size - 1 do
      MergeQueue.push queue
        (Array.unsafe_get !skip_ranks i)
        (Array.unsafe_get !skip_positions i)
        (Array.unsafe_get !skip_new_ids i)
    done;
    skip_size := 0
  in
  while MergeQueue.pop queue do
    let rank = queue.pop_rank in
    let pos = queue.pop_pos in
    let new_id = queue.pop_new_id in
    if Array.unsafe_get sym_len pos > 0 then begin
      let next_pos = Array.unsafe_get sym_next pos in
      if next_pos >= 0 then begin
        let key =
          merge_key (Array.unsafe_get sym_c pos)
            (Array.unsafe_get sym_c next_pos)
        in
        let packed = MergeMap.find merges key in
        if packed >= 0 && merge_new_id packed = new_id then begin
          if use_dropout && Random.float 1.0 < p then
            add_skip rank pos new_id
          else begin
            flush_skips ();
            Array.unsafe_set sym_c pos new_id;
            Array.unsafe_set sym_len pos
              (Array.unsafe_get sym_len pos
              + Array.unsafe_get sym_len next_pos);
            Array.unsafe_set sym_next pos
              (Array.unsafe_get sym_next next_pos);
            Array.unsafe_set sym_len next_pos 0;
            let new_next = Array.unsafe_get sym_next pos in
            if new_next >= 0 then
              Array.unsafe_set sym_prev new_next pos;
            let prev = Array.unsafe_get sym_prev pos in
            if prev >= 0 then begin
              let k =
                merge_key (Array.unsafe_get sym_c prev)
                  (Array.unsafe_get sym_c pos)
              in
              let v = MergeMap.find merges k in
              if v >= 0 then
                MergeQueue.push queue (merge_rank v) prev (merge_new_id v)
            end;
            let next = Array.unsafe_get sym_next pos in
            if next >= 0 then begin
              let k =
                merge_key (Array.unsafe_get sym_c pos)
                  (Array.unsafe_get sym_c next)
              in
              let v = MergeMap.find merges k in
              if v >= 0 then
                MergeQueue.push queue (merge_rank v) pos (merge_new_id v)
            end
          end
        end
      end
    end
  done;
  let j = ref 0 in
  for k = 0 to word.size - 1 do
    if Array.unsafe_get sym_len k > 0 then begin
      if !j <> k then begin
        Array.unsafe_set sym_c !j (Array.unsafe_get sym_c k);
        Array.unsafe_set sym_prev !j (Array.unsafe_get sym_prev k);
        Array.unsafe_set sym_next !j (Array.unsafe_get sym_next k);
        Array.unsafe_set sym_len !j (Array.unsafe_get sym_len k)
      end;
      incr j
    end
  done;
  word.size <- !j

let[@inline] utf8_byte_len b =
  if b land 0x80 = 0 then 1
  else if b land 0xE0 = 0xC0 then 2
  else if b land 0xF0 = 0xE0 then 3
  else 4

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

let merge_word model text =
  let text_len = String.length text in
  let word = create_word text_len in
  let pos = ref 0 in
  let no_prefix = model.continuing_subword_prefix = None in
  let no_suffix = model.end_of_word_suffix = None in
  if no_prefix && no_suffix then begin
    (* Fast path: no prefix/suffix — avoids all per-character string allocation
       for ASCII via pre-computed lookup tables *)
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
        end else begin
          flush_unk ();
          add_symbol word model.unk_id byte_len
        end
      end
    in
    while !pos < text_len do
      let b = Char.code (String.unsafe_get text !pos) in
      if b < 128 then begin
        (* ASCII: direct array lookup, zero allocation *)
        let id = Array.unsafe_get model.ascii_to_id b in
        if id >= 0 then begin
          flush_unk ();
          add_symbol word id 1
        end else if model.byte_fallback then begin
          let fbid = Array.unsafe_get model.byte_fallback_ids b in
          if fbid >= 0 then begin
            flush_unk ();
            add_symbol word fbid 1
          end else
            handle_unk 1
        end else
          handle_unk 1;
        incr pos
      end else begin
        (* Multi-byte UTF-8: packed-int key lookup, zero allocation *)
        let byte_len = utf8_byte_len b in
        let key = pack_char_key text !pos byte_len in
        (match Hashtbl.find_opt model.char_to_id key with
        | Some id ->
            flush_unk ();
            add_symbol word id byte_len
        | None ->
            if model.byte_fallback then begin
              let all_found = ref true in
              for i = 0 to byte_len - 1 do
                if
                  Array.unsafe_get model.byte_fallback_ids
                    (Char.code (String.unsafe_get text (!pos + i)))
                  < 0
                then all_found := false
              done;
              if !all_found then begin
                flush_unk ();
                for i = 0 to byte_len - 1 do
                  add_symbol word
                    (Array.unsafe_get model.byte_fallback_ids
                       (Char.code (String.unsafe_get text (!pos + i))))
                    1
                done
              end else handle_unk byte_len
            end else handle_unk byte_len);
        pos := !pos + byte_len
      end
    done;
    flush_unk ()
  end else begin
    (* Slow path: models with continuing_subword_prefix or end_of_word_suffix *)
    let pending_unk = ref None in
    let flush_unk () =
      match !pending_unk with
      | Some (uid, ulen) ->
          add_symbol word uid ulen;
          pending_unk := None
      | None -> ()
    in
    while !pos < text_len do
      let b = Char.code (String.unsafe_get text !pos) in
      let byte_len = utf8_byte_len b in
      if b land 0xC0 = 0x80 then
        (* continuation byte of invalid sequence *)
        pos := !pos + 1
      else begin
        let start = !pos in
        let char_str = String.sub text start byte_len in
        pos := !pos + byte_len;
        let is_first = start = 0 in
        let is_last = !pos >= text_len in
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
          | false, true, Some prefix, Some suffix ->
              prefix ^ char_str ^ suffix
          | false, true, Some prefix, None -> prefix ^ char_str
          | false, true, None, Some suffix -> char_str ^ suffix
          | false, true, None, None -> char_str
          | false, false, Some prefix, _ -> prefix ^ char_str
          | false, false, None, _ -> char_str
        in
        let unk_handling () =
          if model.unk_id >= 0 then begin
            if model.fuse_unk then
              pending_unk :=
                Some
                  (match !pending_unk with
                  | Some (id, len) -> (id, len + byte_len)
                  | None -> (model.unk_id, byte_len))
            else begin
              flush_unk ();
              add_symbol word model.unk_id byte_len
            end
          end
        in
        (match Hashtbl.find_opt model.vocab token_str with
        | Some id ->
            flush_unk ();
            add_symbol word id byte_len
        | None ->
            if model.byte_fallback then begin
              let all_found = ref true in
              for i = 0 to byte_len - 1 do
                if
                  Array.unsafe_get model.byte_fallback_ids
                    (Char.code (String.unsafe_get char_str i))
                  < 0
                then all_found := false
              done;
              if !all_found then begin
                flush_unk ();
                for i = 0 to byte_len - 1 do
                  add_symbol word
                    (Array.unsafe_get model.byte_fallback_ids
                       (Char.code (String.unsafe_get char_str i)))
                    1
                done
              end else unk_handling ()
            end else unk_handling ())
      end
    done;
    (match !pending_unk with
    | Some (uid, ulen) -> add_symbol word uid ulen
    | None -> ())
  end;
  apply_merges model model.dropout word;
  word

let word_to_tokens model word =
  let offset = ref 0 in
  List.init word.size (fun i ->
      let id = Array.unsafe_get word.sym_c i in
      let value =
        match Hashtbl.find_opt model.vocab_r id with
        | Some v -> v
        | None -> "<unk>"
      in
      let start = !offset in
      let end_ = start + Array.unsafe_get word.sym_len i in
      offset := end_;
      { id; value; offsets = (start, end_) })

let word_to_ids word = Array.init word.size (fun i -> Array.unsafe_get word.sym_c i)

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

let tokenize_ids model text =
  if String.length text = 0 then [||]
  else
    match Hashtbl.find_opt model.vocab text with
    | Some id -> [| id |]
    | None ->
        let get_word text =
          if model.ignore_merges then merge_word model text
          else
            match model.cache with
            | Some cache when String.length text < 1000 -> (
                match Hashtbl.find_opt cache text with
                | Some word -> word
                | None ->
                    let word = merge_word model text in
                    Hashtbl.add cache text word;
                    word)
            | _ -> merge_word model text
        in
        word_to_ids (get_word text)

let token_to_id model token = Hashtbl.find_opt model.vocab token
let id_to_token model id = Hashtbl.find_opt model.vocab_r id
let get_vocab model = Hashtbl.fold (fun k v acc -> (k, v) :: acc) model.vocab []
let get_vocab_size model = Hashtbl.length model.vocab
let get_unk_token model = model.unk_token
let get_continuing_subword_prefix model = model.continuing_subword_prefix
let get_end_of_word_suffix model = model.end_of_word_suffix

let get_merges model =
  MergeMap.fold
    (fun key packed acc ->
      let a_id = key lsr 21 in
      let b_id = key land 0x1FFFFF in
      let rank = merge_rank packed in
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
          | Some new_id -> Some ((a_id, b_id), pack_merge rank new_id)
          | None ->
              failwith
                (Printf.sprintf "Merge token '%s' not in vocabulary" new_token))
      | _ -> failwith (Printf.sprintf "Merge tokens not in vocabulary"))
    merges
  |> List.filter_map (fun x -> x)
  |> fun entries ->
  MergeMap.create
    (List.map (fun ((a_id, b_id), packed) -> (merge_key a_id b_id, packed)) entries)

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
  let ascii_to_id = Array.make 128 (-1) in
  for i = 0 to 127 do
    let s = String.make 1 (Char.chr i) in
    match Hashtbl.find_opt cfg.vocab s with
    | Some id -> ascii_to_id.(i) <- id
    | None -> ()
  done;
  let byte_fallback_ids = Array.make 256 (-1) in
  for i = 0 to 255 do
    let hex = Printf.sprintf "<0x%02X>" i in
    match Hashtbl.find_opt cfg.vocab hex with
    | Some id -> byte_fallback_ids.(i) <- id
    | None -> ()
  done;
  (* Build packed-int char lookup table for zero-allocation multi-byte lookup *)
  let char_to_id = Hashtbl.create 256 in
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
          Hashtbl.replace char_to_id packed id
      end)
    cfg.vocab;
  let unk_id =
    match cfg.unk_token with
    | Some unk -> (
        match Hashtbl.find_opt cfg.vocab unk with Some id -> id | None -> -1)
    | None -> -1
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
    ascii_to_id;
    byte_fallback_ids;
    char_to_id;
    unk_id;
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
        MergeMap.fold
          (fun key packed acc ->
            let a_id = key lsr 21 in
            let b_id = key land 0x1FFFFF in
            let rank = merge_rank packed in
            match
              ( Hashtbl.find_opt model.vocab_r a_id,
                Hashtbl.find_opt model.vocab_r b_id )
            with
            | Some a, Some b -> (rank, a, b) :: acc
            | _ -> acc)
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
      let len = String.length word in
      let chars = ref [] in
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
                  Re.replace_string
                    (Re.compile (Re.str (a ^ " " ^ b)))
                    ~by:new_token word
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
