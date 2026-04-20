(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Two-Level Segregated Fit allocator.

   Maintains two levels of free-list buckets for O(1) best-fit allocation:
     - Level 1 is the most significant bit of the block size.
     - Level 2 subdivides each L1 range into [2^l2_cnt] entries.

   Allocation searches for the smallest block that fits, splitting if
   oversized. Deallocation merges the freed block with its neighbours. *)

let round_up n align = (n + align - 1) / align * align

let bit_length n =
  let rec go acc n = if n = 0 then acc else go (acc + 1) (n lsr 1) in
  go 0 n

type block = {
  size : int;
  next : int option;
  prev : int option;
  is_free : bool;
}

type t = {
  base : int;
  block_size : int;
  l2_cnt : int;
  storage : (int, int list) Hashtbl.t array;
  lv1_entries : int array;
  blocks : (int, block) Hashtbl.t;
}

let lv1 size = bit_length size

let lv2 t size =
  let bl = bit_length size in
  (size - (1 lsl (bl - 1))) / (1 lsl (max 0 (bl - t.l2_cnt)))

let insert_block t start size ?prev () =
  let prev = match prev with
    | Some p -> p
    | None -> (Hashtbl.find t.blocks start).prev in
  let l1 = lv1 size and l2 = lv2 t size in
  let cur = match Hashtbl.find_opt t.storage.(l1) l2 with
    | Some l -> l | None -> [] in
  Hashtbl.replace t.storage.(l1) l2 (start :: cur);
  t.lv1_entries.(l1) <- t.lv1_entries.(l1) + 1;
  Hashtbl.replace t.blocks start
    { size; next = Some (start + size); prev; is_free = true }

let remove_block t start size ?prev () =
  let prev = match prev with
    | Some p -> p
    | None -> (Hashtbl.find t.blocks start).prev in
  let l1 = lv1 size and l2 = lv2 t size in
  let cur = match Hashtbl.find_opt t.storage.(l1) l2 with
    | Some l -> l | None -> [] in
  Hashtbl.replace t.storage.(l1) l2
    (List.filter (fun s -> s <> start) cur);
  t.lv1_entries.(l1) <- t.lv1_entries.(l1) - 1;
  Hashtbl.replace t.blocks start
    { size; next = Some (start + size); prev; is_free = false }

let split_block t start size new_size =
  let blk = Hashtbl.find t.blocks start in
  assert blk.is_free;
  let nxt = blk.next in
  remove_block t start size ();
  insert_block t start new_size ();
  insert_block t (start + new_size) (size - new_size) ~prev:(Some start) ();
  (match nxt with
   | Some n when Hashtbl.mem t.blocks n ->
       let b = Hashtbl.find t.blocks n in
       Hashtbl.replace t.blocks n
         { b with prev = Some (start + new_size) }
   | _ -> ())

let merge_right t start =
  let blk = Hashtbl.find t.blocks start in
  assert blk.is_free;
  let size = ref blk.size in
  let nxt = ref blk.next in
  let continue = ref true in
  while !continue do
    match !nxt with
    | Some n when Hashtbl.mem t.blocks n ->
        let b = Hashtbl.find t.blocks n in
        if not b.is_free then continue := false
        else begin
          remove_block t start !size ();
          remove_block t n b.size ();
          size := !size + b.size;
          insert_block t start !size ();
          assert ((Hashtbl.find t.blocks start).next = b.next);
          nxt := (Hashtbl.find t.blocks n).next;
          Hashtbl.remove t.blocks n
        end
    | _ -> continue := false
  done;
  (match !nxt with
   | Some n when Hashtbl.mem t.blocks n ->
       let b = Hashtbl.find t.blocks n in
       Hashtbl.replace t.blocks n { b with prev = Some start }
   | _ -> ())

let merge_block t start =
  let start = ref start in
  let continue = ref true in
  while !continue do
    match (Hashtbl.find t.blocks !start).prev with
    | Some x when (Hashtbl.find t.blocks x).is_free -> start := x
    | _ -> continue := false
  done;
  merge_right t !start

let create ~size ?(base = 0) ?(block_size = 16) ?(lv2_cnt = 16) () =
  let l2_cnt = bit_length lv2_cnt in
  let n_levels = bit_length size + 1 in
  let storage = Array.init n_levels (fun _ -> Hashtbl.create 4) in
  let lv1_entries = Array.make n_levels 0 in
  let blocks = Hashtbl.create 64 in
  let t = { base; block_size; l2_cnt; storage; lv1_entries; blocks } in
  Hashtbl.replace blocks 0
    { size; next = None; prev = None; is_free = true };
  if size > 0 then insert_block t 0 size ();
  t

let alloc t req_size ?(align = 1) () =
  let req_size = max t.block_size req_size in
  let size = max t.block_size (req_size + align - 1) in
  (* Round up to the next bucket boundary so any entry there fits. *)
  let size = round_up size (1 lsl (bit_length size - t.l2_cnt)) in
  let n_levels = Array.length t.storage in
  let result = ref (-1) in
  let l1 = ref (lv1 size) in
  while !l1 < n_levels && !result = -1 do
    if t.lv1_entries.(!l1) <> 0 then begin
      let l2_start =
        if !l1 = bit_length size then lv2 t size else 0 in
      let l2_end = 1 lsl t.l2_cnt in
      let l2 = ref l2_start in
      while !l2 < l2_end && !result = -1 do
        let entries =
          match Hashtbl.find_opt t.storage.(!l1) !l2 with
          | Some l -> l | None -> [] in
        if entries <> [] then begin
          let start = ref (List.hd entries) in
          let nsize = ref (Hashtbl.find t.blocks !start).size in
          assert (!nsize >= size);
          (* Alignment: split off a prefix if the start isn't aligned. *)
          let new_start = round_up !start align in
          if new_start <> !start then begin
            split_block t !start !nsize (new_start - !start);
            start := new_start;
            nsize := (Hashtbl.find t.blocks new_start).size
          end;
          (* Split off the tail if the block is larger than needed. *)
          if !nsize > req_size then
            split_block t !start !nsize req_size;
          remove_block t !start req_size ();
          result := !start + t.base
        end;
        incr l2
      done
    end;
    incr l1
  done;
  if !result = -1 then raise Out_of_memory;
  !result

let free t start =
  let s = start - t.base in
  let blk = Hashtbl.find t.blocks s in
  insert_block t s blk.size ();
  merge_block t s
