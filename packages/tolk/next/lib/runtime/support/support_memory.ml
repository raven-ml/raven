(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let rec bit_length n = if n = 0 then 0 else 1 + bit_length (n lsr 1)

module Tlsf_allocator = struct
  (* The blocks form a list by address: each block's successor starts where it
     ends. *)
  type block = { size : int; prev : int option; free : bool }

  type t = {
    base : int;
    block_size : int;
    l2_cnt : int;
    storage : int list array array;
        (* The free blocks by first and second level, in the order they became
           free. *)
    lv1_entries : int array;
    blocks : (int, block) Hashtbl.t;
  }

  let lv1 size = bit_length size

  let lv2 a size =
    let b = bit_length size in
    (size - (1 lsl (b - 1))) / (1 lsl max 0 (b - a.l2_cnt))

  let insert_block ?prev a start size =
    let prev =
      match prev with
      | Some p -> Some p
      | None -> (Hashtbl.find a.blocks start).prev
    in
    let l1 = lv1 size and l2 = lv2 a size in
    a.storage.(l1).(l2) <- a.storage.(l1).(l2) @ [ start ];
    a.lv1_entries.(l1) <- a.lv1_entries.(l1) + 1;
    Hashtbl.replace a.blocks start { size; prev; free = true }

  let remove_block a start size =
    let prev = (Hashtbl.find a.blocks start).prev in
    let l1 = lv1 size and l2 = lv2 a size in
    a.storage.(l1).(l2) <- List.filter (fun s -> s <> start) a.storage.(l1).(l2);
    a.lv1_entries.(l1) <- a.lv1_entries.(l1) - 1;
    Hashtbl.replace a.blocks start { size; prev; free = false }

  let set_prev a start prev =
    match Hashtbl.find_opt a.blocks start with
    | Some b -> Hashtbl.replace a.blocks start { b with prev = Some prev }
    | None -> ()

  let split_block a start size new_size =
    remove_block a start size;
    insert_block a start new_size;
    insert_block a (start + new_size) (size - new_size) ~prev:start;
    set_prev a (start + size) (start + new_size)

  let merge_right a start =
    let rec loop size =
      match Hashtbl.find_opt a.blocks (start + size) with
      | Some blk when blk.free ->
          remove_block a start size;
          remove_block a (start + size) blk.size;
          insert_block a start (size + blk.size);
          Hashtbl.remove a.blocks (start + size);
          loop (size + blk.size)
      | _ -> set_prev a (start + size) start
    in
    loop (Hashtbl.find a.blocks start).size

  (* Go left while blocks are free, then merge them all to the right. *)
  let merge_block a start =
    let rec leftmost start =
      match (Hashtbl.find a.blocks start).prev with
      | Some x when (Hashtbl.find a.blocks x).free -> leftmost x
      | _ -> start
    in
    merge_right a (leftmost start)

  let create ?(base = 0) ?(block_size = 16) ?(lv2_cnt = 16) size =
    if size < 0 then invalid_arg (Printf.sprintf "a negative size %d" size);
    if block_size <= 0 || lv2_cnt <= 0 then
      invalid_arg
        (Printf.sprintf "a block size of %d and %d subdivisions" block_size
           lv2_cnt);
    let l2_cnt = bit_length lv2_cnt in
    if bit_length block_size < l2_cnt then
      invalid_arg
        (Printf.sprintf "a block size of %d is too small for %d subdivisions"
           block_size lv2_cnt);
    let levels = bit_length size + 1 in
    let a =
      {
        base;
        block_size;
        l2_cnt;
        storage = Array.init levels (fun _ -> Array.make (1 lsl l2_cnt) []);
        lv1_entries = Array.make levels 0;
        blocks = Hashtbl.create 64;
      }
    in
    Hashtbl.replace a.blocks 0 { size; prev = None; free = true };
    if size > 0 then insert_block a 0 size;
    a

  let alloc ?(align = 1) a req_size =
    if align <= 0 then invalid_arg (Printf.sprintf "an alignment of %d" align);
    let req_size = max a.block_size req_size in
    let size = max a.block_size (req_size + align - 1) in
    (* Rounded up to the next subdivision, any block of which fits. *)
    let size = Helpers.round_up size (1 lsl (bit_length size - a.l2_cnt)) in
    let take start =
      let nsize = (Hashtbl.find a.blocks start).size in
      let start, nsize =
        let aligned = Helpers.round_up start align in
        if aligned = start then (start, nsize)
        else begin
          split_block a start nsize (aligned - start);
          (aligned, (Hashtbl.find a.blocks aligned).size)
        end
      in
      if nsize > req_size then split_block a start nsize req_size;
      remove_block a start req_size;
      start + a.base
    in
    let rec search l1 l2 =
      if l1 >= Array.length a.storage then None
      else if a.lv1_entries.(l1) = 0 || l2 >= 1 lsl a.l2_cnt then
        search (l1 + 1) 0
      else
        match a.storage.(l1).(l2) with
        | start :: _ -> Some (take start)
        | [] -> search l1 (l2 + 1)
    in
    let l1 = lv1 size in
    search l1 (lv2 a size)

  let free a addr =
    let start = addr - a.base in
    match Hashtbl.find_opt a.blocks start with
    | Some b when not b.free ->
        insert_block a start b.size;
        merge_block a start
    | _ -> invalid_arg (Printf.sprintf "no allocated block at %d" addr)
end
