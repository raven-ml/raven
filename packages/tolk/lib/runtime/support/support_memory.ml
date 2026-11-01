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

  (* A free block is kept in the subdivision of its size, an allocated one in
     none. *)
  let set_free a start size free =
    let l1 = lv1 size and l2 = lv2 a size in
    let bucket = a.storage.(l1).(l2) in
    a.storage.(l1).(l2) <-
      (if free then bucket @ [ start ] else List.filter (( <> ) start) bucket);
    a.lv1_entries.(l1) <- (a.lv1_entries.(l1) + if free then 1 else -1);
    let prev = (Hashtbl.find a.blocks start).prev in
    Hashtbl.replace a.blocks start { size; prev; free }

  let set_prev a start prev =
    match Hashtbl.find_opt a.blocks start with
    | Some b -> Hashtbl.replace a.blocks start { b with prev }
    | None -> ()

  (* Splits the allocated block at [start] in two allocated blocks; the caller
     gives the unused one back with [set_free]. *)
  let split_block a start size new_size =
    let b = Hashtbl.find a.blocks start in
    Hashtbl.replace a.blocks start { b with size = new_size };
    Hashtbl.replace a.blocks (start + new_size)
      { size = size - new_size; prev = Some start; free = false };
    set_prev a (start + size) (Some (start + new_size))

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
    if size > 0 then set_free a 0 size true;
    a

  let alloc ?(align = 1) a req_size =
    if align <= 0 then invalid_arg (Printf.sprintf "an alignment of %d" align);
    let req_size = max a.block_size req_size in
    let size = max a.block_size (req_size + align - 1) in
    (* Rounded up to the next subdivision, any block of which fits. *)
    let size = Helpers.round_up size (1 lsl (bit_length size - a.l2_cnt)) in
    (* Takes the whole block, then gives back its unaligned head and its
       unused tail. *)
    let take start =
      let nsize = (Hashtbl.find a.blocks start).size in
      set_free a start nsize false;
      let aligned = Helpers.round_up start align in
      let start, nsize =
        if aligned = start then (start, nsize)
        else begin
          let head = aligned - start in
          split_block a start nsize head;
          set_free a start head true;
          (aligned, nsize - head)
        end
      in
      if nsize > req_size then begin
        split_block a start nsize req_size;
        set_free a (start + req_size) (nsize - req_size) true
      end;
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
        (* A free neighbour merges in. There is at most one on each side, since
           free blocks are always merged. *)
        let size =
          match Hashtbl.find_opt a.blocks (start + b.size) with
          | Some n when n.free ->
              set_free a (start + b.size) n.size false;
              Hashtbl.remove a.blocks (start + b.size);
              b.size + n.size
          | _ -> b.size
        in
        let start, size =
          match b.prev with
          | Some p when (Hashtbl.find a.blocks p).free ->
              let psize = (Hashtbl.find a.blocks p).size in
              set_free a p psize false;
              Hashtbl.remove a.blocks start;
              (p, size + psize)
          | _ -> (start, size)
        in
        set_prev a (start + size) (Some start);
        set_free a start size true
    | _ -> invalid_arg (Printf.sprintf "no allocated block at %d" addr)
end
