(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Blocks tile the range as a list linked by [next] (the address after a block)
   and [prev]. Free blocks are also in the bucket of their size; a bucket keeps
   blocks in the order they were freed. *)
type block = { size : int; prev : int option; free : bool }

type t = {
  base : int;
  length : int;
  block : int;
  subdivisions : int; (* bits of a size that pick its second-level bucket *)
  buckets : int list array array; (* by first level, then second level *)
  counts : int array; (* free blocks per first level *)
  blocks : (int, block) Hashtbl.t; (* by start, relative to [base] *)
}

let bit_length n =
  let rec go n k = if n = 0 then k else go (n lsr 1) (k + 1) in
  go n 0

let round_up n a = (n + a - 1) / a * a
let level1 size = bit_length size

let level2 t size =
  let bl = bit_length size in
  (size - (1 lsl (bl - 1))) / (1 lsl Int.max 0 (bl - t.subdivisions))

let insert t ?prev start size =
  let prev =
    match prev with
    | Some _ -> prev
    | None -> Option.bind (Hashtbl.find_opt t.blocks start) (fun b -> b.prev)
  in
  let l1 = level1 size and l2 = level2 t size in
  t.buckets.(l1).(l2) <- t.buckets.(l1).(l2) @ [ start ];
  t.counts.(l1) <- t.counts.(l1) + 1;
  Hashtbl.replace t.blocks start { size; prev; free = true }

let remove t start size =
  let prev = (Hashtbl.find t.blocks start).prev in
  let l1 = level1 size and l2 = level2 t size in
  t.buckets.(l1).(l2) <- List.filter (( <> ) start) t.buckets.(l1).(l2);
  t.counts.(l1) <- t.counts.(l1) - 1;
  Hashtbl.replace t.blocks start { size; prev; free = false }

let set_prev t start prev =
  match Hashtbl.find_opt t.blocks start with
  | Some b -> Hashtbl.replace t.blocks start { b with prev }
  | None -> ()

let split t start size first =
  let next = start + size in
  remove t start size;
  insert t start first;
  insert t ~prev:start (start + first) (size - first);
  set_prev t next (Some (start + first))

let merge_right t start =
  let rec go size next =
    match Hashtbl.find_opt t.blocks next with
    | Some b when b.free ->
        remove t start size;
        remove t next b.size;
        insert t start (size + b.size);
        Hashtbl.remove t.blocks next;
        go (size + b.size) (next + b.size)
    | _ -> set_prev t next (Some start)
  in
  let b = Hashtbl.find t.blocks start in
  go b.size (start + b.size)

let rec merge t start =
  match (Hashtbl.find t.blocks start).prev with
  | Some p when (Hashtbl.find t.blocks p).free -> merge t p
  | _ -> merge_right t start

let is_power_of_two n = n > 0 && n land (n - 1) = 0

let create ?(block = 16) ?(subdivisions = 16) ~base length =
  if length < 0 then invalid_arg (Printf.sprintf "Tlsf.create: %d" length);
  if block <= 0 then invalid_arg (Printf.sprintf "Tlsf.create: block %d" block);
  if not (is_power_of_two subdivisions) then
    invalid_arg (Printf.sprintf "Tlsf.create: subdivisions %d" subdivisions);
  let levels = bit_length length + 1 in
  let bits = bit_length subdivisions in
  let t =
    {
      base;
      length;
      block;
      subdivisions = bits;
      buckets = Array.init levels (fun _ -> Array.make (1 lsl bits) []);
      counts = Array.make levels 0;
      blocks = Hashtbl.create 64;
    }
  in
  Hashtbl.replace t.blocks 0 { size = length; prev = None; free = true };
  if length > 0 then insert t 0 length;
  t

let base t = t.base
let length t = t.length

(* The first free block of at least [size] bytes, from [size]'s bucket up. *)
let find t size =
  let l1 = level1 size and l2 = level2 t size in
  let rec level i =
    if i >= Array.length t.buckets then None
    else if t.counts.(i) = 0 then level (i + 1)
    else
      let rec bucket j =
        if j >= Array.length t.buckets.(i) then level (i + 1)
        else
          match t.buckets.(i).(j) with s :: _ -> Some s | [] -> bucket (j + 1)
      in
      bucket (if i = l1 then l2 else 0)
  in
  level l1

let alloc ?(align = 1) t n =
  if n < 0 then invalid_arg (Printf.sprintf "Tlsf.alloc: %d bytes" n);
  if align <= 0 then invalid_arg (Printf.sprintf "Tlsf.alloc: align %d" align);
  let req = Int.max t.block n in
  let size = Int.max t.block (req + align - 1) in
  let size =
    round_up size (1 lsl Int.max 0 (bit_length size - t.subdivisions))
  in
  match find t size with
  | None -> None
  | Some start ->
      let nsize = (Hashtbl.find t.blocks start).size in
      let aligned = round_up (t.base + start) align - t.base in
      let start, nsize =
        if aligned = start then (start, nsize)
        else begin
          split t start nsize (aligned - start);
          (aligned, (Hashtbl.find t.blocks aligned).size)
        end
      in
      if nsize > req then split t start nsize req;
      remove t start req;
      Some (t.base + start)

let free t x =
  let start = x - t.base in
  match Hashtbl.find_opt t.blocks start with
  | Some b when not b.free ->
      insert t start b.size;
      merge t start
  | _ -> invalid_arg (Printf.sprintf "Tlsf.free: no block at 0x%x" x)
