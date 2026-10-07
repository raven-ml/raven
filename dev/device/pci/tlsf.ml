(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Blocks tile the range. Each knows the start of the block before it; the one
   after it starts where it ends. Free blocks are also on the list of their
   class: first level [f], the bit length of their size, and second level [s],
   the four bits below the top one. [firsts] has bit [f] set iff level [f] has a
   free block, and [seconds.(f)] bit [s] iff class [(f, s)] has one. No two free
   blocks are neighbours: a freed block merges at once.

   Starts are relative to [base]; [none] ends a chain. *)

let second_bits = 4
let classes = 1 lsl second_bits
let levels = Sys.int_size
let none = -1

type block = {
  mutable size : int;
  mutable prev : int; (* the block before it *)
  mutable free : bool;
  mutable next_free : int; (* on its class's list *)
  mutable prev_free : int;
}

type t = {
  base : int;
  length : int;
  block : int;
  blocks : (int, block) Hashtbl.t;
  heads : int array; (* the first free block of each class *)
  seconds : int array;
  mutable firsts : int;
}

(* Bits *)

(* The number of bits of [n > 0], in six steps. *)
let bit_length n =
  let rec go n k step =
    if step = 0 then k + n
    else if n lsr step <> 0 then go (n lsr step) (k + step) (step / 2)
    else go n k (step / 2)
  in
  go n 0 32

(* The index of the lowest set bit of [m <> 0]. *)
let lowest m = bit_length (m land -m) - 1

(* Bits [f] and above of [m]. *)
let from f m = if f >= levels then 0 else m land (-1 lsl f)

(* Classes *)

(* The class of a block of [size > 0] bytes. *)
let class_of size =
  let f = bit_length size in
  let rest = size - (1 lsl (f - 1)) in
  let s =
    if f - 1 >= second_bits then rest lsr (f - 1 - second_bits)
    else rest lsl (second_bits - (f - 1))
  in
  (f, s)

(* [size] rounded up to the smallest size of its class's successor, unless it
   starts its class: every block of the class found for it fits it. *)
let round_class size =
  let f = bit_length size in
  if f - 1 <= second_bits then size
  else
    let step = 1 lsl (f - 1 - second_bits) in
    (size + step - 1) land lnot (step - 1)

let slot (f, s) = (f * classes) + s

(* Free lists *)

let find t start = Hashtbl.find t.blocks start

let insert t start b =
  let ((f, s) as c) = class_of b.size in
  let i = slot c in
  let head = t.heads.(i) in
  b.free <- true;
  b.prev_free <- none;
  b.next_free <- head;
  if head <> none then (find t head).prev_free <- start;
  t.heads.(i) <- start;
  t.seconds.(f) <- t.seconds.(f) lor (1 lsl s);
  t.firsts <- t.firsts lor (1 lsl f)

let remove t b =
  let ((f, s) as c) = class_of b.size in
  let i = slot c in
  if b.prev_free <> none then (find t b.prev_free).next_free <- b.next_free
  else t.heads.(i) <- b.next_free;
  if b.next_free <> none then (find t b.next_free).prev_free <- b.prev_free;
  if t.heads.(i) = none then begin
    t.seconds.(f) <- t.seconds.(f) land lnot (1 lsl s);
    if t.seconds.(f) = 0 then t.firsts <- t.firsts land lnot (1 lsl f)
  end;
  b.free <- false

(* The first free block of class [c] or a larger one. *)
let suitable t (f, s) =
  let m = from s t.seconds.(f) in
  if m <> 0 then Some t.heads.(slot (f, lowest m))
  else
    let m = from (f + 1) t.firsts in
    if m = 0 then None
    else
      let f = lowest m in
      Some t.heads.(slot (f, lowest t.seconds.(f)))

(* Blocks *)

(* Shrinks the block [b] at [start] to [n] bytes and is the block made of the
   rest, which is on no list. *)
let carve t start b n =
  let rest = start + n in
  let r =
    {
      size = b.size - n;
      prev = start;
      free = false;
      next_free = none;
      prev_free = none;
    }
  in
  (match Hashtbl.find_opt t.blocks (start + b.size) with
  | Some next -> next.prev <- rest
  | None -> ());
  b.size <- n;
  Hashtbl.replace t.blocks rest r;
  (rest, r)

(* Joins the block at [next] to [b], before it. Neither is on a list. *)
let absorb t start b next =
  let n = find t next in
  b.size <- b.size + n.size;
  Hashtbl.remove t.blocks next;
  match Hashtbl.find_opt t.blocks (start + b.size) with
  | Some after -> after.prev <- start
  | None -> ()

(* Allocators *)

let create ?(block = 16) ~base length =
  if length < 0 then invalid_arg (Printf.sprintf "Tlsf.create: %d" length);
  if block <= 0 then invalid_arg (Printf.sprintf "Tlsf.create: block %d" block);
  let t =
    {
      base;
      length;
      block;
      blocks = Hashtbl.create 64;
      heads = Array.make (levels * classes) none;
      seconds = Array.make levels 0;
      firsts = 0;
    }
  in
  if length > 0 then begin
    let b =
      {
        size = length;
        prev = none;
        free = false;
        next_free = none;
        prev_free = none;
      }
    in
    Hashtbl.replace t.blocks 0 b;
    insert t 0 b
  end;
  t

let base t = t.base
let length t = t.length
let round_up n a = (n + a - 1) / a * a

let alloc ?(align = 1) t n =
  if n < 0 then invalid_arg (Printf.sprintf "Tlsf.alloc: %d bytes" n);
  if align <= 0 then invalid_arg (Printf.sprintf "Tlsf.alloc: align %d" align);
  let req = Int.max t.block n in
  (* A block of [req + align - 1] bytes holds the request wherever it starts.
     None is larger than the range, and the sum could wrap past [max_int]. *)
  if req > t.length - align + 1 then None
  else
    let need = round_class (req + align - 1) in
    (* Rounding wraps below 0 only for a range within 2^57 of [max_int]. *)
    if need < 0 then None
    else
      match suitable t (class_of need) with
      | None -> None
      | Some start ->
          let b = find t start in
          remove t b;
          (* The gap below the aligned start stays free. Its neighbour before
             was [b]'s, which is not free. *)
          let gap = round_up (t.base + start) align - t.base - start in
          let start, b =
            if gap = 0 then (start, b)
            else
              let a, ab = carve t start b gap in
              insert t start b;
              (a, ab)
          in
          (* The tail above the request is free, unless smaller than a block. *)
          if b.size - req >= t.block then begin
            let r, rb = carve t start b req in
            insert t r rb
          end;
          Some (t.base + start)

let free t x =
  let start = x - t.base in
  match Hashtbl.find_opt t.blocks start with
  | Some b when not b.free ->
      let start, b =
        match Hashtbl.find_opt t.blocks b.prev with
        | Some p when p.free ->
            remove t p;
            absorb t b.prev p start;
            (b.prev, p)
        | _ -> (start, b)
      in
      (match Hashtbl.find_opt t.blocks (start + b.size) with
      | Some n when n.free ->
          remove t n;
          absorb t start b (start + b.size)
      | _ -> ());
      insert t start b
  | _ -> invalid_arg (Printf.sprintf "Tlsf.free: no block at 0x%x" x)
