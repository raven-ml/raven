(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Blocks tile the range. Each knows the blocks before and after it. Free
   blocks are also on the list of their class: first level [f], the bit length
   of their size, and second level [s], the four bits below the top one.
   [firsts] has bit [f] set iff level [f] has a free block, and [seconds.(f)]
   bit [s] iff class [(f, s)] has one. No two free blocks are neighbours: a
   freed block merges at once.

   A call allocates everything it needs first: the result, and the records of
   the blocks it makes, bound in the table while unlinked. It then changes
   the blocks and their lists with nothing that allocates or polls, and last
   unbinds the blocks a free joined. An exception therefore leaves the blocks
   as they were before the call or as the call made them, and at most
   unlinked records in the table, which a free ignores and a later block at
   their start replaces.

   Starts are relative to [base]; [nil] ends a chain. *)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let second_bits = 4
let classes = 1 lsl second_bits
let levels = Sys.int_size

(* The smallest block handed out: a smaller tail stays with its block. *)
let min_block = 16

type state = Taken | Free | Unlinked

type block = {
  start : int;
  mutable size : int;
  mutable state : state;
  mutable prev : block; (* the block before it *)
  mutable next : block; (* the block after it *)
  mutable prev_free : block; (* on its class's list *)
  mutable next_free : block;
}

let rec nil =
  {
    start = -1;
    size = 0;
    state = Unlinked;
    prev = nil;
    next = nil;
    prev_free = nil;
    next_free = nil;
  }

type t = {
  base : int;
  length : int;
  blocks : block Tables.Address.t;
  bottom : block; (* the block at the range's start, [nil] for none *)
  heads : block array; (* the first free block of each class *)
  seconds : int array;
  mutable firsts : int;
}

(* Bits *)

(* The number of bits of [n > 0], in six steps, with no loop, so that the
   changes of a call reach no poll point. *)
let bit_length n =
  let n = ref n and k = ref 0 in
  if !n lsr 32 <> 0 then begin
    n := !n lsr 32;
    k := !k + 32
  end;
  if !n lsr 16 <> 0 then begin
    n := !n lsr 16;
    k := !k + 16
  end;
  if !n lsr 8 <> 0 then begin
    n := !n lsr 8;
    k := !k + 8
  end;
  if !n lsr 4 <> 0 then begin
    n := !n lsr 4;
    k := !k + 4
  end;
  if !n lsr 2 <> 0 then begin
    n := !n lsr 2;
    k := !k + 2
  end;
  if !n lsr 1 <> 0 then begin
    n := !n lsr 1;
    k := !k + 1
  end;
  !k + !n

(* The index of the lowest set bit of [m <> 0]. *)
let lowest m = bit_length (m land -m) - 1

(* Bits [f] and above of [m]. *)
let from f m = if f >= levels then 0 else m land (-1 lsl f)

(* Classes *)

(* The class of a block of [size > 0] bytes is its first level [f] and its
   second level [second f size]. *)
let first size = bit_length size

let second f size =
  let rest = size - (1 lsl (f - 1)) in
  if f - 1 >= second_bits then rest lsr (f - 1 - second_bits)
  else rest lsl (second_bits - (f - 1))

(* [size] rounded up to the smallest size of its class's successor, unless it
   starts its class: every block of the class found for it fits it. *)
let round_class size =
  let f = bit_length size in
  if f - 1 <= second_bits then size
  else
    let step = 1 lsl (f - 1 - second_bits) in
    (size + step - 1) land lnot (step - 1)

let slot f s = (f * classes) + s

(* Free lists *)

let insert t b =
  let f = first b.size in
  let s = second f b.size in
  let i = slot f s in
  let head = t.heads.(i) in
  b.state <- Free;
  b.prev_free <- nil;
  b.next_free <- head;
  if head != nil then head.prev_free <- b;
  t.heads.(i) <- b;
  t.seconds.(f) <- t.seconds.(f) lor (1 lsl s);
  t.firsts <- t.firsts lor (1 lsl f)

let remove t b =
  let f = first b.size in
  let s = second f b.size in
  let i = slot f s in
  if b.prev_free != nil then b.prev_free.next_free <- b.next_free
  else t.heads.(i) <- b.next_free;
  if b.next_free != nil then b.next_free.prev_free <- b.prev_free;
  if t.heads.(i) == nil then begin
    t.seconds.(f) <- t.seconds.(f) land lnot (1 lsl s);
    if t.seconds.(f) = 0 then t.firsts <- t.firsts land lnot (1 lsl f)
  end

(* The first free block of class [(f, s)] or a larger one, or [nil]. *)
let suitable t f s =
  let m = from s t.seconds.(f) in
  if m <> 0 then t.heads.(slot f (lowest m))
  else
    let m = from (f + 1) t.firsts in
    if m = 0 then nil
    else
      let f = lowest m in
      t.heads.(slot f (lowest t.seconds.(f)))

(* Blocks *)

(* A record for the block at [start], bound in [blocks], unlinked. *)
let unlinked blocks start =
  let b =
    {
      start;
      size = 0;
      state = Unlinked;
      prev = nil;
      next = nil;
      prev_free = nil;
      next_free = nil;
    }
  in
  Tables.Address.replace blocks start b;
  b

(* Shrinks the block [b] to [n] bytes and makes [r], at [b.start + n], the
   block of the rest. *)
let split b r n =
  r.size <- b.size - n;
  r.prev <- b;
  r.next <- b.next;
  if b.next != nil then b.next.prev <- r;
  b.next <- r;
  b.size <- n

(* Joins [n], the block after [b], to [b]. Neither is on a list. *)
let absorb b n =
  b.size <- b.size + n.size;
  b.next <- n.next;
  if n.next != nil then n.next.prev <- b;
  n.state <- Unlinked

(* Allocators *)

let create ~base length =
  if length < 0 then invalid_argf "Tlsf.create: %d" length;
  let blocks = Tables.Address.create 64 in
  let bottom = if length > 0 then unlinked blocks 0 else nil in
  let t =
    {
      base;
      length;
      blocks;
      bottom;
      heads = Array.make (levels * classes) nil;
      seconds = Array.make levels 0;
      firsts = 0;
    }
  in
  if length > 0 then begin
    bottom.size <- length;
    insert t bottom
  end;
  t

let base t = t.base
let length t = t.length
let round_up n a = (n + a - 1) / a * a

(* Takes [req] bytes at the aligned start of the free block [b]: the gap below
   and the tail above stay free. *)
let take_from t ~align b req =
  let gap = round_up (t.base + b.start) align - t.base - b.start in
  let at = b.start + gap in
  (* The tail above the request is free, unless smaller than a block. *)
  let tail = b.size - gap - req >= min_block in
  let got = Some (t.base + at) in
  let ab = if gap = 0 then b else unlinked t.blocks at in
  let rest = if tail then unlinked t.blocks (at + req) else nil in
  remove t b;
  (* The gap below the aligned start stays free. Its neighbour before was
     [b]'s, which is not free. *)
  if gap > 0 then begin
    split b ab gap;
    insert t b
  end;
  if tail then begin
    split ab rest req;
    insert t rest
  end;
  ab.state <- Taken;
  got

(* The first free block, in address order, that holds [req] bytes at an aligned
   start ending at or below [below]: a walk of the blocks, for a request that
   the bound excludes part of the range for. *)
let first_below t ~align ~below req =
  let rec walk b =
    if b == nil || t.base + b.start >= below then None
    else
      let at = round_up (t.base + b.start) align - t.base in
      if
        b.state = Free
        && at + req <= b.start + b.size
        && t.base + at + req <= below
      then take_from t ~align b req
      else walk b.next
  in
  walk t.bottom

let alloc ?(align = 1) ?below t n =
  if n < 0 then invalid_argf "Tlsf.alloc: %d bytes" n;
  if align <= 0 then invalid_argf "Tlsf.alloc: align %d" align;
  let req = Int.max min_block n in
  (* A block of [req + align - 1] bytes holds the request wherever it starts.
     None is larger than the range, and the sum could wrap past [max_int]. *)
  if req > t.length - align + 1 then None
  else
    match below with
    | Some below when below < t.base + t.length ->
        first_below t ~align ~below req
    | _ ->
        let need = round_class (req + align - 1) in
        (* Rounding wraps below 0 only for a range within 2^57 of [max_int]. *)
        if need < 0 then None
        else
          let f = first need in
          let b = suitable t f (second f need) in
          if b == nil then None else take_from t ~align b req

let free t x =
  match Tables.Address.find_opt t.blocks (x - t.base) with
  | Some b when b.state = Taken ->
      let p = b.prev and n = b.next in
      let join_p = p != nil && p.state = Free in
      let join_n = n != nil && n.state = Free in
      let m =
        if join_p then begin
          remove t p;
          absorb p b;
          p
        end
        else b
      in
      if join_n then begin
        remove t n;
        absorb m n
      end;
      insert t m;
      if join_p then Tables.Address.remove t.blocks b.start;
      if join_n then Tables.Address.remove t.blocks n.start;
      true
  | _ -> false
