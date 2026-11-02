(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* The allocator is made by the first allocation: a vendor's space, which its
   library makes when it is linked, puts nothing in the heap of a program that
   drives none of its GPUs. *)
type t = { base : int; length : int; mutable tlsf : Tlsf.t option }

(* A range and the holds on it. A hold is one holder's count of the range,
   on its holder's list. The range goes back to its space with its last
   hold. *)
type range = { space : t; mutable start : int; mutable users : int }

type hold = {
  range : range;
  holder : holder;
  mutable live : bool;
  mutable prev : hold;
  mutable next : hold;
}

and holder = { mutable first : hold }

(* One lock for the ranges of every space and their holders, since a holder
   may hold ranges of several spaces. A hold's count and its link on its
   holder's list change together, with nothing between that allocates or
   polls: the hold is made first, and [Tlsf.alloc] allocates before it
   changes its blocks and [Tlsf.free] not at all. So an exception, an
   asynchronous one too, leaves each range counted by exactly the live holds
   on it. *)
let lock = Lock.create ()

let rec nil =
  {
    range = nil_range;
    holder = nil_holder;
    live = false;
    prev = nil;
    next = nil;
  }

and nil_holder = { first = nil }
and nil_range = { space = nil_space; start = -1; users = 0 }
and nil_space = { base = 0; length = 0; tlsf = None }

let create ~base n =
  if base < 0 || n < 0 || n > max_int - base then
    invalid_argf
      "Space.create: %d addresses at 0x%x, expected both 0 or more and an end \
       at most max_int"
      n base;
  { base; length = n; tlsf = None }

let base s = s.base
let length s = s.length
let holder () = { first = nil }
let address h = h.range.start

(* [s]'s allocator. [lock] is held. *)
let tlsf s =
  match s.tlsf with
  | Some t -> t
  | None ->
      let t = Tlsf.create ~base:s.base s.length in
      s.tlsf <- Some t;
      t

let is_pow2 n = n > 0 && n land (n - 1) = 0

(* The largest power of two not above [n > 0]. *)
let top_bit n =
  let rec go b = if b > n / 2 then b else go (b * 2) in
  go 1

(* Links the hold [x] first on its holder's list. *)
let link x =
  let h = x.holder in
  x.prev <- nil;
  x.next <- h.first;
  if h.first != nil then h.first.prev <- x;
  h.first <- x

let unlink x =
  if x.prev != nil then x.prev.next <- x.next else x.holder.first <- x.next;
  if x.next != nil then x.next.prev <- x.prev

(* Ends the live hold [x], and gives its range back with the last hold.
   [lock] is held. A range a hold counts was taken, so its space has an
   allocator, and [Tlsf.free] finds its block. *)
let give_back x =
  if x.live then begin
    let r = x.range in
    x.live <- false;
    unlink x;
    r.users <- r.users - 1;
    if r.users = 0 then
      match r.space.tlsf with
      | Some t -> ignore (Tlsf.free t r.start : bool)
      | None -> ()
  end

(* The hold and its answer are made before the lock: from the take of the
   range to the hold's link, nothing allocates or polls. *)
let alloc ?(align = 0x1000) s h n =
  if n <= 0 then
    invalid_argf "Space.alloc: %d addresses, expected more than 0" n;
  if not (is_pow2 align) then
    invalid_argf "Space.alloc: align %d is not a positive power of two" align;
  let align = Int.max (top_bit n) align in
  let range = { space = s; start = -1; users = 0 } in
  let x = { range; holder = h; live = false; prev = nil; next = nil } in
  let got = Some x in
  Lock.protect lock @@ fun () ->
  let t = tlsf s in
  match Tlsf.alloc ~align t n with
  | None -> None
  | Some a ->
      range.start <- a;
      range.users <- 1;
      x.live <- true;
      link x;
      got

let share h x =
  let y =
    { range = x.range; holder = h; live = false; prev = nil; next = nil }
  in
  Lock.protect lock (fun () ->
      if not x.live then
        invalid_argf "Space.share: the range at 0x%x was given back"
          x.range.start;
      x.range.users <- x.range.users + 1;
      y.live <- true;
      link y);
  y

let free x = Lock.protect lock (fun () -> give_back x)

(* Each hold goes as one step, so an exception between two leaves the rest
   to the next call. *)
let release h =
  Lock.protect lock (fun () ->
      while h.first != nil do
        give_back h.first
      done)

let taken s =
  Lock.protect lock (fun () ->
      match s.tlsf with None -> 0 | Some t -> Tlsf.taken t)
