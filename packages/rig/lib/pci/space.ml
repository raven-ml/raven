(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* The allocator is made by the first allocation: a vendor's space, which its
   library makes when it is linked, puts nothing in the heap of a program that
   drives none of its GPUs. *)
type t = {
  base : int;
  length : int;
  mutable tlsf : Tlsf.t option;
  lock : Mutex.t;
}

let create ~base n =
  if base < 0 || n < 0 || n > max_int - base then
    invalid_argf
      "Space.create: %d addresses at 0x%x, expected both 0 or more and an end \
       at most max_int"
      n base;
  { base; length = n; tlsf = None; lock = Mutex.create () }

let base s = s.base
let length s = s.length

(* [s]'s allocator. [s]'s lock is held. *)
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

(* Tlsf raises nothing on the arguments Space checks, so [alloc] and [free] take
   the lock without a handler. *)
let alloc ?(align = 0x1000) s n =
  if n <= 0 then
    invalid_argf "Space.alloc: %d addresses, expected more than 0" n;
  if not (is_pow2 align) then
    invalid_argf "Space.alloc: align %d is not a positive power of two" align;
  let align = Int.max (top_bit n) align in
  Mutex.lock s.lock;
  let a = Tlsf.alloc ~align (tlsf s) n in
  Mutex.unlock s.lock;
  a

let free s a =
  Mutex.lock s.lock;
  let freed = Tlsf.free (tlsf s) a in
  Mutex.unlock s.lock;
  if not freed then invalid_argf "Space.free: no range at 0x%x" a
