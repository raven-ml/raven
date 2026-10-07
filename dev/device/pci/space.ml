(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

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
  if base < 0 || n < 0 then
    invalid_arg (Printf.sprintf "Space.create: %d addresses at 0x%x" n base);
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

let is_power_of_two n = n > 0 && n land (n - 1) = 0

(* The largest power of two not above [n > 0]. *)
let top_bit n =
  let rec go b = if b > n / 2 then b else go (b * 2) in
  go 1

let alloc ?(align = 0x1000) s n =
  if n <= 0 then invalid_arg (Printf.sprintf "Space.alloc: %d addresses" n);
  if not (is_power_of_two align) then
    invalid_arg (Printf.sprintf "Space.alloc: align %d" align);
  let align = Int.max (top_bit n) align in
  Mutex.protect s.lock (fun () -> Tlsf.alloc ~align (tlsf s) n)

let free s a =
  Mutex.protect s.lock @@ fun () ->
  try Tlsf.free (tlsf s) a
  with Invalid_argument _ ->
    invalid_arg (Printf.sprintf "Space.free: no range at 0x%x" a)
