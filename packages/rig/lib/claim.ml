(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let reading fn b f =
  Memory.claim_add b;
  match
    Buffer.check_live fn b;
    f ()
  with
  | r ->
      Memory.claim_drop b;
      r
  | exception e ->
      Memory.claim_drop b;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* [reading] of two buffers with no closure, for a staged copy's sides. The
   claims are counted by the store after each, for the handler: a call of an
   OCaml function polls at its entry in bytecode. *)
let reading2 fn a b f x =
  let held = ref 0 in
  match
    Memory.claim_add a;
    held := 1;
    Memory.claim_add b;
    held := 2;
    Buffer.check_live fn a;
    Buffer.check_live fn b;
    f a b x
  with
  | r ->
      Memory.claim_drop a;
      Memory.claim_drop b;
      r
  | exception e ->
      if !held > 0 then Memory.claim_drop a;
      if !held > 1 then Memory.claim_drop b;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* [Buffer.wait]'s checks and wait under a read claim, which it returns
   holding. A loss raises where the wait for [access] meets it, so a read
   depends on every device's last write alone. The caller's next step is the
   trap whose handler drops the claim: an OCaml function applied before it
   polls in bytecode. *)
let wait fn (access : access) b =
  Memory.claim_add b;
  match
    Buffer.check_live fn b;
    Memory.check_owner b.mem;
    Buffer.wait_points b access
  with
  | () -> ()
  | exception e ->
      Memory.claim_drop b;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let stored fn b f x =
  wait fn Read_write b;
  match Memory.stored b.mem.alloc f x with
  | r ->
      Memory.claim_drop b;
      r
  | exception e ->
      Memory.claim_drop b;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* Host access to [b]'s bytes under a read claim: [f] runs on them once the
   work [access] follows is done, and the claim ends as [f] returns or
   raises. A write of [Mapped] memory runs on its window, its stores and the
   flush after them one counted call, so a signal pending as [f] ends lands
   after the flush. *)
let scope fn access b k f =
  wait fn access b;
  match
    let window = access = Read_write && Buffer.windowed b in
    let a = Buffer.array ~window fn k b in
    if window then Memory.stored b.mem.alloc f a else f a
  with
  | r ->
      Memory.claim_drop b;
      r
  | exception e ->
      Memory.claim_drop b;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let read_bytes b k f =
  Dev.check_held "Buffer.read";
  scope "Buffer.read" Read b k f

let write_bytes b k f =
  let fn = "Buffer.write" in
  Dev.check_held fn;
  Buffer.check_live fn b;
  if Buffer.access b = Read then
    invalid_argf "Rig.%s: the buffer admits only reads" fn;
  scope fn Read_write b k f

let mark b =
  Buffer.check_live "Buffer.share" b;
  Memory.mark_outside b.mem.alloc;
  (* A take between the check and the mark killed [b]: the memory stays
     marked, which costs only its takes. *)
  Buffer.check_live "Buffer.share" b

(* Memory a library hands out again is marked already: loads answer, the word
   marked and no step holding it, then [b] live, with no call. *)
let share b =
  let c = b.mem.alloc in
  let n = Atomic.Loc.get [%atomic.loc c.count] in
  if
    n < 0
    || n land Memory.outside = 0
    || Atomic.Loc.get [%atomic.loc c.generation] <> b.generation
    || Memory.lease_ended b.mem
  then mark b

(* Takes *)

let rec check_live fn = function
  | [] -> ()
  | b :: bs ->
      Buffer.check_live fn b;
      check_live fn bs

let rec spans_all = function
  | [] -> true
  | b :: bs -> Buffer.spans b && spans_all bs

let raised b = { b with generation = b.generation + 2 }

(* [take]'s step: -1 if it took every buffer of [bs], -2 if a claim, a hold
   or a bit held a word or a memory is twice in [bs], or the index of a
   buffer it found dead or lost, having changed nothing. It waits out
   another take or free of the memory. It runs no OCaml code and allocates
   nothing, but records the reason with [caml_modify], so it is no
   [noalloc] external. *)
external took : buffer list -> string -> int = "caml_rig_buffers_take"

(* The result is built before the step, so that no allocation after it, where
   an asynchronous exception lands, leaves the memory taken and no buffer
   over it. The buffer [took] names raises: dead, or lost, which the check
   finds under a read claim, so that no free returns the stamps it walks. A
   buffer that passes both now, as when the step met a loss the check sees
   otherwise, is tried again. *)
let rec take ~why bs =
  let fn = "Buffer.take" in
  check_live fn bs;
  if not (spans_all bs) then None
  else
    let taken = Some (List.map raised bs) in
    match took bs why with
    | -1 -> taken
    | -2 -> None
    | i ->
        let b = List.nth bs i in
        Buffer.check_live fn b;
        reading fn b (fun () -> Memory.check b.mem);
        take ~why bs

(* [free]'s step: 1 if it freed [b]'s memory, 0 if a claim, a hold or a bit
   of the word kept it, 2 if [b] is dead. *)
external freed : buffer -> string -> int = "caml_rig_buffer_free"

let free ?(why = "freed") b =
  let fn = "Buffer.free" in
  Dev.check_held fn;
  Buffer.check_live fn b;
  Buffer.spans b
  &&
  match freed b why with
  | 0 -> false
  | 1 ->
      Memory.release_early b.mem.alloc;
      true
  | _ -> Buffer.refuse_dead fn b

(* The lease's step: 1 if it ended [b]'s lease, 0 if a claim or a hold
   through the borrow kept it, 2 if [b] is dead. *)
external lease_freed : buffer -> string -> int = "caml_rig_lease_free"

(* A borrow's end drops its hold of its mapping as its collection would
   ([Memory.section-leases]). *)
let unborrow ?(why = "unborrowed") b =
  let fn = "Buffer.unborrow" in
  Dev.check_held fn;
  Buffer.check_live fn b;
  let l =
    match b.mem.lease with
    | Some l -> l
    | None -> invalid_arg "Rig.Buffer.unborrow: the buffer is no borrow"
  in
  if b.offset <> l.lease_offset || b.length <> l.lease_length then
    invalid_arg "Rig.Buffer.unborrow: the buffer is part of its borrow";
  match lease_freed b why with
  | 0 -> false
  | 1 ->
      Memory.end_lease l;
      true
  | _ -> Buffer.refuse_dead fn b

(* Keeps *)

(* A keep keeps its token: collected, the token puts the record on the list
   of holds, whose next drain releases them. The record is made and the token
   over it before the holds are taken, in one C step that marks each taken in
   the record, so an asynchronous exception anywhere leaves either no hold or
   a token that releases them. *)
type keep = { token : token; record : holds }

let no_record =
  { allocs = Weak.create 0; leases = Weak.create 0; taken = Bytes.empty }

(* Holds the memory of each buffer of [bs] and marks it in [taken]: -1, or
   the index of a dead buffer, holding nothing. *)
external take_holds : buffer array -> Bytes.t -> int = "caml_rig_claim_holds"
[@@noalloc]

let kept fn bs =
  Array.iter (Buffer.check_live fn) bs;
  let n = Array.length bs in
  if n = 0 then { token = Memory.no_hold; record = no_record }
  else begin
    let allocs = Weak.create n and leases = Weak.create n in
    for i = 0 to n - 1 do
      Weak.set allocs i (Some bs.(i).mem.alloc);
      Weak.set leases i bs.(i).mem.lease
    done;
    let record = { allocs; leases; taken = Bytes.make n '\000' } in
    let token = Memory.hold_token record in
    let i = take_holds bs record.taken in
    if i >= 0 then Buffer.refuse_dead fn bs.(i);
    { token; record }
  end

let keep bs = kept "Buffer.keep" (Array.of_list bs)
let release k = Memory.release_hold_now k.token k.record
