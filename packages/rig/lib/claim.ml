(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let count (c : claim) = Atomic.Loc.get [%atomic.loc c.count]
let swap (c : claim) a b = Atomic.Loc.compare_and_set [%atomic.loc c.count] a b

(* Adds a read claim to [c]; the loop retries only a compare-and-set another
   domain's claim beat. *)
let rec read_claim fn c =
  let n = count c in
  if n < 0 then invalid_argf "Rig.%s: the memory is held exclusive" fn
  else if not (swap c n (n + Memory.one_claim)) then read_claim fn c

(* Takes a read claim off [c]. *)
let rec release_claim c =
  let n = count c in
  if n < 0 then invalid_arg "Rig.Claim: the memory is held exclusive"
  else if n < Memory.one_claim then
    invalid_arg "Rig.Claim: the memory has no read claim"
  else if not (swap c n (n - Memory.one_claim)) then release_claim c

(* Claims, then checks [b] under the claim. A donation on another domain that
   consumed the memory before this claim released its claims first, and the
   compare-and-set reads the count after that release, so the check sees the
   consumption. A check before the claim can pass while such a donation runs. *)
let take fn b =
  read_claim fn b.mem.claim;
  match
    Buffer.check_live fn b;
    Memory.check b.mem
  with
  | () -> ()
  | exception e ->
      release_claim b.mem.claim;
      raise e

(* Ends a claim of [take] in C, so that no poll point, and so no asynchronous
   exception, comes between [f]'s return and the release. *)
external drop : buffer -> unit = "caml_rig_claim_drop" [@@noalloc]

(* Host access to [b]'s bytes under a read claim: [f] runs on them once the
   work [access] follows is done, and the claim ends as [f] returns or
   raises. A loss raises where the wait for [access] meets it, so a read
   depends on the last write alone. *)
let scope fn (access : access) b k f =
  read_claim fn b.mem.claim;
  match
    Buffer.check_live fn b;
    Memory.check_owner b.mem;
    Buffer.wait_points b access;
    Buffer.array fn k b
  with
  | exception e ->
      drop b;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())
  | a -> (
      match f a with
      | r ->
          drop b;
          r
      | exception e ->
          drop b;
          Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ()))

let read_bytes b k f = scope "Buffer.read" Read b k f

let write_bytes b k f =
  let fn = "Buffer.write" in
  Buffer.check_live fn b;
  if Buffer.access b = Read then
    invalid_argf "Rig.%s: the buffer's memory admits only reads" fn;
  scope fn Read_write b k f

(* Sets [c]'s outside bit, or is [false] while claims hold it exclusive,
   consumed or not: shared memory is never held exclusive, so [exclusive] need
   not read the word. *)
let rec mark_outside c =
  let n = count c in
  n >= 0
  && (n land Memory.outside <> 0
     || swap c n (n lor Memory.outside)
     || mark_outside c)

let mark b =
  Buffer.check_live "Claim.share" b;
  if not (mark_outside b.mem.claim) then
    invalid_arg "Rig.Claim.share: the memory is held exclusive by claims";
  (* A consumption between the check and the mark killed [b]: the memory stays
     marked, which costs only its donations. *)
  Buffer.check_live "Claim.share" b

(* Memory a library hands out again is marked already: two loads answer, the
   word unheld and marked, then [b] live, with no call. *)
let share b =
  let c = b.mem.claim in
  let n = count c in
  if
    n < 0
    || n land Memory.outside = 0
    || Atomic.Loc.get [%atomic.loc c.generation] <> b.generation
  then mark b

(* The buffers [with_] was given, which it holds for reading: [ended] is set
   once its [f] returned or raised, before any claim is released, and from then
   on they hold nothing. A donated memory is held exclusive iff its claim word
   is negative: the read claim [with_] keeps on it lets no other holder make it
   so. *)
type t = {
  read : buffer list;
  donate : buffer list;
  mutable ended : bool; [@atomic]
}

let refuse () = invalid_arg "Rig.Claim.with_: a donated buffer overlaps another"

(* Up to this many donated buffers, comparing each with every other buffer
   costs less than sorting them all, and allocates nothing. *)
let pairwise = 8

let rec clear_of b = function
  | [] -> ()
  | b' :: bs ->
      if Buffer.overlaps b b' then refuse ();
      clear_of b bs

(* Each donated buffer against every read buffer and every donated one after
   it. *)
let rec clear_pairs read = function
  | [] -> ()
  | b :: bs ->
      clear_of b read;
      clear_of b bs;
      clear_pairs read bs

(* Orders spans by space, then first byte; the polymorphic compare took 40% of
   [with_]. *)
let by_place ((s, a, _), _) ((s', a', _), _) =
  if s <> s' then Int.compare s s' else Int.compare a a'

(* Each buffer's spans are its places ({!Buffer.places}). *)
let sort_overlaps read donate =
  let tag d b =
    List.map (fun (s, a) -> ((s, a, b.length), d)) (Buffer.places b)
  in
  let all =
    List.concat_map (tag false) read
    @ List.concat_map (tag true) donate
    |> List.sort by_place
  in
  (* In each space, a buffer overlaps an earlier one iff it starts before the
     furthest end of those: [ends] is that end over every earlier buffer,
     [donated] over the donated ones. *)
  let rec sweep space ends donated = function
    | [] -> ()
    | ((s, a, n), d) :: rest ->
        let ends, donated =
          if s = space then (ends, donated) else (min_int, min_int)
        in
        if a < donated || (d && a < ends) then refuse ();
        sweep s
          (Int.max ends (a + n))
          (if d then Int.max donated (a + n) else donated)
          rest
  in
  sweep (-1) min_int min_int all

(* Refuses a buffer of [donate] that overlaps another buffer. *)
let refuse_overlaps read donate =
  let n = List.length donate in
  if n = 0 then ()
  else if n <= pairwise then clear_pairs read donate
  else sort_overlaps read donate

let rec check_live = function
  | [] -> ()
  | b :: bs ->
      Buffer.check_live "Claim.with_" b;
      check_live bs

let rec release_all = function
  | [] -> ()
  | b :: bs ->
      release_claim b.mem.claim;
      release_all bs

(* Claims each buffer of [bs] for reading, in order, or none: a refusal
   releases those taken before it. *)
let rec take_all = function
  | [] -> ()
  | b :: bs -> (
      take "Claim.with_" b;
      match take_all bs with
      | () -> ()
      | exception e ->
          release_claim b.mem.claim;
          raise e)

(* Holds each buffer of [donate] exclusive that spans its memory and has no
   other claim on it: its own read claim is the only one. *)
let rec hold_all = function
  | [] -> ()
  | b :: bs ->
      if Buffer.spans b then
        ignore (swap b.mem.claim Memory.one_claim Memory.exclusive : bool);
      hold_all bs

(* Ends an exclusive claim, back to one read claim: outside the claims if the
   consumer exported the memory. Only an export of the consumer's buffer races
   it. *)
let rec unhold cl =
  let w = count cl in
  let back =
    if w = Memory.exported then Memory.one_claim lor Memory.outside
    else Memory.one_claim
  in
  if not (swap cl w back) then unhold cl

let rec unhold_all = function
  | [] -> ()
  | b :: bs ->
      if count b.mem.claim < 0 then unhold b.mem.claim;
      unhold_all bs

let finish c =
  c.ended <- true;
  unhold_all c.donate;
  release_all c.read;
  release_all c.donate

(* Allocates the claims' record alone, before it claims: no list of the
   caller's is copied, and no closure is made. From the first claim to the
   last release nothing allocates outside [f], so an exception raised at an
   allocation, as [Out_of_memory] or a signal handler's is, leaves no claim
   behind. *)
let with_ ~read ~donate f =
  check_live read;
  check_live donate;
  refuse_overlaps read donate;
  let c = { read; donate; ended = false } in
  take_all read;
  (match take_all donate with
  | () -> ()
  | exception e ->
      release_all read;
      raise e);
  match
    hold_all donate;
    f c
  with
  | r ->
      finish c;
      r
  | exception e ->
      finish c;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let rec names cl = function
  | [] -> false
  | b :: bs -> b.mem.claim == cl || names cl bs

let exclusive c b =
  let cl = b.mem.claim in
  (not c.ended) && count cl < 0 && names cl c.donate

let consume c ~why b =
  if c.ended then invalid_arg "Rig.Claim.consume: the claims' with_ returned";
  let cl = b.mem.claim in
  if not (names cl c.read || names cl c.donate) then
    invalid_arg "Rig.Claim.consume: the claims do not hold the memory";
  Buffer.check_live "Claim.consume" b;
  if not (Buffer.spans b) then
    invalid_arg "Rig.Claim.consume: the buffer is part of its memory";
  cl.why <- why;
  let g = Atomic.Loc.fetch_and_add [%atomic.loc cl.generation] 1 + 1 in
  (* After the generation: an export that finds the word consumed then finds
     every earlier buffer dead. The claims hold [cl], so a word exclusive is
     theirs. *)
  ignore (swap cl Memory.exclusive Memory.consumed);
  { b with generation = g }

let take ~why b =
  with_ ~read:[] ~donate:[ b ] (fun c ->
      if exclusive c b then Some (consume c ~why b) else None)
