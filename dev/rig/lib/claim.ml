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
  if n < 0 then invalid_arg "Rig.Claim.release: the memory is held exclusive"
  else if n < Memory.one_claim then
    invalid_arg "Rig.Claim.release: the memory has no read claim"
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

let read b = take "Claim.read" b
let release b = release_claim b.mem.claim

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
   on they hold nothing. *)
type t = {
  read : buffer list;
  donate : buffer list list;
  mutable exclusive : claim list;
  mutable ended : bool; [@atomic]
}

(* Where a buffer's bytes lie, for the overlap check, as [Buffer.overlaps]
   places them: a space and the first byte within it. This process's host
   memory is space 0, at its host addresses. Other memory is a space of its own,
   at its offsets: minus its stamps' address, which no other memory shares. A
   device's addresses or handles would not do: memories of a handle-named device
   lie at handles a few bytes apart. *)
let host_placed m = m.host >= 0 && Option.is_none m.dev.machine

let space b =
  let m = b.mem.root in
  if host_placed m then 0 else -m.entry.stamps

let first b =
  let m = b.mem.root in
  if host_placed m then m.host + b.offset else b.offset

let refuse () = invalid_arg "Rig.Claim.with_: a donated buffer overlaps another"

let overlap b b' =
  b.length > 0 && b'.length > 0
  && space b = space b'
  &&
  let a = first b and a' = first b' in
  a < a' + b'.length && a' < a + b.length

(* Up to this many donated buffers, comparing each with every other buffer
   costs less than sorting them all, and allocates nothing. *)
let pairwise = 8

let rec clear_of b = function
  | [] -> ()
  | b' :: bs ->
      if overlap b b' then refuse ();
      clear_of b bs

let rec clear_of_groups b = function
  | [] -> ()
  | g :: gs ->
      clear_of b g;
      clear_of_groups b gs

(* Each donated buffer of [g] against every read buffer and every donated one
   after it. *)
let rec clear_group read later = function
  | [] -> ()
  | b :: bs ->
      clear_of b read;
      clear_of b bs;
      clear_of_groups b later;
      clear_group read later bs

let rec clear_pairs read = function
  | [] -> ()
  | g :: gs ->
      clear_group read gs g;
      clear_pairs read gs

let rec count_donated n = function
  | [] -> n
  | g :: gs -> count_donated (n + List.length g) gs

(* Orders spans by space, then first byte; the polymorphic compare took 40% of
   [with_]. *)
let by_place ((s, a, _), _) ((s', a', _), _) =
  if s <> s' then Int.compare s s' else Int.compare a a'

(* Refuses a buffer of [donate] that overlaps another buffer, sorting the
   buffers by where they lie and sweeping: n log n. *)
let sort_overlaps read donate =
  let tag d b = ((space b, first b, b.length), d) in
  let all =
    List.map (tag false) read @ List.map (tag true) (List.concat donate)
    |> List.filter (fun ((_, _, n), _) -> n > 0)
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
  let n = count_donated 0 donate in
  if n = 0 then ()
  else if n <= pairwise then clear_pairs read donate
  else sort_overlaps read donate

let rec check_live = function
  | [] -> ()
  | b :: bs ->
      Buffer.check_live "Claim.with_" b;
      check_live bs

let rec check_live_groups = function
  | [] -> ()
  | g :: gs ->
      check_live g;
      check_live_groups gs

let rec release_all = function
  | [] -> ()
  | b :: bs ->
      release_claim b.mem.claim;
      release_all bs

let rec release_groups = function
  | [] -> ()
  | g :: gs ->
      release_all g;
      release_groups gs

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

let rec take_groups = function
  | [] -> ()
  | g :: gs -> (
      take_all g;
      match take_groups gs with
      | () -> ()
      | exception e ->
          release_all g;
          raise e)

let rec all_span = function [] -> true | b :: bs -> Buffer.spans b && all_span bs

(* Makes the memory of every buffer of [bs], each held by one reader, exclusive,
   or none: a swap that another domain's claim beat undoes the ones before
   it. *)
let rec exclusive_all = function
  | [] -> true
  | b :: rest ->
      let cl = b.mem.claim in
      swap cl Memory.one_claim Memory.exclusive
      && (exclusive_all rest
         || begin
           ignore (swap cl Memory.exclusive Memory.one_claim);
           false
         end)

let rec hold_all c = function
  | [] -> ()
  | b :: bs ->
      c.exclusive <- b.mem.claim :: c.exclusive;
      hold_all c bs

(* Holds each value of [donate] exclusive that spans its memories and has no
   other claim on them. *)
let rec hold_groups c = function
  | [] -> ()
  | g :: gs ->
      if all_span g && exclusive_all g then hold_all c g;
      hold_groups c gs

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
  | cl :: cls ->
      unhold cl;
      unhold_all cls

let finish c =
  c.ended <- true;
  unhold_all c.exclusive;
  release_all c.read;
  release_groups c.donate

(* Allocates the claims' record and a cell per exclusive buffer: no list of
   the caller's is copied, and no closure is made. *)
let with_ ~read ~donate f =
  check_live read;
  check_live_groups donate;
  refuse_overlaps read donate;
  take_all read;
  (match take_groups donate with
  | () -> ()
  | exception e ->
      release_all read;
      raise e);
  let c = { read; donate; exclusive = []; ended = false } in
  hold_groups c donate;
  match f c with
  | r ->
      finish c;
      r
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      finish c;
      Printexc.raise_with_backtrace e bt

let exclusive c b = (not c.ended) && List.memq b.mem.claim c.exclusive

let rec names cl = function
  | [] -> false
  | b :: bs -> b.mem.claim == cl || names cl bs

let rec names_groups cl = function
  | [] -> false
  | g :: gs -> names cl g || names_groups cl gs

let consume c ~why b =
  if c.ended then invalid_arg "Rig.Claim.consume: the claims' with_ returned";
  let cl = b.mem.claim in
  if not (names cl c.read || names_groups cl c.donate) then
    invalid_arg "Rig.Claim.consume: the claims do not hold the memory";
  Buffer.check_live "Claim.consume" b;
  if not (Buffer.spans b) then
    invalid_arg "Rig.Claim.consume: the buffer is part of its memory";
  cl.why <- why;
  let g = Atomic.Loc.fetch_and_add [%atomic.loc cl.generation] 1 + 1 in
  (* After the generation: an export that finds the word consumed then finds
     every earlier buffer dead. *)
  if List.memq cl c.exclusive then
    ignore (swap cl Memory.exclusive Memory.consumed);
  { b with generation = g }
