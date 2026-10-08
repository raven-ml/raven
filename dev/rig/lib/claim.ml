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
  else if not (swap c n (n + 1)) then read_claim fn c

let take fn b =
  Buffer.check_live fn b;
  Memory.check_points (Memory.stamps b.mem);
  read_claim fn b.mem.claim

let read b = take "Claim.read" b

let rec release_claim c =
  let n = count c in
  if n = 0 then invalid_arg "Rig.Claim.release: the memory has no read claim"
  else if n < 0 then
    invalid_arg "Rig.Claim.release: the memory is held exclusive"
  else if not (swap c n (n - 1)) then release_claim c

let release b = release_claim b.mem.claim

type t = { reads : claim list; mutable exclusive : claim list }

(* Where a buffer's bytes lie, for the overlap check: a space (0 for this
   process's host memory, else 1 + its memory's device index) and the first and
   last bytes. Memory with no address is placed by its handle. *)
let span b =
  let m = b.mem.root in
  let n = Buffer.length b in
  if m.host >= 0 && m.dev.machine = None then (0, m.host + b.offset, n)
  else
    let base =
      if m.address >= 0 then m.address else Nativeint.to_int m.handle
    in
    (1 + m.dev.index, base + b.offset, n)

(* Orders spans by space, then first byte; the polymorphic compare took 40% of
   [with_]. *)
let by_place ((s, a, _), _) ((s', a', _), _) =
  if s <> s' then Int.compare s s' else Int.compare a a'

(* Refuses a buffer of [donate] that overlaps another buffer, sorting the
   buffers by where they lie and sweeping: n log n. *)
let refuse_overlaps read donate =
  let tag d b = (span b, d) in
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
        if a < donated || (d && a < ends) then
          invalid_arg "Rig.Claim.with_: a donated buffer overlaps another";
        sweep s
          (Int.max ends (a + n))
          (if d then Int.max donated (a + n) else donated)
          rest
  in
  sweep (-1) min_int min_int all

let with_ ~read ~donate f =
  List.iter (Buffer.check_live "Claim.with_") read;
  List.iter (List.iter (Buffer.check_live "Claim.with_")) donate;
  refuse_overlaps read donate;
  let taken = ref [] in
  let undo () = List.iter release_claim !taken in
  (try
     List.iter
       (fun b ->
         take "Claim.with_" b;
         taken := b.mem.claim :: !taken)
       (read @ List.concat donate)
   with e ->
     undo ();
     raise e);
  let c = { reads = !taken; exclusive = [] } in
  List.iter
    (fun group ->
      let claims = List.map (fun b -> b.mem.claim) group in
      if
        List.for_all Buffer.spans group
        && List.for_all (fun cl -> count cl = 1) claims
        && List.for_all (fun cl -> swap cl 1 (-1)) claims
      then c.exclusive <- claims @ c.exclusive)
    donate;
  let finish () =
    List.iter (fun cl -> ignore (swap cl (-1) 1)) c.exclusive;
    List.iter release_claim c.reads
  in
  Fun.protect ~finally:finish (fun () -> f c)

let exclusive c b = List.memq b.mem.claim c.exclusive

let consume c ~why b =
  if not (List.memq b.mem.claim c.reads) then
    invalid_arg "Rig.Claim.consume: the claims do not hold the memory";
  Buffer.check_live "Claim.consume" b;
  if not (Buffer.spans b) then
    invalid_arg "Rig.Claim.consume: the buffer is part of its memory";
  let cl = b.mem.claim in
  cl.why <- why;
  let g = Atomic.Loc.fetch_and_add [%atomic.loc cl.generation] 1 + 1 in
  { b with generation = g }
