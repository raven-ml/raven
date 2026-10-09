(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

type t = hold

(* Marks [e] held, under its device's lock, if its buffers' collection ends it:
   no hold has it, and it is no scratch. *)
let take (e : entry) =
  Dev.protect e.owner (fun () ->
      e.life = Collected
      && begin
        e.life <- Held;
        true
      end)

let give (e : entry) = Dev.protect e.owner (fun () -> e.life <- Collected)

(* Takes every entry of [es], or none: a refusal gives back those taken. *)
let rec take_all = function
  | [] -> ()
  | e :: rest -> (
      if not (take e) then
        invalid_arg "Rig.Hold.make: a buffer's memory is in a hold";
      try take_all rest
      with x ->
        give e;
        raise x)

let make ?(release = ignore) bs =
  List.iter (Buffer.check_live "Hold.make") bs;
  if List.exists (fun b -> b.mem.root.entry.life = Scratch) bs then
    invalid_arg "Rig.Hold.make: a buffer is a scratch";
  let entries =
    List.fold_left
      (fun acc b ->
        let m = b.mem.root in
        if m.entry == Memory.no_entry then Memory.ensure_entry m;
        if List.memq m.entry acc then acc else m.entry :: acc)
      [] bs
  in
  Atomic.set Memory.any_marked true;
  take_all entries;
  let st = Memory.stamps_new () in
  List.iter
    (fun (e : entry) ->
      if e.stamps <> 0 then Memory.stamps_absorb st e.stamps;
      Memory.stamps_ref st;
      e.stamps <- st)
    entries;
  let htoken =
    Memory.token Memory.holds_list
      (Release { stamps = st; release; generation = Dev.generation () })
      0 max_int (-1)
  in
  { hstamps = st; members = bs; htoken }
