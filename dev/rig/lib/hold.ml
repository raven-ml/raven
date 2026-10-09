(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

type t = hold

(* Makes holds one at a time: a memory's stamps link to a hold once, and a
   make links all of its memory or none. *)
let making = Lock.create ()

let make ?(release = ignore) bs =
  List.iter (Buffer.check_live "Hold.make") bs;
  let entries =
    List.fold_left
      (fun acc b ->
        let m = b.mem.root in
        if m.entry == Memory.no_entry then Memory.ensure_entry m;
        if List.memq m.entry acc then acc else m.entry :: acc)
      [] bs
  in
  let st = Memory.stamps_new () in
  Lock.protect making (fun () ->
      if List.exists (fun (e : entry) -> Memory.held e.stamps) entries then begin
        Memory.stamps_unref st;
        invalid_arg "Rig.Hold.make: a buffer's memory is in a hold"
      end;
      List.iter (fun (e : entry) -> Memory.stamps_hold e.stamps st) entries);
  let htoken =
    Memory.token Memory.holds_list
      (Release { stamps = st; release; generation = Dev.generation () })
      0 max_int (-1)
  in
  { hstamps = st; members = bs; htoken }
