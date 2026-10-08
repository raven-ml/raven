(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

type t = hold

let make ?(release = ignore) bs =
  let st = Memory.stamps_new () in
  List.iter
    (fun b ->
      Buffer.check_live "Hold.make" b;
      let m = b.mem.root in
      if m.entry.held then
        invalid_arg "Rig.Hold.make: a buffer's memory is in a hold")
    bs;
  List.iter
    (fun b ->
      let m = b.mem.root in
      if m.entry == Memory.no_entry then Memory.ensure_entry m;
      let e = m.entry in
      if not e.held then begin
        if e.stamps <> 0 then Memory.stamps_absorb st e.stamps;
        Memory.stamps_ref st;
        e.stamps <- st;
        e.held <- true
      end)
    bs;
  let htoken =
    Memory.token Memory.holds_list
      (Release { stamps = st; release })
      0 max_int (-1)
  in
  { hstamps = st; members = bs; htoken }
