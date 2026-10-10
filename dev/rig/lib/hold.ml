(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

type t = hold

(* A hold's stamps hold the uses of the submissions made with it, which its
   release waits for. *)
let make ?(release = ignore) value =
  let st = Memory.stamps_new () in
  let generation = Dev.generation () in
  let htoken =
    Memory.token Memory.holds_list
      (Release (Hold_release { stamps = st; value; release; generation }))
      0 max_int (-1)
  in
  { hstamps = st; htoken }
