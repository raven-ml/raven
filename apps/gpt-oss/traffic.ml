(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [measure d f] is [f ()] and the bytes that copies moved into and out of
   [d]'s memory while it ran, read from rig's profile. *)
let measure d f =
  let y, events = Rig.Profile.take f in
  let count (i, o) = function
    | Rig.Profile.Copy { src; dst; bytes; _ } ->
        ( (if Rig.equal dst d then i + bytes else i),
          if Rig.equal src d then o + bytes else o )
    | _ -> (i, o)
  in
  (y, List.fold_left count (0, 0) events)
