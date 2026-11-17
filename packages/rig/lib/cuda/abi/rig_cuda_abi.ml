(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { symbol : string -> nativeint option }

let key : t Type.Id.t = Type.Id.make ()
let declare c = Rig_edge.Capability (key, c)
