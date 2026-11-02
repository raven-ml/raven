(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Version.t

let name (a : t) = a.name
let version (a : t) = a.version
let digest (a : t) = a.digest
let path (a : t) = a.path
let size_bytes (a : t) = a.size_bytes
let metadata (a : t) = a.metadata
let run (a : t) = a.run
let step (a : t) = a.step
let created_at (a : t) = a.created_at
