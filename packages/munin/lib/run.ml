(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type status = Listing.status
type t = Listing.t

let id (r : t) = r.id
let dir (r : t) = r.dir
let experiment (r : t) = r.start.experiment
let name (r : t) = r.start.name
let parent (r : t) = r.start.parent
let params (r : t) = r.start.params
let steps (r : t) = r.start.steps
let provenance (r : t) = r.start.provenance
let diff (r : t) = r.diff
let started_at (r : t) = r.started_at
let tags (r : t) = r.tags
let notes (r : t) = r.notes
let status (r : t) = r.status
let ended_at (r : t) = r.ended_at
let updated_at (r : t) = r.updated_at
