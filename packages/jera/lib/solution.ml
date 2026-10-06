(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a t = 'a Answer.t

type status = Answer.status =
  | Converged
  | Budget_spent
  | Not_bracketed
  | Not_finite
  | Stalled

let get = Answer.get
let best = Answer.best
let ok = Answer.ok
let is = Answer.is
let error = Answer.error
let evaluations = Answer.evaluations
let ptree = Answer.ptree
let pp = Answer.pp
