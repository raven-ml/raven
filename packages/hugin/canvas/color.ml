(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { r : float; g : float; b : float; a : float }

let v ?(a = 1.) r g b = { r; g; b; a }
let black = v 0. 0. 0.
let white = v 1. 1. 1.
let transparent = v ~a:0. 0. 0. 0.
