(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let allocated () =
  let minor, promoted, major = Gc.counters () in
  minor +. major -. promoted

let start = allocated ()
let before = allocated ()
