(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type version = int * int * int

type t = {
  target : version;
  gc : version;
  sdma : version;
  xccs : int;
  shader_engines : int;
  compute_units : int;
  scratch_slots : int;
}

let processor g =
  let major, minor, stepping = g.target in
  Printf.sprintf "gfx%d%x%x" major minor stepping
