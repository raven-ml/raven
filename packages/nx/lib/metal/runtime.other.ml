(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let get i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_metal.get: %d < 0" i);
  Error "METAL: Metal exists on macOS only"
