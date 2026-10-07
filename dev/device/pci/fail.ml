(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

exception Failed of string

let () = Callback.register_exception "Device_pci.Failed" (Failed "")
let fail fmt = Printf.ksprintf (fun why -> raise (Failed why)) fmt

let step what f =
  try f ()
  with Unix.Unix_error (e, _, _) -> fail "%s: %s" what (Unix.error_message e)
