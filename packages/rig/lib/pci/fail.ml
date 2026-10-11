(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

exception Failed of string

let fail fmt = Printf.ksprintf (fun why -> raise (Failed why)) fmt

let step what f =
  try f ()
  with Unix.Unix_error (e, _, _) -> fail "%s: %s" what (Unix.error_message e)

let memlock = "raise the locked-memory limit (memlock), then log in again"

let err_released fn bus =
  invalid_arg (strf "Function.%s: %s is released" fn bus)

let result f = match f () with v -> Ok v | exception Failed why -> Error why

let bug what f =
  try f ()
  with Unix.Unix_error (e, _, _) ->
    failwith (strf "%s: %s" what (Unix.error_message e))
