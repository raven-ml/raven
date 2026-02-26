(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Io_error of string
  | Format_error of string
  | Unsupported_dtype
  | Unsupported_shape
  | Missing_entry of string
  | Other of string

let to_string = function
  | Io_error msg -> Printf.sprintf "I/O error: %s" msg
  | Format_error msg -> Printf.sprintf "Format error: %s" msg
  | Unsupported_dtype -> "Unsupported dtype"
  | Unsupported_shape -> "Unsupported shape"
  | Missing_entry name -> Printf.sprintf "Missing entry: %s" name
  | Other msg -> msg

let fail_msg fmt = Printf.ksprintf failwith fmt
