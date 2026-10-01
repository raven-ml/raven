(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = string

let of_string s = s

let pp ppf b =
  Format.pp_print_string ppf "0x";
  String.iter (fun c -> Format.fprintf ppf "%02x" (Char.code c)) b
