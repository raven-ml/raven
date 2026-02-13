(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let install_printer name =
  let phrase =
    Printf.sprintf "#install_printer %s;;" name
    |> Lexing.from_string
    |> !Toploop.parse_toplevel_phrase
  in
  Toploop.execute_phrase false Format.err_formatter phrase |> ignore

let () = install_printer "Rune.pp_data"
