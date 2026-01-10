(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let install_printer (ident : string) =
  let lexbuf = Lexing.from_string ident in
  try
    let longident = Parse.longident lexbuf in
    Topdirs.dir_install_printer Format.err_formatter longident
  with ex ->
    Format.fprintf Format.err_formatter "Failed to install printer %s: %s@."
      ident (Printexc.to_string ex)

let pp_hugin_figure fmt figure =
  let image_data = Hugin.render figure in
  let base64_data = Base64.encode_string image_data in
  Format.fprintf fmt "![figure](data:image/png;base64,%s)" base64_data

let () = install_printer "Hugin_top.pp_hugin_figure"
