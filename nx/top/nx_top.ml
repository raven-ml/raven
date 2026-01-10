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

let () = install_printer "Nx.pp_data"
