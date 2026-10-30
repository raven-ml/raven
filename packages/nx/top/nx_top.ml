(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The toplevel lets loaded code call Toploop and Topprinters but not Longident
   or Location, so the identifier is built as data. *)
let noloc txt =
  let loc =
    {
      Location.loc_start = Lexing.dummy_pos;
      loc_end = Lexing.dummy_pos;
      loc_ghost = true;
    }
  in
  { Location.txt; loc }

(* What [#install_printer M.f] does, with its error as a result: the directive
   prints the error and reports success. *)
let install_printer m f =
  let lid = Longident.Ldot (noloc (Longident.Lident m), noloc f) in
  let env = !Toploop.toplevel_env in
  match Topprinters.install Toploop.eval_value_path env lid with
  | Ok () -> ()
  | Error e ->
      let why = Format.asprintf "%a" Topprinters.report_error e in
      Printf.ksprintf failwith "nx.top: cannot install the printer %s.%s: %s" m
        f (String.trim why)

let () = install_printer "Nx" "pp"
