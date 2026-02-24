(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Package configuration ───── *)

let raven_packages =
  [
    "nx";
    "nx.core";
    "nx.backend";
    "nx.c";
    "nx.buffer";
    "nx.io";
    "nx.datasets";
    "rune";
    "kaun";
    "kaun.models";
    "kaun.datasets";
    "hugin";
    "sowilo";
    "talon";
    "talon.csv";
    "brot";
    "fehu";
    "fehu.envs";
    "fehu.algorithms";
  ]

let raven_printers = [ "Nx.pp_data"; "Rune.pp_data" ]

(* ───── Setup ───── *)

let install_printer name =
  try
    let phrase =
      Printf.sprintf "#install_printer %s;;" name
      |> Lexing.from_string
      |> !Toploop.parse_toplevel_phrase
    in
    ignore (Toploop.execute_phrase false Format.err_formatter phrase)
  with _ -> ()

let setup_raven_toplevel () =
  Quill_top.initialize_if_needed ();
  (try
     Findlib.init ();
     List.iter
       (fun pkg ->
         match Findlib.package_directory pkg with
         | dir -> Topdirs.dir_directory dir
         | exception Findlib.No_such_package _ -> ())
       raven_packages
   with _ -> ());
  List.iter install_printer raven_printers

(* ───── Kernel ───── *)

let create ~on_event =
  let base = Quill_top.create ~on_event in
  let first_run = ref true in
  let ensure_setup () =
    if !first_run then (
      setup_raven_toplevel ();
      first_run := false)
  in
  let execute ~cell_id ~code =
    ensure_setup ();
    base.execute ~cell_id ~code
  in
  let complete ~code ~pos =
    ensure_setup ();
    base.complete ~code ~pos
  in
  { base with Quill.Kernel.execute; complete }
