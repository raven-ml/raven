(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A report parses all eight pinned R9700 images. Warm the file cache first so
   the row measures discovery, file identity checks and payload parsing. *)

let fixtures = "../../../test/amd/fixtures"

let table =
  In_channel.with_open_bin
    (Filename.concat fixtures "r9700.bin")
    In_channel.input_all

let report firmware () =
  match
    Rig_amd_pci.report
      ~firmware:(Thumper.black_box firmware)
      (Thumper.black_box table)
  with
  | Ok r -> r
  | Error why -> failwith why

(* The firmware is the machine's, in rig's firmware directory, and lives
   outside the repository. Without it the suite skips, as a GPU suite does
   without its GPU. *)
let () =
  let report = report (Option.to_list (Rig_pci.Firmware.directory ())) in
  let r = report () in
  if List.length r.images <> 8 then failwith "the R9700 needs eight images";
  let missing (i : Rig_amd_pci.image) = Option.is_none i.found in
  if List.exists missing r.images then
    print_endline "rig_amd_pci: skipped, rig's firmware directory lacks images"
  else
    exit
      (Thumper.run "rig_amd_pci" [ Thumper.bench "report/r9700-pinned" report ])
