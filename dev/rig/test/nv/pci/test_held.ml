(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The device file NVIDIA's kernel driver serves a GPU through, from the
   information file its procfs writes (kernel-open/nvidia/nv-procfs.c,
   open-gpu-kernel-modules 570.144: "Device Minor: \t %u\n"). *)

open Windtrap
module Held = Rig_nv_pci.Held

let bus = "0000:01:00.0"
let path = "proc/driver/nvidia/gpus/0000:01:00.0/information"

let information minor =
  String.concat ""
    [
      "Model: \t\t NVIDIA RTX 5000 Ada Generation\n";
      "Bus Location: \t 0000:01:00.0\n";
      Printf.sprintf "Device Minor: \t %u\n" minor;
      "GPU Excluded:\t No\n";
    ]

open Rig_pci_support

(* A machine whose files under its root are [files], as (path, contents). *)
let root_of files =
  let root = Tree.make [] in
  List.iter (fun (p, contents) -> Tree.add root p contents) files;
  root

let test_minor =
  prop "the minor names the device file" (Gen.int_range 0 255) (fun minor ->
      equal (list string)
        [ Printf.sprintf "dev/nvidia%d" minor ]
        (Held.nodes ~root:(root_of [ (path, information minor) ]) bus))

let test_none =
  cases "a GPU the driver does not serve has no device file"
    ~name:(fun (n, _) -> n)
    [
      ("no information file", []);
      ("no minor", [ (path, "Model: \t\t NVIDIA\n") ]);
      ( "another GPU's file",
        [ ("proc/driver/nvidia/gpus/0000:02:00.0/information", information 1) ]
      );
    ]
    (fun (_, files) ->
      equal (list string) [] (Held.nodes ~root:(root_of files) bus))

let () =
  exit
  @@ run "rig_nv_pci.held"
       [ group ~timeout:10. "nodes" [ test_minor; test_none ] ]
