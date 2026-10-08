(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A machine's PCI functions.

   Needs Linux: elsewhere this machine lists no PCI functions.

   The drivers that boot a GPU over PCI, with no kernel driver, start from the
   machine's functions. Listing them, and asking which are GPUs, needs no
   privilege and changes nothing; taking one does both, and is left out. Every
   path numbers a vendor's GPUs in bus order among the functions its library
   calls GPUs, so GPU [i] is the same GPU whichever path opens it. *)

let kind ~vendor ~class_ =
  if Rig_amd.is_gpu ~vendor ~class_ then "AMD GPU"
  else if Rig_nv.is_gpu ~vendor ~class_ then "NVIDIA GPU"
  else ""

let () =
  let m = Rig_pci.Machine.this in
  match Rig_pci.Machine.functions m with
  | [] -> print_endline "no PCI functions on this machine"
  | fs ->
      Printf.printf "%d PCI functions, pages of %d bytes\n" (List.length fs)
        (Rig_pci.Machine.page m);
      List.iter
        (fun ({ bus; vendor; device; class_ } : Rig_pci.Machine.id) ->
          Printf.printf "  %s  %04x:%04x  class %06x  %s\n" bus vendor device
            class_ (kind ~vendor ~class_))
        fs;
      Printf.printf "AMD GPUs: %d; NVIDIA GPUs: %d\n" (Rig_amd_amdgpu.count ())
        (Rig_nv_nvidia.count ())
