(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPUs numbered on machines that fixture trees show, and the opens that answer
   before a GPU is touched. *)

open Windtrap
module Tree = Rig_pci_support.Tree

(* A machine whose NVIDIA GPUs are a display controller and a 3D controller,
   beside an NVIDIA audio function and another vendor's GPU, listed out of bus
   order. *)
let machine () =
  let fn bus vendor class_ = { (Tree.gpu bus) with vendor; class_ } in
  Tree.make
    [
      fn "0000:83:00.0" 0x10de 0x030200;
      fn "0000:03:00.1" 0x10de 0x040300;
      fn "0000:03:00.0" 0x10de 0x030000;
      fn "0000:01:00.0" 0x1002 0x030000;
    ]

let numbering =
  group "numbering"
    [
      test "GPUs are NVIDIA's display and 3D controllers in bus order"
        (fun () ->
          let machine = Rig_pci.Machine.at (machine ()) in
          equal int 2 (Rig_nv_pci.count ~machine ()));
      test "a machine with no PCI functions has no GPU" (fun () ->
          let machine = Rig_pci.Machine.at (Tree.make []) in
          equal int 0 (Rig_nv_pci.count ~machine ()));
      cases ~name:fst "GPUs are named by number"
        [
          ("0", (0, "NV-PCI")); ("1", (1, "NV-PCI:1")); ("12", (12, "NV-PCI:12"));
        ]
        (fun (_, (i, name)) -> equal string name (Rig_nv_pci.device_name i));
    ]

let opening =
  group "opening"
    [
      test "an open past the last GPU names the GPU and the count" (fun () ->
          let machine = Rig_pci.Machine.at (machine ()) in
          match Rig_nv_pci.open_ ~machine ~firmware:[] 2 with
          | Ok _ -> fail "GPU 2 opened"
          | Error why ->
              equal string "NV-PCI:2: no such GPU; the machine has 2" why);
      cases ~name:fst "a negative GPU number raises"
        [
          ("device_name", fun () -> ignore (Rig_nv_pci.device_name (-1)));
          ("open_", fun () -> ignore (Rig_nv_pci.open_ ~firmware:[] (-1)));
          ("detach", fun () -> ignore (Rig_nv_pci.detach (-1)));
          ("attach", fun () -> ignore (Rig_nv_pci.attach (-1)));
          ("reset", fun () -> ignore (Rig_nv_pci.reset (-1)));
        ]
        (fun (_, f) -> raises_match Exn.invalid_arg f);
    ]

(* Stopping *)

let gpus () =
  Rig_pci.Gpus.make ~memory_bar:1
    ~nodes:(fun ~read:_ _ -> [])
    (fun (id : Rig_pci.Machine.id) ->
      Rig_nv.is_gpu ~vendor:id.vendor ~class_:id.class_)

(* [stopped gpu] opens [gpu], runs [before], and gives the GPU up, recording the
   releases that preceded each unload. *)
let stopped ?(before = ignore) (gpu : Rig_nv_pci_support.gpu) =
  let g = gpus () in
  let unloads = ref [] in
  let hold, fn =
    match
      Rig_pci.Gpus.open_ g gpu.machine 0 ~at_exit:ignore (fun h fn ->
          Ok (h, fn))
    with
    | Ok v -> v
    | Error why -> failf "GPU 0 did not open: %s" why
  in
  before ();
  let s =
    Rig_nv_pci.give_up hold fn ~unload:(fun () ->
        unloads := !(gpu.released) :: !unloads)
  in
  (g, s, !unloads)

let state =
  Testable.contramap
    (function `Stopped -> "`Stopped" | `Unknown -> "`Unknown")
    string

let stopping =
  group "stopping"
    [
      test "a stop unloads the GPU before giving it back, and loses it"
        (fun () ->
          let gpu = Rig_nv_pci_support.gpu () in
          let g, s, unloads = stopped gpu in
          equal state `Stopped s;
          equal (list int) ~msg:"releases before each unload" [ 0 ] unloads;
          equal int ~msg:"releases" 1 !(gpu.released);
          match
            Rig_pci.Gpus.open_ g gpu.machine 0 ~at_exit:ignore (fun _ _ ->
                Ok ())
          with
          | Ok () -> fail "the GPU opened again without a reset"
          | Error _ -> ());
      test
        "a stop on a failed machine gives the GPU back unknown, unloading \
         nothing" (fun () ->
          let gpu = Rig_nv_pci_support.gpu () in
          let before () = Rig_pci_support.break gpu.far in
          let _, s, unloads = stopped ~before gpu in
          equal state `Unknown s;
          equal (list int) ~msg:"unloads" [] unloads;
          equal int ~msg:"releases" 1 !(gpu.released));
      test
        "a stop of a GPU that left the bus gives it back unknown, unloading \
         nothing" (fun () ->
          let gone = ref false in
          let gpu =
            Rig_nv_pci_support.gpu
              ~vendor:(fun () -> if !gone then 0xffff else 0x10de)
              ()
          in
          let before () = gone := true in
          let _, s, unloads = stopped ~before gpu in
          equal state `Unknown s;
          equal (list int) ~msg:"unloads" [] unloads;
          equal int ~msg:"releases" 1 !(gpu.released));
    ]

let () = exit (run "rig_nv_pci" [ numbering; opening; stopping ])
