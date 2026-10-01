(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The AMD runtime on a machine without an AMD GPU: opening refuses with a
   message, fixes no interface, and the queries refuse devices of other
   vendors. *)

open Windtrap

let no_gpu () =
  if
    Nx_amd_device.count ~interface:Kernel () > 0
    || Nx_amd_device.count ~interface:Pci () > 0
  then skip ~reason:"this machine has an AMD GPU" ()

(* Another machine without GPUs: this process serves it on the loopback, then
   stops serving it. *)
let test_other_machine () =
  let key = "a key of the test, long enough" in
  let s =
    Nx_remote_device.listen ~key (Unix.ADDR_INET (Unix.inet_addr_loopback, 0))
  in
  let port =
    match Nx_device_support.Remote_server.address s with
    | Unix.ADDR_INET (_, p) -> p
    | Unix.ADDR_UNIX _ -> assert false
  in
  match Nx_remote_device.connect ~port ~key "127.0.0.1" with
  | Error why -> fail why
  | Ok host ->
      if Nx_amd_device.count ~host () > 0 then
        skip ~reason:"the machine has a GPU" ();
      let named = Printf.sprintf "AMD@127.0.0.1:%d: " port in
      (match Nx_amd_device.get ~host 0 with
      | Ok _ -> fail "a GPU opened"
      | Error msg ->
          is_true ~msg:"the GPU's name starts it"
            (String.starts_with ~prefix:named msg));
      (match Nx_amd_device.get ~host ~interface:Kernel 0 with
      | Ok _ -> fail "a GPU opened"
      | Error msg -> contains ~msg:"over PCI only" ~sub:"over PCI" msg);
      Nx_device_support.Remote_server.stop s;
      let lost = function Nx_device.Lost (d, _) -> d == host | _ -> false in
      raises_match ~msg:"count, the machine gone" lost (fun () ->
          Nx_amd_device.count ~host ());
      raises_match ~msg:"get, the machine gone" lost (fun () ->
          Nx_amd_device.get ~host 1)

(* Counters *)

let props ?(xccs = 1) ?(shader_engines = 6) target =
  {
    Nx_amd_device.target;
    gc = target;
    sdma = (6, 0, 0);
    nbio = (0, 0, 0);
    xccs;
    shader_engines;
    compute_units = 48;
    compute_units_per_array = 4;
    waves_per_cu = 32;
    lds_bytes = 65536;
    scratch_slots_per_cu = 32;
  }

let layout (c : Nx_amd_device.counter) =
  ( c.block,
    c.event,
    c.register,
    (c.instances, c.engines, c.arrays, c.wgps),
    c.offset )

let layouts =
  list
    (Testable.make
       ~pp:(fun ppf (b, e, r, (i, s, a, w), o) ->
         Format.fprintf ppf "%s %d r%d (%d, %d, %d, %d) @@%d" b e r i s a w o)
       ~equal:( = ))

let test_counters () =
  equal ~msg:"gfx1100: the SQ is counted per engine, array and WGP" layouts
    [
      ("SQ", 3, 0, (1, 6, 2, 2), 0);
      ("GRBM", 2, 0, (1, 1, 1, 1), 192);
      ("SQ", 62, 1, (1, 6, 2, 2), 200);
      ("GL2C", 42, 0, (32, 1, 1, 1), 392);
    ]
    (List.map layout
       (Nx_amd_device.counters
          (props (11, 0, 0))
          [ "SQ_BUSY_CYCLES"; "GRBM_GUI_ACTIVE"; "SQ_INSTS_VALU"; "GL2C_HIT" ]));
  equal ~msg:"gfx1201's events" layouts
    [ ("SQ", 50, 0, (1, 4, 2, 2), 0) ]
    (List.map layout
       (Nx_amd_device.counters
          (props ~shader_engines:4 (12, 0, 1))
          [ "SQ_INSTS_VALU" ]));
  equal ~msg:"gfx942: the SQ per engine, every block per die" layouts
    [ ("SQ", 26, 0, (1, 4, 1, 1), 0); ("TCC", 17, 0, (16, 1, 1, 1), 256) ]
    (List.map layout
       (Nx_amd_device.counters
          (props ~xccs:8 ~shader_engines:4 (9, 4, 2))
          [ "SQ_INSTS_VALU"; "TCC_HIT" ]));
  raises_match
    (Exn.invalid_arg ~substring:"gfx1100 counts no TCC_HIT; it counts ")
    (fun () -> Nx_amd_device.counters (props (11, 0, 0)) [ "TCC_HIT" ])

let () =
  exit
    (run "nx.amd.device"
       [
         test "without a GPU, an open fails with a message and fixes nothing"
           (fun () ->
             no_gpu ();
             List.iter
               (fun interface ->
                 match Nx_amd_device.get ~interface 0 with
                 | Ok _ -> fail "a GPU opened"
                 | Error msg ->
                     is_true ~msg:"the vendor names the failure"
                       (String.starts_with ~prefix:"AMD: " msg))
               [ Nx_amd_device.Kernel; Pci; Kernel ];
             is_error (Nx_amd_device.get 0);
             raises_match (Exn.failure ~substring:"AMD: ") (fun () ->
                 Nx_amd_device.v 0));
         test "another machine's GPUs are opened over PCI" test_other_machine;
         test "a negative index is refused" (fun () ->
             raises_match (Exn.invalid_arg ~substring:"-1 < 0") (fun () ->
                 Nx_amd_device.get (-1)));
         test
           "counters follow each other in a run's samples, each block's taking \
            its registers in turn"
           test_counters;
         test "a device of another vendor has no AMD queues" (fun () ->
             is_true (Option.is_none (Nx_amd_device.of_device Nx_device.host)));
       ])
