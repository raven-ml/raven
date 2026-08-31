(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Am_register = Amdev.Am_register
module Firmware = Amdev.Firmware
module Helpers = Tolk.Helpers
module System = Tolk_hcq.System

let debug = Helpers.getenv "DEBUG" 0

(* amdev.py:146 AMDev.Version *)
let version = 0xA0000008

type t = {
  adev : Amdev.t;
  fw : Firmware.t;
  soc : Am_ip.Soc.t;
  gmc : Am_ip.Gmc.t;
  ih : Am_ip.Ih.t;
  psp : Am_ip.Psp.t;
  smu : Am_ip.Smu.t;
  gfx : Am_ip.Gfx.t;
  sdma : Am_ip.Sdma.t;
  mutable partial_boot : bool;
}

(* amdev.py:199 init_sw: the firmware set and the blocks, created in a
   fixed order because each takes its state from boot memory and a
   partial boot reuses the previous session's layout address for
   address. *)
let create ?fw adev =
  let fw =
    match fw with
    | Some fw -> fw
    | None -> Firmware.create (Amdev.discovery adev).Amdev.ip_ver
  in
  let soc = Am_ip.Soc.create adev in
  let gmc = Am_ip.Gmc.create adev in
  let ih = Am_ip.Ih.create adev in
  let psp = Am_ip.Psp.create adev ~fw in
  let smu = Am_ip.Smu.create adev in
  let gfx = Am_ip.Gfx.create adev in
  let sdma = Am_ip.Sdma.create adev in
  (* amdev.py:140 AMMemoryManager.on_range_mapped: invalidate the TLBs
     after every mapping, once the hubs that hold them exist. *)
  Amdev.set_on_range_mapped adev (fun () ->
      let xccs = Am_ip.Gfx.xccs gfx in
      Am_ip.Gmc.flush_tlb gmc ~xccs Am_ip.Gmc.Gc ~vmid:0;
      Am_ip.Gmc.flush_tlb gmc ~xccs Am_ip.Gmc.Mm ~vmid:0);
  { adev; fw; soc; gmc; ih; psp; smu; gfx; sdma; partial_boot = false }

let init_hw t blocks =
  List.iter
    (fun (name, init) ->
      init ();
      if debug >= 2 then
        Printf.printf "am %s: %s initialized\n%!" (Amdev.devfmt t.adev) name)
    blocks

(* Linux pci_regs.h: the command register and its bus-master bit. *)
let pci_command = 0x04
let pci_command_master = 0x4

let set_bus_master t enable =
  match Amdev.pci_dev t.adev with
  | None -> ()
  | Some pci_dev ->
      let cmd =
        System.Pci_device.read_config pci_dev ~offset:pci_command ~size:2
      in
      let cmd =
        if enable then cmd lor pci_command_master
        else cmd land lnot pci_command_master
      in
      System.Pci_device.write_config_flush pci_dev ~offset:pci_command
        ~value:cmd ~size:2

let am_power_limit () =
  match float_of_string_opt (Helpers.getenv_str "AM_POWER_LIMIT" "0") with
  | Some v -> v
  | None -> failwith "AM_POWER_LIMIT: expected a number"

(* amdev.py:155-197: the boot state machine.

   The GPU being passed can be in one of several states: 1. Not
   initialized. 2. Initialized by an external driver. 3. Initialized by
   this protocol. The 1st and 2nd states require a full GPU setup since
   their states are unknown; the 2nd state also requires a mode1 reset
   to reinitialize all components.

   The 3rd state can be set up partially to optimize boot time. In this
   case, only the GFX and SDMA blocks need to be initialized. To enable
   this, a separate boot memory region is used that is guaranteed not to
   be overwritten; it backs all blocks that are initialized only during
   the initial boot. regSCRATCH_REG7 flags a device in the third state,
   and regSCRATCH_REG6 whether the previous session finalized
   correctly. *)
let init t =
  let adev = t.adev in
  let reg name = Amdev.reg adev name in
  let partial_boot =
    Am_register.read (reg "regSCRATCH_REG7") = version
    && Helpers.getenv "AM_RESET" 0 <> 1
  in
  let partial_boot =
    if
      partial_boot
      && (Am_register.read (reg "regSCRATCH_REG6") <> 0
         || Am_register.read (reg (Am_ip.Gmc.pf_status_reg t.gmc Am_ip.Gmc.Gc))
            <> 0)
    then begin
      if debug >= 2 then
        Printf.printf "am %s: Malformed state. Issuing a full reset.\n%!"
          (Amdev.devfmt adev);
      false
    end
    else partial_boot
  in
  t.partial_boot <- partial_boot;

  (* Init hw for the blocks where it is needed. *)
  if not partial_boot then begin
    if Am_ip.Psp.is_sos_alive t.psp && Am_ip.Smu.is_smu_alive t.smu then begin
      set_bus_master t false;
      if Amdev.is_hive adev then
        failwith
          (Printf.sprintf
             "am %s: Malformed state. Reset the whole fabric externally and \
              retry."
             (Amdev.devfmt adev));
      Am_ip.Smu.mode1_reset t.smu
    end;
    set_bus_master t true;
    init_hw t
      [
        ("Soc", fun () -> Am_ip.Soc.init_hw t.soc ~vmhubs:(Am_ip.Gmc.vmhubs t.gmc));
        ("Gmc", fun () -> Am_ip.Gmc.init_hw t.gmc ~soc:t.soc);
        ("Ih", fun () -> Am_ip.Ih.init_hw t.ih);
        ("Psp", fun () -> Am_ip.Psp.init_hw t.psp);
        ("Smu", fun () -> Am_ip.Smu.init_hw t.smu);
      ]
  end;

  (* Booting done. *)
  Amdev.set_is_booting adev false;

  (* Re-initialize main blocks. *)
  init_hw t
    [
      ( "Gfx",
        fun () ->
          Am_ip.Gfx.init_hw t.gfx ~soc:t.soc ~gmc:t.gmc ~psp:t.psp ~fw:t.fw
            ~partial_boot );
      ("Sdma", fun () -> Am_ip.Sdma.init_hw t.sdma ~soc:t.soc);
    ];

  let max_power = am_power_limit () in
  if max_power > 0. then begin
    Am_ip.Smu.set_power_limit t.smu max_power;
    Am_ip.Smu.set_clocks t.smu ~level:None
  end
  else Am_ip.Smu.set_clocks t.smu ~level:(Some (-1));
  Am_ip.Soc.set_clockgating_state t.soc;
  Am_ip.Gfx.set_clockgating_state t.gfx;
  Am_register.write (reg "regSCRATCH_REG7") ~value:version [];
  (* Set initialized state. *)
  Am_register.write (reg "regSCRATCH_REG6") ~value:1 [];
  if debug >= 2 then
    Printf.printf "am %s: boot done\n%!" (Amdev.devfmt adev)

(* amdev.py:225 fini *)
let fini t =
  if debug >= 2 then
    Printf.printf "am %s: Finalizing\n%!" (Amdev.devfmt t.adev);
  Am_ip.Sdma.fini_hw t.sdma;
  Am_ip.Gfx.fini_hw t.gfx;
  Am_ip.Smu.set_clocks t.smu ~level:(Some 0);
  Am_ip.Ih.interrupt_handler t.ih ~soc:t.soc ~gmc:t.gmc ~smu:t.smu;
  (* Set finalized state. *)
  Am_register.write
    (Amdev.reg t.adev "regSCRATCH_REG6")
    ~value:(if Amdev.is_err_state t.adev then 1 else 0)
    []

(* amdev.py:232 recover *)
let recover ?(force = false) t =
  if (not force) && not (Amdev.is_err_state t.adev) then false
  else begin
    if debug >= 3 then
      Printf.printf "am %s: Start recovery\n%!" (Amdev.devfmt t.adev);
    Am_ip.Ih.interrupt_handler t.ih ~soc:t.soc ~gmc:t.gmc ~smu:t.smu;
    Am_ip.Gfx.reset_mec t.gfx ~fw:t.fw;
    Amdev.set_err_state t.adev false;
    if debug >= 3 then
      Printf.printf "am %s: Recovery complete\n%!" (Amdev.devfmt t.adev);
    true
  end
