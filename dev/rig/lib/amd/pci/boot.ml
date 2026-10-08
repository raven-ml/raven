(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs
module Function = Rig_pci.Function
module Machine = Rig_pci.Machine
module Window = Rig_pci.Window
module Page_table = Rig_pci.Page_table
module Memory = Rig_pci.Memory

let strf = Printf.sprintf
let ( let* ) = Result.bind

(* Sessions *)

let session = 0x5241_0001

let plan ~mark ~dirty ~fault ~gc ~alive =
  let marked = mark = session in
  if marked && (gc = (9, 5, 0) || (dirty = 0 && fault = 0)) then `Partial
  else if alive then `Booted
  else `Full

(* The GPU *)

type t = {
  f : Function.t;
  r : Regs.t;
  gmc : Gmc.t;
  doorbells : Window.t;
  mmio : Window.t;
  tables : Page_table.t;
  memory : Memory.t;
  ih : Ih.t;
  psp : Psp.t;
  smu : Smu.t;
  gfx : Gfx.t;
  sdma : Sdma.t;
  images : Images.t;
  vf : bool;
  mutable lease : int; (* a VF's access to give back, 0 if none *)
  mutable fault : string option;
      (* the first report, raised again by each sleep *)
  mutable eops : Memory.region list;
  hw : Mutex.t; (* the register sequences and page-table edits *)
}

let space = Rig_pci.Space.create ~base:0x2000_0000_0000 (1 lsl 44)

(* The GPU memory the kernel driver leaves to the firmware at its top: 384 MiB
   on GC 9.4 and 9.5, 64 MiB on the others. *)
let reserved = function 9, (4 | 5), _ -> 384 lsl 20 | _ -> 64 lsl 20

(* The boot pool holds what a partial boot finds where the last one left it. *)
let boot_pool = 3 lsl 20

(* The physical blocks the main pool hands out, largest first: from 4 KiB to 512
   GiB, those of 2 MiB and more aligned on 2 MiB. *)
let main_blocks =
  List.init 28 (fun k ->
      let i = 27 - k in
      (1 lsl (i + 12), if i >= 9 then 2 lsl 20 else 0x1000))

(* The BARs: the GPU's memory, its doorbells and its registers. *)
let vram_bar = 0
let doorbell_bar = 2
let register_bar = 5

(* PCI configuration space: the command register's bus master bit, and the PCI
   Express capability's link control register, whose low two bits enable ASPM
   (PCI Express Base Specification 7.5.3.7). *)
let command = 0x04
let bus_master = 0x4
let capabilities = 0x34
let pcie_capability = 0x10
let link_control = 0x10
let aspm = 0x3

let set_bus_master f on =
  let c = Function.config16 f command in
  Function.set_config16 f command
    (if on then c lor bus_master else c land lnot bus_master)

(* L1 across retimers makes reads oscillate to all ones; clearing the GPU's end
   is enough, since L1 needs both ends. The walk is bounded: a dead link can
   give back pointers for ever. *)
let disable_aspm f =
  let rec walk cap seen =
    if cap = 0 || List.mem cap seen then None
    else if Function.config8 f cap = pcie_capability then Some cap
    else walk (Function.config8 f (cap + 1) land 0xfc) (cap :: seen)
  in
  match walk (Function.config8 f capabilities land 0xfc) [] with
  | Some cap ->
      let at = cap + link_control in
      Function.set_config16 f at (Function.config16 f at land lnot aspm)
  | None -> ()

(* Before the registers' layout is known *)

let wait_raw f ~ms what cond =
  if Machine.wait (Function.machine f) ~ms cond then Ok ()
  else
    match Function.failed f with
    | Some why -> Error (strf "%s: %s" what why)
    | None -> Error (strf "%s did not answer in %d ms" what ms)

(* A function's firmware sets bit 31 of MP0's register 33 once it has laid out
   the GPU, within 2 s of its power, as the kernel's
   amdgpu_discovery_read_binary_from_mem waits for. *)
let firmware_ms = 2_000
let ready = 0x8000_0000

(* A virtual function asks its host through the mailbox for access to the GPU,
   which the host grants with a message; the request after a request gives its
   access back. *)
let mailbox_ms = 1_000

let vf_request mmio f ?(ready = true) req =
  let mb = Window.sub mmio D.nv_maibox_control_trn_offset_byte 2 in
  Window.set8 mb 0 0;
  let* () =
    wait_raw f ~ms:mailbox_ms "the VF mailbox's acknowledgement" (fun () ->
        Window.get8 mb 0 land 2 = 0)
  in
  List.iteri
    (fun i w -> Window.set32 mmio ((D.mmmailbox_msgbuf_trn_dw0 + i) * 4) w)
    [ req; 0; 0; 0 ];
  Window.set8 mb 0 1;
  let* () =
    wait_raw f ~ms:D.nv_mailbox_poll_ack_timedout
      (strf "the VF mailbox's request 0x%x" req) (fun () ->
        Window.get8 mb 0 land 2 = 2)
  in
  Window.set8 mb 0 0;
  let* () =
    if not ready then Ok ()
    else
      let* () =
        wait_raw f ~ms:D.nv_mailbox_poll_msg_timedout
          "the VF's host granting access" (fun () ->
            Window.get32 mmio (D.mmmailbox_msgbuf_rcv_dw0 * 4)
            = D.idh_ready_to_access_gpu)
      in
      Window.set8 mb 1 2;
      Ok ()
  in
  Ok (req + 1)

(* The discovery table, read through the memory BAR, or a word at a time through
   the memory index window when the BAR is smaller than the GPU's memory. *)
let read_table ~vram ~mmio ~memory =
  let at = memory - Discovery.offset in
  if Window.length vram >= memory then Window.read vram at Discovery.bytes
  else begin
    let b = Bytes.create Discovery.bytes in
    for i = 0 to (Discovery.bytes / 4) - 1 do
      let w = at + (4 * i) in
      Window.set32 mmio (D.mmmm_index_hi * 4) (w lsr 31);
      Window.set32 mmio (D.mmmm_index * 4) (w land 0x7fff_ffff lor 0x8000_0000);
      Bytes.set_int32_le b (4 * i)
        (Int32.of_int (Window.get32 mmio (D.mmmm_data * 4)))
    done;
    Bytes.to_string b
  end

(* The windows on the GPU's BARs: its memory combined, its doorbells and
   registers uncached. *)
let map_bars f =
  let* vram = Function.map ~combine:true f vram_bar in
  let* doorbells = Function.map ~combine:false f doorbell_bar in
  let* mmio = Function.map ~combine:false f register_bar in
  Ok (vram, doorbells, mmio)

(* What the GPU says of itself before any block is touched. A VF asks for access
   first, as its registers need it. *)
let survey f =
  let* vram, doorbells, mmio = map_bars f in
  let vf = Window.get32 mmio (D.mmrcc_iov_func_identifier * 4) land 1 = 1 in
  let* lease =
    if vf then vf_request mmio f D.idh_req_gpu_init_access else Ok 0
  in
  let* () =
    if vf then Ok ()
    else
      wait_raw f ~ms:firmware_ms "the GPU's firmware laying out its memory"
        (fun () ->
          Window.get32 mmio (D.mmmp0_smn_c2pmsg_33 * 4) land ready <> 0)
  in
  let size = Window.get32 mmio (D.mmrcc_config_memsize * 4) in
  let* memory =
    if size = 0 || size = 0xffff_ffff then Error "the GPU reports no memory"
    else Ok (size lsl 20)
  in
  let* d = Discovery.of_string (read_table ~vram ~mmio ~memory) in
  let* l = Regs.layout d in
  let r = Regs.make f mmio l ~vf in
  Ok (vram, doorbells, mmio, vf, lease, memory, r)

let stuck f =
  match f () with v -> Ok v | exception Regs.Stuck why -> Error why

(* The boot pool *)

(* What a partial boot finds where the last one left it: the boot pool's layout,
   the same in every session. *)
type pool = {
  scratch : int;
  dummy : int;
  ih_rings : int * int;
  ih_wptr : int;
  psp : Psp.memory;
  smu_table : int;
  mqds : int array;
}

let ih_wptr_bytes = 0x1000
let smu_table_bytes = 0x4000
let psp_ring_bytes = 0x1_0000
let page = 0x1000

let lay_out tables ~vf ~xccs =
  let alloc ?align ?(zero = false) n =
    match Page_table.palloc ?align ~zero ~boot:true tables n with
    | Some pa -> pa
    | None -> raise (Regs.Stuck "the GPU's boot memory is full")
  in
  let scratch = alloc page in
  let dummy = alloc page in
  let ring0 = alloc Ih.bytes in
  let wptr = alloc ih_wptr_bytes in
  let ring1 = alloc Ih.bytes in
  ignore (alloc ih_wptr_bytes);
  let message = alloc ~align:D.psp_1_meg D.psp_1_meg in
  let command = alloc D.psp_cmd_buffer_size in
  let fence = alloc ~zero:true D.psp_fence_buffer_size in
  let ring = alloc psp_ring_bytes in
  let smu_table = alloc smu_table_bytes in
  let mqds = Array.init (if vf then 2 else 1) (fun _ -> alloc (page * xccs)) in
  {
    scratch;
    dummy;
    ih_rings = (ring0, ring1);
    ih_wptr = wptr;
    psp = { Psp.message; command; fence; ring };
    smu_table;
    mqds;
  }

(* Stopping *)

let mark g ~dirty =
  if not g.vf then
    Regs.write ~value:(Bool.to_int dirty) g.r "regSCRATCH_REG6" []

let give_back_access g =
  if g.lease <> 0 then begin
    ignore (vf_request g.mmio g.f ~ready:false g.lease);
    g.lease <- 0
  end

let quietly f = try f () with Regs.Stuck _ -> ()

let stop_locked g =
  if Machine.failed (Function.machine g.f) <> None then `Unknown
  else begin
    (if g.vf && g.lease = 0 then
       match vf_request g.mmio g.f D.idh_req_gpu_fini_access with
       | Ok lease -> g.lease <- lease
       | Error _ -> ());
    let left =
      match
        Sdma.stop g.sdma;
        Gfx.dequeue g.gfx ~wait:(g.fault = None)
      with
      | left -> left
      | exception Regs.Stuck _ -> false
    in
    if not g.vf then quietly (fun () -> Smu.clocks g.smu `Lowest);
    quietly (fun () -> ignore (Ih.read g.ih));
    let lost = g.fault <> None || not left in
    quietly (fun () -> mark g ~dirty:lost);
    give_back_access g;
    List.iter
      (fun m -> try Memory.free g.memory m with Invalid_argument _ -> ())
      g.eops;
    g.eops <- [];
    if not lost then `Clean
    else begin
      (* The GPU reaches no memory outside its own once it masters the bus no
         more, but over a fabric. *)
      set_bus_master g.f false;
      if Gmc.hive g.gmc && not left then `Unknown else `Lost
    end
  end

let stop g = Mutex.protect g.hw (fun () -> stop_locked g)

(* Starting *)

let boot g ~partial ~pool ~kiq =
  let r = g.r in
  let fabric pa = Gmc.fabric g.gmc pa in
  if partial then mark g ~dirty:true;
  disable_aspm g.f;
  (* A partial boot finds the GPU as the last one left it, but for its bus
     mastering, which a server that stopped its client's DMA turned off. *)
  set_bus_master g.f true;
  if not partial then begin
    Soc.start r;
    Gmc.start_hub g.gmc `Mm g.tables ~scratch:(fabric pool.scratch)
      ~dummy:(fabric pool.dummy)
  end;
  Ih.start g.ih;
  if not g.vf then begin
    Psp.start g.psp g.images ~partial;
    if not partial then Smu.start g.smu
  end;
  Page_table.booted g.tables;
  Gmc.start_hub g.gmc `Gc g.tables ~scratch:(fabric pool.scratch)
    ~dummy:(fabric pool.dummy);
  Gfx.start g.gfx g.memory g.images ~partial;
  if g.vf then kiq := Some g.gfx;
  let xccs = (Regs.gpu (Regs.layout_of r)).xccs in
  (* A GPU of several dies runs them as one partition. *)
  if xccs > 1 && (not g.vf) && not partial then Psp.set_partition g.psp 1;
  Sdma.start g.sdma;
  if not g.vf then begin
    Smu.clocks g.smu `Highest;
    Soc.gate r;
    Gfx.gate g.gfx;
    Regs.write ~value:(Psp.tmr g.psp) r "regSCRATCH_REG5" [];
    Regs.write ~value:session r "regSCRATCH_REG7" [];
    mark g ~dirty:true
  end

let fault_status l =
  if Regs.has l "regGCVM_L2_PROTECTION_FAULT_STATUS_LO32" then
    "regGCVM_L2_PROTECTION_FAULT_STATUS_LO32"
  else "regGCVM_L2_PROTECTION_FAULT_STATUS"

let start f find =
  let* vram, doorbells, mmio, vf, lease, memory, r = survey f in
  let l = Regs.layout_of r in
  let* images = Images.load find (Regs.discovery l) in
  let gc = Regs.version l D.gc_hwid in
  let* gmc, p =
    stuck (fun () ->
        let gmc = Gmc.make r vram in
        let alive =
          (not vf) && Psp.running r && Smu.alive (Smu.make r gmc ~table:0)
        in
        let read name = Regs.read r name in
        ( gmc,
          plan ~mark:(read "regSCRATCH_REG7") ~dirty:(read "regSCRATCH_REG6")
            ~fault:(read (fault_status l))
            ~gc ~alive ))
  in
  let* () =
    match p with
    | `Booted when Gmc.hive gmc ->
        Error
          "the GPU is in a fabric left running; reset its GPUs together \
           outside this process"
    | `Booted ->
        Error
          "firmware this library did not start runs on the GPU; a reset stops \
           it"
    | `Partial | `Full -> Ok ()
  in
  (* From here the GPU is written: a failure stops it as lost. *)
  let kiq = ref None in
  let flush () =
    Window.flush vram;
    Gmc.flush_hdp gmc;
    match !kiq with
    | Some gfx -> Gfx.invalidate gfx
    | None -> Gmc.invalidate gmc
  in
  let* tables, pool =
    stuck (fun () ->
        let tables =
          Page_table.create (Gmc.format gmc ~flush) space
            ~memory:(memory - reserved gc)
            ~boot:boot_pool
            ~tables:
              (if Window.length vram < memory then Page_table.Pool
               else Page_table.Main)
            ~pages:main_blocks
        in
        (tables, lay_out tables ~vf ~xccs:(Regs.gpu l).xccs))
  in
  let peer =
    if not (Gmc.hive gmc) then None
    else
      Some
        (fun ranges ->
          ( List.map (fun (pa, n) -> (Gmc.fabric gmc pa, n)) ranges,
            Page_table.Peer 0 ))
  in
  let g =
    {
      f;
      r;
      gmc;
      doorbells;
      mmio;
      tables;
      memory = Memory.create ?peer f tables ~bar:vram_bar;
      ih = Ih.make r gmc vram ~rings:pool.ih_rings ~wptr:pool.ih_wptr;
      psp = Psp.make r gmc vram tables pool.psp;
      smu = Smu.make r gmc ~table:pool.smu_table;
      gfx = Gfx.make r gmc vram doorbells ~mqds:pool.mqds;
      sdma = Sdma.make r;
      images;
      vf;
      lease;
      fault = None;
      eops = [];
      hw = Mutex.create ();
    }
  in
  match boot g ~partial:(p = `Partial) ~pool ~kiq with
  | () -> Ok g
  | exception Regs.Stuck why ->
      g.fault <- Some why;
      ignore (stop g);
      Error why

(* Facts *)

let gpu g = Regs.gpu (Regs.layout_of g.r)
let gc g = (Regs.discovery (Regs.layout_of g.r)).gc
let mec g = g.images.mec
let wgps g = Mutex.protect g.hw (fun () -> Gfx.wgps g.gfx)
let memory g = g.memory
let hive g = Gmc.hive g.gmc
let host w off = if Window.mapped w then Some (Window.address w + off) else None
let hdp g = host g.mmio (Gmc.hdp g.gmc)

(* Queues *)

let eop_bytes = 0x1000
let doorbell_bytes = 8

let queue g kind ~ring ~bytes ~read ~write =
  Mutex.protect g.hw @@ fun () ->
  let doorbell index =
    match host g.doorbells (doorbell_bytes * index) with
    | Some a -> Ok a
    | None -> Error "the GPU's doorbells are not mapped into the process"
  in
  match kind with
  | `Sdma -> doorbell (Sdma.queue g.sdma ~ring ~bytes ~read ~write)
  | (`Pm4 | `Aql) as kind -> (
      match Memory.alloc g.memory Memory.Gpu eop_bytes with
      | Error why -> Error why
      | Ok None -> Error "no GPU memory for a queue's end-of-pipe buffer"
      | Ok (Some m) -> (
          g.eops <- m :: g.eops;
          match
            Gfx.queue g.gfx kind ~ring ~bytes ~read ~write ~eop:m.mapping.va
          with
          | index -> doorbell index
          | exception Regs.Stuck why -> Error why))

(* Sleeping *)

let fault g why =
  g.fault <- Some why;
  raise (Rig_amd.Fault why)

(* The GPU's fatal hardware errors, which NBIO flags outside the interrupt ring,
   with the power manager's machine-check banks. *)
let fatal g =
  let r = g.r in
  let reg = "regBIF_BX0_BIF_DOORBELL_INT_CNTL" in
  let athub = Regs.field r reg "ras_athub_err_event_interrupt_status" in
  let cntlr = Regs.field r reg "ras_cntlr_interrupt_status" in
  if athub = 0 && cntlr = 0 then None
  else begin
    let banks = try Smu.banks g.smu with Regs.Stuck why -> why in
    Regs.write r reg
      [
        ("ras_cntlr_interrupt_clear", cntlr);
        ("ras_athub_err_event_interrupt_clear", athub);
      ];
    Some
      (strf "fatal hardware error%s%s; machine-check banks: %s"
         (if athub <> 0 then " RAS_ATHUB_ERR_EVENT" else "")
         (if cntlr <> 0 then " RAS_CNTLR" else "")
         banks)
  end

let sleep g ~ms =
  (match g.fault with Some why -> raise (Rig_amd.Fault why) | None -> ());
  ignore (Machine.wait (Function.machine g.f) ~ms (fun () -> Ih.pending g.ih));
  Mutex.protect g.hw @@ fun () ->
  match g.fault with
  | Some why -> raise (Rig_amd.Fault why)
  | None -> (
      match Function.failed g.f with
      | Some why -> fault g why
      | None -> (
          match Ih.read g.ih with
          | exception Regs.Stuck why -> fault g why
          | reports -> (
              let reports =
                List.map
                  (function
                    | Ih.Page_fault -> Gmc.fault g.gmc | Ih.Fault why -> why)
                  reports
              in
              let reports =
                if g.vf then reports
                else
                  match fatal g with
                  | Some why -> reports @ [ why ]
                  | None -> reports
              in
              match reports with
              | [] -> ()
              | l -> fault g (String.concat "; " l))))

(* Resetting *)

let quiesce_ms = 100

let reset f =
  let* vram, doorbells, mmio, vf, lease, _, r = survey f in
  if vf then
    (* A VF's physical function resets it: it gives back its access. *)
    Result.map ignore (vf_request mmio f ~ready:false lease)
  else
    stuck (fun () ->
        let gmc = Gmc.make r vram in
        let smu = Smu.make r gmc ~table:0 in
        if Psp.running r && Smu.alive smu then begin
          if Gmc.hive gmc then
            raise
              (Regs.Stuck
                 "the GPU is in a fabric, whose GPUs reset together outside \
                  this process");
          set_bus_master f false;
          Regs.write ~value:0 r "regSCRATCH_REG7" [];
          (* A mode 1 reset over engines running at full clocks can stall the
             GPU until it is power cycled: they are stopped first. *)
          let gfx = Gfx.make r gmc vram doorbells ~mqds:[||] in
          ignore (Gfx.dequeue gfx ~wait:true);
          Smu.clocks smu `Lowest;
          Gfx.halt gfx;
          Sdma.halt (Sdma.make r);
          Regs.pause r quiesce_ms;
          Smu.reset smu
        end)

let give_back g = Mutex.protect g.hw (fun () -> give_back_access g)
