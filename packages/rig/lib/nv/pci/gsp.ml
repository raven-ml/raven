(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* As CPU-RM, the process describes the system and the GPU's memory to the
   GSP, runs the register sequences the GSP asks of the CPU, hands the GSP
   the memory of the objects it allocates (a channel's instance and method
   buffer, a compute engine's context buffers, a virtual address space's
   page directory), and reads the GSP's events. The memory the boot gives
   the GSP is system memory, as the RM places it, but for what the GPU's
   falcons and the GSP's objects read from the GPU's memory. The structures
   are release 570.144's. *)

open Rig_pci
open Field

let strf = Printf.sprintf
let ( let* ) = Result.bind
let page = 0x1000
let mib = 1 lsl 20
let round_up n a = (n + a - 1) / a * a

(* Bytes *)

let fst3 (a, _, _) = a

(* The field [f] of element [i] of the array [a]. *)
let elt (off, size, _) i (f, n) = (off + (i * size) + f, n)

let u64s l =
  record
    (8 * List.length l)
    (fun b -> List.iteri (fun i x -> set b (8 * i, 8) x) l)

(* Booting *)

type placement = {
  chip : Chip.t;
  memory : int;
  tables : Page_table.t;
  tables_lock : Rig_nv.Lock.t;
  bar : Window.t;
  space : Space.t;
}

type buffer = { size : int; phys : bool; virt : bool; local : bool }

(* GPU memory a falcon reads an image from: its physical address and the
   window on it. *)
type region = { pa : int; w : Window.t }

(* How the boot starts the GSP: on Ampere and Ada, FWSEC and then the booter,
   which the falcons read from GPU memory, FWSEC's FRTS command set up for
   the GPU's memory; on Blackwell, the FSP's ops, which point to the FMC in
   system memory. FWSEC's region holds the image of either of its commands,
   and the booters' the larger booter, so that the stop loads its own. *)
type start =
  | Legacy of {
      fwsec : Vbios.fwsec;
      frts : int;
      booters : Images.booters;
      fwsec_at : region;
      booter_at : region;
    }
  | Cot of Falcon.op list

(* Each channel's runlist changes under the GSP's lock by one store of a map
   made first, so an exception leaves the table whole. *)
module Channels = Map.Make (Int)

(* GPU memory the process gave the GSP for an object: a channel's instance
   block and method buffer, or a compute engine's context buffers. *)
type given = Mapped of Page_table.mapping | Block of int

(* An object of one of the process's clients: its parent, and the memory the
   process gave the GSP for it, which goes back once the GSP freed the object
   or an object above it. *)
type held = { parent : int; given : given list }

(* The objects of the process's clients, by client and handle. *)
module Objects = Map.Make (struct
  type t = int * int

  let compare = Stdlib.compare
end)

type t = {
  p : placement;
  q : Msgq.t;
  lock : Rig_nv.Lock.t;
  libos : int; (* the libos arguments' bus address, for a resumption *)
  meta : int; (* the WPR metadata's bus address *)
  flush : (int * int) list; (* the flush page's registers and their values *)
  start : start;
  mutable fault : string option;
  mutable unanswered : bool; (* a call waited out its bound *)
  mutable sequencing : bool; (* a CPU sequence runs, or an exception cut it *)
  mutable next : int; (* the next handle the process names an object with *)
  mutable clients : int; (* the next client handle *)
  mutable runlists : (int * int) list; (* each engine's runlist *)
  mutable channels : int Channels.t; (* each channel's runlist *)
  mutable held : held Objects.t; (* under [lock] *)
  mutable buffers : (int * buffer) list; (* the graphics context's *)
  taken : sys list ref; (* the system memory the boot took *)
}

(* System memory for the GSP: [w], its address in the space and the bus address
   of each 4 KiB page. *)
and sys = { w : Window.t; va : int; pages : int list }

let sys p ~taken ?(contiguous = false) n =
  let n = round_up n page in
  let align =
    if contiguous && n > Machine.page (Function.machine p.chip.fn) then 2 * mib
    else page
  in
  match Space.alloc ~align p.space (round_up n align) with
  | None -> Error "no addresses for the GSP's system memory"
  | Some va -> (
      match Function.alloc_dma ~contiguous ~va p.chip.fn n with
      | Error why ->
          Space.free p.space va;
          Error why
      | Ok None ->
          Space.free p.space va;
          Error
            "the machine has no free system memory for the GSP; free some, or \
             reserve huge pages (vm.nr_hugepages)"
      | Ok (Some (w, runs)) ->
          let pages =
            List.concat_map
              (fun (a, len) -> List.init (len / page) (fun i -> a + (i * page)))
              runs
          in
          let s = { w; va; pages } in
          taken := s :: !taken;
          Ok s)

let give_back p taken =
  List.iter
    (fun s ->
      Function.free_dma p.chip.fn s.w;
      Space.free p.space s.va)
    !taken;
  taken := []

let first s = List.hd s.pages

(* The boot pool: the root table's page, then each image a falcon reads from GPU
   memory, with the room [Page_table.palloc] needs to find a block of [n] bytes,
   twice [n] and its alignment. *)
let boot_pool start =
  let room n = 2 * (round_up n page + page) in
  let images =
    match start with
    | `Booter ({ load; unload } : Images.booters) ->
        room Vbios.window
        + room (Int.max (String.length load.image) (String.length unload.image))
    | `Fmc _ -> 0
  in
  round_up (page + images) (2 * mib)

(* GPU memory a falcon reads, from the boot pool, which the memory BAR reaches
   however small: its physical address and the window on it. Each image is
   written whole before a falcon reads it, so the memory is not zeroed. *)
let vram p n =
  match Page_table.palloc ~zero:false ~boot:true p.tables n with
  | None -> Error "no GPU memory in the boot pool for a falcon's image"
  | Some pa -> Ok { pa; w = Window.sub p.bar pa (round_up n page) }

let blit (r : Images.range) w off =
  Window.blit_string r.contents r.at w off r.length

(* The queues: 256 KiB each, after the table of their pages' addresses. *)
let queue_size = 0x40000

(* The libos regions' size: 64 KiB of log each, the RM's arguments a page. *)
let log_size = 0x10000
let logs = [ "INIT"; "INTR"; "RM"; "MNOC"; "KRNL" ]

let libos_region id ~pa ~size =
  let module L = Defs.Libos_region in
  record L.sizeof (fun b ->
      set b L.kind Defs.libos_memory_region_contiguous;
      set b L.loc Defs.libos_memory_region_loc_sysmem;
      set b L.size size;
      set b L.id8 (String.fold_left (fun n c -> (n lsl 8) lor Char.code c) 0 id);
      set b L.pa pa)

(* The registry keys the GSP boots with: save the PCI configuration across
   resets, and reset by the bus where the function's reset is not enough. *)
let keys = [ ("RMForcePcieConfigSave", 1); ("RMSecBusResetEnable", 1) ]

(* Where BAR 0 mirrors the configuration space: 0x88000 to Ampere and Ada,
   0x92000 to Blackwell, a page each. *)
let config_mirror : Chip.family -> int = function
  | Blackwell -> 0x92000
  | Ampere | Ada -> 0x88000

(* The highest user address of a 64-bit process with 47 bits of address. *)
let max_user_va = 0x7fff_ffff_f000

let bdf bus =
  Scanf.sscanf bus "%x:%x:%x.%x" (fun _ b d f -> (b lsl 8) lor (d lsl 3) lor f)

let system_info p =
  let module S = Defs.System_info in
  let f = p.chip.fn in
  let bar i = match Function.bar f i with Some (a, _) -> a | None -> 0 in
  record S.sizeof (fun b ->
      set b S.gpu_phys_addr (bar 0);
      set b S.gpu_phys_fb_addr (bar 1);
      set b S.gpu_phys_inst_addr (bar 3);
      set b S.pci_config_mirror_base (config_mirror p.chip.family);
      set b S.pci_config_mirror_size page;
      set b S.nv_domain_bus_device_func (bdf (Function.bus f));
      set b S.b_is_passthru 1;
      set b S.pci_device_id (Function.config32 f 0x00);
      set b S.pci_sub_device_id (Function.config32 f 0x2c);
      set b S.pci_revision_id (Function.config8 f 0x08);
      set b S.max_user_va max_user_va)

(* The radix-3 table: each level's pages hold the addresses of the next's, the
   image's pages last. *)
let radix3 sys (image : Images.range) =
  let n = Fb_layout.radix3 image.length in
  let starts =
    Array.init 4 (fun i -> Array.fold_left ( + ) 0 (Array.sub n 0 i) * page)
  in
  let* s = sys (starts.(3) + image.length) in
  blit image s.w starts.(3);
  let pages = Array.of_list s.pages in
  for i = 0 to 2 do
    let next = Array.fold_left ( + ) 0 (Array.sub n 0 (i + 1)) in
    Window.write s.w starts.(i)
      (u64s (Array.to_list (Array.sub pages next n.(i + 1))))
  done;
  Ok s

(* The registers that name the page the GPU reads when it flushes its writes
   to system memory, and their values for the page at bus address [a], the
   high word first, as NVIDIA's RM writes them
   (kmemsysProgramSysmemFlushBuffer_GA100 and _GB10B). The flush, a read of
   the page, runs at every system memory barrier, a falcon's reset among them;
   a reset leaves the registers at address 0. Ampere and Ada take [a]'s bits
   46:8, Blackwell its bits 51:0. *)
let flush_registers (c : Chip.t) a =
  let module L = Defs.Legacy in
  let module B = Defs.Blackwell in
  let low = 0xffff_ffff in
  let regs, limit =
    match c.family with
    | Ampere | Ada ->
        let shift = Defs.nv_pfb_niso_flush_sysmem_addr_shift in
        let s = a lsr shift in
        ( [
            ( L.nv_pfb_niso_flush_sysmem_addr_hi,
              put L.nv_pfb_niso_flush_sysmem_addr_hi_adr_63_40 (s lsr 32) );
            ( L.nv_pfb_niso_flush_sysmem_addr,
              put L.nv_pfb_niso_flush_sysmem_addr_adr_39_08 (s land low) );
          ],
          (Defs.nv_pfb_niso_flush_sysmem_addr_hi_mask + 1) lsl (32 + shift) )
    | Blackwell ->
        ( [
            ( B.nv_pfb_fbhub0_pcie_flush_sysmem_addr_hi,
              put B.nv_pfb_fbhub0_pcie_flush_sysmem_addr_hi_adr (a lsr 32) );
            ( B.nv_pfb_fbhub0_pcie_flush_sysmem_addr_lo,
              put B.nv_pfb_fbhub0_pcie_flush_sysmem_addr_lo_adr (a land low) );
          ],
          (Defs.nv_pfb_fbhub0_pcie_flush_sysmem_addr_hi_adr_mask + 1) lsl 32 )
  in
  if a < limit then Ok regs
  else Error (strf "the GPU cannot name its flush page at bus address 0x%x" a)

(* Calls *)

(* How long the GSP takes to answer a call, and to boot. *)
let answer_ms = 10_000

let cut_short = "the GSP's CPU sequence was cut short"

(* Runs the sequence of [m], which stays in the queue until it ran, and
   records a failure as the GPU's fault before it consumes it. A sequence
   runs once, front to back, as open-gpu-kernel-modules runs it
   (kgspExecuteSequencerBuffer): one an exception cut short is found still
   in the queue, and consumed as a fault without running again. *)
let sequence g (m : Rpc.message) =
  let ran =
    if g.sequencing then Error cut_short
    else begin
      g.sequencing <- true;
      let* ops = Rpc.sequence ~libos:g.libos m.body in
      Falcon.run g.p.chip ops
    end
  in
  (match ran with
  | Error why when g.fault = None -> g.fault <- Some why
  | Ok () | Error _ -> ());
  Msgq.consume g.q;
  g.sequencing <- false;
  ran

(* Each message is handled, its fault recorded, before it is consumed, so an
   exception between leaves it for the next drain. *)
let drain g ~want =
  let rec go () =
    match Msgq.next g.q with
    | None -> Ok None
    | Some (Error why) ->
        if g.fault = None then g.fault <- Some why;
        Msgq.consume g.q;
        Error why
    | Some (Ok (m : Rpc.message)) -> (
        let* () =
          if m.fn <> Defs.nv_vgpu_msg_event_gsp_run_cpu_sequencer then begin
            (match Rpc.fault m with
            | Some f when g.fault = None -> g.fault <- Some f
            | _ -> ());
            Msgq.consume g.q;
            Ok ()
          end
          else sequence g m
        in
        match want with
        | Some fn when m.fn = fn ->
            if m.result <> 0 then
              Error (strf "the GSP failed call %d: 0x%x" fn m.result)
            else Ok (Some m.body)
        | _ -> go ())
  in
  go ()

(* Records a call the GSP left unanswered past its bound as the GPU's fault:
   the GSP's queue is then in a state no later call can trust, and each would
   wait out its bound in vain, the stop's unload too. *)
let unanswered g = function
  | Ok () -> Ok ()
  | Error why ->
      g.unanswered <- true;
      if g.fault = None then g.fault <- Some why;
      Error why

(* [what] names the call in errors: its number, and what it asks. *)
let wait_for g ?(what = "") fn =
  let found = ref (Error (strf "no answer to call %d%s" fn what)) in
  let answered () =
    match drain g ~want:(Some fn) with
    | Ok None -> false
    | Ok (Some body) ->
        found := Ok body;
        true
    | Error _ as e ->
        found := e;
        true
  in
  let* () =
    unanswered g
      (Function.wait g.p.chip.fn ~us:(answer_ms * 1000)
         (strf "the GSP's answer to call %d%s" fn what)
         answered)
  in
  !found

let send g fn body =
  unanswered g
    (Function.wait g.p.chip.fn ~us:(answer_ms * 1000)
       "room in the GSP's command queue" (fun () -> Msgq.send g.q fn body))

let call g ?what fn body =
  Rig_nv.Lock.protect g.lock (fun () ->
      let* () = send g fn body in
      wait_for g ?what fn)

(* Every error of a drain is recorded as the GPU's fault inside it. *)
let drained g = ignore (drain g ~want:None : (_, _) result)

let check g =
  ignore (Rig_nv.Lock.try_protect g.lock (fun () -> drained g) : bool);
  g.fault

let unload g =
  Result.map ignore
    (call g Defs.nv_vgpu_msg_function_unloading_guest_driver Rpc.unloading)

(* The RM *)

(* The GSP's own client, which the golden context belongs to (NV01_ROOT's handle
   in RM's internal range), the first handle of the clients the process makes,
   and the first of the objects it names. *)
let priv_root = 0xc1e00004
let first_client = 0xc1000000
let first_handle = 0xcf000000

let classes : Chip.family -> int * int * int = function
  | Ampere -> Defs.(ampere_channel_gpfifo_a, ampere_compute_b, ampere_dma_copy_b)
  | Ada -> Defs.(ampere_channel_gpfifo_a, ada_compute_a, ampere_dma_copy_b)
  | Blackwell ->
      Defs.
        (blackwell_channel_gpfifo_a, blackwell_compute_b, blackwell_dma_copy_b)

let string_of_params (p : Rig_nv.params) =
  String.init (Bigarray.Array1.dim p) (fun i -> Bigarray.Array1.get p i)

let blit_params s (p : Rig_nv.params) =
  String.iteri (fun i c -> Bigarray.Array1.set p i c) s

(* The RM's statuses that say it lacks memory or another resource
   (nvstatuscodes.h): NV_ERR_NO_MEMORY and NV_ERR_INSUFFICIENT_RESOURCES. *)
let nv_err_no_memory = 0x51
let nv_err_insufficient_resources = 0x1a

(* The refusal [why] of a call the RM answered with [status]. *)
let refusal why status =
  if status = nv_err_no_memory || status = nv_err_insufficient_resources then
    Rig_nv.Lacking why
  else Refused why

let failed r = Result.map_error (fun why -> Rig_nv.Refused why) r
let lacking r = Result.map_error (fun why -> Rig_nv.Lacking why) r
let why = function Rig_nv.Lacking why | Refused why -> why

let control g ~client obj cmd p =
  let body = Option.fold ~none:"" ~some:string_of_params p in
  let* answer =
    failed
      (call g Defs.nv_vgpu_msg_function_gsp_rm_control
         ~what:(strf " (command 0x%x)" cmd)
         (Rpc.rm_control ~client ~obj ~cmd body))
  in
  let* status, out = failed (Rpc.rm_answer `Control answer) in
  if status <> 0 then
    Error
      (refusal (strf "the RM refused command 0x%x: 0x%x" cmd status) status)
  else (
    Option.iter (blit_params out) p;
    Ok ())

let alloc_object g ~client ~parent ~obj cls p =
  let body = Option.fold ~none:"" ~some:string_of_params p in
  let* answer =
    failed
      (call g Defs.nv_vgpu_msg_function_gsp_rm_alloc
         ~what:(strf " (class 0x%x)" cls)
         (Rpc.rm_alloc ~client ~parent ~obj ~cls body))
  in
  let* status, _ = failed (Rpc.rm_answer `Alloc answer) in
  if status <> 0 then
    Error (refusal (strf "the RM refused class 0x%x: 0x%x" cls status) status)
  else Ok obj

(* Objects *)

(* Names a new object of class [cls] of [client] under [parent], and records
   it before the call that makes it, so that what the process gives the GSP
   for it from then on goes back with it, whether the GSP made it or not. Is
   the handle the call names, and the object's, which for a client's root is
   the client. *)
let made g ~client ~parent cls =
  Rig_nv.Lock.protect g.lock (fun () ->
      let h = g.next in
      g.next <- h + 1;
      let obj = if cls = Defs.nv01_root then client else h in
      g.held <- Objects.add (client, obj) { parent; given = [] } g.held;
      (h, obj))

(* Records [m] as given for [obj] of [client]. *)
let given g ~client obj m =
  Rig_nv.Lock.protect g.lock (fun () ->
      g.held <-
        Objects.update (client, obj)
          (Option.map (fun h -> { h with given = m :: h.given }))
          g.held)

(* Under [g]'s lock: takes [obj] of [client] and every object under it out of
   [g]'s records and runlists, and is the memory given for them. *)
let take_freed g ~client obj =
  let rec under gone =
    let more =
      Objects.filter
        (fun (c, o) h ->
          c = client && List.mem h.parent gone && not (List.mem o gone))
        g.held
    in
    if Objects.is_empty more then gone
    else under (Objects.fold (fun (_, o) _ gone -> o :: gone) more gone)
  in
  let gone = under [ obj ] in
  let memory =
    List.concat_map
      (fun o ->
        Option.fold ~none:[] ~some:(fun h -> h.given)
          (Objects.find_opt (client, o) g.held))
      gone
  in
  let drop f t = List.fold_left (fun t o -> f o t) t gone in
  g.held <- drop (fun o -> Objects.remove (client, o)) g.held;
  g.channels <- drop Channels.remove g.channels;
  memory

(* [f] of the GPU's page tables, which the path's memory calls share from
   other domains. *)
let on_tables g f = Rig_nv.Lock.protect g.p.tables_lock (fun () -> f g.p.tables)

(* Memory of the GSP's objects: GPU memory mapped at addresses of the GPU's
   space, recorded for [obj] of [client]. *)
let object_memory g ~client obj n =
  match on_tables g (fun t -> Page_table.alloc ~contiguous:true t n) with
  | None -> Error "no GPU memory for the GSP's objects"
  | Some m ->
      given g ~client obj (Mapped m);
      Ok m

let paddr (m : Page_table.mapping) = fst (List.hd m.pages)

(* A memory descriptor of a channel's allocation, cached: the GSP's RM gives
   system memory described cached the coherent aperture, whose accesses snoop
   the processors' caches, and any other the non-coherent one
   (kgmmuGetHwPteApertureFromMemdesc_GM107). A channel's USERD in system memory,
   which the host writes, must be snooped. *)
let memdesc p field ~base ~size ~space =
  let module M = Defs.Memory_desc in
  Rig_nv.set p (at field M.base) base;
  Rig_nv.set p (at field M.size) size;
  Rig_nv.set p (at field M.address_space) space;
  Rig_nv.set p (at field M.cache_attrib) Defs.nv_memory_cached

(* A channel's instance block, RAMFC and method buffer, which CPU-RM gives the
   GSP: a page, its first 0x200 bytes the RAMFC, and 0x5000 bytes. *)
let instance_size = 0x1000
let ramfc_size = 0x200
let method_buffer_size = 0x5000
let userd_size = 0x400

(* The size the error notifier is described with. *)
let notifier_size = 0xecc

let channel_memory g ~client ~locate obj p =
  let module G = Defs.Gpfifo_alloc in
  let* ramfc = lacking (object_memory g ~client obj instance_size) in
  let* mthd =
    match on_tables g (fun t -> Page_table.palloc t method_buffer_size) with
    | Some pa ->
        given g ~client obj (Block pa);
        Ok pa
    | None ->
        Error (Rig_nv.Lacking "no GPU memory for a channel's method buffer")
  in
  let fb = Defs.addr_fbmem in
  memdesc p G.ramfc_mem ~base:(paddr ramfc) ~size:ramfc_size ~space:fb;
  memdesc p G.instance_mem ~base:(paddr ramfc) ~size:instance_size ~space:fb;
  memdesc p G.mthdbuf_mem ~base:mthd ~size:method_buffer_size ~space:fb;
  if client = priv_root || Rig_nv.get p G.h_object_error = 0 then Ok ()
  else begin
    (* The error notifier is described in no memory: the GSP reports a channel's
       error as an event (RC_TRIGGERED), which the path reads instead. *)
    memdesc p G.error_notifier_mem ~base:0 ~size:notifier_size ~space:0;
    let h = Rig_nv.get p (fst3 G.h_userd_memory, 4) in
    let off = Rig_nv.get p (fst3 G.userd_offset, 8) in
    match locate h off with
    | None ->
        Error
          (Rig_nv.Refused
             (strf "no memory of handle 0x%x for a channel's USERD" h))
    | Some (where, a) ->
        let space =
          match where with `Gpu -> fb | `System -> Defs.addr_sysmem
        in
        memdesc p G.userd_mem ~base:a ~size:userd_size ~space;
        Ok ()
  end

(* The context buffers a compute engine's channel needs, promoted twice: their
   physical memory, then their virtual addresses. The memory is recorded for
   the engine [obj]. *)
let promote g ~client ~subdevice ~obj channel buffers ?(have = []) ~virt ~phys
    () =
  let module Pr = Defs.Promote_ctx in
  let module E = Defs.Promote_entry in
  let p = Rig_nv.params Pr.sizeof in
  Rig_nv.set p Pr.entry_count (List.length buffers);
  Rig_nv.set p Pr.engine_type Defs.nv2080_engine_type_graphics;
  Rig_nv.set p Pr.h_chan_client client;
  Rig_nv.set p Pr.h_object channel;
  let rec fill i made = function
    | [] -> Ok (List.rev made)
    | (id, b) :: rest ->
        let v = Option.value virt ~default:b.virt
        and ph = Option.value phys ~default:b.phys in
        let* m =
          match List.assoc_opt id have with
          | Some m -> Ok m
          | None -> lacking (object_memory g ~client obj b.size)
        in
        let f x = elt Pr.promote_entry i x in
        Rig_nv.set p (f E.buffer_id) id;
        Rig_nv.set p (f E.gpu_virt_addr) (if v then m.Page_table.va else 0);
        Rig_nv.set p (f E.b_initialize) (Bool.to_int ph);
        Rig_nv.set p (f E.gpu_phys_addr) (if ph then paddr m else 0);
        Rig_nv.set p (f E.size) (if ph then b.size else 0);
        (* NV2080_CTRL_GPU_PROMOTE_CTX_PHYS_ATTR: in the GPU's memory,
           cached. *)
        Rig_nv.set p (f E.phys_attr) (if ph then 0x4 else 0);
        Rig_nv.set p (f E.b_nonmapped) (Bool.to_int (ph && not v));
        fill (i + 1) ((id, m) :: made) rest
  in
  let* made = fill 0 [] buffers in
  let* () =
    control g ~client subdevice Defs.nv2080_ctrl_cmd_gpu_promote_ctx (Some p)
  in
  Ok made

(* The root table's entries: 4 on version 2, 2 on version 3. *)
let root_entries (c : Chip.t) =
  let v = Mmu.version c.family in
  let levels = Page_entry.levels v in
  let top = List.nth levels (List.length levels - 1) in
  1 lsl (Page_entry.bits v - top)

(* The subdevice a client made, on which compute engines' contexts are
   promoted. *)
type objects = { mutable subdevice : int }

let alloc g ~client ~locate ~objects ~parent cls p =
  let channel_class, compute_class, _ = classes g.p.chip.family in
  let h, obj = made g ~client ~parent cls in
  let* () =
    match p with
    | Some p when cls = channel_class ->
        channel_memory g ~client ~locate obj p
    | _ -> Ok ()
  in
  let* _ = alloc_object g ~client ~parent ~obj:h cls p in
  if cls = channel_class then
    Option.iter
      (fun p ->
        let e = Rig_nv.get p Defs.Gpfifo_alloc.engine_type in
        let runlist = Option.value ~default:0 (List.assoc_opt e g.runlists) in
        Rig_nv.Lock.protect g.lock (fun () ->
            g.channels <- Channels.add obj runlist g.channels))
      p;
  if cls = Defs.nv20_subdevice_0 then objects.subdevice <- obj;
  (* The golden context's objects get none of this: its address space is the
     RM's own, whose root the GSP takes with the tables of its range
     ([golden]), as SET_PAGE_DIRECTORY applies only to a space another owner
     keeps (gvaspaceExternalRootDirCommit); its context is promoted whole. *)
  let* () =
    if client = priv_root then Ok ()
    else if cls = Defs.fermi_vaspace_a then
      let* _ =
        failed
          (call g Defs.nv_vgpu_msg_function_set_page_directory
             (Rpc.page_directory ~client ~device:parent ~vaspace:obj
                ~root:(Page_table.root g.p.tables)
                ~entries:(root_entries g.p.chip)))
      in
      Ok ()
    else if cls = compute_class then
      let bufs = List.filter (fun (k, _) -> List.mem k [ 0; 1; 2 ]) g.buffers in
      let* phys =
        promote g ~client ~subdevice:objects.subdevice ~obj parent bufs
          ~virt:(Some false) ~phys:None ()
      in
      let* _ =
        promote g ~client ~subdevice:objects.subdevice ~obj parent bufs
          ~have:phys ~virt:None ~phys:(Some false) ()
      in
      Ok ()
    else Ok ()
  in
  Ok obj

(* A channel's work submit token carries its runlist in bits 31:16, and on
   Blackwell bit 30 set. *)
let token g obj p =
  let module W = Defs.Work_submit_token in
  let runlist = Option.value ~default:0 (Channels.find_opt obj g.channels) in
  let blackwell = g.p.chip.family = Blackwell in
  Rig_nv.set p W.work_submit_token
    (Rig_nv.get p W.work_submit_token lor (runlist lsl 16)
    lor if blackwell then 1 lsl 30 else 0)

(* Frees [obj] of [client] and every object under it, as rpcRmApiFree_GSP
   does, whose status is the reply's result. A GPU that left the bus or
   faulted gets no call, which would wait out its bound in vain, as [stop]
   sends it no unload. The GPU reads the memory the process gave the GSP for
   the objects no more once the GSP freed them, so it goes back then; a
   mapping whose unmap the GPU did not confirm stays, as the GPU may still
   reach it. A refusal keeps the records. *)
let free g ~client obj =
  let* () =
    if Option.is_some (Function.failed g.p.chip.fn) || Option.is_some g.fault
    then Error (Rig_nv.Refused "the GSP cannot be reached to free an object")
    else Ok ()
  in
  let* _ =
    failed (call g Defs.nv_vgpu_msg_function_free (Rpc.free ~client ~obj))
  in
  let memory =
    Rig_nv.Lock.protect g.lock (fun () -> take_freed g ~client obj)
  in
  on_tables g (fun t ->
      List.iter
        (function
          | Mapped m -> ignore (Page_table.free t m : bool)
          | Block pa -> Page_table.pfree t pa)
        memory);
  Ok ()

let client_rm g ~client ~locate =
  let objects = { subdevice = 0 } in
  {
    Rig_nv.release = 570;
    client;
    alloc =
      (fun ~parent cls p -> alloc g ~client ~locate ~objects ~parent cls p);
    control =
      (fun obj cmd p ->
        let* () = control g ~client obj cmd p in
        if cmd = Defs.nvc36f_ctrl_cmd_gpfifo_get_work_submit_token then
          Option.iter (token g obj) p;
        Ok ());
    free = Some (fun ~parent:_ obj -> free g ~client obj);
  }

let rm g ~locate =
  let client = g.clients in
  g.clients <- client + 1;
  let r = client_rm g ~client ~locate in
  let* _ =
    Result.map_error why
      (r.alloc ~parent:0 Defs.nv01_root
         (Some (Rig_nv.params Defs.Nv0000_alloc.sizeof)))
  in
  Ok r

(* The objects a path gives the driver *)

(* The address space's first address and size: 49-bit addresses from 4 KiB, less
   80 MiB at their top. *)
let va_base = 0x1000
let va_size = (1 lsl 49) - (80 lsl 20)

let objects (rm : Rig_nv.rm) =
  let module D = Defs.Nv0080_alloc in
  let module V = Defs.Vaspace_alloc in
  let dp = Rig_nv.params D.sizeof in
  Rig_nv.set dp D.h_client_share rm.client;
  Rig_nv.set dp D.va_mode
    Defs.nv_device_allocation_vamode_optional_multiple_vaspaces;
  let alloc ~parent cls p = Result.map_error why (rm.alloc ~parent cls p) in
  let* device = alloc ~parent:rm.client Defs.nv01_device_0 (Some dp) in
  let* subdevice =
    alloc ~parent:device Defs.nv20_subdevice_0
      (Some (Rig_nv.params Defs.Nv2080_alloc.sizeof))
  in
  let vp = Rig_nv.params V.sizeof in
  Rig_nv.set vp V.va_base va_base;
  Rig_nv.set vp V.va_size va_size;
  Rig_nv.set vp V.flags
    (Defs.nv_vaspace_allocation_flags_enable_page_faulting
   lor Defs.nv_vaspace_allocation_flags_is_externally_owned);
  let* vaspace = alloc ~parent:device Defs.fermi_vaspace_a (Some vp) in
  Ok (device, subdevice, vaspace)

(* GR information *)

let gr_indices =
  Defs.
    [
      nv2080_ctrl_gr_info_index_litter_num_gpcs;
      nv2080_ctrl_gr_info_index_litter_num_tpc_per_gpc;
      nv2080_ctrl_gr_info_index_litter_num_sm_per_tpc;
      nv2080_ctrl_gr_info_index_max_warps_per_sm;
      nv2080_ctrl_gr_info_index_sm_version;
    ]

let gpu g (rm : Rig_nv.rm) ~subdevice =
  let module S = Defs.Static_gr_info in
  let module L = Defs.Gr_info_list in
  let module I = Defs.Internal_gr_info in
  let p = Rig_nv.params S.sizeof in
  let* () =
    Result.map_error why
      (rm.control subdevice Defs.nv2080_ctrl_cmd_internal_static_kgr_get_info
         (Some p))
  in
  let info i =
    let engine = elt S.engine_info 0 (0, 0) in
    Rig_nv.get p (elt L.info_list i (fst engine + fst I.data, snd I.data))
  in
  match List.map info gr_indices with
  | [ gpcs; tpcs_per_gpc; sms_per_tpc; warps_per_sm; sm_version ] ->
      let channel_class, compute_class, copy_class = classes g.p.chip.family in
      Ok
        {
          Rig_nv.channel_class;
          compute_class;
          copy_class;
          sm_version;
          gpcs;
          tpcs_per_gpc;
          sms_per_tpc;
          warps_per_sm;
        }
  | _ -> assert false (* five indices asked *)

(* The golden context *)

(* The golden channel: 32 entries, its USERD 0x100 into its page, of the
   graphics engine, with the flags and internal flags CPU-RM gives the GSP's own
   channel. *)
let golden_entries = 32
let golden_userd = 0x100
let golden_userd_size = 0x20
let golden_cid = 3
let golden_flags = 0x200320
let golden_internal_flags = 0x1a

let golden g =
  let objects = { subdevice = 0 } in
  let locate _ _ = None in
  let alloc ~parent cls p =
    Result.map_error why
      (alloc g ~client:priv_root ~locate ~objects ~parent cls p)
  in
  let* _ =
    alloc ~parent:0 Defs.nv01_root
      (Some (Rig_nv.params Defs.Nv0000_alloc.sizeof))
  in
  let dp = Rig_nv.params Defs.Nv0080_alloc.sizeof in
  Rig_nv.set dp Defs.Nv0080_alloc.h_client_share priv_root;
  let* dev = alloc ~parent:priv_root Defs.nv01_device_0 (Some dp) in
  let* subdev =
    alloc ~parent:dev Defs.nv20_subdevice_0
      (Some (Rig_nv.params Defs.Nv2080_alloc.sizeof))
  in
  let* vaspace =
    alloc ~parent:dev Defs.fermi_vaspace_a
      (Some (Rig_nv.params Defs.Vaspace_alloc.sizeof))
  in
  let control obj cmd p =
    Result.map_error why (control g ~client:priv_root obj cmd p)
  in
  (* Each engine's runlist, from the device information table. *)
  let module T = Defs.Device_info_table in
  let module E = Defs.Device_entry in
  let t = Rig_nv.params T.sizeof in
  let* () =
    control subdev Defs.nv2080_ctrl_cmd_fifo_get_device_info_table (Some t)
  in
  g.runlists <-
    List.init (Rig_nv.get t T.num_entries) (fun i ->
        let data k =
          Rig_nv.get t (elt T.entries i (elt E.engine_data k (0, 4)))
        in
        (data 2, data 3));
  (* The address space's tables, as NVIDIA's RM gives the GSP those of its
     own spaces: the GSP takes over the tables of its range down to the dual
     level, the root as the space's page directory, and maps its own buffers
     there; the golden objects take addresses of the process's, which start
     at 64 GiB, above that range. *)
  let v = Mmu.version g.p.chip.family in
  let first, _ = Rpc.reserved in
  let dual = fst (List.nth (Page_entry.pages v) 1) in
  let* tables =
    match on_tables g (fun t -> Page_table.tables t ~va:first dual) with
    | Some l -> Ok l
    | None -> Error "no GPU memory for the tables of the GSP's addresses"
  in
  let r = Rpc.reserved_pdes v tables in
  let p = Rig_nv.params (String.length r) in
  blit_params r p;
  let* () =
    control vaspace Defs.nv90f1_ctrl_cmd_vaspace_copy_server_reserved_pdes
      (Some p)
  in
  (* The channel. The golden context lives as long as the GSP, so its memory
     is recorded for no object. *)
  let* area = object_memory g ~client:priv_root 0 page in
  let module G = Defs.Gpfifo_alloc in
  let gp = Rig_nv.params G.sizeof in
  Rig_nv.set gp G.gp_fifo_offset area.va;
  Rig_nv.set gp G.gp_fifo_entries golden_entries;
  Rig_nv.set gp G.engine_type Defs.nv2080_engine_type_graphics;
  Rig_nv.set gp G.cid golden_cid;
  Rig_nv.set gp G.h_va_space vaspace;
  Rig_nv.set gp (fst3 G.userd_offset, 8) golden_userd;
  memdesc gp G.userd_mem
    ~base:(paddr area + golden_userd)
    ~size:golden_userd_size ~space:Defs.addr_fbmem;
  Rig_nv.set gp G.internal_flags golden_internal_flags;
  Rig_nv.set gp G.flags golden_flags;
  let channel_class, compute_class, copy_class = classes g.p.chip.family in
  let* channel = alloc ~parent:dev channel_class (Some gp) in
  (* The context buffers' sizes. *)
  let module C = Defs.Context_buffers_info in
  let module Cb = Defs.Context_buffers in
  let module B = Defs.Context_buffer in
  let c = Rig_nv.params C.sizeof in
  let* () =
    control subdev
      Defs.nv2080_ctrl_cmd_internal_static_kgr_get_context_buffers_info (Some c)
  in
  let info ?(add = 0) ?align idx =
    let f x =
      Rig_nv.get c (elt C.engine_context_buffers_info 0 (elt Cb.engine idx x))
    in
    round_up (f B.size + add) (Option.value align ~default:(f B.alignment))
  in
  (* The graphics context and its patch buffer, then the global buffers from
     index 14 on of the engine's context properties, the fifth aligned to 2 MiB.
     The graphics context takes 256 KiB more than it says. *)
  let graphics =
    info ~add:0x40000
      Defs.nv0080_ctrl_fifo_get_engine_context_properties_engine_id_graphics
  in
  let patch =
    info
      Defs
      .nv0080_ctrl_fifo_get_engine_context_properties_engine_id_graphics_patch
  in
  let global x =
    info ?align:(if x = 5 then Some (2 * mib) else None) (x + 14)
  in
  let buf ?(local = false) ~phys ~virt size = { size; phys; virt; local } in
  g.buffers <-
    [
      (0, buf graphics ~phys:true ~virt:true);
      (1, buf patch ~phys:true ~virt:true ~local:true);
      (2, buf patch ~phys:true ~virt:true);
    ]
    @ List.map
        (fun x -> (x, buf (global x) ~phys:false ~virt:true))
        [ 3; 4; 5; 6 ]
    @ [
        (9, buf (global 9) ~phys:true ~virt:true);
        (10, buf (global 10) ~phys:true ~virt:false);
        (11, buf (global 10) ~phys:true ~virt:true);
      ];
  let* _ =
    Result.map_error why
      (promote g ~client:priv_root ~subdevice:subdev ~obj:0 channel
         (List.filter (fun (_, b) -> not b.local) g.buffers)
         ~virt:None ~phys:None ())
  in
  let* _ = alloc ~parent:channel compute_class None in
  let* _ = alloc ~parent:channel copy_class None in
  Ok ()

(* The boot *)

let take p (fw : Images.t) fwsec ~taken =
  let c = p.chip in
  let sys = sys p ~taken in
  (* The queues and the table of their pages. *)
  let queue_pages = 2 * queue_size / page in
  let ptes = queue_pages + (round_up (queue_pages * 8) page / page) in
  let pt_size = round_up (ptes * 8) page in
  let* queues = sys (pt_size + (2 * queue_size)) in
  Window.write queues.w 0 (u64s queues.pages);
  let doorbell = Window.sub c.regs (Defs.nv_pgsp_queue_head 0) 4 in
  let q =
    Msgq.create (Window.sub queues.w pt_size (2 * queue_size)) ~doorbell
  in
  (* The GSP's arguments, its libos regions and their logs. *)
  let module Q = Defs.Queue_init_args in
  let module A = Defs.Gsp_arguments in
  let args =
    record A.sizeof (fun b ->
        let qa f = at A.message_queue_init_arguments f in
        set b (qa Q.shared_mem_phys_addr) (first queues);
        set b (qa Q.page_table_entry_count) ptes;
        set b (qa Q.cmd_queue_offset) pt_size;
        set b (qa Q.stat_queue_offset) (pt_size + queue_size);
        set b A.b_dmem_stack 1)
  in
  let* rm_args = sys page in
  Window.write rm_args.w 0 args;
  let* logs_mem = sys ~contiguous:true (List.length logs * log_size) in
  let regions =
    List.mapi
      (fun i n ->
        libos_region ("LOG" ^ n)
          ~pa:(first logs_mem + (log_size * i))
          ~size:log_size)
      logs
    @ [ libos_region "RMARGS" ~pa:(first rm_args) ~size:page ]
  in
  let* libos = sys page in
  Window.write libos.w 0 (String.concat "" regions);
  (* The images. *)
  let* radix = radix3 (fun n -> sys n) fw.gsp in
  let* signature = sys ~contiguous:true fw.signature.length in
  blit fw.signature signature.w 0;
  let* bootloader = sys ~contiguous:true fw.bootloader.image.length in
  blit fw.bootloader.image bootloader.w 0;
  let* meta = sys page in
  let* flush = sys page in
  let* flush = flush_registers c (first flush) in
  let wpr_meta =
    Fb_layout.wpr_meta (Chip.layout c.family) ~memory:p.memory
      ~gsp:{ address = first radix; size = fw.gsp.length }
      ~signature:{ address = first signature; size = fw.signature.length }
      ~bootloader:
        { address = first bootloader; size = fw.bootloader.image.length }
      ~code:fw.bootloader.code ~data:fw.bootloader.data
      ~manifest:fw.bootloader.manifest
  in
  Window.write meta.w 0 wpr_meta;
  (* The start, FWSEC set up for the GPU's memory. *)
  let* start =
    match (fw.start, fwsec, c.family) with
    | ( `Booter ({ load; unload } as booters : Images.booters),
        Some (fwsec : Vbios.fwsec),
        (Ampere | Ada) ) ->
        let* fwsec_at = vram p (String.length fwsec.image) in
        let larger =
          Int.max (String.length load.image) (String.length unload.image)
        in
        let* booter_at = vram p larger in
        Ok
          (Legacy
             {
               fwsec;
               frts = Fb_layout.frts ~memory:p.memory;
               booters;
               fwsec_at;
               booter_at;
             })
    | `Fmc m, None, Blackwell ->
        let* args = sys page in
        Window.write args.w 0
          (Falcon.cot_args ~libos:(first libos)
             ~wpr_meta:{ address = first meta; size = String.length wpr_meta });
        let* fmc = sys ~contiguous:true m.fmc.length in
        blit m.fmc fmc.w 0;
        Ok
          (Cot
             (Falcon.cot
                (Falcon.cot_payload ~args:(first args) ~fmc:(first fmc) m)))
    | _ -> Error "the firmware is of another family than the GPU"
  in
  Ok
    {
      p;
      q;
      lock = Rig_nv.Lock.create ();
      libos = first libos;
      meta = first meta;
      flush;
      start;
      fault = None;
      unanswered = false;
      sequencing = false;
      next = first_handle;
      clients = first_client;
      runlists = [];
      channels = Channels.empty;
      held = Objects.empty;
      buffers = [];
      taken;
    }

(* Nothing was written to the GPU yet, so a failure gives back what it took at
   once. *)
let create p fw fwsec =
  let taken = ref [] in
  match take p fw fwsec ~taken with
  | Ok _ as r -> r
  | Error _ as e ->
      give_back p taken;
      e
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      give_back p taken;
      Printexc.raise_with_backtrace e bt

(* A falcon's view of FWSEC at [pa], and of a booter at [pa]. *)
let fwsec_hs pa (f : Vbios.fwsec) =
  {
    Falcon.image = pa;
    code = { off = 0; pa = f.imem_pa; va = f.imem_va; size = f.imem_size };
    data = { off = f.imem_size; pa = f.dmem_pa; va = 0; size = f.dmem_size };
    pkc = f.pkc;
    engines = f.engines;
    ucode = f.ucode;
  }

let booter_hs pa (b : Images.booter) =
  {
    Falcon.image = pa;
    code = { off = fst b.code; pa = 0; va = fst b.code; size = snd b.code };
    data = { off = fst b.data; pa = 0; va = 0; size = snd b.data };
    pkc = b.pkc;
    engines = b.engines;
    ucode = b.ucode;
  }

let falcons g =
  match g.start with
  | Cot ops -> ops
  | Legacy { fwsec; frts; booters = { load; _ }; fwsec_at; booter_at } ->
      let f = Vbios.patch fwsec (Frts frts) in
      Window.write fwsec_at.w 0 f.image;
      Window.write booter_at.w 0 load.image;
      Falcon.legacy
        ~fwsec:(fwsec_hs fwsec_at.pa f)
        ~booter:(booter_hs booter_at.pa load)
        ~libos:g.libos ~wpr_meta:g.meta

let boot g =
  let c = g.p.chip in
  List.iter (fun (r, v) -> Chip.set c r v) g.flush;
  Function.set_bus_master c.fn true;
  (* What the GSP reads first, before it runs. *)
  let* () =
    send g Defs.nv_vgpu_msg_function_gsp_set_system_info (system_info g.p)
  in
  let* () = send g Defs.nv_vgpu_msg_function_set_registry (Rpc.registry keys) in
  let* () = Falcon.run c (falcons g) in
  (* The GSP runs: its queue, its first answer, then the golden context. *)
  let* () =
    Function.wait c.fn ~us:(answer_ms * 1000) "the GSP's message queue"
      (fun () -> Msgq.ready g.q)
  in
  let* _ =
    Rig_nv.Lock.protect g.lock (fun () ->
        wait_for g Defs.nv_vgpu_msg_event_gsp_init_done)
  in
  Chip.set c Defs.nv_pbus_bar1_block 0;
  if c.family = Blackwell then
    Chip.set c Defs.Blackwell.nv_virtual_function_priv_func_bar1_block_low_addr
      0;
  golden g

(* Takes the boot down as NVIDIA's RM does once the GSP's firmware has
   unloaded (kgspTeardown_TU102 on Ampere and Ada, kgspTeardown_GH100 on
   Blackwell): NVIDIA's driver refuses to boot a GPU whose protected memory
   (WPR2) is up (_kgspBootGspRm), and nouveau, nova-core and the RM each take it
   down at their stop. As the RM does, a wait for the firmware's suspension
   that runs out goes on, since the falcon's reset stops the GSP whatever it
   runs, and so does a FWSEC-SB that fails: the unloading booter's own check
   that WPR2 is down decides. An open finds WPR2 down before its boot, so only
   this boot's FWSEC raised it. A boot whose FWSEC never set it up is left as
   it is: of the RM's steps it skips only SB, which puts back the VBIOS's
   applications, and an attach's reset reruns the GPU's own boot. *)
let teardown g ~unloaded =
  let c = g.p.chip in
  let suspended () =
    if unloaded then Falcon.run c (Falcon.suspended c.family) else Ok ()
  in
  match g.start with
  | Cot _ -> suspended ()
  | Legacy _ when not (Chip.booted c) -> Ok ()
  | Legacy { fwsec; booters = { unload; _ }; fwsec_at; booter_at; _ } ->
      ignore (suspended () : (unit, string) result);
      let sb = Vbios.patch fwsec Sb in
      Window.write fwsec_at.w 0 sb.image;
      ignore
        (Falcon.run c (Falcon.fwsec_sb (fwsec_hs fwsec_at.pa sb))
          : (unit, string) result);
      Window.write booter_at.w 0 unload.image;
      Falcon.run c (Falcon.unload (booter_hs booter_at.pa unload))

(* The GSP stops every channel, unless the GPU cannot be reached or the GSP
   left a call unanswered, which an unload would wait for in vain, or the GSP
   never set its queue up and so runs nothing an unload would stop. A fault an
   event reported leaves the GSP answering, so its unload still runs. The boot
   is then taken down while the GPU still masters the bus, into memory that is
   still the process's. With its bus mastering off the GPU reaches system
   memory no more, whatever its channels do, and the boot's memory goes back.
   That write to the configuration space completes before it returns, so a GPU
   that answers after it took it: one that does not may still read the
   memory, which stays taken. A GPU whose boot raised WPR2 ran: if its unload
   did not stop its channels, or its teardown failed, it is lost, so that its
   next open resets it before any boot. *)
let stop g =
  let c = g.p.chip in
  let answers () = Option.is_none (Function.failed c.fn) in
  let ran = answers () && Chip.booted c in
  let unloaded =
    answers () && Msgq.ready g.q && (not g.unanswered)
    && Result.is_ok (unload g)
  in
  let down = if answers () then teardown g ~unloaded else Ok () in
  Function.set_bus_master c.fn false;
  if not (answers ()) then `Unknown
  else begin
    give_back g.p g.taken;
    match down with
    | Ok () when unloaded || not ran -> `Clean
    | Ok () | Error _ -> `Lost
  end
