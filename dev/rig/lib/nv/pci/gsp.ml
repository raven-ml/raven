(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

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

(* Encodings *)

let rm_alloc ~client ~parent ~obj ~cls params =
  let module A = Defs.Rpc_rm_alloc in
  record A.sizeof (fun b ->
      set b A.h_client client;
      set b A.h_parent parent;
      set b A.h_object obj;
      set b A.h_class cls;
      set b A.params_size (String.length params))
  ^ params

let rm_control ~client ~obj ~cmd params =
  let module C = Defs.Rpc_rm_control in
  record C.sizeof (fun b ->
      set b C.h_client client;
      set b C.h_object obj;
      set b C.cmd cmd;
      set b C.params_size (String.length params))
  ^ params

let rm_answer kind body =
  let size, status, params_size =
    match kind with
    | `Alloc -> Defs.Rpc_rm_alloc.(sizeof, status, params_size)
    | `Control -> Defs.Rpc_rm_control.(sizeof, status, params_size)
  in
  if String.length body < size then Error "an RM answer shorter than its header"
  else
    let n = get body params_size in
    if size + n > String.length body then
      Error (strf "an RM answer of %d bytes of parameters, which it lacks" n)
    else Ok (get body status, String.sub body size n)

let page_directory ~client ~device ~vaspace ~root ~entries =
  let module S = Defs.Rpc_set_page_directory in
  let module P = Defs.Set_page_directory in
  let p f = at S.params f in
  record S.sizeof (fun b ->
      set b S.h_client client;
      set b S.h_device device;
      (* No PASID: the address space is the GPU's own. *)
      set b S.pasid 0xffff_ffff;
      set b (p P.phys_address) root;
      set b (p P.num_entries) entries;
      (* NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE_VIDMEM: the root is
         in the GPU's memory. *)
      set b (p P.flags) 0x8;
      set b (p P.h_va_space) vaspace;
      set b (p P.pasid) 0xffff_ffff;
      set b (p P.sub_device_id) 1)

(* Unloading to level 6, NV2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_3's bit (RM's
   unload to the deepest state the GSP keeps). *)
let unload_level = 1 lsl 6

let unloading =
  record Defs.Rpc_unloading.sizeof (fun b ->
      set b Defs.Rpc_unloading.new_level unload_level)

let registry keys =
  let module T = Defs.Registry_table in
  let module E = Defs.Registry_entry in
  let n = List.length keys in
  let names_at = T.sizeof + (n * E.sizeof) in
  let names = String.concat "" (List.map (fun (k, _) -> k ^ "\000") keys) in
  let entries =
    record (n * E.sizeof) (fun b ->
        ignore
          (List.fold_left
             (fun (i, name) (k, v) ->
               let f (off, w) = ((i * E.sizeof) + off, w) in
               set b (f E.name_offset) name;
               set b (f E.type_) Defs.registry_table_entry_type_dword;
               set b (f E.data) v;
               set b (f E.length) 4;
               (i + 1, name + String.length k + 1))
             (0, names_at) keys))
  in
  record T.sizeof (fun b ->
      set b T.size (names_at + String.length names);
      set b T.num_entries n)
  ^ entries ^ names

(* The CPU sequencer *)

let sequence ~libos body =
  let module S = Defs.Rpc_cpu_sequencer in
  if String.length body < fst3 S.command_buffer then
    Error "a CPU sequence shorter than its header"
  else
    let n = get body S.cmd_index in
    let base = fst3 S.command_buffer in
    if base + (4 * n) > String.length body then
      Error (strf "a CPU sequence of %d words, longer than its message" n)
    else
      let word i = get body (base + (4 * i), 4) in
      let module L = Defs.Legacy in
      let gsp = Falcon.gsp and sec2 = Falcon.sec2 in
      let resume =
        Falcon.reset gsp `Riscv
        @ [
            Falcon.Write (L.nv_pgsp_falcon_mailbox0, libos land 0xffff_ffff);
            Falcon.Write (L.nv_pgsp_falcon_mailbox1, libos lsr 32);
          ]
        @ Falcon.start sec2
        @ [
            Poll
              ( "SEC2 to hand the GSP over",
                L.nv_pgc6_bsi_secure_scratch_14,
                mask L.nv_pgc6_bsi_secure_scratch_14_boot_stage_3_handoff,
                Is
                  (Defs
                   .nv_pgc6_bsi_secure_scratch_14_boot_stage_3_handoff_value_done
                  lsl fst L.nv_pgc6_bsi_secure_scratch_14_boot_stage_3_handoff)
              );
            Expect
              ( "SEC2 failed",
                sec2 + L.nv_pfalcon_falcon_mailbox0,
                0xffff_ffff,
                Falcon.Is 0 );
          ]
      in
      let rec go i acc =
        if i >= n then Ok (List.concat (List.rev acc))
        else
          let op = word i in
          (* A field of the payload after the opcode. *)
          let at (off, _) = word (i + 1 + (off / 4)) in
          let take size f =
            let k = size / 4 in
            if i + k >= n then
              Error (strf "a CPU sequence ending inside opcode %d" op)
            else go (i + k + 1) (f () :: acc)
          in
          if op = Defs.gsp_seq_buf_opcode_reg_write then
            let module P = Defs.Seq_reg_write in
            take P.sizeof (fun () -> [ Falcon.Write (at P.addr, at P.val_) ])
          else if op = Defs.gsp_seq_buf_opcode_reg_modify then
            let module P = Defs.Seq_reg_modify in
            (* The RM writes [(r & ~mask) | val], setting the value's bits
               outside the mask too. *)
            take P.sizeof (fun () ->
                let v = at P.val_ in
                [ Falcon.Modify (at P.addr, at P.mask lor v, v) ])
          else if op = Defs.gsp_seq_buf_opcode_reg_poll then
            let module P = Defs.Seq_reg_poll in
            (* The poll's timeout and error are the runner's. *)
            take P.sizeof (fun () ->
                let r = at P.addr in
                [
                  Falcon.Poll
                    (strf "register 0x%x" r, r, at P.mask, Is (at P.val_));
                ])
          else if op = Defs.gsp_seq_buf_opcode_delay_us then
            let module P = Defs.Seq_delay_us in
            take P.sizeof (fun () -> [ Falcon.Delay (at P.val_) ])
          else if op = Defs.gsp_seq_buf_opcode_reg_store then
            (* A register's value for the GSP's save area, which it does not
               read back from the CPU. *)
            take Defs.Seq_reg_store.sizeof (fun () -> [])
          else if op = Defs.gsp_seq_buf_opcode_core_reset then
            take 0 (fun () ->
                Falcon.reset gsp `Falcon
                @ [
                    Falcon.Modify
                      ( gsp + L.nv_pfalcon_fbif_ctl,
                        mask L.nv_pfalcon_fbif_ctl_allow_phys_no_ctx,
                        1 lsl fst L.nv_pfalcon_fbif_ctl_allow_phys_no_ctx );
                    Write (gsp + L.nv_pfalcon_falcon_dmactl, 0);
                  ])
          else if op = Defs.gsp_seq_buf_opcode_core_start then
            take 0 (fun () -> Falcon.start gsp)
          else if op = Defs.gsp_seq_buf_opcode_core_wait_for_halt then
            take 0 (fun () -> Falcon.wait_halt gsp)
          else if op = Defs.gsp_seq_buf_opcode_core_resume then
            take 0 (fun () -> resume)
          else Error (strf "a CPU sequence of unknown opcode %d" op)
      in
      go 0 []

(* Booting *)

type placement = {
  chip : Chip.t;
  memory : int;
  fn : Function.t;
  tables : Page_table.t;
  bar : Window.t;
  space : Space.t;
}

type buffer = { size : int; phys : bool; virt : bool; local : bool }

type t = {
  p : placement;
  q : Msgq.t;
  lock : Mutex.t;
  libos : int; (* the libos arguments' bus address, for a resumption *)
  mutable fault : string option;
  mutable next : int; (* the next handle the process names an object with *)
  mutable clients : int; (* the next client handle *)
  mutable runlists : (int * int) list; (* each engine's runlist *)
  channels : (int, int) Hashtbl.t; (* each channel's runlist *)
  mutable buffers : (int * buffer) list; (* the graphics context's *)
  taken : sys list ref; (* the system memory the boot took *)
}

(* System memory for the GSP: [w], its address in the space and the bus address
   of each 4 KiB page. *)
and sys = { w : Window.t; va : int; pages : int list }

let sys p ~taken ?(contiguous = false) n =
  let n = round_up n page in
  let align =
    if contiguous && n > Machine.page (Function.machine p.fn) then 2 * mib
    else page
  in
  match Space.alloc ~align p.space (round_up n align) with
  | None -> Error "no addresses for the GSP's system memory"
  | Some va ->
      let* w, runs = Function.alloc_dma ~contiguous ~va p.fn n in
      let pages =
        List.concat_map
          (fun (a, len) -> List.init (len / page) (fun i -> a + (i * page)))
          runs
      in
      let s = { w; va; pages } in
      taken := s :: !taken;
      Ok s

let give_back p taken =
  List.iter
    (fun s ->
      Function.free_dma p.fn s.w;
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
    | `Booter (b : Images.booter) ->
        room Vbios.window + room (String.length b.image)
    | `Fmc _ -> 0
  in
  round_up (page + images) (2 * mib)

(* GPU memory a falcon reads, from the boot pool, which the memory BAR reaches
   however small: its physical address and the window on it. *)
let vram p n =
  match Page_table.palloc ~boot:true p.tables n with
  | None -> Error "no GPU memory in the boot pool for a falcon's image"
  | Some pa -> Ok (pa, Window.sub p.bar pa (round_up n page))

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
  let bar i = match Function.bar p.fn i with Some (a, _) -> a | None -> 0 in
  let f = p.fn in
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
  let n = Layout.radix3 image.length in
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

let wpr_meta (fw : Images.t) family ~memory ~radix3 ~bootloader ~signature =
  let module W = Defs.Wpr_meta in
  record W.sizeof (fun b ->
      let s (f, v) = set b f v in
      Bytes.set_int64_le b (fst W.magic) Defs.gsp_fw_wpr_meta_magic;
      set b W.revision Defs.gsp_fw_wpr_meta_revision;
      set b W.size_of_bootloader fw.bootloader.image.length;
      set b W.sysmem_addr_of_bootloader bootloader;
      set b W.size_of_radix3_elf fw.gsp.length;
      set b W.sysmem_addr_of_radix3_elf radix3;
      set b W.size_of_signature (round_up fw.signature.length page);
      set b W.sysmem_addr_of_signature signature;
      set b W.bootloader_code_offset fw.bootloader.code;
      set b W.bootloader_data_offset fw.bootloader.data;
      set b W.bootloader_manifest_offset fw.bootloader.manifest;
      match (family : Chip.family) with
      | Blackwell -> List.iter s Layout.fmc_sizes
      | Ampere | Ada ->
          List.iter s
            (Layout.wpr ~memory ~boot:fw.bootloader.image.length
               ~image:fw.gsp.length))

(* Calls *)

(* How long the GSP takes to answer a call, and to boot. *)
let answer_ms = 10_000

let drain g ~want =
  let rec go () =
    match Msgq.receive g.q with
    | None -> Ok None
    | Some (Error why) -> Error why
    | Some (Ok m) -> (
        let* () =
          if m.fn <> Defs.nv_vgpu_msg_event_gsp_run_cpu_sequencer then Ok ()
          else
            let* ops = sequence ~libos:g.libos m.body in
            Falcon.run g.p.chip ops
        in
        (match Msgq.fault m with
        | Some f when g.fault = None -> g.fault <- Some f
        | _ -> ());
        match want with
        | Some fn when m.fn = fn ->
            if m.result <> 0 then
              Error (strf "the GSP failed call %d: 0x%x" fn m.result)
            else Ok (Some m.body)
        | _ -> go ())
  in
  go ()

let wait_for g fn =
  let found = ref (Error (strf "no answer to call %d" fn)) in
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
    Chip.wait g.p.chip
      (strf "the GSP's answer to call %d" fn)
      ~ms:answer_ms answered
  in
  !found

let send g fn body =
  Chip.wait g.p.chip "room in the GSP's command queue" ~ms:answer_ms (fun () ->
      Msgq.send g.q fn body)

let call g fn body =
  Mutex.protect g.lock (fun () ->
      let* () = send g fn body in
      wait_for g fn)

let check g =
  if Mutex.try_lock g.lock then begin
    (match drain g ~want:None with
    | Ok _ -> ()
    | Error why -> if g.fault = None then g.fault <- Some why);
    Mutex.unlock g.lock
  end;
  g.fault

let unload g =
  Result.map ignore
    (call g Defs.nv_vgpu_msg_function_unloading_guest_driver unloading)

(* The RM *)

(* The GSP's own client, which the golden context belongs to (NV01_ROOT's handle
   in RM's internal range), the first handle of the clients the process makes,
   and the first of the objects it names. *)
let priv_root = 0xc1e00004
let first_client = 0xc1000000
let first_handle = 0xcf000000

let handle g =
  let h = g.next in
  g.next <- h + 1;
  h

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

let pget p f = get (string_of_params p) f

let pset (p : Rig_nv.params) (off, n) x =
  for i = 0 to n - 1 do
    Bigarray.Array1.set p (off + i) (Char.chr ((x lsr (8 * i)) land 0xff))
  done

let params n =
  Bigarray.Array1.init Bigarray.char Bigarray.c_layout n (fun _ -> '\000')

let control g ~client obj cmd p =
  let body = Option.fold ~none:"" ~some:string_of_params p in
  let* answer =
    call g Defs.nv_vgpu_msg_function_gsp_rm_control
      (rm_control ~client ~obj ~cmd body)
  in
  let* status, out = rm_answer `Control answer in
  if status <> 0 then
    Error (strf "the RM refused command 0x%x: 0x%x" cmd status)
  else (
    Option.iter (blit_params out) p;
    Ok ())

let alloc_object g ~client ~parent cls p =
  let obj = handle g in
  let body = Option.fold ~none:"" ~some:string_of_params p in
  let* answer =
    call g Defs.nv_vgpu_msg_function_gsp_rm_alloc
      (rm_alloc ~client ~parent ~obj ~cls body)
  in
  let* status, _ = rm_answer `Alloc answer in
  if status <> 0 then Error (strf "the RM refused class 0x%x: 0x%x" cls status)
  else Ok obj

(* Memory of the GSP's objects: GPU memory mapped at addresses of the GPU's
   space. *)
let object_memory g n =
  match Page_table.alloc ~contiguous:true g.p.tables n with
  | None -> Error "no GPU memory for the GSP's objects"
  | Some m -> Ok m

let paddr (m : Page_table.mapping) = fst (List.hd m.pages)

(* A memory descriptor of a channel's allocation, cached: the GSP's RM gives
   system memory described cached the coherent aperture, whose accesses snoop
   the processors' caches, and any other the non-coherent one
   (kgmmuGetHwPteApertureFromMemdesc_GM107). A channel's USERD in system memory,
   which the host writes, must be snooped. *)
let memdesc p field ~base ~size ~space =
  let module M = Defs.Memory_desc in
  pset p (at field M.base) base;
  pset p (at field M.size) size;
  pset p (at field M.address_space) space;
  pset p (at field M.cache_attrib) Defs.nv_memory_cached

(* A channel's instance block, RAMFC and method buffer, which CPU-RM gives the
   GSP: a page, its first 0x200 bytes the RAMFC, and 0x5000 bytes. *)
let instance_size = 0x1000
let ramfc_size = 0x200
let method_buffer_size = 0x5000
let userd_size = 0x400

(* The size the error notifier is described with. *)
let notifier_size = 0xecc

let channel_memory g ~client ~locate p =
  let module G = Defs.Gpfifo_alloc in
  let* ramfc = object_memory g instance_size in
  let* mthd =
    match Page_table.palloc g.p.tables method_buffer_size with
    | Some pa -> Ok pa
    | None -> Error "no GPU memory for a channel's method buffer"
  in
  let fb = Defs.addr_fbmem in
  memdesc p G.ramfc_mem ~base:(paddr ramfc) ~size:ramfc_size ~space:fb;
  memdesc p G.instance_mem ~base:(paddr ramfc) ~size:instance_size ~space:fb;
  memdesc p G.mthdbuf_mem ~base:mthd ~size:method_buffer_size ~space:fb;
  if client = priv_root || pget p G.h_object_error = 0 then Ok ()
  else begin
    (* The error notifier is described in no memory: the GSP reports a channel's
       error as an event (RC_TRIGGERED), which the path reads instead. *)
    memdesc p G.error_notifier_mem ~base:0 ~size:notifier_size ~space:0;
    let h = pget p (fst3 G.h_userd_memory, 4) in
    let off = pget p (fst3 G.userd_offset, 8) in
    match locate h off with
    | None -> Error (strf "no memory of handle 0x%x for a channel's USERD" h)
    | Some (where, a) ->
        let space =
          match where with `Gpu -> fb | `System -> Defs.addr_sysmem
        in
        memdesc p G.userd_mem ~base:a ~size:userd_size ~space;
        Ok ()
  end

(* The context buffers a compute engine's channel needs, promoted twice: their
   physical memory, then their virtual addresses. *)
let promote g ~client ~subdevice channel buffers ?(have = []) ~virt ~phys () =
  let module Pr = Defs.Promote_ctx in
  let module E = Defs.Promote_entry in
  let p = params Pr.sizeof in
  pset p Pr.entry_count (List.length buffers);
  pset p Pr.engine_type Defs.nv2080_engine_type_graphics;
  pset p Pr.h_chan_client client;
  pset p Pr.h_object channel;
  let rec fill i made = function
    | [] -> Ok (List.rev made)
    | (id, b) :: rest ->
        let v = Option.value virt ~default:b.virt
        and ph = Option.value phys ~default:b.phys in
        let* m =
          match List.assoc_opt id have with
          | Some m -> Ok m
          | None -> object_memory g b.size
        in
        let f x = elt Pr.promote_entry i x in
        pset p (f E.buffer_id) id;
        pset p (f E.gpu_virt_addr) (if v then m.Page_table.va else 0);
        pset p (f E.b_initialize) (Bool.to_int ph);
        pset p (f E.gpu_phys_addr) (if ph then paddr m else 0);
        pset p (f E.size) (if ph then b.size else 0);
        (* NV2080_CTRL_GPU_PROMOTE_CTX_PHYS_ATTR: in the GPU's memory,
           cached. *)
        pset p (f E.phys_attr) (if ph then 0x4 else 0);
        pset p (f E.b_nonmapped) (Bool.to_int (ph && not v));
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
  let top = List.nth (Mmu.levels v) (List.length (Mmu.levels v) - 1) in
  1 lsl (Mmu.bits v - top)

(* The subdevice a client made, on which compute engines' contexts are
   promoted. *)
type objects = { mutable subdevice : int }

let alloc g ~client ~locate ~objects ~parent cls p =
  let channel_class, compute_class, _ = classes g.p.chip.family in
  let* () =
    match p with
    | Some p when cls = channel_class -> channel_memory g ~client ~locate p
    | _ -> Ok ()
  in
  let* obj =
    if cls = Defs.nv01_root then
      let* _ = alloc_object g ~client ~parent cls p in
      Ok client
    else alloc_object g ~client ~parent cls p
  in
  if cls = channel_class then
    Option.iter
      (fun p ->
        let e = pget p Defs.Gpfifo_alloc.engine_type in
        Hashtbl.replace g.channels obj
          (Option.value ~default:0 (List.assoc_opt e g.runlists)))
      p;
  if cls = Defs.nv20_subdevice_0 then objects.subdevice <- obj;
  let* () =
    if client = priv_root then Ok ()
    else if cls = Defs.fermi_vaspace_a then
      let* _ =
        call g Defs.nv_vgpu_msg_function_set_page_directory
          (page_directory ~client ~device:parent ~vaspace:obj
             ~root:(Page_table.root g.p.tables)
             ~entries:(root_entries g.p.chip))
      in
      Ok ()
    else if cls = compute_class then
      let bufs = List.filter (fun (k, _) -> List.mem k [ 0; 1; 2 ]) g.buffers in
      let* phys =
        promote g ~client ~subdevice:objects.subdevice parent bufs
          ~virt:(Some false) ~phys:None ()
      in
      let* _ =
        promote g ~client ~subdevice:objects.subdevice parent bufs ~have:phys
          ~virt:None ~phys:(Some false) ()
      in
      Ok ()
    else Ok ()
  in
  Ok obj

(* A channel's work submit token carries its runlist in bits 31:16, and on
   Blackwell bit 30 set. *)
let token g obj p =
  let module W = Defs.Work_submit_token in
  let runlist = Option.value ~default:0 (Hashtbl.find_opt g.channels obj) in
  let blackwell = g.p.chip.family = Blackwell in
  pset p W.work_submit_token
    (pget p W.work_submit_token lor (runlist lsl 16)
    lor if blackwell then 1 lsl 30 else 0)

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
    free =
      (fun ~parent:_ _ -> Error "the GSP's objects live as long as the GSP");
  }

let rm g ~locate =
  let client = g.clients in
  g.clients <- client + 1;
  let r = client_rm g ~client ~locate in
  let* _ =
    r.alloc ~parent:0 Defs.nv01_root (Some (params Defs.Nv0000_alloc.sizeof))
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
  let dp = params D.sizeof in
  pset dp D.h_client_share rm.client;
  pset dp D.va_mode Defs.nv_device_allocation_vamode_optional_multiple_vaspaces;
  let* device = rm.alloc ~parent:rm.client Defs.nv01_device_0 (Some dp) in
  let* subdevice =
    rm.alloc ~parent:device Defs.nv20_subdevice_0
      (Some (params Defs.Nv2080_alloc.sizeof))
  in
  let vp = params V.sizeof in
  pset vp V.va_base va_base;
  pset vp V.va_size va_size;
  pset vp V.flags
    (Defs.nv_vaspace_allocation_flags_enable_page_faulting
   lor Defs.nv_vaspace_allocation_flags_is_externally_owned);
  let* vaspace = rm.alloc ~parent:device Defs.fermi_vaspace_a (Some vp) in
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
  let p = params S.sizeof in
  let* () =
    rm.control subdevice Defs.nv2080_ctrl_cmd_internal_static_kgr_get_info
      (Some p)
  in
  let info i =
    let engine = elt S.engine_info 0 (0, 0) in
    pget p (elt L.info_list i (fst engine + fst I.data, snd I.data))
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

(* The GPU memory the GSP's internal address space reserves for its own page
   directory entries: 512 MiB, whose tables the server copies. *)
let reserved = 512 * mib

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
  let alloc = alloc g ~client:priv_root ~locate ~objects in
  let* _ =
    alloc ~parent:0 Defs.nv01_root (Some (params Defs.Nv0000_alloc.sizeof))
  in
  let dp = params Defs.Nv0080_alloc.sizeof in
  pset dp Defs.Nv0080_alloc.h_client_share priv_root;
  let* dev = alloc ~parent:priv_root Defs.nv01_device_0 (Some dp) in
  let* subdev =
    alloc ~parent:dev Defs.nv20_subdevice_0
      (Some (params Defs.Nv2080_alloc.sizeof))
  in
  let* vaspace =
    alloc ~parent:dev Defs.fermi_vaspace_a
      (Some (params Defs.Vaspace_alloc.sizeof))
  in
  let control = control g ~client:priv_root in
  (* Each engine's runlist, from the device information table. *)
  let module T = Defs.Device_info_table in
  let module E = Defs.Device_entry in
  let t = params T.sizeof in
  let* () =
    control subdev Defs.nv2080_ctrl_cmd_fifo_get_device_info_table (Some t)
  in
  g.runlists <-
    List.init (pget t T.num_entries) (fun i ->
        let data k = pget t (elt T.entries i (elt E.engine_data k (0, 4))) in
        (data 2, data 3));
  (* The reserved page directory entries. *)
  let tables = g.p.tables in
  let* va =
    match Space.alloc ~align:reserved (Page_table.space tables) reserved with
    | Some va -> Ok va
    | None -> Error "no addresses for the GSP's page directory"
  in
  let* levels =
    match Page_table.tables tables ~va reserved with
    | Some l -> Ok l
    | None -> Error "no GPU memory for the GSP's page directory"
  in
  let module R = Defs.Reserved_pdes in
  let r = params R.sizeof in
  pset r R.page_size reserved;
  pset r R.num_levels_to_copy (List.length levels);
  pset r R.virt_addr_lo va;
  pset r R.virt_addr_hi (va + reserved - 1);
  let shifts = List.rev (Mmu.levels (Mmu.version g.p.chip.family)) in
  List.iteri
    (fun i table ->
      let f x = elt R.levels i x in
      pset r (f R.levels_phys_address) table;
      pset r (f R.levels_size)
        (if i = 0 then root_entries g.p.chip * 8 else page);
      pset r (f R.levels_page_shift) (List.nth shifts i);
      pset r (f R.levels_aperture) 1)
    levels;
  let* () =
    control vaspace Defs.nv90f1_ctrl_cmd_vaspace_copy_server_reserved_pdes
      (Some r)
  in
  (* The channel. *)
  let* area = object_memory g page in
  let module G = Defs.Gpfifo_alloc in
  let gp = params G.sizeof in
  pset gp G.gp_fifo_offset area.va;
  pset gp G.gp_fifo_entries golden_entries;
  pset gp G.engine_type Defs.nv2080_engine_type_graphics;
  pset gp G.cid golden_cid;
  pset gp G.h_va_space vaspace;
  pset gp (fst3 G.userd_offset, 8) golden_userd;
  memdesc gp G.userd_mem
    ~base:(paddr area + golden_userd)
    ~size:golden_userd_size ~space:Defs.addr_fbmem;
  pset gp G.internal_flags golden_internal_flags;
  pset gp G.flags golden_flags;
  let channel_class, compute_class, copy_class = classes g.p.chip.family in
  let* channel = alloc ~parent:dev channel_class (Some gp) in
  (* The context buffers' sizes. *)
  let module C = Defs.Context_buffers_info in
  let module Cb = Defs.Context_buffers in
  let module B = Defs.Context_buffer in
  let c = params C.sizeof in
  let* () =
    control subdev
      Defs.nv2080_ctrl_cmd_internal_static_kgr_get_context_buffers_info (Some c)
  in
  let info ?(add = 0) ?align idx =
    let f x =
      pget c (elt C.engine_context_buffers_info 0 (elt Cb.engine idx x))
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
    promote g ~client:priv_root ~subdevice:subdev channel
      (List.filter (fun (_, b) -> not b.local) g.buffers)
      ~virt:None ~phys:None ()
  in
  let* _ = alloc ~parent:channel compute_class None in
  let* _ = alloc ~parent:channel copy_class None in
  Ok ()

(* The boot *)

let start p (fw : Images.t) ~taken =
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
  Window.write meta.w 0
    (wpr_meta fw c.family ~memory:p.memory ~radix3:(first radix)
       ~bootloader:(first bootloader) ~signature:(first signature));
  let g =
    {
      p;
      q;
      lock = Mutex.create ();
      libos = first libos;
      fault = None;
      next = first_handle;
      clients = first_client;
      runlists = [];
      channels = Hashtbl.create 8;
      buffers = [];
      taken;
    }
  in
  (* What the GSP reads first, before it runs. *)
  let* () =
    send g Defs.nv_vgpu_msg_function_gsp_set_system_info (system_info p)
  in
  let* () = send g Defs.nv_vgpu_msg_function_set_registry (registry keys) in
  (* The start. *)
  let* ops =
    match (fw.start, c.family) with
    | `Booter b, (Ampere | Ada) ->
        let rom = Vbios.read c in
        let* f = Vbios.fwsec rom ~frts:(Layout.frts ~memory:p.memory) in
        let* fwsec_pa, fwsec_w = vram p (String.length f.image) in
        Window.write fwsec_w 0 f.image;
        let* booter_pa, booter_w = vram p (String.length b.image) in
        Window.write booter_w 0 b.image;
        let fwsec =
          {
            Falcon.image = fwsec_pa;
            code =
              { off = 0; pa = f.imem_pa; va = f.imem_va; size = f.imem_size };
            data =
              { off = f.imem_size; pa = f.dmem_pa; va = 0; size = f.dmem_size };
            pkc = f.pkc;
            engines = f.engines;
            ucode = f.ucode;
          }
        in
        let booter =
          {
            Falcon.image = booter_pa;
            code =
              { off = fst b.code; pa = 0; va = fst b.code; size = snd b.code };
            data = { off = fst b.data; pa = 0; va = 0; size = snd b.data };
            pkc = b.pkc;
            engines = b.engines;
            ucode = b.ucode;
          }
        in
        Ok
          (Falcon.legacy ~fwsec ~booter ~libos:(first libos)
             ~wpr_meta:(first meta))
    | `Fmc m, Blackwell ->
        let* args = sys page in
        Window.write args.w 0
          (Falcon.cot_args ~libos:(first libos) ~wpr_meta:(first meta));
        let* fmc = sys ~contiguous:true m.fmc.length in
        blit m.fmc fmc.w 0;
        Ok
          (Falcon.cot
             (Falcon.cot_payload ~args:(first args) ~fmc:(first fmc) m))
    | _ -> Error "the firmware is of another family than the GPU"
  in
  let* () = Falcon.run c ops in
  (* The GSP runs: its queue, its first answer, then the golden context. *)
  let* () =
    Chip.wait c "the GSP's message queue" ~ms:answer_ms (fun () -> Msgq.ready q)
  in
  let* _ =
    Mutex.protect g.lock (fun () ->
        wait_for g Defs.nv_vgpu_msg_event_gsp_init_done)
  in
  Chip.set c Defs.nv_pbus_bar1_block 0;
  if c.family = Blackwell then
    Chip.set c Defs.Blackwell.nv_virtual_function_priv_func_bar1_block_low_addr
      0;
  let* () = golden g in
  Ok g

(* A failed boot gives back the system memory it took once the GPU masters the
   bus no more: the GSP or a falcon may still be reading it. *)
let boot p fw =
  let taken = ref [] in
  let failed () =
    Chip.bus_master p.fn false;
    give_back p taken
  in
  match start p fw ~taken with
  | Ok _ as r -> r
  | Error _ as e ->
      failed ();
      e
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      failed ();
      Printexc.raise_with_backtrace e bt

let free g = give_back g.p g.taken
