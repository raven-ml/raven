(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Any domain may call any function. A GPU's objects are made once, under
   [gpus_lock], and kept for the process, its registered channels and whether a
   device of it is open under the same lock; the host ranges [map_host] maps are
   kept under [ranges_lock]. *)

module D = Defs

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind
let params = Rm.params
let get = Rm.get
let set = Rm.set
let page = 0x1000
let huge_page = 2 lsl 20

(* Video memory of this size or more takes 2 MiB pages. *)
let huge_from = 8 lsl 20

(* The usermode object's page, whose register at [doorbell_at]
   (NVC361_NOTIFY_CHANNEL_PENDING) wakes a channel. *)
let usermode_bytes = 0x10000
let doorbell_at = 0x90

(* The addresses a channel's registration with unified memory takes. *)
let channel_range = 0x400_0000

(* The address space's first address and size, and the virtual memory object's
   last address: 49 bits. *)
let va_base = 0x1000
let va_size = 0x1fffffb000000
let va_limit = 0x1ffffffffffff
let round_up n a = (n + a - 1) / a * a
let bits (lo, n) v = (v land ((1 lsl n) - 1)) lsl lo

(* GPUs *)

type gpu = {
  c : Rm.t;
  minor : int;
  device : int;
  subdevice : int;
  virtmem : int;
  vaspace : int;
  uuid : string;
  doorbell : int;
  facts : Rig_nv.gpu;
  budget : int;
  index : int; (* its number, in bus order *)
  mutable refused : int list; (* the GPUs peer access was refused with *)
  mutable channels : (int * int) list; (* registered, with their ranges *)
  mutable opened : bool; (* whether a device of it is open and not stopped *)
}

(* GPU memory *)

type mem = {
  va : int;
  size : int;
  handle : int;
  host : int option; (* the process's address of it, if it maps it *)
  video : int option; (* the GPU whose memory it is, by number *)
  of_ : owner;
}

and owner =
  | Own of Va.t (* allocated by this GPU, from these addresses *)
  | Host of range (* host memory [map_host] mapped *)
  | Peer (* another GPU's memory, mapped for this one *)

(* Host memory mapped for the GPUs: described to the RM once, under the first
   GPU that maps it, and mapped for each GPU that maps it, counted. The last
   unmapping frees the range and its description. A range covers whole pages,
   which unified memory requires of its bounds. The GPUs address it from the low
   range: the host's own address may lie above the 40 bits a semaphore takes. *)
and range = {
  addr : int;
  bytes : int;
  gpu_addr : int; (* the GPUs' address of [addr] *)
  descriptor : int;
  parent : gpu;
  mutable users : (gpu * int) list;
}

let key : mem Type.Id.t = Type.Id.make ()
let gpus_lock = Mutex.create ()

(* A file of [g]'s the RM maps memory through: each mapping has its own, as the
   driver keeps one mapping context per file. *)
let with_file g f =
  let* fd = Rm.open_file (strf "/dev/nvidia%d" g.minor) in
  Fun.protect
    ~finally:(fun () -> Rm.close fd)
    (fun () ->
      let* () = Rm.register fd ~ctl:g.c.ctl in
      f fd)

(* Whether the status [s] of [what] is NV_OK: [Ok false] for
   NV_ERR_NO_MEMORY. *)
let fits g what s =
  if s = D.nv_err_no_memory then Ok false
  else
    let* () = Rm.check g.c what s in
    Ok true

(* Maps the memory object [h] of [size] bytes into the process at [va],
   uncached: [Ok false] if the GPU's window onto its memory (BAR1) has no
   room. *)
let map_to_cpu g h size va =
  with_file g @@ fun fd ->
  let module W = D.Nvos33_with_fd in
  let module M = D.Nvos33 in
  let w = params W.sizeof in
  let at (f, n) = (fst W.params + f, n) in
  set w W.fd fd;
  set w (at M.h_client) g.c.root;
  set w (at M.h_device) g.device;
  set w (at M.h_memory) h;
  set w (at M.length) size;
  set w (at M.flags)
    (bits D.nvos33_flags_caching_type D.nvos33_flags_caching_type_uncached);
  let what = "mapping GPU memory into the process" in
  let* () = Rm.escape g.c.ctl D.nv_esc_rm_map_memory w what in
  let* ok = fits g what (get w (at M.status)) in
  if not ok then Ok false
  else
    let* () = Rm.map fd va size in
    Ok true

(* Maps the memory object [h] at [va] of [g]'s virtual memory: [Ok false] if [g]
   has no memory for the mapping's page tables. *)
let map_dma g va size h =
  let (module R : D.RELEASE) = g.c.layouts in
  let module M = R.Nvos46 in
  let m = params M.sizeof in
  set m M.h_client g.c.root;
  set m M.h_device g.device;
  set m M.h_dma g.virtmem;
  set m M.h_memory h;
  set m M.length size;
  set m M.flags
    (bits D.nvos46_flags_page_size D.nvos46_flags_page_size_4kb
    lor bits D.nvos46_flags_cache_snoop D.nvos46_flags_cache_snoop_enable
    lor bits D.nvos46_flags_dma_offset_fixed
          D.nvos46_flags_dma_offset_fixed_true);
  set m M.dma_offset va;
  let what = "mapping GPU memory" in
  let* () = Rm.escape g.c.ctl D.nv_esc_rm_map_memory_dma m what in
  let* ok = fits g what (get m M.status) in
  if ok && get m M.dma_offset <> va then
    Error "mapping GPU memory: the driver chose another address"
  else Ok ok

let set_uuid p (at, _) uuid =
  String.iteri (fun i ch -> Bigarray.Array1.set p (at + i) ch) uuid

(* Maps the memory object [h], at [va] of unified memory, for [g]. *)
let map_external g va size h =
  let module U = D.Uvm_map_external_allocation in
  let module G = D.Uvm_gpu_mapping in
  let u = params U.sizeof in
  set u U.base va;
  set u U.length size;
  set u U.rm_ctrl_fd g.c.ctl;
  set u U.h_client g.c.root;
  set u U.h_memory h;
  set u U.gpu_attributes_count 1;
  let at, _, _ = U.per_gpu_attributes in
  set_uuid u (at + fst G.gpu_uuid, 16) g.uuid;
  set u
    (at + fst G.gpu_mapping_type, snd G.gpu_mapping_type)
    D.uvm_gpu_mapping_type_read_write_atomic;
  let what = "mapping GPU memory" in
  let* s = Rm.uvm g.c D.uvm_map_external_allocation u U.rm_status what in
  fits g what s

(* Frees the unified memory range at [va], unmapping it from every GPU. *)
let uvm_free c va size =
  let (module R : D.RELEASE) = c.Rm.layouts in
  let module U = R.Uvm_free in
  let u = params U.sizeof in
  set u U.base va;
  Option.iter (fun l -> set u l size) U.length;
  Rm.uvm_call c D.uvm_free u U.rm_status "unmapping GPU memory"

(* Makes the unified memory range of [size] bytes at [va] and maps the memory
   object [h] there for [g], in [g]'s virtual memory first. [Ok false] if [g]
   has no memory for it; the range is then freed, so that the addresses stay
   free. *)
let uvm_map g va size h =
  let module E = D.Uvm_create_external_range in
  let e = params E.sizeof in
  set e E.base va;
  set e E.length size;
  let* () =
    Rm.uvm_call g.c D.uvm_create_external_range e E.rm_status
      "reserving GPU addresses"
  in
  let mapped =
    let* ok = map_dma g va size h in
    if ok then map_external g va size h else Ok false
  in
  match mapped with
  | Ok true -> Ok true
  | Ok false | Error _ ->
      ignore (uvm_free g.c va size : (unit, string) result);
      mapped

(* Unmaps for [g] the [size] bytes at [va] of a range that stays. *)
let uvm_unmap g va size =
  let module U = D.Uvm_unmap_external in
  let u = params U.sizeof in
  set u U.base va;
  set u U.length size;
  set_uuid u U.gpu_uuid g.uuid;
  Rm.uvm_call g.c D.uvm_unmap_external u U.rm_status "unmapping GPU memory"

(* The RM's refusals of host memory to describe: an address no memory backs,
   such as an unmapped page, or memory it cannot pin. *)
let refusals =
  [ D.nv_err_invalid_address; D.nv_err_invalid_argument; D.nv_err_no_memory ]

(* Describes the process's memory at [va] to the RM as system memory of [g]: the
   description's handle, or [None] if the RM refuses the memory. *)
let describe g va size =
  let h = Rm.handle () in
  let module W = D.Nvos02_with_fd in
  let module O = D.Nvos02 in
  let w = params W.sizeof in
  let at (f, n) = (fst W.params + f, n) in
  set w W.fd (-1);
  set w (at O.h_root) g.c.root;
  set w (at O.h_object_parent) g.device;
  set w (at O.h_object_new) h;
  set w (at O.h_class) D.nv01_memory_system_os_descriptor;
  set w (at O.p_memory) va;
  set w (at O.limit) (size - 1);
  set w (at O.flags)
    (bits D.nvos02_flags_physicality D.nvos02_flags_physicality_noncontiguous
    lor bits D.nvos02_flags_coherency D.nvos02_flags_coherency_cached
    lor bits D.nvos02_flags_mapping D.nvos02_flags_mapping_no_map);
  let what = "describing host memory to the GPU" in
  let* () =
    with_file g (fun fd -> Rm.escape fd D.nv_esc_rm_alloc_memory w what)
  in
  let s = get w (at O.status) in
  if List.mem s refusals then Ok None
  else
    let* () = Rm.check g.c what s in
    Ok (Some h)

let free_object g h =
  ignore ((Rm.rm g.c).free ~parent:g.device h : (unit, string) result)

(* The parameters of video memory in pages of [page_size]. *)
let video_params g ~contiguous ~page_size size =
  let module A = D.Memory_alloc in
  let huge = page_size > page in
  let p = params A.sizeof in
  set p A.owner g.c.root;
  set p A.alignment page_size;
  set p A.limit (size - 1);
  set p A.format D.nv_mmu_pte_kind_generic_memory;
  set p A.size size;
  set p A.type_ D.nvos32_type_image;
  set p A.attr
    (bits D.nvos32_attr_physicality
       (if contiguous then D.nvos32_attr_physicality_contiguous
        else D.nvos32_attr_physicality_allow_noncontiguous)
    lor (if huge then bits D.nvos32_attr_page_size D.nvos32_attr_page_size_huge
         else 0)
    lor bits D.nvos32_attr_location D.nvos32_attr_location_vidmem);
  set p A.attr2
    (bits D.nvos32_attr2_gpu_cacheable D.nvos32_attr2_gpu_cacheable_yes
    lor (if huge then
           bits D.nvos32_attr2_page_size_huge D.nvos32_attr2_page_size_huge_2mb
         else 0)
    lor bits D.nvos32_attr2_zbc D.nvos32_attr2_zbc_prefer_no_zbc);
  set p A.flags
    (D.nvos32_alloc_flags_map_not_required
   lor D.nvos32_alloc_flags_memory_handle_provided
   lor D.nvos32_alloc_flags_alignment_force
   lor D.nvos32_alloc_flags_ignore_bank_placement
   lor D.nvos32_alloc_flags_persistent_vidmem);
  p

(* Takes [size] addresses of [space] for [f], giving them back if [f] fails or
   answers [None]. *)
let with_addresses space ~align size f =
  match Va.alloc space ~align size with
  | None -> Ok None
  | Some va -> (
      match f va with
      | Ok (Some _) as r -> r
      | (Ok None | Error _) as r ->
          Va.free space va size;
          r)

(* Host memory the process allocates and describes to the RM. *)
let alloc_host g size =
  with_addresses g.c.low ~align:page size @@ fun va ->
  let* () = Rm.map (-1) va size in
  match describe g va size with
  | (Error _ | Ok None) as r ->
      Rm.unmap va size;
      Result.map (fun _ -> None) r
  | Ok (Some h) -> (
      match uvm_map g va size h with
      | Ok true -> Ok (Some (h, va))
      | (Ok false | Error _) as r ->
          free_object g h;
          Rm.unmap va size;
          Result.map (fun _ -> None) r)

(* Video memory, mapped for the host too through BAR1 if [cpu]. The RM allocates
   it first; the addresses follow. *)
let alloc_video g ~cpu size =
  let page_size = if size >= huge_from then huge_page else page in
  let size = round_up size page_size in
  let p = video_params g ~contiguous:cpu ~page_size size in
  let* s, h = Rm.alloc g.c ~parent:g.device D.nv1_memory_user (Some p) in
  let* ok = fits g "allocating GPU memory" s in
  if not ok then Ok None
  else
    let space = if cpu then g.c.low else g.c.main in
    let placed =
      with_addresses space ~align:page_size size @@ fun va ->
      let* bar = if not cpu then Ok true else map_to_cpu g h size va in
      if not bar then Ok None
      else
        match uvm_map g va size h with
        | Ok true -> Ok (Some va)
        | (Ok false | Error _) as r ->
            if cpu then Rm.unmap va size;
            Result.map (fun _ -> None) r
    in
    match placed with
    | Ok (Some va) -> Ok (Some (h, va, size, space))
    | (Ok None | Error _) as r ->
        free_object g h;
        Result.map (fun _ -> None) r

let memory m =
  { Rig_nv.address = m.va; host = m.host; handle = m.handle; data = m }

(* A path function's failure that is no refusal is the driver's fault. *)
let fault = function Ok x -> x | Error e -> raise (Rig_nv.Fault e)

let alloc g kind n =
  fault
  @@
  match kind with
  | `System ->
      let size = round_up n page in
      let* x = alloc_host g size in
      Ok
        (Option.map
           (fun (h, va) ->
             memory
               {
                 va;
                 size;
                 handle = h;
                 host = Some va;
                 video = None;
                 of_ = Own g.c.low;
               })
           x)
  | (`Gpu | `Bar) as k ->
      let* v = alloc_video g ~cpu:(k = `Bar) n in
      Ok
        (Option.map
           (fun (h, va, size, space) ->
             memory
               {
                 va;
                 size;
                 handle = h;
                 host = (if k = `Bar then Some va else None);
                 video = Some g.index;
                 of_ = Own space;
               })
           v)

let ranges_lock = Mutex.create ()
let ranges : range list ref = ref []

let map_host g a n =
  let a0 = a land lnot (page - 1) in
  let a1 = round_up (a + n) page in
  let mem r =
    memory
      {
        va = r.gpu_addr + (a - r.addr);
        size = n;
        handle = r.descriptor;
        host = Some a;
        video = None;
        of_ = Host r;
      }
  in
  Mutex.protect ranges_lock @@ fun () ->
  let inside r = r.addr <= a0 && a1 <= r.addr + r.bytes in
  let overlaps r = a0 < r.addr + r.bytes && r.addr < a1 in
  match List.find_opt inside !ranges with
  | Some r -> (
      match List.assq_opt g r.users with
      | Some k ->
          r.users <- (g, k + 1) :: List.remove_assq g r.users;
          Some (mem r)
      | None -> (
          match fault (map_external g r.gpu_addr r.bytes r.descriptor) with
          | false -> None
          | true ->
              r.users <- (g, 1) :: r.users;
              Some (mem r)))
  | None when List.exists overlaps !ranges -> None
  | None -> (
      let size = a1 - a0 in
      match fault (describe g a0 size) with
      | None -> None
      | Some h -> (
          let placed =
            with_addresses g.c.low ~align:page size @@ fun va ->
            let* ok = uvm_map g va size h in
            Ok (if ok then Some va else None)
          in
          match placed with
          | Ok (Some gpu_addr) ->
              let r =
                {
                  addr = a0;
                  bytes = size;
                  gpu_addr;
                  descriptor = h;
                  parent = g;
                  users = [ (g, 1) ];
                }
              in
              ranges := r :: !ranges;
              Some (mem r)
          | Ok None ->
              free_object g h;
              None
          | Error e ->
              free_object g h;
              raise (Rig_nv.Fault e)))

(* Another GPU's video memory needs peer access, which may be refused; host
   memory needs none. *)
let map_peer g (m : mem Rig_nv.memory) =
  let peer = m.data in
  let refused = function Some u -> List.mem u g.refused | None -> false in
  if refused peer.video then None
  else
    match fault (map_external g peer.va peer.size peer.handle) with
    | false -> None
    | true -> Some (memory { peer with of_ = Peer })

let unmap_host g r =
  Mutex.protect ranges_lock @@ fun () ->
  match List.assq_opt g r.users with
  | Some k when k > 1 -> r.users <- (g, k - 1) :: List.remove_assq g r.users
  | Some _ | None ->
      r.users <- List.remove_assq g r.users;
      if r.users = [] then begin
        ranges := List.filter (fun r' -> r' != r) !ranges;
        fault (uvm_free g.c r.gpu_addr r.bytes);
        Va.free g.c.low r.gpu_addr r.bytes;
        free_object r.parent r.descriptor
      end
      else fault (uvm_unmap g r.gpu_addr r.bytes)

let free g (m : mem Rig_nv.memory) =
  let m = m.data in
  match m.of_ with
  | Host r -> unmap_host g r
  | Peer -> fault (uvm_unmap g m.va m.size)
  | Own space ->
      free_object g m.handle;
      fault (uvm_free g.c m.va m.size);
      (match m.host with Some a -> Rm.unmap a m.size | None -> ());
      Va.free space m.va m.size

(* Channels *)

(* Registers the channel [ch] with unified memory at a range of its own. *)
let register g ch =
  let* base =
    match Va.alloc g.c.low ~align:page channel_range with
    | Some va -> Ok va
    | None -> Error "no address for the GPU channel"
  in
  let module R = D.Uvm_register_channel in
  let r = params R.sizeof in
  set_uuid r R.gpu_uuid g.uuid;
  set r R.rm_ctrl_fd g.c.ctl;
  set r R.h_client g.c.root;
  set r R.h_channel ch;
  set r R.base base;
  set r R.length channel_range;
  match
    Rm.uvm_call g.c D.uvm_register_channel r R.rm_status
      "registering a GPU channel"
  with
  | Error _ as e ->
      Va.free g.c.low base channel_range;
      e
  | Ok () ->
      Mutex.protect gpus_lock (fun () -> g.channels <- (ch, base) :: g.channels);
      Ok ()

let unregister g ch =
  let (module R : D.RELEASE) = g.c.layouts in
  let module U = R.Uvm_unregister_channel in
  let u = params U.sizeof in
  Option.iter (fun f -> set_uuid u f g.uuid) U.gpu_uuid;
  set u U.h_client g.c.root;
  set u U.h_channel ch;
  let* () =
    Rm.uvm_call g.c D.uvm_unregister_channel u U.rm_status
      "unregistering a GPU channel"
  in
  let gone =
    Mutex.protect gpus_lock @@ fun () ->
    let gone, kept = List.partition (fun (c, _) -> c = ch) g.channels in
    g.channels <- kept;
    gone
  in
  List.iter (fun (_, base) -> Va.free g.c.low base channel_range) gone;
  Ok ()

(* Opening a GPU *)

(* The cards the kernel driver holds, as (bus address, GPU id, minor number).
   Its table follows the order in which it took them and leaves out those it
   does not hold, so GPUs are numbered by bus address instead. It reports no
   function number, which is 0 for NVIDIA's GPUs. *)
let cards c =
  let module C = D.Card_info in
  let n = D.nv_max_devices in
  let t = params (n * C.sizeof) in
  let* () = Rm.escape c.Rm.ctl D.nv_esc_card_info t "reading NVIDIA's GPUs" in
  Ok
    (List.filter_map
       (fun i ->
         let at (off, size) = (off + (i * C.sizeof), size) in
         if get t (at C.valid) = 0 then None
         else
           let bus =
             strf "%04x:%02x:%02x.0"
               (get t (at C.pci_info_domain))
               (get t (at C.pci_info_bus))
               (get t (at C.pci_info_slot))
           in
           Some (bus, get t (at C.gpu_id), get t (at C.minor_number)))
       (List.init n Fun.id))

let pick what classes available =
  match List.find_opt (fun c -> List.mem c available) classes with
  | Some c -> Ok c
  | None -> Error (strf "the GPU has no supported %s class" what)

(* The classes the device object offers. *)
let classes rm device =
  let module C = D.Classlist in
  let p = params C.sizeof in
  let* () =
    rm.Rig_nv.control device D.nv0080_ctrl_cmd_gpu_get_classlist (Some p)
  in
  let count = get p C.num_classes in
  let list = params (4 * count) in
  set p C.class_list (Rm.address list);
  let* () = rm.control device D.nv0080_ctrl_cmd_gpu_get_classlist (Some p) in
  ignore (Sys.opaque_identity list);
  Ok (List.init count (fun i -> get list (4 * i, 4)))

let gr_indices =
  [
    D.nv2080_ctrl_gr_info_index_litter_num_gpcs;
    D.nv2080_ctrl_gr_info_index_litter_num_tpc_per_gpc;
    D.nv2080_ctrl_gr_info_index_litter_num_sm_per_tpc;
    D.nv2080_ctrl_gr_info_index_max_warps_per_sm;
    D.nv2080_ctrl_gr_info_index_sm_version;
  ]

(* The graphics engine's facts at [gr_indices], in their order. *)
let gr_info rm subdevice =
  let module G = D.Gr_get_info in
  let module I = D.Gr_info in
  let n = List.length gr_indices in
  let infos = params (n * I.sizeof) in
  List.iteri
    (fun i idx -> set infos (fst I.index + (i * I.sizeof), snd I.index) idx)
    gr_indices;
  let p = params G.sizeof in
  set p G.gr_info_list_size n;
  set p G.gr_info_list (Rm.address infos);
  let* () =
    rm.Rig_nv.control subdevice D.nv2080_ctrl_cmd_gr_get_info (Some p)
  in
  let r =
    List.init n (fun i -> get infos (fst I.data + (i * I.sizeof), snd I.data))
  in
  ignore (Sys.opaque_identity infos);
  Ok r

(* The frame buffer's fact [index], in bytes. *)
let fb_info c rm subdevice index =
  let (module R : D.RELEASE) = c.Rm.layouts in
  let module G = R.Fb_get_info in
  let module F = D.Fb_info in
  let p = params G.sizeof in
  set p G.fb_info_list_size 1;
  (* The first entry's field [f]. *)
  let elt (f, n) =
    let at, _, _ = G.fb_info_list in
    (at + f, n)
  in
  set p (elt F.index) index;
  let* () =
    rm.Rig_nv.control subdevice D.nv2080_ctrl_cmd_fb_get_info_v2 (Some p)
  in
  (* The RM reports the frame buffer's facts in KiB. *)
  Ok (get p (elt F.data) * 1024)

(* The GPUs made, by bus address, under [gpus_lock]. *)
let gpus : (string * gpu) list ref = ref []

(* GPU [bus]'s objects, made at its first open and kept for the process; a
   failed open gives back what it took, last first. *)
let make_gpu c ~index bus =
  let undo = ref [] in
  let taken f = undo := f :: !undo in
  let rm = Rm.rm c in
  let made =
    let* cards = cards c in
    let* gpu_id, minor =
      match List.find_opt (fun (b, _, _) -> b = bus) cards with
      | Some (_, id, minor) -> Ok (id, minor)
      | None -> Error (bus ^ " is not held by NVIDIA's kernel driver")
    in
    let* file = Rm.open_file (strf "/dev/nvidia%d" minor) in
    taken (fun () -> Rm.close file);
    let* () = Rm.register file ~ctl:c.ctl in
    let id = params D.Id_info.sizeof in
    set id D.Id_info.gpu_id gpu_id;
    let* () =
      rm.control c.root D.nv0000_ctrl_cmd_gpu_get_id_info_v2 (Some id)
    in
    let instance = get id D.Id_info.device_instance in
    let alloc ~parent cls f size =
      let p = params size in
      f p;
      rm.alloc ~parent cls (Some p)
    in
    let* device =
      alloc ~parent:c.root D.nv01_device_0
        (fun p ->
          set p D.Nv0080_alloc.device_id instance;
          set p D.Nv0080_alloc.h_client_share c.root;
          set p D.Nv0080_alloc.va_mode
            D.nv_device_allocation_vamode_optional_multiple_vaspaces)
        D.Nv0080_alloc.sizeof
    in
    (* Freeing the device frees every object under it. *)
    taken (fun () -> ignore (rm.free ~parent:c.root device));
    let* subdevice =
      alloc ~parent:device D.nv20_subdevice_0 ignore D.Nv2080_alloc.sizeof
    in
    let* virtmem =
      alloc ~parent:device D.nv01_memory_virtual
        (fun p -> set p D.Memory_virtual_alloc.limit va_limit)
        D.Memory_virtual_alloc.sizeof
    in
    let (module R : D.RELEASE) = c.layouts in
    let* vaspace =
      alloc ~parent:device D.fermi_vaspace_a
        (fun p ->
          set p R.Vaspace_alloc.va_base va_base;
          set p R.Vaspace_alloc.va_size va_size;
          set p R.Vaspace_alloc.flags
            (D.nv_vaspace_allocation_flags_enable_page_faulting
           lor D.nv_vaspace_allocation_flags_is_externally_owned))
        R.Vaspace_alloc.sizeof
    in
    let* available = classes rm device in
    let* usermode_class =
      pick "usermode" [ D.hopper_usermode_a; D.turing_usermode_a ] available
    in
    let* channel_class =
      pick "channel"
        [ D.blackwell_channel_gpfifo_a; D.ampere_channel_gpfifo_a ]
        available
    in
    let* compute_class =
      pick "compute"
        [ D.blackwell_compute_b; D.ada_compute_a; D.ampere_compute_b ]
        available
    in
    let* copy_class =
      pick "copy" [ D.blackwell_dma_copy_b; D.ampere_dma_copy_b ] available
    in
    let* gr = gr_info rm subdevice in
    let* gpcs, tpcs, sms, warps, sm =
      match gr with
      | [ a; b; d; e; f ] -> Ok (a, b, d, e, f)
      | _ -> Error "reading the graphics engine's facts"
    in
    let* budget =
      fb_info c rm subdevice D.nv2080_ctrl_fb_info_index_heap_size
    in
    let module Gi = D.Gid_info in
    let gid = params Gi.sizeof in
    set gid Gi.flags
      (bits D.nv2080_gpu_cmd_gpu_get_gid_flags_format
         D.nv2080_gpu_cmd_gpu_get_gid_flags_format_binary);
    set gid Gi.length 16;
    let* () =
      rm.control subdevice D.nv2080_ctrl_cmd_gpu_get_gid_info (Some gid)
    in
    let data, _, _ = Gi.data in
    let uuid = String.init 16 (fun i -> Bigarray.Array1.get gid (data + i)) in
    let* usermode = rm.alloc ~parent:subdevice usermode_class None in
    let* doorbell =
      match Va.alloc c.low ~align:page usermode_bytes with
      | Some va -> Ok va
      | None -> Error "no address for the GPU's doorbell"
    in
    taken (fun () -> Va.free c.low doorbell usermode_bytes);
    let g =
      {
        c;
        minor;
        device;
        subdevice;
        virtmem;
        vaspace;
        uuid;
        doorbell;
        facts =
          {
            Rig_nv.channel_class;
            compute_class;
            copy_class;
            sm_version = sm;
            gpcs;
            tpcs_per_gpc = tpcs;
            sms_per_tpc = sms;
            warps_per_sm = warps;
          };
        budget;
        index;
        refused = [];
        channels = [];
        opened = false;
      }
    in
    let* mapped = map_to_cpu g usermode usermode_bytes doorbell in
    let* () =
      if mapped then Ok () else Error "no room to map the GPU's doorbell"
    in
    taken (fun () -> Rm.unmap doorbell usermode_bytes);
    let module U = D.Uvm_register_gpu in
    let u = params U.sizeof in
    set_uuid u U.gpu_uuid uuid;
    set u U.rm_ctrl_fd (-1);
    let* () =
      Rm.uvm_call c D.uvm_register_gpu u U.rm_status "registering the GPU"
    in
    let unregister_gpu () =
      let module U = D.Uvm_unregister_gpu in
      let u = params U.sizeof in
      set_uuid u U.gpu_uuid uuid;
      ignore
        (Rm.uvm_call c D.uvm_unregister_gpu u U.rm_status
           "unregistering the GPU"
          : (unit, string) result)
    in
    taken unregister_gpu;
    let module V = D.Uvm_register_gpu_vaspace in
    let v = params V.sizeof in
    set_uuid v V.gpu_uuid uuid;
    set v V.rm_ctrl_fd c.ctl;
    set v V.h_client c.root;
    set v V.h_va_space vaspace;
    let* () =
      Rm.uvm_call c D.uvm_register_gpu_vaspace v V.rm_status
        "registering the GPU's address space"
    in
    Ok g
  in
  match made with
  | Ok g -> Ok g
  | Error _ as e ->
      List.iter (fun f -> f ()) !undo;
      e

(* Enables peer access between [g] and every GPU opened before it, recording
   those that refuse it, both ways. *)
let enable_peers g others =
  List.iter
    (fun g' ->
      let module E = D.Uvm_enable_peer_access in
      let e = params E.sizeof in
      set_uuid e E.gpu_uuid_a g.uuid;
      set_uuid e E.gpu_uuid_b g'.uuid;
      match
        Rm.uvm_call g.c D.uvm_enable_peer_access e E.rm_status
          "enabling peer access"
      with
      | Ok () -> ()
      | Error _ ->
          g.refused <- g'.index :: g.refused;
          g'.refused <- g.index :: g'.refused)
    others

let gpu c ~index bus =
  Mutex.protect gpus_lock @@ fun () ->
  match List.assoc_opt bus !gpus with
  | Some g -> Ok g
  | None ->
      let* g = make_gpu c ~index bus in
      enable_peers g (List.map snd !gpus);
      gpus := (bus, g) :: !gpus;
      Ok g

let path g =
  {
    Rig_nv.key;
    rm = Rm.rm g.c;
    device = g.device;
    subdevice = g.subdevice;
    vaspace = g.vaspace;
    gpu = g.facts;
    budget = g.budget;
    doorbell = g.doorbell + doorbell_at;
    alloc = alloc g;
    map_host = Some (map_host g);
    index = g.index;
    reaches = (fun i -> not (List.mem i g.refused));
    map_peer = map_peer g;
    free = free g;
    register = register g;
    unregister = unregister g;
    (* The RM reports the GPU's faults through the channels, and the kernel
       driver bounds the GPU's work; channels the device could not free may
       still run. *)
    check = ignore;
    hang_ms = None;
    stop =
      (fun () ->
        Mutex.protect gpus_lock (fun () -> g.opened <- false);
        `Unknown);
  }

(* Opening *)

let sysfs = "/"
let gpus_at = Sysfs.gpus
let count () = List.length (Sysfs.gpus sysfs)

let device_name i =
  if i < 0 then invalid_argf "Rig_nv_nvidia.device_name: GPU %d < 0" i;
  if i = 0 then "NV" else strf "NV:%d" i

let open_ i =
  if i < 0 then invalid_argf "Rig_nv_nvidia.open_: GPU %d < 0" i;
  let buses = Sysfs.gpus sysfs in
  match List.nth_opt buses i with
  | None ->
      Error
        (strf "no GPU %d: the machine has %d NVIDIA GPUs" i (List.length buses))
  | Some bus -> (
      let* c = Rm.client () in
      let* g = gpu c ~index:i bus in
      let claimed =
        Mutex.protect gpus_lock @@ fun () ->
        let free = not g.opened in
        g.opened <- true;
        free
      in
      if not claimed then Error (strf "%s has a device open" bus)
      else
        match Rig_nv.make (path g) with
        | Ok d -> Ok d
        | Error _ as e ->
            Mutex.protect gpus_lock (fun () -> g.opened <- false);
            e)
