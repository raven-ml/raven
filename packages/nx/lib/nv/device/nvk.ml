(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* NVIDIA GPUs through the resource manager interface of NVIDIA's kernel driver
   ([/dev/nvidiactl]) and its unified memory driver ([/dev/nvidia-uvm]), which
   maps memory into the GPUs' address spaces. *)

module D = Nv_defs
module P = Params
module Space = Nx_device_support.Page_table.Space
module Sysmem = Nx_device_support.Sysmem
module Pci = Nx_device_support.Pci

external open_file : string -> int = "caml_nx_nv_open"
external close_file : int -> unit = "caml_nx_nv_close"
external ioctl_raw : int -> int -> P.t -> string -> unit = "caml_nx_nv_ioctl"
external map_at : int -> nativeint -> int -> unit = "caml_nx_nv_map"
external release_at : nativeint -> int -> unit = "caml_nx_nv_release"

let ctl_path = "/dev/nvidiactl"

(* The escape ioctls: read and write, of the parameters' size, in the driver's
   magic. *)
let escape fd nr p what =
  let request =
    (3 lsl 30)
    lor ((P.length p land 0x1fff) lsl 16)
    lor (D.nv_ioctl_magic lsl 8) lor nr
  in
  ioctl_raw fd request p what

(* The cards, as (bus address, GPU id, minor number), in the driver's table. The
   table follows the order in which the driver took its GPUs, which a GPU given
   back and bound again leaves last, and leaves out the GPUs it does not hold:
   GPUs are numbered by bus address instead. It reports no function number,
   which is 0 for NVIDIA's GPUs. *)
let ctl = lazy (open_file ctl_path)

let cards () =
  let module C = D.Card_info in
  let n = 64 in
  let t = P.create (n * C.sizeof) in
  escape (Lazy.force ctl) D.nv_esc_card_info t "reading NVIDIA's GPUs";
  List.filter_map
    (fun i ->
      let at (off, size) = (off + (i * C.sizeof), size) in
      if P.get t (at C.valid) = 0 then None
      else
        let bus =
          Pci.address
            ~domain:(P.get t (at C.pci_info_domain))
            ~bus:(P.get t (at C.pci_info_bus))
            ~device:(P.get t (at C.pci_info_slot))
            ~fn:0
        in
        Some (bus, P.get t (at C.gpu_id), P.get t (at C.minor_number)))
    (List.init n Fun.id)

(* The process's client *)

(* The GPU addresses the process allocates, reserved in the process so that
   nothing else maps there: the low range holds what the CPU also maps and the
   channels' ranges, the main range the rest. Both lie below 2^40, the widest
   address of a channel's command segment. *)
let low_base = 0x10_0000_0000
let main_base = 0x20_0000_0000
let top = 1 lsl 40

type client = {
  ctl : int;
  uvm : int;
  mm : int; (* the unified memory driver's memory manager, open for good *)
  root : int;
  release : (module D.RELEASE);
  low : Space.t;
  main : Space.t;
  mutable next_handle : int; (* the handles the process chooses *)
  lock : Mutex.t;
}

(* [f ()], after which [undo ()] runs if it raised. *)
let or_undo undo f =
  match f () with
  | v -> v
  | exception e ->
      undo ();
      raise e

let uvm_status c cmd p field what =
  ioctl_raw c.uvm cmd p what;
  P.get p field

let uvm_call c cmd p field what =
  Rm.check c.release what (uvm_status c cmd p field what)

(* Whether the status [s] of [what] is [NV_OK]: [false] for [NV_ERR_NO_MEMORY],
   and [Failure] for any other. *)
let fits release what s =
  if s = D.nv_err_no_memory then false
  else begin
    Rm.check release what s;
    true
  end

(* An allocation under [root] with the escape [NV_ESC_RM_ALLOC]. *)
let rm_alloc_raw ctl ~root ~parent cls params =
  let module A = D.Nvos21 in
  let a = P.create A.sizeof in
  P.set a A.h_root root;
  P.set a A.h_object_parent parent;
  P.set a A.h_class cls;
  Option.iter
    (fun p -> P.set a A.p_alloc_parms (Nativeint.to_int (P.address p)))
    params;
  escape ctl D.nv_esc_rm_alloc a "allocating a GPU object";
  ignore (Sys.opaque_identity params);
  (P.get a A.status, P.get a A.h_object_new)

let branch version =
  match String.index_opt version '.' with
  | Some i -> int_of_string_opt (String.sub version 0 i)
  | None -> int_of_string_opt version

let supported () = String.concat ", " (List.map string_of_int D.releases)

let open_client () =
  let ctl = Lazy.force ctl in
  let s, root = rm_alloc_raw ctl ~root:0 ~parent:0 D.nv01_root_client None in
  if s <> D.nv_ok then
    failwith
      (Printf.sprintf "creating a client of NVIDIA's driver: status 0x%x" s);
  let module B = D.Build_version in
  let v = P.create B.sizeof in
  let c = D.Nvos54.(P.create sizeof) in
  P.set c D.Nvos54.h_client root;
  P.set c D.Nvos54.h_object root;
  P.set c D.Nvos54.cmd D.nv0000_ctrl_cmd_system_get_build_version_v2;
  P.set c D.Nvos54.params_size B.sizeof;
  P.set c D.Nvos54.params (Nativeint.to_int (P.address v));
  escape ctl D.nv_esc_rm_control c "reading the driver's version";
  ignore (Sys.opaque_identity v);
  let version = P.get_string v B.driver_version_buffer in
  let release =
    match branch version with
    | Some b when List.mem b D.releases -> D.release b
    | _ ->
        failwith
          (Printf.sprintf
             "NVIDIA's kernel driver %s is not supported; the supported \
              releases are %s"
             version (supported ()))
  in
  let uvm = open_file "/dev/nvidia-uvm" in
  let mm = open_file "/dev/nvidia-uvm" in
  Sysmem.reserve ~base:low_base (main_base - low_base);
  Sysmem.reserve ~base:main_base (top - main_base);
  let c =
    {
      ctl;
      uvm;
      mm;
      root;
      release;
      low = Space.create ~base:low_base (main_base - low_base);
      main = Space.create ~base:main_base (top - main_base);
      next_handle = 0x1000;
      lock = Mutex.create ();
    }
  in
  let module I = D.Uvm_initialize in
  uvm_call c D.uvm_initialize (P.create I.sizeof) I.rm_status
    "initializing NVIDIA's unified memory";
  (* The memory manager's registration is made once per process; a second one,
     such as CUDA's in the same process, is refused, and the first serves. *)
  let module M = D.Uvm_mm_initialize in
  let m = P.create M.sizeof in
  P.set m M.uvm_fd uvm;
  (try ioctl_raw mm D.uvm_mm_initialize m "" with Failure _ -> ());
  c

let client = lazy (open_client ())

let rm c =
  let alloc ~parent cls params =
    let s, h = rm_alloc_raw c.ctl ~root:c.root ~parent cls params in
    Rm.check c.release (Printf.sprintf "allocating class 0x%x" cls) s;
    h
  in
  let control obj cmd params =
    let module C = D.Nvos54 in
    let a = P.create C.sizeof in
    P.set a C.h_client c.root;
    P.set a C.h_object obj;
    P.set a C.cmd cmd;
    Option.iter
      (fun p ->
        P.set a C.params_size (P.length p);
        P.set a C.params (Nativeint.to_int (P.address p)))
      params;
    escape c.ctl D.nv_esc_rm_control a "controlling a GPU object";
    ignore (Sys.opaque_identity params);
    Rm.check c.release (Printf.sprintf "command 0x%x" cmd) (P.get a C.status)
  in
  let free ~parent obj =
    let module F = D.Nvos00 in
    let a = P.create F.sizeof in
    P.set a F.h_root c.root;
    P.set a F.h_object_parent parent;
    P.set a F.h_object_old obj;
    escape c.ctl D.nv_esc_rm_free a "freeing a GPU object";
    Rm.check c.release "freeing a GPU object" (P.get a F.status)
  in
  { Rm.root = c.root; alloc; control; free }

(* GPUs *)

type gpu = {
  c : client;
  minor : int;
  file : int;
      (* the GPU's file, which the process holds open while it uses the GPU: the
         driver refuses the GPU to a process without it *)
  instance : int; (* the device instance, as RM numbers devices *)
  mutable device : int; (* the RM device and its virtual memory object *)
  mutable virtmem : int;
  mutable uuid : string;
}

let gpu_file c minor =
  let fd = open_file (Printf.sprintf "/dev/nvidia%d" minor) in
  let module R = D.Register_fd in
  let r = P.create R.sizeof in
  P.set r R.ctl_fd c.ctl;
  or_undo
    (fun () -> close_file fd)
    (fun () -> escape fd D.nv_esc_register_fd r "registering a GPU file");
  fd

let open_gpu bus =
  let c = Lazy.force client in
  let gpu_id, minor =
    match List.find_opt (fun (b, _, _) -> b = bus) (cards ()) with
    | Some (_, id, minor) -> (id, minor)
    | None -> failwith (bus ^ " is not held by NVIDIA's kernel driver")
  in
  let file = gpu_file c minor in
  let module I = D.Id_info in
  let p = P.create I.sizeof in
  P.set p I.gpu_id gpu_id;
  or_undo
    (fun () -> close_file file)
    (fun () ->
      (rm c).control c.root D.nv0000_ctrl_cmd_gpu_get_id_info_v2 (Some p));
  {
    c;
    minor;
    file;
    instance = P.get p I.device_instance;
    device = 0;
    virtmem = 0;
    uuid = "";
  }

(* Maps the memory object [handle] of [size] bytes into the process at [va]:
   through the control device for system memory, and the GPU's file otherwise,
   whose window onto the GPU's memory (BAR1) may have no room left, in which
   case it maps nothing and is [false]. *)
let map_to_cpu g handle size va ~flags ~system =
  let c = g.c in
  let fd = if system then open_file ctl_path else gpu_file c g.minor in
  Fun.protect
    ~finally:(fun () -> close_file fd)
    (fun () ->
      let module W = D.Nvos33_with_fd in
      let module M = D.Nvos33 in
      let w = P.create W.sizeof in
      let at f = (fst W.params + fst f, snd f) in
      P.set w W.fd fd;
      P.set w (at M.h_client) c.root;
      P.set w (at M.h_device) g.device;
      P.set w (at M.h_memory) handle;
      P.set w (at M.length) size;
      P.set w (at M.flags) flags;
      escape c.ctl D.nv_esc_rm_map_memory w
        "mapping GPU memory into the process";
      let status = P.get w (at M.status) in
      if status = D.nv_err_no_memory then false
      else begin
        Rm.check c.release "mapping GPU memory into the process" status;
        map_at fd (Nativeint.of_int va) size;
        true
      end)

let set_uuid p field uuid = P.blit_string uuid p (fst field)

(* Frees the unified memory range at [va], unmapping it from every GPU. *)
let uvm_free c va size =
  let module F = (val c.release : D.RELEASE) in
  let module U = F.Uvm_free in
  let u = P.create U.sizeof in
  P.set u U.base va;
  Option.iter (fun l -> P.set u l size) U.length;
  uvm_call c D.uvm_free u U.rm_status "unmapping GPU memory"

(* Maps the memory object [handle] of [size] bytes, at [va] in the process's
   unified memory, into [g]'s address space: [false] if [g] has no memory for
   the mapping's page tables. *)
let uvm_map_external g va size handle =
  let c = g.c in
  let module U = D.Uvm_map_external_allocation in
  let module G = D.Uvm_gpu_mapping in
  let u = P.create U.sizeof in
  P.set u U.base va;
  P.set u U.length size;
  P.set u U.rm_ctrl_fd c.ctl;
  P.set u U.h_client c.root;
  P.set u U.h_memory handle;
  P.set u U.gpu_attributes_count 1;
  set_uuid u (P.elt_field U.per_gpu_attributes 0 G.gpu_uuid) g.uuid;
  P.set u
    (P.elt_field U.per_gpu_attributes 0 G.gpu_mapping_type)
    D.uvm_gpu_mapping_type_read_write_atomic;
  uvm_status c D.uvm_map_external_allocation u U.rm_status "mapping GPU memory"
  |> fits c.release "mapping GPU memory"

(* Maps the memory object [handle] of [size] bytes at [va] in [g]'s virtual
   memory: [false] if [g] has no memory for the mapping's page tables. *)
let map_dma g va size handle =
  let c = g.c in
  let (module R : D.RELEASE) = c.release in
  let module M = R.Nvos46 in
  let m = P.create M.sizeof in
  P.set m M.h_client c.root;
  P.set m M.h_device g.device;
  P.set m M.h_dma g.virtmem;
  P.set m M.h_memory handle;
  P.set m M.length size;
  P.set m M.flags
    (P.bits D.nvos46_flags_page_size D.nvos46_flags_page_size_4kb
    lor P.bits D.nvos46_flags_cache_snoop D.nvos46_flags_cache_snoop_enable
    lor P.bits D.nvos46_flags_dma_offset_fixed
          D.nvos46_flags_dma_offset_fixed_true);
  P.set m M.dma_offset va;
  escape c.ctl D.nv_esc_rm_map_memory_dma m "mapping GPU memory";
  if not (fits c.release "mapping GPU memory" (P.get m M.status)) then false
  else if P.get m M.dma_offset <> va then
    failwith "mapping GPU memory: the driver chose another address"
  else true

(* Maps [size] bytes of the memory object [handle] at [va] for [g]: first
   creating the unified memory range there and the object's mapping in [g]'s
   virtual memory, if [create]. [false] if [g] has no memory for the mapping. A
   range it created is freed if the mapping fails or does not fit, so that the
   addresses stay free. *)
let uvm_map g ~create va size handle =
  let c = g.c in
  if not create then uvm_map_external g va size handle
  else begin
    let module E = D.Uvm_create_external_range in
    let e = P.create E.sizeof in
    P.set e E.base va;
    P.set e E.length size;
    uvm_call c D.uvm_create_external_range e E.rm_status
      "reserving GPU addresses";
    let mapped =
      or_undo (fun () -> uvm_free c va size) @@ fun () ->
      map_dma g va size handle && uvm_map_external g va size handle
    in
    if not mapped then uvm_free c va size;
    mapped
  end

(* Unmaps for [g] the [size] bytes at [va] of a range that stays. *)
let uvm_unmap g va size =
  let module U = D.Uvm_unmap_external in
  let u = P.create U.sizeof in
  P.set u U.base va;
  P.set u U.length size;
  set_uuid u U.gpu_uuid g.uuid;
  uvm_call g.c D.uvm_unmap_external u U.rm_status "unmapping GPU memory"

(* Memory *)

type mem = {
  va : int;
  size : int;
  handle : int; (* the memory object *)
  cpu : bool; (* whether the process maps it at [va] *)
  space : Space.t;
}

let round_up n a = (n + a - 1) / a * a
let page = 0x1000

(* A new handle for an object the process names itself. *)
let handle c =
  Mutex.protect c.lock (fun () ->
      c.next_handle <- c.next_handle + 1;
      c.next_handle)

(* Describes the process memory at [va] to RM as system memory of [g]. *)
let describe g va size =
  let c = g.c in
  let h = handle c in
  let module W = D.Nvos02_with_fd in
  let module O = D.Nvos02 in
  let w = P.create W.sizeof in
  let at f = (fst W.params + fst f, snd f) in
  P.set w W.fd (-1);
  P.set w (at O.h_root) c.root;
  P.set w (at O.h_object_parent) g.device;
  P.set w (at O.h_object_new) h;
  P.set w (at O.h_class) D.nv01_memory_system_os_descriptor;
  P.set w (at O.p_memory) va;
  P.set w (at O.limit) (size - 1);
  P.set w (at O.flags)
    (P.bits D.nvos02_flags_physicality D.nvos02_flags_physicality_noncontiguous
    lor P.bits D.nvos02_flags_coherency D.nvos02_flags_coherency_cached
    lor P.bits D.nvos02_flags_mapping D.nvos02_flags_mapping_no_map);
  let fd = gpu_file c g.minor in
  Fun.protect
    ~finally:(fun () -> close_file fd)
    (fun () ->
      escape fd D.nv_esc_rm_alloc_memory w "describing host memory to the GPU");
  Rm.check c.release "describing host memory to the GPU" (P.get w (at O.status));
  h

(* The attributes of video memory, or of uncached system memory, in pages of
   [page_size]. *)
let memory_params c ~uncached ~contiguous ~page_size size =
  let huge = page_size > page in
  let module A = D.Memory_alloc in
  let p = P.create A.sizeof in
  P.set p A.owner c.root;
  P.set p A.alignment page_size;
  P.set p A.limit (size - 1);
  P.set p A.format 6;
  P.set p A.size size;
  P.set p A.type_
    (if uncached then D.nvos32_type_notifier else D.nvos32_type_image);
  P.set p A.attr
    (P.bits D.nvos32_attr_physicality
       (if contiguous then D.nvos32_attr_physicality_contiguous
        else D.nvos32_attr_physicality_allow_noncontiguous)
    lor (if huge then
           P.bits D.nvos32_attr_page_size D.nvos32_attr_page_size_huge
         else 0)
    lor P.bits D.nvos32_attr_location
          (if uncached then D.nvos32_attr_location_pci
           else D.nvos32_attr_location_vidmem));
  P.set p A.attr2
    (P.bits D.nvos32_attr2_gpu_cacheable
       (if uncached then D.nvos32_attr2_gpu_cacheable_no
        else D.nvos32_attr2_gpu_cacheable_yes)
    lor (if huge then
           P.bits D.nvos32_attr2_page_size_huge
             D.nvos32_attr2_page_size_huge_2mb
         else 0)
    lor P.bits D.nvos32_attr2_zbc D.nvos32_attr2_zbc_prefer_no_zbc);
  P.set p A.flags
    (D.nvos32_alloc_flags_map_not_required
   lor D.nvos32_alloc_flags_memory_handle_provided
   lor D.nvos32_alloc_flags_alignment_force
   lor D.nvos32_alloc_flags_ignore_bank_placement
    lor if uncached then 0 else D.nvos32_alloc_flags_persistent_vidmem);
  p

(* [n] bytes the GPU addresses at a new address: host memory the process
   allocates and describes to RM if [host], and otherwise video memory, or
   system memory if [uncached], which the process maps too if [cpu_access].
   [None] if RM has no memory. *)
let alloc g ?(host = false) ?(uncached = false) ?(cpu_access = false)
    ?(contiguous = false) ?(map_flags = 0) n =
  let c = g.c in
  let page_size =
    if uncached || host then page else if n >= 8 lsl 20 then 2 lsl 20 else page
  in
  let size = round_up n page_size in
  let space = if cpu_access then c.low else c.main in
  match Space.alloc ~align:page_size space size with
  | None -> None
  | Some va -> (
      let at = Nativeint.of_int va in
      or_undo (fun () -> Space.free space va) @@ fun () ->
      let cpu = host || cpu_access in
      let h =
        if host then begin
          map_at (-1) at size;
          Some
            (or_undo
               (fun () -> release_at at size)
               (fun () -> describe g va size))
        end
        else
          let cls =
            if uncached then D.nv1_memory_system else D.nv1_memory_user
          in
          let p = memory_params c ~uncached ~contiguous ~page_size size in
          let s, h =
            rm_alloc_raw c.ctl ~root:c.root ~parent:g.device cls (Some p)
          in
          if s = D.nv_err_no_memory then None
          else begin
            Rm.check c.release (Printf.sprintf "allocating class 0x%x" cls) s;
            let mapped =
              (not cpu_access)
              || or_undo
                   (fun () -> (rm c).free ~parent:g.device h)
                   (fun () ->
                     map_to_cpu g h size va ~flags:map_flags ~system:uncached)
            in
            if mapped then Some h
            else begin
              (rm c).free ~parent:g.device h;
              None
            end
          end
      in
      match h with
      | None ->
          Space.free space va;
          None
      | Some h ->
          let drop () =
            (rm c).free ~parent:g.device h;
            if cpu then release_at at size
          in
          if or_undo drop (fun () -> uvm_map g ~create:true va size h) then
            Some { va; size; handle = h; cpu; space }
          else begin
            drop ();
            Space.free space va;
            None
          end)

let free g m =
  let c = g.c in
  (rm c).free ~parent:g.device m.handle;
  uvm_free c m.va m.size;
  if m.cpu then release_at (Nativeint.of_int m.va) m.size;
  Space.free m.space m.va

let no_memory = "the GPU has no memory left to map it"

(* Another GPU's memory, mapped for [g] at its address. *)
let map_peer g m =
  match uvm_map g ~create:false m.va m.size m.handle with
  | exception Failure why -> Error why
  | true -> Ok ()
  | false -> Error no_memory

(* Unmaps from [g] another GPU's memory [map_peer] mapped. *)
let unmap_peer g m = uvm_unmap g m.va m.size

(* Host memory mapped for borrows: each range is described to RM once, by the
   first GPU to map it, and mapped for each GPU that borrows it; the last
   unmapping frees the range and the description. A range covers whole pages,
   which unified memory requires of its bounds. *)
type range = {
  addr : int;
  bytes : int;
  descriptor : int;
  parent : int; (* the RM device the descriptor belongs to *)
  mutable users : gpu list;
}

let ranges : (int, range) Hashtbl.t = Hashtbl.create 16
let ranges_lock = Mutex.create ()

let map_host g a n =
  let a = Nativeint.to_int a and n = round_up n page in
  Mutex.protect ranges_lock (fun () ->
      match Hashtbl.find_opt ranges a with
      | Some r when r.bytes = n -> (
          match uvm_map g ~create:false a n r.descriptor with
          | exception Failure why -> Error why
          | false -> Error no_memory
          | true ->
              r.users <- g :: r.users;
              Ok ())
      | Some _ -> Error "another borrow maps a different range at this address"
      | None -> (
          let overlaps =
            Hashtbl.fold
              (fun _ r o -> o || (a < r.addr + r.bytes && r.addr < a + n))
              ranges false
          in
          if overlaps then
            Error "it overlaps host memory mapped for another borrow"
          else
            match
              let h = describe g a n in
              let drop () = (rm g.c).free ~parent:g.device h in
              let mapped =
                or_undo drop (fun () -> uvm_map g ~create:true a n h)
              in
              if not mapped then drop ();
              (h, mapped)
            with
            | exception Failure why -> Error why
            | _, false -> Error no_memory
            | h, true ->
                Hashtbl.replace ranges a
                  {
                    addr = a;
                    bytes = n;
                    descriptor = h;
                    parent = g.device;
                    users = [ g ];
                  };
                Ok ()))

let unmap_host g a =
  let a = Nativeint.to_int a in
  Mutex.protect ranges_lock (fun () ->
      match Hashtbl.find_opt ranges a with
      | None -> ()
      | Some r ->
          r.users <- List.filter (fun u -> u != g) r.users;
          if r.users = [] then begin
            Hashtbl.remove ranges a;
            uvm_free g.c a r.bytes;
            (rm g.c).free ~parent:r.parent r.descriptor
          end
          else uvm_unmap g a r.bytes)

(* Setup *)

(* The usermode object, whose page holds the doorbell, mapped into the
   process. *)
let usermode g ~subdevice cls =
  let h = (rm g.c).alloc ~parent:subdevice cls None in
  let va =
    match Space.alloc g.c.low 0x10000 with
    | Some va -> va
    | None -> failwith "no address for the GPU's doorbell"
  in
  or_undo
    (fun () -> Space.free g.c.low va)
    (fun () ->
      if not (map_to_cpu g h 0x10000 va ~flags:0 ~system:false) then
        failwith "no room to map the GPU's doorbell");
  Nativeint.of_int va

(* Unmaps the usermode page [usermode] mapped at [va]; freeing the RM device
   frees its object. *)
let release_usermode g va =
  release_at va 0x10000;
  Space.free g.c.low (Nativeint.to_int va)

let close_gpu g = close_file g.file

let unregister_gpu g =
  let module U = D.Uvm_unregister_gpu in
  let u = P.create U.sizeof in
  set_uuid u U.gpu_uuid g.uuid;
  uvm_call g.c D.uvm_unregister_gpu u U.rm_status "unregistering the GPU"

(* Unregisters [g] and its address space from unified memory, which detaches its
   channels and disables its peer access. *)
let unregister g =
  let module U = D.Uvm_unregister_gpu_vaspace in
  let u = P.create U.sizeof in
  set_uuid u U.gpu_uuid g.uuid;
  uvm_call g.c D.uvm_unregister_gpu_vaspace u U.rm_status
    "unregistering the GPU's address space";
  unregister_gpu g

(* Registers [g] and its address space [vaspace] with unified memory, and
   enables peer access with [peers], returning those it was refused. *)
let register g ~subdevice ~vaspace ~peers =
  let c = g.c in
  let module Gi = D.Gid_info in
  let p = P.create Gi.sizeof in
  P.set p Gi.flags
    (P.bits D.nv2080_gpu_cmd_gpu_get_gid_flags_format
       D.nv2080_gpu_cmd_gpu_get_gid_flags_format_binary);
  P.set p Gi.length 16;
  (rm c).control subdevice D.nv2080_ctrl_cmd_gpu_get_gid_info (Some p);
  let data, _, _ = Gi.data in
  g.uuid <- P.sub_string p data 16;
  let module R = D.Uvm_register_gpu in
  let r = P.create R.sizeof in
  set_uuid r R.gpu_uuid g.uuid;
  P.set r R.rm_ctrl_fd (-1);
  uvm_call c D.uvm_register_gpu r R.rm_status "registering the GPU";
  let module V = D.Uvm_register_gpu_vaspace in
  let v = P.create V.sizeof in
  set_uuid v V.gpu_uuid g.uuid;
  P.set v V.rm_ctrl_fd c.ctl;
  P.set v V.h_client c.root;
  P.set v V.h_va_space vaspace;
  or_undo
    (fun () -> unregister_gpu g)
    (fun () ->
      uvm_call c D.uvm_register_gpu_vaspace v V.rm_status
        "registering the GPU's address space");
  List.filter
    (fun peer ->
      let module E = D.Uvm_enable_peer_access in
      let e = P.create E.sizeof in
      set_uuid e E.gpu_uuid_a g.uuid;
      set_uuid e E.gpu_uuid_b peer.uuid;
      match
        uvm_call c D.uvm_enable_peer_access e E.rm_status "enabling peer access"
      with
      | () -> false
      | exception Failure _ -> true)
    peers

(* Registers the channel [channel] with unified memory at a new range of
   addresses, and is that range's first address. *)
let register_channel g channel =
  let c = g.c in
  let size = 0x400_0000 in
  let base =
    match Space.alloc c.low size with
    | Some va -> va
    | None -> failwith "no address for the GPU channel"
  in
  let module R = D.Uvm_register_channel in
  let r = P.create R.sizeof in
  set_uuid r R.gpu_uuid g.uuid;
  P.set r R.rm_ctrl_fd c.ctl;
  P.set r R.h_client c.root;
  P.set r R.h_channel channel;
  P.set r R.base base;
  P.set r R.length size;
  or_undo
    (fun () -> Space.free c.low base)
    (fun () ->
      uvm_call c D.uvm_register_channel r R.rm_status
        "registering a GPU channel");
  base

(* Returns the addresses of a channel [register_channel] registered, which
   unregistering the GPU's address space unregistered. *)
let release_channel g base = Space.free g.c.low base
