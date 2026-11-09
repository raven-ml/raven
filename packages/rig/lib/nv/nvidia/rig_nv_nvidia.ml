(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Any domain may call any function. A GPU's objects are made once, under
   [gpus_lock], and kept for the process, its registered channels and whether a
   device of it is open under the same lock; the users of every range are kept
   under [ranges_lock]. Both, and the RM's client lock, are [Rig_nv.Lock]s,
   none taken while another is held; the GPUs' addresses ([Va]) take none.
   An open checks the locks its thread holds as it starts, before it takes
   any.

   References:
   - NVIDIA's open-gpu-kernel-modules, releases 570.144, 580, 610 and 615:
     kernel-open/common/inc/nv-ioctl-numbers.h and nv-ioctl.h (the escape
     ioctls), src/nvidia/arch/nvalloc/unix/include/nv_escape.h,
     src/common/sdk/nvidia/inc/nvos.h (NVOS00, NVOS02, NVOS21, NVOS32,
     NVOS33, NVOS54), ctrl0000gpu.h, ctrl0080gpu.h, ctrl2080fb.h,
     ctrl2080gpu.h, ctrl2080gr.h, ctrl2080perf.h and ctrl2080gpumon.h (the
     utilization samples), and kernel-open/nvidia-uvm/uvm_ioctl.h and
     uvm_linux_ioctl.h.
   - Linux's Documentation/ABI/testing/sysfs-bus-pci: a function's vendor
     and class files. *)

module D = Defs

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind
let params = Rig_nv.params
let get = Rig_nv.get
let set = Rig_nv.set
let page = 0x1000
let huge_page = 2 lsl 20

external clock : int -> int -> int -> int -> int = "caml_rig_nv_nvidia_clock"

(* Video memory of this size or more takes 2 MiB pages. *)
let huge_from = 8 lsl 20

(* The usermode object's page, whose register at [doorbell_at]
   (NVC361_NOTIFY_CHANNEL_PENDING) wakes a channel. *)
let usermode_bytes = 0x10000
let doorbell_at = 0x90

(* The addresses a channel's registration with unified memory takes. *)
let channel_range = 0x400_0000

(* The address space's first address and size: 49 bits. *)
let va_base = 0x1000
let va_size = 0x1fffffb000000
let round_up n a = (n + a - 1) / a * a
let bits = Rm.bits

(* Work that the RM's watchdog does not bound is hung once it has not advanced
   for 30 s, the bound of a GPU driven without a kernel driver. *)
let hang_ms = 30_000

(* GPUs *)

type gpu = {
  c : Rm.t;
  u : Rm.unified;
  root : string; (* the machine's root *)
  file : string; (* its device file, [/dev/nvidiaN] under the machine's root *)
  bus : string;
  device : int;
  subdevice : int;
  vaspace : int;
  uuid : string;
  doorbell : int;
  flush : int option; (* a word of a page of its BAR, mapped for the process *)
  facts : Rig_nv.gpu;
  budget : int;
  cache_bytes : int option;
  hang_ms : int option; (* [None] while the RM's watchdog bounds its work *)
  clock : int; (* its timer's reader, for the process *)
  index : int; (* its number, in bus order *)
  mutable channels : (int * int) list; (* registered, with their ranges *)
  mutable left : (int * int) list;
      (* unregistered, with their ranges, kept while a channel stays
         registered *)
  opened : bool Atomic.t; (* whether a device of it is open and not stopped *)
}

(* GPU memory *)

(* Who chose an RM object's handle: the RM, or the process ({!Rm.handle}),
   which takes it back once the object is freed. *)
type naming = Rm_named | Self_named

(* Where a range's GPU addresses come from: the process's range of memory the
   host maps too ([Rm.unified]'s [low]), or the process's own addresses,
   reserved for the range alone wherever the process has room
   ({!Rm.reserve_any}), out of that range. The GPU's MMU takes 49-bit addresses, and unified
   memory maps a range for a GPU in the lower half of the CPU's addresses
   (open-gpu-kernel-modules' uvm_gpu.c, uvm_gpu_can_address): below 2^47 on
   x86-64, 2^48 on arm64, which bound the addresses the kernel gives a
   mapping placed without a hint. *)
type space = Host_visible | Process

type mem = {
  va : int;
  handle : int;
  host : int option; (* the process's address of it, if it maps it *)
  video : int option; (* the GPU whose memory it is, by number *)
  range : range; (* the range it lies in *)
  mutable user : user;
      (* its GPU's mapping of [range], which its free reads with no search:
         set under [ranges_lock] for a view *)
}

(* A range of unified memory where an RM memory object is mapped for each GPU
   that maps it, counted: memory a GPU allocated, or host memory [map_host]
   described. Every region lies in one, its own allocation's, a host range, or
   another GPU's range it views: unified memory maps a range once per GPU, so
   a GPU's last region unmaps it for that GPU, and the last region of all frees
   the range and its object.

   A range is made, with the parameters its give-back needs, before the first
   call that takes anything for it, and each call's product is stored in it
   as the call returns, with nothing allocated between, so that what it holds
   is always known ([give_back]). *)
and range = {
  bytes : int;
  naming : naming; (* who chooses [memory] *)
  parent : gpu; (* the GPU the object belongs to *)
  space : space; (* where [gpu_addr] is taken *)
  mutable gpu_addr : int; (* the GPUs' address of its first byte, or -1 *)
  mutable memory : int; (* the memory object's handle, or 0 *)
  mutable made : bool; (* whether the RM holds the object *)
  mutable cpu : bool; (* whether the process maps memory at [gpu_addr] *)
  mutable unified : bool; (* whether unified memory holds the range *)
  mutable spare : int; (* a file opened for one call, or -1 *)
  mutable users : user list; (* the GPUs that may map it *)
  mutable mapping : int; (* how many of [users] map it *)
  free_uvm : Rig_nv.params; (* UVM_FREE's, by [uvm_free] *)
  uvm_free : uvm_free;
  free_rm : Rig_nv.params; (* NV_ESC_RM_FREE's, by [Defs.Nvos00] *)
}

(* A GPU's mapping of a range: its regions over the range, and what a free of
   one needs, made with it: the parameters that unmap it, and the free
   itself. *)
and user = {
  gpu : gpu;
  mutable count : int;
  unmap : Rig_nv.params;
  mutable release : unit -> unit;
}

(* The fields of the release's UVM_FREE. *)
and uvm_free = {
  base : int * int;
  length : (int * int) option;
  status : int * int;
}

external reraise : exn -> 'a = "%reraise"

let key : mem Type.Id.t = Type.Id.make ()
let gpus_lock = Rig_nv.Lock.create ()

(* The pairs of GPUs, by number, that refused each other peer access, both
   ways, for the life of the process: one value, so that a GPU's refusals
   are stored with one store as it joins [gpus]. *)
let peer_refusals : (int * int) list Atomic.t = Atomic.make []

(* [f] over a file of [g]'s the RM maps memory through: each mapping has its
   own, as the driver keeps one mapping context per file. [Ok None] if the
   process or the system has no file left, a refusal of the memory. *)
let with_file g f =
  match Rm.open_spare g.file with
  | Ok None -> Ok None
  | Error _ as e -> e
  | Ok (Some fd) ->
      Fun.protect
        ~finally:(fun () -> Rm.close fd)
        (fun () ->
          let* () = Rm.register fd ~ctl:g.c.ctl in
          Result.map Option.some (f fd))

(* Whether the status [s] of [what] is NV_OK: [Ok false] for
   NV_ERR_NO_MEMORY. *)
let fits g what s =
  if s = D.nv_err_no_memory then Ok false
  else
    let* () = Rm.check g.c what s in
    Ok true

(* Maps the memory object [h] of [size] bytes into the process at [va],
   uncached: [Ok false] if the GPU's window onto its memory (BAR1) has no room
   or the process no file left. *)
let map_to_cpu g h size va =
  Result.map (Option.value ~default:false) @@ with_file g
  @@ fun fd ->
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

let set_uuid p (at, _) uuid =
  String.iteri (fun i ch -> Bigarray.Array1.set p (at + i) ch) uuid

(* The RM's refusals of host memory to describe for good: an address no memory
   backs, such as an unmapped page, or memory it cannot pin. The kernel driver
   answers NV_ERR_INVALID_ADDRESS for every pin that falls short, a lack of
   memory included (os-mlock.c, os_lock_user_pages); NV_ERR_NO_MEMORY, which
   returning memory may undo, comes from its own allocations. *)
let refusals = [ D.nv_err_invalid_address; D.nv_err_invalid_argument ]

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

let free_addresses g space va size =
  match space with
  | Host_visible -> Va.free g.u.low va size
  | Process -> Rm.unreserve va size

(* Ranges

   A call that makes a range runs its steps under one handler. A step that
   finds no memory raises [No_memory], and the handler, whatever ended the
   steps, first gives back what the range holds, allocating nothing, then
   its addresses in the process's low range and its handle number, which
   allocate: an exception there leaks those numbers alone. A part whose free
   unified memory or the RM refuses stays held, with every address it
   needs. *)

exception No_memory

(* Host memory the RM refuses for good, which [No_memory]'s returning of memory
   cannot undo. *)
exception Refused

(* The fault of a free that unified memory or the RM refused, made once, so
   that raising it allocates nothing. *)
let refused_free =
  Rig_nv.Fault "unified memory or the RM refused to give back GPU memory"

(* The ranges the process holds: each counted as [range] makes it, before
   anything is taken for it, until it is given back, and for good where its
   give-back left a part. For the suites ([ranges]). [count_held] is one
   atomic operation, which allocates nothing ([Atomic.incr] and [Atomic.decr]
   allocate the location they add to first), so an exception cannot end a
   free between what it gave back and its count. *)
let held = Atomic.make 0
let ranges () = Atomic.get held
let count_held delta = ignore (Atomic.fetch_and_add held delta : int)

(* [g]'s mapping of a range, which no region counts yet. *)
let user g =
  let module U = D.Uvm_unmap_external in
  let unmap = params U.sizeof in
  set_uuid unmap U.gpu_uuid g.uuid;
  { gpu = g; count = 0; unmap; release = ignore }

(* Whether the RM freed [r]'s object [h]. *)
let free_object_raw r h =
  let module F = D.Nvos00 in
  set r.free_rm F.h_object_old h;
  match Rm.escape_raw r.parent.c.ctl D.nv_esc_rm_free r.free_rm with
  | () -> get r.free_rm F.status = D.nv_ok
  | exception Unix.Unix_error _ -> false

(* Gives back what [r] holds but its numbers, allocating nothing: whether
   nothing stays. Unified memory's range goes first, then the object it maps,
   then the process's mapping the object describes. Each part leaves [r] once
   its free returned, so a refused one keeps its addresses, but an object the
   RM named, whose handle the RM may give out again once freed: it leaves
   first, so no free reaches another object. *)
let give_back r =
  let g = r.parent in
  if r.spare >= 0 then begin
    Rm.close r.spare;
    r.spare <- -1
  end;
  if r.unified then begin
    let f = r.uvm_free in
    set r.free_uvm f.base r.gpu_addr;
    (match f.length with Some l -> set r.free_uvm l r.bytes | None -> ());
    match Rm.uvm_raw g.u D.uvm_free r.free_uvm with
    | () -> if get r.free_uvm f.status = D.nv_ok then r.unified <- false
    | exception Unix.Unix_error _ -> ()
  end;
  if r.made && not r.unified then begin
    let h = r.memory in
    match r.naming with
    | Rm_named ->
        r.made <- false;
        r.memory <- 0;
        ignore (free_object_raw r h : bool)
    | Self_named -> if free_object_raw r h then r.made <- false
  end;
  if r.cpu && not r.made then begin
    Rm.unmap r.gpu_addr r.bytes;
    r.cpu <- false
  end;
  not (r.unified || r.made || r.cpu)

(* Gives back [r]'s handle number and addresses, once nothing holds them. *)
let give_numbers r =
  if (not r.made) && r.memory <> 0 then begin
    let h = r.memory in
    r.memory <- 0;
    if r.naming = Self_named then Rm.free_handle h
  end;
  if r.gpu_addr >= 0 && not (r.unified || r.made || r.cpu) then begin
    let va = r.gpu_addr in
    r.gpu_addr <- -1;
    free_addresses r.parent r.space va r.bytes
  end

(* Ends a region of [u]'s GPU over [r]: the GPU's last unmaps [r] for it,
   and the last of all gives [r] back. rig takes the region as given whatever
   a free raises, so a free allocates and polls nothing until it gave back
   what the RM and unified memory hold, reading the counts it needs with no
   search: then [r]'s numbers, whose free an exception may end, leaking them.
   Under [ranges_lock]. *)
let give u r =
  let g = u.gpu in
  if u.count > 1 || r.mapping > 1 then begin
    u.count <- u.count - 1;
    if u.count = 0 then begin
      r.mapping <- r.mapping - 1;
      let module U = D.Uvm_unmap_external in
      set u.unmap U.base r.gpu_addr;
      set u.unmap U.length r.bytes;
      match Rm.uvm_raw g.u D.uvm_unmap_external u.unmap with
      | () -> if get u.unmap U.rm_status <> D.nv_ok then raise refused_free
      | exception Unix.Unix_error _ -> raise refused_free
    end
  end
  else begin
    u.count <- 0;
    r.mapping <- 0;
    if not (give_back r) then raise refused_free;
    count_held (-1);
    give_numbers r
  end

(* [u] made [r]'s, its free made once. *)
let adopt r u = u.release <- (fun () -> give u r)

let range g ~naming ~space bytes =
  let (module R : D.RELEASE) = g.c.Rm.layouts in
  let module U = R.Uvm_free in
  let module F = D.Nvos00 in
  let free_rm = params F.sizeof in
  set free_rm F.h_root g.c.root;
  set free_rm F.h_object_parent g.device;
  let r =
    {
      bytes;
      naming;
      parent = g;
      space;
      gpu_addr = -1;
      memory = 0;
      made = false;
      cpu = false;
      unified = false;
      spare = -1;
      users = [];
      mapping = 0;
      free_uvm = params U.sizeof;
      uvm_free = { base = U.base; length = U.length; status = U.rm_status };
      free_rm;
    }
  in
  let u = user g in
  r.users <- [ u ];
  adopt r u;
  count_held 1;
  r

(* A call that makes a range runs its steps as [try steps r with e -> unmade
   r e], with no closure. The steps make the answer, so nothing allocates
   once the last of them returned, and [unmade], the handler, gives back what
   [r] holds before it allocates anything. A step's exception passes; no
   memory answers [None], or, where the give-back left a part, the fault of a
   refused free, as a free does. *)
let gave_up r e no_room never =
  let gone = give_back r in
  if gone then count_held (-1);
  give_numbers r;
  if e != No_memory && e != Refused then reraise e
  else if not gone then raise refused_free
  else if e == Refused then never
  else no_room

let unmade r e = gave_up r e None None
let unmapped r e = gave_up r e Rig_edge.No_room Rig_edge.Never

(* Steps

   Each step raises [No_memory] where the original call answered no memory,
   and [Rig_nv.Fault] for a failure. It stores what its call took in the
   range as the call returns, with nothing allocated between. *)

let ok = function Ok () -> () | Error why -> raise (Rig_nv.Fault why)
let unix what e =
  raise (Rig_nv.Fault (strf "%s: %s" what (Unix.error_message e)))

(* Raises unless the RM answered [s] to [what] with its memory. *)
let fitted g what s =
  if s = D.nv_err_no_memory then raise No_memory;
  ok (Rm.check g.c what s)

(* Takes [r]'s addresses at a multiple of [align]. *)
let take_addresses r ~align =
  match r.space with
  | Host_visible -> (
      match Va.alloc r.parent.u.low ~align r.bytes with
      | Some va -> r.gpu_addr <- va
      | None -> raise No_memory)
  | Process -> (
      match Rm.reserve_any_raw r.bytes align with
      | 0 -> raise No_memory
      | va -> r.gpu_addr <- va
      | exception Unix.Unix_error (e, _, _) ->
          unix "reserving the GPU's addresses" e)

(* Opens a file of [r]'s GPU for one call, tied to the control file. Each
   mapping has its own, as the RM keeps one mapping context per file. *)
let open_file r =
  let g = r.parent in
  match Rm.open_raw g.file with
  | fd ->
      r.spare <- fd;
      ok (Rm.register fd ~ctl:g.c.ctl)
  | exception Unix.Unix_error ((EMFILE | ENFILE), _, _) -> raise No_memory
  | exception Unix.Unix_error (e, _, _) -> unix g.file e

let close_file r =
  Rm.close r.spare;
  r.spare <- -1

(* Maps anonymous memory of the process at [r]'s addresses. *)
let map_anonymous r =
  match Rm.map_raw (-1) r.gpu_addr r.bytes with
  | () -> r.cpu <- true
  | exception Unix.Unix_error (e, _, _) -> unix "mapping memory for the GPU" e

(* Describes the process's memory at [va] to the RM as [r]'s object, of
   [r]'s GPU. *)
let describe r va =
  let g = r.parent in
  let module W = D.Nvos02_with_fd in
  let module O = D.Nvos02 in
  let w = params W.sizeof in
  let at (f, n) = (fst W.params + f, n) in
  let status = at O.status in
  let what = "describing host memory to the GPU" in
  r.memory <- Rm.handle ();
  set w W.fd (-1);
  set w (at O.h_root) g.c.root;
  set w (at O.h_object_parent) g.device;
  set w (at O.h_object_new) r.memory;
  set w (at O.h_class) D.nv01_memory_system_os_descriptor;
  set w (at O.p_memory) va;
  set w (at O.limit) (r.bytes - 1);
  set w (at O.flags)
    (bits D.nvos02_flags_physicality D.nvos02_flags_physicality_noncontiguous
    lor bits D.nvos02_flags_coherency D.nvos02_flags_coherency_cached
    lor bits D.nvos02_flags_mapping D.nvos02_flags_mapping_no_map);
  open_file r;
  ok (Rm.escape r.spare D.nv_esc_rm_alloc_memory w what);
  let s = get w status in
  if s = D.nv_ok then r.made <- true;
  close_file r;
  if not r.made then begin
    if List.mem s refusals then raise Refused;
    fitted g what s
  end

(* Allocates [r]'s object in its GPU's memory with the parameters [p]. *)
let allocate r p =
  let g = r.parent in
  let module A = D.Nvos21 in
  let a = params A.sizeof in
  let what = "allocating GPU memory" in
  set a A.h_root g.c.root;
  set a A.h_object_parent g.device;
  set a A.h_class D.nv1_memory_user;
  set a A.p_alloc_parms (Rm.address p);
  ok (Rm.escape g.c.ctl D.nv_esc_rm_alloc a what);
  let s = get a A.status in
  if s = D.nv_ok then begin
    r.memory <- get a A.h_object_new;
    r.made <- true
  end;
  ignore (Sys.opaque_identity p);
  if not r.made then fitted g what s

(* Maps [r]'s object into the process at [r]'s addresses, uncached. No room in
   the GPU's window onto its memory (BAR1) is no memory. *)
let map_bar r =
  let g = r.parent in
  let module W = D.Nvos33_with_fd in
  let module M = D.Nvos33 in
  let w = params W.sizeof in
  let at (f, n) = (fst W.params + f, n) in
  let status = at M.status in
  let what = "mapping GPU memory into the process" in
  set w (at M.h_client) g.c.root;
  set w (at M.h_device) g.device;
  set w (at M.h_memory) r.memory;
  set w (at M.length) r.bytes;
  set w (at M.flags)
    (bits D.nvos33_flags_caching_type D.nvos33_flags_caching_type_uncached);
  open_file r;
  set w W.fd r.spare;
  ok (Rm.escape g.c.ctl D.nv_esc_rm_map_memory w what);
  fitted g what (get w status);
  match Rm.map_raw r.spare r.gpu_addr r.bytes with
  | () ->
      r.cpu <- true;
      close_file r
  | exception Unix.Unix_error (e, _, _) -> unix "mapping memory for the GPU" e

(* Maps [r] for [u]'s GPU, counting a region of it once mapped. *)
let map_external r u =
  let g = u.gpu in
  let module U = D.Uvm_map_external_allocation in
  let module G = D.Uvm_gpu_mapping in
  let p = params U.sizeof in
  let what = "mapping GPU memory" in
  set p U.base r.gpu_addr;
  set p U.length r.bytes;
  set p U.rm_ctrl_fd g.c.ctl;
  set p U.h_client g.c.root;
  set p U.h_memory r.memory;
  set p U.gpu_attributes_count 1;
  let at, _, _ = U.per_gpu_attributes in
  set_uuid p (at + fst G.gpu_uuid, 16) g.uuid;
  set p
    (at + fst G.gpu_mapping_type, snd G.gpu_mapping_type)
    D.uvm_gpu_mapping_type_read_write_atomic;
  match Rm.uvm_raw g.u D.uvm_map_external_allocation p with
  | () ->
      let s = get p U.rm_status in
      if s = D.nv_ok then begin
        u.count <- 1;
        r.mapping <- r.mapping + 1
      end
      else fitted g what s
  | exception Unix.Unix_error (e, _, _) -> unix what e

(* Makes [r]'s range of unified memory at its addresses and maps its object
   there for its GPU. *)
let unify r =
  let g = r.parent in
  let module E = D.Uvm_create_external_range in
  let e = params E.sizeof in
  let what = "reserving GPU addresses" in
  set e E.base r.gpu_addr;
  set e E.length r.bytes;
  (match Rm.uvm_raw g.u D.uvm_create_external_range e with
  | () ->
      let s = get e E.rm_status in
      if s = D.nv_ok then r.unified <- true else ok (Rm.check g.c what s)
  | exception Unix.Unix_error (e, _, _) -> unix what e);
  map_external r (List.hd r.users)

let memory m =
  { Rig_nv.address = m.va; host = m.host; handle = m.handle; data = m }

(* [r]'s memory from [va]. *)
let answer r ~va ~host ~video =
  let user = List.hd r.users in
  memory { va; handle = r.memory; host; video; range = r; user }

(* Memory *)

(* Host memory the process allocates and describes to the RM, [size] a
   multiple of the host's page. *)
let host_steps r =
  take_addresses r ~align:(Rm.host_page ());
  map_anonymous r;
  describe r r.gpu_addr;
  unify r;
  Some (answer r ~va:r.gpu_addr ~host:(Some r.gpu_addr) ~video:None)

let alloc_host g size =
  let r = range g ~naming:Self_named ~space:Host_visible size in
  try host_steps r with e -> unmade r e

let video_steps r p ~cpu ~page_size =
  allocate r p;
  take_addresses r ~align:page_size;
  if cpu then map_bar r;
  unify r;
  Some
    (answer r ~va:r.gpu_addr
       ~host:(if cpu then Some r.gpu_addr else None)
       ~video:(Some r.parent.index))

(* Video memory, mapped for the host too through BAR1 if [cpu]. The RM allocates
   it first; the addresses follow. *)
let alloc_video g ~cpu size =
  let page_size = if size >= huge_from then huge_page else page in
  let size = round_up size page_size in
  let p = video_params g ~contiguous:cpu ~page_size size in
  let space = if cpu then Host_visible else Process in
  let r = range g ~naming:Rm_named ~space size in
  try video_steps r p ~cpu ~page_size with e -> unmade r e

let alloc g (kind : Rig_edge.memory) n =
  match kind with
  | Pinned -> alloc_host g (round_up n (Rm.host_page ()))
  | Device -> alloc_video g ~cpu:false n
  | Mapped -> alloc_video g ~cpu:true n

let ranges_lock = Rig_nv.Lock.create ()

(* Counts the view [v] of [g] over [r], mapping [r] for [g] first if [g] has
   none: whether [g] maps it. Under [ranges_lock]. A new GPU's mapping joins
   [r.users] uncounted before the call, and [v] takes it before the count. *)
let take g r v =
  let u =
    match List.find_opt (fun u -> u.gpu == g) r.users with
    | Some u -> u
    | None ->
        let u = user g in
        adopt r u;
        r.users <- u :: r.users;
        u
  in
  v.user <- u;
  if u.count > 0 then begin
    u.count <- u.count + 1;
    true
  end
  else
    match map_external r u with
    | () -> true
    | exception No_memory -> false

let borrowed_steps r ~a ~a0 ~page =
  describe r a0;
  take_addresses r ~align:page;
  unify r;
  Rig_edge.Maps
    (Some (answer r ~va:(r.gpu_addr + (a - a0)) ~host:(Some a) ~video:None))

(* Each call maps a range of its own, over the whole pages that hold the
   memory, which unified memory requires of its bounds: two ranges may describe
   the same pages. The GPUs address it from the low range, as the host's own
   address may lie above the 40 bits a semaphore takes. *)
let map_host g a n : mem Rig_nv.memory option Rig_edge.mapping =
  let page = Rm.host_page () in
  let a0 = a land lnot (page - 1) in
  let size = round_up (a + n) page - a0 in
  let r = range g ~naming:Self_named ~space:Host_visible size in
  try borrowed_steps r ~a ~a0 ~page with e -> unmapped r e

(* Another GPU's video memory needs peer access, which may be refused; host
   memory needs none. A view counts in the range of the region it views, and
   is made before the count. *)
let map_peer g (m : mem Rig_nv.memory) :
    mem Rig_nv.memory option Rig_edge.mapping =
  let viewed = m.data in
  let refused = function
    | Some u -> List.mem (g.index, u) (Atomic.get peer_refusals)
    | None -> false
  in
  if refused viewed.video then Never
  else
    let v = { viewed with user = viewed.user } in
    let view = Rig_edge.Maps (Some (memory v)) in
    Rig_nv.Lock.protect ranges_lock @@ fun () ->
    if take g viewed.range v then view else No_room

let free _ (m : mem Rig_nv.memory) =
  Rig_nv.Lock.protect ranges_lock m.data.user.release

(* Another function's DMA

   The kernel driver exports RM memory objects as one dma-buf through
   NV_ESC_EXPORT_TO_DMABUF_FD on a device file of the GPU (nv.c,
   nv-dmabuf.c's nv_dma_buf_export): with [fd] -1 it makes the dma-buf,
   [totalSize] bytes, from [numObjects] handles with their offsets and sizes,
   dups each object, so the dma-buf keeps its memory past the region's free,
   and returns the descriptor, opened O_RDWR | O_CLOEXEC, in [fd]. It answers
   NV_ERR_NOT_SUPPORTED where the module exports no dma-buf, as only the open
   modules do. The default mapping type serves every importer, mapping video
   memory through BAR1, or as the host's own pages on a GPU coherent with the
   host, and the importer's attach refuses a topology the module does not
   support (nv_dma_buf_attach). A region lies in its range's object, at its
   distance from the range's first byte. *)
let dma g (m : mem Rig_nv.memory) ~fn =
  let* () = Sysfs.known g.root fn in
  let r = m.data.range in
  let page = Rm.host_page () in
  if r.bytes mod page <> 0 then
    Error
      (strf
         "the memory's %d bytes are no whole number of the host's %d-byte pages"
         r.bytes page)
  else
    let module E = D.Export_dma_buf in
    let p = params E.sizeof in
    let first (at, n, _) = (at, n) in
    set p E.fd (-1);
    set p E.h_client g.c.root;
    set p E.total_objects 1;
    set p E.num_objects 1;
    set p E.index 0;
    set p E.total_size r.bytes;
    set p E.mapping_type D.nv_dmabuf_export_mapping_type_default;
    set p (first E.handles) r.memory;
    set p (first E.offsets) 0;
    set p (first E.sizes) r.bytes;
    let what = "exporting GPU memory as a dma-buf" in
    let* exported =
      with_file g (fun fd -> Rm.escape fd D.nv_esc_export_to_dmabuf_fd p what)
    in
    match exported with
    | None -> Error (what ^ ": the process has no file left")
    | Some () when get p E.status = D.nv_err_not_supported ->
        Error
          "NVIDIA's kernel driver exports no dma-buf of this GPU's memory: \
           that needs its open kernel modules"
    | Some () ->
        let* () = Rm.check g.c what (get p E.status) in
        Ok
          (Rig_edge.Dmabuf { fd = get p E.fd; offset = m.data.va - r.gpu_addr })

(* Channels *)

(* Registers the channel [ch] with unified memory at a range of its own. *)
let register g ch =
  let* base =
    match Va.alloc g.u.low ~align:page channel_range with
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
    Rm.uvm_call g.c g.u D.uvm_register_channel r R.rm_status
      "registering a GPU channel"
  with
  | Error _ as e ->
      Va.free g.u.low base channel_range;
      e
  | Ok () ->
      Rig_nv.Lock.protect gpus_lock (fun () ->
          g.channels <- (ch, base) :: g.channels);
      Ok ()

(* Ends [ch]'s registration. Unified memory maps a resource that channels of
   one group share once, in the range of the first channel that registers it,
   and destroys it with the last channel's unregistration
   (uvm_user_channel.c's create_va_range, destroy_va_ranges): a channel's
   range may hold a resource another registered channel still maps. So the
   ranges go back once no channel of the GPU stays registered; a range given
   back earlier would be refused to the next memory placed there. *)
let unregister g ch =
  let (module R : D.RELEASE) = g.c.layouts in
  let module U = R.Uvm_unregister_channel in
  let u = params U.sizeof in
  Option.iter (fun f -> set_uuid u f g.uuid) U.gpu_uuid;
  set u U.h_client g.c.root;
  set u U.h_channel ch;
  let* () =
    Rm.uvm_call g.c g.u D.uvm_unregister_channel u U.rm_status
      "unregistering a GPU channel"
  in
  let free =
    Rig_nv.Lock.protect gpus_lock @@ fun () ->
    let gone, kept = List.partition (fun (c, _) -> c = ch) g.channels in
    let left = gone @ g.left in
    g.channels <- kept;
    if kept = [] then begin
      g.left <- [];
      left
    end
    else begin
      g.left <- left;
      []
    end
  in
  List.iter (fun (_, base) -> Va.free g.u.low base channel_range) free;
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
let classes c device =
  let module C = D.Classlist in
  let p = params C.sizeof in
  let* () = Rm.control c device D.nv0080_ctrl_cmd_gpu_get_classlist p in
  let count = get p C.num_classes in
  let list = params (4 * count) in
  let* () =
    Rm.control c ~refs:[ (C.class_list, list) ] device
      D.nv0080_ctrl_cmd_gpu_get_classlist p
  in
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
let gr_info c subdevice =
  let module G = D.Gr_get_info in
  let module I = D.Gr_info in
  let n = List.length gr_indices in
  let infos = params (n * I.sizeof) in
  List.iteri
    (fun i idx -> set infos (fst I.index + (i * I.sizeof), snd I.index) idx)
    gr_indices;
  let p = params G.sizeof in
  set p G.gr_info_list_size n;
  let* () =
    Rm.control c ~refs:[ (G.gr_info_list, infos) ] subdevice
      D.nv2080_ctrl_cmd_gr_get_info p
  in
  Ok
    (List.init n (fun i -> get infos (fst I.data + (i * I.sizeof), snd I.data)))

let why = function Rig_nv.Lacking why | Refused why -> why

(* The frame buffer's fact [index], as the RM states it. *)
let fb_fact c subdevice index =
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
  let* () = Rm.control c subdevice D.nv2080_ctrl_cmd_fb_get_info_v2 p in
  Ok (get p (elt F.data))

(* The frame buffer's size [index], in bytes: the RM states sizes in KiB. *)
let fb_info c subdevice index =
  Result.map (fun kib -> kib * 1024) (fb_fact c subdevice index)

(* The L2 cache's bytes, which the RM states in bytes, 0 for a GPU without
   one (ctrl2080fb.h, NV2080_CTRL_FB_INFO_INDEX_L2CACHE_SIZE). *)
let l2_bytes c subdevice =
  let* n = fb_fact c subdevice D.nv2080_ctrl_fb_info_index_l2cache_size in
  Ok (if n > 0 then Some n else None)

(* The path's hang bound: [None] while the RM's watchdog runs on the GPU and
   ends work that holds it too long. It does not run on a GPU without a
   display. *)
let hang_bound c subdevice =
  let module W = D.Watchdog_info in
  let p = params W.sizeof in
  let* () = Rm.control c subdevice D.nv2080_ctrl_cmd_rc_get_watchdog_info p in
  let running = D.nv2080_ctrl_rc_get_watchdog_info_flags_running in
  if get p W.watchdog_status_flags land running <> 0 then Ok None
  else Ok (Some hang_ms)

(* The GPUs made, by bus address, under [gpus_lock]. *)
let gpus : (string * gpu) list ref = ref []

(* Enables peer access between [g] and each of [others], the GPUs opened
   before it: the numbers of those that refuse it. *)
let refused_by g others =
  List.filter_map
    (fun g' ->
      let module E = D.Uvm_enable_peer_access in
      let e = params E.sizeof in
      set_uuid e E.gpu_uuid_a g.uuid;
      set_uuid e E.gpu_uuid_b g'.uuid;
      match
        Rm.uvm_call g.c g.u D.uvm_enable_peer_access e E.rm_status
          "enabling peer access"
      with
      | Ok () -> None
      | Error _ -> Some g'.index)
    others

(* A GPU's first objects, which its usage reads through and an open builds on:
   its device file, registered with the client's control file, and the RM's
   device and subdevice of the GPU. Made at the GPU's first use and kept for
   the process, by bus address, under [gpus_lock]. *)
type base = { path : string; device : int; subdevice : int }

let bases : (string * base) list ref = ref []

(* [undoing f] is [f taken], whose [taken g] records [g] to undo it: a failed
   [f], and one an exception ends, gives back what it took, last first. *)
let undoing f =
  let undo = ref [] in
  let taken g = undo := g :: !undo in
  match f taken with
  | Ok _ as x -> x
  | Error _ as e ->
      List.iter (fun g -> g ()) !undo;
      e
  | exception e ->
      List.iter (fun g -> g ()) !undo;
      reraise e

let make_base c ~dev bus =
  undoing @@ fun taken ->
  let rm = Rm.rm c in
  let* cards = cards c in
  let* gpu_id, minor =
    match List.find_opt (fun (b, _, _) -> b = bus) cards with
    | Some (_, id, minor) -> Ok (id, minor)
    | None -> Error (bus ^ " is not held by NVIDIA's kernel driver")
  in
  let path = Filename.concat dev (strf "nvidia%d" minor) in
  let* file = Rm.open_file path in
  taken (fun () -> Rm.close file);
  let* () = Rm.register file ~ctl:c.ctl in
  let id = params D.Id_info.sizeof in
  set id D.Id_info.gpu_id gpu_id;
  let* () = Rm.control c c.root D.nv0000_ctrl_cmd_gpu_get_id_info_v2 id in
  let instance = get id D.Id_info.device_instance in
  let p = params D.Nv0080_alloc.sizeof in
  set p D.Nv0080_alloc.device_id instance;
  set p D.Nv0080_alloc.h_client_share c.root;
  set p D.Nv0080_alloc.va_mode
    D.nv_device_allocation_vamode_optional_multiple_vaspaces;
  let* device =
    Result.map_error why (rm.alloc ~parent:c.root D.nv01_device_0 (Some p))
  in
  (* Freeing the device frees every object under it. *)
  taken (fun () -> ignore (Rm.free c ~parent:c.root device));
  let* subdevice =
    Result.map_error why
      (rm.alloc ~parent:device D.nv20_subdevice_0
         (Some (params D.Nv2080_alloc.sizeof)))
  in
  let b = { path; device; subdevice } in
  Ok (b, (bus, b) :: !bases)

(* GPU [bus]'s base, under [gpus_lock]. *)
let base c ~dev bus =
  match List.assoc_opt bus !bases with
  | Some b -> Ok b
  | None -> (
      match make_base c ~dev bus with
      | Error _ as e -> e
      | Ok (b, bases') ->
          bases := bases';
          Ok b)

(* GPU [bus]'s objects, made at its first open and kept for the process,
   under [gpus_lock], with the values [gpus] and [peer_refusals] take as it
   joins them: the refusals of [others], the GPUs opened before it. A failed
   open, and one an exception ends, gives back what it took, last first. *)
let make_gpu c um ~dev ~index ~others bus =
  let undo = ref [] in
  let taken f = undo := f :: !undo in
  let rm = Rm.rm c in
  let made () =
    let* { path; device; subdevice } = base c ~dev bus in
    let alloc ~parent cls f size =
      let p = params size in
      f p;
      Result.map_error why (rm.alloc ~parent cls (Some p))
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
    taken (fun () -> ignore (Rm.free c ~parent:device vaspace));
    let* available = classes c device in
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
    let* gr = gr_info c subdevice in
    let* gpcs, tpcs, sms, warps, sm =
      match gr with
      | [ a; b; d; e; f ] -> Ok (a, b, d, e, f)
      | _ -> Error "reading the graphics engine's facts"
    in
    let* budget = fb_info c subdevice D.nv2080_ctrl_fb_info_index_heap_size in
    let* cache_bytes = l2_bytes c subdevice in
    let* hang_ms = hang_bound c subdevice in
    let module Gi = D.Gid_info in
    let gid = params Gi.sizeof in
    set gid Gi.flags
      (bits D.nv2080_gpu_cmd_gpu_get_gid_flags_format
         D.nv2080_gpu_cmd_gpu_get_gid_flags_format_binary);
    set gid Gi.length 16;
    let* () = Rm.control c subdevice D.nv2080_ctrl_cmd_gpu_get_gid_info gid in
    let data, _, _ = Gi.data in
    let uuid = String.init 16 (fun i -> Bigarray.Array1.get gid (data + i)) in
    let* usermode =
      Result.map_error why (rm.alloc ~parent:subdevice usermode_class None)
    in
    taken (fun () -> ignore (Rm.free c ~parent:subdevice usermode));
    let* doorbell =
      match Va.alloc um.Rm.low ~align:page usermode_bytes with
      | Some va -> Ok va
      | None -> Error "no address for the GPU's doorbell"
    in
    taken (fun () -> Va.free um.Rm.low doorbell usermode_bytes);
    let g =
      {
        c;
        u = um;
        root = Filename.dirname dev;
        file = path;
        bus;
        device;
        subdevice;
        vaspace;
        uuid;
        doorbell;
        flush = None;
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
        cache_bytes;
        hang_ms;
        clock = clock c.ctl Rm.control_request c.root subdevice;
        index;
        channels = [];
        left = [];
        opened = Atomic.make false;
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
    let unregister_gpu () =
      let module U = D.Uvm_unregister_gpu in
      let u = params U.sizeof in
      set_uuid u U.gpu_uuid uuid;
      ignore
        (Rm.uvm_call c um D.uvm_unregister_gpu u U.rm_status
           "unregistering the GPU"
          : (unit, string) result)
    in
    (* Recorded before the registration, which unified memory makes once per
       process: an exception between the two would refuse the GPU for good.
       Unregistering a GPU never registered is refused, and the refusal
       dropped. *)
    taken unregister_gpu;
    let* () =
      Rm.uvm_call c um D.uvm_register_gpu u U.rm_status "registering the GPU"
    in
    let module V = D.Uvm_register_gpu_vaspace in
    let v = params V.sizeof in
    set_uuid v V.gpu_uuid uuid;
    set v V.rm_ctrl_fd c.ctl;
    set v V.h_client c.root;
    set v V.h_va_space vaspace;
    let* () =
      Rm.uvm_call c um D.uvm_register_gpu_vaspace v V.rm_status
        "registering the GPU's address space"
    in
    (* The flush word's page, kept with the GPU's objects. A GPU whose BAR
       has no room for it has none, and its devices no Mapped memory. *)
    let* bar =
      match alloc_video g ~cpu:true page with
      | bar -> Ok bar
      | exception Rig_nv.Fault why -> Error why
    in
    let g =
      match bar with
      | Some m -> { g with flush = Some m.Rig_nv.address }
      | None -> g
    in
    let refused = refused_by g others in
    let pairs =
      List.concat_map (fun i -> [ (index, i); (i, index) ]) refused
      @ Atomic.get peer_refusals
    in
    Ok (g, (bus, g) :: !gpus, pairs)
  in
  match made () with
  | Ok _ as made -> made
  | Error _ as e ->
      List.iter (fun f -> f ()) !undo;
      e
  | exception e ->
      List.iter (fun f -> f ()) !undo;
      reraise e

let gpu c u ~dev ~index bus =
  Rig_nv.Lock.protect gpus_lock @@ fun () ->
  match List.assoc_opt bus !gpus with
  | Some g -> Ok g
  | None ->
      let others = List.map snd !gpus in
      (* What the GPU joins is made inside the make, whose handler gives the
         GPU back, and nothing allocates from its return to the stores. *)
      match make_gpu c u ~dev ~index ~others bus with
      | Error _ as e -> e
      | Ok (g, gpus', pairs) ->
          gpus := gpus';
          Atomic.set peer_refusals pairs;
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
    cache_bytes = g.cache_bytes;
    doorbell = g.doorbell + doorbell_at;
    flush = g.flush;
    alloc = alloc g;
    map_host = Some (map_host g);
    index = g.index;
    reaches = (fun i -> not (List.mem (g.index, i) (Atomic.get peer_refusals)));
    map_peer = map_peer g;
    free = free g;
    bus = Some g.bus;
    dma = dma g;
    register = register g;
    unregister = unregister g;
    (* The RM reports the GPU's faults through the channels; channels the
       device could not free may still run. *)
    check = ignore;
    hang_ms = g.hang_ms;
    clock = Rig_nv.Source g.clock;
    (* Allocates nothing, so that an exception that ended the device's stop
       before it still lets the GPU open again. *)
    stop =
      (fun () ->
        Atomic.set g.opened false;
        `Unknown);
  }

(* Opening *)

let buses_under ~root = Sysfs.gpus root
let buses () = buses_under ~root:"/"
let count_under ~root = Result.map List.length (buses_under ~root)
let count () = count_under ~root:"/"

let device_name i =
  if i < 0 then invalid_argf "Rig_nv_nvidia.gpu: GPU %d < 0" i;
  if i = 0 then "NV" else strf "NV:%d" i

(* The bus address of GPU [i] of the machine under [root]. *)
let nth_bus ~root i =
  let* buses =
    Result.map_error
      (fun why -> "reading the machine's GPUs: " ^ why)
      (Sysfs.gpus root)
  in
  match List.nth_opt buses i with
  | None ->
      Error
        (strf "no GPU %d: the machine has %d NVIDIA GPUs" i (List.length buses))
  | Some bus -> Ok bus

(* GPU [i] of the machine under [root], its objects made at its first call. *)
let nth_gpu ~root i =
  let* bus = nth_bus ~root i in
  let dev = Filename.concat root "dev" in
  let* c = Rm.client ~dev in
  let* u = Rm.unified c ~dev in
  gpu c u ~dev ~index:i bus

let open_ ?(wrap = Fun.id) ~root ~word i =
  Rig_nv.Lock.guarded "Rig_nv_nvidia.gpu" @@ fun () ->
  if i < 0 then invalid_argf "Rig_nv_nvidia.gpu: GPU %d < 0" i;
  match nth_gpu ~root i with
  | Error _ as e -> e
  | Ok g -> (
      let p = wrap (path g) in
      (* The claim is given back on any end of the make but its [Ok], and
         nothing allocates between the claim and the make. *)
      if not (Atomic.compare_and_set g.opened false true) then
        Error (strf "%s has a device open" g.bus)
      else
        match Rig_nv.make ~word p with
        | Ok _ as d -> d
        | Error _ as e ->
            Atomic.set g.opened false;
            e
        | exception e ->
            Atomic.set g.opened false;
            reraise e)

let gpu_under ~root i : Rig_nv.t Rig_edge.opener =
  {
    name = device_name i;
    driver = (module Rig_nv);
    make = (fun ~word -> open_ ~root ~word i);
  }

let gpu i = gpu_under ~root:"/" i

let gpu_with ~wrap i =
  { (gpu i) with make = (fun ~word -> open_ ~wrap ~root:"/" ~word i) }

(* Usage *)

type usage = { memory_total : int; memory_free : int; busy : float option }

(* The GPU's timer, in nanoseconds since 1970 (NV2080_CTRL_CMD_TIMER_GET_TIME
   of open-gpu-kernel-modules' ctrl2080tmr.h): the clock of the utilization
   samples' time stamps, which runs apart from the host's (2.9 s behind on an
   RTX 5000 Ada, and 4.5 ppm slower). *)
let gpu_time c subdevice =
  let module T = D.Timer_time in
  let p = params T.sizeof in
  let* () = Rm.control c subdevice D.nv2080_ctrl_cmd_timer_get_time p in
  Ok (get p T.time_nsec)

(* The mean of the RM's utilization samples of the graphics engine taken in
   the last [window] seconds of the GPU's timer, from its ring of the last 72
   samples (NV2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2 of
   ctrl2080perf.h, the samples of ctrl2080gpumon.h). A sample's time stamp is
   zero in a slot not yet written, and its utilization in hundredths of a
   percent. The ring's [count] is not used: release 615's GSP leaves it 0
   with every slot written. On an RTX 5000 Ada the RM samples every 200 ms,
   so the ring holds 14.4 s. [None] where the RM takes no samples: it refuses
   the command with NV_ERR_NOT_SUPPORTED, as with MIG. *)
let busy c subdevice ~window =
  let module U = D.Util_samples in
  let module S = D.Util_sample in
  let p = params U.sizeof in
  let at, size, slots = U.samples in
  set p U.type_ D.nv2080_ctrl_gpumon_sample_type_perfmon_util;
  set p U.buf_size (size * slots);
  let* status =
    Rm.control_status c subdevice
      D.nv2080_ctrl_cmd_perf_get_gpumon_perfmon_util_samples_v2 p
  in
  if status = D.nv_err_not_supported then Ok None
  else
    let* () = Rm.check c "reading the GPU's utilization" status in
    let* now = gpu_time c subdevice in
    (* A window past the GPU's time since 1970 holds every sample, and its
       nanoseconds may not fit an [int]. *)
    let span = window *. 1e9 in
    let since =
      if span >= float_of_int now then 0 else now - int_of_float span
    in
    let field i (f, n) = get p (at + (i * size) + f, n) in
    let rec sum i n total =
      if i = slots then (n, total)
      else
        let t = field i S.base_time_stamp in
        if t = 0 || t < since then sum (i + 1) n total
        else sum (i + 1) (n + 1) (total + field i S.gr_util)
    in
    match sum 0 0 0 with
    | 0, _ -> Ok None
    | n, total -> Ok (Some (float_of_int total /. float_of_int n /. 100.))

(* Whether the PCI function [bus] under [root] is powered for a read, by its
   runtime_status (Linux's Documentation/ABI/testing/sysfs-devices-power):
   [active], or [unsupported] where the kernel does not suspend it. A function
   suspended, suspending, resuming or in error, or whose status cannot be read,
   is not: a read could resume it. *)
let powered ~root bus =
  let dir = Filename.concat root ("sys/bus/pci/devices/" ^ bus) in
  let path = Filename.concat dir "power/runtime_status" in
  match In_channel.with_open_bin path In_channel.input_line with
  | Some ("active" | "unsupported") -> true
  | Some _ | None | (exception Sys_error _) -> false

let usage ~window i =
  Rig_nv.Lock.guarded "Rig_nv_nvidia.usage" @@ fun () ->
  if i < 0 then invalid_argf "Rig_nv_nvidia.usage: GPU %d < 0" i;
  (* The message names no value: rig has no float printer of its own, and
     Printf's [%g] writes the C library's exponent, of three digits under
     mingw-w64. *)
  if not (window > 0.) then
    invalid_arg "Rig_nv_nvidia.usage: the window is not positive";
  let root = "/" in
  let* bus = nth_bus ~root i in
  if not (powered ~root bus) then
    Error (bus ^ " is not active: the kernel may have suspended it")
  else
    let dev = Filename.concat root "dev" in
    let* c = Rm.client ~dev in
    let* b = Rig_nv.Lock.protect gpus_lock (fun () -> base c ~dev bus) in
    let* memory_total =
      fb_info c b.subdevice D.nv2080_ctrl_fb_info_index_total_ram_size
    in
    let* memory_free =
      fb_info c b.subdevice D.nv2080_ctrl_fb_info_index_heap_free
    in
    let* busy = busy c b.subdevice ~window in
    Ok { memory_total; memory_free; busy }
