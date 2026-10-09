(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

let strf = Printf.sprintf

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type t = { number : int; params : params; data : params }

external address : params -> int = "caml_rig_amd_amdgpu_address"
external get16 : params -> int -> int = "%caml_bigstring_get16"
external get32 : params -> int -> int32 = "%caml_bigstring_get32"
external get64 : params -> int -> int64 = "%caml_bigstring_get64"
external set16 : params -> int -> int -> unit = "%caml_bigstring_set16"
external set32 : params -> int -> int32 -> unit = "%caml_bigstring_set32"
external set64 : params -> int -> int64 -> unit = "%caml_bigstring_set64"

let params n =
  let p = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill p '\000';
  p

let get p (at, n) =
  match n with
  | 1 -> Char.code (Bigarray.Array1.get p at)
  | 2 -> get16 p at
  | 4 -> Int32.to_int (get32 p at) land 0xffff_ffff
  | _ -> Int64.to_int (get64 p at)

let set p (at, n) v =
  match n with
  | 1 -> Bigarray.Array1.set p at (Char.unsafe_chr (v land 0xff))
  | 2 -> set16 p at (v land 0xffff)
  | 4 -> set32 p at (Int32.of_int v)
  | _ -> set64 p at (Int64.of_int v)

let none = params 0

let request ?(data = none) number sizeof =
  { number; params = params sizeof; data }

(* KFD *)

let version () = request D.amdkfd_ioc_get_version D.Get_version.sizeof

let version_of r =
  let module V = D.Get_version in
  (get r.params V.major_version * 1000) + get r.params V.minor_version

let acquire_vm ~drm ~gpu =
  let module A = D.Acquire_vm in
  let r = request D.amdkfd_ioc_acquire_vm A.sizeof in
  set r.params A.drm_fd drm;
  set r.params A.gpu_id gpu;
  r

let runtime_enable () =
  request D.amdkfd_ioc_runtime_enable D.Runtime_enable.sizeof

type memory = [ `Gpu | `Bar | `System | `Userptr | `Mmio ]

let flags : memory -> int =
  let shared =
    D.kfd_ioc_alloc_mem_flags_coherent lor D.kfd_ioc_alloc_mem_flags_uncached
    lor D.kfd_ioc_alloc_mem_flags_public
  in
  function
  | `Gpu -> D.kfd_ioc_alloc_mem_flags_vram
  | `Bar -> D.kfd_ioc_alloc_mem_flags_vram lor D.kfd_ioc_alloc_mem_flags_public
  | `System -> D.kfd_ioc_alloc_mem_flags_gtt lor shared
  | `Userptr -> D.kfd_ioc_alloc_mem_flags_userptr lor shared
  | `Mmio -> D.kfd_ioc_alloc_mem_flags_mmio_remap

let alloc ~gpu ~va ~bytes m =
  let module A = D.Alloc_memory_of_gpu in
  let r = request D.amdkfd_ioc_alloc_memory_of_gpu A.sizeof in
  let executable =
    if m = `Mmio then 0 else D.kfd_ioc_alloc_mem_flags_executable
  in
  set r.params A.va_addr va;
  set r.params A.size bytes;
  (* The process's own memory is named by its address. *)
  if m = `Userptr then set r.params A.mmap_offset va;
  set r.params A.gpu_id gpu;
  set r.params A.flags
    (flags m lor D.kfd_ioc_alloc_mem_flags_writable
   lor D.kfd_ioc_alloc_mem_flags_no_substitute lor executable);
  r

let allocation r =
  let module A = D.Alloc_memory_of_gpu in
  (get r.params A.handle, get64 r.params (fst A.mmap_offset))

let free handle =
  let module F = D.Free_memory_of_gpu in
  let r = request D.amdkfd_ioc_free_memory_of_gpu F.sizeof in
  set r.params F.handle handle;
  r

(* KFD maps and unmaps with the same layout, which the generator checks: the two
   requests share their fields. *)
module M = D.Map_memory_to_gpu

let mapping number ~gpu handle =
  let ids = params 4 in
  set ids (0, 4) gpu;
  let r = request ~data:ids number M.sizeof in
  set r.params M.handle handle;
  set r.params M.device_ids_array_ptr (address ids);
  set r.params M.n_devices 1;
  r

let map = mapping D.amdkfd_ioc_map_memory_to_gpu
let unmap = mapping D.amdkfd_ioc_unmap_memory_from_gpu
let mapped r = get r.params M.n_success

let event k ~page =
  let module E = D.Create_event in
  let r = request D.amdkfd_ioc_create_event E.sizeof in
  let type_ =
    match k with
    | `Signal -> D.kfd_ioc_event_signal
    | `Memory -> D.kfd_ioc_event_memory
    | `Hardware -> D.kfd_ioc_event_hw_exception
  in
  set r.params E.event_page_offset page;
  set r.params E.event_type type_;
  set r.params E.auto_reset (if k = `Signal then 1 else 0);
  r

let event_id r = get r.params D.Create_event.event_id

let destroy_event id =
  let module E = D.Destroy_event in
  let r = request D.amdkfd_ioc_destroy_event E.sizeof in
  set r.params E.event_id id;
  r

let reset_event id =
  let module E = D.Reset_event in
  let r = request D.amdkfd_ioc_reset_event E.sizeof in
  set r.params E.event_id id;
  r

(* The priority of the path's queues, of 0 to 15. *)
let queue_priority = 7

let queue k ~gpu ~ring ~ring_bytes ~eop ~eop_bytes ~save ~save_bytes ~ctl_stack
    ~write ~read =
  let module Q = D.Create_queue in
  let r = request D.amdkfd_ioc_create_queue Q.sizeof in
  let type_ =
    match k with
    | `Pm4 -> D.kfd_ioc_queue_type_compute
    | `Aql -> D.kfd_ioc_queue_type_compute_aql
    | `Sdma -> D.kfd_ioc_queue_type_sdma
  in
  List.iter
    (fun (f, v) -> set r.params f v)
    [
      (Q.gpu_id, gpu);
      (Q.ring_base_address, ring);
      (Q.ring_size, ring_bytes);
      (Q.eop_buffer_address, eop);
      (Q.eop_buffer_size, eop_bytes);
      (Q.ctx_save_restore_address, save);
      (Q.ctx_save_restore_size, save_bytes);
      (Q.ctl_stack_size, ctl_stack);
      (Q.write_pointer_address, write);
      (Q.read_pointer_address, read);
      (Q.queue_type, type_);
      (Q.queue_percentage, D.kfd_max_queue_percentage);
      (Q.queue_priority, queue_priority);
    ];
  r

let queue_made r =
  let module Q = D.Create_queue in
  (get r.params Q.queue_id, get64 r.params (fst Q.doorbell_offset))

let destroy_queue id =
  let module Q = D.Destroy_queue in
  let r = request D.amdkfd_ioc_destroy_queue Q.sizeof in
  set r.params Q.queue_id id;
  r

module E = D.Event_data

let wait ids ~ms =
  let module W = D.Wait_events in
  let n = Array.length ids in
  let events = params (n * E.sizeof) in
  Array.iteri
    (fun i id -> set events (fst E.event_id + (i * E.sizeof), 4) id)
    ids;
  let r = request ~data:events D.amdkfd_ioc_wait_events W.sizeof in
  set r.params W.events_ptr (address events);
  set r.params W.num_events n;
  set r.params W.timeout ms;
  r

(* The memory exception event is the second to last of a wait's, the hardware
   exception event the last. *)
let field r k (at, w) =
  let n = Bigarray.Array1.dim r.data / E.sizeof in
  let i = match k with `Memory -> n - 2 | `Hardware -> n - 1 in
  get r.data (at + (i * E.sizeof), w)

let exception_event r k =
  let gpu =
    match k with
    | `Memory -> E.memory_exception_data_gpu_id
    | `Hardware -> E.hw_exception_data_gpu_id
  in
  (field r k E.event_id, field r k gpu)

let fault r k =
  let f = field r k in
  match k with
  | `Memory ->
      strf
        "memory fault at 0x%x (not present %d, read-only %d, no execute %d, \
         imprecise %d, error type %d)"
        (f E.memory_exception_data_va)
        (f E.memory_exception_data_failure_not_present)
        (f E.memory_exception_data_failure_read_only)
        (f E.memory_exception_data_failure_no_execute)
        (f E.memory_exception_data_failure_imprecise)
        (f E.memory_exception_data_error_type)
  | `Hardware ->
      strf "hardware exception (reset type %d, reset cause %d, memory lost %d)"
        (f E.hw_exception_data_reset_type)
        (f E.hw_exception_data_reset_cause)
        (f E.hw_exception_data_memory_lost)

(* The render node *)

let device_info () =
  let module I = D.Info in
  let info = params D.Info_device.sizeof in
  let r = request ~data:info D.drm_ioctl_amdgpu_info I.sizeof in
  set r.params I.return_pointer (address info);
  set r.params I.return_size D.Info_device.sizeof;
  set r.params I.query D.amdgpu_info_dev_info;
  r

let clock_khz r = get r.data D.Info_device.gpu_counter_freq

let compute_units r =
  let at, w, n = D.Info_device.cu_bitmap in
  Array.init n (fun i -> get r.data (at + (i * w), w))

let context_request op ctx =
  let module C = D.Ctx in
  let r = request D.drm_ioctl_amdgpu_ctx C.sizeof in
  set r.params C.in_op op;
  set r.params C.in_ctx_id ctx;
  r

let alloc_context () = context_request D.amdgpu_ctx_op_alloc_ctx 0
let context r = get r.params D.Ctx.out_alloc_ctx_id

let stable_pstate ctx =
  let r = context_request D.amdgpu_ctx_op_set_stable_pstate ctx in
  set r.params D.Ctx.in_flags D.amdgpu_ctx_stable_pstate_standard;
  r

let free_context ctx = context_request D.amdgpu_ctx_op_free_ctx ctx
