(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

let strf = Printf.sprintf

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type t = {
  mutable number : int;
  mutable size : int;
  params : params;
  data : params;
  data_address : int;
}

external address : params -> int = "caml_rig_amd_amdgpu_address" [@@noalloc]
external get16 : params -> int -> int = "%caml_bigstring_get16"
external get32 : params -> int -> int32 = "%caml_bigstring_get32"
external get64 : params -> int -> int64 = "%caml_bigstring_get64"
external set16 : params -> int -> int -> unit = "%caml_bigstring_set16"
external set32 : params -> int -> int32 -> unit = "%caml_bigstring_set32"
external set64 : params -> int -> int64 -> unit = "%caml_bigstring_set64"

let get_at p at n =
  match n with
  | 1 -> Char.code (Bigarray.Array1.get p at)
  | 2 -> get16 p at
  | 4 -> Int32.to_int (get32 p at) land 0xffff_ffff
  | _ -> Int64.to_int (get64 p at)

let set_at p at n v =
  match n with
  | 1 -> Bigarray.Array1.set p at (Char.unsafe_chr (v land 0xff))
  | 2 -> set16 p at (v land 0xffff)
  | 4 -> set32 p at (Int32.of_int v)
  | _ -> set64 p at (Int64.of_int v)

let get p (at, n) = get_at p at n
let set p (at, n) v = set_at p at n v

(* Zeroes the first [n] bytes of [p]. *)
let zero p n =
  for i = 0 to (n / 8) - 1 do
    set64 p (8 * i) 0L
  done;
  for i = n land lnot 7 to n - 1 do
    Bigarray.Array1.unsafe_set p i '\000'
  done

(* Requests *)

module E = D.Event_data

(* The most a wait waits for: a signal and the two exception events. *)
let max_events = 3

(* The bytes of the largest request's parameters, and of the largest memory a
   pointer in them names. *)
let params_bytes =
  List.fold_left Int.max 0
    D.
      [
        Get_version.sizeof;
        Acquire_vm.sizeof;
        Runtime_enable.sizeof;
        Alloc_memory_of_gpu.sizeof;
        Free_memory_of_gpu.sizeof;
        Map_memory_to_gpu.sizeof;
        Create_event.sizeof;
        Destroy_event.sizeof;
        Reset_event.sizeof;
        Create_queue.sizeof;
        Destroy_queue.sizeof;
        Wait_events.sizeof;
        Info.sizeof;
        Ctx.sizeof;
      ]

let data_bytes = Int.max D.Info_device.sizeof (max_events * E.sizeof)

let make () =
  let buffer n = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  let params = buffer params_bytes and data = buffer data_bytes in
  { number = 0; size = 0; params; data; data_address = address data }

(* Each domain's request. A thread's request may release the runtime while its
   ioctl runs, and another thread of the domain run meanwhile, so a request is
   taken from its domain's slot, which holds [busy] until it is given back; a
   thread that finds [busy] makes a request of its own. *)
let busy = make ()
let slot = Domain.DLS.new_key (fun () -> Atomic.make (make ()))

let take () =
  let r = Atomic.exchange (Domain.DLS.get slot) busy in
  if r == busy then make () else r

let give r = Atomic.set (Domain.DLS.get slot) r

(* Starts [r] as the request [number] of [size] bytes of parameters, all
   zero. *)
let start r number size =
  r.number <- number;
  r.size <- size;
  zero r.params size

(* KFD *)

let version r = start r D.amdkfd_ioc_get_version D.Get_version.sizeof

let version_of r =
  let module V = D.Get_version in
  (get r.params V.major_version * 1000) + get r.params V.minor_version

let acquire_vm r ~drm ~gpu =
  let module A = D.Acquire_vm in
  start r D.amdkfd_ioc_acquire_vm A.sizeof;
  set r.params A.drm_fd drm;
  set r.params A.gpu_id gpu

let runtime_enable r =
  start r D.amdkfd_ioc_runtime_enable D.Runtime_enable.sizeof

type memory = [ `Gpu | `Bar | `System | `Userptr | `Mmio ]

let shared =
  D.kfd_ioc_alloc_mem_flags_coherent lor D.kfd_ioc_alloc_mem_flags_uncached
  lor D.kfd_ioc_alloc_mem_flags_public

let flags : memory -> int = function
  | `Gpu -> D.kfd_ioc_alloc_mem_flags_vram
  | `Bar -> D.kfd_ioc_alloc_mem_flags_vram lor D.kfd_ioc_alloc_mem_flags_public
  | `System -> D.kfd_ioc_alloc_mem_flags_gtt lor shared
  | `Userptr -> D.kfd_ioc_alloc_mem_flags_userptr lor shared
  | `Mmio -> D.kfd_ioc_alloc_mem_flags_mmio_remap

let alloc r ~gpu ~va ~bytes m =
  let module A = D.Alloc_memory_of_gpu in
  start r D.amdkfd_ioc_alloc_memory_of_gpu A.sizeof;
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
   lor D.kfd_ioc_alloc_mem_flags_no_substitute lor executable)

let handle r = get r.params D.Alloc_memory_of_gpu.handle
let mmap_offset r = get64 r.params (fst D.Alloc_memory_of_gpu.mmap_offset)

let free r handle =
  let module F = D.Free_memory_of_gpu in
  start r D.amdkfd_ioc_free_memory_of_gpu F.sizeof;
  set r.params F.handle handle

(* KFD maps and unmaps with the same layout, which the generator checks: the two
   requests share their fields. *)
module M = D.Map_memory_to_gpu

let mapping number r ~gpu handle =
  start r number M.sizeof;
  set_at r.data 0 4 gpu;
  set r.params M.handle handle;
  set r.params M.device_ids_array_ptr r.data_address;
  set r.params M.n_devices 1

let map r ~gpu handle = mapping D.amdkfd_ioc_map_memory_to_gpu r ~gpu handle

let unmap r ~gpu handle =
  mapping D.amdkfd_ioc_unmap_memory_from_gpu r ~gpu handle

let mapped r = get r.params M.n_success

let event r k ~page =
  let module E = D.Create_event in
  start r D.amdkfd_ioc_create_event E.sizeof;
  let type_ =
    match k with
    | `Signal -> D.kfd_ioc_event_signal
    | `Memory -> D.kfd_ioc_event_memory
    | `Hardware -> D.kfd_ioc_event_hw_exception
  in
  set r.params E.event_page_offset page;
  set r.params E.event_type type_;
  set r.params E.auto_reset (if k = `Signal then 1 else 0)

let event_id r = get r.params D.Create_event.event_id

let destroy_event r id =
  let module E = D.Destroy_event in
  start r D.amdkfd_ioc_destroy_event E.sizeof;
  set r.params E.event_id id

let reset_event r id =
  let module E = D.Reset_event in
  start r D.amdkfd_ioc_reset_event E.sizeof;
  set r.params E.event_id id

(* The priority of the path's queues, of 0 to 15. *)
let queue_priority = 7

let queue r k ~gpu ~ring ~ring_bytes ~eop ~eop_bytes ~save ~save_bytes
    ~ctl_stack ~write ~read =
  let module Q = D.Create_queue in
  start r D.amdkfd_ioc_create_queue Q.sizeof;
  let type_ =
    match k with
    | `Pm4 -> D.kfd_ioc_queue_type_compute
    | `Aql -> D.kfd_ioc_queue_type_compute_aql
    | `Sdma -> D.kfd_ioc_queue_type_sdma
  in
  let p = r.params in
  set p Q.gpu_id gpu;
  set p Q.ring_base_address ring;
  set p Q.ring_size ring_bytes;
  set p Q.eop_buffer_address eop;
  set p Q.eop_buffer_size eop_bytes;
  set p Q.ctx_save_restore_address save;
  set p Q.ctx_save_restore_size save_bytes;
  set p Q.ctl_stack_size ctl_stack;
  set p Q.write_pointer_address write;
  set p Q.read_pointer_address read;
  set p Q.queue_type type_;
  set p Q.queue_percentage D.kfd_max_queue_percentage;
  set p Q.queue_priority queue_priority

let queue_id r = get r.params D.Create_queue.queue_id
let doorbell_offset r = get64 r.params (fst D.Create_queue.doorbell_offset)

let destroy_queue r id =
  let module Q = D.Destroy_queue in
  start r D.amdkfd_ioc_destroy_queue Q.sizeof;
  set r.params Q.queue_id id

module W = D.Wait_events

let wait r ids ~ms =
  let n = Array.length ids in
  if n > max_events then
    invalid_arg (strf "Request.wait: %d events, past %d" n max_events);
  start r D.amdkfd_ioc_wait_events W.sizeof;
  zero r.data (n * E.sizeof);
  let at, w = E.event_id in
  for i = 0 to n - 1 do
    set_at r.data (at + (i * E.sizeof)) w ids.(i)
  done;
  set r.params W.events_ptr r.data_address;
  set r.params W.num_events n;
  set r.params W.timeout ms

(* The field [f] of the exception event [k] of the wait [r]: the memory
   exception event is the second to last of a wait's, the hardware exception
   event the last. *)
let field r k (at, w) =
  let n = get r.params W.num_events in
  let i = match k with `Memory -> n - 2 | `Hardware -> n - 1 in
  get_at r.data (at + (i * E.sizeof)) w

let exception_id r k = field r k E.event_id

let exception_gpu r k =
  match k with
  | `Memory -> field r k E.memory_exception_data_gpu_id
  | `Hardware -> field r k E.hw_exception_data_gpu_id

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

let device_info r =
  let module I = D.Info in
  start r D.drm_ioctl_amdgpu_info I.sizeof;
  zero r.data D.Info_device.sizeof;
  set r.params I.return_pointer r.data_address;
  set r.params I.return_size D.Info_device.sizeof;
  set r.params I.query D.amdgpu_info_dev_info

let clock_khz r = get r.data D.Info_device.gpu_counter_freq

let compute_units r =
  let at, w, n = D.Info_device.cu_bitmap in
  Array.init n (fun i -> get_at r.data (at + (i * w)) w)

let context_request r op ctx =
  let module C = D.Ctx in
  start r D.drm_ioctl_amdgpu_ctx C.sizeof;
  set r.params C.in_op op;
  set r.params C.in_ctx_id ctx

let alloc_context r = context_request r D.amdgpu_ctx_op_alloc_ctx 0
let context r = get r.params D.Ctx.out_alloc_ctx_id

let stable_pstate r ctx =
  context_request r D.amdgpu_ctx_op_set_stable_pstate ctx;
  set r.params D.Ctx.in_flags D.amdgpu_ctx_stable_pstate_standard

let free_context r ctx = context_request r D.amdgpu_ctx_op_free_ctx ctx
