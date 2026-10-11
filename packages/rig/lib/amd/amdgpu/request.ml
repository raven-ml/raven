(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The requests the path makes of KFD and of a GPU's render node, which the
   stubs pack in Linux's uapi structures. A request raises [Unix.Unix_error]
   for the kernel's refusal, and the stubs read and build the constructors of
   the variants here by their order. *)

(* The kinds of memory the path allocates: GPU memory, GPU memory the host maps
   through the BAR, system memory the kernel driver owns, the process's own
   memory, and the page of registers the driver remaps. *)
type memory = Gpu | Bar | System | Userptr | Mmio

(* The kinds of events: a signal, a memory exception, a hardware exception. *)
type event = Signal | Memory | Hardware

(* The kinds of queues. *)
type queue = Pm4 | Aql | Sdma

(* A fault of a GPU, as an exception event reports it: a memory fault at [va]
   with its failure's bits and error type, or a hardware exception with its
   reset's type and cause and whether memory was lost. *)
type fault =
  | Memory_fault of {
      va : int;
      not_present : int;
      read_only : int;
      no_execute : int;
      imprecise : int;
      error_type : int;
    }
  | Hardware_exception of {
      reset_type : int;
      reset_cause : int;
      memory_lost : int;
    }

let fault_message = function
  | Memory_fault f ->
      Printf.sprintf
        "memory fault at 0x%x (not present %d, read-only %d, no execute %d, \
         imprecise %d, error type %d)"
        f.va f.not_present f.read_only f.no_execute f.imprecise f.error_type
  | Hardware_exception h ->
      Printf.sprintf
        "hardware exception (reset type %d, reset cause %d, memory lost %d)"
        h.reset_type h.reset_cause h.memory_lost

(* KFD *)

(* [version kfd] is KFD's version, major * 1000 + minor. *)
external version : int -> int = "caml_rig_amd_amdgpu_version"

(* [acquire_vm kfd drm gpu] gives the process the address space of GPU [gpu],
   whose render node is open as [drm]. *)
external acquire_vm : int -> int -> int -> unit
  = "caml_rig_amd_amdgpu_acquire_vm"

external runtime_enable : int -> unit = "caml_rig_amd_amdgpu_runtime_enable"

(* [alloc kfd gpu va bytes m] allocates [bytes] bytes of [m] for GPU [gpu] at
   the address [va]: its handle and the offset to map it at. *)
external alloc : int -> int -> int -> int -> memory -> int * int64
  = "caml_rig_amd_amdgpu_alloc"

external free : int -> int -> unit = "caml_rig_amd_amdgpu_free"

(* [map_gpu kfd handle gpu map] maps [handle] for GPU [gpu], or unmaps it if
   [map] is [false]: the GPUs it reached. *)
external map_gpu : int -> int -> int -> bool -> int
  = "caml_rig_amd_amdgpu_map_gpu"

(* [event kfd k page] makes an event of kind [k] on the event page whose handle
   is [page], or 0 for events after the page's first: its id. *)
external event : int -> event -> int -> int = "caml_rig_amd_amdgpu_event"

external destroy_event : int -> int -> unit
  = "caml_rig_amd_amdgpu_destroy_event"

external reset_event : int -> int -> unit = "caml_rig_amd_amdgpu_reset_event"

(* [queue kfd k gpu ring ring_bytes eop eop_bytes save save_bytes ctl_stack
   write read] makes a queue of kind [k] on GPU [gpu]: its id and its
   doorbell's offset. *)
external queue :
  int ->
  queue ->
  int ->
  int ->
  int ->
  int ->
  int ->
  int ->
  int ->
  int ->
  int ->
  int ->
  int * int64 = "caml_rig_amd_amdgpu_queue_byte" "caml_rig_amd_amdgpu_queue"

external destroy_queue : int -> int -> unit
  = "caml_rig_amd_amdgpu_destroy_queue"

(* [wait kfd ids ms me] waits at most [ms] for the events [ids], a signal, then
   a memory and a hardware exception event, or the two exception events alone.
   It is [Some f] for a fault [f] of GPU [me] they report, else [None], and
   allocates nothing then. It resets the exception events another GPU's fault
   set, which would end every later wait at once. *)
external wait : int -> int array -> int -> int -> fault option
  = "caml_rig_amd_amdgpu_wait"

(* The render node *)

(* [device_info drm] is the GPU's clock in kHz and its active compute units, 16
   masks engine-major as cu_bitmap lays them out. *)
external device_info : int -> int * int array
  = "caml_rig_amd_amdgpu_device_info"

(* [stable_pstate drm] holds the GPU in its stable power state with a new
   context, until the process closes [drm]. *)
external stable_pstate : int -> unit = "caml_rig_amd_amdgpu_stable_pstate"
