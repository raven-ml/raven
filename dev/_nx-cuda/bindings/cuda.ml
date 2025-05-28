(* Basic types *)
type cu_device = int
type cu_context = int64
type cu_module = int64
type cu_function = int64
type cu_deviceptr = int64
type cu_stream = int64
type cu_event = int64

(* Initialization *)
external cu_init : int -> unit = "caml_cu_init"

(* Device Management *)
external cu_device_get_count : unit -> int = "caml_cu_device_get_count"
external cu_device_get : int -> cu_device = "caml_cu_device_get"

(* Context Management *)
external cu_ctx_create : int -> cu_device -> cu_context = "caml_cu_ctx_create"
external cu_ctx_get_flags : unit -> int = "caml_cu_ctx_get_flags"

external cu_device_primary_ctx_retain : cu_device -> cu_context
  = "caml_cu_device_primary_ctx_retain"

external cu_device_primary_ctx_release : cu_device -> unit
  = "caml_cu_device_primary_ctx_release"

external cu_device_primary_ctx_reset : cu_device -> unit
  = "caml_cu_device_primary_ctx_reset"

external cu_ctx_get_device : unit -> cu_device = "caml_cu_ctx_get_device"
external cu_ctx_get_current : unit -> cu_context = "caml_cu_ctx_get_current"
external cu_ctx_pop_current : unit -> cu_context = "caml_cu_ctx_pop_current"
external cu_ctx_set_current : cu_context -> unit = "caml_cu_ctx_set_current"
external cu_ctx_push_current : cu_context -> unit = "caml_cu_ctx_push_current"

(* Module Management *)
external cu_module_load_data_ex : string -> (int * int64) array -> cu_module
  = "caml_cu_module_load_data_ex"

external cu_module_get_function : cu_module -> string -> cu_function
  = "caml_cu_module_get_function"

(* Memory Management *)
external cu_mem_alloc : int64 -> cu_deviceptr = "caml_cu_mem_alloc"

external cu_mem_alloc_async : int64 -> cu_stream -> cu_deviceptr
  = "caml_cu_mem_alloc_async"

external cu_memcpy_H_to_D : cu_deviceptr -> bytes -> int64 -> unit
  = "caml_cu_memcpy_H_to_D"

external cu_memcpy_H_to_D_async :
  cu_deviceptr -> bytes -> int64 -> cu_stream -> unit
  = "caml_cu_memcpy_H_to_D_async"

external cu_memcpy_D_to_H : bytes -> cu_deviceptr -> int64 -> unit
  = "caml_cu_memcpy_D_to_H"

external cu_memcpy_D_to_H_async :
  bytes -> cu_deviceptr -> int64 -> cu_stream -> unit
  = "caml_cu_memcpy_D_to_H_async"

external cu_memcpy_D_to_D : cu_deviceptr -> cu_deviceptr -> int64 -> unit
  = "caml_cu_memcpy_D_to_D"

external cu_memcpy_D_to_D_async :
  cu_deviceptr -> cu_deviceptr -> int64 -> cu_stream -> unit
  = "caml_cu_memcpy_D_to_D_async"

external cu_memcpy_peer :
  cu_deviceptr -> cu_context -> cu_deviceptr -> cu_context -> int64 -> unit
  = "caml_cu_memcpy_peer"

external cu_memcpy_peer_async :
  cu_deviceptr ->
  cu_context ->
  cu_deviceptr ->
  cu_context ->
  int64 ->
  cu_stream ->
  unit = "caml_cu_memcpy_peer_async_bytecode" "caml_cu_memcpy_peer_async"
[@@noalloc]

external cu_mem_free : cu_deviceptr -> unit = "caml_cu_mem_free"

external cu_mem_free_async : cu_deviceptr -> cu_stream -> unit
  = "caml_cu_mem_free_async"

(* Kernel Launch *)
external cu_launch_kernel :
  cu_function -> int -> int -> int -> int -> int -> int -> int64 array -> unit
  = "caml_cu_launch_kernel_bytecode" "caml_cu_launch_kernel"
[@@noalloc]

(* Synchronization *)
external cu_ctx_synchronize : unit -> unit = "caml_cu_ctx_synchronize"

(* Peer Access *)
external cu_ctx_disable_peer_access : cu_context -> unit
  = "caml_cu_ctx_disable_peer_access"

external cu_ctx_enable_peer_access : cu_context -> int -> unit
  = "caml_cu_ctx_enable_peer_access"

external cu_device_can_access_peer : cu_device -> cu_device -> int
  = "caml_cu_device_can_access_peer"

external cu_device_get_p2p_attribute : int -> cu_device -> cu_device -> int
  = "caml_cu_device_get_p2p_attribute"

(* Module Unload *)
external cu_module_unload : cu_module -> unit = "caml_cu_module_unload"

(* Context Destruction *)
external cu_ctx_destroy : cu_context -> unit = "caml_cu_ctx_destroy"

(* Memory Set *)
external cu_memset_d8 : cu_deviceptr -> int -> int64 -> unit
  = "caml_cu_memset_d8"

external cu_memset_d16 : cu_deviceptr -> int -> int64 -> unit
  = "caml_cu_memset_d16"

external cu_memset_d32 : cu_deviceptr -> int -> int64 -> unit
  = "caml_cu_memset_d32"

external cu_memset_d8_async : cu_deviceptr -> int -> int64 -> cu_stream -> unit
  = "caml_cu_memset_d8_async"

external cu_memset_d16_async : cu_deviceptr -> int -> int64 -> cu_stream -> unit
  = "caml_cu_memset_d16_async"

external cu_memset_d32_async : cu_deviceptr -> int -> int64 -> cu_stream -> unit
  = "caml_cu_memset_d32_async"

(* Memory Info *)
external cu_mem_get_info : unit -> int64 * int64 = "caml_cu_mem_get_info"

(* Module Globals *)
external cu_module_get_global : cu_module -> string -> cu_deviceptr * int64
  = "caml_cu_module_get_global"

(* Device Info *)
external cu_device_get_name : int -> cu_device -> string
  = "caml_cu_device_get_name"

external cu_device_get_attribute : int -> cu_device -> int
  = "caml_cu_device_get_attribute"

(* Context Limits *)
external cu_ctx_set_limit : int -> int64 -> unit = "caml_cu_ctx_set_limit"
external cu_ctx_get_limit : int -> int64 = "caml_cu_ctx_get_limit"

(* Stream Management *)
external cu_stream_attach_mem_async :
  cu_stream -> cu_deviceptr -> int64 -> int -> unit
  = "caml_cu_stream_attach_mem_async"

external cu_stream_create_with_priority : int -> int -> cu_stream
  = "caml_cu_stream_create_with_priority"

external cu_stream_destroy : cu_stream -> unit = "caml_cu_stream_destroy"
external cu_stream_get_ctx : cu_stream -> cu_context = "caml_cu_stream_get_ctx"
external cu_stream_get_id : cu_stream -> int64 = "caml_cu_stream_get_id"
external cu_stream_query : cu_stream -> int = "caml_cu_stream_query"

external cu_stream_synchronize : cu_stream -> unit
  = "caml_cu_stream_synchronize"

(* Event Management *)
external cu_event_create : int -> cu_event = "caml_cu_event_create"
external cu_event_destroy : cu_event -> unit = "caml_cu_event_destroy"

external cu_event_elapsed_time : cu_event -> cu_event -> float
  = "caml_cu_event_elapsed_time"

external cu_event_record_with_flags : cu_event -> cu_stream -> int -> unit
  = "caml_cu_event_record_with_flags"

external cu_event_query : cu_event -> int = "caml_cu_event_query"
external cu_event_synchronize : cu_event -> unit = "caml_cu_event_synchronize"

external cu_stream_wait_event : cu_stream -> cu_event -> int -> unit
  = "caml_cu_stream_wait_event"
