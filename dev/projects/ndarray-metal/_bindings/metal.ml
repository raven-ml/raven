exception Metal_error of string

let () = Callback.register_exception "Metal_error" (Metal_error "")

type resource_options =
  | CPU_Cache_Mode_Default
  | CPU_Cache_Mode_Write_Combined
  | Storage_Mode_Shared
  | Storage_Mode_Managed
  | Storage_Mode_Private
  | Storage_Mode_Memoryless
  | Hazard_Tracking_Default
  | Hazard_Tracking_Untracked
  | Hazard_Tracking_Tracked

type device
type command_queue
type buffer
type library
type function_
type pipeline_state
type command_buffer
type command_encoder
type blit_command_encoder
type compile_options
type grid_size = { x : int64; y : int64; z : int64 }
type thread_group_size = { x : int64; y : int64; z : int64 }

let cpu_cache_mode_shift = 0
let storage_mode_shift = 4
let hazard_tracking_mode_shift = 8

let resource_options_to_int_value = function
  | CPU_Cache_Mode_Default -> 0 lsl cpu_cache_mode_shift
  | CPU_Cache_Mode_Write_Combined -> 1 lsl cpu_cache_mode_shift
  | Storage_Mode_Shared -> 0 lsl storage_mode_shift
  | Storage_Mode_Managed -> 1 lsl storage_mode_shift
  | Storage_Mode_Private -> 2 lsl storage_mode_shift
  | Storage_Mode_Memoryless -> 3 lsl storage_mode_shift
  | Hazard_Tracking_Default -> 0 lsl hazard_tracking_mode_shift
  | Hazard_Tracking_Untracked -> 1 lsl hazard_tracking_mode_shift
  | Hazard_Tracking_Tracked -> 2 lsl hazard_tracking_mode_shift

let resource_options_to_int options =
  List.fold_left
    (fun acc opt -> acc lor resource_options_to_int_value opt)
    0 options

(* Define external functions pointing to the C stubs *)
external create_device : unit -> device = "caml_metal_create_device"
external get_device_name : device -> string = "caml_metal_get_device_name"

external create_command_queue : device -> command_queue
  = "caml_metal_new_command_queue"

external create_buffer_native : device -> int64 -> int -> buffer
  = "caml_metal_new_buffer"

let create_buffer dev size_bytes options =
  let options_mask = resource_options_to_int options in
  create_buffer_native dev size_bytes options_mask

external create_buffer_with_pointer_native :
  device -> nativeint -> int64 -> int -> buffer
  = "caml_metal_new_buffer_with_pointer"

let create_buffer_with_pointer dev ptr size_bytes options =
  let options_mask = resource_options_to_int options in
  create_buffer_with_pointer_native dev ptr size_bytes options_mask

external create_buffer_with_data_native :
  device -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> int -> buffer
  = "caml_metal_new_buffer_with_data"

let create_buffer_with_data dev bigarray options =
  let options_mask = resource_options_to_int options in
  create_buffer_with_data_native dev bigarray options_mask

external create_buffer_with_bytes_native : device -> bytes -> int -> buffer
  = "caml_metal_new_buffer_with_bytes"

let create_buffer_with_bytes dev bytes options =
  let options_mask = resource_options_to_int options in
  create_buffer_with_bytes_native dev bytes options_mask

external buffer_length : buffer -> int64 = "caml_metal_buffer_length"

external buffer_contents_native : buffer -> nativeint
  = "caml_metal_buffer_contents"

let buffer_contents buf =
  buffer_contents_native buf (* Simple wrapper for now *)

external copy_to_buffer_native : buffer -> int64 -> nativeint -> int64 -> unit
  = "caml_metal_copy_to_buffer"

let copy_to_buffer buf ~offset ~ptr ~num_bytes =
  copy_to_buffer_native buf offset ptr num_bytes

external copy_from_buffer_native : buffer -> int64 -> nativeint -> int64 -> unit
  = "caml_metal_copy_from_buffer"

let copy_from_buffer buf ~offset ~ptr ~num_bytes =
  copy_from_buffer_native buf offset ptr num_bytes

external buffer_did_modify_range_native : buffer -> int64 -> int64 -> unit
  = "caml_metal_buffer_did_modify_range"

let buffer_did_modify_range buf ~offset ~length =
  buffer_did_modify_range_native buf offset length

external create_compile_options : unit -> compile_options
  = "caml_metal_create_compile_options"

external set_compile_option_fast_math_enabled : compile_options -> bool -> unit
  = "caml_metal_set_compile_option_fast_math_enabled"

(* Handling optional argument for library creation *)
external create_library_with_source_native :
  device -> string -> compile_options option -> library
  = "caml_metal_new_library_with_source"

let create_library_with_source dev source ?options () =
  create_library_with_source_native dev source options

external create_library_with_data : device -> string -> library
  = "caml_metal_new_library_with_data"

external create_function_with_name : library -> string -> function_
  = "caml_metal_new_function_with_name"

external create_compute_pipeline_state : device -> function_ -> pipeline_state
  = "caml_metal_new_compute_pipeline_state"

external get_pipeline_state_max_total_threads_per_threadgroup :
  pipeline_state -> int64
  = "caml_metal_pipeline_state_get_max_total_threads_per_threadgroup"

external create_command_buffer : command_queue -> command_buffer
  = "caml_metal_new_command_buffer"

external create_compute_command_encoder : command_buffer -> command_encoder
  = "caml_metal_new_compute_command_encoder"

external set_compute_pipeline_state : command_encoder -> pipeline_state -> unit
  = "caml_metal_set_compute_pipeline_state"

external set_buffer_native : command_encoder -> buffer -> int64 -> int -> unit
  = "caml_metal_set_buffer"

let set_buffer encoder buffer ~offset ~index =
  set_buffer_native encoder buffer offset index

external set_bytes_native : command_encoder -> nativeint -> int -> int -> unit
  = "caml_metal_set_bytes"

let set_bytes encoder ~bytes_ptr ~length ~index =
  set_bytes_native encoder bytes_ptr length index

external dispatch_thread_groups_native :
  command_encoder -> grid_size -> thread_group_size -> unit
  = "caml_metal_dispatch_thread_groups"

let dispatch_thread_groups encoder ~grid_size ~thread_group_size =
  dispatch_thread_groups_native encoder grid_size thread_group_size

external end_encoding : command_encoder -> unit = "caml_metal_end_encoding"
external commit : command_buffer -> unit = "caml_metal_commit"

external wait_until_completed : command_buffer -> unit
  = "caml_metal_wait_until_completed"

external create_blit_command_encoder : command_buffer -> blit_command_encoder
  = "caml_metal_new_blit_command_encoder"

external copy_from_buffer_to_buffer_native :
  blit_command_encoder -> buffer -> int64 -> buffer -> int64 -> int64 -> unit
  = "caml_metal_blit_copy_from_buffer_to_buffer_bytecode"
    "caml_metal_blit_copy_from_buffer_to_buffer"

let copy_from_buffer_to_buffer encoder ~src_buffer ~src_offset ~dst_buffer
    ~dst_offset ~size =
  copy_from_buffer_to_buffer_native encoder src_buffer src_offset dst_buffer
    dst_offset size

external end_blit_encoding : blit_command_encoder -> unit
  = "caml_metal_end_blit_encoding"
