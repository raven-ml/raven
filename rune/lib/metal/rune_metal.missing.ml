(* This file is selected when nx.metal is not available *)

let is_available = false

let not_available () =
  failwith "Metal backend is not available on this platform"

type context = unit
type ('a, 'b) t = unit

let create_context () = not_available ()
let view _t = not_available ()
let dtype _t = not_available ()
let context _t = not_available ()
let data _t = not_available ()
let op_buffer _t = not_available ()
let op_const_scalar _t = not_available ()
let op_const_array _t = not_available ()
let op_add _t = not_available ()
let op_mul _t = not_available ()
let op_idiv _t = not_available ()
let op_fdiv _t = not_available ()
let op_max _t = not_available ()
let op_mod _t = not_available ()
let op_pow _t = not_available ()
let op_cmplt _t = not_available ()
let op_cmpne _t = not_available ()
let op_xor _t = not_available ()
let op_or _t = not_available ()
let op_and _t = not_available ()
let op_neg _t = not_available ()
let op_log2 _t = not_available ()
let op_exp2 _t = not_available ()
let op_sin _t = not_available ()
let op_sqrt _t = not_available ()
let op_recip _t = not_available ()
let op_where _t = not_available ()
let op_reduce_sum ~axes:_ ~keepdims:_ _t = not_available ()
let op_reduce_max ~axes:_ ~keepdims:_ _t = not_available ()
let op_reduce_prod ~axes:_ ~keepdims:_ _t = not_available ()
let op_expand _t = not_available ()
let op_reshape _t = not_available ()
let op_permute _t = not_available ()
let op_pad _t = not_available ()
let op_shrink _t = not_available ()
let op_flip _t = not_available ()
let op_unfold _t ~kernel_size:_ ~stride:_ ~dilation:_ ~padding:_ = not_available ()
let op_fold _t ~output_size:_ ~kernel_size:_ ~stride:_ ~dilation:_ ~padding:_ = not_available ()
let op_cat _t = not_available ()
let op_cast _t = not_available ()
let op_contiguous _t = not_available ()
let op_copy _t = not_available ()
let op_assign _t = not_available ()
let op_threefry _t = not_available ()
let op_gather _t = not_available ()
let op_scatter _t = not_available ()

(* Stub JIT backend when Metal is not available *)
module Jit_backend : Rune_jit.Backend_intf.S = struct
  let name = "STUB_BACKEND"

  type device_info = unit
  type device_buffer_native = unit
  type compiled_artifact_native = unit
  type callable_kernel_native = unit

  type nonrec 'a device_buffer =
    ('a, device_buffer_native) Rune_jit.Backend_intf.device_buffer

  type nonrec any_device_buffer =
    device_buffer_native Rune_jit.Backend_intf.any_device_buffer

  type nonrec compiled_artifact =
    compiled_artifact_native Rune_jit.Backend_intf.compiled_artifact

  type nonrec callable_kernel =
    callable_kernel_native Rune_jit.Backend_intf.callable_kernel

  module Device_info = struct
    let get_default () = not_available ()
    let max_shared_memory _ = not_available ()
    let max_workgroup_size _ = not_available ()
    let supports_dtype _ _ = not_available ()
    let renderer_float4_str _ = not_available ()
    let renderer_smem_prefix _ = not_available ()
    let renderer_barrier_str _ = not_available ()
  end

  module Renderer = struct
    let render ~device_info:_ ~lowered_ir:_ ~kernel_name:_ = not_available ()
  end

  module Compiler = struct
    type compile_options = unit

    let default_options _ = not_available ()
    let compile ~device_info:_ ~source_code:_ ~options:_ = not_available ()
  end

  module Runtime = struct
    let allocate_buffer ~device_info:_ ~size_in_bytes:_ ~dtype:_ =
      not_available ()

    let copy_to_device ~dest_buffer:_ ~host_data:_ ~host_data_offset_bytes:_
        ~copy_size_bytes:_ =
      not_available ()

    let copy_from_device ~src_buffer:_ ~host_dest_ptr:_
        ~device_data_offset_bytes:_ ~copy_size_bytes:_ =
      not_available ()

    let get_kernel ~artifact:_ ~kernel_name:_ = not_available ()

    let launch_kernel ?local_dims:_ ~device_info:_ ~global_dims:_ ~args:_ _ =
      not_available ()

    let synchronize ~device_info:_ = not_available ()
  end
end
