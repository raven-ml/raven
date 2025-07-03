open Ctypes

module Functions (F : Ctypes.FOREIGN) = struct
  open F

  module Status = struct
    let error_message =
      foreign "status_error_message" (ptr void @-> returning string)

    let free = foreign "status_free" (ptr void @-> returning void)
  end

  module Shape = struct
    let make_shape_array =
      foreign "make_shape_array"
        (int @-> size_t @-> ptr int64_t @-> returning (ptr void))

    let dimensions_size =
      foreign "shape_dimensions_size" (ptr void @-> returning int)

    let element_type = foreign "shape_element_type" (ptr void @-> returning int)

    let dimensions =
      foreign "shape_dimensions" (ptr void @-> int @-> returning int64_t)

    let tuple_shapes_size =
      foreign "shape_tuple_shapes_size" (ptr void @-> returning int)

    let free = foreign "shape_free" (ptr void @-> returning void)
  end

  module Builder = struct
    let create = foreign "xla_builder_create" (string @-> returning (ptr void))
    let first_error = foreign "first_error" (ptr void @-> returning (ptr void))

    let get_current_status =
      foreign "get_current_status" (ptr void @-> returning (ptr void))

    let free = foreign "xla_builder_free" (ptr void @-> returning void)
  end

  module Literal = struct
    let create_from_shape =
      foreign "literal_create_from_shape"
        (int @-> ptr int64_t @-> size_t @-> returning (ptr void))

    let create_from_shape_and_data =
      foreign "literal_create_from_shape_and_data"
        (int @-> ptr int64_t @-> size_t @-> ptr void @-> size_t
        @-> returning (ptr void))

    let shape =
      foreign "literal_shape" (ptr void @-> ptr (ptr void) @-> returning void)

    let element_type =
      foreign "literal_element_type" (ptr void @-> returning int)

    let element_count =
      foreign "literal_element_count" (ptr void @-> returning int64_t)

    let size_bytes =
      foreign "literal_size_bytes" (ptr void @-> returning int64_t)

    let copy_to =
      foreign "literal_copy_to"
        (ptr void @-> ptr void @-> size_t @-> returning void)

    let copy_from =
      foreign "literal_copy_from"
        (ptr void @-> ptr void @-> size_t @-> returning void)

    let free = foreign "literal_free" (ptr void @-> returning void)

    let create_r0_f32 =
      foreign "create_r0_float" (float @-> returning (ptr void))

    let create_r0_f64 =
      foreign "create_r0_double" (double @-> returning (ptr void))
  end

  module Op = struct
    let constant_literal =
      foreign "constant_literal" (ptr void @-> ptr void @-> returning (ptr void))

    let parameter =
      foreign "parameter"
        (ptr void @-> int64_t @-> int @-> int @-> ptr int64_t @-> string
        @-> returning (ptr void))

    (* Binary operations *)
    let add = foreign "op_add" (ptr void @-> ptr void @-> returning (ptr void))
    let sub = foreign "op_sub" (ptr void @-> ptr void @-> returning (ptr void))
    let mul = foreign "op_mul" (ptr void @-> ptr void @-> returning (ptr void))
    let div = foreign "op_div" (ptr void @-> ptr void @-> returning (ptr void))
    let rem = foreign "op_rem" (ptr void @-> ptr void @-> returning (ptr void))
    let max = foreign "op_max" (ptr void @-> ptr void @-> returning (ptr void))
    let min = foreign "op_min" (ptr void @-> ptr void @-> returning (ptr void))
    let pow = foreign "op_pow" (ptr void @-> ptr void @-> returning (ptr void))
    let dot = foreign "op_dot" (ptr void @-> ptr void @-> returning (ptr void))

    (* Bitwise operations *)
    let and_ = foreign "op_and" (ptr void @-> ptr void @-> returning (ptr void))
    let or_ = foreign "op_or" (ptr void @-> ptr void @-> returning (ptr void))
    let xor = foreign "op_xor" (ptr void @-> ptr void @-> returning (ptr void))

    (* Unary operations *)
    let neg = foreign "op_neg" (ptr void @-> returning (ptr void))
    let abs = foreign "op_abs" (ptr void @-> returning (ptr void))
    let exp = foreign "op_exp" (ptr void @-> returning (ptr void))
    let log = foreign "op_log" (ptr void @-> returning (ptr void))
    let sqrt = foreign "op_sqrt" (ptr void @-> returning (ptr void))
    let sin = foreign "op_sin" (ptr void @-> returning (ptr void))
    let cos = foreign "op_cos" (ptr void @-> returning (ptr void))
    let tanh = foreign "op_tanh" (ptr void @-> returning (ptr void))

    (* Comparison operations *)
    let eq = foreign "op_eq" (ptr void @-> ptr void @-> returning (ptr void))
    let ne = foreign "op_ne" (ptr void @-> ptr void @-> returning (ptr void))
    let lt = foreign "op_lt" (ptr void @-> ptr void @-> returning (ptr void))
    let le = foreign "op_le" (ptr void @-> ptr void @-> returning (ptr void))
    let gt = foreign "op_gt" (ptr void @-> ptr void @-> returning (ptr void))
    let ge = foreign "op_ge" (ptr void @-> ptr void @-> returning (ptr void))

    (* Shape manipulation *)
    let reshape =
      foreign "op_reshape"
        (ptr void @-> size_t @-> ptr int64_t @-> returning (ptr void))

    let transpose =
      foreign "op_transpose"
        (ptr void @-> size_t @-> ptr int64_t @-> returning (ptr void))

    let broadcast =
      foreign "op_broadcast"
        (ptr void @-> size_t @-> ptr int64_t @-> returning (ptr void))

    let broadcast_in_dim =
      foreign "op_broadcast_in_dim"
        (ptr void @-> size_t @-> ptr int64_t @-> size_t @-> ptr int64_t
        @-> returning (ptr void))

    (* Reduction operations *)
    let reduce =
      foreign "op_reduce"
        (ptr void @-> ptr void @-> ptr void @-> ptr int64_t @-> size_t
        @-> returning (ptr void))

    (* Other operations *)
    let select =
      foreign "op_select"
        (ptr void @-> ptr void @-> ptr void @-> returning (ptr void))

    let convert_element_type =
      foreign "op_convert_element_type"
        (ptr void @-> int @-> returning (ptr void))

    let pad =
      foreign "op_pad"
        (ptr void @-> ptr void @-> size_t @-> ptr int64_t @-> ptr int64_t
       @-> ptr int64_t
        @-> returning (ptr void))

    let reverse =
      foreign "op_reverse"
        (ptr void @-> size_t @-> ptr int64_t @-> returning (ptr void))

    let slice =
      foreign "op_slice"
        (ptr void @-> size_t @-> ptr int64_t @-> ptr int64_t @-> ptr int64_t
        @-> returning (ptr void))

    let slice_in_dim =
      foreign "op_slice_in_dim"
        (ptr void @-> int64_t @-> int64_t @-> int64_t @-> int64_t
        @-> returning (ptr void))

    let concat_in_dim =
      foreign "op_concat_in_dim"
        (ptr void
        @-> ptr (ptr void)
        @-> size_t @-> int64_t
        @-> returning (ptr void))

    (* Gather/Scatter operations *)
    let gather =
      foreign "op_gather"
        (ptr void @-> ptr void @-> ptr int64_t @-> size_t @-> ptr int64_t
       @-> size_t @-> ptr int64_t @-> size_t @-> ptr int64_t @-> ptr int64_t
       @-> size_t
        @-> returning (ptr void))

    let scatter =
      foreign "op_scatter"
        (ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr int64_t
       @-> size_t @-> ptr int64_t @-> size_t @-> ptr int64_t @-> size_t
       @-> int64_t
        @-> returning (ptr void))

    (* Convolution *)
    let conv_general_dilated =
      foreign "op_conv_general_dilated"
        (ptr void @-> ptr void @-> ptr int64_t @-> size_t @-> ptr int64_t
       @-> ptr int64_t @-> size_t @-> ptr int64_t @-> size_t @-> ptr int64_t
       @-> size_t @-> ptr int64_t @-> ptr int64_t @-> ptr int64_t @-> size_t
       @-> ptr int64_t @-> ptr int64_t @-> ptr int64_t @-> size_t
       @-> ptr int64_t @-> ptr int64_t @-> ptr int64_t @-> size_t @-> int64_t
       @-> int64_t
        @-> returning (ptr void))

    (* Shape query *)
    let rank = foreign "op_rank" (ptr void @-> returning int)
    let dims = foreign "op_dims" (ptr void @-> ptr int @-> returning void)
    let element_type = foreign "op_element_type" (ptr void @-> returning int)

    (* Constants *)
    let zero = foreign "op_zero" (ptr void @-> int @-> returning (ptr void))
    let one = foreign "op_one" (ptr void @-> int @-> returning (ptr void))

    let min_value =
      foreign "op_min_value" (ptr void @-> int @-> returning (ptr void))

    let max_value =
      foreign "op_max_value" (ptr void @-> int @-> returning (ptr void))

    let free = foreign "xla_op_free" (ptr void @-> returning void)
  end

  module Computation = struct
    let name = foreign "xla_computation_name" (ptr void @-> returning string)

    let build =
      foreign "build"
        (ptr void @-> ptr void @-> ptr (ptr void) @-> returning (ptr void))

    let free = foreign "xla_computation_free" (ptr void @-> returning void)
  end

  module PjRtDevice = struct
    let id = foreign "pjrt_device_id" (ptr void @-> returning int)

    let process_index =
      foreign "pjrt_device_process_index" (ptr void @-> returning int)

    let local_hardware_id =
      foreign "pjrt_device_local_hardware_id" (ptr void @-> returning int)

    let kind = foreign "pjrt_device_kind" (ptr void @-> returning string)

    let debug_string =
      foreign "pjrt_device_debug_string" (ptr void @-> returning string)

    let to_string =
      foreign "pjrt_device_to_string" (ptr void @-> returning string)
  end

  module PjRtClient = struct
    let cpu =
      foreign "pjrt_cpu_client_create" (ptr (ptr void) @-> returning (ptr void))

    let gpu =
      foreign "pjrt_gpu_client_create"
        (ptr (ptr void) @-> double @-> bool @-> returning (ptr void))

    let tpu =
      foreign "pjrt_tpu_client_create"
        (ptr (ptr void) @-> int @-> returning (ptr void))

    let device_count =
      foreign "pjrt_client_device_count" (ptr void @-> returning int)

    let addressable_device_count =
      foreign "pjrt_client_addressable_device_count" (ptr void @-> returning int)

    let devices =
      foreign "pjrt_client_devices"
        (ptr void @-> ptr (ptr void) @-> returning void)

    let addressable_devices =
      foreign "pjrt_client_addressable_devices"
        (ptr void @-> ptr (ptr void) @-> returning void)

    let platform_name =
      foreign "pjrt_client_platform_name" (ptr void @-> returning string)

    let platform_version =
      foreign "pjrt_client_platform_version" (ptr void @-> returning string)

    let free = foreign "pjrt_client_free" (ptr void @-> returning void)
  end

  module PjRtBuffer = struct
    let from_host_literal =
      foreign "pjrt_buffer_from_host_literal"
        (ptr void @-> ptr void @-> ptr void
        @-> ptr (ptr void)
        @-> returning (ptr void))

    let to_literal_sync =
      foreign "pjrt_buffer_to_literal_sync"
        (ptr void @-> ptr (ptr void) @-> returning (ptr void))

    let on_device_shape =
      foreign "pjrt_buffer_on_device_shape" (ptr void @-> returning (ptr void))

    let free = foreign "pjrt_buffer_free" (ptr void @-> returning void)
  end

  module PjRtLoadedExecutable = struct
    let compile =
      foreign "compile"
        (ptr void @-> ptr void @-> ptr (ptr void) @-> returning (ptr void))

    let execute =
      foreign "execute"
        (ptr void
        @-> ptr (ptr void)
        @-> int
        @-> ptr (ptr (ptr (ptr void)))
        @-> returning (ptr void))

    let free =
      foreign "pjrt_loaded_executable_free" (ptr void @-> returning void)
  end
end
