let not_available () =
  failwith "Metal JIT backend is not available on this platform"

include (
  struct
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
  end :
    Rune_jit.Backend_intf.S)
