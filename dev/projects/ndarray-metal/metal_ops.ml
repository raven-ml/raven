open Metal_context
open Internal

let copy _context _t = failwith "todo"
let blit _context _src _dst = failwith "todo"
let fill _context _value _t = failwith "todo"

let run_compute ~ctx ~kernel_name ~buffers ~grid_size ~thread_group_size =
  let pipeline_state = get_or_create_pipeline ctx kernel_name in
  let command_buffer = Metal.create_command_buffer ctx.command_queue in
  let encoder = Metal.create_compute_command_encoder command_buffer in
  Metal.set_compute_pipeline_state encoder pipeline_state;
  List.iteri
    (fun idx buf -> Metal.set_buffer encoder buf ~index:idx ~offset:0L)
    buffers;
  Metal.dispatch_thread_groups encoder ~grid_size ~thread_group_size;
  Metal.end_encoding encoder;
  Metal.commit command_buffer;
  Metal.wait_until_completed command_buffer

let prepare_general_binary_buffers ctx out_shape a_shape a_strides b_shape
    b_strides =
  let ndim = Array.length out_shape in
  let buffer_ndim = create_int32_buffer ctx [| ndim |] in
  let buffer_out_shape = create_int32_buffer ctx out_shape in

  let a_strides_broadcasted =
    compute_broadcasted_strides a_shape a_strides out_shape
  in
  let buffer_a_strides = create_int64_buffer ctx a_strides_broadcasted in

  let b_strides_broadcasted =
    compute_broadcasted_strides b_shape b_strides out_shape
  in
  let buffer_b_strides = create_int64_buffer ctx b_strides_broadcasted in

  (buffer_ndim, buffer_out_shape, buffer_a_strides, buffer_b_strides)

type binary_op =
  | Add
  | Sub
  | Mul
  | Div
  | Pow
  | Remainder
  | Maximum
  | Minimum
  | Equal
  | Greater
  | GreaterEqual
  | Less
  | LessEqual

let binary_op_to_prefix = function
  | Add -> "tensor_add_"
  | Sub -> "tensor_sub_"
  | Mul -> "tensor_mul_"
  | Div -> "tensor_div_"
  | Pow -> "tensor_pow_"
  | Remainder -> "tensor_remainder_"
  | Maximum -> "tensor_max_"
  | Minimum -> "tensor_min_"
  | Equal -> "tensor_equal_"
  | Greater -> "tensor_greater_"
  | GreaterEqual -> "tensor_greater_equal_"
  | Less -> "tensor_less_"
  | LessEqual -> "tensor_less_equal_"

let binary_op ~ctx ~op ~a_buf ~a_shape ~a_strides ~b_buf ~b_shape ~b_strides
    ~out_buf ~out_shape ~dtype =
  let total_elements = Array.fold_left ( * ) 1 out_shape in
  if total_elements = 0 then ()
  else
    let a_strides_broadcast =
      compute_broadcasted_strides a_shape a_strides out_shape
    in
    let b_strides_broadcast =
      compute_broadcasted_strides b_shape b_strides out_shape
    in
    let is_contig_a = is_contiguous out_shape a_strides_broadcast in
    let is_contig_b = is_contiguous out_shape b_strides_broadcast in
    let use_contig_kernel = is_contig_a && is_contig_b in

    let buffer_total_elements = create_int32_buffer ctx [| total_elements |] in
    let suffix = metal_dtype_to_suffix dtype in
    let kernel_name =
      let prefix = binary_op_to_prefix op in
      if use_contig_kernel then prefix ^ suffix ^ "_contiguous"
      else prefix ^ suffix ^ "_general"
    in

    let max_threads_per_group = 256L in
    let total_elements_64 = Int64.of_int total_elements in
    let thread_group_size_x =
      Int64.min max_threads_per_group total_elements_64
    in
    let grid_size_x =
      let ( + ), ( - ), ( / ) = (Int64.add, Int64.sub, Int64.div) in
      (total_elements_64 + thread_group_size_x - 1L) / thread_group_size_x
    in
    let grid_size : Metal.grid_size =
      { Metal.x = grid_size_x; y = 1L; z = 1L }
    in
    let thread_group_size = { Metal.x = thread_group_size_x; y = 1L; z = 1L } in

    let buffers =
      if use_contig_kernel then [ a_buf; b_buf; buffer_total_elements; out_buf ]
      else
        let ndim = Array.length out_shape in
        if ndim > 8 then
          failwith "Number of dimensions exceeds maximum supported (8)";
        let buffer_ndim = create_int32_buffer ctx [| ndim |] in
        let buffer_out_shape_param = create_int32_buffer ctx out_shape in
        let buffer_a_strides_param =
          create_int64_buffer ctx a_strides_broadcast
        in
        let buffer_b_strides_param =
          create_int64_buffer ctx b_strides_broadcast
        in
        [
          a_buf;
          b_buf;
          buffer_total_elements;
          out_buf;
          buffer_ndim;
          buffer_out_shape_param;
          buffer_a_strides_param;
          buffer_b_strides_param;
        ]
    in
    run_compute ~ctx ~kernel_name ~buffers ~grid_size ~thread_group_size

let binary_op_impl : type a b.
    Metal_context.t -> binary_op -> (a, b) t -> (a, b) t -> (a, b) t -> unit =
 fun context op a b out ->
  let a_dtype = Ndarray_core.dtype a.descriptor in
  let b_dtype = Ndarray_core.dtype b.descriptor in
  if a_dtype <> b_dtype then failwith "Dtype mismatch in binary operation";
  let metal_dtype = ndarray_dtype_to_metal_dtype a_dtype in
  let a_shape = Ndarray_core.shape a.descriptor in
  let b_shape = Ndarray_core.shape b.descriptor in
  let a_strides = Ndarray_core.strides a.descriptor in
  let b_strides = Ndarray_core.strides b.descriptor in
  let out_shape = Ndarray_core.shape (descriptor out) in
  let out_buf = out.buffer in
  binary_op ~ctx:context ~op ~a_buf:a.buffer ~a_shape ~a_strides ~b_buf:b.buffer
    ~b_shape ~b_strides ~out_buf ~out_shape ~dtype:metal_dtype

let comparison_op_impl : type a b.
    Metal_context.t ->
    binary_op ->
    (a, b) t ->
    (a, b) t ->
    (int, Ndarray_core.uint8_elt) t ->
    unit =
 fun context op a b out ->
  let a_dtype = Ndarray_core.dtype a.descriptor in
  let b_dtype = Ndarray_core.dtype b.descriptor in
  if a_dtype <> b_dtype then failwith "Dtype mismatch in binary operation";
  let metal_dtype = ndarray_dtype_to_metal_dtype a_dtype in
  let a_shape = Ndarray_core.shape a.descriptor in
  let b_shape = Ndarray_core.shape b.descriptor in
  let a_strides = Ndarray_core.strides a.descriptor in
  let b_strides = Ndarray_core.strides b.descriptor in
  let out_shape = Ndarray_core.shape (descriptor out) in
  let out_buf = out.buffer in
  binary_op ~ctx:context ~op ~a_buf:a.buffer ~a_shape ~a_strides ~b_buf:b.buffer
    ~b_shape ~b_strides ~out_buf ~out_shape ~dtype:metal_dtype

type unary_op =
  | Neg
  | Abs
  | Sign
  | Sqrt
  | Exp
  | Log
  | Sin
  | Cos
  | Tan
  | Asin
  | Acos
  | Atan
  | Sinh
  | Cosh
  | Tanh
  | Asinh
  | Acosh
  | Atanh

let unary_op_to_prefix = function
  | Neg -> "tensor_neg_"
  | Abs -> "tensor_abs_"
  | Sign -> "tensor_sign_"
  | Sqrt -> "tensor_sqrt_"
  | Exp -> "tensor_exp_"
  | Log -> "tensor_log_"
  | Sin -> "tensor_sin_"
  | Cos -> "tensor_cos_"
  | Tan -> "tensor_tan_"
  | Asin -> "tensor_asin_"
  | Acos -> "tensor_acos_"
  | Atan -> "tensor_atan_"
  | Sinh -> "tensor_sinh_"
  | Cosh -> "tensor_cosh_"
  | Tanh -> "tensor_tanh_"
  | Asinh -> "tensor_asinh_"
  | Acosh -> "tensor_acosh_"
  | Atanh -> "tensor_atanh_"

let unary_op ~ctx ~op ~a_buf ~a_shape ~a_strides ~out_buf ~out_shape ~dtype =
  let total_elements = Array.fold_left ( * ) 1 a_shape in
  if total_elements = 0 then ()
  else if a_shape <> out_shape then
    failwith "Unary op input and output shapes must match";

  let is_contig = is_contiguous a_shape a_strides in
  let buffer_total_elements = create_int32_buffer ctx [| total_elements |] in
  let suffix = metal_dtype_to_suffix dtype in
  let kernel_name =
    let prefix = unary_op_to_prefix op in
    if is_contig then prefix ^ suffix ^ "_contiguous"
    else prefix ^ suffix ^ "_general"
  in

  let total_elements_64 = Int64.of_int total_elements in
  let max_threads_per_group = 256L in
  let thread_group_size_x = Int64.min max_threads_per_group total_elements_64 in
  let grid_size_x =
    let ( + ), ( - ), ( / ) = (Int64.add, Int64.sub, Int64.div) in
    (total_elements_64 + thread_group_size_x - 1L) / thread_group_size_x
  in
  let grid_size : Metal.grid_size = { Metal.x = grid_size_x; y = 1L; z = 1L } in
  let thread_group_size = { Metal.x = thread_group_size_x; y = 1L; z = 1L } in

  let buffers =
    if is_contig then [ a_buf; buffer_total_elements; out_buf ]
    else
      let ndim = Array.length a_shape in
      if ndim > 8 then
        failwith "Unary op: Number of dimensions exceeds maximum supported (8)";
      let buffer_ndim = create_int32_buffer ctx [| ndim |] in
      let buffer_shape = create_int32_buffer ctx a_shape in
      let buffer_a_strides = create_int64_buffer ctx a_strides in
      [
        a_buf;
        buffer_total_elements;
        out_buf;
        buffer_ndim;
        buffer_shape;
        buffer_a_strides;
      ]
  in
  run_compute ~ctx ~kernel_name ~buffers ~grid_size ~thread_group_size

let unary_op_impl : type a b.
    Metal_context.t -> unary_op -> (a, b) t -> (a, b) t -> unit =
 fun context op a out ->
  let a_dtype = Ndarray_core.dtype a.descriptor in
  let metal_dtype = ndarray_dtype_to_metal_dtype a_dtype in
  let a_shape = Ndarray_core.shape a.descriptor in
  let a_strides = Ndarray_core.strides a.descriptor in
  let out_buf = out.buffer in
  unary_op ~ctx:context ~op ~a_buf:a.buffer ~a_shape ~a_strides ~out_buf
    ~out_shape:a_shape ~dtype:metal_dtype

type reduction_op = Sum | Prod | Max | Min

let reduction_op_to_prefix = function
  | Sum -> "sum"
  | Prod -> "prod"
  | Max -> "max"
  | Min -> "min"

let calculate_reduction_params input_shape ax_list =
  let n_in = Array.length input_shape in
  let axes =
    let arr = Array.of_list ax_list in
    Array.iteri (fun i x -> if x < 0 then arr.(i) <- x + n_in) arr;
    let unique_sorted_axes =
      arr |> Array.to_list |> List.sort_uniq compare
      |> List.filter (fun x ->
             if x < 0 || x >= n_in then
               failwith (Printf.sprintf "Invalid axis %d for ndim %d" x n_in)
             else true)
      |> Array.of_list
    in
    unique_sorted_axes
  in
  let reduced_axis_flags = Array.make n_in false in
  Array.iter (fun ax -> reduced_axis_flags.(ax) <- true) axes;
  (axes, reduced_axis_flags)

let create_zero_buffer ctx dtype =
  match dtype with
  | Float32 -> create_int32_buffer ctx [| Int.zero |]
  | Int32 -> create_int32_buffer ctx [| Int.zero |]

let reduce ~ctx ~op ~axes ~keepdims ~input_buf ~input_shape ~input_strides
    ~dtype ~out_buf ~out_shape () =
  let _ = (keepdims, op, axes, input_strides) in
  let n_input_dims = Array.length input_shape in
  let n_output_dims = Array.length out_shape in
  let element_size = metal_dtype_size_in_bytes dtype |> Int64.of_int in
  let total_input_elements = max 1 (Array.fold_left ( * ) 1 input_shape) in

  if total_input_elements = 0 then (
    let identity_src_buf = create_zero_buffer ctx dtype in
    let command_buffer = Metal.create_command_buffer ctx.command_queue in
    let blit_encoder = Metal.create_blit_command_encoder command_buffer in
    Metal.copy_from_buffer_to_buffer blit_encoder ~src_buffer:identity_src_buf
      ~src_offset:0L ~dst_buffer:out_buf ~dst_offset:0L ~size:element_size;
    Metal.end_blit_encoding blit_encoder;
    Metal.commit command_buffer;
    Metal.wait_until_completed command_buffer)
  else if n_input_dims = n_output_dims && input_shape = out_shape then (
    if input_buf != out_buf then (
      let command_buffer = Metal.create_command_buffer ctx.command_queue in
      let blit_encoder = Metal.create_blit_command_encoder command_buffer in
      Metal.copy_from_buffer_to_buffer blit_encoder ~src_buffer:input_buf
        ~src_offset:0L ~dst_buffer:out_buf ~dst_offset:0L
        ~size:(Int64.mul element_size (Int64.of_int total_input_elements));
      Metal.end_blit_encoding blit_encoder;
      Metal.commit command_buffer;
      Metal.wait_until_completed command_buffer)
    else
      let suffix = metal_dtype_to_suffix dtype in

      let kernel_name = "sum_reduction_" ^ suffix in
      let thread_group_size_x = 256 in
      let pipeline_state = get_or_create_pipeline ctx kernel_name in
      let device_max_threads =
        Int64.to_int
          (Metal.get_pipeline_state_max_total_threads_per_threadgroup
             pipeline_state)
      in
      let max_allowed_groups = 1024 in
      let max_thread_groups =
        min max_allowed_groups (device_max_threads / thread_group_size_x)
      in
      let max_thread_groups = max 1 max_thread_groups in

      let inter_buffer_size =
        Int64.mul (Int64.of_int max_thread_groups) element_size
      in
      let buffer_intermediate1 =
        Metal.create_buffer ctx.device inter_buffer_size
          [ Metal.Storage_Mode_Shared ]
      in
      let buffer_intermediate2 =
        Metal.create_buffer ctx.device inter_buffer_size
          [ Metal.Storage_Mode_Shared ]
      in

      let current_input_buf = ref input_buf in
      let current_n = ref total_input_elements in
      let is_first_pass = ref true in

      while !current_n > 1 do
        let num_thread_groups =
          min max_thread_groups
            ((!current_n + thread_group_size_x - 1) / thread_group_size_x)
        in
        let num_thread_groups = max 1 num_thread_groups in
        let buffer_n = create_int32_buffer ctx [| !current_n |] in
        let buffer_num_tg = create_int32_buffer ctx [| num_thread_groups |] in
        let output_buf_pass =
          if !is_first_pass || !current_input_buf == buffer_intermediate2 then
            buffer_intermediate1
          else buffer_intermediate2
        in
        let grid_size : Metal.grid_size =
          { Metal.x = Int64.of_int num_thread_groups; y = 1L; z = 1L }
        in
        let thread_group_size =
          { Metal.x = Int64.of_int thread_group_size_x; y = 1L; z = 1L }
        in
        let buffers =
          [ !current_input_buf; output_buf_pass; buffer_n; buffer_num_tg ]
        in

        run_compute ~ctx ~kernel_name ~buffers ~grid_size ~thread_group_size;

        current_input_buf := output_buf_pass;
        current_n := num_thread_groups;
        is_first_pass := false
      done;

      let command_buffer = Metal.create_command_buffer ctx.command_queue in
      let blit_encoder = Metal.create_blit_command_encoder command_buffer in
      Metal.copy_from_buffer_to_buffer blit_encoder
        ~src_buffer:!current_input_buf ~src_offset:0L ~dst_buffer:out_buf
        ~dst_offset:0L ~size:element_size;
      Metal.end_blit_encoding blit_encoder;
      Metal.commit command_buffer;
      Metal.wait_until_completed command_buffer)

let sum : type a b.
    Metal_context.t ->
    axes:int array ->
    keepdims:bool ->
    (a, b) t ->
    (a, b) t ->
    unit =
 fun context ~axes ~keepdims arr out ->
  let ctx = context in
  let input_dtype = Ndarray_core.dtype arr.descriptor in
  let metal_dtype = ndarray_dtype_to_metal_dtype input_dtype in
  let input_shape = Ndarray_core.shape arr.descriptor in
  let input_strides = Ndarray_core.strides arr.descriptor in

  let out_shape = [| 0 |] in
  let out_buf = out.buffer in

  let axes_list = Array.to_list axes in
  reduce ~ctx ~axes:axes_list ~op:Sum ~keepdims ~input_buf:arr.buffer
    ~input_shape ~input_strides ~dtype:metal_dtype ~out_buf ~out_shape ()

let matmul ~ctx ~a_buf ~a_shape ~a_strides ~b_buf ~b_shape ~b_strides ~out_buf
    ~out_shape ~dtype =
  let m = a_shape.(0) in
  let k = a_shape.(1) in
  let n = b_shape.(1) in
  if m = 0 || k = 0 || n = 0 then ()
  else
    let is_contig_a = is_contiguous a_shape a_strides in
    let is_contig_b = is_contiguous b_shape b_strides in
    let can_use_tiled_contiguous = is_contig_a && is_contig_b in

    let buffer_m = create_int32_buffer ctx [| m |] in
    let buffer_n = create_int32_buffer ctx [| n |] in
    let buffer_k = create_int32_buffer ctx [| k |] in

    let suffix = metal_dtype_to_suffix dtype in
    let kernel_name, buffers, grid_size, thread_group_size =
      if can_use_tiled_contiguous then
        let kernel_name = "matmul_" ^ suffix ^ "_tiled_contiguous" in
        let tile_dim = 16L in
        let thread_group_size = { Metal.x = tile_dim; y = tile_dim; z = 1L } in
        let ( + ), ( - ), ( / ) = (Int64.add, Int64.sub, Int64.div) in
        let m_64 = Int64.of_int m in
        let n_64 = Int64.of_int n in
        let grid_size_x = (n_64 + tile_dim - 1L) / tile_dim in
        let grid_size_y = (m_64 + tile_dim - 1L) / tile_dim in
        let grid_size : Metal.grid_size =
          { Metal.x = grid_size_x; y = grid_size_y; z = 1L }
        in
        let buffers = [ a_buf; b_buf; out_buf; buffer_m; buffer_n; buffer_k ] in
        (kernel_name, buffers, grid_size, thread_group_size)
      else
        let kernel_name = "matmul_" ^ suffix in
        let buffer_a_strides = create_int64_buffer ctx a_strides in
        let buffer_b_strides = create_int64_buffer ctx b_strides in
        let buffer_out_strides =
          create_int64_buffer ctx (Ndarray_core.compute_c_strides out_shape)
        in

        let tg_dim_x = 16L in
        let tg_dim_y = 16L in
        let thread_group_size = { Metal.x = tg_dim_x; y = tg_dim_y; z = 1L } in
        let ( + ), ( - ), ( / ) = (Int64.add, Int64.sub, Int64.div) in
        let m_64 = Int64.of_int m in
        let n_64 = Int64.of_int n in
        let grid_size_x = (n_64 + tg_dim_x - 1L) / tg_dim_x in
        let grid_size_y = (m_64 + tg_dim_y - 1L) / tg_dim_y in
        let grid_size : Metal.grid_size =
          { Metal.x = grid_size_x; y = grid_size_y; z = 1L }
        in
        let buffers =
          [
            a_buf;
            b_buf;
            out_buf;
            buffer_m;
            buffer_n;
            buffer_k;
            buffer_a_strides;
            buffer_b_strides;
            buffer_out_strides;
          ]
        in
        (kernel_name, buffers, grid_size, thread_group_size)
    in
    run_compute ~ctx ~kernel_name ~buffers ~grid_size ~thread_group_size

let matmul : type a b.
    Metal_context.t -> (a, b) t -> (a, b) t -> (a, b) t -> unit =
 fun context a b out ->
  let a_dtype = Ndarray_core.dtype a.descriptor in
  let b_dtype = Ndarray_core.dtype b.descriptor in
  if a_dtype <> b_dtype then failwith "Dtype mismatch in matmul";
  let metal_dtype = ndarray_dtype_to_metal_dtype a_dtype in
  let a_shape = Ndarray_core.shape a.descriptor in
  let b_shape = Ndarray_core.shape b.descriptor in
  let a_strides = Ndarray_core.strides a.descriptor in
  let b_strides = Ndarray_core.strides b.descriptor in

  let m = a_shape.(0) in
  let k = a_shape.(1) in
  let n = b_shape.(1) in
  if k <> b_shape.(0) then
    failwith (Printf.sprintf "Incompatible shapes for matmul");

  let out_shape = [| m; n |] in
  let out_buf = out.buffer in
  matmul ~ctx:context ~a_buf:a.buffer ~a_shape ~a_strides ~b_buf:b.buffer
    ~b_shape ~b_strides ~out_buf ~out_shape ~dtype:metal_dtype
