(* OCaml bindings for XLA *)
open Ctypes

(* Module alias for generated bindings *)
module C = Xla_c.C.Functions

(* Re-export Element_type *)
module Element_type = Element_type

let initialize () = ()
(* XLA initialization if needed *)

(* Helper to check status and raise exception if error *)
let check_status status =
  if not (is_null status) then (
    let msg = C.Status.error_message status in
    C.Status.free status;
    failwith msg)

module Client = struct
  type t = unit ptr

  let cpu () =
    let client_ptr = allocate (ptr void) null in
    let status = C.PjRtClient.cpu client_ptr in
    check_status status;
    !@client_ptr

  let gpu ?device_id:_ () =
    (* GPU support uses default parameters for now *)
    let client_ptr = allocate (ptr void) null in
    let status = C.PjRtClient.gpu client_ptr 0.9 true in
    check_status status;
    !@client_ptr

  let _device_count t = C.PjRtClient.device_count t
  let _platform_name t = C.PjRtClient.platform_name t
  let _free t = C.PjRtClient.free t
end

module Shape = struct
  type t = unit ptr

  let create ?layout:_ dims =
    let n = Array.length dims in
    let dims_ptr =
      CArray.of_list int64_t (Array.to_list (Array.map Int64.of_int dims))
    in
    (* For F32 type = 11 according to XLA element types *)
    C.Shape.make_shape_array 11 (Unsigned.Size_t.of_int n)
      (CArray.start dims_ptr)

  let dimensions t =
    let n = C.Shape.dimensions_size t in
    Array.init n (fun i -> Int64.to_int (C.Shape.dimensions t i))

  let rank t = C.Shape.dimensions_size t
  let element_count t = Array.fold_left ( * ) 1 (dimensions t)
  let _free t = C.Shape.free t
end

module Literal = struct
  type t = unit ptr

  let create_r0_f32 value = C.Literal.create_r0_f32 value

  let create_r1_f32 values =
    let n = Array.length values in
    let data = CArray.of_list float (Array.to_list values) in
    let size_bytes = n * 4 in
    (* 4 bytes per float32 *)
    let lit =
      C.Literal.create_from_shape_and_data 11 (* F32 *)
        (CArray.start (CArray.of_list int64_t [ Int64.of_int n ]))
        (Unsigned.Size_t.of_int 1)
        (to_voidp (CArray.start data))
        (Unsigned.Size_t.of_int size_bytes)
    in
    lit

  let create_r2_f32 values =
    let rows = Array.length values in
    let cols = if rows > 0 then Array.length values.(0) else 0 in
    let flat = Array.concat (Array.to_list values) in
    let data = CArray.of_list float (Array.to_list flat) in
    let size_bytes = rows * cols * 4 in
    (* 4 bytes per float32 *)
    let lit =
      C.Literal.create_from_shape_and_data 11 (* F32 *)
        (CArray.start
           (CArray.of_list int64_t [ Int64.of_int rows; Int64.of_int cols ]))
        (Unsigned.Size_t.of_int 2)
        (to_voidp (CArray.start data))
        (Unsigned.Size_t.of_int size_bytes)
    in
    lit

  let of_bigarray (type a b)
      (arr : (a, b, Bigarray.c_layout) Bigarray.Genarray.t) : t =
    let dims = Bigarray.Genarray.dims arr in
    let dtype_code =
      match Bigarray.Genarray.kind arr with
      | Bigarray.Float32 -> 11 (* F32 *)
      | Bigarray.Float64 -> 12 (* F64 *)
      | Bigarray.Int32 -> 5 (* S32 *)
      | Bigarray.Int64 -> 6 (* S64 *)
      | Bigarray.Int8_signed -> 2 (* S8 *)
      | Bigarray.Int8_unsigned -> 3 (* U8 *)
      | Bigarray.Int16_signed -> 4 (* S16 *)
      | Bigarray.Int16_unsigned -> 16 (* U16 *)
      | _ -> failwith "Unsupported bigarray kind for XLA"
    in
    let data_ptr = to_voidp (bigarray_start genarray arr) in
    let size_bytes =
      let elem_size =
        match Bigarray.Genarray.kind arr with
        | Bigarray.Float32 -> 4
        | Bigarray.Float64 -> 8
        | Bigarray.Int32 -> 4
        | Bigarray.Int64 -> 8
        | Bigarray.Int8_signed | Bigarray.Int8_unsigned -> 1
        | Bigarray.Int16_signed | Bigarray.Int16_unsigned -> 2
        | _ -> failwith "Unsupported bigarray kind"
      in
      Array.fold_left ( * ) elem_size dims
    in
    let dims_ptr =
      CArray.of_list int64_t (Array.to_list (Array.map Int64.of_int dims))
    in
    C.Literal.create_from_shape_and_data dtype_code (CArray.start dims_ptr)
      (Unsigned.Size_t.of_int (Array.length dims))
      data_ptr
      (Unsigned.Size_t.of_int size_bytes)

  let to_bigarray (type a b) (lit : t) (kind : (a, b) Bigarray.kind) :
      (a, b, Bigarray.c_layout) Bigarray.Genarray.t =
    let shape_ptr = allocate (ptr void) null in
    C.Literal.shape lit shape_ptr;
    let shape = !@shape_ptr in
    let dims = Shape.dimensions shape in
    let size_bytes = Int64.to_int (C.Literal.size_bytes lit) in

    (* Create bigarray with appropriate dimensions *)
    let arr = Bigarray.Genarray.create kind Bigarray.c_layout dims in

    (* Copy data from literal to bigarray *)
    let data_ptr = to_voidp (bigarray_start genarray arr) in
    C.Literal.copy_to lit data_ptr (Unsigned.Size_t.of_int size_bytes);

    arr

  let shape t =
    let shape_ptr = allocate (ptr void) null in
    C.Literal.shape t shape_ptr;
    !@shape_ptr

  let _free t = C.Literal.free t
end

module Computation = struct
  type t = unit ptr
  type executable = unit ptr

  let compile client computation =
    let exec_ptr = allocate (ptr void) null in
    let status = C.PjRtLoadedExecutable.compile client computation exec_ptr in
    check_status status;
    !@exec_ptr

  let execute executable literals =
    let n = List.length literals in
    let lit_array = CArray.of_list (ptr void) literals in
    let output_ptr =
      allocate (ptr (ptr (ptr void))) (from_voidp (ptr (ptr void)) null)
    in
    let status =
      C.PjRtLoadedExecutable.execute executable (CArray.start lit_array) n
        output_ptr
    in
    check_status status;
    (* The result is a 2D array: replicas x outputs per replica *)
    let outputs = !@output_ptr in
    (* Get first replica (index 0) *)
    let replica0_outputs = !@outputs in
    (* Collect all outputs from this replica *)
    let rec collect_outputs outputs_ptr acc idx =
      let buffer = !@(outputs_ptr +@ idx) in
      if is_null buffer then List.rev acc
      else
        (* Convert buffer to literal *)
        let lit_ptr = allocate (ptr void) null in
        let status = C.PjRtBuffer.to_literal_sync buffer lit_ptr in
        check_status status;
        collect_outputs outputs_ptr (!@lit_ptr :: acc) (idx + 1)
    in
    collect_outputs replica0_outputs [] 0

  let _free t = C.Computation.free t
end

module Builder = struct
  type t = unit ptr
  type op = unit ptr

  let create name = C.Builder.create name

  let parameter builder id shape name =
    let dims = Shape.dimensions shape in
    let rank = Array.length dims in
    let dims_ptr =
      CArray.of_list int64_t (Array.to_list (Array.map Int64.of_int dims))
    in
    C.Op.parameter builder (Int64.of_int id) 11 (* F32 *) rank
      (CArray.start dims_ptr) name

  let constant builder literal = C.Op.constant_literal builder literal
  let add _builder a b = C.Op.add a b
  let multiply _builder a b = C.Op.mul a b
  let subtract _builder a b = C.Op.sub a b
  let divide _builder a b = C.Op.div a b
  let remainder _builder a b = C.Op.rem a b
  let max _builder a b = C.Op.max a b
  let min _builder a b = C.Op.min a b
  let pow _builder a b = C.Op.pow a b
  let dot _builder a b = C.Op.dot a b
  let and_ _builder a b = C.Op.and_ a b
  let or_ _builder a b = C.Op.or_ a b
  let xor _builder a b = C.Op.xor a b
  let neg _builder x = C.Op.neg x
  let abs _builder x = C.Op.abs x
  let exp _builder x = C.Op.exp x
  let log _builder x = C.Op.log x
  let sqrt _builder x = C.Op.sqrt x
  let sin _builder x = C.Op.sin x
  let cos _builder x = C.Op.cos x
  let tanh _builder x = C.Op.tanh x
  let eq _builder a b = C.Op.eq a b
  let ne _builder a b = C.Op.ne a b
  let lt _builder a b = C.Op.lt a b
  let le _builder a b = C.Op.le a b
  let gt _builder a b = C.Op.gt a b
  let ge _builder a b = C.Op.ge a b
  let select _builder cond t f = C.Op.select cond t f

  let convert_element_type _builder x target_type =
    let type_int = Element_type.to_int target_type in
    C.Op.convert_element_type x type_int

  let pad _builder x padding_value ~low_padding ~high_padding ~interior_padding
      =
    let n = Array.length low_padding in
    let low_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int low_padding))
    in
    let high_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int high_padding))
    in
    let interior_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int interior_padding))
    in
    C.Op.pad x padding_value (Unsigned.Size_t.of_int n) (CArray.start low_ptr)
      (CArray.start high_ptr)
      (CArray.start interior_ptr)

  let reverse _builder x axes =
    let n = Array.length axes in
    let axes_ptr =
      CArray.of_list int64_t (Array.to_list (Array.map Int64.of_int axes))
    in
    C.Op.reverse x (Unsigned.Size_t.of_int n) (CArray.start axes_ptr)

  let slice _builder x ~start_indices ~limit_indices ~strides =
    let n = Array.length start_indices in
    let start_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int start_indices))
    in
    let limit_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int limit_indices))
    in
    let strides_ptr =
      CArray.of_list int64_t (Array.to_list (Array.map Int64.of_int strides))
    in
    C.Op.slice x (Unsigned.Size_t.of_int n) (CArray.start start_ptr)
      (CArray.start limit_ptr) (CArray.start strides_ptr)

  let concatenate _builder ops axis =
    let n = Array.length ops in
    let ops_ptr = CArray.of_list (ptr void) (Array.to_list ops) in
    C.Op.concat_in_dim (Obj.magic ()) (CArray.start ops_ptr)
      (Unsigned.Size_t.of_int n) (Int64.of_int axis)

  let reshape _builder x new_shape =
    let n = Array.length new_shape in
    let dims_ptr =
      CArray.of_list int64_t (Array.to_list (Array.map Int64.of_int new_shape))
    in
    C.Op.reshape x (Unsigned.Size_t.of_int n) (CArray.start dims_ptr)

  let transpose _builder x permutation =
    let n = Array.length permutation in
    let perm_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int permutation))
    in
    C.Op.transpose x (Unsigned.Size_t.of_int n) (CArray.start perm_ptr)

  let broadcast _builder x new_shape =
    let n = Array.length new_shape in
    let dims_ptr =
      CArray.of_list int64_t (Array.to_list (Array.map Int64.of_int new_shape))
    in
    C.Op.broadcast x (Unsigned.Size_t.of_int n) (CArray.start dims_ptr)

  let zero builder element_type =
    C.Op.zero builder (Element_type.to_int element_type)

  let min_value builder element_type =
    C.Op.min_value builder (Element_type.to_int element_type)

  let max_value builder element_type =
    C.Op.max_value builder (Element_type.to_int element_type)

  let op_element_type op = Element_type.of_int (C.Op.element_type op)

  let reduce _builder x ~init ~computation ~dims =
    let n = Array.length dims in
    let dims_ptr =
      CArray.of_list int64_t (Array.to_list (Array.map Int64.of_int dims))
    in
    C.Op.reduce x init computation (CArray.start dims_ptr)
      (Unsigned.Size_t.of_int n)

  let build builder root =
    let comp_ptr = allocate (ptr void) null in
    let status = C.Computation.build builder root comp_ptr in
    check_status status;
    !@comp_ptr

  (* Helper to create scalar computations for reductions *)
  let create_scalar_add_computation _builder _element_type =
    let temp_builder = create "add" in
    let scalar_shape = Shape.create [||] in
    let param0 = parameter temp_builder 0 scalar_shape "x" in
    let param1 = parameter temp_builder 1 scalar_shape "y" in
    let sum = add temp_builder param0 param1 in
    build temp_builder sum

  let create_scalar_max_computation _builder _element_type =
    let temp_builder = create "max" in
    let scalar_shape = Shape.create [||] in
    let param0 = parameter temp_builder 0 scalar_shape "x" in
    let param1 = parameter temp_builder 1 scalar_shape "y" in
    let maximum = max temp_builder param0 param1 in
    build temp_builder maximum

  let create_scalar_min_computation _builder _element_type =
    let temp_builder = create "min" in
    let scalar_shape = Shape.create [||] in
    let param0 = parameter temp_builder 0 scalar_shape "x" in
    let param1 = parameter temp_builder 1 scalar_shape "y" in
    let minimum = min temp_builder param0 param1 in
    build temp_builder minimum

  let reduce_sum builder x ~dims ~keep_dims =
    let element_type = op_element_type x in
    let init = zero builder element_type in
    let computation = create_scalar_add_computation builder element_type in
    let result = reduce builder x ~init ~computation ~dims in
    if keep_dims then (
      (* Get original shape and set reduced dimensions to 1 *)
      let rank = C.Op.rank x in
      let orig_dims = Array.make rank 0 in
      C.Op.dims x (CArray.start (CArray.of_list int (Array.to_list orig_dims)));
      let new_shape = Array.copy orig_dims in
      Array.iter (fun d -> new_shape.(d) <- 1) dims;
      reshape builder result new_shape)
    else result

  let reduce_max builder x ~dims ~keep_dims =
    let element_type = op_element_type x in
    let init = min_value builder element_type in
    let computation = create_scalar_max_computation builder element_type in
    let result = reduce builder x ~init ~computation ~dims in
    if keep_dims then (
      let rank = C.Op.rank x in
      let orig_dims = Array.make rank 0 in
      C.Op.dims x (CArray.start (CArray.of_list int (Array.to_list orig_dims)));
      let new_shape = Array.copy orig_dims in
      Array.iter (fun d -> new_shape.(d) <- 1) dims;
      reshape builder result new_shape)
    else result

  let reduce_min builder x ~dims ~keep_dims =
    let element_type = op_element_type x in
    let init = max_value builder element_type in
    let computation = create_scalar_min_computation builder element_type in
    let result = reduce builder x ~init ~computation ~dims in
    if keep_dims then (
      let rank = C.Op.rank x in
      let orig_dims = Array.make rank 0 in
      C.Op.dims x (CArray.start (CArray.of_list int (Array.to_list orig_dims)));
      let new_shape = Array.copy orig_dims in
      Array.iter (fun d -> new_shape.(d) <- 1) dims;
      reshape builder result new_shape)
    else result

  let gather _builder operand indices ~axis =
    (* Simplified gather that works along a single axis *)
    let operand_rank = C.Op.rank operand in
    let offset_dims =
      Array.init (operand_rank - 1) (fun i -> if i < axis then i else i + 1)
    in
    let collapsed_slice_dims = [| axis |] in
    let start_index_map = [| axis |] in
    let index_vector_dim = C.Op.rank indices in
    let slice_sizes = Array.make operand_rank 1 in
    let operand_dims = Array.make operand_rank 0 in
    C.Op.dims operand
      (CArray.start (CArray.of_list int (Array.to_list operand_dims)));
    Array.iteri
      (fun i size -> if i <> axis then slice_sizes.(i) <- size)
      operand_dims;

    let offset_dims_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int offset_dims))
    in
    let collapsed_dims_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int collapsed_slice_dims))
    in
    let start_index_map_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int start_index_map))
    in
    let index_vector_dim_ptr =
      CArray.of_list int64_t [ Int64.of_int index_vector_dim ]
    in
    let slice_sizes_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int slice_sizes))
    in

    C.Op.gather operand indices
      (CArray.start offset_dims_ptr)
      (Unsigned.Size_t.of_int (Array.length offset_dims))
      (CArray.start collapsed_dims_ptr)
      (Unsigned.Size_t.of_int (Array.length collapsed_slice_dims))
      (CArray.start start_index_map_ptr)
      (Unsigned.Size_t.of_int (Array.length start_index_map))
      (CArray.start index_vector_dim_ptr)
      (CArray.start slice_sizes_ptr)
      (Unsigned.Size_t.of_int (Array.length slice_sizes))

  let conv2d _builder input kernel ~strides ~padding ?(dilation = [| 1; 1 |]) ()
      =
    (* 2D convolution for NCHW format with dilation support *)
    let stride_h, stride_w =
      match strides with
      | [| h; w |] -> (h, w)
      | _ -> failwith "Expected 2D strides"
    in
    let dilation_h, dilation_w =
      match dilation with
      | [| h; w |] -> (h, w)
      | _ -> failwith "Expected 2D dilation"
    in
    let (pad_h_before, pad_h_after), (pad_w_before, pad_w_after) =
      match padding with
      | [| (hb, ha); (wb, wa) |] -> ((hb, ha), (wb, wa))
      | _ -> failwith "Expected 2D padding"
    in

    (* Set dimension numbers for NCHW format *)
    let input_batch_dim = 0L in
    let input_feature_dim = 1L in
    let input_spatial_dims = [| 2L; 3L |] in
    let kernel_input_feature_dim = 1L in
    let kernel_output_feature_dim = 0L in
    let kernel_spatial_dims = [| 2L; 3L |] in
    let output_batch_dim = 0L in
    let output_feature_dim = 1L in
    let output_spatial_dims = [| 2L; 3L |] in

    (* Create arrays for FFI *)
    let window_strides =
      CArray.of_list int64_t [ Int64.of_int stride_h; Int64.of_int stride_w ]
    in
    let padding_low =
      CArray.of_list int64_t
        [ Int64.of_int pad_h_before; Int64.of_int pad_w_before ]
    in
    let padding_high =
      CArray.of_list int64_t
        [ Int64.of_int pad_h_after; Int64.of_int pad_w_after ]
    in
    let lhs_dilation = CArray.of_list int64_t [ 1L; 1L ] in
    let rhs_dilation =
      CArray.of_list int64_t
        [ Int64.of_int dilation_h; Int64.of_int dilation_w ]
    in
    let input_spatial_dims_arr =
      CArray.of_list int64_t (Array.to_list input_spatial_dims)
    in
    let kernel_spatial_dims_arr =
      CArray.of_list int64_t (Array.to_list kernel_spatial_dims)
    in
    let output_spatial_dims_arr =
      CArray.of_list int64_t (Array.to_list output_spatial_dims)
    in

    C.Op.conv_general_dilated input kernel
      (CArray.start window_strides)
      (Unsigned.Size_t.of_int 2) (CArray.start padding_low)
      (CArray.start padding_high)
      (Unsigned.Size_t.of_int 2)
      (CArray.start lhs_dilation)
      (Unsigned.Size_t.of_int 2)
      (CArray.start rhs_dilation)
      (Unsigned.Size_t.of_int 2)
      (allocate int64_t input_batch_dim)
      (allocate int64_t input_feature_dim)
      (CArray.start input_spatial_dims_arr)
      (Unsigned.Size_t.of_int 2)
      (allocate int64_t kernel_input_feature_dim)
      (allocate int64_t kernel_output_feature_dim)
      (CArray.start kernel_spatial_dims_arr)
      (Unsigned.Size_t.of_int 2)
      (allocate int64_t output_batch_dim)
      (allocate int64_t output_feature_dim)
      (CArray.start output_spatial_dims_arr)
      (Unsigned.Size_t.of_int 2) 1L
      1L (* feature_group_count, batch_group_count *)

  let scatter _builder operand scatter_indices updates ~axis ~update_computation
      =
    (* Simplified scatter for single axis *)
    (* For a full implementation, we'd need to handle scatter dimension numbers properly *)
    let operand_rank = C.Op.rank operand in
    let update_window_dims =
      Array.init (operand_rank - 1) (fun i -> if i < axis then i else i + 1)
    in
    let inserted_window_dims = [| axis |] in
    let scatter_dims_to_operand_dims = [| axis |] in
    let index_vector_dim = C.Op.rank scatter_indices in

    let update_window_dims_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int update_window_dims))
    in
    let inserted_dims_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int inserted_window_dims))
    in
    let scatter_dims_ptr =
      CArray.of_list int64_t
        (Array.to_list (Array.map Int64.of_int scatter_dims_to_operand_dims))
    in

    C.Op.scatter operand scatter_indices updates update_computation
      (CArray.start update_window_dims_ptr)
      (Unsigned.Size_t.of_int (Array.length update_window_dims))
      (CArray.start inserted_dims_ptr)
      (Unsigned.Size_t.of_int (Array.length inserted_window_dims))
      (CArray.start scatter_dims_ptr)
      (Unsigned.Size_t.of_int (Array.length scatter_dims_to_operand_dims))
      (Int64.of_int index_vector_dim)

  let _free t = C.Builder.free t
end
