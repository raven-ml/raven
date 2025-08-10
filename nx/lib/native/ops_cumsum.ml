open Bigarray
open Nx_core
open Internal

(* Helper function to calculate offset from multi-dimensional index *)
let calculate_offset view_offset strides md_index =
  view_offset + Array.fold_left (+) 0 (Array.mapi (fun d idx -> idx * strides.(d)) md_index)

(* Helper function to convert linear index to multi-dimensional index, skipping axis *)
let linear_to_md_index iter shape axis md_index =
  let temp_iter = ref iter in
  for d = Array.length shape - 1 downto 0 do
    if d <> axis then (
      md_index.(d) <- !temp_iter mod shape.(d);
      temp_iter := !temp_iter / shape.(d)
    )
  done

(* Generic cumsum kernel using higher-order functions *)
let kernel_cumsum_generic (type a) 
    (input : (a, _) t) (output : (a, _) t) ~axis
    ~zero ~add =
  let input_buf = buffer input in
  let output_buf = buffer output in
  let input_view = view input in
  let output_view = view output in
  let shape = View.shape input_view in
  let strides = View.strides input_view in
  let out_strides = View.strides output_view in
  let ndim = Array.length shape in
  
  if ndim = 0 then
    (* Scalar case *)
    let input_offset = View.offset input_view in
    let output_offset = View.offset output_view in
    let value = Array1.unsafe_get input_buf input_offset in
    Array1.unsafe_set output_buf output_offset value
  else
    let axis_size = shape.(axis) in
    
    if axis_size = 0 then
      () (* Empty tensor *)
    else if axis_size = 1 then
      (* Single element - copy all elements *)
      let md_index = Array.make ndim 0 in
      let total_iterations = Array.fold_left ( * ) 1 shape in
      for iter = 0 to total_iterations - 1 do
        let temp_iter = ref iter in
        for d = ndim - 1 downto 0 do
          md_index.(d) <- !temp_iter mod shape.(d);
          temp_iter := !temp_iter / shape.(d)
        done;
        
        let input_offset = calculate_offset (View.offset input_view) strides md_index in
        let output_offset = calculate_offset (View.offset output_view) out_strides md_index in
        
        let value = Array1.unsafe_get input_buf input_offset in
        Array1.unsafe_set output_buf output_offset value
      done
    else
      (* General case with optimized memory access *)
      let md_index = Array.make ndim 0 in
      let total_iterations = Array.fold_left ( * ) 1 shape / axis_size in
      let axis_stride = strides.(axis) in
      let out_axis_stride = out_strides.(axis) in
      
      for iter = 0 to total_iterations - 1 do
        linear_to_md_index iter shape axis md_index;
        
        (* Calculate base offset once *)
        md_index.(axis) <- 0;
        let base_input_offset = calculate_offset (View.offset input_view) strides md_index in
        let base_output_offset = calculate_offset (View.offset output_view) out_strides md_index in
        
        (* Cumulative sum along axis *)
        let acc = ref zero in
        for i = 0 to axis_size - 1 do
          let input_offset = base_input_offset + i * axis_stride in
          let output_offset = base_output_offset + i * out_axis_stride in
          
          let value = Array1.unsafe_get input_buf input_offset in
          acc := add !acc value;
          Array1.unsafe_set output_buf output_offset !acc
        done
      done

(* Specialized kernels for each data type *)
let kernel_cumsum_int32 = kernel_cumsum_generic ~zero:0l ~add:Int32.add
let kernel_cumsum_int64 = kernel_cumsum_generic ~zero:0L ~add:Int64.add  
let kernel_cumsum_float32 = kernel_cumsum_generic ~zero:0.0 ~add:(+.)
let kernel_cumsum_float64 = kernel_cumsum_generic ~zero:0.0 ~add:(+.)

(* Kernel dispatcher *)
let cumsum_kernel : type a b. (a, b) t -> (a, b) t -> axis:int -> unit =
  fun input output ~axis ->
    match dtype input with
    | Int32 -> kernel_cumsum_int32 input output ~axis
    | Int64 -> kernel_cumsum_int64 input output ~axis  
    | Float32 -> kernel_cumsum_float32 input output ~axis
    | Float64 -> kernel_cumsum_float64 input output ~axis
    | _ -> failwith "cumsum: data type not yet supported"
