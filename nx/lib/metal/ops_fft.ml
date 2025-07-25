open Nx_core
open Internal

(* Helper to determine FFT output shape *)
let get_fft_output_shape input_shape axes s =
  let ndim = Array.length input_shape in
  match s with
  | None -> Array.copy input_shape
  | Some sizes ->
      let out_shape = Array.copy input_shape in
      Array.iteri
        (fun i axis ->
          let axis = if axis < 0 then ndim + axis else axis in
          out_shape.(axis) <- sizes.(i))
        axes;
      out_shape

(* Helper to check if a number is a power of 2 *)
let is_power_of_two n = n > 0 && n land (n - 1) = 0

(* Get next power of 2 *)
let next_power_of_two n =
  let rec loop p = if p >= n then p else loop (p * 2) in
  loop 1

(* Get kernel from cache *)
let get_kernel ctx kernel_name =
  try
    let functions = Hashtbl.find ctx.kernels "fft" in
    Hashtbl.find functions kernel_name
  with Not_found ->
    (* Create the function if not cached *)
    let func = Metal.Library.new_function_with_name ctx.library kernel_name in
    let functions =
      try Hashtbl.find ctx.kernels "fft"
      with Not_found ->
        let h = Hashtbl.create 4 in
        Hashtbl.add ctx.kernels "fft" h;
        h
    in
    Hashtbl.add functions kernel_name func;
    func

(* Create buffer helper *)
let create_buffer ctx dtype view =
  let size_elements = View.numel view in
  let elem_size = sizeof_dtype dtype in
  let size_bytes = size_elements * elem_size in
  let buffer = Buffer_pool.allocate ctx.pool size_bytes in
  { context = ctx; dtype; buffer = { buffer; size_bytes }; view }

(* Perform 1D FFT along a single axis *)
let fft_1d_along_axis ctx input output axis inverse =
  let shape = View.shape input.view in
  let ndim = Array.length shape in
  let axis = if axis < 0 then ndim + axis else axis in
  let fft_size = shape.(axis) in

  (* Get strides and offset *)
  let in_strides = View.strides input.view in
  let stride = in_strides.(axis) in
  let offset = View.offset input.view in

  (* Compute number of FFTs to perform *)
  let num_ffts = ref 1 in
  for i = 0 to ndim - 1 do
    if i <> axis then num_ffts := !num_ffts * shape.(i)
  done;

  with_command_buffer ctx (fun command_buffer ->
      let encoder = Metal.ComputeCommandEncoder.on_buffer command_buffer in

      (* Get FFT kernel *)
      let kernel_name = "fft_1d_complex64" in
      let kernel = get_kernel ctx kernel_name in
      let pipeline = Kernels.create_compute_pipeline ctx.device kernel in
      Metal.ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;

      (* Debug print *)
      Printf.printf
        "FFT dispatch: size=%d, stride=%d, offset=%d, num_ffts=%d, inverse=%b\n"
        fft_size stride offset !num_ffts inverse;

      (* Set buffers *)
      Metal.ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        output.buffer.buffer;

      (* Set size *)
      let size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int fft_size))
      in
      Metal.ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp size_val)
        ~length:4 ~index:1;

      (* Set stride *)
      let stride_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int stride))
      in
      Metal.ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp stride_val)
        ~length:4 ~index:2;

      (* Set offset *)
      let offset_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int offset))
      in
      Metal.ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp offset_val)
        ~length:4 ~index:3;

      (* Set inverse flag *)
      let inverse_val = Ctypes.(allocate bool inverse) in
      Metal.ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp inverse_val)
        ~length:1 ~index:4;

      (* Dispatch threads *)
      let threads_per_group = min fft_size 256 in
      let grid_size = { Metal.Size.width = !num_ffts; height = 1; depth = 1 } in
      let group_size =
        { Metal.Size.width = threads_per_group; height = 1; depth = 1 }
      in

      Metal.ComputeCommandEncoder.dispatch_threadgroups encoder
        ~threadgroups_per_grid:grid_size ~threads_per_threadgroup:group_size;

      Metal.ComputeCommandEncoder.end_encoding encoder)

(* Main FFT operation *)
let op_fft (type b) ctx (input : (Complex.t, b) t) ~axes ~s : (Complex.t, b) t =
  match input.dtype with
  | Complex64 ->
      let input_shape = View.shape input.view in
      let output_shape = get_fft_output_shape input_shape axes s in

      (* Check all FFT sizes are powers of 2 *)
      Array.iter
        (fun axis ->
          let axis =
            if axis < 0 then Array.length input_shape + axis else axis
          in
          let size =
            match s with
            | None -> input_shape.(axis)
            | Some sizes ->
                (* Find index of axis in axes array *)
                let rec find_index i =
                  if i >= Array.length axes then input_shape.(axis)
                  else
                    let a = axes.(i) in
                    if a = axis || (a < 0 && Array.length input_shape + a = axis)
                    then sizes.(i)
                    else find_index (i + 1)
                in
                find_index 0
          in
          if not (is_power_of_two size) then
            failwith
              (Printf.sprintf
                 "op_fft: FFT size %d on axis %d must be a power of 2" size axis))
        axes;

      (* Create output buffer *)
      let output_view = View.create output_shape in
      let output = create_buffer ctx input.dtype output_view in

      (* Copy input to output first *)
      Ops_movement.copy ctx input output;

      (* Perform FFT along each axis *)
      Array.iter
        (fun axis -> fft_1d_along_axis ctx output output axis false)
        axes;

      output
  | Complex32 ->
      failwith "op_fft: Complex32 FFT not yet implemented in Metal backend"

(* Inverse FFT operation *)
let op_ifft (type b) ctx (input : (Complex.t, b) t) ~axes ~s : (Complex.t, b) t
    =
  match input.dtype with
  | Complex64 ->
      let input_shape = View.shape input.view in
      let output_shape = get_fft_output_shape input_shape axes s in

      (* Check all FFT sizes are powers of 2 *)
      Array.iter
        (fun axis ->
          let axis =
            if axis < 0 then Array.length input_shape + axis else axis
          in
          let size =
            match s with
            | None -> input_shape.(axis)
            | Some sizes ->
                (* Find index of axis in axes array *)
                let rec find_index i =
                  if i >= Array.length axes then input_shape.(axis)
                  else
                    let a = axes.(i) in
                    if a = axis || (a < 0 && Array.length input_shape + a = axis)
                    then sizes.(i)
                    else find_index (i + 1)
                in
                find_index 0
          in
          if not (is_power_of_two size) then
            failwith
              (Printf.sprintf
                 "op_ifft: FFT size %d on axis %d must be a power of 2" size
                 axis))
        axes;

      (* Create output buffer *)
      let output_view = View.create output_shape in
      let output = create_buffer ctx input.dtype output_view in

      (* Copy input to output first *)
      Ops_movement.copy ctx input output;

      (* Perform inverse FFT along each axis *)
      Array.iter
        (fun axis -> fft_1d_along_axis ctx output output axis true)
        axes;

      output
  | Complex32 ->
      failwith "op_ifft: Complex32 IFFT not yet implemented in Metal backend"

(* Real FFT - not yet implemented *)
let op_rfft ctx _ ~axes:_ ~s:_ =
  failwith "op_rfft: Real FFT not yet implemented in Metal backend"

(* Inverse real FFT - not yet implemented *)
let op_irfft ctx _ ~axes:_ ~s:_ =
  failwith "op_irfft: Inverse real FFT not yet implemented in Metal backend"
