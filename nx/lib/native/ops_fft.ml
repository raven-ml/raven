open Bigarray_ext
module Dtype = Nx_core.Dtype
module Shape = Nx_core.Shape
module View = Nx_core.View
open Internal

let pi = 4.0 *. atan 1.0
let two_pi = 2.0 *. pi

(* Helper to compute twiddle factors *)
let twiddle_factor n k inverse =
  let sign = if inverse then 1.0 else -1.0 in
  let angle = sign *. two_pi *. float_of_int k /. float_of_int n in
  Complex.{ re = cos angle; im = sin angle }

(* Bit reversal permutation for FFT *)
let bit_reverse n x =
  let rec reverse_bits bits num_bits acc =
    if num_bits = 0 then acc
    else reverse_bits (bits lsr 1) (num_bits - 1) ((acc lsl 1) lor (bits land 1))
  in
  (* Compute number of bits using integer arithmetic to avoid floating-point
     errors *)
  let num_bits_for n =
    let rec loop b = if 1 lsl b >= n then b else loop (b + 1) in
    loop 0
  in
  let num_bits = num_bits_for n in
  reverse_bits x num_bits 0

(* Check if n is a power of 2 *)
let is_power_of_2 n = n > 0 && n land (n - 1) = 0

(* DFT for non-power-of-2 sizes *)
let dft_1d ~inverse ~scale data n stride offset =
  (* Create temporary array for result *)
  let temp =
    Array.init n (fun i -> Array1.unsafe_get data (offset + (i * stride)))
  in

  (* Compute DFT *)
  let sign = if inverse then 1.0 else -1.0 in
  for k = 0 to n - 1 do
    let sum = ref Complex.zero in
    for j = 0 to n - 1 do
      let angle = sign *. two_pi *. float_of_int (k * j) /. float_of_int n in
      let w = Complex.{ re = cos angle; im = sin angle } in
      sum := Complex.add !sum (Complex.mul temp.(j) w)
    done;
    (* Apply scaling if needed *)
    (if scale <> 1.0 then
       sum := Complex.{ re = !sum.re *. scale; im = !sum.im *. scale });
    Array1.unsafe_set data (offset + (k * stride)) !sum
  done

(* 1D FFT using Cooley-Tukey algorithm for power-of-2, DFT otherwise *)
let fft_1d ~inverse ~scale data n stride offset =
  if not (is_power_of_2 n) then dft_1d ~inverse ~scale data n stride offset
  else (
    (* Bit reversal permutation *)
    for i = 0 to n - 1 do
      let j = bit_reverse n i in
      if i < j then (
        let idx_i = offset + (i * stride) in
        let idx_j = offset + (j * stride) in
        let tmp = Array1.unsafe_get data idx_i in
        Array1.unsafe_set data idx_i (Array1.unsafe_get data idx_j);
        Array1.unsafe_set data idx_j tmp)
    done;

    (* FFT computation *)
    let rec fft_stage stage_size =
      if stage_size <= n then (
        let half_stage = stage_size / 2 in
        for k = 0 to (n / stage_size) - 1 do
          for j = 0 to half_stage - 1 do
            let twiddle = twiddle_factor stage_size j inverse in
            let idx1 = offset + (((k * stage_size) + j) * stride) in
            let idx2 =
              offset + (((k * stage_size) + j + half_stage) * stride)
            in
            let a = Array1.unsafe_get data idx1 in
            let b = Array1.unsafe_get data idx2 in
            let b_twiddle = Complex.mul b twiddle in
            Array1.unsafe_set data idx1 (Complex.add a b_twiddle);
            Array1.unsafe_set data idx2 (Complex.sub a b_twiddle)
          done
        done;
        fft_stage (stage_size * 2))
    in
    fft_stage 2;

    (* Apply scaling if requested *)
    if scale <> 1.0 then
      for i = 0 to n - 1 do
        let idx = offset + (i * stride) in
        let v = Array1.unsafe_get data idx in
        Array1.unsafe_set data idx
          Complex.{ re = v.re *. scale; im = v.im *. scale }
      done)

(* Validate axes and sizes parameters *)
let validate_axes_and_sizes ndim axes s =
  let module IntSet = Set.Make (Int) in
  let seen = ref IntSet.empty in
  Array.iter
    (fun ax ->
      let ax = if ax < 0 then ndim + ax else ax in
      if ax < 0 || ax >= ndim then
        failwith
          (Printf.sprintf "axis %d out of range for %d-dimensional array" ax
             ndim);
      if IntSet.mem ax !seen then
        failwith (Printf.sprintf "duplicate axis %d" ax);
      seen := IntSet.add ax !seen)
    axes;
  match s with
  | None -> ()
  | Some sizes ->
      if Array.length sizes <> Array.length axes then
        failwith "s parameter length must match axes length"

(* Helper to determine FFT output shape *)
let get_fft_output_shape input_shape axes s =
  let ndim = Array.length input_shape in
  validate_axes_and_sizes ndim axes s;
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

(* Multi-dimensional FFT kernel for complex types *)
let kernel_fft_multi (type b) ~inverse (input : (Complex.t, b) t)
    (output : (Complex.t, b) t) axes =
  let output_shape = Internal.shape output in
  let ndim = Array.length output_shape in

  (* Initialize output - either copy from input or zero-fill *)
  (if not (buffer input == buffer output) then
     (* Check if we need to copy data *)
     let input_shape = Internal.shape input in
     let output_shape = Internal.shape output in

     let same_shape =
       Array.length input_shape = Array.length output_shape
       && Array.for_all2 ( = ) input_shape output_shape
     in

     if same_shape then (
       (* Same shape, can directly copy if buffers are the same size *)
       let input_size = Array1.dim (buffer input) in
       let output_size = Array1.dim (buffer output) in
       if input_size = output_size then
         Array1.blit (buffer input) (buffer output)
       else
         (* Different buffer sizes despite same shape - copy element by element *)
         let size = Array.fold_left ( * ) 1 input_shape in
         for i = 0 to size - 1 do
           Array1.set (buffer output) i (Array1.get (buffer input) i)
         done)
     else (
       (* Different shapes - zero fill and copy what fits *)
       Array1.fill (buffer output) Complex.zero;

       (* Copy the overlapping region *)
       let rec copy_region indices dim =
         if dim = ndim then (
           (* Calculate linear indices *)
           let in_idx = ref 0 in
           let out_idx = ref 0 in
           let in_stride = ref 1 in
           let out_stride = ref 1 in

           for d = ndim - 1 downto 0 do
             in_idx := !in_idx + (indices.(d) * !in_stride);
             out_idx := !out_idx + (indices.(d) * !out_stride);
             in_stride := !in_stride * input_shape.(d);
             out_stride := !out_stride * output_shape.(d)
           done;

           let v = Array1.get (buffer input) !in_idx in
           Array1.set (buffer output) !out_idx v)
         else
           let limit = min input_shape.(dim) output_shape.(dim) in
           for i = 0 to limit - 1 do
             indices.(dim) <- i;
             copy_region indices (dim + 1)
           done
       in
       copy_region (Array.make ndim 0) 0));

  (* Perform FFT along each specified axis *)
  Array.iter
    (fun axis ->
      let axis = if axis < 0 then ndim + axis else axis in
      let n = output_shape.(axis) in

      (* Only perform FFT if n > 1 *)
      if n > 1 then
        if ndim = 1 then
          (* For 1D arrays, just do the FFT directly *)
          fft_1d ~inverse ~scale:1.0 (buffer output) n 1 0
        else
          (* For multi-dimensional arrays, iterate over all other dimensions *)
          let rec iter_dims dim indices =
            if dim = ndim then (
              (* Compute offset for this slice *)
              let offset = ref 0 in
              let stride_accum = ref 1 in
              for d = ndim - 1 downto 0 do
                offset := !offset + (indices.(d) * !stride_accum);
                stride_accum := !stride_accum * output_shape.(d)
              done;

              (* Calculate stride for the FFT axis *)
              let stride =
                let s = ref 1 in
                for d = axis + 1 to ndim - 1 do
                  s := !s * output_shape.(d)
                done;
                !s
              in

              (* Perform 1D FFT along the axis *)
              fft_1d ~inverse ~scale:1.0 (buffer output) n stride !offset)
            else if dim = axis then
              (* Skip the axis we're transforming along *)
              iter_dims (dim + 1) indices
            else
              (* Iterate over this dimension *)
              for i = 0 to output_shape.(dim) - 1 do
                indices.(dim) <- i;
                iter_dims (dim + 1) indices
              done
          in
          iter_dims 0 (Array.make ndim 0))
    axes;

  (* Do not apply scaling in backend - frontend handles normalization *)
  ()

(* Helper to determine RFFT output shape *)
let get_rfft_output_shape input_shape axes =
  let ndim = Array.length input_shape in
  let output_shape = Array.copy input_shape in
  (* Last axis in the transform is halved + 1 *)
  let last_axis_idx = Array.length axes - 1 in
  let last_axis = axes.(last_axis_idx) in
  let last_axis = if last_axis < 0 then ndim + last_axis else last_axis in
  output_shape.(last_axis) <- (input_shape.(last_axis) / 2) + 1;
  output_shape

(* Real to complex copy *)
let real_to_complex_copy (type b c) (real_input : (float, b) t)
    (complex_output : (Complex.t, c) t) =
  let real_buf = buffer real_input in
  let complex_buf = buffer complex_output in
  let n = Internal.size real_input in
  for i = 0 to n - 1 do
    let v = Array1.unsafe_get real_buf i in
    Array1.unsafe_set complex_buf i Complex.{ re = v; im = 0.0 }
  done

let kernel_rfft (type b c) context (input : (float, b) t)
    (output_dtype : (Complex.t, c) Dtype.t) axes s =
  let input_shape = Internal.shape input in
  let ndim = Array.length input_shape in

  (* Validate axes and sizes *)
  validate_axes_and_sizes ndim axes s;
  
  (* Require at least one axis *)
  if Array.length axes = 0 then invalid_arg "rfft requires at least one axis";

  (* For rfft, we only transform the last axis to half size *)
  (* All other axes are transformed normally *)
  let last_axis_idx = Array.length axes - 1 in
  let last_axis = axes.(last_axis_idx) in
  let last_axis = if last_axis < 0 then ndim + last_axis else last_axis in

  (* Apply sizes to get the working shape *)
  let work_shape =
    match s with
    | None -> Array.copy input_shape
    | Some sizes ->
        let shape = Array.copy input_shape in
        Array.iteri
          (fun i axis ->
            let axis = if axis < 0 then ndim + axis else axis in
            shape.(axis) <- sizes.(i))
          axes;
        shape
  in

  (* Create a real work buffer with the desired shape and copy/pad input *)
  let work_real = empty context (Internal.dtype input) work_shape in
  (* Zero-fill the work buffer *)
  let work_buf = buffer work_real in
  let work_size = Internal.size work_real in
  for i = 0 to work_size - 1 do
    Array1.unsafe_set work_buf i 0.0
  done;

  (* Copy the overlapping region from input to work_real *)
  let rec copy_data indices dim =
    if dim = ndim then (
      (* Calculate linear indices - bounds are already checked by limit = min *)
      let idx_from = ref 0 in
      let idx_to = ref 0 in
      let stride_from = ref 1 in
      let stride_to = ref 1 in
      for d = ndim - 1 downto 0 do
        idx_from := !idx_from + (indices.(d) * !stride_from);
        idx_to := !idx_to + (indices.(d) * !stride_to);
        stride_from := !stride_from * input_shape.(d);
        stride_to := !stride_to * work_shape.(d)
      done;
      let v = Array1.unsafe_get (buffer input) !idx_from in
      Array1.unsafe_set work_buf !idx_to v)
    else
      let limit = min input_shape.(dim) work_shape.(dim) in
      for i = 0 to limit - 1 do
        indices.(dim) <- i;
        copy_data indices (dim + 1)
      done
  in
  copy_data (Array.make ndim 0) 0;

  (* First, perform FFT on all axes except the last one *)
  let temp = empty context output_dtype work_shape in
  real_to_complex_copy work_real temp;

  (if Array.length axes > 1 then
     let other_axes = Array.sub axes 0 (Array.length axes - 1) in
     kernel_fft_multi ~inverse:false temp temp other_axes);

  (* Now perform FFT on the last axis and extract only the non-redundant part *)
  let output_shape = get_rfft_output_shape work_shape axes in
  let output = empty context output_dtype output_shape in

  (* Perform 1D FFT on the last axis for each position in other dimensions *)
  let rec process_slices indices dim =
    if dim = ndim then (
      (* Calculate source offset *)
      let src_offset = ref 0 in
      let stride = ref 1 in
      for d = ndim - 1 downto 0 do
        src_offset := !src_offset + (indices.(d) * !stride);
        stride := !stride * work_shape.(d)
      done;

      (* Calculate destination offset *)
      let dst_offset = ref 0 in
      let stride = ref 1 in
      for d = ndim - 1 downto 0 do
        dst_offset := !dst_offset + (indices.(d) * !stride);
        stride := !stride * output_shape.(d)
      done;

      (* Perform 1D FFT and copy only non-redundant part *)
      let n = work_shape.(last_axis) in
      let stride_in =
        let s = ref 1 in
        for d = last_axis + 1 to ndim - 1 do
          s := !s * work_shape.(d)
        done;
        !s
      in
      let stride_out =
        let s = ref 1 in
        for d = last_axis + 1 to ndim - 1 do
          s := !s * output_shape.(d)
        done;
        !s
      in

      (* Do 1D FFT in-place on temp *)
      fft_1d ~inverse:false ~scale:1.0 (buffer temp) n stride_in !src_offset;

      (* Copy only the non-redundant part to output *)
      for i = 0 to output_shape.(last_axis) - 1 do
        let src_idx = !src_offset + (i * stride_in) in
        let dst_idx = !dst_offset + (i * stride_out) in
        let v = Array1.get (buffer temp) src_idx in
        Array1.set (buffer output) dst_idx v
      done)
    else if dim = last_axis then (
      indices.(dim) <- 0;
      process_slices indices (dim + 1))
    else
      for i = 0 to work_shape.(dim) - 1 do
        indices.(dim) <- i;
        process_slices indices (dim + 1)
      done
  in
  process_slices (Array.make ndim 0) 0;
  output

(* IRFFT kernel *)
let kernel_irfft (type b c) context (input : (Complex.t, b) t)
    (output_dtype : (float, c) Dtype.t) axes s =
  let input_shape = Internal.shape input in
  let ndim = Array.length input_shape in

  (* Validate axes and sizes *)
  validate_axes_and_sizes ndim axes s;
  
  (* Require at least one axis *)
  if Array.length axes = 0 then invalid_arg "irfft requires at least one axis";

  (* For irfft, we need to handle the inverse of rfft *)
  (* The input has half-size on the last transformed axis *)
  let last_axis_idx = Array.length axes - 1 in
  let last_axis = axes.(last_axis_idx) in
  let last_axis = if last_axis < 0 then ndim + last_axis else last_axis in

  (* Determine the target output size and frequency domain size *)
  let target_size =
    match s with
    | Some sizes -> sizes.(Array.length sizes - 1)
    | None -> 
        (* Default to even size when ambiguous *)
        (input_shape.(last_axis) - 1) * 2
  in
  
  (* The frequency domain size to use for reconstruction *)
  let freq_domain_size = target_size in
  
  (* Determine real output shape *)
  let output_shape =
    let shape = Array.copy input_shape in
    (* Restore full size for last axis *)
    shape.(last_axis) <-
      (match s with
      | None -> freq_domain_size
      | Some sizes -> sizes.(Array.length sizes - 1));

    match s with
    | None -> shape
    | Some sizes ->
        Array.iteri
          (fun i axis ->
            let axis = if axis < 0 then ndim + axis else axis in
            shape.(axis) <- sizes.(i))
          axes;
        shape
  in

  (* Create real output tensor *)
  let output = empty context output_dtype output_shape in

  (* First, create a complex tensor matching the frequency domain size *)
  let full_complex_shape = Array.copy input_shape in
  full_complex_shape.(last_axis) <- freq_domain_size;
  let full_complex = empty context (Internal.dtype input) full_complex_shape in
  
  (* Zero-fill the full complex buffer for padding case *)
  Array1.fill (buffer full_complex) Complex.zero;

  (* Copy the non-redundant part from input and reconstruct the symmetric
     part *)
  let rec process_slices indices dim =
    if dim = ndim then (
      (* Calculate source and destination offsets *)
      let src_offset = ref 0 in
      let src_stride = ref 1 in
      for d = ndim - 1 downto 0 do
        src_offset := !src_offset + (indices.(d) * !src_stride);
        src_stride := !src_stride * input_shape.(d)
      done;

      let dst_offset = ref 0 in
      let dst_stride = ref 1 in
      for d = ndim - 1 downto 0 do
        dst_offset := !dst_offset + (indices.(d) * !dst_stride);
        dst_stride := !dst_stride * full_complex_shape.(d)
      done;

      (* Copy non-redundant part *)
      let hermitian_size = input_shape.(last_axis) in
      let full_size = freq_domain_size in

      let src_stride_elem =
        let s = ref 1 in
        for d = last_axis + 1 to ndim - 1 do
          s := !s * input_shape.(d)
        done;
        !s
      in

      let dst_stride_elem =
        let s = ref 1 in
        for d = last_axis + 1 to ndim - 1 do
          s := !s * full_complex_shape.(d)
        done;
        !s
      in

      (* Copy the non-redundant part *)
      (* Only copy the frequency bins that correspond to the target size *)
      let target_bins = (full_size / 2) + 1 in
      let copy_limit = min hermitian_size target_bins in
      for i = 0 to copy_limit - 1 do
        let src_idx = !src_offset + (i * src_stride_elem) in
        let dst_idx = !dst_offset + (i * dst_stride_elem) in
        let v = Array1.get (buffer input) src_idx in
        Array1.set (buffer full_complex) dst_idx v
      done;

      (* Only enforce real DC/Nyquist for the actual DC component *)
      (* This is when all indices except the last axis are 0 *)
      let is_dc_slice = 
        let rec check i = 
          if i = ndim then true
          else if i = last_axis then check (i + 1)
          else if indices.(i) = 0 then check (i + 1)
          else false
        in check 0
      in
      (if copy_limit > 0 && is_dc_slice then
         let dc_idx = !dst_offset in
         let v = Array1.get (buffer full_complex) dc_idx in
         Array1.set (buffer full_complex) dc_idx Complex.{ re = v.re; im = 0.0 });

      (* Enforce real Nyquist bin if N is even *)
      (if full_size mod 2 = 0 && copy_limit > full_size / 2 && is_dc_slice then
         let nyquist_idx = !dst_offset + (full_size / 2 * dst_stride_elem) in
         let v = Array1.get (buffer full_complex) nyquist_idx in
         Array1.set (buffer full_complex) nyquist_idx
           Complex.{ re = v.re; im = 0.0 });

      (* Fill the symmetric part *)
      for i = copy_limit to full_size - 1 do
        let mirror_idx = full_size - i in
        if mirror_idx > 0 && mirror_idx < copy_limit then
          let src_idx = !dst_offset + (mirror_idx * dst_stride_elem) in
          let dst_idx = !dst_offset + (i * dst_stride_elem) in
          let v = Array1.get (buffer full_complex) src_idx in
          (* Complex conjugate for Hermitian symmetry *)
          Array1.set (buffer full_complex) dst_idx
            Complex.{ re = v.re; im = -.v.im }
      done)
    else if dim = last_axis then (
      indices.(dim) <- 0;
      process_slices indices (dim + 1))
    else
      (* Use min to avoid out-of-bounds when s shrinks non-last axes *)
      let limit = min input_shape.(dim) full_complex_shape.(dim) in
      for i = 0 to limit - 1 do
        indices.(dim) <- i;
        process_slices indices (dim + 1)
      done
  in
  process_slices (Array.make ndim 0) 0;

  (* Now perform inverse FFT on all axes *)
  kernel_fft_multi ~inverse:true full_complex full_complex axes;

  (* Extract real part, handling truncation if output is smaller *)
  let output_buf = buffer output in
  let complex_buf = buffer full_complex in
  
  (* Copy the real part, handling different shapes *)
  let rec copy_real indices dim =
    if dim = ndim then (
      (* Calculate indices *)
      let out_idx = ref 0 in
      let complex_idx = ref 0 in
      let out_stride = ref 1 in
      let complex_stride = ref 1 in
      
      for d = ndim - 1 downto 0 do
        out_idx := !out_idx + (indices.(d) * !out_stride);
        complex_idx := !complex_idx + (indices.(d) * !complex_stride);
        out_stride := !out_stride * output_shape.(d);
        complex_stride := !complex_stride * full_complex_shape.(d)
      done;
      
      let v = Array1.unsafe_get complex_buf !complex_idx in
      Array1.unsafe_set output_buf !out_idx v.re
    ) else
      let limit = min output_shape.(dim) full_complex_shape.(dim) in
      for i = 0 to limit - 1 do
        indices.(dim) <- i;
        copy_real indices (dim + 1)
      done
  in
  copy_real (Array.make ndim 0) 0;

  output

(* Main FFT operations *)
let fft (type b) (context : context) (input : (Complex.t, b) t) ~axes ~s :
    (Complex.t, b) t =
  let output_shape = get_fft_output_shape (Internal.shape input) axes s in

  match dtype input with
  | Complex16 ->
      let output = empty context Complex16 output_shape in
      kernel_fft_multi ~inverse:false input output axes;
      output
  | Complex32 ->
      let output = empty context Complex32 output_shape in
      kernel_fft_multi ~inverse:false input output axes;
      output
  | Complex64 ->
      let output = empty context Complex64 output_shape in
      kernel_fft_multi ~inverse:false input output axes;
      output

let ifft (type b) (context : context) (input : (Complex.t, b) t) ~axes ~s :
    (Complex.t, b) t =
  let output_shape = get_fft_output_shape (Internal.shape input) axes s in

  match dtype input with
  | Complex16 ->
      let output = empty context Complex16 output_shape in
      kernel_fft_multi ~inverse:true input output axes;
      output
  | Complex32 ->
      let output = empty context Complex32 output_shape in
      kernel_fft_multi ~inverse:true input output axes;
      output
  | Complex64 ->
      let output = empty context Complex64 output_shape in
      kernel_fft_multi ~inverse:true input output axes;
      output

let rfft (type b c) (context : context) (input : (float, b) t)
    ~(dtype : (Complex.t, c) Dtype.t) ~axes ~s : (Complex.t, c) t =
  match Internal.dtype input with
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 ->
      kernel_rfft context input dtype axes s

let irfft (type b c) (context : context) (input : (Complex.t, b) t)
    ~(dtype : (float, c) Dtype.t) ~axes ~s : (float, c) t =
  match Internal.dtype input with
  | Complex16 | Complex32 | Complex64 -> kernel_irfft context input dtype axes s
