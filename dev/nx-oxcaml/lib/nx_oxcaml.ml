(* OxCaml backend for Nx - leverages unboxed types for performance *)

open Nx_core

type ('a, 'b) buffer = ('a, 'b) Internal.buffer
type context = Internal.context

type ('a, 'b) t = ('a, 'b) Internal.t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : View.t;
}

(* Accessors *)
let view t = t.view
let dtype t = t.dtype
let data t = t.buffer
let context t = t.context
let with_view t view = { t with view }

(* Create a new context *)
let create_context ?n_threads () =
  Internal.{ pool = Some (Parallel.get_or_setup_pool ?n_threads ()) }

(* --- Backend Ops Implementation --- *)

let op_buffer ctx dt size_in_elements =
  let kind = Dtype.to_bigarray_kind dt in
  let ba = Bigarray.Array1.create kind Bigarray.c_layout size_in_elements in
  let initial_view =
    if size_in_elements = 0 then View.create [| 0 |]
    else View.create [| size_in_elements |]
  in
  { context = ctx; dtype = dt; buffer = ba; view = initial_view }

let op_const_scalar ctx value dt =
  let kind = Dtype.to_bigarray_kind dt in
  let ba = Bigarray.Array1.create kind Bigarray.c_layout 1 in
  Bigarray.Array1.set ba 0 value;
  let scalar_view = View.create [||] in
  { context = ctx; dtype = dt; buffer = ba; view = scalar_view }

let op_const_array ctx bigarray =
  let dtype = Dtype.of_bigarray_kind (Bigarray.Array1.kind bigarray) in
  let size = Bigarray.Array1.dim bigarray in
  let t = op_buffer ctx dtype size in
  Bigarray.Array1.blit bigarray (data t);
  t

(* Binary Operations *)
let op_add a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.add ctx a b out_tensor;
  out_tensor

let op_mul a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.mul ctx a b out_tensor;
  out_tensor

let op_idiv a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.idiv ctx a b out_tensor;
  out_tensor

let op_fdiv a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.fdiv ctx a b out_tensor;
  out_tensor

let op_max a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.max ctx a b out_tensor;
  out_tensor

let op_mod a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.mod_ ctx a b out_tensor;
  out_tensor

let op_pow a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.pow ctx a b out_tensor;
  out_tensor

let op_cmplt a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx Dtype.Uint8 out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  (* TODO: Implement comparison *)
  failwith "op_cmplt: not implemented yet";
  out_tensor

let op_cmpne a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx Dtype.Uint8 out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  (* TODO: Implement comparison *)
  failwith "op_cmpne: not implemented yet";
  out_tensor

let op_xor a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.xor ctx a b out_tensor;
  out_tensor

let op_or a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.or_ ctx a b out_tensor;
  out_tensor

let op_and a b =
  let ctx = a.context in
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.and_ ctx a b out_tensor;
  out_tensor

(* Unary Operations *)
let op_neg input =
  let ctx = input.context in
  let out_shape = View.shape input.view in
  let out_size = View.numel input.view in
  let out_tensor =
    op_buffer ctx input.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.neg ctx input out_tensor;
  out_tensor

let op_log2 input =
  let ctx = input.context in
  let out_shape = View.shape input.view in
  let out_size = View.numel input.view in
  let out_tensor =
    op_buffer ctx input.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.log2 ctx input out_tensor;
  out_tensor

let op_exp2 input =
  let ctx = input.context in
  let out_shape = View.shape input.view in
  let out_size = View.numel input.view in
  let out_tensor =
    op_buffer ctx input.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.exp2 ctx input out_tensor;
  out_tensor

let op_sin input =
  let ctx = input.context in
  let out_shape = View.shape input.view in
  let out_size = View.numel input.view in
  let out_tensor =
    op_buffer ctx input.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.sin ctx input out_tensor;
  out_tensor

let op_sqrt input =
  let ctx = input.context in
  let out_shape = View.shape input.view in
  let out_size = View.numel input.view in
  let out_tensor =
    op_buffer ctx input.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.sqrt ctx input out_tensor;
  out_tensor

let op_recip input =
  let ctx = input.context in
  let out_shape = View.shape input.view in
  let out_size = View.numel input.view in
  let out_tensor =
    op_buffer ctx input.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.recip ctx input out_tensor;
  out_tensor

(* Ternary Operation *)
let op_where cond if_true if_false =
  (* TODO: Implement where operation *)
  failwith "op_where: not implemented yet"

(* Reduction Operations *)
let op_reduce_sum ~axes ~keepdims input =
  (* TODO: Implement reduction operations *)
  failwith "op_reduce_sum: not implemented yet"

let op_reduce_max ~axes ~keepdims input =
  (* TODO: Implement reduction operations *)
  failwith "op_reduce_max: not implemented yet"

let op_reduce_prod ~axes ~keepdims input =
  (* TODO: Implement reduction operations *)
  failwith "op_reduce_prod: not implemented yet"

(* Movement Operations *)
let op_expand input new_shape =
  { input with view = View.broadcast input.view new_shape }

let op_reshape input new_shape =
  { input with view = View.reshape input.view new_shape }

let op_permute input axes =
  { input with view = View.permute input.view axes }

let op_pad input padding fill_value =
  (* TODO: Implement pad operation *)
  failwith "op_pad: not implemented yet"

let op_shrink input bounds =
  { input with view = View.slice input.view bounds }

let op_flip input axes_to_flip =
  (* TODO: Implement flip operation *)
  failwith "op_flip: not implemented yet"

let op_cat tensors axis =
  (* TODO: Implement concatenation *)
  failwith "op_cat: not implemented yet"

(* Other Operations *)
let op_cast input target_dtype =
  (* TODO: Implement cast operation *)
  failwith "op_cast: not implemented yet"

let op_contiguous input =
  if View.is_c_contiguous input.view then input
  else (
    let out_shape = View.shape input.view in
    let out_size = View.numel input.view in
    let out_tensor =
      op_buffer input.context input.dtype out_size |> fun t ->
      with_view t (View.create out_shape)
    in
    (* TODO: Copy data to make contiguous *)
    failwith "op_contiguous: copy not implemented yet")

let op_copy input =
  let out_shape = View.shape input.view in
  let out_size = View.numel input.view in
  let out_tensor =
    op_buffer input.context input.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  (* TODO: Copy data *)
  failwith "op_copy: not implemented yet"

let op_assign dst src =
  (* TODO: Implement assignment *)
  failwith "op_assign: not implemented yet"

let op_threefry key counter =
  (* TODO: Implement threefry random number generator *)
  failwith "op_threefry: not implemented yet"

(* Element Access Operations *)
let op_gather data indices axis =
  (* TODO: Implement gather *)
  failwith "op_gather: not implemented yet"

let op_scatter ?mode ?unique_indices data_template indices updates axis =
  (* TODO: Implement scatter *)
  failwith "op_scatter: not implemented yet"

let op_unfold input ~kernel_size ~stride ~dilation ~padding =
  (* TODO: Implement unfold (im2col) *)
  failwith "op_unfold: not implemented yet"

let op_fold input ~output_size ~kernel_size ~stride ~dilation ~padding =
  (* TODO: Implement fold (col2im) *)
  failwith "op_fold: not implemented yet"

let op_matmul a b =
  (* TODO: Implement matrix multiplication *)
  failwith "op_matmul: not implemented yet"

(* Fourier Transform Operations *)
let op_fft input ~axes ~s =
  (* TODO: Implement FFT *)
  failwith "op_fft: not implemented yet"

let op_ifft input ~axes ~s =
  (* TODO: Implement inverse FFT *)
  failwith "op_ifft: not implemented yet"

let op_rfft input ~axes ~s =
  (* TODO: Implement real FFT *)
  failwith "op_rfft: not implemented yet"

let op_irfft input ~axes ~s =
  (* TODO: Implement inverse real FFT *)
  failwith "op_irfft: not implemented yet"