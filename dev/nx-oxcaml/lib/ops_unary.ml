open Nx_core
open Internal

(* Generic unary operation implementation *)
let unary_op_impl (type a b) ~op_name ~f (ctx : context) (input : (a, b) t)
    (out : (a, b) t) : unit =
  let in_view = view input in
  let out_view = view out in
  let in_buf = buffer input in
  let out_buf = buffer out in
  
  if View.is_c_contiguous in_view && View.is_c_contiguous out_view then (
    (* Fast path for contiguous tensors *)
    let n = View.numel out_view in
    let in_offset = View.offset in_view in
    let out_offset = View.offset out_view in
    
    Parallel.parallel_chunks (get_pool ctx) ~n_elements:n ~chunk_size:1024
      ~body:(fun ~chunk_idx:_ ~start ~stop ->
        for i = start to stop - 1 do
          let val_ = Array1.get in_buf (in_offset + i) in
          let result = f val_ in
          Array1.set out_buf (out_offset + i) result
        done))
  else
    (* General case: handle strided access *)
    let out_shape = View.shape out_view in
    let ndim = Array.length out_shape in
    let in_strides = View.strides in_view in
    let out_strides = View.strides out_view in
    let in_offset = View.offset in_view in
    let out_offset = View.offset out_view in
    
    let rec iterate indices dim =
      if dim = ndim then (
        let in_idx = ref in_offset in
        let out_idx = ref out_offset in
        for d = 0 to ndim - 1 do
          in_idx := !in_idx + (indices.(d) * in_strides.(d));
          out_idx := !out_idx + (indices.(d) * out_strides.(d))
        done;
        let val_ = Array1.get in_buf !in_idx in
        let result = f val_ in
        Array1.set out_buf !out_idx result)
      else
        for i = 0 to out_shape.(dim) - 1 do
          indices.(dim) <- i;
          iterate indices (dim + 1)
        done
    in
    let indices = Array.make ndim 0 in
    iterate indices 0

(* Specialized unboxed implementations for float operations *)
module Float_ops = struct
  open Unboxed_ops
  
  let neg ctx input out =
    unary_op_impl ~op_name:"neg"
      ~f:(fun x -> box_float (neg_float (unbox_float x)))
      ctx input out
  
  let sqrt ctx input out =
    unary_op_impl ~op_name:"sqrt"
      ~f:(fun x -> box_float (sqrt_float (unbox_float x)))
      ctx input out
  
  let sin ctx input out =
    unary_op_impl ~op_name:"sin"
      ~f:(fun x -> box_float (sin_float (unbox_float x)))
      ctx input out
  
  let exp ctx input out =
    unary_op_impl ~op_name:"exp"
      ~f:(fun x -> box_float (exp_float (unbox_float x)))
      ctx input out
  
  let log ctx input out =
    unary_op_impl ~op_name:"log"
      ~f:(fun x -> box_float (log_float (unbox_float x)))
      ctx input out
  
  let log2 ctx input out =
    unary_op_impl ~op_name:"log2"
      ~f:(fun x -> Float.(log x /. log 2.0))
      ctx input out
  
  let exp2 ctx input out =
    unary_op_impl ~op_name:"exp2"
      ~f:(fun x -> Float.pow 2.0 x)
      ctx input out
  
  let recip ctx input out =
    unary_op_impl ~op_name:"recip"
      ~f:(fun x -> box_float (div_float #1.0 (unbox_float x)))
      ctx input out
end

(* Specialized unboxed implementations for int32 operations *)
module Int32_ops = struct
  open Unboxed_ops
  
  let neg ctx input out =
    unary_op_impl ~op_name:"neg"
      ~f:(fun x -> box_int32 (neg_int32 (unbox_int32 x)))
      ctx input out
end

(* Specialized unboxed implementations for int64 operations *)
module Int64_ops = struct
  open Unboxed_ops
  
  let neg ctx input out =
    unary_op_impl ~op_name:"neg"
      ~f:(fun x -> box_int64 (neg_int64 (unbox_int64 x)))
      ctx input out
end

(* Dispatch based on dtype *)
let neg ctx input out =
  match dtype input with
  | Dtype.Float64 -> Float_ops.neg ctx input out
  | Dtype.Float32 -> Float_ops.neg ctx input out
  | Dtype.Int32 -> Int32_ops.neg ctx input out
  | Dtype.Int64 -> Int64_ops.neg ctx input out
  | _ -> unary_op_impl ~op_name:"neg" ~f:(fun x -> Stdlib.(-x)) ctx input out

let sqrt ctx input out =
  match dtype input with
  | Dtype.Float64 -> Float_ops.sqrt ctx input out
  | Dtype.Float32 -> Float_ops.sqrt ctx input out
  | _ -> failwith "sqrt: only supported for float dtypes"

let sin ctx input out =
  match dtype input with
  | Dtype.Float64 -> Float_ops.sin ctx input out
  | Dtype.Float32 -> Float_ops.sin ctx input out
  | _ -> failwith "sin: only supported for float dtypes"

let exp2 ctx input out =
  match dtype input with
  | Dtype.Float64 -> Float_ops.exp2 ctx input out
  | Dtype.Float32 -> Float_ops.exp2 ctx input out
  | _ -> failwith "exp2: only supported for float dtypes"

let log2 ctx input out =
  match dtype input with
  | Dtype.Float64 -> Float_ops.log2 ctx input out
  | Dtype.Float32 -> Float_ops.log2 ctx input out
  | _ -> failwith "log2: only supported for float dtypes"

let recip ctx input out =
  match dtype input with
  | Dtype.Float64 -> Float_ops.recip ctx input out
  | Dtype.Float32 -> Float_ops.recip ctx input out
  | _ -> failwith "recip: only supported for float dtypes"