open Nx_core
open Internal

(* Generic binary operation implementation *)
let binary_op_impl (type a b) ~op_name ~f (ctx : context) (a : (a, b) t)
    (b : (a, b) t) (out : (a, b) t) : unit =
  let a_view = view a in
  let b_view = view b in
  let out_view = view out in
  let a_buf = buffer a in
  let b_buf = buffer b in
  let out_buf = buffer out in
  
  if View.is_c_contiguous a_view && View.is_c_contiguous b_view
     && View.is_c_contiguous out_view
  then (
    (* Fast path for contiguous tensors *)
    let n = View.numel out_view in
    let a_offset = View.offset a_view in
    let b_offset = View.offset b_view in
    let out_offset = View.offset out_view in
    
    Parallel.parallel_chunks (get_pool ctx) ~n_elements:n ~chunk_size:1024
      ~body:(fun ~chunk_idx:_ ~start ~stop ->
        for i = start to stop - 1 do
          let a_val = Array1.get a_buf (a_offset + i) in
          let b_val = Array1.get b_buf (b_offset + i) in
          let result = f a_val b_val in
          Array1.set out_buf (out_offset + i) result
        done))
  else
    (* General case: handle broadcasting and strided access *)
    let out_shape = View.shape out_view in
    let ndim = Array.length out_shape in
    let a_strides = View.strides a_view in
    let b_strides = View.strides b_view in
    let out_strides = View.strides out_view in
    let a_offset = View.offset a_view in
    let b_offset = View.offset b_view in
    let out_offset = View.offset out_view in
    
    let rec iterate indices dim =
      if dim = ndim then (
        let a_idx = ref a_offset in
        let b_idx = ref b_offset in
        let out_idx = ref out_offset in
        for d = 0 to ndim - 1 do
          a_idx := !a_idx + (indices.(d) * a_strides.(d));
          b_idx := !b_idx + (indices.(d) * b_strides.(d));
          out_idx := !out_idx + (indices.(d) * out_strides.(d))
        done;
        let a_val = Array1.get a_buf !a_idx in
        let b_val = Array1.get b_buf !b_idx in
        let result = f a_val b_val in
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
  
  let add ctx a b out =
    binary_op_impl ~op_name:"add"
      ~f:(fun x y -> box_float (add_float (unbox_float x) (unbox_float y)))
      ctx a b out
  
  let mul ctx a b out =
    binary_op_impl ~op_name:"mul"
      ~f:(fun x y -> box_float (mul_float (unbox_float x) (unbox_float y)))
      ctx a b out
  
  let fdiv ctx a b out =
    binary_op_impl ~op_name:"fdiv"
      ~f:(fun x y -> box_float (div_float (unbox_float x) (unbox_float y)))
      ctx a b out
  
  let max ctx a b out =
    binary_op_impl ~op_name:"max" ~f:Float.max ctx a b out
  
  let pow ctx a b out =
    binary_op_impl ~op_name:"pow" ~f:Float.pow ctx a b out
end

(* Specialized unboxed implementations for int32 operations *)
module Int32_ops = struct
  open Unboxed_ops
  
  let add ctx a b out =
    binary_op_impl ~op_name:"add"
      ~f:(fun x y -> box_int32 (add_int32 (unbox_int32 x) (unbox_int32 y)))
      ctx a b out
  
  let mul ctx a b out =
    binary_op_impl ~op_name:"mul"
      ~f:(fun x y -> box_int32 (mul_int32 (unbox_int32 x) (unbox_int32 y)))
      ctx a b out
  
  let idiv ctx a b out =
    binary_op_impl ~op_name:"idiv"
      ~f:(fun x y -> box_int32 (div_int32 (unbox_int32 x) (unbox_int32 y)))
      ctx a b out
end

(* Specialized unboxed implementations for int64 operations *)
module Int64_ops = struct
  open Unboxed_ops
  
  let add ctx a b out =
    binary_op_impl ~op_name:"add"
      ~f:(fun x y -> box_int64 (add_int64 (unbox_int64 x) (unbox_int64 y)))
      ctx a b out
  
  let mul ctx a b out =
    binary_op_impl ~op_name:"mul"
      ~f:(fun x y -> box_int64 (mul_int64 (unbox_int64 x) (unbox_int64 y)))
      ctx a b out
  
  let idiv ctx a b out =
    binary_op_impl ~op_name:"idiv"
      ~f:(fun x y -> box_int64 (div_int64 (unbox_int64 x) (unbox_int64 y)))
      ctx a b out
end

(* Generic implementations for other types *)
let add ctx a b out =
  match dtype a with
  | Dtype.Float64 -> Float_ops.add ctx a b out
  | Dtype.Float32 -> Float_ops.add ctx a b out  (* Will need float32# later *)
  | Dtype.Int32 -> Int32_ops.add ctx a b out
  | Dtype.Int64 -> Int64_ops.add ctx a b out
  | _ -> binary_op_impl ~op_name:"add" ~f:(fun x y -> Stdlib.(x + y)) ctx a b out

let mul ctx a b out =
  match dtype a with
  | Dtype.Float64 -> Float_ops.mul ctx a b out
  | Dtype.Float32 -> Float_ops.mul ctx a b out
  | Dtype.Int32 -> Int32_ops.mul ctx a b out
  | Dtype.Int64 -> Int64_ops.mul ctx a b out
  | _ -> binary_op_impl ~op_name:"mul" ~f:(fun x y -> Stdlib.(x * y)) ctx a b out

let idiv ctx a b out =
  match dtype a with
  | Dtype.Int32 -> Int32_ops.idiv ctx a b out
  | Dtype.Int64 -> Int64_ops.idiv ctx a b out
  | _ -> binary_op_impl ~op_name:"idiv" ~f:(fun x y -> Stdlib.(x / y)) ctx a b out

let fdiv ctx a b out =
  match dtype a with
  | Dtype.Float64 -> Float_ops.fdiv ctx a b out
  | Dtype.Float32 -> Float_ops.fdiv ctx a b out
  | _ -> failwith "fdiv: expected float dtype"

let max ctx a b out =
  match dtype a with
  | Dtype.Float64 -> Float_ops.max ctx a b out
  | Dtype.Float32 -> Float_ops.max ctx a b out
  | _ -> binary_op_impl ~op_name:"max" ~f:max ctx a b out

let pow ctx a b out =
  match dtype a with
  | Dtype.Float64 -> Float_ops.pow ctx a b out
  | Dtype.Float32 -> Float_ops.pow ctx a b out
  | _ -> failwith "pow: only supported for float dtypes"

(* Other binary operations - placeholders for now *)
let mod_ ctx a b out =
  binary_op_impl ~op_name:"mod" ~f:(fun x y -> Stdlib.(x mod y)) ctx a b out

let xor ctx a b out =
  binary_op_impl ~op_name:"xor" ~f:(fun x y -> x lxor y) ctx a b out

let or_ ctx a b out =
  binary_op_impl ~op_name:"or" ~f:(fun x y -> x lor y) ctx a b out

let and_ ctx a b out =
  binary_op_impl ~op_name:"and" ~f:(fun x y -> x land y) ctx a b out