open Bigarray
module Dtype = Nx_core.Dtype
module Shape = Nx_core.Shape
module View = Nx_core.View
open Internal

(* Tile sizes for the fast path (tune for your CPU) *)
let mc = 128
let nc = 128
let kc = 64

(* Helper to broadcast shapes for matmul *)
let broadcast_matmul_shapes shape_a shape_b =
  let ndim_a = Array.length shape_a in
  let ndim_b = Array.length shape_b in

  if ndim_a < 2 || ndim_b < 2 then failwith "matmul: inputs must be at least 2D";

  (* Check matrix dimensions compatibility *)
  let m = shape_a.(ndim_a - 2) in
  let k_a = shape_a.(ndim_a - 1) in
  let k_b = shape_b.(ndim_b - 2) in
  let n = shape_b.(ndim_b - 1) in

  if k_a <> k_b then
    invalid_arg
      (Printf.sprintf
         "dot: cannot contract %s (last axis: %d) to %s (axis %d: %d) (size \
          %d≠%d)"
         (Shape.to_string shape_a) k_a (Shape.to_string shape_b) (ndim_b - 2)
         k_b k_a k_b);

  (* Extract batch dimensions *)
  let batch_a = Array.sub shape_a 0 (ndim_a - 2) in
  let batch_b = Array.sub shape_b 0 (ndim_b - 2) in

  (* Broadcast batch dimensions *)
  let max_batch_ndim = max (Array.length batch_a) (Array.length batch_b) in
  let batch_shape = Array.make max_batch_ndim 1 in

  (* Fill from the right *)
  for i = 0 to Array.length batch_a - 1 do
    batch_shape.(max_batch_ndim - Array.length batch_a + i) <- batch_a.(i)
  done;

  for i = 0 to Array.length batch_b - 1 do
    let idx = max_batch_ndim - Array.length batch_b + i in
    if batch_shape.(idx) = 1 then batch_shape.(idx) <- batch_b.(i)
    else if batch_b.(i) <> 1 && batch_b.(i) <> batch_shape.(idx) then
      failwith
        (Printf.sprintf "matmul: cannot broadcast shapes %s and %s"
           (Shape.to_string shape_a) (Shape.to_string shape_b))
  done;

  (* Output shape is batch_shape + [m; n] *)
  Array.concat [ batch_shape; [| m; n |] ]

let kernel_block_float32 (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (float, float32_elt) t) ~row0 ~row1 =
  let a_buf, b_buf, c_buf = (buffer a, buffer b, buffer out) in
  let k = dim (ndim a - 1) a in
  let n = dim (ndim b - 1) b in
  let a_rs = k and b_rs = n and c_rs = n in
  let a0 = offset a and b0 = offset b and c0 = offset out in

  let rec jc_loop jc =
    if jc >= n then ()
    else
      let nc' = min nc (n - jc) in
      let rec pc_loop pc =
        if pc >= k then ()
        else
          let kc' = min kc (k - pc) in
          let rec ic_loop ic =
            if ic >= row1 then ()
            else
              let mc' = min mc (row1 - ic) in
              for i = ic to ic + mc' - 1 do
                let a_row = a0 + (i * a_rs) + pc
                and c_row = c0 + (i * c_rs) + jc in
                for j = jc to jc + nc' - 1 do
                  let sum = ref 0. in
                  let a_idx = ref a_row
                  and b_idx = ref (b0 + (pc * b_rs) + j) in
                  for _p = 0 to kc' - 1 do
                    let av = Array1.unsafe_get a_buf !a_idx
                    and bv = Array1.unsafe_get b_buf !b_idx in
                    sum := !sum +. (av *. bv);
                    incr a_idx;
                    b_idx := !b_idx + b_rs
                  done;
                  Array1.unsafe_set c_buf (c_row + j - jc) !sum
                done
              done;
              ic_loop (ic + mc')
          in
          ic_loop row0;
          pc_loop (pc + kc')
      in
      pc_loop 0;
      jc_loop (jc + nc')
  in
  jc_loop 0

let kernel_matmul_fast_float32 pool (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (float, float32_elt) t) =
  let m = dim 0 a in
  Array1.fill (buffer out) 0.0;
  Parallel.parallel_for pool 0 (m - 1) (fun r0 r1 ->
      kernel_block_float32 a b out ~row0:r0 ~row1:r1)

let kernel_matmul_generic_float32 pool (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (float, float32_elt) t) =
  let a_buf, b_buf, c_buf = (buffer a, buffer b, buffer out) in
  let shape_a, shape_b, shape_c = (shape a, shape b, shape out) in
  let nd_a, nd_b, nd_c =
    (Array.length shape_a, Array.length shape_b, Array.length shape_c)
  in
  let m = shape_c.(nd_c - 2)
  and n = shape_c.(nd_c - 1)
  and k = shape_a.(nd_a - 1) in
  let batch_shape = Array.sub shape_c 0 (max 0 (nd_c - 2)) in
  let batch_sz =
    if Array.length batch_shape = 0 then 1 else Shape.numel batch_shape
  in
  let total_units = batch_sz * m in
  (* each unit = one row of C *)

  let a_str = View.strides a.view
  and b_str = View.strides b.view
  and c_str = View.strides out.view in

  Parallel.parallel_for pool 0 (total_units - 1) (fun u0 u1 ->
      let a_idx = Array.make nd_a 0
      and b_idx = Array.make nd_b 0
      and c_idx = Array.make nd_c 0 in
      for work = u0 to u1 - 1 do
        let batch = work / m and i = work mod m in
        (* unravel batch index into leading dims of C *)
        if batch_sz <> 1 then Shape.unravel_index_into batch batch_shape c_idx;
        (* broadcast batch into a_idx / b_idx *)
        Shape.broadcast_index_into c_idx shape_a a_idx;
        Shape.broadcast_index_into c_idx shape_b b_idx;
        (* set row index *)
        c_idx.(nd_c - 2) <- i;
        a_idx.(nd_a - 2) <- i;
        for j = 0 to n - 1 do
          c_idx.(nd_c - 1) <- j;
          b_idx.(nd_b - 1) <- j;
          let sum = ref 0. in
          for l = 0 to k - 1 do
            a_idx.(nd_a - 1) <- l;
            b_idx.(nd_b - 2) <- l;
            let av =
              Array1.unsafe_get a_buf (offset a + Shape.ravel_index a_idx a_str)
            in
            let bv =
              Array1.unsafe_get b_buf (offset b + Shape.ravel_index b_idx b_str)
            in
            sum := !sum +. (av *. bv)
          done;
          let c_off = offset out + Shape.ravel_index c_idx c_str in
          Array1.unsafe_set c_buf c_off !sum
        done
      done)

let kernel_block_float64 (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (float, float64_elt) t) ~row0 ~row1 =
  let a_buf, b_buf, c_buf = (buffer a, buffer b, buffer out) in
  let k = dim (ndim a - 1) a in
  let n = dim (ndim b - 1) b in
  let a_rs = k and b_rs = n and c_rs = n in
  let a0 = offset a and b0 = offset b and c0 = offset out in

  let rec jc_loop jc =
    if jc >= n then ()
    else
      let nc' = min nc (n - jc) in
      let rec pc_loop pc =
        if pc >= k then ()
        else
          let kc' = min kc (k - pc) in
          let rec ic_loop ic =
            if ic >= row1 then ()
            else
              let mc' = min mc (row1 - ic) in
              for i = ic to ic + mc' - 1 do
                let a_row = a0 + (i * a_rs) + pc
                and c_row = c0 + (i * c_rs) + jc in
                for j = jc to jc + nc' - 1 do
                  let sum = ref 0. in
                  let a_idx = ref a_row
                  and b_idx = ref (b0 + (pc * b_rs) + j) in
                  for _p = 0 to kc' - 1 do
                    let av = Array1.unsafe_get a_buf !a_idx
                    and bv = Array1.unsafe_get b_buf !b_idx in
                    sum := !sum +. (av *. bv);
                    incr a_idx;
                    b_idx := !b_idx + b_rs
                  done;
                  Array1.unsafe_set c_buf (c_row + j - jc) !sum
                done
              done;
              ic_loop (ic + mc')
          in
          ic_loop row0;
          pc_loop (pc + kc')
      in
      pc_loop 0;
      jc_loop (jc + nc')
  in
  jc_loop 0

let kernel_matmul_fast_float64 pool (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (float, float64_elt) t) =
  let m = dim 0 a in
  Array1.fill (buffer out) 0.0;
  Parallel.parallel_for pool 0 (m - 1) (fun r0 r1 ->
      kernel_block_float64 a b out ~row0:r0 ~row1:r1)

let kernel_matmul_generic_float64 pool (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (float, float64_elt) t) =
  let a_buf, b_buf, c_buf = (buffer a, buffer b, buffer out) in
  let shape_a, shape_b, shape_c = (shape a, shape b, shape out) in
  let nd_a, nd_b, nd_c =
    (Array.length shape_a, Array.length shape_b, Array.length shape_c)
  in
  let m = shape_c.(nd_c - 2)
  and n = shape_c.(nd_c - 1)
  and k = shape_a.(nd_a - 1) in
  let batch_shape = Array.sub shape_c 0 (max 0 (nd_c - 2)) in
  let batch_sz =
    if Array.length batch_shape = 0 then 1 else Shape.numel batch_shape
  in
  let total_units = batch_sz * m in
  (* each unit = one row of C *)

  let a_str = View.strides a.view
  and b_str = View.strides b.view
  and c_str = View.strides out.view in

  Parallel.parallel_for pool 0 (total_units - 1) (fun u0 u1 ->
      let a_idx = Array.make nd_a 0
      and b_idx = Array.make nd_b 0
      and c_idx = Array.make nd_c 0 in
      for work = u0 to u1 - 1 do
        let batch = work / m and i = work mod m in
        (* unravel batch index into leading dims of C *)
        if batch_sz <> 1 then Shape.unravel_index_into batch batch_shape c_idx;
        (* broadcast batch into a_idx / b_idx *)
        Shape.broadcast_index_into c_idx shape_a a_idx;
        Shape.broadcast_index_into c_idx shape_b b_idx;
        (* set row index *)
        c_idx.(nd_c - 2) <- i;
        a_idx.(nd_a - 2) <- i;
        for j = 0 to n - 1 do
          c_idx.(nd_c - 1) <- j;
          b_idx.(nd_b - 1) <- j;
          let sum = ref 0. in
          for l = 0 to k - 1 do
            a_idx.(nd_a - 1) <- l;
            b_idx.(nd_b - 2) <- l;
            let av =
              Array1.unsafe_get a_buf (offset a + Shape.ravel_index a_idx a_str)
            in
            let bv =
              Array1.unsafe_get b_buf (offset b + Shape.ravel_index b_idx b_str)
            in
            sum := !sum +. (av *. bv)
          done;
          let c_off = offset out + Shape.ravel_index c_idx c_str in
          Array1.unsafe_set c_buf c_off !sum
        done
      done)

let matmul (type a b) (ctx : context) (a : (a, b) t) (b : (a, b) t) : (a, b) t =
  let out_shape = broadcast_matmul_shapes (shape a) (shape b) in
  let out = empty ctx (dtype a) out_shape in
  let fast_path =
    is_c_contiguous a && is_c_contiguous b
    && offset a = 0
    && offset b = 0
    && Array.length (shape a) = 2
    && Array.length (shape b) = 2
  in
  let () =
    match dtype a with
    | Dtype.Float32 when fast_path ->
        kernel_matmul_fast_float32 ctx.pool a b out
    | Dtype.Float32 -> kernel_matmul_generic_float32 ctx.pool a b out
    | Dtype.Float64 when fast_path ->
        kernel_matmul_fast_float64 ctx.pool a b out
    | Dtype.Float64 -> kernel_matmul_generic_float64 ctx.pool a b out
    | Dtype.Int32 | Dtype.Int64 | Dtype.UInt8 | Dtype.UInt16 | Dtype.Int8
    | Dtype.Int16 | Dtype.Float16 | Dtype.Complex32 | Dtype.Complex64
    | Dtype.Int | Dtype.NativeInt ->
        failwith "matmul: dtype not supported"
  in
  out
