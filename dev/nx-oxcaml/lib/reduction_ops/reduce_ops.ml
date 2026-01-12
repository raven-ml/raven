open Bigarray
open Nx_core
open Import

(* Calculates the output shape after reducing along specified axes. *)
let reduction_output_shape (in_shape : int array) (axes : int list)
    (keepdims : bool) : int array =
  let rank = Array.length in_shape in
  let axes =
    List.sort_uniq compare
      (List.map (fun ax -> if ax < 0 then ax + rank else ax) axes)
  in
  let out_dims = ref [] in
  for d = 0 to rank - 1 do
    if List.mem d axes then (if keepdims then out_dims := 1 :: !out_dims)
    else out_dims := in_shape.(d) :: !out_dims
  done;
  Array.of_list (List.rev !out_dims)

(* Initialize the input multi-dimensional index based on an output index and the
   axes being reduced. Works in-place to avoid allocations. *)
let init_input_md_index_inplace (out_md_index : int array)
    (in_md_index : int array) (axes : int list) (rank : int) : unit =
  if Array.length out_md_index = rank then
    for
      (* keepdims = true *)
      d = 0 to rank - 1
    do
      if List.mem d axes then in_md_index.(d) <- 0
      else in_md_index.(d) <- out_md_index.(d)
    done
  else
    (* keepdims = false *)
    let non_reduced_pos = ref 0 in
    for d = 0 to rank - 1 do
      if List.mem d axes then in_md_index.(d) <- 0
      else (
        in_md_index.(d) <- out_md_index.(!non_reduced_pos);
        incr non_reduced_pos)
    done

(* Increments the input multi-dimensional index, primarily along the reduction
   axes. *)
let increment_input_md_index_for_reduction (in_md_index : int array)
    (in_shape : int array) (axes : int list) : bool =
  let rank = Array.length in_shape in
  let rec carry d =
    if d < 0 then false
    else if not (List.mem d axes) then carry (d - 1)
    else (
      in_md_index.(d) <- in_md_index.(d) + 1;
      if in_md_index.(d) < in_shape.(d) then true
      else (
        in_md_index.(d) <- 0;
        carry (d - 1)))
  in
  carry (rank - 1)

(* sum *)
(* 
let add_dtype (type a b) (dtype : (a, b) Dtype.t) (a : a) (b : a) : a =
  match dtype with
  | Float16 -> Float.add a b
  | Float32 -> Float32_u.add a b
  | Float64 -> Float_u.add a b
  | Int8 -> Int.add a b
  | Int16 -> Int.add a b
  | Int32 -> Int32_u.add a b
  | Int64 -> Int64_u.add a b
  | UInt8 -> Int.add a b
  | UInt16 -> Int.add a b
  | Int -> Int.add a b
  | NativeInt -> Nativeint.add a b
  | Complex32 -> Complex.add a b
  | Complex64 -> Complex.add a b
  | _ -> failwith "add_dtype: Unsupported dtype for addition" *)

let kernel_sum_axis_f64 a_arr out_arr va vout axes start_out_idx end_out_idx =
  let a_shape = shape va in
  let a_strides = View.strides va in
  let out_shape = shape vout in
  let rank = Array.length a_shape in

  let axes =
    List.sort_uniq compare
      (List.map (fun ax -> if ax < 0 then ax + rank else ax) axes)
  in

  (* Pre-allocate work arrays to avoid allocations in loop *)
  let out_md_index = Array.make (Array.length out_shape) 0 in
  let in_md_index = Array.make rank 0 in

  for k = start_out_idx to end_out_idx - 1 do
    Shape.unravel_index_into k out_shape out_md_index;

    (* Initialize in_md_index based on out_md_index *)
    init_input_md_index_inplace out_md_index in_md_index axes rank;

    (* let continue_reduction = ref true in
    while !continue_reduction do
      let a_linear_idx = Shape.ravel_index in_md_index a_strides in
      let a_val = Array.unsafe_get a_arr (View.offset va + a_linear_idx) in
      current_sum_ = Float_u.add current_sum a_val;
      continue_reduction :=
        increment_input_md_index_for_reduction in_md_index a_shape axes
    done; *)
    let rec reduce acc =
      let idx =
        View.offset va + Shape.ravel_index in_md_index a_strides
      in
      let acc' = Float_u.add acc (Array.unsafe_get a_arr idx) in
      if increment_input_md_index_for_reduction in_md_index a_shape axes
      then reduce acc'
      else acc'
    in
    let sum = reduce #0.0 in
  Array.unsafe_set out_arr k sum
  done

let kernel_sum_axis_f32 a_arr out_arr va vout axes start_out_idx end_out_idx =
  let a_shape = shape va in
  let a_strides = View.strides va in
  let out_shape = shape vout in
  let rank = Array.length a_shape in

  let axes =
    List.sort_uniq compare
      (List.map (fun ax -> if ax < 0 then ax + rank else ax) axes)
  in

  (* Pre-allocate work arrays to avoid allocations in loop *)
  let out_md_index = Array.make (Array.length out_shape) 0 in
  let in_md_index = Array.make rank 0 in

  for k = start_out_idx to end_out_idx - 1 do
    Shape.unravel_index_into k out_shape out_md_index;

    (* Initialize in_md_index based on out_md_index *)
    init_input_md_index_inplace out_md_index in_md_index axes rank;
    let rec reduce acc =
      let idx =
        View.offset va + Shape.ravel_index in_md_index a_strides
      in
      let acc' = Float32_u.add acc (Array.unsafe_get a_arr idx) in
      if increment_input_md_index_for_reduction in_md_index a_shape axes
      then reduce acc'
      else acc'
    in
    let sum = reduce #0.0s in
  Array.unsafe_set out_arr k sum
  done

let kernel_sum_axis_i64 a_arr out_arr va vout axes start_out_idx end_out_idx =
  let a_shape = shape va in
  let a_strides = View.strides va in
  let out_shape = shape vout in
  let rank = Array.length a_shape in

  let axes =
    List.sort_uniq compare
      (List.map (fun ax -> if ax < 0 then ax + rank else ax) axes)
  in

  (* Pre-allocate work arrays to avoid allocations in loop *)
  let out_md_index = Array.make (Array.length out_shape) 0 in
  let in_md_index = Array.make rank 0 in

  for k = start_out_idx to end_out_idx - 1 do
    Shape.unravel_index_into k out_shape out_md_index;

    (* Initialize in_md_index based on out_md_index *)
    init_input_md_index_inplace out_md_index in_md_index axes rank;

    let rec reduce acc =
      let idx =
        View.offset va + Shape.ravel_index in_md_index a_strides
      in
      let acc' = Int64_u.add acc (Array.unsafe_get a_arr idx) in
      if increment_input_md_index_for_reduction in_md_index a_shape axes
      then reduce acc'
      else acc'
    in
    
    let sum = reduce #0L in
  Array.unsafe_set out_arr k sum
  done

let kernel_sum_axis_i32 a_arr out_arr va vout axes start_out_idx end_out_idx =
  let a_shape = shape va in
  let a_strides = View.strides va in
  let out_shape = shape vout in
  let rank = Array.length a_shape in

  let axes =
    List.sort_uniq compare
      (List.map (fun ax -> if ax < 0 then ax + rank else ax) axes)
  in

  (* Pre-allocate work arrays to avoid allocations in loop *)
  let out_md_index = Array.make (Array.length out_shape) 0 in
  let in_md_index = Array.make rank 0 in

  for k = start_out_idx to end_out_idx - 1 do
    Shape.unravel_index_into k out_shape out_md_index;

    (* Initialize in_md_index based on out_md_index *)
    init_input_md_index_inplace out_md_index in_md_index axes rank;
    let rec reduce acc =
      let idx =
        View.offset va + Shape.ravel_index in_md_index a_strides
      in
      let acc' = Int32_u.add acc (Array.unsafe_get a_arr idx) in
      if increment_input_md_index_for_reduction in_md_index a_shape axes
      then reduce acc'
      else acc'
    in
    
    let sum = reduce #0l in
  Array.unsafe_set out_arr k sum
  done
  let kernel_sum_partial_f64
  a start_linear_idx end_linear_idx=
let a_arr =
  match a.buffer with
  | Float64 arr -> arr
  | _ -> assert false
in
let offset = View.offset a.view in

if View.is_c_contiguous a.view then
  let start_i = offset + start_linear_idx in
  let end_i = offset + end_linear_idx in
  let rec loop i acc =
    if i = end_i then acc
    else
      loop
        (i + 1)
        (Float_u.add acc (Array.unsafe_get a_arr i))
  in
  loop start_i #0.0
else
  let a_shape = shape a.view in
  let a_strides = View.strides a.view in
  let md_index = Array.make (Array.length a_shape) 0 in
  let rec loop k acc =
    if k = end_linear_idx then acc
    else (
      Shape.unravel_index_into k a_shape md_index;
      let idx =
        offset + Shape.ravel_index md_index a_strides
      in
      loop
        (k + 1)
        (Float_u.add acc (Array.unsafe_get a_arr idx))
    )
  in
  loop start_linear_idx #0.0

  let kernel_sum_partial_f32 a start_linear_idx end_linear_idx =
let a_arr =
  match a.buffer with
  | Float32 arr -> arr
  | _ -> assert false
in
let offset = View.offset a.view in

if View.is_c_contiguous a.view then
  let start_i = offset + start_linear_idx in
  let end_i = offset + end_linear_idx in
  let rec loop i acc =
    if i = end_i then acc
    else
      loop
        (i + 1)
        (Float32_u.add acc (Array.unsafe_get a_arr i))
  in
  loop start_i #0.0s
else
  let a_shape = shape a.view in
  let a_strides = View.strides a.view in
  let md_index = Array.make (Array.length a_shape) 0 in
  let rec loop k acc =
    if k = end_linear_idx then acc
    else (
      Shape.unravel_index_into k a_shape md_index;
      let idx =
        offset + Shape.ravel_index md_index a_strides
      in
      loop
        (k + 1)
        (Float32_u.add acc (Array.unsafe_get a_arr idx))
    )
  in
  loop start_linear_idx #0.0s

  let kernel_sum_partial_i64
  a start_linear_idx end_linear_idx =
let a_arr =
  match a.buffer with
  | Int64 arr -> arr
  | _ -> assert false
in
let offset = View.offset a.view in

if View.is_c_contiguous a.view then
  let start_i = offset + start_linear_idx in
  let end_i = offset + end_linear_idx in
  let rec loop i acc =
    if i = end_i then acc
    else
      loop
        (i + 1)
        (Int64_u.add acc (Array.unsafe_get a_arr i))
  in
  loop start_i #0L
else
  let a_shape = shape a.view in
  let a_strides = View.strides a.view in
  let md_index = Array.make (Array.length a_shape) 0 in
  let rec loop k acc =
    if k = end_linear_idx then acc
    else (
      Shape.unravel_index_into k a_shape md_index;
      let idx =
        offset + Shape.ravel_index md_index a_strides
      in
      loop
        (k + 1)
        (Int64_u.add acc (Array.unsafe_get a_arr idx))
    )
  in
  loop start_linear_idx #0L

  let kernel_sum_partial_i32
  a start_linear_idx end_linear_idx =
let a_arr =
  match a.buffer with
  | Int32 arr -> arr
  | _ -> assert false
in
let offset = View.offset a.view in

if View.is_c_contiguous a.view then
  let start_i = offset + start_linear_idx in
  let end_i = offset + end_linear_idx in
  let rec loop i acc =
    if i = end_i then acc
    else
      loop
        (i + 1)
        (Int32_u.add acc (Array.unsafe_get a_arr i))
  in
  loop start_i #0l
else
  let a_shape = shape a.view in
  let a_strides = View.strides a.view in
  let md_index = Array.make (Array.length a_shape) 0 in
  let rec loop k acc =
    if k = end_linear_idx then acc
    else (
      Shape.unravel_index_into k a_shape md_index;
      let idx =
        offset + Shape.ravel_index md_index a_strides
      in
      loop
        (k + 1)
        (Int32_u.add acc (Array.unsafe_get a_arr idx))
    )
  in
  loop start_linear_idx #0l

  let parallel_sum_all_f64 context a =
    let pool = context.pool in
    let numel =
      Array.fold_left ( * ) 1 (shape a.view)
    in
    if numel = 0 then 0.0
    else
      let body start_idx end_idx =
        if start_idx < end_idx then
          kernel_sum_partial_f64 a start_idx end_idx
        else
          #0.0
      in
      let reduce x y = Float_u.add x y in
      Parallel.parallel_for_reduce
        pool
        0
        numel
        body
        reduce
        #0.0
  
        let parallel_sum_all_f32 context a =
          let pool = context.pool in
          let numel =
            Array.fold_left ( * ) 1 (shape a)
          in
          if numel = 0 then 0.0s
          else
            let body start_idx end_idx =
              if start_idx < end_idx then
                kernel_sum_partial_f32 a start_idx end_idx
              else
                #0.0s
            in
            let reduce x y = Float32_u.add x y in
            Parallel.parallel_for_reduce
              pool
              0
              numel
              body
              reduce
              #0.0s

              let parallel_sum_all_i64 context a =
                let pool = context.pool in
                let numel =
                  Array.fold_left ( * ) 1 (shape a)
                in
                if numel = 0 then 0L
                else
                  let body start_idx end_idx =
                    if start_idx < end_idx then
                      kernel_sum_partial_i64 a start_idx end_idx
                    else
                      #0L
                  in
                  let reduce x y = Int64_u.add x y in
                  Parallel.parallel_for_reduce
                    pool
                    0
                    numel
                    body
                    reduce
                    #0L

                    let parallel_sum_all_i32 context (a : (int32, b) t) : int32 =
                      let pool = context.pool in
                      let numel =
                        Array.fold_left ( * ) 1 (shape a)
                      in
                      if numel = 0 then 0l
                      else
                        let body start_idx end_idx =
                          if start_idx < end_idx then
                            kernel_sum_partial_i32 a start_idx end_idx
                          else
                            #0l
                        in
                        let reduce x y = Int32_u.add x y in
                        Parallel.parallel_for_reduce
                          pool
                          0
                          numel
                          body
                          reduce
                          #0l
                    

let sum (type a b) context ~axes ~keepdims (a : (a, b) t) (out : (a, b) t) =
  let in_shape = shape a in
  let rank = Array.length in_shape in
  let axes_to_reduce =
    List.sort_uniq compare
      (List.map
         (fun ax ->
           let ax' = if ax < 0 then ax + rank else ax in
           if ax' < 0 || ax' >= rank then
             invalid_arg
               (Printf.sprintf "sum: invalid axis %d for tensor with rank %d" ax
                  rank)
           else ax')
         (Array.to_list axes))
  in
  if axes_to_reduce = [] then blit a out
  else if rank = 0 then blit a out
  else
    let out_shape = reduction_output_shape in_shape axes_to_reduce keepdims in
    let num_output_elements = Array.fold_left ( * ) 1 out_shape in
    let num_input_elements = Array.fold_left ( * ) 1 in_shape in
    if num_output_elements = 1 && num_input_elements > 0 then
      let total_sum_value : a = parallel_sum_all context a in
      let out_buf = out.buffer in
      Array.unsafe_set out_buf (offset out) total_sum_value
    else if num_output_elements > 0 && num_input_elements > 0 then
      Parallel.parallel_for context.pool 0 (num_output_elements - 1)
        (fun start_idx end_idx ->
          kernel_sum_axis a out axes_to_reduce start_idx end_idx)
