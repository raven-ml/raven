open Bigarray
open Nx_core
open Internal

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

(* Creates an initial multi-dimensional index for the input tensor based on an
   output index and the axes being reduced. *)
let initial_input_md_index (out_md_index : int array) (axes : int list)
    (rank : int) : int array =
  let in_md_index = Array.make rank 0 in
  if Array.length out_md_index = rank then (
    (* keepdims = true *)
    for d = 0 to rank - 1 do
      if List.mem d axes then in_md_index.(d) <- 0
      else in_md_index.(d) <- out_md_index.(d)
    done;
    in_md_index)
  else
    (* keepdims = false *)
    let non_reduced_pos = ref 0 in
    for d = 0 to rank - 1 do
      if List.mem d axes then in_md_index.(d) <- 0
      else (
        in_md_index.(d) <- out_md_index.(!non_reduced_pos);
        incr non_reduced_pos)
    done;
    in_md_index

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

let add_dtype (type a b) (dtype : (a, b) dtype) (a : a) (b : a) : a =
  match dtype with
  | Float16 -> Float.add a b
  | Float32 -> Float.add a b
  | Float64 -> Float.add a b
  | Int8 -> Int.add a b
  | Int16 -> Int.add a b
  | Int32 -> Int32.add a b
  | Int64 -> Int64.add a b
  | UInt8 -> Int.add a b
  | UInt16 -> Int.add a b
  | Complex32 -> Complex.add a b
  | Complex64 -> Complex.add a b

let kernel_sum_axis (type a b) (a : (a, b) t) (out : (a, b) t) (axes : int list)
    start_out_idx end_out_idx =
  let a_buf = buffer a in
  let out_buf = buffer out in
  let a_shape = shape a in
  let a_strides = strides a in
  let out_shape = shape out in
  let rank = Array.length a_shape in

  let axes =
    List.sort_uniq compare
      (List.map (fun ax -> if ax < 0 then ax + rank else ax) axes)
  in

  for k = start_out_idx to end_out_idx - 1 do
    let out_md_index = linear_to_md_c_contig k out_shape in
    let current_sum = ref (zero (dtype out)) in
    let in_md_index = initial_input_md_index out_md_index axes rank in
    let continue_reduction = ref true in
    while !continue_reduction do
      let a_linear_idx = md_to_linear in_md_index a_strides in
      let a_val = Array1.unsafe_get a_buf (offset a + a_linear_idx) in
      current_sum := add_dtype (dtype a) !current_sum a_val;
      continue_reduction :=
        increment_input_md_index_for_reduction in_md_index a_shape axes
    done;
    Array1.unsafe_set out_buf k !current_sum
  done

let kernel_sum_partial (type a b) (a : (a, b) t) (start_linear_idx : int)
    (end_linear_idx : int) : a =
  let a_buf = buffer a in
  let a_offset = offset a in
  let partial_sum = ref (zero (dtype a)) in
  (if is_c_contiguous a then
     let start_buf_idx = a_offset + start_linear_idx in
     let end_buf_idx = a_offset + end_linear_idx in
     for i = start_buf_idx to end_buf_idx - 1 do
       partial_sum :=
         add_dtype (dtype a) !partial_sum (Array1.unsafe_get a_buf i)
     done
   else
     let a_shape = shape a in
     let a_strides = strides a in
     for k = start_linear_idx to end_linear_idx - 1 do
       let md_index = linear_to_md_c_contig k a_shape in
       let buf_idx = md_to_linear md_index a_strides in
       let v = Array1.unsafe_get a_buf (a_offset + buf_idx) in
       partial_sum := add_dtype (dtype a) !partial_sum v
     done);

  !partial_sum

let parallel_sum_all (type a b) context (a : (a, b) t) : a =
  let pool = context.pool in
  let num_input_elements = Array.fold_left ( * ) 1 (shape a) in
  if num_input_elements = 0 then zero (dtype a)
  else
    let zero_val = zero (dtype a) in
    let body start_idx end_idx =
      if start_idx < end_idx then kernel_sum_partial a start_idx end_idx
      else zero_val
    in
    let reduce x y = add_dtype (dtype a) x y in
    Parallel.parallel_for_reduce pool 0 (num_input_elements - 1) body reduce
      zero_val

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
      let out_buf = buffer out in
      Array1.unsafe_set out_buf (offset out) total_sum_value
    else if num_output_elements > 0 && num_input_elements > 0 then
      Parallel.parallel_for context.pool 0 (num_output_elements - 1)
        (fun start_idx end_idx ->
          kernel_sum_axis a out axes_to_reduce start_idx end_idx)

(* prod *)

let mul_dtype (type a b) (dtype : (a, b) dtype) (a : a) (b : a) : a =
  match dtype with
  | Float16 -> Float.mul a b
  | Float32 -> Float.mul a b
  | Float64 -> Float.mul a b
  | Int8 -> Int.mul a b
  | Int16 -> Int.mul a b
  | Int32 -> Int32.mul a b
  | Int64 -> Int64.mul a b
  | UInt8 -> Int.mul a b
  | UInt16 -> Int.mul a b
  | Complex32 -> Complex.mul a b
  | Complex64 -> Complex.mul a b

let kernel_prod_axis (type a b) (a : (a, b) t) (out : (a, b) t)
    (axes : int list) start_out_idx end_out_idx =
  let a_buf = buffer a in
  let out_buf = buffer out in
  let a_shape = shape a in
  let a_strides = strides a in
  let out_shape = shape out in
  let rank = Array.length a_shape in

  let axes =
    List.sort_uniq compare
      (List.map (fun ax -> if ax < 0 then ax + rank else ax) axes)
  in

  for k = start_out_idx to end_out_idx - 1 do
    let out_md_index = linear_to_md_c_contig k out_shape in
    let current_prod = ref (one (dtype out)) in
    let in_md_index = initial_input_md_index out_md_index axes rank in
    let continue_reduction = ref true in
    while !continue_reduction do
      let a_linear_idx = md_to_linear in_md_index a_strides in
      let a_val = Array1.unsafe_get a_buf (offset a + a_linear_idx) in
      current_prod := mul_dtype (dtype a) !current_prod a_val;
      continue_reduction :=
        increment_input_md_index_for_reduction in_md_index a_shape axes
    done;
    Array1.unsafe_set out_buf k !current_prod
  done

let kernel_prod_partial (type a b) (a : (a, b) t) (start_linear_idx : int)
    (end_linear_idx : int) : a =
  let a_buf = buffer a in
  let a_offset = offset a in
  let partial_prod = ref (one (dtype a)) in
  (if is_c_contiguous a then
     let start_buf_idx = a_offset + start_linear_idx in
     let end_buf_idx = a_offset + end_linear_idx in
     for i = start_buf_idx to end_buf_idx - 1 do
       partial_prod :=
         mul_dtype (dtype a) !partial_prod (Array1.unsafe_get a_buf i)
     done
   else
     let a_shape = shape a in
     let a_strides = strides a in
     for k = start_linear_idx to end_linear_idx - 1 do
       let md_index = linear_to_md_c_contig k a_shape in
       let buf_idx = md_to_linear md_index a_strides in
       let v = Array1.unsafe_get a_buf (a_offset + buf_idx) in
       partial_prod := mul_dtype (dtype a) !partial_prod v
     done);
  !partial_prod

let parallel_prod_all (type a b) context (a : (a, b) t) : a =
  let pool = context.pool in
  let num_input_elements = Array.fold_left ( * ) 1 (shape a) in
  if num_input_elements = 0 then one (dtype a)
  else
    let one_val = one (dtype a) in
    let body start_idx end_idx =
      if start_idx < end_idx then kernel_prod_partial a start_idx end_idx
      else one_val
    in
    let reduce x y = mul_dtype (dtype a) x y in
    Parallel.parallel_for_reduce pool 0 (num_input_elements - 1) body reduce
      one_val

let prod (type a b) context ~axes ~keepdims (a : (a, b) t) (out : (a, b) t) =
  let in_shape = shape a in
  let rank = Array.length in_shape in
  let axes_to_reduce =
    List.sort_uniq compare
      (List.map
         (fun ax ->
           let ax' = if ax < 0 then ax + rank else ax in
           if ax' < 0 || ax' >= rank then
             invalid_arg
               (Printf.sprintf "prod: invalid axis %d for tensor with rank %d"
                  ax rank)
           else ax')
         (Array.to_list axes))
  in
  if axes_to_reduce = [] then blit a out
  else if rank = 0 then blit a out
  else
    let out_shape = reduction_output_shape in_shape axes_to_reduce keepdims in
    let num_output_elements = Array.fold_left ( * ) 1 out_shape in
    let num_input_elements = Array.fold_left ( * ) 1 in_shape in
    if num_output_elements = 1 then
      let value =
        if num_input_elements = 0 then one (dtype a)
        else parallel_prod_all context a
      in
      fill value out
    else if num_output_elements > 0 && num_input_elements > 0 then
      if num_input_elements = 0 then fill (one (dtype a)) out
      else if num_output_elements > 0 && num_input_elements > 0 then
        Parallel.parallel_for context.pool 0 (num_output_elements - 1)
          (fun start_idx end_idx ->
            kernel_prod_axis a out axes_to_reduce start_idx end_idx)

(* min *)

let min_dtype (type a b) (dtype : (a, b) dtype) (a : a) (b : a) : a =
  match dtype with
  | Float16 -> Float.min a b
  | Float32 -> Float.min a b
  | Float64 -> Float.min a b
  | Int8 -> Int.min a b
  | Int16 -> Int.min a b
  | Int32 -> Int32.min a b
  | Int64 -> Int64.min a b
  | UInt8 -> Int.min a b
  | UInt16 -> Int.min a b
  | Complex32 ->
      if a.re < b.re then a
      else if a.re > b.re then b
      else if a.im < b.im then a
      else b
  | Complex64 ->
      if a.re < b.re then a
      else if a.re > b.re then b
      else if a.im < b.im then a
      else b

let kernel_min_axis (type a b) (a : (a, b) t) (out : (a, b) t) (axes : int list)
    start_out_idx end_out_idx =
  let a_buf = buffer a in
  let out_buf = buffer out in
  let a_shape = shape a in
  let a_strides = strides a in
  let out_shape = shape out in
  let rank = Array.length a_shape in
  let axes =
    List.sort_uniq compare
      (List.map (fun ax -> if ax < 0 then ax + rank else ax) axes)
  in

  for k = start_out_idx to end_out_idx - 1 do
    let out_md_index = linear_to_md_c_contig k out_shape in
    let current_min = ref None in
    let is_first = ref true in
    let in_md_index = initial_input_md_index out_md_index axes rank in
    let continue_reduction = ref true in

    while !continue_reduction do
      let a_linear_idx = md_to_linear in_md_index a_strides in
      let a_val = Array1.unsafe_get a_buf (offset a + a_linear_idx) in
      (if !is_first then (
         current_min := Some a_val;
         is_first := false)
       else
         let stored_val = Option.get !current_min in
         current_min := Some (min_dtype (dtype a) stored_val a_val));
      continue_reduction :=
        increment_input_md_index_for_reduction in_md_index a_shape axes
    done;

    match !current_min with
    | Some final_val -> Array1.unsafe_set out_buf k final_val
    | None ->
        invalid_arg
          "kernel_min_axis: Reduction over zero elements resulted in no value."
  done

let kernel_min_partial (type a b) (a : (a, b) t) (start_linear_idx : int)
    (end_linear_idx : int) : a =
  let a_buf = buffer a in
  let a_offset = offset a in
  let partial_min = ref None in
  let is_first = ref true in

  if start_linear_idx >= end_linear_idx then
    invalid_arg "kernel_min_partial: Reduction over zero elements.";

  (if is_c_contiguous a then
     let start_buf_idx = a_offset + start_linear_idx in
     let end_buf_idx = a_offset + end_linear_idx in
     for i = start_buf_idx to end_buf_idx - 1 do
       let v = Array1.unsafe_get a_buf i in
       if !is_first then (
         partial_min := Some v;
         is_first := false)
       else
         let stored_val = Option.get !partial_min in
         partial_min := Some (min_dtype (dtype a) stored_val v)
     done
   else
     let a_shape = shape a in
     let a_strides = strides a in
     for k = start_linear_idx to end_linear_idx - 1 do
       let md_index = linear_to_md_c_contig k a_shape in
       let buf_idx = md_to_linear md_index a_strides in
       let v = Array1.unsafe_get a_buf (a_offset + buf_idx) in
       if !is_first then (
         partial_min := Some v;
         is_first := false)
       else
         let stored_val = Option.get !partial_min in
         partial_min := Some (min_dtype (dtype a) stored_val v)
     done);

  match !partial_min with
  | Some final_val -> final_val
  | None ->
      invalid_arg "kernel_min_partial: Failed to find minimum (logic error)."

let parallel_min_all (type a b) context (a : (a, b) t) : a =
  let pool = context.pool in
  let num_input_elements = Array.fold_left ( * ) 1 (shape a) in
  if num_input_elements = 0 then invalid_arg "min: zero-size array reduction"
  else
    let body start_idx end_idx =
      if start_idx < end_idx then Some (kernel_min_partial a start_idx end_idx)
      else None
    in
    let reduce opt_x opt_y =
      match (opt_x, opt_y) with
      | None, None -> None
      | Some x, None -> Some x
      | None, Some y -> Some y
      | Some x, Some y -> Some (min_dtype (dtype a) x y)
    in
    match
      Parallel.parallel_for_reduce pool 0 (num_input_elements - 1) body reduce
        None
    with
    | Some v -> v
    | None -> invalid_arg "parallel_min_all: Could not find minimum."

let min (type a b) context ~axes ~keepdims (a : (a, b) t) (out : (a, b) t) =
  let in_shape = shape a in
  let rank = Array.length in_shape in
  let num_input_elements = Array.fold_left ( * ) 1 in_shape in

  let axes_to_reduce =
    List.sort_uniq compare
      (List.map
         (fun ax ->
           let ax' = if ax < 0 then ax + rank else ax in
           if ax' < 0 || ax' >= rank then
             invalid_arg
               (Printf.sprintf "min: invalid axis %d for tensor with rank %d" ax
                  rank)
           else ax')
         (Array.to_list axes))
  in

  if axes_to_reduce = [] then blit a out
  else if rank = 0 then
    invalid_arg (Printf.sprintf "min: invalid axis for 0-dimensional tensor")
  else if num_input_elements = 0 then
    invalid_arg "min: zero-size array reduction"
  else
    let out_shape = reduction_output_shape in_shape axes_to_reduce keepdims in
    let num_output_elements = Array.fold_left ( * ) 1 out_shape in

    if num_output_elements = 1 then
      let total_min_value : a = parallel_min_all context a in
      fill total_min_value out
    else if num_output_elements > 0 && num_input_elements > 0 then
      Parallel.parallel_for context.pool 0 (num_output_elements - 1)
        (fun start_idx end_idx ->
          kernel_min_axis a out axes_to_reduce start_idx end_idx)

(* max *)

let min_identity (type a b) (dtype : (a, b) dtype) : a =
  match dtype with
  | Float16 -> Float.neg_infinity
  | Float32 -> Float.neg_infinity
  | Float64 -> Float.neg_infinity
  | Int8 -> min_int
  | Int16 -> min_int
  | Int32 -> Int32.min_int
  | Int64 -> Int64.min_int
  | UInt8 -> 0
  | UInt16 -> 0
  | Complex32 -> { re = Float.neg_infinity; im = Float.neg_infinity }
  | Complex64 -> { re = Float.neg_infinity; im = Float.neg_infinity }

let max_dtype (type a b) (dtype : (a, b) dtype) (a : a) (b : a) : a =
  match dtype with
  | Float16 -> Float.max a b
  | Float32 -> Float.max a b
  | Float64 -> Float.max a b
  | Int8 -> Int.max a b
  | Int16 -> Int.max a b
  | Int32 -> Int32.max a b
  | Int64 -> Int64.max a b
  | UInt8 -> Int.max a b
  | UInt16 -> Int.max a b
  | Complex32 ->
      if a.re > b.re then a
      else if a.re < b.re then b
      else if a.im > b.im then a
      else b
  | Complex64 ->
      if a.re > b.re then a
      else if a.re < b.re then b
      else if a.im > b.im then a
      else b

let kernel_max_axis (type a b) (a : (a, b) t) (out : (a, b) t) (axes : int list)
    start_out_idx end_out_idx =
  let a_buf = buffer a in
  let out_buf = buffer out in
  let a_shape = shape a in
  let a_strides = strides a in
  let out_shape = shape out in
  let rank = Array.length a_shape in
  let axes =
    List.sort_uniq compare
      (List.map (fun ax -> if ax < 0 then ax + rank else ax) axes)
  in
  let min_identity_val = min_identity (dtype a) in

  for k = start_out_idx to end_out_idx - 1 do
    let out_md_index = linear_to_md_c_contig k out_shape in
    let current_max = ref min_identity_val in
    let is_first = ref true in
    let in_md_index = initial_input_md_index out_md_index axes rank in
    let continue_reduction = ref true in
    let has_value = ref false in

    while !continue_reduction do
      let a_linear_idx = md_to_linear in_md_index a_strides in
      let a_val = Array1.unsafe_get a_buf (offset a + a_linear_idx) in
      has_value := true;
      if !is_first then (
        current_max := a_val;
        is_first := false)
      else current_max := max_dtype (dtype a) !current_max a_val;
      continue_reduction :=
        increment_input_md_index_for_reduction in_md_index a_shape axes
    done;

    if not !has_value then
      invalid_arg
        "kernel_max_axis: Reduction over zero elements resulted in no value."
    else Array1.unsafe_set out_buf k !current_max
  done

let kernel_max_partial (type a b) (a : (a, b) t) (start_linear_idx : int)
    (end_linear_idx : int) : a =
  let a_buf = buffer a in
  let a_offset = offset a in
  let min_identity_val = min_identity (dtype a) in
  let partial_max = ref min_identity_val in
  let is_first = ref true in
  let has_value = ref false in

  if start_linear_idx >= end_linear_idx then
    invalid_arg "kernel_max_partial: Reduction over zero elements.";

  (if is_c_contiguous a then
     let start_buf_idx = a_offset + start_linear_idx in
     let end_buf_idx = a_offset + end_linear_idx in
     for i = start_buf_idx to end_buf_idx - 1 do
       let v = Array1.unsafe_get a_buf i in
       has_value := true;
       if !is_first then (
         partial_max := v;
         is_first := false)
       else partial_max := max_dtype (dtype a) !partial_max v
     done
   else
     let a_shape = shape a in
     let a_strides = strides a in
     for k = start_linear_idx to end_linear_idx - 1 do
       let md_index = linear_to_md_c_contig k a_shape in
       let buf_idx = md_to_linear md_index a_strides in
       let v = Array1.unsafe_get a_buf (a_offset + buf_idx) in
       has_value := true;
       if !is_first then (
         partial_max := v;
         is_first := false)
       else partial_max := max_dtype (dtype a) !partial_max v
     done);

  if not !has_value then (* Should be unreachable if range > 0 *)
    invalid_arg "kernel_max_partial: Failed to find maximum (logic error)."
  else !partial_max

let parallel_max_all (type a b) context (a : (a, b) t) : a =
  let pool = context.pool in
  let num_input_elements = Array.fold_left ( * ) 1 (shape a) in
  if num_input_elements = 0 then invalid_arg "max: zero-size array reduction"
  else
    let body start_idx end_idx =
      if start_idx < end_idx then Some (kernel_max_partial a start_idx end_idx)
      else None
    in
    let reduce opt_x opt_y =
      match (opt_x, opt_y) with
      | None, None -> None
      | Some x, None -> Some x
      | None, Some y -> Some y
      | Some x, Some y -> Some (max_dtype (dtype a) x y)
    in
    match
      Parallel.parallel_for_reduce pool 0 (num_input_elements - 1) body reduce
        None
    with
    | Some v -> v
    | None -> invalid_arg "parallel_max_all: Could not find maximum."

let max (type a b) context ~axes ~keepdims (a : (a, b) t) (out : (a, b) t) =
  let in_shape = shape a in
  let rank = Array.length in_shape in
  let num_input_elements = Array.fold_left ( * ) 1 in_shape in

  let axes_to_reduce =
    List.sort_uniq compare
      (List.map
         (fun ax ->
           let ax' = if ax < 0 then ax + rank else ax in
           if ax' < 0 || ax' >= rank then
             invalid_arg
               (Printf.sprintf "max: invalid axis %d for tensor with rank %d" ax
                  rank)
           else ax')
         (Array.to_list axes))
  in

  if axes_to_reduce = [] then blit a out
  else if rank = 0 then
    invalid_arg (Printf.sprintf "max: invalid axis for 0-dimensional tensor")
  else if num_input_elements = 0 then
    invalid_arg "max: zero-size array reduction"
  else
    let out_shape = reduction_output_shape in_shape axes_to_reduce keepdims in
    let num_output_elements = Array.fold_left ( * ) 1 out_shape in

    if num_output_elements = 1 then
      let total_max_value : a = parallel_max_all context a in
      fill total_max_value out
    else if num_output_elements > 0 && num_input_elements > 0 then
      Parallel.parallel_for context.pool 0 (num_output_elements - 1)
        (fun start_idx end_idx ->
          kernel_max_axis a out axes_to_reduce start_idx end_idx)

(* var *)

let float_of (type a b) (dtype : (a, b) dtype) (v : a) : float =
  match dtype with
  | Float16 -> v
  | Float32 -> v
  | Float64 -> v
  | Int8 -> float_of_int v
  | Int16 -> float_of_int v
  | Int32 -> Int32.to_float v
  | Int64 -> Int64.to_float v
  | UInt8 -> float_of_int v
  | UInt16 -> float_of_int v
  | Complex32 -> v.re
  | Complex64 -> v.re

let kernel_sum_count_partial (type a b) (a : (a, b) t) (start_linear_idx : int)
    (end_linear_idx : int) : a * int =
  let a_buf = buffer a in
  let a_offset = offset a in
  let partial_sum = ref (zero (dtype a)) in
  let count = ref 0 in

  (if is_c_contiguous a then (
     let start_buf_idx = a_offset + start_linear_idx in
     let end_buf_idx = a_offset + end_linear_idx in
     let actual_count = Stdlib.max 0 (end_buf_idx - start_buf_idx) in
     count := actual_count;
     for i = start_buf_idx to end_buf_idx - 1 do
       partial_sum :=
         add_dtype (dtype a) !partial_sum (Array1.unsafe_get a_buf i)
     done)
   else
     let a_shape = shape a in
     let a_strides = strides a in
     let actual_count = Stdlib.max 0 (end_linear_idx - start_linear_idx) in
     count := actual_count;
     for k = start_linear_idx to end_linear_idx - 1 do
       let md_index = linear_to_md_c_contig k a_shape in
       let buf_idx = md_to_linear md_index a_strides in
       let v = Array1.unsafe_get a_buf (a_offset + buf_idx) in
       partial_sum := add_dtype (dtype a) !partial_sum v
     done);

  (!partial_sum, !count)

let sq_dtype (type a b) (dtype : (a, b) dtype) (a : a) : a =
  match dtype with
  | Float16 -> Float.mul a a
  | Float32 -> Float.mul a a
  | Float64 -> Float.mul a a
  | Int8 -> Int.mul a a
  | Int16 -> Int.mul a a
  | Int32 -> Int32.mul a a
  | Int64 -> Int64.mul a a
  | UInt8 -> Int.mul a a
  | UInt16 -> Int.mul a a
  | Complex32 -> Complex.mul a a
  | Complex64 -> Complex.mul a a

let kernel_sum_sq_count_partial (type a b) (a : (a, b) t)
    (start_linear_idx : int) (end_linear_idx : int) : a * a * int =
  (* sum, sum_sq, count *)
  let a_buf = buffer a in
  let a_offset = offset a in
  let partial_sum = ref (zero (dtype a)) in
  let partial_sum_sq = ref (zero (dtype a)) in
  let count = ref 0 in

  (if is_c_contiguous a then (
     let start_buf_idx = a_offset + start_linear_idx in
     let end_buf_idx = a_offset + end_linear_idx in
     let actual_count = Stdlib.max 0 (end_buf_idx - start_buf_idx) in
     count := actual_count;
     for i = start_buf_idx to end_buf_idx - 1 do
       let v = Array1.unsafe_get a_buf i in
       partial_sum := add_dtype (dtype a) !partial_sum v;
       partial_sum_sq :=
         add_dtype (dtype a) !partial_sum_sq (sq_dtype (dtype a) v)
     done)
   else
     let a_shape = shape a in
     let a_strides = strides a in
     let actual_count = Stdlib.max 0 (end_linear_idx - start_linear_idx) in
     count := actual_count;
     for k = start_linear_idx to end_linear_idx - 1 do
       let md_index = linear_to_md_c_contig k a_shape in
       let buf_idx = md_to_linear md_index a_strides in
       let v = Array1.unsafe_get a_buf (a_offset + buf_idx) in
       partial_sum := add_dtype (dtype a) !partial_sum v;
       partial_sum_sq :=
         add_dtype (dtype a) !partial_sum_sq (sq_dtype (dtype a) v)
     done);

  (!partial_sum, !partial_sum_sq, !count)

let kernel_var_axis (type b) (a : (float, b) t) (out : (float, b) t)
    (axes : int list) (ddof : int) start_out_idx end_out_idx =
  let a_buf = buffer a in
  let out_buf = buffer out in
  let a_shape = shape a in
  let a_strides = strides a in
  let out_shape = shape out in
  let rank = Array.length a_shape in
  let axes =
    List.sort_uniq compare
      (List.map (fun ax -> if ax < 0 then ax + rank else ax) axes)
  in
  let nan_val = Float.nan in
  let ddof_float = float_of_int ddof in

  for k = start_out_idx to end_out_idx - 1 do
    let out_md_index = linear_to_md_c_contig k out_shape in
    let current_sum = ref (zero (dtype a)) in
    let current_sum_sq = ref (zero (dtype a)) in
    let current_count = ref 0 in
    let in_md_index = initial_input_md_index out_md_index axes rank in
    let continue_reduction = ref true in

    while !continue_reduction do
      let a_linear_idx = md_to_linear in_md_index a_strides in
      let a_val = Array1.unsafe_get a_buf (offset a + a_linear_idx) in
      current_sum := add_dtype (dtype a) !current_sum a_val;
      current_sum_sq :=
        add_dtype (dtype a) !current_sum_sq (sq_dtype (dtype a) a_val);
      incr current_count;
      continue_reduction :=
        increment_input_md_index_for_reduction in_md_index a_shape axes
    done;

    let count_float = float_of_int !current_count in
    let adjusted_count_float = count_float -. ddof_float in

    let final_var =
      if adjusted_count_float <= 0.0 then nan_val
        (* Avoid division by zero/negative *)
      else
        let mean = float_of (dtype a) !current_sum /. count_float in
        let mean_sq = float_of (dtype a) !current_sum_sq /. count_float in
        let variance = mean_sq -. (mean *. mean) in
        (* Apply Bessel's correction if count > ddof, clamp negative variance *)
        Stdlib.max 0.0 (variance *. (count_float /. adjusted_count_float))
    in
    Array1.unsafe_set out_buf k final_var
  done

let parallel_var_all (type a b) context (a : (a, b) t) (ddof : int) : float =
  let pool = context.pool in
  let num_input_elements = Array.fold_left ( * ) 1 (shape a) in
  let nan_val = Float.nan in
  let ddof_float = float_of_int ddof in

  if num_input_elements = 0 then nan_val
  else
    let zero_val = zero (dtype a) in
    (* Body function: Compute partial results for a chunk *)
    let body start_idx end_idx =
      if start_idx < end_idx then
        kernel_sum_sq_count_partial a start_idx end_idx
      else (zero_val, zero_val, 0)
    in
    (* Reduce function: Combine two triples *)
    let reduce (sum1, sum_sq1, count1) (sum2, sum_sq2, count2) =
      ( add_dtype (dtype a) sum1 sum2,
        add_dtype (dtype a) sum_sq1 sum_sq2,
        count1 + count2 )
    in
    (* Perform parallel reduction *)
    let total_sum, total_sum_sq, total_count =
      Parallel.parallel_for_reduce pool 0 (num_input_elements - 1) body reduce
        (zero_val, zero_val, 0)
    in
    (* Compute variance *)
    let count_float = float_of_int total_count in
    let adjusted_count_float = count_float -. ddof_float in
    if adjusted_count_float <= 0.0 then nan_val
    else
      let mean = float_of (dtype a) total_sum /. count_float in
      let mean_sq = float_of (dtype a) total_sum_sq /. count_float in
      let variance = mean_sq -. (mean *. mean) in
      Stdlib.max 0.0 (variance *. (count_float /. adjusted_count_float))

let var (type b) context ~axes ~keepdims ?(ddof : int = 0) (a : (float, b) t)
    (out : (float, b) t) =
  let in_shape = shape a in
  let rank = Array.length in_shape in
  let num_input_elements = Array.fold_left ( * ) 1 in_shape in
  let nan_val = Float.nan in

  let axes_to_reduce =
    List.sort_uniq compare
      (List.map
         (fun ax ->
           let ax' = if ax < 0 then ax + rank else ax in
           if ax' < 0 || ax' >= rank then
             invalid_arg
               (Printf.sprintf "var: invalid axis %d for tensor with rank %d" ax
                  rank)
           else ax')
         (Array.to_list axes))
  in

  if axes_to_reduce = [] then fill 0. out
  else if rank = 0 then
    invalid_arg (Printf.sprintf "var: invalid axis for 0-dimensional tensor")
  else
    let out_shape = reduction_output_shape in_shape axes_to_reduce keepdims in
    let num_output_elements = Array.fold_left ( * ) 1 out_shape in

    if
      num_input_elements = 0
      || float_of_int num_input_elements <= float_of_int ddof
    then fill nan_val out
    else if num_output_elements = 1 then
      let total_var_value : float = parallel_var_all context a ddof in
      fill total_var_value out
    else if num_output_elements > 0 && num_input_elements > 0 then
      Parallel.parallel_for context.pool 0 (num_output_elements - 1)
        (fun start_idx end_idx ->
          kernel_var_axis a out axes_to_reduce ddof start_idx end_idx)
