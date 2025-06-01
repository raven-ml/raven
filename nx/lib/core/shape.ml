type t = int array

let numel shape =
  let n = Array.length shape in
  if n = 0 then 1 else Array.fold_left ( * ) 1 shape

let equal = ( = )

let c_contiguous_strides shape =
  let n = Array.length shape in
  if n = 0 then [||]
  else
    let strides = Array.make n 0 in
    strides.(n - 1) <- (if shape.(n - 1) = 0 then 0 else 1);
    for i = n - 2 downto 0 do
      strides.(i) <-
        (if shape.(i) = 0 then 0 else strides.(i + 1) * max 1 shape.(i + 1))
    done;
    strides

let ravel_index indices strides =
  if Array.length indices <> Array.length strides then
    Error.invalid ~op:"ravel_index"
      ~what:
        (Printf.sprintf "rank mismatch: indices[%d] vs strides[%d]"
           (Array.length indices) (Array.length strides))
      ~reason:"dimensions must match" ();
  let o = ref 0 in
  Array.iteri (fun i v -> o := !o + (v * strides.(i))) indices;
  !o

let unravel_index k shape =
  let n = Array.length shape in
  if n = 0 then
    if k = 0 then [||]
    else
      Error.invalid ~op:"unravel_index" ~what:"k"
        ~reason:(Printf.sprintf "%d out of bounds for scalar" k)
        ()
  else if Array.exists (( = ) 0) shape then
    (* zero-size tensor; only k=0 is allowed *)
    if k = 0 then Array.make n 0
    else
      Error.invalid ~op:"unravel_index" ~what:"k"
        ~reason:(Printf.sprintf "%d > 0 for zero-size shape" k)
        ()
  else
    let total_elements = numel shape in
    if k < 0 || k >= total_elements then
      Error.invalid ~op:"unravel_index" ~what:"k"
        ~reason:
          (Printf.sprintf "%d out of bounds for shape (size %d)" k
             total_elements)
        ();

    let idx = Array.make n 0 in
    let temp_k = ref k in
    for i = n - 1 downto 1 do
      let dim_size = shape.(i) in
      idx.(i) <- !temp_k mod dim_size;
      temp_k := !temp_k / dim_size
    done;
    idx.(0) <- !temp_k;

    (* sanity check for the leftmost index *)
    if idx.(0) >= shape.(0) then
      Error.invalid ~op:"unravel_index"
        ~what:(Printf.sprintf "calculated idx.(0)=%d" idx.(0))
        ~reason:(Printf.sprintf "out of bounds for shape.(0)=%d" shape.(0))
        ~hint:"this indicates an issue with k or logic" ();
    idx

let resolve_neg_one current_shape new_shape_spec =
  let new_shape_spec_l = Array.to_list new_shape_spec in
  let current_numel = numel current_shape in
  let neg_one_count =
    new_shape_spec_l |> List.filter (( = ) (-1)) |> List.length
  in
  if neg_one_count > 1 then
    Error.invalid ~op:"reshape" ~what:"shape specification"
      ~reason:"multiple -1 dimensions"
      ~hint:"can only specify one unknown dimension" ()
  else if neg_one_count = 0 then new_shape_spec
  else
    let specified_numel =
      List.filter (( <> ) (-1)) new_shape_spec_l |> Array.of_list |> numel
    in
    (* when shape_spec includes zero dimensions *)
    if specified_numel = 0 then
      if current_numel = 0 then
        Array.map (fun x -> if x = -1 then 0 else x) new_shape_spec
      else
        Error.cannot ~op:"reshape" ~what:"infer -1"
          ~from:"shape with 0-size dimensions" ~to_:"non-zero total size"
          ~reason:"incompatible dimensions" ()
    else if current_numel mod specified_numel <> 0 then
      Error.cannot ~op:"reshape" ~what:"reshape"
        ~from:(Printf.sprintf "%d elements" current_numel)
        ~to_:(Printf.sprintf "shape with %d elements" specified_numel)
        ~reason:"size mismatch" ()
    else
      let inferred_dim = current_numel / specified_numel in
      Array.map (fun s -> if s = -1 then inferred_dim else s) new_shape_spec

let broadcast shape_a shape_b =
  let rank_a = Array.length shape_a and rank_b = Array.length shape_b in
  let rank_out = max rank_a rank_b in
  let out_shape = Array.make rank_out 1 in
  for i = 0 to rank_out - 1 do
    let dim_a =
      if i < rank_out - rank_a then 1 else shape_a.(i - (rank_out - rank_a))
    in
    let dim_b =
      if i < rank_out - rank_b then 1 else shape_b.(i - (rank_out - rank_b))
    in
    if dim_a = dim_b then out_shape.(i) <- dim_a
    else if dim_a = 1 then out_shape.(i) <- dim_b
    else if dim_b = 1 then out_shape.(i) <- dim_a
    else
      Error.broadcast_incompatible ~op:"broadcast" ~shape1:shape_a
        ~shape2:shape_b ()
  done;
  out_shape

let broadcast_index target_multi_idx source_shape =
  let target_ndim = Array.length target_multi_idx in
  let source_ndim = Array.length source_shape in
  let source_multi_idx = Array.make source_ndim 0 in
  for i = 0 to source_ndim - 1 do
    let target_idx_pos = target_ndim - source_ndim + i in
    let source_idx_pos = i in
    if source_idx_pos < 0 || target_idx_pos < 0 then ()
    else if source_shape.(source_idx_pos) = 1 then
      source_multi_idx.(source_idx_pos) <- 0
    else source_multi_idx.(source_idx_pos) <- target_multi_idx.(target_idx_pos)
  done;
  source_multi_idx

let to_string shape =
  let shape_str =
    Array.map string_of_int shape |> Array.to_list |> String.concat ","
  in
  Printf.sprintf "[%s]" shape_str

let pp fmt shape = Format.fprintf fmt "%s" (to_string shape)
