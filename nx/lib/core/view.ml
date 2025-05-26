(* Lightweight view of tensor layout and helpers for reshaping. *)

type layout = C_contiguous | Strided

type t = {
  shape : int array;
  strides : int array;
  offset : int;
  mask : (int * int) array option; (* bounds per dimension *)
  layout : layout;
}

(* ───── helpers ───── *)

let prod arr = Array.fold_left ( * ) 1 arr
let all_eq a b = Array.length a = Array.length b && Array.for_all2 ( = ) a b

let pp_int_array arr =
  "["
  ^ (Array.to_list arr |> List.map string_of_int |> String.concat "; ")
  ^ "]"

(* compute C-contiguous strides *)
let compute_strides shape =
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

(* drop strides for unit dimensions *)
let canonicalize_strides shape strides =
  Array.mapi (fun i s -> if shape.(i) = 1 then 0 else s) strides

(* linearize a multi-dimensional index *)
let index_to_offset md_index strides_arr =
  if Array.length md_index <> Array.length strides_arr then
    invalid_arg
      ("index_to_offset: rank mismatch. Index dim "
      ^ string_of_int (Array.length md_index)
      ^ ", Strides dim "
      ^ string_of_int (Array.length strides_arr));
  let o = ref 0 in
  Array.iteri (fun i v -> o := !o + (v * strides_arr.(i))) md_index;
  !o

(* inverse of [index_to_offset] for C-contiguous shapes *)
let offset_to_index_contig k shape_arr =
  let n = Array.length shape_arr in
  if n = 0 then
    if k = 0 then [||]
    else invalid_arg "offset_to_index_contig: k out of bounds for scalar"
  else if Array.exists (( = ) 0) shape_arr then
    (* zero-size tensor; only k=0 is allowed *)
    if k = 0 then Array.make n 0
    else
      invalid_arg
        ("offset_to_index_contig: k (" ^ string_of_int k
       ^ ") > 0 for zero-size shape " ^ pp_int_array shape_arr)
  else
    let total_elements = prod shape_arr in
    if k < 0 || k >= total_elements then
      invalid_arg
        ("offset_to_index_contig: k (" ^ string_of_int k
       ^ ") out of bounds for C-contiguous shape " ^ pp_int_array shape_arr
       ^ " (size "
        ^ string_of_int total_elements
        ^ ")");

    let idx = Array.make n 0 in
    let temp_k = ref k in
    for i = n - 1 downto 1 do
      let dim_size = shape_arr.(i) in
      idx.(i) <- !temp_k mod dim_size;
      temp_k := !temp_k / dim_size
    done;
    idx.(0) <- !temp_k;

    (* sanity check for the leftmost index *)
    if idx.(0) >= shape_arr.(0) then
      invalid_arg
        ("offset_to_index_contig: calculated idx.(0) ("
        ^ string_of_int idx.(0)
        ^ ") is out of bounds for shape_arr.(0) ("
        ^ string_of_int shape_arr.(0)
        ^ "). This indicates an issue with k or logic.");
    idx

(* infer the dimension corresponding to [-1] in [new_shape_spec] *)
let resolve_neg_one_shape current_shape new_shape_spec =
  let new_shape_spec_l = Array.to_list new_shape_spec in
  let current_numel = prod current_shape in
  let neg_one_count =
    new_shape_spec_l |> List.filter (( = ) (-1)) |> List.length
  in
  if neg_one_count > 1 then
    invalid_arg "reshape: can only specify one unknown dimension"
  else if neg_one_count = 0 then new_shape_spec
  else
    let specified_numel =
      List.filter (( <> ) (-1)) new_shape_spec_l |> Array.of_list |> prod
    in
    (* when shape_spec includes zero dimensions *)
    if specified_numel = 0 then
      if current_numel = 0 then
        Array.map (fun x -> if x = -1 then 0 else x) new_shape_spec
      else
        invalid_arg
          "Reshape cannot infer -1 when other dimensions multiply to 0 but \
           total size is non-zero"
    else if current_numel mod specified_numel <> 0 then
      invalid_arg
        (Printf.sprintf
           "Reshape size mismatch: Cannot reshape %d elements into shape with \
            specified elements %d"
           current_numel specified_numel)
    else
      let inferred_dim = current_numel / specified_numel in
      Array.map (fun s -> if s = -1 then inferred_dim else s) new_shape_spec

(* ───── accessors ───── *)

let shape v = v.shape
let strides v = v.strides

let stride axis v =
  if axis < 0 then invalid_arg "axis must be non-negative";
  if axis >= Array.length v.strides then invalid_arg "axis out of bounds";
  let stride = Array.unsafe_get v.strides axis in
  stride

let offset v = v.offset
let mask v = v.mask
let layout v = v.layout
let is_contiguous v = v.layout = C_contiguous
let dims v = v.shape

let dim axis v =
  if axis < 0 then invalid_arg "axis must be non-negative";
  if axis >= Array.length v.shape then invalid_arg "axis out of bounds";
  let dim = Array.unsafe_get v.shape axis in
  dim

let ndim v = Array.length v.shape

let numel v =
  let n = Array.length v.shape in
  if n = 0 then 1 else Array.fold_left ( * ) 1 v.shape

let size v = numel v

(* allocate a new view for [shape_arr] *)
let create ?(offset = 0) ?mask ?strides shape_arr =
  let is_zero_size = Array.exists (( = ) 0) shape_arr in
  let current_shape =
    if is_zero_size then Array.map (fun s -> max s 0) shape_arr else shape_arr
  in
  let current_strides =
    match strides with
    | Some s -> canonicalize_strides current_shape s
    | None -> compute_strides current_shape
  in
  let current_offset = if is_zero_size then 0 else offset in
  let current_mask =
    if is_zero_size then None
    else
      match mask with
      | Some m
        when Array.for_all2 (fun (b, e) s -> b = 0 && e = s) m current_shape ->
          None
      | _ -> mask
  in
  let new_layout =
    if
      current_offset = 0 && current_mask = None
      && all_eq current_strides (compute_strides current_shape)
    then C_contiguous
    else Strided
  in
  {
    shape = current_shape;
    strides = current_strides;
    offset = current_offset;
    mask = current_mask;
    layout = new_layout;
  }

(* ───── offset & validation ───── *)

let offset_of_indices view indices =
  if Array.length indices <> Array.length view.shape then
    invalid_arg "offset_of_indices: Rank mismatch";
  let physical_offset = ref view.offset in
  Array.iteri
    (fun i idx ->
      physical_offset := !physical_offset + (idx * view.strides.(i)))
    indices;
  !physical_offset

(* Check if an index is valid according to the mask *)
let is_valid view indices =
  match view.mask with
  | None -> true
  | Some mask_array ->
      if Array.length indices <> Array.length mask_array then false
      else
        Array.for_all2
          (fun idx (b, e) -> idx >= b && idx < e)
          indices mask_array

(* ───── view manipulation ───── *)

let expand view new_shape =
  if Array.length new_shape <> Array.length view.shape then
    invalid_arg "expand: Rank must match";
  if Array.exists (( = ) 0) view.shape then create new_shape
  else
    let strides =
      Array.mapi
        (fun i ns ->
          let s = view.shape.(i) in
          if s = ns then view.strides.(i)
          else if s = 1 then 0
          else invalid_arg "expand: Cannot expand non-singleton dimension")
        new_shape
    in
    let mask =
      match view.mask with
      | None -> None
      | Some m ->
          Some
            (Array.mapi
               (fun i (b, e) ->
                 if view.shape.(i) = 1 && new_shape.(i) <> 1 then
                   if b = 0 && e = 1 then (0, new_shape.(i))
                   else invalid_arg "expand: Cannot expand masked singleton"
                 else (b, e))
               m)
    in
    create ~offset:view.offset ?mask ~strides new_shape

let permute view axes =
  let n = ndim view in
  if Array.length axes <> n then invalid_arg "permute: Invalid axes length";
  let seen = Array.make n false in
  Array.iter
    (fun ax ->
      if ax < 0 || ax >= n || seen.(ax) then
        invalid_arg "permute: Invalid axis value or duplicate";
      seen.(ax) <- true)
    axes;
  if not (Array.for_all Fun.id seen) then
    invalid_arg "permute: Axes do not form a permutation";

  let new_shape =
    Array.map (fun i -> view.shape.(axes.(i))) (Array.init n Fun.id)
  in
  let new_strides =
    Array.map (fun i -> view.strides.(axes.(i))) (Array.init n Fun.id)
  in
  let new_mask =
    match view.mask with
    | None -> None
    | Some m -> Some (Array.map (fun i -> m.(axes.(i))) (Array.init n Fun.id))
  in
  create ~offset:view.offset ?mask:new_mask ~strides:new_strides new_shape

(* In view.ml *)
let reshape view new_shape =
  let current_numel = prod view.shape in
  let new_numel = prod new_shape in
  if view.shape = new_shape then view
  else if current_numel <> new_numel && current_numel <> 0 && new_numel <> 0
  then
    (* Allow 0-size to be reshaped if one of them is 0-size *)
    invalid_arg
      (Printf.sprintf "reshape: cannot reshape array of size %d into shape %s"
         current_numel (pp_int_array new_shape))
  else if Array.exists (fun x -> x < 0) new_shape then
    (* Allow new_shape to contain -1 if it's resolved by frontend *)
    invalid_arg
      "reshape: Negative dimensions not allowed in final shape for View.reshape"
  else if Array.exists (( = ) 0) view.shape || Array.exists (( = ) 0) new_shape
  then
    create ~offset:0
      new_shape (* Handle zero-size tensors: new C-contig view, offset 0*)
  else if view.mask = None then
    (* Handle C-contiguous and strided cases without masks *)
    if view.layout = C_contiguous then
      (* Simplest C-contiguous case *)
      create ~offset:view.offset new_shape (* Preserve offset, new C-strides *)
    else
      (* For strided views, check if they're effectively contiguous *)
      let expected_strides = compute_strides view.shape in
      let is_effectively_contiguous = all_eq view.strides expected_strides in

      (* Also check for single-element tensors which can always be reshaped *)
      let is_single_element = current_numel <= 1 in

      (* Check if this is just squeezing (removing dims of size 1) *)
      let is_squeeze_only =
        let old_non_one = Array.to_list view.shape |> List.filter (( <> ) 1) in
        let new_non_one = Array.to_list new_shape |> List.filter (( <> ) 1) in
        old_non_one = new_non_one
      in

      if is_effectively_contiguous || is_single_element then
        (* The view is effectively C-contiguous or single element *)
        create ~offset:view.offset new_shape
      else if is_squeeze_only then
        (* Special case: just removing dimensions of size 1 *)
        let new_strides =
          let old_idx = ref 0 in
          Array.map
            (fun dim ->
              if dim = 1 then 0
              else (
                while
                  !old_idx < Array.length view.shape
                  && view.shape.(!old_idx) = 1
                do
                  incr old_idx
                done;
                let stride = view.strides.(!old_idx) in
                incr old_idx;
                stride))
            new_shape
        in
        {
          shape = new_shape;
          strides = new_strides;
          offset = view.offset;
          mask = view.mask;
          layout = Strided;
        }
      else failwith "View.reshape: cannot reshape non-contiguous strided views"
  else failwith "View.reshape: cannot reshape views with masks"

(* helper used by [pad] and [shrink] *)
let unsafe_resize view arg new_mask_opt =
  if Array.length arg <> Array.length view.shape then
    invalid_arg "unsafe_resize: Argument length mismatch";
  let new_shape = Array.map (fun (a, b) -> b - a) arg in
  let new_offset = ref view.offset in
  Array.iteri
    (fun i (a, _) -> new_offset := !new_offset + (a * view.strides.(i)))
    arg;

  let final_mask =
    let shift_and_combine_mask old_mask_dim_bounds new_mask_dim_bounds
        offset_for_dim =
      let old_b, old_e = old_mask_dim_bounds in
      let new_b, new_e = new_mask_dim_bounds in
      let shifted_old_b = max 0 (old_b - offset_for_dim) in
      let shifted_old_e = max 0 (old_e - offset_for_dim) in
      (max shifted_old_b new_b, min shifted_old_e new_e)
    in
    match (view.mask, new_mask_opt) with
    | None, None -> None
    | Some old_m, None ->
        Some
          (Array.mapi
             (fun i (old_b, old_e) ->
               let a, _ = arg.(i) in
               let new_dim_size = new_shape.(i) in
               (max 0 (old_b - a), min new_dim_size (old_e - a)))
             old_m)
    | None, Some new_m -> Some new_m
    | Some old_m, Some new_m ->
        Some
          (Array.mapi
             (fun i (old_b_i, old_e_i) ->
               let new_m_b_i, new_m_e_i = new_m.(i) in
               let a_i, _ = arg.(i) in
               shift_and_combine_mask (old_b_i, old_e_i) (new_m_b_i, new_m_e_i)
                 a_i)
             old_m)
  in
  create ~offset:!new_offset ?mask:final_mask ~strides:view.strides new_shape

let pad view arg =
  if Array.length arg <> Array.length view.shape then
    invalid_arg "pad: Argument length mismatch";
  if Array.for_all (fun (b, e) -> b = 0 && e = 0) arg then view
  else if Array.exists (fun (b, e) -> b < 0 || e < 0) arg then
    invalid_arg "pad: Negative padding values not allowed (use shrink or slice)"
  else
    let zvarg =
      Array.mapi
        (fun i s ->
          let pad_before, pad_after = arg.(i) in
          (-pad_before, s + pad_after))
        view.shape
    in
    let mask_for_pad =
      Array.mapi
        (fun i s_old ->
          let pad_before, _pad_after = arg.(i) in
          (pad_before, pad_before + s_old))
        view.shape
    in
    unsafe_resize view zvarg (Some mask_for_pad)

let shrink view arg =
  if Array.length arg <> Array.length view.shape then
    invalid_arg "shrink: Argument length mismatch";
  if Array.for_all2 (fun (b, e) s -> b = 0 && e = s) arg view.shape then view
  else if
    Array.exists2
      (fun (b, e) s -> b < 0 || e < 0 || b > s || e > s || b >= e)
      arg view.shape
  then
    invalid_arg
      "shrink: Invalid shrink bounds (must be within old shape and start < end)"
  else unsafe_resize view arg None

let flip view flip_axes_bools =
  if Array.length flip_axes_bools <> Array.length view.shape then
    invalid_arg "flip: Boolean array length mismatch with view rank";
  let new_offset = ref view.offset in
  let new_strides = Array.copy view.strides in
  let new_mask =
    match view.mask with Some m -> Some (Array.copy m) | None -> None
  in
  Array.iteri
    (fun i do_flip ->
      if do_flip then
        let s_i = view.shape.(i) in
        if s_i > 0 then (
          new_offset := !new_offset + ((s_i - 1) * view.strides.(i));
          new_strides.(i) <- -new_strides.(i);
          match new_mask with
          | Some m_arr ->
              let b, e = m_arr.(i) in
              m_arr.(i) <- (s_i - e, s_i - b)
          | None -> ()))
    flip_axes_bools;
  create ~offset:!new_offset ?mask:new_mask ~strides:new_strides view.shape

(* ───── broadcasting ───── *)

let broadcast_shapes shape_a shape_b =
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
      invalid_arg
        (Printf.sprintf
           "broadcast_shapes: shapes %s and %s cannot be broadcast together"
           (pp_int_array shape_a) (pp_int_array shape_b))
  done;
  out_shape

let compute_broadcast_index target_multi_idx source_shape =
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

let multi_index_from_linear linear_idx shape =
  let ndim = Array.length shape in
  let index = Array.make ndim 0 in
  if ndim = 0 then index
  else if Array.exists (( = ) 0) shape then index
  else
    let current_linear_idx = ref linear_idx in
    let strides = Array.make ndim 1 in
    for i = ndim - 2 downto 0 do
      strides.(i) <- strides.(i + 1) * shape.(i + 1)
    done;

    for i = 0 to ndim - 1 do
      if strides.(i) = 0 then index.(i) <- 0
      else (
        index.(i) <- !current_linear_idx / strides.(i);
        current_linear_idx := !current_linear_idx mod strides.(i))
    done;
    index
