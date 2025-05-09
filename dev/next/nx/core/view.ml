(* Shared definitions and helpers for the new V2 *)
type layout_type = C_contiguous | Strided

let prod arr = Array.fold_left ( * ) 1 arr
let all_eq a b = Array.length a = Array.length b && Array.for_all2 ( = ) a b

(* For error messages in offset_to_index_contig *)
let pp_int_array_for_error_msg arr =
  "["
  ^ (Array.to_list arr |> List.map string_of_int |> String.concat "; ")
  ^ "]"

(* V2's Stride computation (remains the same) *)
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

(* V2's Canonicalize strides (remains the same) *)
let canonicalize_strides shape strides =
  Array.mapi (fun i s -> if shape.(i) = 1 then 0 else s) strides

(* New function: index_to_offset Converts a multi-dimensional index into a
   linear offset relative to the view's data start, using the view's strides.
   This does NOT add view.offset. This matches the behavior of V1's
   index_to_offset. *)
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

(* New function: offset_to_index_contig Converts a linear offset (k) into a
   multi-dimensional index, assuming a C-contiguous layout for the given
   'shape_arr'. Handles 0-sized dimensions correctly. This matches the intent of
   V1's offset_to_index_contig but is more robust. *)
let offset_to_index_contig k shape_arr =
  let n = Array.length shape_arr in
  if n = 0 then
    if k = 0 then [||]
    else
      invalid_arg
        "offset_to_index_contig: scalar out of bounds (k must be 0 for 0D \
         tensor)"
  else if
    (* Check for 0-sized dimensions in shape_arr. If any dim is 0, total size is
       0. *)
    Array.exists (( = ) 0) shape_arr
  then
    if k = 0 then
      Array.make n
        0 (* For total size 0, only k=0 is valid, maps to [0,0,...] *)
    else
      invalid_arg
        ("offset_to_index_contig: k (" ^ string_of_int k
       ^ ") > 0 for zero-size shape "
        ^ pp_int_array_for_error_msg shape_arr)
  else
    (* Standard case: all dimensions in shape_arr > 0 *)
    let total_elements = prod shape_arr in
    if k < 0 || k >= total_elements then
      invalid_arg
        ("offset_to_index_contig: k (" ^ string_of_int k
       ^ ") out of bounds for C-contiguous shape "
        ^ pp_int_array_for_error_msg shape_arr
        ^ " (size "
        ^ string_of_int total_elements
        ^ ")");

    let idx = Array.make n 0 in
    let temp_k = ref k in
    (* Logic from V1: iterate from rightmost dimension to second dimension *)
    for i = n - 1 downto 1 do
      let dim_size = shape_arr.(i) in
      (* dim_size cannot be 0 here due to the earlier check *)
      idx.(i) <- !temp_k mod dim_size;
      temp_k := !temp_k / dim_size
    done;
    (* The leftmost dimension's index is the remaining temp_k *)
    idx.(0) <- !temp_k;

    (* Final sanity check for the leftmost index, though covered by
       total_elements check *)
    if idx.(0) >= shape_arr.(0) then
      invalid_arg
        ("offset_to_index_contig: calculated idx.(0) ("
        ^ string_of_int idx.(0)
        ^ ") is out of bounds for shape_arr.(0) ("
        ^ string_of_int shape_arr.(0)
        ^ "). This indicates an issue with k or logic.");
    idx

type t = {
  shape : int array;
  strides : int array;
  offset : int;
  mask : (int * int) array option; (* Array of (min, max) pairs per dimension *)
  layout : layout_type; (* Changed from contiguous: bool *)
}

let shape v = v.shape
let strides v = v.strides

let stride axis v =
  if axis < 0 then invalid_arg "axis must be non-negative";
  if axis >= Array.length v.strides then invalid_arg "axis out of bounds";
  let stride = Array.unsafe_get v.strides axis in
  stride

let offset v = v.offset
let mask v = v.mask
let layout v = v.layout (* Accessor for the layout type *)

let is_contiguous v =
  v.layout = C_contiguous (* Behaviorally same as old v.contiguous *)

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

let size v = numel v (* Alias *)

(* The core factory function - Updated for layout_type *)
let create ?(offset = 0) ?mask ?strides shape_arr =
  (* Renamed shape to shape_arr for clarity *)
  (* Handle 0-size dimensions *)
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
          None (* Canonicalize no-op mask *)
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

(* Physical offset calculation: Important for backend load/store *)
(* This one includes the view's base offset *)
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
  | None ->
      true
      (* If no mask, always valid within shape boundaries (checked elsewhere) *)
  | Some mask_array ->
      if Array.length indices <> Array.length mask_array then
        (* This case should ideally not happen if indices are generated for
           view.shape *)
        false
      else
        Array.for_all2
          (fun idx (b, e) -> idx >= b && idx < e)
          indices mask_array

(* --- View Manipulation Methods (rely on 'create' for layout) --- *)

let expand view new_shape =
  if Array.length new_shape <> Array.length view.shape then
    invalid_arg "expand: Rank must match";
  if Array.exists (( = ) 0) view.shape then
    create new_shape (* Special case for 0-element source *)
  else
    let strides =
      Array.mapi
        (fun i ns ->
          let s = view.shape.(i) in
          if s = ns then view.strides.(i)
          else if s = 1 then 0 (* Expand singleton dim *)
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
                     (* Expand unmasked singleton *)
                   else
                     invalid_arg
                       "expand: Cannot expand masked singleton non-trivially"
                 else (b, e))
                 (* Keep mask for non-expanded or already matching dims *)
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

let reshape view new_shape =
  if view.shape = new_shape then view (* No-op *)
  else if Array.exists (( < ) 0) new_shape then
    invalid_arg "reshape: Negative dimensions not allowed"
  else
    let total_old = numel view in
    let total_new = prod new_shape in
    if total_old <> total_new && total_old <> 0 && total_new <> 0 then
      invalid_arg
        "reshape: Total size must remain unchanged for non-zero shapes";

    if Array.exists (( = ) 0) view.shape || Array.exists (( = ) 0) new_shape
    then
      (* Reshaping involving 0-sized dims results in a new default 0-sized view.
         Offset and mask are reset. Strides become default for the new_shape. *)
      create new_shape
    else if view.layout = C_contiguous then
      (* If the view is C_contiguous (offset=0, mask=None, canonical C strides),
         we can safely reshape it by just computing new C strides for new_shape.
         The offset remains 0. *)
      create ~offset:0
        (* Explicitly 0 as per C_contiguous definition *) new_shape
    else
      (* For Strided views, or C_contiguous views that somehow are not (e.g. by
         direct construction), a more complex stride projection would be needed.
         V2 currently doesn't implement this. *)
      failwith
        "reshape: Non-contiguous (Strided) reshape not fully implemented. View \
         must be C_contiguous."

(* Helper for pad/shrink *)
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
      (* Shift old mask relative to the new view's dimension start *)
      let shifted_old_b = max 0 (old_b - offset_for_dim) in
      let shifted_old_e = max 0 (old_e - offset_for_dim) in
      (* Intersect with the new mask component *)
      (max shifted_old_b new_b, min shifted_old_e new_e)
    in
    match (view.mask, new_mask_opt) with
    | None, None -> None
    | Some old_m, None ->
        (* Shrink case, new_mask_opt is None *)
        Some
          (Array.mapi
             (fun i (old_b, old_e) ->
               let a, _ = arg.(i) in
               (* 'a' is the start of the shrink window in old view's coords *)
               let new_dim_size = new_shape.(i) in
               (max 0 (old_b - a), min new_dim_size (old_e - a)))
               (* Shift and clamp *)
             old_m)
    | None, Some new_m -> Some new_m (* Pad case with no prior mask *)
    | Some old_m, Some new_m ->
        (* Pad case with prior mask *)
        Some
          (Array.mapi
             (fun i (old_b_i, old_e_i) ->
               let new_m_b_i, new_m_e_i = new_m.(i) in
               let a_i, _ = arg.(i) in
               (* 'a_i' is offset due to padding, usually negative for pad *)
               shift_and_combine_mask (old_b_i, old_e_i) (new_m_b_i, new_m_e_i)
                 a_i)
             old_m)
  in
  create ~offset:!new_offset ?mask:final_mask ~strides:view.strides new_shape

let pad view arg =
  if Array.length arg <> Array.length view.shape then
    invalid_arg "pad: Argument length mismatch";
  if Array.for_all (fun (b, e) -> b = 0 && e = 0) arg then view (* No-op *)
  else if Array.exists (fun (b, e) -> b < 0 || e < 0) arg then
    invalid_arg "pad: Negative padding values not allowed (use shrink or slice)"
  else
    let zvarg =
      (* "zero-based view arg": defines the new window in old view's coord
         system *)
      Array.mapi
        (fun i s ->
          let pad_before, pad_after = arg.(i) in
          (-pad_before, s + pad_after))
          (* new_start = 0 - pad_before, new_end = s + pad_after *)
        view.shape
    in
    (* The mask for pad indicates where the *original* data lies within the new
       padded shape *)
    let mask_for_pad =
      Array.mapi
        (fun i s_old ->
          let pad_before, _pad_after = arg.(i) in
          (pad_before, pad_before + s_old))
          (* start of old data, end of old data in new coords *)
        view.shape
    in
    unsafe_resize view zvarg (Some mask_for_pad)

let shrink view arg =
  if Array.length arg <> Array.length view.shape then
    invalid_arg "shrink: Argument length mismatch";
  if Array.for_all2 (fun (b, e) s -> b = 0 && e = s) arg view.shape then view
    (* No-op *)
  else if
    Array.exists2
      (fun (b, e) s -> b < 0 || e < 0 || b > s || e > s || b >= e)
        (* b >= e for non-empty slice *)
      arg view.shape
  then
    invalid_arg
      "shrink: Invalid shrink bounds (must be within old shape and start < end)"
  else
    unsafe_resize view arg
      None (* No *additional* mask needed beyond shifting existing mask *)

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
          (* Avoid issues with 0-sized dimensions *)
          new_offset := !new_offset + ((s_i - 1) * view.strides.(i));
          new_strides.(i) <- -new_strides.(i);
          match new_mask with
          | Some m_arr ->
              let b, e = m_arr.(i) in
              m_arr.(i) <- (s_i - e, s_i - b)
              (* Flip mask bounds *)
          | None -> ()))
    flip_axes_bools;
  create ~offset:!new_offset ?mask:new_mask ~strides:new_strides view.shape

(* Broadcasting utilities for frontend (remains the same) *)
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
    else invalid_arg "broadcast_shapes: Incompatible dimensions"
  done;
  out_shape
