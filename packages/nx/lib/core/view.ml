(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Lightweight view of tensor layout and helpers for reshaping. *)

let err op fmt = Printf.ksprintf (fun msg -> invalid_arg (op ^ ": " ^ msg)) fmt

type layout = C_contiguous | Strided

type t = {
  shape : int array;
  strides : int array;
  offset : int;
  mask : (int * int) array option;
  layout : layout;
}

(* ───── Helpers ───── *)

let prod arr = Array.fold_left ( * ) 1 arr

(* compute C-contiguous strides for concrete shape *)
let compute_strides shape_array =
  let n = Array.length shape_array in
  if n = 0 then [||]
  else
    let strides = Array.make n 0 in
    strides.(n - 1) <- (if shape_array.(n - 1) = 0 then 0 else 1);
    for i = n - 2 downto 0 do
      strides.(i) <-
        (if shape_array.(i) = 0 then 0
         else strides.(i + 1) * max 1 shape_array.(i + 1))
    done;
    strides

(* canonicalize strides - keep original strides, don't force stride 0 for size
   1 *)
let canonicalize_strides _shape_array strides = strides

(* Check if strides represent a contiguous layout *)
let is_c_contiguous_strides shape_arr strides mask =
  mask = None
  &&
  let expected = compute_strides shape_arr in
  let expected_canonical = canonicalize_strides shape_arr expected in
  Array.length strides = Array.length expected_canonical
  && Array.for_all2 ( = ) strides expected_canonical

(* ───── Accessors ───── *)

let shape v = v.shape
let strides v = v.strides

let stride axis v =
  let ndim = Array.length v.shape in
  if axis < 0 || axis >= ndim then
    err "stride" "axis %d out of bounds for %dD tensor" axis ndim;
  Array.unsafe_get v.strides axis

let offset v = v.offset
let mask v = v.mask
let is_c_contiguous v = v.layout = C_contiguous

let dim axis v =
  let ndim = Array.length v.shape in
  if axis < 0 || axis >= ndim then
    err "dim" "axis %d out of bounds for %dD tensor" axis ndim;
  v.shape.(axis)

let ndim v = Array.length v.shape
let numel v = prod v.shape

(* ───── View Creation ───── *)

let create ?(offset = 0) ?strides ?mask shape =
  let is_zero_size = Array.exists (( = ) 0) shape in
  let current_shape =
    if is_zero_size then Array.map (fun s -> max s 0) shape else shape
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
    if is_c_contiguous_strides current_shape current_strides current_mask then
      C_contiguous
    else Strided
  in
  {
    shape = current_shape;
    strides = current_strides;
    offset = current_offset;
    mask = current_mask;
    layout = new_layout;
  }

(* ───── Offset & Validation ───── *)

let linear_index view indices =
  let ndim = Array.length view.shape in
  if Array.length indices <> ndim then
    err "linear_index" "rank mismatch: indices[%d] vs ndim %d"
      (Array.length indices) ndim;
  let physical_offset = ref view.offset in
  Array.iteri
    (fun i idx ->
      physical_offset := !physical_offset + (idx * view.strides.(i)))
    indices;
  !physical_offset

let is_valid view indices =
  match view.mask with
  | None -> true
  | Some mask_array ->
      if Array.length indices <> Array.length mask_array then false
      else
        Array.for_all2
          (fun idx (b, e) -> idx >= b && idx < e)
          indices mask_array

(* ───── View Manipulation ───── *)

let expand view new_shape =
  let old_ndim = Array.length view.shape in
  let new_ndim = Array.length new_shape in
  (* Allow expanding a scalar to any shape *)
  if old_ndim = 0 then
    let strides = Array.make new_ndim 0 in
    { view with shape = new_shape; strides }
  else if new_ndim <> old_ndim then
    err "expand" "rank mismatch: %d vs %d" new_ndim old_ndim
  else
    let old_arr = view.shape in
    let new_arr = new_shape in
    if Array.exists (( = ) 0) old_arr then create new_shape
    else
      let strides =
        Array.mapi
          (fun i ns ->
            let s = old_arr.(i) in
            if s = ns then view.strides.(i)
            else if s = 1 then 0
            else
              err "expand"
                "dimension %d (size %d) cannot expand to size %d, only \
                 singletons expand"
                i s ns)
          new_arr
      in
      let mask =
        match view.mask with
        | None -> None
        | Some m ->
            Some
              (Array.mapi
                 (fun i (b, e) ->
                   if old_arr.(i) = 1 && new_arr.(i) <> 1 then
                     if b = 0 && e = 1 then (0, new_arr.(i))
                     else
                       err "expand"
                         "masked singleton bounds [%d,%d] incompatible with \
                          expansion"
                         b e
                   else (b, e))
                 m)
      in
      create ~offset:view.offset ?mask ~strides new_shape

let permute view axes =
  let n = ndim view in
  if Array.length axes <> n then
    err "permute" "axes length %d != ndim %d" (Array.length axes) n;

  (* Validate permutation *)
  let seen = Array.make n false in
  Array.iter
    (fun ax ->
      if ax < 0 || ax >= n then
        err "permute" "axis %d out of bounds for %dD tensor" ax n;
      if seen.(ax) then err "permute" "duplicate axis %d" ax;
      seen.(ax) <- true)
    axes;

  let new_shape = Array.init n (fun i -> view.shape.(axes.(i))) in
  let new_strides = Array.init n (fun i -> view.strides.(axes.(i))) in
  let new_mask =
    Option.map (fun m -> Array.init n (fun i -> m.(axes.(i)))) view.mask
  in
  create ~offset:view.offset ?mask:new_mask ~strides:new_strides new_shape

let reshape view new_shape =
  (* Early return if shapes are identical *)
  if view.shape = new_shape then view
  else
    let old_arr = view.shape in
    let new_arr = new_shape in
    let old_numel = prod old_arr in
    let new_numel = prod new_arr in

    (* Check size compatibility *)
    if old_numel <> new_numel && old_numel <> 0 && new_numel <> 0 then
      err "reshape" "cannot reshape %s to %s" (Shape.to_string old_arr)
        (Shape.to_string new_arr)
    else if Array.exists (( = ) 0) old_arr || Array.exists (( = ) 0) new_arr
    then create ~offset:0 new_shape
      (* Check for masks - these complicate reshape *)
    else if view.mask <> None then
      invalid_arg
        "reshape: cannot reshape views with masks, call contiguous() first"
      (* Fast path for C-contiguous views *)
    else if view.layout = C_contiguous then create ~offset:view.offset new_shape
    else if
      (* Special case: reshaping to/from scalar *)
      Array.length new_shape = 0
    then create ~offset:view.offset new_shape
      (* Special case: all strides are 0 (broadcast from scalar) *)
    else if Array.for_all (( = ) 0) view.strides then
      let new_strides = Array.make (Array.length new_shape) 0 in
      create ~offset:view.offset ~strides:new_strides new_shape
    (* Special case: only expanding/squeezing size-1 dimensions *)
      else
      let try_squeeze_unsqueeze () =
        let old_non_one = Array.to_list old_arr |> List.filter (( <> ) 1) in
        let new_non_one = Array.to_list new_arr |> List.filter (( <> ) 1) in

        if old_non_one = new_non_one then
          let old_idx = ref 0 in
          let new_strides =
            Array.map
              (fun dim ->
                if dim = 1 then 0
                else (
                  while
                    !old_idx < Array.length old_arr && old_arr.(!old_idx) = 1
                  do
                    incr old_idx
                  done;
                  let stride = view.strides.(!old_idx) in
                  incr old_idx;
                  stride))
              new_arr
          in
          Some new_strides
        else None
      in

      let try_merge_split () =
        let old_dims = ref [] in
        let new_dims = ref [] in

        for i = 0 to Array.length old_arr - 1 do
          if old_arr.(i) > 1 then
            old_dims := (old_arr.(i), view.strides.(i)) :: !old_dims
        done;
        old_dims := List.rev !old_dims;

        for i = 0 to Array.length new_arr - 1 do
          if new_arr.(i) > 1 then new_dims := new_arr.(i) :: !new_dims
        done;
        new_dims := List.rev !new_dims;

        let rec match_dims old_dims new_dims =
          match (old_dims, new_dims) with
          | [], [] -> Some []
          | [], _ | _, [] -> None
          | (old_size, old_stride) :: old_rest, new_size :: new_rest ->
              if old_size = new_size then
                match match_dims old_rest new_rest with
                | Some rest_strides ->
                    Some ((new_size, old_stride) :: rest_strides)
                | None -> None
              else if old_size > new_size && old_size mod new_size = 0 then
                let remaining_size = old_size / new_size in
                let first_stride = old_stride * remaining_size in
                let remaining_dims = (remaining_size, old_stride) :: old_rest in
                match match_dims remaining_dims new_rest with
                | Some rest_strides ->
                    Some ((new_size, first_stride) :: rest_strides)
                | None -> None
              else if new_size > old_size then
                let rec collect_merge size stride dims needed =
                  if size = needed then Some (dims, stride)
                  else if size > needed then None
                  else
                    match dims with
                    | [] -> None
                    | (next_size, next_stride) :: rest ->
                        if stride = next_stride * next_size then
                          collect_merge (size * next_size) next_stride rest
                            needed
                        else None
                in
                match collect_merge old_size old_stride old_rest new_size with
                | Some (remaining, first_stride) -> (
                    match match_dims remaining new_rest with
                    | Some rest_strides ->
                        Some ((new_size, first_stride) :: rest_strides)
                    | None -> None)
                | None -> None
              else None
        in

        match match_dims !old_dims !new_dims with
        | None -> None
        | Some stride_map ->
            let stride_map_arr = Array.of_list stride_map in
            let new_strides = Array.make (Array.length new_arr) 0 in
            let map_idx = ref 0 in

            for i = 0 to Array.length new_arr - 1 do
              if new_arr.(i) = 1 then new_strides.(i) <- 0
              else
                let _, stride = stride_map_arr.(!map_idx) in
                new_strides.(i) <- stride;
                incr map_idx
            done;

            Some new_strides
      in
      (* Try reshape strategies in order *)
      match try_squeeze_unsqueeze () with
      | Some new_strides ->
          create ~offset:view.offset ~strides:new_strides new_shape
      | None -> (
          match try_merge_split () with
          | Some new_strides ->
              create ~offset:view.offset ~strides:new_strides new_shape
          | None ->
              let expected_strides = compute_strides new_arr in
              let stride_str =
                "["
                ^ String.concat ","
                    (Array.to_list (Array.map string_of_int view.strides))
                ^ "]"
              in
              let expected_str =
                "["
                ^ String.concat ","
                    (Array.to_list (Array.map string_of_int expected_strides))
                ^ "]"
              in
              err "reshape"
                "cannot reshape %s to %s, incompatible strides %s (expected \
                 %s), call contiguous() first"
                (Shape.to_string old_arr) (Shape.to_string new_arr) stride_str
                expected_str)

(* helper used by [pad] and [shrink] *)
let unsafe_resize view arg new_mask_opt =
  let ndim = Array.length view.shape in
  if Array.length arg <> ndim then
    err "unsafe_resize" "argument length %d != ndim %d" (Array.length arg) ndim;

  let strides = view.strides in

  let new_shape = Array.map (fun (a, b) -> b - a) arg in
  let new_offset = ref view.offset in
  Array.iteri
    (fun i (a, _) -> new_offset := !new_offset + (a * strides.(i)))
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
  create ~offset:!new_offset ?mask:final_mask ~strides new_shape

let pad view arg =
  let ndim = Array.length view.shape in
  if Array.length arg <> ndim then
    err "pad" "padding length %d != ndim %d" (Array.length arg) ndim;
  if Array.for_all (fun (b, e) -> b = 0 && e = 0) arg then view
  else if Array.exists (fun (b, e) -> b < 0 || e < 0) arg then
    invalid_arg "pad: negative padding values, use shrink or slice instead"
  else
    let shape_arr = view.shape in
    let zvarg =
      Array.mapi
        (fun i s ->
          let pad_before, pad_after = arg.(i) in
          (-pad_before, s + pad_after))
        shape_arr
    in
    let mask_for_pad =
      Array.mapi
        (fun i s_old ->
          let pad_before, _pad_after = arg.(i) in
          (pad_before, pad_before + s_old))
        shape_arr
    in
    unsafe_resize view zvarg (Some mask_for_pad)

let shrink view arg =
  let ndim = Array.length view.shape in
  if Array.length arg <> ndim then
    err "shrink" "bounds length %d != ndim %d" (Array.length arg) ndim;
  let shape_arr = view.shape in
  if Array.for_all2 (fun (b, e) s -> b = 0 && e = s) arg shape_arr then view
  else if
    Array.exists2
      (fun (b, e) s -> b < 0 || e < 0 || b > s || e > s || b >= e)
      arg shape_arr
  then invalid_arg "shrink: bounds must be within shape and start < end"
  else unsafe_resize view arg None

let flip view flip_axes_bools =
  let ndim = Array.length view.shape in
  if Array.length flip_axes_bools <> ndim then
    err "flip" "boolean array length %d != ndim %d"
      (Array.length flip_axes_bools)
      ndim;

  let shape_arr = view.shape in
  let strides = view.strides in

  let new_offset = ref view.offset in
  let new_strides = Array.copy strides in
  let new_mask =
    match view.mask with Some m -> Some (Array.copy m) | None -> None
  in
  Array.iteri
    (fun i do_flip ->
      if do_flip then
        let s_i = shape_arr.(i) in
        if s_i > 0 then (
          new_offset := !new_offset + ((s_i - 1) * strides.(i));
          new_strides.(i) <- -new_strides.(i);
          match new_mask with
          | Some m_arr ->
              let b, e = m_arr.(i) in
              m_arr.(i) <- (s_i - e, s_i - b)
          | None -> ()))
    flip_axes_bools;
  create ~offset:!new_offset ?mask:new_mask ~strides:new_strides view.shape

let simplify view =
  (* Only simplify things that don't change the user-visible shape *)

  (* 1. Canonicalize mask that covers entire dimensions *)
  let mask =
    match view.mask with
    | Some m when Array.for_all2 (fun (b, e) s -> b = 0 && e = s) m view.shape
      ->
        None (* Mask covers everything, remove it *)
    | m -> m
  in

  (* Just return with simplified mask if changed *)
  if mask <> view.mask then
    let new_layout =
      if mask = None && is_c_contiguous_strides view.shape view.strides mask
      then C_contiguous
      else Strided
    in
    { view with mask; layout = new_layout }
  else view

let can_get_strides_simplified simplified =
  match simplified.mask with
  | None -> true
  | Some mask_array ->
      Array.for_all2
        (fun (b, e) s -> b = 0 && e = s)
        mask_array simplified.shape

let can_get_strides view = simplify view |> can_get_strides_simplified

let strides_opt view =
  let simplified = simplify view in
  if can_get_strides_simplified simplified then Some (strides simplified)
  else None

let is_materializable view =
  let simplified = simplify view in
  can_get_strides_simplified simplified
