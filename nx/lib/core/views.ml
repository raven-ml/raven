open Descriptor

let broadcast_to desc target_shape =
  let ndim = Array.length target_shape in
  let orig_ndim = Array.length desc.shape in

  if ndim < orig_ndim then
    invalid_arg
      (Format.asprintf
         "broadcast_to: cannot broadcast shape %a to fewer dimensions (%a)"
         pp_int_array desc.shape pp_int_array target_shape);

  let diff = ndim - orig_ndim in
  let new_strides = Array.make ndim 0 in

  for i = ndim - 1 downto 0 do
    let target_dim = target_shape.(i) in
    if i < diff then
      if target_dim = 1 then new_strides.(i) <- 0
      else if target_dim > 1 then new_strides.(i) <- 0
      else new_strides.(i) <- 0
    else
      let orig_i = i - diff in
      let orig_dim = desc.shape.(orig_i) in
      if orig_dim = target_dim then new_strides.(i) <- desc.strides.(orig_i)
      else if orig_dim = 1 && target_dim >= 0 then new_strides.(i) <- 0
      else
        invalid_arg
          (Format.asprintf
             "broadcast_to: shapes %a and %a are not compatible for \
              broadcasting at dimension %d (original size %d vs target size \
              %d)"
             pp_int_array desc.shape pp_int_array target_shape i orig_dim
             target_dim)
  done;

  let is_contig =
    check_c_contiguity_from_shape_strides target_shape new_strides
  in
  let new_layout =
    if is_contig && desc.offset = 0 then C_contiguous else Strided
  in
  { desc with shape = target_shape; strides = new_strides; layout = new_layout }

let squeeze ?axes (desc : ('a, 'b) descriptor) : ('a, 'b) descriptor =
  let ndim = Array.length desc.shape in
  let axes_to_squeeze =
    match axes with
    | None ->
        Array.to_list
          (Array.mapi (fun i s -> if s = 1 then Some i else None) desc.shape)
        |> List.filter_map Fun.id |> Array.of_list
    | Some arr ->
        Array.map
          (fun ax ->
            let real_axis = if ax < 0 then ndim + ax else ax in
            if real_axis < 0 || real_axis >= ndim then
              invalid_arg
                (Format.asprintf "squeeze: axis %d out of bounds for shape %a"
                   ax pp_int_array desc.shape);
            if desc.shape.(real_axis) <> 1 then
              invalid_arg
                (Format.asprintf
                   "squeeze: cannot select axis %d with size %d != 1 for \
                    squeezing"
                   real_axis desc.shape.(real_axis));
            real_axis)
          arr
  in
  if Array.length axes_to_squeeze = 0 then desc
  else
    let should_squeeze = Array.make ndim false in
    Array.iter (fun ax -> should_squeeze.(ax) <- true) axes_to_squeeze;

    let new_shape_list = ref [] in
    let new_strides_list = ref [] in
    for i = 0 to ndim - 1 do
      if not should_squeeze.(i) then (
        new_shape_list := desc.shape.(i) :: !new_shape_list;
        new_strides_list := desc.strides.(i) :: !new_strides_list)
    done;

    let new_shape = Array.of_list (List.rev !new_shape_list) in
    let new_strides = Array.of_list (List.rev !new_strides_list) in
    let desc_size = Array.fold_left ( * ) 1 desc.shape in
    let final_shape =
      if new_shape = [||] && desc_size = 1 then [||] else new_shape
    in
    let final_strides =
      if new_strides = [||] && desc_size = 1 then [||] else new_strides
    in

    let is_contig =
      check_c_contiguity_from_shape_strides final_shape final_strides
    in
    let new_layout =
      if is_contig && desc.offset = 0 then C_contiguous else Strided
    in
    {
      desc with
      shape = final_shape;
      strides = final_strides;
      layout = new_layout;
    }

let expand_dims axis (desc : ('a, 'b) descriptor) : ('a, 'b) descriptor =
  let ndim = Array.length desc.shape in
  let real_axis = if axis < 0 then ndim + axis + 1 else axis in
  if real_axis < 0 || real_axis > ndim then
    invalid_arg
      (Format.asprintf "expand_dims: axis %d out of bounds for shape %a" axis
         pp_int_array desc.shape);

  let new_ndim = ndim + 1 in
  let new_shape = Array.make new_ndim 0 in
  let new_strides = Array.make new_ndim 0 in

  let j = ref 0 in
  let insert_stride = ref 0 in
  for i = 0 to new_ndim - 1 do
    if i = real_axis then (
      new_shape.(i) <- 1;
      (* Determine stride for inserted dim *)
      if i = new_ndim - 1 then insert_stride := 1
      else if i = 0 && ndim = 0 then insert_stride := 1
      else if i = 0 && ndim > 0 then
        insert_stride := desc.strides.(0) * desc.shape.(0)
      else if i > 0 then
        (* Stride should be stride of next dim times size of next dim *)
        (* We haven't calculated new_strides yet, use original *)
        insert_stride := desc.strides.(!j) * desc.shape.(!j)
      else (* Fallback/Error case, should not happen *) insert_stride := 1;
      new_strides.(i) <- !insert_stride)
    else (
      new_shape.(i) <- desc.shape.(!j);
      new_strides.(i) <- desc.strides.(!j);
      incr j)
  done;

  let is_contig = check_c_contiguity_from_shape_strides new_shape new_strides in
  let new_layout =
    if is_contig && desc.offset = 0 then C_contiguous else Strided
  in
  { desc with shape = new_shape; strides = new_strides; layout = new_layout }

let slice ?steps starts stops desc =
  let ndim = Array.length desc.shape in
  if Array.length starts <> ndim || Array.length stops <> ndim then
    invalid_arg
      (Format.asprintf
         "slice: starts (%d) / stops (%d) length mismatch with tensor rank %d"
         (Array.length starts) (Array.length stops) ndim);

  let actual_steps =
    match steps with
    | None -> Array.make ndim 1
    | Some s ->
        if Array.length s <> ndim then
          invalid_arg
            (Format.asprintf
               "slice: steps (%d) length mismatch with tensor rank %d"
               (Array.length s) ndim);
        Array.iter
          (fun st -> if st = 0 then invalid_arg "slice: step cannot be zero")
          s;
        s
  in

  let new_shape = Array.make ndim 0 in
  let normalized_starts = Array.make ndim 0 in

  for i = 0 to ndim - 1 do
    let dim_size = desc.shape.(i) in
    let start = starts.(i) in
    let stop = stops.(i) in
    let step = actual_steps.(i) in

    let norm_start = if start < 0 then start + dim_size else start in
    let norm_stop = if stop < 0 then stop + dim_size else stop in

    let final_start = max 0 (min norm_start dim_size) in
    let final_stop = max 0 (min norm_stop dim_size) in

    let count =
      if step > 0 then max 0 ((final_stop - final_start + step - 1) / step)
      else if step < 0 then max 0 ((final_stop - final_start + step + 1) / step)
      else 0 (* step = 0 handled above *)
    in

    new_shape.(i) <- count;
    normalized_starts.(i) <- final_start
  done;

  let new_strides = Array.mapi (fun i s -> s * actual_steps.(i)) desc.strides in
  let new_offset = desc.offset + md_to_linear normalized_starts desc.strides in
  let is_contig = check_c_contiguity_from_shape_strides new_shape new_strides in
  let new_layout =
    if is_contig && new_offset = 0 then C_contiguous else Strided
  in

  {
    desc with
    shape = new_shape;
    strides = new_strides;
    offset = new_offset;
    layout = new_layout;
  }

let transpose ?axes desc =
  let ndim = Array.length (shape desc) in
  let axes =
    match axes with
    | None -> Array.init ndim (fun i -> ndim - 1 - i)
    | Some ax ->
        if Array.length ax <> ndim then
          invalid_arg
            (Printf.sprintf
               "transpose: axes length %d does not match tensor rank %d"
               (Array.length ax) ndim);
        let seen = Array.make ndim false in
        let is_perm = ref true in
        Array.iter
          (fun x ->
            if x < 0 || x >= ndim then is_perm := false
            else if seen.(x) then is_perm := false
            else seen.(x) <- true)
          ax;
        if not !is_perm then
          invalid_arg
            (Format.asprintf
               "transpose: axes %a is not a valid permutation for rank %d"
               pp_int_array ax ndim);
        ax
  in

  let new_shape = Array.make ndim 0 in
  let new_strides = Array.make ndim 0 in
  for i = 0 to ndim - 1 do
    let old_dim_index = axes.(i) in
    new_shape.(i) <- (shape desc).(old_dim_index);
    new_strides.(i) <- (strides desc).(old_dim_index)
  done;

  let is_contig = check_c_contiguity_from_shape_strides new_shape new_strides in
  let new_layout = if is_contig then C_contiguous else Strided in

  {
    shape = new_shape;
    strides = new_strides;
    layout = new_layout;
    offset = 0;
    dtype = desc.dtype;
  }

(** [split ?axis sections desc] divides along [axis] into exactly [sections]
    equal slices, raising [Invalid_argument] if the dimension isn’t divisible.
*)
let split ?(axis = 0) sections (desc : ('a, 'b) descriptor) =
  let n = dim axis desc in
  if sections <= 0 then invalid_arg "split: number of sections must be > 0";
  if n mod sections <> 0 then
    invalid_arg
      (Printf.sprintf "split: size %d along axis %d not divisible by %d" n axis
         sections);
  let part_size = n / sections in
  let shape = Array.copy desc.shape in
  let ndim = Array.length shape in
  let rec build i acc =
    if i = sections then List.rev acc
    else
      let start = i * part_size in
      let stop = start + part_size in
      (* set up starts/stops for slice *)
      let starts = Array.make ndim 0 in
      let stops = Array.copy shape in
      starts.(axis) <- start;
      stops.(axis) <- stop;
      let piece = slice starts stops desc in
      build (i + 1) (piece :: acc)
  in
  build 0 []

(** [array_split ?axis sections desc] divides along [axis] into [sections]
    parts, letting the first [n mod sections] slices be one element larger. *)
let array_split ?(axis = 0) sections desc =
  let n = dim axis desc in
  if sections <= 0 then
    invalid_arg "array_split: number of sections must be > 0";
  let base = n / sections in
  let rem = n mod sections in
  (* compute size of each slice *)
  let sizes =
    Array.init sections (fun i -> if i < rem then base + 1 else base)
  in
  let shape = Array.copy desc.shape in
  let ndim = Array.length shape in
  let rec build i offset acc =
    if i = sections then List.rev acc
    else
      let sz = sizes.(i) in
      let start = offset in
      let stop = offset + sz in
      let starts = Array.make ndim 0 in
      let stops = Array.copy shape in
      starts.(axis) <- start;
      stops.(axis) <- stop;
      let piece = slice starts stops desc in
      build (i + 1) stop (piece :: acc)
  in
  build 0 0 []

(* Pure view operations *)
let flip ?axes (desc : ('a, 'b) descriptor) : ('a, 'b) descriptor =
  let shape_arr = desc.shape in
  let ndim = Array.length shape_arr in
  let axes =
    match axes with
    | None -> Array.init ndim (fun i -> i)
    | Some arr ->
        Array.map
          (fun ax ->
            let ax = if ax < 0 then ndim + ax else ax in
            if ax < 0 || ax >= ndim then
              invalid_arg
                (Format.asprintf "flip: axis %d out of bounds for shape %a" ax
                   pp_int_array shape_arr);
            ax)
          arr
  in
  let new_strides = Array.copy desc.strides in
  let new_offset = ref desc.offset in
  Array.iter
    (fun ax ->
      let len = shape_arr.(ax) in
      let stride = desc.strides.(ax) in
      new_strides.(ax) <- -stride;
      new_offset := !new_offset + (stride * (len - 1)))
    axes;
  { desc with strides = new_strides; offset = !new_offset; layout = Strided }

let swapaxes axis1 axis2 (desc : ('a, 'b) descriptor) : ('a, 'b) descriptor =
  let ndim = Array.length desc.shape in
  let ax1 = if axis1 < 0 then axis1 + ndim else axis1 in
  let ax2 = if axis2 < 0 then axis2 + ndim else axis2 in
  if ax1 < 0 || ax1 >= ndim || ax2 < 0 || ax2 >= ndim then
    invalid_arg
      (Format.asprintf "swapaxes: axes (%d, %d) out of bounds for shape %a"
         axis1 axis2 pp_int_array desc.shape);
  let perm =
    Array.init ndim (fun i ->
        if i = ax1 then ax2 else if i = ax2 then ax1 else i)
  in
  transpose ~axes:perm desc

let moveaxis source destination (desc : ('a, 'b) descriptor) :
    ('a, 'b) descriptor =
  let ndim = Array.length desc.shape in
  let src = if source < 0 then source + ndim else source in
  let dst = if destination < 0 then destination + ndim else destination in
  if src < 0 || src >= ndim || dst < 0 || dst > ndim then
    invalid_arg
      (Format.asprintf
         "moveaxis: source %d or destination %d out of bounds for shape %a"
         source destination pp_int_array desc.shape);
  let axes_list = Array.to_list (Array.init ndim (fun i -> i)) in
  let axes_list = List.filter (fun i -> i <> src) axes_list in
  let rec split_at n lst =
    if n <= 0 then ([], lst)
    else
      match lst with
      | [] -> ([], [])
      | x :: xs ->
          let l, r = split_at (n - 1) xs in
          (x :: l, r)
  in
  let left, right = split_at dst axes_list in
  let perm_list = left @ [ src ] @ right in
  let perm = Array.of_list perm_list in
  transpose ~axes:perm desc

let broadcast_arrays (descs : ('a, 'b) descriptor list) :
    ('a, 'b) descriptor list =
  match descs with
  | [] -> []
  | d0 :: ds ->
      let out_shape =
        List.fold_left broadcast_shapes (shape d0) (List.map shape ds)
      in
      List.map (fun d -> broadcast_to d out_shape) descs
