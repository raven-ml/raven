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

(* ───── accessors ───── *)

let shape v = v.shape
let strides v = v.strides

let stride axis v =
  if axis < 0 then
    Error.invalid ~op:"stride" ~what:"axis"
      ~reason:(Printf.sprintf "%d < 0" axis)
      ()
  else if axis >= Array.length v.strides then
    Error.axis_out_of_bounds ~op:"stride" ~axis ~ndim:(Array.length v.strides)
      ();
  let stride = Array.unsafe_get v.strides axis in
  stride

let offset v = v.offset
let mask v = v.mask
let is_c_contiguous v = v.layout = C_contiguous

let dim axis v =
  if axis < 0 then
    Error.invalid ~op:"dim" ~what:"axis"
      ~reason:(Printf.sprintf "%d < 0" axis)
      ()
  else if axis >= Array.length v.shape then
    Error.axis_out_of_bounds ~op:"dim" ~axis ~ndim:(Array.length v.shape) ();
  let dim = Array.unsafe_get v.shape axis in
  dim

let ndim v = Array.length v.shape

let numel v =
  let n = Array.length v.shape in
  if n = 0 then 1 else Array.fold_left ( * ) 1 v.shape

(* allocate a new view for [shape_arr] *)
let create ?(offset = 0) ?strides ?mask shape_arr =
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
      && current_strides = compute_strides current_shape
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

let linear_index view indices =
  if Array.length indices <> Array.length view.shape then
    Error.invalid ~op:"linear_index" ~what:"indices"
      ~reason:
        (Printf.sprintf "rank mismatch: %d vs %d" (Array.length indices)
           (Array.length view.shape))
      ();
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

(* ───── view manipulation ───── *)

let expand view new_shape =
  if Array.length new_shape <> Array.length view.shape then
    Error.invalid ~op:"expand" ~what:"shape dimensions"
      ~reason:
        (Printf.sprintf "rank mismatch: %d vs %d" (Array.length new_shape)
           (Array.length view.shape))
      ();
  if Array.exists (( = ) 0) view.shape then create new_shape
  else
    let strides =
      Array.mapi
        (fun i ns ->
          let s = view.shape.(i) in
          if s = ns then view.strides.(i)
          else if s = 1 then 0
          else
            (Error.cannot ~op:"expand" ~what:"expand"
               ~from:(Printf.sprintf "dimension %d (size %d)" i s)
               ~to_:(Printf.sprintf "size %d" ns)
               ~reason:"can only expand singleton dimensions")
              ())
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
                   else
                     Error.invalid ~op:"expand"
                       ~what:"masked singleton dimension"
                       ~reason:
                         (Printf.sprintf
                            "bounds [%d,%d] incompatible with expansion" b e)
                       ()
                 else (b, e))
               m)
    in
    create ~offset:view.offset ?mask ~strides new_shape

let permute view axes =
  let n = ndim view in
  if Array.length axes <> n then
    Error.invalid ~op:"permute" ~what:"axes array"
      ~reason:(Printf.sprintf "length %d != ndim %d" (Array.length axes) n)
      ();
  let seen = Array.make n false in
  Array.iter
    (fun ax ->
      if ax < 0 || ax >= n || seen.(ax) then
        Error.invalid ~op:"permute"
          ~what:(Printf.sprintf "axis %d" ax)
          ~reason:
            (if ax < 0 || ax >= n then "out of bounds" else "duplicate axis")
          ();
      seen.(ax) <- true)
    axes;
  if not (Array.for_all Fun.id seen) then
    Error.invalid ~op:"permute" ~what:"axes"
      ~reason:"do not form a valid permutation" ();

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
  (* Early return if shapes are identical *)
  if view.shape = new_shape then view
  else
    let current_numel = prod view.shape in
    let new_numel = prod new_shape in

    (* Check size compatibility *)
    if current_numel <> new_numel && current_numel <> 0 && new_numel <> 0 then
      Error.shape_mismatch ~op:"reshape" ~expected:new_shape ~actual:view.shape
        () (* Handle zero-size tensors *)
    else if
      Array.exists (( = ) 0) view.shape || Array.exists (( = ) 0) new_shape
    then create ~offset:0 new_shape
      (* Check for masks - these complicate reshape *)
    else if view.mask <> None then
      Error.failed ~op:"reshape" ~what:"cannot reshape views with masks"
        ~hint:"call contiguous() first to create a mask-free copy" ()
      (* Fast path for C-contiguous views *)
    else if view.layout = C_contiguous then create ~offset:view.offset new_shape
    else if
      (* Special case: reshaping to/from scalar *)
      Array.length new_shape = 0
    then
      (* Reshaping to scalar - always valid if size is 1 *)
      create ~offset:view.offset new_shape
      (* Special case: all strides are 0 (broadcast from scalar) *)
    else if Array.for_all (( = ) 0) view.strides then
      (* When all strides are 0, we have a broadcast view. 
         We can only reshape if all dimensions remain broadcast (stride 0) *)
      let new_strides = Array.make (Array.length new_shape) 0 in
      create ~offset:view.offset ~strides:new_strides new_shape
    (* Special case: only expanding/squeezing size-1 dimensions *)
      else
      let old_non_one = Array.to_list view.shape |> List.filter (( <> ) 1) in
      let new_non_one = Array.to_list new_shape |> List.filter (( <> ) 1) in

      if old_non_one = new_non_one then
        (* Just adding/removing size-1 dimensions *)
        let new_strides =
          let old_idx = ref 0 in
          Array.map
            (fun dim ->
              if dim = 1 then 0
              else (
                (* Skip size-1 dims in old shape *)
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
        create ~offset:view.offset ~strides:new_strides new_shape
      else
        let analyze_reshape old_shape old_strides new_shape =
          (* Can we map old dimensions to new dimensions? *)
          let try_match old_pos new_pos old_size new_size =
            match () with
            | _ when old_size = new_size ->
                (* Sizes match - check if we can reuse stride *)
                if old_pos + 1 <= Array.length old_shape then
                  Some [ (old_pos, new_pos, old_size) ]
                else None
            | _ when old_size < new_size && new_size mod old_size = 0 -> (
                (* Try merging old dimensions *)
                let rec accumulate pos acc =
                  if acc = new_size then
                    Some
                      (List.rev
                         (List.init (pos - old_pos) (fun i -> old_pos + i)))
                  else if pos >= Array.length old_shape || acc > new_size then
                    None
                  else accumulate (pos + 1) (acc * old_shape.(pos))
                in
                match accumulate (old_pos + 1) old_size with
                | Some merged_dims ->
                    (* Check if dimensions are contiguous in memory *)
                    let can_merge =
                      List.fold_left
                        (fun (ok, expected_stride) dim ->
                          if not ok then (false, 0)
                          else if dim = List.hd merged_dims then
                            (true, old_strides.(dim))
                          else
                            ( old_strides.(dim) = expected_stride,
                              old_strides.(dim) * old_shape.(dim) ))
                        (true, 0) merged_dims
                      |> fst
                    in
                    if can_merge then Some [ (old_pos, new_pos, new_size) ]
                    else None
                | None -> None)
            | _ when new_size < old_size && old_size mod new_size = 0 ->
                (* Try splitting old dimension *)
                let factor = old_size / new_size in
                (* For splitting to work, we need stride = 1 (contiguous) *)
                if old_strides.(old_pos) = 1 then
                  (* Can split: outer dim gets stride*factor, inner gets
                     stride *)
                  Some
                    [
                      (old_pos, new_pos, new_size);
                      (old_pos, new_pos + 1, factor);
                    ]
                else None
            | _ -> None
          in

          (* Build mapping between old and new dimensions *)
          let rec build_mapping old_idx new_idx mappings =
            if
              old_idx >= Array.length old_shape
              && new_idx >= Array.length new_shape
            then Some (List.rev mappings)
            else if
              old_idx >= Array.length old_shape
              || new_idx >= Array.length new_shape
            then None
            else
              match
                try_match old_idx new_idx old_shape.(old_idx)
                  new_shape.(new_idx)
              with
              | Some maps ->
                  let old_advance =
                    List.length
                      (List.filter (fun (o, _, _) -> o = old_idx) maps)
                  in
                  let new_advance = List.length maps in
                  build_mapping (old_idx + old_advance) (new_idx + new_advance)
                    (maps @ mappings)
              | None -> None
          in

          build_mapping 0 0 []
        in

        (* Compute new strides based on mapping *)
        match analyze_reshape view.shape view.strides new_shape with
        | Some mapping ->
            let new_strides = Array.make (Array.length new_shape) 0 in
            List.iter
              (fun (old_dim, new_dim, _size) ->
                new_strides.(new_dim) <- view.strides.(old_dim))
              mapping;
            create ~offset:view.offset ~strides:new_strides new_shape
        | None ->
            (* Fall back to existing logic or error *)
            Error.cannot ~op:"reshape" ~what:"reshape strided view"
              ~from:(Shape.to_string view.shape)
              ~to_:(Shape.to_string new_shape)
              ~reason:
                (Printf.sprintf "incompatible strides %s (expected %s)"
                   (Shape.to_string view.strides)
                   (Shape.to_string (compute_strides view.shape)))
              ~hint:
                "call contiguous() before reshape to create a C-contiguous copy"
              ()

(* helper used by [pad] and [shrink] *)
let unsafe_resize view arg new_mask_opt =
  if Array.length arg <> Array.length view.shape then
    Error.invalid ~op:"unsafe_resize" ~what:"argument array"
      ~reason:
        (Printf.sprintf "length %d != ndim %d" (Array.length arg)
           (Array.length view.shape))
      ();
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
    Error.invalid ~op:"pad" ~what:"padding array"
      ~reason:
        (Printf.sprintf "length %d != ndim %d" (Array.length arg)
           (Array.length view.shape))
      ();
  if Array.for_all (fun (b, e) -> b = 0 && e = 0) arg then view
  else if Array.exists (fun (b, e) -> b < 0 || e < 0) arg then
    Error.invalid ~op:"pad" ~what:"padding values"
      ~reason:"negative values not allowed"
      ~hint:"use shrink or slice to remove elements" ()
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
    Error.invalid ~op:"shrink" ~what:"bounds array"
      ~reason:
        (Printf.sprintf "length %d != ndim %d" (Array.length arg)
           (Array.length view.shape))
      ();
  if Array.for_all2 (fun (b, e) s -> b = 0 && e = s) arg view.shape then view
  else if
    Array.exists2
      (fun (b, e) s -> b < 0 || e < 0 || b > s || e > s || b >= e)
      arg view.shape
  then
    Error.invalid ~op:"shrink" ~what:"bounds"
      ~reason:"must be within shape and start < end" ()
  else unsafe_resize view arg None

let flip view flip_axes_bools =
  if Array.length flip_axes_bools <> Array.length view.shape then
    Error.invalid ~op:"flip" ~what:"boolean array"
      ~reason:
        (Printf.sprintf "length %d != ndim %d"
           (Array.length flip_axes_bools)
           (Array.length view.shape))
      ();
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
