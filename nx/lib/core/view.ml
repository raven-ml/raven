(* Lightweight view of tensor layout and helpers for reshaping. *)

type layout = C_contiguous | Strided

type t = {
  shape : Symbolic_shape.t;
  strides : int array; (* Always present, even for symbolic shapes *)
  offset : int;
  mask : (int * int) array option; (* bounds per dimension *)
  layout : layout;
}

(* ───── helpers ───── *)

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

(* drop strides for unit dimensions *)
let canonicalize_strides shape_array strides =
  Array.mapi (fun i s -> if shape_array.(i) = 1 then 0 else s) strides

(* Try to get concrete shape, return None if symbolic *)
let eval_shape_opt shape = Symbolic_shape.eval shape

(* Check if strides represent a contiguous layout *)
let is_c_contiguous_strides shape_arr strides offset mask =
  offset = 0 && mask = None
  &&
  let expected = compute_strides shape_arr in
  let expected_canonical = canonicalize_strides shape_arr expected in
  Array.length strides = Array.length expected_canonical
  && Array.for_all2 ( = ) strides expected_canonical

(* ───── accessors ───── *)

let shape v = v.shape
let strides v = v.strides

let stride axis v =
  let ndim = Symbolic_shape.rank v.shape in
  if axis < 0 || axis >= ndim then
    Error.axis_out_of_bounds ~op:"stride" ~axis ~ndim ();
  Array.unsafe_get v.strides axis

let offset v = v.offset
let mask v = v.mask
let is_c_contiguous v = v.layout = C_contiguous

let dim axis v =
  let ndim = Symbolic_shape.rank v.shape in
  if axis < 0 || axis >= ndim then
    Error.axis_out_of_bounds ~op:"dim" ~axis ~ndim ();
  v.shape.(axis)

let ndim v = Symbolic_shape.rank v.shape

let numel v =
  let n = Symbolic_shape.rank v.shape in
  if n = 0 then Symbolic_shape.static 1
  else
    let rec prod_dims i acc =
      if i >= n then acc
      else
        let next_acc =
          match
            (Symbolic_shape.eval_dim acc, Symbolic_shape.eval_dim v.shape.(i))
          with
          | Some a, Some b -> Symbolic_shape.static (a * b)
          | _ -> Symbolic_shape.mul acc v.shape.(i)
        in
        prod_dims (i + 1) next_acc
    in
    prod_dims 1 v.shape.(0)

(* ───── view creation ───── *)

let create ?(offset = 0) ?strides ?mask shape =
  match Symbolic_shape.eval shape with
  | Some shape_arr ->
      (* Shape is fully static - compute strides normally *)
      let is_zero_size = Array.exists (( = ) 0) shape_arr in
      let current_shape =
        if is_zero_size then
          Symbolic_shape.of_ints (Array.map (fun s -> max s 0) shape_arr)
        else shape
      in
      let current_strides =
        match strides with
        | Some s -> canonicalize_strides shape_arr s
        | None -> compute_strides shape_arr
      in
      let current_offset = if is_zero_size then 0 else offset in
      let current_mask =
        if is_zero_size then None
        else
          match mask with
          | Some m
            when Array.for_all2 (fun (b, e) s -> b = 0 && e = s) m shape_arr ->
              None
          | _ -> mask
      in
      let new_layout =
        if
          is_c_contiguous_strides shape_arr current_strides current_offset
            current_mask
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
  | None ->
      (* Shape has symbolic dimensions - create symbolic strides *)
      let symbolic_strides =
        match strides with
        | Some s -> s
        | None ->
            (* For symbolic shapes, we can't compute exact strides, but we can
               create placeholder strides that maintain C-contiguous pattern *)
            let n = Symbolic_shape.rank shape in
            if n = 0 then [||]
            else
              (* Create strides assuming unit size for unknown dimensions *)
              let strides = Array.make n 1 in
              for i = n - 2 downto 0 do
                strides.(i) <- strides.(i + 1)
              done;
              strides
      in
      let layout =
        if offset = 0 && mask = None && strides = None then C_contiguous
        else Strided
      in
      { shape; strides = symbolic_strides; offset; mask; layout }

(* ───── offset & validation ───── *)

let linear_index view indices =
  let ndim = Symbolic_shape.rank view.shape in
  if Array.length indices <> ndim then
    Error.invalid ~op:"linear_index" ~what:"indices"
      ~reason:
        (Printf.sprintf "rank mismatch: %d vs %d" (Array.length indices) ndim)
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

let can_be_strided view =
  (* All views now have strides, so check other conditions *)
  match (view.mask, eval_shape_opt view.shape) with
  | None, _ -> true
  | Some mask_array, Some shape_arr ->
      Array.for_all2 (fun (b, e) s -> b = 0 && e = s) mask_array shape_arr
  | Some _, None ->
      (* With symbolic shape and mask, conservatively return false *)
      false

(* ───── view manipulation ───── *)

let expand view new_shape =
  let old_ndim = Symbolic_shape.rank view.shape in
  let new_ndim = Symbolic_shape.rank new_shape in
  (* Allow expanding a scalar to any shape *)
  if old_ndim = 0 then
    (* Scalar case: broadcast to any shape *)
    let strides = Array.make new_ndim 0 in
    { view with shape = new_shape; strides }
  else if new_ndim <> old_ndim then
    Error.invalid ~op:"expand" ~what:"shape dimensions"
      ~reason:(Printf.sprintf "rank mismatch: %d vs %d" new_ndim old_ndim)
      ()
  else
    match (Symbolic_shape.eval view.shape, Symbolic_shape.eval new_shape) with
  | Some old_arr, Some new_arr ->
      if Array.exists (( = ) 0) old_arr then create new_shape
      else
        let strides =
          Array.mapi
            (fun i ns ->
              let s = old_arr.(i) in
              if s = ns then view.strides.(i)
              else if s = 1 then 0
              else
                Error.cannot ~op:"expand" ~what:"expand"
                  ~from:(Printf.sprintf "dimension %d (size %d)" i s)
                  ~to_:(Printf.sprintf "size %d" ns)
                  ~reason:"can only expand singleton dimensions" ())
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
  | _, _ ->
      (* At least one shape is symbolic - create view with appropriate
         strides *)
      create ~offset:view.offset ?mask:view.mask ~strides:view.strides new_shape

let permute view axes =
  let n = ndim view in
  if Array.length axes <> n then
    Error.invalid ~op:"permute" ~what:"axes array"
      ~reason:(Printf.sprintf "length %d != ndim %d" (Array.length axes) n)
      ();

  (* Validate permutation *)
  let seen = Array.make n false in
  Array.iter
    (fun ax ->
      if ax < 0 || ax >= n then
        Error.axis_out_of_bounds ~op:"permute" ~axis:ax ~ndim:n ();
      if seen.(ax) then
        Error.invalid ~op:"permute"
          ~what:(Printf.sprintf "axis %d" ax)
          ~reason:"duplicate axis" ();
      seen.(ax) <- true)
    axes;

  let new_shape = Array.init n (fun i -> view.shape.(axes.(i))) in
  let new_strides = Array.init n (fun i -> view.strides.(axes.(i))) in
  let new_mask =
    Option.map (fun m -> Array.init n (fun i -> m.(axes.(i)))) view.mask
  in
  create ~offset:view.offset ?mask:new_mask ~strides:new_strides
    (Array.of_list (Array.to_list new_shape))

let reshape view new_shape =
  (* Early return if shapes are identical *)
  if Symbolic_shape.equal view.shape new_shape then view
  else
    match (Symbolic_shape.eval view.shape, Symbolic_shape.eval new_shape) with
    | Some old_arr, Some new_arr -> (
        (* Both shapes are concrete *)
        let old_numel = prod old_arr in
        let new_numel = prod new_arr in

        (* Check size compatibility *)
        if old_numel <> new_numel && old_numel <> 0 && new_numel <> 0 then
          Error.shape_mismatch ~op:"reshape" ~expected:new_arr ~actual:old_arr
            () (* Handle zero-size tensors *)
        else if Array.exists (( = ) 0) old_arr || Array.exists (( = ) 0) new_arr
        then create ~offset:0 new_shape
          (* Check for masks - these complicate reshape *)
        else if view.mask <> None then
          Error.failed ~op:"reshape" ~what:"cannot reshape views with masks"
            ~hint:"call contiguous() first to create a mask-free copy" ()
          (* Fast path for C-contiguous views *)
        else if view.layout = C_contiguous then
          create ~offset:view.offset new_shape
        else if
          (* Special case: reshaping to/from scalar *)
          Array.length new_shape = 0
        then
          (* Reshaping to scalar - always valid if size is 1 *)
          create ~offset:view.offset new_shape
          (* Special case: all strides are 0 (broadcast from scalar) *)
        else if Array.for_all (( = ) 0) view.strides then
          (* When all strides are 0, we have a broadcast view. We can only
             reshape if all dimensions remain broadcast (stride 0) *)
          let new_strides = Array.make (Array.length new_shape) 0 in
          create ~offset:view.offset ~strides:new_strides new_shape
        (* Special case: only expanding/squeezing size-1 dimensions *)
          else
          let _old_non_one = Array.to_list old_arr |> List.filter (( <> ) 1) in
          let _new_non_one = Array.to_list new_arr |> List.filter (( <> ) 1) in

          (* Helper: try to reshape by only adding/removing size-1 dimensions *)
          let try_squeeze_unsqueeze () =
            let old_non_one = Array.to_list old_arr |> List.filter (( <> ) 1) in
            let new_non_one = Array.to_list new_arr |> List.filter (( <> ) 1) in

            if old_non_one = new_non_one then
              (* Just adding/removing size-1 dimensions *)
              let old_idx = ref 0 in
              let new_strides =
                Array.map
                  (fun dim ->
                    if dim = 1 then 0
                    else (
                      (* Skip size-1 dims in old shape *)
                      while
                        !old_idx < Array.length old_arr
                        && old_arr.(!old_idx) = 1
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

          (* Helper: check if we can reshape by merging/splitting dimensions *)
          let try_merge_split () =
            (* Build list of non-unit dimensions with their properties *)
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

            (* Try to match old and new dimensions *)
            let rec match_dims old_dims new_dims =
              match (old_dims, new_dims) with
              | [], [] -> Some []
              | [], _ | _, [] -> None
              | (old_size, old_stride) :: old_rest, new_size :: new_rest ->
                  if old_size = new_size then
                    (* Direct match *)
                    match match_dims old_rest new_rest with
                    | Some rest_strides ->
                        Some ((new_size, old_stride) :: rest_strides)
                    | None -> None
                  else if old_size > new_size && old_size mod new_size = 0 then
                    (* Split dimension - this works even for non-contiguous
                       dims *)
                    let remaining_size = old_size / new_size in
                    let remaining_stride = old_stride * new_size in
                    let remaining_dims =
                      (remaining_size, remaining_stride) :: old_rest
                    in
                    match match_dims remaining_dims new_rest with
                    | Some rest_strides ->
                        Some ((new_size, old_stride) :: rest_strides)
                    | None -> None
                  else if new_size > old_size then
                    (* Try merging dimensions - they must be contiguous *)
                    let rec collect_merge size stride dims needed =
                      if size = needed then Some (dims, stride)
                      else if size > needed then None
                      else
                        match dims with
                        | [] -> None
                        | (next_size, next_stride) :: rest ->
                            (* For C-contiguous merging: current_stride = next_stride * next_size *)
                            (* Check if next dimension is contiguous with current *)
                            if stride = next_stride * next_size then
                              collect_merge (size * next_size) next_stride rest
                                needed
                            else None (* Not contiguous, can't merge *)
                    in
                    match
                      collect_merge old_size old_stride old_rest new_size
                    with
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
                (* Build final strides array, including size-1 dimensions *)
                let new_strides = Array.make (Array.length new_arr) 0 in
                let map_idx = ref 0 in

                for i = 0 to Array.length new_arr - 1 do
                  if new_arr.(i) = 1 then new_strides.(i) <- 0
                  else
                    let _, stride = List.nth stride_map !map_idx in
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
                  (* Include stride information in error message *)
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
                        (Array.to_list
                           (Array.map string_of_int expected_strides))
                    ^ "]"
                  in
                  Error.cannot ~op:"reshape" ~what:"reshape strided view"
                    ~from:(Shape.to_string old_arr)
                    ~to_:(Shape.to_string new_arr)
                    ~reason:
                      (Printf.sprintf "incompatible strides %s (expected %s)"
                         stride_str expected_str)
                    ~hint:
                      "call contiguous() before reshape to create a \
                       C-contiguous copy"
                    ()))
    | _, _ ->
        (* At least one shape is symbolic *)
        if view.layout = C_contiguous then create ~offset:view.offset new_shape
        else
          Error.failed ~op:"reshape"
            ~what:"cannot reshape symbolic non-contiguous view"
            ~hint:"bind all symbolic dimensions before reshaping" ()

(* helper used by [pad] and [shrink] *)
let unsafe_resize view arg new_mask_opt =
  let ndim = Symbolic_shape.rank view.shape in
  if Array.length arg <> ndim then
    Error.invalid ~op:"unsafe_resize" ~what:"argument array"
      ~reason:(Printf.sprintf "length %d != ndim %d" (Array.length arg) ndim)
      ();

  (* Don't require concrete shape here - work with what we have *)
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
  create ~offset:!new_offset ?mask:final_mask ~strides
    (Symbolic_shape.of_ints new_shape)

let pad view arg =
  let ndim = Symbolic_shape.rank view.shape in
  if Array.length arg <> ndim then
    Error.invalid ~op:"pad" ~what:"padding array"
      ~reason:(Printf.sprintf "length %d != ndim %d" (Array.length arg) ndim)
      ();
  if Array.for_all (fun (b, e) -> b = 0 && e = 0) arg then view
  else if Array.exists (fun (b, e) -> b < 0 || e < 0) arg then
    Error.invalid ~op:"pad" ~what:"padding values"
      ~reason:"negative values not allowed"
      ~hint:"use shrink or slice to remove elements" ()
  else
    match eval_shape_opt view.shape with
    | None ->
        Error.failed ~op:"pad" ~what:"cannot pad symbolic shape"
          ~hint:"bind all symbolic dimensions before padding" ()
    | Some shape_arr ->
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
  let ndim = Symbolic_shape.rank view.shape in
  if Array.length arg <> ndim then
    Error.invalid ~op:"shrink" ~what:"bounds array"
      ~reason:(Printf.sprintf "length %d != ndim %d" (Array.length arg) ndim)
      ();
  match eval_shape_opt view.shape with
  | None ->
      Error.failed ~op:"shrink" ~what:"cannot shrink symbolic shape"
        ~hint:"bind all symbolic dimensions before shrinking" ()
  | Some shape_arr ->
      if Array.for_all2 (fun (b, e) s -> b = 0 && e = s) arg shape_arr then view
      else if
        Array.exists2
          (fun (b, e) s -> b < 0 || e < 0 || b > s || e > s || b >= e)
          arg shape_arr
      then
        Error.invalid ~op:"shrink" ~what:"bounds"
          ~reason:"must be within shape and start < end" ()
      else unsafe_resize view arg None

let flip view flip_axes_bools =
  let ndim = Symbolic_shape.rank view.shape in
  if Array.length flip_axes_bools <> ndim then
    Error.invalid ~op:"flip" ~what:"boolean array"
      ~reason:
        (Printf.sprintf "length %d != ndim %d"
           (Array.length flip_axes_bools)
           ndim)
      ();

  match eval_shape_opt view.shape with
  | None ->
      Error.failed ~op:"flip" ~what:"cannot flip symbolic shape"
        ~hint:"bind all symbolic dimensions before flipping" ()
  | Some shape_arr ->
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

(* ───── view merging ───── *)

(* Try to merge two views into one *)
let merge v1 v2 =
  (* Case 1: v2 is contiguous - v1 doesn't matter *)
  if v2.layout = C_contiguous then Some v2
    (* Case 2: v1 is contiguous and same shape - use v2 *)
  else if v1.layout = C_contiguous && Symbolic_shape.equal v1.shape v2.shape
  then Some v2
    (* Case 3: Handle symbolic shapes - if either shape is symbolic, try
       conservative merge *)
  else if
    (not (Symbolic_shape.is_static v1.shape))
    || not (Symbolic_shape.is_static v2.shape)
  then
    (* For symbolic shapes, we can still merge in some cases *)
    if v1.layout = C_contiguous then
      (* v1 is contiguous, v2 operates on it - often mergeable *)
      Some { v2 with offset = v1.offset + v2.offset }
    else
      (* Don't try to merge non-contiguous views with symbolic shapes *)
      None (* Case 4: v1 is contiguous and can be reshaped to v2 *)
  else if v1.layout = C_contiguous then
    match (Symbolic_shape.eval v1.shape, Symbolic_shape.eval v2.shape) with
    | Some s1_arr, Some s2_arr when prod s1_arr = prod s2_arr -> (
        (* v1 is C-contiguous and has same numel as v2 *)
        (* Check if v2 represents a permutation that would be lost by reshape *)
        let v2_is_permutation =
          (* A view is a permutation if it has the same strides values but in different order *)
          (* For example, transpose of [2,3] has strides [1,3] instead of [3,1] *)
          (* But we need to be careful with zero strides from size-1 dimensions *)
          match (Symbolic_shape.eval v2.shape, v2.strides) with
          | Some shape_arr, strides ->
              (* Filter out strides for size-1 dimensions (which have stride
                 0) *)
              let non_unit_pairs =
                Array.to_list
                  (Array.mapi
                     (fun i _ -> (shape_arr.(i), strides.(i)))
                     shape_arr)
                |> List.filter (fun (size, _) -> size > 1)
              in
              if List.length non_unit_pairs < 2 then
                (* Less than 2 non-unit dimensions - can't be a meaningful
                   permutation *)
                false
              else
                (* Check if the non-zero strides form a permutation *)
                (* For transpose, we'd have different stride ordering *)
                let strides_only =
                  List.map snd non_unit_pairs |> Array.of_list
                in
                let expected_strides =
                  let sizes_only =
                    List.map fst non_unit_pairs |> Array.of_list
                  in
                  compute_strides sizes_only
                in
                (* Check if strides match expected pattern for this shape *)
                (* A permutation would have different stride ordering *)
                not (Array.for_all2 ( = ) strides_only expected_strides)
          | _ -> false
        in
        if v2_is_permutation then
          (* v2 is a permutation (like transpose) - can't merge without losing
             info *)
          None
        else
          (* v2 is not a permutation - safe to reshape v1 to v2's shape *)
          try Some (reshape v1 v2.shape) with _ -> None)
    | _ -> None (* Case 5: Handle operations on broadcast views *)
  else if Array.exists (( = ) 0) v1.strides then
    (* v1 is a broadcast view - operations on it maintain broadcast semantics *)
    match Symbolic_shape.eval v2.shape with
    | Some _s2_arr ->
        (* For broadcast views, most operations just update shape/offset *)
        (* The key is that strides remain broadcast (0) for broadcast dimensions *)
        Some v2
    | _ -> None
    (* Case 6: Handle expand operations - if v2 is an expand of v1 *)
  else if v1.mask = None && v2.mask = None then
    match (Symbolic_shape.eval v1.shape, Symbolic_shape.eval v2.shape) with
    | Some s1_arr, Some s2_arr when Array.length s1_arr = Array.length s2_arr ->
        (* Check if v2 could be an expand of v1 *)
        let is_expand = ref true in
        for i = 0 to Array.length s1_arr - 1 do
          if s1_arr.(i) <> s2_arr.(i) && s1_arr.(i) <> 1 then is_expand := false
        done;
        (* Also check that strides have the same sign (no flips) *)
        let has_negative_v1 = Array.exists (fun s -> s < 0) v1.strides in
        let has_negative_v2 = Array.exists (fun s -> s < 0) v2.strides in
        if !is_expand && has_negative_v1 = has_negative_v2 then
          (* v2 is an expand of v1 - we can merge by applying expand to v1 *)
          try Some (expand v1 v2.shape) with _ -> None
        else None
    | _ -> None (* Case 7: Handle shrink views - if v2 has a mask *)
  else if v2.mask <> None && v1.mask = None then
    (* v2 is a shrink/slice operation - try to preserve it *)
    (* This handles the common case of slicing a regular tensor *)
    Some v2
  (* Default: can't merge *)
    else None

let simplify view =
  match Symbolic_shape.eval view.shape with
  | None -> view (* Can't simplify symbolic shapes *)
  | Some shape_arr ->
      (* Only simplify things that don't change the user-visible shape *)

      (* 1. Canonicalize mask that covers entire dimensions *)
      let mask =
        match view.mask with
        | Some m
          when Array.for_all2 (fun (b, e) s -> b = 0 && e = s) m shape_arr ->
            None (* Mask covers everything, remove it *)
        | m -> m
      in

      (* 2. Don't reshape - that changes user-visible shape! *)
      (* Just return with simplified mask if changed *)
      if mask <> view.mask then
        (* Don't use create here - it canonicalizes strides and loses negative values *)
        (* Just update the mask field and recalculate layout *)
        let new_layout =
          if
            mask = None && view.offset = 0
            && is_c_contiguous_strides shape_arr view.strides 0 None
          then C_contiguous
          else Strided
        in
        { view with mask; layout = new_layout }
      else view (* No simplification possible *)
