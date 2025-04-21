open Descriptor
open Views

module Make (B : Backend_intf.S) = struct
  let transpose _ctx ?axes t =
    let desc = transpose ?axes (B.descriptor t) in
    B.view desc t

  let broadcast_to _ctx new_shape t =
    let desc = broadcast_to (B.descriptor t) new_shape in
    B.view desc t

  let squeeze _ctx ?axes t =
    let desc = squeeze ?axes (B.descriptor t) in
    B.view desc t

  let expand_dims _ctx axis t =
    let desc = expand_dims axis (B.descriptor t) in
    B.view desc t

  let slice _ctx ?steps starts stops t =
    let desc = slice ?steps starts stops (B.descriptor t) in
    B.view desc t

  let split _ctx ?(axis = 0) sections t =
    let descs = split ~axis sections (B.descriptor t) in
    List.map (fun d -> B.view d t) descs

  let array_split _ctx ?(axis = 0) sections t =
    let descs = array_split ~axis sections (B.descriptor t) in
    List.map (fun d -> B.view d t) descs

  (* Pure view wrappers *)
  let flip _ctx ?axes t =
    let desc = flip ?axes (B.descriptor t) in
    B.view desc t

  let swapaxes _ctx axis1 axis2 t =
    let desc = swapaxes axis1 axis2 (B.descriptor t) in
    B.view desc t

  let moveaxis _ctx src dst t =
    let desc = moveaxis src dst (B.descriptor t) in
    B.view desc t

  (** [concatenate ctx ~axis ts] joins [ts] along [axis]. *)
  let concatenate ctx ?(axis = 0) ts =
    match ts with
    | [] -> invalid_arg "concatenate: empty list"
    | t0 :: _ ->
        (* 1) Descriptor checks *)
        let d0 = B.descriptor t0 in
        let ndim = Array.length d0.shape in
        if axis < 0 || axis >= ndim then
          invalid_arg
            (Format.asprintf "concatenate: axis %d out of bounds for shape %a"
               axis pp_int_array d0.shape);
        (* verify uniform dtype & shapes (except along axis) *)
        List.iter
          (fun t ->
            let d = B.descriptor t in
            if d.dtype <> d0.dtype then
              invalid_arg "concatenate: all arrays must have the same dtype";
            if Array.length d.shape <> ndim then
              invalid_arg "concatenate: rank mismatch";
            for i = 0 to ndim - 1 do
              if i <> axis && d.shape.(i) <> d0.shape.(i) then
                invalid_arg
                  (Printf.sprintf
                     "concatenate: shape mismatch at dimension %d (%d vs %d)" i
                     d.shape.(i) d0.shape.(i))
            done)
          ts;
        (* 2) build output shape *)
        let out_shape = Array.copy d0.shape in
        out_shape.(axis) <-
          List.fold_left (fun acc t -> acc + (B.descriptor t).shape.(axis)) 0 ts;
        (* 3) allocate and scatter *)
        let result = B.empty ctx d0.dtype out_shape in
        let offset = ref 0 in
        List.iter
          (fun t ->
            let dt = B.descriptor t in
            let len = dt.shape.(axis) in
            (* slice region [offset, offset+len) along [axis] *)
            let starts = Array.make ndim 0 in
            let stops = Array.copy out_shape in
            starts.(axis) <- !offset;
            stops.(axis) <- !offset + len;
            let slot_desc = Views.slice starts stops (B.descriptor result) in
            let slot = B.view slot_desc result in
            B.blit ctx t slot;
            offset := !offset + len)
          ts;
        result

  (** [stack ctx ~axis ts] adds a new axis at [axis], then concatenates. *)
  let stack ctx ?(axis = 0) ts =
    match ts with
    | [] -> invalid_arg "stack: need at least one tensor"
    | _ ->
        (* expand each into a 1‑slice along the new axis *)
        let expanded = List.map (fun t -> expand_dims ctx axis t) ts in
        (* now just concatenate along that axis *)
        concatenate ctx ~axis expanded

  let reshape ctx new_shape t =
    let t_desc = B.descriptor t in
    let old_size = size t_desc in
    let new_size = Array.fold_left ( * ) 1 new_shape in
    if old_size <> new_size then
      invalid_arg
        (Format.asprintf
           "reshape: cannot reshape array of size %d into shape %a" old_size
           pp_int_array new_shape);

    if is_c_contiguous t_desc then
      let new_strides = compute_c_strides new_shape in
      B.view
        {
          t_desc with
          shape = new_shape;
          strides = new_strides;
          layout = C_contiguous;
        }
        t
    else
      (* Attempt to calculate strides for a view *)
      let rec can_compute_strides old_shape old_strides new_shape new_strides
          o_idx n_idx current_stride =
        if n_idx < 0 then true
        else if o_idx < 0 then
          if new_shape.(n_idx) = 1 then (
            new_strides.(n_idx) <- 0;
            (* Or some other appropriate stride *)
            can_compute_strides old_shape old_strides new_shape new_strides
              o_idx (n_idx - 1) current_stride)
          else false
        else if old_shape.(o_idx) = new_shape.(n_idx) then (
          new_strides.(n_idx) <- old_strides.(o_idx);
          can_compute_strides old_shape old_strides new_shape new_strides
            (o_idx - 1) (n_idx - 1)
            (current_stride * new_shape.(n_idx)))
        else if old_shape.(o_idx) = 1 then
          can_compute_strides old_shape old_strides new_shape new_strides
            (o_idx - 1) n_idx current_stride
        else if new_shape.(n_idx) = 1 then (
          new_strides.(n_idx) <- 0;
          (* Or some other appropriate stride *)
          can_compute_strides old_shape old_strides new_shape new_strides o_idx
            (n_idx - 1) current_stride)
        else false
      in
      let temp_new_strides = Array.make (Array.length new_shape) 0 in
      if
        can_compute_strides (shape t_desc) (strides t_desc) new_shape
          temp_new_strides
          (ndim t_desc - 1)
          (Array.length new_shape - 1)
          1
      then
        (* Create view *)
        let is_contig =
          check_c_contiguity_from_shape_strides new_shape temp_new_strides
        in
        let new_layout = if is_contig then C_contiguous else Strided in
        B.view
          {
            t_desc with
            shape = new_shape;
            strides = temp_new_strides;
            layout = new_layout;
          }
          t
      else
        (* Cannot create a view, fallback to copy *)
        let contiguous_t = B.copy ctx t in
        let final_strides = compute_c_strides new_shape in
        B.view
          {
            (B.descriptor contiguous_t) with
            shape = new_shape;
            strides = final_strides;
            layout = C_contiguous;
          }
          contiguous_t

  let flatten ctx t =
    let t_desc = B.descriptor t in
    let s = size t_desc in
    let new_shape = if s = 0 then [| 0 |] else [| s |] in
    if is_c_contiguous t_desc then
      let new_strides = compute_c_strides new_shape in
      B.view
        {
          t_desc with
          shape = new_shape;
          strides = new_strides;
          layout = C_contiguous;
        }
        t
    else
      let contiguous_copy = B.copy ctx t in
      let new_strides = compute_c_strides new_shape in
      B.view
        {
          (B.descriptor contiguous_copy) with
          shape = new_shape;
          strides = new_strides;
          layout = C_contiguous;
        }
        contiguous_copy

  let ravel = flatten

  (* Circularly roll array elements along a given axis. *)
  let rec roll ctx ?axis shift t =
    let d = B.descriptor t in
    let dims = shape d in
    let ndim = Array.length dims in
    match axis with
    | None ->
        let flat = flatten ctx t in
        let rolled = roll ctx ~axis:0 shift flat in
        reshape ctx (shape d) rolled
    | Some ax ->
        let ax = if ax < 0 then ax + ndim else ax in
        if ax < 0 || ax >= ndim then
          invalid_arg
            (Format.asprintf "roll: axis %d out of bounds for shape %a" ax
               pp_int_array dims);
        let n = dims.(ax) in
        if n = 0 then t
        else
          let s = ((shift mod n) + n) mod n in
          if s = 0 then t
          else
            let starts1 = Array.make ndim 0 in
            let stops1 = Array.copy dims in
            let starts2 = Array.make ndim 0 in
            let stops2 = Array.copy dims in
            stops1.(ax) <- n - s;
            starts2.(ax) <- n - s;
            let part1 = slice ctx starts1 stops1 t in
            let part2 = slice ctx starts2 stops2 t in
            concatenate ctx ~axis:ax [ part2; part1 ]

  let pad ctx padding value t =
    let t_desc = B.descriptor t in
    let ndim = ndim t_desc in
    let t_shape = shape t_desc in
    let t_dtype = dtype t_desc in

    if Array.length padding <> ndim then
      invalid_arg
        (Format.asprintf "pad: padding length %d does not match tensor rank %d"
           (Array.length padding) ndim);

    let starts = Array.make ndim 0 in
    (* Start index for slice in result *)
    let stops = Array.make ndim 0 in
    (* Stop index for slice in result *)
    let new_shape =
      Array.mapi
        (fun i dim_size ->
          if i >= Array.length padding then
            failwith "pad: internal error - padding length";
          let pad_before, pad_after = padding.(i) in
          if pad_before < 0 || pad_after < 0 then
            invalid_arg
              (Format.asprintf
                 "pad: padding values must be non-negative, got (%d, %d) for \
                  axis %d"
                 pad_before pad_after i);
          starts.(i) <- pad_before;
          stops.(i) <- pad_before + dim_size;
          (* End coordinate is exclusive *)
          dim_size + pad_before + pad_after)
        t_shape
    in

    let result = B.empty ctx t_dtype new_shape in
    B.fill ctx value result;
    let target_view = slice ctx starts stops result in
    B.blit ctx t target_view;
    result

  let broadcast_arrays _ctx ts =
    let descs = broadcast_arrays (List.map B.descriptor ts) in
    List.map2 (fun d t -> B.view d t) descs ts

  let vstack ctx ts =
    match ts with
    | [] -> invalid_arg "vstack: need at least one tensor"
    | _ :: _ ->
        let ts' =
          List.map
            (fun t ->
              let d = B.descriptor t in
              if Array.length (shape d) = 1 then expand_dims ctx 0 t else t)
            ts
        in
        concatenate ctx ~axis:0 ts'

  let hstack ctx ts =
    match ts with
    | [] -> invalid_arg "hstack: need at least one tensor"
    | t0 :: _ ->
        let d0 = B.descriptor t0 in
        let axis = if Array.length (shape d0) <= 1 then 0 else 1 in
        concatenate ctx ~axis ts

  let dstack ctx ts =
    match ts with
    | [] -> invalid_arg "dstack: need at least one tensor"
    | _ :: _ ->
        let ts' =
          List.map
            (fun t ->
              let d = B.descriptor t in
              match Array.length (shape d) with
              | 0 ->
                  expand_dims ctx 0 t |> expand_dims ctx 1 |> expand_dims ctx 2
              | 1 -> expand_dims ctx 0 t |> expand_dims ctx 2
              | 2 -> expand_dims ctx 2 t
              | _ -> t)
            ts
        in
        concatenate ctx ~axis:2 ts'

  let repeat ctx ?axis count t =
    if count < 0 then invalid_arg "repeat: count must be non-negative";
    let t', axis =
      match axis with Some a -> (t, a) | None -> (flatten ctx t, 0)
    in
    let d = B.descriptor t' in
    let sh = shape d in
    if axis < 0 || axis >= Array.length sh then
      invalid_arg "repeat: axis out of bounds";
    let old = sh.(axis) in
    let new_ = old * count in
    let new_sh = Array.copy sh in
    new_sh.(axis) <- new_;
    let t_dtype = dtype d in
    let result = B.empty ctx t_dtype new_sh in
    let rec aux i =
      if i >= old then ()
      else
        let starts = Array.make (Array.length sh) 0 in
        let stops = Array.copy sh in
        starts.(axis) <- i;
        stops.(axis) <- i + 1;
        let src = slice ctx starts stops t' in
        for j = 0 to count - 1 do
          let dstarts = Array.copy starts in
          let dstops = Array.copy stops in
          dstarts.(axis) <- (i * count) + j;
          dstops.(axis) <- (i * count) + j + 1;
          let dest = slice ctx dstarts dstops result in
          B.blit ctx src dest
        done;
        aux (i + 1)
    in
    aux 0;
    result

  let tile ctx reps t =
    let d = B.descriptor t in
    let sh = shape d in
    let n = Array.length sh in
    let rlen = Array.length reps in
    let reps' =
      if rlen < n then Array.append (Array.make (n - rlen) 1) reps
      else if rlen > n then
        invalid_arg "tile: reps length must be <= tensor rank"
      else reps
    in
    let new_sh = Array.init n (fun i -> sh.(i) * reps'.(i)) in
    let t_dtype = dtype d in
    let result = B.empty ctx t_dtype new_sh in
    let idx = Array.make n 0 in
    let rec loop axis =
      if axis = n then
        let starts = Array.init n (fun i -> idx.(i) * sh.(i)) in
        let stops = Array.init n (fun i -> starts.(i) + sh.(i)) in
        let dest = slice ctx starts stops result in
        B.blit ctx t dest
      else
        for k = 0 to reps'.(axis) - 1 do
          idx.(axis) <- k;
          loop (axis + 1)
        done
    in
    loop 0;
    result
end
