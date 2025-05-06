type layout = C_contiguous | Strided

type view = {
  shape : int array;
  strides : int array;
  offset : int;
  layout : layout;
  (* todo: add mask *)
}

(* Helpers *)

let pp_int_array fmt arr =
  let n = Array.length arr in
  Format.fprintf fmt "[";
  for i = 0 to n - 1 do
    Format.fprintf fmt "%d%s" arr.(i) (if i < n - 1 then "; " else "")
  done;
  Format.fprintf fmt "]"

let split_n lst n =
  (* Return (first n elems, rest). If [lst] is shorter than [n], the first
     component contains the whole list. *)
  let rec aux i acc = function
    | [] as l -> (List.rev acc, l)
    | l when i = 0 -> (List.rev acc, l)
    | h :: t -> aux (i - 1) (h :: acc) t
  in
  aux n [] lst

(* Basic utilities *)

let ndim v = Array.length v.shape
let size v = if ndim v = 0 then 1 else Array.fold_left ( * ) 1 v.shape
let dim v axis = v.shape.(axis)
let stride v axis = v.strides.(axis)
let shape v = v.shape

(** Compute canonical C‑contiguous strides given a shape. Zero‑length dimensions
    produce stride 0 to avoid overflow and match NumPy / PyTorch behaviour. *)
let compute_c_strides shape =
  let n = Array.length shape in
  if n = 0 then [||]
  else
    let s = Array.make n 0 in
    s.(n - 1) <- 1;
    for k = n - 2 downto 0 do
      s.(k) <-
        (if shape.(k + 1) = 0 then 0 else s.(k + 1) * max 1 shape.(k + 1))
    done;
    s

(* Accurate contiguity check – ignores offset and sign of stride. *)
let check_c_contiguity_from_shape_strides shape strides =
  let n = Array.length shape in
  if Array.length strides <> n then false
  else
    let expected = ref 1 in
    let ok = ref true in
    for i = n - 1 downto 0 do
      let size = shape.(i) in
      let st = abs strides.(i) in
      if size <= 1 then ()
        (* singleton & broadcast dims don’t break contiguity *)
      else if st <> !expected then ok := false;
      expected := st * size
    done;
    !ok

let is_c_contiguous v = check_c_contiguity_from_shape_strides v.shape v.strides

let make_view ?(offset = 0) ?strides shape =
  let strides =
    match strides with Some s -> s | None -> compute_c_strides shape
  in
  let layout =
    if check_c_contiguity_from_shape_strides shape strides then C_contiguous
    else Strided
  in
  { shape; strides; offset; layout }

(* Index/offset helpers *)

(** Convert a linear offset ([k]) into a multi‑dimensional index for a C‑
    contiguous tensor described by [shape]. Raises if [k] is out of range. *)
let offset_to_index_contig k shape =
  let n = Array.length shape in
  if n = 0 then
    if k = 0 then [||]
    else invalid_arg "offset_to_index_contig: scalar out of bounds"
  else
    let idx = Array.make n 0 in
    let tmp = ref k in
    for i = n - 1 downto 1 do
      idx.(i) <- !tmp mod shape.(i);
      tmp := !tmp / shape.(i)
    done;
    idx.(0) <- !tmp;
    idx

(** Convert a multi‑dimensional index into a linear offset using [strides]. *)
let index_to_offset md strides =
  if Array.length md <> Array.length strides then
    invalid_arg "index_to_offset: rank mismatch";
  let o = ref 0 in
  Array.iteri (fun i v -> o := !o + (v * strides.(i))) md;
  !o

(** Given a *target* multi‑index and a *source* shape (broadcasted to target),
    compute the corresponding index into the source view. *)
let broadcast_index target_idx source_shape =
  let tgt_nd = Array.length target_idx and src_nd = Array.length source_shape in
  let out = Array.make src_nd 0 in
  for i = 0 to src_nd - 1 do
    let tgt_pos = tgt_nd - src_nd + i in
    if source_shape.(i) <> 1 then out.(i) <- target_idx.(tgt_pos)
  done;
  out

(** Iterate over all valid indices of the given [shape]. *)
let iter_indices shape f =
  let n = Array.length shape in
  if n = 0 then f [||]
  else if Array.exists (( = ) 0) shape then ()
  else
    let idx = Array.make n 0 in
    let rec loop d =
      if d = n then f (Array.copy idx)
      else
        for i = 0 to shape.(d) - 1 do
          idx.(d) <- i;
          loop (d + 1)
        done
    in
    loop 0

(* Broadcasting *)

let broadcast_shapes a b =
  let na = Array.length a and nb = Array.length b in
  let n = max na nb in
  let out = Array.make n 1 in
  for i = 0 to n - 1 do
    let da = if i < n - na then 1 else a.(i - (n - na)) in
    let db = if i < n - nb then 1 else b.(i - (n - nb)) in
    match (da, db) with
    | x, y when x = y -> out.(i) <- x
    | 1, y -> out.(i) <- y
    | x, 1 -> out.(i) <- x
    | _ -> invalid_arg "broadcast_shapes: incompatible"
  done;
  out

let broadcast_to v target_shape =
  let nd_old = Array.length v.shape in
  let nd_new = Array.length target_shape in
  if nd_new < nd_old then
    invalid_arg "broadcast_to: cannot broadcast to lower rank";
  let diff = nd_new - nd_old in
  let new_strides = Array.make nd_new 0 in
  for i = 0 to nd_new - 1 do
    let tgt = target_shape.(i) in
    if i < diff then (
      if
        (* padded dim on the left *)
        tgt <> 1
      then new_strides.(i) <- 0 (* broadcast *))
    else
      let src_dim = v.shape.(i - diff) in
      if src_dim = tgt then new_strides.(i) <- v.strides.(i - diff)
      else if src_dim = 1 then new_strides.(i) <- 0
      else invalid_arg "broadcast_to: incompatible sizes"
  done;
  let layout =
    if
      check_c_contiguity_from_shape_strides target_shape new_strides
      && v.offset = 0
    then C_contiguous
    else Strided
  in
  { v with shape = Array.copy target_shape; strides = new_strides; layout }

(* Rank‑preserving broadcast *)
let expand v new_shape =
  if Array.length new_shape <> Array.length v.shape then
    invalid_arg "expand: must keep rank";
  let new_strides =
    Array.mapi
      (fun i ns ->
        let old = v.shape.(i) in
        if old = ns then v.strides.(i)
        else if old = 1 then 0
        else invalid_arg "expand: cannot expand non‑singleton dim")
      new_shape
  in
  let layout =
    if
      check_c_contiguity_from_shape_strides new_shape new_strides
      && v.offset = 0
    then C_contiguous
    else Strided
  in
  { v with shape = Array.copy new_shape; strides = new_strides; layout }

(* Dim manipulation *)

let expand_dims v axis =
  let n = ndim v in
  let axis = if axis < 0 then axis + n + 1 else axis in
  if axis < 0 || axis > n then invalid_arg "expand_dims: axis out of range";
  let new_shape = Array.make (n + 1) 0 in
  let new_strides = Array.make (n + 1) 0 in
  for i = 0 to n do
    if i < axis then (
      new_shape.(i) <- v.shape.(i);
      new_strides.(i) <- v.strides.(i))
    else if i = axis then (
      new_shape.(i) <- 1;
      (* stride for inserted dim is product of later dims’ sizes times their
         stride magnitude *)
      if n = 0 then new_strides.(i) <- 1
      else if i = 0 then new_strides.(i) <- v.strides.(0) * max 1 v.shape.(0)
      else new_strides.(i) <- v.strides.(i - 1) * max 1 v.shape.(i - 1))
    else (
      new_shape.(i) <- v.shape.(i - 1);
      new_strides.(i) <- v.strides.(i - 1))
  done;
  let layout =
    if
      check_c_contiguity_from_shape_strides new_shape new_strides
      && v.offset = 0
    then C_contiguous
    else Strided
  in
  { v with shape = new_shape; strides = new_strides; layout }

let permute v axes =
  let n = ndim v in
  if Array.length axes <> n then invalid_arg "permute: bad rank";
  let seen = Array.make n false in
  Array.iter
    (fun ax ->
      let ax = if ax < 0 then ax + n else ax in
      if ax < 0 || ax >= n || seen.(ax) then
        invalid_arg "permute: axes not a permutation";
      seen.(ax) <- true)
    axes;
  let new_shape = Array.map (fun ax -> v.shape.(ax)) axes in
  let new_strides = Array.map (fun ax -> v.strides.(ax)) axes in
  let layout =
    if
      check_c_contiguity_from_shape_strides new_shape new_strides
      && v.offset = 0
    then C_contiguous
    else Strided
  in
  { v with shape = new_shape; strides = new_strides; layout }

let swapaxes v a b =
  let n = ndim v in
  let a = if a < 0 then a + n else a in
  let b = if b < 0 then b + n else b in
  let axes =
    Array.init n (fun i -> if i = a then b else if i = b then a else i)
  in
  permute v axes

let moveaxis v src dst =
  let n = ndim v in
  let src = if src < 0 then src + n else src in
  let dst = if dst < 0 then dst + n else dst in
  if src < 0 || src >= n || dst < 0 || dst > n then
    invalid_arg "moveaxis: bounds";
  let axes_l =
    Array.to_list (Array.init n Fun.id) |> List.filter (fun x -> x <> src)
  in
  let left, right = split_n axes_l dst in
  let perm = Array.of_list (left @ [ src ] @ right) in
  permute v perm

(* Slicing *)

let slice ?steps ~starts ~stops v =
  let n = ndim v in
  if Array.length starts <> n || Array.length stops <> n then
    invalid_arg "slice: rank mismatch";
  let steps =
    match steps with
    | None -> Array.make n 1
    | Some s when Array.length s = n ->
        Array.iter (fun x -> if x = 0 then invalid_arg "slice: step = 0") s;
        s
    | _ -> invalid_arg "slice: bad steps length"
  in
  let new_shape = Array.make n 0 in
  let new_strides = Array.init n (fun i -> v.strides.(i) * steps.(i)) in
  let new_offset = ref v.offset in
  for i = 0 to n - 1 do
    let dim = v.shape.(i) in
    let s = if starts.(i) < 0 then starts.(i) + dim else starts.(i) in
    let e = if stops.(i) < 0 then stops.(i) + dim else stops.(i) in
    let s = max 0 (min s dim) in
    let e = max 0 (min e dim) in
    let step = steps.(i) in
    let len =
      if step > 0 then max 0 ((e - s + step - 1) / step)
      else max 0 ((s - e - step - 1) / -step)
    in
    new_shape.(i) <- len;
    new_offset := !new_offset + (s * v.strides.(i))
  done;
  let layout =
    if
      check_c_contiguity_from_shape_strides new_shape new_strides
      && !new_offset = 0
    then C_contiguous
    else Strided
  in
  { shape = new_shape; strides = new_strides; offset = !new_offset; layout }

(* Flip *)

let flip ?axes v =
  let n = ndim v in
  let axes =
    match axes with
    | None -> Array.init n Fun.id
    | Some a -> Array.map (fun ax -> if ax < 0 then ax + n else ax) a
  in
  Array.iter
    (fun ax -> if ax < 0 || ax >= n then invalid_arg "flip: axis out of bounds")
    axes;
  let new_strides = Array.copy v.strides in
  let new_offset = ref v.offset in
  Array.iter
    (fun ax ->
      new_strides.(ax) <- -new_strides.(ax);
      new_offset := !new_offset + ((v.shape.(ax) - 1) * v.strides.(ax)))
    axes;
  { v with strides = new_strides; offset = !new_offset; layout = Strided }

(* Reshape *)

(* Merge contiguous dimensions to simplify non‑contiguous reshape. *)
let merge_dims shape strides =
  let n = Array.length shape in
  let rec loop i acc =
    if i = n then List.rev acc
    else if shape.(i) = 1 then
      loop (i + 1) acc (* skip singleton dims – they’re free to merge *)
    else
      match acc with
      | (sz, st) :: tl when st = shape.(i) * strides.(i) ->
          loop (i + 1) ((sz * shape.(i), strides.(i)) :: tl)
      | _ -> loop (i + 1) ((shape.(i), strides.(i)) :: acc)
  in
  Array.of_list (loop 0 [])
(* array of (merged_size, repr_stride) *)

let reshape v new_shape =
  (* Compute total elements – allowing zero‑dim tensors. *)
  let total_old = size v in
  let total_new =
    if Array.length new_shape = 0 then 1 else Array.fold_left ( * ) 1 new_shape
  in
  if total_old <> total_new then invalid_arg "reshape: mismatched element count";
  (* Trivial case – same shape. *)
  if
    Array.length new_shape = Array.length v.shape
    && Array.for_all2 ( = ) new_shape v.shape
  then v
  else if is_c_contiguous v then
    (* Contiguous: just recompute C strides. *)
    let strides = compute_c_strides new_shape in
    { v with shape = Array.copy new_shape; strides; layout = C_contiguous }
  else
    (* Non‑contiguous: attempt stride projection as in tinygrad. *)
    let merged = merge_dims v.shape v.strides in
    let rev_new_shape = List.rev (Array.to_list new_shape) in
    let new_strides_rev = ref [] in
    let idx_ns = ref rev_new_shape in
    Array.iteri
      (fun _ (merged_sz, base_stride) ->
        let acc = ref 1 in
        let cur_stride = ref base_stride in
        while !acc < merged_sz && !idx_ns <> [] do
          match !idx_ns with
          | dim :: tl ->
              new_strides_rev := !cur_stride :: !new_strides_rev;
              cur_stride := !cur_stride * dim;
              acc := !acc * dim;
              idx_ns := tl
          | [] -> ()
        done;
        if !acc <> merged_sz then
          invalid_arg "reshape: cannot map strides for non‑contiguous view")
      merged;
    (* If there are leftover dims (new_shape longer than merged result), those
       are leading singleton/broadcast dims. *)
    let leftover = List.length !idx_ns in
    new_strides_rev := List.init leftover (fun _ -> 0) @ !new_strides_rev;
    let new_strides = Array.of_list (List.rev !new_strides_rev) in
    let layout =
      if
        check_c_contiguity_from_shape_strides new_shape new_strides
        && v.offset = 0
      then C_contiguous
      else Strided
    in
    { v with shape = Array.copy new_shape; strides = new_strides; layout }

(* Split *)

let split ?(axis = 0) sections v =
  let n = dim v axis in
  if sections <= 0 then invalid_arg "split: sections <= 0";
  if n mod sections <> 0 then invalid_arg "split: not divisible";
  let part = n / sections in
  let rec build i acc =
    if i = sections then List.rev acc
    else
      let starts = Array.make (ndim v) 0 in
      let stops = Array.copy v.shape in
      let s = i * part in
      starts.(axis) <- s;
      stops.(axis) <- s + part;
      let piece = slice ~starts ~stops v in
      build (i + 1) (piece :: acc)
  in
  build 0 []

let array_split ?(axis = 0) sections v =
  let n = dim v axis in
  if sections <= 0 then invalid_arg "array_split: sections <= 0";
  let base = n / sections in
  let rem = n mod sections in
  let sizes =
    Array.init sections (fun i -> if i < rem then base + 1 else base)
  in
  let rec build i offset acc =
    if i = sections then List.rev acc
    else
      let starts = Array.make (ndim v) 0 in
      let stops = Array.copy v.shape in
      starts.(axis) <- offset;
      stops.(axis) <- offset + sizes.(i);
      let piece = slice ~starts ~stops v in
      build (i + 1) (offset + sizes.(i)) (piece :: acc)
  in
  build 0 0 []
