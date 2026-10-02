(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Lightweight view of tensor layout and helpers for reshaping. *)

let err op fmt = Printf.ksprintf (fun msg -> invalid_arg (op ^ ": " ^ msg)) fmt

type layout = C_contiguous | Strided

(* The order of [shape], [strides] and [offset] is nx.cpu's C ABI (nx_c.h). *)
type t = {
  shape : int array;
  strides : int array;
  offset : int;
  layout : layout;
}

(* ───── Helpers ───── *)

let prod arr = Array.fold_left ( * ) 1 arr

(* Check if strides represent a contiguous layout *)
(* Row-major order whatever the strides of axes of size 1, which are never
   stepped; an empty view holds nothing out of order. *)
let has_zero shape =
  let n = Array.length shape and i = ref 0 in
  while !i < n && Array.unsafe_get shape !i <> 0 do
    incr i
  done;
  !i < n

let is_c_contiguous_strides shape_arr strides =
  has_zero shape_arr
  ||
  let expected = ref 1 and ordered = ref true in
  for i = Array.length shape_arr - 1 downto 0 do
    if shape_arr.(i) <> 1 then begin
      if strides.(i) <> !expected then ordered := false;
      expected := !expected * shape_arr.(i)
    end
  done;
  !ordered

(* ───── Accessors ───── *)

let shape v = v.shape
let strides v = v.strides

let stride axis v =
  let ndim = Array.length v.shape in
  if axis < 0 || axis >= ndim then
    err "stride" "axis %d out of bounds for %dD tensor" axis ndim;
  Array.unsafe_get v.strides axis

let offset v = v.offset
let is_c_contiguous v = v.layout = C_contiguous

let dim axis v =
  let ndim = Array.length v.shape in
  if axis < 0 || axis >= ndim then
    err "dim" "axis %d out of bounds for %dD tensor" axis ndim;
  v.shape.(axis)

let ndim v = Array.length v.shape
let numel v = prod v.shape

let extent v =
  let lo = ref v.offset and hi = ref v.offset in
  Array.iteri
    (fun a n ->
      let s = v.strides.(a) * (n - 1) in
      if s < 0 then lo := !lo + s else hi := !hi + s)
    v.shape;
  (!lo, !hi + 1)

(* Each product and sum is bounded before it is formed, so none wraps. A stride
   [s] over [d > 1] elements stays within [n - 1] positions only if [s] is at
   most [m = (n - 1) / (d - 1)] in magnitude, and the positions reached so far
   stay in [0, n - 1]. *)
let within v n =
  let shape = v.shape and strides = v.strides in
  let rank = Array.length shape in
  let rec go a count lo hi =
    if a = rank then true
    else
      let d = Array.unsafe_get shape a in
      if count > max_int / d then false
      else if d = 1 then go (a + 1) count lo hi
      else
        let m = (n - 1) / (d - 1) and s = Array.unsafe_get strides a in
        if s < -m || s > m then false
        else
          let t = s * (d - 1) in
          if t < 0 then t >= -lo && go (a + 1) (count * d) (lo + t) hi
          else t <= n - 1 - hi && go (a + 1) (count * d) lo (hi + t)
  in
  if Array.exists (fun d -> d < 0) shape then false
  else if has_zero shape then true
  else v.offset >= 0 && v.offset < n && go 0 1 v.offset v.offset

(* ───── View Creation ───── *)

let create ?(offset = 0) ?strides shape =
  let is_zero_size = has_zero shape in
  let shape =
    if is_zero_size then Array.map (fun s -> max s 0) shape else shape
  in
  let offset = if is_zero_size then 0 else offset in
  match strides with
  | None ->
      {
        shape;
        strides = Shape.c_contiguous_strides shape;
        offset;
        layout = C_contiguous;
      }
  | Some strides ->
      if Array.length strides <> Array.length shape then
        err "create" "strides length %d != shape length %d"
          (Array.length strides) (Array.length shape);
      let layout =
        if is_c_contiguous_strides shape strides then C_contiguous
        else Strided
      in
      { shape; strides; offset; layout }

(* ───── View Manipulation ───── *)

let expand view new_shape =
  let old_ndim = Array.length view.shape in
  let new_ndim = Array.length new_shape in
  (* Allow expanding a scalar to any shape *)
  if old_ndim = 0 then
    create ~offset:view.offset ~strides:(Array.make new_ndim 0) new_shape
  else if new_ndim <> old_ndim then
    err "expand" "rank mismatch: %d vs %d" new_ndim old_ndim
  else
    let old_arr = view.shape in
    let new_arr = new_shape in
    if has_zero old_arr then create new_shape
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
      create ~offset:view.offset ~strides new_shape

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
  create ~offset:view.offset ~strides:new_strides new_shape

(* The strides that view [view]'s elements as [new_shape], if any. The axes of
   both shapes group into runs of equal size; the old axes of a run must merge
   into one stride, which its new axes split. Axes of size 1 take no part, and
   get stride 0. *)
let viewing_strides view new_shape =
  let old_dims =
    List.filter
      (fun (d, _) -> d <> 1)
      (List.combine (Array.to_list view.shape) (Array.to_list view.strides))
    |> Array.of_list
  in
  let new_dims =
    Array.of_list (List.filter (( <> ) 1) (Array.to_list new_shape))
  in
  let strides = Array.make (Array.length new_dims) 0 in
  let rec runs oi ni =
    if oi = Array.length old_dims then true
    else
      let rec grow oj nj op np =
        if op = np then (oj, nj)
        else if op < np then grow (oj + 1) nj (op * fst old_dims.(oj)) np
        else grow oj (nj + 1) op (np * new_dims.(nj))
      in
      let oj, nj = grow (oi + 1) (ni + 1) (fst old_dims.(oi)) new_dims.(ni) in
      let merges = ref true in
      for k = oi to oj - 2 do
        let d, s = old_dims.(k + 1) in
        if snd old_dims.(k) <> d * s then merges := false
      done;
      !merges
      && begin
        strides.(nj - 1) <- snd old_dims.(oj - 1);
        for k = nj - 1 downto ni + 1 do
          strides.(k - 1) <- strides.(k) * new_dims.(k)
        done;
        runs oj nj
      end
  in
  if not (runs 0 0) then None
  else
    let k = ref 0 in
    Some
      (Array.map
         (fun d ->
           if d = 1 then 0
           else
             let s = strides.(!k) in
             incr k;
             s)
         new_shape)

let can_reshape view new_shape =
  prod view.shape = prod new_shape
  && (view.shape = new_shape
     || has_zero new_shape
     || view.layout = C_contiguous
     || Option.is_some (viewing_strides view new_shape))

let reshape view new_shape =
  if view.shape = new_shape then view
  else if prod view.shape <> prod new_shape then
    err "reshape" "cannot reshape %s to %s"
      (Shape.to_string view.shape)
      (Shape.to_string new_shape)
  else if has_zero new_shape then create ~offset:0 new_shape
  else if view.layout = C_contiguous then create ~offset:view.offset new_shape
  else
    match viewing_strides view new_shape with
    | Some strides -> create ~offset:view.offset ~strides new_shape
    | None ->
        err "reshape"
          "cannot reshape %s to %s, strides %s cannot view it, call \
           contiguous() first"
          (Shape.to_string view.shape)
          (Shape.to_string new_shape)
          (Shape.to_string view.strides)

let shrink view arg =
  let ndim = Array.length view.shape in
  if Array.length arg <> ndim then
    err "shrink" "bounds length %d != ndim %d" (Array.length arg) ndim;
  let shape_arr = view.shape in
  if Array.for_all2 (fun (b, e) s -> b = 0 && e = s) arg shape_arr then view
  else if
    Array.exists2
      (fun (b, e) s -> b < 0 || e < 0 || b > s || e > s || b > e)
      arg shape_arr
  then invalid_arg "shrink: bounds must be within shape and start <= end"
  else
    let new_shape = Array.map (fun (a, b) -> b - a) arg in
    let new_offset = ref view.offset in
    Array.iteri
      (fun i (a, _) -> new_offset := !new_offset + (a * view.strides.(i)))
      arg;
    create ~offset:!new_offset ~strides:view.strides new_shape

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
  Array.iteri
    (fun i do_flip ->
      if do_flip then
        let s_i = shape_arr.(i) in
        if s_i > 0 then (
          new_offset := !new_offset + ((s_i - 1) * strides.(i));
          new_strides.(i) <- -new_strides.(i)))
    flip_axes_bools;
  create ~offset:!new_offset ~strides:new_strides view.shape

let sliding_window view ~axis ~window ~step =
  let ndim = Array.length view.shape in
  if axis < 0 || axis >= ndim then
    err "sliding_window" "axis %d out of bounds for %dD tensor" axis ndim;
  if window < 1 then err "sliding_window" "window %d < 1" window;
  if step < 1 then err "sliding_window" "step %d < 1" step;
  let size = view.shape.(axis) in
  if window > size then
    err "sliding_window" "window %d > size %d of axis %d" window size axis;
  let new_shape = Array.make (ndim + 1) window in
  let new_strides = Array.make (ndim + 1) view.strides.(axis) in
  Array.blit view.shape 0 new_shape 0 ndim;
  Array.blit view.strides 0 new_strides 0 ndim;
  new_shape.(axis) <- ((size - window) / step) + 1;
  new_strides.(axis) <- view.strides.(axis) * step;
  create ~offset:view.offset ~strides:new_strides new_shape
