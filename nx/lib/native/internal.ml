open Nx_core
open Bigarray

type ('a, 'b) buffer = ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
type context = { pool : Parallel.pool }

type ('a, 'b) t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : Lazy_view.t;
}

(* Helper to map logical indices through a chain of view transformations *)
(* This is needed when views can't be composed into a single view *)
let iterate_view_indices shape indices f =
  (* Helper to iterate through all indices of a tensor *)
  let ndim = Array.length shape in
  if ndim = 0 then f indices
  else
    let rec iter_dim d =
      if d = ndim then f (Array.copy indices)
      else
        for i = 0 to shape.(d) - 1 do
          indices.(d) <- i;
          iter_dim (d + 1)
        done
    in
    iter_dim 0

(* Basic Accessors for Internal.t *)
let dtype { dtype; _ } = dtype
let buffer { buffer; _ } = buffer
let view { view; _ } = view

let shape { view; _ } =
  match Symbolic_shape.eval (Lazy_view.shape view) with
  | Some arr -> arr
  | None ->
      Error.failed ~op:"shape"
        ~what:"cannot get shape with unbound symbolic dimensions" ()

let strides { view; _ } =
  match Lazy_view.strides view with
  | Some s ->
      (* Note: strides are in elements, not bytes *)
      s
  | None ->
      Error.failed ~op:"strides"
        ~what:"cannot get strides for non-composable views" ()

let stride axis { view; _ } =
  match Lazy_view.strides view with
  | Some s -> s.(axis)
  | None ->
      Error.failed ~op:"stride"
        ~what:"cannot get stride for non-composable views" ()

let offset { view; _ } =
  match Lazy_view.offset view with
  | Symbolic_shape.Static n -> n
  | Symbolic_shape.Dynamic _ ->
      Error.failed ~op:"offset"
        ~what:"cannot get offset with symbolic dimensions" ()

let size { view; _ } =
  match Lazy_view.numel view with
  | Symbolic_shape.Static n -> n
  | Symbolic_shape.Dynamic _ ->
      Error.failed ~op:"size" ~what:"cannot get size with symbolic dimensions"
        ()

let numel { view; _ } =
  match Lazy_view.numel view with
  | Symbolic_shape.Static n -> n
  | Symbolic_shape.Dynamic _ ->
      Error.failed ~op:"numel" ~what:"cannot get numel with symbolic dimensions"
        ()

let dims t = shape t
let dim axis t = (shape t).(axis)
let ndim { view; _ } = Lazy_view.ndim view
let is_c_contiguous { view; _ } = Lazy_view.is_contiguous view

(* Low-level helper to create a Bigarray.Array1.t *)
let create_buffer_unsafe (type a b) (dt : (a, b) Dtype.t)
    (size_in_elements : int) : (a, b) buffer =
  Bigarray.Array1.create
    (Dtype.to_bigarray_kind dt)
    Bigarray.c_layout size_in_elements

(* Operations (These seem like they might belong in a higher-level API or were
   part of an older structure, but correcting them as requested.) Note: These
   operations do not use a 'context' and thus won't be parallelized by
   default. *)

let empty : type a b. context -> (a, b) Dtype.t -> int array -> (a, b) t =
 fun ctx dt shp ->
  let num_elements = Array.fold_left ( * ) 1 shp in
  let buf = create_buffer_unsafe dt num_elements in
  (* Backend ops like op_buffer typically create a flat view. If these are
     higher-level ops, they should create a view matching the shape. *)
  let vw = Lazy_view.create (Symbolic_shape.of_ints shp) in
  { context = ctx; dtype = dt; buffer = buf; view = vw }

let full : type a b. context -> (a, b) Dtype.t -> int array -> a -> (a, b) t =
 fun ctx dt shp value ->
  let t = empty ctx dt shp in
  (* Fill the entire buffer; assumes the view of 'empty' covers the whole buffer
     contiguously if size > 0 *)
  if Array.fold_left ( * ) 1 shp > 0 then Array1.fill t.buffer value;
  t

let empty_like t = empty t.context (dtype t) (shape t)

let copy : type a b. (a, b) t -> (a, b) t =
 fun t_src ->
  let src_shape = shape t_src in
  let total_elements = numel t_src in

  (* If source has non-composable views, materialize it first *)
  let t_src =
    if Lazy_view.can_get_strides t_src.view then t_src
    else
      (* This is a bit circular, but op_contiguous should handle this case *)
      (* by creating a new buffer and copying data element by element *)
      (* For now, we'll handle the simple case *)
      t_src
  in

  (* Create new C-contiguous buffer *)
  let new_buffer = create_buffer_unsafe (dtype t_src) total_elements in
  let new_view = Lazy_view.create (Symbolic_shape.of_ints src_shape) in
  let new_t =
    {
      context = t_src.context;
      dtype = dtype t_src;
      buffer = new_buffer;
      view = new_view;
    }
  in

  (* Copy data respecting the view *)
  (if total_elements > 0 then
     match Lazy_view.strides t_src.view with
     | Some strides ->
         (* We have strides - do element-by-element copy *)
         let src_offset = offset t_src in
         let indices = Array.make (ndim t_src) 0 in
         let dst_idx = ref 0 in

         iterate_view_indices src_shape indices (fun idx ->
             let src_idx = src_offset + Shape.ravel_index idx strides in
             new_buffer.{!dst_idx} <- (buffer t_src).{src_idx};
             incr dst_idx)
     | None ->
         (* This shouldn't happen if ensure_materializable works correctly *)
         failwith "Cannot copy tensor with non-materializable view");
  new_t

let fill : type a b. a -> (a, b) t -> unit =
 fun value t_fill ->
  let fill_buffer = buffer t_fill in
  let total_elements = numel t_fill in

  if total_elements = 0 then () (* No elements to fill *)
  else if is_c_contiguous t_fill && offset t_fill = 0 then
    (* If the view is fully C-contiguous on its buffer, use fast fill *)
    Array1.fill fill_buffer value
  else
    (* Generic case: Iterate and fill element by element respecting the view *)
    let n_dims = ndim t_fill in
    if n_dims = 0 then
      (* Scalar case *)
      Bigarray.Array1.set fill_buffer (offset t_fill) value
    else
      (* Try to get strides; if not possible, we can't fill element by
         element *)
      match Lazy_view.strides t_fill.view with
      | Some strides_arr ->
          let current_md_idx = Array.make n_dims 0 in
          let rec fill_slice dim =
            if dim = n_dims then
              let physical_idx =
                offset t_fill + Shape.ravel_index current_md_idx strides_arr
              in
              fill_buffer.{physical_idx} <- value
            else
              for i = 0 to (shape t_fill).(dim) - 1 do
                current_md_idx.(dim) <- i;
                fill_slice (dim + 1)
              done
          in
          fill_slice 0
      | None ->
          (* Can't get strides - views are not composable *)
          (* We should NOT fill the entire buffer! *)
          Error.failed ~op:"fill"
            ~what:"cannot fill tensor with non-composable views"
            ~hint:"call contiguous() first to create a fillable copy" ()

let blit : type a b. (a, b) t -> (a, b) t -> unit =
 fun src dst ->
  if ndim src <> ndim dst then
    invalid_arg "blit: tensors must have the same number of dimensions";
  if not (Shape.equal (shape src) (shape dst)) then
    invalid_arg "blit: tensors must have the same shape";

  let total_elements = numel src in
  if total_elements = 0 then () (* Nothing to blit *)
  else
    let src_buffer = buffer src in
    let dst_buffer = buffer dst in
    let n_dims = ndim src in

    if n_dims = 0 then
      (* Scalar case *)
      dst_buffer.{offset dst} <- src_buffer.{offset src}
    else
      (* Get strides using the fixed Lazy_view.strides that returns last view's
         strides *)
      match (Lazy_view.strides src.view, Lazy_view.strides dst.view) with
      | Some src_strides, Some dst_strides ->
          (* Both have valid strides - use element-by-element copy *)
          let current_md_idx = Array.make n_dims 0 in
          let rec blit_slice dim =
            if dim = n_dims then
              let src_physical_offset =
                offset src + Shape.ravel_index current_md_idx src_strides
              in
              let dst_physical_offset =
                offset dst + Shape.ravel_index current_md_idx dst_strides
              in
              dst_buffer.{dst_physical_offset} <-
                src_buffer.{src_physical_offset}
            else
              for i = 0 to (shape src).(dim) - 1 do
                current_md_idx.(dim) <- i;
                blit_slice (dim + 1)
              done
          in
          blit_slice 0
      | _ ->
          (* Can't get strides for one or both tensors *)
          Error.failed ~op:"blit"
            ~what:"source or destination tensor has non-composable views"
            ~hint:"materialize with contiguous() before blit" ()
