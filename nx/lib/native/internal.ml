open Nx_core
open Bigarray

type ('a, 'b) buffer = ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
type context = { pool : Parallel.pool }

type ('a, 'b) t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : View.t;
}

(* Basic Accessors for Internal.t *)
let dtype { dtype; _ } = dtype
let buffer { buffer; _ } = buffer
let view { view; _ } = view
let shape { view; _ } = View.shape view
let strides { view; _ } = View.strides view
let stride axis { view; _ } = View.stride axis view
let offset { view; _ } = View.offset view
let size { view; _ } = View.numel view (* Delegates to View.numel (view t) *)
let numel { view; _ } = View.numel view (* Explicit numel accessor *)
let dims { view; _ } = View.shape view (* Alias for shape *)
let dim axis { view; _ } = View.dim axis view
let ndim { view; _ } = View.ndim view
let is_c_contiguous { view; _ } = View.is_c_contiguous view

(* Low-level helper to create a Bigarray.Array1.t *)
let create_buffer_unsafe (type a b) (dt : (a, b) Dtype.t)
    (size_in_elements : int) : (a, b) buffer =
  Bigarray.Array1.create (Dtype.to_bigarray_kind dt) Bigarray.c_layout
    size_in_elements

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
  let vw = View.create shp in
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
  let src_view = view t_src in
  let src_shape = View.shape src_view in
  let total_elements = View.numel src_view in

  let new_buffer = create_buffer_unsafe (dtype t_src) total_elements in
  (* Create a new C-contiguous view for the destination *)
  let new_view = View.create src_shape in
  let new_t =
    {
      context = t_src.context;
      dtype = dtype t_src;
      buffer = new_buffer;
      view = new_view;
    }
  in

  if total_elements = 0 then new_t (* Handle zero-element tensor *)
  else if
    is_c_contiguous t_src
    && View.offset src_view = 0
    && Array1.dim (buffer t_src) = total_elements
  then (
    (* If source is fully C-contiguous and view covers entire buffer, use fast
       Bigarray.Array1.blit *)
    Array1.blit (buffer t_src) new_buffer;
    new_t)
  else
    (* General case: iterate based on logical indices and copy element by
       element This assumes the new_t is C-contiguous from its own view's
       perspective. *)
    let n_dims = View.ndim src_view in
    if n_dims = 0 then (
      (* Scalar case *)
      let v = Bigarray.Array1.get (buffer t_src) (View.offset src_view) in
      Bigarray.Array1.set new_buffer (View.offset new_view) v;
      new_t)
    else
      let current_md_idx = Array.make n_dims 0 in
      let rec copy_slice dim =
        if dim = n_dims then
          let src_physical_idx =
            View.offset src_view
            + Shape.ravel_index current_md_idx (View.strides src_view)
          in
          (* For new_t, its view is C-contiguous and offset 0 on its own
             buffer *)
          let dst_physical_idx =
            Shape.ravel_index current_md_idx (View.strides new_view)
          in
          new_buffer.{dst_physical_idx} <- (buffer t_src).{src_physical_idx}
        else
          for i = 0 to View.dim dim src_view - 1 do
            current_md_idx.(dim) <- i;
            copy_slice (dim + 1)
          done
      in
      copy_slice 0;
      new_t

let fill : type a b. a -> (a, b) t -> unit =
 fun value t_fill ->
  let fill_view = view t_fill in
  let fill_buffer = buffer t_fill in
  let total_elements = View.numel fill_view in

  if total_elements = 0 then () (* No elements to fill *)
  else if is_c_contiguous t_fill && View.offset fill_view = 0 then
    (* If the view is fully C-contiguous on its buffer, use fast fill *)
    Array1.fill fill_buffer value
  else
    (* Generic case: Iterate and fill element by element respecting the view *)
    let n_dims = View.ndim fill_view in
    if n_dims = 0 then
      (* Scalar case *)
      Bigarray.Array1.set fill_buffer (View.offset fill_view) value
    else
      let current_md_idx = Array.make n_dims 0 in
      let rec fill_slice dim =
        if dim = n_dims then
          let physical_idx =
            View.offset fill_view
            + Shape.ravel_index current_md_idx (View.strides fill_view)
          in
          fill_buffer.{physical_idx} <- value
        else
          for i = 0 to View.dim dim fill_view - 1 do
            current_md_idx.(dim) <- i;
            fill_slice (dim + 1)
          done
      in
      fill_slice 0

let blit : type a b. (a, b) t -> (a, b) t -> unit =
 fun src dst ->
  let src_view = view src in
  let dst_view = view dst in

  if View.ndim src_view <> View.ndim dst_view then
    invalid_arg "blit: tensors must have the same number of dimensions";
  if not (Shape.equal (View.shape src_view) (View.shape dst_view)) then
    invalid_arg "blit: tensors must have the same shape";

  let total_elements = View.numel src_view in
  if total_elements = 0 then () (* Nothing to blit *)
  else
    let src_buffer = buffer src in
    let dst_buffer = buffer dst in
    let n_dims = View.ndim src_view in

    (* TODO: Handle overlapping bigarrays correctly. Currently, when src and dst
       are views of the same underlying buffer with overlapping regions, the
       copy may produce incorrect results as source data can be overwritten
       before being read.

       Consider using https://github.com/dinosaure/overlap which provides a
       library for checking if bigarrays overlap. If overlap is detected, we
       should either: 1. Make a copy of the source data first 2. Copy in the
       appropriate order (backward if dst > src) 3. Use memmove-like semantics

       See test_blit_overlapping_views for expected behavior. *)
    if n_dims = 0 then
      (* Scalar case *)
      dst_buffer.{View.offset dst_view} <- src_buffer.{View.offset src_view}
    else
      (* Iterate through logical elements based on common shape *)
      let current_md_idx = Array.make n_dims 0 in
      let rec blit_slice dim =
        if dim = n_dims then (
          let src_physical_offset =
            View.offset src_view
            + Shape.ravel_index current_md_idx (View.strides src_view)
          in
          let dst_physical_offset =
            View.offset dst_view
            + Shape.ravel_index current_md_idx (View.strides dst_view)
          in
          (* Debug output *)
          if false then
            Printf.printf "Copying from src[%d] to dst[%d]\n"
              src_physical_offset dst_physical_offset;
          dst_buffer.{dst_physical_offset} <- src_buffer.{src_physical_offset})
        else
          for i = 0 to View.dim dim src_view - 1 do
            (* Use src_view's shape, same as dst_view's *)
            current_md_idx.(dim) <- i;
            blit_slice (dim + 1)
          done
      in
      blit_slice 0
