type ('a, 'b) buffer = ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t

type ('a, 'b) t = {
  dtype : ('a, 'b) Nx_core.Dtype.dtype;
  buffer : ('a, 'b) buffer;
  view : Nx_core.View.view;
}

type context = { pool : Parallel.pool }

let buffer { buffer; _ } = buffer
let view { view; _ } = view
let offset { view; _ } = view.Nx_core.View.offset
let shape { view; _ } = view.Nx_core.View.shape
let strides { view; _ } = view.Nx_core.View.strides
let size { view; _ } = Nx_core.View.size view
let is_c_contiguous { view; _ } = Nx_core.View.is_c_contiguous view
