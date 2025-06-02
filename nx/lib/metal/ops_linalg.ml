open Nx_core
open Internal

(* Helper to get concrete shape from view *)
let get_shape view =
  match Symbolic_shape.eval (Lazy_view.shape view) with
  | Some arr -> arr
  | None ->
      Error.failed ~op:"get_shape" ~what:"cannot evaluate symbolic shape" ()

(* Helper to check if view is C-contiguous *)
let is_c_contiguous view = Lazy_view.is_contiguous view

let op_cholesky ctx ~upper t =
  let view = t.view in
  let shape = get_shape view in
  let ndim = Array.length shape in

  if ndim < 2 then
    Error.invalid ~op:"cholesky" ~what:"tensor shape"
      ~reason:"must be at least 2D" ~hint:"" ();

  let n = shape.(ndim - 1) in
  let m = shape.(ndim - 2) in

  if n <> m then
    Error.invalid ~op:"cholesky" ~what:"tensor shape"
      ~reason:"last two dimensions must be square"
      ~hint:(Printf.sprintf "got shape [..., %d, %d]" m n)
      ();

  let t_contig =
    if is_c_contiguous view then t else Ops_movement.make_contiguous ctx t
  in

  let out_buffer = Buffer_pool.allocate ctx.pool t_contig.buffer.size_bytes in
  let out =
    {
      t_contig with
      buffer = { buffer = out_buffer; size_bytes = t_contig.buffer.size_bytes };
    }
  in

  Dispatch.run_cholesky ctx ~upper t_contig out;
  out

let op_qr ctx ~reduced t =
  let dtype = t.dtype in
  let view = t.view in
  let shape = get_shape view in
  let ndim = Array.length shape in

  if ndim < 2 then
    Error.invalid ~op:"qr" ~what:"tensor shape" ~reason:"must be at least 2D"
      ~hint:"" ();

  let n = shape.(ndim - 1) in
  let m = shape.(ndim - 2) in

  let t_contig =
    if is_c_contiguous view then t else Ops_movement.make_contiguous ctx t
  in

  let batch_shape = Array.sub shape 0 (ndim - 2) in
  let k = min m n in

  let q_shape =
    if reduced then Array.append batch_shape [| m; k |]
    else Array.append batch_shape [| m; m |]
  in

  let r_shape =
    if reduced then Array.append batch_shape [| k; n |]
    else Array.append batch_shape [| m; n |]
  in

  let q_size = Array.fold_left ( * ) 1 q_shape in
  let r_size = Array.fold_left ( * ) 1 r_shape in

  let q_buffer = Buffer_pool.allocate ctx.pool (q_size * sizeof_dtype dtype) in
  let r_buffer = Buffer_pool.allocate ctx.pool (r_size * sizeof_dtype dtype) in

  let q =
    {
      context = ctx;
      dtype;
      buffer = { buffer = q_buffer; size_bytes = q_size * sizeof_dtype dtype };
      view = Lazy_view.create (Symbolic_shape.of_ints q_shape);
    }
  in
  let r =
    {
      context = ctx;
      dtype;
      buffer = { buffer = r_buffer; size_bytes = r_size * sizeof_dtype dtype };
      view = Lazy_view.create (Symbolic_shape.of_ints r_shape);
    }
  in

  Dispatch.run_qr ctx ~reduced t_contig q r;
  (q, r)

let op_svd ctx ~full_matrices t =
  let dtype = t.dtype in
  let view = t.view in
  let shape = get_shape view in
  let ndim = Array.length shape in

  if ndim < 2 then
    Error.invalid ~op:"svd" ~what:"tensor shape" ~reason:"must be at least 2D"
      ~hint:"" ();

  let n = shape.(ndim - 1) in
  let m = shape.(ndim - 2) in

  let t_contig =
    if is_c_contiguous view then t else Ops_movement.make_contiguous ctx t
  in

  let batch_shape = Array.sub shape 0 (ndim - 2) in
  let k = min m n in

  let u_shape =
    if full_matrices then Array.append batch_shape [| m; m |]
    else Array.append batch_shape [| m; k |]
  in

  let s_shape = Array.append batch_shape [| k |] in

  let vh_shape =
    if full_matrices then Array.append batch_shape [| n; n |]
    else Array.append batch_shape [| k; n |]
  in

  let u_size = Array.fold_left ( * ) 1 u_shape in
  let s_size = Array.fold_left ( * ) 1 s_shape in
  let vh_size = Array.fold_left ( * ) 1 vh_shape in

  let u_buffer = Buffer_pool.allocate ctx.pool (u_size * sizeof_dtype dtype) in
  let vh_buffer =
    Buffer_pool.allocate ctx.pool (vh_size * sizeof_dtype dtype)
  in

  let u =
    {
      context = ctx;
      dtype;
      buffer = { buffer = u_buffer; size_bytes = u_size * sizeof_dtype dtype };
      view = Lazy_view.create (Symbolic_shape.of_ints u_shape);
    }
  in
  let vh =
    {
      context = ctx;
      dtype;
      buffer = { buffer = vh_buffer; size_bytes = vh_size * sizeof_dtype dtype };
      view = Lazy_view.create (Symbolic_shape.of_ints vh_shape);
    }
  in

  (* Singular values are always float64 according to the interface *)
  let s_dtype = Dtype.Float64 in
  let s_buffer =
    Buffer_pool.allocate ctx.pool (s_size * sizeof_dtype s_dtype)
  in
  let s =
    {
      context = ctx;
      dtype = s_dtype;
      buffer = { buffer = s_buffer; size_bytes = s_size * sizeof_dtype s_dtype };
      view = Lazy_view.create (Symbolic_shape.of_ints s_shape);
    }
  in

  Dispatch.run_svd ctx ~full_matrices t_contig u s vh;
  (u, s, vh)

let op_eig : type a b.
    context ->
    vectors:bool ->
    (a, b) t ->
    (Complex.t, Dtype.complex64_elt) t
    * (Complex.t, Dtype.complex64_elt) t option =
 fun ctx ~vectors t ->
  let _dtype = t.dtype in
  let view = t.view in
  let shape = get_shape view in
  let ndim = Array.length shape in

  if ndim < 2 then
    Error.invalid ~op:"eig" ~what:"tensor shape" ~reason:"must be at least 2D"
      ~hint:"" ();

  let n = shape.(ndim - 1) in
  let m = shape.(ndim - 2) in

  if n <> m then
    Error.invalid ~op:"eig" ~what:"tensor shape"
      ~reason:"last two dimensions must be square"
      ~hint:(Printf.sprintf "got shape [..., %d, %d]" m n)
      ();

  let t_contig =
    if is_c_contiguous view then t else Ops_movement.make_contiguous ctx t
  in

  let batch_shape = Array.sub shape 0 (ndim - 2) in
  let eigenvalues_shape = Array.append batch_shape [| n |] in

  let eigenvalues_size = Array.fold_left ( * ) 1 eigenvalues_shape in

  (* Eigenvalues are always complex64 for general matrices *)
  let eigenvalues_dtype = Dtype.Complex64 in
  let eigenvalues_buffer =
    Buffer_pool.allocate ctx.pool
      (eigenvalues_size * sizeof_dtype eigenvalues_dtype)
  in
  let eigenvalues =
    {
      context = ctx;
      dtype = eigenvalues_dtype;
      buffer =
        {
          buffer = eigenvalues_buffer;
          size_bytes = eigenvalues_size * sizeof_dtype eigenvalues_dtype;
        };
      view = Lazy_view.create (Symbolic_shape.of_ints eigenvalues_shape);
    }
  in

  (* Eigenvectors are also complex64 for general matrices *)
  let eigenvectors =
    if vectors then
      let eigenvectors_shape = shape in
      let eigenvectors_size = Array.fold_left ( * ) 1 eigenvectors_shape in
      let eigenvectors_dtype = Dtype.Complex64 in
      let eigenvectors_buffer =
        Buffer_pool.allocate ctx.pool
          (eigenvectors_size * sizeof_dtype eigenvectors_dtype)
      in
      Some
        {
          context = ctx;
          dtype = eigenvectors_dtype;
          buffer =
            {
              buffer = eigenvectors_buffer;
              size_bytes = eigenvectors_size * sizeof_dtype eigenvectors_dtype;
            };
          view = Lazy_view.create (Symbolic_shape.of_ints eigenvectors_shape);
        }
    else None
  in

  (* Cast input to match Metal's expectations if needed *)
  let t_for_dispatch =
    match eigenvectors with
    | Some evecs -> t_contig (* Use original type for dispatch *)
    | None -> t_contig
  in

  (* Create temporary buffers for dispatch if types don't match *)
  let dispatch_eigenvectors =
    match eigenvectors with
    | Some _ ->
        (* Create temporary buffer with input's dtype for dispatch *)
        let eigenvectors_shape = shape in
        let eigenvectors_size = Array.fold_left ( * ) 1 eigenvectors_shape in
        let eigenvectors_buffer =
          Buffer_pool.allocate ctx.pool
            (eigenvectors_size * sizeof_dtype t.dtype)
        in
        Some
          {
            context = ctx;
            dtype = t.dtype;
            buffer =
              {
                buffer = eigenvectors_buffer;
                size_bytes = eigenvectors_size * sizeof_dtype t.dtype;
              };
            view = Lazy_view.create (Symbolic_shape.of_ints eigenvectors_shape);
          }
    | None -> None
  in

  Dispatch.run_eig ctx ~symmetric:false ~vectors t_for_dispatch eigenvalues
    dispatch_eigenvectors;

  (* For now, return the pre-allocated complex64 eigenvectors *)
  let final_eigenvectors = eigenvectors in

  (eigenvalues, final_eigenvectors)

let op_eigh : type a b.
    context ->
    vectors:bool ->
    (a, b) t ->
    (float, Dtype.float64_elt) t * (a, b) t option =
 fun ctx ~vectors t ->
  let dtype = t.dtype in
  let view = t.view in
  let shape = get_shape view in
  let ndim = Array.length shape in

  if ndim < 2 then
    Error.invalid ~op:"eigh" ~what:"tensor shape" ~reason:"must be at least 2D"
      ~hint:"" ();

  let n = shape.(ndim - 1) in
  let m = shape.(ndim - 2) in

  if n <> m then
    Error.invalid ~op:"eigh" ~what:"tensor shape"
      ~reason:"last two dimensions must be square"
      ~hint:(Printf.sprintf "got shape [..., %d, %d]" m n)
      ();

  let t_contig =
    if is_c_contiguous view then t else Ops_movement.make_contiguous ctx t
  in

  let batch_shape = Array.sub shape 0 (ndim - 2) in
  let eigenvalues_shape = Array.append batch_shape [| n |] in

  let eigenvalues_size = Array.fold_left ( * ) 1 eigenvalues_shape in

  (* Eigenvalues are always float64 for symmetric/Hermitian matrices *)
  let eigenvalues_dtype = Dtype.Float64 in
  let eigenvalues_buffer =
    Buffer_pool.allocate ctx.pool
      (eigenvalues_size * sizeof_dtype eigenvalues_dtype)
  in
  let eigenvalues =
    {
      context = ctx;
      dtype = eigenvalues_dtype;
      buffer =
        {
          buffer = eigenvalues_buffer;
          size_bytes = eigenvalues_size * sizeof_dtype eigenvalues_dtype;
        };
      view = Lazy_view.create (Symbolic_shape.of_ints eigenvalues_shape);
    }
  in

  (* Eigenvectors have the same type as input for symmetric/Hermitian
     matrices *)
  let eigenvectors =
    if vectors then
      let eigenvectors_shape = shape in
      let eigenvectors_size = Array.fold_left ( * ) 1 eigenvectors_shape in
      let eigenvectors_buffer =
        Buffer_pool.allocate ctx.pool (eigenvectors_size * sizeof_dtype dtype)
      in
      Some
        {
          context = ctx;
          dtype;
          buffer =
            {
              buffer = eigenvectors_buffer;
              size_bytes = eigenvectors_size * sizeof_dtype dtype;
            };
          view = Lazy_view.create (Symbolic_shape.of_ints eigenvectors_shape);
        }
    else None
  in

  Dispatch.run_eig ctx ~symmetric:true ~vectors t_contig eigenvalues
    eigenvectors;
  (eigenvalues, eigenvectors)

let op_triangular_solve ctx ~upper ~transpose ~unit_diag a b =
  let a_dtype = a.dtype in
  let b_dtype = b.dtype in

  (if a_dtype <> b_dtype then
     let expected = Dtype.to_string a_dtype in
     let actual = Dtype.to_string b_dtype in
     Error.dtype_mismatch ~op:"triangular_solve" ~expected ~actual ());

  let a_shape = get_shape a.view in
  let b_shape = get_shape b.view in
  let a_ndim = Array.length a_shape in
  let b_ndim = Array.length b_shape in

  if a_ndim < 2 then
    Error.invalid ~op:"triangular_solve" ~what:"matrix shape"
      ~reason:"must be at least 2D" ~hint:"" ();

  if b_ndim < 1 then
    Error.invalid ~op:"triangular_solve" ~what:"rhs shape"
      ~reason:"must be at least 1D" ~hint:"" ();

  let n = a_shape.(a_ndim - 1) in
  let m = a_shape.(a_ndim - 2) in

  if n <> m then
    Error.invalid ~op:"triangular_solve" ~what:"matrix shape"
      ~reason:"last two dimensions must be square"
      ~hint:(Printf.sprintf "got shape [..., %d, %d]" m n)
      ();

  let a_contig =
    if is_c_contiguous a.view then a else Ops_movement.make_contiguous ctx a
  in
  let b_contig =
    if is_c_contiguous b.view then b else Ops_movement.make_contiguous ctx b
  in

  let out_buffer = Buffer_pool.allocate ctx.pool b_contig.buffer.size_bytes in
  let out =
    {
      b_contig with
      buffer = { buffer = out_buffer; size_bytes = b_contig.buffer.size_bytes };
    }
  in

  Dispatch.run_triangular_solve ctx ~upper ~transpose ~unit_diag a_contig
    b_contig out;
  out
