open Bigarray_ext
module Dtype = Nx_core.Dtype
module Shape = Nx_core.Shape
module View = Nx_core.View
module Lazy_view = Nx_core.Lazy_view
module Symbolic_shape = Nx_core.Symbolic_shape
module Error = Nx_core.Error
open Internal

(* Constants *)
let eps = 1e-10

(* Dtype-specific machine epsilon *)
let eps_of_dtype (type a b) (dtype : (a, b) Dtype.t) : float =
  match dtype with
  | Dtype.Float32 -> 1.1920929e-7
  | Dtype.Float64 -> 2.220446049250313e-16
  | _ -> failwith "eps_of_dtype: only float dtypes are supported"

(* Helper to get concrete shape from view *)
let get_shape view =
  match Symbolic_shape.eval (Lazy_view.shape view) with
  | Some arr -> arr
  | None ->
      Error.failed ~op:"get_shape" ~what:"cannot evaluate symbolic shape" ()

(* Helper functions for matrix operations *)

(* Convert value to float based on dtype *)
let[@inline] to_float (type a b) (dtype : (a, b) Dtype.t) (value : a) : float =
  match dtype with
  | Dtype.Float32 -> value
  | Dtype.Float64 -> value
  | _ -> failwith "to_float: only float dtypes are supported"

(* Generic N-dimensional indexing helper *)
let[@inline] get_nd t indices =
  let strides =
    match Lazy_view.strides t.view with
    | Some s -> s
    | None ->
        Error.failed ~op:"get_nd"
          ~what:"cannot get strides for non-contiguous view" ()
  in
  let offset =
    match Symbolic_shape.eval_dim (Lazy_view.offset t.view) with
    | Some n -> n
    | None ->
        Error.failed ~op:"get_nd" ~what:"cannot evaluate symbolic offset" ()
  in
  let acc = ref offset in
  for d = 0 to Array.length indices - 1 do
    acc := !acc + (indices.(d) * strides.(d))
  done;
  Array1.unsafe_get t.buffer !acc

(* Generic N-dimensional indexing setter *)
let[@inline] set_nd t indices value =
  let strides =
    match Lazy_view.strides t.view with
    | Some s -> s
    | None ->
        Error.failed ~op:"set_nd"
          ~what:"cannot get strides for non-contiguous view" ()
  in
  let offset =
    match Symbolic_shape.eval_dim (Lazy_view.offset t.view) with
    | Some n -> n
    | None ->
        Error.failed ~op:"set_nd" ~what:"cannot evaluate symbolic offset" ()
  in
  let acc = ref offset in
  for d = 0 to Array.length indices - 1 do
    acc := !acc + (indices.(d) * strides.(d))
  done;
  Array1.unsafe_set t.buffer !acc value

(* Helper to decompose a flattened batch index into multi-dimensional batch
   coordinates *)
let decompose_batch_index shape batch_idx =
  let ndim = Array.length shape in
  let batch_ndim = ndim - 2 in
  let coords = Array.make ndim 0 in
  let rem = ref batch_idx in
  for d = batch_ndim - 1 downto 0 do
    let size = shape.(d) in
    coords.(d) <- !rem mod size;
    rem := !rem / size
  done;
  coords

(* Batched matrix element getters/setters using proper N-dimensional indexing *)
let[@inline] get_batched_2d t shape batch_idx i j =
  let coords = decompose_batch_index shape batch_idx in
  let ndim = Array.length shape in
  coords.(ndim - 2) <- i;
  coords.(ndim - 1) <- j;
  get_nd t coords

let[@inline] set_batched_2d t shape batch_idx i j value =
  let coords = decompose_batch_index shape batch_idx in
  let ndim = Array.length shape in
  coords.(ndim - 2) <- i;
  coords.(ndim - 1) <- j;
  set_nd t coords value

(* Legacy 2D accessor for backward compatibility *)
let[@inline] get_2d t i j = get_nd t [| i; j |]
let[@inline] set_2d t i j value = set_nd t [| i; j |] value

(* Get matrix dimensions, handling batched case *)
let get_matrix_dims t =
  let shape =
    match Symbolic_shape.eval (Lazy_view.shape t.view) with
    | Some arr -> arr
    | None ->
        Error.failed ~op:"ops_linalg" ~what:"cannot evaluate symbolic shape" ()
  in
  let ndim = Array.length shape in
  if ndim < 2 then failwith "ops_linalg: input must be at least 2D"
  else if ndim = 2 then (1, shape.(0), shape.(1)) (* batch_size, m, n *)
  else
    let batch_size = Array.fold_left ( * ) 1 (Array.sub shape 0 (ndim - 2)) in
    (batch_size, shape.(ndim - 2), shape.(ndim - 1))

(* Create output tensor for matrix operations *)
let create_output ctx dtype shape = Internal.empty ctx dtype shape

(* Helper function for matrix multiplication of 2D arrays *)
let matmul_2d (type a b) (dtype : (a, b) Dtype.t) m n k a_get b_get =
  let result = Array.make_matrix m n (Dtype.zero dtype) in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let sum = ref (Dtype.zero dtype) in
      for l = 0 to k - 1 do
        sum := Dtype.add dtype !sum (Dtype.mul dtype (a_get i l) (b_get l j))
      done;
      result.(i).(j) <- !sum
    done
  done;
  result

(* Helper to compute dot product of two vectors *)
let[@inline] dot_product (type a b) (dtype : (a, b) Dtype.t) n v1_get v2_get =
  let sum = ref (Dtype.zero dtype) in
  for i = 0 to n - 1 do
    sum := Dtype.add dtype !sum (Dtype.mul dtype (v1_get i) (v2_get i))
  done;
  !sum

(* Helper to compute vector norm *)
let[@inline] vector_norm (type a b) (dtype : (a, b) Dtype.t) n v_get =
  let sum = ref (Dtype.zero dtype) in
  for i = 0 to n - 1 do
    let v = v_get i in
    sum := Dtype.add dtype !sum (Dtype.mul dtype v v)
  done;
  Dtype.of_float dtype (sqrt (to_float dtype !sum))

(* Helper to normalize a vector in place *)
let normalize_vector (type a b) (dtype : (a, b) Dtype.t) n v_array =
  let norm = vector_norm dtype n (fun i -> v_array.(i)) in
  if to_float dtype norm > eps then
    for i = 0 to n - 1 do
      v_array.(i) <- Dtype.div dtype v_array.(i) norm
    done

(* Apply Givens rotation to eliminate element at (i,j) *)
let givens_rotation (type a b) (dtype : (a, b) Dtype.t) a_ii a_ji =
  let a_ii_f = to_float dtype a_ii in
  let a_ji_f = to_float dtype a_ji in
  if abs_float a_ji_f < eps then (Dtype.one dtype, Dtype.zero dtype)
  else
    let r = sqrt ((a_ii_f *. a_ii_f) +. (a_ji_f *. a_ji_f)) in
    let c = a_ii_f /. r in
    let s = a_ji_f /. r in
    (Dtype.of_float dtype c, Dtype.of_float dtype s)

(* Apply Givens rotation to two rows *)
let apply_givens_rows (type a b) (dtype : (a, b) Dtype.t) n row1 row2 c s =
  for k = 0 to n - 1 do
    let temp1 =
      Dtype.add dtype (Dtype.mul dtype c row1.(k)) (Dtype.mul dtype s row2.(k))
    in
    let temp2 =
      Dtype.sub dtype (Dtype.mul dtype c row2.(k)) (Dtype.mul dtype s row1.(k))
    in
    row1.(k) <- temp1;
    row2.(k) <- temp2
  done

(* ===== Cholesky Decomposition ===== *)

(* Kernel for single matrix Cholesky decomposition *)
let kernel_cholesky_single (type a b) ~upper (dtype : (a, b) Dtype.t) input
    output input_shape output_shape n batch_size batch =
  (* Compute tolerance based on dtype and matrix size *)
  let tol = eps_of_dtype dtype *. float_of_int n in

  if upper then
    (* Upper triangular Cholesky: A = U^T * U *)
    for i = 0 to n - 1 do
      for j = i to n - 1 do
        let sum =
          ref
            (if batch_size = 1 then get_2d input i j
             else get_batched_2d input input_shape batch i j)
        in
        for k = 0 to i - 1 do
          let u_ki =
            if batch_size = 1 then get_2d output k i
            else get_batched_2d output output_shape batch k i
          in
          let u_kj =
            if batch_size = 1 then get_2d output k j
            else get_batched_2d output output_shape batch k j
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype u_ki u_kj)
        done;

        let value =
          if i = j then (
            (* Diagonal element *)
            let s_float = to_float dtype !sum in
            if s_float < -.tol then
              failwith "op_cholesky: matrix is not positive definite";
            (* Clamp small negative values to zero for numerical stability *)
            Dtype.of_float dtype (sqrt (max 0.0 s_float)))
          else
            (* Off-diagonal element *)
            let u_ii =
              if batch_size = 1 then get_2d output i i
              else get_batched_2d output output_shape batch i i
            in
            Dtype.div dtype !sum u_ii
        in
        if batch_size = 1 then set_2d output i j value
        else set_batched_2d output output_shape batch i j value
      done
    done
  else
    (* Lower triangular Cholesky: A = L * L^T *)
    for i = 0 to n - 1 do
      for j = 0 to i do
        let sum =
          ref
            (if batch_size = 1 then get_2d input i j
             else get_batched_2d input input_shape batch i j)
        in
        for k = 0 to j - 1 do
          let l_ik =
            if batch_size = 1 then get_2d output i k
            else get_batched_2d output output_shape batch i k
          in
          let l_jk =
            if batch_size = 1 then get_2d output j k
            else get_batched_2d output output_shape batch j k
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype l_ik l_jk)
        done;

        let value =
          if i = j then (
            (* Diagonal element *)
            let s_float = to_float dtype !sum in
            if s_float < -.tol then
              failwith "op_cholesky: matrix is not positive definite";
            (* Clamp small negative values to zero for numerical stability *)
            Dtype.of_float dtype (sqrt (max 0.0 s_float)))
          else
            (* Off-diagonal element *)
            let l_jj =
              if batch_size = 1 then get_2d output j j
              else get_batched_2d output output_shape batch j j
            in
            Dtype.div dtype !sum l_jj
        in
        if batch_size = 1 then set_2d output i j value
        else set_batched_2d output output_shape batch i j value
      done
    done

(* Cholesky decomposition using Cholesky-Banachiewicz algorithm *)
let cholesky (type a b) ~upper ctx (input : (a, b) t) =
  let pool = ctx.pool in
  let batch_size, n, m = get_matrix_dims input in
  if n <> m then failwith "op_cholesky: input must be square matrix";

  let output_shape = get_shape input.view in
  let output = create_output ctx input.dtype output_shape in

  (* Initialize output to zero *)
  let total_elements = Array.fold_left ( * ) 1 output_shape in
  for i = 0 to total_elements - 1 do
    Array1.unsafe_set output.buffer i (Dtype.zero input.dtype)
  done;

  (* Get input shape for batch indexing *)
  let input_shape = get_shape input.view in

  (* Process batches in parallel if batch_size > 1 *)
  if batch_size > 1 then
    Parallel.parallel_for pool 0 batch_size (fun batch_start batch_end ->
        for batch = batch_start to batch_end - 1 do
          kernel_cholesky_single ~upper input.dtype input output input_shape
            output_shape n batch_size batch
        done)
  else
    kernel_cholesky_single ~upper input.dtype input output input_shape
      output_shape n batch_size 0;

  output

(* ===== QR Decomposition ===== *)

(* Kernel for applying Householder reflection to matrix R *)
let apply_householder_to_r (type a b) (dtype : (a, b) Dtype.t) r_work v tau j n
    =
  for l = j to n - 1 do
    (* Compute v^T * R[:,l] *)
    let dot = ref (Dtype.zero dtype) in
    for i = j to Array.length v - 1 do
      dot := Dtype.add dtype !dot (Dtype.mul dtype v.(i) r_work.(i).(l))
    done;

    (* Update R[:,l] = R[:,l] - tau * dot * v *)
    let scale = Dtype.mul dtype tau !dot in
    for i = j to Array.length v - 1 do
      r_work.(i).(l) <-
        Dtype.sub dtype r_work.(i).(l) (Dtype.mul dtype scale v.(i))
    done
  done

(* Kernel for applying Householder reflection to matrix Q *)
let apply_householder_to_q (type a b) (dtype : (a, b) Dtype.t) q_work v tau j m
    =
  for i = 0 to m - 1 do
    (* Compute Q[i,:] * v *)
    let dot = ref (Dtype.zero dtype) in
    for l = j to m - 1 do
      dot := Dtype.add dtype !dot (Dtype.mul dtype q_work.(i).(l) v.(l))
    done;

    (* Update Q[i,:] = Q[i,:] - tau * dot * v^T *)
    let scale = Dtype.mul dtype tau !dot in
    for l = j to m - 1 do
      q_work.(i).(l) <-
        Dtype.sub dtype q_work.(i).(l) (Dtype.mul dtype scale v.(l))
    done
  done

(* Kernel for single matrix QR decomposition *)
let kernel_qr_single (type a b) ~reduced (dtype : (a, b) Dtype.t) input q r
    input_shape q_shape r_shape m n k batch_size batch =
  (* Copy input to R and initialize Q as identity *)
  let r_work =
    Array.init m (fun i ->
        Array.init n (fun j ->
            if batch_size = 1 then get_2d input i j
            else get_batched_2d input input_shape batch i j))
  in

  let q_work =
    Array.init m (fun i ->
        Array.init m (fun j ->
            if i = j then Dtype.one dtype else Dtype.zero dtype))
  in

  (* Householder QR decomposition *)
  let tau = Array.make k (Dtype.zero dtype) in
  (* scaling factors *)

  for j = 0 to k - 1 do
    (* Compute Householder vector for column j *)
    let col_norm_sq = ref (Dtype.zero dtype) in
    for i = j to m - 1 do
      let v = r_work.(i).(j) in
      col_norm_sq := Dtype.add dtype !col_norm_sq (Dtype.mul dtype v v)
    done;
    let col_norm = Dtype.of_float dtype (sqrt (to_float dtype !col_norm_sq)) in

    if to_float dtype col_norm > eps then (
      (* Determine the sign for numerical stability *)
      let sign =
        if to_float dtype r_work.(j).(j) >= 0.0 then Dtype.one dtype
        else Dtype.minus_one dtype
      in

      (* Compute Householder vector v *)
      let alpha = r_work.(j).(j) in
      let beta = Dtype.add dtype alpha (Dtype.mul dtype sign col_norm) in

      (* Guard against beta = 0 case *)
      if abs_float (to_float dtype beta) < eps_of_dtype dtype then
        (* Column is already zero, no reflection needed *)
        tau.(j) <- Dtype.zero dtype
      else
        (* Store Householder vector (normalized) in lower part of R *)
        let v = Array.make m (Dtype.zero dtype) in
        v.(j) <- Dtype.one dtype;
        for i = j + 1 to m - 1 do
          v.(i) <- Dtype.div dtype r_work.(i).(j) beta
        done;

        (* Compute tau = 2 / ||v||^2 *)
        let v_norm_sq = ref (Dtype.one dtype) in
        (* v[j] = 1 *)
        for i = j + 1 to m - 1 do
          v_norm_sq := Dtype.add dtype !v_norm_sq (Dtype.mul dtype v.(i) v.(i))
        done;
        tau.(j) <- Dtype.div dtype (Dtype.of_float dtype 2.0) !v_norm_sq;

        (* Apply Householder reflections *)
        apply_householder_to_r dtype r_work v tau.(j) j n;
        apply_householder_to_q dtype q_work v tau.(j) j m;

        (* Store the essential part of v below diagonal of R for potential later
           use *)
        for i = j + 1 to m - 1 do
          r_work.(i).(j) <- v.(i)
        done;

        (* Set diagonal element to negative norm for stability *)
        r_work.(j).(j) <-
          Dtype.mul dtype (Dtype.minus_one dtype)
            (Dtype.mul dtype sign col_norm))
  done;

  (* Copy results to output tensors *)
  if batch_size = 1 then (
    (* Copy Q (only needed columns for reduced QR) *)
    let q_cols = if reduced then k else m in
    for i = 0 to m - 1 do
      for j = 0 to q_cols - 1 do
        set_2d q i j q_work.(i).(j)
      done
    done;

    (* Copy R (upper triangular part) *)
    let r_rows = if reduced then k else m in
    for i = 0 to r_rows - 1 do
      for j = 0 to n - 1 do
        if i <= j then set_2d r i j r_work.(i).(j)
        else set_2d r i j (Dtype.zero dtype)
      done
    done)
  else
    (* Copy Q (only needed columns for reduced QR) *)
    let q_cols = if reduced then k else m in
    for i = 0 to m - 1 do
      for j = 0 to q_cols - 1 do
        set_batched_2d q q_shape batch i j q_work.(i).(j)
      done
    done;

    (* Copy R (upper triangular part) *)
    let r_rows = if reduced then k else m in
    for i = 0 to r_rows - 1 do
      for j = 0 to n - 1 do
        if i <= j then set_batched_2d r r_shape batch i j r_work.(i).(j)
        else set_batched_2d r r_shape batch i j (Dtype.zero dtype)
      done
    done

(* QR decomposition using Householder reflections *)
let qr (type a b) ~reduced ctx (input : (a, b) t) =
  let pool = ctx.pool in
  let batch_size, m, n = get_matrix_dims input in
  let k = min m n in

  let q_shape =
    if reduced then
      if batch_size = 1 then [| m; k |]
      else
        Array.append
          (Array.sub (get_shape input.view) 0
             (Array.length (get_shape input.view) - 2))
          [| m; k |]
    else if batch_size = 1 then [| m; m |]
    else
      Array.append
        (Array.sub (get_shape input.view) 0
           (Array.length (get_shape input.view) - 2))
        [| m; m |]
  in

  let r_shape =
    if reduced then
      if batch_size = 1 then [| k; n |]
      else
        Array.append
          (Array.sub (get_shape input.view) 0
             (Array.length (get_shape input.view) - 2))
          [| k; n |]
    else if batch_size = 1 then [| m; n |]
    else
      Array.append
        (Array.sub (get_shape input.view) 0
           (Array.length (get_shape input.view) - 2))
        [| m; n |]
  in

  let q = create_output ctx input.dtype q_shape in
  let r = create_output ctx input.dtype r_shape in

  (* Initialize Q and R to zero *)
  let q_elements = Array.fold_left ( * ) 1 q_shape in
  let r_elements = Array.fold_left ( * ) 1 r_shape in
  for i = 0 to q_elements - 1 do
    Array1.unsafe_set q.buffer i (Dtype.zero input.dtype)
  done;
  for i = 0 to r_elements - 1 do
    Array1.unsafe_set r.buffer i (Dtype.zero input.dtype)
  done;

  (* Get shapes for batch indexing *)
  let input_shape = get_shape input.view in

  (* Process batches in parallel if batch_size > 1 *)
  if batch_size > 1 then
    Parallel.parallel_for pool 0 batch_size (fun batch_start batch_end ->
        for batch = batch_start to batch_end - 1 do
          kernel_qr_single ~reduced input.dtype input q r input_shape q_shape
            r_shape m n k batch_size batch
        done)
  else
    kernel_qr_single ~reduced input.dtype input q r input_shape q_shape r_shape
      m n k batch_size 0;

  (q, r)

(* ===== Eigenvalue Decomposition ===== *)

(* General eigenvalue decomposition - returns complex
   eigenvalues/eigenvectors *)
let eig (type a b) ~vectors:_ _ctx (input : (a, b) t) =
  let _batch_size, n, m = get_matrix_dims input in
  if n <> m then failwith "op_eig: input must be square matrix";

  (* For now, we don't support general eigenvalue decomposition *)
  (* Would need Hessenberg reduction + Francis QR algorithm *)
  failwith "op_eig: general eigenvalue decomposition not yet implemented"

(* Kernel for single matrix symmetric eigenvalue decomposition *)
let kernel_eigh_single (type a b) ~vectors (dtype : (a, b) Dtype.t) input
    eigenvalues eigenvectors input_shape evecs_shape n batch_size batch =
  (* Copy input matrix to working array *)
  let a_work = Array.make_matrix n n (Dtype.zero dtype) in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      a_work.(i).(j) <-
        (if batch_size = 1 then get_2d input i j
         else get_batched_2d input input_shape batch i j)
    done
  done;

  (* Initialize eigenvector matrix as identity if needed *)
  let q_accum =
    if vectors then (
      let q = Array.make_matrix n n (Dtype.zero dtype) in
      for i = 0 to n - 1 do
        q.(i).(i) <- Dtype.one dtype
      done;
      Some q)
    else None
  in

  (* Tridiagonalization using Householder reflections for symmetric matrices *)
  let tridiag = Array.make n (Dtype.zero dtype) in
  let offdiag = Array.make (n - 1) (Dtype.zero dtype) in

  for k = 0 to n - 3 do
    (* Compute Householder vector for column k *)
    let sigma = ref (Dtype.zero dtype) in
    for i = k + 1 to n - 1 do
      let v = a_work.(i).(k) in
      sigma := Dtype.add dtype !sigma (Dtype.mul dtype v v)
    done;

    if to_float dtype !sigma > eps then (
      let alpha = a_work.(k + 1).(k) in
      let sign_alpha =
        if to_float dtype alpha >= 0.0 then Dtype.one dtype
        else Dtype.minus_one dtype
      in
      let s = Dtype.of_float dtype (sqrt (to_float dtype !sigma)) in
      let mu =
        Dtype.of_float dtype
          (sqrt
             (to_float dtype
                (Dtype.add dtype !sigma
                   (Dtype.mul dtype alpha (Dtype.mul dtype s sign_alpha)))))
      in

      (* Householder vector *)
      let v = Array.make n (Dtype.zero dtype) in
      v.(k + 1) <-
        Dtype.div dtype
          (Dtype.add dtype alpha (Dtype.mul dtype s sign_alpha))
          mu;
      for i = k + 2 to n - 1 do
        v.(i) <- Dtype.div dtype a_work.(i).(k) mu
      done;

      (* Apply Householder transformation: A = (I - 2vv^T)A(I - 2vv^T) *)
      (* First compute w = Av *)
      let w = Array.make n (Dtype.zero dtype) in
      for i = 0 to n - 1 do
        for j = k + 1 to n - 1 do
          w.(i) <- Dtype.add dtype w.(i) (Dtype.mul dtype a_work.(i).(j) v.(j))
        done
      done;

      (* Compute scalar beta = v^T w *)
      let beta = ref (Dtype.zero dtype) in
      for i = k + 1 to n - 1 do
        beta := Dtype.add dtype !beta (Dtype.mul dtype v.(i) w.(i))
      done;

      (* Update w: w = w - beta*v *)
      for i = 0 to n - 1 do
        w.(i) <- Dtype.sub dtype w.(i) (Dtype.mul dtype !beta v.(i))
      done;

      (* Update A: A = A - 2(vw^T + wv^T) *)
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let update =
            Dtype.add dtype
              (Dtype.mul dtype v.(i) w.(j))
              (Dtype.mul dtype w.(i) v.(j))
          in
          a_work.(i).(j) <-
            Dtype.sub dtype
              a_work.(i).(j)
              (Dtype.mul dtype (Dtype.of_float dtype 2.0) update)
        done
      done;

      (* Update eigenvector matrix if needed *)
      match q_accum with
      | Some q ->
          (* Q = Q(I - 2vv^T) *)
          for i = 0 to n - 1 do
            let dot = ref (Dtype.zero dtype) in
            for j = k + 1 to n - 1 do
              dot := Dtype.add dtype !dot (Dtype.mul dtype q.(i).(j) v.(j))
            done;
            for j = k + 1 to n - 1 do
              q.(i).(j) <-
                Dtype.sub dtype
                  q.(i).(j)
                  (Dtype.mul dtype (Dtype.of_float dtype 2.0)
                     (Dtype.mul dtype !dot v.(j)))
            done
          done
      | None -> ())
  done;

  (* Extract tridiagonal elements *)
  for i = 0 to n - 1 do
    tridiag.(i) <- a_work.(i).(i);
    if i < n - 1 then offdiag.(i) <- a_work.(i).(i + 1)
  done;

  (* QR algorithm on tridiagonal matrix with block deflation *)
  let max_iter = 100 * n in
  let tol = eps_of_dtype dtype *. float_of_int n in

  (* Deflate small off-diagonal elements *)
  let deflate () =
    for i = 0 to n - 2 do
      let t_i = to_float dtype tridiag.(i) in
      let t_i1 = to_float dtype tridiag.(i + 1) in
      let o_i = to_float dtype offdiag.(i) in
      if abs_float o_i <= tol *. (abs_float t_i +. abs_float t_i1) then
        offdiag.(i) <- Dtype.zero dtype
    done
  in

  for _iter = 0 to max_iter - 1 do
    (* Deflate negligible off-diagonals *)
    deflate ();

    (* Find active sub-blocks [p_start, p_end] *)
    let finished = ref true in
    let p_end = ref (n - 1) in

    while !p_end > 0 do
      (* Find the end of the current active block *)
      while !p_end > 0 && to_float dtype offdiag.(!p_end - 1) = 0.0 do
        decr p_end
      done;

      if !p_end = 0 then () (* All blocks processed *)
      else
        (* Find the start of the current active block *)
        let p_start = ref !p_end in
        while !p_start > 0 && to_float dtype offdiag.(!p_start - 1) <> 0.0 do
          decr p_start
        done;

        if !p_end > !p_start then (
          finished := false;
          let block_size = !p_end - !p_start + 1 in

          (* Wilkinson shift for this block *)
          let d =
            Dtype.div dtype
              (Dtype.sub dtype tridiag.(!p_end - 1) tridiag.(!p_end))
              (Dtype.of_float dtype 2.0)
          in
          let sign_d =
            if to_float dtype d >= 0.0 then Dtype.one dtype
            else Dtype.minus_one dtype
          in
          let shift =
            if block_size = 1 then tridiag.(!p_end)
            else
              Dtype.sub dtype tridiag.(!p_end)
                (Dtype.div dtype
                   (Dtype.mul dtype offdiag.(!p_end - 1) offdiag.(!p_end - 1))
                   (Dtype.add dtype d
                      (Dtype.mul dtype sign_d
                         (Dtype.of_float dtype
                            (sqrt
                               (to_float dtype
                                  (Dtype.add dtype (Dtype.mul dtype d d)
                                     (Dtype.mul dtype
                                        offdiag.(!p_end - 1)
                                        offdiag.(!p_end - 1)))))))))
          in

          (* QR step with shift on this block *)
          let x = ref (Dtype.sub dtype tridiag.(!p_start) shift) in
          let y =
            ref
              (if !p_start < n - 1 then offdiag.(!p_start) else Dtype.zero dtype)
          in

          for k = !p_start to !p_end - 1 do
            (* Compute Givens rotation *)
            let c, s = givens_rotation dtype !x !y in

            (* Apply rotation to tridiagonal matrix *)
            if k > !p_start then
              offdiag.(k - 1) <-
                Dtype.add dtype (Dtype.mul dtype c !x) (Dtype.mul dtype s !y);

            let temp1 =
              Dtype.add dtype
                (Dtype.mul dtype c (Dtype.mul dtype c tridiag.(k)))
                (Dtype.mul dtype s (Dtype.mul dtype s tridiag.(k + 1)))
            in
            let temp2 =
              Dtype.mul dtype (Dtype.of_float dtype 2.0)
                (Dtype.mul dtype c (Dtype.mul dtype s offdiag.(k)))
            in
            let new_diag_k = Dtype.add dtype temp1 temp2 in

            let temp3 =
              Dtype.add dtype
                (Dtype.mul dtype s (Dtype.mul dtype s tridiag.(k)))
                (Dtype.mul dtype c (Dtype.mul dtype c tridiag.(k + 1)))
            in
            let temp4 =
              Dtype.mul dtype (Dtype.of_float dtype 2.0)
                (Dtype.mul dtype c (Dtype.mul dtype s offdiag.(k)))
            in
            let new_diag_k1 = Dtype.sub dtype temp3 temp4 in

            let cs_diff =
              Dtype.sub dtype (Dtype.mul dtype c c) (Dtype.mul dtype s s)
            in
            let new_offdiag_k =
              Dtype.add dtype
                (Dtype.mul dtype cs_diff offdiag.(k))
                (Dtype.mul dtype c
                   (Dtype.mul dtype s
                      (Dtype.sub dtype tridiag.(k) tridiag.(k + 1))))
            in

            tridiag.(k) <- new_diag_k;
            tridiag.(k + 1) <- new_diag_k1;
            offdiag.(k) <- new_offdiag_k;

            if k < n - 2 then (
              x := Dtype.mul dtype s offdiag.(k + 1);
              y := Dtype.mul dtype c offdiag.(k + 1));

            (* Update eigenvector matrix if needed *)
            match q_accum with
            | Some q ->
                for i = 0 to n - 1 do
                  let q_ik = q.(i).(k) in
                  let q_ik1 = q.(i).(k + 1) in
                  q.(i).(k) <-
                    Dtype.add dtype (Dtype.mul dtype c q_ik)
                      (Dtype.mul dtype s q_ik1);
                  q.(i).(k + 1) <-
                    Dtype.sub dtype (Dtype.mul dtype c q_ik1)
                      (Dtype.mul dtype s q_ik)
                done
            | None -> ()
          done);

        (* Move to next block *)
        p_end := !p_start - 1
    done;

    (* Check if all blocks are converged *)
    if !finished then ()
  done;

  (* Sort eigenvalues in descending order *)
  let indices = Array.init n (fun i -> i) in
  Array.sort
    (fun i j ->
      compare (to_float dtype tridiag.(j)) (to_float dtype tridiag.(i)))
    indices;

  (* Copy eigenvalues to output (convert to float64) *)
  if batch_size = 1 then
    for i = 0 to n - 1 do
      let eval =
        Dtype.of_float Dtype.float64 (to_float dtype tridiag.(indices.(i)))
      in
      Array1.unsafe_set eigenvalues.buffer i eval
    done
  else
    for i = 0 to n - 1 do
      let idx = (batch * n) + i in
      let eval =
        Dtype.of_float Dtype.float64 (to_float dtype tridiag.(indices.(i)))
      in
      Array1.unsafe_set eigenvalues.buffer idx eval
    done;

  (* Copy eigenvectors to output if requested *)
  match (eigenvectors, q_accum) with
  | Some evecs, Some q ->
      if batch_size = 1 then
        for i = 0 to n - 1 do
          for j = 0 to n - 1 do
            set_2d evecs i j q.(i).(indices.(j))
          done
        done
      else
        for i = 0 to n - 1 do
          for j = 0 to n - 1 do
            set_batched_2d evecs evecs_shape batch i j q.(i).(indices.(j))
          done
        done
  | _ -> ()

(* Symmetric/Hermitian eigenvalue decomposition - returns real eigenvalues *)
let eigh (type a b) ~vectors ctx (input : (a, b) t) =
  let pool = ctx.pool in
  let batch_size, n, m = get_matrix_dims input in
  if n <> m then failwith "op_eigh: input must be square matrix";

  (* Output shapes *)
  let eigenvalues_shape =
    if batch_size = 1 then [| n |]
    else
      let batch_dims =
        Array.sub (get_shape input.view) 0
          (Array.length (get_shape input.view) - 2)
      in
      Array.append batch_dims [| n |]
  in

  let eigenvectors_shape = get_shape input.view in

  let eigenvalues = create_output ctx Dtype.float64 eigenvalues_shape in
  (* Always float64 *)
  let eigenvectors =
    if vectors then Some (create_output ctx input.dtype eigenvectors_shape)
    else None
  in

  (* Get shapes for batch indexing *)
  let input_shape = get_shape input.view in
  let evecs_shape =
    match eigenvectors with Some evecs -> get_shape evecs.view | None -> [||]
  in

  (* Process batches in parallel if batch_size > 1 *)
  if batch_size > 1 then
    Parallel.parallel_for pool 0 batch_size (fun batch_start batch_end ->
        for batch = batch_start to batch_end - 1 do
          kernel_eigh_single ~vectors input.dtype input eigenvalues eigenvectors
            input_shape evecs_shape n batch_size batch
        done)
  else
    kernel_eigh_single ~vectors input.dtype input eigenvalues eigenvectors
      input_shape evecs_shape n batch_size 0;

  (eigenvalues, eigenvectors)

(* ===== Triangular Solve ===== *)

(* Kernel for triangular solve on single matrix/vector pair *)
let kernel_triangular_solve_single (type a b) ~upper ~transpose ~unit_diag
    (dtype : (a, b) Dtype.t) a output a_shape output_shape n nb batch_size_a
    batch_size batch_a batch_b =
  (* Compute tolerance for zero diagonal check *)
  let tol = eps_of_dtype dtype *. float_of_int n in

  (* Solve for each column of b *)
  for col = 0 to nb - 1 do
    if upper && not transpose then
      (* Solve U*x = b using backward substitution *)
      for i = n - 1 downto 0 do
        let sum =
          ref
            (if batch_size = 1 then get_2d output i col
             else get_batched_2d output output_shape batch_b i col)
        in
        for j = i + 1 to n - 1 do
          let a_ij =
            if batch_size_a = 1 then get_2d a i j
            else get_batched_2d a a_shape batch_a i j
          in
          let x_j =
            if batch_size = 1 then get_2d output j col
            else get_batched_2d output output_shape batch_b j col
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype a_ij x_j)
        done;
        let a_ii =
          if unit_diag then Dtype.one dtype
          else if batch_size_a = 1 then get_2d a i i
          else get_batched_2d a a_shape batch_a i i
        in
        (* Check for near-zero diagonal *)
        if (not unit_diag) && abs_float (to_float dtype a_ii) < tol then
          invalid_arg "solve: matrix is singular";
        let x_i = Dtype.div dtype !sum a_ii in
        if batch_size = 1 then set_2d output i col x_i
        else set_batched_2d output output_shape batch_b i col x_i
      done
    else if (not upper) && not transpose then
      (* Solve L*x = b using forward substitution *)
      for i = 0 to n - 1 do
        let sum =
          ref
            (if batch_size = 1 then get_2d output i col
             else get_batched_2d output output_shape batch_b i col)
        in
        for j = 0 to i - 1 do
          let a_ij =
            if batch_size_a = 1 then get_2d a i j
            else get_batched_2d a a_shape batch_a i j
          in
          let x_j =
            if batch_size = 1 then get_2d output j col
            else get_batched_2d output output_shape batch_b j col
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype a_ij x_j)
        done;
        let a_ii =
          if unit_diag then Dtype.one dtype
          else if batch_size_a = 1 then get_2d a i i
          else get_batched_2d a a_shape batch_a i i
        in
        (* Check for near-zero diagonal *)
        if (not unit_diag) && abs_float (to_float dtype a_ii) < tol then
          invalid_arg "solve: matrix is singular";
        let x_i = Dtype.div dtype !sum a_ii in
        if batch_size = 1 then set_2d output i col x_i
        else set_batched_2d output output_shape batch_b i col x_i
      done
    else if upper && transpose then
      (* Solve U^T*x = b using forward substitution *)
      for i = 0 to n - 1 do
        let sum =
          ref
            (if batch_size = 1 then get_2d output i col
             else get_batched_2d output output_shape batch_b i col)
        in
        for j = 0 to i - 1 do
          let a_ji =
            if batch_size_a = 1 then get_2d a j i
            else get_batched_2d a a_shape batch_a j i
          in
          let x_j =
            if batch_size = 1 then get_2d output j col
            else get_batched_2d output output_shape batch_b j col
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype a_ji x_j)
        done;
        let a_ii =
          if unit_diag then Dtype.one dtype
          else if batch_size_a = 1 then get_2d a i i
          else get_batched_2d a a_shape batch_a i i
        in
        (* Check for near-zero diagonal *)
        if (not unit_diag) && abs_float (to_float dtype a_ii) < tol then
          invalid_arg "solve: matrix is singular";
        let x_i = Dtype.div dtype !sum a_ii in
        if batch_size = 1 then set_2d output i col x_i
        else set_batched_2d output output_shape batch_b i col x_i
      done
    else (* not upper && transpose *)
      (* Solve L^T*x = b using backward substitution *)
      for i = n - 1 downto 0 do
        let sum =
          ref
            (if batch_size = 1 then get_2d output i col
             else get_batched_2d output output_shape batch_b i col)
        in
        for j = i + 1 to n - 1 do
          let a_ji =
            if batch_size_a = 1 then get_2d a j i
            else get_batched_2d a a_shape batch_a j i
          in
          let x_j =
            if batch_size = 1 then get_2d output j col
            else get_batched_2d output output_shape batch_b j col
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype a_ji x_j)
        done;
        let a_ii =
          if unit_diag then Dtype.one dtype
          else if batch_size_a = 1 then get_2d a i i
          else get_batched_2d a a_shape batch_a i i
        in
        (* Check for near-zero diagonal *)
        if (not unit_diag) && abs_float (to_float dtype a_ii) < tol then
          invalid_arg "solve: matrix is singular";
        let x_i = Dtype.div dtype !sum a_ii in
        if batch_size = 1 then set_2d output i col x_i
        else set_batched_2d output output_shape batch_b i col x_i
      done
  done

(* Triangular solve using forward/backward substitution *)
let triangular_solve (type a b) ~upper ~transpose ~unit_diag ctx (a : (a, b) t)
    (b : (a, b) t) =
  let pool = ctx.pool in
  let batch_size_a, n, m = get_matrix_dims a in
  if n <> m then failwith "op_triangular_solve: matrix A must be square";

  (* Handle 1D input b by expanding to 2D *)
  let b_shape = get_shape b.view in
  let b_ndim = Array.length b_shape in
  let b_is_1d = b_ndim = 1 in

  let b_expanded =
    if b_is_1d then
      (* Expand 1D to 2D by adding a trailing dimension *)
      let new_shape = [| b_shape.(0); 1 |] in
      {
        b with
        view = Lazy_view.reshape (Symbolic_shape.of_ints new_shape) b.view;
      }
    else b
  in

  let batch_size_b, mb, nb = get_matrix_dims b_expanded in
  if batch_size_a <> 1 && batch_size_b <> 1 && batch_size_a <> batch_size_b then
    failwith
      (Printf.sprintf
         "op_triangular_solve: batch dimensions must match (a=%d, b=%d)"
         batch_size_a batch_size_b);

  let batch_size = max batch_size_a batch_size_b in

  if mb <> n then
    failwith "op_triangular_solve: dimensions of A and b are incompatible";

  (* Create output with expanded shape *)
  let output_shape = get_shape b_expanded.view in
  let output = create_output ctx b.dtype output_shape in

  (* Copy b to output *)
  Array1.blit b_expanded.buffer output.buffer;

  (* Get shapes for batch indexing *)
  let a_shape = get_shape a.view in

  (* Process batches in parallel if batch_size > 1 *)
  (if batch_size > 1 then
     Parallel.parallel_for pool 0 batch_size (fun batch_start batch_end ->
         for batch = batch_start to batch_end - 1 do
           let batch_a = if batch_size_a = 1 then 0 else batch in
           let batch_b = if batch_size_b = 1 then 0 else batch in
           kernel_triangular_solve_single ~upper ~transpose ~unit_diag a.dtype a
             output a_shape output_shape n nb batch_size_a batch_size batch_a
             batch_b
         done)
   else
     let batch_a = 0 in
     let batch_b = 0 in
     kernel_triangular_solve_single ~upper ~transpose ~unit_diag a.dtype a
       output a_shape output_shape n nb batch_size_a batch_size batch_a batch_b);

  (* Reshape output to match input b shape *)
  if b_is_1d then
    {
      output with
      view =
        Lazy_view.reshape (Symbolic_shape.of_ints [| b_shape.(0) |]) output.view;
    }
  else output
