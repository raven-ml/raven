open Bigarray
module Dtype = Nx_core.Dtype
module Shape = Nx_core.Shape
module View = Nx_core.View
open Internal

(* Constants *)
let eps = 1e-10

(* Helper functions for matrix operations *)

(* Convert value to float based on dtype *)
let[@inline] to_float (type a b) (dtype : (a, b) Dtype.t) (value : a) : float =
  match dtype with
  | Dtype.Float32 -> value
  | Dtype.Float64 -> value
  | _ -> failwith "to_float: only float dtypes are supported"

let[@inline] get_2d t i j =
  let strides = View.strides t.view in
  let stride0 = strides.(0) in
  let stride1 = strides.(1) in
  let offset = View.offset t.view in
  Array1.unsafe_get t.buffer (offset + (i * stride0) + (j * stride1))

let[@inline] set_2d t i j value =
  let strides = View.strides t.view in
  let stride0 = strides.(0) in
  let stride1 = strides.(1) in
  let offset = View.offset t.view in
  Array1.unsafe_set t.buffer (offset + (i * stride0) + (j * stride1)) value

let[@inline] get_3d t b i j =
  let strides = View.strides t.view in
  let stride0 = strides.(0) in
  let stride1 = strides.(1) in
  let stride2 = strides.(2) in
  let offset = View.offset t.view in
  Array1.unsafe_get t.buffer
    (offset + (b * stride0) + (i * stride1) + (j * stride2))

let[@inline] set_3d t b i j value =
  let strides = View.strides t.view in
  let stride0 = strides.(0) in
  let stride1 = strides.(1) in
  let stride2 = strides.(2) in
  let offset = View.offset t.view in
  Array1.unsafe_set t.buffer
    (offset + (b * stride0) + (i * stride1) + (j * stride2))
    value

(* Get matrix dimensions, handling batched case *)
let get_matrix_dims t =
  let shape = View.shape t.view in
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
    output n batch_size batch =
  if upper then
    (* Upper triangular Cholesky: A = U^T * U *)
    for i = 0 to n - 1 do
      for j = i to n - 1 do
        let sum =
          ref
            (if batch_size = 1 then get_2d input i j else get_3d input batch i j)
        in
        for k = 0 to i - 1 do
          let u_ki =
            if batch_size = 1 then get_2d output k i
            else get_3d output batch k i
          in
          let u_kj =
            if batch_size = 1 then get_2d output k j
            else get_3d output batch k j
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype u_ki u_kj)
        done;

        let value =
          if i = j then (
            (* Diagonal element *)
            let s_float = to_float dtype !sum in
            if s_float <= 0.0 then
              failwith "op_cholesky: matrix is not positive definite";
            Dtype.of_float dtype (sqrt s_float))
          else
            (* Off-diagonal element *)
            let u_ii =
              if batch_size = 1 then get_2d output i i
              else get_3d output batch i i
            in
            Dtype.div dtype !sum u_ii
        in
        if batch_size = 1 then set_2d output i j value
        else set_3d output batch i j value
      done
    done
  else
    (* Lower triangular Cholesky: A = L * L^T *)
    for i = 0 to n - 1 do
      for j = 0 to i do
        let sum =
          ref
            (if batch_size = 1 then get_2d input i j else get_3d input batch i j)
        in
        for k = 0 to j - 1 do
          let l_ik =
            if batch_size = 1 then get_2d output i k
            else get_3d output batch i k
          in
          let l_jk =
            if batch_size = 1 then get_2d output j k
            else get_3d output batch j k
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype l_ik l_jk)
        done;

        let value =
          if i = j then (
            (* Diagonal element *)
            let s_float = to_float dtype !sum in
            if s_float <= 0.0 then
              failwith "op_cholesky: matrix is not positive definite";
            Dtype.of_float dtype (sqrt s_float))
          else
            (* Off-diagonal element *)
            let l_jj =
              if batch_size = 1 then get_2d output j j
              else get_3d output batch j j
            in
            Dtype.div dtype !sum l_jj
        in
        if batch_size = 1 then set_2d output i j value
        else set_3d output batch i j value
      done
    done

(* Cholesky decomposition using Cholesky-Banachiewicz algorithm *)
let cholesky (type a b) ~upper ctx (input : (a, b) t) =
  let pool = ctx.pool in
  let batch_size, n, m = get_matrix_dims input in
  if n <> m then failwith "op_cholesky: input must be square matrix";

  let output_shape = View.shape input.view in
  let output = create_output ctx input.dtype output_shape in

  (* Initialize output to zero *)
  let total_elements = Array.fold_left ( * ) 1 output_shape in
  for i = 0 to total_elements - 1 do
    Array1.unsafe_set output.buffer i (Dtype.zero input.dtype)
  done;

  (* Process batches in parallel if batch_size > 1 *)
  if batch_size > 1 then
    Parallel.parallel_for pool 0 (batch_size - 1) (fun batch_start batch_end ->
        for batch = batch_start to batch_end - 1 do
          kernel_cholesky_single ~upper input.dtype input output n batch_size
            batch
        done)
  else kernel_cholesky_single ~upper input.dtype input output n batch_size 0;

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
let kernel_qr_single (type a b) ~reduced (dtype : (a, b) Dtype.t) input q r m n
    k batch_size batch =
  (* Copy input to R and initialize Q as identity *)
  let r_work =
    Array.init m (fun i ->
        Array.init n (fun j ->
            if batch_size = 1 then get_2d input i j else get_3d input batch i j))
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
      done)
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
        set_3d q batch i j q_work.(i).(j)
      done
    done;

    (* Copy R (upper triangular part) *)
    let r_rows = if reduced then k else m in
    for i = 0 to r_rows - 1 do
      for j = 0 to n - 1 do
        if i <= j then set_3d r batch i j r_work.(i).(j)
        else set_3d r batch i j (Dtype.zero dtype)
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
          (Array.sub (View.shape input.view) 0
             (Array.length (View.shape input.view) - 2))
          [| m; k |]
    else if batch_size = 1 then [| m; m |]
    else
      Array.append
        (Array.sub (View.shape input.view) 0
           (Array.length (View.shape input.view) - 2))
        [| m; m |]
  in

  let r_shape =
    if reduced then
      if batch_size = 1 then [| k; n |]
      else
        Array.append
          (Array.sub (View.shape input.view) 0
             (Array.length (View.shape input.view) - 2))
          [| k; n |]
    else if batch_size = 1 then [| m; n |]
    else
      Array.append
        (Array.sub (View.shape input.view) 0
           (Array.length (View.shape input.view) - 2))
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

  (* Process batches in parallel if batch_size > 1 *)
  if batch_size > 1 then
    Parallel.parallel_for pool 0 (batch_size - 1) (fun batch_start batch_end ->
        for batch = batch_start to batch_end - 1 do
          kernel_qr_single ~reduced input.dtype input q r m n k batch_size batch
        done)
  else kernel_qr_single ~reduced input.dtype input q r m n k batch_size 0;

  (q, r)

(* ===== SVD Decomposition ===== *)

(* Apply Householder reflection for bidiagonalization *)
let apply_householder_bidiag_left (type a b) (dtype : (a, b) Dtype.t) a_work
    u_vec beta i m n =
  (* A = (I - beta * u * u^T) * A *)
  for j = i to n - 1 do
    let dot = ref (Dtype.zero dtype) in
    for l = i to m - 1 do
      dot := Dtype.add dtype !dot (Dtype.mul dtype u_vec.(l) a_work.(l).(j))
    done;
    let scale = Dtype.mul dtype beta !dot in
    for l = i to m - 1 do
      a_work.(l).(j) <-
        Dtype.sub dtype a_work.(l).(j) (Dtype.mul dtype scale u_vec.(l))
    done
  done

let apply_householder_bidiag_right (type a b) (dtype : (a, b) Dtype.t) a_work
    v_vec beta i m n =
  (* A = A * (I - beta * v * v^T) *)
  for j = i to m - 1 do
    let dot = ref (Dtype.zero dtype) in
    for l = i + 1 to n - 1 do
      dot := Dtype.add dtype !dot (Dtype.mul dtype a_work.(j).(l) v_vec.(l))
    done;
    let scale = Dtype.mul dtype beta !dot in
    for l = i + 1 to n - 1 do
      a_work.(j).(l) <-
        Dtype.sub dtype a_work.(j).(l) (Dtype.mul dtype scale v_vec.(l))
    done
  done

(* Update U or V matrix during bidiagonalization *)
let update_orthogonal_matrix (type a b) (dtype : (a, b) Dtype.t) mat vec beta
    start_idx m =
  for j = 0 to m - 1 do
    let dot = ref (Dtype.zero dtype) in
    for l = start_idx to m - 1 do
      dot := Dtype.add dtype !dot (Dtype.mul dtype mat.(l).(j) vec.(l))
    done;
    let scale = Dtype.mul dtype beta !dot in
    for l = start_idx to m - 1 do
      mat.(l).(j) <- Dtype.sub dtype mat.(l).(j) (Dtype.mul dtype scale vec.(l))
    done
  done

(* Kernel for single matrix SVD *)
let kernel_svd_single (type a b) ~full_matrices (dtype : (a, b) Dtype.t) input u
    s v m n k batch_size batch =
  (* Copy input matrix to working array *)
  let a_work = Array.make_matrix m n (Dtype.zero dtype) in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      a_work.(i).(j) <-
        (if batch_size = 1 then get_2d input i j else get_3d input batch i j)
    done
  done;

  (* Initialize U and V as identity matrices *)
  let u_work = Array.make_matrix m m (Dtype.zero dtype) in
  let v_work = Array.make_matrix n n (Dtype.zero dtype) in
  for i = 0 to m - 1 do
    u_work.(i).(i) <- Dtype.one dtype
  done;
  for i = 0 to n - 1 do
    v_work.(i).(i) <- Dtype.one dtype
  done;

  (* Golub-Kahan bidiagonalization *)
  for i = 0 to k - 1 do
    (* Step 1: Left Householder reflection to zero out column i below
       diagonal *)
    if i < m - 1 then (
      let col_norm_sq = ref (Dtype.zero dtype) in
      for j = i to m - 1 do
        let v = a_work.(j).(i) in
        col_norm_sq := Dtype.add dtype !col_norm_sq (Dtype.mul dtype v v)
      done;
      let col_norm =
        Dtype.of_float dtype (sqrt (to_float dtype !col_norm_sq))
      in

      if to_float dtype col_norm > eps then (
        (* Compute Householder vector *)
        let sign =
          if to_float dtype a_work.(i).(i) >= 0.0 then Dtype.one dtype
          else Dtype.minus_one dtype
        in
        let u_0 =
          Dtype.add dtype a_work.(i).(i) (Dtype.mul dtype sign col_norm)
        in
        let beta =
          Dtype.div dtype (Dtype.of_float dtype 2.0)
            (Dtype.mul dtype u_0 col_norm)
        in

        (* Apply reflection to A *)
        let u_vec = Array.make m (Dtype.zero dtype) in
        u_vec.(i) <- u_0;
        for j = i + 1 to m - 1 do
          u_vec.(j) <- a_work.(j).(i)
        done;

        apply_householder_bidiag_left dtype a_work u_vec beta i m n;
        update_orthogonal_matrix dtype u_work u_vec beta i m));

    (* Step 2: Right Householder reflection to zero out row i to the right of
       superdiagonal *)
    if i < n - 2 then (
      let row_norm_sq = ref (Dtype.zero dtype) in
      for j = i + 1 to n - 1 do
        let v = a_work.(i).(j) in
        row_norm_sq := Dtype.add dtype !row_norm_sq (Dtype.mul dtype v v)
      done;
      let row_norm =
        Dtype.of_float dtype (sqrt (to_float dtype !row_norm_sq))
      in

      if to_float dtype row_norm > eps then (
        (* Compute Householder vector *)
        let sign =
          if to_float dtype a_work.(i).(i + 1) >= 0.0 then Dtype.one dtype
          else Dtype.minus_one dtype
        in
        let v_0 =
          Dtype.add dtype a_work.(i).(i + 1) (Dtype.mul dtype sign row_norm)
        in
        let beta =
          Dtype.div dtype (Dtype.of_float dtype 2.0)
            (Dtype.mul dtype v_0 row_norm)
        in

        (* Apply reflection to A from the right *)
        let v_vec = Array.make n (Dtype.zero dtype) in
        v_vec.(i + 1) <- v_0;
        for j = i + 2 to n - 1 do
          v_vec.(j) <- a_work.(i).(j)
        done;

        apply_householder_bidiag_right dtype a_work v_vec beta i m n;

        (* Update V *)
        for j = 0 to n - 1 do
          let dot = ref (Dtype.zero dtype) in
          for l = i + 1 to n - 1 do
            dot :=
              Dtype.add dtype !dot (Dtype.mul dtype v_work.(j).(l) v_vec.(l))
          done;
          let scale = Dtype.mul dtype beta !dot in
          for l = i + 1 to n - 1 do
            v_work.(j).(l) <-
              Dtype.sub dtype v_work.(j).(l) (Dtype.mul dtype scale v_vec.(l))
          done
        done))
  done;

  (* Extract diagonal and superdiagonal from bidiagonal matrix *)
  let diag = Array.make k (Dtype.zero dtype) in
  let superdiag = Array.make (max 0 (k - 1)) (Dtype.zero dtype) in
  for i = 0 to k - 1 do
    diag.(i) <- a_work.(i).(i);
    if i < k - 1 && i < n - 1 then superdiag.(i) <- a_work.(i).(i + 1)
  done;

  (* Implicit QR algorithm with Wilkinson shift for singular values *)
  let max_iter = 30 * k in
  (* Usually converges much faster *)

  for _ = 0 to max_iter - 1 do
    (* Check for convergence and deflation *)
    let converged = ref true in
    for i = 0 to k - 2 do
      if abs_float (to_float dtype superdiag.(i)) > eps then converged := false
    done;
    if !converged then ()
    else
      (* Find the largest unconverged block *)
      let p = ref (k - 1) in
      while !p > 0 && abs_float (to_float dtype superdiag.(!p - 1)) < eps do
        decr p
      done;

      if !p > 0 then
        (* Apply QR step with shift *)
        let shift =
          let a = to_float dtype diag.(!p) in
          let b = to_float dtype diag.(!p - 1) in
          let c = if !p > 1 then to_float dtype superdiag.(!p - 2) else 0.0 in
          (* Wilkinson shift *)
          let d = (b -. a) /. 2.0 in
          let sign_d = if d >= 0.0 then 1.0 else -1.0 in
          a -. (c *. c /. (d +. (sign_d *. sqrt ((d *. d) +. (c *. c)))))
        in

        (* QR factorization of shifted matrix *)
        for i = 0 to !p - 1 do
          let d_shifted =
            if i = 0 then Dtype.sub dtype diag.(i) (Dtype.of_float dtype shift)
            else diag.(i)
          in

          if i < !p - 1 then (
            (* Compute Givens rotation *)
            let c, s = givens_rotation dtype d_shifted superdiag.(i) in

            (* Apply rotation *)
            let new_diag =
              Dtype.add dtype
                (Dtype.mul dtype c d_shifted)
                (Dtype.mul dtype s superdiag.(i))
            in
            let new_super =
              if i < !p - 2 then Dtype.mul dtype s superdiag.(i + 1)
              else Dtype.zero dtype
            in

            diag.(i) <- new_diag;
            if i < !p - 2 then superdiag.(i + 1) <- new_super;

            if i < !p - 1 then (
              let temp =
                Dtype.sub dtype
                  (Dtype.mul dtype c diag.(i + 1))
                  (Dtype.mul dtype s superdiag.(i))
              in
              superdiag.(i) <-
                Dtype.add dtype
                  (Dtype.mul dtype s diag.(i + 1))
                  (Dtype.mul dtype c superdiag.(i));
              diag.(i + 1) <- temp);

            (* Update U and V *)
            for j = 0 to m - 1 do
              let u_i = u_work.(j).(i) in
              let u_i1 =
                if i + 1 < m then u_work.(j).(i + 1) else Dtype.zero dtype
              in
              u_work.(j).(i) <-
                Dtype.add dtype (Dtype.mul dtype c u_i) (Dtype.mul dtype s u_i1);
              if i + 1 < m then
                u_work.(j).(i + 1) <-
                  Dtype.sub dtype (Dtype.mul dtype c u_i1)
                    (Dtype.mul dtype s u_i)
            done;

            for j = 0 to n - 1 do
              let v_i = v_work.(j).(i) in
              let v_i1 =
                if i + 1 < n then v_work.(j).(i + 1) else Dtype.zero dtype
              in
              v_work.(j).(i) <-
                Dtype.add dtype (Dtype.mul dtype c v_i) (Dtype.mul dtype s v_i1);
              if i + 1 < n then
                v_work.(j).(i + 1) <-
                  Dtype.sub dtype (Dtype.mul dtype c v_i1)
                    (Dtype.mul dtype s v_i)
            done)
        done
  done;

  (* Ensure singular values are positive and sort in descending order *)
  for i = 0 to k - 1 do
    if to_float dtype diag.(i) < 0.0 then (
      diag.(i) <- Dtype.mul dtype (Dtype.minus_one dtype) diag.(i);
      (* Flip corresponding column of U *)
      for j = 0 to m - 1 do
        u_work.(j).(i) <- Dtype.mul dtype (Dtype.minus_one dtype) u_work.(j).(i)
      done)
  done;

  (* Sort singular values in descending order *)
  let indices = Array.init k (fun i -> i) in
  Array.sort
    (fun i j -> compare (to_float dtype diag.(j)) (to_float dtype diag.(i)))
    indices;

  (* Reorder based on sorted indices *)
  let sorted_diag = Array.map (fun i -> diag.(i)) indices in
  let sorted_u =
    Array.make_matrix m (if full_matrices then m else k) (Dtype.zero dtype)
  in
  let sorted_v =
    Array.make_matrix n (if full_matrices then n else k) (Dtype.zero dtype)
  in

  for idx = 0 to k - 1 do
    let orig_idx = indices.(idx) in
    for i = 0 to m - 1 do
      sorted_u.(i).(idx) <- u_work.(i).(orig_idx)
    done;
    for i = 0 to n - 1 do
      sorted_v.(i).(idx) <- v_work.(i).(orig_idx)
    done
  done;

  (* Copy to output tensors *)
  if batch_size = 1 then (
    (* Copy singular values (convert to float64) *)
    for i = 0 to k - 1 do
      let s_val =
        Dtype.of_float Dtype.float64 (to_float dtype sorted_diag.(i))
      in
      Array1.unsafe_set s.buffer i s_val
    done;

    (* Copy U *)
    let u_cols = if full_matrices then m else k in
    for i = 0 to m - 1 do
      for j = 0 to u_cols - 1 do
        set_2d u i j sorted_u.(i).(j)
      done
    done;

    (* Copy V^H (transpose of V) *)
    let v_rows = if full_matrices then n else k in
    for i = 0 to v_rows - 1 do
      for j = 0 to n - 1 do
        set_2d v i j sorted_v.(j).(i)
      done
    done)
  else (
    (* Copy for batched case *)
    (* Copy singular values (convert to float64) *)
    for i = 0 to k - 1 do
      let idx = (batch * k) + i in
      let s_val =
        Dtype.of_float Dtype.float64 (to_float dtype sorted_diag.(i))
      in
      Array1.unsafe_set s.buffer idx s_val
    done;

    (* Copy U *)
    let u_cols = if full_matrices then m else k in
    for i = 0 to m - 1 do
      for j = 0 to u_cols - 1 do
        set_3d u batch i j sorted_u.(i).(j)
      done
    done;

    (* Copy V^H (transpose of V) *)
    let v_rows = if full_matrices then n else k in
    for i = 0 to v_rows - 1 do
      for j = 0 to n - 1 do
        set_3d v batch i j sorted_v.(j).(i)
      done
    done)

(* SVD using Golub-Kahan bidiagonalization followed by QR iteration *)
let svd (type a b) ~full_matrices ctx (input : (a, b) t) =
  let pool = ctx.pool in
  let batch_size, m, n = get_matrix_dims input in
  let k = min m n in

  (* Determine output shapes based on full_matrices flag *)
  let u_shape =
    if batch_size = 1 then if full_matrices then [| m; m |] else [| m; k |]
    else
      let batch_dims =
        Array.sub (View.shape input.view) 0
          (Array.length (View.shape input.view) - 2)
      in
      if full_matrices then Array.append batch_dims [| m; m |]
      else Array.append batch_dims [| m; k |]
  in

  let v_shape =
    if batch_size = 1 then if full_matrices then [| n; n |] else [| k; n |]
    else
      let batch_dims =
        Array.sub (View.shape input.view) 0
          (Array.length (View.shape input.view) - 2)
      in
      if full_matrices then Array.append batch_dims [| n; n |]
      else Array.append batch_dims [| k; n |]
  in

  let s_shape =
    if batch_size = 1 then [| k |]
    else
      let batch_dims =
        Array.sub (View.shape input.view) 0
          (Array.length (View.shape input.view) - 2)
      in
      Array.append batch_dims [| k |]
  in

  let u = create_output ctx input.dtype u_shape in
  let s = create_output ctx Dtype.float64 s_shape in
  (* Always float64 for singular values *)
  let v = create_output ctx input.dtype v_shape in

  (* Process batches in parallel if batch_size > 1 *)
  if batch_size > 1 then
    Parallel.parallel_for pool 0 (batch_size - 1) (fun batch_start batch_end ->
        for batch = batch_start to batch_end - 1 do
          kernel_svd_single ~full_matrices input.dtype input u s v m n k
            batch_size batch
        done)
  else
    kernel_svd_single ~full_matrices input.dtype input u s v m n k batch_size 0;

  (u, s, v)

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
    eigenvalues eigenvectors n batch_size batch =
  (* Copy input matrix to working array *)
  let a_work = Array.make_matrix n n (Dtype.zero dtype) in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      a_work.(i).(j) <-
        (if batch_size = 1 then get_2d input i j else get_3d input batch i j)
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

  (* QR algorithm on tridiagonal matrix *)
  let max_iter = 100 * n in

  for _iter = 0 to max_iter - 1 do
    (* Check convergence *)
    let converged = ref true in
    for i = 0 to n - 2 do
      if
        abs_float (to_float dtype offdiag.(i))
        > eps
          *. (abs_float (to_float dtype tridiag.(i))
             +. abs_float (to_float dtype tridiag.(i + 1)))
      then converged := false
    done;

    if !converged then ()
    else
      (* Wilkinson shift *)
      let d =
        Dtype.div dtype
          (Dtype.sub dtype tridiag.(n - 2) tridiag.(n - 1))
          (Dtype.of_float dtype 2.0)
      in
      let sign_d =
        if to_float dtype d >= 0.0 then Dtype.one dtype
        else Dtype.minus_one dtype
      in
      let shift =
        Dtype.sub dtype
          tridiag.(n - 1)
          (Dtype.div dtype
             (Dtype.mul dtype offdiag.(n - 2) offdiag.(n - 2))
             (Dtype.add dtype d
                (Dtype.mul dtype sign_d
                   (Dtype.of_float dtype
                      (sqrt
                         (to_float dtype
                            (Dtype.add dtype (Dtype.mul dtype d d)
                               (Dtype.mul dtype offdiag.(n - 2) offdiag.(n - 2)))))))))
      in

      (* QR step with shift *)
      let x = ref (Dtype.sub dtype tridiag.(0) shift) in
      let y = ref offdiag.(0) in

      for k = 0 to n - 2 do
        (* Compute Givens rotation *)
        let c, s = givens_rotation dtype !x !y in

        (* Apply rotation to tridiagonal matrix *)
        if k > 0 then
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
               (Dtype.mul dtype s (Dtype.sub dtype tridiag.(k) tridiag.(k + 1))))
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
      done
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
            set_3d evecs batch i j q.(i).(indices.(j))
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
        Array.sub (View.shape input.view) 0
          (Array.length (View.shape input.view) - 2)
      in
      Array.append batch_dims [| n |]
  in

  let eigenvectors_shape = View.shape input.view in

  let eigenvalues = create_output ctx Dtype.float64 eigenvalues_shape in
  (* Always float64 *)
  let eigenvectors =
    if vectors then Some (create_output ctx input.dtype eigenvectors_shape)
    else None
  in

  (* Process batches in parallel if batch_size > 1 *)
  if batch_size > 1 then
    Parallel.parallel_for pool 0 (batch_size - 1) (fun batch_start batch_end ->
        for batch = batch_start to batch_end - 1 do
          kernel_eigh_single ~vectors input.dtype input eigenvalues eigenvectors
            n batch_size batch
        done)
  else
    kernel_eigh_single ~vectors input.dtype input eigenvalues eigenvectors n
      batch_size 0;

  (eigenvalues, eigenvectors)

(* ===== Triangular Solve ===== *)

(* Kernel for triangular solve on single matrix/vector pair *)
let kernel_triangular_solve_single (type a b) ~upper ~transpose ~unit_diag
    (dtype : (a, b) Dtype.t) a output n nb batch_size_a batch_size batch_a
    batch_b =
  (* Solve for each column of b *)
  for col = 0 to nb - 1 do
    if upper && not transpose then
      (* Solve U*x = b using backward substitution *)
      for i = n - 1 downto 0 do
        let sum =
          ref
            (if batch_size = 1 then get_2d output i col
             else get_3d output batch_b i col)
        in
        for j = i + 1 to n - 1 do
          let a_ij =
            if batch_size_a = 1 then get_2d a i j else get_3d a batch_a i j
          in
          let x_j =
            if batch_size = 1 then get_2d output j col
            else get_3d output batch_b j col
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype a_ij x_j)
        done;
        let a_ii =
          if unit_diag then Dtype.one dtype
          else if batch_size_a = 1 then get_2d a i i
          else get_3d a batch_a i i
        in
        let x_i = Dtype.div dtype !sum a_ii in
        if batch_size = 1 then set_2d output i col x_i
        else set_3d output batch_b i col x_i
      done
    else if (not upper) && not transpose then
      (* Solve L*x = b using forward substitution *)
      for i = 0 to n - 1 do
        let sum =
          ref
            (if batch_size = 1 then get_2d output i col
             else get_3d output batch_b i col)
        in
        for j = 0 to i - 1 do
          let a_ij =
            if batch_size_a = 1 then get_2d a i j else get_3d a batch_a i j
          in
          let x_j =
            if batch_size = 1 then get_2d output j col
            else get_3d output batch_b j col
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype a_ij x_j)
        done;
        let a_ii =
          if unit_diag then Dtype.one dtype
          else if batch_size_a = 1 then get_2d a i i
          else get_3d a batch_a i i
        in
        let x_i = Dtype.div dtype !sum a_ii in
        if batch_size = 1 then set_2d output i col x_i
        else set_3d output batch_b i col x_i
      done
    else if upper && transpose then
      (* Solve U^T*x = b using forward substitution *)
      for i = 0 to n - 1 do
        let sum =
          ref
            (if batch_size = 1 then get_2d output i col
             else get_3d output batch_b i col)
        in
        for j = 0 to i - 1 do
          let a_ji =
            if batch_size_a = 1 then get_2d a j i else get_3d a batch_a j i
          in
          let x_j =
            if batch_size = 1 then get_2d output j col
            else get_3d output batch_b j col
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype a_ji x_j)
        done;
        let a_ii =
          if unit_diag then Dtype.one dtype
          else if batch_size_a = 1 then get_2d a i i
          else get_3d a batch_a i i
        in
        let x_i = Dtype.div dtype !sum a_ii in
        if batch_size = 1 then set_2d output i col x_i
        else set_3d output batch_b i col x_i
      done
    else (* not upper && transpose *)
      (* Solve L^T*x = b using backward substitution *)
      for i = n - 1 downto 0 do
        let sum =
          ref
            (if batch_size = 1 then get_2d output i col
             else get_3d output batch_b i col)
        in
        for j = i + 1 to n - 1 do
          let a_ji =
            if batch_size_a = 1 then get_2d a j i else get_3d a batch_a j i
          in
          let x_j =
            if batch_size = 1 then get_2d output j col
            else get_3d output batch_b j col
          in
          sum := Dtype.sub dtype !sum (Dtype.mul dtype a_ji x_j)
        done;
        let a_ii =
          if unit_diag then Dtype.one dtype
          else if batch_size_a = 1 then get_2d a i i
          else get_3d a batch_a i i
        in
        let x_i = Dtype.div dtype !sum a_ii in
        if batch_size = 1 then set_2d output i col x_i
        else set_3d output batch_b i col x_i
      done
  done

(* Triangular solve using forward/backward substitution *)
let triangular_solve (type a b) ~upper ~transpose ~unit_diag ctx (a : (a, b) t)
    (b : (a, b) t) =
  let pool = ctx.pool in
  let batch_size_a, n, m = get_matrix_dims a in
  if n <> m then failwith "op_triangular_solve: matrix A must be square";

  (* Handle 1D input b by expanding to 2D *)
  let b_shape = View.shape b.view in
  let b_ndim = Array.length b_shape in
  let b_is_1d = b_ndim = 1 in

  let b_expanded =
    if b_is_1d then
      (* Expand 1D to 2D by adding a trailing dimension *)
      let new_shape = [| b_shape.(0); 1 |] in
      { b with view = View.reshape b.view new_shape }
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
  let output_shape = View.shape b_expanded.view in
  let output = create_output ctx b.dtype output_shape in

  (* Copy b to output *)
  Bigarray.Array1.blit b_expanded.buffer output.buffer;

  (* Process batches in parallel if batch_size > 1 *)
  (if batch_size > 1 then
     Parallel.parallel_for pool 0 (batch_size - 1) (fun batch_start batch_end ->
         for batch = batch_start to batch_end - 1 do
           let batch_a = if batch_size_a = 1 then 0 else batch in
           let batch_b = if batch_size_b = 1 then 0 else batch in
           kernel_triangular_solve_single ~upper ~transpose ~unit_diag a.dtype a
             output n nb batch_size_a batch_size batch_a batch_b
         done)
   else
     let batch_a = 0 in
     let batch_b = 0 in
     kernel_triangular_solve_single ~upper ~transpose ~unit_diag a.dtype a
       output n nb batch_size_a batch_size batch_a batch_b);

  (* Reshape output to match input b shape *)
  if b_is_1d then
    { output with view = View.reshape output.view [| b_shape.(0) |] }
  else output
