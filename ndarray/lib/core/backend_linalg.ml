open Descriptor
open Views

module Make (B : Backend_intf.S) = struct
  module Transform = Backend_transform.Make (B)

  let zeros context dtype shape =
    let t = B.empty context dtype shape in
    B.fill context (zero dtype) t;
    t

  let matmul context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let a_shape = shape a_desc in
    let b_shape = shape b_desc in
    let a_ndim = Array.length a_shape in
    let b_ndim = Array.length b_shape in

    if a_ndim = 0 || b_ndim = 0 then
      invalid_arg "matmul: inputs must not be scalars (0D)";

    let a_is_1d = a_ndim = 1 in
    let b_is_1d = b_ndim = 1 in

    let a_op =
      if a_is_1d then Transform.reshape context [| 1; a_shape.(0) |] a else a
    in
    let b_op =
      if b_is_1d then Transform.reshape context [| b_shape.(0); 1 |] b else b
    in

    let a_op_shape = shape (B.descriptor a_op) in
    let b_op_shape = shape (B.descriptor b_op) in
    let a_op_ndim = Array.length a_op_shape in
    let b_op_ndim = Array.length b_op_shape in

    let m = a_op_shape.(a_op_ndim - 2) in
    let k1 = a_op_shape.(a_op_ndim - 1) in
    let k2 = b_op_shape.(b_op_ndim - 2) in
    let n = b_op_shape.(b_op_ndim - 1) in

    if k1 <> k2 then
      invalid_arg
        (Format.asprintf
           "matmul: incompatible shapes for matrix multiplication (%a vs %a -> \
            inner dimensions %d vs %d mismatch)"
           pp_int_array a_shape pp_int_array b_shape k1 k2);

    let a_batch_shape = Array.sub a_op_shape 0 (Stdlib.max 0 (a_op_ndim - 2)) in
    let b_batch_shape = Array.sub b_op_shape 0 (Stdlib.max 0 (b_op_ndim - 2)) in

    let out_batch_shape =
      try broadcast_shapes a_batch_shape b_batch_shape
      with Failure msg ->
        invalid_arg (Printf.sprintf "matmul: broadcast error: %s" msg)
    in

    let out_core_shape = [| m; n |] in
    let out_shape = Array.append out_batch_shape out_core_shape in
    let num_batch_dims = Array.length out_batch_shape in

    let final_output_shape =
      match (a_is_1d, b_is_1d) with
      | true, true -> out_batch_shape
      | true, false -> Array.append out_batch_shape [| n |]
      | false, true -> Array.append out_batch_shape [| m |]
      | false, false -> out_shape
    in

    if Array.exists (( = ) 0) out_shape then
      zeros context (dtype a_desc) final_output_shape
    else
      let c = zeros context (dtype a_desc) out_shape in
      B.matmul context a_op b_op c;
      let c_desc = B.descriptor c in

      let final_c_desc =
        match (a_is_1d, b_is_1d) with
        | true, true ->
            squeeze ~axes:[| num_batch_dims; num_batch_dims + 1 |] c_desc
        | true, false -> squeeze ~axes:[| num_batch_dims |] c_desc
        | false, true -> squeeze ~axes:[| num_batch_dims + 1 |] c_desc
        | false, false -> c_desc
      in
      B.view final_c_desc c

  let dot context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let a_shape = shape a_desc in
    let b_shape = shape b_desc in
    let a_ndim = Array.length a_shape in
    let b_ndim = Array.length b_shape in

    if a_ndim = 0 || b_ndim = 0 then
      invalid_arg
        "dot: scalar (0D) inputs not supported by this implementation \
         (requires element-wise multiply)"
    else if a_ndim = 1 && b_ndim = 1 then (
      let k1 = a_shape.(0) in
      let k2 = b_shape.(0) in
      if k1 <> k2 then
        invalid_arg
          (Printf.sprintf "dot: 1D vectors have incompatible lengths %d and %d"
             k1 k2);

      if k1 = 0 then
        (* Handle empty vector case: dot product is 0 *)
        zeros context (dtype a_desc) [||]
      else
        let a_row = Transform.reshape context [| 1; k1 |] a in
        let b_col = Transform.reshape context [| k1; 1 |] b in
        (* Perform matmul: (1, k1) @ (k1, 1) -> (1, 1) *)
        let result_1x1 = matmul context a_row b_col in
        let result_desc = squeeze (B.descriptor result_1x1) ~axes:[| 0; 1 |] in
        B.view result_desc result_1x1)
    else if a_ndim = 2 && b_ndim = 2 then matmul context a b
    else if b_ndim = 1 then (
      let k1 = a_shape.(a_ndim - 1) in
      let k2 = b_shape.(0) in
      if k1 <> k2 then
        invalid_arg
          (Printf.sprintf
             "dot: N-D array last dimension %d doesn't match 1-D array length \
              %d"
             k1 k2);
      let b_mat = Transform.reshape context [| k2; 1 |] b in
      let c_mat = matmul context a b_mat in
      let c_desc = B.descriptor c_mat in
      let c_mat_ndim = Array.length (shape c_desc) in
      if c_mat_ndim > 0 then
        let new_desc = squeeze c_desc ~axes:[| c_mat_ndim - 1 |] in
        B.view new_desc c_mat
      else c_mat)
    else
      let k1 = a_shape.(a_ndim - 1) in
      let k2 = b_shape.(b_ndim - 2) in
      if k1 <> k2 then
        invalid_arg
          (Printf.sprintf
             "dot: last dim of N-D array (%d) doesn't match second-to-last dim \
              of M-D array (%d)"
             k1 k2);

      let a_core_dims =
        [| Array.fold_left ( * ) 1 (Array.sub a_shape 0 (a_ndim - 1)); k1 |]
      in
      let a_reshaped = Transform.reshape context a_core_dims a in

      let b_pre_dims = Array.sub b_shape 0 (b_ndim - 2) in
      let b_last_dim = b_shape.(b_ndim - 1) in
      let b_core_dims =
        [| k2; Array.fold_left ( * ) 1 b_pre_dims * b_last_dim |]
      in

      let b_perm = Array.make b_ndim 0 in
      b_perm.(0) <- b_ndim - 2;
      (* k2 dimension first *)
      let current_idx = ref 1 in
      for i = 0 to b_ndim - 2 - 1 do
        (* Dimensions before k2 *)
        b_perm.(!current_idx) <- i;
        incr current_idx
      done;
      b_perm.(!current_idx) <- b_ndim - 1;

      (* Last dimension *)
      let b_transposed =
        let new_desc = transpose ~axes:b_perm b_desc in
        B.view new_desc b
      in
      let b_reshaped = Transform.reshape context b_core_dims b_transposed in

      let c_reshaped = matmul context a_reshaped b_reshaped in

      let final_shape_list =
        Array.to_list (Array.sub a_shape 0 (a_ndim - 1))
        @ Array.to_list b_pre_dims @ [ b_last_dim ]
      in
      let final_shape = Array.of_list final_shape_list in

      Transform.reshape context final_shape c_reshaped

  let convolve1d_inner mode a v out =
    let ad = B.descriptor a in
    let vd = B.descriptor v in
    let n = (shape ad).(0) in
    let m = (shape vd).(0) in
    let dt = dtype ad in
    let zero_val = zero dt in

    (* Assume n > 0, m > 0, and output length >= 0 based on checks in Ops *)
    let out_len =
      match mode with `Full -> n + m - 1 | `Valid -> n - m + 1 | `Same -> n
    in

    let out_buf = B.buffer out in
    let a_buf = B.buffer a in
    let v_buf = B.buffer v in
    let a_stride = (strides ad).(0) in
    let v_stride = (strides vd).(0) in
    let a_offset = offset ad in
    let v_offset = offset vd in

    let convolve_loop () =
      let k_start, k_end =
        match mode with
        | `Full -> (0, n + m - 2)
        | `Valid -> (0, n - m) (* Should not be called for Valid *)
        | `Same -> (0, n - 1)
      in

      let _out_offset, physical_k_adjust =
        match mode with
        | `Full -> (0, 0)
        | `Valid -> (0, 0) (* Should not be called for Valid *)
        | `Same ->
            let start = (m - 1) / 2 in
            (start, -start)
      in

      for physical_k = k_start to k_end do
        let current_sum = ref zero_val in
        for j = 0 to m - 1 do
          let i = physical_k - j in
          if i >= 0 && i < n then
            let a_val =
              Bigarray.Array1.unsafe_get a_buf (a_offset + (i * a_stride))
            in
            let v_val =
              Bigarray.Array1.unsafe_get v_buf (v_offset + (j * v_stride))
            in
            current_sum := add_dtype dt !current_sum (mul_dtype dt a_val v_val)
        done;

        let logical_out_k = physical_k + physical_k_adjust in
        (* Boundary check remains necessary based on how indices are
           calculated *)
        if logical_out_k >= 0 && logical_out_k < out_len then
          Bigarray.Array1.unsafe_set out_buf logical_out_k !current_sum
      done
    in

    let convolve_valid_loop () =
      for k = 0 to out_len - 1 do
        let current_sum = ref zero_val in
        for j = 0 to m - 1 do
          let i = k + j in
          (* Bounds check on 'i' implicitly handled by loop range and n - m + 1
             length *)
          let a_val =
            Bigarray.Array1.unsafe_get a_buf (a_offset + (i * a_stride))
          in
          let v_val =
            Bigarray.Array1.unsafe_get v_buf (v_offset + (j * v_stride))
          in
          current_sum := add_dtype dt !current_sum (mul_dtype dt a_val v_val)
        done;
        Bigarray.Array1.unsafe_set out_buf k !current_sum
      done
    in

    if mode = `Valid then convolve_valid_loop () else convolve_loop ()

  let convolve1d context ?(mode = `Full) a v =
    let a_desc = B.descriptor a in
    let v_desc = B.descriptor v in
    let a_ndim = Array.length (shape a_desc) in
    let v_ndim = Array.length (shape v_desc) in
    if a_ndim <> 1 || v_ndim <> 1 then
      invalid_arg
        (Format.asprintf
           "convolve1d: Inputs must be 1-dimensional arrays, got shapes %a and \
            %a"
           pp_int_array (shape a_desc) pp_int_array (shape v_desc));
    let n = (shape a_desc).(0) in
    let m = (shape v_desc).(0) in
    let dtype = dtype a_desc in
    if m = 0 then
      match mode with
      | `Full -> zeros context dtype [| Stdlib.max 0 (n - 1) |]
      | `Valid -> zeros context dtype [| n |]
      | `Same -> zeros context dtype [| n |]
    else if n = 0 then
      match mode with
      | `Full -> zeros context dtype [| Stdlib.max 0 (m - 1) |]
      | `Valid -> zeros context dtype [| 0 |]
      | `Same -> zeros context dtype [| 0 |]
    else
      let out_len =
        match mode with
        | `Full -> n + m - 1
        | `Valid -> Stdlib.max 0 (n - m + 1)
        | `Same -> n
      in
      let out_shape = [| out_len |] in
      let out = B.empty context dtype out_shape in
      convolve1d_inner mode a v out;
      out

  (* Read/write an element from our descriptor into an OCaml float matrix *)
  let get_elem buf offset strides i j =
    let idx =
      offset
      + (i * strides.(Array.length strides - 2))
      + (j * strides.(Array.length strides - 1))
    in
    Bigarray.Array1.get buf idx

  let set_elem buf offset strides i j v =
    let idx =
      offset
      + (i * strides.(Array.length strides - 2))
      + (j * strides.(Array.length strides - 1))
    in
    Bigarray.Array1.unsafe_set buf idx v

  (* Gaussian elimination + back‐substitution *)
  let inv_inner _context (t : (float, _) B.b_t) (out : (float, _) B.b_t) =
    let d = B.descriptor t in
    let shape = d.shape in
    if Array.length shape <> 2 || shape.(0) <> shape.(1) then
      invalid_arg "inv: input must be a square matrix"
    else
      let n = shape.(0) in
      let buf = B.buffer t in
      let obuf = B.buffer out in
      let od = B.descriptor out in

      (* copy t into a_work, and init out to identity *)
      let a_work =
        Array.init n (fun i ->
            Array.init n (fun j -> get_elem buf d.offset d.strides i j))
      in
      let inv_work =
        Array.init n (fun i ->
            Array.init n (fun j -> if i = j then 1.0 else 0.0))
      in

      (* forward elimination with partial pivoting *)
      for k = 0 to n - 1 do
        (* find pivot row *)
        let pivot, piv_row =
          Array.fold_left
            (fun (best_v, best_r) r ->
              let v = abs_float a_work.(r).(k) in
              if v > best_v then (v, r) else (best_v, best_r))
            (0.0, k)
            (Array.init (n - k) (fun i -> i + k))
        in
        if pivot = 0. then invalid_arg "inv: matrix is singular";
        (* swap rows k and piv_row in both a_work and inv_work *)
        if piv_row <> k then (
          let tmp = a_work.(k) in
          a_work.(k) <- a_work.(piv_row);
          a_work.(piv_row) <- tmp;
          let tmp2 = inv_work.(k) in
          inv_work.(k) <- inv_work.(piv_row);
          inv_work.(piv_row) <- tmp2);
        (* normalize pivot row *)
        let diag = a_work.(k).(k) in
        for j = k to n - 1 do
          a_work.(k).(j) <- a_work.(k).(j) /. diag
        done;
        for j = 0 to n - 1 do
          inv_work.(k).(j) <- inv_work.(k).(j) /. diag
        done;
        (* eliminate below *)
        for i = k + 1 to n - 1 do
          let factor = a_work.(i).(k) in
          for j = k to n - 1 do
            a_work.(i).(j) <- a_work.(i).(j) -. (factor *. a_work.(k).(j))
          done;
          for j = 0 to n - 1 do
            inv_work.(i).(j) <- inv_work.(i).(j) -. (factor *. inv_work.(k).(j))
          done
        done
      done;

      (* back-substitution to clear above-diagonal *)
      for k = n - 1 downto 0 do
        for i = 0 to k - 1 do
          let factor = a_work.(i).(k) in
          for j = 0 to n - 1 do
            inv_work.(i).(j) <- inv_work.(i).(j) -. (factor *. inv_work.(k).(j))
          done
        done
      done;

      (* write back into out buffer *)
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          set_elem obuf od.offset od.strides i j inv_work.(i).(j)
        done
      done

  let inv context (a : (float, _) B.b_t) =
    let ad = B.descriptor a in
    let shape = ad.shape in
    if Array.length shape <> 2 || shape.(0) <> shape.(1) then
      invalid_arg "inv: input must be a square matrix";
    let out = zeros context (dtype ad) shape in
    inv_inner context a out;
    out

  (* Pure-OCaml QR decomposition for general real matrices *)
  let qr_decomp a_mat =
    let n = Array.length a_mat in
    (* a_mat: row-major 2D float array *)
    let q = Array.init n (fun _ -> Array.make n 0.) in
    let r = Array.init n (fun _ -> Array.make n 0.) in
    for k = 0 to n - 1 do
      (* Copy column k of a_mat into v *)
      let v = Array.init n (fun i -> a_mat.(i).(k)) in
      (* Orthogonalize against previous q-columns *)
      for i = 0 to k - 1 do
        (* dot = q[:,i] \cdot v *)
        let dot = ref 0. in
        for j = 0 to n - 1 do
          dot := !dot +. (q.(j).(i) *. v.(j))
        done;
        r.(i).(k) <- !dot;
        (* v := v - dot * q[:,i] *)
        for j = 0 to n - 1 do
          v.(j) <- v.(j) -. (!dot *. q.(j).(i))
        done
      done;
      (* Normalize v to get q[:,k] *)
      let norm = sqrt (Array.fold_left (fun acc x -> acc +. (x *. x)) 0. v) in
      if norm = 0. then invalid_arg "qr_decomp: rank deficient";
      r.(k).(k) <- norm;
      for j = 0 to n - 1 do
        q.(j).(k) <- v.(j) /. norm
      done
    done;
    (q, r)

  (* Matrix-matrix multiply: c = a * b *)
  let matmul_local a b =
    let n = Array.length a in
    let c = Array.init n (fun _ -> Array.make n 0.) in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let sum = ref 0. in
        for k = 0 to n - 1 do
          sum := !sum +. (a.(i).(k) *. b.(k).(j))
        done;
        c.(i).(j) <- !sum
      done
    done;
    c

  (* Eigen-decomposition for general (possibly non-symmetric) via unshifted
     QR *)
  let eig_inner (t : (float, 'b) B.b_t) (w : (float, 'b) B.b_t)
      (vr : (float, 'b) B.b_t) =
    let desc = B.descriptor t in
    let shape = desc.shape in
    if Array.length shape <> 2 || shape.(0) <> shape.(1) then
      invalid_arg "eig: input must be a square matrix";
    let n = shape.(0) in
    (* Load into local 2D array *)
    let a_mat =
      Array.init n (fun i ->
          Array.init n (fun j ->
              Bigarray.Array1.unsafe_get (B.buffer t)
                (desc.offset + (i * desc.strides.(0)) + (j * desc.strides.(1)))))
    in
    (* Initialize Q_total = identity *)
    let q_tot =
      Array.init n (fun i -> Array.init n (fun j -> if i = j then 1. else 0.))
    in
    let max_iter = 1000 in
    for _ = 1 to max_iter do
      let q, r = qr_decomp a_mat in
      (* A := R * Q *)
      let a_next = matmul_local r q in
      (* Q_tot := Q_tot * Q *)
      let qtot_next = matmul_local q_tot q in
      (* copy back *)
      for i = 0 to n - 1 do
        Array.blit a_next.(i) 0 a_mat.(i) 0 n
      done;
      for i = 0 to n - 1 do
        Array.blit qtot_next.(i) 0 q_tot.(i) 0 n
      done
    done;
    (* Write eigenvalues and eigenvectors *)
    let w_desc = B.descriptor w in
    let vr_desc = B.descriptor vr in
    for i = 0 to n - 1 do
      (* eigenvalue = A[i,i] *)
      Bigarray.Array1.unsafe_set (B.buffer w) (w_desc.offset + i) a_mat.(i).(i);
      (* eigenvector = column i of Q_tot *)
      for j = 0 to n - 1 do
        Bigarray.Array1.unsafe_set (B.buffer vr)
          (vr_desc.offset
          + (j * vr_desc.strides.(0))
          + (i * vr_desc.strides.(1)))
          q_tot.(j).(i)
      done
    done

  let eig context (a : (float, _) B.b_t) =
    let ad = B.descriptor a in
    let shape = ad.shape in
    if Array.length shape <> 2 || shape.(0) <> shape.(1) then
      invalid_arg "eig: input must be a square matrix";
    let n = shape.(0) in
    let w = B.empty context ad.dtype [| n |] in
    let vr = B.empty context ad.dtype [| n; n |] in
    eig_inner a w vr;
    (w, vr)

  (* Symmetric eigen-decomposition via Jacobi rotations *)
  let eigh_inner _context (t : (float, 'b) B.b_t) (w : (float, 'b) B.b_t)
      (vr : (float, 'b) B.b_t) =
    let desc = B.descriptor t in
    let shape = desc.shape in
    if Array.length shape <> 2 || shape.(0) <> shape.(1) then
      invalid_arg "eigh: input must be a square matrix";
    let n = shape.(0) in
    (* Load symmetric matrix and init V=I *)
    let a_mat =
      Array.init n (fun i ->
          Array.init n (fun j ->
              Bigarray.Array1.unsafe_get (B.buffer t)
                (desc.offset + (i * desc.strides.(0)) + (j * desc.strides.(1)))))
    in
    let v_mat =
      Array.init n (fun i -> Array.init n (fun j -> if i = j then 1. else 0.))
    in
    let tol = 1e-12 in
    let max_sweeps = n * 10 in
    let () =
      try
        for _ = 1 to max_sweeps do
          let max_off = ref 0. in
          for p = 0 to n - 2 do
            for q = p + 1 to n - 1 do
              let app = a_mat.(p).(p) in
              let aqq = a_mat.(q).(q) in
              let apq = a_mat.(p).(q) in
              max_off := max !max_off (abs_float apq);
              if abs_float apq > tol then (
                let phi = 0.5 *. atan2 (2. *. apq) (aqq -. app) in
                let c = cos phi and s = sin phi in
                (* Rotate rows p,q *)
                for i = 0 to n - 1 do
                  let aip = a_mat.(i).(p) and aiq = a_mat.(i).(q) in
                  a_mat.(i).(p) <- (c *. aip) -. (s *. aiq);
                  a_mat.(i).(q) <- (s *. aip) +. (c *. aiq)
                done;
                (* Rotate cols p,q *)
                for j = 0 to n - 1 do
                  let apj = a_mat.(p).(j) and aqj = a_mat.(q).(j) in
                  a_mat.(p).(j) <- (c *. apj) -. (s *. aqj);
                  a_mat.(q).(j) <- (s *. apj) +. (c *. aqj)
                done;
                (* Accumulate V *)
                for i = 0 to n - 1 do
                  let vip = v_mat.(i).(p) and viq = v_mat.(i).(q) in
                  v_mat.(i).(p) <- (c *. vip) -. (s *. viq);
                  v_mat.(i).(q) <- (s *. vip) +. (c *. viq)
                done)
            done
          done;
          if !max_off < tol then raise Exit
        done
      with Exit -> ()
    in
    (* Write back eigenvalues and vectors *)
    let w_desc = B.descriptor w in
    let vr_desc = B.descriptor vr in
    for i = 0 to n - 1 do
      Bigarray.Array1.unsafe_set (B.buffer w) (w_desc.offset + i) a_mat.(i).(i);
      for j = 0 to n - 1 do
        Bigarray.Array1.unsafe_set (B.buffer vr)
          (vr_desc.offset
          + (j * vr_desc.strides.(0))
          + (i * vr_desc.strides.(1)))
          v_mat.(j).(i)
      done
    done

  (* Top‑level symmetric eigendecomposition *)
  let eigh context (a : (float, _) B.b_t) =
    let ad = B.descriptor a in
    let shape = ad.shape in
    if Array.length shape <> 2 || shape.(0) <> shape.(1) then
      invalid_arg "eigh: input must be a square matrix";
    let n = shape.(0) in
    let w = B.empty context ad.dtype [| n |] in
    let vr = B.empty context ad.dtype [| n; n |] in
    eigh_inner context a w vr;
    (w, vr)

  let svd_inner context (t : (float, _) B.b_t) (u : (float, _) B.b_t)
      (s : (float, _) B.b_t) (v : (float, _) B.b_t) =
    (* 1) only float64, only 2‑D *)
    let td = B.descriptor t in
    let shape_t = td.shape in
    if Array.length shape_t <> 2 then invalid_arg "svd: input must be 2D";
    let m, n = (shape_t.(0), shape_t.(1)) in

    (* 2) length of s must be ≤ min(m,n) *)
    let sd = B.descriptor s in
    if Array.length sd.shape <> 1 then invalid_arg "svd: s must be 1D";
    let r = sd.shape.(0) in
    if r > min m n then invalid_arg "svd: length of s cannot exceed min(m,n)";

    (* 3) build A^T A in a temp tensor ata *)
    let ata = B.empty context Float64 [| n; n |] in
    let atad = B.descriptor ata in
    let ata_buf = B.buffer ata in
    let ata_off = offset atad in
    let ata_str = strides atad in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let sum = ref 0. in
        for k = 0 to m - 1 do
          let a_ki = get_elem (B.buffer t) (offset td) (strides td) k i in
          let a_kj = get_elem (B.buffer t) (offset td) (strides td) k j in
          sum := !sum +. (a_ki *. a_kj)
        done;
        set_elem ata_buf ata_off ata_str i j !sum
      done
    done;

    (* 4) eigen‑decompose ata = V_full Λ V_full^T *)
    let w_full = B.empty context Float64 [| n |] in
    let w_desc = B.descriptor w_full in
    let v_full = B.empty context Float64 [| n; n |] in
    let v_desc = B.descriptor v_full in
    eigh_inner context ata w_full v_full;

    (* 5) singular values = sqrt(eigenvalues) *)
    let s_buf = B.buffer s in
    let s_off = sd.offset in
    for i = 0 to r - 1 do
      let _l = Bigarray.Array1.unsafe_get s_buf (s_off + i) in
      (* w_full.buf holds λ_i on its diagonal *)
      let l =
        Bigarray.Array1.unsafe_get (B.buffer w_full) (offset w_desc + i)
      in
      Bigarray.Array1.unsafe_set s_buf (s_off + i) (sqrt l)
    done;

    (* 6) build U(:,0:r) = A * V_full(:,0:r) * Σ^{-1} *)
    let ud = B.descriptor u in
    let u_buf = B.buffer u in
    let u_off = ud.offset in
    let u_str0, u_str1 = (ud.strides.(0), ud.strides.(1)) in
    for i = 0 to m - 1 do
      for j = 0 to ud.shape.(1) - 1 do
        if j < r then (
          let dot = ref 0. in
          for k = 0 to n - 1 do
            let a_ik = get_elem (B.buffer t) (offset td) (strides td) i k in
            let v_kj =
              get_elem (B.buffer v_full) (offset v_desc) (strides v_desc) k j
            in
            dot := !dot +. (a_ik *. v_kj)
          done;
          let o = Bigarray.Array1.unsafe_get s_buf (s_off + j) in
          let v = if o = 0. then 0. else !dot /. o in
          Bigarray.Array1.unsafe_set u_buf
            (u_off + (i * u_str0) + (j * u_str1))
            v)
        else
          Bigarray.Array1.unsafe_set u_buf
            (u_off + (i * u_str0) + (j * u_str1))
            0.
      done
    done;

    (* 7) write out V (either full or economy) *)
    let vd = B.descriptor v in
    let v_buf = B.buffer v in
    let v_off = vd.offset in
    let v_str0, v_str1 = (vd.strides.(0), vd.strides.(1)) in
    let vn = vd.shape.(1) in
    if vd.shape.(0) <> n then invalid_arg "svd: v must have n rows";
    if vn = n then
      (* full V <- v_full *)
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let x =
            get_elem (B.buffer v_full) (offset v_desc) (strides v_desc) i j
          in
          Bigarray.Array1.unsafe_set v_buf
            (v_off + (i * v_str0) + (j * v_str1))
            x
        done
      done
    else if vn >= r then
      (* economy: only first r cols of V *)
      for i = 0 to n - 1 do
        for j = 0 to vn - 1 do
          let x =
            if j < r then
              get_elem (B.buffer v_full) (offset v_desc) (strides v_desc) i j
            else 0.
          in
          Bigarray.Array1.unsafe_set v_buf
            (v_off + (i * v_str0) + (j * v_str1))
            x
        done
      done
    else invalid_arg "svd: v must have at least as many columns as s"

  let svd context (t : (float, _) B.b_t) =
    let td = B.descriptor t in
    let shape = td.shape in
    if Array.length shape <> 2 then invalid_arg "svd: input must be 2D";
    let m = shape.(0) and n = shape.(1) in
    let r = min m n in
    (* full U has shape m×r, full S has length r, full V has shape n×n *)
    let u = B.empty context td.dtype [| m; r |] in
    let s = B.empty context td.dtype [| r |] in
    let v = B.empty context td.dtype [| n; n |] in
    svd_inner context t u s v;
    (u, s, v)

  let solve context (a : (float, _) B.b_t) (b : (float, _) B.b_t) =
    let ad = B.descriptor a in
    let bd = B.descriptor b in
    let shape_a = ad.shape in
    let shape_b = shape bd in
    if Array.length shape_a <> 2 then invalid_arg "solve: A must be 2D";
    let b_op =
      if Array.length shape_b = 1 then
        Transform.reshape context [| shape_b.(0); 1 |] b
      else b
    in
    let u, s, v = svd context a in
    let u_t = Transform.transpose context ~axes:[| 1; 0 |] u in
    let c = matmul context u_t b_op in
    let dt = dtype (B.descriptor s) in
    let r = (shape (B.descriptor s)).(0) in
    let ones = B.empty context dt [| r |] in
    B.fill context (one dt) ones;
    let s_inv = B.empty context dt [| r |] in
    B.div context ones s s_inv;
    let c_desc = B.descriptor c in
    let c_shape = shape c_desc in
    let k = c_shape.(1) in
    let c_scaled = B.empty context dt c_shape in
    let c_buf = B.buffer c in
    let c_offset = c_desc.offset in
    let c_strides = c_desc.strides in
    let s_inv_buf = B.buffer s_inv in
    let s_inv_desc = B.descriptor s_inv in
    let s_inv_offset = s_inv_desc.offset in
    let s_inv_strides = s_inv_desc.strides in
    let c_scaled_buf = B.buffer c_scaled in
    let c_scaled_desc = B.descriptor c_scaled in
    let c_scaled_offset = c_scaled_desc.offset in
    let c_scaled_strides = c_scaled_desc.strides in
    for i = 0 to r - 1 do
      let s_inv_i =
        Bigarray.Array1.unsafe_get s_inv_buf
          (s_inv_offset + (i * s_inv_strides.(0)))
      in
      for j = 0 to k - 1 do
        let c_ij =
          Bigarray.Array1.unsafe_get c_buf
            (c_offset + (i * c_strides.(0)) + (j * c_strides.(1)))
        in
        let value = s_inv_i *. c_ij in
        Bigarray.Array1.unsafe_set c_scaled_buf
          (c_scaled_offset
          + (i * c_scaled_strides.(0))
          + (j * c_scaled_strides.(1)))
          value
      done
    done;
    let v_desc = B.descriptor v in
    let n_v = v_desc.shape.(0) in
    let v_red = Transform.slice context [| 0; 0 |] [| n_v; r |] v in
    let x = matmul context v_red c_scaled in
    if Array.length shape_b = 1 then
      let x_desc = B.descriptor x in
      let x_shape = shape x_desc in
      if Array.length x_shape = 2 && x_shape.(1) = 1 then
        Transform.reshape context [| x_shape.(0) |] x
      else x
    else x
end
