open Bigarray_ext
module Dtype = Nx_core.Dtype
module Shape = Nx_core.Shape
module View = Nx_core.View
module Lazy_view = Nx_core.Lazy_view
module Symbolic_shape = Nx_core.Symbolic_shape
module Error = Nx_core.Error
open Internal

(* Helper to get concrete shape from view *)
let get_shape view =
  match Symbolic_shape.eval (Lazy_view.shape view) with
  | Some arr -> arr
  | None ->
      Error.failed ~op:"get_shape" ~what:"cannot evaluate symbolic shape" ()

(* Helper to get strides from view *)
let get_strides view =
  match Lazy_view.strides view with
  | Some s -> s
  | None ->
      Error.failed ~op:"get_strides"
        ~what:"cannot get strides for non-contiguous view" ()

(* Helper to get offset from view *)
let get_offset view =
  match Symbolic_shape.eval_dim (Lazy_view.offset view) with
  | Some n -> n
  | None ->
      Error.failed ~op:"get_offset" ~what:"cannot evaluate symbolic offset" ()

(* Helper to calculate batch offset *)
let batch_offset b shape strides =
  let ndim = Array.length shape in
  if ndim <= 2 then 0
  else
    let offset = ref 0 in
    let remaining = ref b in
    for i = ndim - 3 downto 0 do
      let dim_size = shape.(i) in
      let coord = !remaining mod dim_size in
      remaining := !remaining / dim_size;
      offset := !offset + (coord * strides.(i))
    done;
    !offset

module Float_ops = struct
  let epsilon32 = 1.192092896e-07
  let epsilon64 = epsilon_float
  let sign x = if x >= 0.0 then 1.0 else -1.0

  let hypot a b =
    let absa = abs_float a in
    let absb = abs_float b in
    if absa > absb then
      let ratio = absb /. absa in
      absa *. sqrt (1.0 +. (ratio *. ratio))
    else if absb > 0.0 then
      let ratio = absa /. absb in
      absb *. sqrt (1.0 +. (ratio *. ratio))
    else 0.0

  let givens a b =
    if b = 0.0 then (1.0, 0.0)
    else if abs_float b > abs_float a then
      let t = a /. b in
      let sign_b = sign b in
      let s = sign_b /. sqrt (1.0 +. (t *. t)) in
      let c = s *. t in
      (c, s)
    else
      let t = b /. a in
      let sign_a = sign a in
      let c = sign_a /. sqrt (1.0 +. (t *. t)) in
      let s = c *. t in
      (c, s)

  let apply_givens_left pool a _m n i j c s =
    let loop start_k end_k =
      for k = start_k to end_k - 1 do
        let temp = (c *. a.{(i * n) + k}) +. (s *. a.{(j * n) + k}) in
        a.{(j * n) + k} <- (-.s *. a.{(i * n) + k}) +. (c *. a.{(j * n) + k});
        a.{(i * n) + k} <- temp
      done
    in
    if n > 100 then Parallel.parallel_for pool 0 (n - 1) loop else loop 0 n

  let apply_givens_right pool a m n i j c s =
    let loop start_k end_k =
      for k = start_k to end_k - 1 do
        let temp = (c *. a.{(k * n) + i}) +. (s *. a.{(k * n) + j}) in
        a.{(k * n) + j} <- (-.s *. a.{(k * n) + i}) +. (c *. a.{(k * n) + j});
        a.{(k * n) + i} <- temp
      done
    in
    if m > 100 then Parallel.parallel_for pool 0 (m - 1) loop else loop 0 m
end

module Complex_ops = struct
  include Complex

  let cabs = norm

  let sign x =
    let mag = norm x in
    if mag = 0.0 then one else div x { re = mag; im = 0.0 }

  let hypot a b = norm a +. norm b

  let givens a b =
    let na = norm a in
    let nb = norm b in
    if nb = 0.0 then (1.0, zero)
    else if nb > na then
      let t = div a b in
      let s =
        mul
          (div one { re = Stdlib.sqrt (1.0 +. (norm t ** 2.0)); im = 0.0 })
          (sign b)
      in
      let c = (mul s t).re in
      (c, s)
    else
      let t = div b a in
      let c = 1.0 /. Stdlib.sqrt (1.0 +. (norm t ** 2.0)) in
      let s = mul { re = c; im = 0.0 } (mul t (sign a)) in
      (c, s)

  let apply_givens_left pool a _m n i j c s =
    let loop start_k end_k =
      for k = start_k to end_k - 1 do
        let a_ik = a.{(i * n) + k} in
        let a_jk = a.{(j * n) + k} in
        let temp = add (mul { re = c; im = 0.0 } a_ik) (mul s a_jk) in
        a.{(j * n) + k} <-
          add (mul (neg (conj s)) a_ik) (mul { re = c; im = 0.0 } a_jk);
        a.{(i * n) + k} <- temp
      done
    in
    if n > 100 then Parallel.parallel_for pool 0 (n - 1) loop else loop 0 n

  let apply_givens_right pool a m n i j c s =
    let loop start_k end_k =
      for k = start_k to end_k - 1 do
        let a_ki = a.{(k * n) + i} in
        let a_kj = a.{(k * n) + j} in
        let temp = add (mul { re = c; im = 0.0 } a_ki) (mul (conj s) a_kj) in
        a.{(k * n) + j} <-
          add (mul (neg s) a_ki) (mul { re = c; im = 0.0 } a_kj);
        a.{(k * n) + i} <- temp
      done
    in
    if m > 100 then Parallel.parallel_for pool 0 (m - 1) loop else loop 0 m
end

(* Bidiagonalize for real *)
let bidiagonalize_real pool a u v diag superdiag m n =
  let minmn = min m n in
  (* Debug: check matrix dimensions *)
  let a_size = Array1.dim a in
  if a_size < m * n then
    failwith
      (Printf.sprintf
         "bidiagonalize_real: matrix size mismatch, expected %d elements \
          (m=%d, n=%d), got %d"
         (m * n) m n a_size);
  let init_loop size mat =
    if size > 100 then
      Parallel.parallel_for pool 0 (size - 1) (fun start_i end_i ->
          for i = start_i to end_i - 1 do
            for j = 0 to size - 1 do
              mat.{(i * size) + j} <- (if i = j then 1.0 else 0.0)
            done
          done)
    else
      for i = 0 to size - 1 do
        for j = 0 to size - 1 do
          mat.{(i * size) + j} <- (if i = j then 1.0 else 0.0)
        done
      done
  in
  init_loop m u;
  init_loop n v;
  for p = 0 to minmn - 1 do
    (* if m = 2 && n = 2 then Printf.printf "=== Processing column p=%d
       (minmn=%d) ===\n" p minmn; *)
    let norm2 = ref 0.0 in
    for i = p to m - 1 do
      let x = a.{(i * n) + p} in
      norm2 := !norm2 +. (x *. x)
    done;
    (* if m = 2 && n = 2 && p = 1 then Printf.printf " p=%d, loop from %d to
       %d\n" p p (m-1); *)
    let norm = sqrt !norm2 in
    (* Only apply column Householder if there are elements below diagonal to zero out *)
    if norm > 0.0 && p < m - 1 then (
      let sign = Float_ops.sign a.{(p * n) + p} in
      let alpha = -.sign *. norm in
      a.{(p * n) + p} <- a.{(p * n) + p} -. alpha;
      let beta = -1.0 /. (alpha *. a.{(p * n) + p}) in
      let col_loop start_j end_j =
        for j = start_j to end_j - 1 do
          let gamma = ref 0.0 in
          for i = p to m - 1 do
            gamma := !gamma +. (a.{(i * n) + p} *. a.{(i * n) + j})
          done;
          let gamma = !gamma *. beta in
          for i = p to m - 1 do
            a.{(i * n) + j} <- a.{(i * n) + j} -. (gamma *. a.{(i * n) + p})
          done
        done
      in
      if n - p - 1 > 100 then
        Parallel.parallel_for pool (p + 1) (n - 1) col_loop
      else col_loop (p + 1) n;
      (* Apply Householder from right: U = U * H 
         H = I - beta * v * v^T
         U * H = U - beta * (U * v) * v^T
         For each row i: U[i,:] = U[i,:] - beta * (U[i,:] · v) * v^T *)
      for i = 0 to m - 1 do
        let dot_prod = ref 0.0 in
        for k = p to m - 1 do
          dot_prod := !dot_prod +. (u.{(i * m) + k} *. a.{(k * n) + p})
        done;
        let gamma = !dot_prod *. beta in
        for k = p to m - 1 do
          u.{(i * m) + k} <- u.{(i * m) + k} -. (gamma *. a.{(k * n) + p})
        done
      done;
      (* Store alpha as the diagonal element (the reduced value) *)
      diag.(p) <- alpha;
      a.{(p * n) + p} <- alpha)
    else
      (* No Householder needed, just store the diagonal element *)
      diag.(p) <- a.{(p * n) + p};
    if p < n - 1 then (
      let norm2r = ref 0.0 in
      for j = p + 1 to n - 1 do
        let x = a.{(p * n) + j} in
        norm2r := !norm2r +. (x *. x)
      done;
      let normr = sqrt !norm2r in
      if normr > 0.0 then (
        let sign = Float_ops.sign a.{(p * n) + p + 1} in
        let alphar = -.sign *. normr in
        a.{(p * n) + p + 1} <- a.{(p * n) + p + 1} -. alphar;
        let beta = -1.0 /. (alphar *. a.{(p * n) + p + 1}) in
        let row_loop start_i end_i =
          for i = start_i to end_i - 1 do
            let gamma = ref 0.0 in
            for j = p + 1 to n - 1 do
              gamma := !gamma +. (a.{(i * n) + j} *. a.{(p * n) + j})
            done;
            let gamma = !gamma *. beta in
            for j = p + 1 to n - 1 do
              a.{(i * n) + j} <- a.{(i * n) + j} -. (gamma *. a.{(p * n) + j})
            done
          done
        in
        if m - p - 1 > 100 then
          Parallel.parallel_for pool (p + 1) (m - 1) row_loop
        else row_loop (p + 1) m;
        (* Apply Householder from right to V: V = V * H For each row i: V[i,:] =
           V[i,:] - beta * (V[i,:] · v) * v^T *)
        let v_loop start_i end_i =
          for i = start_i to end_i - 1 do
            let dot_prod = ref 0.0 in
            for k = p + 1 to n - 1 do
              dot_prod := !dot_prod +. (v.{(i * n) + k} *. a.{(p * n) + k})
            done;
            let gamma = !dot_prod *. beta in
            for k = p + 1 to n - 1 do
              v.{(i * n) + k} <- v.{(i * n) + k} -. (gamma *. a.{(p * n) + k})
            done
          done
        in
        if n > 100 then Parallel.parallel_for pool 0 (n - 1) v_loop
        else v_loop 0 n;
        (* Store alphar as the superdiagonal element (the reduced value) *)
        superdiag.(p) <- (if p < minmn - 1 then alphar else 0.0);
        a.{(p * n) + p + 1} <- (if p < minmn - 1 then alphar else 0.0))
      else superdiag.(p) <- 0.0)
  done

(* Bidiagonalize for complex *)
let bidiagonalize_complex pool a u v diag superdiag m n =
  let minmn = min m n in
  let init_loop size mat =
    if size > 100 then
      Parallel.parallel_for pool 0 (size - 1) (fun start_i end_i ->
          for i = start_i to end_i - 1 do
            for j = 0 to size - 1 do
              mat.{(i * size) + j} <-
                (if i = j then Complex_ops.one else Complex_ops.zero)
            done
          done)
    else
      for i = 0 to size - 1 do
        for j = 0 to size - 1 do
          mat.{(i * size) + j} <-
            (if i = j then Complex_ops.one else Complex_ops.zero)
        done
      done
  in
  init_loop m u;
  init_loop n v;
  for p = 0 to minmn - 1 do
    let norm2 = ref 0.0 in
    for i = p to m - 1 do
      norm2 := !norm2 +. Complex_ops.norm2 a.{(i * n) + p}
    done;
    let norm = sqrt !norm2 in
    if norm > 0.0 then (
      let a_pp = a.{(p * n) + p} in
      let phase =
        Complex_ops.div a_pp { re = Complex_ops.norm a_pp; im = 0.0 }
      in
      let alpha = Complex_ops.mul { re = -.norm; im = 0.0 } phase in
      a.{(p * n) + p} <- Complex_ops.sub a_pp alpha;
      let a_pp_new = a.{(p * n) + p} in
      let beta =
        -1.0 /. ((Complex_ops.mul (Complex_ops.conj alpha) a_pp_new).re /. norm)
      in
      let col_loop start_j end_j =
        for j = start_j to end_j - 1 do
          let gamma = ref Complex_ops.zero in
          for i = p to m - 1 do
            gamma :=
              Complex_ops.add !gamma
                (Complex_ops.mul
                   (Complex_ops.conj a.{(i * n) + p})
                   a.{(i * n) + j})
          done;
          let gamma = Complex_ops.mul !gamma { re = beta; im = 0.0 } in
          for i = p to m - 1 do
            a.{(i * n) + j} <-
              Complex_ops.sub
                a.{(i * n) + j}
                (Complex_ops.mul gamma a.{(i * n) + p})
          done
        done
      in
      if n - p - 1 > 100 then
        Parallel.parallel_for pool (p + 1) (n - 1) col_loop
      else col_loop (p + 1) n;
      let u_loop start_j end_j =
        for j = start_j to end_j - 1 do
          let gamma = ref Complex_ops.zero in
          for i = p to m - 1 do
            gamma :=
              Complex_ops.add !gamma
                (Complex_ops.mul
                   (Complex_ops.conj a.{(i * n) + p})
                   u.{(i * m) + j})
          done;
          let gamma = Complex_ops.mul !gamma { re = beta; im = 0.0 } in
          for i = p to m - 1 do
            u.{(i * m) + j} <-
              Complex_ops.sub
                u.{(i * m) + j}
                (Complex_ops.mul gamma a.{(i * n) + p})
          done
        done
      in
      if m > 100 then Parallel.parallel_for pool 0 (m - 1) u_loop
      else u_loop 0 m;
      (* Store the norm as the diagonal element *)
      diag.(p) <- norm)
    else
      (* No Householder needed, just store the diagonal element *)
      diag.(p) <- (if norm > 0.0 then norm else 0.0);
    a.{(p * n) + p} <- { re = diag.(p); im = 0.0 };
    if p < n - 1 then (
      let norm2r = ref 0.0 in
      for j = p + 1 to n - 1 do
        norm2r := !norm2r +. Complex_ops.norm2 a.{(p * n) + j}
      done;
      let normr = sqrt !norm2r in
      if normr > 0.0 then (
        let a_pq = a.{(p * n) + p + 1} in
        let phase =
          Complex_ops.div a_pq { re = Complex_ops.norm a_pq; im = 0.0 }
        in
        let alpha = Complex_ops.mul { re = -.normr; im = 0.0 } phase in
        a.{(p * n) + p + 1} <- Complex_ops.sub a_pq alpha;
        let a_pq_new = a.{(p * n) + p + 1} in
        let beta =
          -1.0
          /. ((Complex_ops.mul a_pq_new (Complex_ops.conj alpha)).re /. normr)
        in
        let row_loop start_i end_i =
          for i = start_i to end_i - 1 do
            let gamma = ref Complex_ops.zero in
            for j = p + 1 to n - 1 do
              gamma :=
                Complex_ops.add !gamma
                  (Complex_ops.mul
                     a.{(i * n) + j}
                     (Complex_ops.conj a.{(p * n) + j}))
            done;
            let gamma = Complex_ops.mul !gamma { re = beta; im = 0.0 } in
            for j = p + 1 to n - 1 do
              a.{(i * n) + j} <-
                Complex_ops.sub
                  a.{(i * n) + j}
                  (Complex_ops.mul gamma a.{(p * n) + j})
            done
          done
        in
        if m - p - 1 > 100 then
          Parallel.parallel_for pool (p + 1) (m - 1) row_loop
        else row_loop (p + 1) m;
        let v_loop start_j end_j =
          for j = start_j to end_j - 1 do
            let gamma = ref Complex_ops.zero in
            for t = p + 1 to n - 1 do
              gamma :=
                Complex_ops.add !gamma
                  (Complex_ops.mul
                     v.{(t * n) + j}
                     (Complex_ops.conj a.{(p * n) + t}))
            done;
            let gamma = Complex_ops.mul !gamma { re = beta; im = 0.0 } in
            for t = p + 1 to n - 1 do
              v.{(t * n) + j} <-
                Complex_ops.sub
                  v.{(t * n) + j}
                  (Complex_ops.mul gamma a.{(p * n) + t})
            done
          done
        in
        if n > 100 then Parallel.parallel_for pool 0 (n - 1) v_loop
        else v_loop 0 n;
        (* Store the norm as the superdiagonal element *)
        superdiag.(p) <- (if p < minmn - 1 then normr else 0.0);
        a.{(p * n) + p + 1} <- { re = superdiag.(p); im = 0.0 })
      else (
        superdiag.(p) <- 0.0;
        a.{(p * n) + p + 1} <- { re = 0.0; im = 0.0 }))
    else ()
  done

(* SVD QR iteration for real *)
let svd_qr_iteration_real pool diag superdiag u v m n p q =
  let d = (diag.(q - 1) -. diag.(q)) /. 2.0 in
  let shift =
    diag.(q)
    -. superdiag.(q - 1)
       *. superdiag.(q - 1)
       /. (d +. (Float_ops.sign d *. Float_ops.hypot d superdiag.(q - 1)))
  in
  let f = ref (diag.(p) -. shift) in
  let g = ref superdiag.(p) in
  for k = p to q - 1 do
    let c, s = Float_ops.givens !f !g in
    if k > p then superdiag.(k - 1) <- Float_ops.hypot !f !g;
    f := (c *. diag.(k)) +. (s *. superdiag.(k));
    superdiag.(k) <- (-.s *. diag.(k)) +. (c *. superdiag.(k));
    g := s *. diag.(k + 1);
    diag.(k + 1) <- c *. diag.(k + 1);
    Float_ops.apply_givens_right pool v n n k (k + 1) c s;
    let c, s = Float_ops.givens !f !g in
    diag.(k) <- Float_ops.hypot !f !g;
    f := (c *. superdiag.(k)) +. (s *. diag.(k + 1));
    diag.(k + 1) <- (-.s *. superdiag.(k)) +. (c *. diag.(k + 1));
    if k < q - 1 then (
      g := s *. superdiag.(k + 1);
      superdiag.(k + 1) <- c *. superdiag.(k + 1));
    Float_ops.apply_givens_left pool u m m k (k + 1) c s
  done;
  superdiag.(q - 1) <- !f

(* SVD QR iteration for complex *)
let svd_qr_iteration_complex pool diag superdiag u v m n p q =
  let d = (diag.(q - 1) -. diag.(q)) /. 2.0 in
  let shift =
    diag.(q)
    -. superdiag.(q - 1)
       *. superdiag.(q - 1)
       /. (d +. (Float_ops.sign d *. Float_ops.hypot d superdiag.(q - 1)))
  in
  let f = ref (diag.(p) -. shift) in
  let g = ref superdiag.(p) in
  for k = p to q - 1 do
    let c, s = Complex_ops.givens { re = !f; im = 0.0 } { re = !g; im = 0.0 } in
    if k > p then superdiag.(k - 1) <- Float_ops.hypot !f !g;
    f := (c *. diag.(k)) +. (s.re *. superdiag.(k));
    superdiag.(k) <-
      (-.(Complex_ops.conj s).re *. diag.(k)) +. (c *. superdiag.(k));
    g := s.re *. diag.(k + 1);
    diag.(k + 1) <- c *. diag.(k + 1);
    Complex_ops.apply_givens_right pool v n n k (k + 1) c s;
    let c, s = Complex_ops.givens { re = !f; im = 0.0 } { re = !g; im = 0.0 } in
    diag.(k) <- Float_ops.hypot !f !g;
    f := (c *. superdiag.(k)) +. (s.re *. diag.(k + 1));
    diag.(k + 1) <-
      (-.(Complex_ops.conj s).re *. superdiag.(k)) +. (c *. diag.(k + 1));
    if k < q - 1 then (
      g := s.re *. superdiag.(k + 1);
      superdiag.(k + 1) <- c *. superdiag.(k + 1));
    Complex_ops.apply_givens_left pool u m m k (k + 1) c s
  done;
  superdiag.(q - 1) <- !f

(* SVD iterate for real *)
let svd_iterate_real pool diag superdiag u v m n =
  let minmn = min m n in
  let tol = Float_ops.epsilon64 *. float (max m n) in
  (* adjust for precision if needed *)
  let max_iter = 75 * minmn in
  let iter = ref 0 in
  try
    while !iter < max_iter do
      incr iter;
      let converged = ref true in
      for i = 0 to minmn - 2 do
        if
          abs_float superdiag.(i)
          > tol *. (abs_float diag.(i) +. abs_float diag.(i + 1))
        then converged := false
      done;
      if !converged then raise Exit;
      let q_pos = ref (minmn - 1) in
      while
        !q_pos > 0
        && abs_float superdiag.(!q_pos - 1)
           <= tol *. (abs_float diag.(!q_pos - 1) +. abs_float diag.(!q_pos))
      do
        superdiag.(!q_pos - 1) <- 0.0;
        q_pos := !q_pos - 1
      done;
      let p_pos = ref !q_pos in
      while
        !p_pos > 0
        && abs_float superdiag.(!p_pos - 1)
           > tol *. (abs_float diag.(!p_pos - 1) +. abs_float diag.(!p_pos))
      do
        p_pos := !p_pos - 1
      done;
      if !p_pos < !q_pos then
        svd_qr_iteration_real pool diag superdiag u v m n !p_pos !q_pos
    done
  with Exit -> ()

(* SVD iterate for complex *)
let svd_iterate_complex pool diag superdiag u v m n =
  let minmn = min m n in
  let tol = Float_ops.epsilon64 *. float (max m n) in
  let max_iter = 75 * minmn in
  let iter = ref 0 in
  try
    while !iter < max_iter do
      incr iter;
      let converged = ref true in
      for i = 0 to minmn - 2 do
        if
          abs_float superdiag.(i)
          > tol *. (abs_float diag.(i) +. abs_float diag.(i + 1))
        then converged := false
      done;
      if !converged then raise Exit;
      let q_pos = ref (minmn - 1) in
      while
        !q_pos > 0
        && abs_float superdiag.(!q_pos - 1)
           <= tol *. (abs_float diag.(!q_pos - 1) +. abs_float diag.(!q_pos))
      do
        superdiag.(!q_pos - 1) <- 0.0;
        q_pos := !q_pos - 1
      done;
      let p_pos = ref !q_pos in
      while
        !p_pos > 0
        && abs_float superdiag.(!p_pos - 1)
           > tol *. (abs_float diag.(!p_pos - 1) +. abs_float diag.(!p_pos))
      do
        p_pos := !p_pos - 1
      done;
      if !p_pos < !q_pos then
        svd_qr_iteration_complex pool diag superdiag u v m n !p_pos !q_pos
    done
  with Exit -> ()

(* SVD core for real *)
let svd_real pool a u s v m n _full_matrices =
  let minmn = min m n in
  let diag = Array.make minmn 0.0 in
  let superdiag = Array.make (minmn - 1) 0.0 in
  bidiagonalize_real pool a u v diag superdiag m n;
  (* Debug: print U after bidiagonalization for 2x2 case if m = 2 && n = 2 then
     begin Printf.printf "After bidiagonalization:\n"; Printf.printf " U =
     [[%.6f, %.6f], [%.6f, %.6f]]\n" u.{0} u.{1} u.{2} u.{3}; Printf.printf " V
     = [[%.6f, %.6f], [%.6f, %.6f]]\n" v.{0} v.{1} v.{2} v.{3}; Printf.printf "
     diag = [%.6f, %.6f]\n" diag.(0) diag.(1); Printf.printf " superdiag =
     [%.6f]\n" superdiag.(0) end; *)
  svd_iterate_real pool diag superdiag u v m n;
  (* if m = 2 && n = 2 then begin Printf.printf "After QR iteration:\n";
     Printf.printf " U = [[%.6f, %.6f], [%.6f, %.6f]]\n" u.{0} u.{1} u.{2}
     u.{3}; Printf.printf " V = [[%.6f, %.6f], [%.6f, %.6f]]\n" v.{0} v.{1}
     v.{2} v.{3}; Printf.printf " diag = [%.6f, %.6f]\n" diag.(0) diag.(1)
     end; *)
  let s_arr = Array1.of_array float64 c_layout diag in
  for i = 0 to minmn - 1 do
    if s_arr.{i} < 0.0 then (
      (* If singular value is negative, make it positive and negate
         corresponding column in U *)
      s.{i} <- -.s_arr.{i};
      for j = 0 to m - 1 do
        u.{(j * m) + i} <- -.u.{(j * m) + i}
      done)
    else s.{i} <- s_arr.{i}
  done;
  for i = 0 to minmn - 2 do
    let max_idx = ref i in
    for j = i + 1 to minmn - 1 do
      if s.{j} > s.{!max_idx} then max_idx := j
    done;
    if !max_idx <> i then (
      let temp = s.{i} in
      s.{i} <- s.{!max_idx};
      s.{!max_idx} <- temp;
      let swap_loop size mat =
        if size > 100 then
          Parallel.parallel_for pool 0 (size - 1) (fun start_kk end_kk ->
              for kk = start_kk to end_kk - 1 do
                let temp = mat.{(kk * size) + i} in
                mat.{(kk * size) + i} <- mat.{(kk * size) + !max_idx};
                mat.{(kk * size) + !max_idx} <- temp
              done)
        else
          for kk = 0 to size - 1 do
            let temp = mat.{(kk * size) + i} in
            mat.{(kk * size) + i} <- mat.{(kk * size) + !max_idx};
            mat.{(kk * size) + !max_idx} <- temp
          done
      in
      swap_loop m u;
      swap_loop n v)
  done

let svd_complex pool a u s v m n _full_matrices =
  let minmn = min m n in
  let diag = Array.make minmn 0.0 in
  let superdiag = Array.make (minmn - 1) 0.0 in
  bidiagonalize_complex pool a u v diag superdiag m n;
  svd_iterate_complex pool diag superdiag u v m n;
  for i = 0 to minmn - 1 do
    s.{i} <- diag.(i)
  done;
  for i = 0 to minmn - 2 do
    let max_idx = ref i in
    for j = i + 1 to minmn - 1 do
      if s.{j} > s.{!max_idx} then max_idx := j
    done;
    if !max_idx <> i then (
      let temp = s.{i} in
      s.{i} <- s.{!max_idx};
      s.{!max_idx} <- temp;
      let swap_loop size mat =
        if size > 100 then
          Parallel.parallel_for pool 0 (size - 1) (fun start_kk end_kk ->
              for kk = start_kk to end_kk - 1 do
                let temp = mat.{(kk * size) + i} in
                mat.{(kk * size) + i} <- mat.{(kk * size) + !max_idx};
                mat.{(kk * size) + !max_idx} <- temp
              done)
        else
          for kk = 0 to size - 1 do
            let temp = mat.{(kk * size) + i} in
            mat.{(kk * size) + i} <- mat.{(kk * size) + !max_idx};
            mat.{(kk * size) + !max_idx} <- temp
          done
      in
      swap_loop m u;
      swap_loop n v)
  done

(* Pack/unpack general, for both real and complex *)
let pack pool dst src_buf off_src m n stride_row stride_col =
  if m > 100 then
    Parallel.parallel_for pool 0 (m - 1) (fun start_i end_i ->
        for i = start_i to end_i - 1 do
          for j = 0 to n - 1 do
            dst.{(i * n) + j} <-
              src_buf.{off_src + (i * stride_row) + (j * stride_col)}
          done
        done)
  else
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        dst.{(i * n) + j} <-
          src_buf.{off_src + (i * stride_row) + (j * stride_col)}
      done
    done

let unpack pool dst_buf off_dst src m n stride_row stride_col =
  if m > 100 then
    Parallel.parallel_for pool 0 (m - 1) (fun start_i end_i ->
        for i = start_i to end_i - 1 do
          for j = 0 to n - 1 do
            dst_buf.{off_dst + (i * stride_row) + (j * stride_col)} <-
              src.{(i * n) + j}
          done
        done)
  else
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        dst_buf.{off_dst + (i * stride_row) + (j * stride_col)} <-
          src.{(i * n) + j}
      done
    done

(* Kernel for SVD *)
let kernel_svd (type a b) pool (input : (a, b) t) (u : (a, b) t)
    (s : ('c, 'd) t) (vt : (a, b) t) full_matrices =
  let shape = get_shape input.view in
  let ndim = Array.length shape in
  let m = shape.(ndim - 2) in
  let n = shape.(ndim - 1) in
  let minmn = min m n in
  let batch_size =
    if ndim = 2 then 1 else Shape.numel (Array.sub shape 0 (ndim - 2))
  in
  let input_strides = get_strides input.view in
  let u_strides = get_strides u.view in
  let s_strides = get_strides s.view in
  let vt_strides = get_strides vt.view in
  let input_buf = buffer input in
  let u_buf = buffer u in
  let s_buf = buffer s in
  let vt_buf = buffer vt in
  let input_offset = get_offset input.view in
  let u_offset = get_offset u.view in
  let s_offset = get_offset s.view in
  let vt_offset = get_offset vt.view in
  let input_row_stride = input_strides.(ndim - 2) in
  let input_col_stride = input_strides.(ndim - 1) in
  let u_row_stride = u_strides.(ndim - 2) in
  let u_col_stride = u_strides.(ndim - 1) in
  let s_ndim = Array.length (get_shape s.view) in
  let s_stride = if s_ndim > 0 then s_strides.(s_ndim - 1) else 1 in
  let vt_row_stride = vt_strides.(ndim - 2) in
  let vt_col_stride = vt_strides.(ndim - 1) in
  let u_cols = if full_matrices then m else minmn in
  let vt_rows = if full_matrices then n else minmn in
  let kind_a = Dtype.to_bigarray_ext_kind (dtype input) in
  let kind_u = Dtype.to_bigarray_ext_kind (dtype u) in
  let kind_vt = Dtype.to_bigarray_ext_kind (dtype vt) in
  let kind_s = Dtype.to_bigarray_ext_kind (dtype s) in
  for b = 0 to batch_size - 1 do
    let off_input = input_offset + batch_offset b shape input_strides in
    let off_u = u_offset + batch_offset b (get_shape u.view) u_strides in
    let off_s = s_offset + batch_offset b (get_shape s.view) s_strides in
    let off_vt = vt_offset + batch_offset b (get_shape vt.view) vt_strides in

    (* Handle m < n case by transposing *)
    if m < n then (
      (* Transpose the matrix and swap U and V *)
      let packed_at = Array1.create kind_a c_layout (n * m) in
      (* Pack transposed: A^T *)
      for i = 0 to m - 1 do
        for j = 0 to n - 1 do
          let src_idx =
            off_input + (i * input_row_stride) + (j * input_col_stride)
          in
          let dst_idx = (j * m) + i in
          packed_at.{dst_idx} <- input_buf.{src_idx}
        done
      done;
      let packed_u_t = Array1.create kind_u c_layout (n * n) in
      let packed_vt_t = Array1.create kind_vt c_layout (m * m) in
      let packed_s = Array1.create kind_s c_layout minmn in
      let () =
        match dtype input with
        | Dtype.Float32 ->
            svd_real pool packed_at packed_u_t packed_s packed_vt_t n m
              full_matrices
        | Dtype.Float64 ->
            svd_real pool packed_at packed_u_t packed_s packed_vt_t n m
              full_matrices
        | Dtype.Complex32 ->
            svd_complex pool packed_at packed_u_t packed_s packed_vt_t n m
              full_matrices
        | Dtype.Complex64 ->
            svd_complex pool packed_at packed_u_t packed_s packed_vt_t n m
              full_matrices
        | _ -> Error.failed ~op:"svd" ~what:"unsupported dtype" ()
      in
      (* Copy results, swapping U and V^T *)
      (* packed_vt_t (m×m) contains V' which becomes U for original matrix *)
      unpack pool u_buf off_u packed_vt_t m u_cols u_row_stride u_col_stride;
      for i = 0 to minmn - 1 do
        let off = off_s + (i * s_stride) in
        s_buf.{off} <- packed_s.{i}
      done;
      (* packed_u_t (n×n) contains U' which needs to be transposed to become V^T *)
      (* We need to transpose packed_u_t when copying to vt_buf *)
      for ii = 0 to vt_rows - 1 do
        for jj = 0 to n - 1 do
          vt_buf.{off_vt + (ii * vt_row_stride) + (jj * vt_col_stride)} <-
            packed_u_t.{(jj * n) + ii}
        done
      done)
    else
      (* Original case for m >= n *)
      let packed_a = Array1.create kind_a c_layout (m * n) in
      pack pool packed_a input_buf off_input m n input_row_stride
        input_col_stride;
      let packed_u = Array1.create kind_u c_layout (m * m) in
      let packed_v = Array1.create kind_vt c_layout (n * n) in
      let packed_s = Array1.create kind_s c_layout minmn in
      let () =
        match dtype input with
        | Dtype.Float32 ->
            svd_real pool packed_a packed_u packed_s packed_v m n full_matrices
        | Dtype.Float64 ->
            svd_real pool packed_a packed_u packed_s packed_v m n full_matrices
        | Dtype.Complex32 ->
            svd_complex pool packed_a packed_u packed_s packed_v m n
              full_matrices
        | Dtype.Complex64 ->
            svd_complex pool packed_a packed_u packed_s packed_v m n
              full_matrices
        | _ -> Error.failed ~op:"svd" ~what:"unsupported dtype" ()
      in
      unpack pool u_buf off_u packed_u m u_cols u_row_stride u_col_stride;
      for i = 0 to minmn - 1 do
        s_buf.{off_s + (i * s_stride)} <- packed_s.{i}
      done;
      (* For vt, we need to transpose v *)
      if vt_rows > 100 then
        Parallel.parallel_for pool 0 (vt_rows - 1) (fun start_ii end_ii ->
            for ii = start_ii to end_ii - 1 do
              for jj = 0 to n - 1 do
                vt_buf.{off_vt + (ii * vt_row_stride) + (jj * vt_col_stride)} <-
                  packed_v.{(jj * n) + ii}
              done
            done)
      else
        for ii = 0 to vt_rows - 1 do
          for jj = 0 to n - 1 do
            vt_buf.{off_vt + (ii * vt_row_stride) + (jj * vt_col_stride)} <-
              packed_v.{(jj * n) + ii}
          done
        done
  done

(* Main svd function *)
let svd (type a b) (ctx : context) ~full_matrices (input : (a, b) t) =
  let shape = get_shape input.view in
  let ndim = Array.length shape in
  if ndim < 2 then
    Error.failed ~op:"svd" ~what:"input must have at least 2 dimensions" ();
  let batch_shape = if ndim = 2 then [||] else Array.sub shape 0 (ndim - 2) in
  let m = shape.(ndim - 2) in
  let n = shape.(ndim - 1) in
  let minmn = min m n in

  (* For m < n, we need to handle the case specially *)
  if m < n then (
    (* The kernel_svd already handles m < n case internally by transposing *)
    let u_shape =
      Array.append batch_shape [| m; (if full_matrices then m else m) |]
    in
    let s_shape = Array.append batch_shape [| m |] in
    let vt_shape =
      Array.append batch_shape [| (if full_matrices then n else m); n |]
    in
    let input_dtype = dtype input in
    let s_dtype = Dtype.Float64 in
    let u = empty ctx input_dtype u_shape in
    let s = empty ctx s_dtype s_shape in
    let vt = empty ctx input_dtype vt_shape in
    kernel_svd ctx.pool input u s vt full_matrices;
    (u, s, vt))
  else
    let u_shape =
      Array.append batch_shape [| m; (if full_matrices then m else minmn) |]
    in
    let s_shape = Array.append batch_shape [| minmn |] in
    let vt_shape =
      Array.append batch_shape [| (if full_matrices then n else minmn); n |]
    in
    let input_dtype = dtype input in
    let s_dtype = Dtype.Float64 in
    let u = empty ctx input_dtype u_shape in
    let s = empty ctx s_dtype s_shape in
    let vt = empty ctx input_dtype vt_shape in
    kernel_svd ctx.pool input u s vt full_matrices;
    (u, s, vt)
