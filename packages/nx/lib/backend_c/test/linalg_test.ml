(* @backend-stress — exhaustive internal branch coverage for nx_c_linalg.c.

   Residual-property checks complement the backend contract: factor,
   reconstruct, and bound the relative residual by c·n·ε. Positive-definite
   inputs are built as A = M·Mᴴ + n·I (Hermitian PD by construction). Buffers
   come from Nx_buffer so every compute dtype and the upcast (f16→f32) path is
   exercised.

   Public linalg semantics run under the normal Nx test suite. This direct-FFI
   gauntlet is kept separate because its large eig/SVD fixtures deliberately
   force internal algorithm and workspace branches. *)

module Buf = Nx_buffer
open Bigarray
open Windtrap

let ok name cond = is_true ~msg:name cond

type ('a, 'b) ffi = {
  buffer : ('a, 'b, c_layout) Genarray.t;
  shape : int array;
  strides : int array;
  offset : int;
}

external cholesky : ('a, 'b) ffi -> ('a, 'b) ffi -> bool -> unit
  = "caml_nx_c_cholesky"

(* flags: bit0 upper, bit1 transpose, bit2 unit_diag *)
external trsm : ('a, 'b) ffi -> ('a, 'b) ffi -> ('a, 'b) ffi -> int -> unit
  = "caml_nx_c_triangular_solve"

(* qr q r in reduced *)
external qr : ('a, 'b) ffi -> ('a, 'b) ffi -> ('a, 'b) ffi -> bool -> unit
  = "caml_nx_c_qr"

(* eigh w v in vectors — w is float64, v is input dtype. When vectors=false the
   binding re-passes the values tensor in the v slot (the C never touches
   it). *)
external eigh :
  (float, float64_elt) ffi -> ('a, 'b) ffi -> ('a, 'b) ffi -> bool -> unit
  = "caml_nx_c_eigh"

(* svd u s vt in — u/vt input dtype, s float64; thin/full encoded in the
   shapes. *)
external svd :
  ('a, 'b) ffi ->
  (float, float64_elt) ffi ->
  ('a, 'b) ffi ->
  ('a, 'b) ffi ->
  unit = "caml_nx_c_svd"

let ffi buf shape strides =
  {
    buffer = Buf.to_genarray buf [| Buf.length buf |];
    shape;
    strides;
    offset = 0;
  }

let contig shape =
  let n = Array.length shape in
  let s = Array.make n 1 in
  for i = n - 2 downto 0 do
    s.(i) <- s.(i + 1) * shape.(i + 1)
  done;
  s

(* ── Real symmetric PD cholesky ───────────────────────────────────────────*)

(* A[i][j] = sum_t M[i][t]*M[j][t] + (i=j ? n : 0), a deterministic
   pseudo-random symmetric positive-definite matrix. Returns the flat n*n array
   (row-major). *)
let make_spd n =
  let m =
    Array.init (n * n) (fun idx -> sin (float_of_int (idx * 13 mod 251) -. 3.0))
  in
  let a = Array.make (n * n) 0.0 in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      let s = ref 0.0 in
      for t = 0 to n - 1 do
        s := !s +. (m.((i * n) + t) *. m.((j * n) + t))
      done;
      a.((i * n) + j) <- (!s +. if i = j then float_of_int n else 0.0)
    done
  done;
  a

let test_chol_real (type b) ~(kind : (float, b) Buf.kind) ~name ~tol ~n ~upper
    () =
  let a = make_spd n in
  let ain = Buf.create kind (n * n) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let out = Buf.create kind (n * n) in
  cholesky
    (ffi out [| n; n |] (contig [| n; n |]))
    (ffi ain [| n; n |] (contig [| n; n |]))
    upper;
  (* reconstruct: lower L·Lᵀ, or upper Uᵀ·U *)
  let g i j = Buf.get out ((i * n) + j) in
  let recon i j =
    let s = ref 0.0 in
    for t = 0 to n - 1 do
      s := !s +. if upper then g t i *. g t j else g i t *. g j t
    done;
    !s
  in
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      let d = recon i j -. a.((i * n) + j) in
      num := !num +. (d *. d);
      den := !den +. (a.((i * n) + j) *. a.((i * n) + j))
    done
  done;
  let res = sqrt (!num /. !den) in
  (* triangle zeroing: the unused triangle must be exactly 0 *)
  let zeros = ref true in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      if ((not upper) && j > i) || (upper && j < i) then
        if g i j <> 0.0 then zeros := false
    done
  done;
  ok
    (Printf.sprintf "%s chol n=%d upper=%b residual=%.2e" name n upper res)
    (res <= tol);
  ok
    (Printf.sprintf "%s chol n=%d upper=%b other-triangle zero" name n upper)
    !zeros

(* ── Complex Hermitian PD cholesky ────────────────────────────────────────*)

let make_hpd n =
  let mre idx = sin (float_of_int (idx * 13 mod 251)) in
  let mim idx = cos (float_of_int (idx * 7 mod 241)) in
  let a = Array.make (n * n) Complex.zero in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      let s = ref Complex.zero in
      for t = 0 to n - 1 do
        let mi = { Complex.re = mre ((i * n) + t); im = mim ((i * n) + t) } in
        let mj = { Complex.re = mre ((j * n) + t); im = mim ((j * n) + t) } in
        (* M[i][t] * conj(M[j][t]) *)
        s := Complex.add !s (Complex.mul mi (Complex.conj mj))
      done;
      let d =
        if i = j then { Complex.re = float_of_int n; im = 0.0 }
        else Complex.zero
      in
      a.((i * n) + j) <- Complex.add !s d
    done
  done;
  a

let test_chol_complex (type b) ~(kind : (Complex.t, b) Buf.kind) ~name ~tol ~n
    ~upper () =
  let a = make_hpd n in
  let ain = Buf.create kind (n * n) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let out = Buf.create kind (n * n) in
  cholesky
    (ffi out [| n; n |] (contig [| n; n |]))
    (ffi ain [| n; n |] (contig [| n; n |]))
    upper;
  let g i j = Buf.get out ((i * n) + j) in
  (* lower: L·Lᴴ -> sum_t g i t * conj(g j t); upper: Uᴴ·U -> sum_t conj(g t
     i)*g t j *)
  let recon i j =
    let s = ref Complex.zero in
    for t = 0 to n - 1 do
      let term =
        if upper then Complex.mul (Complex.conj (g t i)) (g t j)
        else Complex.mul (g i t) (Complex.conj (g j t))
      in
      s := Complex.add !s term
    done;
    !s
  in
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      let d = Complex.sub (recon i j) a.((i * n) + j) in
      num :=
        !num +. (d.Complex.re *. d.Complex.re) +. (d.Complex.im *. d.Complex.im);
      let av = a.((i * n) + j) in
      den :=
        !den
        +. (av.Complex.re *. av.Complex.re)
        +. (av.Complex.im *. av.Complex.im)
    done
  done;
  let res = sqrt (!num /. !den) in
  ok
    (Printf.sprintf "%s chol n=%d upper=%b residual=%.2e" name n upper res)
    (res <= tol)

(* ── Batched ──────────────────────────────────────────────────────────────*)

let test_chol_batched ~n ~batch () =
  (* independent PD matrices per batch element (scale each by 1+b) *)
  let a = Array.init batch (fun _ -> make_spd n) in
  let ain = Buf.create Buf.float64 (batch * n * n) in
  for b = 0 to batch - 1 do
    for t = 0 to (n * n) - 1 do
      Buf.set ain ((b * n * n) + t) (a.(b).(t) *. float_of_int (1 + b))
    done
  done;
  let out = Buf.create Buf.float64 (batch * n * n) in
  cholesky
    (ffi out [| batch; n; n |] (contig [| batch; n; n |]))
    (ffi ain [| batch; n; n |] (contig [| batch; n; n |]))
    false;
  let worst = ref 0.0 in
  for b = 0 to batch - 1 do
    let g i j = Buf.get out ((b * n * n) + (i * n) + j) in
    let num = ref 0.0 and den = ref 0.0 in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let r = ref 0.0 in
        for t = 0 to n - 1 do
          r := !r +. (g i t *. g j t)
        done;
        let av = a.(b).((i * n) + j) *. float_of_int (1 + b) in
        let d = !r -. av in
        num := !num +. (d *. d);
        den := !den +. (av *. av)
      done
    done;
    let res = sqrt (!num /. !den) in
    if res > !worst then worst := res
  done;
  ok
    (Printf.sprintf "f64 chol batched=%d n=%d worst=%.2e" batch n !worst)
    (!worst <= 1e-10)

(* ── Non-PD raises ────────────────────────────────────────────────────────*)

let test_chol_not_pd () =
  let n = 4 in
  (* a matrix that is symmetric but indefinite: diagonal of -1 *)
  let ain = Buf.create Buf.float64 (n * n) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t 0.0
  done;
  for i = 0 to n - 1 do
    Buf.set ain ((i * n) + i) (-1.0)
  done;
  let out = Buf.create Buf.float64 (n * n) in
  let raised =
    try
      cholesky
        (ffi out [| n; n |] (contig [| n; n |]))
        (ffi ain [| n; n |] (contig [| n; n |]))
        false;
      false
    with Failure _ -> true
  in
  ok "chol non-PD raises Failure" raised;
  (* all-zeros: first pivot is 0, also not PD *)
  let z = Buf.create Buf.float64 (n * n) in
  let raised0 =
    try
      cholesky
        (ffi out [| n; n |] (contig [| n; n |]))
        (ffi z [| n; n |] (contig [| n; n |]))
        false;
      false
    with Failure _ -> true
  in
  ok "chol zero-matrix (pivot 0) raises Failure" raised0

(* ── Triangular solve ─────────────────────────────────────────────────────

   Build a well-conditioned triangular A and a known X, form B = op(A)·X, solve,
   and check the solver recovers X. op(A) is A or (conj-)transpose per the
   flags; with unit_diag the stored diagonal (kept != 1) must be ignored. *)

let flags ~upper ~transpose ~unit =
  (if upper then 1 else 0)
  lor (if transpose then 2 else 0)
  lor if unit then 4 else 0

let test_trsm_real (type b) ~(kind : (float, b) Buf.kind) ~name ~tol ~n ~nrhs
    ~upper ~transpose ~unit () =
  let a = Array.make (n * n) 0.0 in
  for i = 0 to n - 1 do
    for k = 0 to n - 1 do
      let intri = if upper then k >= i else k <= i in
      if intri then
        a.((i * n) + k) <-
          (if i = k then 1.5 +. (0.1 *. float_of_int i)
           else 0.3 *. sin (float_of_int ((i * n) + k)))
    done
  done;
  let xt = Array.init (n * nrhs) (fun idx -> cos (float_of_int (idx * 7))) in
  let opa i k = if transpose then a.((k * n) + i) else a.((i * n) + k) in
  let opad i k = if i = k && unit then 1.0 else opa i k in
  let bmat = Array.make (n * nrhs) 0.0 in
  for i = 0 to n - 1 do
    for j = 0 to nrhs - 1 do
      let s = ref 0.0 in
      for k = 0 to n - 1 do
        s := !s +. (opad i k *. xt.((k * nrhs) + j))
      done;
      bmat.((i * nrhs) + j) <- !s
    done
  done;
  let ain = Buf.create kind (n * n) and bin = Buf.create kind (n * nrhs) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  for t = 0 to (n * nrhs) - 1 do
    Buf.set bin t bmat.(t)
  done;
  let out = Buf.create kind (n * nrhs) in
  trsm
    (ffi out [| n; nrhs |] (contig [| n; nrhs |]))
    (ffi ain [| n; n |] (contig [| n; n |]))
    (ffi bin [| n; nrhs |] (contig [| n; nrhs |]))
    (flags ~upper ~transpose ~unit);
  let err = ref 0.0 in
  for t = 0 to (n * nrhs) - 1 do
    let e = abs_float (Buf.get out t -. xt.(t)) in
    if e > !err then err := e
  done;
  ok
    (Printf.sprintf "%s trsm n=%d nrhs=%d up=%b tr=%b unit=%b err=%.2e" name n
       nrhs upper transpose unit !err)
    (!err <= tol)

let test_trsm_complex (type b) ~(kind : (Complex.t, b) Buf.kind) ~name ~tol ~n
    ~upper ~transpose () =
  let a = Array.make (n * n) Complex.zero in
  for i = 0 to n - 1 do
    for k = 0 to n - 1 do
      let intri = if upper then k >= i else k <= i in
      if intri then
        a.((i * n) + k) <-
          (if i = k then
             { Complex.re = 1.5 +. (0.1 *. float_of_int i); im = 0.0 }
           else
             {
               Complex.re = 0.3 *. sin (float_of_int ((i * n) + k));
               im = 0.2 *. cos (float_of_int ((i * n) + k));
             })
    done
  done;
  let xt =
    Array.init n (fun idx ->
        {
          Complex.re = cos (float_of_int (idx * 7));
          im = sin (float_of_int (idx * 5));
        })
  in
  (* op(A): transpose ⇒ conjugate transpose *)
  let opa i k =
    if transpose then Complex.conj a.((k * n) + i) else a.((i * n) + k)
  in
  let bvec = Array.make n Complex.zero in
  for i = 0 to n - 1 do
    let s = ref Complex.zero in
    for k = 0 to n - 1 do
      s := Complex.add !s (Complex.mul (opa i k) xt.(k))
    done;
    bvec.(i) <- !s
  done;
  let ain = Buf.create kind (n * n) and bin = Buf.create kind n in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  for t = 0 to n - 1 do
    Buf.set bin t bvec.(t)
  done;
  let out = Buf.create kind n in
  trsm
    (ffi out [| n; 1 |] (contig [| n; 1 |]))
    (ffi ain [| n; n |] (contig [| n; n |]))
    (ffi bin [| n; 1 |] (contig [| n; 1 |]))
    (flags ~upper ~transpose ~unit:false);
  let err = ref 0.0 in
  for t = 0 to n - 1 do
    let d = Complex.sub (Buf.get out t) xt.(t) in
    let e =
      sqrt ((d.Complex.re *. d.Complex.re) +. (d.Complex.im *. d.Complex.im))
    in
    if e > !err then err := e
  done;
  ok
    (Printf.sprintf "%s trsm n=%d up=%b tr=%b err=%.2e" name n upper transpose
       !err)
    (!err <= tol)

let test_trsm_singular () =
  let n = 4 in
  let ain = Buf.create Buf.float64 (n * n) in
  for i = 0 to n - 1 do
    for k = 0 to n - 1 do
      if k <= i then Buf.set ain ((i * n) + k) (if i = k then 1.0 else 0.5)
    done
  done;
  Buf.set ain ((2 * n) + 2) 0.0;
  (* zero pivot *)
  let bin = Buf.create Buf.float64 n in
  for t = 0 to n - 1 do
    Buf.set bin t 1.0
  done;
  let out = Buf.create Buf.float64 n in
  let raised =
    try
      trsm
        (ffi out [| n; 1 |] (contig [| n; 1 |]))
        (ffi ain [| n; n |] (contig [| n; n |]))
        (ffi bin [| n; 1 |] (contig [| n; 1 |]))
        (flags ~upper:false ~transpose:false ~unit:false);
      false
    with Failure _ -> true
  in
  ok "trsm singular (zero pivot) raises Failure" raised

let test_trsm_batched () =
  let n = 6 and nrhs = 3 and batch = 4 in
  let a = Array.make (n * n) 0.0 in
  for i = 0 to n - 1 do
    for k = 0 to i do
      a.((i * n) + k) <- (if i = k then 2.0 else 0.2 *. float_of_int (k + 1))
    done
  done;
  let xt =
    Array.init (batch * n * nrhs) (fun idx -> sin (float_of_int (idx + 1)))
  in
  let bmat = Array.make (batch * n * nrhs) 0.0 in
  for bb = 0 to batch - 1 do
    for i = 0 to n - 1 do
      for j = 0 to nrhs - 1 do
        let s = ref 0.0 in
        for k = 0 to i do
          s := !s +. (a.((i * n) + k) *. xt.((bb * n * nrhs) + (k * nrhs) + j))
        done;
        bmat.((bb * n * nrhs) + (i * nrhs) + j) <- !s
      done
    done
  done;
  let ain = Buf.create Buf.float64 (batch * n * n) in
  for bb = 0 to batch - 1 do
    for t = 0 to (n * n) - 1 do
      Buf.set ain ((bb * n * n) + t) a.(t)
    done
  done;
  let bin = Buf.create Buf.float64 (batch * n * nrhs) in
  for t = 0 to (batch * n * nrhs) - 1 do
    Buf.set bin t bmat.(t)
  done;
  let out = Buf.create Buf.float64 (batch * n * nrhs) in
  trsm
    (ffi out [| batch; n; nrhs |] (contig [| batch; n; nrhs |]))
    (ffi ain [| batch; n; n |] (contig [| batch; n; n |]))
    (ffi bin [| batch; n; nrhs |] (contig [| batch; n; nrhs |]))
    (flags ~upper:false ~transpose:false ~unit:false);
  let err = ref 0.0 in
  for t = 0 to (batch * n * nrhs) - 1 do
    let e = abs_float (Buf.get out t -. xt.(t)) in
    if e > !err then err := e
  done;
  ok (Printf.sprintf "f64 trsm batched=%d err=%.2e" batch !err) (!err <= 1e-9)

(* ── QR ───────────────────────────────────────────────────────────────────

   Property gate: ‖A − QR‖/‖A‖ small, QᴴQ = I (orthonormal columns), R upper
   triangular. Covers reduced/full and tall/wide/square. *)

let test_qr_real (type b) ~(kind : (float, b) Buf.kind) ~name ~tol ~m ~n
    ~reduced () =
  let a =
    Array.init (m * n) (fun idx ->
        sin (float_of_int (idx * 13 mod 251)) +. (0.1 *. cos (float_of_int idx)))
  in
  let ain = Buf.create kind (m * n) in
  for t = 0 to (m * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let k = min m n in
  let nq = if reduced then k else m in
  let qb = Buf.create kind (m * nq) and rb = Buf.create kind (nq * n) in
  qr
    (ffi qb [| m; nq |] (contig [| m; nq |]))
    (ffi rb [| nq; n |] (contig [| nq; n |]))
    (ffi ain [| m; n |] (contig [| m; n |]))
    reduced;
  let getq i c = Buf.get qb ((i * nq) + c) in
  let getr i c = Buf.get rb ((i * n) + c) in
  (* reconstruction *)
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0.0 in
      for c = 0 to nq - 1 do
        s := !s +. (getq i c *. getr c j)
      done;
      let d = !s -. a.((i * n) + j) in
      num := !num +. (d *. d);
      den := !den +. (a.((i * n) + j) *. a.((i * n) + j))
    done
  done;
  let recon = sqrt (!num /. !den) in
  (* orthonormality QᵀQ = I *)
  let orth = ref 0.0 in
  for c1 = 0 to nq - 1 do
    for c2 = 0 to nq - 1 do
      let s = ref 0.0 in
      for i = 0 to m - 1 do
        s := !s +. (getq i c1 *. getq i c2)
      done;
      let target = if c1 = c2 then 1.0 else 0.0 in
      let e = abs_float (!s -. target) in
      if e > !orth then orth := e
    done
  done;
  (* R upper triangular *)
  let rlow = ref 0.0 in
  for i = 0 to nq - 1 do
    for c = 0 to min (i - 1) (n - 1) do
      let e = abs_float (getr i c) in
      if e > !rlow then rlow := e
    done
  done;
  ok
    (Printf.sprintf "%s qr %dx%d reduced=%b recon=%.2e orth=%.2e Rlow=%.2e" name
       m n reduced recon !orth !rlow)
    (recon <= tol && !orth <= tol && !rlow <= tol)

let test_qr_complex (type b) ~(kind : (Complex.t, b) Buf.kind) ~name ~tol ~m ~n
    ~reduced () =
  let a =
    Array.init (m * n) (fun idx ->
        {
          Complex.re = sin (float_of_int (idx * 13 mod 251));
          im = cos (float_of_int (idx * 7 mod 241));
        })
  in
  let ain = Buf.create kind (m * n) in
  for t = 0 to (m * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let k = min m n in
  let nq = if reduced then k else m in
  let qb = Buf.create kind (m * nq) and rb = Buf.create kind (nq * n) in
  qr
    (ffi qb [| m; nq |] (contig [| m; nq |]))
    (ffi rb [| nq; n |] (contig [| nq; n |]))
    (ffi ain [| m; n |] (contig [| m; n |]))
    reduced;
  let getq i c = Buf.get qb ((i * nq) + c) in
  let getr i c = Buf.get rb ((i * n) + c) in
  let cabs2 (z : Complex.t) =
    (z.Complex.re *. z.Complex.re) +. (z.Complex.im *. z.Complex.im)
  in
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref Complex.zero in
      for c = 0 to nq - 1 do
        s := Complex.add !s (Complex.mul (getq i c) (getr c j))
      done;
      num := !num +. cabs2 (Complex.sub !s a.((i * n) + j));
      den := !den +. cabs2 a.((i * n) + j)
    done
  done;
  let recon = sqrt (!num /. !den) in
  (* QᴴQ = I *)
  let orth = ref 0.0 in
  for c1 = 0 to nq - 1 do
    for c2 = 0 to nq - 1 do
      let s = ref Complex.zero in
      for i = 0 to m - 1 do
        s := Complex.add !s (Complex.mul (Complex.conj (getq i c1)) (getq i c2))
      done;
      let target = if c1 = c2 then Complex.one else Complex.zero in
      let e = sqrt (cabs2 (Complex.sub !s target)) in
      if e > !orth then orth := e
    done
  done;
  ok
    (Printf.sprintf "%s qr %dx%d reduced=%b recon=%.2e orth=%.2e" name m n
       reduced recon !orth)
    (recon <= tol && !orth <= tol)

let test_qr_batched () =
  let batch = 4 and m = 7 and n = 5 in
  let a =
    Array.init
      (batch * m * n)
      (fun idx -> sin (float_of_int (idx * 11 mod 263)))
  in
  let ain = Buf.create Buf.float64 (batch * m * n) in
  for t = 0 to (batch * m * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let k = min m n in
  let qb = Buf.create Buf.float64 (batch * m * k)
  and rb = Buf.create Buf.float64 (batch * k * n) in
  qr
    (ffi qb [| batch; m; k |] (contig [| batch; m; k |]))
    (ffi rb [| batch; k; n |] (contig [| batch; k; n |]))
    (ffi ain [| batch; m; n |] (contig [| batch; m; n |]))
    true;
  let worst = ref 0.0 in
  for bb = 0 to batch - 1 do
    let getq i c = Buf.get qb ((bb * m * k) + (i * k) + c) in
    let getr i c = Buf.get rb ((bb * k * n) + (i * n) + c) in
    let num = ref 0.0 and den = ref 0.0 in
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        let s = ref 0.0 in
        for c = 0 to k - 1 do
          s := !s +. (getq i c *. getr c j)
        done;
        let av = a.((bb * m * n) + (i * n) + j) in
        let d = !s -. av in
        num := !num +. (d *. d);
        den := !den +. (av *. av)
      done
    done;
    let res = sqrt (!num /. !den) in
    if res > !worst then worst := res
  done;
  ok
    (Printf.sprintf "f64 qr batched=%d worst-recon=%.2e" batch !worst)
    (!worst <= 1e-10)

(* ── Closing-items coverage (tier-1 review) ───────────────────────────────*)

(* Batched cross-worker error aggregation: one non-PD among many PD (batch big
   enough to fan across workers) must raise Failure. *)
let test_chol_batched_fail () =
  let batch = 20 and n = 6 in
  let a = make_spd n in
  let ain = Buf.create Buf.float64 (batch * n * n) in
  for b = 0 to batch - 1 do
    for t = 0 to (n * n) - 1 do
      Buf.set ain ((b * n * n) + t) a.(t)
    done
  done;
  (* corrupt element 13: negate its (0,0) pivot → first pivot < 0 → not PD *)
  Buf.set ain ((13 * n * n) + 0) (-.Buf.get ain ((13 * n * n) + 0));
  let out = Buf.create Buf.float64 (batch * n * n) in
  let raised =
    try
      cholesky
        (ffi out [| batch; n; n |] (contig [| batch; n; n |]))
        (ffi ain [| batch; n; n |] (contig [| batch; n; n |]))
        false;
      false
    with Failure _ -> true
  in
  ok "chol batched=20 one-non-PD raises Failure (cross-worker werr)" raised

let test_trsm_batched_fail () =
  let batch = 20 and n = 5 in
  let ain = Buf.create Buf.float64 (batch * n * n) in
  for b = 0 to batch - 1 do
    for i = 0 to n - 1 do
      for k = 0 to i do
        Buf.set ain ((b * n * n) + (i * n) + k) (if i = k then 2.0 else 0.3)
      done
    done
  done;
  (* element 7: zero a diagonal → singular *)
  Buf.set ain ((7 * n * n) + (2 * n) + 2) 0.0;
  let bin = Buf.create Buf.float64 (batch * n) in
  for t = 0 to (batch * n) - 1 do
    Buf.set bin t 1.0
  done;
  let out = Buf.create Buf.float64 (batch * n) in
  let raised =
    try
      trsm
        (ffi out [| batch; n; 1 |] (contig [| batch; n; 1 |]))
        (ffi ain [| batch; n; n |] (contig [| batch; n; n |]))
        (ffi bin [| batch; n; 1 |] (contig [| batch; n; 1 |]))
        (flags ~upper:false ~transpose:false ~unit:false);
      false
    with Failure _ -> true
  in
  ok "trsm batched=20 one-singular raises Failure (cross-worker werr)" raised

(* Offline fixture pinning the complex transpose = CONJUGATE-transpose
   convention (scipy solve_triangular trans='C' / 2), so the check is not
   self-referential. A lower = [[1+i, 0],[2-i, 1+2i]], b = [1, 4]; solving Aᴴ x
   = b by hand gives x = [2.5-1.5i, 0.8+1.6i]. *)
let test_trsm_conj_fixture () =
  let c re im = { Complex.re; im } in
  let n = 2 in
  let ain = Buf.create Buf.complex128 (n * n) in
  Buf.set ain 0 (c 1.0 1.0);
  Buf.set ain 1 Complex.zero;
  Buf.set ain 2 (c 2.0 (-1.0));
  Buf.set ain 3 (c 1.0 2.0);
  let bin = Buf.create Buf.complex128 n in
  Buf.set bin 0 (c 1.0 0.0);
  Buf.set bin 1 (c 4.0 0.0);
  let out = Buf.create Buf.complex128 n in
  trsm
    (ffi out [| n; 1 |] (contig [| n; 1 |]))
    (ffi ain [| n; n |] (contig [| n; n |]))
    (ffi bin [| n; 1 |] (contig [| n; 1 |]))
    (flags ~upper:false ~transpose:true ~unit:false);
  let expect = [| c 2.5 (-1.5); c 0.8 1.6 |] in
  let bad = ref 0 in
  for i = 0 to n - 1 do
    let g = Buf.get out i and e = expect.(i) in
    if
      abs_float (g.Complex.re -. e.Complex.re) > 1e-12
      || abs_float (g.Complex.im -. e.Complex.im) > 1e-12
    then incr bad
  done;
  ok (Printf.sprintf "c64 trsm conj-transpose fixture (bad=%d)" !bad) (!bad = 0)

(* QR with an interior zero column exercises the tau=0 (no-reflection) path;
   reconstruction and orthonormality must still hold. *)
let test_qr_zerocol ?(m = 6) ?(n = 4) ?(zc = 2) () =
  let a = Array.make (m * n) 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      if j <> zc then a.((i * n) + j) <- sin (float_of_int (((i * n) + j) * 5))
    done
  done;
  let ain = Buf.create Buf.float64 (m * n) in
  for t = 0 to (m * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let k = min m n in
  let qb = Buf.create Buf.float64 (m * k)
  and rb = Buf.create Buf.float64 (k * n) in
  qr
    (ffi qb [| m; k |] (contig [| m; k |]))
    (ffi rb [| k; n |] (contig [| k; n |]))
    (ffi ain [| m; n |] (contig [| m; n |]))
    true;
  let getq i c = Buf.get qb ((i * k) + c)
  and getr i c = Buf.get rb ((i * n) + c) in
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0.0 in
      for c = 0 to k - 1 do
        s := !s +. (getq i c *. getr c j)
      done;
      let d = !s -. a.((i * n) + j) in
      num := !num +. (d *. d);
      den := !den +. (a.((i * n) + j) *. a.((i * n) + j))
    done
  done;
  let recon = sqrt (!num /. !den) in
  let orth = ref 0.0 in
  for c1 = 0 to k - 1 do
    for c2 = 0 to k - 1 do
      let s = ref 0.0 in
      for i = 0 to m - 1 do
        s := !s +. (getq i c1 *. getq i c2)
      done;
      let e = abs_float (!s -. if c1 = c2 then 1.0 else 0.0) in
      if e > !orth then orth := e
    done
  done;
  ok
    (Printf.sprintf
       "f64 qr zero-column %dx%d zc=%d (tau=0 path) recon=%.2e orth=%.2e" m n zc
       recon !orth)
    (recon <= 1e-10 && !orth <= 1e-10)

(* n=0 empty matrices are no-ops (must not raise or crash). *)
let test_zero_dim () =
  let e0 = Buf.create Buf.float64 0 in
  let raised =
    try
      cholesky
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        false;
      trsm
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        0;
      qr
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        true;
      eigh (ffi e0 [| 0 |] [| 1 |])
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        true;
      svd
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        (ffi e0 [| 0 |] [| 1 |])
        (ffi e0 [| 0; 0 |] [| 0; 1 |])
        (ffi e0 [| 0; 0 |] [| 0; 1 |]);
      false
    with _ -> true
  in
  ok "n=0 chol/trsm/qr/eigh/svd are no-ops (no raise)" (not raised)

(* ── eigh ─────────────────────────────────────────────────────────────────

   Property gate: ‖A − V diag(w) Vᴴ‖/‖A‖ small, VᴴV = I, w ascending and real. A
   is built symmetric/Hermitian; the C reads the lower triangle only. *)

let test_eigh_real (type b) ~(kind : (float, b) Buf.kind) ~name ~tol ~n () =
  let m =
    Array.init (n * n) (fun idx -> sin (float_of_int (idx * 13 mod 251)))
  in
  let a = Array.make (n * n) 0.0 in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      a.((i * n) + j) <- m.((i * n) + j) +. m.((j * n) + i)
    done
  done;
  let ain = Buf.create kind (n * n) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let wout = Buf.create Buf.float64 n and vout = Buf.create kind (n * n) in
  eigh (ffi wout [| n |] [| 1 |])
    (ffi vout [| n; n |] (contig [| n; n |]))
    (ffi ain [| n; n |] (contig [| n; n |]))
    true;
  let wv i = Buf.get wout i and vv i j = Buf.get vout ((i * n) + j) in
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to n - 1 do
    for k = 0 to n - 1 do
      let s = ref 0.0 in
      for j = 0 to n - 1 do
        s := !s +. (vv i j *. wv j *. vv k j)
      done;
      let d = !s -. a.((i * n) + k) in
      num := !num +. (d *. d);
      den := !den +. (a.((i * n) + k) *. a.((i * n) + k))
    done
  done;
  let recon = if !den < 1e-30 then sqrt !num else sqrt (!num /. !den) in
  let orth = ref 0.0 in
  for c1 = 0 to n - 1 do
    for c2 = 0 to n - 1 do
      let s = ref 0.0 in
      for i = 0 to n - 1 do
        s := !s +. (vv i c1 *. vv i c2)
      done;
      let e = abs_float (!s -. if c1 = c2 then 1.0 else 0.0) in
      if e > !orth then orth := e
    done
  done;
  let asc = ref true in
  for i = 0 to n - 2 do
    if wv i > wv (i + 1) +. 1e-6 then asc := false
  done;
  ok
    (Printf.sprintf "%s eigh n=%d recon=%.2e orth=%.2e asc=%b" name n recon
       !orth !asc)
    (recon <= tol && !orth <= tol && !asc)

let test_eigh_complex (type b) ~(kind : (Complex.t, b) Buf.kind) ~name ~tol ~n
    () =
  let mre idx = sin (float_of_int (idx * 13 mod 251)) in
  let mim idx = cos (float_of_int (idx * 7 mod 241)) in
  let a = Array.make (n * n) Complex.zero in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      (* A = M + Mᴴ is Hermitian: A[i][j] = M[i][j] + conj(M[j][i]) *)
      let mij = { Complex.re = mre ((i * n) + j); im = mim ((i * n) + j) } in
      let mji = { Complex.re = mre ((j * n) + i); im = mim ((j * n) + i) } in
      a.((i * n) + j) <- Complex.add mij (Complex.conj mji)
    done
  done;
  let ain = Buf.create kind (n * n) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let wout = Buf.create Buf.float64 n and vout = Buf.create kind (n * n) in
  eigh (ffi wout [| n |] [| 1 |])
    (ffi vout [| n; n |] (contig [| n; n |]))
    (ffi ain [| n; n |] (contig [| n; n |]))
    true;
  let wv i = Buf.get wout i and vv i j = Buf.get vout ((i * n) + j) in
  let cabs2 (z : Complex.t) =
    (z.Complex.re *. z.Complex.re) +. (z.Complex.im *. z.Complex.im)
  in
  (* A' = V diag(w) Vᴴ : A'[i][k] = sum_j V[i][j] w[j] conj(V[k][j]) *)
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to n - 1 do
    for k = 0 to n - 1 do
      let s = ref Complex.zero in
      for j = 0 to n - 1 do
        let term =
          Complex.mul (vv i j)
            (Complex.mul
               { Complex.re = wv j; im = 0.0 }
               (Complex.conj (vv k j)))
        in
        s := Complex.add !s term
      done;
      num := !num +. cabs2 (Complex.sub !s a.((i * n) + k));
      den := !den +. cabs2 a.((i * n) + k)
    done
  done;
  let recon = if !den < 1e-30 then sqrt !num else sqrt (!num /. !den) in
  (* VᴴV = I *)
  let orth = ref 0.0 in
  for c1 = 0 to n - 1 do
    for c2 = 0 to n - 1 do
      let s = ref Complex.zero in
      for i = 0 to n - 1 do
        s := Complex.add !s (Complex.mul (Complex.conj (vv i c1)) (vv i c2))
      done;
      let e =
        sqrt
          (cabs2
             (Complex.sub !s (if c1 = c2 then Complex.one else Complex.zero)))
      in
      if e > !orth then orth := e
    done
  done;
  let asc = ref true in
  for i = 0 to n - 2 do
    if wv i > wv (i + 1) +. 1e-6 then asc := false
  done;
  ok
    (Printf.sprintf "%s eigh n=%d recon=%.2e orth=%.2e asc=%b" name n recon
       !orth !asc)
    (recon <= tol && !orth <= tol && !asc)

(* Diagonal input: eigenvalues are the sorted diagonal, exact. *)
let test_eigh_diag () =
  let vals = [| 5.0; 2.0; 8.0; 1.0; 2.0 |] in
  let n = Array.length vals in
  let ain = Buf.create Buf.float64 (n * n) in
  Buf.fill ain 0.0;
  for i = 0 to n - 1 do
    Buf.set ain ((i * n) + i) vals.(i)
  done;
  let wout = Buf.create Buf.float64 n
  and vout = Buf.create Buf.float64 (n * n) in
  eigh (ffi wout [| n |] [| 1 |])
    (ffi vout [| n; n |] (contig [| n; n |]))
    (ffi ain [| n; n |] (contig [| n; n |]))
    true;
  let sorted = Array.copy vals in
  Array.sort compare sorted;
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if abs_float (Buf.get wout i -. sorted.(i)) > 1e-12 then incr bad
  done;
  ok
    (Printf.sprintf "f64 eigh diagonal exact eigenvalues (bad=%d)" !bad)
    (!bad = 0)

(* vectors=false must give the same eigenvalues; the values tensor is re-passed
   in the v slot (must be untouched). *)
let test_eigh_novec ~n () =
  let m = Array.init (n * n) (fun idx -> sin (float_of_int (idx + 3))) in
  let a = Array.make (n * n) 0.0 in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      a.((i * n) + j) <- m.((i * n) + j) +. m.((j * n) + i)
    done
  done;
  let ain = Buf.create Buf.float64 (n * n) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let w1 = Buf.create Buf.float64 n and vout = Buf.create Buf.float64 (n * n) in
  eigh (ffi w1 [| n |] [| 1 |])
    (ffi vout [| n; n |] (contig [| n; n |]))
    (ffi ain [| n; n |] (contig [| n; n |]))
    true;
  let w2 = Buf.create Buf.float64 n in
  let ain_ffi = ffi ain [| n; n |] (contig [| n; n |]) in
  eigh (ffi w2 [| n |] [| 1 |]) ain_ffi ain_ffi false;
  (* re-pass values in the v slot *)
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if abs_float (Buf.get w1 i -. Buf.get w2 i) > 1e-12 then incr bad
  done;
  ok (Printf.sprintf "f64 eigh vectors=false matches (bad=%d)" !bad) (!bad = 0)

let test_eigh_batched () =
  let batch = 6 and n = 7 in
  let mats =
    Array.init batch (fun b ->
        let m =
          Array.init (n * n) (fun idx ->
              sin (float_of_int (((b * 31) + idx) mod 251)))
        in
        let a = Array.make (n * n) 0.0 in
        for i = 0 to n - 1 do
          for j = 0 to n - 1 do
            a.((i * n) + j) <- m.((i * n) + j) +. m.((j * n) + i)
          done
        done;
        a)
  in
  let ain = Buf.create Buf.float64 (batch * n * n) in
  for b = 0 to batch - 1 do
    for t = 0 to (n * n) - 1 do
      Buf.set ain ((b * n * n) + t) mats.(b).(t)
    done
  done;
  let wout = Buf.create Buf.float64 (batch * n)
  and vout = Buf.create Buf.float64 (batch * n * n) in
  eigh
    (ffi wout [| batch; n |] (contig [| batch; n |]))
    (ffi vout [| batch; n; n |] (contig [| batch; n; n |]))
    (ffi ain [| batch; n; n |] (contig [| batch; n; n |]))
    true;
  let worst = ref 0.0 in
  for b = 0 to batch - 1 do
    let wv i = Buf.get wout ((b * n) + i) in
    let vv i j = Buf.get vout ((b * n * n) + (i * n) + j) in
    let num = ref 0.0 and den = ref 0.0 in
    for i = 0 to n - 1 do
      for k = 0 to n - 1 do
        let s = ref 0.0 in
        for j = 0 to n - 1 do
          s := !s +. (vv i j *. wv j *. vv k j)
        done;
        let d = !s -. mats.(b).((i * n) + k) in
        num := !num +. (d *. d);
        den := !den +. (mats.(b).((i * n) + k) *. mats.(b).((i * n) + k))
      done
    done;
    let res = sqrt (!num /. !den) in
    if res > !worst then worst := res
  done;
  ok
    (Printf.sprintf "f64 eigh batched=%d worst-recon=%.2e" batch !worst)
    (!worst <= 1e-9)

(* ── eigh divide-and-conquer gauntlet ──────────────────────────────────────

   The D&C activates for eigh with vectors:true and n > 25 (SMLSIZ); tql2 stays
   the eigenvalues-only path, so eigh ~vectors:false is an INDEPENDENT oracle
   for the D&C eigenvalues (a different algorithm entirely). Each case runs a
   symmetric f64 matrix through both, requiring the eigenvalue sets to agree to
   a tight relative tol AND the D&C vectors to pass ‖A−VΛVᴴ‖/‖A‖ + ‖VᴴV−I‖ at
   c·n·ε. The builders stress the deflation machinery: tight eigenvalue
   clusters, glued Wilkinson pairs, and a spectrum graded across 16 orders of
   magnitude. *)

let test_eigh_crosscheck ~name ~n ~build ~tol ?(orth_c = 128.0) () =
  let a = build n in
  let ain = Buf.create Buf.float64 (n * n) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let ffi_a () = ffi ain [| n; n |] (contig [| n; n |]) in
  let w_dc = Buf.create Buf.float64 n and v = Buf.create Buf.float64 (n * n) in
  eigh (ffi w_dc [| n |] [| 1 |])
    (ffi v [| n; n |] (contig [| n; n |]))
    (ffi_a ()) true;
  let w_ql = Buf.create Buf.float64 n in
  let fa = ffi_a () in
  (* tql2 is the independent oracle where it converges; on a spectrum too graded
     for the QL iteration (the very case D&C exists for) it caps out, and the
     D&C's own ‖A−VΛVᴴ‖/‖VᴴV−I‖ residual gates carry the proof instead. *)
  let ql_ok = ref true in
  (try eigh (ffi w_ql [| n |] [| 1 |]) fa fa false
   with Failure _ -> ql_ok := false);
  let scale = ref 1.0 in
  for i = 0 to n - 1 do
    scale := Float.max !scale (abs_float (Buf.get w_dc i))
  done;
  let dmax = ref 0.0 in
  if !ql_ok then
    for i = 0 to n - 1 do
      let e = abs_float (Buf.get w_dc i -. Buf.get w_ql i) in
      if e > !dmax then dmax := e
    done;
  let vv i j = Buf.get v ((i * n) + j) and wv i = Buf.get w_dc i in
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to n - 1 do
    for k = 0 to n - 1 do
      let s = ref 0.0 in
      for j = 0 to n - 1 do
        s := !s +. (vv i j *. wv j *. vv k j)
      done;
      let d = !s -. a.((i * n) + k) in
      num := !num +. (d *. d);
      den := !den +. (a.((i * n) + k) *. a.((i * n) + k))
    done
  done;
  let recon = if !den < 1e-30 then sqrt !num else sqrt (!num /. !den) in
  let orth = ref 0.0 in
  for c1 = 0 to n - 1 do
    for c2 = 0 to n - 1 do
      let s = ref 0.0 in
      for i = 0 to n - 1 do
        s := !s +. (vv i c1 *. vv i c2)
      done;
      let e = abs_float (!s -. if c1 = c2 then 1.0 else 0.0) in
      if e > !orth then orth := e
    done
  done;
  let asc = ref true in
  for i = 0 to n - 2 do
    if wv i > wv (i + 1) +. (1e-6 *. !scale) then asc := false
  done;
  (* Orthogonality is the sharp gate for deflation-aggressiveness / Gu-Eisenstat
     degradation: a real c·n·ε sentinel, not a fixed slack. clustered
     eigenvalues leave the in-cluster eigenvector basis rotation-free only to
     c·n·ε, so it carries a larger c. *)
  let orth_tol = orth_c *. float_of_int n *. epsilon_float in
  ok
    (Printf.sprintf
       "%s eigh D&C n=%d xcheck-Δw=%.2e%s recon=%.2e orth=%.2e(tol %.1e) asc=%b"
       name n (!dmax /. !scale)
       (if !ql_ok then "" else "(ql-capped)")
       recon !orth orth_tol !asc)
    (!dmax <= tol *. !scale && recon <= tol && !orth <= orth_tol && !asc)

let build_smooth n =
  let a = Array.make (n * n) 0.0 in
  for i = 0 to n - 1 do
    a.((i * n) + i) <- sin (float_of_int i *. 1.3)
  done;
  for i = 0 to n - 1 do
    for j = 0 to i - 1 do
      let v = 0.3 *. cos (float_of_int ((i * 7) + j)) in
      a.((i * n) + j) <- v;
      a.((j * n) + i) <- v
    done
  done;
  a

(* Tight eigenvalue clusters (three levels of exactly-equal diagonal + a 1e-6
   symmetric perturbation): the negligible-gap deflation stressor. *)
let build_clustered n =
  let a = Array.make (n * n) 0.0 in
  for i = 0 to n - 1 do
    a.((i * n) + i) <-
      (if i < n / 2 then 1.0 else if i < 3 * n / 4 then 2.0 else 3.0)
  done;
  for i = 0 to n - 1 do
    for j = 0 to i - 1 do
      let v = 1e-6 *. sin (float_of_int ((i * n) + j)) in
      a.((i * n) + j) <- v;
      a.((j * n) + i) <- v
    done
  done;
  a

(* Glued Wilkinson: repeated W-block diagonals with one near-zero coupling at
   the tear — the classic D&C stress (pairs of nearly-degenerate eigenvalues
   split across the split point). *)
let build_glued_wilkinson n =
  let a = Array.make (n * n) 0.0 in
  let half = (n - 1) / 2 in
  for i = 0 to n - 1 do
    let k = i mod (half + 1) in
    a.((i * n) + i) <- abs_float (float_of_int (k - (half / 2)))
  done;
  for i = 0 to n - 2 do
    let v = if i = (n / 2) - 1 then 1e-7 else 1.0 in
    a.(((i + 1) * n) + i) <- v;
    a.((i * n) + (i + 1)) <- v
  done;
  a

(* Spectrum graded 1e-8..1e8 (tridiagonal, diagonal spanning 16 orders of
   magnitude, coupling scaled to the geometric mean). The recon gate is ‖A‖-
   relative, so the tiny eigenvalues sit at the dense-solver ‖A‖·ε floor. *)
let build_graded n =
  let d =
    Array.init n (fun i ->
        10.0 ** ((16.0 *. (float_of_int i /. float_of_int (n - 1))) -. 8.0))
  in
  let a = Array.make (n * n) 0.0 in
  for i = 0 to n - 1 do
    a.((i * n) + i) <- d.(i)
  done;
  for i = 0 to n - 2 do
    let v = 1e-3 *. sqrt (abs_float (d.(i) *. d.(i + 1))) in
    a.(((i + 1) * n) + i) <- v;
    a.((i * n) + (i + 1)) <- v
  done;
  a

let build_random n =
  let a = Array.make (n * n) 0.0 in
  let seed = ref 20260717 in
  let rnd () =
    seed := ((!seed * 1103515245) + 12345) land 0x7fffffff;
    (float_of_int !seed /. 2147483648.0) -. 0.5
  in
  for i = 0 to n - 1 do
    for j = 0 to i do
      let v = rnd () in
      a.((i * n) + j) <- v;
      a.((j * n) + i) <- v
    done
  done;
  a

(* ── SVD ───────────────────────────────────────────────────────────────────

   A = U diag(S) Vᴴ via the Jordan–Wielandt embedding. Gates: ‖A−U S Vᴴ‖/‖A‖,
   UᴴU=I, Vᴴ(Vᴴ)ᴴ=I, S descending ≥0, and (fixtures) S matches an offline numpy
   np.linalg.svd oracle. *)
let test_svd_real (type b) ~(kind : (float, b) Buf.kind) ~name ~tol ~m ~n ~full
    ?a ?expect_s () =
  let a =
    match a with
    | Some x -> x
    | None ->
        Array.init (m * n) (fun i ->
            sin (float_of_int (i * 13 mod 251)) +. (0.1 *. cos (float_of_int i)))
  in
  let k = min m n in
  let ncu = if full then m else k in
  let nrv = if full then n else k in
  let abuf = Buf.create kind (m * n) in
  for t = 0 to (m * n) - 1 do
    Buf.set abuf t a.(t)
  done;
  let ubuf = Buf.create kind (m * ncu)
  and sbuf = Buf.create Buf.float64 k
  and vtbuf = Buf.create kind (nrv * n) in
  svd
    (ffi ubuf [| m; ncu |] (contig [| m; ncu |]))
    (ffi sbuf [| k |] [| 1 |])
    (ffi vtbuf [| nrv; n |] (contig [| nrv; n |]))
    (ffi abuf [| m; n |] (contig [| m; n |]));
  let gu i t = Buf.get ubuf ((i * ncu) + t) in
  let gvt t j = Buf.get vtbuf ((t * n) + j) in
  let gs t = Buf.get sbuf t in
  let sok = ref true in
  for t = 0 to k - 1 do
    if gs t < -.tol then sok := false;
    if t > 0 && gs (t - 1) < gs t -. tol then sok := false
  done;
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0.0 in
      for t = 0 to k - 1 do
        s := !s +. (gu i t *. gs t *. gvt t j)
      done;
      let d = !s -. a.((i * n) + j) in
      num := !num +. (d *. d);
      den := !den +. (a.((i * n) + j) *. a.((i * n) + j))
    done
  done;
  let recon = sqrt (!num /. !den) in
  let uo = ref 0.0 in
  for c1 = 0 to ncu - 1 do
    for c2 = 0 to ncu - 1 do
      let s = ref 0.0 in
      for i = 0 to m - 1 do
        s := !s +. (gu i c1 *. gu i c2)
      done;
      let e = abs_float (!s -. if c1 = c2 then 1.0 else 0.0) in
      if e > !uo then uo := e
    done
  done;
  let vo = ref 0.0 in
  for r1 = 0 to nrv - 1 do
    for r2 = 0 to nrv - 1 do
      let s = ref 0.0 in
      for j = 0 to n - 1 do
        s := !s +. (gvt r1 j *. gvt r2 j)
      done;
      let e = abs_float (!s -. if r1 = r2 then 1.0 else 0.0) in
      if e > !vo then vo := e
    done
  done;
  let smatch =
    match expect_s with
    | None -> true
    | Some es ->
        let ok = ref true in
        Array.iteri
          (fun t v ->
            if abs_float (gs t -. v) > tol *. (abs_float v +. 1.0) then
              ok := false)
          es;
        !ok
  in
  ok
    (Printf.sprintf
       "%s svd %dx%d full=%b recon=%.2e Uo=%.2e Vo=%.2e Sok=%b Smatch=%b" name m
       n full recon !uo !vo !sok smatch)
    (recon <= tol && !uo <= tol && !vo <= tol && !sok && smatch)

let test_svd_complex (type b) ~(kind : (Complex.t, b) Buf.kind) ~name ~tol ~m ~n
    ?(full = false) () =
  let a =
    Array.init (m * n) (fun i ->
        {
          Complex.re = sin (float_of_int (i * 13 mod 251));
          im = 0.3 *. cos (float_of_int (i * 7 mod 241));
        })
  in
  let k = min m n in
  let ncu = if full then m else k in
  let nrv = if full then n else k in
  let abuf = Buf.create kind (m * n) in
  for t = 0 to (m * n) - 1 do
    Buf.set abuf t a.(t)
  done;
  let ubuf = Buf.create kind (m * ncu)
  and sbuf = Buf.create Buf.float64 k
  and vtbuf = Buf.create kind (nrv * n) in
  svd
    (ffi ubuf [| m; ncu |] (contig [| m; ncu |]))
    (ffi sbuf [| k |] [| 1 |])
    (ffi vtbuf [| nrv; n |] (contig [| nrv; n |]))
    (ffi abuf [| m; n |] (contig [| m; n |]));
  let gu i t = Buf.get ubuf ((i * ncu) + t) in
  let gvt t j = Buf.get vtbuf ((t * n) + j) in
  let gs t = Buf.get sbuf t in
  let cabs2 (z : Complex.t) = (z.re *. z.re) +. (z.im *. z.im) in
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref Complex.zero in
      for t = 0 to k - 1 do
        s :=
          Complex.add !s
            (Complex.mul (gu i t)
               (Complex.mul { Complex.re = gs t; im = 0.0 } (gvt t j)))
      done;
      num := !num +. cabs2 (Complex.sub !s a.((i * n) + j));
      den := !den +. cabs2 a.((i * n) + j)
    done
  done;
  let recon = sqrt (!num /. !den) in
  (* UᴴU = I *)
  let uo = ref 0.0 in
  for c1 = 0 to ncu - 1 do
    for c2 = 0 to ncu - 1 do
      let s = ref Complex.zero in
      for i = 0 to m - 1 do
        s := Complex.add !s (Complex.mul (Complex.conj (gu i c1)) (gu i c2))
      done;
      let e =
        sqrt
          (cabs2
             (Complex.sub !s (if c1 = c2 then Complex.one else Complex.zero)))
      in
      if e > !uo then uo := e
    done
  done;
  (* Vᴴ rows orthonormal: Σ_j Vᴴ[a][j]·conj(Vᴴ[b][j]) = δ *)
  let vo = ref 0.0 in
  for r1 = 0 to nrv - 1 do
    for r2 = 0 to nrv - 1 do
      let s = ref Complex.zero in
      for j = 0 to n - 1 do
        s := Complex.add !s (Complex.mul (gvt r1 j) (Complex.conj (gvt r2 j)))
      done;
      let e =
        sqrt
          (cabs2
             (Complex.sub !s (if r1 = r2 then Complex.one else Complex.zero)))
      in
      if e > !vo then vo := e
    done
  done;
  ok
    (Printf.sprintf "%s svd %dx%d full=%b recon=%.2e Uo=%.2e Vo=%.2e" name m n
       full recon !uo !vo)
    (recon <= tol && !uo <= tol && !vo <= tol)

(* Scale-aware relative-accuracy gate against an offline numpy oracle: each σ
   matches to [reltol] relatively, with an [abstol]·σ_max absolute floor for the
   values that sit near the Householder-bidiagonalization noise level. The
   Demmel–Kahan zero-shift sweep is what keeps the small σ (well above that
   floor) to high RELATIVE accuracy — the property the eigh-embedding lacked. *)
let test_svd_relacc ~name ~m ~n ~a ~es ~reltol ~abstol () =
  let k = min m n in
  let abuf = Buf.create Buf.float64 (m * n) in
  Array.iteri (fun t v -> Buf.set abuf t v) a;
  let ubuf = Buf.create Buf.float64 (m * k)
  and sbuf = Buf.create Buf.float64 k
  and vtbuf = Buf.create Buf.float64 (k * n) in
  svd
    (ffi ubuf [| m; k |] (contig [| m; k |]))
    (ffi sbuf [| k |] [| 1 |])
    (ffi vtbuf [| k; n |] (contig [| k; n |]))
    (ffi abuf [| m; n |] (contig [| m; n |]));
  let smax = es.(0) in
  let bad = ref 0 and worst = ref 0.0 in
  Array.iteri
    (fun t v ->
      let ad = abs_float (Buf.get sbuf t -. v) in
      if ad > (reltol *. abs_float v) +. (abstol *. smax) then incr bad;
      let rel = ad /. (abs_float v +. (abstol *. smax)) in
      if rel > !worst then worst := rel)
    es;
  ok
    (Printf.sprintf "%s svd %dx%d rel-acc worst=%.2e (bad=%d)" name m n !worst
       !bad)
    (!bad = 0)

let svd_fix_graded5 =
  [|
    32.386797746303365;
    16.827033870464344;
    14.584698473625766;
    -18.850569830179445;
    40.42840547068902;
    0.046132882854762004;
    0.02344450153202795;
    0.015648132918435656;
    -0.024303383986049882;
    0.043136264240973894;
    -12.129897333388689;
    -5.889762551323407;
    -5.79014615560565;
    6.926071545200442;
    -14.853552436553978;
    -27.77607326992134;
    -14.616090675580534;
    -12.360592663421496;
    16.2264304812836;
    -34.79871712862178;
    31.88425186254641;
    15.835254878984497;
    14.940170870726424;
    -18.3212109624654;
    39.29466835322517;
  |]

let svd_fix_clustered4 =
  [|
    1.0319022220349705;
    -1.1692421800252757;
    -1.2519467991343354;
    -0.02301362382118173;
    1.4605164703680484;
    1.1209106287001123;
    0.16983450111505743;
    -0.7556567362558355;
    -0.4701649161878702;
    0.1385554176909072;
    -0.48267787450226246;
    -1.0258956403995578;
    -0.5414448548498132;
    -0.4068288671037338;
    -0.049613307019892874;
    -1.5079502874438235;
  |]

let svd_fix_tall63 =
  [|
    0.4155026142980106;
    -0.9235446567132176;
    -0.19602731230796547;
    -0.5907698188631366;
    -0.29971123732782745;
    1.296885192726673;
    1.5295796333931557;
    0.6694181934096611;
    0.5487451197783295;
    0.6766289895558858;
    -0.012242186606443108;
    -0.07566346148706628;
    -0.6736451873760739;
    -0.05586745005007389;
    2.2599469866262614;
    0.8690393292538257;
    -0.3421170234268632;
    -0.4719266521335216;
  |]

(* Offline np.linalg.svd oracles (seed 20260717) for the rewrite's new gates.
   dyn5: σ spanning 1e8..1e-8 (dense, rotated) — relative accuracy on the
   big/mid σ, absolute floor on the smallest (‖A‖·ε ≈ 1e-8 bounds any dense
   solver, numpy included). clus6: three-fold σ=3 cluster plus a 1e-7 pair
   (deflation + near-zero). rank54: 5×4 rank-3, one exactly-zero σ. *)
let svd_fix_dyn5 =
  [|
    12539723.02952069;
    -17505815.700353518;
    8917634.289808983;
    -16937631.63217063;
    4742540.701164636;
    -2762955.81560511;
    3857177.694615324;
    -1964885.2312229422;
    3731986.3108871873;
    -1045011.2446501966;
    -31291444.481013276;
    43683751.27652291;
    -22252930.88896055;
    42265915.521155044;
    -11834427.366774203;
    17868284.075847495;
    -24944618.11789646;
    12707031.17294191;
    -24134993.23686137;
    6757697.663380319;
    -19519069.286400933;
    27249173.575499415;
    -13880995.406145057;
    26364750.557379056;
    -7382084.710354824;
  |]

let svd_fix_dyn5_s =
  [|
    99999999.99999999;
    100.00000000017037;
    1.000000000739783;
    9.999692044246688e-05;
    1.0152727663761473e-08;
  |]

let svd_fix_clus6 =
  [|
    1.3762482976891157;
    -1.211578303678782;
    1.7029830637032164;
    1.1969969710698656;
    -0.23525908046506158;
    -0.3296615614040424;
    -0.6749225211531032;
    -0.46187176340362057;
    -0.16028257965153886;
    0.5914655454141511;
    -1.750989547527381;
    1.5334592779542342;
    2.0410556265995954;
    -0.2994038720024088;
    -1.3925101322350015;
    -1.0694069124241379;
    -1.0384666443579424;
    0.2936203371031439;
    -0.25987274966933854;
    -0.6227662353143555;
    -1.453093833319683;
    0.8234424169539848;
    0.5980158161545885;
    0.4190279860305053;
    0.9063803901859939;
    -0.7608715199431577;
    -0.17622785291524268;
    0.578868492851017;
    0.21671273669207655;
    -0.10349301644334462;
    -0.6191436082567583;
    -0.35719544629461536;
    0.22807085774378452;
    0.6553843800842688;
    -0.9070567424045388;
    0.8710376317212473;
  |]

let svd_fix_clus6_s =
  [|
    2.999999999999999;
    2.9999999999999987;
    2.999999999999998;
    1.9999999999999998;
    1.0000000006201816e-07;
    9.999999990522373e-08;
  |]

let svd_fix_rank54 =
  [|
    -2.1696103122284724;
    3.9831908107256426;
    2.3661310821741566;
    2.9319476599936753;
    -0.4973397612223236;
    1.2107073289003105;
    0.7855771904017829;
    2.3244441993188394;
    0.020612554164313977;
    1.8714810832971354;
    0.9543147010704648;
    -0.027118102233273104;
    3.596196166404184;
    1.3673206175009527;
    0.44047011478031906;
    0.32674110459701894;
    -1.2435439739734906;
    0.8178173438705062;
    0.38894380501552506;
    -2.275394443284693;
  |]

let svd_fix_rank54_s =
  [|
    6.61055515551338;
    4.151029831382346;
    2.938020265275789;
    9.106250838955923e-17;
  |]

(* Offline np.linalg.svd U/S/Vᴴ literal oracles, sign-pinned (each Vᴴ row's
   dominant-magnitude entry made positive, U column flipped to match) and
   order-pinned (S descending) — deterministic because each fixture has DISTINCT
   singular values. Element-wise, not reconstruction: pins the emitted vectors
   and the Vᴴ (not V) convention, independent of this backend. [usv_*_jmax] is
   the per-row dominant column used to canonicalize the backend output the same
   way. usv_wit3 is the reviewer's idir==2 witness; usv_wide35 is m<n. *)
let usv_m4x3_a =
  [| 1.0; 2.0; -1.0; 0.0; 3.0; 1.0; 2.0; -2.0; 1.0; 1.0; 0.0; 4.0 |]

let usv_m4x3_s = [| 4.616105113376634; 4.072590492050938; 2.02622315313847 |]

let usv_m4x3_u =
  [|
    -0.27185566989730237;
    0.3629941087048192;
    0.7386039781731422;
    -0.04265797572872631;
    0.7748371448568426;
    0.028984263161124287;
    0.48751552783020147;
    -0.3606731510041366;
    0.6580809252062578;
    0.8286153522448495;
    0.3711839896068217;
    -0.1433651693944226;
  |]

let usv_m4x3_vh =
  [|
    0.3318361909847113;
    -0.35673284775714975;
    0.8732848433837794;
    0.0031507700880835134;
    0.9261539949451082;
    0.3771324041965486;
    0.9433317629554355;
    0.12239466074257882;
    -0.30845539713400827;
  |]

let usv_m4x3_jmax = [| 2; 1; 0 |]
let usv_wit3_a = [| -1.0; 0.0; 1.0; 0.0; -2.0; 1.0; 0.0; 0.0; 1.0 |]
let usv_wit3_s = [| 2.376078978288514; 1.4142135623730943; 0.5951879442120863 |]

let usv_wit3_u =
  [|
    -0.32198227875427493;
    0.816496580927726;
    -0.4792293245425806;
    -0.9089159362082553;
    -0.40824829046386313;
    -0.08488317995930554;
    -0.2649513786997062;
    0.4082482904638631;
    0.8735754691258553;
  |]

let usv_wit3_vh =
  [|
    0.13550992273253398;
    0.7650553239294643;
    -0.6295454011969308;
    -0.5773502691896255;
    0.577350269189626;
    0.5773502691896255;
    0.8051731040637718;
    0.28523151648064504;
    0.5199415875831266;
  |]

let usv_wit3_jmax = [| 1; 1; 0 |]

let usv_wide35_a =
  [|
    2.0; 1.0; 0.0; -1.0; 3.0; 1.0; -2.0; 1.0; 0.0; 1.0; 0.0; 1.0; 4.0; -1.0; 2.0;
  |]

let usv_wide35_s = [| 5.328793586020537; 3.129271756205937; 2.410729597735552 |]

let usv_wide35_u =
  [|
    0.5363139419060934;
    -0.822405346846727;
    0.18978092948202696;
    0.227165194472677;
    -0.07590566086922558;
    -0.9708935601126402;
    0.8128735019283987;
    0.5638153741482408;
    0.1461126063580448;
  |]

let usv_wide35_vh =
  [|
    0.24391882652289582;
    0.16792863908947245;
    0.6528042691148939;
    -0.2531881601445263;
    0.6496506138142639;
    -0.5498775717226141;
    -0.034122524113884764;
    0.6964418578864765;
    0.08263583122355979;
    -0.4523387750858456;
    -0.2452915920989384;
    0.9448096784495553;
    -0.16030131917053486;
    -0.13933272987380393;
    -0.04534957344579904;
  |]

let usv_wide35_jmax = [| 2; 2; 1 |]

(* Offline np.linalg.svd S oracles for the divide-and-conquer path (min(m,n) >
   SMLSIZ=25). dc40/dc_tall/dc_wide use test_svd_real's default deterministic
   generator, reproduced verbatim in numpy — dc40 is naturally rank-deficient
   (six σ at the ε floor). The bd_* fixtures are dense upper bidiagonals built
   from closed-form d/e (formulas duplicated in the harness below), so they hit
   the D&C on chosen spectra: clus40 = near-equal d with tiny e (maximal
   deflation), grad40 = 1e-8..1e8 grading, zero33 = exact-zero d and e entries
   (splits + zero σ), wilk41 = Wilkinson-analog |i-20| diagonal (paired
   near-equal σ + a zero). *)
let svd_dc40_s =
  [|
    20.17424707200857;
    19.640233723255268;
    2.14293702896625;
    2.0428112269319865;
    1.251429157347747;
    1.0855976090341564;
    0.9823667157161405;
    0.9606224225815796;
    0.5369478679265075;
    0.51456563036363;
    0.45401133983339204;
    0.4057044074696483;
    0.395285209380375;
    0.3465289449356177;
    0.31541992014607273;
    0.29749178774553026;
    0.27044756766215233;
    0.2626909557108073;
    0.2536023890760674;
    0.2346509742465725;
    0.20112849623131365;
    0.17961403135883136;
    0.1706197379410516;
    0.15013651620182156;
    0.14355990372715594;
    0.1339544070884951;
    0.12640922830911336;
    0.12097321775360632;
    0.08886418189875162;
    0.07581233787137467;
    0.04781813440772818;
    0.03185734780313648;
    0.010884003124594553;
    0.007415655865180321;
    4.0586153625248276e-16;
    2.9095790411778316e-16;
    1.9275379222239941e-16;
    8.223206559820745e-17;
    6.51617069665745e-17;
    3.7107304145228335e-17;
  |]

let svd_dc_tall_s =
  [|
    25.580305193011746;
    24.784974333367693;
    2.6435276042711364;
    2.5820432538502183;
    1.5746785202455242;
    1.3778987929812676;
    1.2541943672506817;
    1.2345775052331636;
    0.6947793110763337;
    0.6136110693682875;
    0.5445036467263014;
    0.5115428999518211;
    0.4791871541038212;
    0.42607303645958516;
    0.36936792427037834;
    0.3614688691601677;
    0.3425285963252584;
    0.322464502008018;
    0.3111777377094667;
    0.2978470094907816;
    0.26558139043744405;
    0.260093260730698;
    0.2315670372855433;
    0.21855166961389708;
    0.20441737974727126;
    0.19405446901595777;
    0.17849274925588599;
    0.17501652450489474;
    0.1639750503258515;
    0.1525675912901452;
    0.1381845513517985;
    0.11222156843789756;
    0.10535681261253438;
    0.08127876781619013;
    0.061551765018734036;
    0.04720327070622464;
    0.024686260911173322;
    0.013779632932317449;
    5.276318102462372e-16;
    4.742027290528569e-16;
  |]

let svd_dc_wide_s =
  [|
    25.255846988214795;
    25.110900898312718;
    2.6265964182040475;
    2.552415697832638;
    1.4672636143478575;
    1.3595548138440094;
    1.2652988442095172;
    1.214995310112537;
    0.7004461167602294;
    0.6810892689806833;
    0.6278037624658546;
    0.5844902774548739;
    0.48552486726546606;
    0.4605299249189175;
    0.444118655908279;
    0.4211627013694147;
    0.3920704651788255;
    0.38541573669574364;
    0.34243452672159297;
    0.3157505330578907;
    0.28318706067948574;
    0.27833547044591544;
    0.25874442824616956;
    0.2505398944523026;
    0.23922135694885915;
    0.2338456499555341;
    0.2265900038566958;
    0.2121116309906526;
    0.20789135481524937;
    0.20506412704969895;
    0.19419468721404168;
    0.19257969634030303;
    0.1856796713373598;
    0.18431854753479482;
    0.17423547760030214;
    0.1583939277764557;
    0.14423249101836508;
    0.12222477657189072;
    0.08658145934115978;
    0.07570371647351017;
  |]

let svd_bd_clus40_s =
  [|
    1.0000000000004552;
    1.0000000000004277;
    1.0000000000003986;
    1.000000000000378;
    1.0000000000003681;
    1.0000000000003455;
    1.0000000000003382;
    1.0000000000003162;
    1.0000000000003086;
    1.000000000000307;
    1.000000000000286;
    1.0000000000002782;
    1.0000000000002756;
    1.0000000000002558;
    1.0000000000002482;
    1.0000000000002456;
    1.0000000000002258;
    1.0000000000002183;
    1.0000000000002156;
    1.0000000000001958;
    1.0000000000001883;
    1.0000000000001856;
    1.0000000000001659;
    1.0000000000001585;
    1.000000000000156;
    1.000000000000136;
    1.0000000000001286;
    1.0000000000001261;
    1.0000000000001057;
    1.000000000000098;
    1.0000000000000957;
    1.0000000000000755;
    1.0000000000000657;
    1.0000000000000475;
    1.000000000000036;
    1.000000000000026;
    1.0000000000000062;
    0.9999999999999889;
    0.999999999999975;
    0.999999999999944;
  |]

let svd_bd_grad40_s =
  [|
    100000022.90322469;
    38881551.80308503;
    15117750.706156598;
    5878016.072274924;
    2285463.864134994;
    888623.8162743407;
    345510.7294592218;
    134339.93325988983;
    52233.45074266833;
    20309.17620904739;
    7896.522868499733;
    3070.29062975785;
    1193.7766417144355;
    464.1588833612773;
    180.4721766827174;
    70.17038286703837;
    27.283333764867706;
    10.608183551394484;
    4.124626382901348;
    1.6037187437513274;
    0.6235507341273916;
    0.24244620170823314;
    0.0942668455117885;
    0.036652412370796264;
    0.014251026703029992;
    0.005541020330009493;
    0.002154434690031882;
    0.0008376776400682923;
    0.0003257020655659783;
    0.00012663801734674022;
    4.923882631706742e-05;
    1.9144819761699567e-05;
    7.443803013251697e-06;
    2.8942661247167537e-06;
    1.1253355826007646e-06;
    4.3754793750741814e-07;
    1.7012542798525891e-07;
    6.614740641230145e-08;
    2.57191380905934e-08;
    9.999997709677007e-09;
  |]

let svd_bd_zero33_s =
  [|
    1.6839902448903397;
    1.6126256353992048;
    1.5943055651751838;
    1.5850523995755001;
    1.5366302774637621;
    1.444177351153566;
    1.3671706449867962;
    1.3384978042100268;
    1.2581061847874977;
    1.2467485554070765;
    1.214284996995534;
    1.1740301046398476;
    1.1470415351593835;
    1.102589253529079;
    1.101940121030122;
    1.0629781329062375;
    1.009396599291299;
    1.0016612179567417;
    0.7662832160350473;
    0.705225469734421;
    0.6561439274729741;
    0.5483791568402014;
    0.5318729597193121;
    0.46464434623681933;
    0.4103381307826223;
    0.2422633969559378;
    0.19755340972195773;
    0.17433822959296214;
    0.17165146097483558;
    0.09472915460929923;
    0.033210166904989645;
    0.0;
    0.0;
  |]

let svd_bd_wilk41_s =
  [|
    20.231765920239;
    20.231765920239;
    19.036585297985386;
    19.036585297985383;
    18.01504909703887;
    18.01504909703887;
    17.014744407790822;
    17.01474440779082;
    16.015636893688484;
    16.015636893688473;
    15.016680555438686;
    15.016680555438679;
    14.017874218411217;
    14.017874218411217;
    13.019252094705747;
    13.01925209470574;
    12.020860444516636;
    12.020860444516632;
    11.02276246662289;
    11.022762466622888;
    10.025046836465645;
    10.025046836465645;
    9.027842013151702;
    9.027842013151702;
    8.03134143610338;
    8.031341436103379;
    7.035850721802344;
    7.035850721802342;
    6.041883198534905;
    6.041883198534904;
    5.050373826697282;
    5.05037382669728;
    4.063229009991975;
    4.063229009991974;
    3.085057131288046;
    3.0850571312880453;
    2.1308656387760907;
    2.1308656387760907;
    1.2834973309298923;
    1.283497330929892;
    0.0;
  |]

(* Element-wise oracle: run the backend, canonicalize its sign per [jmax], then
   compare S, U, Vᴴ to the numpy literals. *)
let test_svd_usv ~name ~a ~s ~u ~vh ~jmax ~m ~n ~tol () =
  let k = min m n in
  let abuf = Buf.create Buf.float64 (m * n) in
  Array.iteri (fun t v -> Buf.set abuf t v) a;
  let ubuf = Buf.create Buf.float64 (m * k)
  and sbuf = Buf.create Buf.float64 k
  and vtbuf = Buf.create Buf.float64 (k * n) in
  svd
    (ffi ubuf [| m; k |] (contig [| m; k |]))
    (ffi sbuf [| k |] [| 1 |])
    (ffi vtbuf [| k; n |] (contig [| k; n |]))
    (ffi abuf [| m; n |] (contig [| m; n |]));
  let gu = Array.make (m * k) 0.0 and gv = Array.make (k * n) 0.0 in
  for t = 0 to k - 1 do
    let sgn = if Buf.get vtbuf ((t * n) + jmax.(t)) < 0.0 then -1.0 else 1.0 in
    for j = 0 to n - 1 do
      gv.((t * n) + j) <- sgn *. Buf.get vtbuf ((t * n) + j)
    done;
    for i = 0 to m - 1 do
      gu.((i * k) + t) <- sgn *. Buf.get ubuf ((i * k) + t)
    done
  done;
  let bad = ref 0 and worst = ref 0.0 in
  let chk got exp =
    let e = abs_float (got -. exp) in
    if e > tol then incr bad;
    if e > !worst then worst := e
  in
  Array.iteri (fun t v -> chk (Buf.get sbuf t) v) s;
  Array.iteri (fun idx v -> chk gu.(idx) v) u;
  Array.iteri (fun idx v -> chk gv.(idx) v) vh;
  ok
    (Printf.sprintf "%s svd U/S/Vh literal worst=%.2e bad=%d" name !worst !bad)
    (!bad = 0)

(* idir==2 regression: a matrix whose bidiagonal has |d[first]| < |d[last]|
   takes dbdsqr's backward chase — the path the rotation-pair bug corrupted.
   Gates BOTH reconstruction AND UᵀAV diagonality (the residual off-diagonal the
   bug left, which reconstruction alone can miss when U/Vᴴ stay orthonormal). *)
let test_svd_idir2 ~name ~a ~m ~n ~tol () =
  let k = min m n in
  let abuf = Buf.create Buf.float64 (m * n) in
  Array.iteri (fun t v -> Buf.set abuf t v) a;
  let ubuf = Buf.create Buf.float64 (m * k)
  and sbuf = Buf.create Buf.float64 k
  and vtbuf = Buf.create Buf.float64 (k * n) in
  svd
    (ffi ubuf [| m; k |] (contig [| m; k |]))
    (ffi sbuf [| k |] [| 1 |])
    (ffi vtbuf [| k; n |] (contig [| k; n |]))
    (ffi abuf [| m; n |] (contig [| m; n |]));
  let gu i t = Buf.get ubuf ((i * k) + t)
  and gvt t j = Buf.get vtbuf ((t * n) + j)
  and gs t = Buf.get sbuf t in
  let num = ref 0.0 and den = ref 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0.0 in
      for t = 0 to k - 1 do
        s := !s +. (gu i t *. gs t *. gvt t j)
      done;
      let d = !s -. a.((i * n) + j) in
      num := !num +. (d *. d);
      den := !den +. (a.((i * n) + j) *. a.((i * n) + j))
    done
  done;
  let recon = sqrt (!num /. !den) in
  (* M[s][t] = Uᵀ A V, V[j][t] = Vᴴ[t][j]; off-diagonal must vanish *)
  let offdiag = ref 0.0 in
  for s = 0 to k - 1 do
    for t = 0 to k - 1 do
      if s <> t then begin
        let v = ref 0.0 in
        for i = 0 to m - 1 do
          for j = 0 to n - 1 do
            v := !v +. (gu i s *. a.((i * n) + j) *. gvt t j)
          done
        done;
        if abs_float !v > !offdiag then offdiag := abs_float !v
      end
    done
  done;
  ok
    (Printf.sprintf "%s svd idir2 recon=%.2e offdiag=%.2e" name recon !offdiag)
    (recon <= tol && !offdiag <= tol)

(* ── SVD divide-and-conquer gauntlet ──────────────────────────────────────

   min(m,n) > SMLSIZ=25 forces the dbdsdc/dlasd path in nx_c_svd.c.
   Reconstruction + orthogonality alone can miss role/pairing bugs (a U/Vᴴ swap
   once survived them), so these gates PIN the roles: per-vector residuals
   ‖A·vₜ−σₜ·uₜ‖ and ‖Aᵀ·uₜ−σₜ·vₜ‖, full UᵀAV diagonality (off-diagonal ≤
   tol·σmax and diagonal matches S), orthogonality of both factors, descending
   nonnegative S, and (where given) an offline numpy S oracle at |σₜ−oracle| ≤
   tol·(|oracle|+1). *)
let test_svd_dc ~name ~tol ~m ~n ?(full = false) ?a ?expect_s () =
  let a =
    match a with
    | Some x -> x
    | None ->
        Array.init (m * n) (fun i ->
            sin (float_of_int (i * 13 mod 251)) +. (0.1 *. cos (float_of_int i)))
  in
  let k = min m n in
  let ncu = if full then m else k in
  let nrv = if full then n else k in
  let abuf = Buf.create Buf.float64 (m * n) in
  Array.iteri (fun t v -> Buf.set abuf t v) a;
  let ubuf = Buf.create Buf.float64 (m * ncu)
  and sbuf = Buf.create Buf.float64 k
  and vtbuf = Buf.create Buf.float64 (nrv * n) in
  svd
    (ffi ubuf [| m; ncu |] (contig [| m; ncu |]))
    (ffi sbuf [| k |] [| 1 |])
    (ffi vtbuf [| nrv; n |] (contig [| nrv; n |]))
    (ffi abuf [| m; n |] (contig [| m; n |]));
  let gu i t = Buf.get ubuf ((i * ncu) + t)
  and gvt t j = Buf.get vtbuf ((t * n) + j)
  and gs t = Buf.get sbuf t in
  let smax = max (gs 0) 1e-30 in
  let sok = ref true in
  for t = 0 to k - 1 do
    if gs t < 0.0 then sok := false;
    if t > 0 && gs (t - 1) < gs t then sok := false
  done;
  let pv = ref 0.0 in
  for t = 0 to k - 1 do
    let e1 = ref 0.0 in
    for i = 0 to m - 1 do
      let av = ref 0.0 in
      for j = 0 to n - 1 do
        av := !av +. (a.((i * n) + j) *. gvt t j)
      done;
      let d = !av -. (gs t *. gu i t) in
      e1 := !e1 +. (d *. d)
    done;
    let e2 = ref 0.0 in
    for j = 0 to n - 1 do
      let au = ref 0.0 in
      for i = 0 to m - 1 do
        au := !au +. (a.((i * n) + j) *. gu i t)
      done;
      let d = !au -. (gs t *. gvt t j) in
      e2 := !e2 +. (d *. d)
    done;
    let e = sqrt (max !e1 !e2) /. smax in
    if e > !pv then pv := e
  done;
  let offd = ref 0.0 and diagd = ref 0.0 in
  for s = 0 to k - 1 do
    for t = 0 to k - 1 do
      let v = ref 0.0 in
      for i = 0 to m - 1 do
        for j = 0 to n - 1 do
          v := !v +. (gu i s *. a.((i * n) + j) *. gvt t j)
        done
      done;
      if s = t then (
        let e = abs_float (!v -. gs t) /. smax in
        if e > !diagd then diagd := e)
      else
        let e = abs_float !v /. smax in
        if e > !offd then offd := e
    done
  done;
  let uo = ref 0.0 in
  for c1 = 0 to ncu - 1 do
    for c2 = 0 to ncu - 1 do
      let s = ref 0.0 in
      for i = 0 to m - 1 do
        s := !s +. (gu i c1 *. gu i c2)
      done;
      let e = abs_float (!s -. if c1 = c2 then 1.0 else 0.0) in
      if e > !uo then uo := e
    done
  done;
  let vo = ref 0.0 in
  for r1 = 0 to nrv - 1 do
    for r2 = 0 to nrv - 1 do
      let s = ref 0.0 in
      for j = 0 to n - 1 do
        s := !s +. (gvt r1 j *. gvt r2 j)
      done;
      let e = abs_float (!s -. if r1 = r2 then 1.0 else 0.0) in
      if e > !vo then vo := e
    done
  done;
  let smatch = ref true in
  (match expect_s with
  | None -> ()
  | Some es ->
      Array.iteri
        (fun t v ->
          if abs_float (gs t -. v) > tol *. (abs_float v +. 1.0) then
            smatch := false)
        es);
  ok
    (Printf.sprintf
       "%s svd-dc %dx%d full=%b pervec=%.2e offd=%.2e diagd=%.2e Uo=%.2e \
        Vo=%.2e Sok=%b Smatch=%b"
       name m n full !pv !offd !diagd !uo !vo !sok !smatch)
    (!pv <= tol && !offd <= tol && !diagd <= tol && !uo <= tol && !vo <= tol
   && !sok && !smatch)

(* Complex D&C: the real double core plus the complex lift/back-multiply and the
   conjugated Qp materialization in la_sd_apply. Same role-pinning gates (A·vₜ =
   σₜ·uₜ, Aᴴ·uₜ = σₜ·vₜ with vₜ = conj(Vᴴ row t)). *)
let test_svd_dc_c ~name ~tol ~m ~n () =
  let a =
    Array.init (m * n) (fun i ->
        {
          Complex.re = sin (float_of_int (i * 13 mod 251));
          im = 0.3 *. cos (float_of_int (i * 7 mod 241));
        })
  in
  let k = min m n in
  let abuf = Buf.create Buf.complex128 (m * n) in
  Array.iteri (fun t v -> Buf.set abuf t v) a;
  let ubuf = Buf.create Buf.complex128 (m * k)
  and sbuf = Buf.create Buf.float64 k
  and vtbuf = Buf.create Buf.complex128 (k * n) in
  svd
    (ffi ubuf [| m; k |] (contig [| m; k |]))
    (ffi sbuf [| k |] [| 1 |])
    (ffi vtbuf [| k; n |] (contig [| k; n |]))
    (ffi abuf [| m; n |] (contig [| m; n |]));
  let gu i t : Complex.t = Buf.get ubuf ((i * k) + t)
  and gvt t j : Complex.t = Buf.get vtbuf ((t * n) + j)
  and gs t = Buf.get sbuf t in
  let smax = max (gs 0) 1e-30 in
  let pv = ref 0.0 in
  for t = 0 to k - 1 do
    let e1 = ref 0.0 in
    for i = 0 to m - 1 do
      (* Σⱼ aᵢⱼ·conj(vtₜⱼ) − σₜ·uᵢₜ *)
      let re = ref 0.0 and im = ref 0.0 in
      for j = 0 to n - 1 do
        let av = a.((i * n) + j) and vt = gvt t j in
        re := !re +. (av.Complex.re *. vt.Complex.re) +. (av.im *. vt.im);
        im := !im +. (av.im *. vt.Complex.re) -. (av.Complex.re *. vt.im)
      done;
      let u = gu i t in
      let dr = !re -. (gs t *. u.Complex.re) and di = !im -. (gs t *. u.im) in
      e1 := !e1 +. (dr *. dr) +. (di *. di)
    done;
    let e2 = ref 0.0 in
    for j = 0 to n - 1 do
      (* Σᵢ conj(aᵢⱼ)·uᵢₜ − σₜ·conj(vtₜⱼ) *)
      let re = ref 0.0 and im = ref 0.0 in
      for i = 0 to m - 1 do
        let av = a.((i * n) + j) and u = gu i t in
        re := !re +. (av.Complex.re *. u.Complex.re) +. (av.im *. u.im);
        im := !im +. (av.Complex.re *. u.im) -. (av.im *. u.Complex.re)
      done;
      let vt = gvt t j in
      let dr = !re -. (gs t *. vt.Complex.re) and di = !im +. (gs t *. vt.im) in
      e2 := !e2 +. (dr *. dr) +. (di *. di)
    done;
    let e = sqrt (max !e1 !e2) /. smax in
    if e > !pv then pv := e
  done;
  let sok = ref true in
  for t = 0 to k - 1 do
    if gs t < 0.0 then sok := false;
    if t > 0 && gs (t - 1) < gs t then sok := false
  done;
  ok
    (Printf.sprintf "%s svd-dc %dx%d pervec=%.2e Sok=%b" name m n !pv !sok)
    (!pv <= tol && !sok)

(* Batched D&C: independent matrices per batch element through the pooled
   driver; per-matrix per-vector gates. *)
let test_svd_dc_batched ~m ~n ~batch () =
  let k = min m n in
  let a =
    Array.init batch (fun b ->
        Array.init (m * n) (fun i ->
            sin (float_of_int ((i + (37 * b)) * 13 mod 251))
            +. (0.1 *. cos (float_of_int (i + b)))))
  in
  let abuf = Buf.create Buf.float64 (batch * m * n) in
  Array.iteri
    (fun b ab -> Array.iteri (fun t v -> Buf.set abuf ((b * m * n) + t) v) ab)
    a;
  let ubuf = Buf.create Buf.float64 (batch * m * k)
  and sbuf = Buf.create Buf.float64 (batch * k)
  and vtbuf = Buf.create Buf.float64 (batch * k * n) in
  svd
    (ffi ubuf [| batch; m; k |] (contig [| batch; m; k |]))
    (ffi sbuf [| batch; k |] (contig [| batch; k |]))
    (ffi vtbuf [| batch; k; n |] (contig [| batch; k; n |]))
    (ffi abuf [| batch; m; n |] (contig [| batch; m; n |]));
  let worst = ref 0.0 in
  for b = 0 to batch - 1 do
    let gu i t = Buf.get ubuf ((b * m * k) + (i * k) + t)
    and gvt t j = Buf.get vtbuf ((b * k * n) + (t * n) + j)
    and gs t = Buf.get sbuf ((b * k) + t) in
    let smax = max (gs 0) 1e-30 in
    for t = 0 to k - 1 do
      let e1 = ref 0.0 in
      for i = 0 to m - 1 do
        let av = ref 0.0 in
        for j = 0 to n - 1 do
          av := !av +. (a.(b).((i * n) + j) *. gvt t j)
        done;
        let d = !av -. (gs t *. gu i t) in
        e1 := !e1 +. (d *. d)
      done;
      let e2 = ref 0.0 in
      for j = 0 to n - 1 do
        let au = ref 0.0 in
        for i = 0 to m - 1 do
          au := !au +. (a.(b).((i * n) + j) *. gu i t)
        done;
        let d = !au -. (gs t *. gvt t j) in
        e2 := !e2 +. (d *. d)
      done;
      let e = sqrt (max !e1 !e2) /. smax in
      if e > !worst then worst := e
    done
  done;
  ok
    (Printf.sprintf "f64 svd-dc batched=%d %dx%d pervec=%.2e" batch m n !worst)
    (!worst <= 1e-9)

(* Signed zero singular value: dbdsqr.f normalizes D(I)==0 to +0 ("Avoid -ZERO")
   before its `< 0` sign fold, which alone lets -0.0 through. The matrix
   [[1,1],[0,-0.0]] survives gebrd unchanged (zero-tail larfg keeps tau=0 and
   the diagonal as-is), so the bidiagonal SVD's 2x2 la_lasv2 deflation sees
   h=-0.0 and its ssmin copysign yields an exact -0.0 — the pre-fix S was [sqrt
   2, -0.0]. The 1/s sign probe has teeth: 1/(-0.) is -infinity. *)
let test_svd_signed_zero (type b) ~(kind : (float, b) Buf.kind) ~name () =
  let n = 2 in
  let a = [| 1.0; 1.0; 0.0; -0.0 |] in
  let abuf = Buf.create kind (n * n) in
  Array.iteri (fun t v -> Buf.set abuf t v) a;
  let ubuf = Buf.create kind (n * n)
  and sbuf = Buf.create Buf.float64 n
  and vtbuf = Buf.create kind (n * n) in
  svd
    (ffi ubuf [| n; n |] (contig [| n; n |]))
    (ffi sbuf [| n |] [| 1 |])
    (ffi vtbuf [| n; n |] (contig [| n; n |]))
    (ffi abuf [| n; n |] (contig [| n; n |]));
  let s0 = Buf.get sbuf 0 and s1 = Buf.get sbuf 1 in
  ok
    (Printf.sprintf "%s svd signed-zero S=[%h,%h]" name s0 s1)
    (abs_float (s0 -. sqrt 2.0) <= 1e-6 && s1 = 0.0 && 1.0 /. s1 = infinity)

(* ── Run ──────────────────────────────────────────────────────────────────*)

let test_internal_branches () =
  List.iter
    (fun n ->
      List.iter
        (fun upper ->
          test_chol_real ~kind:Buf.float64 ~name:"f64" ~tol:1e-10 ~n ~upper ();
          test_chol_real ~kind:Buf.float32 ~name:"f32" ~tol:1e-3 ~n ~upper ();
          test_chol_complex ~kind:Buf.complex128 ~name:"c64" ~tol:1e-10 ~n
            ~upper ();
          test_chol_complex ~kind:Buf.complex64 ~name:"c32" ~tol:1e-3 ~n ~upper
            ())
        [ false; true ])
    (* n>=128 forces the trailing update through nx_c_gemm2d_ct's BLOCKED path
       (64x64x64 >= the 48^3 direct cutoff) — the integration gap the linalg
       reviewer flagged (cholesky's only exercise of the blocked GEMM entry). *)
    [ 1; 2; 3; 5; 8; 16; 33; 64; 100; 128; 200 ];
  (* low precision upcast path (store L in f16 → looser tol) *)
  test_chol_real ~kind:Buf.float16 ~name:"f16" ~tol:5e-2 ~n:16 ~upper:false ();
  test_chol_real ~kind:Buf.bfloat16 ~name:"bf16" ~tol:1e-1 ~n:16 ~upper:false ();

  test_chol_batched ~n:8 ~batch:5 ();
  test_chol_batched ~n:40 ~batch:9 ();

  List.iter
    (fun n ->
      List.iter
        (fun (upper, transpose, unit) ->
          test_trsm_real ~kind:Buf.float64 ~name:"f64" ~tol:1e-9 ~n ~nrhs:3
            ~upper ~transpose ~unit ();
          test_trsm_real ~kind:Buf.float32 ~name:"f32" ~tol:1e-3 ~n ~nrhs:3
            ~upper ~transpose ~unit ())
        [
          (false, false, false);
          (true, false, false);
          (false, true, false);
          (true, true, false);
          (false, false, true);
          (true, true, true);
        ])
    (* 63/65 straddle LA_TRSM_NB=64 (unblocked vs one trailing block); 128 gives
       two diagonal blocks. Same residual gate proves the blocked decomposition
       matches the substitution across the crossover. *)
    [ 1; 2; 5; 16; 40; 63; 65; 128 ];
  List.iter
    (fun (upper, transpose) ->
      test_trsm_complex ~kind:Buf.complex128 ~name:"c64" ~tol:1e-9 ~n:20 ~upper
        ~transpose ();
      test_trsm_complex ~kind:Buf.complex64 ~name:"c32" ~tol:1e-3 ~n:20 ~upper
        ~transpose ();
      (* complex crossover just above LA_TRSM_NB *)
      test_trsm_complex ~kind:Buf.complex128 ~name:"c64" ~tol:1e-9 ~n:65 ~upper
        ~transpose ())
    [ (false, false); (true, false); (false, true); (true, true) ];
  (* Wide RHS so the trailing update clears the GEMM direct cutoff — exercises
     the blocked nx_c_gemm2d_ct_ws path inside TRSM, not just the block
     decomposition. *)
  test_trsm_real ~kind:Buf.float64 ~name:"f64-wide" ~tol:1e-9 ~n:192 ~nrhs:256
    ~upper:false ~transpose:false ~unit:false ();
  test_trsm_real ~kind:Buf.float64 ~name:"f64-wide" ~tol:1e-9 ~n:192 ~nrhs:256
    ~upper:true ~transpose:true ~unit:false ();
  test_trsm_batched ();

  List.iter
    (fun (m, n) ->
      List.iter
        (fun reduced ->
          test_qr_real ~kind:Buf.float64 ~name:"f64" ~tol:1e-10 ~m ~n ~reduced
            ();
          test_qr_real ~kind:Buf.float32 ~name:"f32" ~tol:1e-3 ~m ~n ~reduced ();
          test_qr_complex ~kind:Buf.complex128 ~name:"c64" ~tol:1e-10 ~m ~n
            ~reduced ();
          test_qr_complex ~kind:Buf.complex64 ~name:"c32" ~tol:1e-3 ~m ~n
            ~reduced ())
        [ true; false ])
    (* k=min(m,n): with the raised unblocked crossover LA_QR_UNB=128, everything
       up to k=128 runs the unblocked Householder; only k>128 (129 here) drives
       the compact-WY blocked panels (larft + two-GEMM). Same recon +
       orthonormality gate proves the block reflectors match the unblocked path
       across the crossover, tall/wide/square, reduced and full Q. *)
    [
      (1, 1);
      (3, 3);
      (5, 3);
      (3, 5);
      (8, 8);
      (16, 9);
      (9, 16);
      (40, 30);
      (31, 31);
      (33, 33);
      (48, 48);
      (100, 64);
      (64, 100);
      (129, 129);
    ];
  test_qr_batched ();

  List.iter
    (fun n ->
      test_eigh_real ~kind:Buf.float64 ~name:"f64" ~tol:1e-9 ~n ();
      test_eigh_real ~kind:Buf.float32 ~name:"f32" ~tol:1e-3 ~n ();
      test_eigh_complex ~kind:Buf.complex128 ~name:"c64" ~tol:1e-9 ~n ();
      test_eigh_complex ~kind:Buf.complex64 ~name:"c32" ~tol:1e-3 ~n ())
    (* 33/64 straddle LA_EIGH_NB=32 (one latrd panel + tail, then two); 100/129
       force several panels. Recon + orthonormality gates prove the rank-2k GEMM
       trailing update matches the unblocked two-sided reduction, and that the
       real-β subdiagonal (hence tql2's feed) is preserved. *)
    [ 1; 2; 3; 5; 8; 16; 33; 64; 100; 129 ];
  test_eigh_real ~kind:Buf.float16 ~name:"f16" ~tol:5e-2 ~n:16 ();
  test_eigh_real ~kind:Buf.bfloat16 ~name:"bf16" ~tol:1e-1 ~n:16 ();
  test_eigh_diag ();
  test_eigh_novec ~n:12 ();
  test_eigh_batched ();

  List.iter
    (fun n ->
      test_eigh_crosscheck ~name:"smooth" ~n ~tol:1e-9 ~build:build_smooth ();
      (* clustered/glued: in-cluster eigenvectors are basis-rotation-free only
         to a larger c·n·ε (near-degenerate subspaces), so a wider orth_c. *)
      test_eigh_crosscheck ~name:"clustered" ~n ~tol:1e-8 ~orth_c:2048.0
        ~build:build_clustered ();
      test_eigh_crosscheck ~name:"glued-W" ~n ~tol:1e-8 ~orth_c:2048.0
        ~build:build_glued_wilkinson ();
      test_eigh_crosscheck ~name:"graded" ~n ~tol:1e-9 ~build:build_graded ();
      test_eigh_crosscheck ~name:"random" ~n ~tol:1e-9 ~build:build_random ())
    [ 26; 33; 50; 64; 100; 128; 200; 500 ];
  (* complex Hermitian rides the real D&C + the complex Q back-multiply *)
  test_eigh_complex ~kind:Buf.complex128 ~name:"c64-dc" ~tol:1e-9 ~n:40 ();
  test_eigh_complex ~kind:Buf.complex128 ~name:"c64-dc" ~tol:1e-9 ~n:100 ();
  test_eigh_complex ~kind:Buf.complex64 ~name:"c32-dc" ~tol:1e-3 ~n:64 ();
  (* odd n forces the recursion's unequal split; several tear levels *)
  test_eigh_real ~kind:Buf.float64 ~name:"f64-dc" ~tol:1e-9 ~n:257 ();

  List.iter
    (fun (m, n) ->
      List.iter
        (fun full ->
          test_svd_real ~kind:Buf.float64 ~name:"f64" ~tol:1e-9 ~m ~n ~full ();
          test_svd_real ~kind:Buf.float32 ~name:"f32" ~tol:1e-3 ~m ~n ~full ())
        [ false; true ])
    [ (1, 1); (3, 3); (5, 3); (3, 5); (6, 4); (4, 6); (8, 8) ];
  (* Blocked bidiagonalization crossover: min(m,n) straddles LA_SVD_NB=32 (33/48
     drive one labrd panel + tail, then two; 100 several), tall and wide so both
     the m>=n and the m<n (transpose) reduction paths run, reduced and full. The
     recon + orthonormality + descending-S gates prove the two-GEMM trailing
     update matches the unblocked two-sided Householder that feeds la_bdsvd. *)
  List.iter
    (fun (m, n) ->
      List.iter
        (fun full ->
          test_svd_real ~kind:Buf.float64 ~name:"f64" ~tol:1e-9 ~m ~n ~full ();
          test_svd_real ~kind:Buf.float32 ~name:"f32" ~tol:1e-3 ~m ~n ~full ())
        [ false; true ])
    [ (33, 33); (48, 48); (64, 40); (40, 64); (100, 100); (129, 96) ];
  test_svd_complex ~kind:Buf.complex128 ~name:"c64-blk" ~tol:1e-9 ~m:48 ~n:48 ();
  test_svd_complex ~kind:Buf.complex128 ~name:"c64-blk-wide" ~tol:1e-9 ~m:40
    ~n:70 ();
  test_svd_complex ~kind:Buf.complex64 ~name:"c32-blk" ~tol:1e-3 ~m:64 ~n:50 ();
  (* offline np.linalg.svd oracles: graded (1e2..1e-6), clustered, tall *)
  test_svd_real ~kind:Buf.float64 ~name:"graded" ~tol:1e-6 ~m:5 ~n:5 ~full:false
    ~a:svd_fix_graded5
    ~expect_s:[| 100.; 1.; 0.01; 1e-4; 1e-6 |]
    ();
  test_svd_real ~kind:Buf.float64 ~name:"clustered" ~tol:1e-9 ~m:4 ~n:4
    ~full:false ~a:svd_fix_clustered4 ~expect_s:[| 2.; 2.; 2.; 0.5 |] ();
  test_svd_real ~kind:Buf.float64 ~name:"tall" ~tol:1e-9 ~m:6 ~n:3 ~full:true
    ~a:svd_fix_tall63
    ~expect_s:[| 2.895986520101071; 1.911860947586696; 1.163883857289863 |]
    ();
  test_svd_complex ~kind:Buf.complex128 ~name:"c64" ~tol:1e-9 ~m:5 ~n:5 ();
  test_svd_complex ~kind:Buf.complex64 ~name:"c32" ~tol:1e-3 ~m:4 ~n:6 ();
  (* Golub–Kahan + Demmel–Kahan additions: hand-checkable 2×2 (σ = φ, 1/φ), n=1
     edges (thin & full m<n), extreme dynamic range and clustered/tiny with
     scale-aware relative accuracy, rank-deficient exact zero, tightened graded
     relative accuracy (the zero-shift win over the embedding's
     absolute-only). *)
  test_svd_real ~kind:Buf.float64 ~name:"phi2" ~tol:1e-12 ~m:2 ~n:2 ~full:false
    ~a:[| 1.; 1.; 0.; 1. |]
    ~expect_s:[| 1.618033988749895; 0.6180339887498949 |]
    ();
  test_svd_real ~kind:Buf.float64 ~name:"col1" ~tol:1e-12 ~m:2 ~n:1 ~full:false
    ~a:[| 3.; 4. |] ~expect_s:[| 5. |] ();
  test_svd_real ~kind:Buf.float64 ~name:"row1" ~tol:1e-12 ~m:1 ~n:3 ~full:false
    ~a:[| 3.; 4.; 0. |] ~expect_s:[| 5. |] ();
  test_svd_real ~kind:Buf.float64 ~name:"row1" ~tol:1e-12 ~m:1 ~n:3 ~full:true
    ~a:[| 3.; 4.; 0. |] ~expect_s:[| 5. |] ();
  (* Pure-relative gate (abstol=0): asserts the RELATIVE bound directly on the
     small σ — 1e-6 at ‖A‖≈100 to 1e-7 relative — the zero-shift accuracy that
     the embedding lacked, with no absolute floor masking it. *)
  test_svd_relacc ~name:"graded-rel" ~m:5 ~n:5 ~a:svd_fix_graded5
    ~es:[| 100.; 1.; 0.01; 1e-4; 1e-6 |]
    ~reltol:1e-7 ~abstol:0. ();
  test_svd_relacc ~name:"clustered-tiny" ~m:6 ~n:6 ~a:svd_fix_clus6
    ~es:svd_fix_clus6_s ~reltol:1e-8 ~abstol:0. ();
  (* dyn's 1e-8 σ and rank's exact-zero σ sit AT the ‖A‖·ε floor (numpy too), so
     these keep the absolute floor — no relative bound exists to assert
     there. *)
  test_svd_relacc ~name:"dyn(1e8..1e-8)" ~m:5 ~n:5 ~a:svd_fix_dyn5
    ~es:svd_fix_dyn5_s ~reltol:1e-9 ~abstol:1e-13 ();
  test_svd_relacc ~name:"rank-deficient" ~m:5 ~n:4 ~a:svd_fix_rank54
    ~es:svd_fix_rank54_s ~reltol:1e-9 ~abstol:1e-13 ();
  (* idir==2 backward-chase regression (the rel-0.257 conformance failure): the
     reviewer's witness, the conformance m4x3, and upper-bidiagonal inputs with
     |d[first]|<|d[last]| — all force dbdsqr's backward sweep. *)
  test_svd_idir2 ~name:"witness" ~a:usv_wit3_a ~m:3 ~n:3 ~tol:1e-12 ();
  test_svd_idir2 ~name:"m4x3" ~a:usv_m4x3_a ~m:4 ~n:3 ~tol:1e-12 ();
  test_svd_idir2 ~name:"bidiag-1,3,5"
    ~a:[| 1.; 2.; 0.; 0.; 3.; 2.; 0.; 0.; 5. |]
    ~m:3 ~n:3 ~tol:1e-12 ();
  test_svd_idir2 ~name:"bidiag-2,3,4"
    ~a:[| 2.; 1.; 0.; 0.; 3.; 1.; 0.; 0.; 4. |]
    ~m:3 ~n:3 ~tol:1e-12 ();
  (* element-wise numpy U/S/Vᴴ literal oracles (sign/order-pinned) *)
  test_svd_usv ~name:"m4x3" ~a:usv_m4x3_a ~s:usv_m4x3_s ~u:usv_m4x3_u
    ~vh:usv_m4x3_vh ~jmax:usv_m4x3_jmax ~m:4 ~n:3 ~tol:1e-9 ();
  test_svd_usv ~name:"wit3" ~a:usv_wit3_a ~s:usv_wit3_s ~u:usv_wit3_u
    ~vh:usv_wit3_vh ~jmax:usv_wit3_jmax ~m:3 ~n:3 ~tol:1e-9 ();
  test_svd_usv ~name:"wide35" ~a:usv_wide35_a ~s:usv_wide35_s ~u:usv_wide35_u
    ~vh:usv_wide35_vh ~jmax:usv_wide35_jmax ~m:3 ~n:5 ~tol:1e-9 ();
  (* complex m<n full_matrices + unitarity (previously recon-only) *)
  test_svd_complex ~kind:Buf.complex128 ~name:"c64-wide-full" ~tol:1e-9 ~m:3
    ~n:5 ~full:true ();

  test_svd_dc ~name:"dc40" ~tol:1e-9 ~m:40 ~n:40 ~expect_s:svd_dc40_s ();
  test_svd_dc ~name:"dc40-full" ~tol:1e-9 ~m:40 ~n:40 ~full:true ();
  test_svd_dc ~name:"dc-tall" ~tol:1e-9 ~m:64 ~n:40 ~expect_s:svd_dc_tall_s ();
  (* m<n runs the transpose path: U/Vᴴ swap+conjugate on output — the exact
     role-confusion habitat the per-vector gates exist for *)
  test_svd_dc ~name:"dc-wide" ~tol:1e-9 ~m:40 ~n:64 ~expect_s:svd_dc_wide_s ();
  test_svd_dc ~name:"dc-wide-full" ~tol:1e-9 ~m:40 ~n:64 ~full:true ();
  (* odd sizes tear unevenly through several recursion levels; 129 spans
     multiple SMLSIZ leaves and secular merges *)
  test_svd_dc ~name:"dc-odd" ~tol:1e-9 ~m:53 ~n:53 ();
  test_svd_dc ~name:"dc-129" ~tol:1e-9 ~m:129 ~n:129 ();
  (* bidiagonal-direct spectra (dense upper bidiagonal inputs; d/e formulas
     duplicated in the fixture generator): clustered = maximal deflation, graded
     = 16 orders of magnitude, zeros = splits + exact-zero σ, Wilkinson-analog =
     paired near-equal σ + a zero *)
  (let bidiag nn d e =
     let a = Array.make (nn * nn) 0.0 in
     for i = 0 to nn - 1 do
       a.((i * nn) + i) <- d.(i)
     done;
     for i = 0 to nn - 2 do
       a.((i * nn) + i + 1) <- e.(i)
     done;
     a
   in
   let clus_d = Array.init 40 (fun i -> 1.0 +. (1e-14 *. float_of_int i))
   and clus_e =
     Array.init 39 (fun i ->
         1e-13 *. (0.5 +. (float_of_int (i * 7 mod 3) /. 3.0)))
   in
   test_svd_dc ~name:"bd-clus40" ~tol:1e-9 ~m:40 ~n:40
     ~a:(bidiag 40 clus_d clus_e) ~expect_s:svd_bd_clus40_s ();
   let grad_d =
     Array.init 40 (fun i -> 10.0 ** ((16.0 *. float_of_int i /. 39.0) -. 8.0))
   in
   let grad_e =
     Array.init 39 (fun i -> 1e-3 *. sqrt (grad_d.(i) *. grad_d.(i + 1)))
   in
   test_svd_dc ~name:"bd-grad40" ~tol:1e-9 ~m:40 ~n:40
     ~a:(bidiag 40 grad_d grad_e) ~expect_s:svd_bd_grad40_s ();
   let zero_d =
     Array.init 33 (fun i ->
         sin (float_of_int (i * 13 mod 251)) +. (0.1 *. cos (float_of_int i)))
   and zero_e =
     Array.init 32 (fun i ->
         sin (float_of_int (i * 17 mod 241))
         +. (0.1 *. cos (float_of_int (2 * i))))
   in
   zero_d.(11) <- 0.0;
   zero_d.(22) <- 0.0;
   zero_e.(16) <- 0.0;
   test_svd_dc ~name:"bd-zero33" ~tol:1e-9 ~m:33 ~n:33
     ~a:(bidiag 33 zero_d zero_e) ~expect_s:svd_bd_zero33_s ();
   let wilk_d = Array.init 41 (fun i -> float_of_int (abs (i - 20)))
   and wilk_e = Array.make 40 1.0 in
   test_svd_dc ~name:"bd-wilk41" ~tol:1e-9 ~m:41 ~n:41
     ~a:(bidiag 41 wilk_d wilk_e) ~expect_s:svd_bd_wilk41_s ());
  (* complex rides the real double core + the complex lift/conjugated Qp *)
  test_svd_dc_c ~name:"c64-dc" ~tol:1e-9 ~m:40 ~n:40 ();
  test_svd_dc_c ~name:"c64-dc-wide" ~tol:1e-9 ~m:30 ~n:50 ();
  (* f32 rides the double core through the f32 lift/back-multiply *)
  test_svd_real ~kind:Buf.float32 ~name:"f32-dc" ~tol:1e-3 ~m:40 ~n:40
    ~full:false ();
  test_svd_dc_batched ~m:40 ~n:30 ~batch:3 ();

  test_chol_batched ~n:128 ~batch:3 ();
  (* batched blocked-GEMM trailing update *)
  test_trsm_real ~kind:Buf.float16 ~name:"f16" ~tol:5e-2 ~n:16 ~nrhs:2
    ~upper:false ~transpose:false ~unit:false ();
  test_trsm_real ~kind:Buf.bfloat16 ~name:"bf16" ~tol:1e-1 ~n:16 ~nrhs:2
    ~upper:false ~transpose:false ~unit:false ();
  test_qr_real ~kind:Buf.float16 ~name:"f16" ~tol:5e-2 ~m:16 ~n:16 ~reduced:true
    ();
  test_qr_real ~kind:Buf.bfloat16 ~name:"bf16" ~tol:1e-1 ~m:16 ~n:16
    ~reduced:true ();
  test_qr_zerocol ();
  (* zero column on the UNBLOCKED path (k<=LA_QR_UNB=128): tau=0 must skip the
     reflector and leave that column's R entry as-is. *)
  test_qr_zerocol ~m:80 ~n:60 ~zc:40 ();
  test_qr_zerocol ~m:70 ~n:70 ~zc:33 ();
  (* zero column INSIDE a blocked compact-WY panel: k=140 > LA_QR_UNB=128 forces
     the blocked path, exercising la_larft's tc==0 branch and la_buildv's
     identity-column (v[l]=1, tail 0) handling — no coverage otherwise. *)
  test_qr_zerocol ~m:150 ~n:140 ~zc:70 ();
  test_trsm_conj_fixture ();
  test_zero_dim ();
  test_svd_signed_zero ~kind:Buf.float64 ~name:"f64" ();
  test_svd_signed_zero ~kind:Buf.float32 ~name:"f32" ();

  test_chol_not_pd ();
  test_trsm_singular ();
  test_chol_batched_fail ();
  test_trsm_batched_fail ()

let () =
  Windtrap.run "nx C backend linalg stress"
    [
      group "internal-algorithm-branches"
        [ test "factorization and workspace gauntlet" test_internal_branches ];
    ]
