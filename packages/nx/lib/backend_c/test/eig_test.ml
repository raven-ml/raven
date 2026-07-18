(* Direct-FFI maintenance tests for nx_c_eig.c, the general nonsymmetric
   eigensolver.

   Each fixture is small enough to hand-verify its spectrum, embedded as a
   literal, and gated three ways — * per-eigenpair residual ‖A v − λ v‖ / (‖A‖
   ‖v‖) ≤ tol when vectors are requested (phase- and scale-invariant, so
   eigenvector normalization is free); * eigenvalue-SET match against the
   hand-computed spectrum (sorted by (re, im), so order is free); *
   conjugate-pair symmetry of the computed spectrum for real inputs. The oracle
   is independent of the solver (spectra derived by hand from characteristic
   polynomials / construction), never self-referential.

   Coverage: complex eigenvalues from real input, a defective Jordan block,
   graded/extreme scaling (balancing earns its keep), clustered spectra, a
   rotation+scaling composite, a mixed real+complex spectrum, the exact
   conformance eig-general matrix, complex-input matrices (the single-shift
   path), plus batched and f32-upcast cases.

   Test-local externals bind the C stub directly; nx_buffer builds the typed
   input buffers (incl. low-precision upcast and single/double complex) and the
   complex128 output buffers. The hard convergence fixtures run only under the
   backend-stress alias. *)

module Buf = Nx_buffer
open Bigarray

let ok name cond = if not cond then failwith name

type ('a, 'b) ffi = {
  buffer : ('a, 'b, c_layout) Genarray.t;
  shape : int array;
  strides : int array;
  offset : int;
}

(* eig w v in vectors — w and v are ALWAYS complex128; the dispatch dtype comes
   from `in`. When vectors=false the C never touches the v slot, so a
   placeholder complex128 tensor is passed there. *)
external eig_ext :
  (Complex.t, 'c) ffi -> (Complex.t, 'c) ffi -> ('a, 'b) ffi -> bool -> unit
  = "caml_nx_c_eig"

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

(* ── Complex oracle helpers ────────────────────────────────────────────────*)

let c re im : Complex.t = { Complex.re; im }
let ac_of_real n (a : float array) = Array.init (n * n) (fun i -> c a.(i) 0.0)

let frob n (a : Complex.t array) =
  let s = ref 0.0 in
  for i = 0 to (n * n) - 1 do
    s := !s +. Complex.norm2 a.(i)
  done;
  sqrt !s

(* v col j is eigenvector j: v.(i*n + j) is component i of eigenvector j. *)
let residual n (ac : Complex.t array) (w : Complex.t array)
    (v : Complex.t array) =
  let anorm = frob n ac in
  let anorm = if anorm = 0.0 then 1.0 else anorm in
  let worst = ref 0.0 in
  for j = 0 to n - 1 do
    let vn = ref 0.0 in
    for i = 0 to n - 1 do
      vn := !vn +. Complex.norm2 v.((i * n) + j)
    done;
    let vnorm = sqrt !vn in
    let vnorm = if vnorm = 0.0 then 1.0 else vnorm in
    let res = ref 0.0 in
    for i = 0 to n - 1 do
      let av = ref Complex.zero in
      for k = 0 to n - 1 do
        av := Complex.add !av (Complex.mul ac.((i * n) + k) v.((k * n) + j))
      done;
      let d = Complex.sub !av (Complex.mul w.(j) v.((i * n) + j)) in
      res := !res +. Complex.norm2 d
    done;
    let r = sqrt !res /. (anorm *. vnorm) in
    if r > !worst then worst := r
  done;
  !worst

let sort_evals (a : Complex.t array) =
  let b = Array.copy a in
  Array.sort
    (fun (x : Complex.t) (y : Complex.t) ->
      if x.re <> y.re then compare x.re y.re else compare x.im y.im)
    b;
  b

let evals_match ~n ~expected ~got =
  let e = sort_evals expected and g = sort_evals got in
  let worst = ref 0.0 in
  for i = 0 to n - 1 do
    let m = Complex.norm (Complex.sub e.(i) g.(i)) in
    if m > !worst then worst := m
  done;
  !worst

let conj_sym ~got ~tol =
  Array.for_all
    (fun (z : Complex.t) ->
      Array.exists
        (fun (w : Complex.t) ->
          Complex.norm (Complex.sub w (Complex.conj z)) <= tol)
        got)
    got

(* ── Runners ───────────────────────────────────────────────────────────────*)

let run_real (type b) ~(kind : (float, b) Buf.kind) ~n (a : float array)
    ~vectors =
  let ain = Buf.create kind (n * n) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let wout = Buf.create Buf.complex128 n in
  if vectors then (
    let vout = Buf.create Buf.complex128 (n * n) in
    eig_ext (ffi wout [| n |] [| 1 |])
      (ffi vout [| n; n |] (contig [| n; n |]))
      (ffi ain [| n; n |] (contig [| n; n |]))
      true;
    let w = Array.init n (fun i -> Buf.get wout i) in
    let v = Array.init (n * n) (fun i -> Buf.get vout i) in
    (w, Some v))
  else (
    (* vectors=false: reuse wout as the ignored v-slot placeholder. *)
    eig_ext (ffi wout [| n |] [| 1 |]) (ffi wout [| n |] [| 1 |])
      (ffi ain [| n; n |] (contig [| n; n |]))
      false;
    let w = Array.init n (fun i -> Buf.get wout i) in
    (w, None))

let run_complex (type b) ~(kind : (Complex.t, b) Buf.kind) ~n
    (a : Complex.t array) =
  let ain = Buf.create kind (n * n) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let wout = Buf.create Buf.complex128 n in
  let vout = Buf.create Buf.complex128 (n * n) in
  eig_ext (ffi wout [| n |] [| 1 |])
    (ffi vout [| n; n |] (contig [| n; n |]))
    (ffi ain [| n; n |] (contig [| n; n |]))
    true;
  let w = Array.init n (fun i -> Buf.get wout i) in
  let v = Array.init (n * n) (fun i -> Buf.get vout i) in
  (w, v)

(* Eigenvalues only (vectors=false): the C never touches the v slot, so a
   complex128 placeholder (wout) is passed there. Used for the self-consistency
   gate — the values must agree with the vectors=true path. *)
let run_complex_vals (type b) ~(kind : (Complex.t, b) Buf.kind) ~n
    (a : Complex.t array) =
  let ain = Buf.create kind (n * n) in
  for t = 0 to (n * n) - 1 do
    Buf.set ain t a.(t)
  done;
  let wout = Buf.create Buf.complex128 n in
  eig_ext (ffi wout [| n |] [| 1 |]) (ffi wout [| n |] [| 1 |])
    (ffi ain [| n; n |] (contig [| n; n |]))
    false;
  Array.init n (fun i -> Buf.get wout i)

let run_real_batched ~b ~n (mats : float array array) =
  let ain = Buf.create Buf.float64 (b * n * n) in
  for bi = 0 to b - 1 do
    for t = 0 to (n * n) - 1 do
      Buf.set ain ((bi * n * n) + t) mats.(bi).(t)
    done
  done;
  let wout = Buf.create Buf.complex128 (b * n) in
  let vout = Buf.create Buf.complex128 (b * n * n) in
  eig_ext
    (ffi wout [| b; n |] (contig [| b; n |]))
    (ffi vout [| b; n; n |] (contig [| b; n; n |]))
    (ffi ain [| b; n; n |] (contig [| b; n; n |]))
    true;
  Array.init b (fun bi ->
      let w = Array.init n (fun i -> Buf.get wout ((bi * n) + i)) in
      let v = Array.init (n * n) (fun i -> Buf.get vout ((bi * n * n) + i)) in
      (w, v))

(* ── Checkers ───────────────────────────────────────────────────────────────*)

let check_real (type b) ~(kind : (float, b) Buf.kind) ~name ~n ~tol_res ~tol_ev
    ~a ~expected () =
  let w, vo = run_real ~kind ~n a ~vectors:true in
  let v = match vo with Some v -> v | None -> assert false in
  let ac = ac_of_real n a in
  let res = residual n ac w v in
  let evm = evals_match ~n ~expected ~got:w in
  let cs = conj_sym ~got:w ~tol:(tol_ev *. 100.0) in
  ok
    (Printf.sprintf "%s res=%.2e evset=%.2e conj=%b" name res evm cs)
    (res <= tol_res && evm <= tol_ev && cs)

let check_complex (type b) ~(kind : (Complex.t, b) Buf.kind) ~name ~n ~tol_res
    ~tol_ev ~a ~expected () =
  let w, v = run_complex ~kind ~n a in
  let res = residual n a w v in
  let evm = evals_match ~n ~expected ~got:w in
  ok
    (Printf.sprintf "%s res=%.2e evset=%.2e" name res evm)
    (res <= tol_res && evm <= tol_ev)

(* ── Fixtures (spectra hand-derived, see comments) ─────────────────────────*)

(* [[0,-1],[1,0]]: characteristic λ²+1 ⇒ eigenvalues ±i. Real input, purely
   imaginary spectrum — the smallest complex-from-real case. *)
let () =
  check_real ~kind:Buf.float64 ~name:"rot90 (±i)" ~n:2 ~tol_res:1e-9
    ~tol_ev:1e-9 ~a:[| 0.; -1.; 1.; 0. |]
    ~expected:[| c 0. 1.; c 0. (-1.) |]
    ()

(* [[2,1],[0,2]]: a defective Jordan block. Eigenvalue 2 (algebraic mult 2,
   geometric mult 1). The solver returns correct eigenvalues and a valid
   per-column eigenpair (both columns ≈ the single eigenvector [1,0]); the
   eigenvector matrix is intentionally rank-deficient — documented, not a bug.
   Per-pair residual still holds. *)
let () =
  check_real ~kind:Buf.float64 ~name:"jordan defective (2,2)" ~n:2 ~tol_res:1e-9
    ~tol_ev:1e-9 ~a:[| 2.; 1.; 0.; 2. |]
    ~expected:[| c 2. 0.; c 2. 0. |]
    ()

(* Graded/extreme scaling: A = D C D⁻¹ with D = diag(1e8, 1, 1e-8) and C the
   companion matrix of (x−1)(x−2)(x−3). The similarity leaves the spectrum
   {1,2,3} but wrecks the row/column norms — balancing must recover it. A = [[0,
   1e8, 0], [0, 0, 1e8], [6e-16, -1.1e-7, 6]]. *)
let () =
  check_real ~kind:Buf.float64 ~name:"graded scaling (1,2,3)" ~n:3 ~tol_res:1e-9
    ~tol_ev:1e-5
    ~a:[| 0.; 1e8; 0.; 0.; 0.; 1e8; 6e-16; -1.1e-7; 6. |]
    ~expected:[| c 1. 0.; c 2. 0.; c 3. 0. |]
    ()

(* Clustered spectrum: companion of (x−0.999)(x−1)(x−1.001); the three
   near-equal real roots stress deflation. Coefficients: x³ − 3x² + 2.999999x −
   0.999999. *)
let () =
  check_real ~kind:Buf.float64 ~name:"clustered (~1)" ~n:3 ~tol_res:1e-9
    ~tol_ev:1e-6
    ~a:[| 0.; 1.; 0.; 0.; 0.; 1.; 0.999999; -2.999999; 3. |]
    ~expected:[| c 0.999 0.; c 1.0 0.; c 1.001 0. |]
    ()

(* Rotation + uniform scaling by 2 at θ=π/4: [[√2,−√2],[√2,√2]] has eigenvalues
   2·e^{±iπ/4} = √2 ± i√2. *)
let () =
  let r = sqrt 2.0 in
  check_real ~kind:Buf.float64 ~name:"rotation+scale (√2±i√2)" ~n:2
    ~tol_res:1e-9 ~tol_ev:1e-9 ~a:[| r; -.r; r; r |]
    ~expected:[| c r r; c r (-.r) |]
    ()

(* Mixed real+complex spectrum: companion of (x−2)(x²+1) ⇒ eigenvalues 2, ±i. x³
   − 2x² + x − 2 ⇒ companion [[0,1,0],[0,0,1],[2,-1,2]]. *)
let () =
  check_real ~kind:Buf.float64 ~name:"mixed spectrum (2,±i)" ~n:3 ~tol_res:1e-9
    ~tol_ev:1e-9
    ~a:[| 0.; 1.; 0.; 0.; 0.; 1.; 2.; -1.; 2. |]
    ~expected:[| c 2. 0.; c 0. 1.; c 0. (-1.) |]
    ()

(* Backend-contract eig-general regression: [[2,-1,0],[1,3,-1],[0,1,2]]. char
   poly λ³−7λ²+18λ−16 = (λ−2)(λ²−5λ+8) ⇒ 2, 2.5 ± i·√7/2. *)
let () =
  let s7 = sqrt 7.0 /. 2.0 in
  check_real ~kind:Buf.float64 ~name:"conformance eig-general" ~n:3
    ~tol_res:1e-9 ~tol_ev:1e-9
    ~a:[| 2.; -1.; 0.; 1.; 3.; -1.; 0.; 1.; 2. |]
    ~expected:[| c 2. 0.; c 2.5 s7; c 2.5 (-.s7) |]
    ()

(* Upcast: rotation+scale through the f32 path (packed to double, factored in
   double). √2 is not f32-exact, so this genuinely exercises lossy input
   rounding — the eigenvalues of the f32-rounded matrix differ from the true
   √2±i√2 by ~1e-7, well inside the loosened tolerances. *)
let () =
  let r = sqrt 2.0 in
  check_real ~kind:Buf.float32 ~name:"rotation+scale upcast f32" ~n:2
    ~tol_res:1e-4 ~tol_ev:1e-4 ~a:[| r; -.r; r; r |]
    ~expected:[| c r r; c r (-.r) |]
    ()

(* Complex input, upper triangular: eigenvalues are the diagonal 1+i, 3−i, 2.
   Exercises the complex single-shift path (dispatch on c64). *)
let () =
  check_complex ~kind:Buf.complex128 ~name:"complex triangular" ~n:3
    ~tol_res:1e-9 ~tol_ev:1e-9
    ~a:
      [|
        c 1. 1.;
        c 2. 0.;
        c 0. 0.;
        c 0. 0.;
        c 3. (-1.);
        c 1. 0.;
        c 0. 0.;
        c 0. 0.;
        c 2. 0.;
      |]
    ~expected:[| c 1. 1.; c 3. (-1.); c 2. 0. |]
    ()

(* Complex input, non-triangular: [[1,i],[i,1]] ⇒ trace 2, det 1−i²=2, λ²−2λ+2=0
   ⇒ 1 ± i. Full complex QR + eigenvectors. *)
let () =
  check_complex ~kind:Buf.complex128 ~name:"complex 2x2 (1±i)" ~n:2
    ~tol_res:1e-9 ~tol_ev:1e-9
    ~a:[| c 1. 0.; c 0. 1.; c 0. 1.; c 1. 0. |]
    ~expected:[| c 1. 1.; c 1. (-1.) |]
    ()

(* Complex input through the single-precision (complex64) path, upcast to
   double-complex for the factorization. *)
let () =
  check_complex ~kind:Buf.complex64 ~name:"complex 2x2 upcast c32" ~n:2
    ~tol_res:1e-4 ~tol_ev:1e-4
    ~a:[| c 1. 0.; c 0. 1.; c 0. 1.; c 1. 0. |]
    ~expected:[| c 1. 1.; c 1. (-1.) |]
    ()

(* REGRESSION GATE for the complex Householder Hessenberg reflector (Hᴴ-vs-H τ
   conjugation). A = [[0,1,1],[i,0,1],[1,1,0]] is NOT upper Hessenberg
   (a[2][0]=1≠0) AND has a[1][0]=i, so the m=1 reflector's leading element is
   imaginary ⇒ τ is genuinely COMPLEX — the exact case the earlier fixtures
   dodged (companions have a[1][0]=0 ⇒ real τ, which cannot detect the swap).
   char poly λ³−(2+i)λ−(1+i) = (λ+1)(λ²−λ−(1+i)) ⇒ eigenvalues −1 and (1 ±
   √(5+4i))/2, derived offline and computed here from the closed form. *)
let () =
  let disc = Complex.sqrt (c 5. 4.) in
  let half = c 0.5 0.0 in
  let l1 = Complex.mul half (Complex.add Complex.one disc) in
  let l2 = Complex.mul half (Complex.sub Complex.one disc) in
  check_complex ~kind:Buf.complex128 ~name:"complex non-Hessenberg (complex τ)"
    ~n:3 ~tol_res:1e-9 ~tol_ev:1e-9
    ~a:
      [|
        c 0. 0.;
        c 1. 0.;
        c 1. 0.;
        c 0. 1.;
        c 0. 0.;
        c 1. 0.;
        c 1. 0.;
        c 1. 0.;
        c 0. 0.;
      |]
    ~expected:[| c (-1.) 0.; l1; l2 |]
    ()

(* Random dense complex stress (n=24): dense ⇒ every column drives a complex-τ
   reflector, and the run reduces through the full Hessenberg + QR pipeline. No
   hand oracle — gated on the residual (independent: A v = λ v for each computed
   pair) and eigenvalue self-consistency (vectors=true values == vectors=false
   values, sorted). A broken similarity blows the residual (the reviewer's repro
   went 0.31 -> 2.6e-16). *)
let () =
  let n = 24 in
  let a =
    Array.init (n * n) (fun idx ->
        c
          (sin (float_of_int (idx * 13 mod 251) -. 3.0))
          (cos (float_of_int (idx * 7 mod 241) -. 2.0)))
  in
  let w, v = run_complex ~kind:Buf.complex128 ~n a in
  let res = residual n a w v in
  let w2 = run_complex_vals ~kind:Buf.complex128 ~n a in
  let self = evals_match ~n ~expected:w ~got:w2 in
  ok
    (Printf.sprintf "complex random n=24 res=%.2e self=%.2e" res self)
    (res <= 1e-9 && self <= 1e-12)

(* Batched: stack the mixed-spectrum and eig-general matrices; each batch is
   checked independently against its own spectrum. *)
let () =
  let s7 = sqrt 7.0 /. 2.0 in
  let m0 = [| 0.; 1.; 0.; 0.; 0.; 1.; 2.; -1.; 2. |] in
  let m1 = [| 2.; -1.; 0.; 1.; 3.; -1.; 0.; 1.; 2. |] in
  let exp0 = [| c 2. 0.; c 0. 1.; c 0. (-1.) |] in
  let exp1 = [| c 2. 0.; c 2.5 s7; c 2.5 (-.s7) |] in
  let results = run_real_batched ~b:2 ~n:3 [| m0; m1 |] in
  let check bi a expected =
    let w, v = results.(bi) in
    let ac = ac_of_real 3 a in
    let res = residual 3 ac w v in
    let evm = evals_match ~n:3 ~expected ~got:w in
    ok
      (Printf.sprintf "batched[%d] res=%.2e evset=%.2e" bi res evm)
      (res <= 1e-9 && evm <= 1e-9)
  in
  check 0 m0 exp0;
  check 1 m1 exp1

(* vectors=false: eigenvalues only, no eigenvector slot touched. *)
let () =
  let w, vo =
    run_real ~kind:Buf.float64 ~n:3
      [| 0.; 1.; 0.; 0.; 0.; 1.; 2.; -1.; 2. |]
      ~vectors:false
  in
  let evm =
    evals_match ~n:3 ~expected:[| c 2. 0.; c 0. 1.; c 0. (-1.) |] ~got:w
  in
  ok (Printf.sprintf "vectors=false evset=%.2e" evm) (evm <= 1e-9 && vo = None)

(* n=1: the trivial 1×1 case — eigenvalue is the sole entry, eigenvector [1]. *)
let () =
  let w, vo = run_real ~kind:Buf.float64 ~n:1 [| 7. |] ~vectors:true in
  let v = match vo with Some v -> v | None -> assert false in
  ok "n=1 eigenvalue" (Complex.norm (Complex.sub w.(0) (c 7. 0.)) <= 1e-12);
  ok "n=1 eigenvector unit" (abs_float (Complex.norm v.(0) -. 1.0) <= 1e-12)

(* Convergence: every fixture converges well inside the iteration cap (30·n
   total for the real Francis QR, 30·(igh−low+1) for the complex single-shift
   QR); the exceptional shifts at iterations 10 and 20 break the cyclic stalls
   that killed round one. A matrix that blows the cap raises Failure
   ("eigenvalue iteration did not converge") rather than returning a silent
   wrong answer — not reproducible on the gauntlet, so asserted by construction
   in the C, not forced here. *)
