(* Internal correctness checks for nx_c_matmul.c.

   Test-local externals bind the GEMM stubs directly over FFI records mirroring
   the four slots the C reads (buffer/shape/strides/offset). Buffers come from
   Nx_buffer, so every dtype — including the extended kinds (bf16, fp8, u32/u64)
   that have no standard Bigarray kind — is exercised through to_genarray.

   Correctness gate (before any tuning): every path is checked against an
   independent f64 (or exact-integer / complex) reference, and the blocked
   kernel is cross-checked against the fully independent naive triple-loop path
   (mode 2) over every size, transpose combo, offset, batch broadcast, and
   dtype. Inputs are read back through Nx_buffer.get so the reference uses the
   ACTUAL quantized operands: for low-precision dtypes the only slack is
   f32-accumulation plus a single output-quantization step.

   Public matmul semantics live in the Nx backend contract; this suite retains
   owned-kernel, workspace, worker-partition, and Accelerate-routing checks. *)

module Buf = Nx_buffer
open Bigarray
open Windtrap

let ok name cond = is_true ~msg:name cond

(* FFI operand: slots 0-3 are buffer/shape/strides/offset, exactly what the C
   reads (the nx_c.h NX_C_FFI slots). A flat 1-D genarray carries the storage
   and its kind; the logical shape/strides/offset drive the C entirely. *)
type ('a, 'b) ffi = {
  buffer : ('a, 'b, c_layout) Genarray.t;
  shape : int array;
  strides : int array;
  offset : int;
}

external mm : ('a, 'b) ffi -> ('a, 'b) ffi -> ('a, 'b) ffi -> unit
  = "caml_nx_c_matmul"

(* mode: 0 owned+policy, 1 owned+single-thread, 2 owned direct naive, 3 accel *)
external mm_ex : ('a, 'b) ffi -> ('a, 'b) ffi -> ('a, 'b) ffi -> int -> unit
  = "caml_nx_c_matmul_ex"

(* Test-only: nx_c_gemm2d_ct_ws (linalg's caller-workspace GEMM) over a single
   2-D matmul. The stub sizes + allocates the workspace and runs it; f32/f64
   only (its dt is compute-typed). Exercises the k > MM_KC_FULLK_MAX Cacc-alloc
   branch. *)
external mm_ws : ('a, 'b) ffi -> ('a, 'b) ffi -> ('a, 'b) ffi -> unit
  = "caml_nx_c_gemm2d_ct_ws_test"

(* Accelerate hook introspection + control (macOS only; the stubs report 0 / are
   no-ops off macOS). available: hook compiled in. enabled: a call would route
   eligible ops to cblas now. set_override: -1 automatic / 0 force owned / 1
   force Accelerate — drives both routes in one process for the differential. *)
external accel_available : unit -> int = "caml_nx_c_matmul_accel_available"
external accel_enabled : unit -> int = "caml_nx_c_matmul_accel_enabled"

external accel_set_override : int -> unit
  = "caml_nx_c_matmul_accel_set_override"

let ffi ?(offset = 0) buf shape strides =
  { buffer = Buf.to_genarray buf [| Buf.length buf |]; shape; strides; offset }

(* ── Real dtypes (float-valued kinds: f16 f32 f64 bf16 fp8e4m3 fp8e5m2) ──── *)

(* Build a 2-D operand of logical shape rows x cols. trans=false stores
   row-major (strides cols,1); trans=true stores column-major (strides 1,rows) —
   a transposed view the pack must resolve for free. off shifts the data
   base. *)
let mk_real (type b) (kind : (float, b) Buf.kind) ~rows ~cols ~trans ~off fill =
  let rs, cs = if trans then (1, rows) else (cols, 1) in
  let buf = Buf.create kind (off + (rows * cols)) in
  for i = 0 to rows - 1 do
    for j = 0 to cols - 1 do
      Buf.set buf (off + (i * rs) + (j * cs)) (fill i j)
    done
  done;
  (buf, rs, cs)

let test_real (type b) ~(kind : (float, b) Buf.kind) ~name ~tol_rel ~tol_abs ~m
    ~k ~n ?(a_trans = false) ?(b_trans = false) ?(a_off = 0) ?(b_off = 0)
    ?(c_off = 0) ?(scale = 1.0) ~modes () =
  let fa i p = sin (float_of_int (((i * k) + p) * 13 mod 1009)) *. scale in
  let fb p j = cos (float_of_int (((p * n) + j) * 7 mod 1013)) *. scale in
  let abuf, ars, acs =
    mk_real kind ~rows:m ~cols:k ~trans:a_trans ~off:a_off fa
  in
  let bbuf, brs, bcs =
    mk_real kind ~rows:k ~cols:n ~trans:b_trans ~off:b_off fb
  in
  let geta i p = Buf.get abuf (a_off + (i * ars) + (p * acs)) in
  let getb p j = Buf.get bbuf (b_off + (p * brs) + (j * bcs)) in
  let refc = Array.make (m * n) 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0.0 in
      for p = 0 to k - 1 do
        s := !s +. (geta i p *. getb p j)
      done;
      refc.((i * n) + j) <- !s
    done
  done;
  let a_ffi = ffi ~offset:a_off abuf [| m; k |] [| ars; acs |] in
  let b_ffi = ffi ~offset:b_off bbuf [| k; n |] [| brs; bcs |] in
  let run label call =
    let cbuf = Buf.create kind (c_off + (m * n)) in
    for t = 0 to c_off + (m * n) - 1 do
      Buf.set cbuf t 123456.0
    done;
    let c_ffi = ffi ~offset:c_off cbuf [| m; n |] [| n; 1 |] in
    call c_ffi a_ffi b_ffi;
    let bad = ref 0 and maxerr = ref 0.0 in
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        let got = Buf.get cbuf (c_off + (i * n) + j) in
        let r = refc.((i * n) + j) in
        let e = abs_float (got -. r) in
        if e > !maxerr then maxerr := e;
        if e > (tol_rel *. abs_float r) +. tol_abs then incr bad
      done
    done;
    ok
      (Printf.sprintf
         "%s %s %dx%dx%d aT=%b bT=%b off=%d/%d/%d (bad=%d maxerr=%.3g)" name
         label m k n a_trans b_trans a_off b_off c_off !bad !maxerr)
      (!bad = 0)
  in
  List.iter
    (function
      | `Prod -> run "prod" (fun c a b -> mm c a b)
      | `Owned -> run "owned" (fun c a b -> mm_ex c a b 0)
      | `St -> run "1t" (fun c a b -> mm_ex c a b 1)
      | `Direct -> run "direct" (fun c a b -> mm_ex c a b 2))
    modes

(* Cross-check the OWNED blocked kernel (mode 0, engine policy — the
   multi-thread panel / M-split path) against the naive path with no OCaml
   reference (both are C-speed) — for sizes where an OCaml reference triple loop
   would be slow. This forces the owned path explicitly, not `mm`: on macOS the
   frontend routes large floats to Accelerate, and this test's whole point is
   the owned kernel (incl. mm_mpar_body). Both paths sum in the same k-order, so
   they agree to within one FMA rounding: the blocked microkernel fuses (vfma),
   the naive loop fuses only if the compiler contracts a*b+c, so compare within
   a tolerance rather than bitwise (an exact test would be hostage to that
   contraction). A real bug moves the result by orders of magnitude, far outside
   tol. *)
let test_diff (type b) ~(kind : (float, b) Buf.kind) ~name ~tol ~m ~k ~n () =
  let fa i p = sin (float_of_int (((i * k) + p) * 13 mod 4099)) in
  let fb p j = cos (float_of_int (((p * n) + j) * 7 mod 4093)) in
  let abuf, ars, acs = mk_real kind ~rows:m ~cols:k ~trans:false ~off:0 fa in
  let bbuf, brs, bcs = mk_real kind ~rows:k ~cols:n ~trans:false ~off:0 fb in
  let a_ffi = ffi abuf [| m; k |] [| ars; acs |] in
  let b_ffi = ffi bbuf [| k; n |] [| brs; bcs |] in
  let c1 = Buf.create kind (m * n) and c2 = Buf.create kind (m * n) in
  mm_ex (ffi c1 [| m; n |] [| n; 1 |]) a_ffi b_ffi 0;
  mm_ex (ffi c2 [| m; n |] [| n; 1 |]) a_ffi b_ffi 2;
  let bad = ref 0 and maxerr = ref 0.0 in
  for t = 0 to (m * n) - 1 do
    let g = Buf.get c1 t and r = Buf.get c2 t in
    let e = abs_float (g -. r) in
    if e > !maxerr then maxerr := e;
    if e > tol +. (tol *. abs_float r) then incr bad
  done;
  ok
    (Printf.sprintf "%s diff blocked-vs-naive %dx%dx%d (bad=%d maxerr=%.3g)"
       name m k n !bad !maxerr)
    (!bad = 0)

(* KC accumulator-precision witness, sharpened for adversarial cancellation.
   Every A row is +a on the first half of k and -a on the second; B is uniform
   +b. So every output is exactly (k/2)(ab) - (k/2)(ab) = 0, but the running
   partial climbs to ~(k/2)(ab) before it cancels. A compute-typed (f32)
   accumulator sums the ±(KC·ab) panel partials back to ~0; a storage-typed
   (f16/bf16) Cacc rounds each panel partial on that large running sum, and the
   rounding does NOT cancel — it leaves a residual of order ulp((k/2)·ab), which
   for these values is O(1), ~20x over tol. The true answer being exactly 0
   turns any storage-precision leak into a large ABSOLUTE error, unlike a
   benign-input test where it hides under the f16 output quantization. Checks
   the blocked KC path (mode 0 policy, mode 1 forced-1t) and direct (mode 2,
   also compute-typed); a and b are chosen so the panel partial (KC·ab) is NOT
   f16-representable, so the leak actually rounds. *)
let test_kc_cancel (type b) ~(kind : (float, b) Buf.kind) ~name ~m ~k ~n () =
  let a = 1.3 and b = 1.5 in
  let abuf = Buf.create kind (m * k) in
  for i = 0 to m - 1 do
    for p = 0 to k - 1 do
      Buf.set abuf ((i * k) + p) (if p < k / 2 then a else -.a)
    done
  done;
  let bbuf = Buf.create kind (k * n) in
  for t = 0 to (k * n) - 1 do
    Buf.set bbuf t b
  done;
  (* Reference from the ACTUAL quantized operands, in f64: the ± halves cancel
     to exactly 0 (equal magnitudes), so the target is 0 regardless of
     quantization. *)
  let geta i p = Buf.get abuf ((i * k) + p) in
  let getb p j = Buf.get bbuf ((p * n) + j) in
  let refc = Array.make (m * n) 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0.0 in
      for p = 0 to k - 1 do
        s := !s +. (geta i p *. getb p j)
      done;
      refc.((i * n) + j) <- !s
    done
  done;
  let a_ffi = ffi abuf [| m; k |] [| k; 1 |] in
  let b_ffi = ffi bbuf [| k; n |] [| n; 1 |] in
  let run label call =
    let cbuf = Buf.create kind (m * n) in
    call (ffi cbuf [| m; n |] [| n; 1 |]) a_ffi b_ffi;
    let bad = ref 0 and maxerr = ref 0.0 in
    for t = 0 to (m * n) - 1 do
      let e = abs_float (Buf.get cbuf t -. refc.(t)) in
      if e > !maxerr then maxerr := e;
      if e > 0.1 then incr bad
    done;
    ok
      (Printf.sprintf "%s KC-cancel %s %dx%dx%d (bad=%d maxerr=%.3g)" name label
         m k n !bad !maxerr)
      (!bad = 0)
  in
  run "prod" (fun c a b -> mm c a b);
  run "1t" (fun c a b -> mm_ex c a b 1);
  run "direct" (fun c a b -> mm_ex c a b 2)

(* nx_c_gemm2d_ct_ws (linalg's caller-workspace GEMM) vs the direct oracle, over
   compute-typed f32/f64. Run at k > MM_KC_FULLK_MAX so the wrapper's Cacc-alloc
   branch (size query + sub-slot handoff) executes; a mis-sized/mis-offset
   scratch or wrong Cacc pointer would move the result far outside tol. *)
let test_ws (type b) ~(kind : (float, b) Buf.kind) ~name ~tol ~m ~k ~n () =
  let fa i p = sin (float_of_int (((i * k) + p) * 13 mod 4099)) in
  let fb p j = cos (float_of_int (((p * n) + j) * 7 mod 4093)) in
  let abuf, _, _ = mk_real kind ~rows:m ~cols:k ~trans:false ~off:0 fa in
  let bbuf, _, _ = mk_real kind ~rows:k ~cols:n ~trans:false ~off:0 fb in
  let a_ffi = ffi abuf [| m; k |] [| k; 1 |] in
  let b_ffi = ffi bbuf [| k; n |] [| n; 1 |] in
  let cws = Buf.create kind (m * n) and cref = Buf.create kind (m * n) in
  mm_ws (ffi cws [| m; n |] [| n; 1 |]) a_ffi b_ffi;
  mm_ex (ffi cref [| m; n |] [| n; 1 |]) a_ffi b_ffi 2;
  let bad = ref 0 and maxerr = ref 0.0 in
  for t = 0 to (m * n) - 1 do
    let r = Buf.get cref t in
    let e = abs_float (Buf.get cws t -. r) in
    if e > !maxerr then maxerr := e;
    if e > tol +. (tol *. abs_float r) then incr bad
  done;
  ok
    (Printf.sprintf "%s gemm2d_ct_ws-vs-direct %dx%dx%d (bad=%d maxerr=%.3g)"
       name m k n !bad !maxerr)
    (!bad = 0)

(* ── Accelerate hook (macOS) ───────────────────────────────────────────────

   The frontend's default-on cblas route (caml_nx_c_matmul). Three properties,
   proven in one process via the test-only override: (1) the automatic enabled
   state reflects the platform; (2) forced-on (cblas) agrees with the owned
   direct oracle within rounding — the hook-path vs owned differential; (3)
   forced-owned is BIT-identical to the owned policy path. When the hook is
   unavailable (off macOS) every route is owned, so (2)/(3) hold as
   owned-vs-owned and the suite stays green everywhere. *)
let test_accel_real (type b) ~(kind : (float, b) Buf.kind) ~name ~tol ~m ~k ~n
    () =
  let fa i p = sin (float_of_int (((i * k) + p) * 13 mod 4099)) in
  let fb p j = cos (float_of_int (((p * n) + j) * 7 mod 4093)) in
  let abuf, _, _ = mk_real kind ~rows:m ~cols:k ~trans:false ~off:0 fa in
  let bbuf, _, _ = mk_real kind ~rows:k ~cols:n ~trans:false ~off:0 fb in
  let a_ffi = ffi abuf [| m; k |] [| k; 1 |] in
  let b_ffi = ffi bbuf [| k; n |] [| n; 1 |] in
  let run mode =
    let c = Buf.create kind (m * n) in
    mm_ex (ffi c [| m; n |] [| n; 1 |]) a_ffi b_ffi mode;
    c
  in
  let oracle = run 2 and owned = run 0 in
  accel_set_override 1;
  let forced_on = run 3 in
  accel_set_override 0;
  let forced_off = run 3 in
  accel_set_override (-1);
  let cmp label tol ca cb =
    let bad = ref 0 and maxerr = ref 0.0 in
    for t = 0 to (m * n) - 1 do
      let e = abs_float (Buf.get ca t -. Buf.get cb t) in
      if e > !maxerr then maxerr := e;
      if e > tol +. (tol *. abs_float (Buf.get cb t)) then incr bad
    done;
    ok
      (Printf.sprintf "%s %s %dx%dx%d (bad=%d maxerr=%.3g)" name label m k n
         !bad !maxerr)
      (!bad = 0)
  in
  cmp "accel-vs-oracle" tol forced_on oracle;
  cmp "optout-eq-owned" 0.0 forced_off owned

let test_accel_complex (type b) ~(kind : (Complex.t, b) Buf.kind) ~name ~tol ~m
    ~k ~n () =
  let fa i p =
    {
      Complex.re = sin (float_of_int (((i * k) + p) mod 97));
      im = cos (float_of_int (((i * k) + p) mod 89));
    }
  in
  let fb p j =
    {
      Complex.re = cos (float_of_int (((p * n) + j) mod 83));
      im = sin (float_of_int (((p * n) + j) mod 79));
    }
  in
  let abuf = Buf.create kind (m * k) and bbuf = Buf.create kind (k * n) in
  for i = 0 to m - 1 do
    for p = 0 to k - 1 do
      Buf.set abuf ((i * k) + p) (fa i p)
    done
  done;
  for p = 0 to k - 1 do
    for j = 0 to n - 1 do
      Buf.set bbuf ((p * n) + j) (fb p j)
    done
  done;
  let a_ffi = ffi abuf [| m; k |] [| k; 1 |] in
  let b_ffi = ffi bbuf [| k; n |] [| n; 1 |] in
  let run mode =
    let c = Buf.create kind (m * n) in
    mm_ex (ffi c [| m; n |] [| n; 1 |]) a_ffi b_ffi mode;
    c
  in
  let oracle = run 2 and owned = run 0 in
  accel_set_override 1;
  let forced_on = run 3 in
  accel_set_override 0;
  let forced_off = run 3 in
  accel_set_override (-1);
  (* gate on the magnitude of the complex difference, relative to |cb| *)
  let cmp label tol ca cb =
    let bad = ref 0 and maxerr = ref 0.0 in
    for t = 0 to (m * n) - 1 do
      let za = Buf.get ca t and zb = Buf.get cb t in
      let e = Complex.norm (Complex.sub za zb) in
      if e > !maxerr then maxerr := e;
      if e > tol +. (tol *. Complex.norm zb) then incr bad
    done;
    ok
      (Printf.sprintf "%s %s %dx%dx%d (bad=%d maxerr=%.3g)" name label m k n
         !bad !maxerr)
      (!bad = 0)
  in
  cmp "accel-vs-oracle" tol forced_on oracle;
  cmp "optout-eq-owned" 0.0 forced_off owned

let test_accel () =
  accel_set_override (-1);
  let avail = accel_available () = 1 in
  ok
    (Printf.sprintf "accel enabled reflects platform (avail=%b)" avail)
    (accel_enabled () = if avail then 1 else 0);
  List.iter
    (fun (m, k, n) ->
      test_accel_real ~kind:Buf.float32 ~name:"f32-accel" ~tol:2e-3 ~m ~k ~n ();
      test_accel_real ~kind:Buf.float64 ~name:"f64-accel" ~tol:1e-9 ~m ~k ~n ();
      test_accel_complex ~kind:Buf.complex64 ~name:"c32-accel" ~tol:2e-3 ~m ~k
        ~n ();
      test_accel_complex ~kind:Buf.complex128 ~name:"c64-accel" ~tol:1e-9 ~m ~k
        ~n ())
    [ (128, 128, 128); (256, 200, 192); (129, 300, 65) ]

(* ── Batch broadcast ──────────────────────────────────────────────────────*)

let test_batch_real (type b) ~(kind : (float, b) Buf.kind) ~name () =
  (* A:[G,m,k] B:[k,n] (broadcast over batch) C:[G,m,n]. Plus a both-batched
     case with a size-1 (stride-0) broadcast dim on B. *)
  let g = 3 and m = 5 and k = 4 and n = 6 in
  let fa idx = sin (float_of_int (idx * 13 mod 997)) in
  let fb idx = cos (float_of_int (idx * 7 mod 991)) in
  let abuf = Buf.create kind (g * m * k) in
  for t = 0 to (g * m * k) - 1 do
    Buf.set abuf t (fa t)
  done;
  let bbuf = Buf.create kind (k * n) in
  for t = 0 to (k * n) - 1 do
    Buf.set bbuf t (fb t)
  done;
  let a_ffi = ffi abuf [| g; m; k |] [| m * k; k; 1 |] in
  (* two equivalent RHS presentations: rank-2 (right-aligned, missing batch dim)
     and rank-3 with an explicit size-1 batch dim (stride-0 broadcast). *)
  let b_ffi = ffi bbuf [| k; n |] [| n; 1 |] in
  let b_ffi3 = ffi bbuf [| 1; k; n |] [| k * n; n; 1 |] in
  let refc = Array.make (g * m * n) 0.0 in
  for gg = 0 to g - 1 do
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        let s = ref 0.0 in
        for p = 0 to k - 1 do
          s :=
            !s
            +. Buf.get abuf ((gg * m * k) + (i * k) + p)
               *. Buf.get bbuf ((p * n) + j)
        done;
        refc.((gg * m * n) + (i * n) + j) <- !s
      done
    done
  done;
  let run label bf call =
    let cbuf = Buf.create kind (g * m * n) in
    call (ffi cbuf [| g; m; n |] [| m * n; n; 1 |]) a_ffi bf;
    let bad = ref 0 in
    for t = 0 to (g * m * n) - 1 do
      if
        abs_float (Buf.get cbuf t -. refc.(t))
        > 1e-4 +. (1e-4 *. abs_float refc.(t))
      then incr bad
    done;
    ok (Printf.sprintf "%s batch-bcast %s (bad=%d)" name label !bad) (!bad = 0)
  in
  run "rhs-rank2 prod" b_ffi (fun c a b -> mm c a b);
  run "rhs-rank2 direct" b_ffi (fun c a b -> mm_ex c a b 2);
  run "rhs-size1 prod" b_ffi3 (fun c a b -> mm c a b);
  run "rhs-size1 direct" b_ffi3 (fun c a b -> mm_ex c a b 2)

(* ── Integer dtypes ───────────────────────────────────────────────────────*)

let wrap_signed bits x =
  let m = x land ((1 lsl bits) - 1) in
  if m >= 1 lsl (bits - 1) then m - (1 lsl bits) else m

let wrap_unsigned bits x = x land ((1 lsl bits) - 1)

let test_int_small (type b) ~(kind : (int, b) Buf.kind) ~name ~wrap ~m ~k ~n
    ~vlo ~vhi ~modes () =
  let span = vhi - vlo + 1 in
  let fa i p = vlo + (((((i * k) + p) * 7) + 3) mod span) in
  let fb p j = vlo + (((((p * n) + j) * 5) + 1) mod span) in
  let abuf = Buf.create kind (m * k) in
  for i = 0 to m - 1 do
    for p = 0 to k - 1 do
      Buf.set abuf ((i * k) + p) (fa i p)
    done
  done;
  let bbuf = Buf.create kind (k * n) in
  for p = 0 to k - 1 do
    for j = 0 to n - 1 do
      Buf.set bbuf ((p * n) + j) (fb p j)
    done
  done;
  let a_ffi = ffi abuf [| m; k |] [| k; 1 |] in
  let b_ffi = ffi bbuf [| k; n |] [| n; 1 |] in
  let refc = Array.make (m * n) 0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0 in
      for p = 0 to k - 1 do
        s := !s + (Buf.get abuf ((i * k) + p) * Buf.get bbuf ((p * n) + j))
      done;
      refc.((i * n) + j) <- wrap !s
    done
  done;
  let run label call =
    let cbuf = Buf.create kind (m * n) in
    call (ffi cbuf [| m; n |] [| n; 1 |]) a_ffi b_ffi;
    let bad = ref 0 in
    for t = 0 to (m * n) - 1 do
      if Buf.get cbuf t <> refc.(t) then incr bad
    done;
    ok
      (Printf.sprintf "%s %s %dx%dx%d (bad=%d)" name label m k n !bad)
      (!bad = 0)
  in
  List.iter
    (function
      | `Prod -> run "prod" (fun c a b -> mm c a b)
      | `Direct -> run "direct" (fun c a b -> mm_ex c a b 2))
    modes

let test_i32 ~name ~m ~k ~n () =
  let fa i p = Int32.of_int ((((((i * k) + p) * 7) + 3) mod 21) - 10) in
  let fb p j = Int32.of_int ((((((p * n) + j) * 5) + 1) mod 21) - 10) in
  let abuf = Buf.create Buf.int32 (m * k) in
  for i = 0 to m - 1 do
    for p = 0 to k - 1 do
      Buf.set abuf ((i * k) + p) (fa i p)
    done
  done;
  let bbuf = Buf.create Buf.int32 (k * n) in
  for p = 0 to k - 1 do
    for j = 0 to n - 1 do
      Buf.set bbuf ((p * n) + j) (fb p j)
    done
  done;
  let a_ffi = ffi abuf [| m; k |] [| k; 1 |] in
  let b_ffi = ffi bbuf [| k; n |] [| n; 1 |] in
  let refc = Array.make (m * n) 0l in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0L in
      for p = 0 to k - 1 do
        s :=
          Int64.add !s
            (Int64.mul
               (Int64.of_int32 (Buf.get abuf ((i * k) + p)))
               (Int64.of_int32 (Buf.get bbuf ((p * n) + j))))
      done;
      refc.((i * n) + j) <- Int64.to_int32 !s
    done
  done;
  let cbuf = Buf.create Buf.int32 (m * n) in
  mm (ffi cbuf [| m; n |] [| n; 1 |]) a_ffi b_ffi;
  let bad = ref 0 in
  for t = 0 to (m * n) - 1 do
    if Buf.get cbuf t <> refc.(t) then incr bad
  done;
  ok (Printf.sprintf "%s %dx%dx%d (bad=%d)" name m k n !bad) (!bad = 0)

let test_i64 ~name ~m ~k ~n () =
  let fa i p = Int64.of_int ((((((i * k) + p) * 7) + 3) mod 21) - 10) in
  let fb p j = Int64.of_int ((((((p * n) + j) * 5) + 1) mod 21) - 10) in
  let abuf = Buf.create Buf.int64 (m * k) in
  for i = 0 to m - 1 do
    for p = 0 to k - 1 do
      Buf.set abuf ((i * k) + p) (fa i p)
    done
  done;
  let bbuf = Buf.create Buf.int64 (k * n) in
  for p = 0 to k - 1 do
    for j = 0 to n - 1 do
      Buf.set bbuf ((p * n) + j) (fb p j)
    done
  done;
  let a_ffi = ffi abuf [| m; k |] [| k; 1 |] in
  let b_ffi = ffi bbuf [| k; n |] [| n; 1 |] in
  let refc = Array.make (m * n) 0L in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0L in
      for p = 0 to k - 1 do
        s :=
          Int64.add !s
            (Int64.mul
               (Buf.get abuf ((i * k) + p))
               (Buf.get bbuf ((p * n) + j)))
      done;
      refc.((i * n) + j) <- !s
    done
  done;
  let cbuf = Buf.create Buf.int64 (m * n) in
  mm (ffi cbuf [| m; n |] [| n; 1 |]) a_ffi b_ffi;
  let bad = ref 0 in
  for t = 0 to (m * n) - 1 do
    if Buf.get cbuf t <> refc.(t) then incr bad
  done;
  ok (Printf.sprintf "%s %dx%dx%d (bad=%d)" name m k n !bad) (!bad = 0)

(* Modular-wrap contract (i64): the rows above keep values in -10..10 so the
   int64 accumulator never wraps — this one fills full-range random bits
   (splitmix64) salted with INT64_MIN/MAX, so nearly every product and partial
   sum overflows, pinning the unsigned-width wrap idiom (MM_MAC/MM_ADD, the
   u64-shared microkernel) that keeps the wrap defined without -fwrapv. The
   oracle is OCaml's Int64.add/mul — modular by definition, independent of the
   C. Runs policy (0), single-thread blocked (1), and direct (2): all three must
   match the oracle bit-for-bit, so blocked == direct follows. Callers pick
   shapes that land on the full-k register path (k <= MM_KC_FULLK_MAX) and the
   KC path (k > MM_KC_FULLK_MAX, mm_acc live). *)
let test_i64_wrap ~name ~m ~k ~n () =
  let st = ref 0x1234_5678_9ABC_DEF0L in
  let sm64 () =
    st := Int64.add !st 0x9E3779B97F4A7C15L;
    let z = !st in
    let z =
      Int64.mul
        (Int64.logxor z (Int64.shift_right_logical z 30))
        0xBF58476D1CE4E5B9L
    in
    let z =
      Int64.mul
        (Int64.logxor z (Int64.shift_right_logical z 27))
        0x94D049BB133111EBL
    in
    Int64.logxor z (Int64.shift_right_logical z 31)
  in
  let abuf = Buf.create Buf.int64 (m * k) in
  for t = 0 to (m * k) - 1 do
    Buf.set abuf t (sm64 ())
  done;
  let bbuf = Buf.create Buf.int64 (k * n) in
  for t = 0 to (k * n) - 1 do
    Buf.set bbuf t (sm64 ())
  done;
  (* salt the extremes so the outer products certainly include MAX*MAX
     (A(0,0)*B(0,1) at p=0), MAX*MIN (A(0,0)*B(0,0) at p=0) and MIN*MIN
     (A(0,1)*B(1,0) at p=1) — the contraction indices must line up for a product
     to form, so B needs a MIN at p=1 too. *)
  Buf.set abuf 0 Int64.max_int;
  Buf.set abuf 1 Int64.min_int;
  Buf.set bbuf 0 Int64.min_int;
  Buf.set bbuf 1 Int64.max_int;
  Buf.set bbuf n Int64.min_int;
  let refc = Array.make (m * n) 0L in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0L in
      for p = 0 to k - 1 do
        s :=
          Int64.add !s
            (Int64.mul
               (Buf.get abuf ((i * k) + p))
               (Buf.get bbuf ((p * n) + j)))
      done;
      refc.((i * n) + j) <- !s
    done
  done;
  let a_ffi = ffi abuf [| m; k |] [| k; 1 |] in
  let b_ffi = ffi bbuf [| k; n |] [| n; 1 |] in
  List.iter
    (fun mode ->
      let cbuf = Buf.create Buf.int64 (m * n) in
      mm_ex (ffi cbuf [| m; n |] [| n; 1 |]) a_ffi b_ffi mode;
      let bad = ref 0 in
      for t = 0 to (m * n) - 1 do
        if Buf.get cbuf t <> refc.(t) then incr bad
      done;
      ok
        (Printf.sprintf "%s %dx%dx%d mode%d wrap (bad=%d)" name m k n mode !bad)
        (!bad = 0))
    [ 0; 1; 2 ]

(* Hand-derived literal, independent of every code path: [MAX MIN]·[MAX; MIN] =
   MAX^2 + MIN^2 = (2^126 - 2^64 + 1) + 2^126 == 1 (mod 2^64). m=1 < MR forces
   the direct path (MM_MAC). *)
let test_i64_wrap_fixture () =
  let abuf = Buf.create Buf.int64 2 in
  Buf.set abuf 0 Int64.max_int;
  Buf.set abuf 1 Int64.min_int;
  let bbuf = Buf.create Buf.int64 2 in
  Buf.set bbuf 0 Int64.max_int;
  Buf.set bbuf 1 Int64.min_int;
  let cbuf = Buf.create Buf.int64 1 in
  mm
    (ffi cbuf [| 1; 1 |] [| 1; 1 |])
    (ffi abuf [| 1; 2 |] [| 2; 1 |])
    (ffi bbuf [| 2; 1 |] [| 1; 1 |]);
  ok
    (Printf.sprintf "i64 wrap fixture MAX^2+MIN^2 = %Ld" (Buf.get cbuf 0))
    (Buf.get cbuf 0 = 1L)

(* u32/u64: small non-negative values, unambiguous in the int32/int64
   storage. *)
let test_u32 ~m ~k ~n () =
  let abuf = Buf.create Buf.uint32 (m * k) in
  for t = 0 to (m * k) - 1 do
    Buf.set abuf t (Int32.of_int (t * 3 mod 11))
  done;
  let bbuf = Buf.create Buf.uint32 (k * n) in
  for t = 0 to (k * n) - 1 do
    Buf.set bbuf t (Int32.of_int (t * 5 mod 7))
  done;
  let refc = Array.make (m * n) 0l in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0l in
      for p = 0 to k - 1 do
        s :=
          Int32.add !s
            (Int32.mul
               (Buf.get abuf ((i * k) + p))
               (Buf.get bbuf ((p * n) + j)))
      done;
      refc.((i * n) + j) <- !s
    done
  done;
  let cbuf = Buf.create Buf.uint32 (m * n) in
  mm
    (ffi cbuf [| m; n |] [| n; 1 |])
    (ffi abuf [| m; k |] [| k; 1 |])
    (ffi bbuf [| k; n |] [| n; 1 |]);
  let bad = ref 0 in
  for t = 0 to (m * n) - 1 do
    if Buf.get cbuf t <> refc.(t) then incr bad
  done;
  ok (Printf.sprintf "u32 %dx%dx%d (bad=%d)" m k n !bad) (!bad = 0)

let test_u64 ~m ~k ~n () =
  let abuf = Buf.create Buf.uint64 (m * k) in
  for t = 0 to (m * k) - 1 do
    Buf.set abuf t (Int64.of_int (t * 3 mod 11))
  done;
  let bbuf = Buf.create Buf.uint64 (k * n) in
  for t = 0 to (k * n) - 1 do
    Buf.set bbuf t (Int64.of_int (t * 5 mod 7))
  done;
  let refc = Array.make (m * n) 0L in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0L in
      for p = 0 to k - 1 do
        s :=
          Int64.add !s
            (Int64.mul
               (Buf.get abuf ((i * k) + p))
               (Buf.get bbuf ((p * n) + j)))
      done;
      refc.((i * n) + j) <- !s
    done
  done;
  let cbuf = Buf.create Buf.uint64 (m * n) in
  mm
    (ffi cbuf [| m; n |] [| n; 1 |])
    (ffi abuf [| m; k |] [| k; 1 |])
    (ffi bbuf [| k; n |] [| n; 1 |]);
  let bad = ref 0 in
  for t = 0 to (m * n) - 1 do
    if Buf.get cbuf t <> refc.(t) then incr bad
  done;
  ok (Printf.sprintf "u64 %dx%dx%d (bad=%d)" m k n !bad) (!bad = 0)

(* ── Complex dtypes ───────────────────────────────────────────────────────*)

let test_complex (type b) ~(kind : (Complex.t, b) Buf.kind) ~name ~tol ~m ~k ~n
    ~modes () =
  let fa i p =
    {
      Complex.re = sin (float_of_int (((i * k) + p) mod 97));
      im = cos (float_of_int (((i * k) + p) mod 89));
    }
  in
  let fb p j =
    {
      Complex.re = cos (float_of_int (((p * n) + j) mod 83));
      im = sin (float_of_int (((p * n) + j) mod 79));
    }
  in
  let abuf = Buf.create kind (m * k) in
  for i = 0 to m - 1 do
    for p = 0 to k - 1 do
      Buf.set abuf ((i * k) + p) (fa i p)
    done
  done;
  let bbuf = Buf.create kind (k * n) in
  for p = 0 to k - 1 do
    for j = 0 to n - 1 do
      Buf.set bbuf ((p * n) + j) (fb p j)
    done
  done;
  let a_ffi = ffi abuf [| m; k |] [| k; 1 |] in
  let b_ffi = ffi bbuf [| k; n |] [| n; 1 |] in
  let refc = Array.make (m * n) Complex.zero in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref Complex.zero in
      for p = 0 to k - 1 do
        s :=
          Complex.add !s
            (Complex.mul
               (Buf.get abuf ((i * k) + p))
               (Buf.get bbuf ((p * n) + j)))
      done;
      refc.((i * n) + j) <- !s
    done
  done;
  let run label call =
    let cbuf = Buf.create kind (m * n) in
    call (ffi cbuf [| m; n |] [| n; 1 |]) a_ffi b_ffi;
    let bad = ref 0 in
    for t = 0 to (m * n) - 1 do
      let g = Buf.get cbuf t and r = refc.(t) in
      if
        abs_float (g.Complex.re -. r.Complex.re) > tol
        || abs_float (g.Complex.im -. r.Complex.im) > tol
      then incr bad
    done;
    ok
      (Printf.sprintf "%s %s %dx%dx%d (bad=%d)" name label m k n !bad)
      (!bad = 0)
  in
  List.iter
    (function
      | `Prod -> run "prod" (fun c a b -> mm c a b)
      | `Direct -> run "direct" (fun c a b -> mm_ex c a b 2))
    modes

(* ── bool: matmul is arithmetic, unsupported (frontend promotes bool away) ──*)

let test_bool_unsupported () =
  let m = 4 and k = 4 and n = 4 in
  let mkbool sz = Buf.create Buf.bool sz in
  let a = mkbool (m * k) and b = mkbool (k * n) and c = mkbool (m * n) in
  let raised =
    try
      mm
        (ffi c [| m; n |] [| n; 1 |])
        (ffi a [| m; k |] [| k; 1 |])
        (ffi b [| k; n |] [| n; 1 |]);
      false
    with Failure _ -> true
  in
  ok "bool matmul raises Failure (unsupported)" raised

(* ── int4/uint4: packed, no compute kernel — must raise (not corrupt nibbles)
   ─*)

let test_packed_unsupported (type b) ~(kind : (int, b) Buf.kind) ~name () =
  let m = 4 and k = 4 and n = 4 in
  let a = Buf.create kind (m * k)
  and b = Buf.create kind (k * n)
  and c = Buf.create kind (m * n) in
  let raised =
    try
      mm
        (ffi c [| m; n |] [| n; 1 |])
        (ffi a [| m; k |] [| k; 1 |])
        (ffi b [| k; n |] [| n; 1 |]);
      false
    with Failure _ -> true
  in
  ok
    (Printf.sprintf "%s matmul raises Failure (packed, unsupported)" name)
    raised

(* ── Run ──────────────────────────────────────────────────────────────────*)

let test_maintenance_paths () =
  (* f32 / f64: full size sweep, all transpose combos, three modes, vs f64
     ref *)
  let sizes =
    [
      1; 2; 3; 4; 5; 7; 8; 9; 16; 31; 32; 33; 48; 63; 64; 65; 100; 127; 128; 129;
    ]
  in
  List.iter
    (fun s ->
      List.iter
        (fun (at, bt) ->
          test_real ~kind:Buf.float32 ~name:"f32" ~tol_rel:1e-3 ~tol_abs:1e-3
            ~m:s ~k:s ~n:s ~a_trans:at ~b_trans:bt
            ~modes:[ `Prod; `St; `Direct ] ();
          test_real ~kind:Buf.float64 ~name:"f64" ~tol_rel:1e-9 ~tol_abs:1e-9
            ~m:s ~k:s ~n:s ~a_trans:at ~b_trans:bt
            ~modes:[ `Prod; `St; `Direct ] ())
        [ (false, false); (true, false); (false, true); (true, true) ])
    sizes;

  (* rectangular shapes (non-square m,k,n), edge tile fractions *)
  List.iter
    (fun (m, k, n) ->
      test_real ~kind:Buf.float32 ~name:"f32" ~tol_rel:1e-3 ~tol_abs:1e-3 ~m ~k
        ~n ~modes:[ `Prod; `Direct ] ();
      test_real ~kind:Buf.float64 ~name:"f64" ~tol_rel:1e-9 ~tol_abs:1e-9 ~m ~k
        ~n ~modes:[ `Prod; `Direct ] ())
    [
      (1, 200, 1);
      (200, 1, 200);
      (17, 200, 13);
      (130, 70, 90);
      (7, 8, 200);
      (200, 3, 5);
    ];

  (* offsets on all three operands *)
  test_real ~kind:Buf.float32 ~name:"f32" ~tol_rel:1e-3 ~tol_abs:1e-3 ~m:40
    ~k:40 ~n:40 ~a_off:5 ~b_off:7 ~c_off:3 ~modes:[ `Prod; `Direct ] ();
  test_real ~kind:Buf.float64 ~name:"f64" ~tol_rel:1e-9 ~tol_abs:1e-9 ~m:130
    ~k:70 ~n:90 ~a_off:11 ~b_off:0 ~c_off:9 ~a_trans:true
    ~modes:[ `Prod; `Direct ] ();

  (* larger sizes: blocked-vs-naive differential (no OCaml reference) + one
     ref *)
  List.iter
    (fun s ->
      test_diff ~kind:Buf.float32 ~name:"f32" ~tol:1e-4 ~m:s ~k:s ~n:s ();
      test_diff ~kind:Buf.float64 ~name:"f64" ~tol:1e-10 ~m:s ~k:s ~n:s ())
    [ 200; 256; 384 ];
  test_real ~kind:Buf.float32 ~name:"f32" ~tol_rel:2e-3 ~tol_abs:2e-3 ~m:512
    ~k:512 ~n:512 ~modes:[ `Prod; `Owned ] ();
  test_real ~kind:Buf.float64 ~name:"f64" ~tol_rel:1e-9 ~tol_abs:1e-9 ~m:300
    ~k:300 ~n:300 ~modes:[ `Prod; `Owned ] ();

  (* Large-k KC blocking: k > MM_KC_FULLK_MAX drives the contraction sub-block
     path, summing many KC panels into a compute-typed accumulator. The witness:
     f16 inputs whose per-element sum crosses KC-panel boundaries must stay
     f32-accumulated (compute type) to match the f64 reference — accumulating in
     f16 storage across the boundary would drift out of tolerance. Every mode
     (prod/1t/direct) is checked against the same f64/quantized reference, and
     1t forces the KC path single-threaded (the panel path, small m/n). *)
  List.iter
    (fun (m, k, n) ->
      test_real ~kind:Buf.float32 ~name:"f32-largek" ~tol_rel:1e-3 ~tol_abs:1e-3
        ~m ~k ~n ~modes:[ `Prod; `St; `Direct ] ();
      test_real ~kind:Buf.float64 ~name:"f64-largek" ~tol_rel:1e-9 ~tol_abs:1e-9
        ~m ~k ~n ~modes:[ `Prod; `St; `Direct ] ();
      test_real ~kind:Buf.float16 ~name:"f16-largek" ~tol_rel:2e-2 ~tol_abs:2e-2
        ~m ~k ~n ~scale:0.5 ~modes:[ `Prod; `St; `Direct ] ())
    [ (16, 4096, 16); (24, 5000, 40); (8, 8192, 12) ];
  (* The sharp storage-precision detector (cancellation, true answer = 0): a
     storage-typed Cacc leaves a large residual, a compute-typed one stays ~0.
     f16 and bf16 (bf16's 8-bit mantissa makes the leak even larger). *)
  List.iter
    (fun (m, k, n) ->
      test_kc_cancel ~kind:Buf.float16 ~name:"f16" ~m ~k ~n ();
      test_kc_cancel ~kind:Buf.bfloat16 ~name:"bf16" ~m ~k ~n ())
    [ (16, 4096, 16); (32, 6144, 24) ];

  (* Large-k over the non-float compute microkernels, so the
     int64/uint64/c32/c64 mm_acc instances actually execute on the KC path (k >
     MM_KC_FULLK_MAX). Exact modular / complex references; values bounded so
     int64 accumulation never overflows across the KC-panel sum. *)
  List.iter
    (fun (m, k, n) ->
      test_i64 ~name:"i64-largek" ~m ~k ~n ();
      test_u64 ~m ~k ~n ();
      test_complex ~kind:Buf.complex64 ~name:"c32-largek" ~tol:1e-3 ~m ~k ~n
        ~modes:[ `Prod; `Direct ] ();
      test_complex ~kind:Buf.complex128 ~name:"c64-largek" ~tol:1e-9 ~m ~k ~n
        ~modes:[ `Prod; `Direct ] ())
    [ (16, 2100, 16); (24, 4096, 20) ];

  (* Modular-wrap differential: full-range + MIN/MAX-salted i64, blocked full-k
     and KC-blocked vs direct vs the Int64 oracle, plus the hand-derived
     MAX^2+MIN^2 == 1 fixture. *)
  test_i64_wrap ~name:"i64-wrap" ~m:24 ~k:200 ~n:24 ();
  test_i64_wrap ~name:"i64-wrap-kc" ~m:16 ~k:2100 ~n:16 ();
  test_i64_wrap_fixture ();

  (* M-partition: a narrow matrix (n <= NC, or too few panels to fill the pool)
     drives the (batch x panel x MC-block) split with B pre-packed once into a
     shared buffer. Cross-check the M-split blocked result against the naive
     triple loop. Includes m NOT divisible by MC=256 (last MC-block has mr < MR,
     the remainder-tile path in mm_mpar_body) and one case crossing
     MM_KC_FULLK_MAX (both levers at once) — disjoint (ic, jc) tiles must
     reconstruct the full product. *)
  List.iter
    (fun (m, k, n) ->
      test_diff ~kind:Buf.float32 ~name:"f32-msplit" ~tol:1e-4 ~m ~k ~n ();
      test_diff ~kind:Buf.float64 ~name:"f64-msplit" ~tol:1e-10 ~m ~k ~n ())
    [
      (2048, 256, 128);
      (4096, 128, 64);
      (1024, 300, 200);
      (512, 4096, 128);
      (2050, 256, 128);
      (1300, 260, 130);
    ];

  (* nx_c_gemm2d_ct_ws (linalg caller-workspace GEMM), incl. the large-k
     Cacc-alloc branch, vs the direct oracle. *)
  List.iter
    (fun (m, k, n) ->
      test_ws ~kind:Buf.float32 ~name:"f32-ws" ~tol:1e-4 ~m ~k ~n ();
      test_ws ~kind:Buf.float64 ~name:"f64-ws" ~tol:1e-10 ~m ~k ~n ())
    [ (96, 300, 80); (64, 2100, 48); (128, 4096, 96) ];

  (* Accelerate hook (macOS default-on): platform reflection, the hook-vs-owned
     oracle differential, and forced-owned policy equivalence. Owned-vs-owned
     off macOS, so green everywhere. *)
  test_accel ();

  (* batch broadcast *)
  test_batch_real ~kind:Buf.float32 ~name:"f32" ();
  test_batch_real ~kind:Buf.float64 ~name:"f64" ();

  (* low precision through-pack: f16 bf16 fp8 (small k, generous tol) *)
  List.iter
    (fun s ->
      test_real ~kind:Buf.float16 ~name:"f16" ~tol_rel:2e-2 ~tol_abs:2e-2 ~m:s
        ~k:s ~n:s ~scale:0.5 ~modes:[ `Prod; `Direct ] ();
      test_real ~kind:Buf.bfloat16 ~name:"bf16" ~tol_rel:6e-2 ~tol_abs:6e-2 ~m:s
        ~k:s ~n:s ~scale:0.5 ~modes:[ `Prod; `Direct ] ())
    [ 8; 16; 33; 64 ];
  List.iter
    (fun s ->
      test_real ~kind:Buf.float8_e4m3 ~name:"fp8e4m3" ~tol_rel:0.2 ~tol_abs:0.3
        ~m:s ~k:s ~n:s ~scale:0.4 ~modes:[ `Prod; `Direct ] ();
      test_real ~kind:Buf.float8_e5m2 ~name:"fp8e5m2" ~tol_rel:0.35 ~tol_abs:0.4
        ~m:s ~k:s ~n:s ~scale:0.4 ~modes:[ `Prod; `Direct ] ())
    [ 8; 16; 32 ];
  (* transposed low-precision, to exercise the strided pack path *)
  test_real ~kind:Buf.float16 ~name:"f16" ~tol_rel:2e-2 ~tol_abs:2e-2 ~m:40
    ~k:40 ~n:40 ~a_trans:true ~b_trans:true ~scale:0.5 ~modes:[ `Prod; `Direct ]
    ();

  (* integers: exact modular reference, incl. edge sizes *)
  List.iter
    (fun s ->
      test_int_small ~kind:Buf.int8 ~name:"i8" ~wrap:(wrap_signed 8) ~m:s ~k:s
        ~n:s ~vlo:(-4) ~vhi:4 ~modes:[ `Prod; `Direct ] ();
      test_int_small ~kind:Buf.uint8 ~name:"u8" ~wrap:(wrap_unsigned 8) ~m:s
        ~k:s ~n:s ~vlo:0 ~vhi:5 ~modes:[ `Prod; `Direct ] ();
      test_int_small ~kind:Buf.int16 ~name:"i16" ~wrap:(wrap_signed 16) ~m:s
        ~k:s ~n:s ~vlo:(-9) ~vhi:9 ~modes:[ `Prod; `Direct ] ();
      test_int_small ~kind:Buf.uint16 ~name:"u16" ~wrap:(wrap_unsigned 16) ~m:s
        ~k:s ~n:s ~vlo:0 ~vhi:12 ~modes:[ `Prod; `Direct ] ())
    [ 3; 8; 33; 64 ];
  List.iter
    (fun s ->
      test_i32 ~name:"i32" ~m:s ~k:s ~n:s ();
      test_i64 ~name:"i64" ~m:s ~k:s ~n:s ();
      test_u32 ~m:s ~k:s ~n:s ();
      test_u64 ~m:s ~k:s ~n:s ())
    [ 8; 40 ];

  (* i8/i16 modular wrap: partial sums overflow the storage width *)
  test_int_small ~kind:Buf.int8 ~name:"i8-wrap" ~wrap:(wrap_signed 8) ~m:4 ~k:20
    ~n:4 ~vlo:10 ~vhi:12 ~modes:[ `Prod; `Direct ] ();
  test_int_small ~kind:Buf.uint8 ~name:"u8-wrap" ~wrap:(wrap_unsigned 8) ~m:4
    ~k:20 ~n:4 ~vlo:10 ~vhi:15 ~modes:[ `Prod; `Direct ] ();
  test_int_small ~kind:Buf.int16 ~name:"i16-wrap" ~wrap:(wrap_signed 16) ~m:4
    ~k:64 ~n:4 ~vlo:200 ~vhi:250 ~modes:[ `Prod; `Direct ] ();

  (* complex: 3M-free direct complex microkernel vs complex reference *)
  List.iter
    (fun s ->
      test_complex ~kind:Buf.complex64 ~name:"c32" ~tol:1e-3 ~m:s ~k:s ~n:s
        ~modes:[ `Prod; `Direct ] ();
      test_complex ~kind:Buf.complex128 ~name:"c64" ~tol:1e-9 ~m:s ~k:s ~n:s
        ~modes:[ `Prod; `Direct ] ())
    [ 3; 8; 33; 64 ];

  (* bool + packed int4/uint4 unsupported *)
  test_bool_unsupported ();
  test_packed_unsupported ~kind:Buf.int4 ~name:"i4" ();
  test_packed_unsupported ~kind:Buf.uint4 ~name:"u4" ()

let () =
  Windtrap.run "nx C backend matmul"
    [
      group "internal-path-equivalence"
        [ test "owned, workspace, and Accelerate paths" test_maintenance_paths ];
    ]
