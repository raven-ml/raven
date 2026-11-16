(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The spectral routines' launches: spectral.hip's steps and the
   contraction's products, one call.

     operand --(copy, unless C-contiguous)--> workspace
       --> eigh: Jacobi (n <= 32), else tridiagonalisation, divide and
           conquer (or bisection for values alone), back-transformation
       --> svd: Jacobi (both extents <= 32), else bidiagonalisation, QR
           sweeps (or bisection), back-transformation
       --> eig: one group a matrix, then Z X by a product

   The steps keep each matrix's workspace as the layouts below lay it
   out, which their words pass to the device. A product's ranges of k
   follow each matrix's extents alone, so no result's bits depend on the
   batch.

   An iteration that does not converge within its limit makes its matrix
   NaN in every result: 60 Jacobi sweeps (sp_jacobi.h), 30 steps of a
   secular equation and 40 iterations of a step's equation over three
   poles (sp_stedc.h), 30 max(10, h) QR iterations of eig, h the order
   balancing leaves to iterate on (sp_eig.h), and 6 k² rotations of the
   bidiagonal QR sweeps (sp_bd.h). A call reads all but the 40 once, and
   each of its launches passes them as words, which tests lower. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels

let cdiv a b = (a + b - 1) / b
let round x = (x + 255) land lnot 255

(* Limits *)

type limits = { sweeps : int; steps : int; qr : int; rotations : int }

let limits = Atomic.make { sweeps = 60; steps = 30; qr = 30; rotations = 6 }

let bound_iterations ~sweeps ~steps ~qr ~rotations =
  if sweeps < 1 then invalid_arg "Nx_amd.bound_iterations: sweeps";
  if steps < 1 then invalid_arg "Nx_amd.bound_iterations: steps";
  if qr < 0 then invalid_arg "Nx_amd.bound_iterations: qr";
  if rotations < 0 then invalid_arg "Nx_amd.bound_iterations: rotations";
  Atomic.set limits { sweeps; steps; qr; rotations }

(* Layouts

   [t] and [r] the bytes of an element and of its real part. *)

(* A Jacobi group's dynamic LDS: what every Jacobi kernel leaves, so that
   a matrix's group depends on its format and extents alone. *)
let jacobi_lds =
  List.fold_left
    (fun most k -> Int.min most (Plan.shared_most k))
    max_int
    K.
      [
        Sp_eigh_jacobi_f32;
        Sp_eigh_jacobi_f64;
        Sp_eigh_jacobi_c64;
        Sp_eigh_jacobi_c128;
        Sp_svd_jacobi_f32;
        Sp_svd_jacobi_f64;
        Sp_svd_jacobi_c64;
        Sp_svd_jacobi_c128;
      ]

(* Jacobi's matrices a group, as many as its work-items take, and their
   LDS: eigh's of order n, svd's of m x n. *)
let eigh_per n = if n * n >= K.sp_threads then 1 else K.sp_threads / (n * n)

let eigh_lds n ~t ~r =
  let per = eigh_per n in
  (per * ((2 * n * n) + (2 * n)) * t) + (((per * n) + 1) * 4) + (per * n * r)

let svd_span ~mm ~k = (mm * k) + (k * k) + (mm * mm) + (2 * k) + mm

let svd_per m n ~t ~r =
  let mm = Int.max m n and k = Int.min m n in
  let slot = (svd_span ~mm ~k * t) + (2 * k * 4) + ((k + mm) * r) in
  let per = ref (if m * n >= K.sp_threads then 1 else K.sp_threads / (m * n)) in
  while !per > 1 && !per * slot > jacobi_lds do
    decr per
  done;
  !per

let svd_lds m n ~t ~r =
  let mm = Int.max m n and k = Int.min m n in
  let per = svd_per m n ~t ~r in
  (per * svd_span ~mm ~k * t) + (per * 2 * k * 4) + (per * (k + mm) * r)

(* A matrix's workspace, on 256 bytes: the tridiagonal reduction's (A, W,
   Vc, T, G, tau; d, e; two words), divide and conquer's (Q, Qt, S; ten
   vectors; six of ints and four words) and eig's (H, Z, X; X's complex
   copy; tau; four vectors of R; a word). *)
let tnb = K.sp_tnb
let leaf = K.sp_leaf

let td_per n ~t ~r =
  round
    ((((n * n) + (2 * n * tnb) + (2 * tnb * tnb) + n) * t) + (2 * n * r) + 16)

let dc_per n ~r = round ((((3 * n * n) + (10 * n)) * r) + (((6 * n) + 4) * 4))

let eig_per n ~t ~r =
  round ((3 * n * n * t) + (n * n * 2 * r) + (n * t) + (4 * n * r) + 16)

(* A large svd's workspace a matrix, as byte offsets: its arrays of
   elements (g, x, y, vc, tm, gg, w1, w2, wv, tauq, taup), of reals (d, e,
   U_B, V_B, 4 a rotation record, scalars) and of ints (sweeps, state);
   [cap] the records, and the sweeps, a pass holds. g is M x K, M >= K, gc
   the columns of its left factor the call stores. *)
type bd = {
  g : int;
  x : int;
  y : int;
  vc : int;
  tm : int;
  gg : int;
  w1 : int;
  w2 : int;
  wv : int;
  tauq : int;
  taup : int;
  d : int;
  e : int;
  ub : int;
  vb : int;
  rec_ : int;
  sc : int;
  sw : int;
  st : int;
  per : int;
  cap : int;
}

let bd_layout ~mm ~k ~gc ~t ~r =
  let o = ref 0 in
  let field n w =
    let at = !o in
    o := !o + (n * w);
    at
  in
  let mk = Int.max mm k and wc = Int.max gc k and nb = K.sp_bd_nb in
  let cap = (64 * k) + 64 in
  let g = field (mm * k) t in
  let x = field (mm * gc) t in
  let y = field (k * k) t in
  let vc = field (mk * nb) t in
  let tm = field (nb * nb) t in
  let gg = field (nb * nb) t in
  let w1 = field (nb * wc) t in
  let w2 = field (nb * wc) t in
  let wv = field mk t in
  let tauq = field k t in
  let taup = field k t in
  let d = field k r in
  let e = field k r in
  let ub = field (k * k) r in
  let vb = field (k * k) r in
  let rec_ = field (4 * cap) r in
  let sc = field 4 r in
  let sw = field (3 * cap) 4 in
  let st = field K.sp_st_words 4 in
  let per = round !o in
  {
    g; x; y; vc; tm; gg; w1; w2; wv; tauq; taup; d; e; ub; vb; rec_; sc; sw;
    st; per; cap;
  }

let bd_words w l =
  List.iter
    (fun (i, v) -> Plan.set w i v)
    [
      (K.sp_bd_g, l.g); (K.sp_bd_x, l.x); (K.sp_bd_y, l.y);
      (K.sp_bd_vc, l.vc); (K.sp_bd_tm, l.tm); (K.sp_bd_gg, l.gg);
      (K.sp_bd_w1, l.w1); (K.sp_bd_w2, l.w2); (K.sp_bd_wv, l.wv);
      (K.sp_bd_tauq, l.tauq); (K.sp_bd_taup, l.taup); (K.sp_bd_d, l.d);
      (K.sp_bd_e, l.e); (K.sp_bd_ub, l.ub); (K.sp_bd_vb, l.vb);
      (K.sp_bd_rec, l.rec_); (K.sp_bd_sc, l.sc); (K.sp_bd_sw, l.sw);
      (K.sp_bd_st, l.st); (K.sp_bd_per, l.per); (K.sp_bd_cap, l.cap);
    ]

(* Formats *)

(* A routine's arithmetic: its NX_SP_ code, the bytes of its element and
   of its real part, its dtype, and the call's limits, read once when the
   call is planned. *)
type fmt = { code : int; t : int; r : int; dt : D.any; lim : limits }

let fmt_of ~lim (D.Any d as dt) =
  match d with
  | D.Float32 -> { code = K.sp_f32; t = 4; r = 4; dt; lim }
  | D.Float64 -> { code = K.sp_f64; t = 8; r = 8; dt; lim }
  | D.Complex64 -> { code = K.sp_c64; t = 8; r = 4; dt; lim }
  | D.Complex128 -> { code = K.sp_c128; t = 16; r = 8; dt; lim }
  | _ -> invalid_arg "Spectral: not a routine's dtype"

let real_fmt f =
  let dt = if f.r = 4 then D.Any D.Float32 else D.Any D.Float64 in
  fmt_of ~lim:f.lim dt

(* The kernel of [f]'s format among four, or two of real ones. *)
let by f k32 k64 c64 c128 =
  if f.code = K.sp_f32 then k32
  else if f.code = K.sp_f64 then k64
  else if f.code = K.sp_c64 then c64
  else c128

let by_real f k32 k64 = if f.r = 4 then k32 else k64

(* Calls *)

(* A launch's groups: [Over (n, e)] along x, striding over [n] units, [e]
   a group ({!Plan.grid}); [Tiles (x, y, z)] one a tile, those along z
   striding past 65,535, since the dispatch keeps the counts along y and z
   modulo 65,536. *)
type groups = Over of int * int | Tiles of int * int * int

let yz_most = 65535

(* An argument of a step: an address read or written, an integer, or
   NULL. *)
type arg = Read of int * int | Write of int * int | N of int | Null

let byte = function Read (_, at) | Write (_, at) -> at | N _ | Null -> 0

let plus a n =
  match a with
  | Read (s, at) -> Read (s, at + n)
  | Write (s, at) -> Write (s, at + n)
  | x -> x

(* [n] bytes of workspace, as a written address. *)
let take b n = Write (Plan.workspace b, Plan.take b n)

(* Adds to [b] a launch of [k] in [f]'s format over [groups] with [args]
   at words 0 on, and the workspace's offsets: [per] the bytes of a
   matrix's, [span] Jacobi's elements of one, [bd] a large svd's
   layout. *)
let launch b f k ~groups ?(shared = 0) ?(per = 0) ?(span = 0) ?bd args =
  let w = Plan.words k and refs = ref [] in
  let address i s at write =
    Plan.set w i at;
    refs := { Plan.word = i; slot = s; write } :: !refs
  in
  List.iteri
    (fun i a ->
      match a with
      | Read (s, at) -> address i s at false
      | Write (s, at) -> address i s at true
      | N v -> Plan.set w i v
      | Null -> ())
    args;
  let gx, gy, gz =
    match groups with
    | Over (n, e) -> (Plan.grid ~per:e k n, 1, 1)
    | Tiles (x, y, z) -> (x, Int.min y yz_most, Int.min z yz_most)
  in
  Plan.set w K.sp_fmt f.code;
  Plan.set w K.sp_gx gx;
  Plan.set w K.sp_gy gy;
  Plan.set w K.sp_gz gz;
  Plan.set w K.sp_threads_of (K.threads k);
  Plan.set w K.sp_per per;
  Plan.set w K.sp_span span;
  Plan.set w K.sp_sweeps f.lim.sweeps;
  Plan.set w K.sp_steps f.lim.steps;
  Plan.set w K.sp_qr f.lim.qr;
  Plan.set w K.sp_rotations f.lim.rotations;
  Option.iter (bd_words w) bd;
  Plan.launch b k ~groups:(gx, gy, gz) ~shared w (Array.of_list !refs)

(* [c <- op(a) op(b)], plus [c] where [acc], negated where [neg], over
   [nz] matrices of the workspace at the bytes [cb], [ab] and [bb]:
   row-major, [op] 0 as it lies, 1 transposed, 2 conjugate-transposed,
   rows [lda], [ldb], [ldc] and matrices [sa], [sb], [sc] elements apart.
   The contraction's product, in the ranges of k each matrix's extents
   give, so that no result's bits depend on the batch. The workspace it
   takes returns once it is planned. *)
let gemm b f ~cb ~ab ~bb ~m ~n ~k ~lda ~ldb ~ldc ~opa ~opb ~sa ~sb ~sc ~acc
    ~neg ~nz =
  if m > 0 && n > 0 && nz > 0 then begin
    let view at ~s ~rs ~cs ~rows ~cols =
      let l =
        L.v
          ~strides:(Iarray.of_array [| s; rs; cs |])
          (Iarray.of_array [| nz; rows; cols |])
      in
      Contract.work ~at f.dt l
    in
    let op o at ~ld ~s ~rows ~cols =
      if o = 0 then view at ~s ~rs:ld ~cs:1 ~rows ~cols
      else view at ~s ~rs:1 ~cs:ld ~rows ~cols
    in
    let o =
      {
        Contract.plain with
        neg;
        conj_a = opa = 2;
        conj_b = opb = 2;
        ranges = Some (Contract.ranges ~rows:m ~cols:n ~k);
      }
    in
    let mark = Plan.mark b in
    Contract.product b o ~acc:f.dt ~init:acc
      ~y:(view cb ~s:sc ~rs:ldc ~cs:1 ~rows:m ~cols:n)
      (op opa ab ~ld:lda ~s:sa ~rows:m ~cols:k)
      (op opb bb ~ld:ldb ~s:sb ~rows:k ~cols:n);
    Plan.release b mark
  end

(* eigh *)

let rec eigh_dev b f ~a ~w ~v ~nb ~n =
  if nb > 0 && n > 0 then
    if n <= K.sp_jacobi then
      let per = eigh_per n in
      launch b f
        (by f K.Sp_eigh_jacobi_f32 K.Sp_eigh_jacobi_f64 K.Sp_eigh_jacobi_c64
           K.Sp_eigh_jacobi_c128)
        ~groups:(Over (nb, per))
        ~shared:(eigh_lds n ~t:f.t ~r:f.r)
        ~per [ a; w; v; N nb; N n ]
    else eigh_large b f ~a ~w ~v ~nb ~n

(* Tridiagonalisation, divide and conquer or bisection, the reflectors'
   blocks applied. *)
and eigh_large b f ~a ~w ~v ~nb ~n =
  let rf = real_fmt f in
  let t = f.t and r = f.r and nn = n * n in
  let tp = td_per n ~t ~r in
  let ws = take b (nb * tp) in
  let wsb = byte ws in
  let td = launch b f ~per:tp in
  td K.Sp_td_stage ~groups:(Over (nb, 1)) [ a; ws; N nb; N n ];
  let st = tp / t in
  let j0 = ref 0 in
  while !j0 < n do
    let nbk = Int.min tnb (n - !j0) in
    for i = 0 to nbk do
      td
        (by f K.Sp_td_step_f32 K.Sp_td_step_f64 K.Sp_td_step_c64
           K.Sp_td_step_c128)
        ~groups:(Over (nb, 1))
        [ ws; N nb; N n; N !j0; N nbk; N i ];
      let col = !j0 + i in
      if i < nbk && col < n - 1 then
        td K.Sp_td_hemv
          ~groups:(Over (nb * cdiv (n - col - 1) 8, 1))
          [ ws; N nb; N n; N col; N i ]
    done;
    let r0 = !j0 + nbk in
    let mt = n - r0 in
    if mt > 0 then begin
      (* A22 -= V W^H + W V^H, row-major views of the column-major
         buffers. *)
      let a22 = wsb + ((r0 + (r0 * n)) * t) in
      let wv = wsb + ((nn + r0) * t) and vv = wsb + ((r0 + (!j0 * n)) * t) in
      gemm b f ~cb:a22 ~ab:wv ~bb:vv ~m:mt ~n:mt ~k:nbk ~lda:n ~ldb:n ~ldc:n
        ~opa:2 ~opb:0 ~sa:st ~sb:st ~sc:st ~acc:true ~neg:true ~nz:nb;
      gemm b f ~cb:a22 ~ab:vv ~bb:wv ~m:mt ~n:mt ~k:nbk ~lda:n ~ldb:n ~ldc:n
        ~opa:2 ~opb:0 ~sa:st ~sb:st ~sc:st ~acc:true ~neg:true ~nz:nb
    end;
    j0 := !j0 + tnb
  done;
  td K.Sp_td_diag ~groups:(Over (nb * n, 256)) [ ws; N nb; N n ];
  let dstride = tp / r in
  let dsrc = plus ws ((nn + (2 * n * tnb) + (2 * tnb * tnb) + n) * t) in
  let esrc = plus dsrc (n * r) in
  match v with
  | Null ->
      let wt = take b (nb * n * r) in
      launch b rf K.Sp_tri_bisect
        ~groups:(Over (nb * n, 128))
        [ dsrc; esrc; N dstride; wt; N nb; N n; N 0; N n ];
      td K.Sp_td_out
        ~groups:(Over (nb * nn, 256))
        [ ws; wt; N n; Null; N 0; Null; w; Null; N nb; N n ]
  | _ ->
      (* Divide and conquer on T. *)
      let dp = dc_per n ~r in
      let dw = take b (nb * dp) in
      let nl = cdiv n leaf in
      let last = n - ((nl - 1) * leaf) in
      let nfull = nb * (nl - 1) in
      let lf =
        take b
          ((((nfull * leaf * leaf) + (nb * last * last)) * r * 2)
          + (((nfull * leaf) + (nb * last)) * r))
      in
      let lt = plus lf (nfull * leaf * leaf * r) in
      let vf = plus lt (nb * last * last * r) in
      let vt = plus vf (nfull * leaf * leaf * r) in
      let wf = plus vt (nb * last * last * r) in
      let wt = plus wf (nfull * leaf * r) in
      let dc = launch b rf ~per:dp in
      dc K.Sp_dc_leaves ~groups:(Over (nb, 1))
        [ dw; dsrc; esrc; N dstride; lf; lt; N nb; N n ];
      eigh_dev b rf ~a:lf ~w:wf ~v:vf ~nb:nfull ~n:leaf;
      eigh_dev b rf ~a:lt ~w:wt ~v:vt ~nb ~n:last;
      dc K.Sp_dc_scatter
        ~groups:(Over (nb * nn, 256))
        [ dw; wf; vf; wt; vt; N nb; N n ];
      let lev = ref 1 in
      while leaf lsl (!lev - 1) < n do
        let span = leaf lsl !lev in
        let merges = nb * cdiv n span in
        let la = [ dw; N nb; N n; N !lev ] in
        dc
          (by_real rf K.Sp_dc_deflate_f32 K.Sp_dc_deflate_f64)
          ~groups:(Over (merges, 1)) la;
        for phase = 0 to 2 do
          dc
            (by_real rf K.Sp_dc_secular_f32 K.Sp_dc_secular_f64)
            ~groups:(Over (nb * n, 128))
            [ dw; N nb; N n; N !lev; N phase ]
        done;
        let tiles = cdiv (Int.min span n) 64 in
        dc K.Sp_dc_product ~groups:(Tiles (tiles, tiles, merges)) la;
        dc K.Sp_dc_sort ~groups:(Over (merges, 1)) la;
        incr lev
      done;
      (* v = Q Z: Z complex where a is, the panels' blocks last first. *)
      let z = take b (nb * nn * t) and w2 = take b (nb * tnb * n * t) in
      let zb = byte z and w2b = byte w2 in
      td K.Sp_td_zinit
        ~groups:(Over (nb * nn, 256))
        [ dw; N (dp / r); z; N nb; N n ];
      let vco = nn + (n * tnb) in
      let tto = vco + (n * tnb) in
      let go = tto + (tnb * tnb) in
      let j0 = ref ((n - 1) / tnb * tnb) in
      while !j0 >= 0 do
        let nbk = Int.min tnb (n - !j0) and m = n - !j0 - 1 in
        if m > 0 then begin
          td K.Sp_td_vcopy
            ~groups:(Over (nb * m * nbk, 256))
            [ ws; N nb; N n; N !j0; N nbk ];
          let vc = wsb + (vco * t)
          and tm = wsb + (tto * t)
          and g = wsb + (go * t) in
          gemm b f ~cb:g ~ab:vc ~bb:vc ~m:nbk ~n:nbk ~k:m ~lda:nbk ~ldb:nbk
            ~ldc:nbk ~opa:2 ~opb:0 ~sa:st ~sb:st ~sc:st ~acc:false ~neg:false
            ~nz:nb;
          td K.Sp_larft ~groups:(Over (nb, 1))
            [
              plus ws (tto * t); plus ws (go * t);
              plus ws ((go + (tnb * tnb) + !j0) * t); N nb; N nbk;
            ];
          let wv = wsb + (nn * t) and zs = zb + ((!j0 + 1) * t) in
          gemm b f ~cb:wv ~ab:vc ~bb:zs ~m:nbk ~n ~k:m ~lda:nbk ~ldb:n ~ldc:n
            ~opa:2 ~opb:1 ~sa:st ~sb:nn ~sc:st ~acc:false ~neg:false ~nz:nb;
          gemm b f ~cb:w2b ~ab:tm ~bb:wv ~m:nbk ~n ~k:nbk ~lda:tnb ~ldb:n ~ldc:n
            ~opa:0 ~opb:0 ~sa:st ~sb:st ~sc:(tnb * n) ~acc:false ~neg:false
            ~nz:nb;
          gemm b f ~cb:zs ~ab:w2b ~bb:vc ~m:n ~n:m ~k:nbk ~lda:n ~ldb:nbk ~ldc:n
            ~opa:1 ~opb:1 ~sa:(tnb * n) ~sb:st ~sc:nn ~acc:true ~neg:true ~nz:nb
        end;
        j0 := !j0 - tnb
      done;
      let dd = plus dw (3 * nn * r) in
      let fx = plus dw ((((3 * nn) + (10 * n)) * r) + (5 * n * 4)) in
      td K.Sp_td_out
        ~groups:(Over (nb * nn, 256))
        [ ws; dd; N (dp / r); fx; N (dp / 4); z; w; v; N nb; N n ]

(* svd *)

let rec svd_dev b f ~a ~s ~u ~vh ~nb ~m ~n ~uc ~vr =
  if nb > 0 then
    if Int.min m n = 0 then begin
      if u <> Null then
        launch b f K.Sp_eye
          ~groups:(Over (nb * m * uc, 256))
          [ u; N nb; N m; N uc ];
      if vh <> Null then
        launch b f K.Sp_eye
          ~groups:(Over (nb * vr * n, 256))
          [ vh; N nb; N vr; N n ]
    end
    else if m <= K.sp_jacobi && n <= K.sp_jacobi then
      let per = svd_per m n ~t:f.t ~r:f.r in
      let mm = Int.max m n and k = Int.min m n in
      launch b f
        (by f K.Sp_svd_jacobi_f32 K.Sp_svd_jacobi_f64 K.Sp_svd_jacobi_c64
           K.Sp_svd_jacobi_c128)
        ~groups:(Over (nb, per))
        ~shared:(svd_lds m n ~t:f.t ~r:f.r)
        ~per ~span:(svd_span ~mm ~k)
        [ a; s; u; vh; N nb; N m; N n; N uc; N vr ]
    else svd_large b f ~a ~s ~u ~vh ~nb ~m ~n ~uc ~vr

(* Bidiagonalisation, then bisection on the Golub-Kahan matrix for values
   alone, or recorded QR sweeps applied in a wavefront and the reflectors'
   blocks for vectors. *)
and svd_large b f ~a ~s ~u ~vh ~nb ~m ~n ~uc ~vr =
  let rf = real_fmt f in
  let mm = Int.max m n and k = Int.min m n in
  let wide = m < n and vec = u <> Null in
  let gc = if not vec then 0 else if wide then vr else uc in
  let t = f.t and r = f.r in
  let l = bd_layout ~mm ~k ~gc ~t ~r in
  let ws = take b (nb * l.per) in
  let wsb = byte ws in
  let base = [ ws; N nb; N mm; N k; N gc ] in
  let bd = launch b f ~bd:l in
  bd K.Sp_bd_stage ~groups:(Over (nb, 1)) ((a :: base) @ [ N m; N n ]);
  for col = 0 to k - 1 do
    let args = base @ [ N col ] in
    bd
      (by f K.Sp_bd_left_f32 K.Sp_bd_left_f64 K.Sp_bd_left_c64
         K.Sp_bd_left_c128)
      ~groups:(Over (nb, 1)) args;
    if col < k - 1 then begin
      bd K.Sp_bd_wh ~groups:(Over (nb * (k - col - 1), 1)) args;
      bd K.Sp_bd_updl
        ~groups:(Over (nb * (mm - col) * (k - col - 1), 256))
        args;
      bd
        (by f K.Sp_bd_right_f32 K.Sp_bd_right_f64 K.Sp_bd_right_c64
           K.Sp_bd_right_c128)
        ~groups:(Over (nb, 1)) args;
      bd K.Sp_bd_wn ~groups:(Over (nb * (mm - col - 1), 128)) args;
      bd K.Sp_bd_updr
        ~groups:(Over (nb * (mm - col - 1) * (k - col - 1), 256))
        args
    end
  done;
  if not vec then begin
    let dt = take b ((nb * 2 * k * r * 2) + (nb * k * r)) in
    let et = plus dt (nb * 2 * k * r) in
    let lam = plus et (nb * 2 * k * r) in
    bd K.Sp_bd_tgk ~groups:(Over (nb * 2 * k, 256)) (base @ [ dt; et ]);
    launch b rf K.Sp_tri_bisect
      ~groups:(Over (nb * k, 128))
      [ dt; et; N (2 * k); lam; N nb; N (2 * k); N k; N k ];
    bd K.Sp_bd_values ~groups:(Over (nb * k, 256)) (base @ [ lam; s ])
  end
  else begin
    (* Rounds: every one but a matrix's last records more than cap - K
       rotations (a sweep's at most K - 1 more would pass cap), and a
       matrix records fewer than R + 2 K before it converges or fails, R =
       max(l K², K) its limit of rotations, l 6 but in tests: its counted
       rotations, below R + K, and a rotation for each 2 x 2 block, at
       most K / 2. A matrix done earlier skips the rest, and one left
       unfinished is NaN in every result. *)
    bd K.Sp_bd_ginit ~groups:(Over (nb, 1)) base;
    let most = Int.max (f.lim.rotations * k * k) k in
    let rounds = cdiv (most + (2 * k)) (l.cap - k) + 1 in
    let gs = if 2 * k * r <= 48 * 1024 then 2 * k * r else 0 in
    for _ = 1 to rounds do
      bd
        (by f K.Sp_bd_gen_f32 K.Sp_bd_gen_f64 K.Sp_bd_gen_c64
           K.Sp_bd_gen_c128)
        ~groups:(Over (nb, 1)) ~shared:gs base;
      bd K.Sp_bd_apply ~groups:(Over (nb * 2 * k, 128)) base
    done;
    bd K.Sp_bd_xinit
      ~groups:(Over (nb * ((mm * gc) + (k * k)), 256))
      base;
    (* u's side, then v's: X <- Q X, Q's panels last first. *)
    let st = l.per / t and wc = Int.max gc k in
    let x = wsb + l.x and y = wsb + l.y and vc = wsb + l.vc in
    let tm = wsb + l.tm and gg = wsb + l.gg in
    let w1 = wsb + l.w1 and w2 = wsb + l.w2 in
    let bdnb = K.sp_bd_nb in
    for side = 0 to 1 do
      let refl = if side = 1 then k - 1 else k in
      let xm = if side = 1 then y else x in
      let ld = if side = 1 then k else mm
      and cols = if side = 1 then k else gc in
      let j0 = ref ((refl - 1) / bdnb * bdnb) in
      while !j0 >= 0 && refl > 0 do
        let nbk = Int.min bdnb (refl - !j0) in
        let r0 = if side = 1 then !j0 + 1 else !j0 in
        let mr = ld - r0 in
        if mr > 0 && cols > 0 then begin
          let args = base @ [ N !j0; N nbk; N side ] in
          bd K.Sp_bd_vcopy ~groups:(Over (nb * mr * nbk, 256)) args;
          gemm b f ~cb:gg ~ab:vc ~bb:vc ~m:nbk ~n:nbk ~k:mr ~lda:nbk ~ldb:nbk
            ~ldc:nbk ~opa:2 ~opb:0 ~sa:st ~sb:st ~sc:st ~acc:false ~neg:false
            ~nz:nb;
          let tau = if side = 1 then l.taup else l.tauq in
          launch b f K.Sp_larft ~groups:(Over (nb, 1)) ~per:l.per
            [
              plus ws l.tm; plus ws l.gg; plus ws (tau + (!j0 * t)); N nb;
              N nbk;
            ];
          let xs = xm + (r0 * t) in
          gemm b f ~cb:w1 ~ab:vc ~bb:xs ~m:nbk ~n:cols ~k:mr ~lda:nbk ~ldb:ld
            ~ldc:wc ~opa:2 ~opb:1 ~sa:st ~sb:st ~sc:st ~acc:false ~neg:false
            ~nz:nb;
          gemm b f ~cb:w2 ~ab:tm ~bb:w1 ~m:nbk ~n:cols ~k:nbk ~lda:bdnb ~ldb:wc
            ~ldc:wc ~opa:0 ~opb:0 ~sa:st ~sb:st ~sc:st ~acc:false ~neg:false
            ~nz:nb;
          gemm b f ~cb:xs ~ab:w2 ~bb:vc ~m:cols ~n:mr ~k:nbk ~lda:wc ~ldb:nbk
            ~ldc:ld ~opa:1 ~opb:1 ~sa:st ~sb:st ~sc:st ~acc:true ~neg:true
            ~nz:nb
        end;
        j0 := !j0 - bdnb
      done
    done;
    bd K.Sp_bd_rank ~groups:(Over (nb * k, 256)) base;
    let per = k + (m * uc) + (vr * n) in
    bd K.Sp_bd_out
      ~groups:(Over (nb * per, 256))
      (base @ [ s; u; vh; N m; N n; N uc; N vr ])
  end

(* eig *)

(* A matrix's group: 128 work-items, 64 where the reals are doubles,
   nx.cuda's sizes, which no timing on RDNA4 has yet chosen. The group's
   size orders eig's sums, so its bits follow the format alone. *)
let eig_schur f =
  by f K.Sp_eig_schur_f32 K.Sp_eig_schur_f64 K.Sp_eig_schur_c64
    K.Sp_eig_schur_c128

let eig_dev b f ~a ~w ~v ~nb ~n =
  if nb > 0 && n > 0 then begin
    let per = eig_per n ~t:f.t ~r:f.r and schur = eig_schur f in
    let wsp = take b (nb * per) in
    let vec = v <> Null in
    let eig = launch b f ~per in
    eig schur ~groups:(Over (nb, 1))
      ~shared:(K.threads schur * (f.r + 4))
      [ a; w; wsp; N (Bool.to_int vec); N nb; N n ];
    if vec then begin
      let g = Over (nb * n, 128) in
      eig K.Sp_eig_norms ~groups:g [ wsp; N nb; N n ];
      eig
        (by f K.Sp_eig_vectors_f32 K.Sp_eig_vectors_f64
           K.Sp_eig_vectors_c64 K.Sp_eig_vectors_c128)
        ~groups:g [ wsp; N nb; N n ];
      let nn = n * n and st = per / f.t in
      let zp = take b (nb * nn * f.t) in
      let wb = byte wsp in
      gemm b f ~cb:(byte zp)
        ~ab:(wb + (nn * f.t))
        ~bb:(wb + (2 * nn * f.t))
        ~m:n ~n ~k:n ~lda:n ~ldb:n ~ldc:n ~opa:1 ~opb:1 ~sa:st ~sb:st ~sc:nn
        ~acc:false ~neg:false ~nz:nb;
      eig K.Sp_eig_finish ~groups:(Over (nb, 1)) [ zp; v; wsp; N nb; N n ]
    end
  end

(* Calls *)

let batch_of (A.Any x) =
  let l = A.layout x in
  let n = ref 1 in
  for i = 0 to L.rank l - 3 do
    n := !n * L.dim l i
  done;
  !n

let dims (A.Any x) =
  let l = A.layout x in
  let r = L.rank l in
  (L.dim l (r - 2), L.dim l (r - 1))

(* [x] read in place where C-contiguous, else copied into workspace in C
   order. *)
let operand b f (A.Any a as x) =
  let l = A.layout a in
  if L.is_contiguous l then Read (Plan.slot b (A.buffer a), L.offset l * f.t)
  else begin
    let at = Plan.take b (L.numel l * f.t) in
    Contract.copy b ~src:(Contract.array x)
      ~dst:(Contract.work ~at f.dt (L.contiguous (L.shape l)));
    Read (Plan.workspace b, at)
  end

let numel (A.Any y) = L.numel (A.layout y)

(* A destination's address. *)
let out b (A.Any y) =
  let l = A.layout y in
  Write (Plan.slot b (A.buffer y), L.offset l * D.bytes (A.dtype y) 1)

(* A call on [x]: [plan] adds its launches, its operand read as {!operand}
   gives it where [reads], else [Null]: a submission names no buffer that
   none of its launches reads or writes. *)
let run ?(reads = true) x plan =
  let (A.Any a) = x in
  let f = fmt_of ~lim:(Atomic.get limits) (D.Any (A.dtype a)) in
  let b = Plan.make () in
  let src = if reads then operand b f x else Null in
  plan b f src;
  Plan.finish_some b

let eigh ~w ?v x =
  let nb = batch_of x and n, _ = dims x in
  if nb = 0 || n = 0 then None
  else
    run x (fun b f src ->
        let v = match v with None -> Null | Some v -> out b v in
        eigh_dev b f ~a:src ~w:(out b w) ~v ~nb ~n)

let svd ~s ?vectors x =
  let nb = batch_of x and m, n = dims x in
  if nb = 0 then None
  else
    let uc, vr =
      match vectors with
      | None -> (0, 0)
      | Some (u, vh) -> (snd (dims u), fst (dims vh))
    in
    (* Where k = 0, s holds nothing and the factors are identities: a
       call names an empty destination with no slot, so its submission
       binds only what its launches use. *)
    let k = Int.min m n in
    run ~reads:(k > 0) x (fun b f src ->
        let some y = if k > 0 || numel y > 0 then out b y else Null in
        let u, vh =
          match vectors with
          | None -> (Null, Null)
          | Some (u, vh) -> (some u, some vh)
        in
        svd_dev b f ~a:src ~s:(some s) ~u ~vh ~nb ~m ~n ~uc ~vr)

let eig ~w ?v x =
  let nb = batch_of x and n, _ = dims x in
  if nb = 0 || n = 0 then None
  else
    run x (fun b f src ->
        let v = match v with None -> Null | Some v -> out b v in
        eig_dev b f ~a:src ~w:(out b w) ~v ~nb ~n)
