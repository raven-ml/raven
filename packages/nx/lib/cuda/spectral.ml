(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The spectral routines' launches: spectral.cu's steps and the
   contraction's products, one submission a call.

     operand --(copy, unless C-contiguous)--> scratch
       --> eigh: Jacobi (n <= 32), else tridiagonalisation, divide and
           conquer (or bisection for values alone), back-transformation
       --> svd: Jacobi (both extents <= 32), else bidiagonalisation, QR
           sweeps (or bisection), back-transformation
       --> eig: one block a matrix, then Z X by a product

   The steps keep each matrix's workspace as kernels.h lays it out, in
   memory of the call's own. A product's ranges of k follow each matrix's
   extents alone, so no result's bits depend on the batch.

   An iteration that does not converge within its limit makes its matrix
   NaN in every result: 60 Jacobi sweeps (sp_jacobi.cuh), 30 steps of a
   secular equation (sp_stedc.cuh), 30 max(10, n) QR iterations of eig
   (sp_eig.cuh) and 6 k² bidiagonal rotations (sp_bd.cuh). *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels
module C = Contract
module B = Kernels.Nx_sp_bd

(* kernels.h's layouts, from nx_cuda_stubs.c, [t] and [r] the bytes of an
   element and of its real part: Jacobi's matrices a group and their
   shared bytes, and a matrix's workspace. *)
external sp_eigh_per : int -> int = "nx_cuda_sp_eigh_per" [@@noalloc]

external sp_eigh_smem : int -> t:int -> r:int -> int = "nx_cuda_sp_eigh_smem"
[@@noalloc]

external sp_svd_per : int -> int -> t:int -> r:int -> int = "nx_cuda_sp_svd_per"
[@@noalloc]

external sp_svd_smem : int -> int -> t:int -> r:int -> int
  = "nx_cuda_sp_svd_smem"
[@@noalloc]

external sp_td_per : int -> t:int -> r:int -> int = "nx_cuda_sp_td_per"
[@@noalloc]

external sp_dc_per : int -> r:int -> int = "nx_cuda_sp_dc_per" [@@noalloc]

external sp_eig_per : int -> t:int -> r:int -> int = "nx_cuda_sp_eig_per"
[@@noalloc]

external sp_bd_layout : int -> int -> int -> int -> int -> Bytes.t
  = "nx_cuda_sp_bd_layout"

(* A large svd's workspace a matrix, nx_sp_bd_layout's for g of [m] x [k]:
   its field at the offset [f] of B, a size_t but [cap], an int. *)
let sp_bd ~m ~k ~gc ~t ~r =
  let l = sp_bd_layout m k gc t r in
  fun f ->
    if f = B.cap then Int32.to_int (Bytes.get_int32_le l f)
    else Int64.to_int (Bytes.get_int64_le l f)

(* Formats *)

(* A routine's arithmetic: its kernels' suffix, its NX_FMT_ code, the
   bytes of its element and of its real part, and its dtype. *)
type fmt = { name : string; code : int; t : int; r : int; dt : D.any }

let fmt_of (D.Any dt as any) =
  match dt with
  | D.Float32 -> { name = "f32"; code = K.fmt_f32; t = 4; r = 4; dt = any }
  | D.Float64 -> { name = "f64"; code = K.fmt_f64; t = 8; r = 8; dt = any }
  | D.Complex64 -> { name = "c64"; code = K.fmt_c64; t = 8; r = 4; dt = any }
  | D.Complex128 ->
      { name = "c128"; code = K.fmt_c128; t = 16; r = 8; dt = any }
  | _ -> invalid_arg "Spectral: not a routine's dtype"

let real_fmt f =
  if f.r = 4 then fmt_of (D.Any D.Float32) else fmt_of (D.Any D.Float64)

let cdiv a b = (a + b - 1) / b

(* A grid of [blocks] groups, the kernels looping over those past it. *)
let grid_of blocks = Int.max 1 (Int.min blocks (1 lsl 30))
let blocks work per = grid_of (cdiv work per)

(* Calls *)

(* A call: its builder and its scratch, one buffer of the call's own of
   [size] bytes, made at its first piece and carved into pieces on 256
   bytes as the plan takes them. *)
type call = {
  b : C.builder;
  size : int;
  device : Rig.t;
  mutable scratch : (Rig.Buffer.t * int) option;
  ws : Arena.t;
}

let scratch c =
  match c.scratch with
  | Some s -> s
  | None ->
      let buf = C.fresh c.b c.device c.size in
      let s = (buf, C.slot c.b buf) in
      c.scratch <- Some s;
      s

(* An argument of a step: an address read or written, an integer, or
   NULL. *)
type arg = Read of int * int | Write of int * int | N of int | Null

(* [n] bytes of the scratch, as a written address. *)
let take c n =
  let at = Arena.take c.ws n in
  Write (snd (scratch c), at)

let byte = function Read (_, at) | Write (_, at) -> at | N _ | Null -> 0

let plus a n =
  match a with
  | Read (s, at) -> Read (s, at + n)
  | Write (s, at) -> Write (s, at + n)
  | x -> x

(* The steps one kernel takes in every format, which it reads from the
   word K.sp_fmt. *)
let any_format =
  let t = Hashtbl.create 32 in
  Array.iteri
    (fun k (name, i) ->
      match (i : K.instance) with
      | Spectral_any _ | Spectral_real _ -> Hashtbl.replace t name k
      | _ -> ())
    K.kernels;
  t

let launch c f entry ?(gy = 1) ?(gz = 1) ~gx ~threads ?(shared = 0) args =
  let w = Bytes.make (8 * K.sp_args) '\000' and refs = ref [] in
  List.iteri
    (fun i a ->
      let at = 8 * i in
      match a with
      | Read (s, off) -> C.address refs w at s off ~write:false
      | Write (s, off) -> C.address refs w at s off ~write:true
      | N v -> Bytes.set_int64_le w at (Int64.of_int v)
      | Null -> ())
    args;
  let kernel =
    match Hashtbl.find_opt any_format ("sp_" ^ entry) with
    | Some k ->
        Bytes.set_int64_le w (8 * K.sp_fmt) (Int64.of_int f.code);
        k
    | None -> Launch.kernel ("sp_" ^ entry ^ "_" ^ f.name)
  in
  C.launch c.b ~kernel ~gx ~gy ~gz ~threads ~shared w refs

(* [c <- op(a) op(b)], plus [c] where [acc], negated where [neg], over
   [nz] matrices of the scratch at the bytes [cb], [ab] and [bb]:
   row-major, [op] 0 as it lies, 1 transposed, 2 conjugate-transposed. Its
   ranges of k are each matrix's: few tiles over a long k split it. *)
let gemm c f ~cb ~ab ~bb ~m ~n ~k ~lda ~ldb ~ldc ~opa ~opb ~sa ~sb ~sc ~acc ~neg
    ~nz =
  if m > 0 && n > 0 && nz > 0 then begin
    let (D.Any dt) = f.dt in
    let view at ~sb ~rs ~cs ~rows ~cols =
      let l =
        L.v ~offset:(at / f.t)
          ~strides:(Iarray.of_array [| sb; rs; cs |])
          (Iarray.of_array [| nz; rows; cols |])
      in
      A.Any (A.v dt l (fst (scratch c)))
    in
    let op o at ~ld ~s ~rows ~cols =
      if o = 0 then view at ~sb:s ~rs:ld ~cs:1 ~rows ~cols
      else view at ~sb:s ~rs:1 ~cs:ld ~rows ~cols
    in
    let ranges = C.ranges ~rows:m ~cols:n ~k in
    let o =
      {
        C.init_first = false;
        neg;
        conj_a = opa = 2;
        conj_b = opb = 2;
        tri = 0;
        ranges = Some ranges;
      }
    in
    C.product c.b o ~acc:f.dt ~init:acc
      ~y:(view cb ~sb:sc ~rs:ldc ~cs:1 ~rows:m ~cols:n)
      (op opa ab ~ld:lda ~s:sa ~rows:m ~cols:k)
      (op opb bb ~ld:ldb ~s:sb ~rows:k ~cols:n)
  end

(* eigh *)

let tnb = K.sp_tnb
let leaf = K.sp_leaf

let rec eigh_dev c f ~a ~w ~v ~nb ~n =
  if nb > 0 && n > 0 then
    if n <= 32 then
      let per = sp_eigh_per n in
      launch c f "eigh_jacobi"
        ~gx:(grid_of (cdiv nb per))
        ~threads:K.sp_threads
        ~shared:(sp_eigh_smem n ~t:f.t ~r:f.r)
        [ a; w; v; N nb; N n ]
    else eigh_large c f ~a ~w ~v ~nb ~n

(* Tridiagonalisation, divide and conquer or bisection, the reflectors'
   blocks applied. *)
and eigh_large c f ~a ~w ~v ~nb ~n =
  let rf = real_fmt f in
  let t = f.t and r = f.r and nn = n * n in
  let tp = sp_td_per n ~t ~r in
  let ws = take c (nb * tp) in
  let wsb = byte ws in
  launch c f "td_stage" ~gx:(grid_of nb) ~threads:256 [ a; ws; N nb; N n ];
  let st = tp / t in
  let j0 = ref 0 in
  while !j0 < n do
    let nbk = Int.min tnb (n - !j0) in
    for i = 0 to nbk do
      launch c f "td_step" ~gx:(grid_of nb) ~threads:256
        [ ws; N nb; N n; N !j0; N nbk; N i ];
      let col = !j0 + i in
      if i < nbk && col < n - 1 then
        launch c f "td_hemv"
          ~gx:(blocks (nb * cdiv (n - col - 1) 8) 1)
          ~threads:256
          [ ws; N nb; N n; N col; N i ]
    done;
    let r0 = !j0 + nbk in
    let mt = n - r0 in
    if mt > 0 then begin
      (* A22 -= V W^H + W V^H, row-major views of the column-major
         buffers. *)
      let a22 = wsb + ((r0 + (r0 * n)) * t) in
      let wv = wsb + ((nn + r0) * t) and vv = wsb + ((r0 + (!j0 * n)) * t) in
      gemm c f ~cb:a22 ~ab:wv ~bb:vv ~m:mt ~n:mt ~k:nbk ~lda:n ~ldb:n ~ldc:n
        ~opa:2 ~opb:0 ~sa:st ~sb:st ~sc:st ~acc:true ~neg:true ~nz:nb;
      gemm c f ~cb:a22 ~ab:vv ~bb:wv ~m:mt ~n:mt ~k:nbk ~lda:n ~ldb:n ~ldc:n
        ~opa:2 ~opb:0 ~sa:st ~sb:st ~sc:st ~acc:true ~neg:true ~nz:nb
    end;
    j0 := !j0 + tnb
  done;
  launch c f "td_diag" ~gx:(blocks (nb * n) 256) ~threads:256 [ ws; N nb; N n ];
  let dstride = tp / r in
  let dsrc = plus ws ((nn + (2 * n * tnb) + (2 * tnb * tnb) + n) * t) in
  let esrc = plus dsrc (n * r) in
  match v with
  | Null ->
      let wt = take c (nb * n * r) in
      launch c rf "tri_bisect"
        ~gx:(blocks (nb * n) 128)
        ~threads:128
        [ dsrc; esrc; N dstride; wt; N nb; N n; N 0; N n ];
      launch c f "td_out"
        ~gx:(blocks (nb * nn) 256)
        ~threads:256
        [ ws; wt; N n; Null; N 0; Null; w; Null; N nb; N n ]
  | _ ->
      (* Divide and conquer on T. *)
      let dp = sp_dc_per n ~r in
      let dw = take c (nb * dp) in
      let nl = cdiv n leaf in
      let last = n - ((nl - 1) * leaf) in
      let nfull = nb * (nl - 1) in
      let lf =
        take c
          ((((nfull * leaf * leaf) + (nb * last * last)) * r * 2)
          + (((nfull * leaf) + (nb * last)) * r))
      in
      let lt = plus lf (nfull * leaf * leaf * r) in
      let vf = plus lt (nb * last * last * r) in
      let vt = plus vf (nfull * leaf * leaf * r) in
      let wf = plus vt (nb * last * last * r) in
      let wt = plus wf (nfull * leaf * r) in
      launch c rf "dc_leaves" ~gx:(grid_of nb) ~threads:256
        [ dw; dsrc; esrc; N dstride; lf; lt; N nb; N n ];
      eigh_dev c rf ~a:lf ~w:wf ~v:vf ~nb:nfull ~n:leaf;
      eigh_dev c rf ~a:lt ~w:wt ~v:vt ~nb ~n:last;
      launch c rf "dc_scatter"
        ~gx:(blocks (nb * nn) 256)
        ~threads:256
        [ dw; wf; vf; wt; vt; N nb; N n ];
      let lev = ref 1 in
      while leaf lsl (!lev - 1) < n do
        let span = leaf lsl !lev in
        let merges = nb * cdiv n span in
        let la = [ dw; N nb; N n; N !lev ] in
        launch c rf "dc_deflate" ~gx:(grid_of merges) ~threads:256 la;
        for phase = 0 to 2 do
          launch c rf "dc_secular"
            ~gx:(blocks (nb * n) 128)
            ~threads:128
            [ dw; N nb; N n; N !lev; N phase ]
        done;
        let tiles = cdiv (Int.min span n) 64 in
        launch c rf "dc_product" ~gx:tiles ~gy:tiles ~gz:(Int.min merges 65535)
          ~threads:256 la;
        launch c rf "dc_sort" ~gx:(grid_of merges) ~threads:256 la;
        incr lev
      done;
      (* v = Q Z: Z complex where a is, the panels' blocks last first. *)
      let z = take c (nb * nn * t) and w2 = take c (nb * tnb * n * t) in
      let zb = byte z and w2b = byte w2 in
      launch c f "td_zinit"
        ~gx:(blocks (nb * nn) 256)
        ~threads:256
        [ dw; N (dp / r); z; N nb; N n ];
      let vco = nn + (n * tnb) in
      let tto = vco + (n * tnb) in
      let go = tto + (tnb * tnb) in
      let j0 = ref ((n - 1) / tnb * tnb) in
      while !j0 >= 0 do
        let nbk = Int.min tnb (n - !j0) and m = n - !j0 - 1 in
        if m > 0 then begin
          launch c f "td_vcopy"
            ~gx:(blocks (nb * m * nbk) 256)
            ~threads:256
            [ ws; N nb; N n; N !j0; N nbk ];
          let vc = wsb + (vco * t)
          and tm = wsb + (tto * t)
          and g = wsb + (go * t) in
          gemm c f ~cb:g ~ab:vc ~bb:vc ~m:nbk ~n:nbk ~k:m ~lda:nbk ~ldb:nbk
            ~ldc:nbk ~opa:2 ~opb:0 ~sa:st ~sb:st ~sc:st ~acc:false ~neg:false
            ~nz:nb;
          launch c f "td_larft" ~gx:(grid_of nb) ~threads:32
            [ ws; N nb; N n; N !j0; N nbk ];
          let wv = wsb + (nn * t) and zs = zb + ((!j0 + 1) * t) in
          gemm c f ~cb:wv ~ab:vc ~bb:zs ~m:nbk ~n ~k:m ~lda:nbk ~ldb:n ~ldc:n
            ~opa:2 ~opb:1 ~sa:st ~sb:nn ~sc:st ~acc:false ~neg:false ~nz:nb;
          gemm c f ~cb:w2b ~ab:tm ~bb:wv ~m:nbk ~n ~k:nbk ~lda:tnb ~ldb:n ~ldc:n
            ~opa:0 ~opb:0 ~sa:st ~sb:st ~sc:(tnb * n) ~acc:false ~neg:false
            ~nz:nb;
          gemm c f ~cb:zs ~ab:w2b ~bb:vc ~m:n ~n:m ~k:nbk ~lda:n ~ldb:nbk ~ldc:n
            ~opa:1 ~opb:1 ~sa:(tnb * n) ~sb:st ~sc:nn ~acc:true ~neg:true ~nz:nb
        end;
        j0 := !j0 - tnb
      done;
      let dd = plus dw (3 * nn * r) in
      let fx = plus dw ((((3 * nn) + (10 * n)) * r) + (5 * n * 4)) in
      launch c f "td_out"
        ~gx:(blocks (nb * nn) 256)
        ~threads:256
        [ ws; dd; N (dp / r); fx; N (dp / 4); z; w; v; N nb; N n ]

(* svd *)

let rec svd_dev c f ~a ~s ~u ~vh ~nb ~m ~n ~uc ~vr =
  if nb > 0 then
    if Int.min m n = 0 then
      begin if u <> Null then begin
        launch c f "eye"
          ~gx:(blocks (nb * m * uc) 256)
          ~threads:256 [ u; N nb; N m; N uc ];
        launch c f "eye"
          ~gx:(blocks (nb * vr * n) 256)
          ~threads:256 [ vh; N nb; N vr; N n ]
      end
      end
    else if m <= 32 && n <= 32 then
      let per = sp_svd_per m n ~t:f.t ~r:f.r in
      launch c f "svd_jacobi"
        ~gx:(grid_of (cdiv nb per))
        ~threads:K.sp_threads
        ~shared:(sp_svd_smem m n ~t:f.t ~r:f.r)
        [ a; s; u; vh; N nb; N m; N n; N uc; N vr ]
    else svd_large c f ~a ~s ~u ~vh ~nb ~m ~n ~uc ~vr

(* Bidiagonalisation, then bisection on the Golub-Kahan matrix for values
   alone, or recorded QR sweeps applied in a wavefront and the reflectors'
   blocks for vectors. *)
and svd_large c f ~a ~s ~u ~vh ~nb ~m ~n ~uc ~vr =
  let rf = real_fmt f in
  let mm = Int.max m n and k = Int.min m n in
  let wide = m < n and vec = u <> Null in
  let gc = if not vec then 0 else if wide then vr else uc in
  let t = f.t and r = f.r in
  let l = sp_bd ~m:mm ~k ~gc ~t ~r in
  let ws = take c (nb * l B.per) in
  let wsb = byte ws in
  let base = [ ws; N nb; N mm; N k; N gc ] in
  launch c f "bd_stage" ~gx:(grid_of nb) ~threads:256
    ((a :: base) @ [ N m; N n ]);
  for col = 0 to k - 1 do
    let args = base @ [ N col ] in
    launch c f "bd_left" ~gx:(grid_of nb) ~threads:256 args;
    if col < k - 1 then begin
      launch c f "bd_wh" ~gx:(grid_of (nb * (k - col - 1))) ~threads:256 args;
      launch c f "bd_updl"
        ~gx:(blocks (nb * (mm - col) * (k - col - 1)) 256)
        ~threads:256 args;
      launch c f "bd_right" ~gx:(grid_of nb) ~threads:256 args;
      launch c f "bd_wn"
        ~gx:(blocks (nb * (mm - col - 1)) 128)
        ~threads:128 args;
      launch c f "bd_updr"
        ~gx:(blocks (nb * (mm - col - 1) * (k - col - 1)) 256)
        ~threads:256 args
    end
  done;
  if not vec then begin
    let dt = take c ((nb * 2 * k * r * 2) + (nb * k * r)) in
    let et = plus dt (nb * 2 * k * r) in
    let lam = plus et (nb * 2 * k * r) in
    launch c f "bd_tgk"
      ~gx:(blocks (nb * 2 * k) 256)
      ~threads:256
      (base @ [ dt; et ]);
    launch c rf "tri_bisect"
      ~gx:(blocks (nb * k) 128)
      ~threads:128
      [ dt; et; N (2 * k); lam; N nb; N (2 * k); N k; N k ];
    launch c f "bd_values"
      ~gx:(blocks (nb * k) 256)
      ~threads:256
      (base @ [ lam; s ])
  end
  else begin
    (* Rounds: every one but a matrix's last records more than cap - K
       rotations (a sweep's at most K - 1 more would pass cap), and a
       matrix records fewer than 6 K² + 2 K before it converges or fails:
       its counted rotations, below 6 K² + K, and a rotation for each 2 x 2
       block, at most K / 2. A matrix done earlier skips the rest. *)
    launch c f "bd_ginit" ~gx:(grid_of nb) ~threads:256 base;
    let rounds = cdiv ((6 * k * k) + (2 * k)) (l B.cap - k) + 1 in
    let gs = if 2 * k * r <= 48 * 1024 then 2 * k * r else 0 in
    for _ = 1 to rounds do
      launch c f "bd_gen" ~gx:(grid_of nb) ~threads:32 ~shared:gs base;
      launch c f "bd_apply" ~gx:(blocks (nb * 2 * k) 128) ~threads:128 base
    done;
    launch c f "bd_xinit"
      ~gx:(blocks (nb * ((mm * gc) + (k * k))) 256)
      ~threads:256 base;
    (* u's side, then v's: X <- Q X, Q's panels last first. *)
    let st = l B.per / t and wc = Int.max gc k in
    let x = wsb + l B.x and y = wsb + l B.y and vc = wsb + l B.vc in
    let tm = wsb + l B.tm and gg = wsb + l B.gg in
    let w1 = wsb + l B.w1 and w2 = wsb + l B.w2 in
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
          launch c f "bd_vcopy"
            ~gx:(blocks (nb * mr * nbk) 256)
            ~threads:256 args;
          gemm c f ~cb:gg ~ab:vc ~bb:vc ~m:nbk ~n:nbk ~k:mr ~lda:nbk ~ldb:nbk
            ~ldc:nbk ~opa:2 ~opb:0 ~sa:st ~sb:st ~sc:st ~acc:false ~neg:false
            ~nz:nb;
          launch c f "bd_larft" ~gx:(grid_of nb) ~threads:32 args;
          let xs = xm + (r0 * t) in
          gemm c f ~cb:w1 ~ab:vc ~bb:xs ~m:nbk ~n:cols ~k:mr ~lda:nbk ~ldb:ld
            ~ldc:wc ~opa:2 ~opb:1 ~sa:st ~sb:st ~sc:st ~acc:false ~neg:false
            ~nz:nb;
          gemm c f ~cb:w2 ~ab:tm ~bb:w1 ~m:nbk ~n:cols ~k:nbk ~lda:bdnb ~ldb:wc
            ~ldc:wc ~opa:0 ~opb:0 ~sa:st ~sb:st ~sc:st ~acc:false ~neg:false
            ~nz:nb;
          gemm c f ~cb:xs ~ab:w2 ~bb:vc ~m:cols ~n:mr ~k:nbk ~lda:wc ~ldb:nbk
            ~ldc:ld ~opa:1 ~opb:1 ~sa:st ~sb:st ~sc:st ~acc:true ~neg:true
            ~nz:nb
        end;
        j0 := !j0 - bdnb
      done
    done;
    launch c f "bd_rank" ~gx:(blocks (nb * k) 256) ~threads:256 base;
    let per = k + (m * uc) + (vr * n) in
    launch c f "bd_out"
      ~gx:(blocks (nb * per) 256)
      ~threads:256
      (base @ [ s; u; vh; N m; N n; N uc; N vr ])
  end

(* eig *)

(* A matrix's group: past 32, fewer threads where the reals are doubles,
   whose arithmetic is slow enough on sm_89 that the sweeps' barriers
   and serial steps cost more than the wider rows save. *)
let eig_threads f n =
  if n <= 32 then 32 else if f.r = 8 then 64 else if n <= 128 then 128 else 256

let eig_dev c f ~a ~w ~v ~nb ~n =
  if nb > 0 && n > 0 then begin
    let per = sp_eig_per n ~t:f.t ~r:f.r and th = eig_threads f n in
    let wsp = take c (nb * per) in
    let vec = v <> Null in
    launch c f "eig_schur" ~gx:(grid_of nb) ~threads:th
      ~shared:(th * (f.r + 4))
      [ a; w; wsp; N (Bool.to_int vec); N nb; N n ];
    if vec then begin
      let g = grid_of (cdiv (nb * n) 128) in
      launch c f "eig_norms" ~gx:g ~threads:128 [ wsp; N nb; N n ];
      launch c f "eig_vectors" ~gx:g ~threads:128 [ wsp; N nb; N n ];
      let nn = n * n and st = per / f.t in
      let zp = take c (nb * nn * f.t) in
      let wb = byte wsp in
      gemm c f ~cb:(byte zp)
        ~ab:(wb + (nn * f.t))
        ~bb:(wb + (2 * nn * f.t))
        ~m:n ~n ~k:n ~lda:n ~ldb:n ~ldc:n ~opa:1 ~opb:1 ~sa:st ~sb:st ~sc:nn
        ~acc:false ~neg:false ~nz:nb;
      launch c f "eig_finish" ~gx:(grid_of nb) ~threads:th
        [ zp; v; wsp; N nb; N n ]
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

let elt_bytes (A.Any x) = D.bytes (A.dtype x) 1

(* A call's scratch of [bytes]; [plan] adds its launches, its operand
   read from [x] in place where C-contiguous, else from a copy, the
   scratch's first piece. *)
let run x ~bytes plan =
  let (A.Any a) = x in
  let f = fmt_of (D.Any (A.dtype a)) in
  let b = C.builder () in
  let contiguous = L.is_contiguous (A.layout a) in
  let copy =
    if contiguous then 0 else ((L.numel (A.layout a) * f.t) + 255) land lnot 255
  in
  let size = Int.max 1 (bytes + copy) in
  let c =
    { b; size; device = A.device a; scratch = None; ws = Arena.make () }
  in
  let src =
    if contiguous then Read (C.slot b (A.buffer a), L.offset (A.layout a) * f.t)
    else begin
      let at = take c (L.numel (A.layout a) * f.t) in
      let shape = L.shape (A.layout a) in
      let r = Iarray.length shape in
      let strides = Array.make r 1 in
      for i = r - 2 downto 0 do
        strides.(i) <- strides.(i + 1) * Iarray.get shape (i + 1)
      done;
      let dst =
        A.Any
          (A.v (A.dtype a)
             (L.v
                ~offset:(byte at / f.t)
                ~strides:(Iarray.of_array strides) shape)
             (fst (scratch c)))
      in
      C.copy b ~src:x ~dst;
      Read (snd (scratch c), byte at)
    end
  in
  plan c f src;
  Ok [ C.finish ~most:8 b ]

(* A destination's address. *)
let out c (A.Any y) =
  Write (C.slot c.b (A.buffer y), L.offset (A.layout y) * elt_bytes (A.Any y))

let pieces sizes =
  List.fold_left (fun n s -> n + ((s + 255) land lnot 255)) 0 sizes

let eigh ~w ?v x =
  let nb = batch_of x and n, _ = dims x in
  if nb = 0 || n = 0 then Ok []
  else
    let (A.Any a) = x in
    let f = fmt_of (D.Any (A.dtype a)) in
    let t = f.t and r = f.r and nn = n * n in
    let dp = sp_dc_per n ~r and nl = cdiv n leaf in
    let last = n - ((nl - 1) * leaf) and nfull = nb * (nl - 1) in
    let bytes =
      if n <= 32 then 0
      else
        pieces
          [
            nb * sp_td_per n ~t ~r;
            nb * n * r;
            nb * dp;
            (((nfull * leaf * leaf) + (nb * last * last)) * r * 2)
            + (((nfull * leaf) + (nb * last)) * r);
            nb * nn * t;
            nb * tnb * n * t;
          ]
    in
    run x ~bytes (fun c f src ->
        let v = match v with None -> Null | Some v -> out c v in
        eigh_dev c f ~a:src ~w:(out c w) ~v ~nb ~n)

let svd ~s ?vectors x =
  let nb = batch_of x and m, n = dims x in
  if nb = 0 then Ok []
  else
    let (A.Any a) = x in
    let f = fmt_of (D.Any (A.dtype a)) in
    let mm = Int.max m n and k = Int.min m n in
    let uc, vr =
      match vectors with
      | None -> (0, 0)
      | Some (u, vh) -> (snd (dims u), fst (dims vh))
    in
    let gc =
      match vectors with None -> 0 | Some _ -> if m < n then vr else uc
    in
    let bytes =
      if k = 0 || (m <= 32 && n <= 32) then 0
      else
        let l = sp_bd ~m:mm ~k ~gc ~t:f.t ~r:f.r in
        pieces [ nb * l B.per; (nb * 2 * k * f.r * 2) + (nb * k * f.r) ]
    in
    run x ~bytes (fun c f src ->
        let u, vh =
          match vectors with
          | None -> (Null, Null)
          | Some (u, vh) -> (out c u, out c vh)
        in
        svd_dev c f ~a:src ~s:(out c s) ~u ~vh ~nb ~m ~n ~uc ~vr)

let eig ~w ?v x =
  let nb = batch_of x and n, _ = dims x in
  if nb = 0 || n = 0 then Ok []
  else
    let (A.Any a) = x in
    let f = fmt_of (D.Any (A.dtype a)) in
    let bytes = pieces [ nb * sp_eig_per n ~t:f.t ~r:f.r; nb * n * n * f.t ] in
    run x ~bytes (fun c f src ->
        let v = match v with None -> Null | Some v -> out c v in
        eig_dev c f ~a:src ~w:(out c w) ~v ~nb ~n)
