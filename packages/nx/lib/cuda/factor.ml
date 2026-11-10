(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The factorisations' launches: factor.cu's steps and the contraction's
   products, one submission a call.

     operand --(copy, unless read in place)--> destination or scratch
       --> panels, leaves and blocks (factor.cu) --> products (contract.cu,
       one range of k from the output) --> finishing steps

   LU, Cholesky and the solves recurse or block as LAPACK's unblocked
   algorithm's arithmetic allows: every element takes its updates in step
   order, each a fused multiply-add from its own value, so their bits do
   not depend on the blocking. QR blocks its reflectors (xGEQRT, xORGQR)
   and splits a long product's k by each matrix's extents alone. A
   result's bits never depend on the batch.

   A call reads its operand in place where it can, and copies it
   otherwise: LU's and Cholesky's a that is not C-contiguous into their
   destination, as every a of more than 64 columns (32 at complex128 for
   LU); a solve's b likewise, past 64 rows (32 at complex128), and its a
   whose batch axes do not form one run of strides into memory of its
   own; QR's a into memory of its own unless it is C-contiguous and
   factored in one launch. A panel's or a block's tile lies in the
   workspace where shared memory does not hold it or the panel's rows
   span several groups. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels
module C = Contract

(* Formats *)

(* A routine's arithmetic: its kernels' suffix, its NX_FMT_ code, its
   element's bytes and its dtype. *)
type fmt = { name : string; code : int; bytes : int; dt : D.any }

let fmt_of (D.Any dt as any) =
  match dt with
  | D.Float32 -> { name = "f32"; code = K.fmt_f32; bytes = 4; dt = any }
  | D.Float64 -> { name = "f64"; code = K.fmt_f64; bytes = 8; dt = any }
  | D.Complex64 -> { name = "c64"; code = K.fmt_c64; bytes = 8; dt = any }
  | D.Complex128 -> { name = "c128"; code = K.fmt_c128; bytes = 16; dt = any }
  | _ -> invalid_arg "Factor: not a routine's dtype"

(* The most rows of a solve's leaf, and the columns of an LU panel: half
   at 16 bytes an element, so that their tiles fit shared memory. *)
let leaf f = if f.bytes = 16 then K.la_leaf / 2 else K.la_leaf
let panel_width f = if f.bytes = 16 then 32 else 64
let qr_width f = if f.bytes = 16 then 16 else 32

(* Cholesky's diagonal blocks. *)
let chol_nb = 64

(* A group's shared memory, as factor.cu's launches opt into it. *)
let smem_most = 96 * 1024

(* The most blocks a QR panel's rows split into. A block sums its rows'
   partials one block of NX_LA_QBLK rows after another, a chain of loads,
   and combines every block's partials at each step: on 65536 rows 128
   blocks took a third of 16's time, and 256 a quarter more than 128. *)
let qr_blocks = 128

(* A call of at most this many launches keeps its submission, of a key
   among finitely many. A larger one, on larger matrices, makes its own at
   each call, so its words grow with its launches. *)
let keyed_most = 8
let cdiv a b = (a + b - 1) / b
let r16 a = (a + 15) / 16 * 16

(* The split of a recursive routine's [n]: half, rounded to a multiple of
   [q]. *)
let split n q =
  let h = n / 2 / q * q in
  if h > 0 then h else q

(* Views *)

(* Matrices of a buffer: element (z, i, j) at [off + z sb + i rs + j cs],
   in elements of the format. *)
type view = { buf : Rig.Buffer.t; off : int; sb : int; rs : int; cs : int }

let at v i j = { v with off = v.off + (i * v.rs) + (j * v.cs) }
let tr v = { v with rs = v.cs; cs = v.rs }

(* [v] read from its last row and column: an upper triangle as a lower
   one. *)
let reversed v ~n =
  { v with off = v.off + ((n - 1) * (v.rs + v.cs)); rs = -v.rs; cs = -v.cs }

let contiguous_view buf ~off ~rows ~cols =
  { buf; off; sb = rows * cols; rs = cols; cs = 1 }

(* [v]'s [batch] matrices of [rows] x [cols] as an array, for a product. *)
let arr f v ~batch ~rows ~cols =
  let (D.Any dt) = f.dt in
  let l =
    L.v ~offset:v.off
      ~strides:(Iarray.of_array [| v.sb; v.rs; v.cs |])
      (Iarray.of_array [| batch; rows; cols |])
  in
  A.Any (A.v dt l v.buf)

(* A C-contiguous array of [x]'s shape from [v]'s first element. *)
let like (A.Any x) v =
  let shape = L.shape (A.layout x) in
  let r = Iarray.length shape in
  let strides = Array.make r 1 in
  for i = r - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * Iarray.get shape (i + 1)
  done;
  A.Any
    (A.v (A.dtype x)
       (L.v ~offset:v.off ~strides:(Iarray.of_array strides) shape)
       v.buf)

(* [x], square, transposed: its last two axes' strides swapped. *)
let transposed (A.Any x) =
  let l = A.layout x in
  let r = L.rank l in
  let strides = Array.init r (L.stride l) in
  strides.(r - 1) <- L.stride l (r - 2);
  strides.(r - 2) <- L.stride l (r - 1);
  A.Any
    (A.v (A.dtype x)
       (L.v ~offset:(L.offset l) ~strides:(Iarray.of_array strides) (L.shape l))
       (A.buffer x))

(* An array's matrices as a view, where its batch axes form one run of
   strides. *)
let view_of (A.Any x) =
  let l = A.layout x in
  let r = L.rank l in
  let rs = L.stride l (r - 2) and cs = L.stride l (r - 1) in
  let extent = ref 1 and stride = ref 0 and runs = ref 0 in
  for i = 0 to r - 3 do
    let e = L.dim l i in
    if e = 0 then extent := 0
    else if e > 1 then begin
      let s = L.stride l i in
      if !runs = 0 || !stride <> s * e then incr runs;
      stride := s;
      extent := !extent * e
    end
  done;
  if !runs > 1 then None
  else Some { buf = A.buffer x; off = L.offset l; sb = !stride; rs; cs }

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

let is_contiguous (A.Any x) = L.is_contiguous (A.layout x)
let buffer (A.Any x) = A.buffer x
let offset (A.Any x) = L.offset (A.layout x)

(* Calls *)

(* A call under construction: its builder, format and batch. *)
type call = { b : C.builder; f : fmt; batch : int }

(* A buffer of [n] bytes on [x]'s device, of the call's own. *)
let fresh c (A.Any x) n = C.fresh c.b (A.device x) n
let call f ~batch = { b = C.builder (); f; batch }

(* A pointer field: the byte [at] of [slot], or NULL where [slot] is
   [null]. *)
let null = min_int
let set w at v = Bytes.set_int64_le w at (Int64.of_int v)

let ptr refs w field (slot, at) ~write =
  if slot <> null then C.address refs w field slot at ~write

let of_view c v = (C.slot c.b v.buf, v.off * c.f.bytes)
let ws bytes c = (C.ws_slot, C.take c.b bytes)

let of_array (A.Any x as a) =
  (buffer a, L.offset (A.layout x) * D.bytes (A.dtype x) 1)

let launch c name ?(fmt = true) ~gx ~threads ~shared w refs =
  let name = if fmt then name ^ "_" ^ c.f.name else name in
  C.launch c.b ~kernel:(Launch.kernel name) ~gx ~threads ~shared w refs

(* A launch of a kernel of every format, which reads it at the byte [at]
   of [w]. *)
let launch_any c name ~at ~gx ~threads w refs =
  set w at c.f.code;
  launch c name ~fmt:false ~gx ~threads ~shared:0 w refs

(* Words of [value] at the workspace's byte [at], [count] of them. *)
let fill c at ~count ~value =
  if count > 0 then begin
    let module P = K.La_fill_params in
    let w = Bytes.make P.size '\000' and refs = ref [] in
    C.address refs w P.x C.ws_slot at ~write:true;
    set w P.count count;
    set w P.value value;
    launch c "la_fill" ~fmt:false ~gx:(cdiv count 256) ~threads:256 ~shared:0 w
      refs
  end

(* [y <- y - op(a) op(m)] (or [+] without [neg]) from [y]'s own value,
   one range of k unless [ranges]: the unblocked algorithm's chain. *)
let product c ?(neg = true) ?(conj_a = false) ?(tri = 0) ?(ranges = 1)
    ?(init = true) y a m ~rows ~cols ~k =
  if rows > 0 && cols > 0 && c.batch > 0 && (k > 0 || not init) then begin
    let o =
      {
        C.init_first = init;
        neg;
        conj_a;
        conj_b = false;
        tri;
        ranges = Some ranges;
      }
    in
    let f = c.f and batch = c.batch in
    C.product c.b o ~acc:f.dt ~init
      ~y:(arr f y ~batch ~rows ~cols)
      (arr f a ~batch ~rows ~cols:k)
      (arr f m ~batch ~rows:k ~cols)
  end

(* Triangular solves *)

(* [x <- op(l)^-1 x] for the lower triangular view [l] of [n] rows and the
   [n] x [r] view [x]: leaves of [leaf] rows, the rest by products
   (ReLAPACK's recursive xTRSM). *)
let rec trsm c ~l ~conj ~unit ~x ~n ~r =
  if n = 0 || r = 0 || c.batch = 0 then ()
  else if n <= leaf c.f then
    trsm_leaf c ~l ~conj ~unit ~x ~n ~r ~src:(null, 0) ~whole:false
  else
    let n1 = split n (leaf c.f) in
    let n2 = n - n1 in
    trsm c ~l ~conj ~unit ~x ~n:n1 ~r;
    product c ~conj_a:conj (at x n1 0) (at l n1 0) x ~rows:n2 ~cols:r ~k:n1;
    trsm c ~l:(at l n1 n1) ~conj ~unit ~x:(at x n1 0) ~n:n2 ~r

and trsm_leaf c ~l ~conj ~unit ~x ~n ~r ~src ~whole =
  let module P = K.La_trsm_params in
  let w = Bytes.make P.size '\000' and refs = ref [] in
  ptr refs w P.l (of_view c l) ~write:false;
  ptr refs w P.x (of_view c x) ~write:true;
  ptr refs w P.src src ~write:false;
  set w P.sl l.sb;
  set w P.lrs l.rs;
  set w P.lcs l.cs;
  set w P.sx x.sb;
  set w P.xrs x.rs;
  set w P.xcs x.cs;
  set w P.n n;
  set w P.r r;
  set w P.matrices c.batch;
  set w P.conj (Bool.to_int conj);
  set w P.unit (Bool.to_int unit);
  set w P.whole (Bool.to_int whole);
  let threads = K.la_trsm_threads in
  launch c "la_trsm"
    ~gx:(cdiv (c.batch * r) threads)
    ~threads
    ~shared:(n * threads * c.f.bytes)
    w refs

(* A solve's diagonal: [flag] set for a matrix of a zero on [l]'s
   diagonal, then that matrix's [x] NaN. *)
let diag c step ~a ~sd ~x ~flag ~n ~xn =
  let module P = K.La_diag_params in
  let w = Bytes.make P.size '\000' and refs = ref [] in
  ptr refs w P.a (of_view c a) ~write:false;
  ptr refs w P.x (of_view c x) ~write:true;
  ptr refs w P.flag flag ~write:true;
  set w P.sa a.sb;
  set w P.sd sd;
  set w P.n n;
  set w P.sx x.sb;
  set w P.xn xn;
  set w P.matrices c.batch;
  let total = (if step = "la_diag_zero" then n else xn) * c.batch in
  launch_any c step ~at:P.fmt ~gx:(cdiv total 256) ~threads:256 w refs

(* LU *)

(* A pivot candidate's bytes in shared memory (factor.cu's Cand), 64 of
   them. *)
let cand = 64 * 32

let lu_panel c ~a ~m ~n ~c0 ~w:wd ~piv ~k ~perm ?src () =
  let f = c.f in
  let mp = m - c0 in
  let steps = Int.min mp wd in
  if steps > 0 then begin
    let mark = C.mark c.b in
    let module P = K.La_panel_params in
    let ldt = wd + 1 in
    let src_of = C.take c.b (c.batch * mp * 8) in
    let pw = Bytes.make P.size '\000' and refs = ref [] in
    ptr refs pw P.a (of_view c a) ~write:true;
    ptr refs pw P.piv piv ~write:true;
    ptr refs pw P.src_of (C.ws_slot, src_of) ~write:true;
    (match src with
    | None -> ()
    | Some s ->
        ptr refs pw P.src s ~write:false;
        ptr refs pw P.perm perm ~write:true);
    set pw P.sbat (m * n);
    set pw P.lda n;
    set pw P.spiv k;
    set pw P.sperm m;
    set pw P.m m;
    set pw P.c0 c0;
    set pw P.w wd;
    set pw P.ld ldt;
    set pw P.matrices c.batch;
    if mp <= 16 && wd <= 16 then begin
      (* A half-warp a matrix, 8 a block. *)
      let spill = r16 (mp * ldt * f.bytes) + (mp * 8) in
      set pw P.g_ 16;
      set pw P.p_ 1;
      set pw P.per mp;
      set pw P.spill_bytes spill;
      set pw P.in_smem 1;
      launch c "la_lu_panel" ~gx:(cdiv c.batch 8) ~threads:128
        ~shared:(cand + (8 * spill))
        pw refs
    end
    else begin
      (* A block a matrix, or P blocks of 128 rows where the batch leaves
         the GPU idle and the rows are many: then one launch a step, each
         block's tile at [spill]. *)
      let p = if c.batch < C.sms / 2 && mp > 512 then cdiv mp 128 else 1 in
      let per = cdiv mp p in
      let bytes = r16 (per * ldt * f.bytes) + (per * 8) in
      let threads = Int.min 256 (Int.max 64 (cdiv per 32 * 32)) in
      set pw P.g_ threads;
      set pw P.p_ p;
      set pw P.per per;
      set pw P.spill_bytes bytes;
      if p = 1 then begin
        let in_smem = cand + bytes <= smem_most in
        set pw P.in_smem (Bool.to_int in_smem);
        if not in_smem then
          ptr refs pw P.spill (ws (c.batch * bytes) c) ~write:true;
        launch c "la_lu_panel" ~gx:c.batch ~threads
          ~shared:(cand + if in_smem then bytes else 0)
          pw refs
      end
      else begin
        let slot_bytes = r16 (K.la_slot_head + (wd * f.bytes)) in
        set pw P.slot_bytes slot_bytes;
        ptr refs pw P.slots (ws (c.batch * 2 * p * slot_bytes) c) ~write:true;
        ptr refs pw P.spill (ws (c.batch * p * bytes) c) ~write:true;
        for k = 0 to steps do
          let w = Bytes.copy pw in
          set w P.k k;
          launch c "la_lu_panel" ~gx:(c.batch * p) ~threads
            ~shared:(r16 (cand + (wd * f.bytes)))
            w refs
        done
      end
    end;
    (* With [src], the whole matrix: perm stored by the panel. *)
    if Option.is_none src then begin
      let module G = K.La_gather_params in
      let w = Bytes.make G.size '\000' and refs = ref [] in
      ptr refs w G.a (of_view c a) ~write:true;
      ptr refs w G.piv piv ~write:false;
      ptr refs w G.src_of (C.ws_slot, src_of) ~write:false;
      ptr refs w G.perm perm ~write:true;
      set w G.sbat (m * n);
      set w G.lda n;
      set w G.n n;
      set w G.spiv k;
      set w G.sperm m;
      set w G.m m;
      set w G.c0 c0;
      set w G.w wd;
      set w G.steps steps;
      set w G.matrices c.batch;
      let total = (n - wd + 1) * c.batch in
      launch_any c "la_gather" ~at:G.fmt ~gx:(cdiv total 128) ~threads:128 w
        refs
    end;
    C.release c.b mark
  end

let iota c perm ~m =
  let module P = K.La_iota_params in
  let total = c.batch * m in
  if total > 0 then begin
    let w = Bytes.make P.size '\000' and refs = ref [] in
    ptr refs w P.perm perm ~write:true;
    set w P.m m;
    set w P.total total;
    launch c "la_iota" ~fmt:false ~gx:(cdiv total 256) ~threads:256 ~shared:0 w
      refs
  end

let rec lu_rec c ~a ~m ~n ~c0 ~w ~piv ~k ~perm =
  let wp = panel_width c.f in
  if w <= wp then lu_panel c ~a ~m ~n ~c0 ~w ~piv ~k ~perm ()
  else
    let n1 = split w wp in
    let n2 = w - n1 in
    lu_rec c ~a ~m ~n ~c0 ~w:n1 ~piv ~k ~perm;
    trsm c ~l:(at a c0 c0) ~conj:false ~unit:true
      ~x:(at a c0 (c0 + n1))
      ~n:n1 ~r:n2;
    product c
      (at a (c0 + n1) (c0 + n1))
      (at a (c0 + n1) c0)
      (at a c0 (c0 + n1))
      ~rows:(m - c0 - n1)
      ~cols:n2 ~k:n1;
    lu_rec c ~a ~m ~n ~c0:(c0 + n1) ~w:n2 ~piv ~k ~perm

let lu ~lu:dst ~pivots ~perm x =
  let f =
    fmt_of
      (let (A.Any a) = x in
       D.Any (A.dtype a))
  in
  let batch = batch_of x and m, n = dims x in
  let k = Int.min m n in
  if batch = 0 || m = 0 then Ok []
  else begin
    let c = call f ~batch in
    let slot_of a =
      let buf, at = of_array a in
      (C.slot c.b buf, at)
    in
    let piv = slot_of pivots and pm = slot_of perm in
    let a = contiguous_view (buffer dst) ~off:(offset dst) ~rows:m ~cols:n in
    if k > 0 && n <= panel_width f then begin
      (* One launch: the operand read in place where it is C-contiguous,
         LU, pivots and perm stored. *)
      let src =
        if is_contiguous x then slot_of x
        else begin
          C.copy c.b ~src:x ~dst;
          of_view c a
        end
      in
      lu_panel c ~a ~m ~n ~c0:0 ~w:n ~piv ~k ~perm:pm ~src ()
    end
    else begin
      C.copy c.b ~src:x ~dst;
      iota c pm ~m;
      if k > 0 then begin
        lu_rec c ~a ~m ~n ~c0:0 ~w:k ~piv ~k ~perm:pm;
        if n > k then
          trsm c ~l:a ~conj:false ~unit:true ~x:(at a 0 k) ~n:k ~r:(n - k)
      end
    end;
    Ok [ C.finish ~most:keyed_most c.b ]
  end

(* Cholesky *)

let chol_block c ~a ~n ~k ~nb ~fail ?src ~upper () =
  let module P = K.La_chol_params in
  let w = Bytes.make P.size '\000' and refs = ref [] in
  ptr refs w P.a (of_view c a) ~write:true;
  ptr refs w P.fail fail ~write:true;
  (match src with None -> () | Some s -> ptr refs w P.src s ~write:false);
  set w P.sbat (n * n);
  set w P.lda a.rs;
  set w P.cs a.cs;
  set w P.k k;
  set w P.nb nb;
  set w P.matrices c.batch;
  set w P.ld (nb + 1);
  set w P.upper (Bool.to_int upper);
  let tile = nb * (nb + 1) * c.f.bytes in
  if nb <= 16 then begin
    set w P.g_ 16;
    launch c "la_chol" ~gx:(cdiv c.batch 8) ~threads:128 ~shared:(8 * tile) w
      refs
  end
  else begin
    let threads = Int.min 256 (cdiv nb 32 * 32) in
    set w P.g_ threads;
    launch c "la_chol" ~gx:c.batch ~threads ~shared:tile w refs
  end

let cholesky (t : Nx_kernel.triangle) ~dst x =
  let f =
    fmt_of
      (let (A.Any a) = x in
       D.Any (A.dtype a))
  in
  let batch = batch_of x and n, _ = dims x in
  if batch = 0 || n = 0 then Ok []
  else begin
    let c = call f ~batch in
    let upper = t = Upper in
    let a = contiguous_view (buffer dst) ~off:(offset dst) ~rows:n ~cols:n in
    if n <= chol_nb then begin
      (* One launch: the operand read, the whole result stored. *)
      let src =
        if is_contiguous x then
          let buf, at = of_array x in
          (C.slot c.b buf, at)
        else begin
          C.copy c.b ~src:x ~dst;
          of_view c a
        end
      in
      chol_block c ~a ~n ~k:0 ~nb:n ~fail:(null, 0) ~src ~upper ()
    end
    else begin
      (* A complex operand transposed into dst: the trailing updates
         compute A22's transpose, which a complex product writes in place
         only where its columns are neighbours, and through two packs of
         the whole of it otherwise. *)
      let flip = f.code = K.fmt_c64 || f.code = K.fmt_c128 in
      let a = if flip then tr a else a in
      C.copy c.b ~src:(if flip then transposed x else x) ~dst;
      let fail = ws (8 * cdiv batch 2) c in
      fill c (snd fail) ~count:(cdiv batch 2) ~value:0;
      let k = ref 0 in
      while !k < n do
        let kb = Int.min chol_nb (n - !k) in
        let rest = n - !k - kb in
        chol_block c ~a ~n ~k:!k ~nb:kb ~fail ~upper:false ();
        if rest > 0 then begin
          (* L21 = A21 L11^-H: conj(L11) L21^T = A21^T. *)
          trsm c ~l:(at a !k !k) ~conj:true ~unit:false
            ~x:(tr (at a (!k + kb) !k))
            ~n:kb ~r:rest;
          (* A22 -= L21 L21^H, its transpose computed: (i, c) takes
             conj(L_ip) L_cp, the tiles below its diagonal skipped. *)
          let l21 = at a (!k + kb) !k in
          product c ~conj_a:true ~tri:K.contract_upper
            (tr (at a (!k + kb) (!k + kb)))
            l21 (tr l21) ~rows:rest ~cols:rest ~k:kb
        end;
        k := !k + kb
      done;
      let module P = K.La_chol_finish_params in
      let w = Bytes.make P.size '\000' and refs = ref [] in
      ptr refs w P.a (of_view c a) ~write:true;
      ptr refs w P.fail fail ~write:false;
      set w P.n n;
      set w P.matrices batch;
      set w P.upper (Bool.to_int upper);
      set w P.tr (Bool.to_int flip);
      launch_any c "la_chol_finish" ~at:P.fmt
        ~gx:(cdiv (n * n * batch) 256)
        ~threads:256 w refs
    end;
    Ok [ C.finish ~most:keyed_most c.b ]
  end

(* QR *)

let qr_panel c ~f:fv ~m ~n ~c0 ~w:wd ~tau ~k ?v ?t ?qr ~qc ?(ldto = 0) ?(st = 0)
    () =
  let fm = c.f in
  let mark = C.mark c.b in
  let module P = K.La_qr_params in
  let w = Bytes.make P.size '\000' in
  let most = Int.max (wd + 2) 64 in
  let mp = m - c0 in
  let qblk = K.la_qblk in
  let b_first = c0 / qblk in
  let nblocks = ((m - 1) / qblk) - b_first + 1 in
  let hdr =
    r16
      ((most + wd + (wd * wd)
       + (wd * (wd - 1) / 2)
       + if Option.is_some qr then wd * qc else 0)
      * fm.bytes)
  in
  let p =
    if Option.is_none qr && c.batch < C.sms / 2 && nblocks > 1 then
      Int.min qr_blocks nblocks
    else 1
  in
  let bpc = cdiv nblocks p in
  let p = cdiv nblocks bpc in
  let per = bpc * qblk in
  let rows = Int.min per mp in
  let spill = r16 (rows * (wd + 1) * fm.bytes) in
  let sv = mp * wd and st = if st > 0 then st else wd * wd in
  (* With P > 1, the Gram pairs' store after the partials and rows. *)
  let parts_stride =
    (2 * nblocks * most) + (2 * wd) + if p > 1 then wd * (wd - 1) / 2 else 0
  in
  set w P.sbat (m * n);
  set w P.lda n;
  set w P.stau k;
  set w P.sv sv;
  set w P.st st;
  set w P.ldto (if ldto > 0 then ldto else wd);
  set w P.qc qc;
  set w P.n n;
  set w P.m m;
  set w P.c0 c0;
  set w P.w wd;
  set w P.ld (wd + 1);
  set w P.most most;
  set w P.parts_stride parts_stride;
  set w P.p_ p;
  set w P.per per;
  set w P.spill_bytes spill;
  set w P.matrices c.batch;
  let parts = parts_stride * fm.bytes in
  let refs = ref [] in
  ptr refs w P.a (of_view c fv) ~write:true;
  ptr refs w P.tau tau ~write:true;
  Option.iter (fun v -> ptr refs w P.v (of_view c v) ~write:true) v;
  Option.iter (fun t -> ptr refs w P.t (of_view c t) ~write:true) t;
  Option.iter
    (fun (q, r) ->
      ptr refs w P.q q ~write:true;
      ptr refs w P.r r ~write:true)
    qr;
  if mp <= 16 && wd <= 16 && p = 1 then begin
    set w P.g_ 16;
    set w P.parts_in_smem 1;
    set w P.in_smem 1;
    let per_group = hdr + parts + spill in
    let groups = Int.max 1 (Int.min 8 (smem_most / per_group)) in
    launch c "la_qr_panel" ~gx:(cdiv c.batch groups) ~threads:(16 * groups)
      ~shared:(groups * per_group) w refs
  end
  else begin
    let threads = Int.min 256 (Int.max 64 (cdiv rows 32 * 32)) in
    set w P.g_ threads;
    if p = 1 then begin
      let parts_in_smem = hdr + parts <= smem_most / 2 in
      set w P.parts_in_smem (Bool.to_int parts_in_smem);
      let used = hdr + if parts_in_smem then parts else 0 in
      let in_smem = used + spill <= smem_most in
      set w P.in_smem (Bool.to_int in_smem);
      if not parts_in_smem then
        ptr refs w P.parts (ws (c.batch * parts) c) ~write:true;
      if not in_smem then
        ptr refs w P.spill (ws (c.batch * spill) c) ~write:true;
      launch c "la_qr_panel" ~gx:c.batch ~threads
        ~shared:(used + if in_smem then spill else 0)
        w refs
    end
    else begin
      (* P blocks a matrix, one launch between two meetings of theirs: two
         a step, then the Gram pairs' batches (factor.cu's stages). *)
      ptr refs w P.parts (ws (c.batch * parts) c) ~write:true;
      ptr refs w P.spill (ws (c.batch * p * spill) c) ~write:true;
      let steps = Int.min mp wd in
      let batches = Int.max 1 (cdiv (wd * (wd - 1) / 2) most) in
      let stage s k =
        let o = Bytes.copy w in
        set o P.stage s;
        set o P.k k;
        launch c "la_qr_panel" ~gx:(c.batch * p) ~threads ~shared:hdr o refs
      in
      for k = 0 to steps do
        stage K.la_qr_step_a k;
        if k < steps then stage K.la_qr_step_b k
      done;
      if Option.is_some t then
        for i = 1 to batches do
          stage K.la_qr_step_g i
        done
    end
  end;
  C.release c.b mark

let qr_out c ~f:fv ~q ~r ~m ~n ~qc =
  let module P = K.La_qr_out_params in
  let w = Bytes.make P.size '\000' and refs = ref [] in
  Option.iter (fun fv -> ptr refs w P.f (of_view c fv) ~write:false) fv;
  ptr refs w P.q q ~write:true;
  ptr refs w P.r r ~write:true;
  set w P.m m;
  set w P.n n;
  set w P.qc qc;
  set w P.matrices c.batch;
  let total = ((m * qc) + (qc * n)) * c.batch in
  if total > 0 then
    launch_any c "la_qr_out" ~at:P.fmt ~gx:(cdiv total 256) ~threads:256 w refs

let dense_v c ~f:fv ~v ~m ~n ~c0 ~w:wd =
  let module P = K.La_dense_params in
  let w = Bytes.make P.size '\000' and refs = ref [] in
  ptr refs w P.f (of_view c fv) ~write:false;
  ptr refs w P.v (of_view c v) ~write:true;
  set w P.m m;
  set w P.n n;
  set w P.c0 c0;
  set w P.w wd;
  set w P.matrices c.batch;
  launch_any c "la_dense_v" ~at:P.fmt
    ~gx:(cdiv ((m - c0) * wd * c.batch) 256)
    ~threads:256 w refs

(* Byte pieces of one buffer, each on 256 bytes. *)
let pieces sizes =
  let at = ref 0 in
  let offs =
    List.map
      (fun s ->
        let o = !at in
        at := (o + s + 255) land lnot 255;
        o)
      sizes
  in
  (offs, Int.max 1 !at)

let qr ~q ~r x =
  let f =
    fmt_of
      (let (A.Any a) = x in
       D.Any (A.dtype a))
  in
  let batch = batch_of x and m, n = dims x in
  let k = Int.min m n in
  let _, qc = dims q in
  if batch = 0 then Ok []
  else begin
    let c = call f ~batch in
    let slot_of a =
      let buf, at = of_array a in
      (C.slot c.b buf, at)
    in
    let qp = slot_of q and rp = slot_of r in
    let wq = qr_width f in
    if m = 0 || k = 0 then qr_out c ~f:None ~q:qp ~r:rp ~m ~n ~qc
    else begin
      let tau = ws (batch * k * f.bytes) c in
      let hdr =
        r16
          ((Int.max (n + 2) 64 + n + (n * n) + (n * (n - 1) / 2) + (n * qc))
          * f.bytes)
      in
      if n <= wq && m <= 1024 && hdr <= smem_most / 2 then begin
        (* One launch: a read, Q and R stored; the factored panel is not. *)
        let fv =
          if is_contiguous x then
            contiguous_view (buffer x) ~off:(offset x) ~rows:m ~cols:n
          else begin
            let buf = fresh c x (batch * m * n * f.bytes) in
            let fv = contiguous_view buf ~off:0 ~rows:m ~cols:n in
            C.copy c.b ~src:x ~dst:(like x fv);
            fv
          end
        in
        qr_panel c ~f:fv ~m ~n ~c0:0 ~w:n ~tau ~k ~qr:(qp, rp) ~qc ()
      end
      else begin
        (* Blocks of qb columns, each of panels of wq: a panel's
           reflectors applied to the rest of its block, the block's T built
           from its panels' (T12 = -T1 (V1^H V2) T2, xGEQRT), then the
           columns right of the block updated once, by products of k =
           qb. *)
        let qb = 4 * wq in
        let nblk = cdiv k qb and big = Int.max n qc in
        let tb = Int.min qb k in
        let e = f.bytes in
        let offs, total =
          pieces
            [
              batch * m * n * e;
              batch * m * tb * e;
              batch * nblk * tb * tb * e;
              batch * tb * big * e;
              batch * tb * big * e;
            ]
        in
        let buf = fresh c x total in
        let el i = List.nth offs i / e in
        let fv = contiguous_view buf ~off:(el 0) ~rows:m ~cols:n in
        let vbuf = el 1 and tbig = el 2 and w1 = el 3 and w2 = el 4 in
        C.copy c.b ~src:x ~dst:(like x fv);
        (* T's blocks below their diagonal stay zero. *)
        let words = cdiv (batch * nblk * tb * tb * e) 8 in
        let tz = C.slot c.b buf in
        let module Fl = K.La_fill_params in
        let wf = Bytes.make Fl.size '\000' and refs = ref [] in
        C.address refs wf Fl.x tz (tbig * e) ~write:true;
        set wf Fl.count words;
        launch c "la_fill" ~fmt:false ~gx:(cdiv words 256) ~threads:256
          ~shared:0 wf refs;
        (* c <- (I - V op(T) V^H) c, V h x kb, c h x cols: three products. *)
        let apply v tv ~conj_t cv ~h ~kb ~cols =
          if cols > 0 then begin
            let wv1 = { buf; off = w1; sb = kb * cols; rs = cols; cs = 1 } in
            let wv2 = { wv1 with off = w2 } in
            product c ~init:false ~neg:false ~conj_a:true
              ~ranges:(C.ranges ~rows:kb ~cols ~k:h)
              wv1 (tr v) cv ~rows:kb ~cols ~k:h;
            product c ~init:false ~neg:false ~conj_a:conj_t
              ~ranges:(C.ranges ~rows:kb ~cols ~k:kb)
              wv2
              (if conj_t then tr tv else tv)
              wv1 ~rows:kb ~cols ~k:kb;
            product c
              ~ranges:(C.ranges ~rows:h ~cols ~k:kb)
              cv v wv2 ~rows:h ~cols ~k:kb
          end
        in
        let b0 = ref 0 in
        while !b0 < k do
          let bb = Int.min qb (k - !b0) and hb = m - !b0 in
          let tbo = tbig + (!b0 / qb * batch * tb * tb) in
          let tbv = { buf; off = tbo; sb = tb * tb; rs = tb; cs = 1 } in
          let j0 = ref !b0 in
          while !j0 < !b0 + bb do
            let jb = Int.min wq (!b0 + bb - !j0)
            and h = m - !j0
            and jo = !j0 - !b0 in
            let vv = { buf; off = vbuf; sb = h * jb; rs = jb; cs = 1 } in
            qr_panel c ~f:fv ~m ~n ~c0:!j0 ~w:jb ~tau ~k ~v:vv ~t:(at tbv jo jo)
              ~qc ~ldto:tb ~st:(tb * tb) ();
            apply vv (at tbv jo jo) ~conj_t:true
              (at fv !j0 (!j0 + jb))
              ~h ~kb:jb
              ~cols:(!b0 + bb - !j0 - jb);
            j0 := !j0 + wq
          done;
          (* The block's dense V, then T's blocks above its panels'
             diagonals. *)
          let vv = { buf; off = vbuf; sb = hb * bb; rs = bb; cs = 1 } in
          dense_v c ~f:fv ~v:vv ~m ~n ~c0:!b0 ~w:bb;
          let jo = ref wq in
          while !jo < bb do
            let jb = Int.min wq (bb - !jo) in
            let g = { buf; off = w1; sb = !jo * jb; rs = jb; cs = 1 } in
            let xv = { g with off = w2 } in
            product c ~init:false ~neg:false ~conj_a:true
              ~ranges:(C.ranges ~rows:!jo ~cols:jb ~k:hb)
              g (tr vv) (at vv 0 !jo) ~rows:!jo ~cols:jb ~k:hb;
            product c ~init:false ~neg:false xv tbv g ~rows:!jo ~cols:jb ~k:!jo;
            product c (at tbv 0 !jo) xv (at tbv !jo !jo) ~rows:!jo ~cols:jb
              ~k:jb;
            jo := !jo + wq
          done;
          apply vv tbv ~conj_t:true
            (at fv !b0 (!b0 + bb))
            ~h:hb ~kb:bb
            ~cols:(n - !b0 - bb);
          b0 := !b0 + qb
        done;
        qr_out c ~f:(Some fv) ~q:qp ~r:rp ~m ~n ~qc;
        let qv = contiguous_view (buffer q) ~off:(offset q) ~rows:m ~cols:qc in
        let b0 = ref ((k - 1) / qb * qb) in
        while !b0 >= 0 do
          let bb = Int.min qb (k - !b0) and hb = m - !b0 in
          let vv = { buf; off = vbuf; sb = hb * bb; rs = bb; cs = 1 } in
          dense_v c ~f:fv ~v:vv ~m ~n ~c0:!b0 ~w:bb;
          let tbv =
            {
              buf;
              off = tbig + (!b0 / qb * batch * tb * tb);
              sb = tb * tb;
              rs = tb;
              cs = 1;
            }
          in
          apply vv tbv ~conj_t:false (at qv !b0 !b0) ~h:hb ~kb:bb
            ~cols:(qc - !b0);
          b0 := !b0 - qb
        done
      end
    end;
    Ok [ C.finish ~most:keyed_most c.b ]
  end

(* Triangular solves *)

let solve_triangular (t : Nx_kernel.triangle) ~adjoint ~unit_diagonal ~dst a bm
    =
  let f =
    fmt_of
      (let (A.Any x) = a in
       D.Any (A.dtype x))
  in
  let batch = batch_of bm and n, r = dims bm in
  if batch = 0 || n = 0 || r = 0 then Ok []
  else begin
    let c = call f ~batch in
    (* a in place where its batch axes form one run, else a copy. *)
    let av =
      match view_of a with
      | Some v -> v
      | None ->
          let buf = fresh c a (batch * n * n * f.bytes) in
          let v = contiguous_view buf ~off:0 ~rows:n ~cols:n in
          C.copy c.b ~src:a ~dst:(like a v);
          v
    in
    let op_upper = t = Upper <> adjoint in
    let l = if adjoint then tr av else av in
    let x = contiguous_view (buffer dst) ~off:(offset dst) ~rows:n ~cols:r in
    (* An upper op(a) is a lower one on reversed indices: it substitutes
       from its last row. *)
    let l, xv =
      if op_upper then
        (reversed l ~n, { x with off = x.off + ((n - 1) * r); rs = -r })
      else (l, x)
    in
    if n <= leaf f then begin
      (* One launch: b read in place where it is C-contiguous, x stored,
         the diagonal checked. *)
      let src =
        if is_contiguous bm then
          let buf, at = of_array bm in
          (C.slot c.b buf, at + ((xv.off - x.off) * f.bytes))
        else begin
          C.copy c.b ~src:bm ~dst;
          (null, 0)
        end
      in
      trsm_leaf c ~l ~conj:adjoint ~unit:unit_diagonal ~x:xv ~n ~r ~src
        ~whole:true
    end
    else begin
      C.copy c.b ~src:bm ~dst;
      let flag = ws (8 * cdiv batch 2) c in
      if not unit_diagonal then begin
        fill c (snd flag) ~count:(cdiv batch 2) ~value:0;
        diag c "la_diag_zero" ~a:av ~sd:(av.rs + av.cs) ~x ~flag ~n ~xn:(n * r)
      end;
      trsm c ~l ~conj:adjoint ~unit:unit_diagonal ~x:xv ~n ~r;
      if not unit_diagonal then
        diag c "la_nan_where" ~a:av ~sd:(av.rs + av.cs) ~x ~flag ~n ~xn:(n * r)
    end;
    Ok [ C.finish ~most:keyed_most c.b ]
  end
