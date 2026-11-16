(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The factorisations' launches: factor.hip's steps and the contraction's
   products, one call a routine.

   operand --(copy, unless read in place)--> destination or workspace -->
   panels, leaves and blocks (factor.hip) --> products (Contract, one range of k
   from the output) --> finishing steps

   LU, Cholesky and the solves recurse or block as LAPACK's unblocked
   algorithm's arithmetic allows: every element takes its updates in step order,
   each a fused multiply-add from its own value, so their bits do not depend on
   the blocking. QR blocks its reflectors (xGEQRT, xORGQR) and splits a long
   product's k by each matrix's extents alone. A result's bits never depend on
   the batch.

   A call reads its operand in place where it can, and copies it otherwise: LU's
   and Cholesky's a that is not C-contiguous into their destination, as every a
   of more than one panel or block; a solve's b likewise, past one leaf, and its
   a whose batch axes do not form one run of strides into workspace; QR's a into
   workspace unless it is C-contiguous and factored in one launch. A panel's or
   a block's tile lies in the workspace where LDS does not hold it or the
   panel's rows span several groups. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels
module C = Contract

(* Formats *)

(* A routine's arithmetic: its NX_LA_ code, its element's bytes and its
   dtype. *)
type fmt = { code : int; bytes : int; dt : D.any }

let fmt_of (A.Any x) =
  let dt = D.Any (A.dtype x) in
  match A.dtype x with
  | D.Float32 -> { code = K.la_f32; bytes = 4; dt }
  | D.Float64 -> { code = K.la_f64; bytes = 8; dt }
  | D.Complex64 -> { code = K.la_c64; bytes = 8; dt }
  | D.Complex128 -> { code = K.la_c128; bytes = 16; dt }
  | _ -> invalid_arg "Factor: not a routine's dtype"

let complex f = f.code = K.la_c64 || f.code = K.la_c128

let pick f (f32, f64, c64, c128) =
  if f.code = K.la_f32 then f32
  else if f.code = K.la_f64 then f64
  else if f.code = K.la_c64 then c64
  else c128

(* The most rows of a solve's leaf, and the columns of an LU panel and a
   Cholesky block: half at 16 bytes an element, so that their tiles fit LDS. *)
let leaf f = if f.bytes = 16 then K.la_leaf / 2 else K.la_leaf
let panel_width = leaf
let chol_nb = leaf
let qr_width f = if f.bytes = 16 then 16 else 32

(* The most bytes of the one launch of a QR's header, nx_amd.mli's 32 KiB. *)
let qr_header_most = 32 * 1024

(* The compute units of the largest RDNA4 GPU: a batch of fewer than half leaves
   it idle with a group a matrix, of at least twice fills it with a wave a
   matrix. Neither choice changes a result's bits. *)
let units = 64

(* The most groups a QR panel's rows split into, each summing its rows' partials
   one block of NX_LA_QBLK rows after another. *)
let qr_groups = 128

let cdiv a b = (a + b - 1) / b
let r16 a = (a + 15) / 16 * 16

(* The split of a recursive routine's [n]: half, rounded to a multiple of
   [q]. *)
let split n q =
  let h = n / 2 / q * q in
  if h > 0 then h else q

(* Views *)

(* Where a routine's matrices lie: an array's buffer, or the call's
   workspace. *)
type mem = Buf of Rig.Buffer.t | Work

(* Matrices of [mem]: element (z, i, j) at [off + z sb + i rs + j cs], in
   elements of the format from the buffer's or the workspace's first byte. *)
type view = { mem : mem; off : int; sb : int; rs : int; cs : int }

let at v i j = { v with off = v.off + (i * v.rs) + (j * v.cs) }
let tr v = { v with rs = v.cs; cs = v.rs }

(* [v] read from its last row and column: an upper triangle as a lower one. *)
let reversed v ~n =
  { v with off = v.off + ((n - 1) * (v.rs + v.cs)); rs = -v.rs; cs = -v.cs }

let contiguous mem ~off ~rows ~cols =
  { mem; off; sb = rows * cols; rs = cols; cs = 1 }

let layout (A.Any x) = A.layout x
let buffer (A.Any x) = A.buffer x
let offset x = L.offset (layout x)
let is_contiguous x = L.is_contiguous (layout x)

let batch_of x =
  let l = layout x in
  let n = ref 1 in
  for i = 0 to L.rank l - 3 do
    n := !n * L.dim l i
  done;
  !n

let dims x =
  let l = layout x in
  let r = L.rank l in
  (L.dim l (r - 2), L.dim l (r - 1))

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

(* Calls *)

(* A call under construction: its builder, format and batch. *)
type call = { b : Plan.builder; f : fmt; batch : int }

let call x ~batch = { b = Plan.make (); f = fmt_of x; batch }

(* A place: a slot and a byte in it. *)
let slot_of c = function Buf x -> Plan.slot c.b x | Work -> Plan.workspace c.b
let place_of c v = (slot_of c v.mem, v.off * c.f.bytes)
let ws c bytes = (Plan.workspace c.b, Plan.take c.b bytes)

(* The [off] of a [Work] view of a piece of the workspace of [bytes]. *)
let piece c bytes = Plan.take c.b bytes / c.f.bytes
let elt_bytes (A.Any a) = D.bytes (A.dtype a) 1
let of_array c x = (Plan.slot c.b (buffer x), offset x * elt_bytes x)
let of_dst x = contiguous (Buf (buffer x)) ~off:(offset x)

(* A launch's words and refs. *)
type words = { w : Bytes.t; mutable refs : Plan.ref list }

let words k = { w = Plan.words k; refs = [] }
let set l i v = Plan.set l.w i v

(* Word [i] of [l] holds the address of [place]. *)
let address l i (slot, at) ~write =
  set l i at;
  l.refs <- { Plan.word = i; slot; write } :: l.refs

let copy_words l = { w = Bytes.copy l.w; refs = l.refs }

(* A launch of [k] over [count] units, [per] a group, its groups' count in word
   [groups]. *)
let launch c k l ~groups ~count ~per ?(shared = 0) () =
  let g = Plan.grid ~per k count in
  set l groups g;
  Plan.launch c.b k ~groups:(g, 1, 1) ~shared l.w
    (Array.of_list (List.rev l.refs))

(* Zeroes the [bytes], a multiple of 4, of the workspace from byte [at]. *)
let zero c at ~bytes =
  if bytes > 0 then Plan.zero c.b ~slot:(Plan.workspace c.b) ~at bytes

(* Products and copies *)

(* [v]'s [batch] matrices of [rows] x [cols], for the contraction. *)
let matrix c v ~rows ~cols =
  let l =
    L.v ~offset:v.off
      ~strides:(Iarray.of_array [| v.sb; v.rs; v.cs |])
      (Iarray.of_array [| c.batch; rows; cols |])
  in
  match v.mem with
  | Work -> C.work ~at:0 c.f.dt l
  | Buf x ->
      let (D.Any dt) = c.f.dt in
      C.array (A.Any (A.v dt l x))

(* [y <- y - op(a) m] (or [+] without [neg]) from [y]'s own value, or from +0
   without [init], in [ranges] ranges of k: one is the unblocked algorithm's
   chain. Under [tri], only the tiles that hold an element of that triangle of
   [y]. The product's copies return to the workspace once it is planned. *)
let product c ?(neg = true) ?(conj_a = false) ?tri ?(ranges = 1) ?(init = true)
    y a m ~rows ~cols ~k =
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
    let mark = Plan.mark c.b in
    C.product c.b o ~acc:c.f.dt ~init ~y:(matrix c y ~rows ~cols)
      (matrix c a ~rows ~cols:k) (matrix c m ~rows:k ~cols);
    Plan.release c.b mark
  end

(* [x]'s elements, in C order, into the consecutive elements from [v]'s
   first. *)
let copy c (A.Any x) v =
  let shape = L.shape (A.layout x) in
  let l = L.contiguous shape in
  let l = L.v ~offset:v.off ~strides:(L.strides l) shape in
  let dst =
    match v.mem with
    | Work -> C.work ~at:0 c.f.dt l
    | Buf b -> C.array (A.Any (A.v (A.dtype x) l b))
  in
  C.copy c.b ~src:(C.array (A.Any x)) ~dst

(* Triangular solves *)

(* [x <- op(l)^-1 x] for the lower triangular view [l] of [n] rows and the [n] x
   [r] view [x]: leaves of [leaf] rows, the rest by products (ReLAPACK's
   recursive xTRSM). *)
let rec trsm c ~l ~conj ~unit ~x ~n ~r =
  if n = 0 || r = 0 || c.batch = 0 then ()
  else if n <= leaf c.f then
    trsm_leaf c ~l ~conj ~unit ~x ~n ~r ~src:None ~whole:false
  else
    let n1 = split n (leaf c.f) in
    let n2 = n - n1 in
    trsm c ~l ~conj ~unit ~x ~n:n1 ~r;
    product c ~conj_a:conj (at x n1 0) (at l n1 0) x ~rows:n2 ~cols:r ~k:n1;
    trsm c ~l:(at l n1 n1) ~conj ~unit ~x:(at x n1 0) ~n:n2 ~r

and trsm_leaf c ~l ~conj ~unit ~x ~n ~r ~src ~whole =
  let k = pick c.f K.(La_trsm_f32, La_trsm_f64, La_trsm_c64, La_trsm_c128) in
  let w = words k in
  address w K.la_trsm_l (place_of c l) ~write:false;
  address w K.la_trsm_x (place_of c x) ~write:true;
  Option.iter (fun s -> address w K.la_trsm_src s ~write:false) src;
  set w K.la_trsm_sl l.sb;
  set w K.la_trsm_lrs l.rs;
  set w K.la_trsm_lcs l.cs;
  set w K.la_trsm_sx x.sb;
  set w K.la_trsm_xrs x.rs;
  set w K.la_trsm_xcs x.cs;
  set w K.la_trsm_n n;
  set w K.la_trsm_r r;
  set w K.la_trsm_matrices c.batch;
  set w K.la_trsm_conj (Bool.to_int conj);
  set w K.la_trsm_unit (Bool.to_int unit);
  set w K.la_trsm_whole (Bool.to_int whole);
  let threads = K.la_trsm_threads in
  launch c k w ~groups:K.la_trsm_groups ~count:(c.batch * r) ~per:threads
    ~shared:(n * threads * c.f.bytes)
    ()

(* A solve's diagonal: [flag] set for a matrix of a zero on [a]'s diagonal, then
   that matrix's [x] NaN. *)
let diag c k ~a ~sd ~x ~flag ~n ~xn =
  let l = words k in
  address l K.la_diag_a (place_of c a) ~write:false;
  address l K.la_diag_x (place_of c x) ~write:true;
  address l K.la_diag_flag flag ~write:true;
  set l K.la_diag_sa a.sb;
  set l K.la_diag_sd sd;
  set l K.la_diag_n n;
  set l K.la_diag_sx x.sb;
  set l K.la_diag_xn xn;
  set l K.la_diag_matrices c.batch;
  set l K.la_diag_fmt c.f.code;
  let count = (if k = K.La_diag_zero then n else xn) * c.batch in
  launch c k l ~groups:K.la_diag_groups ~count ~per:K.la_threads ()

(* LU *)

(* A panel's pivot candidates in LDS: a slot of each of the group's waves, a
   slot's head each, for each of two step parities; the second parity's byte,
   and their bytes. *)
let cands_parity = K.la_threads / 32 * K.la_slot_head
let cand = 2 * cands_parity

(* A matrix's G work-items and the matrices a group holds, for a tile of [bytes]
   a matrix of [rows] rows: 16 lanes where the panel is at most 16 x 16, a wave
   where the batch fills the GPU and the rows are few, else the whole group; as
   many matrices as [kernel]'s LDS holds past [head]. *)
let lanes c kernel ~rows ~small ~head ~bytes =
  let fit g =
    Int.min (K.la_threads / g)
      ((Plan.shared_most kernel - head) / Int.max 1 bytes)
  in
  if small && fit 16 >= 1 then (16, fit 16)
  else if c.batch >= 2 * units && rows <= 64 && fit 32 >= 2 then (32, fit 32)
  else (K.la_threads, 1)

let lu_panel c ~a ~m ~n ~c0 ~w:wd ~piv ~k ~perm ~src =
  let f = c.f in
  let mp = m - c0 in
  let steps = Int.min mp wd in
  if steps > 0 then begin
    let mark = Plan.mark c.b in
    let kernel =
      pick f
        K.(La_lu_panel_f32, La_lu_panel_f64, La_lu_panel_c64, La_lu_panel_c128)
    in
    let ldt = wd + 1 in
    let src_of = ws c (c.batch * mp * 8) in
    let pw = words kernel in
    address pw K.la_panel_a (place_of c a) ~write:true;
    address pw K.la_panel_piv piv ~write:true;
    address pw K.la_panel_src_of src_of ~write:true;
    (match src with
    | None -> ()
    | Some s ->
        address pw K.la_panel_src s ~write:false;
        address pw K.la_panel_perm perm ~write:true);
    set pw K.la_panel_sbat (m * n);
    set pw K.la_panel_lda n;
    set pw K.la_panel_spiv k;
    set pw K.la_panel_sperm m;
    set pw K.la_panel_m m;
    set pw K.la_panel_c0 c0;
    set pw K.la_panel_w wd;
    set pw K.la_panel_ld ldt;
    set pw K.la_panel_matrices c.batch;
    (* P groups of 128 rows a matrix where the batch leaves the GPU idle and the
       rows are many: then one launch a step, each group's tile in the
       workspace. *)
    let p = if c.batch < units / 2 && mp > 512 then cdiv mp 128 else 1 in
    let per = cdiv mp p in
    (* A matrix's tile, then its rows' positions. *)
    let tile = r16 (per * ldt * f.bytes) in
    let bytes = tile + (per * 8) in
    (* The candidates, then with P > 1 the pivot row, then the tiles. *)
    let head = r16 (cand + if p > 1 then wd * f.bytes else 0) in
    set pw K.la_panel_per per;
    set pw K.la_panel_p p;
    set pw K.la_panel_spill_bytes bytes;
    set pw K.la_panel_pos tile;
    set pw K.la_panel_row cand;
    set pw K.la_panel_cands_parity cands_parity;
    set pw K.la_panel_head head;
    if p = 1 then begin
      let g, mats =
        lanes c kernel ~rows:mp ~small:(mp <= 16 && wd <= 16) ~head ~bytes
      in
      let in_lds = head + (mats * bytes) <= Plan.shared_most kernel in
      set pw K.la_panel_g g;
      set pw K.la_panel_mats mats;
      set pw K.la_panel_in_lds (Bool.to_int in_lds);
      if not in_lds then
        address pw K.la_panel_spill (ws c (c.batch * bytes)) ~write:true;
      launch c kernel pw ~groups:K.la_panel_groups ~count:c.batch ~per:mats
        ~shared:(head + if in_lds then mats * bytes else 0)
        ()
    end
    else begin
      let slot_bytes = r16 (K.la_slot_head + (wd * f.bytes)) in
      set pw K.la_panel_g K.la_threads;
      set pw K.la_panel_mats 1;
      set pw K.la_panel_slot_bytes slot_bytes;
      (* Each group's slot of each step parity. *)
      set pw K.la_panel_slots_parity (p * slot_bytes);
      set pw K.la_panel_slots_stride (2 * p * slot_bytes);
      address pw K.la_panel_slots
        (ws c (c.batch * 2 * p * slot_bytes))
        ~write:true;
      address pw K.la_panel_spill (ws c (c.batch * p * bytes)) ~write:true;
      for k = 0 to steps do
        let w = copy_words pw in
        set w K.la_panel_k k;
        launch c kernel w ~groups:K.la_panel_groups ~count:(c.batch * p) ~per:1
          ~shared:head ()
      done
    end;
    (* With [src], the whole matrix: perm stored by the panel. *)
    if Option.is_none src then begin
      let w = words K.La_gather in
      address w K.la_gather_a (place_of c a) ~write:true;
      address w K.la_gather_piv piv ~write:false;
      address w K.la_gather_src_of src_of ~write:false;
      address w K.la_gather_perm perm ~write:true;
      set w K.la_gather_sbat (m * n);
      set w K.la_gather_lda n;
      set w K.la_gather_n n;
      set w K.la_gather_spiv k;
      set w K.la_gather_sperm m;
      set w K.la_gather_m m;
      set w K.la_gather_c0 c0;
      set w K.la_gather_w wd;
      set w K.la_gather_steps steps;
      set w K.la_gather_matrices c.batch;
      set w K.la_gather_fmt f.code;
      let blocks = Int.max 1 (cdiv ((n - wd) * f.bytes / 4) 64) in
      launch c K.La_gather w ~groups:K.la_gather_groups
        ~count:(c.batch * blocks) ~per:1 ()
    end;
    Plan.release c.b mark
  end

let iota c perm ~m =
  let total = c.batch * m in
  if total > 0 then begin
    let l = words K.La_iota in
    address l K.la_iota_perm perm ~write:true;
    set l K.la_iota_m m;
    set l K.la_iota_total total;
    launch c K.La_iota l ~groups:K.la_iota_groups ~count:total ~per:K.la_threads
      ()
  end

let rec lu_rec c ~a ~m ~n ~c0 ~w ~piv ~k ~perm =
  let wp = panel_width c.f in
  if w <= wp then lu_panel c ~a ~m ~n ~c0 ~w ~piv ~k ~perm ~src:None
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
  let batch = batch_of x and m, n = dims x in
  let k = Int.min m n in
  if batch = 0 || m = 0 then None
  else begin
    let c = call x ~batch in
    let piv = of_array c pivots and pm = of_array c perm in
    (* No column: perm alone, the identity. *)
    if k = 0 then iota c pm ~m
    else begin
      let a = of_dst dst ~rows:m ~cols:n in
      if n <= panel_width c.f then begin
        (* One panel, of one launch or, where P groups hold its rows, one a
           step: the operand read in place where it is C-contiguous, LU, pivots
           and perm stored. *)
        let src =
          if is_contiguous x then of_array c x
          else begin
            copy c x a;
            place_of c a
          end
        in
        lu_panel c ~a ~m ~n ~c0:0 ~w:n ~piv ~k ~perm:pm ~src:(Some src)
      end
      else begin
        copy c x a;
        iota c pm ~m;
        lu_rec c ~a ~m ~n ~c0:0 ~w:k ~piv ~k ~perm:pm;
        if n > k then
          trsm c ~l:a ~conj:false ~unit:true ~x:(at a 0 k) ~n:k ~r:(n - k)
      end
    end;
    Plan.finish_some c.b
  end

(* Cholesky *)

let chol_block c ~a ~n ~k ~nb ~fail ~src ~upper =
  let kernel =
    pick c.f K.(La_chol_f32, La_chol_f64, La_chol_c64, La_chol_c128)
  in
  let w = words kernel in
  address w K.la_chol_a (place_of c a) ~write:true;
  Option.iter (fun f -> address w K.la_chol_fail f ~write:true) fail;
  Option.iter (fun s -> address w K.la_chol_src s ~write:false) src;
  set w K.la_chol_sbat (n * n);
  set w K.la_chol_lda a.rs;
  set w K.la_chol_cs a.cs;
  set w K.la_chol_k k;
  set w K.la_chol_nb nb;
  set w K.la_chol_matrices c.batch;
  set w K.la_chol_ld (nb + 1);
  set w K.la_chol_upper (Bool.to_int upper);
  let tile = nb * (nb + 1) * c.f.bytes in
  let g, mats = lanes c kernel ~rows:nb ~small:(nb <= 16) ~head:0 ~bytes:tile in
  set w K.la_chol_stride tile;
  set w K.la_chol_g g;
  set w K.la_chol_mats mats;
  launch c kernel w ~groups:K.la_chol_groups ~count:c.batch ~per:mats
    ~shared:(mats * tile) ()

let cholesky (t : Nx_kernel.triangle) ~dst x =
  let batch = batch_of x and n, _ = dims x in
  if batch = 0 || n = 0 then None
  else begin
    let c = call x ~batch in
    let upper = t = Upper in
    let a = of_dst dst ~rows:n ~cols:n in
    if n <= chol_nb c.f then begin
      (* One launch: the operand read, the whole result stored. *)
      let src =
        if is_contiguous x then of_array c x
        else begin
          copy c x a;
          place_of c a
        end
      in
      chol_block c ~a ~n ~k:0 ~nb:n ~fail:None ~src:(Some src) ~upper
    end
    else begin
      (* A complex operand transposed into dst: the trailing updates compute
         A22's transpose, which a complex product writes in place only where its
         columns are neighbours, and through two copies of the whole of it
         otherwise. *)
      let flip = complex c.f in
      let a = if flip then tr a else a in
      copy c (if flip then transposed x else x) (of_dst dst ~rows:n ~cols:n);
      let fail = ws c (4 * batch) in
      zero c (snd fail) ~bytes:(4 * batch);
      let nb = chol_nb c.f in
      let k = ref 0 in
      while !k < n do
        let kb = Int.min nb (n - !k) in
        let rest = n - !k - kb in
        chol_block c ~a ~n ~k:!k ~nb:kb ~fail:(Some fail) ~src:None ~upper:false;
        if rest > 0 then begin
          (* L21 = A21 L11^-H: conj(L11) L21^T = A21^T. *)
          trsm c ~l:(at a !k !k) ~conj:true ~unit:false
            ~x:(tr (at a (!k + kb) !k))
            ~n:kb ~r:rest;
          (* A22 -= L21 L21^H, through its transpose: (i, c) of A22, c <= i,
             takes conj(L_cp) L_ip, the order the unblocked algorithm's updates
             take their operands in. *)
          let l21 = at a (!k + kb) !k in
          product c ~conj_a:true ~tri:`Upper
            (tr (at a (!k + kb) (!k + kb)))
            l21 (tr l21) ~rows:rest ~cols:rest ~k:kb
        end;
        k := !k + kb
      done;
      let w = words K.La_chol_finish in
      address w K.la_finish_a (place_of c a) ~write:true;
      address w K.la_finish_fail fail ~write:false;
      set w K.la_finish_n n;
      set w K.la_finish_matrices batch;
      set w K.la_finish_upper (Bool.to_int upper);
      set w K.la_finish_tr (Bool.to_int flip);
      set w K.la_finish_fmt c.f.code;
      launch c K.La_chol_finish w ~groups:K.la_finish_groups
        ~count:(n * n * batch)
        ~per:K.la_threads ()
    end;
    Plan.finish_some c.b
  end

(* QR *)

(* A QR panel's LDS a matrix from byte 0: its sums' [most] values, a step's w_c,
   T, the Gram pairs and, where Q is stored, M ([qm] elements); each region's
   byte and the header's bytes. *)
type header = { tc : int; ts : int; gram : int; mx : int; size : int }

let qr_most w = Int.max (w + 2) 64

let qr_head ~bytes ~w ~qm =
  let tc = qr_most w * bytes in
  let ts = tc + (w * bytes) in
  let gram = ts + (w * w * bytes) in
  let mx = gram + (w * (w - 1) / 2 * bytes) in
  { tc; ts; gram; mx; size = r16 (mx + (qm * bytes)) }

let qr_panel c ~f:fv ~m ~n ~c0 ~w:wd ~tau ~k ?v ?t ?qr ~qc ?(ldto = 0) ?(st = 0)
    () =
  let fm = c.f in
  let mark = Plan.mark c.b in
  let kernel =
    pick fm
      K.(La_qr_panel_f32, La_qr_panel_f64, La_qr_panel_c64, La_qr_panel_c128)
  in
  let w = words kernel in
  let most = qr_most wd in
  let mp = m - c0 in
  let qblk = K.la_qblk in
  let b_first = c0 / qblk in
  let nblocks = ((m - 1) / qblk) - b_first + 1 in
  let h =
    qr_head ~bytes:fm.bytes ~w:wd ~qm:(if Option.is_some qr then wd * qc else 0)
  in
  let hdr = h.size in
  let p =
    if Option.is_none qr && c.batch < units / 2 && nblocks > 1 then
      Int.min qr_groups nblocks
    else 1
  in
  let bpc = cdiv nblocks p in
  let p = cdiv nblocks bpc in
  let per = bpc * qblk in
  let rows = Int.min per mp in
  let spill = r16 (rows * (wd + 1) * fm.bytes) in
  let sv = mp * wd and st = if st > 0 then st else wd * wd in
  (* The partials, two of each block's by step parity, then two rows of R, and
     with P > 1 the Gram pairs' store. *)
  let store = nblocks * most in
  let drow = 2 * store in
  let gstore = drow + (2 * wd) in
  let parts_stride = gstore + if p > 1 then wd * (wd - 1) / 2 else 0 in
  set w K.la_qr_sbat (m * n);
  set w K.la_qr_lda n;
  set w K.la_qr_stau k;
  set w K.la_qr_sv sv;
  set w K.la_qr_st st;
  set w K.la_qr_ldto (if ldto > 0 then ldto else wd);
  set w K.la_qr_qc qc;
  set w K.la_qr_n n;
  set w K.la_qr_m m;
  set w K.la_qr_c0 c0;
  set w K.la_qr_w wd;
  set w K.la_qr_ld (wd + 1);
  set w K.la_qr_most most;
  set w K.la_qr_parts_stride parts_stride;
  set w K.la_qr_tc h.tc;
  set w K.la_qr_ts h.ts;
  set w K.la_qr_gram h.gram;
  set w K.la_qr_mx h.mx;
  set w K.la_qr_store store;
  set w K.la_qr_drow drow;
  set w K.la_qr_gstore gstore;
  set w K.la_qr_parts_at hdr;
  set w K.la_qr_p p;
  set w K.la_qr_per per;
  set w K.la_qr_spill_bytes spill;
  set w K.la_qr_matrices c.batch;
  let parts = parts_stride * fm.bytes in
  (* The panel stores its factored columns, except where it is the whole matrix:
     then it only reads them, from an operand it may not write. *)
  address w K.la_qr_a (place_of c fv) ~write:(Option.is_none qr);
  address w K.la_qr_tau tau ~write:true;
  Option.iter (fun v -> address w K.la_qr_v (place_of c v) ~write:true) v;
  Option.iter (fun t -> address w K.la_qr_t (place_of c t) ~write:true) t;
  Option.iter
    (fun (q, r) ->
      address w K.la_qr_q q ~write:true;
      address w K.la_qr_r r ~write:true)
    qr;
  if p = 1 then begin
    (* Several matrices a group where the whole of each fits LDS, else one
       matrix a group, its partials and tile in LDS where they fit. 16 lanes sum
       one block's partials alone. *)
    let whole = hdr + parts + spill in
    let g, mats =
      lanes c kernel ~rows
        ~small:(mp <= 16 && wd <= 16 && nblocks = 1)
        ~head:0 ~bytes:whole
    in
    if mats > 1 || g < K.la_threads then begin
      set w K.la_qr_g g;
      set w K.la_qr_mats mats;
      set w K.la_qr_parts_in_lds 1;
      set w K.la_qr_in_lds 1;
      set w K.la_qr_tile_at (hdr + parts);
      set w K.la_qr_stride whole;
      launch c kernel w ~groups:K.la_qr_groups ~count:c.batch ~per:mats
        ~shared:(mats * whole) ()
    end
    else begin
      set w K.la_qr_g K.la_threads;
      set w K.la_qr_mats 1;
      let lds = Plan.shared_most kernel in
      let parts_in_lds = hdr + parts <= lds / 2 in
      set w K.la_qr_parts_in_lds (Bool.to_int parts_in_lds);
      let used = hdr + if parts_in_lds then parts else 0 in
      let in_lds = used + spill <= lds in
      set w K.la_qr_in_lds (Bool.to_int in_lds);
      set w K.la_qr_tile_at used;
      set w K.la_qr_stride (used + if in_lds then spill else 0);
      if not parts_in_lds then
        address w K.la_qr_parts (ws c (c.batch * parts)) ~write:true;
      if not in_lds then
        address w K.la_qr_spill (ws c (c.batch * spill)) ~write:true;
      launch c kernel w ~groups:K.la_qr_groups ~count:c.batch ~per:1
        ~shared:(used + if in_lds then spill else 0)
        ()
    end
  end
  else begin
    (* P groups a matrix, one launch between two meetings of theirs: two a step,
       then the Gram pairs' batches (factor.hip's stages). *)
    set w K.la_qr_g K.la_threads;
    set w K.la_qr_mats 1;
    set w K.la_qr_stride hdr;
    address w K.la_qr_parts (ws c (c.batch * parts)) ~write:true;
    address w K.la_qr_spill (ws c (c.batch * p * spill)) ~write:true;
    let steps = Int.min mp wd in
    let batches = Int.max 1 (cdiv (wd * (wd - 1) / 2) most) in
    let stage s k =
      let o = copy_words w in
      set o K.la_qr_stage s;
      set o K.la_qr_k k;
      launch c kernel o ~groups:K.la_qr_groups ~count:(c.batch * p) ~per:1
        ~shared:hdr ()
    in
    for k = 0 to steps do
      stage K.la_qr_step_a k;
      if k < steps then stage K.la_qr_step_b k
    done;
    if Option.is_some t then
      for i = 1 to batches do
        stage K.la_qr_step_g i
      done
  end;
  Plan.release c.b mark

let qr_out c ~f:fv ~q ~r ~m ~n ~qc =
  let total = ((m * qc) + (qc * n)) * c.batch in
  if total > 0 then begin
    let w = words K.La_qr_out in
    Option.iter
      (fun fv -> address w K.la_qr_out_f (place_of c fv) ~write:false)
      fv;
    address w K.la_qr_out_q q ~write:true;
    address w K.la_qr_out_r r ~write:true;
    set w K.la_qr_out_m m;
    set w K.la_qr_out_n n;
    set w K.la_qr_out_qc qc;
    set w K.la_qr_out_matrices c.batch;
    set w K.la_qr_out_fmt c.f.code;
    launch c K.La_qr_out w ~groups:K.la_qr_out_groups ~count:total
      ~per:K.la_threads ()
  end

let dense_v c ~f:fv ~v ~m ~n ~c0 ~w:wd =
  let w = words K.La_dense_v in
  address w K.la_dense_f (place_of c fv) ~write:false;
  address w K.la_dense_v (place_of c v) ~write:true;
  set w K.la_dense_m m;
  set w K.la_dense_n n;
  set w K.la_dense_c0 c0;
  set w K.la_dense_w wd;
  set w K.la_dense_matrices c.batch;
  set w K.la_dense_fmt c.f.code;
  launch c K.La_dense_v w ~groups:K.la_dense_groups
    ~count:((m - c0) * wd * c.batch)
    ~per:K.la_threads ()

let qr ~q ~r x =
  let batch = batch_of x and m, n = dims x in
  let k = Int.min m n in
  let _, qc = dims q in
  if batch = 0 then None
  else begin
    let c = call x ~batch in
    let f = c.f in
    let qp = of_array c q and rp = of_array c r in
    let wq = qr_width f in
    if m = 0 || k = 0 then qr_out c ~f:None ~q:qp ~r:rp ~m ~n ~qc
    else begin
      let tau = ws c (batch * k * f.bytes) in
      let hdr = (qr_head ~bytes:f.bytes ~w:n ~qm:(n * qc)).size in
      if n <= wq && m <= 1024 && hdr <= qr_header_most then begin
        (* One launch: a read, Q and R stored; the factored panel is not. *)
        let fv =
          if is_contiguous x then of_dst x ~rows:m ~cols:n
          else begin
            let fv =
              contiguous Work
                ~off:(piece c (batch * m * n * f.bytes))
                ~rows:m ~cols:n
            in
            copy c x fv;
            fv
          end
        in
        qr_panel c ~f:fv ~m ~n ~c0:0 ~w:n ~tau ~k ~qr:(qp, rp) ~qc ()
      end
      else begin
        (* Blocks of qb columns, each of panels of wq: a panel's reflectors
           applied to the rest of its block, the block's T built from its
           panels' (T12 = -T1 (V1^H V2) T2, xGEQRT), then the columns right of
           the block updated once, by products of k = qb. *)
        let qb = 4 * wq in
        let nblk = cdiv k qb and big = Int.max n qc in
        let tb = Int.min qb k in
        let e = f.bytes in
        let fo = piece c (batch * m * n * e) in
        let vbuf = piece c (batch * m * tb * e) in
        let tbytes = batch * nblk * tb * tb * e in
        let tbig = piece c tbytes in
        let w1 = piece c (batch * tb * big * e) in
        let w2 = piece c (batch * tb * big * e) in
        let fv = contiguous Work ~off:fo ~rows:m ~cols:n in
        copy c x fv;
        (* T's blocks below their diagonal stay zero. *)
        zero c (tbig * e) ~bytes:tbytes;
        (* c <- (I - V op(T) V^H) c, V h x kb, c h x cols: three products. *)
        let apply v tv ~conj_t cv ~h ~kb ~cols =
          if cols > 0 then begin
            let wv1 =
              { mem = Work; off = w1; sb = kb * cols; rs = cols; cs = 1 }
            in
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
          let tbv = { mem = Work; off = tbo; sb = tb * tb; rs = tb; cs = 1 } in
          let j0 = ref !b0 in
          while !j0 < !b0 + bb do
            let jb = Int.min wq (!b0 + bb - !j0)
            and h = m - !j0
            and jo = !j0 - !b0 in
            let vv = { mem = Work; off = vbuf; sb = h * jb; rs = jb; cs = 1 } in
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
          let vv = { mem = Work; off = vbuf; sb = hb * bb; rs = bb; cs = 1 } in
          dense_v c ~f:fv ~v:vv ~m ~n ~c0:!b0 ~w:bb;
          let jo = ref wq in
          while !jo < bb do
            let jb = Int.min wq (bb - !jo) in
            let g = { mem = Work; off = w1; sb = !jo * jb; rs = jb; cs = 1 } in
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
        let qv = of_dst q ~rows:m ~cols:qc in
        let b0 = ref ((k - 1) / qb * qb) in
        while !b0 >= 0 do
          let bb = Int.min qb (k - !b0) and hb = m - !b0 in
          let vv = { mem = Work; off = vbuf; sb = hb * bb; rs = bb; cs = 1 } in
          dense_v c ~f:fv ~v:vv ~m ~n ~c0:!b0 ~w:bb;
          let tbv =
            {
              mem = Work;
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
    Plan.finish_some c.b
  end

(* Triangular solves *)

(* An array's matrices as a view, where its batch axes form one run of
   strides. *)
let view_of x =
  let l = layout x in
  let r = L.rank l in
  let rs = L.stride l (r - 2) and cs = L.stride l (r - 1) in
  let stride = ref 0 and runs = ref 0 in
  for i = 0 to r - 3 do
    let e = L.dim l i in
    if e > 1 then begin
      let s = L.stride l i in
      if !runs = 0 || !stride <> s * e then incr runs;
      stride := s
    end
  done;
  if !runs > 1 then None
  else Some { mem = Buf (buffer x); off = L.offset l; sb = !stride; rs; cs }

let solve_triangular (t : Nx_kernel.triangle) ~adjoint ~unit_diagonal ~dst a bm
    =
  let batch = batch_of bm and n, r = dims bm in
  if batch = 0 || n = 0 || r = 0 then None
  else begin
    let c = call a ~batch in
    let f = c.f in
    (* a in place where its batch axes form one run, else a copy. *)
    let av =
      match view_of a with
      | Some v -> v
      | None ->
          let v =
            contiguous Work
              ~off:(piece c (batch * n * n * f.bytes))
              ~rows:n ~cols:n
          in
          copy c a v;
          v
    in
    let op_upper = t = Upper <> adjoint in
    let l = if adjoint then tr av else av in
    let x = of_dst dst ~rows:n ~cols:r in
    (* An upper op(a) is a lower one on reversed indices: it substitutes from
       its last row. *)
    let l, xv =
      if op_upper then
        (reversed l ~n, { x with off = x.off + ((n - 1) * r); rs = -r })
      else (l, x)
    in
    if n <= leaf f then begin
      (* One launch: b read in place where it is C-contiguous, x stored, the
         diagonal checked. *)
      let src =
        if is_contiguous bm then
          let slot, at = of_array c bm in
          Some (slot, at + ((xv.off - x.off) * f.bytes))
        else begin
          copy c bm x;
          None
        end
      in
      trsm_leaf c ~l ~conj:adjoint ~unit:unit_diagonal ~x:xv ~n ~r ~src
        ~whole:true
    end
    else begin
      copy c bm x;
      if unit_diagonal then trsm c ~l ~conj:adjoint ~unit:true ~x:xv ~n ~r
      else begin
        (* A matrix with a zero on its diagonal is NaN: flagged first, its x set
           after the solve. *)
        let flag = ws c (4 * batch) in
        let sd = av.rs + av.cs and xn = n * r in
        zero c (snd flag) ~bytes:(4 * batch);
        diag c K.La_diag_zero ~a:av ~sd ~x ~flag ~n ~xn;
        trsm c ~l ~conj:adjoint ~unit:false ~x:xv ~n ~r;
        diag c K.La_nan_where ~a:av ~sd ~x ~flag ~n ~xn
      end
    end;
    Plan.finish_some c.b
  end
