(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The factorisations' launches: factor.metal's steps and the contraction's
   products, one call a routine.

   operand --(copy, unless read in place)--> destination --> panels, leaves and
   blocks (factor.metal) --> products (Contract, one range of k from the output)
   --> finishing steps

   LU, Cholesky and the solves follow nx.amd's factor.ml: every element takes
   the unblocked right-looking algorithm's updates in step order, each a fused
   multiply-add from its own value, so their bits do not depend on the blocking,
   nor on the batch.

   A step launch covers units: a group's matrices, one group's rows of a matrix,
   a thread's column or a thread's element. The units of a step split into as
   many launches as keep each launch's estimate within the budget
   ({!Plan.allowed}), each from its first unit (its kernel's FIRST word) below
   the step's count (COUNT). The split changes no bit: each unit computes what
   it computes in any launch.

   A step declines where one group passes the budget, and every step's group is
   bounded, so none does at the budget: the operands' and the workspace's memory
   bounds a call. An element step's group takes 256 elements, a leaf's (solve,
   Cholesky) at most the leaf's order, a gather's a panel's rows. A product's
   least launch is one 64 x 64 tile over one span of 32 terms, 3.9 us, whose
   sums the next launch resumes. An LU panel's group holds 128 rows and searches
   at most 256 candidates: past 256 groups, launches of levels between two steps
   take each run's best. A QR panel's group holds at most 16 blocks of rows, and
   past 128 groups it reads a meeting's totals, which a sum stage adds over as
   many launches as the superblocks take. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels
module O = Operands

(* Formats *)

(* A routine's arithmetic: its NX_LA_ code, its element's bytes, whether it is
   complex, and its dtype. *)
type fmt = { code : int; bytes : int; cx : bool; dt : D.any }

let fmt_of (x : O.operand) =
  match x.dtype with
  | D.Any D.Float32 as dt -> Some { code = K.la_f32; bytes = 4; cx = false; dt }
  | D.Any D.Complex64 as dt ->
      Some { code = K.la_c64; bytes = 8; cx = true; dt }
  | D.Any _ -> None

let pick f (f32, c64) = if f.cx then c64 else f32

(* The most rows of a solve's leaf, and the columns of an LU panel and the order
   of a Cholesky block: half at complex64, whose block of 64 x 65 elements
   passes a group's 32 KiB. *)
let leaf f = if f.cx then K.la_leaf / 2 else K.la_leaf
let panel_width = leaf
let chol_nb = leaf

(* The M1 Max's GPU cores: a batch of fewer than half leaves it idle with a
   group a matrix, of at least twice fills it with a SIMD-group a matrix.
   Neither choice changes a result's bits. *)
let units = 32
let cdiv a b = (a + b - 1) / b
let r16 a = (a + 15) / 16 * 16

(* The split of a recursive routine's [n]: half, rounded to a multiple of
   [q]. *)
let split n q =
  let h = n / 2 / q * q in
  if h > 0 then h else q

(* Views *)

(* Where a view's matrices lie: the buffer of the call's operand [k]
   ({!Operands}) from its first byte, or the workspace from byte [at], a piece
   {!Plan.take} gave. *)
type mem = Op of int | Work of int

(* Matrices of [mem]: element (z, i, j) at [off + z sb + i rs + j cs], in
   elements of the format. *)
type view = { mem : mem; off : int; sb : int; rs : int; cs : int }

let at v i j = { v with off = v.off + (i * v.rs) + (j * v.cs) }
let tr v = { v with rs = v.cs; cs = v.rs }

(* [v] read from its last row and column: an upper triangle as a lower one. *)
let reversed v ~n =
  { v with off = v.off + ((n - 1) * (v.rs + v.cs)); rs = -v.rs; cs = -v.cs }

let contiguous mem ~off ~rows ~cols =
  { mem; off; sb = rows * cols; rs = cols; cs = 1 }

let layout (x : O.operand) = x.layout
let buffer (x : O.operand) = x.buffer
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

(* [x]'s matrices as a view where its batch axes form one run of strides, each
   of extent 1 ignored. *)
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
  else Some { mem = Op (buffer x); off = L.offset l; sb = !stride; rs; cs }

(* Calls *)

(* A call under construction: its builder, format, batch, and the threads Metal
   lets a group of each kernel have ({!Loop.map}'s [holds]). *)
type call = { b : Plan.builder; f : fmt; batch : int; holds : K.t -> int }

(* A place: a slot and a byte in it. *)
let place_of c v =
  match v.mem with
  | Op k -> (Plan.slot c.b k, v.off * c.f.bytes)
  | Work at -> (Plan.workspace c.b, at + (v.off * c.f.bytes))

let ws c bytes = (Plan.workspace c.b, Plan.take c.b bytes)

let elt_bytes (x : O.operand) =
  let (D.Any dt) = x.dtype in
  D.bytes dt 1

let of_array c x = (Plan.slot c.b (buffer x), offset x * elt_bytes x)
let of_dst x = contiguous (Op (buffer x)) ~off:(offset x)

(* A piece of the workspace of the batch's matrices of [rows] x [cols]. *)
let work c ~rows ~cols =
  let at = Plan.take c.b (c.batch * rows * cols * c.f.bytes) in
  contiguous (Work at) ~off:0 ~rows ~cols

(* A launch's words and refs. *)
type words = { w : Bytes.t; mutable refs : Plan.ref list }

let words k = { w = Plan.words k; refs = [] }
let set l i v = Plan.set l.w i v

(* Word [i] of [l] holds the address of [place]. *)
let address l i (slot, at) ~write =
  set l i at;
  l.refs <- { Plan.word = i; slot; write } :: l.refs

let copy_words l = { w = Bytes.copy l.w; refs = l.refs }

(* Launches *)

(* A launch's counts. *)
let cost ~bytes ~ops ~chain = { Plan.bytes; ops; chain }

(* The links of a dependent division or square root, and of a group's meeting:
   estimates. *)
let div_links = 16
let barrier_links = 4

(* A call that a step finds no launches for, which {!routine} declines. *)
exception Decline

(* The bound below which a launch's units take the divisor's 32-bit division:
   2^32, or less for tests. *)
let divide_below = Atomic.make (1 lsl 32)

let bound_divide ~below =
  if below < 1 || below > 1 lsl 32 then
    invalid_arg "Nx_metal.Factor.bound_divide: below";
  Atomic.set divide_below below;
  Plan.renew ()

(* The word [wide] of {!divided} names for a kernel that divides no unit. *)
let undivided = -1

(* The launches of [k] over [count] units, [per] a group: the units' end in word
   [last], each launch's first unit in word [first], and as many groups a launch
   as keep its estimate within the budget, [cost g] the counts of [g] groups.
   Where its kernel divides its units by [divisor], word [wide] of a launch is 1
   where its last unit or [divisor] reaches {!divide_below}, which the divisor's
   32-bit division cannot take, and 0 otherwise. Raises [Decline] where one
   group passes the budget. *)
let divided c k l ~first ~last ~count ~per ~shared ~wide ~divisor cost =
  if count > 0 then begin
    let resident = Plan.resident ~granted:c.holds k ~shared in
    let threads = K.threads k in
    let fits g =
      Plan.seconds ~threads ~groups:(g, 1, 1) ~resident (cost g)
      <= Plan.allowed ()
    in
    let groups = cdiv count per in
    let most =
      match Plan.most ~fits groups with
      | Some g -> g
      | None -> raise_notrace Decline
    in
    set l last count;
    let refs = Array.of_list (List.rev l.refs) in
    let g0 = ref 0 in
    while !g0 < groups do
      let g = Int.min most (groups - !g0) in
      let w = copy_words l in
      set w first (!g0 * per);
      if wide <> undivided then begin
        let below = Atomic.get divide_below in
        let last = Int.min count ((!g0 + g) * per) - 1 in
        set w wide (Bool.to_int (last >= below || divisor >= below))
      end;
      Plan.launch c.b k ~groups:(g, 1, 1) ~shared ~resident ~cost:(cost g) w.w
        refs;
      g0 := !g0 + g
    done
  end

(* {!divided}'s launches of a kernel that divides no unit. *)
let launches c k l ~first ~last ~count ~per ?(shared = 0) cost =
  divided c k l ~first ~last ~count ~per ~shared ~wide:undivided ~divisor:0 cost

(* The counts of [g] groups of a kernel of [threads] that touches [bytes] and
   runs [chain] links a thread. *)
let busy ~threads ~bytes ~chain g =
  cost ~bytes:(g * bytes) ~ops:(g * threads * chain) ~chain

(* An element kernel's launches over [count] elements, each touching [bytes]: a
   line at each end of a launch's run, a dependent read and a few operations a
   thread. *)
let elementwise c k l ~first ~last ~count ~bytes =
  let per = K.la_threads in
  launches c k l ~first ~last ~count ~per (fun g ->
      cost
        ~bytes:((g * per * bytes) + (2 * Plan.line))
        ~ops:(g * per * 8)
        ~chain:(Plan.read_links + 8))

(* The operations of a multiply-add of the format: four at complex64. *)
let mac_ops f = if f.cx then 4 else 1

(* Products and copies *)

(* [v]'s [batch] matrices of [rows] x [cols], for the contraction. *)
let matrix c v ~rows ~cols =
  let l =
    L.v ~offset:v.off
      ~strides:(Iarray.of_array [| v.sb; v.rs; v.cs |])
      (Iarray.of_array [| c.batch; rows; cols |])
  in
  match v.mem with
  | Op k -> Contract.operand k c.f.dt l
  | Work at -> Contract.work ~at c.f.dt l

(* [y <- y - op(a) m] (or [+] without [neg]) from [y]'s own value, or from +0
   without [init], in [ranges] ranges of k: one is the unblocked algorithm's
   chain. Under [tri], only the outputs of that triangle of [y]. The product's
   copies return to the workspace once it is planned. *)
let product c ?(neg = true) ?(conj_a = false) ?tri ?(ranges = 1) ?(init = true)
    y a m ~rows ~cols ~k =
  if rows > 0 && cols > 0 && c.batch > 0 && (k > 0 || not init) then begin
    let o =
      {
        Contract.init_first = init;
        neg;
        conj_a;
        conj_b = false;
        tri;
        ranges = Some ranges;
      }
    in
    Contract.product c.b o ~acc:c.f.dt ~init ~y:(matrix c y ~rows ~cols)
      (matrix c a ~rows ~cols:k) (matrix c m ~rows:k ~cols)
  end

(* [x]'s matrices, of [rows] x [cols], read through [xv], into [v]. *)
let copy c xv v ~rows ~cols =
  Contract.copy c.b ~src:(matrix c xv ~rows ~cols) ~dst:(matrix c v ~rows ~cols)

(* Zeroes the [bytes], a multiple of 4, of the workspace from byte [at], in as
   many launches as keep within the budget. *)
let zero c at ~bytes =
  let count = bytes / 4 in
  if count > 0 then begin
    let per = K.threads K.Zero in
    let resident = Plan.resident ~granted:c.holds K.Zero ~shared:0 in
    let cost g =
      cost ~bytes:((4 * g * per) + (2 * Plan.line)) ~ops:(g * per * 8) ~chain:8
    in
    let fits g =
      Plan.seconds ~threads:per ~groups:(g, 1, 1) ~resident (cost g)
      <= Plan.allowed ()
    in
    let most =
      match Plan.most ~fits (cdiv count per) with
      | Some g -> g * per
      | None -> raise_notrace Decline
    in
    let first = ref 0 in
    while !first < count do
      let n = Int.min most (count - !first) in
      let l = words K.Zero in
      address l K.zero_dst (Plan.workspace c.b, at + (4 * !first)) ~write:true;
      set l K.zero_count n;
      let g = cdiv n per in
      Plan.launch c.b K.Zero ~groups:(g, 1, 1) ~resident ~cost:(cost g) l.w
        (Array.of_list (List.rev l.refs));
      first := !first + n
    done
  end

(* Triangular solves *)

(* [x <- op(l)^-1 x] for the lower triangular view [l] of [n <= leaf] rows and
   the [n] x [r] view [x], a thread a column: x read from [src] where given, and
   under [whole] NaN for a matrix of a zero on its diagonal. *)
let trsm_leaf c ~l ~conj ~unit ~x ~n ~r ~src ~whole =
  let k = pick c.f K.(La_trsm_f32, La_trsm_c64) in
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
  set w K.la_trsm_rdiv (Plan.divisor r);
  set w K.la_trsm_conj (Bool.to_int conj);
  set w K.la_trsm_unit (Bool.to_int unit);
  set w K.la_trsm_whole (Bool.to_int whole);
  let threads = K.la_trsm_threads and e = c.f.bytes in
  (* A column: its n elements read and written, its triangle's n²/2 macs and n
     divisions; a group reads l whole. *)
  let chain = (n * (n + 1) / 2 * mac_ops c.f) + (div_links * n) in
  let bytes = (n * n * e) + (2 * threads * n * Int.max e (Plan.line / 4)) in
  divided c k w ~first:K.la_trsm_first ~last:K.la_trsm_count
    ~count:(c.batch * r) ~per:threads ~wide:K.la_trsm_wide ~divisor:r
    ~shared:(n * threads * e)
    (busy ~threads ~bytes ~chain)

(* [x <- op(l)^-1 x] for the lower triangular view [l] of [n] rows and the [n] x
   [r] view [x]: leaves of [leaf] rows, the rest by products (ReLAPACK's
   recursive xTRSM). Each element takes its updates in increasing step order,
   whatever the split. *)
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

(* A solve's diagonal: [flag] set for a matrix of a zero on [a]'s diagonal, [sd]
   apart, by La_diag_zero, then that matrix's [xn] elements of [x] NaN by
   La_nan_where. *)
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
  set l K.la_diag_fmt c.f.code;
  let each = if k = K.La_diag_zero then n else xn in
  elementwise c k l ~first:K.la_diag_first ~last:K.la_diag_count
    ~count:(each * c.batch) ~bytes:(Plan.line / 4)

(* LU *)

(* Whether a panel of [mp] rows spans groups where it fits one: the batch leaves
   the GPU idle and the rows are many. *)
let spans c ~mp = c.batch < units / 2 && mp > 512

(* The columns of an LU's leaf where its panel spans groups: each step then
   moves [narrow] columns of every row, not the panel width's, and the
   recursion's products take the rest on the matrix units. *)
let narrow = 8

(* A panel over several groups: groups of [least_rows] rows or a multiple,
   [spread] groups a GPU core where the rows are many. Each group's fixed work a
   step (the search of the slots, two meetings, the pivot row's copy) is then
   paid [spread x units] times a step, not once every 128 rows. *)
let least_rows = 128
let spread = 2

(* For tests: the rows a group of a panel over several, in place of the
   rule's. *)
let rows_bound = Atomic.make None

let bound_lu ~rows =
  (match rows with
  | Some r when r < 1 -> invalid_arg "Nx_metal.Factor.bound_lu: rows"
  | _ -> ());
  Atomic.set rows_bound rows;
  Plan.renew ()

(* The candidates each level of a panel over [p] groups reduces, a run of
   [K.la_threads] a group, until one run is left. *)
let rec level_sizes p =
  if p <= K.la_threads then []
  else
    let n = cdiv p K.la_threads in
    n :: level_sizes n

(* A panel's pivot candidates in threadgroup memory: a slot of each of the
   group's SIMD-groups, a slot's head each, for each of two step parities; the
   second parity's byte, and their bytes. *)
let cands_parity = K.la_threads / 32 * K.la_slot_head
let cand = 2 * cands_parity

(* A matrix's G threads and the matrices a group holds, for a tile of [bytes] a
   matrix of [rows] rows: 16 lanes where the panel is at most 16 x 16, a
   SIMD-group where the batch fills the GPU and the rows are few, else the whole
   group; as many matrices as a group's memory holds past [head], an even count
   at 16 lanes, two to a SIMD-group (factor.metal's [member]). *)
let lanes c ~rows ~small ~head ~bytes =
  let fit g =
    Int.min (K.la_threads / g) ((Plan.group_memory - head) / Int.max 1 bytes)
  in
  if small && fit 16 >= 2 then (16, fit 16 / 2 * 2)
  else if c.batch >= 2 * units && rows <= 64 && fit 32 >= 2 then (32, fit 32)
  else (K.la_threads, 1)

(* The links of one step of a panel whose G threads hold [rows] rows of [w]
   columns: a thread's rows' updates, the division, and the search's shuffles
   and meeting. *)
let step_links c ~rows ~g ~w =
  (cdiv rows g * w * mac_ops c.f) + div_links + 16 + (2 * barrier_links)

(* LU of the panel of columns [c0, c0 + w) of rows [c0, m) of [a]'s matrices of
   [m] x [n]: pivots into [piv] ([k] a matrix), and either the row interchanges
   on the panel's other columns and [perm] (gather), or, reading the whole
   matrix from [src], [perm] stored by the panel. P = 1 group a matrix in one
   launch where it fits the budget, else P groups a matrix, one launch a
   step. *)
let lu_panel c ~a ~m ~n ~c0 ~w:wd ~piv ~k ~perm ~src =
  let f = c.f in
  let mp = m - c0 in
  let steps = Int.min mp wd in
  if steps > 0 then begin
    let mark = Plan.mark c.b in
    let kernel = pick f K.(La_lu_panel_f32, La_lu_panel_c64) in
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
    set pw K.la_panel_row cand;
    set pw K.la_panel_cands_parity cands_parity;
    (* A matrix's tile, then its rows' positions, of [per] rows. *)
    let tile per = r16 (per * ldt * f.bytes) in
    let bytes per = tile per + (per * 8) in
    let threads = K.la_threads in
    let one_group =
      let head = r16 cand in
      let g, _ =
        lanes c ~rows:mp ~small:(mp <= 16 && wd <= 16) ~head ~bytes:(bytes mp)
      in
      let chain = steps * step_links c ~rows:mp ~g ~w:wd in
      Plan.seconds ~threads ~groups:(1, 1, 1)
        (busy ~threads ~bytes:(2 * mp * wd * f.bytes) ~chain 1)
      <= Plan.allowed ()
    in
    let slot_bytes = r16 (K.la_slot_head + (wd * f.bytes)) in
    (* The links and bytes of a group of [per] rows in a step launch over [p]
       groups: the search of the slots or of the last level, a step's update of
       its rows, the next step's candidates; the first loads the rows and the
       last stores them. *)
    let step_cost ~p ~per =
      let sizes = level_sizes p in
      let cands = match List.rev sizes with n :: _ -> n | [] -> p in
      let chain =
        step_links c ~rows:per ~g:threads ~w:wd
        + (cdiv cands threads * Plan.read_links)
        + (2 * cdiv per threads * wd)
      in
      let searched =
        if sizes = [] then p * slot_bytes
        else (cands * K.la_slot_head) + slot_bytes
      in
      (chain, (2 * per * wd * f.bytes) + searched)
    in
    let spread_head = r16 (cand + (wd * f.bytes)) in
    (* The rows of a group of P > 1: [spread] groups a core over the batch, at
       least two a matrix, halved while one group's step passes the budget. *)
    let group_rows () =
      let fits per =
        let chain, bytes = step_cost ~p:(cdiv mp per) ~per in
        let resident =
          Plan.resident ~granted:c.holds kernel ~shared:spread_head
        in
        Plan.seconds ~threads ~groups:(1, 1, 1) ~resident
          (busy ~threads ~bytes ~chain 1)
        <= Plan.allowed ()
      in
      let round per = Int.max least_rows (per / least_rows * least_rows) in
      let groups = Int.max 2 (cdiv (spread * units) c.batch) in
      let rec go per =
        if per <= least_rows || fits per then per else go (round (per / 2))
      in
      go (round (cdiv (cdiv mp groups) least_rows * least_rows))
    in
    (* P groups a matrix where the panel spans groups, or where one group's
       panel passes the budget: then one launch a step, each group's tile in the
       workspace. *)
    let p =
      if spans c ~mp || not one_group then
        match Atomic.get rows_bound with
        | Some r -> cdiv mp r
        | None -> cdiv mp (group_rows ())
      else 1
    in
    let per = cdiv mp p in
    (* The candidates, then with P > 1 the pivot row, then the tiles. *)
    let head = if p > 1 then spread_head else r16 cand in
    set pw K.la_panel_per per;
    set pw K.la_panel_p p;
    set pw K.la_panel_pdiv (Plan.divisor p);
    set pw K.la_panel_spill_bytes (bytes per);
    set pw K.la_panel_pos (tile per);
    set pw K.la_panel_head head;
    if p = 1 then begin
      let g, mats =
        lanes c ~rows:mp ~small:(mp <= 16 && wd <= 16) ~head ~bytes:(bytes mp)
      in
      let in_lds = head + (mats * bytes mp) <= Plan.group_memory in
      set pw K.la_panel_g g;
      set pw K.la_panel_mats mats;
      set pw K.la_panel_in_lds (Bool.to_int in_lds);
      if not in_lds then begin
        (* In device memory, the tile is column-major. *)
        set pw K.la_panel_ld mp;
        address pw K.la_panel_spill (ws c (c.batch * bytes mp)) ~write:true
      end;
      let chain = steps * step_links c ~rows:mp ~g ~w:wd in
      launches c kernel pw ~first:K.la_panel_first ~last:K.la_panel_count
        ~count:(cdiv c.batch mats) ~per:1
        ~shared:(head + if in_lds then mats * bytes mp else 0)
        (busy ~threads ~bytes:(2 * mats * mp * wd * f.bytes) ~chain)
    end
    else begin
      set pw K.la_panel_g threads;
      set pw K.la_panel_mats 1;
      set pw K.la_panel_in_lds 0;
      set pw K.la_panel_ld per;
      set pw K.la_panel_slot_bytes slot_bytes;
      (* Each group's slot of each step parity. *)
      set pw K.la_panel_slots_parity (p * slot_bytes);
      set pw K.la_panel_slots_stride (2 * p * slot_bytes);
      address pw K.la_panel_slots
        (ws c (c.batch * 2 * p * slot_bytes))
        ~write:true;
      address pw K.la_panel_spill (ws c (c.batch * p * bytes per)) ~write:true;
      (* Where the slots pass one a thread, levels between two steps: each takes
         the best of every run of [threads] candidates, the first of the slots,
         until one run is left, which the next step searches. Each level's
         candidates a matrix, from the first's. *)
      let sizes = level_sizes p in
      let lv_stride = List.fold_left ( + ) 0 sizes in
      (* Each level's input (its candidate and count) and output. *)
      let levels =
        let rec go at n out = function
          | [] -> []
          | size :: rest -> (at, n, out) :: go out size (out + size) rest
        in
        go 0 p 0 sizes
      in
      let lv_in, cands =
        match (List.rev levels, List.rev sizes) with
        | (_, _, out) :: _, n :: _ -> (out, n)
        | _ -> (-1, p)
      in
      set pw K.la_panel_lv_in lv_in;
      set pw K.la_panel_n_in cands;
      if sizes <> [] then begin
        set pw K.la_panel_lv_stride lv_stride;
        address pw K.la_panel_levels
          (ws c (c.batch * lv_stride * K.la_slot_head))
          ~write:true
      end;
      let chain, traffic = step_cost ~p ~per in
      (* A level's group: a candidate a thread, a slot's in its own line, then
         the group's best. *)
      let level step i (at_in, n_in, out) =
        let w = copy_words pw in
        set w K.la_panel_k step;
        set w K.la_panel_level (i + 1);
        set w K.la_panel_lv_in at_in;
        set w K.la_panel_lv_out out;
        set w K.la_panel_n_in n_in;
        set w K.la_panel_runs_div (Plan.divisor (cdiv n_in threads));
        let read =
          if i = 0 then Int.min Plan.line slot_bytes else K.la_slot_head
        in
        divided c kernel w ~first:K.la_panel_first ~last:K.la_panel_count
          ~count:(c.batch * cdiv n_in threads)
          ~per:1 ~wide:K.la_panel_wide ~divisor:(cdiv n_in threads)
          ~shared:(r16 (threads / 32 * K.la_slot_head))
          (busy ~threads ~bytes:(threads * read)
             ~chain:(Plan.read_links + 16 + (2 * barrier_links)))
      in
      for step = 0 to steps do
        let w = copy_words pw in
        set w K.la_panel_k step;
        divided c kernel w ~first:K.la_panel_first ~last:K.la_panel_count
          ~count:(c.batch * p) ~per:1 ~shared:head ~wide:K.la_panel_wide
          ~divisor:p
          (busy ~threads ~bytes:traffic ~chain);
        if step < steps then List.iteri (level step) levels
      done
    end;
    (* With [src], the whole matrix: perm stored by the panel. *)
    if Option.is_none src then begin
      let span = K.la_gather_span in
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
      set w K.la_gather_fmt f.code;
      let blocks = Int.max 1 (cdiv ((n - wd) * f.bytes / 4) span) in
      set w K.la_gather_per blocks;
      set w K.la_gather_pdiv (Plan.divisor blocks);
      (* A group: 2 steps rows of [span] words read and written, each row its
         own line, and the positions. *)
      let rows = 2 * steps in
      let chain =
        (2 * Plan.read_links) + (2 * barrier_links)
        + (cdiv (rows * span) threads * 4)
      in
      divided c K.La_gather w ~first:K.la_gather_first ~last:K.la_gather_count
        ~count:(c.batch * blocks) ~per:1 ~wide:K.la_gather_wide ~divisor:blocks
        ~shared:((3 * 2 * K.la_leaf * 8) + (2 * K.la_leaf * span * 4))
        (busy ~threads ~bytes:(rows * ((2 * Plan.line) + 24)) ~chain)
    end;
    Plan.release c.b mark
  end

(* perm = the identity, each matrix's [m] rows. *)
let iota c perm ~m =
  let l = words K.La_iota in
  address l K.la_iota_perm perm ~write:true;
  set l K.la_iota_m m;
  elementwise c K.La_iota l ~first:K.la_iota_first ~last:K.la_iota_count
    ~count:(c.batch * m) ~bytes:8

(* The columns of an LU's leaf, or of its one panel, for a panel of [mp]
   rows. *)
let lu_width c ~mp =
  if spans c ~mp then Int.min narrow (panel_width c.f) else panel_width c.f

(* LU of the columns [c0, c0 + w) of rows [c0, m) of [a]'s matrices of [m] x [n]
   (ReLAPACK's recursive xGETRF): panels of at most the panel width, each half's
   columns right of it solved by trsm and the rows below updated by a product.
   Each element takes its updates in increasing step order, whatever the
   split. *)
let rec lu_rec c ~a ~m ~n ~c0 ~w ~piv ~k ~perm =
  let wp = lu_width c ~mp:(m - c0) in
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

(* Cholesky *)

(* L of the diagonal block of order [nb] at ([k], [k]) of [a]'s matrices of
   order [n]: [fail] set for a matrix whose pivot is not positive, or, reading
   the whole matrix from [src], its whole result stored, the [upper] triangle as
   U = Lᴴ. *)
let chol_block c ~a ~n ~k ~nb ~fail ~src ~upper =
  let kernel = pick c.f K.(La_chol_f32, La_chol_c64) in
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
  let g, mats = lanes c ~rows:nb ~small:(nb <= 16) ~head:0 ~bytes:tile in
  set w K.la_chol_stride tile;
  set w K.la_chol_g g;
  set w K.la_chol_mats mats;
  (* A step: the column's division and square root, a thread's rows' updates,
     three meetings. *)
  let step =
    (cdiv nb g * nb * mac_ops c.f) + (2 * div_links) + (3 * barrier_links)
  in
  let threads = K.la_threads in
  launches c kernel w ~first:K.la_chol_first ~last:K.la_chol_count
    ~count:(cdiv c.batch mats) ~per:1 ~shared:(mats * tile)
    (busy ~threads
       ~bytes:(2 * mats * nb * nb * c.f.bytes)
       ~chain:((nb * step) + (2 * cdiv (nb * nb) g)))

(* Each matrix of order [n]: NaN where [fail] is set; else U = Lᴴ under [upper],
   the other triangle +0. Under [tr], L lies above the diagonal, as its
   transpose. *)
let chol_finish c ~a ~fail ~n ~upper ~tr =
  let w = words K.La_chol_finish in
  address w K.la_finish_a (place_of c a) ~write:true;
  address w K.la_finish_fail fail ~write:false;
  set w K.la_finish_n n;
  set w K.la_finish_upper (Bool.to_int upper);
  set w K.la_finish_tr (Bool.to_int tr);
  set w K.la_finish_fmt c.f.code;
  elementwise c K.La_chol_finish w ~first:K.la_finish_first
    ~last:K.la_finish_count
    ~count:(n * n * c.batch)
    ~bytes:(2 * c.f.bytes)

(* QR *)

(* The columns of a QR panel, and of the one launch's matrix. *)
let qr_width = 32

(* The most bytes of the one launch's header. *)
let qr_header_most = 32 * 1024

(* The groups a QR panel's rows split into, each summing its rows' partials one
   block of NX_LA_QBLK rows after another: at most 128, each of as few blocks as
   that allows; past 128 x 16 blocks, groups of 16 blocks, and a meeting's sums
   in a stage of their own. *)
let qr_groups = 128
let qr_blocks_most = 16

(* A QR panel's threadgroup memory a matrix from byte 0: its sums' [most]
   values, a step's w_c, T, the Gram pairs and, where Q is stored, M ([qm]
   elements); each region's byte and the header's bytes. *)
type header = { tc : int; ts : int; gram : int; mx : int; size : int }

let qr_most w = Int.max (w + 2) 64

let qr_head ~bytes ~w ~qm =
  let tc = qr_most w * bytes in
  let ts = tc + (w * bytes) in
  let gram = ts + (w * w * bytes) in
  let mx = gram + (w * (w - 1) / 2 * bytes) in
  { tc; ts; gram; mx; size = r16 (mx + (qm * bytes)) }

(* The links of one step of a QR panel whose G threads hold [rows] rows of [w]
   columns: the column's sums and the reflector's application to the others, its
   blocks' tree, xLARFG's divisions and square root, and the meetings. *)
let qr_step_links c ~rows ~g ~w ~nblocks =
  (2 * cdiv rows g * (w + 1) * mac_ops c.f)
  + (cdiv nblocks g * 8)
  + 16 + (3 * div_links) + (6 * barrier_links)

(* Householder QR of the panel of columns [c0, c0 + w) of rows [c0, m) of [fv]'s
   matrices of [m] x [n]: tau into [tau] ([k] a matrix); dense V into [v] and T
   into [t] where given; where the panel is the whole matrix, [qr]'s Q (m x qc)
   and R stored and the panel only read. P = 1 group a matrix in one launch, or
   with no [qr], where the batch leaves the GPU idle or one group's launch
   passes the budget, P groups a matrix, two launches a step. *)
let qr_panel c ~f:fv ~m ~n ~c0 ~w:wd ~tau ~k ?v ?t ?qr ~qc ?(ldto = 0) ?(st = 0)
    () =
  let fm = c.f in
  let mark = Plan.mark c.b in
  let kernel = pick fm K.(La_qr_panel_f32, La_qr_panel_c64) in
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
  let threads = K.la_threads in
  let steps = Int.min mp wd in
  (* Q's columns formed in the one launch: a thread's rows' sums over the
     panel's reflectors. *)
  let q_links g =
    if Option.is_some qr then cdiv m g * qc * wd * mac_ops fm else 0
  in
  let one_group =
    let chain =
      (steps * qr_step_links c ~rows:mp ~g:threads ~w:wd ~nblocks)
      + q_links threads
    in
    Plan.seconds ~threads ~groups:(1, 1, 1)
      (busy ~threads ~bytes:(2 * mp * wd * fm.bytes) ~chain 1)
    <= Plan.allowed ()
  in
  let bpc =
    if Option.is_none qr && nblocks > 1 && (c.batch < units / 2 || not one_group)
    then Int.min (cdiv nblocks qr_groups) qr_blocks_most
    else nblocks
  in
  let p = cdiv nblocks bpc in
  (* Each group reads the stored totals of a meeting's sums, which a stage
     computes once, rather than every block's partials. *)
  let summed = p > qr_groups in
  let per = bpc * qblk in
  let rows = Int.min per mp in
  let spill = r16 (rows * (wd + 1) * fm.bytes) in
  let sv = mp * wd and st = if st > 0 then st else wd * wd in
  (* The partials, two of each block's by step parity, then two rows of R, with
     P > 1 the Gram pairs' store, and where summed a meeting's totals. *)
  let store = nblocks * most in
  let drow = 2 * store in
  let gstore = drow + (2 * wd) in
  let tot = gstore + if p > 1 then wd * (wd - 1) / 2 else 0 in
  let parts_stride = tot + if summed then most else 0 in
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
  set w K.la_qr_tot tot;
  set w K.la_qr_summed (Bool.to_int summed);
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
    (* Several matrices a group where the whole of each fits a group's memory,
       else one matrix a group, its partials and tile there where they fit. 16
       lanes sum one block's partials alone. *)
    let whole = hdr + parts + spill in
    let g, mats =
      lanes c ~rows
        ~small:(mp <= 16 && wd <= 16 && nblocks = 1)
        ~head:0 ~bytes:whole
    in
    let traffic mats = mats * ((m * n) + (m * qc) + (qc * n)) * fm.bytes in
    let chain g =
      (steps * qr_step_links c ~rows:mp ~g ~w:wd ~nblocks) + q_links g
    in
    if mats > 1 || g < threads then begin
      set w K.la_qr_g g;
      set w K.la_qr_mats mats;
      set w K.la_qr_parts_in_lds 1;
      set w K.la_qr_in_lds 1;
      set w K.la_qr_tile_at (hdr + parts);
      set w K.la_qr_stride whole;
      launches c kernel w ~first:K.la_qr_first ~last:K.la_qr_count
        ~count:(cdiv c.batch mats) ~per:1 ~shared:(mats * whole)
        (busy ~threads ~bytes:(traffic mats) ~chain:(chain g))
    end
    else begin
      set w K.la_qr_g threads;
      set w K.la_qr_mats 1;
      let lds = Plan.group_memory in
      let parts_in_lds = hdr + parts <= lds / 2 in
      set w K.la_qr_parts_in_lds (Bool.to_int parts_in_lds);
      let used = hdr + if parts_in_lds then parts else 0 in
      (* The kernel places a tile in the group's memory only with its
         partials. *)
      let in_lds = parts_in_lds && used + spill <= lds in
      set w K.la_qr_in_lds (Bool.to_int in_lds);
      set w K.la_qr_tile_at used;
      set w K.la_qr_stride (used + if in_lds then spill else 0);
      if not parts_in_lds then
        address w K.la_qr_parts (ws c (c.batch * parts)) ~write:true;
      if not in_lds then
        address w K.la_qr_spill (ws c (c.batch * spill)) ~write:true;
      launches c kernel w ~first:K.la_qr_first ~last:K.la_qr_count
        ~count:c.batch ~per:1
        ~shared:(used + if in_lds then spill else 0)
        (busy ~threads ~bytes:(traffic 1) ~chain:(chain threads))
    end
  end
  else begin
    (* P groups a matrix, one launch between two meetings of theirs: two a step,
       then the Gram pairs' batches (factor.metal's stages). *)
    set w K.la_qr_g threads;
    set w K.la_qr_mats 1;
    set w K.la_qr_stride hdr;
    address w K.la_qr_parts (ws c (c.batch * parts)) ~write:true;
    address w K.la_qr_spill (ws c (c.batch * p * spill)) ~write:true;
    let batches = Int.max 1 (cdiv (wd * (wd - 1) / 2) most) in
    (* A launch: a step over a group's rows, the partials' reads of the others;
       the first loads the rows and the last stores them. *)
    let chain =
      qr_step_links c ~rows ~g:threads ~w:wd ~nblocks:bpc
      + (if summed then Plan.read_links
         else cdiv nblocks threads * Plan.read_links)
      + (2 * cdiv rows threads * wd)
    in
    let reads = if summed then (most + (2 * wd)) * fm.bytes else parts in
    let traffic = (2 * rows * wd * fm.bytes) + reads in
    let stage s k =
      let o = copy_words w in
      set o K.la_qr_stage s;
      set o K.la_qr_k k;
      launches c kernel o ~first:K.la_qr_first ~last:K.la_qr_count
        ~count:(c.batch * p) ~per:1 ~shared:hdr
        (busy ~threads ~bytes:traffic ~chain)
    in
    (* A sum stage: a group a matrix, a SIMD-group's values each taking a
       superblock's partials, a line each, by a tree of shuffles; as many
       superblocks a launch as keep it within the budget, the next launch
       resuming each total. *)
    let sups = cdiv nblocks K.la_qsup in
    let sup_cost n =
      busy ~threads
        ~bytes:(most * n * K.la_qsup * Plan.line)
        ~chain:(n * cdiv most (threads / 32) * (Plan.read_links + 16))
    in
    let sum s k =
      let fits n =
        Plan.seconds ~threads ~groups:(1, 1, 1) (sup_cost n 1)
        <= Plan.allowed ()
      in
      let most_sups =
        match Plan.most ~fits sups with
        | Some n -> n
        | None -> raise_notrace Decline
      in
      let from = ref 0 in
      while !from < sups do
        let upto = Int.min sups (!from + most_sups) in
        let o = copy_words w in
        set o K.la_qr_stage s;
        set o K.la_qr_k k;
        set o K.la_qr_from !from;
        set o K.la_qr_to upto;
        launches c kernel o ~first:K.la_qr_first ~last:K.la_qr_count
          ~count:c.batch ~per:1 ~shared:hdr
          (sup_cost (upto - !from));
        from := upto
      done
    in
    for k = 0 to steps do
      if summed && k > 0 then sum K.la_qr_sum_a k;
      stage K.la_qr_step_a k;
      if k < steps then begin
        if summed then sum K.la_qr_sum_b k;
        stage K.la_qr_step_b k
      end
    done;
    if Option.is_some t then
      for i = 1 to batches do
        if summed then sum K.la_qr_sum_g i;
        stage K.la_qr_step_g i
      done
  end;
  Plan.release c.b mark

(* Q, the first [qc] columns of the m x m identity, and R, the upper triangle of
   [fv]'s first rows, or zero without [fv]. *)
let qr_out c ~f:fv ~q ~r ~m ~n ~qc =
  let count = ((m * qc) + (qc * n)) * c.batch in
  let w = words K.La_qr_out in
  Option.iter
    (fun fv -> address w K.la_qr_out_f (place_of c fv) ~write:false)
    fv;
  address w K.la_qr_out_q q ~write:true;
  address w K.la_qr_out_r r ~write:true;
  set w K.la_qr_out_m m;
  set w K.la_qr_out_n n;
  set w K.la_qr_out_qc qc;
  set w K.la_qr_out_fmt c.f.code;
  elementwise c K.La_qr_out w ~first:K.la_qr_out_first ~last:K.la_qr_out_count
    ~count ~bytes:(2 * c.f.bytes)

(* Dense V of the panel of columns [c0, c0 + w) of [fv]'s matrices of [m] x [n]:
   rows [c0, m), its unit diagonal and zeros above it, into [v]. *)
let dense_v c ~f:fv ~v ~m ~n ~c0 ~w:wd =
  let w = words K.La_dense_v in
  address w K.la_dense_f (place_of c fv) ~write:false;
  address w K.la_dense_v (place_of c v) ~write:true;
  set w K.la_dense_m m;
  set w K.la_dense_n n;
  set w K.la_dense_c0 c0;
  set w K.la_dense_w wd;
  set w K.la_dense_fmt c.f.code;
  elementwise c K.La_dense_v w ~first:K.la_dense_first ~last:K.la_dense_count
    ~count:((m - c0) * wd * c.batch)
    ~bytes:(2 * c.f.bytes)

(* QR past the one launch (xGEQRT, xORGQR): the operand copied into the
   workspace, blocks of [qb] columns, each of panels of the panel width; a
   panel's reflectors applied to the rest of its block, the block's T built from
   its panels' (T12 = -T1 (V1ᴴ V2) T2), then the columns right of the block
   updated once, by products of k = qb. Q is formed from the identity's columns
   by the blocks in reverse. *)
let qr_blocked c ~xv ~q:(qp, qv) ~r:rp ~m ~n ~qc ~k =
  let wq = qr_width and batch = c.batch in
  let qb = 4 * wq in
  let nblk = cdiv k qb and big = Int.max n qc and tb = Int.min qb k in
  let e = c.f.bytes in
  let tau = ws c (batch * k * e) in
  let fv = work c ~rows:m ~cols:n in
  let vbuf = Plan.take c.b (batch * m * tb * e) in
  let tbytes = batch * nblk * tb * tb * e in
  let tbig = Plan.take c.b tbytes in
  let w1 = Plan.take c.b (batch * tb * big * e) in
  let w2 = Plan.take c.b (batch * tb * big * e) in
  copy c xv fv ~rows:m ~cols:n;
  (* T's blocks below their diagonal stay zero. *)
  zero c tbig ~bytes:tbytes;
  let vview ~h ~w = { mem = Work vbuf; off = 0; sb = h * w; rs = w; cs = 1 } in
  let tview b0 =
    {
      mem = Work tbig;
      off = b0 / qb * batch * tb * tb;
      sb = tb * tb;
      rs = tb;
      cs = 1;
    }
  in
  (* x <- (I - V op(T) Vᴴ) x, V h x kb, x h x cols: three products. *)
  let apply v tv ~conj_t x ~h ~kb ~cols =
    if cols > 0 then begin
      let wv1 = { mem = Work w1; off = 0; sb = kb * cols; rs = cols; cs = 1 } in
      let wv2 = { wv1 with mem = Work w2 } in
      product c ~init:false ~neg:false ~conj_a:true
        ~ranges:(Contract.ranges ~rows:kb ~cols ~k:h)
        wv1 (tr v) x ~rows:kb ~cols ~k:h;
      product c ~init:false ~neg:false ~conj_a:conj_t
        ~ranges:(Contract.ranges ~rows:kb ~cols ~k:kb)
        wv2
        (if conj_t then tr tv else tv)
        wv1 ~rows:kb ~cols ~k:kb;
      product c
        ~ranges:(Contract.ranges ~rows:h ~cols ~k:kb)
        x v wv2 ~rows:h ~cols ~k:kb
    end
  in
  let b0 = ref 0 in
  while !b0 < k do
    let bb = Int.min qb (k - !b0) and hb = m - !b0 in
    let tbv = tview !b0 in
    let j0 = ref !b0 in
    while !j0 < !b0 + bb do
      let jb = Int.min wq (!b0 + bb - !j0) and h = m - !j0 and jo = !j0 - !b0 in
      let vv = vview ~h ~w:jb in
      qr_panel c ~f:fv ~m ~n ~c0:!j0 ~w:jb ~tau ~k ~v:vv ~t:(at tbv jo jo) ~qc
        ~ldto:tb ~st:(tb * tb) ();
      apply vv (at tbv jo jo) ~conj_t:true
        (at fv !j0 (!j0 + jb))
        ~h ~kb:jb
        ~cols:(!b0 + bb - !j0 - jb);
      j0 := !j0 + wq
    done;
    (* The block's dense V, then T's blocks above its panels' diagonals. *)
    let vv = vview ~h:hb ~w:bb in
    dense_v c ~f:fv ~v:vv ~m ~n ~c0:!b0 ~w:bb;
    let jo = ref wq in
    while !jo < bb do
      let jb = Int.min wq (bb - !jo) in
      let g = { mem = Work w1; off = 0; sb = !jo * jb; rs = jb; cs = 1 } in
      let xg = { g with mem = Work w2 } in
      product c ~init:false ~neg:false ~conj_a:true
        ~ranges:(Contract.ranges ~rows:!jo ~cols:jb ~k:hb)
        g (tr vv) (at vv 0 !jo) ~rows:!jo ~cols:jb ~k:hb;
      product c ~init:false ~neg:false xg tbv g ~rows:!jo ~cols:jb ~k:!jo;
      product c (at tbv 0 !jo) xg (at tbv !jo !jo) ~rows:!jo ~cols:jb ~k:jb;
      jo := !jo + wq
    done;
    apply vv tbv ~conj_t:true
      (at fv !b0 (!b0 + bb))
      ~h:hb ~kb:bb
      ~cols:(n - !b0 - bb);
    b0 := !b0 + qb
  done;
  qr_out c ~f:(Some fv) ~q:qp ~r:rp ~m ~n ~qc;
  let b0 = ref ((k - 1) / qb * qb) in
  while !b0 >= 0 do
    let bb = Int.min qb (k - !b0) and hb = m - !b0 in
    let vv = vview ~h:hb ~w:bb in
    dense_v c ~f:fv ~v:vv ~m ~n ~c0:!b0 ~w:bb;
    apply vv (tview !b0) ~conj_t:false (at qv !b0 !b0) ~h:hb ~kb:bb
      ~cols:(qc - !b0);
    b0 := !b0 - qb
  done

(* Routines *)

let call ~holds f ~batch = { b = Plan.make (); f; batch; holds }

(* Plans [body c] of the operand [x]'s format, or declines a format the family
   has no arithmetic for and a call a step of which finds no launches. *)
let routine ~holds x ~batch body =
  match fmt_of x with
  | None -> Error A.Declined
  | Some f -> (
      match body (call ~holds f ~batch) with
      | r -> r
      | exception (Decline | Contract.Decline) -> Error A.Declined)

(* Each routine's operands: its destinations, then its loads. *)

let cholesky ~holds (t : Nx_kernel.triangle) (xs : O.t) =
  let dst = xs.dsts.(0) and x = xs.loads.(0) in
  let batch = batch_of x and n, _ = dims x in
  if batch = 0 || n = 0 then Ok None
  else
    routine ~holds x ~batch @@ fun c ->
    match view_of x with
    | None -> Error A.Declined
    | Some xv ->
        let upper = t = Upper and d = of_dst dst ~rows:n ~cols:n in
        if n <= chol_nb c.f then begin
          (* One launch: the operand read in place where it is C-contiguous,
             else from its copy in [d], the whole result stored. *)
          let src =
            if is_contiguous x then of_array c x
            else begin
              copy c xv d ~rows:n ~cols:n;
              place_of c d
            end
          in
          chol_block c ~a:d ~n ~k:0 ~nb:n ~fail:None ~src:(Some src) ~upper
        end
        else begin
          (* Blocked right-looking, as nx.amd's: a complex operand transposed
             into [d], so that the trailing updates compute A22's transpose,
             which a complex product writes in place only where its columns are
             neighbours. *)
          let flip = c.f.cx in
          let a = if flip then tr d else d in
          copy c (if flip then tr xv else xv) d ~rows:n ~cols:n;
          let fail = ws c (4 * batch) in
          zero c (snd fail) ~bytes:(4 * batch);
          let nb = chol_nb c.f in
          let k = ref 0 in
          while !k < n do
            let kb = Int.min nb (n - !k) in
            let rest = n - !k - kb in
            chol_block c ~a ~n ~k:!k ~nb:kb ~fail:(Some fail) ~src:None
              ~upper:false;
            if rest > 0 then begin
              (* L21 = A21 L11^-H: conj(L11) L21^T = A21^T. *)
              trsm c ~l:(at a !k !k) ~conj:true ~unit:false
                ~x:(tr (at a (!k + kb) !k))
                ~n:kb ~r:rest;
              (* A22 -= L21 L21^H, through its transpose: (i, c) of A22, c <= i,
                 takes conj(L_cp) L_ip, the order the unblocked algorithm's
                 updates take their operands in. *)
              let l21 = at a (!k + kb) !k in
              product c ~conj_a:true ~tri:`Upper
                (tr (at a (!k + kb) (!k + kb)))
                l21 (tr l21) ~rows:rest ~cols:rest ~k:kb
            end;
            k := !k + kb
          done;
          chol_finish c ~a:d ~fail ~n ~upper ~tr:flip
        end;
        Ok (Plan.finish_some ~most:8 c.b)

let lu ~holds (xs : O.t) =
  let dst = xs.dsts.(0) and pivots = xs.dsts.(1) and perm = xs.dsts.(2) in
  let x = xs.loads.(0) in
  let batch = batch_of x and m, n = dims x in
  let k = Int.min m n in
  if batch = 0 || m = 0 then Ok None
  else
    routine ~holds x ~batch @@ fun c ->
    let pm = of_array c perm in
    if k = 0 then begin
      (* No column: perm alone, the identity. *)
      iota c pm ~m;
      Ok (Plan.finish_some ~most:8 c.b)
    end
    else
      match view_of x with
      | None -> Error A.Declined
      | Some xv ->
          let a = of_dst dst ~rows:m ~cols:n and piv = of_array c pivots in
          if n <= lu_width c ~mp:m then begin
            (* One panel, reading the operand in place where it is C-contiguous,
               else its copy in [a]: LU, pivots and perm stored. *)
            let src =
              if is_contiguous x then of_array c x
              else begin
                copy c xv a ~rows:m ~cols:n;
                place_of c a
              end
            in
            lu_panel c ~a ~m ~n ~c0:0 ~w:n ~piv ~k ~perm:pm ~src:(Some src)
          end
          else begin
            (* The operand copied into [a], factored there, then the columns
               past the k-th of a wide matrix solved. *)
            copy c xv a ~rows:m ~cols:n;
            iota c pm ~m;
            lu_rec c ~a ~m ~n ~c0:0 ~w:k ~piv ~k ~perm:pm;
            if n > k then
              trsm c ~l:a ~conj:false ~unit:true ~x:(at a 0 k) ~n:k ~r:(n - k)
          end;
          Ok (Plan.finish_some ~most:8 c.b)

let solve_triangular ~holds (t : Nx_kernel.triangle) ~adjoint ~unit_diagonal
    (xs : O.t) =
  let dst = xs.dsts.(0) and a = xs.loads.(0) and bm = xs.loads.(1) in
  let batch = batch_of bm and n, r = dims bm in
  if batch = 0 || n = 0 || r = 0 then Ok None
  else
    routine ~holds a ~batch @@ fun c ->
    match (view_of a, view_of bm) with
    | Some av, Some bv ->
        let op_upper = t = Upper <> adjoint in
        let l = if adjoint then tr av else av in
        let x = of_dst dst ~rows:n ~cols:r in
        (* An upper op(a) is a lower one on reversed indices: it substitutes
           from its last row. *)
        let l, xv =
          if op_upper then
            (reversed l ~n, { x with off = x.off + ((n - 1) * r); rs = -r })
          else (l, x)
        in
        if n <= leaf c.f && is_contiguous bm then begin
          (* One launch: b read in place, x stored, the diagonal checked. *)
          let slot, at = of_array c bm in
          let src = Some (slot, at + ((xv.off - x.off) * c.f.bytes)) in
          trsm_leaf c ~l ~conj:adjoint ~unit:unit_diagonal ~x:xv ~n ~r ~src
            ~whole:true
        end
        else begin
          (* b copied into x, flushed, then solved there. *)
          copy c bv x ~rows:n ~cols:r;
          if unit_diagonal then trsm c ~l ~conj:adjoint ~unit:true ~x:xv ~n ~r
          else begin
            (* A matrix with a zero on its diagonal is NaN: flagged first, its x
               set after the solve. *)
            let flag = ws c (4 * batch) in
            let sd = av.rs + av.cs and xn = n * r in
            zero c (snd flag) ~bytes:(4 * batch);
            diag c K.La_diag_zero ~a:av ~sd ~x ~flag ~n ~xn;
            trsm c ~l ~conj:adjoint ~unit:false ~x:xv ~n ~r;
            diag c K.La_nan_where ~a:av ~sd ~x ~flag ~n ~xn
          end
        end;
        Ok (Plan.finish_some ~most:8 c.b)
    | _ -> Error A.Declined

let qr ~holds (xs : O.t) =
  let q = xs.dsts.(0) and r = xs.dsts.(1) and x = xs.loads.(0) in
  let batch = batch_of x and m, n = dims x in
  let k = Int.min m n in
  let _, qc = dims q in
  if batch = 0 || (m * qc) + (qc * n) = 0 then Ok None
  else
    routine ~holds x ~batch @@ fun c ->
    let qp = of_array c q and rp = of_array c r in
    let one_launch =
      n <= qr_width && m <= 1024
      && (qr_head ~bytes:c.f.bytes ~w:n ~qm:(n * qc)).size <= qr_header_most
    in
    if m = 0 || k = 0 then begin
      (* No reflector: Q the identity's columns, R zero. *)
      qr_out c ~f:None ~q:qp ~r:rp ~m ~n ~qc;
      Ok (Plan.finish_some ~most:8 c.b)
    end
    else
      match view_of x with
      | None -> Error A.Declined
      | Some xv ->
          if one_launch then begin
            (* One launch: a read of the operand, or of its copy where it is not
               C-contiguous, Q and R stored; the factored panel is not. *)
            let tau = ws c (batch * k * c.f.bytes) in
            let fv =
              if is_contiguous x then of_dst x ~rows:m ~cols:n
              else begin
                let fv = work c ~rows:m ~cols:n in
                copy c xv fv ~rows:m ~cols:n;
                fv
              end
            in
            qr_panel c ~f:fv ~m ~n ~c0:0 ~w:n ~tau ~k ~qr:(qp, rp) ~qc ()
          end
          else
            qr_blocked c ~xv
              ~q:(qp, of_dst q ~rows:m ~cols:qc)
              ~r:rp ~m ~n ~qc ~k;
          Ok (Plan.finish_some ~most:8 c.b)
