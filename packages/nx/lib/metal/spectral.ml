(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The spectral routines' launches: spectral.metal's steps, one call a routine.

   operand (read in place where C-contiguous, else copied) --> eigh: Jacobi (n
   <= 32), else the tridiagonal reduction in panels, then bisection for w alone,
   or divide and conquer and the panels' reflectors applied for v --> svd:
   Jacobi (both extents <= 32), the identity's columns where k = 0, else for the
   values bidiagonalisation and bisection on the Golub-Kahan matrix --> eig,
   values: load, Hessenberg in ranges of columns, QR in slices of sweeps, w

   This follows nx.amd's spectral.ml, its orders of operations and workspace
   layouts. A step's units, a group's or a thread's as kernels.h states, split
   into as many launches as keep each launch's estimate within the budget
   ({!Plan.allowed}), each from its first unit (its kernel's FIRST word) to
   below its last (COUNT): the kernels loop over their launch's units alone, and
   no bit follows where a launch's units start or end. A unit whose estimate
   alone passes the budget declines the call.

   An iteration that does not converge within its limit makes its matrix NaN in
   every result: 60 Jacobi sweeps (sp_jacobi.h) and 30 max(10, h) QR sweeps of
   eig in all, h the order balancing leaves to iterate on (sp_eig.h). A call
   reads the limits once, and each of its launches passes them as words, which
   tests lower. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels
module O = Operands

let cdiv a b = (a + b - 1) / b
let r16 a = (a + 15) / 16 * 16
let r256 x = (x + 255) land lnot 255

(* Limits *)

type limits = {
  sweeps : int;
  steps : int;
  qr : int;
  passes : int;
  secular : int;
}

(* Balancing's scaling stops after 256 passes: each scale stays within xGEBAL's
   guards, 2^±103, so a scale drifting an exponent a pass settles within 206
   passes. *)
let passes = 256

(* The most steps of a root of divide and conquer's secular equation (xLAED4's
   limit). *)
let secular = 30

let limits =
  Atomic.make { sweeps = 60; steps = max_int; qr = 30; passes; secular }

let bound_iterations ~sweeps ~steps ~qr ~passes ~secular =
  if sweeps < 1 then invalid_arg "Nx_metal.bound_iterations: sweeps";
  if steps < 1 then invalid_arg "Nx_metal.bound_iterations: steps";
  if qr < 0 then invalid_arg "Nx_metal.bound_iterations: qr";
  if passes < 1 then invalid_arg "Nx_metal.bound_iterations: passes";
  if secular < 1 then invalid_arg "Nx_metal.bound_iterations: secular";
  Atomic.set limits { sweeps; steps; qr; passes; secular };
  Plan.renew ()

(* Formats *)

(* A routine's arithmetic: whether it is complex, its element's bytes and the
   floats of one. *)
type fmt = { cx : bool; t : int; parts : int }

let fmt_of (x : O.operand) =
  match x.dtype with
  | D.Any D.Float32 -> Some { cx = false; t = 4; parts = 1 }
  | D.Any D.Complex64 -> Some { cx = true; t = 8; parts = 2 }
  | D.Any _ -> None

let pick f (f32, c64) = if f.cx then c64 else f32

(* Calls *)

(* A call under construction: its builder, format, limits, and the threads Metal
   lets a group of each kernel have. *)
type call = { b : Plan.builder; f : fmt; lim : limits; holds : K.t -> int }

(* Raised while planning a call one of whose units alone passes the budget, or
   one of whose kernels Metal grants fewer threads than its group: the call
   declines. *)
exception Decline

(* A place: a slot and a byte in it. *)
type place = int * int

let of_array c (x : O.operand) =
  let (D.Any dt) = x.dtype in
  (Plan.slot c.b x.buffer, L.offset x.layout * D.bytes dt 1)

let ws c bytes = (Plan.workspace c.b, Plan.take c.b bytes)
let plus ((s, at) : place) n = (s, at + n)

(* A launch's words and refs. *)
type words = { w : Bytes.t; mutable refs : Plan.ref list }

let words k = { w = Plan.words k; refs = [] }
let set l i v = Plan.set l.w i v

let address l i (slot, at) ~write =
  set l i at;
  l.refs <- { Plan.word = i; slot; write } :: l.refs

let copy_words l = { w = Bytes.copy l.w; refs = l.refs }

(* Estimates *)

(* The links of a dependent division or square root, a group's meeting, a Givens
   rotation computed, and a reduction over a group by its fixed tree of eight
   levels: estimates (nx_metal.mli, Time). *)
let div_links = 16
let barrier_links = 4
let givens_links = 30
let tree_links = 8 * (barrier_links + 1)

(* The links of a loop's step over device memory whose reads are in flight
   together, a quarter of a dependent read: td_step's and larft's, measured on
   the M1 Max. *)
let overlapped_links = Plan.read_links / 4

(* The operations of a multiply-add of the format: four at complex64. *)
let mac f = if f.cx then 4 else 1
let cost ~bytes ~ops ~chain = { Plan.bytes; ops; chain }

(* The counts of [g] groups of [threads] that each touch [bytes] and run [chain]
   links a thread. *)
let busy ~threads ~bytes ~chain g =
  cost ~bytes:(g * bytes) ~ops:(g * threads * chain) ~chain

(* The groups of [k] a core holds of [shared] bytes. A kernel Metal grants fewer
   threads than its group declines the call. *)
let resident c k shared =
  if c.holds k < K.threads k then raise Decline;
  Plan.resident ~granted:c.holds k ~shared

(* The most of [n] that [fits], or the call declines. *)
let most ~fits n =
  match Plan.most ~fits n with Some m -> m | None -> raise Decline

let fits c k ~shared ~groups cost =
  Plan.seconds ~threads:(K.threads k) ~groups:(groups, 1, 1)
    ~resident:(resident c k shared) cost
  <= Plan.allowed ()

(* Launches *)

(* The launches of [k] over units [0, count), [per] units a group, [l]'s words
   with each launch's units in words [first] and [last], and as many groups a
   launch as keep its estimate within the budget, [cost g] the counts of [g]
   groups. *)
let steps c k l ~first ~last ~count ~per ?(shared = 0) cost =
  if count > 0 then begin
    let resident = resident c k shared in
    let groups = cdiv count per in
    let fits g = fits c k ~shared ~groups:g (cost g) in
    let most = Int.min (most ~fits groups) Plan.most_groups in
    let refs = Array.of_list (List.rev l.refs) in
    let g0 = ref 0 in
    while !g0 < groups do
      let g = Int.min most (groups - !g0) in
      let w = copy_words l in
      set w first (!g0 * per);
      set w last (Int.min count ((!g0 + g) * per));
      Plan.launch c.b k ~groups:(g, 1, 1) ~shared ~resident ~cost:(cost g) w.w
        refs;
      g0 := !g0 + g
    done
  end

(* The links of a unit of [count] that runs [links] and [divides] divisions of
   its index by the host's divisor words (sp_common.h's [split]): device.h's
   divide below 2^32 units, within [links], and a 64-bit division of 1024 links
   past them, as loop.ml weighs it. *)
let divided ~count ~divides links =
  if count lsr 32 = 0 then links else links + (divides * 1024)

(* A step of a thread a unit over [count] units, each touching [bytes] and
   running [links] links after a dependent read. *)
let elementwise c k l ~first ~last ~count ~bytes ~links =
  let per = K.threads k in
  steps c k l ~first ~last ~count ~per (fun g ->
      cost
        ~bytes:((g * per * bytes) + (2 * Plan.line))
        ~ops:(g * per * links)
        ~chain:(Plan.read_links + links))

(* A resumed step: [total] steps of each of [count] units, [per] a group, each
   step at most [step] links, as launches of R steps, in word [q]: R the most
   that keep a wave of groups within the budget, at most the limit's steps. Word
   [start], where given, is 1 in the first round of launches alone. Every launch
   runs R steps of each unfinished unit and saves its state, so the launches
   finish every unit whatever its values. [cost g chain] counts [g] groups whose
   threads run [chain] links. *)
let resumed c k l ~first ~last ~count ~per ~shared ~q ?start ~step ~total cost =
  let wave = Int.min (cdiv count per) (resident c k shared * Plan.cores) in
  let fits r = fits c k ~shared ~groups:wave (cost wave (r * step)) in
  let r = Int.min (most ~fits total) c.lim.steps in
  for round = 1 to cdiv total r do
    let l = copy_words l in
    set l q r;
    Option.iter (fun i -> set l i (Bool.to_int (round = 1))) start;
    steps c k l ~first ~last ~count ~per ~shared (fun g -> cost g (r * step))
  done

(* Jacobi *)

(* A Jacobi group's meeting word and its matrices' slots: 16 bytes, then 20 a
   matrix, rounded up to 16. *)
let jacobi_head per = 16 + r16 (20 * per)
let eigh_per n = Int.max 1 (K.sp_threads / (n * n))

let eigh_shared f n =
  let per = eigh_per n in
  jacobi_head per
  + (per * ((2 * n * n) + (2 * n)) * f.t)
  + (((2 * per * n) + (per * n land 1)) * 4)

let svd_span ~mm ~k = (mm * k) + (k * k) + (mm * mm) + (2 * k) + mm

let svd_shared f ~per ~mm ~k =
  jacobi_head per
  + (per * svd_span ~mm ~k * f.t)
  + (per * 2 * k * 4)
  + (per * (k + mm) * 4)

(* svd's matrices a group: as many as its threads take that a group's memory
   holds. *)
let svd_per f m n =
  let mm = Int.max m n and k = Int.min m n in
  let per = ref (Int.max 1 (K.sp_threads / (m * n))) in
  while !per > 1 && svd_shared f ~per:!per ~mm ~k > Plan.group_memory do
    decr per
  done;
  !per

(* The links of a group's Jacobi on its [per] matrices at the limit of sweeps:
   [np - 1] rounds a sweep, each a rotation computed, its updates of [elements]
   spread over a matrix's threads, and the meetings. *)
let jacobi_chain c ~per ~np ~elements ~extra =
  let gs = K.sp_threads / per in
  let round =
    givens_links + tree_links + (4 * barrier_links)
    + (cdiv elements gs * 2 * mac c.f)
  in
  (c.lim.sweeps * (np - 1) * round) + extra

let eigh_jacobi c ~a ~w ~v ~nb ~n =
  let k = pick c.f K.(Sp_eigh_jacobi_f32, Sp_eigh_jacobi_c64) in
  let l = words k in
  let per = eigh_per n in
  address l K.sp_eigh_jacobi_a a ~write:false;
  address l K.sp_eigh_jacobi_w w ~write:true;
  Option.iter (fun v -> address l K.sp_eigh_jacobi_v v ~write:true) v;
  set l K.sp_eigh_jacobi_nb nb;
  set l K.sp_eigh_jacobi_n n;
  set l K.sp_eigh_jacobi_per per;
  set l K.sp_eigh_jacobi_sweeps c.lim.sweeps;
  (* A round updates a's rows and columns and v's columns of its pairs: 3 n²
     elements; loading and storing a matrix, 3 n² more. *)
  let gs = K.sp_threads / per in
  let chain =
    jacobi_chain c ~per
      ~np:(n + (n land 1))
      ~elements:(3 * n * n)
      ~extra:((cdiv (3 * n * n) gs * 4) + (2 * Plan.read_links))
  in
  let bytes = per * ((n * n * c.f.t * 2) + (n * 4)) in
  let shared = eigh_shared c.f n in
  steps c k l ~first:K.sp_eigh_jacobi_first ~last:K.sp_eigh_jacobi_count
    ~count:(cdiv nb per) ~per:1 ~shared
    (busy ~threads:K.sp_threads ~bytes ~chain)

let svd_jacobi c ~a ~s ~vectors ~nb ~m ~n ~uc ~vr =
  let k = pick c.f K.(Sp_svd_jacobi_f32, Sp_svd_jacobi_c64) in
  let l = words k in
  let mm = Int.max m n and kk = Int.min m n in
  let per = svd_per c.f m n in
  let span = svd_span ~mm ~k:kk in
  address l K.sp_svd_jacobi_a a ~write:false;
  address l K.sp_svd_jacobi_s s ~write:true;
  Option.iter
    (fun (u, vh) ->
      address l K.sp_svd_jacobi_u u ~write:true;
      address l K.sp_svd_jacobi_vh vh ~write:true)
    vectors;
  set l K.sp_svd_jacobi_nb nb;
  set l K.sp_svd_jacobi_m m;
  set l K.sp_svd_jacobi_n n;
  set l K.sp_svd_jacobi_uc uc;
  set l K.sp_svd_jacobi_vr vr;
  set l K.sp_svd_jacobi_per per;
  set l K.sp_svd_jacobi_span span;
  set l K.sp_svd_jacobi_sweeps c.lim.sweeps;
  (* A round: each pair's three sums over M rows, its rotation of g's and v's
     columns; u formed and the factors stored, M² and K² more. *)
  let gs = K.sp_threads / per in
  let chain =
    jacobi_chain c ~per
      ~np:(kk + (kk land 1))
      ~elements:(kk * ((5 * mm) + (2 * kk)) / 2)
      ~extra:
        ((cdiv ((mm * mm * kk) + (3 * span)) gs * 2 * mac c.f)
        + (2 * Plan.read_links))
  in
  let bytes = per * ((m * n) + (m * uc) + (vr * n) + kk) * c.f.t in
  let shared = svd_shared c.f ~per ~mm ~k:kk in
  steps c k l ~first:K.sp_svd_jacobi_first ~last:K.sp_svd_jacobi_count
    ~count:(cdiv nb per) ~per:1 ~shared
    (busy ~threads:K.sp_threads ~bytes ~chain)

(* The identity's first [r] rows and [cols] columns into [nb] matrices of [x]:
   sp_eye takes no units, so its launches split by matrices. An empty [x] names
   no slot, since a plan's slots are each some launch's. *)
let eye c x ~nb ~r ~cols =
  let per = r * cols in
  if nb > 0 && per > 0 then begin
    let x = of_array c x in
    let k = K.Sp_eye in
    let resident = resident c k 0 in
    let threads = K.threads k in
    let counts mats =
      let els = mats * per in
      ( cdiv els threads,
        cost
          ~bytes:((els * c.f.t) + (2 * Plan.line))
          ~ops:(cdiv els threads * threads * 8)
          ~chain:(Plan.read_links + 8) )
    in
    let fits mats =
      let g, n = counts mats in
      g <= Plan.most_groups && fits c k ~shared:0 ~groups:g n
    in
    let mats = most ~fits nb in
    let z = ref 0 in
    while !z < nb do
      let mz = Int.min mats (nb - !z) in
      let l = words k in
      address l K.sp_eye_x (plus x (!z * per * c.f.t)) ~write:true;
      set l K.sp_eye_nz mz;
      set l K.sp_eye_r r;
      set l K.sp_eye_c cols;
      set l K.sp_eye_parts c.f.parts;
      let g, n = counts mz in
      Plan.launch c.b k ~groups:(g, 1, 1) ~resident ~cost:n l.w
        (Array.of_list (List.rev l.refs));
      z := !z + mz
    done
  end

(* Bisection *)

(* The links of a step of a Sturm count: a division and the pivot's tests, each
   on the last. *)
let sturm_links = div_links + 4

(* Eigenvalues [i0, i0 + cnt) of each of [nb] tridiagonals of order [n],
   diagonal [d] and subdiagonal [e] [stride] floats a matrix, into [w], [cnt] a
   matrix: Gershgorin's bounds, a thread a matrix, then the bisection, a thread
   an eigenvalue, whose steps are each halving's test and Sturm rows,
   resumed. *)
let tri_bisect c ~d ~e ~stride ~w ~nb ~n ~i0 ~cnt =
  let bd = ws c (nb * K.sp_bounds * 4) in
  let st = ws c (nb * cnt * K.sp_bisect * 4) in
  (let k = K.Sp_tri_bounds in
   let l = words k in
   address l K.sp_tri_bounds_d d ~write:false;
   address l K.sp_tri_bounds_e e ~write:false;
   address l K.sp_tri_bounds_bd bd ~write:true;
   set l K.sp_tri_bounds_stride stride;
   set l K.sp_tri_bounds_n n;
   elementwise c k l ~first:K.sp_tri_bounds_first ~last:K.sp_tri_bounds_count
     ~count:nb
     ~bytes:((2 * n * 4) + (K.sp_bounds * 4))
     ~links:(n * 8));
  let k = K.Sp_tri_bisect in
  let l = words k in
  address l K.sp_tri_bisect_d d ~write:false;
  address l K.sp_tri_bisect_e e ~write:false;
  address l K.sp_tri_bisect_bd bd ~write:false;
  address l K.sp_tri_bisect_st st ~write:true;
  address l K.sp_tri_bisect_w w ~write:true;
  set l K.sp_tri_bisect_stride stride;
  set l K.sp_tri_bisect_n n;
  set l K.sp_tri_bisect_i0 i0;
  set l K.sp_tri_bisect_cnt cnt;
  let threads = K.threads k in
  (* A launch reads a value's bounds and state, and its threads share their
     matrix's rows. *)
  let state = K.sp_bisect * 4 in
  resumed c k l ~first:K.sp_tri_bisect_first ~last:K.sp_tri_bisect_count
    ~count:(nb * cnt) ~per:threads ~shared:0 ~q:K.sp_tri_bisect_q
    ~start:K.sp_tri_bisect_start ~step:sturm_links
    ~total:((K.sp_halvings * (n + 1)) + 1)
    (fun g chain ->
      cost
        ~bytes:((g * threads * 2 * state) + (g * 2 * 2 * n * 4))
        ~ops:(g * threads * chain)
        ~chain:(chain + (2 * Plan.read_links)))

(* eigh past Jacobi *)

(* The tridiagonal reduction's workspace a matrix (sp_tridiag.h): A, n²
   elements, W and Vc, n x NB each, T and G, NB² each, and τ, n, of the format's
   elements; d and e, n floats each; the flag and exponent and the stage's two
   scan words. *)
let td_per f n =
  let nb = K.sp_tnb in
  r256 ((((n * n) + (2 * n * nb) + (2 * nb * nb) + n) * f.t) + (2 * n * 4) + 16)

(* Divide and conquer's workspace a matrix (sp_stedc.h): Q, Qt and S, n² floats
   each, ten arrays of n floats, five of n ints, two words and each root's
   secular state. *)
let dc_per n =
  r256
    ((((3 * n * n) + (10 * n)) * 4)
    + (((5 * n) + 2) * 4)
    + (n * K.sp_secular * 4))

(* The real format divide and conquer computes in. *)
let real = { cx = false; t = 4; parts = 1 }

(* A matrix of a product: [nb] of them in the workspace from byte [at], the
   first at element [off] past it, each [s] elements apart. *)
type operand = { at : int; off : int; s : int }

(* [y <- op(a) op(m)], or [y <- y − op(a) op(m)] where [init], over the [nb]
   matrices: rows x cols outputs of [k] terms, each operand row-major of its
   leading dimension where its op is 0, transposed where 1, and conjugated too
   where 2. *)
let gemm c ~nb ~y ~a ~m ~rows ~cols ~k ~lda ~ldb ~ldc ~opa ~opb ~init =
  if rows > 0 && cols > 0 && k > 0 then begin
    let dt = if c.f.cx then D.Any D.Complex64 else D.Any D.Float32 in
    let view x ~rs ~cs ~r ~cl =
      let l =
        L.v ~offset:x.off
          ~strides:(Iarray.of_array [| x.s; rs; cs |])
          (Iarray.of_array [| nb; r; cl |])
      in
      Contract.work ~at:x.at dt l
    in
    let op o x ~ld ~r ~cl =
      if o = 0 then view x ~rs:ld ~cs:1 ~r ~cl else view x ~rs:1 ~cs:ld ~r ~cl
    in
    let k' = if c.f.cx then 2 * k else k in
    let o =
      {
        Contract.plain with
        neg = init;
        conj_a = opa = 2;
        conj_b = opb = 2;
        ranges = Some (Contract.ranges ~rows ~cols ~k:k');
      }
    in
    Contract.product c.b o ~acc:dt ~init
      ~y:(view y ~rs:ldc ~cs:1 ~r:rows ~cl:cols)
      (op opa a ~ld:lda ~r:rows ~cl:k)
      (op opb m ~ld:ldb ~r:k ~cl:cols)
  end

(* A step of a group a matrix of the [nb] matrices, each running [chain] links
   and touching [bytes]. *)
let each c k l ~first ~last ~nb ~shared ~chain ~bytes =
  steps c k l ~first ~last ~count:nb ~per:1 ~shared
    (busy ~threads:(K.threads k) ~bytes ~chain)

(* The tridiagonal reduction (blocked xHETRD, lower) of the [nb] n x n matrices
   at [a] into the workspace from [tw], [tp] bytes a matrix: each panel's
   columns by td_step and td_hemv, its trailing update as two products, then T's
   diagonal into d. *)
let tridiagonal c ~a ~tw ~tp ~nb ~n =
  let f = c.f and t = c.f.t and nn = n * n and tnb = K.sp_tnb in
  let threads = K.sp_threads in
  (let k = K.Sp_td_stage in
   let l = words k in
   address l K.sp_td_stage_src a ~write:false;
   address l K.sp_td_stage_ws tw ~write:true;
   set l K.sp_td_stage_n n;
   set l K.sp_td_stage_parts f.parts;
   set l K.sp_td_stage_per tp;
   set l K.sp_td_stage_dn (Plan.divisor n);
   let phase p ~units =
     let l = copy_words l in
     set l K.sp_td_stage_phase p;
     set l K.sp_td_stage_du (Plan.divisor units);
     l
   in
   let first = K.sp_td_stage_first and last = K.sp_td_stage_count in
   (* A matrix's scan words cleared; a row's scan of its elements; an element
      staged. *)
   elementwise c k (phase 0 ~units:1) ~first ~last ~count:nb ~bytes:8 ~links:2;
   elementwise c k (phase 1 ~units:n) ~first ~last ~count:(nb * n) ~bytes:(n * t)
     ~links:(divided ~count:(nb * n) ~divides:1 (n * 8));
   elementwise c k (phase 2 ~units:nn) ~first ~last ~count:(nb * nn)
     ~bytes:(2 * t)
     ~links:(divided ~count:(nb * nn) ~divides:2 8));
  let st = tp / t in
  let base = snd tw in
  let j0 = ref 0 in
  while !j0 < n do
    let nbk = Int.min tnb (n - !j0) in
    for i = 0 to nbk do
      (let k = pick f K.(Sp_td_step_f32, Sp_td_step_c64) in
       let l = words k in
       address l K.sp_td_step_ws tw ~write:true;
       set l K.sp_td_step_n n;
       set l K.sp_td_step_j0 !j0;
       set l K.sp_td_step_nbk nbk;
       set l K.sp_td_step_i i;
       set l K.sp_td_step_per tp;
       (* W's column p = i - 1 finished, then column i's update by the panel's i
          reflectors and its reflector. Each step over a reflector is two
          multiply-adds and overlapped reads. W's column takes a SIMD-group's
          sums over every 32nd row, a reflector a SIMD-group, then each thread's
          rows corrected by p reflectors; the column, each thread's rows updated
          by i. Each part passes over its rows at four dependent reads a row. *)
       let len = n - !j0 and p = Int.max 0 (i - 1) in
       let step = (2 * mac f) + overlapped_links in
       let passes = 4 * Plan.read_links in
       let rows = cdiv len threads in
       let w =
         if i = 0 then 0
         else
           (cdiv p (threads / 32) * cdiv len 32 * step)
           + (rows * ((p * step) + passes))
       in
       let column = if i = nbk then 0 else rows * ((i * step) + passes) in
       let trees = (6 * tree_links) + (3 * div_links) + (8 * barrier_links) in
       each c k l ~first:K.sp_td_step_first ~last:K.sp_td_step_count ~nb
         ~shared:K.sp_td_lds
         ~chain:(w + column + trees)
         ~bytes:(4 * len * tnb * t));
      let col = !j0 + i in
      if i < nbk && col < n - 1 then begin
        let k = K.Sp_td_hemv in
        let l = words k in
        address l K.sp_td_hemv_ws tw ~write:true;
        set l K.sp_td_hemv_n n;
        set l K.sp_td_hemv_c col;
        set l K.sp_td_hemv_i i;
        set l K.sp_td_hemv_parts f.parts;
        set l K.sp_td_hemv_per tp;
        (* A SIMD-group a row: its sum over the row's elements, a lane every
           32nd, then the lanes' tree. *)
        let rows = n - col - 1 in
        steps c k l ~first:K.sp_td_hemv_first ~last:K.sp_td_hemv_count
          ~count:(nb * cdiv rows 8)
          ~per:1 ~shared:0
          (busy ~threads
             ~bytes:(8 * rows * t)
             ~chain:((cdiv rows 32 * 2 * mac f) + 5 + Plan.read_links))
      end
    done;
    let r0 = !j0 + nbk in
    let mt = n - r0 in
    (* A22 −= V Wᴴ + W Vᴴ, through row-major views of the column-major
       matrices. *)
    let at off = { at = base; off; s = st } in
    let a22 = at (r0 + (r0 * n)) and wv = at (nn + r0) in
    let vv = at (r0 + (!j0 * n)) in
    gemm c ~nb ~y:a22 ~a:wv ~m:vv ~rows:mt ~cols:mt ~k:nbk ~lda:n ~ldb:n ~ldc:n
      ~opa:2 ~opb:0 ~init:true;
    gemm c ~nb ~y:a22 ~a:vv ~m:wv ~rows:mt ~cols:mt ~k:nbk ~lda:n ~ldb:n ~ldc:n
      ~opa:2 ~opb:0 ~init:true;
    j0 := !j0 + tnb
  done;
  let k = K.Sp_td_diag in
  let l = words k in
  address l K.sp_td_diag_ws tw ~write:true;
  set l K.sp_td_diag_n n;
  set l K.sp_td_diag_parts f.parts;
  set l K.sp_td_diag_per tp;
  elementwise c k l ~first:K.sp_td_diag_first ~last:K.sp_td_diag_count
    ~count:(nb * n) ~bytes:(2 * t) ~links:4

(* w and v from [d], [ds] floats a matrix, and Z: td_out, a thread an element of
   v. *)
let td_out c ~tw ~tp ~d ~ds ~dcx ~z ~w ~v ~nb ~n =
  let k = K.Sp_td_out in
  let l = words k in
  address l K.sp_td_out_ws tw ~write:false;
  address l K.sp_td_out_d d ~write:false;
  Option.iter
    (fun (x, es) ->
      address l K.sp_td_out_dcx x ~write:false;
      set l K.sp_td_out_es es)
    dcx;
  Option.iter (fun z -> address l K.sp_td_out_z z ~write:false) z;
  address l K.sp_td_out_w w ~write:true;
  Option.iter (fun v -> address l K.sp_td_out_v v ~write:true) v;
  set l K.sp_td_out_ds ds;
  set l K.sp_td_out_n n;
  set l K.sp_td_out_parts c.f.parts;
  set l K.sp_td_out_per tp;
  elementwise c k l ~first:K.sp_td_out_first ~last:K.sp_td_out_count
    ~count:(nb * n * n)
    ~bytes:(3 * c.f.t) ~links:8

(* Divide and conquer (xSTEDC) on the [nb] tridiagonals at [dsrc] and [esrc],
   [dstride] floats a matrix, into the workspace from [dw], [dp] bytes a matrix:
   the leaves torn and solved by Jacobi, then the merges level by level. *)
let divide c ~dw ~dp ~dsrc ~esrc ~dstride ~nb ~n =
  let leaf = K.sp_leaf and threads = K.sp_threads in
  let nn = n * n in
  let nl = cdiv n leaf in
  let last = n - ((nl - 1) * leaf) in
  let nfull = nb * (nl - 1) in
  let lf =
    ws c
      ((((nfull * leaf * leaf) + (nb * last * last)) * 4 * 2)
      + (((nfull * leaf) + (nb * last)) * 4))
  in
  let lt = plus lf (nfull * leaf * leaf * 4) in
  let vf = plus lt (nb * last * last * 4) in
  let vt = plus vf (nfull * leaf * leaf * 4) in
  let wf = plus vt (nb * last * last * 4) in
  let wt = plus wf (nfull * leaf * 4) in
  (let k = K.Sp_dc_leaves in
   let l = words k in
   address l K.sp_dc_leaves_ws dw ~write:true;
   address l K.sp_dc_leaves_d dsrc ~write:false;
   address l K.sp_dc_leaves_e esrc ~write:false;
   address l K.sp_dc_leaves_full lf ~write:true;
   address l K.sp_dc_leaves_tail lt ~write:true;
   set l K.sp_dc_leaves_stride dstride;
   set l K.sp_dc_leaves_n n;
   set l K.sp_dc_leaves_per dp;
   (* Thread 0's scan for the largest magnitude, then the leaves' n LEAF
      elements spread over the group. *)
   each c k l ~first:K.sp_dc_leaves_first ~last:K.sp_dc_leaves_count ~nb
     ~shared:K.sp_dc_lds
     ~chain:((n * 8) + (cdiv (n * leaf) threads * 8) + (3 * barrier_links))
     ~bytes:(4 * n * leaf * 4));
  let rc = { c with f = real } in
  eigh_jacobi rc ~a:lf ~w:wf ~v:(Some vf) ~nb:nfull ~n:leaf;
  eigh_jacobi rc ~a:lt ~w:wt ~v:(Some vt) ~nb ~n:last;
  (let k = K.Sp_dc_scatter in
   let l = words k in
   address l K.sp_dc_scatter_ws dw ~write:true;
   address l K.sp_dc_scatter_wf wf ~write:false;
   address l K.sp_dc_scatter_vf vf ~write:false;
   address l K.sp_dc_scatter_wt wt ~write:false;
   address l K.sp_dc_scatter_vt vt ~write:false;
   set l K.sp_dc_scatter_n n;
   set l K.sp_dc_scatter_per dp;
   elementwise c k l ~first:K.sp_dc_scatter_first ~last:K.sp_dc_scatter_count
     ~count:(nb * nn) ~bytes:12 ~links:8);
  let lev = ref 1 in
  while leaf lsl (!lev - 1) < n do
    let span = leaf lsl !lev in
    let big = Int.min span n in
    let merges = nb * cdiv n span in
    let merge ?(units = 1) ?(rows = 1) k =
      let l = words k in
      address l K.sp_merge_ws dw ~write:true;
      set l K.sp_merge_n n;
      set l K.sp_merge_lev !lev;
      set l K.sp_merge_per dp;
      set l K.sp_merge_du (Plan.divisor units);
      set l K.sp_merge_dr (Plan.divisor rows);
      l
    in
    (* The merges a matrix, and a merge's elements' units: n columns of [big]
       rows. *)
    let per_matrix = cdiv n span and elements = n * big in
    let first = K.sp_merge_first and last = K.sp_merge_count in
    (* Deflation: a merge's z, merged order and tol, passes over its N values;
       its decisions, a value a step; a row's rotations, at most N; a set-aside
       value's rank among at most N; a set-aside column's element. *)
    elementwise c K.Sp_dc_prepare
      (merge ~units:per_matrix K.Sp_dc_prepare)
      ~first ~last ~count:merges
      ~bytes:(5 * big * 4)
      ~links:(divided ~count:merges ~divides:1 (24 * big));
    (let k = K.Sp_dc_deflate in
     let l = words k in
     address l K.sp_dc_deflate_ws dw ~write:true;
     set l K.sp_dc_deflate_n n;
     set l K.sp_dc_deflate_lev !lev;
     set l K.sp_dc_deflate_per dp;
     set l K.sp_dc_deflate_du (Plan.divisor per_matrix);
     let per = K.threads k and state = K.sp_secular * 4 in
     resumed c k l ~first:K.sp_dc_deflate_first ~last:K.sp_dc_deflate_count
       ~count:merges ~per ~shared:0 ~q:K.sp_dc_deflate_q ~step:60
       ~total:(big + 1) (fun g chain ->
         cost
           ~bytes:
             ((g * per * ((chain / 60 * 24) + (2 * state))) + (2 * Plan.line))
           ~ops:(g * per * chain)
           ~chain:(chain + (2 * Plan.read_links))));
    elementwise c K.Sp_dc_rotate (merge K.Sp_dc_rotate) ~first ~last
      ~count:(nb * n)
      ~bytes:(4 * big * 4)
      ~links:(big * 8);
    elementwise c K.Sp_dc_rank (merge K.Sp_dc_rank) ~first ~last ~count:(nb * n)
      ~bytes:(big * 4) ~links:(big * 4);
    elementwise c K.Sp_dc_gather
      (merge ~units:elements ~rows:big K.Sp_dc_gather)
      ~first ~last ~count:(nb * elements) ~bytes:8
      ~links:(divided ~count:(nb * elements) ~divides:2 4);
    for phase = 0 to 2 do
      let k = K.Sp_dc_secular in
      let l = words k in
      address l K.sp_dc_secular_ws dw ~write:true;
      set l K.sp_dc_secular_n n;
      set l K.sp_dc_secular_lev !lev;
      set l K.sp_dc_secular_phase phase;
      set l K.sp_dc_secular_steps c.lim.secular;
      set l K.sp_dc_secular_per dp;
      let first = K.sp_dc_secular_first and last = K.sp_dc_secular_count in
      if phase = 0 then begin
        (* A root's start and each of its secular steps: a sum or two over the
           merge's values, and its state read and saved. *)
        let per = K.threads k and state = K.sp_secular * 4 in
        resumed c k l ~first ~last ~count:(nb * n) ~per ~shared:0
          ~q:K.sp_dc_secular_q ~start:K.sp_dc_secular_start ~step:(big * 20)
          ~total:(c.lim.secular + 1) (fun g chain ->
            cost
              ~bytes:
                ((g * per * ((chain / 20 * 8) + (2 * state))) + (2 * Plan.line))
              ~ops:(g * per * chain)
              ~chain:(chain + (2 * Plan.read_links)))
      end
      else
        (* A root's weight and vector, a pass or two over the values. *)
        elementwise c k l ~first ~last ~count:(nb * n)
          ~bytes:(2 * big * 4)
          ~links:(big * (div_links + 4))
    done;
    (* Tiles of 64 x 64 of Q's rows by the roots, along the grid's x and y, a
       merge along z: a tile's 16 x 16 threads each 16 products a term. A launch
       takes whole merges where one fits, else a merge's rows of tiles, else its
       tiles of a row. *)
    (let k = K.Sp_dc_product in
     let l = merge k in
     let tiles = cdiv big 64 in
     let refs = Array.of_list (List.rev l.refs) in
     let shared = K.sp_dc_product_lds in
     let resident = resident c k shared in
     let cost groups =
       busy ~threads
         ~bytes:(2 * 64 * big * 4)
         ~chain:((big * 16 * 2) + (cdiv big 16 * 2 * barrier_links))
         groups
     in
     let fits (x, y, z) =
       Plan.seconds ~threads ~groups:(x, y, z) ~resident (cost (x * y * z))
       <= Plan.allowed ()
     in
     let launch ~tx ~ty ~z (x, y, g) =
       let w = copy_words l in
       set w K.sp_merge_tx tx;
       set w K.sp_merge_ty ty;
       set w K.sp_merge_first z;
       set w K.sp_merge_count (z + g);
       Plan.launch c.b k ~groups:(x, y, g) ~shared ~resident
         ~cost:(cost (x * y * g))
         w.w refs
     in
     if fits (tiles, tiles, 1) then begin
       let most = most ~fits:(fun g -> fits (tiles, tiles, g)) merges in
       let z = ref 0 in
       while !z < merges do
         let g = Int.min most (merges - !z) in
         launch ~tx:0 ~ty:0 ~z:!z (tiles, tiles, g);
         z := !z + g
       done
     end
     else begin
       let xs, ys =
         match Plan.most ~fits:(fun y -> fits (tiles, y, 1)) tiles with
         | Some ys -> (tiles, ys)
         | None -> (most ~fits:(fun x -> fits (x, 1, 1)) tiles, 1)
       in
       for z = 0 to merges - 1 do
         for by = 0 to cdiv tiles ys - 1 do
           for bx = 0 to cdiv tiles xs - 1 do
             let tx = bx * xs and ty = by * ys in
             launch ~tx ~ty ~z
               (Int.min xs (tiles - tx), Int.min ys (tiles - ty), 1)
           done
         done
       done
     end);
    (* A merge's order of its N values; an element of Q placed. *)
    elementwise c K.Sp_dc_sort
      (merge ~units:per_matrix K.Sp_dc_sort)
      ~first ~last ~count:merges
      ~bytes:(5 * big * 4)
      ~links:(divided ~count:merges ~divides:1 (big * 12));
    elementwise c K.Sp_dc_place
      (merge ~units:elements ~rows:big K.Sp_dc_place)
      ~first ~last ~count:(nb * elements) ~bytes:12
      ~links:(divided ~count:(nb * elements) ~divides:2 (4 + Plan.read_links));
    incr lev
  done

(* v = Q Z, the reduction's panels' block reflectors applied to Z, the last
   panel first (xUNMTR's xLARFB). *)
let back c ~tw ~tp ~z ~nb ~n =
  let f = c.f and t = c.f.t and nn = n * n and tnb = K.sp_tnb in
  let st = tp / t and base = snd tw and zb = snd z in
  let w2 = ws c (nb * tnb * n * t) in
  let vco = nn + (n * tnb) in
  let tto = vco + (n * tnb) in
  let go = tto + (tnb * tnb) in
  let at off = { at = base; off; s = st } in
  let j0 = ref ((n - 1) / tnb * tnb) in
  while !j0 >= 0 do
    let nbk = Int.min tnb (n - !j0) and m = n - !j0 - 1 in
    if m > 0 then begin
      (let k = K.Sp_td_vcopy in
       let l = words k in
       address l K.sp_td_vcopy_ws tw ~write:true;
       set l K.sp_td_vcopy_n n;
       set l K.sp_td_vcopy_j0 !j0;
       set l K.sp_td_vcopy_nbk nbk;
       set l K.sp_td_vcopy_parts f.parts;
       set l K.sp_td_vcopy_per tp;
       elementwise c k l ~first:K.sp_td_vcopy_first ~last:K.sp_td_vcopy_count
         ~count:(nb * m * nbk)
         ~bytes:(2 * t) ~links:4);
      let vc = at vco and tm = at tto and g = at go in
      gemm c ~nb ~y:g ~a:vc ~m:vc ~rows:nbk ~cols:nbk ~k:m ~lda:nbk ~ldb:nbk
        ~ldc:nbk ~opa:2 ~opb:0 ~init:false;
      (let k = K.Sp_larft in
       let l = words k in
       address l K.sp_larft_tm (plus tw (tto * t)) ~write:true;
       address l K.sp_larft_g (plus tw (go * t)) ~write:false;
       address l K.sp_larft_tau
         (plus tw ((go + (tnb * tnb) + !j0) * t))
         ~write:false;
       set l K.sp_larft_nbk nbk;
       set l K.sp_larft_parts f.parts;
       set l K.sp_larft_per tp;
       (* Column i's thread 0 sums i products, each a multiply-add and
          overlapped reads, then the group meets twice. *)
       each c k l ~first:K.sp_larft_first ~last:K.sp_larft_count ~nb ~shared:0
         ~chain:
           ((nbk * (nbk - 1) / 2 * (mac f + overlapped_links))
           + (nbk * 2 * barrier_links))
         ~bytes:(3 * tnb * tnb * t));
      let wv = at nn and zs = { at = zb; off = !j0 + 1; s = nn } in
      let w2 = { at = snd w2; off = 0; s = tnb * n } in
      gemm c ~nb ~y:wv ~a:vc ~m:zs ~rows:nbk ~cols:n ~k:m ~lda:nbk ~ldb:n ~ldc:n
        ~opa:2 ~opb:1 ~init:false;
      gemm c ~nb ~y:w2 ~a:tm ~m:wv ~rows:nbk ~cols:n ~k:nbk ~lda:tnb ~ldb:n
        ~ldc:n ~opa:0 ~opb:0 ~init:false;
      gemm c ~nb ~y:zs ~a:w2 ~m:vc ~rows:n ~cols:m ~k:nbk ~lda:n ~ldb:nbk ~ldc:n
        ~opa:1 ~opb:1 ~init:true
    end;
    j0 := !j0 - tnb
  done

(* w and, where given, v of the [nb] n x n Hermitian matrices at [a], past
   Jacobi's order: the tridiagonal reduction, then for w alone bisection on T,
   else divide and conquer and the back-transformation. *)
let eigh_large c ~a ~w ~v ~nb ~n =
  let tp = td_per c.f n in
  let tw = ws c (nb * tp) in
  tridiagonal c ~a ~tw ~tp ~nb ~n;
  let nn = n * n and tnb = K.sp_tnb in
  let dstride = tp / 4 in
  let dsrc = plus tw ((nn + (2 * n * tnb) + (2 * tnb * tnb) + n) * c.f.t) in
  let esrc = plus dsrc (n * 4) in
  match v with
  | None ->
      let wt = ws c (nb * n * 4) in
      tri_bisect c ~d:dsrc ~e:esrc ~stride:dstride ~w:wt ~nb ~n ~i0:0 ~cnt:n;
      td_out c ~tw ~tp ~d:wt ~ds:n ~dcx:None ~z:None ~w ~v:None ~nb ~n
  | Some v ->
      let dp = dc_per n in
      let dw = ws c (nb * dp) in
      divide c ~dw ~dp ~dsrc ~esrc ~dstride ~nb ~n;
      let z = ws c (nb * nn * c.f.t) in
      (let k = K.Sp_td_zinit in
       let l = words k in
       address l K.sp_td_zinit_q dw ~write:false;
       address l K.sp_td_zinit_z z ~write:true;
       set l K.sp_td_zinit_qs (dp / 4);
       set l K.sp_td_zinit_n n;
       set l K.sp_td_zinit_parts c.f.parts;
       elementwise c k l ~first:K.sp_td_zinit_first ~last:K.sp_td_zinit_count
         ~count:(nb * nn) ~bytes:(2 * c.f.t) ~links:4);
      back c ~tw ~tp ~z ~nb ~n;
      let dd = plus dw (3 * nn * 4) in
      let fx = plus dw ((((3 * nn) + (10 * n)) * 4) + (5 * n * 4)) in
      td_out c ~tw ~tp ~d:dd ~ds:(dp / 4)
        ~dcx:(Some (fx, dp / 4))
        ~z:(Some z) ~w ~v:(Some v) ~nb ~n

(* svd *)

(* A large svd's workspace a matrix, as byte offsets: its arrays of elements (g,
   x, y, vc, wv, tauq, taup), of reals (d, e, U_B, V_B, 4 a rotation record,
   scalars) and of ints (sweeps, state); [cap] the records, and the sweeps, a
   pass holds; [pt] a column's partials past a chunk, of max(2, K − 1) elements
   a chunk, where M passes one. g is M x K, M >= K. For the values alone, x, y,
   vc, U_B, V_B, the records and the sweeps take no byte. *)
type bd = {
  g : int;
  x : int;
  y : int;
  vc : int;
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
  pt : int;
  per : int;
  cap : int;
}

(* The rows of a chunk (kernels.h's NX_SP_BD_CHUNK). *)
let chunk = K.sp_bd_chunk

(* The partials' elements a matrix: a sum and a maximum a chunk of x, or a sum a
   chunk of each of K − 1 columns, where M passes a chunk. *)
let bd_partials ~mm ~k =
  if mm > chunk then Int.max 2 (k - 1) * cdiv mm chunk else 0

let bd_values_layout ~mm ~k ~t =
  let o = ref 0 in
  let field n w =
    let at = !o in
    o := !o + (n * w);
    at
  in
  let g = field (mm * k) t in
  let wv = field (Int.max mm k) t in
  let tauq = field k t in
  let taup = field k t in
  let d = field k 4 in
  let e = field k 4 in
  let pt = field (bd_partials ~mm ~k) t in
  let st = field K.sp_st_words 4 in
  let none = !o in
  {
    g;
    x = none;
    y = none;
    vc = none;
    wv;
    tauq;
    taup;
    d;
    e;
    ub = none;
    vb = none;
    rec_ = none;
    sc = none;
    sw = none;
    st;
    pt;
    per = r256 !o;
    cap = 0;
  }

(* [l]'s words from word [at] on, as kernels.h's NX_SP_BDL orders them, each set
   in place: a list of the pairs would allocate 108 words a step. *)
let bd_words w ~at l =
  set w (at + K.sp_bdl_per) l.per;
  set w (at + K.sp_bdl_g) l.g;
  set w (at + K.sp_bdl_x) l.x;
  set w (at + K.sp_bdl_y) l.y;
  set w (at + K.sp_bdl_vc) l.vc;
  set w (at + K.sp_bdl_wv) l.wv;
  set w (at + K.sp_bdl_tauq) l.tauq;
  set w (at + K.sp_bdl_taup) l.taup;
  set w (at + K.sp_bdl_d) l.d;
  set w (at + K.sp_bdl_e) l.e;
  set w (at + K.sp_bdl_ub) l.ub;
  set w (at + K.sp_bdl_vb) l.vb;
  set w (at + K.sp_bdl_rec) l.rec_;
  set w (at + K.sp_bdl_sc) l.sc;
  set w (at + K.sp_bdl_sw) l.sw;
  set w (at + K.sp_bdl_st) l.st;
  set w (at + K.sp_bdl_cap) l.cap;
  set w (at + K.sp_bdl_pt) l.pt

(* A bidiagonal step's words: the workspace, M, K and the column [col] where
   [col] is given, the format's parts, and the layout. *)
let bd_col c k ~wsp ~l ~mm ~kk ~col =
  let w = words k in
  address w K.sp_bd_col_ws wsp ~write:true;
  set w K.sp_bd_col_m mm;
  set w K.sp_bd_col_k kk;
  set w K.sp_bd_col_c col;
  set w K.sp_bd_col_parts c.f.parts;
  bd_words w ~at:K.sp_bd_col_layout l;
  w

(* Bidiagonalisation (xGEBD2) of the [nb] m x n matrices at [a] into the
   workspace at [wsp]: g, M x K, a or aᴴ, reduced to Q B Pᴴ, B's diagonal and
   superdiagonal into d and e. *)
let bidiagonal c ~a ~wsp ~l ~nb ~m ~n =
  let f = c.f and t = c.f.t in
  let mm = Int.max m n and kk = Int.min m n in
  let threads = K.sp_threads in
  let shared = K.sp_shared in
  let group_step k w ~links ~bytes =
    steps c k w ~first:K.sp_bd_col_first ~last:K.sp_bd_col_count ~count:nb
      ~per:1 ~shared
      (busy ~threads ~bytes ~chain:links)
  in
  (let k = K.Sp_bd_stage in
   let w = words k in
   address w K.sp_bd_stage_a a ~write:false;
   address w K.sp_bd_stage_ws wsp ~write:true;
   set w K.sp_bd_stage_m mm;
   set w K.sp_bd_stage_k kk;
   set w K.sp_bd_stage_rows m;
   set w K.sp_bd_stage_cols n;
   set w K.sp_bd_stage_parts f.parts;
   bd_words w ~at:K.sp_bd_stage_layout l;
   set w K.sp_bd_stage_dm (Plan.divisor mm);
   let phase p ~units =
     let w = copy_words w in
     set w K.sp_bd_stage_phase p;
     set w K.sp_bd_stage_du (Plan.divisor units);
     w
   in
   let first = K.sp_bd_stage_first and last = K.sp_bd_stage_count in
   let run = K.sp_scan_run in
   let runs = cdiv (m * n * f.parts) run in
   (* A matrix's scan words cleared; a run of a's floats scanned; an element
      staged. *)
   elementwise c k (phase 0 ~units:1) ~first ~last ~count:nb ~bytes:8 ~links:2;
   elementwise c k (phase 1 ~units:runs) ~first ~last ~count:(nb * runs)
     ~bytes:(run * 4)
     ~links:(divided ~count:(nb * runs) ~divides:1 (run * 4));
   elementwise c k
     (phase 2 ~units:(m * n))
     ~first ~last
     ~count:(nb * m * n)
     ~bytes:(4 * t)
     ~links:(divided ~count:(nb * m * n) ~divides:2 8));
  (* A step of column [col] past a chunk, a group a unit of [count]: [phase],
     the column's [chunks] and the divisors of them and of the columns past
     [col]. *)
  let pass = cdiv chunk threads in
  let chunked k ~col ~phase ~chunks ~count ~bytes ~links =
    let w = words k in
    address w K.sp_bd_chunk_ws wsp ~write:true;
    set w K.sp_bd_chunk_m mm;
    set w K.sp_bd_chunk_k kk;
    set w K.sp_bd_chunk_c col;
    set w K.sp_bd_chunk_parts f.parts;
    set w K.sp_bd_chunk_phase phase;
    set w K.sp_bd_chunk_chunks chunks;
    set w K.sp_bd_chunk_dchunks (Plan.divisor chunks);
    set w K.sp_bd_chunk_dcols (Plan.divisor (kk - col - 1));
    bd_words w ~at:K.sp_bd_chunk_layout l;
    steps c k w ~first:K.sp_bd_chunk_first ~last:K.sp_bd_chunk_count ~count
      ~per:1 ~shared
      (busy ~threads ~bytes ~chain:links)
  in
  for col = 0 to kk - 1 do
    let rows = mm - col and cols = kk - col - 1 in
    (* A reflector over [len] elements: its norm's two reductions, a division
       and square roots, its scaling. *)
    let reflector len =
      (cdiv len threads * 3 * mac f)
      + (2 * tree_links) + (3 * div_links) + (2 * barrier_links)
    in
    if rows - 1 > chunk then begin
      (* x's partials and their close, again scaled where the sum left the safe
         range, then x scaled: the norm's passes, the close's passes over the
         chunks and the scalar part's roots and divisions. *)
      let q = cdiv (rows - 1) chunk in
      let close = (cdiv q threads * 2) + (2 * tree_links) + (6 * div_links) in
      let units = nb * q in
      let at = divided ~count:units ~divides:1 0 in
      chunked K.Sp_bd_lsum ~col ~phase:0 ~chunks:q ~count:units
        ~bytes:(chunk * t)
        ~links:((pass * (mac f + 2)) + (2 * tree_links) + at);
      chunked K.Sp_bd_lfin ~col ~phase:0 ~chunks:q ~count:nb ~bytes:(8 * q)
        ~links:close;
      chunked K.Sp_bd_lsum ~col ~phase:1 ~chunks:q ~count:units
        ~bytes:(chunk * t)
        ~links:((pass * (mac f + (div_links * f.parts))) + tree_links + at);
      chunked K.Sp_bd_lfin ~col ~phase:1 ~chunks:q ~count:nb ~bytes:(4 * q)
        ~links:close;
      chunked K.Sp_bd_lscale ~col ~phase:0 ~chunks:q ~count:units
        ~bytes:(2 * chunk * t)
        ~links:((pass * ((div_links * f.parts) + mac f)) + at)
    end
    else begin
      let k = pick f K.(Sp_bd_left_f32, Sp_bd_left_c64) in
      group_step k
        (bd_col c k ~wsp ~l ~mm ~kk ~col)
        ~links:(reflector rows)
        ~bytes:(2 * rows * t)
    end;
    if cols > 0 && rows > chunk then begin
      (* w's partials, a chunk of a column each, then each column's close. *)
      let q = cdiv rows chunk in
      let units = nb * cols * q in
      chunked K.Sp_bd_whc ~col ~phase:0 ~chunks:q ~count:units
        ~bytes:(2 * chunk * t)
        ~links:
          ((pass * mac f) + (2 * tree_links) + divided ~count:units ~divides:2 0);
      chunked K.Sp_bd_whfin ~col ~phase:0 ~chunks:q ~count:(nb * cols)
        ~bytes:(q * t)
        ~links:
          ((cdiv q threads * mac f)
          + (2 * tree_links)
          + divided ~count:(nb * cols) ~divides:1 0)
    end;
    if cols > 0 then begin
      if rows <= chunk then begin
        let w = bd_col c K.Sp_bd_wh ~wsp ~l ~mm ~kk ~col in
        steps c K.Sp_bd_wh w ~first:K.sp_bd_col_first ~last:K.sp_bd_col_count
          ~count:(nb * cols) ~per:1 ~shared
          (busy ~threads
             ~bytes:(2 * rows * t)
             ~chain:((cdiv rows threads * mac f) + (2 * tree_links)))
      end;
      let w = bd_col c K.Sp_bd_updl ~wsp ~l ~mm ~kk ~col in
      elementwise c K.Sp_bd_updl w ~first:K.sp_bd_col_first
        ~last:K.sp_bd_col_count
        ~count:(nb * rows * cols)
        ~bytes:(4 * t)
        ~links:(2 * mac f);
      let k = pick f K.(Sp_bd_right_f32, Sp_bd_right_c64) in
      group_step k
        (bd_col c k ~wsp ~l ~mm ~kk ~col)
        ~links:(reflector cols + cdiv cols threads + barrier_links)
        ~bytes:(4 * cols * Int.max t (Plan.line / 4));
      let rest = rows - 1 in
      if rest > 0 then begin
        (* A thread's row: its sum over the row's [cols] elements, each of its
           own line, which the group's neighbouring rows share. *)
        let w = bd_col c K.Sp_bd_wn ~wsp ~l ~mm ~kk ~col in
        elementwise c K.Sp_bd_wn w ~first:K.sp_bd_col_first
          ~last:K.sp_bd_col_count ~count:(nb * rest)
          ~bytes:((cols * t) + t)
          ~links:(cols * mac f);
        let w = bd_col c K.Sp_bd_updr ~wsp ~l ~mm ~kk ~col in
        elementwise c K.Sp_bd_updr w ~first:K.sp_bd_col_first
          ~last:K.sp_bd_col_count
          ~count:(nb * rest * cols)
          ~bytes:(4 * t)
          ~links:(2 * mac f)
      end
    end
  done

(* s alone of the [nb] m x n matrices at [a], past Jacobi's extents:
   bidiagonalisation, then bisection on the Golub-Kahan matrix, whose
   eigenvalues are ±σ. *)
let svd_values c ~a ~s ~nb ~m ~n =
  let mm = Int.max m n and kk = Int.min m n in
  let l = bd_values_layout ~mm ~k:kk ~t:c.f.t in
  let wsp = ws c (nb * l.per) in
  bidiagonal c ~a ~wsp ~l ~nb ~m ~n;
  let dt = ws c ((nb * 2 * kk * 4 * 2) + (nb * kk * 4)) in
  let et = plus dt (nb * 2 * kk * 4) in
  let lam = plus et (nb * 2 * kk * 4) in
  (let k = K.Sp_bd_tgk in
   let w = words k in
   address w K.sp_bd_tgk_ws wsp ~write:false;
   address w K.sp_bd_tgk_dt dt ~write:true;
   address w K.sp_bd_tgk_et et ~write:true;
   set w K.sp_bd_tgk_k kk;
   bd_words w ~at:K.sp_bd_tgk_layout l;
   elementwise c k w ~first:K.sp_bd_tgk_first ~last:K.sp_bd_tgk_count
     ~count:(nb * 2 * kk)
     ~bytes:12 ~links:4);
  tri_bisect c ~d:dt ~e:et ~stride:(2 * kk) ~w:lam ~nb ~n:(2 * kk) ~i0:kk
    ~cnt:kk;
  let k = K.Sp_bd_values in
  let w = words k in
  address w K.sp_bd_values_ws wsp ~write:false;
  address w K.sp_bd_values_lam lam ~write:false;
  address w K.sp_bd_values_s s ~write:true;
  set w K.sp_bd_values_k kk;
  bd_words w ~at:K.sp_bd_values_layout l;
  elementwise c k w ~first:K.sp_bd_values_first ~last:K.sp_bd_values_count
    ~count:(nb * kk) ~bytes:8 ~links:8

(* eig *)

(* A matrix's workspace (sp_eig.h): H, Z and X of n² elements, a complex scratch
   of n², τ, four arrays of n reals, four words and the iteration's state. *)
let eig_per f n =
  r256
    ((3 * n * n * f.t)
    + (n * n * 8)
    + (n * f.t)
    + (5 * n * 4)
    + ((4 + K.sp_eig_state) * 4))

(* eig's words, every step's: [a], [w] and [ws] where given. *)
let eig_words c k ~a ~w ~wsp ~n ~per ~vectors =
  let l = words k in
  Option.iter (fun a -> address l K.sp_eig_a a ~write:false) a;
  Option.iter (fun w -> address l K.sp_eig_w w ~write:true) w;
  address l K.sp_eig_ws wsp ~write:true;
  set l K.sp_eig_n n;
  set l K.sp_eig_vectors (Bool.to_int vectors);
  set l K.sp_eig_itq c.lim.qr;
  set l K.sp_eig_parts c.f.parts;
  set l K.sp_eig_per per;
  set l K.sp_eig_passes c.lim.passes;
  l

(* A step of a group a matrix of [nb] matrices, each running [chain] links and
   touching [bytes]. *)
let eig_step c k l ~nb ~shared ~chain ~bytes =
  steps c k l ~first:K.sp_eig_first ~last:K.sp_eig_count ~count:nb ~per:1
    ~shared
    (busy ~threads:K.sp_half ~bytes ~chain)

(* The links of a group's pass over [len] elements, a thread's share [ops] links
   each. *)
let pass len ~ops = cdiv len K.sp_half * ops

(* A resumed step of a group a matrix of [nb] matrices, each step at most [step]
   links and touching [bytes]. *)
let eig_resumed c k l ~nb ~shared ~bytes ~step ~total =
  resumed c k l ~first:K.sp_eig_first ~last:K.sp_eig_count ~count:nb ~per:1
    ~shared ~q:K.sp_eig_q ~step ~total (fun g chain ->
      busy ~threads:K.sp_half ~bytes ~chain g)

(* A step of a group a row or column of each of the [nb] matrices. *)
let column_step c k l ~nb ~n ~shared ~chain ~bytes =
  steps c k l ~first:K.sp_eig_first ~last:K.sp_eig_count ~count:(nb * n) ~per:1
    ~shared
    (busy ~threads:K.sp_half ~bytes ~chain)

(* xGEBAL's scaling of a row brings f into range a factor of 2 at a time, at
   most once over each of the 2 · 103 exponents its guards allow. *)
let scale_links = 2 * 4 * 206

(* Load and balance: a into H, flushed; isolation, at most n + 1 steps of rows
   and n + 1 of columns, each O(n) with the counts; at most the limit's passes
   of scaling, n + 1 steps each; the scaling into the safe range; τ zeroed and,
   for the vectors, Z = I. *)
let balance c ~a ~wsp ~nb ~n ~per ~vectors =
  let f = c.f in
  let words k a = eig_words c k ~a ~w:None ~wsp ~n ~per ~vectors in
  let small = K.sp_half * 4 and two = 2 * K.sp_half * 4 in
  (let k = pick f K.(Sp_eig_in_f32, Sp_eig_in_c64) in
   column_step c k (words k (Some a)) ~nb ~n ~shared:small
     ~chain:(pass n ~ops:4 + tree_links)
     ~bytes:(2 * n * f.t));
  (let k = pick f K.(Sp_eig_count_f32, Sp_eig_count_c64) in
   elementwise c k (words k None) ~first:K.sp_eig_first ~last:K.sp_eig_count
     ~count:(nb * n) ~bytes:(n * f.t) ~links:(2 * n));
  let iso cols =
    let k = pick f K.(Sp_eig_iso_f32, Sp_eig_iso_c64) in
    let l = words k None in
    set l K.sp_eig_from (Bool.to_int cols);
    eig_resumed c k l ~nb ~shared:small
      ~bytes:(4 * n * f.t)
      ~step:
        (pass n ~ops:(8 + (2 * mac f)) + (2 * tree_links) + (6 * barrier_links))
      ~total:(n + 1)
  in
  iso false;
  (let k = pick f K.(Sp_eig_colcnt_f32, Sp_eig_colcnt_c64) in
   elementwise c k (words k None) ~first:K.sp_eig_first ~last:K.sp_eig_count
     ~count:(nb * n) ~bytes:(n * f.t) ~links:(2 * n));
  iso true;
  (let k = pick f K.(Sp_eig_rows_f32, Sp_eig_rows_c64) in
   let reduce = pass n ~ops:(2 * mac f) + tree_links in
   eig_resumed c k (words k None) ~nb ~shared:two
     ~bytes:(4 * n * f.t)
     ~step:
       ((6 * reduce)
       + (2 * pass n ~ops:(2 * mac f))
       + scale_links + (4 * barrier_links))
     ~total:(c.lim.passes * (n + 1)));
  (let k = pick f K.(Sp_eig_big_f32, Sp_eig_big_c64) in
   column_step c k (words k None) ~nb ~n ~shared:small
     ~chain:(pass n ~ops:2 + tree_links)
     ~bytes:(n * f.t));
  let k = pick f K.(Sp_eig_scale_f32, Sp_eig_scale_c64) in
  column_step c k (words k None) ~nb ~n ~shared:small
    ~chain:(pass n ~ops:6 + tree_links + (2 * barrier_links))
    ~bytes:((2 + Bool.to_int vectors) * n * f.t)

(* Hessenberg's column [i] of [n]: its reflector, then its applications from the
   right to the rows above [hi] and from the left to the columns past [i]. *)
let hess_links c ~n i =
  let m = n - i - 1 in
  pass m ~ops:(3 * mac c.f)
  + (2 * tree_links) + (3 * div_links) + (4 * barrier_links)
  + (2 * pass n ~ops:(2 * m * mac c.f))
  + (2 * pass m ~ops:(2 * m * mac c.f))

(* A QR sweep over an active block of [n] rows at most: its bulge chased down,
   each step a reflector of three and its application to H's rows and columns
   and, for the vectors, Z's rows; then the next deflation's scan. At the
   budget's 1.25·10⁶ links of a lane, the values' sweep passes near order 3,400
   at float32 and 1,700 at complex64, the frontier nx_metal.mli states. *)
let sweep_links c ~n ~vectors =
  let rows = if vectors then 3 else 2 in
  let step =
    givens_links + (2 * barrier_links) + (rows * pass n ~ops:(3 * 2 * mac c.f))
  in
  (n * step) + (n * 4)

(* The columns of each launch of a step over columns [lo, hi), [links i] column
   i's: consecutive ranges, each as long as one wave of groups fits the
   budget. *)
let column_ranges c k ~shared ~nb ~lo ~hi links =
  let wave = Int.min nb (resident c k shared * Plan.cores) in
  let fits chain =
    fits c k ~shared ~groups:wave
      (busy ~threads:(K.threads k) ~bytes:0 ~chain wave)
  in
  let rec go from acc =
    if from >= hi then List.rev acc
    else begin
      if not (fits (links from)) then raise Decline;
      let rec extend to_ chain =
        if to_ < hi && fits (chain + links to_) then
          extend (to_ + 1) (chain + links to_)
        else (to_, chain)
      in
      let to_, chain = extend (from + 1) (links from) in
      go to_ ((from, to_, chain) :: acc)
    end
  in
  go lo []

(* w alone of the [nb] n x n matrices at [a]: one group a matrix, from the load
   to w, the QR iteration in slices of [q] sweeps, as many as the limit's 30
   max(10, n) sweeps need. *)
let eig_values c ~a ~w ~nb ~n =
  let f = c.f in
  let per = eig_per f n in
  let wsp = ws c (nb * per) in
  let nn = n * n in
  let step_words k =
    eig_words c k ~a:None ~w:None ~wsp ~n ~per ~vectors:false
  in
  balance c ~a ~wsp ~nb ~n ~per ~vectors:false;
  (let k = pick f K.(Sp_eig_hess_f32, Sp_eig_hess_c64) in
   let shared = K.sp_half * 4 in
   List.iter
     (fun (from, to_, chain) ->
       let l = step_words k in
       set l K.sp_eig_from from;
       set l K.sp_eig_to to_;
       eig_step c k l ~nb ~shared ~chain ~bytes:(4 * n * (to_ - from) * f.t))
     (column_ranges c k ~shared ~nb ~lo:0 ~hi:(n - 1) (hess_links c ~n)));
  (let k = pick f K.(Sp_eig_qr_f32, Sp_eig_qr_c64) in
   let sweep = sweep_links c ~n ~vectors:false in
   let wave = Int.min nb (resident c k 0 * Plan.cores) in
   let bytes = 4 * nn * f.t in
   (* The estimate the launch's steps make of one wave, so that a slice that
      fits here fits there. *)
   let fits q =
     fits c k ~shared:0 ~groups:wave
       (busy ~threads:K.sp_half ~bytes ~chain:(q * sweep) wave)
   in
   let limit = c.lim.qr * Int.max 10 n in
   let q = most ~fits (Int.max 1 limit) in
   let q = Int.min q c.lim.steps in
   for _ = 1 to Int.max 1 (cdiv limit q) do
     let l = step_words k in
     set l K.sp_eig_q q;
     eig_step c k l ~nb ~shared:0 ~chain:(q * sweep) ~bytes
   done);
  let k = pick f K.(Sp_eig_w_f32, Sp_eig_w_c64) in
  let l = eig_words c k ~a:None ~w:(Some w) ~wsp ~n ~per ~vectors:false in
  eig_step c k l ~nb ~shared:0
    ~chain:(pass n ~ops:8 + (3 * barrier_links))
    ~bytes:(4 * n * 8)

(* Calls *)

let batch_of (x : O.operand) =
  let l = x.layout in
  let n = ref 1 in
  for i = 0 to L.rank l - 3 do
    n := !n * L.dim l i
  done;
  !n

let dims (x : O.operand) =
  let l = x.layout in
  let r = L.rank l in
  (L.dim l (r - 2), L.dim l (r - 1))

(* [x]'s [nb] matrices of [rows] x [cols] as a C-contiguous place: [x] itself
   where it is C-contiguous, else its copy in the workspace by the
   contraction's, which flushes subnormal elements as the routines read them. An
   [x] whose batch axes do not merge into one strided axis declines the call. *)
let source c (x : O.operand) ~nb ~rows ~cols =
  let l = x.layout in
  if L.is_contiguous l then of_array c x
  else begin
    let r = L.rank l in
    let stride = ref 0 and runs = ref 0 in
    for i = 0 to r - 3 do
      let e = L.dim l i in
      if e > 1 then begin
        let s = L.stride l i in
        if !runs = 0 || !stride <> s * e then incr runs;
        stride := s
      end
    done;
    if !runs > 1 then raise Decline;
    let shape = Iarray.of_array [| nb; rows; cols |] in
    let strides =
      Iarray.of_array [| !stride; L.stride l (r - 2); L.stride l (r - 1) |]
    in
    let src = L.v ~offset:(L.offset l) ~strides shape in
    let dst =
      L.v ~offset:0 ~strides:(Iarray.of_array [| rows * cols; cols; 1 |]) shape
    in
    let ((_, at) as place) = ws c (nb * rows * cols * c.f.t) in
    Contract.copy c.b
      ~src:(Contract.operand x.buffer x.dtype src)
      ~dst:(Contract.work ~at x.dtype dst);
    place
  end

(* Plans [body c] of the operand [x]'s format, or declines a format the family
   has no arithmetic for, or a call one of whose units passes the budget. *)
let routine ~holds (x : O.operand) body =
  match fmt_of x with
  | None -> Error A.Declined
  | Some f -> (
      let c = { b = Plan.make (); f; lim = Atomic.get limits; holds } in
      match body c with
      | () -> Ok (Plan.finish_some ~most:8 c.b)
      | exception (Decline | Contract.Decline) -> Error A.Declined)

(* Each routine's operands: its destinations, then its load. *)

let eigh ~holds (xs : O.t) =
  let x = xs.loads.(0) and w = xs.dsts.(0) in
  let v = if Array.length xs.dsts > 1 then Some xs.dsts.(1) else None in
  let nb = batch_of x and n, _ = dims x in
  if nb = 0 || n = 0 then Ok None
  else
    routine ~holds x @@ fun c ->
    let a = source c x ~nb ~rows:n ~cols:n in
    let w = of_array c w and v = Option.map (of_array c) v in
    if n <= K.sp_jacobi then eigh_jacobi c ~a ~w ~v ~nb ~n
    else eigh_large c ~a ~w ~v ~nb ~n

let svd_of_facts ~holds ~s ?vectors x =
  let nb = batch_of x and m, n = dims x in
  let k = Int.min m n in
  if nb = 0 then Ok None
  else
    let uc, vr =
      match vectors with
      | None -> (0, 0)
      | Some (u, vh) -> (snd (dims u), fst (dims vh))
    in
    (* Where k = 0, s holds nothing and the factors are identities: the call
       names neither a nor s, so that its submission binds only what its
       launches use. *)
    routine ~holds x @@ fun c ->
    if k = 0 then
      Option.iter
        (fun (u, vh) ->
          eye c u ~nb ~r:m ~cols:uc;
          eye c vh ~nb ~r:vr ~cols:n)
        vectors
    else if m <= K.sp_jacobi && n <= K.sp_jacobi then
      svd_jacobi c
        ~a:(source c x ~nb ~rows:m ~cols:n)
        ~s:(of_array c s)
        ~vectors:
          (Option.map (fun (u, vh) -> (of_array c u, of_array c vh)) vectors)
        ~nb ~m ~n ~uc ~vr
    else if Option.is_some vectors then raise Decline
    else
      svd_values c
        ~a:(source c x ~nb ~rows:m ~cols:n)
        ~s:(of_array c s) ~nb ~m ~n

let svd ~holds (xs : O.t) =
  let vectors =
    if Array.length xs.dsts > 1 then Some (xs.dsts.(1), xs.dsts.(2)) else None
  in
  svd_of_facts ~holds ~s:xs.dsts.(0) ?vectors xs.loads.(0)

let eig ~holds (xs : O.t) =
  let x = xs.loads.(0) and w = xs.dsts.(0) in
  let nb = batch_of x and n, _ = dims x in
  if nb = 0 || n = 0 then Ok None
  else if Array.length xs.dsts > 1 then Error A.Declined
  else
    routine ~holds x @@ fun c ->
    eig_values c ~a:(source c x ~nb ~rows:n ~cols:n) ~w:(of_array c w) ~nb ~n
