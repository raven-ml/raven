(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The transforms' launches of fft.cu.

   A transform runs along one axis at a time, the last of [axes] first,
   over lines: the points along that axis at one index of the others. The
   first axis reads the operand and writes [dst]; later axes run in place
   on [dst]. A line's FFT of N points runs as Stockham steps (kernels.h),
   each a launch of fft:

     N fits a group      one step: x -> dst
     longer              a step per factor of N, balanced, each fitting a
                         group: x -> dst -> dst for two (the last step runs
                         in place), through the workspace for more or where
                         dst cannot hold the line's points

   A line of a 7-smooth n is such an FFT of n points. A prime n whose
   n - 1 is 7-smooth and fits a group runs Rader's convolution, two FFTs
   of n - 1 points in one launch. Any other n runs Bluestein's convolution
   of a 7-smooth m >= 2n - 1 points: both FFTs in one launch where m fits
   a group, else two stages of steps, the chirp and the kernel's product
   fused into their first load and last store. A real line of even n is a
   complex line of n / 2 points, split into bins after it (r2c) or made
   from the bins before it (c2r), in its launch where it fits a group,
   else by a launch of fft_real; a real line of odd n is the complex line
   of its n points, completed from its bins by a c2r's load.

   Error. Let u be the unit roundoff. A butterfly of radix R, as fft.cu
   computes it, errs by at most η(R) u ‖z‖₂ √R, with η(2) = 1,
   η(3) = 4.17, η(4) = 2.13, η(5) = 5.63, η(7) = 7.40, η(8) = 3.97 and
   η(16) = 7.63: each rounding adds δ t (|δ| <= u) to its operation's
   value t and each stored constant one more, so the error is
   C diag(δ) A z and its norm at most u ‖C‖₂ ‖A‖₂ ‖z‖₂, C and A the maps
   after and before the rounding. A twiddle from a table rounded once costs
   3 (u for the table, 2u for the product's two fused multiply-adds,
   Jeannerod et al., "Further analysis of Kahan's algorithm", Math. Comp.
   2017), one from two tables 6. Each pass is a scaled unitary map, so the
   passes' relative errors add: a 7-smooth line's FFT errs by
   e(n) u √n ‖x‖₂, e(n) the sum over its passes of η(R), 3 for each pass
   after a step's first and 6 for each step after the first. A pass costs
   at most 4.52 per bit of n (radix 3 with its twiddle), so
   e(n) <= 4.6 log₂ n + 6 (q - 1) for q steps, and [factors] takes
   q <= ⌈log₂ n / 8⌉: its factors are each within 7 of the least, which is
   at most n^(1/q), so q = ⌈log₂ n / 8⌉ factors are each at most 1792, and
   a factor of at most 2048 points fits a group of either width: a pass
   of radix r gives each thread ⌊PM / r⌋ butterflies, r ⌊PM / r⌋ >= 5
   of its PM points for every radix [inner] takes (14 at complex64), so
   tl <= ⌈2048 / 5⌉ = 410 <= fft_threads. Hence
   e(n) <= 5.35 log₂ n.

   Rader: the permuted points' FFT A errs by e(m) u √m ‖x‖ (m = n - 1);
   the kernel's |B'_k| <= √n / m (a Gauss sum) and its product 3u; the
   second FFT e(m), so the convolution errs by √n u ‖x‖ (2 e(m) + 3), and
   y_0's sum and the additions of x_0 add e(m) + 1: e(n) = 3 e(n - 1) + 4.
   Bluestein: the chirp's product 3u, the FFT e(m), the kernel's product
   with |B'| <= K √n / m and 3u, the second FFT e(m) and the last chirp's
   product 3u on ‖y‖ = √n ‖x‖: e(n) = K (2 e(m) + 6) + 3, K the kernel's
   max |B_k| / √n. B_k = Σ_(|j|<n) e(f(j)), e(t) = e^(2πit),
   f(t) = t² / (2n) - t k / m, a sum whose f' = t / n - k / m is monotone
   and spans less than 2. Take δ = √(4 / (3πn)): f' passes within δ of at
   most 3 integers, at fewer than 2δn + 1 points each, and the other
   points form at most 4 runs on each of which ‖f'‖ >= δ. f' grows by
   1/n < 2δ from one integer to the next, so it cannot skip a band
   |f' - N| < δ between two adjacent points of a run. Each difference
   f(j+1) - f(j) lies between f'(j) and f'(j+1), so on a run the
   differences increase within one [N + δ, N + 1 - δ], and the run's sum
   is at most cot(πδ/2) <= 2 / (πδ) (Landau's bound, Mordell, "On the
   Kusmin-Landau inequality for exponential sums", Acta Arith. 4 (1958),
   eq. (3)); δ < 1/2 for n >= 2, and n = 1 is a single term. So
   |B_k| <= 6δn + 3 + 8 / (πδ) = 7.82 √n + 3, and
   with |B_k| <= 2n - 1, K <= 8.5 for every n: e(n) <= 17 e(m) + 54 and,
   with m < 4n, e(n) <= 91 log₂ n + 236. A real line's split adds 11 to e(n/2)
   and its pre-pass 19.5, by the same tracking over the angles of their
   twiddles, relative to the completed bins' norm, at most √2 times the
   bins'. A c2r's other axes run first on the operand's bins, whatever
   the last axis's completion drops, and their errors pass through the
   last axis's c2r, which multiplies norms by at most √(2 n): measured
   against √2 times the operand's norm, each axis counts its e(n). *)

module A = Nx_array
module L = Nx_array.Layout
module D = Nx_array.Dtype
module K = Kernels
module Run = Rig.Submission.Run

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external roots : bool -> int -> int -> int -> int -> bigbytes
  = "nx_cuda_fft_roots"

external chirp_table : bool -> int -> bigbytes = "nx_cuda_fft_chirp"

external bluestein_table : bool -> int -> int -> bigbytes
  = "nx_cuda_fft_bluestein"

external rader_tables : bool -> int -> bigbytes * bigbytes = "nx_cuda_fft_rader"

(* Widths *)

(* The bytes of a complex point, the points a thread holds in a pass
   (fft.cu's PM), and the shift of shared memory's padding (PS). *)
let cplx f64 = if f64 then 16 else 8
let per_thread f64 = if f64 then 8 else 16
let pad_shift f64 = if f64 then 3 else 4

(* The most points of a line's FFT in one group, and of a long line's
   step. A complex128 step keeps to 2048 points, 36 KiB of shared memory,
   so that an SM holds two groups and overlaps one's loads with the
   other's arithmetic. *)
let block_points f64 = if f64 then 4096 else 8192
let step_points f64 = if f64 then 2048 else 8192

(* Shared memory: what a group holds of a step whose groups may hold
   several lines, and of a complex64 step whose lines lie side by side,
   which takes more where fewer would leave its accesses in pieces of
   less than a 32-byte sector: two lines of 4096 points, four of 2048. *)
let group_bytes = 48 * 1024
let pair_bytes = 98 * 1024

(* The inner FFT *)

(* A line of [l] points in a group: its passes' radices, 16s (8s at
   complex128) for its power of two and one smaller power for the rest,
   then 7s, 5s and 3s, and [tl] threads, so that each pass's butterflies
   fit a thread's points. *)
type inner = { l : int; radices : int array; tl : int }

let inner ~f64 l =
  let big, bits = if f64 then (8, 3) else (16, 4) in
  let rec strip n p acc =
    if n mod p = 0 then strip (n / p) p (p :: acc) else (n, acc)
  in
  let rest, twos = strip l 2 [] in
  let rest, sevens = strip rest 7 [] in
  let rest, fives = strip rest 5 [] in
  let _, threes = strip rest 3 [] in
  let a = List.length twos in
  let small = if a mod bits > 0 then [ 1 lsl (a mod bits) ] else [] in
  let radices =
    Array.of_list
      (List.init (a / bits) (fun _ -> big) @ small @ sevens @ fives @ threes)
  in
  let pm = per_thread f64 in
  let tl =
    Array.fold_left
      (fun t r ->
        let per = pm / r in
        Int.max t (((l / r) + per - 1) / per))
      1 radices
  in
  { l; radices; tl }

let smooth n =
  let rec strip n p = if n mod p = 0 then strip (n / p) p else n in
  n >= 1 && strip (strip (strip (strip n 2) 3) 5) 7 = 1

(* A 7-smooth line of [l] points fits a group of [most]. *)
let fits ~f64 l most = l <= most && (inner ~f64 l).tl <= K.fft_threads

(* The steps of a long line of [n] 7-smooth points: the fewest factors that
   each fit a step, each prime given to the least factor, largest first. *)
let factors ~f64 n =
  let rec strip n p acc =
    if n mod p = 0 then strip (n / p) p (p :: acc) else (n, acc)
  in
  let n', sevens = strip n 7 [] in
  let n', fives = strip n' 5 [] in
  let n', threes = strip n' 3 [] in
  let _, twos = strip n' 2 [] in
  let primes = sevens @ fives @ threes @ twos in
  let rec from q =
    let f = Array.make q 1 in
    List.iter
      (fun p ->
        let least = ref 0 in
        for i = 1 to q - 1 do
          if f.(i) < f.(!least) then least := i
        done;
        f.(!least) <- f.(!least) * p)
      primes;
    if Array.for_all (fun x -> fits ~f64 x (step_points f64)) f then begin
      Array.sort (fun a b -> Int.compare b a) f;
      Array.map (inner ~f64) f
    end
    else from (q + 1)
  in
  from 1

(* Plans *)

(* How a line's FFT runs: its own points, Rader's convolution of n - 1, or
   Bluestein's of m. *)
type how = Smooth | Rader | Bluestein of int

(* A line's plan: how it runs, its FFT's points [l1], its one step where
   it fits a group or else its steps, and its tables in [buffer] by their
   bytes: each inner FFT's by its points, a long FFT's w_l1^m by [g_bits],
   a real line's w_2n^k by [r_bits], and Bluestein's or Rader's. *)
type plan = {
  how : how;
  l1 : int;
  block : inner option;
  steps : inner array;
  tw : (int * int) list;
  glo : int;
  ghi : int;
  g_bits : int;
  rlo : int;
  rhi : int;
  r_bits : int;
  chirp : int;
  bspec : int;
  rlog : int;
  buffer : Rig.Buffer.t;
}

let is_prime n =
  let rec from d = d * d > n || (n mod d <> 0 && from (d + 1)) in
  n >= 2 && from 2

let rader_ok ~f64 n =
  n > 7
  && n < 1 lsl 31
  && is_prime n
  && smooth (n - 1)
  && fits ~f64 (n - 1) (block_points f64)

(* The steps a 7-smooth FFT of [l] points runs. *)
let line_steps ~f64 l =
  if fits ~f64 l (block_points f64) then [| inner ~f64 l |] else factors ~f64 l

(* A relative cost of an FFT of [m] points: its passes' operations a
   point. *)
let cost ~f64 m =
  let radix = function
    | 2 -> 4.
    | 3 -> 7.3
    | 4 -> 7.
    | 5 -> 10.4
    | 7 -> 12.8
    | 8 -> 10.
    | _ -> 13.3
  in
  Array.fold_left
    (fun c i -> Array.fold_left (fun c r -> c +. radix r) c i.radices)
    0. (line_steps ~f64 m)
  *. Float.of_int m

(* The 7-smooth numbers from [lo] to [hi]. *)
let smooth_between lo hi =
  let rec from x primes acc =
    match primes with
    | [] -> if x >= lo then x :: acc else acc
    | p :: rest ->
        let rec powers x acc =
          if x > hi then acc else powers (x * p) (from x rest acc)
        in
        powers x acc
  in
  from 1 [ 2; 3; 5; 7 ] []

(* Bluestein's length for [n] and its kernel's table: the cheapest 7-smooth
   m from 2n - 1 to 2 (2n - 1), the least of equal cost. *)
let bluestein ~f64 n =
  let lo = (2 * n) - 1 in
  let cheapest (m, c) m' =
    let c' = cost ~f64 m' in
    if c' < c then (m', c') else (m, c)
  in
  let m, _ =
    List.fold_left cheapest (0, infinity)
      (List.sort Int.compare (smooth_between lo (2 * lo)))
  in
  (m, bluestein_table f64 n m)

(* w_n^m = lo[m mod 2^bits] hi[m >> bits]: the two tables and [bits]. *)
let two_tables ~f64 n =
  let rec log2 b = if 1 lsl b < n then log2 (b + 1) else b in
  let bits = (log2 0 + 1) / 2 in
  let nl = 1 lsl bits in
  (roots f64 n 0 1 nl, roots f64 n 0 nl ((n + nl - 1) / nl), bits)

let make_plan device n ~f64 ~real =
  let tables = ref [] and size = ref 0 in
  let add (b : bigbytes) =
    let at = (!size + 255) land lnot 255 in
    tables := (at, b) :: !tables;
    size := at + Bigarray.Array1.dim b;
    at
  in
  let how, l1, chirp, bspec, rlog =
    if smooth n then (Smooth, n, 0, 0, 0)
    else if (not real) && rader_ok ~f64 n then begin
      let kernel, logs = rader_tables f64 n in
      (Rader, n - 1, 0, add kernel, add logs)
    end
    else begin
      let m, kernel = bluestein ~f64 n in
      let chirp = add (chirp_table f64 n) in
      (Bluestein m, m, chirp, add kernel, 0)
    end
  in
  let block =
    if fits ~f64 l1 (block_points f64) then Some (inner ~f64 l1) else None
  in
  let steps = match block with Some _ -> [||] | None -> factors ~f64 l1 in
  let ls =
    List.sort_uniq Int.compare
      (List.map (fun i -> i.l) (Option.to_list block @ Array.to_list steps))
  in
  let tw =
    List.map
      (fun l -> (l, add (roots f64 l 0 1 (if l mod 2 = 0 then l / 2 else l))))
      ls
  in
  let glo, ghi, g_bits =
    if steps = [||] then (0, 0, 0)
    else
      let lo, hi, bits = two_tables ~f64 l1 in
      (add lo, add hi, bits)
  in
  let rlo, rhi, r_bits =
    if not real then (0, 0, 0)
    else
      let lo, hi, bits = two_tables ~f64 (2 * n) in
      (add lo, add hi, bits)
  in
  let bytes = Bigarray.Array1.create Int8_unsigned C_layout !size in
  List.iter
    (fun (at, b) ->
      let len = Bigarray.Array1.dim b in
      Bigarray.Array1.blit b (Bigarray.Array1.sub bytes at len))
    !tables;
  let buffer = Rig.Buffer.create device !size in
  Rig.Buffer.copy ~src:(Rig.Buffer.of_bigarray bytes) ~dst:buffer;
  {
    how;
    l1;
    block;
    steps;
    tw;
    glo;
    ghi;
    g_bits;
    rlo;
    rhi;
    r_bits;
    chirp;
    bspec;
    rlog;
    buffer;
  }

(* A device's plans, by points, precision and whether of a halved real
   line, the [kept_plans] most recently made whose buffers take at most
   [kept_bytes] together, newest first. A plan past [kept_bytes], such as
   a long Bluestein line's chirp and kernel, is made at each call. A
   transform holds its plans' buffers in the work it submits, so a plan
   dropped here lives until that work is done. *)
let kept_plans = 16
let kept_bytes = 64 * 1024 * 1024
let plan_bytes p = Rig.Buffer.length p.buffer

type cache = { device : Rig.t; plans : (int * bool * bool, plan) Cache.t }

let cache device = { device; plans = Cache.make () }

let plan c n ~f64 ~real =
  let key = (n, f64, real) in
  match Cache.find c.plans key with
  | Some p -> p
  | None ->
      Cache.keep ~count:kept_plans ~bytes:(kept_bytes, plan_bytes) c.plans key
        (make_plan c.device n ~f64 ~real)

(* Lines *)

(* Where an array lies: its buffer, or the workspace, from a byte. *)
type mem = Mem of Rig.Buffer.t | Workspace
type where = { mem : mem; byte : int }

(* An array: where, its first element, its extents and its strides, in
   elements. *)
type arr = { where : where; off : int; dims : int array; strides : int array }

let arr_of (type v s) (x : (v, s) A.t) =
  let l = A.layout x in
  {
    where = { mem = Mem (A.buffer x); byte = 0 };
    off = L.offset l;
    dims = Iarray.to_array (L.shape l);
    strides = Iarray.to_array (L.strides l);
  }

let numel a = Array.fold_left ( * ) 1 a.dims

(* [a]'s shape C-contiguous in the workspace from [byte]. *)
let contiguous byte dims =
  let r = Array.length dims in
  let strides = Array.make r 1 in
  for i = r - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * Int.max dims.(i + 1) 1
  done;
  { where = { mem = Workspace; byte }; off = 0; dims; strides }

(* fft_side: a launch's lines on one side. *)
type side = {
  at : where;
  first : int;
  ps : int;
  dim : int array;
  str : int array;
}

(* The lines along [axis] of [x] and [y], arrays of one batch shape: their
   count and each side's, the batch axes merged where both sides allow
   and extents of 1 dropped. *)
let lines_of x y axis =
  let axes = ref [] in
  for i = Array.length x.dims - 1 downto 0 do
    if i <> axis && x.dims.(i) <> 1 then
      axes := (x.dims.(i), x.strides.(i), y.strides.(i)) :: !axes
  done;
  let merged =
    List.fold_right
      (fun (d, s1, s2) acc ->
        match acc with
        | (d', s1', s2') :: rest when s1 = s1' * d' && s2 = s2' * d' ->
            (d * d', s1', s2') :: rest
        | _ -> (d, s1, s2) :: acc)
      !axes []
  in
  let dim = Array.of_list (List.map (fun (d, _, _) -> d) merged) in
  let side a str =
    { at = a.where; first = a.off; ps = a.strides.(axis); dim; str }
  in
  ( Array.fold_left ( * ) 1 dim,
    side x (Array.of_list (List.map (fun (_, s, _) -> s) merged)),
    side y (Array.of_list (List.map (fun (_, _, s) -> s) merged)) )

(* [nl] contiguous lines of [n] points in the workspace from [byte]. *)
let ws_side byte nl n =
  {
    at = { mem = Workspace; byte };
    first = 0;
    ps = 1;
    dim = [| nl |];
    str = [| n |];
  }

let unit s = s = 1 || s = -1

(* Lines side by side: a strided axis whose innermost batch axis steps one
   element. *)
let side_by_side s =
  let r = Array.length s.dim in
  (not (unit s.ps)) && r > 0 && unit s.str.(r - 1)

(* Launches *)

(* A launch of fft (kernels.h's fft_params), but its tables, which its
   plan holds, and the ones it computes as it is written. *)
type step = {
  src : side;
  dst : side;
  inner : inner;
  nv : int;
  m : int;
  ns : int;
  nsrc : int;
  limit : int;
  gts : int;
  nh : int;
  lld : int;
  lst : int;
  lp : int;
  g : int;
  lf_ld : bool;
  lf_st : bool;
  ld : int;
  st : int;
  conj_in : bool;
  conj_out : bool;
  chirp_in : bool;
  post : int;
  ops : int array;
  rader : bool;
}

(* A launch's parameters: a step of fft, or a split or a merge of
   fft_real, with the plan whose tables it reads. *)
type params =
  | Step of step * plan
  | Real of {
      src : side;
      dst : side;
      nl : int;
      h : int;
      split : bool;
      plan : plan;
    }

(* A call's plan as it grows: its precision, its launches in reverse, the
   workspace's pieces they take and the buffers of the plans the cache did
   not keep. *)
type call = {
  c : cache;
  f64 : bool;
  mutable launches : params Launch.t list;
  mutable count : int;
  ws : Arena.t;
  mutable frees : Rig.Buffer.t list;
}

(* Each kernel's index in {!Kernels.kernels}: fft or fft_real, by width. *)
let fft_f32 = Launch.kernel "fft_f32"
let fft_f64 = Launch.kernel "fft_f64"
let real_f32 = Launch.kernel "fft_real_f32"
let real_f64 = Launch.kernel "fft_real_f64"

(* Each launch takes three slots from the one before's: the side it reads,
   the side it writes and its plan. A split reads and writes [dst]. *)
let slots = 3

let refs_of i o p =
  [| i + K.Fft_side.ptr; 0; 0; o + K.Fft_side.ptr; 1; 1; p; 2; 0 |]

let step_refs = K.Fft_params.(refs_of in_ out plan)
let real_refs = K.Fft_real_params.(refs_of in_ out plan)

(* Pushes a launch of [kernel] over [grid] groups of [threads] threads. *)
let push k ~kernel ~size ~refs ~grid ~threads ~shared params =
  k.launches <-
    {
      Launch.kernel;
      size;
      base = slots * k.count;
      refs;
      gx = grid;
      gy = 1;
      gz = 1;
      threads;
      shared;
      params;
    }
    :: k.launches;
  k.count <- k.count + 1

(* [f ()], whose workspace later launches take again. *)
let within k f =
  let m = Arena.mark k.ws in
  f ();
  Arena.release k.ws m

let max_grid = 0x7fffffff
let ceil_div a b = (a + b - 1) / b
let rec pow2_floor g x = if 2 * g <= x then pow2_floor (2 * g) x else g
let rec log2 g = if g <= 1 then 0 else 1 + log2 (g / 2)

(* The bytes before a group's lines: five words a line. *)
let header g = ((5 * g * 8) + 15) land lnot 15

(* A step's group: [g] lines of [inner.tl] threads, [lp] slots apart for
   lines of [lmax] points, the slot count odd. About 128 threads of whole
   warps, at least 64 bytes of lines side by side where a side reads them
   so, within [group_bytes], or a sector's within [pair_bytes]. *)
let shape ~f64 (s : step) lmax =
  let es = cplx f64 and tl = s.inner.tl in
  let lp = lmax + (lmax lsr pad_shift f64) + 1 in
  let lp = if lp mod 2 = 0 then lp + 1 else lp in
  let warps x = 32 * (((x * tl) + 31) / 32) in
  (* No more than a seventh of a group's lanes idle. *)
  let idle x = 7 * (warps x - (x * tl)) > warps x in
  let rec grow g =
    if idle g && 2 * g * tl <= K.fft_threads && 2 * g * lp * es <= group_bytes
    then grow (2 * g)
    else g
  in
  let g = grow (pow2_floor 1 (Int.max 1 (128 / tl))) in
  let lf = s.lf_ld || s.lf_st in
  let g = if lf then Int.max g (64 / es) else g in
  let rec fit g =
    if g > 1 && (g * lp * es > group_bytes || g * tl > K.fft_threads) then
      fit (g / 2)
    else g
  in
  let g = fit g in
  let rec sector g =
    if
      (not f64) && lf
      && g * es < 32
      && 2 * g * lp * es <= pair_bytes
      && 2 * g * tl <= K.fft_threads
    then sector (2 * g)
    else g
  in
  let g = sector g in
  let rec few g = if g > 1 && g / 2 >= s.nv then few (g / 2) else g in
  let g = few g in
  let lf_ld = s.lf_ld && g > 1 and lf_st = s.lf_st && g > 1 in
  { s with lp; g; lf_ld; lf_st }

let push_step k p (s : step) lmax =
  if s.nv > 0 then begin
    let s = shape ~f64:k.f64 s lmax in
    push k
      ~kernel:(if k.f64 then fft_f64 else fft_f32)
      ~size:K.Fft_params.size ~refs:step_refs
      ~grid:(Int.min (ceil_div s.nv s.g) max_grid)
      ~threads:(s.g * s.inner.tl)
      ~shared:(header s.g + (s.g * s.lp * cplx k.f64))
      (Step (s, p))
  end

let push_real k p ~src ~dst ~nl ~h ~split =
  if nl > 0 then
    push k
      ~kernel:(if k.f64 then real_f64 else real_f32)
      ~size:K.Fft_real_params.size ~refs:real_refs
      ~grid:(Int.max 1 (Int.min (ceil_div (nl * ((h / 2) + 1)) 256) max_grid))
      ~threads:256 ~shared:0
      (Real { src; dst; nl; h; split; plan = p })

(* The core: [nl] lines of [nc] complex points from [src] to [dst], the
   lines' first launch loading as [ld], their last storing as [st], below
   [limit]. [real] 1 splits an r2c's bins after, 2 makes a c2r's points
   before. [alias] reads [dst] ([src] is [dst]'s lines); [scratch] lets
   [dst] hold a long line's points between steps. *)
type core = {
  inv : bool;
  alias : bool;
  scratch : bool;
  nl : int;
  nc : int;
  limit : int;
  src : side;
  dst : side;
  ld : int;
  st : int;
  real : int;
}

let blank_step ~src ~dst inner =
  {
    src;
    dst;
    inner;
    nv = 0;
    m = 1;
    ns = 1;
    nsrc = 0;
    limit = 0;
    gts = 0;
    nh = 0;
    lld = 0;
    lst = 0;
    lp = 0;
    g = 1;
    lf_ld = false;
    lf_st = false;
    ld = K.fft_ld_c;
    st = K.fft_st_c;
    conj_in = false;
    conj_out = false;
    chirp_in = false;
    post = K.fft_post_none;
    ops = [| K.fft_op_fft |];
    rader = false;
  }

(* A line in one group: its FFT, with Rader's or Bluestein's steps and a
   real line's around it, in shared memory. *)
let in_block k p (c : core) inner =
  let n = c.nc and rader = p.how = Rader in
  let fft =
    match p.how with
    | Smooth -> [ K.fft_op_fft ]
    | Rader -> [ K.fft_op_fft; K.fft_op_rmul; K.fft_op_fft ]
    | Bluestein _ ->
        [
          K.fft_op_chirp;
          K.fft_op_fft;
          K.fft_op_bmul;
          K.fft_op_fft;
          K.fft_op_bfin;
        ]
  in
  let ops =
    (if c.real = 2 then [ K.fft_op_hpre ] else [])
    @ fft
    @ if c.real = 1 then [ K.fft_op_split ] else []
  in
  let s =
    {
      (blank_step ~src:c.src ~dst:c.dst inner) with
      nv = c.nl;
      nsrc = (if c.real = 2 then n + 1 else n);
      limit = c.limit;
      nh = n;
      lld = (if c.real = 2 then n + 1 else if rader then n else p.l1);
      lst = Int.min (if c.real = 1 then n + 1 else n) c.limit;
      lf_ld = side_by_side c.src;
      lf_st = side_by_side c.dst;
      ld = c.ld;
      st = c.st;
      conj_in = c.inv && c.real <> 2;
      conj_out = c.inv;
      ops = Array.of_list ops;
      rader;
    }
  in
  (* A Rader line holds x_0 and y_0 past its n - 1 points, a real one its
     bin n / 2. *)
  push_step k p s (Int.max p.l1 (n + 1) + if rader then 1 else 0)

(* A long line: a step per factor of its FFT. Buffers: 0 the operand, 1
   dst, from 2 the workspace's lines of l1 points, Bluestein's convolution
   in 2 and temporaries in 3 and 4, or temporaries in 2 and 3. *)
let long_lines k p (c : core) =
  let f64 = k.f64 and n = c.nc and nl = c.nl in
  let es = cplx f64 and smooth = p.how = Smooth in
  let src, first =
    if c.real = 2 then begin
      (* c2r's points from the bins, conjugated, into the workspace. *)
      let w = ws_side (Arena.take k.ws (nl * n * es)) nl n in
      push_real k p ~src:c.src ~dst:w ~nl ~h:n ~split:false;
      (w, (K.fft_ld_c, n, false, not smooth))
    end
    else (c.src, (c.ld, n, c.inv, not smooth))
  in
  let l1 = p.l1 and steps = p.steps in
  let q = Array.length steps and tn = nl * l1 * es in
  let t1 = if smooth then 2 else 3 in
  let used = ref 0 in
  (* The buffers a stage's steps write, from [i] to [o]: through [o] where
     it may hold a line's points ([scratch]), else the temporaries, never
     back into their own input; the last writes [o]. *)
  let assign i o scratch =
    let outs = Array.make q o and cur = ref i in
    for j = 0 to q - 2 do
      let b =
        List.find
          (fun b -> b <> !cur && b <> 0 && (b <> o || scratch))
          [ o; t1; t1 + 1 ]
      in
      outs.(j) <- b;
      cur := b
    done;
    Array.iter (fun b -> if b >= 2 then used := Int.max !used (b - 1)) outs;
    outs
  in
  let i = if c.alias then 1 else 0 in
  (* r2c stores the half-length transform, then splits it. *)
  let limit = if c.real = 1 then n else c.limit in
  let stages =
    if smooth then
      [
        (i, assign i 1 c.scratch, first, (c.st, K.fft_post_none, c.inv, limit));
      ]
    else
      let o1 = assign i 2 true in
      let o2 = assign 2 1 false in
      [
        (i, o1, first, (K.fft_st_c, K.fft_post_bmul, false, l1));
        ( 2,
          o2,
          (K.fft_ld_c, l1, false, false),
          (c.st, K.fft_post_bfin, c.inv, limit) );
      ]
  in
  let base = if !used > 0 then Arena.take k.ws (!used * tn) else 0 in
  let side_of b =
    if b = 0 then src
    else if b = 1 then c.dst
    else ws_side (base + ((b - 2) * tn)) nl l1
  in
  let stage (i, outs, (ld, nsrc, conj_in, chirp_in), (st, post, conj_out, limit))
      =
    let ns = ref 1 and from = ref i in
    Array.iteri
      (fun j (inner : inner) ->
        let fi = j = 0 and la = j = q - 1 and l = inner.l in
        let src = side_of !from and dst = side_of outs.(j) in
        let s =
          {
            (blank_step ~src ~dst inner) with
            nv = nl * (l1 / l);
            m = l1 / l;
            ns = !ns;
            nsrc = (if fi then nsrc else l1);
            limit = (if la then limit else l1);
            gts = (if fi then 0 else l1 / (!ns * l));
            nh = n;
            lld = l;
            lst = l;
            lf_ld = unit src.ps;
            lf_st = !ns > 1 && unit dst.ps;
            ld = (if fi then ld else K.fft_ld_c);
            st = (if la then st else K.fft_st_c);
            conj_in = fi && conj_in;
            conj_out = la && conj_out;
            chirp_in = fi && chirp_in;
            post = (if la then post else K.fft_post_none);
          }
        in
        push_step k p s l;
        from := outs.(j);
        ns := !ns * l)
      steps
  in
  List.iter stage stages;
  if c.real = 1 then push_real k p ~src:c.dst ~dst:c.dst ~nl ~h:n ~split:true

let core k (c : core) =
  let p = plan k.c c.nc ~f64:k.f64 ~real:(c.real > 0) in
  if plan_bytes p > kept_bytes then k.frees <- p.buffer :: k.frees;
  match p.block with
  | Some inner -> in_block k p c inner
  | None -> long_lines k p c

(* The entries *)

let f64_of (type v s) (x : (v, s) A.t) =
  match A.dtype x with D.Complex128 | D.Float64 -> true | _ -> false

let call c ~f64 =
  { c; f64; launches = []; count = 0; ws = Arena.make (); frees = [] }

let complex_core ~inv ~alias nl n src dst =
  {
    inv;
    alias;
    scratch = true;
    nl;
    nc = n;
    limit = n;
    src;
    dst;
    ld = K.fft_ld_c;
    st = K.fft_st_c;
    real = 0;
  }

(* The complex transform along [axes], the last first: from [x] into
   [dst], then in place, or in place throughout where [in_place]. *)
let c2c_axes k ~inv ~in_place axes dst x =
  if numel dst > 0 then
    for i = Array.length axes - 1 downto 0 do
      let a = axes.(i) and first = i = Array.length axes - 1 in
      let nl, src, d = lines_of (if first then x else dst) dst a in
      within k (fun () ->
          core k
            (complex_core ~inv ~alias:(in_place || not first) nl dst.dims.(a)
               src d))
    done

(* The call of [k]'s launches, the buffers of their slots in order, the
   workspace's where a side lies there. *)
let verdict k : params Launch.plan =
  match k.launches with
  | [] -> Ok []
  | l :: _ as ls ->
      let n = k.count in
      let launches = Array.make n l in
      let buffers = Array.make (slots * n) Launch.workspace in
      let of_where w =
        match w.mem with Mem b -> b | Workspace -> Launch.workspace
      in
      List.iteri
        (fun j (l : params Launch.t) ->
          let i = n - 1 - j in
          launches.(i) <- l;
          let src, dst, plan =
            match l.params with
            | Step (s, plan) -> (s.src, s.dst, plan)
            | Real { src; dst; plan; _ } -> (src, dst, plan)
          in
          buffers.(slots * i) <- of_where src.at;
          buffers.((slots * i) + 1) <- of_where dst.at;
          buffers.((slots * i) + 2) <- plan.buffer)
        ls;
      Ok
        [
          {
            Launch.launches;
            buffers;
            workspace = Arena.peak k.ws;
            tickets = 0;
            kept = true;
            frees = k.frees;
          };
        ]

let c2c c (d : Nx_kernel.direction) ~axes ~dst x =
  let k = call c ~f64:(f64_of x) in
  c2c_axes k ~inv:(d = Inverse) ~in_place:false (Iarray.to_array axes)
    (arr_of dst) (arr_of x);
  verdict k

let r2c c ~axes ~dst x =
  let k = call c ~f64:(f64_of x) in
  let axes = Iarray.to_array axes in
  let x = arr_of x and y = arr_of dst in
  let last = Array.length axes - 1 in
  let a = axes.(last) in
  let n = x.dims.(a) in
  if n > 0 && numel y > 0 then begin
    let nl, src, d = lines_of x y a in
    let h = n / 2 in
    let even = n mod 2 = 0 in
    within k (fun () ->
        core k
          {
            inv = false;
            alias = false;
            scratch = even;
            nl;
            nc = (if even then h else n);
            limit = h + 1;
            src;
            dst = d;
            ld = (if even then K.fft_ld_rp else K.fft_ld_r);
            st = K.fft_st_c;
            real = (if even then 1 else 0);
          });
    c2c_axes k ~inv:false ~in_place:true (Array.sub axes 0 last) y y
  end;
  verdict k

let c2r c ~axes ~dst x =
  let k = call c ~f64:(f64_of x) in
  let axes = Iarray.to_array axes in
  let x = arr_of x and y = arr_of dst in
  let last = Array.length axes - 1 in
  let a = axes.(last) in
  let n = y.dims.(a) in
  if numel y > 0 then begin
    (* The other axes first, into a C-contiguous copy of x's shape. *)
    let x =
      if last = 0 then x
      else
        let w = contiguous (Arena.take k.ws (numel x * cplx k.f64)) x.dims in
        c2c_axes k ~inv:true ~in_place:false (Array.sub axes 0 last) w x;
        w
    in
    let nl, src, d = lines_of x y a in
    let even = n mod 2 = 0 in
    within k (fun () ->
        core k
          {
            inv = true;
            alias = false;
            scratch = false;
            nl;
            nc = (if even then n / 2 else n);
            limit = (if even then n / 2 else n);
            src;
            dst = d;
            ld = (if even then K.fft_ld_c else K.fft_ld_hc);
            st = (if even then K.fft_st_rp else K.fft_st_re);
            real = (if even then 2 else 0);
          })
  end;
  verdict k

(* Parameters *)

let int64 r at off v = Run.int64 r at off (Int64.of_int v)
let int32 r at off v = Run.int32 r at off v
let flag b = Bool.to_int b

let write_side r at base (s : side) =
  let module S = K.Fft_side in
  int64 r at (base + S.ptr) s.at.byte;
  int64 r at (base + S.off) s.first;
  int64 r at (base + S.ps) s.ps;
  int32 r at (base + S.rank) (Array.length s.dim);
  Array.iteri (fun i d -> int64 r at (base + S.dim + (8 * i)) d) s.dim;
  Array.iteri (fun i d -> int64 r at (base + S.str + (8 * i)) d) s.str

(* The multiplier by which fft.cu divides by [d] < 2^16: 0 for 1. *)
let magic d = if d = 1 then 0 else ((1 lsl 32) / d) + 1

let write_step r at p (s : step) =
  let module P = K.Fft_params in
  write_side r at P.in_ s.src;
  write_side r at P.out s.dst;
  int64 r at P.plan 0;
  int64 r at P.nv s.nv;
  int64 r at P.m s.m;
  int64 r at P.ns s.ns;
  int64 r at P.nsrc s.nsrc;
  int64 r at P.limit s.limit;
  int64 r at P.gts s.gts;
  int64 r at P.nh s.nh;
  let l = s.inner.l in
  int64 r at P.tw (List.assoc l p.tw);
  int64 r at P.glo p.glo;
  int64 r at P.ghi p.ghi;
  int64 r at P.rlo p.rlo;
  int64 r at P.rhi p.rhi;
  int64 r at P.chirp p.chirp;
  int64 r at P.bspec p.bspec;
  int64 r at P.rlog p.rlog;
  int32 r at P.l l;
  int32 r at P.lld s.lld;
  int32 r at P.lst s.lst;
  int32 r at P.lp s.lp;
  int32 r at P.g s.g;
  int32 r at P.tl s.inner.tl;
  int32 r at P.lf_ld (flag s.lf_ld);
  int32 r at P.lf_st (flag s.lf_st);
  int32 r at P.ld s.ld;
  int32 r at P.st s.st;
  int32 r at P.conj_in (flag s.conj_in);
  int32 r at P.conj_out (flag s.conj_out);
  int32 r at P.chirp_in (flag s.chirp_in);
  int32 r at P.post s.post;
  int32 r at P.np (Array.length s.inner.radices);
  int32 r at P.half (flag (l mod 2 = 0));
  int32 r at P.nops (Array.length s.ops);
  int32 r at P.gshift (log2 s.g);
  int32 r at P.g_bits p.g_bits;
  int32 r at P.r_bits p.r_bits;
  int32 r at P.rader (flag s.rader);
  Array.iteri (fun i o -> int32 r at (P.ops + (4 * i)) o) s.ops;
  let ns = ref 1 in
  Array.iteri
    (fun i radix ->
      let module F = K.Fft_pass in
      let base = P.pass + (F.size * i) in
      int32 r at (base + F.r) radix;
      int32 r at (base + F.ns) !ns;
      int32 r at (base + F.q) (l / radix);
      int32 r at (base + F.ts) (l / (!ns * radix));
      int32 r at (base + F.nsm) (magic !ns);
      ns := !ns * radix)
    s.inner.radices

let write params r at =
  match params with
  | Step (st, plan) -> write_step r at plan st
  | Real { src; dst; nl; h; split; plan } ->
      let module P = K.Fft_real_params in
      write_side r at P.in_ src;
      write_side r at P.out dst;
      int64 r at P.plan 0;
      int64 r at P.nl nl;
      int64 r at P.h h;
      int64 r at P.rlo plan.rlo;
      int64 r at P.rhi plan.rhi;
      int32 r at P.r_bits plan.r_bits;
      int32 r at P.split (flag split)
