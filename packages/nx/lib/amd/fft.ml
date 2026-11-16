(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The transforms' launches of fft.hip.

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

   Error. Let u be the unit roundoff. A butterfly of radix R, as fft.hip
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
   a factor of at most 2048 points fits a step of either width
   ([step_points]): a pass of radix r gives each work-item ⌊PM / r⌋
   butterflies, r ⌊PM / r⌋ >= 5 of its PM points for every radix [inner]
   takes (14 at complex64), so tl <= ⌈2048 / 5⌉ = 410 <= fft_threads.
   Hence e(n) <= 5.35 log₂ n.

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

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external roots : bool -> int -> int -> int -> int -> bigbytes
  = "nx_amd_fft_roots"

external chirp_table : bool -> int -> bigbytes = "nx_amd_fft_chirp"

external bluestein_table : bool -> int -> int -> bigbytes
  = "nx_amd_fft_bluestein"

external rader_tables : bool -> int -> bigbytes * bigbytes = "nx_amd_fft_rader"

(* Widths *)

(* The bytes of a complex point, the points a work-item holds in a pass
   (fft.hip's PM), and the shift of LDS's padding (PS). *)
let cplx f64 = if f64 then 16 else 8
let per_thread f64 = if f64 then 8 else 16
let pad_shift f64 = if f64 then 3 else 4

(* The most points of a line's FFT in one group, and of a long line's
   step: 32 KiB of points and their padding, so that the longest line
   leaves a group's 64 KiB of LDS room for its words and a WGP's 128 KiB
   room for a second group. *)
let block_points f64 = if f64 then 2048 else 4096
let step_points = block_points

(* LDS: what a group holds of a step whose groups may hold several lines,
   48 KiB, a bound no timing has tuned; and of a complex64 step whose
   lines lie side by side, which takes up to the most a group of fft may
   ({!Plan.shared_most}, its copy of its words aside) where fewer lines
   would leave its accesses in pieces of less than 32 bytes. *)
let group_bytes = 48 * 1024
let pair_bytes = Plan.shared_most K.Fft_f32

(* The inner FFT *)

(* A line of [l] points in a group: its passes' radices, 16s (8s at
   complex128) for its power of two and one smaller power for the rest,
   then 7s, 5s and 3s, and [tl] work-items, so that each pass's
   butterflies fit a work-item's points. *)
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
   it fits a group or else its steps, and its tables, the constant [key]
   of [length] bytes that [bytes] makes, by their bytes: each inner FFT's
   by its points, a long FFT's w_l1^m by [g_bits], a real line's w_2n^k by
   [r_bits], and Bluestein's or Rader's. *)
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
  key : string;
  length : int;
  bytes : unit -> bigbytes;
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

(* Bluestein's length for [n]: the cheapest 7-smooth m from 2n - 1 to
   2 (2n - 1), the least of equal cost. *)
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
  m

(* A plan's tables as they are laid out: their bytes, and how to write
   each into the plan's bytes. *)
type tables = { mutable size : int; mutable fills : (bigbytes -> unit) list }

(* The offset of a table of [len] bytes, on 256 bytes past the others. *)
let reserve t len =
  let at = (t.size + 255) land lnot 255 in
  t.size <- at + len;
  at

let blit dst at (b : bigbytes) =
  Bigarray.Array1.blit b (Bigarray.Array1.sub dst at (Bigarray.Array1.dim b))

(* The offset of a table of [len] bytes that [make] makes. *)
let add t len make =
  let at = reserve t len in
  t.fills <- (fun dst -> blit dst at (make ())) :: t.fills;
  at

(* w_n^m = lo[m mod 2^bits] hi[m >> bits]: the two tables' offsets and
   [bits]. *)
let two_tables t ~f64 n =
  let rec log2 b = if 1 lsl b < n then log2 (b + 1) else b in
  let bits = (log2 0 + 1) / 2 in
  let nl = 1 lsl bits and nh = (n + (1 lsl bits) - 1) lsr bits in
  let lo = add t (nl * cplx f64) (fun () -> roots f64 n 0 1 nl) in
  let hi = add t (nh * cplx f64) (fun () -> roots f64 n 0 nl nh) in
  (lo, hi, bits)

let make_plan n ~f64 ~real =
  let t = { size = 0; fills = [] } in
  let how, l1, chirp, bspec, rlog =
    if smooth n then (Smooth, n, 0, 0, 0)
    else if (not real) && rader_ok ~f64 n then begin
      let kernel = reserve t ((n - 1) * cplx f64) and logs = reserve t (4 * n) in
      let fill dst =
        let k, l = rader_tables f64 n in
        blit dst kernel k;
        blit dst logs l
      in
      t.fills <- fill :: t.fills;
      (Rader, n - 1, 0, kernel, logs)
    end
    else begin
      let m = bluestein ~f64 n in
      let chirp = add t (n * cplx f64) (fun () -> chirp_table f64 n) in
      let bspec = add t (m * cplx f64) (fun () -> bluestein_table f64 n m) in
      (Bluestein m, m, chirp, bspec, 0)
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
      (fun l ->
        let count = if l mod 2 = 0 then l / 2 else l in
        (l, add t (count * cplx f64) (fun () -> roots f64 l 0 1 count)))
      ls
  in
  let glo, ghi, g_bits =
    if steps = [||] then (0, 0, 0) else two_tables t ~f64 l1
  in
  let rlo, rhi, r_bits =
    if not real then (0, 0, 0) else two_tables t ~f64 (2 * n)
  in
  let length = t.size and fills = t.fills in
  (* Every byte written, the gaps between tables zero, so that the bytes
     are a function of the key. *)
  let bytes () =
    let b = Bigarray.Array1.create Int8_unsigned C_layout length in
    Bigarray.Array1.fill b 0;
    List.iter (fun fill -> fill b) fills;
    b
  in
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
    key =
      Printf.sprintf "fft %d %s%s" n
        (if f64 then "complex128" else "complex64")
        (if real then " real" else "");
    length;
    bytes;
  }

(* The plans of the lines last planned, by points, precision and whether
   of a halved real line, the [kept_plans] most recent: a plan holds no
   table, which the door makes, but planning a Bluestein line weighs each
   length it may take. *)
let kept_plans = 64
let plans : (int * bool * bool, plan) Cache.t = Cache.make ()

let plan n ~f64 ~real =
  let key = (n, f64, real) in
  match Cache.find plans key with
  | Some p -> p
  | None -> Cache.keep ~count:kept_plans plans key (make_plan n ~f64 ~real)

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

(* A launch's lines on one side, over the launch's extents ([lines]):
   line l's first element at [first] + Σ idx_d str[d], its points [ps]
   apart. *)
type side = { at : where; first : int; ps : int; str : int array }

(* The lines along [axis] of [x] and [y], arrays of one batch shape: their
   extents, the batch axes merged where both sides allow and extents of 1
   dropped, and each side. *)
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
    { at = a.where; first = a.off; ps = a.strides.(axis); str }
  in
  ( dim,
    side x (Array.of_list (List.map (fun (_, s, _) -> s) merged)),
    side y (Array.of_list (List.map (fun (_, _, s) -> s) merged)) )

let count dim = Array.fold_left ( * ) 1 dim

(* Lines of [n] points in the workspace from [byte], one after the other
   in C order over [dim]. *)
let ws_side byte dim n =
  let r = Array.length dim in
  let str = Array.make r n in
  for i = r - 2 downto 0 do
    str.(i) <- str.(i + 1) * dim.(i + 1)
  done;
  { at = { mem = Workspace; byte }; first = 0; ps = 1; str }

let unit s = s = 1 || s = -1

(* Lines side by side: a strided axis whose innermost batch axis steps one
   element. *)
let side_by_side s =
  let r = Array.length s.str in
  (not (unit s.ps)) && r > 0 && unit s.str.(r - 1)

(* Launches *)

(* A launch of fft (kernels.h's fft words), but its tables, which its plan
   holds, and the ones it computes as it is written. *)
type step = {
  dim : int array;
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

(* A call's plan as it grows: its precision, and its launches and the
   workspace's pieces they take. *)
type call = { f64 : bool; b : Plan.builder }

let kernel k = if k.f64 then K.Fft_f64 else K.Fft_f32

let real_kernel k = if k.f64 then K.Fft_real_f64 else K.Fft_real_f32

(* [f ()], whose workspace later launches take again. *)
let within k f =
  let m = Plan.mark k.b in
  f ();
  Plan.release k.b m

let rec pow2_floor g x = if 2 * g <= x then pow2_floor (2 * g) x else g
let rec log2 g = if g <= 1 then 0 else 1 + log2 (g / 2)

(* The bytes before a group's lines: five words a line. *)
let header g = ((5 * g * 8) + 15) land lnot 15

(* A step's group: [g] lines of [inner.tl] work-items, [lp] slots apart for
   lines of [lmax] points, the slot count odd. About 128 work-items of
   whole waves, at least 64 bytes of lines side by side where a side reads
   them so, within [group_bytes], or 32 bytes within [pair_bytes]. *)
let shape ~f64 (s : step) lmax =
  let es = cplx f64 and tl = s.inner.tl in
  let lp = lmax + (lmax lsr pad_shift f64) + 1 in
  let lp = if lp mod 2 = 0 then lp + 1 else lp in
  let bytes g = header g + (g * lp * es) in
  let waves x = 32 * (((x * tl) + 31) / 32) in
  (* No more than a seventh of a group's lanes idle. *)
  let idle x = 7 * (waves x - (x * tl)) > waves x in
  let rec grow g =
    if idle g && 2 * g * tl <= K.fft_threads && bytes (2 * g) <= group_bytes
    then grow (2 * g)
    else g
  in
  let g = grow (pow2_floor 1 (Int.max 1 (128 / tl))) in
  let lf = s.lf_ld || s.lf_st in
  let g = if lf then Int.max g (64 / es) else g in
  let rec fit g =
    if g > 1 && (bytes g > group_bytes || g * tl > K.fft_threads) then
      fit (g / 2)
    else g
  in
  let g = fit g in
  let rec sector g =
    if
      (not f64) && lf
      && g * es < 32
      && bytes (2 * g) <= pair_bytes
      && 2 * g * tl <= K.fft_threads
    then sector (2 * g)
    else g
  in
  let g = sector g in
  let rec few g = if g > 1 && g / 2 >= s.nv then few (g / 2) else g in
  let g = few g in
  let lf_ld = s.lf_ld && g > 1 and lf_st = s.lf_st && g > 1 in
  { s with lp; g; lf_ld; lf_st }

let flag b = Bool.to_int b

(* The words of the sides [s] and [d], their lines over [dim], as layouts
   from [i] (kernels.h); the word after them. *)
let set_sides w i dim (s : side) (d : side) =
  let shape = Iarray.of_array dim in
  let layout i (s : side) =
    Plan.layout w i (L.v ~offset:s.first ~strides:(Iarray.of_array s.str) shape)
  in
  layout (layout i s) d

(* The slot of where [w] lies in [b]. *)
let slot_of b w =
  match w.mem with Mem x -> Plan.slot b x | Workspace -> Plan.workspace b

(* Adds to [k] a launch of [kernel] over [groups], with words [w] whose
   [src], [dst] and [plan] words hold those sides' and the plan's
   offsets. *)
let push k kernel ~groups ~shared ~used w ~src ~dst ~plan (s : side)
    (d : side) p =
  Plan.set w src s.at.byte;
  Plan.set w dst d.at.byte;
  Plan.set w plan 0;
  let refs =
    [|
      { Plan.word = src; slot = slot_of k.b s.at; write = false };
      { Plan.word = dst; slot = slot_of k.b d.at; write = true };
      {
        Plan.word = plan;
        slot = Plan.constant k.b ~key:p.key ~length:p.length p.bytes;
        write = false;
      };
    |]
  in
  Plan.launch k.b kernel ~groups:(groups, 1, 1) ~shared ~used w refs

let push_step k p (s : step) lmax =
  if s.nv > 0 then begin
    let s = shape ~f64:k.f64 s lmax in
    let w = Plan.words (kernel k) in
    let groups = Plan.grid ~per:s.g (kernel k) s.nv in
    let l = s.inner.l in
    Plan.set w K.fft_groups groups;
    Plan.set w K.fft_nv s.nv;
    Plan.set w K.fft_small (flag (Plan.small s.nv));
    Plan.axis w K.fft_m s.m;
    Plan.axis w K.fft_ns s.ns;
    Plan.set w K.fft_nsrc s.nsrc;
    Plan.set w K.fft_limit s.limit;
    Plan.set w K.fft_gts s.gts;
    Plan.set w K.fft_nh s.nh;
    Plan.set w K.fft_tw (List.assoc l p.tw);
    Plan.set w K.fft_glo p.glo;
    Plan.set w K.fft_ghi p.ghi;
    Plan.set w K.fft_rlo p.rlo;
    Plan.set w K.fft_rhi p.rhi;
    Plan.set w K.fft_chirp p.chirp;
    Plan.set w K.fft_bspec p.bspec;
    Plan.set w K.fft_rlog p.rlog;
    Plan.set w K.fft_l l;
    Plan.set w K.fft_lld s.lld;
    Plan.set w K.fft_lst s.lst;
    Plan.set w K.fft_lp s.lp;
    Plan.set w K.fft_g s.g;
    Plan.set w K.fft_tl s.inner.tl;
    Plan.set w K.fft_gshift (log2 s.g);
    Plan.set w K.fft_lf_ld (flag s.lf_ld);
    Plan.set w K.fft_lf_st (flag s.lf_st);
    Plan.set w K.fft_ld s.ld;
    Plan.set w K.fft_st s.st;
    Plan.set w K.fft_conj_in (flag s.conj_in);
    Plan.set w K.fft_conj_out (flag s.conj_out);
    Plan.set w K.fft_chirp_in (flag s.chirp_in);
    Plan.set w K.fft_post s.post;
    Plan.set w K.fft_np (Array.length s.inner.radices);
    Plan.set w K.fft_half (flag (l mod 2 = 0));
    Plan.set w K.fft_nops (Array.length s.ops);
    Plan.set w K.fft_g_bits p.g_bits;
    Plan.set w K.fft_r_bits p.r_bits;
    Plan.set w K.fft_rader (flag s.rader);
    Array.iteri (fun i o -> Plan.set w (K.fft_op + i) o) s.ops;
    let first = K.fft_op + Array.length s.ops in
    let ns = ref 1 in
    Array.iteri
      (fun i radix ->
        let at = first + (K.fft_pass_words * i) in
        Plan.set w (at + K.fft_pass_r) radix;
        Plan.axis w (at + K.fft_pass_ns) !ns;
        Plan.set w (at + K.fft_pass_q) (l / radix);
        Plan.set w (at + K.fft_pass_ts) (l / (!ns * radix));
        ns := !ns * radix)
      s.inner.radices;
    let sides = first + (K.fft_pass_words * Array.length s.inner.radices) in
    Plan.set w K.fft_ps_in s.src.ps;
    Plan.set w K.fft_ps_out s.dst.ps;
    let used = set_sides w sides s.dim s.src s.dst in
    Plan.set w K.fft_used used;
    push k (kernel k) ~groups
      ~shared:(header s.g + (s.g * s.lp * cplx k.f64))
      ~used w ~src:K.fft_src ~dst:K.fft_dst ~plan:K.fft_plan s.src s.dst p
  end

let push_real k p ~dim ~src ~dst ~h ~split =
  let nl = count dim in
  if nl > 0 then begin
    let kernel = real_kernel k in
    let w = Plan.words kernel in
    let groups = Plan.grid kernel (nl * ((h / 2) + 1)) in
    Plan.set w K.fft_real_groups groups;
    Plan.set w K.fft_real_nl nl;
    Plan.set w K.fft_real_h h;
    Plan.axis w K.fft_real_pairs ((h / 2) + 1);
    Plan.set w K.fft_real_small (flag (Plan.small (nl * ((h / 2) + 1))));
    Plan.set w K.fft_real_rlo p.rlo;
    Plan.set w K.fft_real_rhi p.rhi;
    Plan.set w K.fft_real_r_bits p.r_bits;
    Plan.set w K.fft_real_split (flag split);
    Plan.set w K.fft_real_ps_in src.ps;
    Plan.set w K.fft_real_ps_out dst.ps;
    let used = set_sides w K.fft_real_in dim src dst in
    Plan.set w K.fft_real_used used;
    push k kernel ~groups ~shared:0 ~used w ~src:K.fft_real_src
      ~dst:K.fft_real_dst ~plan:K.fft_real_plan src dst p
  end

(* The core: lines over [dim] of [nc] complex points from [src] to [dst],
   the lines' first launch loading as [ld], their last storing as [st],
   below [limit]. [real] 1 splits an r2c's bins after, 2 makes a c2r's
   points before. [alias] reads [dst] ([src] is [dst]'s lines); [scratch]
   lets [dst] hold a long line's points between steps. *)
type core = {
  inv : bool;
  alias : bool;
  scratch : bool;
  dim : int array;
  nc : int;
  limit : int;
  src : side;
  dst : side;
  ld : int;
  st : int;
  real : int;
}

let blank_step ~dim ~src ~dst inner =
  {
    dim;
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
   real line's around it, in LDS. *)
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
      (blank_step ~dim:c.dim ~src:c.src ~dst:c.dst inner) with
      nv = count c.dim;
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
  let f64 = k.f64 and n = c.nc and dim = c.dim in
  let nl = count dim in
  let es = cplx f64 and smooth = p.how = Smooth in
  let src, first =
    if c.real = 2 then begin
      (* c2r's points from the bins, conjugated, into the workspace. *)
      let w = ws_side (Plan.take k.b (nl * n * es)) dim n in
      push_real k p ~dim ~src:c.src ~dst:w ~h:n ~split:false;
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
  let base = if !used > 0 then Plan.take k.b (!used * tn) else 0 in
  let side_of b =
    if b = 0 then src
    else if b = 1 then c.dst
    else ws_side (base + ((b - 2) * tn)) dim l1
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
            (blank_step ~dim ~src ~dst inner) with
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
  if c.real = 1 then push_real k p ~dim ~src:c.dst ~dst:c.dst ~h:n ~split:true

let core k (c : core) =
  let p = plan c.nc ~f64:k.f64 ~real:(c.real > 0) in
  match p.block with
  | Some inner -> in_block k p c inner
  | None -> long_lines k p c

(* The entries *)

let f64_of (type v s) (x : (v, s) A.t) =
  match A.dtype x with D.Complex128 | D.Float64 -> true | _ -> false

let call ~f64 = { f64; b = Plan.make () }

let complex_core ~inv ~alias dim n src dst =
  {
    inv;
    alias;
    scratch = true;
    dim;
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
      let dim, src, d = lines_of (if first then x else dst) dst a in
      within k (fun () ->
          core k
            (complex_core ~inv ~alias:(in_place || not first) dim dst.dims.(a)
               src d))
    done

let finish k = Plan.finish_some k.b

(* Lines of no point: one bin of +0 each, which a step of one point stores
   without loading any. *)
let zeros k (dim, src, dst) =
  let p = plan 1 ~f64:k.f64 ~real:false in
  push_step k p
    {
      (blank_step ~dim ~src ~dst (inner ~f64:k.f64 1)) with
      nv = count dim;
      limit = 1;
      nh = 1;
      lld = 1;
      lst = 1;
      ops = [||];
    }
    1

let c2c (d : Nx_kernel.direction) ~axes ~dst x =
  let k = call ~f64:(f64_of x) in
  c2c_axes k ~inv:(d = Inverse) ~in_place:false (Iarray.to_array axes)
    (arr_of dst) (arr_of x);
  finish k

let r2c ~axes ~dst x =
  let k = call ~f64:(f64_of x) in
  let axes = Iarray.to_array axes in
  let x = arr_of x and y = arr_of dst in
  let last = Array.length axes - 1 in
  let a = axes.(last) in
  let n = x.dims.(a) in
  if n = 0 && numel y > 0 then zeros k (lines_of x y a)
  else if numel y > 0 then begin
    let dim, src, d = lines_of x y a in
    let h = n / 2 in
    let even = n mod 2 = 0 in
    within k (fun () ->
        core k
          {
            inv = false;
            alias = false;
            scratch = even;
            dim;
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
  finish k

let c2r ~axes ~dst x =
  let k = call ~f64:(f64_of x) in
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
        let w = contiguous (Plan.take k.b (numel x * cplx k.f64)) x.dims in
        c2c_axes k ~inv:true ~in_place:false (Array.sub axes 0 last) w x;
        w
    in
    let dim, src, d = lines_of x y a in
    let even = n mod 2 = 0 in
    within k (fun () ->
        core k
          {
            inv = true;
            alias = false;
            scratch = false;
            dim;
            nc = (if even then n / 2 else n);
            limit = (if even then n / 2 else n);
            src;
            dst = d;
            ld = (if even then K.fft_ld_c else K.fft_ld_hc);
            st = (if even then K.fft_st_rp else K.fft_st_re);
            real = (if even then 2 else 0);
          })
  end;
  finish k
