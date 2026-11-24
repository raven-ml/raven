(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The transforms' plans.

   A line of n points runs as an FFT of l1 points, as Stockham steps of at most
   4096 points, each one group's work: one step where l1 fits a group, else a
   step per factor of a balanced split of l1, each fitting a group, with tiled
   transposes between them.

   l1 is n where n is 7-smooth, n - 1 for Rader's convolution where n is a prime
   whose n - 1 is 7-smooth and fits a group, else Bluestein's 7-smooth m >= 2n -
   1. A step's radices are a function of its points alone, so a line's
   arithmetic is a function of n: a layout, a batch or a kernel only decides how
   a step moves its points.

   Error. nx.amd's derivation holds, with fft.hip's butterflies and their η(R)
   (1, 4.17, 2.13, 5.63, 7.40, 3.97, 7.63 for 2, 3, 4, 5, 7, 8, 16): a pass
   costs at most 4.52 u per bit of n for every radix, a twiddle from one table
   rounded once 3 u, from two tables 6 u, and [factors] takes q <= ⌈log₂ n / 8⌉
   steps, so e(n) <= 5.35 log₂ n for a 7-smooth n. *)

(* Steps *)

let most_points = 4096

type step = { l : int; radices : int array; p : int; tl : int; threads : int }

let smooth n =
  let rec strip n p = if n mod p = 0 then strip (n / p) p else n in
  n >= 1 && strip (strip (strip (strip n 2) 3) 5) 7 = 1

(* [n]'s factors [p] and what is left. *)
let rec strip n p k = if n mod p = 0 then strip (n / p) p (k + 1) else (n, k)

(* [l]'s radices: a [2^k] for each [k] of its 2s, one power of two for the 2s
   left, then its 7s, 5s and 3s. *)
let radices ~k l =
  let n, twos = strip l 2 0 in
  let n, sevens = strip n 7 0 in
  let n, fives = strip n 5 0 in
  let _, threes = strip n 3 0 in
  Array.concat
    [
      Array.make (twos / k) (1 lsl k);
      (if twos mod k > 0 then [| 1 lsl (twos mod k) |] else [||]);
      Array.make sevens 7;
      Array.make fives 5;
      Array.make threes 3;
    ]

(* The threads a line of [l] points takes at [p] points a thread: each of a
   pass's [l / R] butterflies on one, [⌊p / R⌋] a thread. *)
let threads_of l radices p =
  Array.fold_left
    (fun t r ->
      let per = p / r in
      Int.max t (((l / r) + per - 1) / per))
    1 radices

let simd = 32

let step l =
  if l < 1 || l > most_points || not (smooth l) then
    invalid_arg (Printf.sprintf "Fft.step: %d points" l);
  (* At 8 points a thread a line takes twice the threads, so more of a pass's
     butterflies run at once, but its 2s take more passes: 8s where the line
     then fits 256 threads, else 16s. The rule reads [l] alone, so the bits
     do. *)
  let eights = radices ~k:3 l in
  let radices, p =
    if threads_of l eights 8 <= 256 then (eights, 8) else (radices ~k:4 l, 16)
  in
  let tl = threads_of l radices p in
  { l; radices; p; tl; threads = Int.max simd ((tl + simd - 1) / simd * simd) }

(* For q from 2 up, [n]'s primes largest first, each to the least factor: the
   largest factor took its last prime when it was the least, at most n^(1/q), so
   at q = ⌈log₂ n / 8⌉ each factor is at most 7 · 256. *)
let factors n =
  if n <= most_points || not (smooth n) then
    invalid_arg (Printf.sprintf "Fft.factors: %d points" n);
  let primes =
    List.concat_map
      (fun p -> List.init (snd (strip n p 0)) (fun _ -> p))
      [ 7; 5; 3; 2 ]
  in
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
    if Array.for_all (fun x -> x <= most_points) f then begin
      Array.sort (fun a b -> Int.compare b a) f;
      Array.map step f
    end
    else from (q + 1)
  in
  from 2

(* Groups *)

let group_memory = Plan.group_memory

let plane s =
  let pow2 = s.l land (s.l - 1) = 0 in
  8 * if pow2 && s.l <= 2048 then s.l + (s.l / 16) else s.l

(* Contiguous lines fill this many threads, while their planes fit a group; a
   line of more threads takes a group alone. *)
let rows_threads = 256
let rows s = Int.max 1 (Int.min (rows_threads / s.tl) (group_memory / plane s))
let side_most = 2048
let side_threads = 512

(* Side by side, a group's [side_threads] take at least this many lines, so that
   a group reads more than two lines' bytes of each 128-byte cache line. *)
let side_least = 3

let side_step l =
  if l < 1 || l > side_most || not (smooth l) then
    invalid_arg (Printf.sprintf "Fft.side_step: %d points" l);
  let s = step l in
  if side_threads / s.tl >= side_least then s
  else
    (* p arranges a pass's butterflies among threads and leaves the bits to the
       radices: 16 points a thread halve the threads a line. *)
    let tl = threads_of l s.radices 16 in
    {
      s with
      p = 16;
      tl;
      threads = Int.max simd ((tl + simd - 1) / simd * simd);
    }

let side l = Int.min (8192 / l) (side_threads / (side_step l).tl)
let column_lines = 16

(* On an M1 Max the column step crosses the transposes between 16 and 32 KiB
   between a line's consecutive points; half of 16 KiB leaves a margin for parts
   with more groups in flight or fewer translations cached. The bits do not
   depend on it. *)
let columns_stride_most = 8192

(* Lines *)

type how = Smooth | Rader | Bluestein of int
type line = { n : int; real : bool; how : how; l1 : int; steps : step array }

let is_prime n =
  let rec from d = d * d > n || (n mod d <> 0 && from (d + 1)) in
  n >= 2 && from 2

let steps_of l1 = if l1 <= most_points then [| step l1 |] else factors l1

(* A radix's weight in Bluestein's choice of m: its operations a point. *)
let weight = function
  | 2 -> 4.
  | 3 -> 7.3
  | 4 -> 7.
  | 5 -> 10.4
  | 7 -> 12.8
  | 8 -> 10.
  | _ -> 13.3

let cost m =
  Float.of_int m
  *. Array.fold_left
       (fun c s -> Array.fold_left (fun c r -> c +. weight r) c s.radices)
       0. (steps_of m)

(* The 7-smooth numbers from [lo] to [hi], in order. *)
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
  List.sort Int.compare (from 1 [ 2; 3; 5; 7 ] [])

(* The cheapest 7-smooth [m] in [2n - 1, 2 (2n - 1)], the least among equals:
   each product of powers of 3, 5 and 7 within the window's top, times the
   powers of 2 that bring it into the window, at most two. *)
let bluestein n =
  let lo = (2 * n) - 1 in
  let hi = 2 * lo in
  let best = ref 0 and least = ref infinity in
  let weigh m =
    let c = cost m in
    if c < !least || (c = !least && m < !best) then begin
      best := m;
      least := c
    end
  in
  let rec odd y = function
    | [] ->
        let m = ref y in
        while !m < lo do
          m := 2 * !m
        done;
        while !m <= hi do
          weigh !m;
          m := 2 * !m
        done
    | p :: rest ->
        let y = ref y in
        while !y <= hi do
          odd !y rest;
          y := !y * p
        done
  in
  odd 1 [ 3; 5; 7 ];
  !best

(* [bluestein]'s [m] for each of [ns], in ascending order, with each window slid
   over the 7-smooth numbers, each costed once: a queue holds the window's
   numbers that no later one in it undercuts, so its first is the cheapest, the
   least among equals. *)
let windows ns =
  let top = if Array.length ns = 0 then 1 else ns.(Array.length ns - 1) in
  let s = Array.of_list (smooth_between 1 (2 * ((2 * top) - 1))) in
  let c = Array.map cost s in
  let queue = Array.make (Array.length s) 0 in
  let first = ref 0 and last = ref 0 and next = ref 0 in
  Array.map
    (fun n ->
      let lo = (2 * n) - 1 in
      while !next < Array.length s && s.(!next) <= 2 * lo do
        while !last > !first && c.(queue.(!last - 1)) > c.(!next) do
          decr last
        done;
        queue.(!last) <- !next;
        incr last;
        incr next
      done;
      while s.(queue.(!first)) < lo do
        incr first
      done;
      s.(queue.(!first)))
    ns

(* For each [m] past [most_points] that some [n] up to [2{^32}] takes, the least
   such [n], in order. [bluestein]'s [m] is nondecreasing in [n]: a larger [n]'s
   window loses its least numbers and gains larger ones, so the cheapest stays
   or moves up. Its window changes only where [n] passes [(s + 1) / 2], losing
   [s], or reaches [⌈(s + 2) / 4⌉], gaining it, so [m] changes only at those
   [n]. *)
let takers () =
  let top = 1 lsl 32 in
  let events =
    smooth_between 1 (2 * ((2 * top) - 1))
    |> List.concat_map (fun s -> [ ((s + 1) / 2) + 1; (s + 5) / 4 ])
    |> List.filter (fun n -> n >= 2 && n <= top)
    |> List.cons 2 |> List.sort_uniq Int.compare |> Array.of_list
  in
  let ms = windows events in
  let last = Array.length events - 1 in
  let takers = ref [] in
  Array.iteri
    (fun i m ->
      if m > most_points && (i = 0 || ms.(i - 1) <> m) then begin
        (* [m] changes only at an event, so [n + 1] takes [m] unless it is the
           next event. *)
        let next =
          if i < last && events.(i + 1) = events.(i) + 1 then ms.(i + 1) else m
        in
        takers := (m, events.(i), next) :: !takers
      end)
    ms;
  List.rev !takers

let least_takers () = List.map (fun (m, n, _) -> (m, n)) (takers ())

(* The host cut: the greatest Bluestein [m] whose kernel the host makes in at
   most about 1 s, 1.0 s and 50 MB at [2^20]; and [n_B], the greatest length
   whose [m] is within it. Since [m] does not fall as [n] grows, a Bluestein
   line of more points takes a kernel the device makes. *)
let bluestein_most = 1 lsl 20
let bluestein_points = 524_288

let past_cut line =
  match line.how with
  | Bluestein m -> m > bluestein_most
  | Smooth | Rader -> false

(* A line of [n] points, its Bluestein [m] by [bluestein]. *)
let make ?(bluestein = bluestein) n ~real =
  let how, l1 =
    if smooth n then (Smooth, n)
    else if (not real) && is_prime n && smooth (n - 1) && n - 1 <= most_points
    then (Rader, n - 1)
    else
      let m = bluestein n in
      (Bluestein m, m)
  in
  { n; real; how; l1; steps = steps_of l1 }

(* Planning a Bluestein line weighs each length it may take, so lines are
   kept. *)
let kept_lines = 64
let lines : (int * bool, line) Cache.t = Cache.make ()

let line n ~real =
  if n < 1 then invalid_arg (Printf.sprintf "Fft.line: %d points" n);
  match Cache.find lines (n, real) with
  | Some l -> l
  | None -> Cache.keep ~count:kept_lines lines (n, real) (make n ~real)

(* Calls

   Along an axis, a launch moves lines from one side to another, and the sides
   alone choose the path, never the passes. A line of one step with both sides
   rows takes the contiguous step; one of at most 2048 points with no side
   scattered or wide takes the side-by-side step; any other one step moves each
   side that is not rows through a transpose to or from the workspace. A long
   line runs in two levels, [n = s_0 M], each step in place on rows of its
   points, the transposes between them recording what they read, scale and
   write.

   Costs are Plan's counts, at one group resident a core, the bound every
   pipeline meets, since planning makes none. A barrier counts 4 links until the
   kernels' are measured. *)

module L = Nx_array.Layout

type access = Rows | Columns | Scattered
type buffer = Operand | Dst | Workspace of int

type side = {
  buffer : buffer;
  first : int;
  ps : int;
  dims : int array;
  strides : int array;
  points : int;
  width : int;
  base : int;
  access : access;
}

type path = Contiguous | Side_by_side
type part = First_pass | Last_passes
type input = Complex | Real | Bins
type scale = Plain | Twiddle | Chirp_in | Kernel | Chirp_out
type output = Complex | Half | Real

type work =
  | Step of { line : line; step : step; path : path; lines : int }
  | Column_step of { line : line; step : step; part : part }
  | Transpose of {
      line : line;
      split : int * int;
      wide : bool;
      input : input;
      scale : scale;
      output : output;
    }
  | Split of { wide : bool }
  | Prepass of { wide : bool }

type launch = {
  work : work;
  src : side;
  dst : side;
  first_group : int;
  groups : int;
  threads : int;
  shared : int;
  cost : Plan.cost;
  seconds : float;
}

type call = { launches : launch array; workspace : int; seconds : float }

(* A complex64 point's bytes. *)
let width = 8

(* A float32 element's bytes. *)
let real_width = 4

(* The most elements a step's 32-bit point offset [j · ps] reaches. *)
let span_most = (1 lsl 31) - 1

(* The span past which a side is wide, [span_most] but under a test's bound. A
   step's rows hold a halved line's [2 · most_points] floats at most, so a bound
   at or past their span keeps every step's side narrow. *)
let span_bound = Atomic.make span_most
let wide_least = (2 * most_points) - 1

let bound_wide n =
  if n < wide_least || n > span_most then invalid_arg "Nx_metal.bound_fft_wide";
  Atomic.set span_bound n;
  Plan.renew ()

let wide s =
  s.points > 1 && Int.abs s.ps * (s.points - 1) > Atomic.get span_bound

(* fft_s4096 and a column step read complex elements at 32-bit offsets from each
   line's first. *)
let narrow (s : side) = s.width = width && not (wide s)
let unit s = s = 1 || s = -1
let count dims = Array.fold_left ( * ) 1 dims

(* A batch of one line has no adjacent line, so it is never columns. *)
let access_of s =
  if s.points <= 1 || unit s.ps then Rows
  else if count s.dims > 1 && unit s.strides.(Array.length s.strides - 1) then
    Columns
  else Scattered

(* The lines of [x] along [axis] and of [y], one shape: the batch's axes merged
   where both lay them out as one run. *)
let sides x y axis =
  let batch l =
    let keep a = Iarray.of_list (List.filteri (fun i _ -> i <> axis) a) in
    L.v ~offset:(L.offset l)
      ~strides:(keep (Iarray.to_list (L.strides l)))
      (keep (Iarray.to_list (L.shape l)))
  in
  let merged = L.coalesce [| batch x; batch y |] in
  let side buffer ~width ~base l b =
    let s =
      {
        buffer;
        base;
        first = L.offset l;
        ps = L.stride l axis;
        dims = Iarray.to_array (L.shape b);
        strides = Iarray.to_array (L.strides b);
        points = L.dim l axis;
        width;
        access = Rows;
      }
    in
    { s with access = access_of s }
  in
  (side, merged)

(* Lines of [points] over [dims], one after another in C order from [first]. *)
let packed ?(width = width) ?(base = 0) buffer ~first dims points =
  let r = Array.length dims in
  let strides = Array.make r points in
  for i = r - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * dims.(i + 1)
  done;
  { buffer; base; first; ps = 1; dims; strides; points; width; access = Rows }

(* The same points as rows of [l]. *)
let rows_of s l =
  let total = count s.dims * s.points in
  { s with dims = [| total / l |]; strides = [| l |]; ps = 1; points = l }

(* Costs *)

let ceil_div a b = (a + b - 1) / b
let barrier_links = 4

(* A transpose's tile, [tile_side] by [tile_side] points, and its group's
   threads and bytes; a split's or a pre-pass's pairs a block and threads. *)
let tile_side = Kernels.fft_tile
let tile = tile_side * tile_side
let tile_threads = Kernels.fft_move_threads
let tile_shared = Kernels.fft_move_shared
let pair_block = Kernels.fft_pair_block
let pair_threads = Kernels.fft_pair_threads
let rec gcd a b = if b = 0 then Int.abs a else gcd b (a mod b)

(* The worst byte of a line at which a group's segment of a columns side's run
   starts. A segment starts at the run's lowest element plus a multiple of [gcd
   g d] lines, its first line's place in the run; runs' lowest elements step by
   the point stride and the outer batch strides. So every segment starts in
   [base + lowest w + ⟨gcd (128, ps w, S_k w, gcd (g, d) w)⟩] mod 128, whose
   worst is [128 - q] past the start's residue mod [q]. *)
let worst_start s ~g ~d =
  let inner = Array.length s.dims - 1 in
  let lowest = if s.strides.(inner) < 0 then s.first - (d - 1) else s.first in
  let q = ref (gcd Plan.line (s.ps * s.width)) in
  Array.iteri
    (fun k st -> if k <> inner then q := gcd !q (st * s.width))
    s.strides;
  let q = gcd !q (gcd g d * s.width) in
  let at = (((s.base + (lowest * s.width)) mod q) + q) mod q in
  Plan.line - q + at

(* A columns side's traffic for one group's [g] lines in runs of [d] adjacent
   lines, from the worst start [o]: [r = min g d] lines a segment, [⌈g / r⌉]
   segments, one more where neither of [g] and [d] divides the other, since the
   group may then straddle the innermost axis, each the lines its [w r] bytes
   touch from [o]. *)
let run_bytes ~width ~points ~g ~d ~o =
  let r = Int.min g d in
  let straddles = d mod g <> 0 && g mod d <> 0 in
  let runs = ((g + r - 1) / r) + Bool.to_int straddles in
  let lines = (o + (width * r) + Plan.line - 1) / Plan.line in
  Plan.line * points * runs * lines

(* A side's traffic for one group's [g] lines: a whole cache line for each point
   no neighbour shares, or a columns side's runs from their worst start by the
   layout. *)
let side_bytes s ~g =
  match s.access with
  | Rows -> s.width * s.points * g
  | Scattered -> Plan.line * s.points * g
  | Columns ->
      let d = s.dims.(Array.length s.dims - 1) in
      run_bytes ~width:s.width ~points:s.points ~g ~d ~o:(worst_start s ~g ~d)

(* A pass's butterfly's dependent depth, in links. *)
let depth = function 2 -> 1 | 4 -> 2 | 8 -> 4 | 16 -> 6 | r -> r

(* An FFT of [s]'s points: its operations, about [5 l log₂ l] and a twiddle
   product a point each pass after the first, and its chain: each pass's
   butterfly, its twiddle's product and issue, and its barriers. *)
let fft_ops s =
  let l = Float.of_int s.l in
  (5. *. l *. Float.log2 (Float.max l 1.))
  +. (6. *. l *. Float.of_int (Int.max 0 (Array.length s.radices - 1)))

(* A twiddle load's issue a SIMD group, in links. A pass's twiddle address comes
   from its thread and the pass, so nothing in the chain waits on the read but
   its product. *)
let issue = 0.15

(* The chain of [s]'s passes at [tl] threads a line in groups of [simd] SIMD
   groups: past the first, a pass's twiddle product, 2 links, and the issue of a
   thread's loads, [r - 1] for each of its [⌈(l / r) / tl⌉] butterflies. *)
let pass_links s ~tl ~simd ~barriers i r =
  let twiddle =
    if i = 0 then 0.
    else
      let loads = ((s.l / r) + tl - 1) / tl * (r - 1) in
      2. +. (issue *. Float.of_int (loads * simd))
  in
  Float.of_int (depth r + (barriers * barrier_links)) +. twiddle

let fft_chain s ~tl ~simd ~barriers =
  let c = ref 0. in
  Array.iteri
    (fun i r -> c := !c +. pass_links s ~tl ~simd ~barriers i r)
    s.radices;
  Float.to_int (Float.ceil !c)

(* The groups of a launch over [blocks], in launches each within
   {!Plan.allowed}, each of the most blocks that fit from where the last ends:
   [bytes first c], the bytes of blocks [first] to [first + c - 1], a group's
   operations, the launch's tables. *)
let split ~name ~threads ~bytes ~ops ~chain ~tables blocks =
  let cost first c =
    { Plan.bytes = bytes first c + tables; ops = c * ops; chain }
  in
  let seconds first c =
    Plan.seconds ~threads ~groups:(c, 1, 1) (cost first c)
  in
  let allowed = Plan.allowed () in
  let rec most first lo hi =
    if lo >= hi then lo
    else
      let mid = lo + ((hi - lo + 1) / 2) in
      if seconds first mid <= allowed then most first mid hi
      else most first lo (mid - 1)
  in
  let rec from first acc =
    if first >= blocks then List.rev acc
    else begin
      if seconds first 1 > allowed then
        invalid_arg (name ^ ": a group takes past the launch budget");
      (* Most launches take every block left: one reading settles them. *)
      let hi = Int.min (blocks - first) Plan.most_groups in
      let c = if seconds first hi <= allowed then hi else most first 1 hi in
      from (first + c) ((first, c, cost first c, seconds first c) :: acc)
    end
  in
  from 0 []

(* Blocks of [b] bytes each. *)
let per_block b _ c = c * b

(* Planning *)

(* The launches planned so far, the last first, the workspace's bytes, and the
   byte from which the axis being planned takes its own. A [tally] plans no
   launch: it sums each work's seconds as one launch, [whole]. Without
   [columns], no line takes the column step. *)
type acc = {
  name : string;
  tally : bool;
  columns : bool;
  mutable launches : launch list;
  mutable workspace : int;
  mutable base : int;
  mutable whole : float;
  bases : int * int;
}

let start ?(columns = true) ~bases name =
  {
    bases;
    name;
    tally = false;
    columns;
    launches = [];
    workspace = 0;
    base = 0;
    whole = 0.;
  }

(* The axis's workspace from its byte [at], and its first [bytes] taken. *)
let ws acc at = Workspace (acc.base + at)
let take acc bytes = acc.workspace <- Int.max acc.workspace (acc.base + bytes)

let push acc ~work ~src ~dst ~threads ~shared ~bytes ~ops ~chain ~tables blocks
    =
  if blocks > 0 then
    acc.whole <-
      acc.whole
      +. Plan.seconds ~threads ~groups:(blocks, 1, 1)
           { Plan.bytes = bytes 0 blocks + tables; ops = blocks * ops; chain };
  if blocks > 0 && not acc.tally then
    List.iter
      (fun (first_group, groups, cost, seconds) ->
        acc.launches <-
          {
            work;
            src;
            dst;
            first_group;
            groups;
            threads;
            shared;
            cost;
            seconds;
          }
          :: acc.launches)
      (split ~name:acc.name ~threads ~bytes ~ops ~chain ~tables blocks)

(* An inverse conjugates each point a launch loads and each it stores, a
   negation each: two operations a point, charged in both directions on every
   launch an inverse runs, so that both directions plan alike. *)
let flip_ops = 2

(* Whether a transpose copies bits, which an inverse runs as forward: it
   commutes with conjugation. *)
let copies (input : input) scale (output : output) =
  scale = Plain
  &&
  match (input, output) with
  | Complex, (Complex | Real) | Real, Real -> true
  | Real, (Complex | Half) | Complex, Half | Bins, _ -> false

(* Whether an inverse runs a transpose and conjugates in it: neither a copy nor
   an r2c's, which reads [Real] into [Complex] or writes [Half]. *)
let flipped (input : input) scale (output : output) =
  (not (copies input scale output)) && input <> Real && output <> Half

(* The roots a step of [l] points reads: the first half of an even step's, since
   [w_l^(e + l/2) = -w_l^e], all of an odd one's. *)
let roots l = if l mod 2 = 0 then l / 2 else l

(* A step of [line] over its sides' lines of [step.l] points, or of the line's
   [n] where it fits a group. Each group reads its roots and, for each of its
   lines, [per_line] bytes of the line's other tables: Bluestein's chirp and
   kernel, Rader's kernel and logarithms. A long line's step is its FFT
   alone. *)
(* A group's lines, capped by the call's: a group's empty slots would still run
   every pass, on the one core a lone group takes (1x1024 at two slots took
   14.4 us a call, at one 11.4). A power of two bounds the keys a step takes
   to log₂ of its slots. *)
let group_lines ~path (s : step) n =
  let most =
    match path with Contiguous -> rows s | Side_by_side -> side s.l
  in
  let rec pow2 p = if p >= n then p else pow2 (2 * p) in
  Int.min most (pow2 1)

let push_step acc ~line ~(step : step) ~path src dst =
  let step =
    match path with Contiguous -> step | Side_by_side -> side_step step.l
  in
  let one = Array.length line.steps = 1 in
  let g = group_lines ~path step (count src.dims) in
  let in_simds t = Int.max simd ((t + simd - 1) / simd * simd) in
  let threads, shared, tl, barriers =
    match path with
    | Contiguous -> (in_simds (g * step.tl), g * plane step, step.tl, 2)
    | Side_by_side ->
        let threads =
          if g = side step.l then side_threads else in_simds (g * step.tl)
        in
        (threads, 4 * g * step.l, step.tl, 4)
  in
  let fft = fft_ops step and n = Float.of_int line.n in
  let l = Float.of_int step.l in
  let once = fft_chain step ~tl ~simd:(threads / simd) ~barriers in
  let line_ops, chain, per_line =
    match line.how with
    | Rader when one ->
        ( (2. *. fft) +. (6. *. l) +. (2. *. n),
          (2 * once) + 2,
          (width * step.l) + (4 * line.n) )
    | Bluestein _ when one ->
        ( (2. *. fft) +. (6. *. l) +. (12. *. n),
          (2 * once) + 6,
          width * (line.n + step.l) )
    | _ -> (fft, once, 0)
  in
  let line_ops = line_ops +. (Float.of_int flip_ops *. if one then n else l) in
  let chain = Plan.read_links + chain in
  (* A halved line in a group splits its points into bins, or makes them, in its
     store or load: a root of [2n] and about 10 operations a point. *)
  let split = line.real && one in
  let line_ops = if split then line_ops +. (10. *. n) else line_ops in
  let per_line =
    if split then per_line + (width * ((line.n / 2) + 1)) else per_line
  in
  let chain = if split then chain + Plan.read_links + 4 else chain in
  let ops = threads * Float.to_int (Float.ceil (line_ops /. Float.of_int tl)) in
  push acc
    ~work:(Step { line; step; path; lines = g })
    ~src ~dst ~threads ~shared
    ~bytes:(per_block (side_bytes src ~g + side_bytes dst ~g + (g * per_line)))
    ~ops ~chain
    ~tables:(width * roots step.l)
    (ceil_div (count src.dims) g)

(* A point's bytes as a transpose reads or writes it. A halved real line's side
   is [Complex]: its points are pairs of floats. *)
let in_bytes : input -> int = function
  | Complex | Bins -> width
  | Real -> real_width

let out_bytes : output -> int = function
  | Complex | Half -> width
  | Real -> real_width

(* Twiddled transposes *)

(* A twiddled transpose's tiles are [tile_side] columns [c] by [tile_side] rows
   [r] of the rows it writes. A SIMD group's lanes lie along [r] at one [c],
   each point times [w^e = lo.(e mod low) hi.(e / low)], [e = r c]: two
   products. *)
let low = 1 lsl Kernels.fft_low_bits
let twiddle_links = 4

(* The high table's lines the lanes at column [c] touch: their entries step by
   [c / low], at most [⌊(tile_side - 1) c / low⌋ + 2] of them, adjacent, so at
   most one line more than they fill. *)
let hi_lines c =
  ceil_div (width * (((tile_side - 1) * c / low) + 2)) Plan.line + 1

(* [hi_lines] summed over the columns [lo <= c < hi], a run of columns at a
   time: [hi_lines] does not decrease in [c], so each run of one value ends
   where a search finds it, a run about every 530 columns. *)
let hi_lines_over lo hi =
  let rec last v lo hi =
    if lo >= hi then lo
    else
      let mid = lo + ((hi - lo + 1) / 2) in
      if hi_lines mid = v then last v mid hi else last v lo (mid - 1)
  in
  let rec from c acc =
    if c >= hi then acc
    else
      let v = hi_lines c in
      let e = last v c (hi - 1) in
      from (e + 1) (acc + (v * (e - c + 1)))
  in
  from lo 0

(* A transpose's tiles: [tile_side] columns [c] by [tile_side] rows [r] of each
   run of [a b] points written as [b] rows of [a], a run's by column, then by
   row. A move [(1, n)] takes the side's lines as its rows, one run of
   [tile_side] lines by [tile_side] points a tile. *)
let grid ~split:(a, b) src =
  let lines = count src.dims in
  if a = 1 then ((lines, b), 1) else ((a, b), lines)

(* The bytes of a transpose's tiles [first] to [first + c - 1] over a grid of
   runs of [b] rows of [a]: their points at [point] bytes, and for a twiddle
   their lanes' lines of the high table, in closed form over the columns. *)
let tile_bytes ~grid:(a, b) ~point ~twiddle =
  let rows = ceil_div a tile_side and columns = ceil_div b tile_side in
  let tiles = rows * columns in
  (* [below j]: the high table's lines of the columns before tile column [j],
     summed once for all of [split]'s searches. A move's grid, whose [b] is a
     line's points, takes no twiddle. *)
  let below =
    if not twiddle then fun _ -> 0
    else begin
      let sums = Array.make (columns + 1) 0 in
      for j = 1 to columns do
        let c = (j - 1) * tile_side in
        sums.(j) <- sums.(j - 1) + hi_lines_over c (Int.min b (c + tile_side))
      done;
      fun j -> sums.(j)
    end
  in
  let before g =
    let runs = g / tiles and t = g mod tiles in
    let j = t / rows and m = t mod rows in
    let c0 = j * tile_side in
    let c1 = Int.min b (c0 + tile_side) in
    let points =
      (runs * a * b) + (c0 * a) + ((c1 - c0) * Int.min a (m * tile_side))
    in
    let lines =
      (runs * rows * below columns)
      + (rows * below j)
      + (m * (below (j + 1) - below j))
    in
    (point * points) + (Plan.line * lines)
  in
  fun first c -> before (first + c) - before first

(* A division by an extent's divisor, the remainder's product and the stride's:
   130.3 float32 [fma]s' time a thread in a full grid of the M1 Max, its [fma]s
   at 0.91 of a lane a cycle, taken at 131 operations; and in a chain [mulhi],
   an add and a shift, 2.15 links in a lone thread, taken at 3. *)
let division_ops = 131
let division_links = 3

(* A tail's bytes: both sides' axes, where one is past what the words hold. *)
let tail_bytes src dst =
  let rs = Array.length src.dims and rd = Array.length dst.dims in
  if rs <= Kernels.fft_side_holds && rd <= Kernels.fft_side_holds then 0
  else 8 * Kernels.fft_axis_words * (rs + rd)

(* A group's divisions: [first] at each of its [threads] for its block, and for
   a line's coordinates one an axis past a side's outermost at each of [by]
   threads a side; their operations, and their chain, the sides' lines found
   side by side. *)
let divisions ~first ~threads ~by src dst =
  let rs = Int.max 0 (Array.length src.dims - 1)
  and rd = Int.max 0 (Array.length dst.dims - 1) in
  ( division_ops * ((threads * first) + (by * (rs + rd))),
    division_links * (first + Int.max rs rd) )

(* A transpose over tiles of up to [tile] points: each point read and written at
   its bytes on rows and a cache line an element off them. A chirp or a kernel
   adds a table's number a point; a twiddle, its tiles' high table lines, the
   low table and the products. *)
let push_transpose acc ~line ~split:(a, b) ?(input : input = Complex)
    ?(scale = Plain) ?(output : output = Complex) src dst =
  let wide = wide src || wide dst in
  (* A side off rows is charged a cache line a point. The column step's bound
     against the transposes (fft.mli's c2c) rests on this charge: refining it
     must revisit that bound and its test. *)
  let charge s bytes =
    if s.access = Rows then bytes else Plan.line * (bytes / s.width)
  in
  let point = charge src (in_bytes input) + charge dst (out_bytes output) in
  let scaled = scale <> Plain and twiddle = scale = Twiddle in
  let flip = if flipped input scale output then flip_ops else 0 in
  let ((ga, gb) as grid), runs = grid ~split:(a, b) src in
  let point = if scaled && not twiddle then point + width else point in
  (* The block's tile takes a division by the tiles of a run, but in a move,
     which is one run, and one by the rows of tiles, but in one column of them;
     a move's lines' positions take a barrier more. *)
  let move = a = 1 in
  let rows = ceil_div ga tile_side and columns = ceil_div gb tile_side in
  let first =
    Bool.to_int ((not move) && rows * columns > 1)
    + Bool.to_int (columns > 1 && rows > 1)
  in
  let by = if move then tile_side else tile_threads in
  let div_ops, div_links = divisions ~first ~threads:tile_threads ~by src dst in
  let chain =
    Plan.read_links + barrier_links + 2 + div_links
    + if move then barrier_links else 0
  in
  push acc
    ~work:(Transpose { line; split = (a, b); wide; input; scale; output })
    ~src ~dst ~threads:tile_threads ~shared:tile_shared
    ~bytes:(tile_bytes ~grid ~point ~twiddle)
    ~ops:
      ((tile * (2 + flip + (if scaled then 6 else 0) + if wide then 6 else 0))
      + div_ops)
    ~chain:(if twiddle then chain + twiddle_links else chain)
    ~tables:((if twiddle then width * low else 0) + tail_bytes src dst)
    (runs * ceil_div ga tile_side * ceil_div gb tile_side)

(* A split or a pre-pass over the pairs of a halved line's bins, [k] and [h - k]
   for [k] up to [h / 2]: [pair_block] pairs a block, and a root of [2h] a pair;
   [split] for a split. Wide where a side is. *)
let push_pairs acc ~split src dst =
  let wide = wide src || wide dst in
  let work = if split then Split { wide } else Prepass { wide } in
  let bins = if split then dst else src in
  let pairs = ((bins.points - 1) / 2) + 1 in
  let charge s = if s.access = Rows then s.width else Plan.line in
  let blocks = ceil_div pairs pair_block in
  let div_ops, div_links =
    divisions
      ~first:(Bool.to_int (blocks > 1))
      ~threads:pair_threads ~by:pair_threads src dst
  in
  push acc ~work ~src ~dst ~threads:pair_threads ~shared:0
    ~bytes:(per_block (pair_block * ((2 * (charge src + charge dst)) + width)))
    ~ops:
      (pair_block
       * ((if wide then 26 else 20) + if split then 0 else 2 * flip_ops)
      + div_ops)
    ~chain:((2 * Plan.read_links) + 4 + div_links)
    ~tables:(tail_bytes src dst)
    (count bins.dims * blocks)

(* A line of one step through the transposes: each side that is not rows moves
   to or from the workspace, which keeps its points and elements; where both
   move and differ, the step's two copies lie one after the other. *)
let transposed acc line src dst =
  let step = line.steps.(0) in
  let step_on = push_step acc ~line ~step in
  let copy s at = packed ~width:s.width (ws acc at) ~first:0 s.dims s.points in
  let bytes s = s.width * count s.dims * s.points in
  let move s d =
    if s.width = real_width then
      push_transpose acc ~line ~split:(1, s.points) ~input:Real ~output:Real s d
    else push_transpose acc ~line ~split:(1, s.points) s d
  in
  match (src.access = Rows, dst.access = Rows) with
  | true, _ ->
      let w = copy dst 0 in
      take acc (bytes dst);
      step_on ~path:Contiguous src w;
      move w dst
  | false, true ->
      let w = copy src 0 in
      take acc (bytes src);
      move src w;
      step_on ~path:Contiguous w dst
  | false, false ->
      let same = src.width = dst.width && src.points = dst.points in
      let into = copy src 0 in
      let out = if same then into else copy dst (bytes src) in
      take acc (if same then bytes src else bytes src + bytes dst);
      move src into;
      step_on ~path:Contiguous into out;
      move out dst

(* Column steps *)

(* A column step's group: 256 threads on [column_lines] lines, 256 points of
   each, so a block of lines takes [column_parts] groups. *)
let column_threads = Kernels.fft_step_threads
let column_points = Kernels.fft_step_points / column_lines
let column_parts = Kernels.fft_step_points / column_points

(* Whether [line], of [step], is one fft_c4096_first and fft_c4096_last compute:
   the only step of a complex 7-smooth line, of 4096 points of radices 16 16
   16. *)
let column_kernel line (step : step) =
  Array.length line.steps = 1
  && line.how = Smooth && (not line.real)
  && step.l = Kernels.fft_step_points
  && step.radices = [| 16; 16; 16 |]

(* Whether a line of one step may take the column step by its sides: the step of
   fft_c4096_first and fft_c4096_last on complex sides in columns, each run at
   least [column_lines] lines, and at most [columns_stride_most] bytes between a
   line's consecutive points on both sides and on the workspace, [width] bytes a
   line of a run. *)
let column_fits line src dst =
  let d = src.dims.(Array.length src.dims - 1) in
  let near s = Int.abs s.ps * s.width <= columns_stride_most in
  column_kernel line line.steps.(0)
  && src.access = Columns && dst.access = Columns && narrow src && narrow dst
  && d >= column_lines && near src && near dst
  && width * d <= columns_stride_most

(* A column step's workspace: line [c] of run [i] at element [4096 d i + c], its
   points [d] apart, [d] the lines of a run. *)
let column_side acc s =
  let r = Array.length s.dims in
  let strides = Array.make r 1 in
  for k = r - 2 downto 0 do
    strides.(k) <-
      (if k = r - 2 then s.points * s.dims.(r - 1)
       else strides.(k + 1) * s.dims.(k + 1))
  done;
  let w =
    {
      buffer = ws acc 0;
      base = 0;
      first = 0;
      ps = s.dims.(r - 1);
      dims = Array.copy s.dims;
      strides;
      points = s.points;
      width;
      access = Rows;
    }
  in
  { w with access = access_of w }

(* A segment of a group's side: its lines' same point, 128 bytes from the worst
   start of a full block; a last block of fewer lines lies within. *)
let segment s =
  let o = worst_start s ~g:column_lines ~d:column_lines in
  Plan.line * ceil_div (o + (width * column_lines)) Plan.line

(* A part's operations a group: its DFTs of 16, 5 · 16 · 4 operations each, the
   last passes' products, and an inverse's flips, each thread's share rounded
   up. *)
let column_ops part =
  let l = Float.of_int Kernels.fft_step_points in
  let dfts =
    match part with
    | First_pass -> 5. *. l *. 4.
    | Last_passes -> (5. *. l *. 8.) +. (6. *. l *. 2.)
  in
  let ops = dfts +. (Float.of_int flip_ops *. l) in
  column_threads
  * Float.to_int (Float.ceil (ops /. Float.of_int column_threads))

(* A part's chain: a dependent read, then the first pass's butterfly, or the
   step's passes 1 and 2 with one barrier between them. *)
let column_chain (step : step) part =
  let pass ~barriers i =
    pass_links step ~tl:column_threads ~simd:(column_threads / simd) ~barriers i
      step.radices.(i)
  in
  let passes =
    match part with
    | First_pass -> pass ~barriers:0 0
    | Last_passes -> pass ~barriers:1 1 +. pass ~barriers:0 2
  in
  Plan.read_links + Float.to_int (Float.ceil passes)

(* A line of one step through the column step: its first pass into the
   workspace, its last passes into [dst]. *)
let columns acc line src dst =
  let step = line.steps.(0) in
  let w = column_side acc src in
  take acc (width * count src.dims * step.l);
  let d = src.dims.(Array.length src.dims - 1) in
  let blocks = count src.dims / d * ceil_div d column_lines * column_parts in
  let part p a b ~shared ~tables =
    push acc
      ~work:(Column_step { line; step; part = p })
      ~src:a ~dst:b ~threads:column_threads ~shared
      ~bytes:(per_block (column_points * (segment a + segment b)))
      ~ops:(column_ops p) ~chain:(column_chain step p) ~tables blocks
  in
  part First_pass src w ~shared:0 ~tables:0;
  part Last_passes w dst ~shared:Kernels.fft_step_shared
    ~tables:(width * roots step.l)

(* A path's estimate with each work one launch, which no launch budget changes:
   a tally from [acc]'s workspace, apart from it. *)
let estimate acc f =
  let t = { acc with tally = true; launches = []; whole = 0. } in
  f t;
  t.whole

(* A line of one step: the contiguous step on rows; side by side where no side
   is scattered or wide and its estimate, each work one launch, is at most the
   transposes'; the column step where it fits, whose bytes alone keep its
   estimate below theirs; else the transposes. *)
let short acc line src dst =
  let step = line.steps.(0) in
  let apart s = s.access = Scattered || wide s in
  let cheaper path =
    let moved t = transposed t line src dst in
    if estimate acc path <= estimate acc moved then path acc else moved acc
  in
  if src.access = Rows && dst.access = Rows then
    push_step acc ~line ~step ~path:Contiguous src dst
  else if step.l <= side_most && (not (apart src)) && not (apart dst) then
    cheaper (fun t -> push_step t ~line ~step ~path:Side_by_side src dst)
  else if acc.columns && column_fits line src dst then columns acc line src dst
  else transposed acc line src dst

(* A long line's moves, in two levels: [n = s_0 M], [M] the rest of its steps'
   points. With [j = M j_1 + j_2] and [k = k_1 + s_0 k_2]: a transpose [(s_0,
   M)], [[j_1][j_2]] to [[j_2][j_1]]; the step on rows of [s_0]; a twiddled
   transpose [(M, s_0)] to [[k_1][j_2]]; the [M]-point transform of each row,
   the same scheme on lines of [M], or its last step; a transpose [(s_0, M)] to
   [[k_2][k_1]], natural order. *)
type move = Swap of int * int * scale | Run of step

let rec level (steps : step array) i =
  if i = Array.length steps - 1 then [ Run steps.(i) ]
  else
    let s = steps.(i).l in
    let rest = ref 1 in
    for k = i + 1 to Array.length steps - 1 do
      rest := !rest * steps.(k).l
    done;
    [ Swap (s, !rest, Plain); Run steps.(i); Swap (!rest, s, Twiddle) ]
    @ level steps (i + 1)
    @ [ Swap (s, !rest, Plain) ]

let swaps line =
  List.length
    (List.filter
       (function Swap _ -> true | Run _ -> false)
       (level line.steps 0))

(* A stage of a long line on [l1] points a line, from [src]: transpose [i] into
   [targets.(i)], the last as it is, the others as lines of the points they
   move; each step in place. The first reads [input] with [first]'s scale, the
   last writes [output] with [last]'s. *)
let stage acc line ~src ~targets ~input ~first ~last ~output =
  let t = Array.length targets in
  let cur = ref src and i = ref 0 in
  List.iter
    (function
      | Swap (a, b, scale) ->
          let k = !i in
          let scale =
            if k = 0 then first else if k = t - 1 then last else scale
          in
          let input : input = if k = 0 then input else Complex in
          let output : output = if k = t - 1 then output else Complex in
          let into =
            if k = t - 1 then targets.(k) else rows_of targets.(k) (a * b)
          in
          let from = if k = 0 then !cur else rows_of !cur (a * b) in
          push_transpose acc ~line ~split:(a, b) ~input ~scale ~output from into;
          cur := targets.(k);
          incr i
      | Run s ->
          let rows = rows_of !cur s.l in
          push_step acc ~line ~step:s ~path:Contiguous rows rows)
    (level line.steps 0)

(* A long line, its workspace lines [w0] and [w1] of [l1] points from the axis's
   first byte; [dst_scratch] where [dst] holds the line's points. *)
(* What a long line transforms: its own points, or Bluestein's convolution of
   them at [m]. A Rader line's [n - 1] fits a step, so it is never long. *)
type core = Direct | Chirped

let long acc line ~core ~dst_scratch ~input ~output src dst =
  let t = swaps line and lines = count src.dims in
  let piece = width * lines * line.l1 in
  let w0 = packed (ws acc 0) ~first:0 src.dims line.l1 in
  let w1 = packed (ws acc piece) ~first:0 src.dims line.l1 in
  match core with
  | Chirped ->
      (* Each transpose from one of [w0], [w1] to the other; the second stage
         from where the first ends, its last into [dst]. *)
      take acc (2 * piece);
      let alternate i = if i mod 2 = 0 then w0 else w1 in
      let t1 = Array.init t alternate in
      let t2 =
        Array.init t (fun i -> if i = t - 1 then dst else alternate (t + i))
      in
      stage acc line ~src ~targets:t1 ~input ~first:Chirp_in ~last:Kernel
        ~output:Complex;
      stage acc line
        ~src:t1.(t - 1)
        ~targets:t2 ~input:Complex ~first:Plain ~last:Chirp_out ~output
  | Direct ->
      (* Back from the last into [dst]: the workspace, then [dst] where it holds
         the line, else a second workspace line, in turn; the first takes the
         second workspace line where it would read and write one buffer. *)
      let other =
        if dst_scratch then
          packed ~base:dst.base dst.buffer ~first:dst.first src.dims line.l1
        else w1
      in
      let targets =
        Array.init t (fun i ->
            if i = t - 1 then dst
            else if (t - 2 - i) mod 2 = 0 then w0
            else other)
      in
      if src.buffer = targets.(0).buffer then targets.(0) <- w1;
      let two = Array.exists (fun s -> s.buffer = w1.buffer) targets in
      take acc (if two then 2 * piece else piece);
      stage acc line ~src ~targets ~input ~first:Plain ~last:Plain ~output

(* The transform along [axis] of [line], from [x]'s lines into [dst]'s, each
   with its buffer and its elements' bytes. [dst] holds a long line's points
   where it is C-contiguous, complex, and as large. *)
let along acc line ~x:(x, xb, xw) ~dst:(dst, db, dw) axis =
  let side, merged = sides x dst axis in
  let base = function
    | Operand -> fst acc.bases
    | Dst -> snd acc.bases
    | Workspace _ -> 0
  in
  let src = side xb ~width:xw ~base:(base xb) x merged.(0) in
  let d = side db ~width:dw ~base:(base db) dst merged.(1) in
  let dst_scratch =
    L.is_contiguous dst && dw = width && L.numel dst >= count d.dims * line.l1
  in
  let h = line.n in
  (* An odd real line's first transpose reads its reals or bins, its last writes
     bins up to [n / 2] or the real parts. *)
  let r2c = xw = real_width and c2r = dw = real_width in
  let input : input =
    if r2c && not line.real then Real
    else if c2r && not line.real then Bins
    else Complex
  in
  let output : output =
    if r2c && not line.real then Half
    else if c2r && not line.real then Real
    else Complex
  in
  let run core =
    let long = long acc line ~core ~dst_scratch ~input ~output in
    if line.real && r2c then begin
      (* An r2c's long halved line: its FFT's [h] points into [dst]'s first [h]
         bins, then the split into its [h + 1] in place. *)
      long src { d with points = h };
      push_pairs acc ~split:true d d
    end
    else if line.real then begin
      (* A c2r's: its [h + 1] bins into [h] points at the axis's workspace, its
         FFT's own workspace past them. *)
      let made = packed (ws acc 0) ~first:0 src.dims h in
      let bytes = width * count src.dims * h in
      push_pairs acc ~split:false src made;
      take acc bytes;
      let base = acc.base in
      acc.base <- base + bytes;
      long made d;
      acc.base <- base
    end
    else long src d
  in
  match line.how with
  | Rader -> short acc line src d
  | (Smooth | Bluestein _) when Array.length line.steps = 1 ->
      short acc line src d
  | Smooth -> run Direct
  | Bluestein _ -> run Chirped

let check_axes name axes r =
  if Array.length axes = 0 then invalid_arg (name ^ ": no axis");
  Array.iteri
    (fun i a ->
      if a < 0 || a >= r || (i > 0 && a <= axes.(i - 1)) then
        invalid_arg (name ^ ": axes"))
    axes

let finish acc =
  let launches = Array.of_list (List.rev acc.launches) in
  let seconds =
    Array.fold_left (fun t (l : launch) -> t +. l.seconds) 0. launches
  in
  { launches; workspace = acc.workspace; seconds }

(* An axis to plan: its line, its sides' layouts, buffers and widths. *)
type job = {
  line : line;
  x : L.t * buffer * int;
  dst : L.t * buffer * int;
  axis : int;
}

(* The most real points of a line, those its steps are planned for. *)
let line_most = 1 lsl 32
let real_points line = if line.real then 2 * line.n else line.n

(* [jobs]' call, or [None] where a line is past the host cut or [line_most].
   [before] runs ahead of the first job. *)
let run acc ?(before = ignore) jobs =
  let past j = past_cut j.line || real_points j.line > line_most in
  if List.exists past jobs then None
  else begin
    if jobs <> [] then before ();
    List.iter (fun j -> along acc j.line ~x:j.x ~dst:j.dst j.axis) jobs;
    Some (finish acc)
  end

let start_call ?columns name ~bases:((bx, bd) as bases) =
  let line b = b >= 0 && b < Plan.line in
  if not (line bx && line bd) then invalid_arg (name ^ ": bases");
  start ?columns ~bases name

let line_offset b =
  if Rig.Buffer.aligned b ~at:0 Plan.line then 0
  else begin
    (* The bytes [x] that bring the first byte onto a line, a bit at a time. *)
    let x = ref 0 and n = ref 2 in
    while !n <= Plan.line do
      if not (Rig.Buffer.aligned b ~at:!x !n) then x := !x + (!n / 2);
      n := 2 * !n
    done;
    (Plan.line - !x) mod Plan.line
  end

let plan_c2c ~columns ~bases ~axes x ~dst =
  let name = "Fft.c2c" in
  if L.shape x <> L.shape dst then invalid_arg (name ^ ": shapes differ");
  check_axes name axes (L.rank x);
  let acc = start_call ~columns name ~bases in
  let last = Array.length axes - 1 in
  let on_dst = (dst, Dst, width) in
  let job i =
    let axis = axes.(i) in
    let x = if i = last then (x, Operand, width) else on_dst in
    { line = line (L.dim dst axis) ~real:false; x; dst = on_dst; axis }
  in
  let jobs =
    if L.numel dst = 0 then []
    else List.init (last + 1) (fun i -> job (last - i))
  in
  run acc jobs

let c2c = plan_c2c ~columns:true
let c2c_moved = plan_c2c ~columns:false

(* A real shape's bins' shape along [axis]. *)
let bins shape axis =
  Iarray.of_array
    (Array.mapi
       (fun i d -> if i = axis then (d / 2) + 1 else d)
       (Iarray.to_array shape))

(* The line of a real line of [n] points: the halved line of [n / 2] complex
   ones where [n] is even. *)
let real_line n =
  if n mod 2 = 0 then line (n / 2) ~real:true else line n ~real:false

let r2c ~bases ~axes x ~dst =
  let name = "Fft.r2c" in
  check_axes name axes (L.rank x);
  let last = Array.length axes - 1 in
  let a = axes.(last) in
  if L.shape dst <> bins (L.shape x) a then
    invalid_arg (name ^ ": shapes differ");
  let acc = start_call name ~bases in
  let n = L.dim x a in
  let reals = (x, Operand, real_width) and on_dst = (dst, Dst, width) in
  (* Lines of no point store a bin each from a step of one point; the other
     axes' transforms of these zeros are zeros. *)
  let others () =
    List.init last (fun i ->
        let axis = axes.(last - 1 - i) in
        {
          line = line (L.dim dst axis) ~real:false;
          x = on_dst;
          dst = on_dst;
          axis;
        })
  in
  let jobs =
    if L.numel dst = 0 then []
    else if n = 0 then
      [ { line = line 1 ~real:false; x = reals; dst = on_dst; axis = a } ]
    else { line = real_line n; x = reals; dst = on_dst; axis = a } :: others ()
  in
  run acc jobs

let c2r ~bases ~axes x ~dst =
  let name = "Fft.c2r" in
  check_axes name axes (L.rank dst);
  let last = Array.length axes - 1 in
  let a = axes.(last) in
  if L.shape x <> bins (L.shape dst) a then
    invalid_arg (name ^ ": shapes differ");
  let acc = start_call name ~bases in
  let operand = (x, Operand, width) in
  (* The other axes into a C-contiguous copy at the workspace's first byte, the
     axes' own workspace past it. *)
  let copy = (L.contiguous (L.shape x), Workspace 0, width) in
  let others () =
    List.init last (fun i ->
        let axis = axes.(last - 1 - i) in
        {
          line = line (L.dim x axis) ~real:false;
          x = (if i = 0 then operand else copy);
          dst = copy;
          axis;
        })
  in
  let final () =
    {
      line = real_line (L.dim dst a);
      x = (if last = 0 then operand else copy);
      dst = (dst, Dst, real_width);
      axis = a;
    }
  in
  let jobs = if L.numel dst = 0 then [] else others () @ [ final () ] in
  let before () =
    if last > 0 then begin
      acc.base <- width * L.numel x;
      take acc 0
    end
  in
  run acc ~before jobs

(* Launches *)

(* A transpose's kernel and its 64-bit twin, of the triples [push_transpose] is
   given. *)
let move_kernels (input : input) scale (output : output) =
  let open Kernels in
  match (input, scale, output) with
  | Complex, Plain, Complex -> (Fft_t_cpc, Fft_t_cpc_w)
  | Real, Plain, Real -> (Fft_t_rpr, Fft_t_rpr_w)
  | Real, Plain, Complex -> (Fft_t_rpc, Fft_t_rpc_w)
  | Bins, Plain, Complex -> (Fft_t_bpc, Fft_t_bpc_w)
  | Complex, Twiddle, Complex -> (Fft_t_ctc, Fft_t_ctc_w)
  | Complex, Plain, Half -> (Fft_t_cph, Fft_t_cph_w)
  | Complex, Plain, Real -> (Fft_t_cpr, Fft_t_cpr_w)
  | Complex, Chirp_in, Complex -> (Fft_t_cic, Fft_t_cic_w)
  | Real, Chirp_in, Complex -> (Fft_t_ric, Fft_t_ric_w)
  | Bins, Chirp_in, Complex -> (Fft_t_bic, Fft_t_bic_w)
  | Complex, Kernel, Complex -> (Fft_t_ckc, Fft_t_ckc_w)
  | Complex, Chirp_out, Complex -> (Fft_t_coc, Fft_t_coc_w)
  | Complex, Chirp_out, Half -> (Fft_t_coh, Fft_t_coh_w)
  | Complex, Chirp_out, Real -> (Fft_t_cor, Fft_t_cor_w)
  | (Complex | Real | Bins), _, _ ->
      invalid_arg "Fft: no transpose reads, scales and writes so"

(* [d]'s divisor for device.h's [nx_divide]: [m = ⌊2^64 (2^l - d) / d⌋ + 1 mod
   2^64], [l = ⌈log₂ d⌉], the quotient by long division, one bit a step from
   [2^l - d < d]. *)
let divisor d =
  if d < 1 then invalid_arg "Fft.divisor";
  let rec bits x = if x = 0 then 0 else 1 + bits (x lsr 1) in
  let d64 = Int64.of_int d in
  let r = ref (Int64.sub (Int64.shift_left 1L (bits (d - 1))) d64)
  and q = ref 0L in
  for _ = 1 to 64 do
    r := Int64.shift_left !r 1;
    q := Int64.shift_left !q 1;
    if Int64.unsigned_compare !r d64 >= 0 then begin
      r := Int64.sub !r d64;
      q := Int64.succ !q
    end
  done;
  Int64.succ !q

let set_divisor w i d = Bytes.set_int64_le w (8 * i) (divisor d)

(* [s]'s axes from word [at]: an extent, a stride and the extent's divisor
   each. *)
let axis_words w ~at (s : side) =
  Array.iteri
    (fun k e ->
      let axis = at + (Kernels.fft_axis_words * k) in
      Plan.set w (axis + Kernels.fft_axis_extent) e;
      Plan.set w (axis + Kernels.fft_axis_stride) s.strides.(k);
      set_divisor w (axis + Kernels.fft_axis_divisor) e)
    s.dims

let side_words w ~at (s : side) =
  Plan.set w (at + Kernels.fft_side_first) s.first;
  Plan.set w (at + Kernels.fft_side_ps) s.ps;
  Plan.set w (at + Kernels.fft_side_rank) (Array.length s.dims);
  Plan.set w (at + Kernels.fft_side_width) s.width;
  if Array.length s.dims <= Kernels.fft_side_holds then
    axis_words w ~at:(at + Kernels.fft_side_axes) s

(* The axes of a transpose whose side is past what its words hold, [src]'s then
   [dst]'s: the call's own bytes, so that no layout takes a kept constant. *)
let tail_table (src : side) (dst : side) =
  let r = Array.length src.dims + Array.length dst.dims in
  let w = Bytes.make (8 * Kernels.fft_axis_words * r) '\000' in
  axis_words w ~at:0 src;
  axis_words w ~at:(Kernels.fft_axis_words * Array.length src.dims) dst;
  Bytes.unsafe_to_string w

(* The constant [key]. Tables are the forward transform's: an inverse conjugates
   around the forward operation. *)
let table b key length f =
  Plan.constant b ~key:(Printf.sprintf "fft %s" key) ~length f

(* The twiddle's two tables of [l], whose product gives each [w_l^k]. *)
let twiddles b l =
  ( table b (Printf.sprintf "glo %d" l) (width * low) (fun () ->
        fst (Fft_tables.global l)),
    table b
      (Printf.sprintf "ghi %d" l)
      (width * ceil_div l low)
      (fun () -> snd (Fft_tables.global l)) )

(* The slots of the tables a transpose's scale reads, [lo] and [hi]. *)
let scale_tables b (line : line) ~split:(a, c) = function
  | Plain -> (None, None)
  | Twiddle ->
      let lo, hi = twiddles b (a * c) in
      (Some lo, Some hi)
  | Chirp_in | Chirp_out ->
      let n = line.n in
      ( Some
          (table b (Printf.sprintf "chirp %d" n) (width * n) (fun () ->
               Fft_tables.chirp n)),
        None )
  | Kernel -> (
      match line.how with
      | Bluestein m ->
          let n = line.n in
          ( Some
              (table b (Printf.sprintf "bluestein %d %d" n m) (width * m)
                 (fun () -> Fft_tables.bluestein ~n ~m)),
            None )
      | Smooth | Rader -> invalid_arg "Fft: a kernel off Bluestein's line")

(* A launch of [k] and its refs: its sides' buffers at [place], [src]'s where it
   [read]s it, the tables [lo] and [hi], and the sides' axes, held from the
   words [sides] or in the tail. A pair's buffers, tables and tail take a
   transpose's words (fft.metal checks it). *)
let launch_of ?constants ?(more = []) b k ~place ~read ~tables:(lo, hi)
    ~sides:(i, o) (l : launch) set =
  let w = Plan.words k in
  set w;
  side_words w ~at:i l.src;
  side_words w ~at:o l.dst;
  let ref word (slot, at) ~write =
    Plan.set w word at;
    { Plan.word; slot; write }
  in
  let src =
    if read then [ ref Kernels.fft_move_src (place l.src.buffer) ~write:false ]
    else []
  and dst = ref Kernels.fft_move_dst (place l.dst.buffer) ~write:true in
  let held (s : side) = Array.length s.dims <= Kernels.fft_side_holds in
  let tail =
    if held l.src && held l.dst then []
    else
      let c = Plan.uploaded b (tail_table l.src l.dst) in
      [ ref Kernels.fft_move_tail (c, 0) ~write:false ]
  in
  let table word = function
    | Some slot -> [ ref word (slot, 0) ~write:false ]
    | None -> []
  in
  Plan.launch b k ~groups:(l.groups, 1, 1) ~shared:l.shared ~threads:l.threads
    ?constants ~cost:l.cost w
    (Array.of_list
       ((dst :: src) @ tail
       @ table Kernels.fft_move_lo lo
       @ table Kernels.fft_move_hi hi
       @ List.concat_map (fun (word, slot) -> table word (Some slot)) more))

let emit_transpose ~wide b ~inverse ~place (l : launch) t =
  let line, split, input, scale, output = t in
  let narrow, twin = move_kernels input scale output in
  let (ga, gb), _ = grid ~split l.src in
  let rows = ceil_div ga tile_side in
  (* A move's rows are lines: lanes along them where adjacent lines are
     adjacent. Else lanes along a run's points, which a twiddle's writing takes
     along [r]. *)
  let move = fst split = 1 in
  let along s = move && s.access <> Rows in
  let mode =
    (if move then Kernels.fft_move_one else 0)
    lor (if along l.src then Kernels.fft_read_along_r else 0)
    lor (if along l.dst || not move then Kernels.fft_write_along_r else 0)
    lor
    if inverse && not (copies input scale output) then Kernels.fft_conj else 0
  in
  launch_of b
    (if wide then twin else narrow)
    ~place ~read:true
    ~tables:(scale_tables b line ~split scale)
    ~sides:(Kernels.fft_move_in, Kernels.fft_move_out)
    l
    (fun w ->
      Plan.set w Kernels.fft_move_first l.first_group;
      set_divisor w Kernels.fft_move_rows_divisor rows;
      set_divisor w Kernels.fft_move_tiles_divisor (rows * ceil_div gb tile_side);
      Plan.set w Kernels.fft_move_a ga;
      Plan.set w Kernels.fft_move_b gb;
      Plan.set w Kernels.fft_move_n line.n;
      Plan.set w Kernels.fft_move_mode mode)

(* A split reads and writes its bins on [dst], in place; a pre-pass reads them
   from [src]. *)
let emit_pairs ~wide b ~inverse ~place ~split (l : launch) =
  let k : Kernels.t =
    match (split, wide) with
    | true, false -> Fft_split
    | true, true -> Fft_split_w
    | false, false -> Fft_prepass
    | false, true -> Fft_prepass_w
  in
  let h = (if split then l.dst else l.src).points - 1 in
  let lo, hi = twiddles b (2 * h) in
  launch_of b k ~place ~read:(not split) ~tables:(Some lo, Some hi)
    ~sides:(Kernels.fft_pair_in, Kernels.fft_pair_out) l (fun w ->
      Plan.set w Kernels.fft_pair_first l.first_group;
      set_divisor w Kernels.fft_pair_blocks_divisor
        (ceil_div ((h / 2) + 1) pair_block);
      Plan.set w Kernels.fft_pair_h h;
      Plan.set w Kernels.fft_pair_mode (if inverse then Kernels.fft_conj else 0))

(* The step's inner table: the forward roots it reads, [roots l]. *)
let inner b l =
  table b (Printf.sprintf "tw %d" l)
    (width * roots l)
    (fun () -> Fft_tables.roots l ~start:0 ~step:1 ~count:(roots l))

(* A halved line's roots of [2 h], its pairs' [w_2h^k] for [k <= h / 2]. *)
let halves b h =
  let count = (h / 2) + 1 in
  table b (Printf.sprintf "split %d" h) (width * count) (fun () ->
      Fft_tables.roots (2 * h) ~start:0 ~step:1 ~count)

(* A step whose kernel a test runs: 4096 points of radices 16 16 16, contiguous,
   one line a group, a step of a c2c's 7-smooth line, its only one, or one of a
   long line's, a Bluestein line's included. *)
let step_kernel ~line ~step ~path ~lines ~shared =
  step.l = Kernels.fft_step_points
  && step.radices = [| 16; 16; 16 |]
  && path = Contiguous && lines = 1
  && shared = Kernels.fft_step_shared
  && (line.how = Smooth || Array.length line.steps > 1)
  && not line.real

(* The flushing term's factor, the least upper bound over every sequence of
   passes; a test derives it from the passes' layers. *)
let flush_scale = 13.29
let rader_flush_scale = 42.
let bluestein_flush_scale n = (76. *. sqrt (Float.of_int n)) +. 72.

(* A specialised step's key: its function by its path and the points a thread
   holds, and its constants (kernels.h), its points, threads a line, lines a
   group and radices; what its sides hold, and its line's convolution in its
   group. A step takes one where it is not fft_s4096's: every step of a line, a
   long Bluestein line's an FFT of its own. *)
let key_step ~step ~path ~lines =
  let most =
    match path with Contiguous -> rows step | Side_by_side -> side step.l
  in
  lines = most || (lines < most && lines land (lines - 1) = 0)

(* Whether a step is a halved line's only one, which splits or makes its bins
   itself; a long halved line's steps take complex sides and leave them to a
   [Split] or a [Prepass]. *)
let halved_in_group ~line = line.real && Array.length line.steps = 1

(* What a step's sides hold, from their widths and points: a halved line in a
   group splits into bins where it reads floats, an r2c's, and makes its points
   from bins where it writes floats, a c2r's; an odd line reads floats or a
   c2r's bins, and writes floats or an r2c's bins; an r2c's line of no point
   reads nothing. *)
let step_sides ~line (l : launch) =
  let open Kernels in
  let reads = l.src.width = real_width and writes = l.dst.width = real_width in
  let halved =
    if not (halved_in_group ~line) then fft_halved_none
    else if reads then fft_halved_split
    else fft_halved_prepass
  in
  let input =
    if l.src.points = 0 then fft_in_none
    else if reads then fft_in_real
    else if writes && not line.real then fft_in_bins
    else fft_in_complex
  in
  let output =
    if writes then fft_out_real
    else if reads && not line.real then fft_out_half
    else fft_out_complex
  in
  (halved, input, output)

(* The convolution a step runs: its line's where the line runs in its group,
   none for a long line's steps. *)
let conv ~line = if Array.length line.steps > 1 then Smooth else line.how

let step_key (l : launch) ~line step ~path ~lines =
  let open Kernels in
  let c = Array.make fft_fc_count 0 in
  let halved, input, output = step_sides ~line l in
  c.(fft_fc_halved) <- halved;
  c.(fft_fc_input) <- input;
  c.(fft_fc_output) <- output;
  c.(fft_fc_conv) <-
    (match conv ~line with
    | Smooth -> fft_conv_none
    | Rader -> fft_conv_rader
    | Bluestein _ -> fft_conv_bluestein);
  let k =
    match path with
    | Contiguous -> if step.p = 8 then Fft_step_p8 else Fft_step_p16
    | Side_by_side -> if step.p = 8 then Fft_side_p8 else Fft_side_p16
  in
  c.(fft_fc_l) <- step.l;
  c.(fft_fc_tl) <- step.tl;
  c.(fft_fc_lines) <- lines;
  c.(fft_fc_passes) <- Array.length step.radices;
  Array.iteri (fun i r -> c.(fft_fc_r0 + i) <- r) step.radices;
  (k, c)

(* What runs a step's launch, on sides it reads. *)
type code = Unrun | S4096 | Key of (Kernels.t * int array)

let step_code (l : launch) ~line ~step ~path ~lines =
  if wide l.src || wide l.dst then Unrun
  else if
    narrow l.src && narrow l.dst
    && step_kernel ~line ~step ~path ~lines ~shared:l.shared
  then S4096
  else if key_step ~step ~path ~lines then
    Key (step_key l ~line step ~path ~lines)
  else Unrun

let key (l : launch) =
  match l.work with
  | Step { line; step; path; lines } -> (
      match step_code l ~line ~step ~path ~lines with
      | Key k -> Some k
      | S4096 | Unrun -> None)
  | Column_step _ | Transpose _ | Split _ | Prepass _ -> None

(* Whether [l], a step's launch, is one a kernel computes. *)
let step_runs (l : launch) ~line ~step ~path ~lines =
  step_code l ~line ~step ~path ~lines <> Unrun

(* Whether [l], a column step's launch, is one fft_c4096_first or fft_c4096_last
   computes. *)
let column_runs (l : launch) ~line ~step =
  column_kernel line step && narrow l.src && narrow l.dst

let runs (c : call) =
  Array.for_all
    (fun (l : launch) ->
      match l.work with
      | Step { line; step; path; lines } -> step_runs l ~line ~step ~path ~lines
      | Column_step { line; step; _ } -> column_runs l ~line ~step
      | Transpose _ | Split _ | Prepass _ -> true)
    c.launches

(* A line's convolution's tables, by their words: Rader's kernel and logarithms,
   Bluestein's kernel and chirp. *)
let conv_tables b line =
  let n = line.n in
  match line.how with
  | Smooth -> []
  | Rader ->
      [
        ( Kernels.fft_key_kernel,
          table b
            (Printf.sprintf "rader %d" n)
            (width * (n - 1))
            (fun () -> fst (Fft_tables.rader n)) );
        ( Kernels.fft_key_logs,
          table b (Printf.sprintf "logs %d" n) (4 * n) (fun () ->
              snd (Fft_tables.rader n)) );
      ]
  | Bluestein m ->
      [
        ( Kernels.fft_key_kernel,
          table b (Printf.sprintf "bluestein %d %d" n m) (width * m) (fun () ->
              Fft_tables.bluestein ~n ~m) );
        ( Kernels.fft_key_chirp,
          table b (Printf.sprintf "chirp %d" n) (width * n) (fun () ->
              Fft_tables.chirp n) );
      ]

let emit_step b ~inverse ~place (l : launch) ~line ~step ~path ~lines =
  let mode = if inverse then Kernels.fft_conj else 0 in
  let tables =
    ( Some (inner b step.l),
      if halved_in_group ~line then Some (halves b line.n) else None )
  in
  match step_code l ~line ~step ~path ~lines with
  | Unrun -> invalid_arg (Printf.sprintf "Fft.emit: a step of %d points" step.l)
  | S4096 ->
      launch_of b Fft_s4096 ~place ~read:true ~tables
        ~sides:(Kernels.fft_step_in, Kernels.fft_step_out) l (fun w ->
          Plan.set w Kernels.fft_step_first l.first_group;
          Plan.set w Kernels.fft_step_mode mode)
  | Key (k, constants) ->
      let more = if conv ~line = Smooth then [] else conv_tables b line in
      let n = if conv ~line = Smooth then step.l else line.n in
      launch_of ~constants ~more b k ~place ~read:true ~tables
        ~sides:(Kernels.fft_key_in, Kernels.fft_key_out) l (fun w ->
          Plan.set w Kernels.fft_key_first l.first_group;
          Plan.set w Kernels.fft_key_mode mode;
          Plan.set w Kernels.fft_key_lines (count l.src.dims);
          Plan.set w Kernels.fft_key_n n)

let emit_column b ~inverse ~place (l : launch) ~line ~step ~part =
  if not (column_runs l ~line ~step) then
    invalid_arg (Printf.sprintf "Fft.emit: a column step of %d points" step.l);
  let d = l.src.dims.(Array.length l.src.dims - 1) in
  let blocks = ceil_div d column_lines in
  let k, tables =
    match part with
    | First_pass -> (Kernels.Fft_c4096_first, (None, None))
    | Last_passes -> (Kernels.Fft_c4096_last, (Some (inner b step.l), None))
  in
  launch_of b k ~place ~read:true ~tables
    ~sides:(Kernels.fft_cols_in, Kernels.fft_cols_out) l (fun w ->
      Plan.set w Kernels.fft_cols_first l.first_group;
      Plan.set w Kernels.fft_cols_mode (if inverse then Kernels.fft_conj else 0);
      Plan.set w Kernels.fft_cols_blocks blocks;
      set_divisor w Kernels.fft_cols_blocks_divisor blocks;
      Plan.set w Kernels.fft_cols_lines d)

let emit ?wide b ~inverse ~place (l : launch) =
  match l.work with
  | Step { line; step; path; lines } ->
      emit_step b ~inverse ~place l ~line ~step ~path ~lines
  | Column_step { line; step; part } ->
      emit_column b ~inverse ~place l ~line ~step ~part
  | Transpose t ->
      let wide = Option.value wide ~default:t.wide in
      emit_transpose ~wide b ~inverse ~place l
        (t.line, t.split, t.input, t.scale, t.output)
  | Split { wide = w } ->
      emit_pairs
        ~wide:(Option.value wide ~default:w)
        b ~inverse ~place ~split:true l
  | Prepass { wide = w } ->
      emit_pairs
        ~wide:(Option.value wide ~default:w)
        b ~inverse ~place ~split:false l
